"""Phase 4: tensor-parallel Llama-70B layer boundary  down-proj -> residual add -> RMSNorm -> QKV.

Per rank r of ``world`` (one process / GPU each), sharding:
  h_r       (M, Kr)       K-shard of the FFN hidden, Kr = 28672 / world
  Wdown_r   (8192, Kr)    W_down[:, r*Kr:(r+1)*Kr]
  Wqkv_r    (N2r, 8192)   W_qkv[r*N2r:(r+1)*N2r]  (gamma folded in for CtappBoundary)
  out_r     (M, N2r)      QKV output column shard
The reduced residual stream x_new = resid + sum_r h_r Wdown_r^T is left in full on every rank (``x()``).

``CtappBoundary`` runs the CTA-pipelined NVSwitch-multicast protocol (csrc/gemm_tp4.cu); the other classes are baselines
with the same ``forward(h_r, resid)`` signature. Also: reference / 1-GPU chains and a small process-spawn + timing harness.
"""

from __future__ import annotations

import os
import statistics
import sys
import traceback

import torch
import torch.distributed as dist

DTYPE = torch.bfloat16
EPS = 1e-5
N1 = 8192          # hidden size (down-proj N, RMSNorm width)
K_FFN = 28672
N2 = 10240         # fused QKV width


def _symm(shape, dtype, dev, group_name):
    import torch.distributed._symmetric_memory as symm_mem
    t = symm_mem.empty(*shape, dtype=dtype, device=dev) if isinstance(shape, tuple) else symm_mem.empty(shape, dtype=dtype, device=dev)
    hdl = symm_mem.rendezvous(t, group_name)
    return t, hdl


def shard_weights(Wdown, Wqkv, gamma, rank, world):
    """Full weights -> (Wdown_r, Wqkv_r unfolded, Wqkv_r with gamma folded), all bf16 contiguous."""
    Kr, N2r = Wdown.shape[1] // world, Wqkv.shape[0] // world
    Wd = Wdown[:, rank * Kr:(rank + 1) * Kr].contiguous()
    Wq = Wqkv[rank * N2r:(rank + 1) * N2r].contiguous()
    Wqf = (Wqkv.float() * gamma.float()[None, :])[rank * N2r:(rank + 1) * N2r].to(DTYPE).contiguous()
    return Wd, Wq, Wqf


def rmsnorm_gamma(x, gamma):
    """Eager, unfused: x * rsqrt(mean(x^2) + eps) * gamma (fp32 inner, bf16 round, then gamma)."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)).to(x.dtype) * gamma


def fold_gamma(Wq, gamma):
    """RMSNorm gamma folded into the QKV weight columns: (Wq * gamma[None, :]) in fp32, rounded to bf16, contiguous.
    Same arithmetic as ``shard_weights`` (fold before sharding or after: rows are independent)."""
    return (Wq.float() * gamma.float()[None, :]).to(DTYPE).contiguous()


# --------------------------------------------------------------------------- CTA-pipelined boundary
class CtappBoundary:
    """Production CTA-pipelined boundary (Phase-4 decisions).

    Per forward: producer GEMM mode 4 (consumers only signal tiles) + a concurrent reducer kernel (version 3, R blocks) on a
    second stream, then the QKV GEMM mode 2 (per-panel wait). Both GEMMs run on ``num_SMs - R`` SMs so the reducer is always
    resident. The epoch lives on the device (``epoch_dev``, bumped by ``add_(1)`` at the start of each forward), so the same
    kernel launches can be CUDA-graph captured and replayed.

    use_graph=True: the QKV epilogue's ``rowss`` pointer is baked into the EVT arguments, and rowss is double-buffered by epoch
    parity, so the forward is captured TWICE (one graph per parity) and the graphs are replayed alternately. The first forward
    runs un-captured (warm-up, counts as a real forward); each later parity is captured on first use. Inputs are copied into
    static buffers before every replay; the returned ``out`` / ``x()`` buffers are the same tensors every call.
    protocol=False: mode-0 GEMMs with the same SM split and rasters, no reducer / waits / residual (raw two-GEMM floor).
    protocol=True or "v3": the multimem protocol above (instance serves exactly M rows).

    protocol="v5" (Phase 6, TP4 only): panel ownership. Rank r owns the 128-row panels m with m % 4 == r. Per forward:
    ``tp4_step`` (epoch += 1, zero rowss_local / panel_done) -> producer GEMM mode 7 (each tile TMA-stored straight into its
    owner's inbox, unicast value flag) || owner reducer v5 (R CTAs of ``red_threads`` on a second stream: sums the 4 inbox
    slices + resid, pushes x and the panel's row sums of squares to every rank, value flag per panel) -> QKV GEMM mode 5
    (value-flag panel wait). No multimem. All flags are epoch values, so one instance sized ``max_M`` serves every
    M <= max_M with M % 128 == 0 (``forward`` takes M from h_r; ``out`` / ``x()`` are views of the first M rows).
    ``down_raster``: the producer's raster (1 = m fast, 2 = n fast); the reducer walks its units in the same order.
    ``red_threads`` / ``red_blocks`` (default R): reducer CTA shape; both GEMMs run on SM_count - R SMs.
    ``red_variant``: 1 (default) = warp-per-unit, no CTA barrier (red_threads 1024 | 512); 0 = 31 worker warps + 1 signaller
    warp per CTA, batch-synchronous (red_threads 1024 | 544; slower: the per-batch fence/atomic round trips set the pace);
    2 = the S3 role reducer as a stand-alone kernel (red_threads 384, ``role_rows`` rows per lane).

    ``fusion`` (Phase 6 S3, v5 only; programmatic dependent launch, one stream for the GEMMs):
      "none" (default): the S2 3-kernel form above.
      "pdl"  (Variant A'): as "none", but the producer triggers its PDL dependent right after its prologue (``pdl_trigger``) and
             the QKV GEMM (mode 5, grid SM_count - R) is launched as that dependent, so it starts the moment producer CTAs
             retire (no launch / stream gap). The consumer grid stays at SM_count - R: a launch-order inversion (consumer CTAs
             on the reducer's SMs) can only serialise, never deadlock.
      "pdl1" (single stream, S3 addition): tp4_step, then the reducer kernel (triggers its PDL dependent at entry), the producer
             as its PDL dependent (so the reducer is resident before the producer launches: no inversion, early producer trigger
             safe), then the QKV GEMM as the producer's PDL dependent (mode 5, grid SM_count - R). No second stream / events.
      "role" (Variant B, 2 kernels + tp4_step, no second stream, no events): producer mode 7 on SM_count - R SMs with
             ``pdl_trigger``, then the QKV GEMM in mode 8 on all SMs as its PDL dependent: the first R consumer CTAs to start
             (they land on the R SMs the producer leaves free) run the owner reducer (384 threads, ``role_rows`` rows per lane)
             before their own GEMM tiles; the others start as producer CTAs retire and wait per panel.
    ``role_stages`` (role / red_variant 2): 0 = register loads; 2 | 3 = cp.async (L1-bypass) staging of the reducer's loads in
    a per-warp smem ring (needs role_rows 2). The role CTAs run with the GEMM's 214 KB smem carve-out (~28 KB L1), which caps
    plain loads in flight; the staged loop is not L1-bound.
    ``down_tail`` (S3): producer tile order. 0 = stock (raster ``down_raster``, swizzle ``down_swizzle``); C > 0 = m-fast over
    the first 32 - C n-tiles, then the last C n-tiles panel-major (needs raster 1 / swizzle 1), so the panels of the
    second-to-last wave(s) are reduced and consumable while the last wave runs; "auto" (default) = 16 for M >= 2048, 8 below
    (S3 sweep at 1k-8k), 0 unless raster 1 / swizzle 1. The reducer's unit order follows it.
    ``graph_bind`` (v5, use_graph): capture the caller's h_r / resid directly (graphs keyed by (parity, M, pointers)) instead of
    copying them into static buffers before each replay (the copy costs ~0.07 ms at 4k); the caller must pass the same,
    still-alive tensors to replay (bench use).
    ``steal`` (S3 work stealing): the reducer's units are claimed dynamically from a shared counter, and the QKV GEMM runs in
    mode 8 (role preamble) with every CTA claiming units before its GEMM tiles: the standalone reducer ("none" / "pdl") or the R
    early role CTAs ("role") keep pace with the producer, and the consumer CTAs that start as producer CTAs retire drain the
    remaining units (the producer's last wave) on all SMs instead of R. Consumer grid: SM_count - R ("none" / "pdl"), SM_count
    ("role").
    ``qkv_tile``: 256 (128x256 tile) | 128 (128x128). ``down_swizzle``: the producer's swizzle (raster ``down_raster``); the
    reducer's unit order follows it. ``trace`` (or env CTAPP_TRACE=1): %globaltimer stamps per CTA and per panel publish on
    every eager forward (``trace_summary()``).
    """

    def __init__(self, M, Kr, N2r, rank, world, group_name, device, resid=True, protocol=True, use_graph=False, R=None,
                 qkv_raster=2, qkv_swizzle=1, max_M=None, down_raster=None, red_threads=1024, red_blocks=None, red_variant=1,
                 fusion="none", qkv_tile=256, role_rows=4, down_swizzle=1, pdl_trigger=None, trace=None, role_stages=0, steal=False,
                 down_tail="auto", graph_bind=False):
        from ctapp.ext import load_tp4
        assert N1 == 8192 and M % 128 == 0, (N1, M)
        assert protocol in (True, False, "v3", "v5"), protocol
        self.v5 = protocol == "v5"
        self.ext = load_tp4()
        self.M, self.Kr, self.N2r, self.rank, self.world = M, Kr, N2r, rank, world
        self.dev, self.use_resid, self.protocol, self.use_graph = device, resid, bool(protocol), use_graph
        if self.v5:
            self.R = R or 4
        else:
            self.R = R or (12 if M <= 2048 else 8)
        self.qkv_raster, self.qkv_swizzle = qkv_raster, qkv_swizzle
        self.nsm = torch.cuda.get_device_properties(device).multi_processor_count
        self.sms = self.nsm - self.R
        if self.v5:
            assert fusion in ("none", "pdl", "pdl1", "role"), fusion
            assert qkv_tile in (256, 128) and self.N2r % qkv_tile == 0, (qkv_tile, self.N2r)
            self.fusion, self.qkv_tile, self.role_rows, self.down_swizzle = fusion, qkv_tile, role_rows, down_swizzle
            assert role_stages in (0, 2, 3) and (role_stages == 0 or role_rows == 2), (role_stages, role_rows)
            self.role_stages = role_stages
            self.steal = bool(steal)
            assert down_tail == "auto" or (isinstance(down_tail, int) and 0 <= down_tail <= 32), down_tail
            self.down_tail = down_tail
            self.graph_bind = bool(graph_bind)
            assert not (self.steal and role_stages), "steal uses the register reducer loop (role_stages 0)"
            # producer trigger default: "pdl" 0 (an early trigger lets the queued consumer CTAs take the free SMs before the reducer on
            # the side stream launches: measured +0.06-0.11 ms b2b at 4k), "pdl1" / "role" 1 (no inversion possible)
            self.pdl_trigger = (1 if fusion in ("pdl1", "role") else 0) if pdl_trigger is None else int(pdl_trigger)
            self.qkv_sms = self.nsm if fusion == "role" else self.sms
            self.trace = bool(int(os.environ.get("CTAPP_TRACE", "0"))) if trace is None else bool(trace)
            self._init_v5(M, group_name, max_M, down_raster, red_threads, red_blocks, red_variant)
            return
        assert fusion == "none", "fusion needs protocol v5"
        gn = group_name
        self.partials, h1 = _symm((M, N1), DTYPE, device, gn)
        self.xbuf, h2 = _symm((M, N1), DTYPE, device, gn)
        self.xbuf1, h2b = _symm((M, N1), DTYPE, device, gn)   # second x buffer: x double-buffered by epoch parity
        self.tile_cnt, h3 = _symm((M // 128 * (N1 // 256),), torch.int32, device, gn)
        self.panel_cnt, h4 = _symm((M // 128,), torch.int32, device, gn)
        self.rowss, h5 = _symm((2, M), torch.float32, device, gn)
        for h, t in ((h1, self.partials), (h2, self.xbuf), (h2b, self.xbuf1), (h3, self.tile_cnt), (h4, self.panel_cnt), (h5, self.rowss)):
            assert t.data_ptr() == h.buffer_ptrs[rank]
        self.partials_mc, self.tile_mc = h1.multicast_ptr, h3.multicast_ptr
        self.xbufs, self.x_mcs = [self.xbuf, self.xbuf1], [h2.multicast_ptr, h2b.multicast_ptr]
        self.x_mc = self.x_mcs[0]
        self._last_par = 0
        self.panel_mc, self.rowss_mc = h4.multicast_ptr, h5.multicast_ptr
        self._hdls = (h1, h2, h2b, h3, h4, h5)
        for t in (self.tile_cnt, self.panel_cnt, self.rowss):
            t.zero_()
        self.epoch_dev = torch.zeros(1, dtype=torch.int32, device=device)
        self.s_red = torch.cuda.Stream(device=device)
        self.ev_start, self.ev_red = torch.cuda.Event(), torch.cuda.Event()
        torch.cuda.synchronize(device)
        dist.barrier()
        self.out = torch.empty(M, N2r, dtype=DTYPE, device=device)
        self.epoch = 0
        self.Wd = self.Wq = None
        self.h_static = torch.empty(M, Kr, dtype=DTYPE, device=device) if use_graph else None
        self.resid_static = torch.empty(M, N1, dtype=DTYPE, device=device) if use_graph and resid else None
        self._graphs = {}
        self._warm = False
        self._warmed = False

    def _init_v5(self, M, gn, max_M, down_raster, red_threads, red_blocks, red_variant):
        assert self.world == 4, "protocol v5 is TP4 only"
        max_M = max_M or M
        assert max_M % 128 == 0 and M <= max_M, (M, max_M)
        dev, rank, world = self.dev, self.rank, self.world
        self.max_M = max_M
        self.down_raster = down_raster or 1
        self.red_threads, self.red_variant = red_threads, red_variant
        self.red_blocks = red_blocks or self.R               # reducer CTAs (default: one per reserved SM)
        self.ppo = ppo = -(-max_M // 128 // world)            # panels per owner (ceil)
        self.inbox, hi = _symm((world, ppo * 128, N1), DTYPE, dev, gn)   # this rank's inbox; slot s = source rank s
        self.tile_flags, ht = _symm((world, ppo, N1 // 256), torch.int32, dev, gn)
        self.panel_flag, hp = _symm((max_M // 128,), torch.int32, dev, gn)
        self.rowss, hr = _symm((2, max_M), torch.float32, dev, gn)
        self.xbuf, hx0 = _symm((max_M, N1), DTYPE, dev, gn)
        self.xbuf1, hx1 = _symm((max_M, N1), DTYPE, dev, gn)
        for h, t in ((hi, self.inbox), (ht, self.tile_flags), (hp, self.panel_flag), (hr, self.rowss), (hx0, self.xbuf), (hx1, self.xbuf1)):
            assert t.data_ptr() == h.buffer_ptrs[rank]
        self._hdls = (hi, ht, hp, hr, hx0, hx1)
        self.xbufs = [self.xbuf, self.xbuf1]
        slot = ppo * 128 * N1 * 2                             # bytes of one inbox slot
        self._dst_ptrs = [hi.buffer_ptrs[d] + rank * slot for d in range(world)]      # producer: my slot at every owner
        self._peer_tf = [ht.buffer_ptrs[d] for d in range(world)]
        self._inbox_local = [self.inbox.data_ptr() + s * slot for s in range(world)]  # reducer: my inbox, per source
        self._peer_x = [[hx0.buffer_ptrs[d] for d in range(world)], [hx1.buffer_ptrs[d] for d in range(world)]]
        self._peer_rowss = [[hr.buffer_ptrs[d] + par * max_M * 4 for d in range(world)] for par in (0, 1)]
        self._peer_pf = [hp.buffer_ptrs[d] for d in range(world)]
        self._dview = self.inbox.view(-1, N1)                 # unused D argument of tp4_down mode 7 (shape only)
        self.rowss_local = torch.zeros(max_M, dtype=torch.float32, device=dev)
        self.panel_done = torch.zeros(max_M // 128, dtype=torch.int32, device=dev)
        self.role_ctr = torch.zeros(2, dtype=torch.int32, device=dev)          # [mode-8 role assignment, steal unit counter], zeroed by tp4_step
        self.trace_down = torch.zeros(self.nsm * 8, dtype=torch.int64, device=dev)   # CTAPP_TRACE buffers (see trace_summary)
        self.trace_qkv = torch.zeros(self.nsm * 8, dtype=torch.int64, device=dev)
        self.trace_red = torch.zeros(max_M // 128, dtype=torch.int64, device=dev)
        for t in (self.tile_flags, self.panel_flag, self.rowss):
            t.zero_()
        self.epoch_dev = torch.zeros(1, dtype=torch.int32, device=dev)
        self.s_red = torch.cuda.Stream(device=dev)
        self.ev_start, self.ev_red = torch.cuda.Event(), torch.cuda.Event()
        torch.cuda.synchronize(dev)
        dist.barrier()
        self._out_full = torch.empty(max_M, self.N2r, dtype=DTYPE, device=dev)
        self.out = self._out_full[:M]
        self.epoch = 0
        self._last_par, self._last_M = 0, M
        self.Wd = self.Wq = None
        self.h_static = torch.empty(max_M, self.Kr, dtype=DTYPE, device=dev) if self.use_graph else None
        self.resid_static = torch.empty(max_M, N1, dtype=DTYPE, device=dev) if self.use_graph and self.use_resid else None
        self._graphs = {}
        self._warm = False
        self._warmed = False

    def _warmup_v5(self):
        """Every kernel of the v5 forward is launched once while the GPU is idle (CUDA lazy loading: the first launch of a
        kernel while a spinning kernel occupies SMs can deadlock). No cross-rank writes: mode-0 GEMMs into local scratch,
        tp4_step (epoch reset afterwards), a zero-unit reducer launch."""
        e, M = self.ext, 128
        hz = torch.zeros(M, self.Kr, dtype=DTYPE, device=self.dev)
        wd = torch.zeros(N1, self.Kr, dtype=DTYPE, device=self.dev)
        wq = torch.zeros(self.N2r, N1, dtype=DTYPE, device=self.dev)
        dz = torch.empty(M, N1, dtype=DTYPE, device=self.dev)   # local scratch: a peer may already be writing into the inbox
        # mode 0 + the PDL path (trigger + dependent launch; mode 0 still waits for the primary grid) of both GEMM instances used
        e.tp4_down(hz, wd, dz, self.down_raster, 1, 0, self.rank, self.world, 0, 0, self.tile_flags, 0, 0, None, 0, 0, sms=self.sms,
                   pdl_trigger=self.pdl_trigger, pdl=1 if self.fusion == "pdl1" else 0)
        e.tp4_qkv(self.xbuf[:M], wq, self._out_full[:M], self.qkv_raster, self.qkv_swizzle, 0, 0, self.panel_flag, 1, self.rowss[0],
                  sms=self.qkv_sms, pdl=1 if self.fusion != "none" else 0, tile=self.qkv_tile)
        e.tp4_step(self.epoch_dev, self.rowss_local, self.panel_done, M, self.role_ctr)
        self._reduce_v5(0, None, 0)
        torch.cuda.synchronize(self.dev)
        self.epoch_dev.zero_()
        self.panel_done.zero_()
        self.role_ctr.zero_()
        torch.cuda.synchronize(self.dev)

    def _tail(self, M):
        if self.down_tail != "auto":
            return self.down_tail
        if self.down_raster != 1 or self.down_swizzle != 1:
            return 0
        return 16 if M // 128 >= 16 else 8

    def _tr(self, t):
        return t.data_ptr() if self.trace else 0

    def _reduce_v5(self, M, resid, parity):
        self.ext.tp4_reduce(M, self.rank, self.world, 0, self.down_raster, self.red_blocks, self.tile_flags, 0, 0, resid, 0, 0,
                            peer_x=self._peer_x[parity], version=5, epoch_dev=self.epoch_dev, inbox_ptrs=self._inbox_local,
                            peer_rowss=self._peer_rowss[parity], peer_panel_flag=self._peer_pf,
                            rowss_local=self.rowss_local.data_ptr(), panel_done=self.panel_done.data_ptr(),
                            panels_per_owner=self.ppo, threads=self.red_threads, variant=self.red_variant,
                            swz=self.down_swizzle, rows=self.role_rows, red_trace=self._tr(self.trace_red),
                            stages=self.role_stages if self.red_variant == 2 else 0,
                            unit_ctr=self.role_ctr.data_ptr() + 4 if self.steal else 0, tail_cols=self._tail(M),
                            pdl_trigger=1 if self.fusion == "pdl1" else 0)

    def _down_v5(self, h_r, Wd, M):
        self.ext.tp4_down(h_r, Wd, self._dview[:M], self.down_raster, self.down_swizzle, 7, self.rank, self.world, 0, 0, self.tile_flags,
                          0, 0, None, 0, 0, sms=self.sms, epoch_dev=self.epoch_dev, dst_ptrs=self._dst_ptrs, dst_rows=self.ppo * 128,
                          peer_tile_flags=self._peer_tf, panels_per_owner=self.ppo, pdl_trigger=self.pdl_trigger,
                          trace=self._tr(self.trace_down), tail_cols=self._tail(M), pdl=1 if self.fusion == "pdl1" else 0)

    def _qkv_v5(self, Wq, M, parity, resid=None):
        if self.fusion == "role" or self.steal:   # mode 8: role-switching / unit-stealing consumer
            self.ext.tp4_qkv(self.xbufs[parity][:M], Wq, self._out_full[:M], self.qkv_raster, self.qkv_swizzle, 8, 0, self.panel_flag, 1,
                             self.rowss[parity], sms=self.qkv_sms, pdl=1 if self.fusion != "none" else 0, epoch_dev=self.epoch_dev,
                             tile=self.qkv_tile, R=self.R if self.fusion == "role" else 0,
                             role_ctr=self.role_ctr, rank=self.rank, red_order=self.down_raster, red_swz=self.down_swizzle,
                             tile_flags=self.tile_flags, resid=resid, inbox_ptrs=self._inbox_local, peer_x=self._peer_x[parity],
                             peer_rowss=self._peer_rowss[parity], peer_panel_flag=self._peer_pf,
                             rowss_local=self.rowss_local.data_ptr(), panel_done=self.panel_done.data_ptr(), panels_per_owner=self.ppo,
                             red_rows=self.role_rows, trace=self._tr(self.trace_qkv), red_trace=self._tr(self.trace_red),
                             red_stages=self.role_stages, red_dyn=1 if self.steal else 0, red_tail=self._tail(M))
            return
        self.ext.tp4_qkv(self.xbufs[parity][:M], Wq, self._out_full[:M], self.qkv_raster, self.qkv_swizzle, 5, 0, self.panel_flag, 1,
                         self.rowss[parity], sms=self.qkv_sms, pdl=1 if self.fusion != "none" else 0, epoch_dev=self.epoch_dev,
                         tile=self.qkv_tile, trace=self._tr(self.trace_qkv))

    def _enqueue_v5(self, h_r, resid, parity, Wd, Wq, M):
        e = self.ext
        cur = torch.cuda.current_stream()
        if self.trace:
            for t in (self.trace_down, self.trace_qkv, self.trace_red):
                t.zero_()
        e.tp4_step(self.epoch_dev, self.rowss_local, self.panel_done, M, self.role_ctr)
        if self.fusion == "role":        # Variant B: 2 GEMM launches on this stream, no second stream / events
            self._down_v5(h_r, Wd, M)
            self._qkv_v5(Wq, M, parity, resid)
            return
        if self.fusion == "pdl1":        # single stream: reducer (triggers at entry) -> producer (PDL) -> consumer (PDL)
            self._reduce_v5(M, resid, parity)
            self._down_v5(h_r, Wd, M)
            self._qkv_v5(Wq, M, parity, resid)
            return
        self.ev_start.record(cur)
        self.s_red.wait_event(self.ev_start)
        with torch.cuda.stream(self.s_red):
            self._reduce_v5(M, resid, parity)
            self.ev_red.record(self.s_red)
        self._down_v5(h_r, Wd, M)
        self._qkv_v5(Wq, M, parity, resid)
        cur.wait_event(self.ev_red)

    def warmup(self):
        """Force lazy CUDA module loading of both GEMM kernels (mode-0 launches on zeros), once per instance. If the first
        launch of a GEMM happens while the (spinning) reducer kernel already occupies SMs, the run faults (spin-limit trap)
        on every rank. Needs no real weights (dummy zero weights are used)."""
        if self._warmed:
            return
        if self.v5:
            self._warmup_v5()
            self._warmed = True
            return
        e = self.ext
        hz = torch.zeros(self.M, self.Kr, dtype=DTYPE, device=self.dev)
        wd = torch.zeros(N1, self.Kr, dtype=DTYPE, device=self.dev)
        wq = torch.zeros(self.N2r, N1, dtype=DTYPE, device=self.dev)
        e.tp4_down(hz, wd, self.partials, 1, 1, 0, self.rank, self.world, 0, self.tile_mc, self.tile_cnt,
                   self.partials_mc, self.x_mc, None, self.rowss_mc, self.panel_mc, sms=self.sms)
        e.tp4_qkv(self.xbuf, wq, self.out, self.qkv_raster, self.qkv_swizzle, 0, 0, self.panel_cnt, 32 * self.world, self.rowss[0], sms=self.sms)
        torch.cuda.synchronize(self.dev)
        self._warmed = True

    def _check_w(self, Wd, Wq):
        assert Wd.shape == (N1, self.Kr) and Wd.dtype == DTYPE and Wd.is_contiguous(), (Wd.shape, Wd.dtype)
        assert Wq.shape == (self.N2r, N1) and Wq.dtype == DTYPE and Wq.is_contiguous(), (Wq.shape, Wq.dtype)

    def set_weights(self, Wdown_r, Wqkv_r_folded):
        """Default weights (used when forward() gets none; baked into the graphs when use_graph). Also runs warmup()."""
        self._check_w(Wdown_r, Wqkv_r_folded)
        self.Wd, self.Wq = Wdown_r, Wqkv_r_folded
        self.warmup()

    def _enqueue(self, h_r, resid, parity, Wd, Wq, M=None):
        e = self.ext
        if self.v5:
            self._enqueue_v5(h_r, resid, parity, Wd, Wq, M)
            return
        if not self.protocol:
            e.tp4_down(h_r, Wd, self.partials, 1, 1, 0, self.rank, self.world, 0, self.tile_mc, self.tile_cnt,
                       self.partials_mc, self.x_mcs[0], None, self.rowss_mc, self.panel_mc, sms=self.sms)
            e.tp4_qkv(self.xbufs[0], Wq, self.out, self.qkv_raster, self.qkv_swizzle, 0, 0, self.panel_cnt, 32 * self.world, self.rowss[0], sms=self.sms)
            return
        cur = torch.cuda.current_stream()
        xb, xm = self.xbufs[parity], self.x_mcs[parity]
        self.epoch_dev.add_(1)
        self.rowss[parity].zero_()
        self.ev_start.record(cur)
        self.s_red.wait_event(self.ev_start)
        with torch.cuda.stream(self.s_red):
            e.tp4_reduce(self.M, self.rank, self.world, 0, 1, self.R, self.tile_cnt, self.partials_mc, xm, resid,
                         self.rowss_mc + parity * self.M * 4, self.panel_mc, version=3, epoch_dev=self.epoch_dev)
            self.ev_red.record(self.s_red)
        e.tp4_down(h_r, Wd, self.partials, 1, 1, 4, self.rank, self.world, 0, self.tile_mc, self.tile_cnt,
                   self.partials_mc, xm, None, self.rowss_mc + parity * self.M * 4, self.panel_mc,
                   sms=self.sms, epoch_dev=self.epoch_dev)
        e.tp4_qkv(xb, Wq, self.out, self.qkv_raster, self.qkv_swizzle, 2, 0, self.panel_cnt, 32 * self.world, self.rowss[parity],
                  sms=self.sms, epoch_dev=self.epoch_dev)
        cur.wait_event(self.ev_red)

    def forward(self, h_r, resid=None, Wd=None, Wq=None):
        """Wd (8192, Kr) / Wq (N2r, 8192, gamma folded): bf16 contiguous weights for this call (no copy; must stay alive
        until the stream work finishes); default = those from set_weights. Not supported with use_graph."""
        if self.use_graph:
            assert Wd is None and Wq is None, "per-call weights are not supported with use_graph"
        Wd = self.Wd if Wd is None else Wd
        Wq = self.Wq if Wq is None else Wq
        assert Wd is not None and Wq is not None, "no weights: call set_weights() or pass Wd/Wq"
        self._check_w(Wd, Wq)
        self.warmup()
        r = resid if self.use_resid else None
        if self.v5:
            return self._forward_v5(h_r, r, Wd, Wq)
        self.epoch += 1
        par = self.epoch & 1
        self._last_par = par
        if r is not None:
            assert r.data_ptr() != self.xbufs[par].data_ptr(), "resid aliases the x buffer written by this forward"
        if not self.use_graph:
            self._enqueue(h_r, r, par, Wd, Wq)
            return self.out
        self.h_static.copy_(h_r)
        if r is not None:
            self.resid_static.copy_(r)
        rs = self.resid_static if r is not None else None
        if not self._warm:               # un-captured warm-up; it is a real forward (device epoch advances with host epoch)
            self._warm = True
            self._enqueue(self.h_static, rs, par, Wd, Wq)
            return self.out
        g = self._graphs.get(par)
        if g is None:
            torch.cuda.synchronize(self.dev)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._enqueue(self.h_static, rs, par, Wd, Wq)
            self._graphs[par] = g
        g.replay()
        return self.out

    def _forward_v5(self, h_r, r, Wd, Wq):
        M = h_r.shape[0]
        assert M % 128 == 0 and 0 < M <= self.max_M, (M, self.max_M)
        assert h_r.shape == (M, self.Kr) and h_r.dtype == DTYPE and h_r.is_contiguous(), (h_r.shape, h_r.dtype)
        if r is not None:
            assert r.shape == (M, N1) and r.dtype == DTYPE and r.is_contiguous(), (r.shape, r.dtype)
        self.epoch += 1
        par = self.epoch & 1
        self._last_par, self._last_M = par, M
        self.out = self._out_full[:M]
        if r is not None:
            assert r.data_ptr() != self.xbufs[par].data_ptr(), "resid aliases the x buffer written by this forward"
        if not self.use_graph:
            self._enqueue_v5(h_r, r, par, Wd, Wq, M)
            return self.out
        if self.graph_bind:
            hs, rs = h_r, r
            key = (par, M, h_r.data_ptr(), r.data_ptr() if r is not None else 0)
        else:
            self.h_static[:M].copy_(h_r)
            if r is not None:
                self.resid_static[:M].copy_(r)
            hs = self.h_static[:M]
            rs = self.resid_static[:M] if r is not None else None
            key = (par, M)
        if not self._warm:               # un-captured warm-up; it is a real forward (device epoch advances with host epoch)
            self._warm = True
            self._enqueue_v5(hs, rs, par, Wd, Wq, M)
            return self.out
        g = self._graphs.get(key)
        if g is None:
            torch.cuda.synchronize(self.dev)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._enqueue_v5(hs, rs, par, Wd, Wq, M)
            self._graphs[key] = g
        g.replay()
        return self.out

    def x(self):
        """New residual (M, 8192) of the most recent forward; valid until two forwards later (x is double-buffered)."""
        if self.v5:
            return self.xbufs[self._last_par][:self._last_M]
        return self.xbufs[self._last_par]

    def trace_summary(self):
        """CTAPP_TRACE (v5, eager): stage stamps of the most recent forward in us relative to the first producer CTA entry (this
        GPU's %globaltimer). Synchronises the device."""
        torch.cuda.synchronize(self.dev)
        td, tq = self.trace_down.view(-1, 8).cpu(), self.trace_qkv.view(-1, 8).cpu()
        P = self._last_M // 128
        own = [m for m in range(P) if m % 4 == self.rank]
        tr = self.trace_red.cpu()[own]
        td, tq = td[td[:, 0] > 0], tq[tq[:, 0] > 0]
        t0 = int(td[:, 0].min())
        us = lambda v: (int(v) - t0) / 1e3  # noqa: E731
        d = {"prod_ctas": int(td.shape[0]), "prod_entry_max": us(td[:, 0].max()), "prod_prologue_max": us(td[:, 1].max()),
             "prod_end_min": us(td[:, 6].min()), "prod_end_max": us(td[:, 6].max()), "cons_ctas": int(tq.shape[0])}
        if tr.numel() and int(tr.min()) > 0:
            d["red_first_panel"], d["red_last_panel"] = us(tr.min()), us(tr.max())
        if tq.shape[0]:
            role = (tq[:, 7] >> 32) > 0
            nr = tq[~role]
            d["cons_entry_min"], d["cons_entry_max"] = us(tq[:, 0].min()), us(tq[:, 0].max())
            if nr.shape[0]:
                d["nonrole_entry_min"], d["nonrole_entry_med"] = us(nr[:, 0].min()), us(nr[:, 0].median())
                d["nonrole_first_tile_min"], d["nonrole_first_tile_med"] = us(nr[:, 2].min()), us(nr[:, 2].median())
                d["nonrole_end_max"] = us(nr[:, 6].max())
            if role.any():
                rr = tq[role]
                d["role_ctas"], d["role_sms"] = int(rr.shape[0]), sorted(int(v) & 0xffffffff for v in rr[:, 7])
                d["role_start_min"], d["role_red_end_max"] = us(rr[:, 4].min()), us(rr[:, 5].max())
                d["role_first_tile_min"], d["role_end_max"] = us(rr[:, 2].min()), us(rr[:, 6].max())
                d["prod_sms_free"] = sorted(set(range(self.nsm)) - {int(v) & 0xffffffff for v in td[:, 7]})
            red = tq[tq[:, 5] > 0]
            if red.shape[0] and self.steal:
                d["steal_ctas"], d["steal_red_end_max"] = int(red.shape[0]), us(red[:, 5].max())
            d["cons_last_tile_start_max"], d["cons_end_max"] = us(tq[:, 3].max()), us(tq[:, 6].max())
        return d


class CtappBoundaryPool:
    """One CtappBoundary per exact token count M, shared by all layers (weights are passed per forward call).

    ``get(M)`` returns None when M is unsupported (M % 128 != 0 or M < min_M): the caller uses its stock path. Otherwise it
    returns the cached instance for that M, creating it lazily. Creation is COLLECTIVE (symmetric-memory rendezvous +
    barrier): every rank must call ``get`` with the same M in the same order. Instances share nothing and are never evicted;
    requesting more than ``max_instances`` distinct M raises RuntimeError. Memory per instance: 2*M*8192*2 B symmetric
    (partials + xbuf) + M*N2r*2 B (out) + small counters. warmup() runs on creation.

    protocol="v5": ONE instance sized ``max_M`` (created on the first valid ``get``, also collective) serves every
    M <= max_M; ``get`` returns None for M > max_M as well. ``v5kw``: extra CtappBoundary kwargs for that instance (e.g.
    fusion="role", qkv_tile=128).
    """

    def __init__(self, Kr, N2r, rank, world, group_name, device, R=None, min_M=512, max_instances=4,
                 qkv_raster=2, qkv_swizzle=1, protocol="v3", max_M=8192, **v5kw):
        assert protocol in ("v3", "v5"), protocol
        self.Kr, self.N2r, self.rank, self.world, self.gn, self.dev = Kr, N2r, rank, world, group_name, device
        self.R, self.min_M, self.max_instances = R, min_M, max_instances
        self.qkv_raster, self.qkv_swizzle = qkv_raster, qkv_swizzle
        self.protocol, self.max_M = protocol, max_M
        assert protocol == "v5" or not v5kw, "v5kw needs protocol v5"
        self.v5kw = v5kw
        self._inst = {}

    def get(self, M):
        if M % 128 != 0 or M < self.min_M:
            return None
        if self.protocol == "v5":
            if M > self.max_M:
                return None
            bd = self._inst.get(self.max_M)
            if bd is None:
                bd = CtappBoundary(self.max_M, self.Kr, self.N2r, self.rank, self.world, self.gn, self.dev, R=self.R,
                                  qkv_raster=self.qkv_raster, qkv_swizzle=self.qkv_swizzle, protocol="v5", max_M=self.max_M, **self.v5kw)
                bd.warmup()
                self._inst[self.max_M] = bd
            return bd
        bd = self._inst.get(M)
        if bd is None:
            if len(self._inst) >= self.max_instances:
                raise RuntimeError(f"CtappBoundaryPool: more than {self.max_instances} distinct M requested (have "
                                   f"{sorted(self._inst)}, asked {M})")
            bd = CtappBoundary(M, self.Kr, self.N2r, self.rank, self.world, self.gn, self.dev, R=self.R,
                              qkv_raster=self.qkv_raster, qkv_swizzle=self.qkv_swizzle)
            bd.warmup()
            self._inst[M] = bd
        return bd


# --------------------------------------------------------------------------- baselines
class _Base:
    def __init__(self, M, rank, world, group_name, device, Wdown_r, Wqkv_r, gamma):
        self.M, self.rank, self.world, self.gn, self.dev = M, rank, world, group_name, device
        self.Wd, self.Wq, self.gamma = Wdown_r, Wqkv_r, gamma.to(DTYPE)
        self._x = None

    def x(self):
        return self._x

    def _post(self, y, resid):
        x = y + resid if resid is not None else y
        self._x = x
        return torch.mm(rmsnorm_gamma(x, self.gamma), self.Wq.t())


class MultimemBoundary(_Base):
    def __init__(self, *a):
        super().__init__(*a)
        self.y, _h = _symm((self.M, N1), DTYPE, self.dev, self.gn)
        self._h = _h

    def forward(self, h_r, resid=None):
        torch.mm(h_r, self.Wd.t(), out=self.y)
        torch.ops.symm_mem.multimem_all_reduce_(self.y, "sum", self.gn)
        return self._post(self.y, resid)


class FlashInferBoundary(_Base):
    """FlashInfer trtllm_allreduce_fusion (vLLM/TRT-LLM production path): one kernel does all-reduce + residual add
    + RMSNorm and writes the new residual. Reduction is NOT overlapped with the GEMM."""

    def __init__(self, *a, use_oneshot=None):
        super().__init__(*a)
        from flashinfer.comm import trtllm_create_ipc_workspace_for_all_reduce_fusion as create
        self.use_oneshot = use_oneshot
        self.ipc, self.ws = create(self.rank, self.world, self.M, N1, group=dist.group.WORLD)
        self.y = torch.empty(self.M, N1, dtype=DTYPE, device=self.dev)
        self.resid_out = torch.empty(self.M, N1, dtype=DTYPE, device=self.dev)
        self.norm_out = torch.empty(self.M, N1, dtype=DTYPE, device=self.dev)
        self.zeros = None
        self.gamma_bf16 = self.gamma.contiguous()

    def forward(self, h_r, resid=None):
        from flashinfer.comm import trtllm_allreduce_fusion, AllReduceFusionPattern
        if resid is None:
            if self.zeros is None:
                self.zeros = torch.zeros(self.M, N1, dtype=DTYPE, device=self.dev)
            resid = self.zeros
        torch.mm(h_r, self.Wd.t(), out=self.y)
        trtllm_allreduce_fusion(
            allreduce_in=self.y, world_size=self.world, world_rank=self.rank, token_num=self.M, hidden_dim=N1,
            workspace_ptrs=self.ws, launch_with_pdl=False, trigger_completion_at_end=True, fp32_acc=True,
            pattern_code=AllReduceFusionPattern.kARResidualRMSNorm, use_oneshot=self.use_oneshot,
            allreduce_out=None, residual_in=resid, residual_out=self.resid_out, norm_out=self.norm_out,
            quant_out=None, scale_out=None, rms_gamma=self.gamma_bf16, rms_eps=EPS, scale_factor=None, layout_code=None)
        self._x = self.resid_out
        return torch.mm(self.norm_out, self.Wq.t())

    def destroy(self):
        from flashinfer.comm import trtllm_destroy_ipc_workspace_for_all_reduce_fusion as destroy
        torch.cuda.synchronize(self.dev)
        dist.barrier()
        destroy(self.ipc)


class AsyncTpBoundary(_Base):
    def __init__(self, *a):
        super().__init__(*a)
        import torch.distributed._symmetric_memory as symm_mem
        symm_mem.enable_symm_mem_for_group(self.gn)
        self.full = torch.empty(self.M, N1, dtype=DTYPE, device=self.dev)

    def forward(self, h_r, resid=None):
        from torch.distributed._symmetric_memory import _fused_matmul_reduce_scatter
        m = self.M // self.world
        ys = _fused_matmul_reduce_scatter(h_r, self.Wd.t(), "sum", 0, self.gn)
        if resid is not None:
            ys = ys + resid[self.rank * m:(self.rank + 1) * m]
        self._x = ys  # own rows only
        dist.all_gather_into_tensor(self.full, rmsnorm_gamma(ys, self.gamma))
        return torch.mm(self.full, self.Wq.t())


class NcclBoundary(_Base):
    def forward(self, h_r, resid=None):
        y = torch.mm(h_r, self.Wd.t())
        dist.all_reduce(y)
        return self._post(y, resid)


# --------------------------------------------------------------------------- references
@torch.no_grad()
def one_gpu_reference(h_full, Wdown_full, resid, gamma, Wqkv_full):
    """fp32 chain -> (x_new fp32 (M, 8192), out_full fp32 (M, 10240))."""
    x = h_full.float() @ Wdown_full.float().t()
    if resid is not None:
        x = x + resid.float()
    n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * gamma.float()
    return x, n @ Wqkv_full.float().t()


@torch.no_grad()
def one_gpu_chain_bf16(h_full, Wdown_full, resid, gamma, Wqkv_full):
    """Eager bf16 chain (the ideal-timing 1-GPU time); returns (x, out)."""
    x = torch.mm(h_full, Wdown_full.t())
    if resid is not None:
        x = x + resid
    return x, torch.mm(rmsnorm_gamma(x, gamma), Wqkv_full.t())


def make_inputs(M, dev, seed=1, K=K_FFN):
    """Deterministic (same on every rank, same GPU arch) full-size inputs. K: full down-proj K (default the FFN's 28672; the
    o_proj boundary "A" uses 8192)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    r = lambda *s, sc=1.0: torch.randn(*s, device=dev, generator=g) * sc  # noqa: E731
    h = r(M, K, sc=1.0).to(DTYPE)
    resid = r(M, N1).to(DTYPE)
    return h, resid


def make_weights(dev, seed=0, K=K_FFN, N2_=N2):
    """Full weights: Wdown (8192, K), gamma (8192), Wqkv (N2_, 8192). Boundary "A" (o_proj -> RMSNorm -> gate_up): K=8192,
    N2_=57344."""
    g = torch.Generator(device=dev).manual_seed(seed)
    Wd = (torch.randn(N1, K, device=dev, generator=g) * K ** -0.5).to(DTYPE)
    gamma = (1 + 0.1 * torch.randn(N1, device=dev, generator=g)).to(DTYPE)
    Wq = (torch.randn(N2_, N1, device=dev, generator=g) * N1 ** -0.5).to(DTYPE)
    return Wd, gamma, Wq


# --------------------------------------------------------------------------- process harness + timing
GLOO = None


def barrier():
    dist.barrier(group=GLOO)


def time_step(step, dev, iters, warmup):
    """(per-step median ms with barrier+sync before each call, back-to-back mean ms), both max over ranks."""
    for _ in range(warmup):
        step()
    torch.cuda.synchronize(dev)
    barrier()
    times = []
    for _ in range(iters):
        barrier()
        torch.cuda.synchronize(dev)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        step()
        b.record()
        torch.cuda.synchronize(dev)
        times.append(a.elapsed_time(b))
    ps = statistics.median(times)
    barrier()
    torch.cuda.synchronize(dev)
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        step()
    b.record()
    torch.cuda.synchronize(dev)
    t = torch.tensor([ps, a.elapsed_time(b) / iters], dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=GLOO)
    return t.tolist()


def time_one_gpu(fn, dev, iters, warmup):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize(dev)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize(dev)
        times.append(a.elapsed_time(b))
    return statistics.median(times)


def _entry(rank, world, port, fn, args, queue):
    global GLOO
    os.environ.setdefault("TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES", "0")
    os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    code = 0
    try:
        dev = torch.device("cuda", rank)
        torch.cuda.set_device(dev)
        dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world, device_id=dev)
        GLOO = dist.new_group(backend="gloo")
        res = fn(rank, world, dev, dist.group.WORLD.group_name, *args)
        torch.cuda.synchronize(dev)
        barrier()
        queue.put((rank, "ok", res))
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        queue.put((rank, "err", traceback.format_exc()))
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    queue.close()
    queue.join_thread()
    os._exit(code)  # skip NCCL / symm-mem teardown (can hang)


def launch(fn, world, port, *args):
    """Run fn(rank, world, dev, group_name, *args) in `world` processes; returns list of per-rank results (None on error)."""
    import queue as _q
    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_entry, args=(r, world, port, fn, args, q)) for r in range(world)]
    for p in procs:
        p.start()
    results, got = [None] * world, 0
    while got < world:
        try:
            r, st, res = q.get(timeout=5)
            results[r] = res if st == "ok" else None
            got += 1
        except _q.Empty:
            if all(not p.is_alive() for p in procs):
                break
    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.kill()
    bad = got < world or any(r is None for r in results)
    return results, bad
