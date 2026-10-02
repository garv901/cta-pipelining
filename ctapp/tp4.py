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
    """

    def __init__(self, M, Kr, N2r, rank, world, group_name, device, resid=True, protocol=True, use_graph=False, R=None):
        from ctapp.ext import load_tp4
        assert N1 == 8192 and M % 128 == 0, (N1, M)
        self.ext = load_tp4()
        self.M, self.Kr, self.N2r, self.rank, self.world = M, Kr, N2r, rank, world
        self.dev, self.use_resid, self.protocol, self.use_graph = device, resid, protocol, use_graph
        self.R = R or (12 if M <= 2048 else 8)
        self.sms = torch.cuda.get_device_properties(device).multi_processor_count - self.R
        gn = group_name
        self.partials, h1 = _symm((M, N1), DTYPE, device, gn)
        self.xbuf, h2 = _symm((M, N1), DTYPE, device, gn)
        self.tile_cnt, h3 = _symm((M // 128 * (N1 // 256),), torch.int32, device, gn)
        self.panel_cnt, h4 = _symm((M // 128,), torch.int32, device, gn)
        self.rowss, h5 = _symm((2, M), torch.float32, device, gn)
        for h, t in ((h1, self.partials), (h2, self.xbuf), (h3, self.tile_cnt), (h4, self.panel_cnt), (h5, self.rowss)):
            assert t.data_ptr() == h.buffer_ptrs[rank]
        self.partials_mc, self.x_mc, self.tile_mc = h1.multicast_ptr, h2.multicast_ptr, h3.multicast_ptr
        self.panel_mc, self.rowss_mc = h4.multicast_ptr, h5.multicast_ptr
        self._hdls = (h1, h2, h3, h4, h5)
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

    def set_weights(self, Wdown_r, Wqkv_r_folded):
        assert Wdown_r.shape == (N1, self.Kr) and Wqkv_r_folded.shape == (self.N2r, N1)
        self.Wd, self.Wq = Wdown_r.contiguous(), Wqkv_r_folded.contiguous()
        # Force lazy CUDA module loading of both GEMM kernels now. If the first launch of a GEMM happens while the (spinning)
        # reducer kernel already occupies SMs, the run faults (spin-limit trap) on every rank.
        e = self.ext
        hz = torch.zeros(self.M, self.Kr, dtype=DTYPE, device=self.dev)
        e.tp4_down(hz, self.Wd, self.partials, 1, 1, 0, self.rank, self.world, 0, self.tile_mc, self.tile_cnt,
                   self.partials_mc, self.x_mc, None, self.rowss_mc, self.panel_mc, sms=self.sms)
        e.tp4_qkv(self.xbuf, self.Wq, self.out, 2, 1, 0, 0, self.panel_cnt, 32 * self.world, self.rowss[0], sms=self.sms)
        torch.cuda.synchronize(self.dev)

    def _enqueue(self, h_r, resid, parity):
        e = self.ext
        if not self.protocol:
            e.tp4_down(h_r, self.Wd, self.partials, 1, 1, 0, self.rank, self.world, 0, self.tile_mc, self.tile_cnt,
                       self.partials_mc, self.x_mc, None, self.rowss_mc, self.panel_mc, sms=self.sms)
            e.tp4_qkv(self.xbuf, self.Wq, self.out, 2, 1, 0, 0, self.panel_cnt, 32 * self.world, self.rowss[0], sms=self.sms)
            return
        cur = torch.cuda.current_stream()
        self.epoch_dev.add_(1)
        self.rowss[parity].zero_()
        self.ev_start.record(cur)
        self.s_red.wait_event(self.ev_start)
        with torch.cuda.stream(self.s_red):
            e.tp4_reduce(self.M, self.rank, self.world, 0, 1, self.R, self.tile_cnt, self.partials_mc, self.x_mc, resid,
                         self.rowss_mc + parity * self.M * 4, self.panel_mc, version=3, epoch_dev=self.epoch_dev)
            self.ev_red.record(self.s_red)
        e.tp4_down(h_r, self.Wd, self.partials, 1, 1, 4, self.rank, self.world, 0, self.tile_mc, self.tile_cnt,
                   self.partials_mc, self.x_mc, None, self.rowss_mc + parity * self.M * 4, self.panel_mc,
                   sms=self.sms, epoch_dev=self.epoch_dev)
        e.tp4_qkv(self.xbuf, self.Wq, self.out, 2, 1, 2, 0, self.panel_cnt, 32 * self.world, self.rowss[parity],
                  sms=self.sms, epoch_dev=self.epoch_dev)
        cur.wait_event(self.ev_red)

    def forward(self, h_r, resid=None):
        r = resid if self.use_resid else None
        self.epoch += 1
        par = self.epoch & 1
        if not self.use_graph:
            self._enqueue(h_r, r, par)
            return self.out
        self.h_static.copy_(h_r)
        if r is not None:
            self.resid_static.copy_(r)
        rs = self.resid_static if r is not None else None
        if not self._warm:               # un-captured warm-up; it is a real forward (device epoch advances with host epoch)
            self._warm = True
            self._enqueue(self.h_static, rs, par)
            return self.out
        g = self._graphs.get(par)
        if g is None:
            torch.cuda.synchronize(self.dev)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._enqueue(self.h_static, rs, par)
            self._graphs[par] = g
        g.replay()
        return self.out

    def x(self):
        return self.xbuf


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


def make_inputs(M, dev, seed=1):
    """Deterministic (same on every rank, same GPU arch) full-size inputs."""
    g = torch.Generator(device=dev).manual_seed(seed)
    r = lambda *s, sc=1.0: torch.randn(*s, device=dev, generator=g) * sc  # noqa: E731
    h = r(M, K_FFN, sc=1.0).to(DTYPE)
    resid = r(M, N1).to(DTYPE)
    return h, resid


def make_weights(dev, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    Wd = (torch.randn(N1, K_FFN, device=dev, generator=g) * K_FFN ** -0.5).to(DTYPE)
    gamma = (1 + 0.1 * torch.randn(N1, device=dev, generator=g)).to(DTYPE)
    Wq = (torch.randn(N2, N1, device=dev, generator=g) * N1 ** -0.5).to(DTYPE)
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
