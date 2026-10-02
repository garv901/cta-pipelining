"""Phase-6 S1 go/no-go probes (TP4, one process per GPU): owner-side unicast reducer throughput, interference with the producer
GEMM, remote-epilogue producer GEMM, flag-chain ordering/latency, PDL early launch.

    python -u bench/push_bw.py [--probes 1,2,3,4,5] [--iters 20] [--warmup 5] [--json build/p6_s1.json]

Probe kernels live in csrc/gemm_tp4.cu (p6_*). Every timing: `warmup` untimed calls, then `iters` timed calls each preceded by
barrier + device sync; reported value = median over calls, max over ranks. Results are merged into the JSON file (per probe key)
and printed as markdown tables by rank 0.

1  p6_reduce_probe: per rank, M x 8192 bf16 payload in 128-row panels owned by panel % 4 == rank; per unit (owned panel, 256-col
   n-tile, 32-row quarter): 4 local inbox loads + resid load, fp32 add, st.relaxed.sys.v4 to local x + 3 peers' x, red.add.f32 row
   sum of squares; signaller warp: fence.acq_rel.sys + value flag per T units. Counted bytes (full): 4 x M/4 x 16 KB read +
   M/4 x 16 KB local write + 3 x M/4 x 16 KB remote write = 32 KB x M (128 MiB at M=4096; resid reads, 16 MiB, not counted).
   placement "spread": no other kernel, CTAs land on `blocks` SMs. "packed": a blocker kernel occupies 132 - blocks/2 SMs first
   (one CTA/SM via 200 KB smem; probe CTAs carry a 32 KB smem pad so they cannot co-reside) -> 2 probe CTAs per free SM.
2  interference: probe 1 (threads/T/blocks 544/2/4, 544/2/8, 384/2/4, 384/4/4) on a second stream behind a 320 us delay kernel while tp4_down mode 0 runs on the
   current stream behind a 300 us delay (GEMM's 128 CTAs resident first; probe CTAs on the 4 free SMs), tp4_down mode 0 (M=4096, K=7168, N=8192, raster 1, swizzle 1, sms=128) runs on the current stream.
3  tp4_down mode 0 with D local vs D = right neighbour's symmetric buffer (d_ptr), all 4 ranks concurrently.
4  p6_chain_stress: two-hop chain (A remote st + fence.sys + local gpu atomic -> B acquire.gpu + fence.sys + remote flag ->
   checker acquire.sys + verify 16 KB) and one-hop (A writes data + fence + flag), with no-fence negative controls.
5  PDL: primary (grid 128, 384 thr, 214016 B smem, griddepcontrol.launch_dependents at start, spins 500 us) + dependent (grid 132,
   programmatic stream serialization, stamps %globaltimer/%smid at start), eager and CUDA-graph captured. Rank 0 only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from ctapp import tp4  # noqa: E402
from ctapp.tp4 import _symm  # noqa: E402

BF = torch.bfloat16
NSM = 132
BLOCKER_SMEM = 200 * 1024
PAD_SMEM = 32 * 1024
K_DOWN = tp4.K_FFN // 4     # 7168
N1 = tp4.N1                 # 8192


def mx(v):
    t = torch.tensor([float(v)], dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=tp4.GLOO)
    return t.item()


def sm_(v):
    t = torch.tensor([float(v)], dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM, group=tp4.GLOO)
    return t.item()


def timed(step, dev, iters, warmup, after=None):
    """step() enqueues and returns (ev_a, ev_b); per-call median ms (barrier + sync before each call), max over ranks.
    after(): called after each timed call's sync (e.g. to read device stamps); its return values are collected."""
    for _ in range(warmup):
        step()
    torch.cuda.synchronize(dev)
    tp4.barrier()
    ts, extra = [], []
    for _ in range(iters):
        tp4.barrier()
        torch.cuda.synchronize(dev)
        a, b = step()
        torch.cuda.synchronize(dev)
        ts.append(a.elapsed_time(b))
        if after is not None:
            extra.append(after())
    return mx(statistics.median(ts)), extra


def ev():
    return torch.cuda.Event(enable_timing=True)


# ----------------------------------------------------------------------------------------------------------------- probe 1
class Reducer:
    def __init__(self, ext, rank, dev, gn, M):
        self.ext, self.rank, self.dev, self.M = ext, rank, dev, M
        g = torch.Generator(device=dev).manual_seed(100 + rank)
        self.inbox = (torch.randn(4, M // 4, N1, device=dev, generator=g) * 0.25).to(BF)
        g2 = torch.Generator(device=dev).manual_seed(99)      # resid is replicated: same on every rank
        self.resid = torch.randn(M, N1, device=dev, generator=g2).to(BF)
        self.x, self.hx = _symm((M, N1), BF, dev, gn)
        self.x.zero_()
        self.x_ptrs = [int(self.hx.buffer_ptrs[r]) for r in range(4)]
        self.rowss = torch.zeros(M, dtype=torch.float32, device=dev)
        self.flags = torch.zeros(M, dtype=torch.int32, device=dev)
        self.ctl = torch.zeros(4, dtype=torch.int32, device=dev)   # [0] done, [1] release, [2] blocker started
        self.smid = torch.zeros(64, dtype=torch.int32, device=dev)
        self.stamps = torch.zeros(128, dtype=torch.int64, device=dev)
        self.s2 = torch.cuda.Stream(device=dev)
        self.epoch = self.done = self.started = 0
        torch.cuda.synchronize(dev)

    def bytes(self, no_remote):
        return self.M * (16384 + 4096 + (0 if no_remote else 12288))

    def launch(self, blocks, threads, T, no_remote=0, no_fence=0, smem_pad=0):
        self.epoch += 1
        self.done += blocks
        self.ext.p6_reduce_probe(self.inbox, self.resid, self.x_ptrs, self.rowss, self.flags, self.ctl, self.smid, self.stamps,
                                 self.rank, self.M, blocks, threads, T, no_remote, no_fence, self.epoch, self.done, smem_pad)

    def step(self, blocks, threads, T, no_remote, no_fence, free=None):
        """free=None: plain launch (CTAs spread over `blocks` SMs). free=F: a blocker first occupies NSM - F SMs, so the probe runs
        on F SMs (needs blocks <= 2 F: two CTAs per SM)."""
        cur = torch.cuda.current_stream()
        a, b = ev(), ev()
        if free is None:
            a.record(cur)
            self.launch(blocks, threads, T, no_remote, no_fence, 0)
            b.record(cur)
            return a, b
        assert blocks <= 2 * free
        grid_b = NSM - free
        e0, eb = torch.cuda.Event(), torch.cuda.Event()
        e0.record(cur)
        self.s2.wait_event(e0)
        with torch.cuda.stream(self.s2):
            self.ext.p6_blocker(self.ctl, self.epoch + 1, grid_b, BLOCKER_SMEM)   # released by the probe's last CTA (flag = epoch)
            eb.record(self.s2)
        self.started += grid_b
        self.ext.p6_wait(self.ctl, self.started)
        a.record(cur)
        self.launch(blocks, threads, T, no_remote, no_fence, PAD_SMEM)
        b.record(cur)
        cur.wait_event(eb)
        return a, b

    def span(self, blocks):
        s = self.stamps[:2 * blocks].view(blocks, 2).cpu()
        return (s[:, 1].max() - s[:, 0].min()).item() / 1e6, len(set(self.smid[:blocks].cpu().tolist()))

    def check(self, blocks, threads, T):
        """One full run from zeroed x / rowss; x must equal the 4-owner reduction on every rank, rowss the owned rows' sum of squares."""
        self.x.zero_(); self.rowss.zero_()
        torch.cuda.synchronize(self.dev); tp4.barrier()
        self.launch(blocks, threads, T)
        torch.cuda.synchronize(self.dev); tp4.barrier()
        M = self.M
        exp = torch.empty(M, N1, dtype=BF, device=self.dev)
        ev_ = exp.view(M // 128, 128, N1)
        rs = self.resid.view(M // 128, 128, N1)
        for o in range(4):
            g = torch.Generator(device=self.dev).manual_seed(100 + o)
            ib = (torch.randn(4, M // 4, N1, device=self.dev, generator=g) * 0.25).to(BF)
            ev_[o::4] = (ib.float().sum(0).view(M // 512, 128, N1) + rs[o::4].float()).to(BF)
            del ib
        d = (self.x.float() - exp.float()).abs()
        bad = int((d > 1e-2 * exp.float().abs().clamp_min(1.0)).sum())
        own = self.x.view(M // 128, 128, N1)[self.rank::4].float()
        ss_ref = own.pow(2).sum(-1).flatten()
        ss = self.rowss.view(M // 128, 128)[self.rank::4].flatten()
        ss_err = ((ss - ss_ref).abs() / ss_ref).max().item()
        del exp
        return int(sm_(bad)), mx(ss_err)


def probe1(ext, rank, dev, gn, iters, warmup, quick):
    out = {"rows": [], "check": []}
    for M in (4096, 1024):
        R = Reducer(ext, rank, dev, gn, M)
        for th in (544, 384):
            R.launch(8, th, 2)                    # warm (module load) before any timed / blocker use
        torch.cuda.synchronize(dev)
        if M == 4096:
            for th in (544, 384):
                bad, sserr = R.check(8, th, 2)
                out["check"].append({"M": M, "threads": th, "blocks": 8, "T": 2, "x_mismatch_all_ranks": bad, "rowss_max_rel_err": sserr})
                if rank == 0:
                    print(f"[p1 check] M={M} threads={th}: x mismatches (sum over ranks) {bad}, rowss max rel err {sserr:.2e}", flush=True)
        if M == 4096:
            cfgs = [(pl, th, T, b, var) for pl in ("spread", "packed") for th in (544, 384) for T in (2, 4) for b in (2, 4, 8, 16)
                    for var in ("full", "no_fence", "no_remote", "no_remote_no_fence")]
            if quick:
                cfgs = [c for c in cfgs if c[4] == "full" and c[2] == 2]
        else:
            cfgs = [(pl, th, 2, b, "full") for pl in ("spread", "packed") for th in (544, 384) for b in (4, 8)]
        for pl, th, T, b, var in cfgs:
            nr, nf = int("no_remote" in var), int("no_fence" in var)
            spans, sms = [], []

            def after():
                s, n = R.span(b)
                return s, n
            ms, extra = timed(lambda: R.step(b, th, T, nr, nf, (b + 1) // 2 if pl == "packed" else None), dev, iters, warmup, after)
            span = mx(statistics.median(e[0] for e in extra))
            nsm = mx(max(e[1] for e in extra))
            b2b = float("nan")
            if pl == "spread":
                _, b2b = tp4.time_step(lambda: R.launch(b, th, T, nr, nf, 0), dev, iters, warmup)
            gbs = R.bytes(nr) / (ms * 1e6)
            row = {"M": M, "placement": pl, "threads": th, "T": T, "blocks": b, "variant": var, "ms": ms, "kernel_span_ms": span,
                   "b2b_ms": b2b, "GBps": gbs, "GBps_span": R.bytes(nr) / (span * 1e6), "distinct_SMs": nsm, "bytes": R.bytes(nr)}
            out["rows"].append(row)
            if rank == 0:
                print(f"[p1] M={M} {pl:6s} thr={th} T={T} blocks={b:2d} {var:19s} {ms:7.3f} ms (span {span:7.3f}, b2b {b2b:7.3f}) "
                      f"{gbs:6.1f} GB/s  SMs={nsm:.0f}", flush=True)
        del R
        torch.cuda.empty_cache()
    return out


# ----------------------------------------------------------------------------------------------------------------- probe 2
def down_inputs(dev, M, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    A = torch.randn(M, K_DOWN, device=dev, generator=g).to(BF)
    gw = torch.Generator(device=dev).manual_seed(7)
    W = (torch.randn(N1, K_DOWN, device=dev, generator=gw) * K_DOWN ** -0.5).to(BF)
    return A, W


def probe2(ext, rank, dev, gn, iters, warmup):
    out = {"rows": []}
    M = 4096
    A, W = down_inputs(dev, M, 300 + rank)
    D = torch.empty(M, N1, dtype=BF, device=dev)
    dummy = torch.zeros(1, dtype=torch.int32, device=dev)

    def gemm(sms):
        ext.tp4_down(A, W, D, 1, 1, 0, rank, 4, 0, 0, dummy, 0, 0, None, 0, 0, sms=sms)
    alone = {}
    for sms in (128, 132):
        alone[sms] = tp4.time_step(lambda: gemm(sms), dev, iters, warmup)
        if rank == 0:
            print(f"[p2] GEMM alone sms={sms}: per-step {alone[sms][0]:.3f} b2b {alone[sms][1]:.3f} ms", flush=True)
    out["gemm_alone"] = {str(k): {"per_step": v[0], "b2b": v[1]} for k, v in alone.items()}
    R = Reducer(ext, rank, dev, gn, M)
    for th_ in (544, 384):
        for T_ in (2, 4):
            R.launch(8, th_, T_)
    torch.cuda.synchronize(dev)
    s2 = torch.cuda.Stream(device=dev)
    st = torch.zeros(4, dtype=torch.int64, device=dev)
    # (threads, T, blocks): 544 x {4, 8} = 3-kernel form (two reducer CTAs fit per free SM); 384 x 4 = D2 role-CTA form (one 384-thread
    # CTA per free SM, like a consumer-GEMM CTA)
    for th, T, b in ((544, 2, 4), (544, 2, 8), (384, 2, 4), (384, 4, 4)):
        probe_alone = timed(lambda: R.step(b, th, T, 0, 0, None), dev, iters, warmup)[0]          # b SMs, nothing else running
        probe_packed = timed(lambda: R.step(b, th, T, 0, 0, 4), dev, iters, warmup)[0]            # on the same 4 free SMs as with the GEMM
        rec = {}

        def step(stamp=False):
            cur = torch.cuda.current_stream()
            e00, e0, eg, ep, ee = ev(), ev(), ev(), ev(), ev()
            # Both streams start with a delay kernel so host enqueue lag (~60 us for the GEMM) does not decide the order: GEMM starts
            # at +300 us, probe at +320 us (GEMM's 128 CTAs resident first, probe CTAs land on the 4 free SMs).
            e00.record(cur)
            ext.p6_delay(300000)
            s2.wait_event(e00)
            with torch.cuda.stream(s2):
                ext.p6_delay(320000)
                R.launch(b, th, T)
                ep.record(s2)
            e0.record(cur)
            if stamp:
                ext.p6_stamp(st, 0)
            gemm(128)
            eg.record(cur)
            if stamp:
                ext.p6_stamp(st, 1)
            cur.wait_event(ep)
            ee.record(cur)
            rec["e"] = (e0, eg, ep, ee)
            return e0, ee

        def after():
            e0, eg, ep, ee = rec["e"]
            sp, nsm = R.span(b)
            return e0.elapsed_time(eg), e0.elapsed_time(ep), sp, nsm
        comb, extra = timed(step, dev, iters, warmup, after)
        t_g = mx(statistics.median(e[0] for e in extra))
        t_p = mx(statistics.median(e[1] for e in extra))
        span = mx(statistics.median(e[2] for e in extra))
        nsm = mx(max(e[3] for e in extra))
        n_spill = int(sm_(sum(1 for e in extra if e[3] > 4)))   # calls (all ranks) where the probe was not confined to 4 SMs
        # overlap verification (one stamped call): GEMM interval from stamp kernels around it, probe interval from its CTA stamps
        torch.cuda.synchronize(dev); tp4.barrier()
        step(stamp=True)
        torch.cuda.synchronize(dev)
        g0, g1 = st[0].item(), st[1].item()
        ps = R.stamps[:2 * b].view(b, 2).cpu()
        p0, p1 = ps[:, 0].min().item(), ps[:, 1].max().item()
        ov = max(0, min(g1, p1) - max(g0, p0)) / max(1, p1 - p0)
        ov = mx(-ov) * -1   # min over ranks
        slow = t_g / alone[128][0] - 1
        row = {"blocks": b, "threads": th, "T": T, "gemm_with_probe_ms": t_g, "probe_stream_ms": t_p, "probe_span_ms": span,
               "combined_ms": comb, "probe_alone_spread_ms": probe_alone, "probe_alone_packed_ms": probe_packed,
               "gemm_alone_128_ms": alone[128][0], "gemm_alone_132_ms": alone[132][0], "gemm_slowdown": slow,
               "probe_distinct_SMs": nsm, "calls_probe_on_more_than_4_SMs_all_ranks": n_spill, "overlap_frac_of_probe_min_ranks": ov,
               "stamp_rank0_us": {"gemm": [0, (g1 - g0) / 1e3], "probe": [(p0 - g0) / 1e3, (p1 - g0) / 1e3]}}
        out["rows"].append(row)
        if rank == 0:
            print(f"[p2] threads={th} T={T} blocks={b}: GEMM {t_g:.3f} ms (alone128 {alone[128][0]:.3f}, slowdown {slow * 100:+.1f} %), probe end-GEMM start {t_p:.3f} "
                  f"(span {span:.3f}, alone spread {probe_alone:.3f} packed {probe_packed:.3f}), combined {comb:.3f}; probe SMs {nsm:.0f} (calls on >4 SMs: {n_spill}/{4 * iters}); "
                  f"overlap {ov * 100:.0f} % of probe; rank0 stamps GEMM [0, {(g1 - g0) / 1e3:.0f}] us probe [{(p0 - g0) / 1e3:.0f}, "
                  f"{(p1 - g0) / 1e3:.0f}] us", flush=True)
    del R
    torch.cuda.empty_cache()
    return out


# ----------------------------------------------------------------------------------------------------------------- probe 3
def probe3(ext, rank, dev, gn, iters, warmup):
    out = {"rows": []}
    Dsym, hd = _symm((4096, N1), BF, dev, gn)
    right, left = (rank + 1) % 4, (rank + 3) % 4
    peer_ptr = int(hd.buffer_ptrs[right])
    dummy = torch.zeros(1, dtype=torch.int32, device=dev)
    for M in (1024, 4096):
        A, W = down_inputs(dev, M, 300 + rank)
        Al, _ = down_inputs(dev, M, 300 + left)
        ref_left = torch.mm(Al, W.t())
        del Al
        Dv = Dsym[:M]
        for sms in (128, 132):
            for swz in (1, 2):
                res = {}
                for where in ("local", "peer"):
                    dp = 0 if where == "local" else peer_ptr

                    def fn():
                        ext.tp4_down(A, W, Dv, 1, swz, 0, rank, 4, 0, 0, dummy, 0, 0, None, 0, 0, sms=sms, d_ptr=dp)
                    Dsym.zero_(); torch.cuda.synchronize(dev); tp4.barrier()
                    res[where] = tp4.time_step(fn, dev, iters, warmup)
                    torch.cuda.synchronize(dev); tp4.barrier()
                    if where == "peer":   # my buffer now holds the left neighbour's result
                        err = ((Dv.float() - ref_left.float()).abs().max() / ref_left.float().abs().max()).item()
                    else:
                        r = torch.mm(A, W.t())
                        err = ((Dv.float() - r.float()).abs().max() / r.float().abs().max()).item()
                    res[where + "_err"] = mx(err)
                row = {"M": M, "sms": sms, "raster": 1, "swizzle": swz,
                       "local_ms": res["local"][0], "peer_ms": res["peer"][0], "ratio": res["peer"][0] / res["local"][0],
                       "local_b2b": res["local"][1], "peer_b2b": res["peer"][1], "ratio_b2b": res["peer"][1] / res["local"][1],
                       "local_err": res["local_err"], "peer_err": res["peer_err"]}
                out["rows"].append(row)
                if rank == 0:
                    print(f"[p3] M={M} sms={sms} r1 s{swz}: local {row['local_ms']:.3f} peer {row['peer_ms']:.3f} ratio {row['ratio']:.3f} | "
                          f"b2b {row['local_b2b']:.3f} / {row['peer_b2b']:.3f} = {row['ratio_b2b']:.3f} | err {row['local_err']:.1e} / "
                          f"{row['peer_err']:.1e}", flush=True)
        del A, W, ref_left
        torch.cuda.empty_cache()
    return out


# ----------------------------------------------------------------------------------------------------------------- probe 4
def probe4(ext, rank, dev, gn, iters_chain):
    out = {"rows": []}
    S = 16
    NBIG = 16384                                   # 256 KB per slot variant (makes the A-side store drain long)
    data, hd = _symm((S * NBIG * 4,), torch.int32, dev, gn)
    flag, hf = _symm((S,), torch.int32, dev, gn)
    ack, ha = _symm((S,), torch.int32, dev, gn)
    counter = torch.zeros(S, dtype=torch.int32, device=dev)
    right, left = (rank + 1) % 4, (rank + 3) % 4
    names = {0: "two-hop chain (A fence.sys + gpu atomic -> B fence.sys + flag)", 1: "one-hop (A fence.sys + flag)",
             2: "two-hop, NO fence in A (control)", 3: "one-hop, NO fence (control)"}
    cfgs = [(0, 16, 1024), (0, 8, 1024), (0, 1, 1024), (1, 16, 1024), (1, 1, 1024), (2, 16, 1024), (3, 16, 1024),
            (0, 16, NBIG), (2, 16, NBIG), (1, 16, NBIG), (3, 16, NBIG)]
    for mode, slots, n4 in cfgs:
        its = iters_chain if n4 == 1024 else max(2000, iters_chain // 2)
        for t in (data, flag, ack, counter):
            t.zero_()
        torch.cuda.synchronize(dev); tp4.barrier()
        r = ext.p6_chain_stress(int(hd.buffer_ptrs[right]), data, int(hf.buffer_ptrs[right]), flag, int(ha.buffer_ptrs[left]), ack, counter,
                                mode, its, slots, n4)
        torch.cuda.synchronize(dev); tp4.barrier()
        viol = int(sm_(r[0]))
        cyc = mx(statistics.median(r[1:1 + slots]) / its / 1e3)
        row = {"mode": mode, "name": names[mode], "slots": slots, "payload_KB": n4 * 16 // 1024, "iters": its, "violations_all_ranks": viol,
               "cycle_us": cyc}
        out["rows"].append(row)
        if rank == 0:
            print(f"[p4] mode {mode} slots {slots:2d} payload {n4 * 16 // 1024:3d} KB iters {its}: violations {viol}, cycle {cyc:.2f} us  "
                  f"({names[mode]})", flush=True)
    return out


# ----------------------------------------------------------------------------------------------------------------- probe 5
def probe5(ext, dev, reps=5):
    out = {"rows": []}
    GP, GD, SMEM, SPIN = 128, 132, 214016, 500_000
    tp = torch.zeros(2 * GP, dtype=torch.int64, device=dev)
    sp = torch.zeros(GP, dtype=torch.int32, device=dev)
    td = torch.zeros(GD, dtype=torch.int64, device=dev)
    sd = torch.zeros(GD, dtype=torch.int32, device=dev)

    def analyse():
        t = tp.view(GP, 2).cpu()
        p0, p1 = t[:, 0].min().item(), t[:, 1].max().item()
        d, s = td.cpu(), sd.cpu()
        early = d < p1
        free = sorted(set(range(NSM)) - set(sp.cpu().tolist()))
        esm = sorted(set(s[early].tolist()))
        late = d[~early]
        return {"n_early": int(early.sum()), "early_SMs": esm, "free_SMs": free, "early_on_free": all(x in free for x in esm),
                "first_dep_minus_primary_start_us": (d.min().item() - p0) / 1e3,
                "first_late_dep_minus_primary_end_us": ((late.min().item() - p1) / 1e3) if len(late) else float("nan"),
                "primary_span_us": (p1 - p0) / 1e3}

    def run_once():
        for t in (tp, sp, td, sd):
            t.zero_()
    for dep_spin in (0, 600_000):
        for pdl in (1, 0):
            ext.p6_pdl_probe(tp, sp, td, sd, SPIN, pdl, SMEM, GP, GD, dep_spin)   # warm (module load, smem attribute)
            torch.cuda.synchronize(dev)
            for mode in ("eager", "graph"):
                g = None
                if mode == "graph":
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        ext.p6_pdl_probe(tp, sp, td, sd, SPIN, pdl, SMEM, GP, GD, dep_spin)
                    torch.cuda.synchronize(dev)
                res = []
                for _ in range(reps):
                    run_once(); torch.cuda.synchronize(dev)
                    if g is None:
                        ext.p6_pdl_probe(tp, sp, td, sd, SPIN, pdl, SMEM, GP, GD, dep_spin)
                    else:
                        g.replay()
                    torch.cuda.synchronize(dev)
                    res.append(analyse())
                row = {"pdl_attr": pdl, "mode": mode, "dep_spin_us": dep_spin / 1e3, "reps": res}
                out["rows"].append(row)
                r0 = res[0]
                print(f"[p5] dep_spin {dep_spin / 1e3:.0f} us pdl={pdl} {mode:5s}: early CTAs per rep {[r['n_early'] for r in res]}, "
                      f"early SMs {r0['early_SMs']} (free {r0['free_SMs']}, on free: {all(r['early_on_free'] for r in res)}), "
                      f"first dep - primary start {[round(r['first_dep_minus_primary_start_us'], 1) for r in res]} us, first late dep - "
                      f"primary end {[round(r['first_late_dep_minus_primary_end_us'], 1) for r in res]} us, primary span "
                      f"{r0['primary_span_us']:.0f} us", flush=True)
                del g
    return out


# ----------------------------------------------------------------------------------------------------------------- driver
def worker(rank, world, dev, gn, probes, iters, warmup, quick, chain_iters):
    from ctapp.ext import load_tp4
    ext = load_tp4()
    assert world == 4
    # CUDA lazy loading: the first launch of a kernel may need a context synchronisation, which deadlocks against an already-
    # spinning blocker (the first packed-placement p6_wait trapped this way). Load every helper kernel once while the GPU is idle.
    scratch = torch.zeros(4, dtype=torch.int32, device=dev)
    ext.p6_wait(scratch, 0)
    ext.p6_blocker(scratch, 0, 1, BLOCKER_SMEM)
    ext.p6_delay(0)
    ext.p6_stamp(torch.zeros(1, dtype=torch.int64, device=dev), 0)
    torch.cuda.synchronize(dev)
    res = {}
    if 5 in probes:
        if rank == 0:
            res["p5"] = probe5(ext, dev)
        tp4.barrier()
    if 4 in probes:
        res["p4"] = probe4(ext, rank, dev, gn, chain_iters)
    if 1 in probes:
        res["p1"] = probe1(ext, rank, dev, gn, iters, warmup, quick)
    if 3 in probes:
        res["p3"] = probe3(ext, rank, dev, gn, iters, warmup)
    if 2 in probes:
        res["p2"] = probe2(ext, rank, dev, gn, iters, warmup)
    return res if rank == 0 else {}


def f3(v):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"


def tables(r):
    if "p1" in r:
        print("\n### Probe 1: reducer-shaped push (per rank; GB/s = counted bytes / median per-call ms)\n")
        print("| M | placement | threads | T | blocks | variant | ms | kernel span ms | b2b ms | GB/s | SMs |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for x in r["p1"]["rows"]:
            print(f"| {x['M']} | {x['placement']} | {x['threads']} | {x['T']} | {x['blocks']} | {x['variant']} | {f3(x['ms'])} | "
                  f"{f3(x['kernel_span_ms'])} | {f3(x['b2b_ms'])} | {x['GBps']:.1f} | {x['distinct_SMs']:.0f} |")
        for c in r["p1"]["check"]:
            print(f"\ncheck M={c['M']} threads={c['threads']}: x mismatches {c['x_mismatch_all_ranks']}, rowss max rel err {c['rowss_max_rel_err']:.2e}")
    if "p2" in r:
        p = r["p2"]
        print("\n### Probe 2: interference (GEMM mode-0 down M=4096 sms=128 on stream 1, probe on stream 2)\n")
        print(f"GEMM alone: sms=128 {f3(p['gemm_alone']['128']['per_step'])} ms (b2b {f3(p['gemm_alone']['128']['b2b'])}), "
              f"sms=132 {f3(p['gemm_alone']['132']['per_step'])} ms (b2b {f3(p['gemm_alone']['132']['b2b'])})\n")
        print("| threads | T | blocks | GEMM alone 128 | GEMM with probe | slowdown | probe alone (blocks SMs) | probe alone (4 free SMs, blocker) | probe end - GEMM start | probe span | combined | probe SMs (max) | calls on >4 SMs | overlap |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for x in p["rows"]:
            print(f"| {x['threads']} | {x['T']} | {x['blocks']} | {f3(x['gemm_alone_128_ms'])} | {f3(x['gemm_with_probe_ms'])} | {x['gemm_slowdown'] * 100:+.1f} % | "
                  f"{f3(x['probe_alone_spread_ms'])} | {f3(x['probe_alone_packed_ms'])} | {f3(x['probe_stream_ms'])} | {f3(x['probe_span_ms'])} | "
                  f"{f3(x['combined_ms'])} | {x['probe_distinct_SMs']:.0f} | {x.get('calls_probe_on_more_than_4_SMs_all_ranks', 'n/a')} | {x['overlap_frac_of_probe_min_ranks'] * 100:.0f} % |")
    if "p3" in r:
        print("\n### Probe 3: down GEMM mode 0, D local vs D on right neighbour (all ranks concurrently), raster 1\n")
        print("| M | sms | swizzle | local ms | peer ms | peer/local | local b2b | peer b2b | ratio b2b | err local / peer |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for x in r["p3"]["rows"]:
            print(f"| {x['M']} | {x['sms']} | {x['swizzle']} | {f3(x['local_ms'])} | {f3(x['peer_ms'])} | {x['ratio']:.3f} | {f3(x['local_b2b'])} | "
                  f"{f3(x['peer_b2b'])} | {x['ratio_b2b']:.3f} | {x['local_err']:.1e} / {x['peer_err']:.1e} |")
    if "p4" in r:
        print("\n### Probe 4: flag chain (16 KB payload per iteration per slot)\n")
        print("| mode | slots | payload KB | iters | violations (all ranks) | cycle us |")
        print("|---|---|---|---|---|---|")
        for x in r["p4"]["rows"]:
            print(f"| {x['name']} | {x['slots']} | {x['payload_KB']} | {x['iters']} | {x['violations_all_ranks']} | {x['cycle_us']:.2f} |")
    if "p5" in r:
        print("\n### Probe 5: PDL early launch (primary 128 CTAs x 214016 B smem, spins 500 us; dependent 132 CTAs)\n")
        print("| dependent spin us | PDL attr | mode | early CTAs per rep | distinct early SMs (rep 0) | free SMs | early on free | first dep - primary start us | first late dep - primary end us |")
        print("|---|---|---|---|---|---|---|---|---|")
        for x in r["p5"]["rows"]:
            rs = x["reps"]
            print(f"| {x['dep_spin_us']:.0f} | {x['pdl_attr']} | {x['mode']} | {[q['n_early'] for q in rs]} | {rs[0]['early_SMs']} | {rs[0]['free_SMs']} | "
                  f"{all(q['early_on_free'] for q in rs)} | {[round(q['first_dep_minus_primary_start_us'], 1) for q in rs]} | "
                  f"{[round(q['first_late_dep_minus_primary_end_us'], 1) for q in rs]} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", default="1,2,3,4,5")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--chain-iters", type=int, default=5000)
    ap.add_argument("--quick", action="store_true", help="probe 1: only T=2, full variant")
    ap.add_argument("--port", type=int, default=29931)
    ap.add_argument("--json", default="build/p6_s1.json")
    a = ap.parse_args()
    probes = {int(p) for p in a.probes.split(",")}
    res, bad = tp4.launch(worker, 4, a.port, probes, a.iters, a.warmup, a.quick, a.chain_iters)
    if bad:
        print("push_bw FAILED (see tracebacks above)")
        return 1
    r = res[0]
    path = a.json
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(r)
    old["meta"] = {"iters": a.iters, "warmup": a.warmup, "chain_iters": a.chain_iters}
    with open(path, "w") as f:
        json.dump(old, f, indent=1)
    tables(r)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
