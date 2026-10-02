"""Phase-4 tests: multimem probes + end-to-end CtappBoundary vs fp32 reference.

    python tests/test_tp4.py --world 4 [--m 1024] [--port 29900]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from ctapp import tp4  # noqa: E402
from ctapp.tp4 import _symm, barrier  # noqa: E402

FAILS = []


def check(rank, name, ok, detail=""):
    t = torch.tensor([1 if ok else 0], dtype=torch.int32)
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=tp4.GLOO)
    if not ok:
        print(f"  [rank {rank}] local failure in '{name}': {detail}", flush=True)
    if rank == 0:
        print(f"{'PASS' if t.item() else 'FAIL'}  {name}  {detail}", flush=True)
    if not t.item():
        FAILS.append(name)


def sync(dev):
    torch.cuda.synchronize(dev)
    barrier()


def probes(rank, world, dev, gn):
    from ctapp.ext import load_tp4
    ext = load_tp4()
    # mm_red_u32
    buf, hdl = _symm((4096,), torch.int32, dev, gn)
    ok, det = True, ""
    for n in (1, 100, 4096):
        buf.zero_(); sync(dev)
        ext.mm_red_u32(hdl.multicast_ptr, n, 3); sync(dev)
        b = buf.cpu()
        good = bool((b[:n] == 3 * world).all()) and bool((b[n:] == 0).all())
        ok &= good
        det += f"n={n}:{'ok' if good else 'BAD'} "
    check(rank, "probe mm_red_u32 (copy == world*v)", ok, det)
    # mm_red_f32
    n = 8192
    fb, fh = _symm((n,), torch.float32, dev, gn)
    srcs = [torch.randn(n, generator=torch.Generator().manual_seed(100 + r)) for r in range(world)]
    fb.zero_(); sync(dev)
    ext.mm_red_f32(fh.multicast_ptr, srcs[rank].to(dev)); sync(dev)
    exp = sum(srcs)
    err = (fb.cpu() - exp).abs().max().item()
    check(rank, "probe mm_red_f32 (sum of src)", err < 1e-4, f"max|err|={err:.2e}")
    # mm_st_bf16 from rank 0
    bb, bh = _symm((n,), DT := torch.bfloat16, dev, gn)
    bb.zero_(); sync(dev)
    src = torch.randn(n, generator=torch.Generator().manual_seed(7)).to(DT)
    if rank == 0:
        ext.mm_st_bf16(bh.multicast_ptr, src.to(dev))
    sync(dev)
    check(rank, "probe mm_st_bf16 (broadcast from rank 0)", torch.equal(bb.cpu(), src))
    # mm_ld_reduce_bf16, random
    n = 65536
    lb, lh = _symm((n,), DT, dev, gn)
    datas = [(torch.randn(n, generator=torch.Generator().manual_seed(200 + r)) * 0.1).to(DT) for r in range(world)]
    lb.copy_(datas[rank]); sync(dev)
    out = torch.empty(n, dtype=DT, device=dev)
    ext.mm_ld_reduce_bf16(lh.multicast_ptr, out); sync(dev)
    exp = sum(d.float() for d in datas).to(DT)
    mism = (out.cpu() != exp).sum().item()
    check(rank, "probe mm_ld_reduce_bf16 (random, == fp32 sum rounded)", mism <= n * 1e-4, f"mismatches {mism}/{n}")
    # fp32-accumulation proof (needs >= 3 ranks)
    if world >= 3:
        vals = [256.0] + [1.0] * (world - 1)
        lb.fill_(vals[rank]); sync(dev)
        ext.mm_ld_reduce_bf16(lh.multicast_ptr, out); sync(dev)
        exp = torch.tensor(sum(vals)).to(DT)
        seq = torch.tensor(vals[0]).to(DT)
        for v in vals[1:]:
            seq = (seq.float() + v).to(DT)
        assert exp != seq
        got = out.cpu()
        check(rank, "probe mm_ld_reduce_bf16 fp32 accumulation", bool((got == exp).all()),
              f"got {got[0].item()} fp32-sum {exp.item()} (bf16-sequential would be {seq.item()})")
    elif rank == 0:
        print("SKIP  fp32-accumulation proof needs world >= 3", flush=True)
    # order stress
    slots, iters = 64, 2000
    db, dh = _symm((slots * world * 8 * 64,), DT, dev, gn)
    fl, fh2 = _symm((slots,), torch.int32, dev, gn)
    db.zero_(); fl.zero_(); sync(dev)
    viol = ext.mm_order_stress(dh.multicast_ptr, db, fh2.multicast_ptr, fl, rank, world, iters, slots)
    sync(dev)
    check(rank, "probe mm_order_stress (0 violations)", viol == 0, f"violations={viol}")


def rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()


def e2e(rank, world, dev, gn, M):
    from ctapp.tp4 import CtappBoundary, MultimemBoundary
    Kr, N2r = tp4.K_FFN // world, tp4.N2 // world
    Wd, gamma, Wq = tp4.make_weights(dev)
    h, resid = tp4.make_inputs(M, dev)
    x_ref, out_ref = tp4.one_gpu_reference(h, Wd, resid, gamma, Wq)
    Wd_r, Wq_r, Wqf_r = tp4.shard_weights(Wd, Wq, gamma, rank, world)
    h_r = h[:, rank * Kr:(rank + 1) * Kr].contiguous()
    ref_r = out_ref[:, rank * N2r:(rank + 1) * N2r]
    del Wd, Wq, h, out_ref
    mm = MultimemBoundary(M, rank, world, gn, dev, Wd_r, Wq_r, gamma)
    mm.forward(h_r, resid); sync(dev)
    ex, eo = rel(mm.x(), x_ref), rel(mm.forward(h_r, resid), ref_r)
    sync(dev)
    if rank == 0:
        print(f"multimem baseline: x rel {ex:.2e}, out rel {eo:.2e}", flush=True)
    tol_x, tol_o = 1e-2, 2e-2

    def check_state(bd, tag):
        cx, co = rel(bd.x(), x_ref), rel(bd.out, ref_r)
        check(rank, f"{tag}: x rel err", cx < tol_x, f"{cx:.2e} (multimem {ex:.2e})")
        check(rank, f"{tag}: out rel err", co < tol_o, f"{co:.2e} (multimem {eo:.2e})")

    def identical(bd, tag):
        xi = bd.x().contiguous().view(torch.int32).cpu()
        lst = [torch.empty_like(xi) for _ in range(world)]
        dist.all_gather(lst, xi, group=tp4.GLOO)
        diffs = [int((l != lst[0]).sum()) for l in lst]
        check(rank, f"{tag}: x identical on all ranks", sum(diffs) == 0, f"differing elements per rank {diffs}")

    for graph, first in ((False, 3), (True, 4)):
        tag0 = f"ctapp{'_graph' if graph else ''} M={M}"
        bd = CtappBoundary(M, Kr, N2r, rank, world, gn, dev, resid=True, protocol=True, use_graph=graph)
        bd.set_weights(Wd_r, Wqf_r)
        for ep in range(1, first + 1):
            bd.forward(h_r, resid)
            torch.cuda.synchronize(dev)
            check_state(bd, f"{tag0} forward {ep}")
            sync(dev)
        identical(bd, tag0)
        for _ in range(20):
            bd.forward(h_r, resid)
        torch.cuda.synchronize(dev)
        check_state(bd, f"{tag0} after +20 back-to-back (epoch {bd.epoch})")
        identical(bd, tag0 + " +20")
        sync(dev)
        if not graph:
            bd.ext.tp4_down(h_r, bd.Wd, bd.partials, 1, 1, 0, rank, world, 0, bd.tile_mc, bd.tile_cnt, bd.partials_mc, bd.x_mc,
                            None, bd.rowss_mc, bd.panel_mc)
            torch.cuda.synchronize(dev)
            d0 = rel(bd.partials, torch.mm(h_r, Wd_r.t()))
            check(rank, f"{tag0}: tp4_down mode 0 vs torch.mm", d0 < 2e-3, f"rel {d0:.2e}")
            sync(dev)
        del bd
    # protocol=False floor path runs (no correctness claim on out; partials are the plain GEMM)
    for graph in (False, True):
        bn = CtappBoundary(M, Kr, N2r, rank, world, gn, dev, resid=True, protocol=False, use_graph=graph)
        bn.set_weights(Wd_r, Wqf_r)
        for _ in range(5):
            bn.forward(h_r, resid)
        torch.cuda.synchronize(dev)
        d0 = rel(bn.partials, torch.mm(h_r, Wd_r.t()))
        check(rank, f"ctapp_nowait{'_graph' if graph else ''} M={M} runs", d0 < 2e-3, f"partials vs torch.mm rel {d0:.2e}")
        sync(dev)
        del bn
    return len(FAILS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--m", type=int, default=1024)
    ap.add_argument("--port", type=int, default=29900)
    a = ap.parse_args()
    res, bad = tp4.launch(_run, a.world, a.port, a.m)
    ok = (not bad) and all(r == 0 for r in res)
    print("ALL PASS" if ok else f"FAILED (results {res}, process error: {bad})")
    return 0 if ok else 1


def _run(rank, world, dev, gn, M):
    probes(rank, world, dev, gn)
    return e2e(rank, world, dev, gn, M)


if __name__ == "__main__":
    sys.exit(main())
