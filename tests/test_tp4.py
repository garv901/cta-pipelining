"""Phase-4 tests: multimem probes + end-to-end CtappBoundary vs fp32 reference.

    python tests/test_tp4.py --world 4 [--m 1024] [--port 29900] [--protocol v3|v5] [--red-variant 0|1|2] [--red-threads N]
                             [--fusion none|pdl|role] [--qkv-tile 256|128] [--R 4] [--role-rows 2|4|8]

--protocol v5 additionally runs: flags == epoch after a forward, a variable-M sequence on ONE instance (max_M 8192, fresh
weights per call), a 30-forward back-to-back stress with alternating M and inputs (every forward's out / x checked), and a
50-forward back-to-back variable-M (4096/1024/8192) hang check, eager and CUDA graph (every forward checked).
--fusion (Phase 6 S3): "pdl" = PDL consumer (Variant A'), "role" = role-switching 2-kernel boundary (Variant B); with "role" the
first eager forward is traced (CTAPP_TRACE stamps) and the role CTAs' SMs are printed next to the SMs the producer left free.
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


V5KW = {}   # extra CtappBoundary kwargs for protocol v5 (--red-variant / --red-threads / --fusion / --qkv-tile / --R / --role-rows)


def fusion_tag():
    f, t = V5KW.get("fusion", "none"), V5KW.get("qkv_tile", 256)
    return (("" if f == "none" else f" fusion={f}") + (" steal" if V5KW.get("steal") else "") + ("" if t == 256 else f" tile={t}")
            + (f" R={V5KW['R']}" if "R" in V5KW else "") + (f" stages={V5KW['role_stages']}" if V5KW.get("role_stages") else "")
            + (f" tail={V5KW['down_tail']}" if V5KW.get("down_tail") else ""))


def e2e(rank, world, dev, gn, M, protocol="v3"):
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

    proto = "v5" if protocol == "v5" else True
    for graph, first in ((False, 3), (True, 4)):
        tag0 = f"ctapp{'_graph' if graph else ''} {protocol}{fusion_tag()} M={M}"
        bd = CtappBoundary(M, Kr, N2r, rank, world, gn, dev, resid=True, protocol=proto, use_graph=graph, **(V5KW if proto == "v5" else {}))
        bd.set_weights(Wd_r, Wqf_r)
        for ep in range(1, first + 1):
            if proto == "v5" and not graph and ep == 2:
                bd.trace = True
            bd.forward(h_r, resid)
            torch.cuda.synchronize(dev)
            if proto == "v5" and bd.trace and not graph:
                bd.trace = False
                ts = bd.trace_summary()
                if rank == 0:
                    print(f"  trace (rank 0, us from producer entry): {ts}", flush=True)
            check_state(bd, f"{tag0} forward {ep}")
            sync(dev)
        identical(bd, tag0)
        for _ in range(20):
            bd.forward(h_r, resid)
        torch.cuda.synchronize(dev)
        check_state(bd, f"{tag0} after +20 back-to-back (epoch {bd.epoch})")
        identical(bd, tag0 + " +20")
        sync(dev)
        if protocol == "v5":
            v5_flags(rank, bd, M, tag0)
        if not graph and protocol != "v5":
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


def v5_flags(rank, bd, M, tag):
    """After a completed forward: every tile flag of this rank's owned panels (all 4 sources) and every panel flag == epoch."""
    torch.cuda.synchronize(bd.dev)
    P = M // 128
    Pr = (P - 1 - rank) // 4 + 1 if P > rank else 0
    tf = bd.tile_flags[:, :Pr, :].cpu()
    pf = bd.panel_flag[:P].cpu()
    ok = bool((tf == bd.epoch).all()) and bool((pf == bd.epoch).all())
    check(rank, f"{tag}: tile_flags / panel_flag == epoch {bd.epoch}", ok,
          f"tile_flags min {tf.min().item() if tf.numel() else '-'} max {tf.max().item() if tf.numel() else '-'}, "
          f"panel_flag min {pf.min().item()} max {pf.max().item()}")
    sync(bd.dev)


def v5_extra(rank, world, dev, gn):
    """Variable M on ONE instance (max_M 8192) with fresh weights per call; 30-forward back-to-back stress."""
    from ctapp.tp4 import CtappBoundary
    Kr, N2r = tp4.K_FFN // world, tp4.N2 // world
    bd = CtappBoundary(8192, Kr, N2r, rank, world, gn, dev, resid=True, protocol="v5", max_M=8192, **V5KW)
    bd.warmup()
    for i, M in enumerate((4096, 1024, 8192, 512, 4096)):
        Wd, gamma, Wq = tp4.make_weights(dev, seed=10 + i)
        Wd_r, _, Wqf_r = tp4.shard_weights(Wd, Wq, gamma, rank, world)
        h, resid = tp4.make_inputs(M, dev, seed=20 + i)
        h_r = h[:, rank * Kr:(rank + 1) * Kr].contiguous()
        y = bd.forward(h_r, resid, Wd_r, Wqf_r)
        torch.cuda.synchronize(dev)
        x_ref, out_ref = tp4.one_gpu_reference(h, Wd, resid, gamma, Wq)
        ref_r = out_ref[:, rank * N2r:(rank + 1) * N2r]
        cx, co = rel(bd.x(), x_ref), rel(y, ref_r)
        tag = f"v5 variable-M step {i + 1} M={M} (one instance, max_M 8192)"
        check(rank, f"{tag}: x rel", cx < 1e-2 and bd.x().shape[0] == M, f"{cx:.2e}")
        check(rank, f"{tag}: out rel", co < 2e-2 and y.shape[0] == M, f"{co:.2e}")
        v5_flags(rank, bd, M, tag)
        del Wd, Wq, h, x_ref, out_ref, ref_r
        torch.cuda.empty_cache()
    # stress: 30 forwards back to back, alternating (M, inputs, weights); every output / x is cloned on the stream and checked after
    Wd, gamma, Wq = tp4.make_weights(dev, seed=3)
    Wd_r, _, Wqf_r = tp4.shard_weights(Wd, Wq, gamma, rank, world)
    sets = []
    for M, seed in ((4096, 31), (1024, 32), (2048, 33)):
        h, resid = tp4.make_inputs(M, dev, seed=seed)
        x_ref, out_ref = tp4.one_gpu_reference(h, Wd, resid, gamma, Wq)
        sets.append((h[:, rank * Kr:(rank + 1) * Kr].contiguous(), resid, x_ref.to(tp4.DTYPE), out_ref[:, rank * N2r:(rank + 1) * N2r].contiguous()))
        del h
    sync(dev)
    keep = []
    for i in range(30):
        h_r, resid, _, _ = sets[i % 3]
        y = bd.forward(h_r, resid, Wd_r, Wqf_r)
        keep.append((i % 3, y.clone(), bd.x().clone()))
    torch.cuda.synchronize(dev)
    worst_x = worst_o = 0.0
    for k, y, x in keep:
        worst_x = max(worst_x, rel(x, sets[k][2]))
        worst_o = max(worst_o, rel(y, sets[k][3]))
    check(rank, f"v5{fusion_tag()} stress 30 b2b forwards (M 4096/1024/2048): worst x rel", worst_x < 1e-2, f"{worst_x:.2e}")
    check(rank, f"v5{fusion_tag()} stress 30 b2b forwards: worst out rel", worst_o < 2e-2, f"{worst_o:.2e}")
    v5_flags(rank, bd, 4096 if keep[-1][0] == 0 else (1024 if keep[-1][0] == 1 else 2048), "v5 stress")
    del bd, keep
    torch.cuda.empty_cache()
    return len(FAILS)


def v5_hang(rank, world, dev, gn, graph):
    """50 back-to-back forwards cycling M = 4096 / 1024 / 8192 on one max_M 8192 instance (default weights; eager or CUDA graph:
    one graph per (parity, M)); every forward's out / x cloned on the stream and checked after one final sync."""
    from ctapp.tp4 import CtappBoundary
    Kr, N2r = tp4.K_FFN // world, tp4.N2 // world
    Wd, gamma, Wq = tp4.make_weights(dev, seed=5)
    Wd_r, _, Wqf_r = tp4.shard_weights(Wd, Wq, gamma, rank, world)
    sets = []
    for M, seed in ((4096, 41), (1024, 42), (8192, 43)):
        h, resid = tp4.make_inputs(M, dev, seed=seed)
        x_ref, out_ref = tp4.one_gpu_reference(h, Wd, resid, gamma, Wq)
        sets.append((h[:, rank * Kr:(rank + 1) * Kr].contiguous(), resid, x_ref.to(tp4.DTYPE), out_ref[:, rank * N2r:(rank + 1) * N2r].contiguous()))
        del h, x_ref, out_ref
    del Wd, Wq
    torch.cuda.empty_cache()
    bd = CtappBoundary(8192, Kr, N2r, rank, world, gn, dev, resid=True, protocol="v5", max_M=8192, use_graph=graph, **V5KW)
    bd.set_weights(Wd_r, Wqf_r)
    sync(dev)
    keep = []
    for i in range(50):
        h_r, resid, _, _ = sets[i % 3]
        y = bd.forward(h_r, resid)
        keep.append((i % 3, y.clone(), bd.x().clone()))
    torch.cuda.synchronize(dev)
    worst_x = worst_o = 0.0
    for k, y, x in keep:
        worst_x = max(worst_x, rel(x, sets[k][2]))
        worst_o = max(worst_o, rel(y, sets[k][3]))
    tag = f"v5{fusion_tag()} hang check 50 b2b forwards M 4096/1024/8192 {'graph' if graph else 'eager'}"
    check(rank, f"{tag}: worst x rel", worst_x < 1e-2, f"{worst_x:.2e}")
    check(rank, f"{tag}: worst out rel", worst_o < 2e-2, f"{worst_o:.2e}")
    sync(dev)
    del bd, keep, sets
    torch.cuda.empty_cache()
    return len(FAILS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--m", type=int, default=1024)
    ap.add_argument("--port", type=int, default=29900)
    ap.add_argument("--protocol", default="v3", choices=["v3", "v5"])
    ap.add_argument("--red-variant", type=int, default=None, help="v5 reducer variant (default: CtappBoundary's)")
    ap.add_argument("--red-threads", type=int, default=None)
    ap.add_argument("--fusion", default=None, choices=["none", "pdl", "pdl1", "role"], help="v5 only (default: CtappBoundary's, none)")
    ap.add_argument("--qkv-tile", type=int, default=None, choices=[256, 128])
    ap.add_argument("--R", type=int, default=None, help="v5 reserved SMs / role CTAs")
    ap.add_argument("--role-rows", type=int, default=None, choices=[2, 4, 8])
    ap.add_argument("--qkv-raster", type=int, default=None)
    ap.add_argument("--qkv-swizzle", type=int, default=None)
    ap.add_argument("--down-swizzle", type=int, default=None)
    ap.add_argument("--steal", action="store_true", help="v5: work-stealing reducer units (mode-8 consumer on every CTA)")
    ap.add_argument("--role-stages", type=int, default=None, choices=[0, 2, 3])
    ap.add_argument("--down-tail", default=None, help="v5: producer panel-major tail columns (int) or auto")
    a = ap.parse_args()
    v5kw = {k: v for k, v in (("red_variant", a.red_variant), ("red_threads", a.red_threads), ("fusion", a.fusion),
                              ("qkv_tile", a.qkv_tile), ("R", a.R), ("role_rows", a.role_rows), ("qkv_raster", a.qkv_raster),
                              ("qkv_swizzle", a.qkv_swizzle), ("down_swizzle", a.down_swizzle), ("steal", a.steal or None),
                              ("role_stages", a.role_stages),
                              ("down_tail", None if a.down_tail is None else (a.down_tail if a.down_tail == "auto" else int(a.down_tail))))
            if v is not None}
    assert a.protocol == "v5" or not v5kw, "v5-only options given with --protocol v3"
    res, bad = tp4.launch(_run, a.world, a.port, a.m, a.protocol, v5kw)
    ok = (not bad) and all(r == 0 for r in res)
    print("ALL PASS" if ok else f"FAILED (results {res}, process error: {bad})")
    return 0 if ok else 1


def _run(rank, world, dev, gn, M, protocol="v3", v5kw=None):
    V5KW.update(v5kw or {})
    probes(rank, world, dev, gn)
    e2e(rank, world, dev, gn, M, protocol)
    if protocol == "v5":
        v5_extra(rank, world, dev, gn)
        for graph in (False, True):
            v5_hang(rank, world, dev, gn, graph)
    return len(FAILS)


if __name__ == "__main__":
    sys.exit(main())
