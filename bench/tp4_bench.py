"""Phase-4 bench: TP down-proj -> resid -> RMSNorm -> QKV, CTA-pipelined (ctapp) vs baselines.

    python bench/tp4_bench.py --world 4 --tokens 1024,2048,4096,8192,16384 --iters 30 --warmup 5 --json build/tp4_4.json
    python bench/tp4_bench.py --world 4 --protocol v5 --R 4 --methods ctapp,ctapp_graph,ctapp_nowait   # -> build/tp4_4_v5_R4.json

Per M: one_gpu = eager bf16 chain on rank 0, ideal = one_gpu / world. Methods: ctapp (eager launches), ctapp_graph (CUDA graph
replay, one graph per epoch parity), ctapp_nowait (mode-0 GEMMs only: the raw two-GEMM floor), multimem, flashinfer (fused AR+resid+RMSNorm), asynctp, nccl. Times: per-step (barrier+sync each call, median) and back-to-back (mean),
both max over ranks. exposed % = (t - ideal) / t.
--protocol v3 (default) | v5 | both: which CtappBoundary protocol the ctapp methods use (v5 methods are named ctapp_v5,
ctapp_v5_graph, ctapp_v5_nowait; v5 instances use max_M = M). ctapp_graphb (v5 only): CUDA graph that binds the caller's input
tensors (graph_bind=True: no static-buffer copy before replay). --R: reserved SMs (default: per-protocol default; with
--protocol both it applies to the v5 methods only).
--methods: comma list of base names (ctapp, ctapp_graph, ctapp_nowait, multimem, ...) to run; default all.
Phase 6 S3 (v5 only): --fusion none,pdl,role (comma list; v5 method names get _pdl / _role), --qkv-tile 256,128 (_t128),
--role-rows, --down-swizzle, --qkv-raster / --qkv-swizzle (default per tile / shape), --pdl-trigger 0|1 (Variant A' only);
--shape B (default: down 28672 -> QKV 10240) | A (o_proj boundary: down K 8192 -> gate_up 57344, i.e. Kr 2048, N2r 14336).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from ctapp import tp4  # noqa: E402

METHODS = ["ctapp", "ctapp_graph", "ctapp_graphb", "ctapp_nowait", "multimem", "flashinfer", "asynctp", "nccl"]
QKV128_B = (2, 2)   # QKV raster / swizzle of the 128x128 tile at N2r = 2560 (S3 1-GPU sweep: r2 s2 0.239 ms b2b at 4k, 128 SMs)
BASE = ("multimem", "flashinfer", "asynctp", "nccl")


SHAPES = {"B": (tp4.K_FFN, tp4.N2), "A": (8192, 57344)}   # (full down K, full consumer N)


def worker(rank, world, dev, gn, tokens, iters, warmup, specs, shape, repeat=1):
    K, N2f = SHAPES[shape]
    Kr, N2r = K // world, N2f // world
    Wd, gamma, Wq = tp4.make_weights(dev, K=K, N2_=N2f)
    Wd_r, Wq_r, Wqf_r = tp4.shard_weights(Wd, Wq, gamma, rank, world)
    methods = [n for n, _ in specs]
    out = {}
    for M in tokens:
        h, resid = tp4.make_inputs(M, dev, K=K)
        x_ref, out_ref = tp4.one_gpu_reference(h, Wd, resid, gamma, Wq)
        ref_r = out_ref[:, rank * N2r:(rank + 1) * N2r]
        h_r = h[:, rank * Kr:(rank + 1) * Kr].contiguous()
        t = torch.zeros(1, dtype=torch.float64)
        if rank == 0:
            t[0] = tp4.time_one_gpu(lambda: tp4.one_gpu_chain_bf16(h, Wd, resid, gamma, Wq), dev, iters, warmup)
            print(f"M={M:6d} one_gpu {t.item():8.3f} ms  ideal {t.item() / world:8.3f} ms", flush=True)
        dist.broadcast(t, src=0, group=tp4.GLOO)
        one = t.item()
        out[M] = {"one_gpu": one, "ideal": one / world}
        live = {}   # --repeat > 1: every instance stays alive, timings are interleaved over the methods, median over repeats
        for name, kw in specs:
            tp4.barrier()
            try:
                if kw is not None:
                    kw = dict(kw, max_M=M) if kw.get("protocol") == "v5" else kw
                    b = tp4.CtappBoundary(M, Kr, N2r, rank, world, gn, dev, resid=True, **kw)
                    b.set_weights(Wd_r, Wqf_r)
                else:
                    cls = {"multimem": tp4.MultimemBoundary, "flashinfer": tp4.FlashInferBoundary, "asynctp": tp4.AsyncTpBoundary, "nccl": tp4.NcclBoundary}[name]
                    b = cls(M, rank, world, gn, dev, Wd_r, Wq_r, gamma)
                y = b.forward(h_r, resid)
                if "_graph" in name:   # check the replayed graphs (parity 0 and 1), not only the un-captured warm-up call
                    y = b.forward(h_r, resid)
                    y = b.forward(h_r, resid)
                torch.cuda.synchronize(dev)
                err = float("nan") if name.endswith("_nowait") else ((y.float() - ref_r.float()).abs().max() / ref_r.abs().max()).item()
                e = torch.tensor([err], dtype=torch.float64)
                dist.all_reduce(e, op=dist.ReduceOp.MAX, group=tp4.GLOO)
                if repeat > 1:
                    live[name] = b
                    out[M][name] = {"rel_err": e.item(), "per_step_all": [], "b2b_all": []}
                    del y
                    continue
                ps, bb = tp4.time_step(lambda: b.forward(h_r, resid), dev, iters, warmup)
                out[M][name] = {"per_step": ps, "b2b": bb, "rel_err": e.item()}
                if rank == 0:
                    print(f"M={M:6d} {name:22s} per-step {ps:8.3f}  b2b {bb:8.3f} ms  out rel err {e.item():.2e}", flush=True)
                if hasattr(b, "destroy"):
                    b.destroy()
                del b, y
            except Exception as ex:  # noqa: BLE001
                out[M][name] = {"per_step": float("nan"), "b2b": float("nan"), "rel_err": float("nan")}
                if rank == 0:
                    print(f"M={M:6d} {name} FAILED {type(ex).__name__}: {str(ex)[:200]}", flush=True)
                torch.cuda.synchronize(dev)
                tp4.barrier()
            torch.cuda.empty_cache()
        for rep_i in range(repeat if live else 0):
            for name, b in live.items():
                tp4.barrier()
                ps, bb = tp4.time_step(lambda: b.forward(h_r, resid), dev, iters, warmup)
                out[M][name]["per_step_all"].append(ps)
                out[M][name]["b2b_all"].append(bb)
                if rank == 0:
                    print(f"M={M:6d} rep {rep_i} {name:22s} per-step {ps:8.3f}  b2b {bb:8.3f} ms", flush=True)
        for name in live:
            d = out[M][name]
            d["per_step"], d["b2b"] = sorted(d["per_step_all"])[repeat // 2], sorted(d["b2b_all"])[repeat // 2]
            if rank == 0:
                print(f"M={M:6d} {name:22s} median of {repeat}: per-step {d['per_step']:8.3f}  b2b {d['b2b']:8.3f} ms (b2b min {min(d['b2b_all']):.3f} "
                      f"max {max(d['b2b_all']):.3f})  out rel err {d['rel_err']:.2e}", flush=True)
        for b in live.values():
            if hasattr(b, "destroy"):
                b.destroy()
        live.clear()
        torch.cuda.empty_cache()
        del h, resid, x_ref, out_ref, ref_r, h_r
    return out if rank == 0 else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--tokens", default="1024,2048,4096,8192,16384")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--port", type=int, default=29910)
    ap.add_argument("--json", default=None)
    ap.add_argument("--protocol", default="v3", choices=["v3", "v5", "both"])
    ap.add_argument("--R", type=int, default=None)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--red-threads", type=int, default=1024)
    ap.add_argument("--red-variant", type=int, default=None)
    ap.add_argument("--down-raster", type=int, default=None)
    ap.add_argument("--down-swizzle", type=int, default=None)
    ap.add_argument("--fusion", default="none", help="v5: comma list of none,pdl,role, each optionally +steal (e.g. role+steal)")
    ap.add_argument("--role-stages", type=int, default=None, help="v5: role reducer cp.async staging 0 | 2 | 3 (needs --role-rows 2)")
    ap.add_argument("--qkv-tile", default="256", help="v5: comma list of 256,128")
    ap.add_argument("--qkv-raster", type=int, default=None, help="default: 2 (shape B tile 256), 1 (shape A); see QKV_DEFAULTS")
    ap.add_argument("--qkv-swizzle", type=int, default=None)
    ap.add_argument("--role-rows", type=int, default=None)
    ap.add_argument("--pdl-trigger", type=int, default=None)
    ap.add_argument("--down-tail", default=None, help="v5: producer panel-major tail columns (int) or auto")
    ap.add_argument("--shape", default="B", choices=list(SHAPES))
    ap.add_argument("--repeat", type=int, default=1, help="> 1: keep all instances alive, interleave the timings, report the median")
    a = ap.parse_args()
    tokens = [int(t) for t in a.tokens.split(",")]
    want = a.methods.split(",")
    # per (shape, tile) QKV raster / swizzle defaults (1-GPU sweeps: build/logs/p6_s3_qkv_sweep.log)
    QKV_DEFAULTS = {("B", 256): (2, 1), ("B", 128): QKV128_B, ("A", 256): (1, 2), ("A", 128): (1, 2)}
    v5kw = {"red_threads": a.red_threads, "down_raster": a.down_raster}
    for k, v in (("red_variant", a.red_variant), ("down_swizzle", a.down_swizzle), ("role_rows", a.role_rows), ("pdl_trigger", a.pdl_trigger),
                 ("role_stages", a.role_stages),
                 ("down_tail", None if a.down_tail is None else (a.down_tail if a.down_tail == "auto" else int(a.down_tail)))):
        if v is not None:
            v5kw[k] = v
    specs = []   # (method name, CtappBoundary kwargs | None for the baselines)
    for p in (["v3", "v5"] if a.protocol == "both" else [a.protocol]):
        for m in METHODS:
            if m not in want or not (m.startswith("ctapp") or p == ("v5" if a.protocol == "v5" else "v3")):
                continue
            if not m.startswith("ctapp"):
                specs.append((m, None))
                continue
            graph, nowait, bind = m in ("ctapp_graph", "ctapp_graphb"), m == "ctapp_nowait", m == "ctapp_graphb"
            if bind and p == "v3":
                continue
            tiles = [256] if p == "v3" or nowait else [int(t) for t in a.qkv_tile.split(",")]
            for fu in (["none"] if p == "v3" or nowait else a.fusion.split(",")):
                for tile in tiles:
                    qr, qs = QKV_DEFAULTS[(a.shape, tile)]
                    qr = a.qkv_raster if a.qkv_raster is not None else qr
                    qs = a.qkv_swizzle if a.qkv_swizzle is not None else qs
                    kw = dict(use_graph=graph, qkv_raster=qr, qkv_swizzle=qs)
                    if p == "v3":
                        kw.update(protocol=False if nowait else True, R=None if a.protocol == "both" else a.R)
                        specs.append((m, kw))
                    elif nowait:
                        kw.update(protocol=False, R=a.R or 4)
                        specs.append((m.replace("ctapp", "ctapp_v5", 1), kw))
                    else:
                        fb, _, suf = fu.partition("+")
                        assert suf in ("", "steal"), fu
                        kw.update(v5kw, protocol="v5", R=a.R or 4, fusion=fb, qkv_tile=tile, steal=suf == "steal", graph_bind=bind)
                        name = ("ctapp_v5" + ("" if fb == "none" else f"_{fb}") + ("_steal" if suf else "") + ("" if tile == 256 else f"_t{tile}")
                                + ("_graphb" if bind else "_graph" if graph else ""))
                        specs.append((name, kw))
    methods = [n for n, _ in specs]
    res, bad = tp4.launch(worker, a.world, a.port, tokens, a.iters, a.warmup, specs, a.shape, a.repeat)
    if bad:
        print("bench failed")
        return 1
    r = res[0]
    nan = float("nan")
    for title, key in (("per step", "per_step"), ("back-to-back", "b2b")):
        print(f"\n== TP{a.world} down->resid->RMSNorm->QKV, ms, {title}; cell = ms (exposed %) ; ideal = 1-GPU/{a.world} ==")
        print("| M | one_gpu | ideal | " + " | ".join(methods) + " | best ctapp vs best baseline |")
        print("|---" * (len(methods) + 4) + "|")
        for M in tokens:
            d = r[M]
            cells = []
            for m in methods:
                v = d[m][key]
                cells.append("n/a" if math.isnan(v) else f"{v:.3f} ({(v - d['ideal']) / v * 100:.0f}%)")
            base = [d[m][key] for m in BASE if m in d and not math.isnan(d[m][key])]
            cv = [d[m][key] for m in methods if m.startswith("ctapp") and not m.endswith("_nowait") and not math.isnan(d[m][key])]
            spd = f"{min(base) / min(cv):.2f}x" if base and cv else "n/a"
            print(f"| {M} | {d['one_gpu']:.3f} | {d['ideal']:.3f} | " + " | ".join(cells) + f" | {spd} |")
    tag = "" if a.protocol == "v3" and a.R is None else f"_{a.protocol}" + (f"_R{a.R}" if a.R else "")
    tag += "" if a.shape == "B" else f"_shape{a.shape}"
    tag += "" if a.fusion == "none" else "_" + a.fusion.replace(",", "-").replace("+", "")
    path = a.json or f"build/tp4_{a.world}{tag}.json"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        results = {}
        for M in tokens:
            for m in methods:
                v = r[M][m]
                results[f"{M}/{m}"] = {"per_step_ms": v["per_step"], "b2b_ms": v["b2b"], "rel_err": v["rel_err"]}
        json.dump({"shape": "l70b_qkv" if a.shape == "B" else "l70b_oproj_gateup", "world": a.world, "iters": a.iters, "warmup": a.warmup,
                   "protocol": a.protocol, "R": a.R, "v5": v5kw, "fusion": a.fusion, "qkv_tile": a.qkv_tile,
                   "dims": [["K1", SHAPES[a.shape][0]], ["N1", tp4.N1], ["N2", SHAPES[a.shape][1]]],
                   "one_gpu_ms": {str(M): r[M]["one_gpu"] for M in tokens}, "ideal_ms": {str(M): r[M]["ideal"] for M in tokens},
                   "results": results}, f, indent=1)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
