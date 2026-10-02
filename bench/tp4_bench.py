"""Phase-4 bench: TP down-proj -> resid -> RMSNorm -> QKV, CTA-pipelined (ctapp) vs baselines.

    python bench/tp4_bench.py --world 4 --tokens 1024,2048,4096,8192,16384 --iters 30 --warmup 5 --json build/tp4_4.json

Per M: one_gpu = eager bf16 chain on rank 0, ideal = one_gpu / world. Methods: ctapp (eager launches), ctapp_graph (CUDA graph
replay, one graph per epoch parity), ctapp_nowait (mode-0 GEMMs only: the raw two-GEMM floor), multimem, asynctp, nccl. Times: per-step (barrier+sync each call, median) and back-to-back (mean),
both max over ranks. exposed % = (t - ideal) / t.
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

METHODS = ["ctapp", "ctapp_graph", "ctapp_nowait", "multimem", "asynctp", "nccl"]
CTAPP = ("ctapp", "ctapp_graph")
BASE = ("multimem", "asynctp", "nccl")


def worker(rank, world, dev, gn, tokens, iters, warmup):
    Kr, N2r = tp4.K_FFN // world, tp4.N2 // world
    Wd, gamma, Wq = tp4.make_weights(dev)
    Wd_r, Wq_r, Wqf_r = tp4.shard_weights(Wd, Wq, gamma, rank, world)
    out = {}
    for M in tokens:
        h, resid = tp4.make_inputs(M, dev)
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
        for name in METHODS:
            tp4.barrier()
            try:
                if name.startswith("ctapp"):
                    b = tp4.CtappBoundary(M, Kr, N2r, rank, world, gn, dev, resid=True, protocol=(name != "ctapp_nowait"),
                                          use_graph=(name == "ctapp_graph"))
                    b.set_weights(Wd_r, Wqf_r)
                else:
                    cls = {"multimem": tp4.MultimemBoundary, "asynctp": tp4.AsyncTpBoundary, "nccl": tp4.NcclBoundary}[name]
                    b = cls(M, rank, world, gn, dev, Wd_r, Wq_r, gamma)
                y = b.forward(h_r, resid)
                if name == "ctapp_graph":   # check the replayed graphs (parity 0 and 1), not only the un-captured warm-up call
                    y = b.forward(h_r, resid)
                    y = b.forward(h_r, resid)
                torch.cuda.synchronize(dev)
                err = float("nan") if name == "ctapp_nowait" else ((y.float() - ref_r.float()).abs().max() / ref_r.abs().max()).item()
                e = torch.tensor([err], dtype=torch.float64)
                dist.all_reduce(e, op=dist.ReduceOp.MAX, group=tp4.GLOO)
                ps, bb = tp4.time_step(lambda: b.forward(h_r, resid), dev, iters, warmup)
                out[M][name] = {"per_step": ps, "b2b": bb, "rel_err": e.item()}
                if rank == 0:
                    print(f"M={M:6d} {name:13s} per-step {ps:8.3f}  b2b {bb:8.3f} ms  out rel err {e.item():.2e}", flush=True)
                del b, y
            except Exception as ex:  # noqa: BLE001
                out[M][name] = {"per_step": float("nan"), "b2b": float("nan"), "rel_err": float("nan")}
                if rank == 0:
                    print(f"M={M:6d} {name} FAILED {type(ex).__name__}: {str(ex)[:200]}", flush=True)
                torch.cuda.synchronize(dev)
                tp4.barrier()
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
    a = ap.parse_args()
    tokens = [int(t) for t in a.tokens.split(",")]
    res, bad = tp4.launch(worker, a.world, a.port, tokens, a.iters, a.warmup)
    if bad:
        print("bench failed")
        return 1
    r = res[0]
    nan = float("nan")
    for title, key in (("per step", "per_step"), ("back-to-back", "b2b")):
        print(f"\n== TP{a.world} down->resid->RMSNorm->QKV, ms, {title}; cell = ms (exposed %) ; ideal = 1-GPU/{a.world} ==")
        print("| M | one_gpu | ideal | " + " | ".join(METHODS) + " | best ctapp vs best baseline |")
        print("|---" * (len(METHODS) + 4) + "|")
        for M in tokens:
            d = r[M]
            cells = []
            for m in METHODS:
                v = d[m][key]
                cells.append("n/a" if math.isnan(v) else f"{v:.3f} ({(v - d['ideal']) / v * 100:.0f}%)")
            base = [d[m][key] for m in BASE if not math.isnan(d[m][key])]
            cv = [d[m][key] for m in CTAPP if not math.isnan(d[m][key])]
            spd = f"{min(base) / min(cv):.2f}x" if base and cv else "n/a"
            print(f"| {M} | {d['one_gpu']:.3f} | {d['ideal']:.3f} | " + " | ".join(cells) + f" | {spd} |")
    path = a.json or f"build/tp4_{a.world}.json"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        results = {}
        for M in tokens:
            for m in METHODS:
                v = r[M][m]
                results[f"{M}/{m}"] = {"per_step_ms": v["per_step"], "b2b_ms": v["b2b"], "rel_err": v["rel_err"]}
        json.dump({"shape": "l70b_qkv", "world": a.world, "iters": a.iters, "warmup": a.warmup,
                   "dims": [["K1", tp4.K_FFN], ["N1", tp4.N1]],
                   "one_gpu_ms": {str(M): r[M]["one_gpu"] for M in tokens}, "ideal_ms": {str(M): r[M]["ideal"] for M in tokens},
                   "results": results}, f, indent=1)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
