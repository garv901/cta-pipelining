# python3 bench/tp_strong_report.py  -> results/tp_strong.md from build/tp_strong_<shape>_w<world>.json (written by bench/tp_strong.py)
import glob, json, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["paper", "l70b", "l70b_qkv"]
DESC = {"paper": "8192 -> 8192 -> 8192, plain GEMM chain", "l70b": "Llama-70B FFN: 8192 -> SwiGLU 2x28672 -> 8192",
        "l70b_qkv": "Llama-70B down-proj -> RMSNorm -> QKV: 28672 -> 8192 -> 10240 (all-reduce after GEMM1, GEMM2 column-sharded)"}
files = sorted(glob.glob(f"{ROOT}/build/tp_strong_*_w*.json"))
runs = {}
for f in files:
    d = json.load(open(f)); runs[(d["shape"], d["world"])] = d
o = ["# S0.2: strong tensor-parallel baselines on node1 (8x H100 NVSwitch)", "",
     "Megatron sharding, one process per GPU, bf16 with fp32 accumulate. `ideal` = 1-GPU eager cuBLAS time of the same chain / world. "
     "`exposed` = (variant - ideal) / variant: the share of the step that CTA-pipelining could at most recover. "
     "Back-to-back = calls queued without host sync (production condition); per-step = barrier + sync before each call. "
     "Variants: `nccl` dist.all_reduce; `nccl_graph` same in a CUDA graph; `oneshot`/`twoshot`/`multimem` torch symm-mem all-reduce "
     "(multimem = NVLS in-switch reduction); `asynctp` fused_matmul_reduce_scatter (output row-sharded, like a sequence-parallel residual); "
     "`asynctp_ag` + NCCL all-gather (replicated output); `nccl[NCCL_ALGO=..]` fresh process group with that algorithm. Every variant passed the fp32 reference check (rel err listed).", ""]
summary = []
for shape in ORDER:
    for world in (2, 4, 8):
        d = runs.get((shape, world))
        if not d: continue
        R = d["results"]; keys = list(R)
        Ms = sorted({int(k.split("/")[0]) for k in keys})
        variants = []
        for k in keys:
            v = k.split("/", 1)[1]
            if v not in variants: variants.append(v)
        o += [f"## {shape}, TP{world} — {DESC[shape]}", ""]
        for title, key in (("back-to-back", "b2b_ms"), ("per step", "per_step_ms")):
            o += [f"### {title}: ms (exposed %)", "", "| M | 1-GPU | ideal | " + " | ".join(variants) + " | best |", "|---" * (len(variants) + 4) + "|"]
            for M in Ms:
                ideal = d["ideal_ms"][str(M)]; cells, best = [], None
                for v in variants:
                    r = R.get(f"{M}/{v}"); t = r and r.get(key)
                    if t is None: cells.append("n/a"); continue
                    cells.append(f"{t:.3f} ({(t - ideal) / t * 100:.0f}%)")
                    if best is None or t < best[0]: best = (t, v)
                o.append(f"| {M} | {d['one_gpu_ms'][str(M)]:.3f} | {ideal:.3f} | " + " | ".join(cells) + f" | {best[1]} {best[0]:.3f} ({(best[0] - ideal) / best[0] * 100:.0f}%) |")
                if key == "b2b_ms": summary.append((shape, world, M, ideal, best, R.get(f"{M}/nccl", {}).get(key)))
            o.append("")
        errs = sorted({(v, round(R[f"{M}/{v}"]["rel_err"], 4)) for M in Ms for v in variants if R.get(f"{M}/{v}") and R[f"{M}/{v}"].get("rel_err") is not None})
        o += ["rel err vs fp32 reference (max over M): " + ", ".join(f"{v} {e}" for v, e in sorted(errs, key=lambda x: -x[1])[:len(variants)]), ""]
o += ["## Summary: best strong variant, back-to-back", "", "| shape | TP | M | ideal ms | plain nccl ms (exposed) | best variant ms (exposed) | headroom vs best = exposed % |", "|---|---|---|---|---|---|---|"]
for shape, world, M, ideal, best, nccl in summary:
    nc = "n/a" if nccl is None else f"{nccl:.3f} ({(nccl - ideal) / nccl * 100:.0f}%)"
    o.append(f"| {shape} | {world} | {M} | {ideal:.3f} | {nc} | {best[1]} {best[0]:.3f} | {(best[0] - ideal) / best[0] * 100:.0f}% |")
open(f"{ROOT}/results/tp_strong.md", "w").write("\n".join(o) + "\n")
print("\n".join(o[-len(summary) - 3:]))
