# python3 bench/fig5_report.py  -> results/fig5_h100.md from results/fig5_h100.csv, build/fig5_smi.txt, build/nsys_table.md
# usage: python3 bench/fig5_report.py [--shape paper|l70b|l70b_qkv] [--csv PATH] [--md PATH]
import argparse, csv, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--shape", default="paper"); ap.add_argument("--csv", default=None); ap.add_argument("--md", default=None)
args = ap.parse_args()
paper = args.shape == "paper"
base = "fig5_h100" if paper else f"fig5_{args.shape}"
CSV = args.csv or f"{ROOT}/results/{base}.csv"; MD = args.md or CSV[:-4] + ".md"
SHAPE = dict(paper="N=K=8192 plain", l70b="K=8192, SwiGLU I=28672, N2=8192", l70b_qkv="K=28672, N1=8192, N2=10240 plain")[args.shape]
rows = list(csv.DictReader(open(CSV)))
smi_path = f"{ROOT}/build/fig5_smi" + ("" if paper else f"_{args.shape}") + ".txt"
smi = open(smi_path).read() if os.path.exists(smi_path) else "harness floor n/a\n"
have = {r["method"] for r in rows}
MS = sorted({int(r["M"]) for r in rows})
PAPER_MB = dict(zip([1024, 2048, 4096, 8192, 16384, 32768], [1.3, 9.1, 18.3, 26.3, 31.8, 22.7]))
PAPER_TP = dict(zip([1024, 2048, 4096, 8192, 16384, 32768], [1.9, 6.3, 23.9, 25.1, 29.0, 29.6]))
def sel(M, meth, pre=""): return [r for r in rows if int(r["M"]) == M and r["method"] == meth and r["variant"].startswith(pre)]
def best(M, meth, pre=""):
    r = min(sel(M, meth, pre), key=lambda r: float(r["median_us"])); return float(r["median_us"]), r["variant"]
red = lambda a, b: f"{(1 - a / b) * 100:+.1f}"
o = [f"# Phase 2: Fig-5-style sweep on 2x H100 (producer = physical GPU 3, consumer = physical GPU 0), {SHAPE} BF16, 2 layers", "",
     "Latency = gated end-to-end (cuStreamWaitValue32 on a pinned host flag; start event after A's gate wait, end event after A waits on B's done event), median of 20 reps after 5 warmup, us.",
     "Harness floor (empty method on A+B): " + smi.splitlines()[0].replace("harness floor", "median/p10/p90 us =") + " in the sweep script (100 reps); ~12 us in bench/harness_check.py (50 reps), the join is bimodal (~4.5 or ~12 us).", "",
     "Reduction = 1 - CTAPP/baseline, positive means CTAPP faster. Paper (B200) reference in the last column of each pair." if paper else "Reduction = 1 - CTAPP/baseline, positive means CTAPP faster.", ""]
T1 = ["## Table 1: paper-style (CTAPP uses our KernelTma kernels; baselines are cuBLAS)", "",
     "| M | CTAPP best (variant) | micro-batching best cuBLAS (chunk) | TP2 | 1-GPU cuBLAS | red. vs micro-batching % | paper % | red. vs TP % | paper % |" if paper else "| M | CTAPP best (variant) | micro-batching best cuBLAS (chunk) | TP2 | 1-GPU cuBLAS | red. vs micro-batching % | red. vs TP % |", "|---|---|---|---|---|---|---|---|---|" if paper else "|---|---|---|---|---|---|---|"]
T1R = {"Ctapp", "MicroBatchCublas", "TensorParallel2", "OneGpuCublas"}
if T1R <= have:
    o += T1
    for M in MS:
        c, cv = best(M, "Ctapp"); mb, mv = best(M, "MicroBatchCublas"); tp, _ = best(M, "TensorParallel2"); og, _ = best(M, "OneGpuCublas")
        pm = f" {PAPER_MB[M]} |" if paper and M in PAPER_MB else (" n/a |" if paper else "")
        pt = f" {PAPER_TP[M]} |" if paper and M in PAPER_TP else (" n/a |" if paper else "")
        o.append(f"| {M} | {c:.0f} ({cv}) | {mb:.0f} ({mv}) | {tp:.0f} | {og:.0f} | {red(c, mb)} |{pm} {red(c, tp)} |{pt}")
T2R = {"Ctapp", "MicroBatchTma", "OneGpuTma"}
if T2R <= have:
  o += ["", "## Table 2: like-for-like (same KernelTma kernel, same cfg)", "",
      "| M | CTAPP best (variant) | MicroBatchTma best (cfg, chunk) | OneGpuTma cfg4 | OneGpuTma cfg7 | red. vs MicroBatchTma % | red. vs sequential-2-GPU TMA (chunk=M, same cfg) % |", "|---|---|---|---|---|---|---|"]
  for M in MS:
    c, cv = best(M, "Ctapp"); cfg = cv.split("/")[0]  # e.g. cfg7
    mb, mv = best(M, "MicroBatchTma"); seq = float(sel(M, "MicroBatchTma", cfg + f" chunk={M}")[0]["median_us"])
    o.append(f"| {M} | {c:.0f} ({cv}) | {mb:.0f} ({mv}) | {best(M, 'OneGpuTma', 'cfg4')[0]:.0f} | {best(M, 'OneGpuTma', 'cfg7')[0]:.0f} | {red(c, mb)} | {red(c, seq)} |")
o += ["", "## All results (median / p10 / p90 us)", "", "| M | method | variant | median | p10 | p90 |", "|---|---|---|---|---|---|"]
o += [f"| {r['M']} | {r['method']} | {r['variant']} | {r['median_us']} | {r['p10_us']} | {r['p90_us']} |" for r in rows]
nsys = f"{ROOT}/build/nsys_table.md"
if paper and os.path.exists(nsys):
    o += ["", "## Nsight Systems cross-check, M=16384", "",
          "5 gated reps per method (harness in the profiled process, NVTX range around each enqueue). Span = earliest kernel/memcpy start to latest end over both GPUs, taken from `nsys stats --report cuda_gpu_trace`; ops are assigned to a rep by the NVTX range containing their launch API call (prepare() ops and gate waits are outside).", "",
          open(nsys).read()]
o += ["## nvidia-smi snapshots (before / after each M; a sweep with a foreign process on our GPUs afterwards is discarded and rerun)", "", "```", smi, "```", ""]
open(MD, "w").write("\n".join(o))
