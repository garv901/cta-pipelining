# Phase 2c: CTAPP tile-scoreboard vs row-panel scoreboard, plus drift-check baselines. Run:
#   timeout 3000 env CUDA_VISIBLE_DEVICES=3,0 TMPDIR=/mnt/storage/garv901/cta-pipelining/build python3 -u bench/rowpanel.py
# (torch dev 0 = physical GPU 3 = A, dev 1 = physical GPU 0 = B). Writes results/rowpanel.csv and build/rowpanel_smi.txt; results/rowpanel.md is made from them.
import os, sys, csv, gc, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from ctapp.ext import ROOT
from ctapp.timing import measure
from ctapp.methods import *

# reuse the clean-GPU helpers of fig5.py without running its sweep
src = open(f"{ROOT}/bench/fig5.py").read()
exec(src[src.index("UUIDS ="):src.index("MS = [")])
MS = [1024, 2048, 4096, 8192, 16384, 32768]
CHUNKS = [256, 512, 1024, 2048, 4096, 8192]
CTAPP = [(7, 7, 8), (7, 7, 1), (4, 4, 8), (6, 6, 8)]
TILE_M = {4: 64, 6: 64, 7: 128}

def sweep(M, rows):
    torch.manual_seed(0)
    X = torch.randn(M, K, device="cuda:0", dtype=BF); W1 = torch.randn(N, K, device="cuda:0", dtype=BF) / K**0.5
    W2b = torch.randn(N, K, device="cuda:1", dtype=BF) / K**0.5; W2a = W2b.to("cuda:0")
    def go(name, variant, m):
        r = measure(m); rows.append([M, name, variant, *("%.1f" % v for v in r)]); print(rows[-1], flush=True)
    go("OneGpuCublas", "", OneGpuCublas(M, X, W1, W2a))
    for c in [c for c in CHUNKS if c < M] + [M]: go("MicroBatchCublas", f"chunk={c}", MicroBatchCublas(M, X, W1, W2b, c))
    go("TensorParallel2", "", TensorParallel2(M, X, W1, W2b))
    for cp, cc, g in CTAPP:
        if M % TILE_M[cp] or M % TILE_M[cc] or (M // TILE_M[cp]) % g: continue
        for sb in ("tile", "rowpanel"): go("Ctapp", f"cfg{cp}/{cc} group{g} {sb}", Ctapp(M, X, W1, W2b, cp, cc, g, scoreboard=sb))

allrows, log = [], []
for M in MS:
    for attempt in range(4):
        ok = wait_clean()
        s0 = smi(); rows = []
        sweep(M, rows)
        s1 = smi(); f = foreign()
        log.append((M, attempt, s0, s1, f))
        if not f and ok: break
        print(f"M={M} attempt {attempt} contaminated / not clean: {f}; rerun", flush=True)
        gc.collect(); torch.cuda.empty_cache()
    else: raise SystemExit("could not get a clean run")
    allrows += rows; gc.collect(); torch.cuda.empty_cache()

with open(f"{ROOT}/results/rowpanel.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["M", "method", "variant", "median_us", "p10_us", "p90_us"]); w.writerows(allrows)
with open(f"{ROOT}/build/rowpanel_smi.txt", "w") as fh:
    for M, a, s0, s1, f in log: fh.write(f"=== M={M} attempt {a} clean={not f}\n--- before\n{s0}--- after\n{s1}foreign after: {f}\n")
