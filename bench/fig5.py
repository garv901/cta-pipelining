# Run: timeout 3000 env CUDA_VISIBLE_DEVICES=3,0 python3 -u bench/fig5.py   (torch dev 0 = physical GPU 3 = A, dev 1 = physical GPU 0 = B)
import os, subprocess, sys, time, csv, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from ctapp.ext import ROOT
from ctapp.timing import measure
from ctapp.methods import *

UUIDS = ["GPU-e82ab913-2d6b-1a55-89a8-9ad23a5a09cf", "GPU-cfd9ce0a-b7af-895a-f54c-4f175b610977"]  # physical GPU 0 (B) and 3 (A)
def smi():
    a = subprocess.run("nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv", shell=True, capture_output=True, text=True).stdout
    b = subprocess.run("nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv", shell=True, capture_output=True, text=True).stdout
    return a + b
def foreign():
    out = subprocess.run("nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader", shell=True, capture_output=True, text=True).stdout
    return [l for l in out.splitlines() if any(u in l for u in UUIDS) and int(l.split(",")[1]) != os.getpid()]
def wait_clean():
    for _ in range(30):
        if not foreign(): return True
        time.sleep(60)
    return False

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
        go("Ctapp", f"cfg{cp}/{cc} group{g}", Ctapp(M, X, W1, W2b, cp, cc, g))
    for cfg in (4, 7):
        go("OneGpuTma", f"cfg{cfg}", OneGpuTma(M, X, W1, W2a, cfg))
        for c in [c for c in CHUNKS if c < M] + [M]: go("MicroBatchTma", f"cfg{cfg} chunk={c}", MicroBatchTma(M, X, W1, W2b, c, cfg))

allrows, log = [], []
class Empty:
    def prepare(self): pass
    def enqueue(self): pass
wait_clean(); floor = measure(Empty(), reps=100)
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

with open(f"{ROOT}/results/fig5_h100.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["M", "method", "variant", "median_us", "p10_us", "p90_us"]); w.writerows(allrows)
with open(f"{ROOT}/build/fig5_smi.txt", "w") as fh:
    fh.write("harness floor %.1f %.1f %.1f\n" % floor)
    for M, a, s0, s1, f in log: fh.write(f"=== M={M} attempt {a} clean={not f}\n--- before\n{s0}--- after\n{s1}foreign after: {f}\n")
print("floor", floor)
