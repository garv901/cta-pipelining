# Run: timeout 3000 env CUDA_VISIBLE_DEVICES=<A>,<B> python3 -u bench/fig5.py   (torch dev 0 = producer GPU A, dev 1 = consumer GPU B; see scripts/pick_gpus.sh)
import argparse, os, subprocess, sys, time, csv, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from ctapp.ext import ROOT
from ctapp.timing import measure
from ctapp.methods import *

def _uuid_map():
    out = subprocess.run("nvidia-smi --query-gpu=index,uuid --format=csv,noheader", shell=True, capture_output=True, text=True).stdout
    return dict(l.split(", ", 1) for l in out.splitlines() if ", " in l)
def _uuids():  # physical GPUs in use; CUDA_VISIBLE_DEVICES may hold UUIDs, or indices nvidia-smi does not list (Slurm remap)
    m = _uuid_map(); vis = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
    return [m[v].strip() if v in m else v for v in vis if v in m or v.startswith(("GPU-", "MIG-"))]  # unmapped indices: ignored
UUIDS = _uuids()
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

CHUNKS = [256, 512, 1024, 2048, 4096, 8192]
CTAPP = [(7, 7, 8), (7, 7, 1), (4, 4, 8), (6, 6, 8)]
TILE_M = {4: 64, 6: 64, 7: 128}
# shape -> (K, N1 (= I for swiglu), N2, act)
SHAPES = dict(paper=(8192, 8192, 8192, None), l70b=(8192, 28672, 8192, "swiglu"), l70b_qkv=(28672, 8192, 10240, None))
ALL_METHODS = "onegpu,mb,tp2,ctapp,onegpu_tma,mb_tma"
ap = argparse.ArgumentParser()
ap.add_argument("--shape", choices=list(SHAPES), default="paper")
ap.add_argument("--methods", default=ALL_METHODS)
ap.add_argument("--ms", default="1024,2048,4096,8192,16384,32768")
ap.add_argument("--out", default=None)
ap.add_argument("--reps", type=int, default=20)
args = ap.parse_args()
KD, N1, N2, ACT = SHAPES[args.shape]
METHODS = set(args.methods.split(",")); assert METHODS <= set(ALL_METHODS.split(",")), METHODS
if ACT: METHODS -= {"ctapp", "onegpu_tma", "mb_tma"}  # TMA / CTAPP kernels are plain-GEMM only
MS = [int(m) for m in args.ms.split(",")]
OUT = args.out or f"{ROOT}/results/" + ("fig5_h100.csv" if args.shape == "paper" else f"fig5_{args.shape}.csv")
H = N1 // 2 if ACT == "swiglu" else N1  # width fed to GEMM2

def sweep(M, rows):
    torch.manual_seed(0)
    X = torch.randn(M, KD, device="cuda:0", dtype=BF); W1 = torch.randn(N1, KD, device="cuda:0", dtype=BF) / KD**0.5
    W2b = torch.randn(N2, H, device="cuda:1", dtype=BF) / H**0.5; W2a = W2b.to("cuda:0")
    def go(name, variant, m):
        r = measure(m, reps=args.reps); rows.append([M, name, variant, *("%.1f" % v for v in r)]); print(rows[-1], flush=True)
    chunks = [c for c in CHUNKS if c < M] + [M]
    if "onegpu" in METHODS: go("OneGpuCublas", "", OneGpuCublas(M, X, W1, W2a, act=ACT))
    if "mb" in METHODS:
        for c in chunks: go("MicroBatchCublas", f"chunk={c}", MicroBatchCublas(M, X, W1, W2b, c, act=ACT))
    if "tp2" in METHODS: go("TensorParallel2", "", TensorParallel2(M, X, W1, W2b, act=ACT))
    if "ctapp" in METHODS:
        for cp, cc, g in CTAPP:
            if M % TILE_M[cp] or M % TILE_M[cc] or (M // TILE_M[cp]) % g: continue
            go("Ctapp", f"cfg{cp}/{cc} group{g}", Ctapp(M, X, W1, W2b, cp, cc, g))
    for cfg in (4, 7):
        if "onegpu_tma" in METHODS: go("OneGpuTma", f"cfg{cfg}", OneGpuTma(M, X, W1, W2a, cfg))
        if "mb_tma" in METHODS:
            for c in chunks: go("MicroBatchTma", f"cfg{cfg} chunk={c}", MicroBatchTma(M, X, W1, W2b, c, cfg))

allrows, log = [], []
class Empty:
    def prepare(self): pass
    def enqueue(self): pass
wait_clean(); floor = measure(Empty(), reps=100)
print('shape', args.shape, KD, N1, N2, ACT, 'methods', sorted(METHODS), flush=True)
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

with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["M", "method", "variant", "median_us", "p10_us", "p90_us"]); w.writerows(allrows)
with open(f"{ROOT}/build/fig5_smi" + ("" if args.shape == "paper" else "_" + args.shape) + ".txt", "w") as fh:
    fh.write("harness floor %.1f %.1f %.1f\n" % floor)
    for M, a, s0, s1, f in log: fh.write(f"=== M={M} attempt {a} clean={not f}\n--- before\n{s0}--- after\n{s1}foreign after: {f}\n")
print("floor", floor)
