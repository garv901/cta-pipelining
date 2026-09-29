# Driver:  timeout 900 env CUDA_VISIBLE_DEVICES=3,0 python3 bench/fig5_nsys.py     (picks best variants from results/fig5_h100.csv, profiles M=16384)
# Workload: python3 bench/fig5_nsys.py --run <ctapp|mb|tp> <variant...>  (5 gated reps each, NVTX range around each enqueue)
import os, sys, csv, io, json, subprocess, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = 16384
if len(sys.argv) > 1 and sys.argv[1] == "--run":
    import torch
    from ctapp.timing import measure
    from ctapp.methods import *
    torch.manual_seed(0)
    X = torch.randn(M, K, device="cuda:0", dtype=BF); W1 = torch.randn(N, K, device="cuda:0", dtype=BF) / K**0.5
    W2b = torch.randn(N, K, device="cuda:1", dtype=BF) / K**0.5
    class Tag:
        def __init__(self, m, name): self.m, self.name, self.i = m, name, 0
        def prepare(self): self.m.prepare()
        def enqueue(self):
            torch.cuda.nvtx.range_push(f"{self.name}/{self.i}"); self.i += 1; self.m.enqueue(); torch.cuda.nvtx.range_pop()
    kind, v = sys.argv[2], sys.argv[3:]
    m = {"ctapp": lambda: Ctapp(M, X, W1, W2b, int(v[0]), int(v[1]), int(v[2])), "mb": lambda: MicroBatchCublas(M, X, W1, W2b, int(v[0])),
         "tp": lambda: TensorParallel2(M, X, W1, W2b)}[kind]()
    print("HARNESS", kind, *("%.1f" % x for x in measure(Tag(m, kind), reps=5, warmup=1)))  # 1 ungated + 1 warmup + 5 timed enqueues (last 5 used)
    sys.exit()

rows = [r for r in csv.DictReader(open(f"{ROOT}/results/fig5_h100.csv")) if int(r["M"]) == M]
best = lambda meth: min((r for r in rows if r["method"] == meth), key=lambda r: float(r["median_us"]))
bc, bm = best("Ctapp"), best("MicroBatchCublas")
cp, cc, g = bc["variant"].replace("cfg", "").replace("/", " ").replace("group", "").split()
runs = [("ctapp", ["ctapp", cp, cc, g], bc), ("mb", ["mb", bm["variant"].split("=")[1]], bm), ("tp", ["tp"], best("TensorParallel2"))]
out = ["| method | variant | harness median us (fig5 sweep) | harness median us (in nsys run) | nsys span us per rep (median, min, max) |", "|---|---|---|---|---|"]
def stats_csv(rep, f):
    t = subprocess.run(f"/usr/local/bin/nsys stats --report {rep} --format csv --force-export=true {f}", shell=True, capture_output=True, text=True, cwd=ROOT + "/build").stdout
    return list(csv.DictReader(io.StringIO(t[t.index("Start (ns)"):])))
for kind, args, ref in runs:
    base = f"{ROOT}/build/nsys_{kind}"
    p = subprocess.run(f"/usr/local/bin/nsys profile -o {base} --force-overwrite=true --trace=cuda,nvtx timeout 600 python3 -u {os.path.abspath(__file__)} --run {' '.join(args)}", shell=True, capture_output=True, text=True, cwd=ROOT + "/build")
    hn = [l for l in p.stdout.splitlines() if l.startswith("HARNESS")][0].split()[2]
    gpu = stats_csv("cuda_gpu_trace", base + ".nsys-rep"); api = stats_csv("cuda_api_trace", base + ".nsys-rep"); nv = stats_csv("nvtx_pushpop_trace", base + ".nsys-rep")
    start_api = {r["CorrID"]: int(r["Start (ns)"]) for r in api}
    rng = sorted((int(r["Start (ns)"]), int(r["Start (ns)"]) + int(r["Duration (ns)"])) for r in nv if r["Name"].lstrip(":").startswith(kind + "/"))[-5:]
    spans = []
    for lo, hi in rng:
        k = [r for r in gpu if lo <= start_api.get(r["CorrId"], -1) <= hi]
        spans.append((max(int(r["Start (ns)"]) + int(r["Duration (ns)"]) for r in k) - min(int(r["Start (ns)"]) for r in k)) / 1e3)
        print(kind, "range", (lo, hi), "gpu ops", len(k), "span us %.1f" % spans[-1], flush=True)
    out.append(f"| {kind} | {ref['variant'] or '-'} | {float(ref['median_us']):.0f} | {hn} | {statistics.median(spans):.0f}, {min(spans):.0f}, {max(spans):.0f} |")
open(f"{ROOT}/build/nsys_table.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
