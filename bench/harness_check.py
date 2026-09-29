# Run: timeout 1500 env CUDA_VISIBLE_DEVICES=3,0 python3 bench/harness_check.py   (harness floor, gate sanity, per-method correctness at M=4096)
import os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from ctapp.timing import measure, ext
from ctapp.methods import *

print(subprocess.run("nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv", shell=True, capture_output=True, text=True).stdout)

class Empty:
    def prepare(self): pass
    def enqueue(self): pass
print("harness floor (empty, A+B) us: median/p10/p90 = %.1f / %.1f / %.1f" % measure(Empty(), reps=50))

# sanity 1: the GPUs wait for the gate. 200 tiny kernels enqueued, once fast and once with a slow host (sleep per launch).
a = torch.zeros(1024, device="cuda:0")
class Tiny:
    def __init__(self, sleep): self.sleep = sleep
    def prepare(self): pass
    def enqueue(self):
        for _ in range(200):
            a.add_(1)
            if self.sleep: time.sleep(self.sleep)
t0 = time.perf_counter(); r_fast = measure(Tiny(0), reps=10); t1 = time.perf_counter()
r_slow = measure(Tiny(1e-3), reps=10); t2 = time.perf_counter()
print("sanity 200 tiny kernels: fast host %.1f us (host wall/rep %.1f ms), 1ms/launch host %.1f us (host wall/rep %.1f ms)" % (r_fast[0], (t1-t0)/15*1e3, r_slow[0], (t2-t1)/15*1e3))

# sanity 2: start event is recorded after the gate opens: delay gate_set by 50 ms; start->end of an empty method must stay ~floor, and wall >= 50 ms.
sA = torch.cuda.current_stream(0)
ext.gate_set(0)
for d in (0, 1):
    with torch.cuda.device(d): ext.gate_wait()
s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
s.record(sA); e.record(sA)
time.sleep(0.05); q = s.query(); t = time.perf_counter(); ext.gate_set(1); torch.cuda.synchronize(0); torch.cuda.synchronize(1)
print("sanity gate: start event completed before gate opened (50 ms sleep)? %s; start->end after open: %.1f us; sync returned %.2f ms after gate_set" % (q, s.elapsed_time(e) * 1e3, (time.perf_counter() - t) * 1e3))

# correctness at M=4096
torch.manual_seed(0)
M = 4096
X = torch.randn(M, K, device="cuda:0", dtype=BF); W1 = torch.randn(N, K, device="cuda:0", dtype=BF) / K**0.5
W2b = torch.randn(N, K, device="cuda:1", dtype=BF) / K**0.5; W2a = W2b.to("cuda:0")
def run(m):
    m.prepare(); m.enqueue(); torch.cuda.synchronize(0); torch.cuda.synchronize(1); return m.result().to("cuda:0").clone()
ref = run(OneGpuCublas(M, X, W1, W2a))
def rel(y): return ((y.float() - ref.float()).norm() / ref.float().norm()).item()
refT = {c: run(OneGpuTma(M, X, W1, W2a, c)) for c in (4, 7)}
print("OneGpuTma vs cuBLAS rel err:", {c: "%.2e" % rel(y) for c, y in refT.items()})
for ch in (256, 1024, 4096):
    y = run(MicroBatchCublas(M, X, W1, W2b, ch)); print(f"MicroBatchCublas chunk={ch}: rel err {rel(y):.2e}, max abs diff {(y.float()-ref.float()).abs().max().item():.3g}, bit-equal {torch.equal(y, ref)}")
for c in (4, 7):
    for ch in (256, 4096):
        y = run(MicroBatchTma(M, X, W1, W2b, ch, c)); print(f"MicroBatchTma cfg{c} chunk={ch}: bit-equal to OneGpuTma {torch.equal(y, refT[c])}")
y = run(TensorParallel2(M, X, W1, W2b)); print(f"TensorParallel2: rel err {rel(y):.2e}")
for cp, cc, g in [(7, 7, 8), (7, 7, 1), (4, 4, 8), (6, 6, 8)]:
    for _ in range(3): y = run(Ctapp(M, X, W1, W2b, cp, cc, g))
    print(f"Ctapp ({cp},{cc},{g}): bit-equal to OneGpuTma cfg{cp} {torch.equal(y, refT[cp]) if cp in refT else 'n/a'}, rel err vs cuBLAS {rel(y):.2e}")
