# Run: CUDA_VISIBLE_DEVICES=3,0 python3 bench/gate.py   (torch dev 0 = physical GPU 3, dev 1 = physical GPU 0)
import os, subprocess, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ctapp.ext import load, ROOT
import torch
import torch.nn.functional as F

ext = load()
ext.enable_peer_access(0, 1); ext.enable_peer_access(1, 0)
torch.manual_seed(0)
N = K = 8192
CFGS = range(4)
dev, peer = torch.device("cuda:0"), torch.device("cuda:1")
out = []
def p(s=""): print(s); out.append(s)
def table(head, rows):
    p("| " + " | ".join(head) + " |"); p("|" + "---|" * len(head))
    for r in rows: p("| " + " | ".join(str(c) for c in r) + " |")
    p()

def bench(fn, sync_dev=dev):
    with torch.cuda.device(sync_dev):
        for _ in range(5): fn()
        ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(20)]
        for s, e in ev: s.record(); fn(); e.record()
        torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in ev) * 1e3  # us

smi = subprocess.run("nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv", shell=True, capture_output=True, text=True).stdout
commit = open(f"{ROOT}/third_party/CUTLASS_COMMIT").read().strip()
W = torch.randn(N, K, device=dev, dtype=torch.bfloat16) / K**0.5

# a. correctness
X = torch.randn(1024, K, device=dev, dtype=torch.bfloat16)
ref = F.linear(X, W)
rows = []
for c in CFGS:
    Y = torch.empty_like(ref); ext.gemm_tma(X, W, Y, c)
    d = (Y.float() - ref.float()).abs()
    rows.append([ext.config_info(c), f"{d.max().item():.4g}", f"{(d / ref.float().abs().clamp_min(1e-2)).max().item():.4g}"])
p("## Correctness (M=1024, vs F.linear; rel = |d|/max(|ref|,1e-2))")
table(["config", "max abs err", "max rel err"], rows)

# b. throughput
rows, best = [], {}
for M in [1024, 2048, 4096, 8192, 16384, 32768]:
    X = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    Y = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    tf = lambda t: 2 * M * N * K / t / 1e6
    tb = bench(lambda: torch.mm(X, W.t(), out=Y))
    ts = [bench(lambda: ext.gemm_tma(X, W, Y, c)) for c in CFGS]
    bc = min(CFGS, key=lambda c: ts[c]); best[M] = bc
    rows.append([M, f"{tb:.0f} / {tf(tb):.0f}"] + [f"{t:.0f} / {tf(t):.0f}" for t in ts] + [bc, f"{tb / ts[bc]:.3f}"])
p("## Throughput, N=K=8192 (median us / TFLOP/s); cuBLAS = torch.mm(X, W.t(), out=Y)")
table(["M", "cuBLAS"] + [f"cfg{c} " + ext.config_info(c).split()[1] for c in CFGS] + ["best cfg", "best/cuBLAS speed ratio"], rows)

# c. remote-store cost
rows = []
for M in [4096, 16384]:
    c = best[M]
    X = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    Yl = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    Yp = torch.empty(M, N, device=peer, dtype=torch.bfloat16)
    tl = bench(lambda: ext.gemm_tma(X, W, Yl, c)); tp = bench(lambda: ext.gemm_tma(X, W, Yp, c))
    assert torch.equal(Yl, Yp.to(dev))
    gb = lambda t: 2 * M * N / t / 1e3
    rows.append([M, c, f"{tl:.0f}", f"{tp:.0f}", f"{tp / tl:.3f}", f"{gb(tl):.0f}", f"{gb(tp):.0f}"])
p("## Remote-store cost (best config, kernel on GPU 3)")
table(["M", "cfg", "local us", "peer us", "peer/local", "local write GB/s", "peer write GB/s"], rows)

# d. P2P bandwidth
nb = 256 << 20
rows = []
for name, s, d in [("3->0", dev, peer), ("0->3", peer, dev)]:
    src = torch.empty(nb, device=s, dtype=torch.uint8); dst = torch.empty(nb, device=d, dtype=torch.uint8)
    t_ce = bench(lambda: dst.copy_(src), s)
    t_sm = bench(lambda: ext.p2p_store_bw_kernel(src, dst), s)
    rows.append([name, f"{nb / t_ce / 1e3:.1f}", f"{nb / t_sm / 1e3:.1f}"])
p("## P2P bandwidth, 256 MiB (GB/s)")
table(["direction", "copy engine", "SM stores (16B)"], rows)

os.makedirs(f"{ROOT}/results", exist_ok=True)
with open(f"{ROOT}/results/gate.md", "w") as f:
    f.write("# Phase 0 gate\n\nnvidia-smi before run:\n```\n" + smi + "```\nCUTLASS: " + commit + "\n\n" + "\n".join(out))
