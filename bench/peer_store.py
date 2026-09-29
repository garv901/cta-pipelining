# Run: timeout 600 env CUDA_VISIBLE_DEVICES=3,0 python3 bench/peer_store.py   (torch dev 0 = physical GPU 3, dev 1 = physical GPU 0)
import os, subprocess, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ctapp.ext import load, ROOT
import torch

ext = load()
ext.enable_peer_access(0, 1); ext.enable_peer_access(1, 0)
torch.manual_seed(0)
N = K = 8192
dev, peer = torch.device("cuda:0"), torch.device("cuda:1")
out = []
def p(s=""): print(s); out.append(s)
def table(head, rows):
    p("| " + " | ".join(head) + " |"); p("|" + "---|" * len(head))
    for r in rows: p("| " + " | ".join(str(c) for c in r) + " |")
    p()

def bench(fn):
    for _ in range(5): fn()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(20)]
    for s, e in ev: s.record(); fn(); e.record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in ev) * 1e3  # us

smi = subprocess.run("nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv", shell=True, capture_output=True, text=True).stdout
W = torch.randn(N, K, device=dev, dtype=torch.bfloat16) / K**0.5
rows = []
for M in [4096, 16384]:
    X = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    Yl = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    Yp = torch.empty(M, N, device=peer, dtype=torch.bfloat16)
    ref = torch.empty_like(Yl); ext.gemm_tma(X, W, ref, 2)
    for c in [2, 1, 4, 5]:
        tl = bench(lambda: ext.gemm_tma(X, W, Yl, c)); tp = bench(lambda: ext.gemm_tma(X, W, Yp, c))
        assert torch.equal(Yl, Yp.to(dev))
        same = torch.equal(Yl, ref)
        rows.append([M, c, ext.config_info(c).split(" threads")[0], f"{tl:.0f}", f"{2*M*N*K/tl/1e6:.0f}", f"{tp:.0f}", f"{tp/tl:.3f}", f"{2*M*N/tp/1e3:.0f}", same])
p("## Local vs peer D (kernel on GPU 3, D on GPU 0 for peer)")
table(["M", "cfg", "config", "local us", "local TFLOP/s", "peer us", "peer/local", "peer write GB/s", "bit-equal to cfg2"], rows)
with open(f"{ROOT}/results/peer_store.md", "w") as f:
    f.write("# Part 1a: smem-staged vectorized epilogue\n\nnvidia-smi before run:\n```\n" + smi + "```\n\n" + "\n".join(out))
