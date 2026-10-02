# Run: timeout 600 env CUDA_VISIBLE_DEVICES=3,0 python3 tests/test_ctapp.py   (torch dev 0 = physical GPU 3 producer, dev 1 = physical GPU 0 consumer)
import os, subprocess, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from ctapp.pipeline import Pipeline

print(subprocess.run("nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv", shell=True, capture_output=True, text=True).stdout)
torch.manual_seed(0)
N = K = 8192
A, B = torch.device("cuda:0"), torch.device("cuda:1")
def weights(K, N1, N2): return (torch.randn(N1, K, device=A, dtype=torch.bfloat16) / K**0.5, torch.randn(N2, N1, device=B, dtype=torch.bfloat16) / N1**0.5)
W1, W2 = weights(K, N, N)
CFG64, CFG128 = 4, 5  # smem-epilogue 64x256x64 and 128x128x64
CFG64_2CTA, CFG128_2CTA = 8, 7  # 2 CTAs/SM: 64x256x32 (5 stages), 128x128x64 (3 stages)


def sequential(pl, X, W1=W1, W2=W2):
    N, N2 = W1.shape[0], W2.shape[0]
    with torch.cuda.device(A):
        Y1a = torch.empty(pl.M, N, device=A, dtype=torch.bfloat16); pl.ext.gemm_tma(X, W1, Y1a, pl.cfg1)
    Y1b = Y1a.to(B)
    with torch.cuda.device(B):
        Y2 = torch.empty(pl.M, N2, device=B, dtype=torch.bfloat16); pl.ext.gemm_tma(Y1b, W2, Y2, pl.cfg2)
    return Y1b, Y2


def case(M, cfg1, cfg2, runs, W1=W1, W2=W2, **kw):
    pl = Pipeline(M, W1.shape[1], W1.shape[0], W2.shape[0], cfg1, cfg2, **kw)
    X = torch.randn(M, W1.shape[1], device=A, dtype=torch.bfloat16)
    Y1_ref, Y2_ref = sequential(pl, X, W1, W2)
    torch.cuda.synchronize(A); torch.cuda.synchronize(B)
    lin = F.linear(F.linear(X, W1).to(B), W2).float()
    ok = True
    for r in range(runs):
        with torch.cuda.device(B): pl.Y1.fill_(float("nan")); pl.Y2.fill_(float("nan"))
        pl.run(X, W1, W2)
        torch.cuda.synchronize(A); torch.cuda.synchronize(B)
        if not (torch.equal(pl.Y1, Y1_ref) and torch.equal(pl.Y2, Y2_ref)):
            ok = False; print(f"  MISMATCH run {r}: Y1 {(pl.Y1 != Y1_ref).sum().item()} Y2 {(pl.Y2 != Y2_ref).sum().item()}")
    rel = ((pl.Y2.float() - lin).abs() / lin.abs().clamp_min(1e-2)).max().item()
    print(f"M={M} K={W1.shape[1]} N1={W1.shape[0]} N2={W2.shape[0]} prod cfg{cfg1} cons cfg{cfg2} {kw} runs={runs}: bit-identical={ok} max rel err vs torch chain={rel:.3g}")
    assert ok


for M, c1, c2, runs in [(1024, CFG64, CFG64, 50), (4096, CFG64, CFG64, 50), (16384, CFG64, CFG64, 20), (4096, CFG128, CFG64, 50)]:
    case(M, c1, c2, runs)
# fence F1 and grouped producer order, incl. 2-CTA/SM configs and BM1 != BM2
for M, c1, c2, runs, kw in [(4096, CFG64, CFG64, 50, dict(fence=1)), (4096, CFG128, CFG64, 50, dict(fence=1)), (16384, CFG64, CFG64, 20, dict(fence=1, group_rows=8)),
                            (4096, CFG64_2CTA, CFG64_2CTA, 50, dict(fence=1)), (4096, CFG128_2CTA, CFG128_2CTA, 50, dict(fence=1)),
                            (4096, CFG128_2CTA, CFG64_2CTA, 50, dict(fence=1, group_rows=8)), (16384, CFG128_2CTA, CFG128_2CTA, 20, dict(fence=1, group_rows=8)),
                            (16384, CFG64_2CTA, CFG64_2CTA, 20, dict(fence=1, group_rows=8)), (16384, CFG128_2CTA, CFG128_2CTA, 20, dict(fence=0, group_rows=8))]:
    case(M, c1, c2, runs, **kw)

# Phase 2c: row-panel scoreboard (Pipeline default); explicit scoreboard="tile" keeps the old path
for M, c1, c2, runs, kw in [(M, c, c, 50 if M < 16384 else 20, dict(fence=1, group_rows=g)) for M in (1024, 4096, 16384) for c, g in ((CFG128_2CTA, 1), (CFG128_2CTA, 8), (CFG64, 1))] + \
                            [(4096, CFG128_2CTA, CFG64_2CTA, 50, dict(fence=1)), (4096, CFG128_2CTA, CFG64_2CTA, 50, dict(fence=1, group_rows=8)),
                             (4096, CFG128_2CTA, CFG128_2CTA, 50, dict(fence=1, group_rows=8, scoreboard="tile"))]:
    case(M, c1, c2, runs, **kw)

# non-square chain: N2 != N1 (QKV/O-proj-like), and N2 < N1
W1n, W2n = weights(8192, 8192, 10240)
for M, c1, c2, runs, kw in [(1024, CFG128_2CTA, CFG128_2CTA, 50, dict(fence=1)), (4096, CFG128_2CTA, CFG128_2CTA, 50, dict(fence=1, group_rows=8)),
                            (4096, CFG128_2CTA, CFG64_2CTA, 50, dict(fence=1))]:
    case(M, c1, c2, runs, W1=W1n, W2=W2n, **kw)
W1s, W2s = weights(4096, 8192, 4096)
case(4096, CFG128_2CTA, CFG128_2CTA, 50, W1=W1s, W2=W2s, fence=1)

# negative controls (skip_wait): default config, and the chosen best variant (fence F1, grouped order, 2 CTAs/SM)
for cfg, kw in [(CFG64, {}), (CFG128_2CTA, dict(fence=1, group_rows=8))]:
  M = 16384
  pl = Pipeline(M, K, N, N, cfg, cfg, skip_wait=1, **kw)
  X = torch.randn(M, K, device=A, dtype=torch.bfloat16)
  Y1_ref, Y2_ref = sequential(pl, X)
  with torch.cuda.device(B): pl.Y1.fill_(float("nan"))
  pl.run(X, W1, W2)
  torch.cuda.synchronize(A); torch.cuda.synchronize(B)
  bad = (pl.Y2 != Y2_ref).sum().item(); nan = pl.Y2.isnan().sum().item()
  print(f"negative control skip_wait=1 cfg{cfg} {kw} M={M}: mismatching Y2 elements {bad} (NaN {nan}) of {pl.Y2.numel()} -> {'SENSITIVE' if bad else 'INSENSITIVE (test cannot catch races!)'}")

# rough sanity timing
pl = Pipeline(M, K, N, N, CFG64, CFG64)
def timed(fn):
    fn(); torch.cuda.synchronize(A); torch.cuda.synchronize(B)
    ts = []
    for _ in range(10):
        t = time.perf_counter(); fn(); torch.cuda.synchronize(A); torch.cuda.synchronize(B); ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e3
print(f"M={M} pipelined {timed(lambda: pl.run(X, W1, W2)):.2f} ms, sequential {timed(lambda: sequential(pl, X)):.2f} ms")
