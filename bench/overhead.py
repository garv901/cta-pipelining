# Run: timeout 600 env CUDA_VISIBLE_DEVICES=3,0 python3 bench/overhead.py   (torch dev 0 = physical GPU 3 producer, dev 1 = physical GPU 0 consumer)
import os, subprocess, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ctapp.ext import load, ROOT
from ctapp.pipeline import Pipeline
import torch
import torch.nn.functional as F

ext = load()
ext.enable_peer_access(0, 1); ext.enable_peer_access(1, 0)
torch.manual_seed(0)
N = K = 8192
A, B = torch.device("cuda:0"), torch.device("cuda:1")
out = []
def p(s=""): print(s); out.append(s)
def table(head, rows):
    p("| " + " | ".join(head) + " |"); p("|" + "---|" * len(head))
    for r in rows: p("| " + " | ".join(str(c) for c in r) + " |")
    p()

def bench(fn, pre=lambda: None):  # CUDA events on the current (producer) device, median us
    for _ in range(5): pre(); fn()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(20)]
    for s, e in ev: pre(); s.record(); fn(); e.record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in ev) * 1e3

def host_ms(fn):
    fn(); torch.cuda.synchronize(A); torch.cuda.synchronize(B)
    ts = []
    for _ in range(10):
        t = time.perf_counter(); fn(); torch.cuda.synchronize(A); torch.cuda.synchronize(B); ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e3

smi = subprocess.run("nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv", shell=True, capture_output=True, text=True).stdout
FENCES = {0: "current", 1: "F1", 2: "F0 (unsafe)"}
# label, config id (ids: 4 = 64x256x64 auto, 6 = 64x256x64 3 stages, 8 = 64x256x32 5 stages, 5 = 128x128 auto, 7 = 128x128 3 stages)
CFGS = [("64x256x64 auto", 4), ("64x256x64 s3", 6), ("64x256x32 s5", 8), ("128x128x64 auto", 5), ("128x128x64 s3", 7)]
W1 = torch.randn(N, K, device=A, dtype=torch.bfloat16) / K**0.5
W2 = torch.randn(N, K, device=B, dtype=torch.bfloat16) / K**0.5

p("## Occupancy (ctapp kernel)")
table(["cfg", "config", "CTAs/SM"], [[c, ext.config_info(c), ext.occupancy(c)] for _, c in CFGS])

ORDERS = {"row-major": 0, "grouped8": 8}
rows1, rows2 = [], []
for M in [4096, 16384]:
    X = torch.randn(M, K, device=A, dtype=torch.bfloat16)
    Yl = torch.empty(M, N, device=A, dtype=torch.bfloat16)
    Yp = torch.empty(M, N, device=B, dtype=torch.bfloat16)
    for label, c in CFGS:
        bm, bn = [int(v) for v in ext.config_info(c).split("tile ")[1].split(" ")[0].split("x")[:2]]
        waves = -(-M // bm) * -(-N // bn) / (132 * ext.occupancy(c))
        tl = bench(lambda: ext.gemm_tma(X, W1, Yl, c)); tp = bench(lambda: ext.gemm_tma(X, W1, Yp, c))
        tf = lambda t: f"{2*M*N*K/t/1e6:.0f}"
        rows1.append([M, label, f"{waves:.1f}", f"{tl:.0f}", tf(tl), f"{tp:.0f}", tf(tp)])
        for oname, gr in ORDERS.items():
            pl = Pipeline(M, N, K, c, c, fence=0, group_rows=gr)
            ts = {}
            for f in [-1] + list(FENCES):  # -1: same ctapp kernel/tile order, queue pop only, no dependency signalling
                pl.fence = f
                def sig():
                    with torch.cuda.device(pl.b): pl.src_b.fill_(-1); pl.head_b.zero_(); pl.tail_b.zero_()
                    with torch.cuda.device(pl.a): pl.head_a.zero_(); pl.sb.copy_(pl.sb_pristine)
                    torch.cuda.synchronize(B)
                with torch.cuda.device(A):
                    sg = f >= 0
                    ts[f] = bench(lambda: ext.ctapp_gemm(X, W1, pl.Y1, c, pl.src_a, pl.head_a, pl.tail_a, pl.dep_off if sg else None, pl.dep_cons if sg else None,
                                                         pl.sb if sg else None, pl.src_b if sg else None, pl.head_b if sg else None, pl.tail_b if sg else None,
                                                         pl.tn1, 0, max(f, 0), tiles_n2=pl.tn2, rowpanel=pl.rowpanel), pre=sig)
            rows2.append([M, label, oname, f"{ts[-1]:.0f}"] + sum([[f"{ts[f]:.0f}", f"{(ts[f]-tp)/waves:.1f}", f"{(ts[f]-ts[-1])/waves:.1f}"] for f in FENCES], []))
p("## Producer alone, no protocol")
table(["M", "cfg", "waves", "local D us", "TFLOP/s", "peer D us", "TFLOP/s"], rows1)
p("## Producer alone with signalling (consumer not launched). ovh/wave = (t_signal - t_peer_noprotocol) / waves, us; ovh/wave(q) = relative to 'queue only' (same kernel and same tile order, no signalling)")
table(["M", "cfg", "order", "queue-only us"] + sum([[f"{n} us", "ovh/wave", "ovh/wave(q)"] for n in FENCES.values()], []), rows2)

# Full 2-layer: pipelined (safe fences 0, 1) vs sequential (same kernels) vs cuBLAS sequential
rows3 = []
for M in [4096, 16384]:
    X = torch.randn(M, K, device=A, dtype=torch.bfloat16)
    for label, c in CFGS:
        def seq():
            with torch.cuda.device(A):
                Y1 = torch.empty(M, N, device=A, dtype=torch.bfloat16); ext.gemm_tma(X, W1, Y1, c)
            Y1b = Y1.to(B)
            with torch.cuda.device(B):
                Y2 = torch.empty(M, N, device=B, dtype=torch.bfloat16); ext.gemm_tma(Y1b, W2, Y2, c)
            return Y2
        Yref = seq(); torch.cuda.synchronize(A); torch.cuda.synchronize(B)
        def cub():
            Y1 = F.linear(X, W1); Y1b = Y1.to(B); return F.linear(Y1b, W2)
        ms_seq, ms_cub = host_ms(seq), host_ms(cub)
        for oname, gr in ORDERS.items():
            pl = Pipeline(M, N, K, c, c, group_rows=gr)
            r = [M, label, oname]
            for f in (0, 1):
                pl.fence = f
                eq = torch.equal(pl.run(X, W1, W2), Yref)
                r += [f"{host_ms(lambda: pl.run(X, W1, W2)):.2f}", eq]
            rows3.append(r + [f"{ms_seq:.2f}", f"{ms_cub:.2f}"])
p("## 2-layer pipeline, host-timed median of 10, ms (bit-equal = vs same-kernel sequential)")
table(["M", "cfg", "order", "pipe F-current", "bit-equal", "pipe F1", "bit-equal", "sequential (same kernels)", "cuBLAS sequential"], rows3)
smi2 = subprocess.run("nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv", shell=True, capture_output=True, text=True).stdout
with open(f"{ROOT}/results/overhead.md", "w") as f:
    f.write("# Phase 1c: protocol overhead\n\nnvidia-smi before run:\n```\n" + smi + "```\nnvidia-smi after run:\n```\n" + smi2 + "```\n\n" + "\n".join(out))
