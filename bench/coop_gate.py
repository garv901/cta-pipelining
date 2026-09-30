# Run: CUDA_VISIBLE_DEVICES=<gemm gpu>,<peer gpu> python3 bench/coop_gate.py part1|part2|report   (torch dev 0 = physical GPU 3, dev 1 = physical GPU 0)
import os, subprocess, statistics, sys, json, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ctapp.ext import load, ROOT
import torch

ext = load()
ext.enable_peer_access(0, 1); ext.enable_peer_access(1, 0)
torch.manual_seed(0)
N = K = 8192
MS = [1024, 2048, 4096, 8192, 16384, 32768]
CFGS = range(3)
RASTERS, SWIZZLES = (1, 2), (1, 2, 4, 8)
RN = {1: "AlongN", 2: "AlongM"}
LINK_GBS = 124.2
dev, peer = torch.device("cuda:0"), torch.device("cuda:1")
JS = f"{ROOT}/build/coop_gate.json"

def bench(fn):
    for _ in range(5): fn()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(20)]
    for s, e in ev: s.record(); fn(); e.record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in ev) * 1e3  # us

def smi():
    return subprocess.run("nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv",
                          shell=True, capture_output=True, text=True).stdout
def load_js(): return json.load(open(JS)) if os.path.exists(JS) else {}
def save_js(d): json.dump(d, open(JS, "w"), indent=1)
tf = lambda M, t: 2 * M * N * K / t / 1e6

class Obj:  # for ctapp.timing.measure
    def __init__(s, fn): s.fn = fn
    def prepare(s): pass
    def enqueue(s): s.fn()

def part1():
    d = load_js(); d["smi1"] = smi()
    W = torch.randn(N, K, device=dev, dtype=torch.bfloat16) / K**0.5
    rows = []
    # sanity
    X = torch.randn(4096, K, device=dev, dtype=torch.bfloat16); ref = torch.mm(X, W.t())
    san = []
    for c in CFGS:
        Y = torch.full_like(ref, float("nan")); ext.coop_gemm(X, W, Y, c, 1, 1)
        diff = (Y.float() - ref.float()).abs().max().item()
        san.append(dict(cfg=c, info=ext.coop_info(c), max_abs=diff, nan=bool(torch.isnan(Y).any()), close=bool(torch.allclose(Y.float(), ref.float(), rtol=2e-2, atol=2e-2))))
        print(san[-1], flush=True)
    d["sanity"] = san
    for M in MS:
        X = torch.randn(M, K, device=dev, dtype=torch.bfloat16); Y = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
        tb = bench(lambda: torch.mm(X, W.t(), out=Y))
        rows.append(dict(M=M, cfg="cublas", raster=0, swizzle=0, us=tb, tflops=tf(M, tb)))
        for c in CFGS:
            for r in RASTERS:
                for s in SWIZZLES:
                    t = bench(lambda: ext.coop_gemm(X, W, Y, c, r, s))
                    rows.append(dict(M=M, cfg=c, raster=r, swizzle=s, us=t, tflops=tf(M, t)))
        b = min((x for x in rows if x["M"] == M and x["cfg"] != "cublas"), key=lambda x: x["us"])
        print(M, f"cublas {tb:.0f}us best {b}", flush=True)
    d["rows"] = rows
    # cross-check with gated timing at M=16384
    from ctapp.timing import measure
    M = 16384; X = torch.randn(M, K, device=dev, dtype=torch.bfloat16); Y = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    b = min((x for x in rows if x["M"] == M and x["cfg"] != "cublas"), key=lambda x: x["us"])
    m_c = measure(Obj(lambda: torch.mm(X, W.t(), out=Y)), devs=(0,))
    m_k = measure(Obj(lambda: ext.coop_gemm(X, W, Y, b["cfg"], b["raster"], b["swizzle"])), devs=(0,))
    d["xcheck"] = dict(best=b, cublas_gated=m_c, coop_gated=m_k)
    print(d["xcheck"])
    save_js(d)

def part2():
    d = load_js(); d["smi2"] = smi(); d["pair"] = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    nb = 256 << 20  # SM-store P2P bandwidth on this pair, as in bench/gate.py
    src = torch.empty(nb, device=dev, dtype=torch.uint8); dst = torch.empty(nb, device=peer, dtype=torch.uint8)
    d["link_gbs"] = link_gbs = nb / bench(lambda: ext.p2p_store_bw_kernel(src, dst)) / 1e3
    print("pair", d["pair"], "link GB/s", link_gbs, flush=True); del src, dst
    W = torch.randn(N, K, device=dev, dtype=torch.bfloat16) / K**0.5
    rows = []
    for M in MS:
        cand = sorted((x for x in d["rows"] if x["M"] == M and x["cfg"] != "cublas"), key=lambda x: x["us"])
        picks, seen = [], set()
        for x in cand:
            if x["cfg"] not in seen: picks.append(x); seen.add(x["cfg"])
            if len(picks) == 2: break
        X = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        Yl = torch.empty(M, N, device=dev, dtype=torch.bfloat16); Yp = torch.empty(M, N, device=peer, dtype=torch.bfloat16)
        for x in picks:
            c, r, s = x["cfg"], x["raster"], x["swizzle"]
            tl = bench(lambda: ext.coop_gemm(X, W, Yl, c, r, s)); tp = bench(lambda: ext.coop_gemm(X, W, Yp, c, r, s))
            ok = torch.equal(Yl, Yp.to(dev))
            link = M * N * 2 / link_gbs / 1e3
            rows.append(dict(M=M, cfg=c, raster=r, swizzle=s, local_us=tl, peer_us=tp, link_us=link, bitequal=ok))
            print(rows[-1], flush=True)
    d["peer"] = rows; save_js(d)

def report():
    d = load_js(); rows = d["rows"]
    with open(f"{ROOT}/results/coop_gate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["M", "cfg", "raster", "swizzle", "us", "tflops"]); w.writeheader()
        for x in rows: w.writerow({k: (round(v, 2) if isinstance(v, float) else v) for k, v in x.items()})
    o = []
    def p(s=""): o.append(s)
    p("# Phase 3.0: stock CUTLASS cooperative persistent GEMM\n")
    p("nvidia-smi before 1-GPU run:\n```\n" + d["smi1"] + "```\n")
    p("## Config info\n")
    for c in CFGS: p(f"- cfg{c}: {ext.coop_info(c)}")
    p("\n## Sanity (M=4096, raster AlongN, swizzle 1, output pre-filled with NaN; allclose rtol=atol=2e-2 vs torch.mm)\n")
    p("| cfg | max abs diff | NaN | allclose |\n|---|---|---|---|")
    for s in d["sanity"]: p(f"| {s['info'].split()[2]} | {s['max_abs']:.4g} | {s['nan']} | {s['close']} |")
    p("\n## Throughput, N=K=8192 (median us / TFLOP/s; best over raster x swizzle per config)\n")
    p("| M | cuBLAS | " + " | ".join(f"cfg{c}" for c in CFGS) + " | best overall | best/cuBLAS speed ratio |\n|" + "---|" * (len(CFGS) + 4))
    g1 = []
    for M in MS:
        rm = [x for x in rows if x["M"] == M]; cb = next(x for x in rm if x["cfg"] == "cublas")
        cells, bests = [], {}
        for c in CFGS:
            b = min((x for x in rm if x["cfg"] == c), key=lambda x: x["us"]); bests[c] = b
            cells.append(f"{b['us']:.0f} / {b['tflops']:.0f} ({RN[b['raster']][-1]}{b['swizzle']})")
        b = min(bests.values(), key=lambda x: x["us"])
        ratio = cb["us"] / b["us"]; g1.append((M, ratio))
        p(f"| {M} | {cb['us']:.0f} / {cb['tflops']:.0f} | " + " | ".join(cells) + f" | cfg{b['cfg']} {RN[b['raster']]} sw{b['swizzle']} | {ratio:.3f} |")
    p("\nCell suffix: raster (N/M) + swizzle.")
    xc = d["xcheck"]
    p(f"\n## Cross-check M=16384 with ctapp.timing.measure (gated; median, p10, p90 us)\n\n- cuBLAS {xc['cublas_gated']}\n- coop {xc['coop_gated']} (cfg{xc['best']['cfg']} raster {xc['best']['raster']} swizzle {xc['best']['swizzle']}); event-timed: cuBLAS {next(x for x in rows if x['M']==16384 and x['cfg']=='cublas')['us']:.0f}, coop {xc['best']['us']:.0f}")
    p("\n## Gate G1: best/cuBLAS >= 0.85 at M >= 4096 (target 0.90)\n")
    p(", ".join(f"M={M}: {r:.3f}" for M, r in g1 if M >= 4096) + " -> " + ("PASS" if all(r >= 0.85 for M, r in g1 if M >= 4096) else "FAIL") + (" (target 0.90 met)" if all(r >= 0.90 for M, r in g1 if M >= 4096) else " (target 0.90 not met)"))
    if "peer" in d:
        p("\nnvidia-smi before peer run:\n```\n" + d["smi2"] + "```\n")
        p(d.get("peer_note", ""))
        a, b = d.get("pair", "3,0").split(",")
        p(f"## Peer D (kernel on physical GPU {a}, D on physical GPU {b}); link bound = M*8192*2 B / {d.get('link_gbs', LINK_GBS):.1f} GB/s (SM-store P2P, measured on this pair)\n")
        p("| M | cfg/raster/swizzle | local us | peer us | link bound us | peer/local | peer / max(local, link) | within 15% | D equal |\n|---|---|---|---|---|---|---|---|---|")
        for x in d["peer"]:
            ref = max(x["local_us"], x["link_us"]); q = x["peer_us"] / ref
            p(f"| {x['M']} | cfg{x['cfg']} {RN[x['raster']]} sw{x['swizzle']} | {x['local_us']:.0f} | {x['peer_us']:.0f} | {x['link_us']:.0f} | {x['peer_us']/x['local_us']:.3f} | {q:.3f} | {'yes' if q <= 1.15 else 'NO'} | {x['bitequal']} |")
    open(f"{ROOT}/results/coop_gate.md", "w").write("\n".join(o) + "\n")

{"part1": part1, "part2": part2, "report": report}[sys.argv[1]]()
