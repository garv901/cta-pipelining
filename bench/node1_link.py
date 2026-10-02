# Step 0 / S0.1: link characterisation of the GPUs visible to this process (dev 0 = source, the rest = peers).
# Run inside the Slurm allocation: srun --jobid=<id> --overlap bash -c 'export PATH=<venv>/bin:$PATH CUDA_HOME=... TMPDIR=$PWD/build CUDA_VISIBLE_DEVICES=0,1,2,3; python bench/node1_link.py'
import os, subprocess, statistics, sys, json, ctypes
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from ctapp.ext import load, ROOT

ext = load()
ND = torch.cuda.device_count()
PEERS = list(range(1, ND))
for p in PEERS: ext.enable_peer_access(0, p); ext.enable_peer_access(p, 0)
NB = 256 << 20
TF = 700e12                      # BF16 dense peak used in PLAN.md models
BM, BN, SMS = 128, 256, 132      # coop cfg0 tile, H100 SXM SM count
OUT_MD, OUT_JS = f"{ROOT}/results/node1_link.md", f"{ROOT}/build/node1_link.json"
RUNS = 3


def bench(fn, reps=20, warm=5):
    for _ in range(warm): fn()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
    for s, e in ev: s.record(); fn(); e.record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in ev) * 1e3  # us


def gbs(nbytes, us): return nbytes / us / 1e3


def repeat(fn):
    v = [fn() for _ in range(RUNS)]
    return dict(median=statistics.median(v), spread_pct=(max(v) - min(v)) / statistics.median(v) * 100, runs=v)


def topo():
    return subprocess.run("nvidia-smi topo -m", shell=True, capture_output=True, text=True).stdout


def multicast():
    out = {}
    try:
        from torch._C._autograd import DeviceType
        from torch._C._distributed_c10d import _SymmetricMemory
        out["torch_has_multicast"] = [bool(_SymmetricMemory.has_multicast_support(DeviceType.CUDA, i)) for i in range(ND)]
    except Exception as e:  # noqa
        out["torch_has_multicast"] = f"unknown ({type(e).__name__})"
    try:
        cu = ctypes.CDLL("libcuda.so.1"); cu.cuInit(0); v, d = ctypes.c_int(), ctypes.c_int(); attr = []
        for i in range(ND):
            cu.cuDeviceGet(ctypes.byref(d), i); cu.cuDeviceGetAttribute(ctypes.byref(v), 132, d); attr.append(v.value)  # CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED
        out["cu_multicast_supported"] = attr
    except Exception as e:  # noqa
        out["cu_multicast_supported"] = f"unknown ({type(e).__name__})"
    return out


def sm_store(dst_dev):
    """SM 16 B store stream dev0 -> dst_dev (dst_dev == 0: local HBM write for reference). GB/s."""
    with torch.cuda.device(0):
        src = torch.empty(NB, device="cuda:0", dtype=torch.uint8); dst = torch.empty(NB, device=f"cuda:{dst_dev}", dtype=torch.uint8)
        r = repeat(lambda: gbs(NB, bench(lambda: ext.p2p_store_bw_kernel(src, dst))))
    del src, dst; return r


def fanout(peers):
    """dev0 streams one 256 MiB SM-store copy to each peer concurrently (one stream per peer). Aggregate egress GB/s + per-stream min."""
    with torch.cuda.device(0):
        srcs = [torch.empty(NB, device="cuda:0", dtype=torch.uint8) for _ in peers]
        dsts = [torch.empty(NB, device=f"cuda:{p}", dtype=torch.uint8) for p in peers]
        streams = [torch.cuda.Stream() for _ in peers]
        main = torch.cuda.current_stream()

        def once():
            start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
            start.record(main)
            for s, a, b in zip(streams, srcs, dsts):
                s.wait_event(start)
                with torch.cuda.stream(s): ext.p2p_store_bw_kernel(a, b)
            for s in streams:
                e = torch.cuda.Event(); e.record(s); main.wait_event(e)
            end.record(main); torch.cuda.synchronize()
            return start.elapsed_time(end) * 1e3

        for _ in range(5): once()
        r = repeat(lambda: gbs(NB * len(peers), statistics.median(once() for _ in range(15))))
    del srcs, dsts; return r


def copy_engine(p):
    with torch.cuda.device(0):
        src = torch.empty(NB, device="cuda:0", dtype=torch.uint8); dst = torch.empty(NB, device=f"cuda:{p}", dtype=torch.uint8)
        r = repeat(lambda: gbs(NB, bench(lambda: ext.peer_copy(dst, src))))
    del src, dst; return r


def pingpong(p, n=200, trials=5):
    """Flag round trip dev0 <-> peer p via system-scope atomics (bench/timeline.py clock_offset). Returns min / median RTT us."""
    mins, meds = [], []
    i64 = dict(dtype=torch.int64)
    for _ in range(trials):
        fa, fb = torch.zeros(1, device="cuda:0", dtype=torch.int32), torch.zeros(1, device=f"cuda:{p}", dtype=torch.int32)
        oa, ob = torch.zeros(2 * n, device="cuda:0", **i64), torch.zeros(n, device=f"cuda:{p}", **i64)
        torch.cuda.synchronize(0); torch.cuda.synchronize(p)
        with torch.cuda.device(p): ext.pingpong(fb, fa, ob, n, 0)
        with torch.cuda.device(0): ext.pingpong(fa, fb, oa, n, 1)
        torch.cuda.synchronize(0); torch.cuda.synchronize(p)
        a = oa.cpu().numpy().reshape(n, 2); rtt = (a[1:, 1] - a[1:, 0]) / 1e3
        mins.append(float(rtt.min())); meds.append(float(np.median(rtt)))
    return dict(min_us=min(mins), median_us=statistics.median(meds))


def tma_store(p, N=8192, K=8192):
    """Stock coop cfg0 GEMM with D local vs D in peer p's memory; best raster/swizzle picked locally at M=4096."""
    out = []
    with torch.cuda.device(0):
        W = torch.randn(N, K, device="cuda:0", dtype=torch.bfloat16) / K**0.5
        X = torch.randn(4096, K, device="cuda:0", dtype=torch.bfloat16); Y = torch.empty(4096, N, device="cuda:0", dtype=torch.bfloat16)
        best = min(((bench(lambda: ext.coop_gemm(X, W, Y, 0, r, s)), r, s) for r in (1, 2) for s in (1, 2, 4, 8)))
        _, r, s = best
        for M in (1024, 4096, 16384):
            X = torch.randn(M, K, device="cuda:0", dtype=torch.bfloat16)
            Yl = torch.empty(M, N, device="cuda:0", dtype=torch.bfloat16); Yp = torch.empty(M, N, device=f"cuda:{p}", dtype=torch.bfloat16)
            tl = bench(lambda: ext.coop_gemm(X, W, Yl, 0, r, s)); tp = bench(lambda: ext.coop_gemm(X, W, Yp, 0, r, s))
            tc = bench(lambda: torch.mm(X, W.t(), out=Yl))
            out.append(dict(M=M, raster=r, swizzle=s, local_us=tl, peer_us=tp, cublas_us=tc, bytes=M * N * 2, bitequal=bool(torch.equal(Yl, Yp.to("cuda:0")))))
            print(out[-1], flush=True)
    return out


def model(bw1, bwfan):
    """Fixed-cost model from PLAN.md with the measured link: drain floor, paper-shape CTAPP/ideal, TP link ratios (70B FFN)."""
    drain = SMS * BM * BN * 2 / bw1 / 1e3
    rows = []
    for M in (1024, 2048, 4096, 8192, 16384):
        N = K = 8192
        ideal = 2 * 2 * M * N * K / TF / 2 * 1e6
        fill = 2 * BM * K * N / TF * 1e6
        rows.append(dict(M=M, ideal_us=ideal, ctapp_us=ideal + fill + drain + 15, ratio=(ideal + fill + drain + 15) / ideal))
    inter = 28672
    return dict(drain_floor_us=drain, paper_shape=rows,
                tp_link_ratio={"TP2 (1 peer)": TF / bw1 / 1e9 / inter, "TP4 (3 peers, fan-out bw)": 3 * TF / bwfan / 1e9 / inter if bwfan else None,
                               "TP8 (7 peers, fan-out bw)": 7 * TF / bwfan / 1e9 / inter if bwfan else None})


def main():
    d = dict(host=os.uname().nodename, cvd=os.environ.get("CUDA_VISIBLE_DEVICES"), ndev=ND, topo=topo(), multicast=multicast())
    print(d["topo"]); print(d["multicast"], flush=True)
    d["sm_store_local"] = sm_store(0); print("local HBM store", d["sm_store_local"], flush=True)
    d["sm_store_peer"] = {p: sm_store(p) for p in PEERS}; print("sm store peer", d["sm_store_peer"], flush=True)
    d["fanout"] = {len(PEERS[:k]): fanout(PEERS[:k]) for k in range(2, len(PEERS) + 1)} if len(PEERS) >= 2 else {}
    print("fanout", d["fanout"], flush=True)
    d["copy_engine"] = {p: copy_engine(p) for p in PEERS[:1]}; print("CE", d["copy_engine"], flush=True)
    d["pingpong"] = {p: pingpong(p) for p in PEERS}; print("pingpong", d["pingpong"], flush=True)
    d["tma_store"] = tma_store(PEERS[0])
    bw1 = statistics.median(v["median"] for v in d["sm_store_peer"].values())
    bwfan = max((v["median"] for v in d["fanout"].values()), default=None)
    d["model"] = model(bw1, bwfan); print(d["model"], flush=True)
    json.dump(d, open(OUT_JS, "w"), indent=1, default=str)

    o = [f"# S0.1 link characterisation: {d['host']}, CUDA_VISIBLE_DEVICES={d['cvd']} ({ND} GPUs; dev 0 = source)", "",
         "```", d["topo"].rstrip(), "```", "",
         f"Multicast (NVLS) support: torch `has_multicast_support` = {d['multicast']['torch_has_multicast']}, "
         f"`CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED` = {d['multicast']['cu_multicast_supported']}", "",
         "## Bandwidth (256 MiB, median of 20 reps, 3 runs; spread = (max-min)/median)", "",
         "| path | GB/s | spread % |", "|---|---|---|",
         f"| SM 16 B stores, local HBM | {d['sm_store_local']['median']:.1f} | {d['sm_store_local']['spread_pct']:.1f} |"]
    for p, v in d["sm_store_peer"].items(): o.append(f"| SM 16 B stores, dev0 -> dev{p} | {v['median']:.1f} | {v['spread_pct']:.1f} |")
    for k, v in d["fanout"].items(): o.append(f"| SM stores fan-out, dev0 -> {k} peers concurrently (aggregate egress) | {v['median']:.1f} | {v['spread_pct']:.1f} |")
    for p, v in d["copy_engine"].items(): o.append(f"| copy engine (cudaMemcpyPeerAsync), dev0 -> dev{p} | {v['median']:.1f} | {v['spread_pct']:.1f} |")
    o += ["", "## Flag round trip (system-scope release/acquire ping-pong, 200 iterations x 5 trials)", "", "| peer | min RTT us | median RTT us |", "|---|---|---|"]
    for p, v in d["pingpong"].items(): o.append(f"| dev{p} | {v['min_us']:.2f} | {v['median_us']:.2f} |")
    o += ["", f"## TMA-store D local vs peer (stock coop cfg0 128x256x64, N=K=8192, raster {d['tma_store'][0]['raster']} swizzle {d['tma_store'][0]['swizzle']}; link bound = bytes / peer SM-store GB/s)", "",
          "| M | cuBLAS us | local us | peer us | link bound us | peer/local | peer / max(local, link) | D equal |", "|---|---|---|---|---|---|---|---|"]
    for x in d["tma_store"]:
        link = x["bytes"] / bw1 / 1e3; q = x["peer_us"] / max(x["local_us"], link)
        o.append(f"| {x['M']} | {x['cublas_us']:.0f} | {x['local_us']:.0f} | {x['peer_us']:.0f} | {link:.0f} | {x['peer_us']/x['local_us']:.3f} | {q:.3f} | {x['bitequal']} |")
    m = d["model"]
    o += ["", "## Model with the measured link (PLAN.md pen-and-paper, 700 TFLOP/s)", "",
          f"- Peer SM-store bandwidth used: {bw1:.1f} GB/s (1 peer); fan-out aggregate: {bwfan if bwfan is None else round(bwfan, 1)} GB/s.",
          f"- Last-wave drain floor (132 x 128x256 bf16 tiles = 8.65 MB): **{m['drain_floor_us']:.1f} us** (was 69 us at 124 GB/s).", "",
          "| M (paper square) | ideal us | CTAPP model us | model/ideal |", "|---|---|---|---|"]
    for r in m["paper_shape"]: o.append(f"| {r['M']} | {r['ideal_us']:.0f} | {r['ctapp_us']:.0f} | {r['ratio']:.3f} |")
    o += ["", "70B FFN per-tile-reduce link ratio = (t-1) * (TF/BW) / inter (must be << 1 to hide under GEMM2):", ""]
    for k, v in m["tp_link_ratio"].items(): o.append(f"- {k}: {'n/a' if v is None else f'{v:.2f}'}")
    open(OUT_MD, "w").write("\n".join(o) + "\n"); print("\n".join(o))


if __name__ == "__main__":
    main()
