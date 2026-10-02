"""Phase-6 S1 probe 6 (plan D4): why does vLLM's eager 64 MB all-reduce take 0.35-0.38 ms while our bench's NcclBoundary
(torch.distributed.all_reduce) measures ~0.75 ms?  Run in the vLLM venv (torch 2.13 / NCCL 2.29.7, vLLM 0.30.0):

    CUDA_MODULE_LOADING=LAZY python -u bench/nccl_ar.py [--iters 20] [--warmup 5] [--json build/p6_nccl.json]

4 ranks, bf16 (4096, 8192) = 64 MiB. Per NCCL env setting (default, NCCL_PROTO=LL|LL128|Simple, NCCL_ALGO=Ring|Tree|NVLS) a fresh
set of 4 processes (env set before any NCCL init) times
  torch_dist : torch.distributed.all_reduce(x) on the default ProcessGroupNCCL (in place; what NcclBoundary does)
  pynccl     : vLLM PyNcclCommunicator(gloo group, device).all_reduce(x) (out of place, current stream; what vLLM eager does)
plus an in-chain measurement (torch.mm down GEMM M=4096 K=7168 N=8192, then dist.all_reduce of its output; AR in chain =
(mm + AR) - mm), all with ctapp.tp4.time_step (per-call median with barrier+sync before each call, back-to-back mean; max over ranks), and lists the
CUDA kernels of 3 profiled calls (torch.profiler).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from ctapp import tp4  # noqa: E402

SETTINGS = [("default", {}), ("NCCL_PROTO=LL", {"NCCL_PROTO": "LL"}), ("NCCL_PROTO=LL128", {"NCCL_PROTO": "LL128"}),
            ("NCCL_PROTO=Simple", {"NCCL_PROTO": "Simple"}), ("NCCL_ALGO=Ring", {"NCCL_ALGO": "Ring"}),
            ("NCCL_ALGO=Tree", {"NCCL_ALGO": "Tree"}), ("NCCL_ALGO=NVLS", {"NCCL_ALGO": "NVLS"})]


def kernels(fn, dev, n=3):
    from torch.profiler import ProfilerActivity, profile
    fn(); torch.cuda.synchronize(dev)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            fn()
        torch.cuda.synchronize(dev)
    out = {}
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA and "nccl" in e.name.lower():
            d = out.setdefault(e.name, [0, 0.0])
            d[0] += 1
            d[1] += e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
    return {k: {"count": v[0], "avg_us": v[1] / max(1, v[0])} for k, v in out.items()}


def boundary_decomp(rank, world, dev, gn, h, wd, iters, warmup):
    """tp4_bench's `nccl` row (tp4.NcclBoundary.forward = mm -> dist.all_reduce -> + resid -> eager rmsnorm_gamma -> QKV mm) and its
    pieces, M=4096, so the old '~0.75 ms NCCL' figure can be attributed."""
    M = 4096
    g = torch.Generator(device=dev).manual_seed(11)
    wq = (torch.randn(tp4.N2 // world, tp4.N1, device=dev, generator=g) * tp4.N1 ** -0.5).to(torch.bfloat16)
    gamma = (1 + 0.1 * torch.randn(tp4.N1, device=dev, generator=g)).to(torch.bfloat16)
    resid = torch.randn(M, tp4.N1, device=dev, generator=g).to(torch.bfloat16)
    b = tp4.NcclBoundary(M, rank, world, gn, dev, wd, wq, gamma)
    y = torch.empty(M, tp4.N1, device=dev, dtype=torch.bfloat16)
    torch.mm(h, wd.t(), out=y)
    x = y + resid
    xn = tp4.rmsnorm_gamma(x, b.gamma)
    pieces = {"forward (whole nccl row)": lambda: b.forward(h, resid),
              "down mm": lambda: torch.mm(h, wd.t(), out=y),
              "all_reduce": lambda: dist.all_reduce(y),
              "resid add": lambda: y + resid,
              "rmsnorm_gamma (eager, fp32 temporaries)": lambda: tp4.rmsnorm_gamma(x, b.gamma),
              "QKV mm": lambda: torch.mm(xn, wq.t()),
              "post = add + rmsnorm + QKV": lambda: b._post(y, resid)}
    return {k: dict(zip(("per_step", "b2b"), tp4.time_step(fn, dev, iters, warmup))) for k, fn in pieces.items()}


def worker(rank, world, dev, gn, iters, warmup):
    x = (torch.randn(4096, 8192, device=dev) * 1e-3).to(torch.bfloat16)
    res = {"nccl_version": ".".join(map(str, torch.cuda.nccl.version())), "torch": torch.__version__,
           "env": {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}}
    ps, bb = tp4.time_step(lambda: dist.all_reduce(x), dev, iters, warmup)
    res["torch_dist"] = {"per_step": ps, "b2b": bb, "kernels": kernels(lambda: dist.all_reduce(x), dev)}
    # in-chain: the tp4_bench `nccl` row's AR follows the down GEMM (torch.mm, M=4096, K=7168, N=8192) on the same stream
    g = torch.Generator(device=dev).manual_seed(5)
    h = torch.randn(4096, 7168, device=dev, generator=g).to(torch.bfloat16)
    wd = (torch.randn(8192, 7168, device=dev, generator=g) * 7168 ** -0.5).to(torch.bfloat16)
    ybuf = torch.empty(4096, 8192, device=dev, dtype=torch.bfloat16)

    def mm():
        torch.mm(h, wd.t(), out=ybuf)

    def mm_ar():
        torch.mm(h, wd.t(), out=ybuf)
        dist.all_reduce(ybuf)
    c = {}
    for name, fn in (("mm", mm), ("mm_ar", mm_ar), ("ar", lambda: dist.all_reduce(ybuf))):
        c[name] = dict(zip(("per_step", "b2b"), tp4.time_step(fn, dev, iters, warmup)))
    c["ar_in_chain_per_step"] = c["mm_ar"]["per_step"] - c["mm"]["per_step"]
    c["ar_in_chain_b2b"] = c["mm_ar"]["b2b"] - c["mm"]["b2b"]
    res["chain"] = c
    if not res["env"]:
        res["boundary"] = boundary_decomp(rank, world, dev, gn, h, wd, iters, warmup)
    try:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    except ImportError:
        return res if rank == 0 else {}
    comm = PyNcclCommunicator(tp4.GLOO, dev)
    ps, bb = tp4.time_step(lambda: comm.all_reduce(x), dev, iters, warmup)
    res["pynccl"] = {"per_step": ps, "b2b": bb, "kernels": kernels(lambda: comm.all_reduce(x), dev), "nccl_version": comm.nccl.ncclGetVersion()}
    # check both give the same sum (one call each from the same input)
    y = (torch.randn(4096, 8192, device=dev, generator=torch.Generator(device=dev).manual_seed(rank)) * 1e-2).to(torch.bfloat16)
    a = y.clone(); dist.all_reduce(a)
    b = comm.all_reduce(y)
    torch.cuda.synchronize(dev)
    res["max_abs_diff_torch_vs_pynccl"] = (a.float() - b.float()).abs().max().item()
    return res if rank == 0 else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--port", type=int, default=29960)
    ap.add_argument("--settings", default=",".join(s[0] for s in SETTINGS))
    ap.add_argument("--json", default="build/p6_nccl.json")
    a = ap.parse_args()
    want = a.settings.split(",")
    out = {}
    for i, (name, env) in enumerate(SETTINGS):
        if name not in want:
            continue
        saved = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        res, bad = tp4.launch(worker, 4, a.port + i, a.iters, a.warmup)
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        out[name] = res[0] if not bad else {"error": True}
        r = out[name]
        if bad:
            print(f"[{name}] FAILED", flush=True)
            continue
        for m in ("torch_dist", "pynccl"):
            if m not in r:
                continue
            ks = "; ".join(f"{k} x{v['count']} avg {v['avg_us']:.0f} us" for k, v in r[m]["kernels"].items())
            print(f"[{name}] {m:10s} per-call {r[m]['per_step']:.3f} ms  b2b {r[m]['b2b']:.3f} ms  kernels: {ks}", flush=True)
        c = r["chain"]
        print(f"[{name}] chain: mm {c['mm']['per_step']:.3f} / {c['mm']['b2b']:.3f}, mm+AR {c['mm_ar']['per_step']:.3f} / {c['mm_ar']['b2b']:.3f}, "
              f"AR {c['ar']['per_step']:.3f} / {c['ar']['b2b']:.3f} -> AR in chain {c['ar_in_chain_per_step']:.3f} / {c['ar_in_chain_b2b']:.3f} ms "
              f"(per-call / b2b)", flush=True)
        print(f"[{name}] nccl {r['nccl_version']} (pynccl lib {r.get('pynccl', {}).get('nccl_version', 'n/a')}), torch {r['torch']}, "
              f"env {r['env']}, max|torch - pynccl| {r.get('max_abs_diff_torch_vs_pynccl', float('nan')):.2e}", flush=True)
    print("\n### Probe 6: 64 MiB bf16 all-reduce, 4 ranks (ms; per-call median / back-to-back mean, max over ranks)\n")
    print("| setting | method | per-call ms | b2b ms | kernel (avg us) |")
    print("|---|---|---|---|---|")
    for name, r in out.items():
        if r.get("error"):
            print(f"| {name} | - | FAILED | | |")
            continue
        for m in ("torch_dist", "pynccl"):
            if m not in r:
                continue
            ks = "; ".join(f"`{k}` ({v['avg_us']:.0f})" for k, v in r[m]["kernels"].items())
            print(f"| {name} | {m} | {r[m]['per_step']:.3f} | {r[m]['b2b']:.3f} | {ks} |")
    print("\n| setting | mm (per-call / b2b) | mm + AR | AR alone (same buffer) | AR in chain = (mm + AR) - mm |")
    print("|---|---|---|---|---|")
    for name, r in out.items():
        if r.get("error"):
            continue
        c = r["chain"]
        print(f"| {name} | {c['mm']['per_step']:.3f} / {c['mm']['b2b']:.3f} | {c['mm_ar']['per_step']:.3f} / {c['mm_ar']['b2b']:.3f} | "
              f"{c['ar']['per_step']:.3f} / {c['ar']['b2b']:.3f} | {c['ar_in_chain_per_step']:.3f} / {c['ar_in_chain_b2b']:.3f} |")
    for name, r in out.items():
        if "boundary" in r:
            print(f"\n#### `nccl` boundary row decomposition ({name}, M=4096; per-call / b2b ms)\n")
            print("| piece | per-call | b2b |")
            print("|---|---|---|")
            for k, v in r["boundary"].items():
                print(f"| {k} | {v['per_step']:.3f} | {v['b2b']:.3f} |")
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
