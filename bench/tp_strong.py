"""Strong tensor-parallel baselines (bf16, fp32 accumulate) on 2 or 4 GPUs, for CTA-pipelining comparisons.

Shapes (``--shape``), all bias-free; ``world`` = number of GPUs / processes (one process per GPU, NCCL):

* ``paper``    K1=8192 -> N1=8192 -> N2=8192, plain GEMM -> GEMM, no activation. Megatron: fc1 column-
               sharded (N1/world per rank), fc2 row-sharded (K=N1/world per rank), all-reduce of fc2 output.
* ``l70b``     Llama-70B FFN: fc1 produces gate and up (28672 each), h = silu(gate) * up, fc2 28672 -> 8192.
               Per-rank fc1 weight is (2*28672/world, 8192) laid out [gate_shard; up_shard]; fc2 K sharded.
* ``l70b_qkv`` down-proj K1=28672 -> N1=8192, RMSNorm(8192, eps 1e-5, weight=1), QKV GEMM N2=10240.
               GEMM1 row-sharded on K1 (w1s (8192, 28672/world), x column-shard), the all-reduce variant is
               applied to GEMM1's output, RMSNorm replicated, GEMM2 column-sharded (w2s (10240/world, 8192)),
               output (M, 10240/world), no second all-reduce. ``asynctp``: reduce-scatter GEMM1, RMSNorm on
               own M/world rows, NCCL all-gather rows, GEMM2 (``asynctp_ag``/``asynctp_mmag`` skipped).

Variants (``--variants``): ``nccl`` (eager, dist.all_reduce), ``nccl_compiled`` / ``nccl_mat``
(torch.compile on compute), ``nccl_graph`` (whole step in a CUDA graph), ``oneshot`` / ``twoshot`` /
``multimem`` (symm_mem all-reduces), ``asynctp`` (fused_matmul_reduce_scatter, output row-sharded: rank r holds
rows [r*M/world, (r+1)*M/world)), ``asynctp_ag`` (+NCCL all-gather), ``asynctp_mmag`` (+multimem all-gather).
``--nccl-env KEY=VAL ...`` reruns plain ``nccl`` in fresh process groups per env setting (default: world 2
Ring, Tree; world>=4 NVLS, Ring, Tree).

Timing: per step (barrier + sync before each call, median) and back-to-back (iters queued, mean); both are the
max over ranks. ``one_gpu_ms`` = eager bf16 cuBLAS of the whole unsharded chain on rank 0 (median);
``ideal_ms = one_gpu_ms / world``; exposed % = (variant - ideal) / variant * 100. Every variant is checked
against an fp32 single-GPU reference of the full chain (max|err|, rel = max|err| / max|ref|).
Results are also dumped to ``--json`` (default build/tp_strong_<shape>_w<world>.json).

Run (from repo root)::

    python bench/tp_strong.py --world 2 --shape l70b
    python bench/tp_strong.py --world 4 --shape l70b_qkv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import traceback

import torch
import torch.distributed as dist
import torch.nn.functional as F

DTYPE = torch.bfloat16
EPS = 1e-5
# kind: mlp (GEMM -> [act] -> GEMM -> all-reduce) or qkv (GEMM -> all-reduce -> norm -> GEMM)
SHAPES = {
    "paper": dict(K1=8192, N1=8192, N2=8192, act="none"),
    "l70b": dict(K1=8192, N1=28672, N2=8192, act="swiglu"),
    "l70b_qkv": dict(K1=28672, N1=8192, N2=10240, act="qkv"),
}


def make_weights(cfg, dev, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    K1, N1, N2 = cfg["K1"], cfg["N1"], cfg["N2"]
    n1 = 2 * N1 if cfg["act"] == "swiglu" else N1
    w1 = (torch.randn(n1, K1, device=dev, generator=g) * K1**-0.5).to(DTYPE)
    w2 = (torch.randn(N2, N1, device=dev, generator=g) * N1**-0.5).to(DTYPE)
    return w1, w2


def rmsnorm(x, weight):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)).to(x.dtype) * weight


def act_hidden(cfg, x, w1):
    h = F.linear(x, w1)
    if cfg["act"] == "swiglu":
        g, u = h.chunk(2, dim=-1)
        h = F.silu(g) * u
    return h


def chain(cfg, x, w1, w2, nw, ref=False):
    """Full unsharded chain; ref=True: fp32 compute with bf16 rounding at the same points as the bf16 path."""
    c = (lambda t: t.float()) if ref else (lambda t: t)
    d = (lambda t: t.to(DTYPE)) if ref else (lambda t: t)
    h = d(act_hidden(cfg, c(x), c(w1)))
    if cfg["act"] == "qkv":
        h = rmsnorm(h, nw)
    return d(F.linear(c(h), c(w2)))


def _max_err(y, ref):
    return (y.float() - ref.float()).abs().max().item()


# --------------------------------------------------------------------------- variants
class TPStep:
    """One TP forward for a fixed M; ``__call__`` returns y (full, row-sharded, or column-sharded)."""

    def __init__(self, name, rank, world, cfg, x, w1s, w2s, nw, group_name):
        self.name, self.rank, self.world, self.cfg = name, rank, world, cfg
        self.x, self.w1s, self.w2s, self.nw, self.group_name = x, w1s, w2s, nw, group_name
        self.dev = x.device
        self.qkv = cfg["act"] == "qkv"
        # last GEMM whose output is the crossing (all-reduced) tensor: fc2 (mlp) or GEMM1 (qkv)
        self.wl = w1s if self.qkv else w2s
        M = x.shape[0]
        Nc = cfg["N1"] if self.qkv else cfg["N2"]
        self.graph = None
        if name in ("nccl", "nccl_graph", "nccl_env"):
            self.compute = self._partial
        elif name == "nccl_compiled":
            self.compute = torch.compile(self._partial, dynamic=False)
        elif name == "nccl_mat":
            self.compute = torch.compile(self._partial, dynamic=False, mode="max-autotune-no-cudagraphs")
        elif name in ("oneshot", "twoshot", "multimem"):
            import torch.distributed._symmetric_memory as symm_mem
            self.y_symm = symm_mem.empty(M, Nc, dtype=DTYPE, device=self.dev)
            symm_mem.rendezvous(self.y_symm, group_name)
            self.compute = self._partial_into_symm
        elif name.startswith("asynctp"):
            import torch.distributed._symmetric_memory as symm_mem
            if self.qkv and name != "asynctp":
                raise NotImplementedError("skipped")
            symm_mem.enable_symm_mem_for_group(group_name)
            if name == "asynctp_mmag":
                self.y_full = symm_mem.empty(M, Nc, dtype=DTYPE, device=self.dev)
                symm_mem.rendezvous(self.y_full, group_name)
            elif name == "asynctp_ag" or self.qkv:
                self.y_full = torch.empty(M, Nc, dtype=DTYPE, device=self.dev)
        else:
            raise ValueError(name)

    # compute pieces ------------------------------------------------------
    def _hidden(self):  # input of the crossing GEMM
        return self.x if self.qkv else act_hidden(self.cfg, self.x, self.w1s)

    def _partial(self):
        return F.linear(self._hidden(), self.wl)

    def _partial_into_symm(self):
        torch.mm(self._hidden(), self.wl.t(), out=self.y_symm)
        return self.y_symm

    def _post(self, y):  # after the all-reduce: qkv does norm + column-sharded GEMM2
        return F.linear(rmsnorm(y, self.nw), self.w2s) if self.qkv else y

    # full step -----------------------------------------------------------
    def step(self):
        n = self.name
        if n in ("nccl", "nccl_env", "nccl_compiled", "nccl_mat", "nccl_graph"):
            y = self.compute()
            dist.all_reduce(y)
            return self._post(y)
        if n in ("oneshot", "twoshot", "multimem"):
            y = self.compute()
            op = {"oneshot": torch.ops.symm_mem.one_shot_all_reduce,
                  "twoshot": torch.ops.symm_mem.two_shot_all_reduce_,
                  "multimem": torch.ops.symm_mem.multimem_all_reduce_}[n]
            return self._post(op(y, "sum", self.group_name))
        if n.startswith("asynctp"):
            from torch.distributed._symmetric_memory import _fused_matmul_reduce_scatter
            ys = _fused_matmul_reduce_scatter(self._hidden(), self.wl.t(), "sum", 0, self.group_name)
            if self.qkv:  # norm on own rows, all-gather, GEMM2
                dist.all_gather_into_tensor(self.y_full, rmsnorm(ys, self.nw))
                return F.linear(self.y_full, self.w2s)
            if n == "asynctp":
                return ys  # [M/world, N2], this rank's own rows
            if n == "asynctp_ag":
                dist.all_gather_into_tensor(self.y_full, ys)
                return self.y_full
            torch.ops.symm_mem.multimem_all_gather_out(ys, self.group_name, self.y_full)
            return self.y_full
        raise ValueError(n)

    def capture(self):
        """CUDA-graph the whole step (nccl_graph)."""
        s = torch.cuda.Stream(self.dev)
        s.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(s):
            for _ in range(3):
                self.step()
        torch.cuda.current_stream(self.dev).wait_stream(s)
        torch.cuda.synchronize(self.dev)
        _barrier()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            self.y_static = self.step()
        torch.cuda.synchronize(self.dev)
        _barrier()
        self.graph = g

    def __call__(self):
        if self.graph is not None:
            self.graph.replay()
            return self.y_static
        return self.step()


GLOO = None  # side-channel group for barriers / timing reduce (never NCCL)


def _barrier():
    dist.barrier(group=GLOO)


def _time(step, dev, iters, warmup, extra=()):
    for _ in range(warmup):
        step()
    torch.cuda.synchronize(dev)
    _barrier()
    times = []
    for _ in range(iters):
        _barrier()
        torch.cuda.synchronize(dev)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        step()
        b.record()
        torch.cuda.synchronize(dev)
        times.append(a.elapsed_time(b))
    per_step = statistics.median(times)
    _barrier()
    torch.cuda.synchronize(dev)
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        step()
    b.record()
    torch.cuda.synchronize(dev)
    b2b = a.elapsed_time(b) / iters
    t = torch.tensor([per_step, b2b, *extra], dtype=torch.float64)  # CPU; rank-max over gloo
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=GLOO)
    return t.tolist()


def time_one_gpu(fn, dev, iters, warmup):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize(dev)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize(dev)
        times.append(a.elapsed_time(b))
    return statistics.median(times)


def worker(rank, world, port, shape, tokens, variants, iters, warmup, queue, tag, do_one_gpu):
    global GLOO
    os.environ.setdefault("TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES", "0")
    os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")  # watchdog hung 480 s at teardown in b7
    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world,
                            device_id=dev)
    GLOO = dist.new_group(backend="gloo")
    group_name = dist.group.WORLD.group_name
    cfg = SHAPES[shape]
    K1, N1, N2, qkv = cfg["K1"], cfg["N1"], cfg["N2"], cfg["act"] == "qkv"
    w1, w2 = make_weights(cfg, dev)
    nw = torch.ones(N1, device=dev, dtype=DTYPE)
    sl = lambda n: slice(rank * (n // world), (rank + 1) * (n // world))  # noqa: E731
    if qkv:  # GEMM1 row-sharded on K1, GEMM2 column-sharded on N2
        w1s, w2s = w1[:, sl(K1)].contiguous(), w2[sl(N2)].contiguous()
    elif cfg["act"] == "swiglu":  # [gate_shard; up_shard]
        w1s = torch.cat([w1[:N1][sl(N1)], w1[N1:][sl(N1)]]).contiguous()
        w2s = w2[:, sl(N1)].contiguous()
    else:
        w1s, w2s = w1[sl(N1)].contiguous(), w2[:, sl(N1)].contiguous()
    try:
        from torch._C._autograd import DeviceType
        from torch._C._distributed_c10d import _SymmetricMemory
        mc = _SymmetricMemory.has_multicast_support(DeviceType.CUDA, rank)
    except Exception as e:  # noqa: BLE001
        mc = f"unknown ({type(e).__name__})"
    if rank == 0:
        print(f"[{tag}] torch {torch.__version__}, nccl {torch.cuda.nccl.version()}, world {world}, "
              f"shape {shape}, multicast support: {mc}", flush=True)
    results, one_gpu = {}, {}
    for M in tokens:
        x = torch.randn(M, K1, device=dev, dtype=DTYPE, generator=torch.Generator(device=dev).manual_seed(1))
        with torch.no_grad():
            ref = chain(cfg, x, w1, w2, nw, ref=True)  # every rank computes the same fp32 reference locally
            if do_one_gpu:
                t = torch.zeros(1, dtype=torch.float64)
                if rank == 0:
                    t[0] = time_one_gpu(lambda: chain(cfg, x, w1, w2, nw), dev, iters, warmup)
                dist.broadcast(t, src=0, group=GLOO)
                one_gpu[M] = t.item()
                if rank == 0:
                    print(f"[{tag}] M={M:6d} one_gpu {one_gpu[M]:8.3f} ms  ideal {one_gpu[M] / world:8.3f} ms",
                          flush=True)
        xin = x[:, sl(K1)].contiguous() if qkv else x
        for v in variants:
            _barrier()
            t0 = time.time()
            try:
                st = TPStep(v, rank, world, cfg, xin, w1s, w2s, nw, group_name)
                if v == "nccl_graph":
                    st.capture()
                y = st()
                torch.cuda.synchronize(dev)
                if qkv:  # column shard
                    r_ = ref[:, sl(N2)]
                elif y.shape[0] == M:
                    r_ = ref
                else:  # row-sharded: rank r holds rows [r*M/world, (r+1)*M/world)
                    r_ = ref[sl(M)]
                err = _max_err(y, r_)
                rel = err / r_.float().abs().max().item()
                ps, bb, err, rel = _time(st, dev, iters, warmup, (err, rel))
                results[(M, v)] = (ps, bb, err, rel)
                if rank == 0:
                    print(f"[{tag}] M={M:6d} {v:14s} per-step {ps:8.3f} ms  b2b {bb:8.3f} ms  "
                          f"max|err| {err:.4f}  rel {rel:.2e}  ({time.time() - t0:.0f}s)", flush=True)
                del st, y
            except Exception as e:  # noqa: BLE001
                results[(M, v)] = (float("nan"),) * 4
                if rank == 0:
                    msg = str(e).splitlines()[0][:200] if str(e) else ""
                    if isinstance(e, NotImplementedError):
                        print(f"[{tag}] M={M:6d} {v:14s} skipped", flush=True)
                    else:
                        print(f"[{tag}] M={M:6d} {v:14s} FAILED: {type(e).__name__}: {msg}", flush=True)
                        traceback.print_exc(limit=3)
                torch.cuda.synchronize(dev)
                _barrier()
            torch.cuda.empty_cache()
        del x, xin, ref
    if rank == 0:
        queue.put((results, one_gpu))
    torch.cuda.synchronize(dev)
    _barrier()
    sys.stdout.flush()
    os._exit(0)  # skip NCCL / symm-mem teardown (hung the watchdog for 480 s in b7)


def run_group(world, shape, tokens, variants, iters, warmup, port, tag, env=None, do_one_gpu=False):
    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    old = {}
    if env:
        for k, v in env.items():
            old[k] = os.environ.get(k)
            os.environ[k] = v
    procs = [ctx.Process(target=worker, args=(r, world, port, shape, tokens, variants, iters, warmup, queue,
                                              tag, do_one_gpu)) for r in range(world)]
    for p in procs:
        p.start()
    res = queue.get()
    for p in procs:
        p.join()
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--shape", choices=list(SHAPES), default="paper")
    ap.add_argument("--tokens", default="1024,2048,4096,8192,16384")
    ap.add_argument("--variants", default="nccl,nccl_graph,oneshot,twoshot,multimem,asynctp,asynctp_ag")
    ap.add_argument("--nccl-env", nargs="*", default=None,
                    help="extra runs of the plain nccl variant, one fresh process group per KEY=VAL[,KEY=VAL]")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--port", type=int, default=29611)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    world = args.world
    if args.nccl_env is None:
        args.nccl_env = (["NCCL_ALGO=Ring", "NCCL_ALGO=Tree"] if world == 2
                         else ["NCCL_ALGO=NVLS", "NCCL_ALGO=Ring", "NCCL_ALGO=Tree"])
    json_path = args.json or f"build/tp_strong_{args.shape}_w{world}.json"
    tokens = [int(t) for t in args.tokens.split(",")]
    variants = args.variants.split(",")

    rows = {}
    res, one_gpu = run_group(world, args.shape, tokens, variants, args.iters, args.warmup, args.port, "main",
                             do_one_gpu=True)
    rows.update(res)
    for i, spec in enumerate(args.nccl_env):
        env = dict(kv.split("=", 1) for kv in spec.split(","))
        res, _ = run_group(world, args.shape, tokens, ["nccl"], args.iters, args.warmup, args.port + 1 + i,
                           spec, env)
        rows.update({(M, f"nccl[{spec}]"): r for (M, v), r in res.items()})

    nan4 = (float("nan"),) * 4
    cols = variants + [f"nccl[{s}]" for s in args.nccl_env]
    for title, idx in (("per step (barrier + sync before each call)", 0), ("back-to-back", 1)):
        print(f"\n== TP{world} {args.shape}, ms, {title}; cell = ms (exposed %, % vs nccl); "
              f"ideal = 1-GPU/{world} ==")
        print("| M | ideal | " + " | ".join(cols) + " |")
        print("|---" * (len(cols) + 2) + "|")
        for M in tokens:
            base = rows.get((M, "nccl"), nan4)[idx]
            ideal = one_gpu[M] / world
            cells = []
            for c in cols:
                v = rows.get((M, c), nan4)[idx]
                cells.append("n/a" if math.isnan(v) else
                             f"{v:.3f} ({(v - ideal) / v * 100:.0f}%, {(v - base) / base * 100:+.0f}%)")
            print(f"| {M} | {ideal:.3f} | " + " | ".join(cells) + " |")
    print("\n== max|err| (rel err = max|err| / max|ref|) vs fp32 reference ==")
    for M in tokens:
        print(f"M={M}: " + ", ".join(
            f"{c} {rows.get((M, c), nan4)[2]:.4f} ({rows.get((M, c), nan4)[3]:.1e})" for c in cols))

    nn = lambda v: None if math.isnan(v) else v  # noqa: E731
    out = {"shape": args.shape, "world": world, "iters": args.iters, "warmup": args.warmup,
           "dims": SHAPES[args.shape], "one_gpu_ms": {str(M): one_gpu[M] for M in tokens},
           "ideal_ms": {str(M): one_gpu[M] / world for M in tokens}, "results": {}}
    for (M, c), (ps, bb, err, rel) in rows.items():
        out["results"][f"{M}/{c}"] = dict(per_step_ms=nn(ps), b2b_ms=nn(bb), max_err=nn(err), rel_err=nn(rel))
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
