"""Stock-vLLM TP4 prefill latency (Llama-3.1-70B config, dummy weights).
One generate() of b prompts x seq tokens, max_tokens=1  => prefill of seq*b tokens + 1 decode step + scheduler overhead."""
import argparse, glob, json, os, statistics, time, random

p = argparse.ArgumentParser()
p.add_argument("--variant", default="compiled", choices=["eager", "compiled", "compiled_fi", "compiled_asynctp", "compiled_fi_big", "ctapp"])
p.add_argument("--r", type=int, default=None, help="ctapp: CTAPP_R reducer blocks (default: rule 12 for M<=2048 else 8)")
p.add_argument("--boundary", default=None, choices=["both", "down", "oproj"], help="ctapp: CTAPP_BOUNDARY (default both)")
p.add_argument("--proto", default=None, choices=["v3", "v5"], help="ctapp: CTAPP_PROTO (plugin default v5)")
p.add_argument("--fusion", default=None, choices=["none", "pdl", "pdl1", "role"], help="ctapp v5: CTAPP_FUSION (plugin default pdl)")
p.add_argument("--steal", default=None, choices=["0", "1"], help="ctapp v5: CTAPP_STEAL for both boundaries (plugin default A 1, B 0)")
p.add_argument("--qkv-tile", type=int, default=None, choices=[128, 256], help="ctapp v5: CTAPP_QKV_TILE (boundary B consumer)")
p.add_argument("--max-m", type=int, default=None, help="ctapp: CTAPP_MAX_M (plugin default 8192)")
p.add_argument("--tag", default="", help="suffix of the result tag / JSON name (e.g. a session id)")
p.add_argument("--seq", type=int, default=4096)
p.add_argument("--batch", default="1,2,4")
p.add_argument("--iters", type=int, default=10)
p.add_argument("--warmup", type=int, default=3)
p.add_argument("--tp", type=int, default=4)
p.add_argument("--model", default=glob.glob("/data/garv901-55613a/hf/hub/models--NousResearch--Meta-Llama-3.1-70B/snapshots/*")[0])
a = p.parse_args()
TAG = a.variant + (f"_{a.boundary}" if a.variant == "ctapp" and a.boundary else "") + (f"_R{a.r}" if a.variant == "ctapp" and a.r else "")
if a.variant == "ctapp":
    TAG += (f"_{a.proto}" if a.proto else "") + (f"_{a.fusion}" if a.fusion else "") + (f"_steal{a.steal}" if a.steal else "")
    TAG += (f"_t{a.qkv_tile}" if a.qkv_tile else "") + (f"_maxM{a.max_m}" if a.max_m else "")
TAG += (f"_{a.tag}" if a.tag else "") + (f"_seq{a.seq}" if a.seq != 4096 else "")
if a.variant == "ctapp":   # must be set before importing vllm: spawned workers inherit the environment
    os.environ["CTAPP_VLLM"] = "1"
    for k, v in (("CTAPP_R", a.r), ("CTAPP_BOUNDARY", a.boundary), ("CTAPP_PROTO", a.proto), ("CTAPP_FUSION", a.fusion),
                 ("CTAPP_STEAL", a.steal), ("CTAPP_QKV_TILE", a.qkv_tile), ("CTAPP_MAX_M", a.max_m)):
        if v:
            os.environ[k] = str(v)
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

CC = {
    "eager": None,
    "compiled": {},
    "compiled_fi": {"pass_config": {"fuse_allreduce_rms": True}},
    "compiled_fi_big": {"pass_config": {"fuse_allreduce_rms": True, "fi_allreduce_fusion_max_size_mb": 256}},
    "ctapp": None,
    "compiled_asynctp": {"pass_config": {"enable_sp": True, "fuse_gemm_comms": True}},
}[a.variant]

def main():
    from vllm import LLM, SamplingParams
    kw = dict(model=a.model, tensor_parallel_size=a.tp, distributed_executor_backend="mp", load_format="dummy",
              dtype="bfloat16", gpu_memory_utilization=0.6, max_model_len=a.seq + 104,
              max_num_batched_tokens=max(16640, a.seq * 4 + 256), max_num_seqs=8,
              enable_prefix_caching=False, enforce_eager=(a.variant in ("eager", "ctapp")))
    if CC is not None:
        kw["compilation_config"] = CC
    llm = LLM(**kw)
    sp = SamplingParams(max_tokens=1, ignore_eos=True, temperature=0.0)
    rng = random.Random(0)
    rows = []
    for b in [int(x) for x in a.batch.split(",")]:
        prompts = [{"prompt_token_ids": [rng.randint(1000, 50000) for _ in range(a.seq)]} for _ in range(b)]
        for _ in range(a.warmup):
            llm.generate(prompts, sp, use_tqdm=False)
        ts = []
        for _ in range(a.iters):
            t0 = time.perf_counter()
            llm.generate(prompts, sp, use_tqdm=False)
            ts.append((time.perf_counter() - t0) * 1e3)
        med, mn = statistics.median(ts), min(ts)
        rows.append(dict(batch=b, tokens=a.seq * b, median_ms=med, min_ms=mn, tok_per_s_median=a.seq * b / med * 1e3, all_ms=ts))
        print(f"[{TAG}] b={b} M={a.seq*b} median {med:.2f} ms  min {mn:.2f} ms  {a.seq*b/med*1e3:.0f} tok/s", flush=True)
    print(f"\n{TAG}\n  b      M   median_ms   min_ms   tok/s(median)")
    for r in rows:
        print(f"{r['batch']:3d} {r['tokens']:6d} {r['median_ms']:11.2f} {r['min_ms']:8.2f} {r['tok_per_s_median']:12.0f}")
    os.makedirs("build", exist_ok=True)
    json.dump(dict(variant=TAG, seq=a.seq, iters=a.iters, rows=rows, env={k: v for k, v in os.environ.items() if k.startswith("CTAPP_")}), open(f"build/vllm_prefill_{TAG}.json", "w"), indent=1)

if __name__ == "__main__":
    main()
