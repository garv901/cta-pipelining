"""Torch-profiler trace of vLLM TP4 eager prefill (4-layer Llama-70B config). CTAPP_VLLM=0/1 selects stock/ctapp.
Usage: CTAPP_VLLM=1 python bench/vllm_prof.py --out build/prof/ctapp"""
import argparse, glob, os, random, time

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--seq", type=int, default=4096)
p.add_argument("--warmup", type=int, default=3)
p.add_argument("--iters", type=int, default=3)
p.add_argument("--model", default="/data/garv901-55613a/hf/llama70b-4layer")
a = p.parse_args()
a.out = os.path.abspath(a.out)
os.makedirs(a.out, exist_ok=True)

def main():
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, tensor_parallel_size=4, distributed_executor_backend="mp", load_format="dummy", dtype="bfloat16",
              gpu_memory_utilization=0.5, max_model_len=a.seq + 104, max_num_batched_tokens=16640, max_num_seqs=8,
              enable_prefix_caching=False, enforce_eager=True,
              profiler_config=dict(profiler="torch", torch_profiler_dir=a.out, torch_profiler_with_stack=False,
                                   torch_profiler_use_gzip=True))
    sp = SamplingParams(max_tokens=1, ignore_eos=True, temperature=0.0)
    rng = random.Random(0)
    prompts = [{"prompt_token_ids": [rng.randint(1000, 50000) for _ in range(a.seq)]}]
    for _ in range(a.warmup):
        llm.generate(prompts, sp, use_tqdm=False)
    llm.start_profile()
    for _ in range(a.iters):
        t0 = time.perf_counter()
        llm.generate(prompts, sp, use_tqdm=False)
        print(f"timed generate {(time.perf_counter()-t0)*1e3:.2f} ms", flush=True)
    llm.stop_profile()
    time.sleep(5)

if __name__ == "__main__":
    main()
