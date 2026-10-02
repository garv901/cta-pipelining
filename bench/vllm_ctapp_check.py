"""Correctness run on the 4-layer Llama-70B config (dummy weights): prints top-20 logprobs of the first generated token.
Usage: CTAPP_VLLM=0|1 [CTAPP_CHECK=1] python bench/vllm_ctapp_check.py --lens 4096 --out build/ck_stock.json
       python bench/vllm_ctapp_check.py --compare build/ck_stock.json build/ck_ctapp.json"""
import argparse, json, os, random, sys

p = argparse.ArgumentParser()
p.add_argument("--lens", default="4096", help="comma list of prompt lengths (one prompt each)")
p.add_argument("--out", default=None)
p.add_argument("--model", default="/data/garv901-55613a/hf/llama70b-4layer")
p.add_argument("--rescale", type=int, default=1, help="re-init dummy weights at a realistic scale (default dummy init is +-1e-3: all logits tie)")
p.add_argument("--max-model-len", type=int, default=4200)
p.add_argument("--compare", nargs=2, default=None)
a = p.parse_args()




def worker_cksum(self):
    m = self.model_runner.get_model()
    w = m.model.layers[0].mlp.down_proj.weight
    return (tuple(w.shape), w.double().sum().item(), w.double().abs().sum().item())


def worker_rescale(self, rank_seed=0):
    import math, torch
    from vllm.distributed import get_tensor_model_parallel_rank
    m = self.model_runner.get_model()
    g = torch.Generator(device="cuda").manual_seed(1234 + get_tensor_model_parallel_rank())
    with torch.no_grad():
        for n, q in m.named_parameters():
            if q.dim() == 2:
                fan = q.shape[1] * (4 if ("o_proj" in n or "down_proj" in n) else 1)
                bnd = 1.7 if "embed" in n else math.sqrt(3.0 / fan)
                q.copy_(((torch.rand(q.shape, device=q.device, generator=g, dtype=torch.float32) * 2 - 1) * bnd).to(q.dtype))
            else:
                q.copy_((1 + 0.1 * torch.randn(q.shape, device=q.device, generator=g)).to(q.dtype))
    if hasattr(m.model, "_refold"):   # plugin folds gammas in place into gate_up/qkv: re-init invalidates it, re-fold on next forward
        m.model._refold()
    return True


def main():
    if a.compare:
        s, o = (json.load(open(f)) for f in a.compare)
        for i, (rs, ro) in enumerate(zip(s["results"], o["results"])):
            ts, to = rs["top"], ro["top"]
            common = set(ts) & set(to)
            d = max((abs(ts[t] - to[t]) for t in common), default=float("nan"))
            top1s, top1o = max(ts, key=ts.get), max(to, key=to.get)
            print(f"prompt {i}: top1 stock {top1s} ours {top1o} match={top1s == top1o}; top20 overlap {len(common)}/20; max|dlogprob| {d:.4e}")
            print("   cksum stock", s["cksum"], " ours", o["cksum"])
        return
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, tensor_parallel_size=4, load_format="dummy", dtype="bfloat16", enforce_eager=True,
              gpu_memory_utilization=0.5, max_model_len=a.max_model_len, max_num_batched_tokens=16640, max_num_seqs=8,
              enable_prefix_caching=False, seed=0)
    if a.rescale:
        llm.collective_rpc(worker_rescale)
    ck = llm.collective_rpc(worker_cksum)[0]
    print("CKSUM layers[0].down_proj rank0:", ck, flush=True)
    sp = SamplingParams(max_tokens=1, temperature=0, logprobs=20)
    rng = random.Random(0)
    lens = [int(x) for x in a.lens.split(",")]
    prompts = [{"prompt_token_ids": [rng.randint(1000, 50000) for _ in range(n)]} for n in lens]
    outs = llm.generate(prompts, sp, use_tqdm=False)
    results = []
    for o in outs:
        lp = o.outputs[0].logprobs[0]
        top = {int(t): float(v.logprob) for t, v in lp.items()}
        print("generated", o.outputs[0].token_ids, "top20:", sorted(top.items(), key=lambda kv: -kv[1]), flush=True)
        results.append(dict(top=top, gen=list(o.outputs[0].token_ids)))
    if a.out:
        json.dump(dict(cksum=ck, lens=lens, results=results), open(a.out, "w"))


if __name__ == "__main__":
    main()
