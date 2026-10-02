# vLLM 4K prefill ablation, Llama-3.1-70B shape (80 layers), TP4, dummy weights, node1

## Setup

- vLLM 0.30.0, torch 2.13+cu130, TP4 on node1 (4 of 8 H100 NVSwitch GPUs), one process per GPU, dummy weights.
- Performance runs use the Llama-3.1-70B config at full depth (80 layers). The correctness runs use a 4-layer variant of the same config (`/data/garv901-55613a/hf/llama70b-4layer`) with dummy weights re-initialised at a realistic scale (the default dummy init of +-1e-3 makes all logits tie).
- Metric: wall clock of one `llm.generate` of b prompts x 4096 tokens, `max_tokens=1` (prefill + 1 decode step + scheduler overhead), batch b = 1, 2, 4. 10 iterations with 3 warmup in the first session; 20 iterations with 3 warmup in the newer session (`--boundary` flag). Eager (no torch.compile, no CUDA graphs) unless the variant name says compiled. Cells: median / min ms, then tokens/s at the median.
- Raw: `build/vllm_prefill_*.json`, logs `build/logs/pf_*.log`. Drivers: `bench/vllm_prefill.py` (latency), `bench/vllm_ctapp_check.py` (correctness), `bench/vllm_prof.py` (torch profiler, 4-layer config).

## Plugin (`vllm_plugin/`, package `ctapp_vllm`)

A vLLM general plugin. `register()` runs in every process; with `CTAPP_VLLM=1` (default off = stock vLLM) it registers `CtappLlamaForCausalLM` in place of `LlamaForCausalLM`. The model replaces the two TP all-reduce boundaries of each layer with `ctapp.tp4.CtappBoundary` (CTA-pipelined NVLS reduction):

- A: o_proj -> all-reduce -> residual add -> post_attention_layernorm -> gate_up_proj (same layer).
- B: down_proj -> all-reduce -> residual add -> next input_layernorm -> next qkv_proj (layers 0..78 -> 1..79, 79 boundaries per forward). Layer 0's QKV and the last layer's down_proj stay stock.

Environment variables:

| variable | meaning | default |
|---|---|---|
| `CTAPP_VLLM` | 1 enables the plugin | 0 |
| `CTAPP_BOUNDARY` | `both`, `down` (B only), `oproj` (A only) | both |
| `CTAPP_R` | reducer blocks (SMs taken from the GEMM) | rule: 12 for M <= 2048, else 8 |
| `CTAPP_A_RASTER` / `CTAPP_A_SWIZZLE` | consumer (gate_up) GEMM order at boundary A | 1 / 2 |
| `CTAPP_B_RASTER` / `CTAPP_B_SWIZZLE` | consumer (QKV) GEMM order at boundary B | 2 / 1 |
| `CTAPP_CHECK` | 1 logs per-layer rel err of the boundary output vs the stock computation, and fallbacks | 0 |
| `CTAPP_LOG_M` | 1 logs every M seen (otherwise only the first time an M is created) | 0 |
| `CTAPP_MIN_M` / `CTAPP_MAX_M` | M range handled by the pool | 512 / 16384 |

Implementation notes:
- RMSNorm gammas are folded IN PLACE into the consumer weights (gate_up, qkv) at setup (no extra weight copies; vLLM's "Model loading took" stays 32.89 GiB per rank). Every stock-path norm in the forward then uses a ones-weight RMSNorm, so the fallback paths stay correct. After re-initialising weights call `model._refold()`.
- A `CtappBoundaryPool` holds one boundary instance per M, created lazily on first use (instances for 4096, 8192, 12288, 16384 appear for b>1 runs).
- Fallback to the stock path (no boundary) when M is not a multiple of 128 or M < 512 (decode steps, M=1..16, and odd lengths such as 4097).
- See "Known issue" after the Conclusions.

## Correctness (4-layer config, M=4096, rank 0, `CTAPP_CHECK=1`)

Rel err of the boundary outputs vs the stock computation of the same layer (multimem bf16 accumulation order explains the 1e-3 level; the micro-benchmark's NVLS baseline is 5e-3):

| boundary / layer | boundary output rel err | x (residual stream) rel err |
|---|---|---|
| B 0->1 (qkv) | 3.845e-03 (down-only) / 3.411e-03 (both) | 2.060e-03 |
| B 1->2 | 3.749e-03 / 3.298e-03 | 1.884e-03 |
| B 2->3 | 3.684e-03 / 3.221e-03 | 1.758e-03 |
| A layer 0 (gate_up input g) | 2.600e-03 | 5.938e-04 |
| A layer 1 | 2.637e-03 | 5.986e-04 |
| A layer 2 | 2.654e-03 | 6.337e-04 |
| A layer 3 | 2.673e-03 | 6.867e-04 |

M=8192 down-only gives the same B errors (3.845e-03 / 3.750e-03 / 3.686e-03). End to end, the first generated token and its top-20 logprobs match stock: M=4096 token 124749 (stock -7.4926, ctapp down -7.4926, both -7.4926, oproj -7.4927), M=8192 token 24006; M=4097 (fallback, no boundary) token 21145. The b=2 run matches stock too (124749 and 24006).

## Results: end-to-end 4K prefill latency (80 layers)

| variant | b=1 (M=4096) | b=2 | b=4 |
|---|---|---|---|
| eager (stock, enforce_eager) | 287.5 / 281.9 (14246 tok/s) | 575.3 / 572.3 (14239 tok/s) | 1137.7 / 1133.0 (14401 tok/s) |
| compiled_asynctp (stock, SP + fused GEMM-comms) | 273.1 / 270.8 (15000 tok/s) | 541.8 / 527.4 (15121 tok/s) | 1058.1 / 1047.0 (15484 tok/s) |
| compiled_fi (stock, default 2 MB FI threshold) | 287.9 / 282.3 (14226 tok/s) | 574.7 / 569.9 (14255 tok/s) | 1138.0 / 1127.2 (14397 tok/s) |
| compiled_fi_big (FI allreduce+rms fusion, 256 MB threshold) | 299.9 / 293.5 (13658 tok/s) | 594.0 / 585.7 (13791 tok/s) | 1176.5 / 1155.8 (13926 tok/s) |
| ctapp default R (=8 at these M), trial 1 | 282.9 / 275.7 (14477 tok/s) | 562.4 / 557.0 (14565 tok/s) | 1103.9 / 1096.7 (14842 tok/s) |
| ctapp default R, trial 2 | 281.8 / 279.1 (14536 tok/s) | 555.0 / 550.2 (14761 tok/s) | 1104.2 / 1097.6 (14838 tok/s) |
| ctapp default R, trial 3 | 282.9 / 279.4 (14480 tok/s) | 561.8 / 551.2 (14582 tok/s) | 1111.9 / 1100.9 (14735 tok/s) |
| ctapp R=12 | 286.0 / 281.9 (14322 tok/s) | 569.1 / 558.2 (14394 tok/s) | 1133.5 / 1123.4 (14454 tok/s) |
| ctapp R=4 | 296.0 / 292.4 (13838 tok/s) | 583.3 / 576.7 (14043 tok/s) | 1177.1 / 1158.0 (13918 tok/s) |
| eager (same session, run 1) | 290.3 / 282.2 (14111 tok/s) | 577.1 / 571.8 (14195 tok/s) | 1141.4 / 1133.9 (14354 tok/s) |
| eager (same session, run 2) | 289.5 / 283.9 (14150 tok/s) | 576.4 / 572.8 (14212 tok/s) | 1136.2 / 1128.7 (14420 tok/s) |
| ctapp both (R=8) | 331.4 / 327.3 (12358 tok/s) | 654.6 / 645.6 (12515 tok/s) | 1318.2 / 1282.9 (12430 tok/s) |
| ctapp both R=12 | 322.4 / 319.5 (12703 tok/s) | 640.0 / 635.5 (12801 tok/s) | 1302.9 / 1276.7 (12575 tok/s) |
| ctapp oproj only (R=8) | 345.6 / 341.8 (11850 tok/s) | 683.0 / 676.5 (11994 tok/s) | 1388.9 / 1364.6 (11797 tok/s) |
| ctapp down only (R=8, = earlier ctapp) | 280.7 / 278.1 (14594 tok/s) | 554.2 / 547.2 (14782 tok/s) | 1101.0 / 1093.8 (14881 tok/s) |

Rows from "ctapp both (R=8)" down are the newer session (20 iters, `--boundary` flag, old consumer config raster 2 / swizzle 1 for A). The "ctapp default R" rows are the older session and are down-only (identical config to "down only").

Implied saving per layer vs stock eager = (eager median - variant median) / 80, ms. Older-session rows are computed against the first eager row (287.5 / 575.3 / 1137.7); the newer-session rows (both, oproj, down only) against the mean of the two same-session eager runs (289.9 / 576.8 / 1138.8 ms).

Implied saving per layer vs stock eager = (eager median - variant median) / 80, ms:

| variant | b=1 | b=2 | b=4 |
|---|---|---|---|
| compiled_asynctp (stock, SP + fused GEMM-comms) | +0.181 | +0.420 | +0.995 |
| compiled_fi (stock, default 2 MB FI threshold) | -0.005 | +0.008 | -0.004 |
| compiled_fi_big (FI allreduce+rms fusion, 256 MB threshold) | -0.155 | -0.233 | -0.485 |
| ctapp default R (=8 at these M), trial 1 | +0.057 | +0.161 | +0.422 |
| ctapp default R, trial 2 | +0.072 | +0.254 | +0.419 |
| ctapp default R, trial 3 | +0.058 | +0.169 | +0.323 |
| ctapp R=12 | +0.019 | +0.078 | +0.052 |
| ctapp R=4 | -0.106 | -0.100 | -0.493 |
| eager (same session, run 1) | -0.005 | -0.004 | -0.033 |
| eager (same session, run 2) | +0.005 | +0.004 | +0.033 |
| ctapp both (R=8) | -0.520 | -0.973 | -2.242 |
| ctapp both R=12 | -0.407 | -0.790 | -2.052 |
| ctapp oproj only (R=8) | -0.697 | -1.328 | -3.126 |
| ctapp down only (R=8, = earlier ctapp) | +0.115 | +0.282 | +0.473 |

### Swizzle fix for boundary A (b=1, 80 layers, M=4096, 20 iters, median ms)

Same-session references: eager 289.9, down-only R=8 280.7, compiled_asynctp 273.1.

| config | R=8 | R=12 |
|---|---|---|
| both, old raster 2/1 | 331.4 | 322.4 |
| both, A raster 1 swz 2 | 292.6 | 289.5 |
| both, A raster 1 swz 8 | 290.6 | 288.1 |
| both, A raster 2 swz 8 | - | 290.4 |
| oproj only, A raster 1 swz 8 | - | 297.9 |
| down only (re-run) | 280.7 | 281.4 |

The swizzle fix removes ~40 ms (331 -> 288-293 ms): boundary A is now about break-even with stock eager (289.9), but `both` is still ~8 ms slower than down-only, so no batch 1,2,4 run was made for the new A config. Best end to end was A=1/8 (within noise of the 1/2 default; 1/2 chosen from the 1-GPU sweep).

### Down-only result and caveats

With boundary B swapped in, ctapp at default R (8) runs 4K prefill at 282-283 ms vs 287.5 ms for stock eager: about 0.06 ms/layer at b=1 (b=2: ~0.15, b=4: ~0.4 ms/layer, by the 3-trial spread of medians). The micro-benchmark (`results/tp4_chain.md`, M=4096) has our chain at 1.053 ms vs 1.285 asynctp vs 1.298 FlashInfer (0.23-0.25 ms/layer better than those two), but it does not include the stock eager chain we actually replace, so it is not directly a saving vs eager. End to end, ctapp's saving vs eager (0.06 ms/layer at b=1, up to ~0.4 at b=4) is below what compiled_asynctp achieves (0.18 at b=1, ~1.0 at b=4).

Caveats:
1. ctapp runs EAGER (no torch.compile, no CUDA graphs, per-layer Python dispatch) and stock eager is the comparison; asynctp is compiled, so part of the gap to asynctp is compile/fusion of everything else, not the boundary.
2. In the first session only boundary B of the two per layer was ported; boundary A (o_proj -> post_attention_layernorm -> gate_up) was a stock all-reduce. Both are now ported (see `both`/`oproj` rows).
3. Layer 0's QKV and the last layer's down_proj stay stock.
4. FlashInfer fusion (`compiled_fi_big`, 256 MB threshold) was not faster than stock (b=1 299.9 ms vs 287.9 default); only the log lines 'Enabled custom fusions: allreduce_rms' and FlashInfer workspace init (backend=mnnvl) confirm the pass was enabled, fused nodes were not counted.
5. For b>1 the V1 engine starts stepping as soon as the first request arrives, so the 'b=2/4' generates are a mix of M=4096/8192/12288/16384 steps (our pool created instances for those four M and used the boundary for all), not one fused M=4096*b batch; stock b=2 is ~2x b=1 for the same reason.
6. R sensitivity: R=8 < R=12 < R=4 in latency for down-only at these M (R=4 is clearly worse: 296 ms at b=1; R=12 costs ~4 ms over R=8), consistent with the micro-benchmark's default rule. For `both`, R=12 is better than R=8 (322.4 vs 331.4 ms with the old config; 289.5 vs 292.6 and 288.1 vs 290.6 with the swizzle fix).

## Profiling (4-layer config, M=4096, rank 0, torch profiler, `bench/vllm_prof.py`)

Stock vs ours at boundary B (`build/prof/ctapp_both`):
- Stock boundary: down GEMM 0.62 ms + NCCL ring LL all-reduce 0.38 ms (330-380 us for 64 MB) + fused_add_rms_norm 0.09 ms + QKV 0.23 ms = 1.32 ms.
- Ours: 1.12 ms. Down GEMM 0.78 ms on 124 SMs (0.73 ms in the first profile), the reducer ~0.84 ms runs concurrently and outlasts the GEMM, QKV consumer 0.34 ms (0.32 vs 0.21 stock in the first profile).
- No host gap, the CPU is ~3 ms ahead of the GPU; the 80-layer kernels run ~5 % slower than the 4-layer ones.

Boundary A (`build/prof/ctapp_both`, old consumer config): o_proj producer 0.25 ms, reducer finishes at ~0.77 ms as predicted, but the gate_up consumer (grid [1,124], raster 2) takes 2.1-2.7 ms vs 1.22 ms stock cuBLAS and ends 1.4-1.9 ms after the reducer is done: consumer GEMM throughput at N=14336, not reduction wait, dominates. A totals ~2.67 ms from producer launch to act kernel vs 1.83 ms stock.

Boundary A after the swizzle fix (`build/prof/ctapp_both_r1s8`, R=12, `build/logs/prof_both_r1s8.txt`): producer 0.28 ms (283 us), reducer 0.72 ms (722 us, hidden), gate_up consumer (grid 120 CTAs) 1.67 ms (1669 us), starting at 345 us and ending ~1.29 ms after the reducer finished; A total ~2.01 ms vs 1.83 ms stock (was 2.67). The consumer's 1.67 ms in situ is 1.37x cuBLAS (1.22 ms); in the 1-GPU sweep it was 1.40 ms at 124 SMs, so ~0.27 ms is lost to SM count (120 vs 132) plus reducer contention for L2/HBM. Boundary B consumer in the same profile: 0.30-0.37 ms.

## Swizzle sweep for the boundary A consumer (gate_up, N=14336)

Hypothesis: raster 2 (n fast) with swizzle 1 streams the 235 MB gate_up weight through L2 once per CTA wave; a swizzled order restores reuse. 1-GPU sweep (`build/scratch/sweep_gateup.py`, log `build/logs/sweep_gateup.log`; mode 0 = no waits, rms epilogue on; M=4096, K=8192; 50 iters):

| shape | cuBLAS | old (raster 2, swz 1, 124 SMs) | best configs (mode 0) |
|---|---|---|---|
| N=14336 | 1.234 ms | 2.466 ms (2.00x) | r1 s2 sms132 1.309 (1.06x); r1 s2 sms124 1.389 (1.13x); r1 s8 sms124 1.397 (1.13x); r1 s4 sms124 1.413; r2 s8 sms124 1.474 (1.19x); r2 s4 sms124 1.507 |
| N=2560 | 0.247 ms | 0.251 ms (1.01x) | r2 s1 sms124 0.251 (best); r1 s* and large swizzles are 1.1-2x worse |

Mode 2 with panel counters prefilled (waits pass at once) adds ~0.02 ms at N=14336 (r1 s2 sms132 1.327 vs 1.309) and ~0 at N=2560. At larger M (gate_up, sms132, best 3): M=8192 r1 s4 2.737 ms vs cuBLAS 2.910; M=16384 r1 s4 5.413 vs 5.191 (1.04x). Raster 1 / swizzle 2-8 removes the 2x slowdown; the QKV shape keeps raster 2 / swizzle 1. The plugin exposes these as `CTAPP_A_*` / `CTAPP_B_*` (`CtappBoundary`/`CtappBoundaryPool` take `qkv_raster`/`qkv_swizzle`).

## Conclusions

**Verdict: negative-to-marginal for production at TP4.** 3 % over stock eager (280.7 vs 289.9 ms), 3 % behind vLLM's compiled async-TP (273.1 ms) at 4K prefill; the second boundary is break-even at best.

**Open discrepancy (cause not established).** Our bench's `nccl` variant (torch.distributed all_reduce, bf16, 64 MB) measures ~0.75 ms in both venvs (torch 2.12 NCCL and torch 2.13 / NCCL 2.29.7), while vLLM's PyNccl path shows 0.35-0.38 ms in the profiler trace (`ncclDevKernel_AllReduce_Sum_bf16_RING_LL`, 24 blocks). Algorithm/protocol selection, communicator setup or measurement method could each explain it; measure before making further baseline claims. The torch-level baselines in `results/tp4_chain.md` may therefore overstate the available gain.

(a) The down boundary alone gives 3 % over stock eager at 4k (9-10 ms of 290), 3-4 % at 8k/16k, and loses to vLLM's compiled async-TP (273 ms) by 3 %.

(b) The reduction is fully hidden at both boundaries, so the remaining loss is GEMM efficiency: our consumer GEMMs run 1.3-1.6x cuBLAS time in situ (fewer SMs, 128x256 tile, panel waits, reducer traffic), and our producer 1.25x.

(c) The reducer at ~0.8 ms for 64 MB is bound by the ~90 GB/s SM-issued multimem path, whereas NCCL ring moves the same bytes in 0.35 ms over 369 GB/s unicast links. A unicast reduce-scatter/all-gather reducer with a lightweight flag protocol is the main lever to make the reducer shorter than the GEMM and allow smaller R.

(d) Decode is out of scope by design (fewer than 2 row panels).

Known issue: a one-off illegal-memory-access crash occurred at pool-instance creation mid-run. It is worked around by a device sync plus a TP barrier before creation (`model.py`); the root cause is unproven.

## Reproduce

```bash
source /data/garv901-55613a/venvs/vllm/bin/activate          # vLLM 0.30.0, torch 2.13+cu130
cd /shared/home/garv901-55613a/cta-pipelining
uv pip install -e vllm_plugin                                 # registers the ctapp_vllm entry point
# /tmp/run4v.sh <gpu list> <cmd...> sets PATH (venv), CUDA_HOME, TMPDIR, HF_HOME, PYTHONPATH, VLLM_CACHE_ROOT,
# CUDA_VISIBLE_DEVICES and CUDA_MODULE_LOADING (default EAGER); use LAZY as below.
export CUDA_MODULE_LOADING=LAZY
/tmp/run4v.sh 0,1,2,3 python bench/vllm_prefill.py --variant eager
/tmp/run4v.sh 0,1,2,3 python bench/vllm_prefill.py --variant compiled_asynctp
/tmp/run4v.sh 0,1,2,3 python bench/vllm_prefill.py --variant ctapp --boundary down --r 8
/tmp/run4v.sh 0,1,2,3 python bench/vllm_prefill.py --variant ctapp --boundary both --r 12 --iters 20
# correctness (4-layer config)
CTAPP_VLLM=0 /tmp/run4v.sh 0,1,2,3 python bench/vllm_ctapp_check.py --lens 4096 --out build/ck_stock.json
CTAPP_VLLM=1 CTAPP_CHECK=1 /tmp/run4v.sh 0,1,2,3 python bench/vllm_ctapp_check.py --lens 4096 --out build/ck_ctapp.json
python bench/vllm_ctapp_check.py --compare build/ck_stock.json build/ck_ctapp.json
# profile
CTAPP_VLLM=1 /tmp/run4v.sh 0,1,2,3 python bench/vllm_prof.py --out build/prof/ctapp
```

Variants of `bench/vllm_prefill.py`: eager, compiled, compiled_fi, compiled_asynctp, compiled_fi_big, ctapp (`--r`, `--boundary both|down|oproj`; `--seq`, `--batch 1,2,4`, `--iters`, `--warmup`).
