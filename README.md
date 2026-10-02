# CTA-Pipelining on H100

This repository explores **CTA-pipelining** ([arXiv:2607.07862](https://arxiv.org/abs/2607.07862)) on a 4x H100 SXM node. The technique runs dependent GEMMs on different GPUs at the same time:
- The producer GEMM writes each finished output tile straight into the next GPU's memory over NVLink, then raises a flag.
- A consumer CTA on the next GPU starts a tile as soon as every tile it depends on has arrived. It does not wait for the whole first GEMM to finish, as a plain layer-per-GPU pipeline would.

The paper's end-to-end results come from B200 GPUs with NVLink 5 and an NVSwitch. This repo asks how much of the benefit carries over to Hopper (H100), with NVLink 4 point-to-point links and no switch. It compares against the two usual ways of splitting a model across GPUs: micro-batching and tensor parallelism.

## Status and conclusion (2026-10-02)

**Verdict: for production (vLLM, Llama-70B, TP4, H100) this is a negative-to-marginal result.** The mechanism works and is checked against a reference, but the end-to-end gain is within a few percent of stock and below vLLM's compiled async-TP.

- **Built:** a CTA-pipelined TP4 down-proj -> RMSNorm -> QKV boundary (CUTLASS SM90 cooperative GEMM + standalone NVLS multimem reducer + panel-waiting consumer GEMM), productionised in a vLLM plugin that covers both all-reduce boundaries of every Llama layer.
- **Results:** stand-alone, the chain is 18 % faster than async-TP at 4k tokens (1.053 vs 1.285 ms) against the strongest torch-level baselines. Inside vLLM it is only 3 % faster than stock eager at 4K prefill (280.7 vs 289.9 ms) and 3 % slower than vLLM's compiled async-TP (273.1 ms). The second boundary (o_proj -> gate_up) is break-even at best. Decode is out of scope.
- **Why (negative findings):**
  - (a) *Wrong baseline.* The Step-0 "strong baselines" were torch symmetric-memory variants and torch.distributed NCCL, where the 64 MB all-reduce costs ~0.7 ms. vLLM's own NCCL path does it in 0.35-0.38 ms (ring, LL protocol, ~275 GB/s effective over unicast links), and that is what the kernel actually competes with.
  - (b) *Wrong mechanism.* NVLS multimem was chosen by comparing byte counts (multimem moves P once, ring 1.5 P), but SM-issued multimem throughput is only ~90 GB/s per GPU vs 369 GB/s unicast. The reducer (0.8 ms) outlasts the GEMM it hides behind and costs 8 SMs.
  - (c) *GEMM efficiency.* Our GEMMs run 1.25-1.6x cuBLAS time in situ (fewer SMs, one tile shape, panel waits, traffic contention), which eats the hidden communication at the o-proj boundary.
- **What would change the outcome:** first a unicast reduce-scatter/all-gather reducer with NCCL-class bandwidth (a go/no-go experiment), then consumer-GEMM efficiency. See `results/vllm_prefill_4k.md` and PLAN.md Phase 5 ("Conclusion and incorrect assumptions"). One discrepancy is still open: our bench's NCCL all-reduce measures ~0.75 ms, vLLM's 0.35-0.38 ms.

All measured numbers below and in `results/` are kept as measured; the stand-alone gains in particular are against torch-level baselines and overstate what an engine can gain.

## Workload and baselines
- **Workload:** a 2-layer MLP on 2 GPUs, `Y1 = X·W1ᵀ` on GPU A and `Y2 = Y1·W2ᵀ` on GPU B.
  - `N = K = 8192`, BF16 inputs and outputs, FP32 accumulation, no activation between the layers.
  - M (tokens) ranges from 1k to 32k.
- **Baselines:**
  - **Micro-batching:** cuBLAS GEMMs on chunks of M, with peer copies overlapped between chunks. The chunk size is swept and the best is kept.
  - **Tensor parallelism (TP2):** Megatron-style column/row sharding with cuBLAS and an NCCL all-reduce.
  - **Ideal:** one GEMM's time for two layers. This is what perfect overlap would give.
- **Timing:** all methods are timed with a gated harness. Every stream blocks on a host flag until all work is enqueued, so launch overhead is excluded. The harness was checked against Nsight Systems.

## How it works
The protocol follows the paper and is plain user-level CUDA:
- **Dependency array:** a CSR list for each producer tile, giving the consumer tiles that need it. For an MLP this is one row panel of the output.
- **Scoreboard:** atomic counters in producer memory, one per consumer row panel.
- **Workqueue:** a ring buffer in consumer memory.
  - The producer pushes the IDs of consumer tiles that have become ready.
  - Consumer CTAs poll the queue locally for their next tile.
- **Ordering:** a system-scope fence plus acquire/release atomics make sure a consumer sees a tile's data before it sees the flag. A proxy fence then lets the consumer's TMA loads read that peer-written data.

The GEMMs are CUTLASS 4.8 SM90 kernels:
- The CTA-pipelining logic is a small prologue (fetch a tile ID from the queue) and a small epilogue (signal, then push to the queue).
- These live in `csrc/`. CUTLASS itself is a pinned, unmodified submodule.

## Optimization passes so far
| Pass | What changed | Outcome |
|---|---|---|
| 0. Baseline | CUTLASS's classical `KernelTma` GEMM (one tile per CTA) | 0.62–0.68x cuBLAS speed. Writing its output from registers straight to a peer GPU was 3.1–3.6x slower than local. |
| 1a. Peer-store epilogue | Stage each output tile through shared memory and write it with 16-byte coalesced stores | Peer-write overhead dropped from 3.3x to 1.12x of a local write |
| 1b. Protocol | Dependency array, scoreboard and workqueue added as prologue/epilogue hooks | Bit-exact against a sequential run. A negative control (consumer skips the wait) reliably fails. |
| 1c. Producer tile order | Grouped order (8 tile-rows at a time) instead of row-major | Row-major re-streamed the weights from HBM on every wave and was 2.2x slower |
| 2. Harness and baselines | Gated timing, micro-batching, TP2 | CTA-pipelining beats TP2 at large M but loses to micro-batching |
| 2b. Per-CTA timeline | `%globaltimer` stamps on both GPUs, with the clocks aligned between GPUs | The overhead comes from each wave's burst of remote stores draining while no CTA computes, plus a one-wave tail |
| 2c. Row-panel scoreboard | One counter per consumer row panel instead of one per consumer tile, so each producer tile does 1 atomic instead of 64 | Small gains. Confirmed that the cost is store drain, not atomic contention. |
| 3.0 Persistent kernel | CUTLASS's warp-specialized persistent cooperative kernel | 0.90–1.07x cuBLAS speed. Writing its output to the peer GPU is 1.2–1.7x slower, because all persistent CTAs burst their stores at the same time. |

Current standing, from the classical-kernel version (µs, 2x H100):

| M | 1k | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|---|
| CTA-pipelining (`KernelTma`) | 476 | 786 | 1415 | 2433 | 4519 | 8668 |
| Micro-batching (cuBLAS) | 336 | 548 | 908 | 1626 | 3086 | 5959 |
| Tensor parallel (TP2) | 368 | 700 | 1348 | 2649 | 5191 | 10200 |
| Ideal (one GEMM's time) | 178 | 346 | 694 | 1398 | 3163 | 6384 |

- **What stands out on H100:** micro-batching is already close to ideal at large M. The room for CTA-pipelining is at small and medium M, and against TP at every M.
- **Where the classical kernel falls short:** the kernel itself and the non-overlapped store drain. The persistent kernel removes the first.

## Roadmap
- **Persistent kernel, in progress.**
  - A custom tile scheduler whose scheduler warp polls the workqueue. This is the Hopper analogue of the paper's Blackwell design.
  - An epilogue wrapper that signals a tile only after its TMA stores have landed. Signalling can be deferred by one tile, so the drain overlaps the next tile's math.
  - Full-tile store buffering.
- **4 GPUs:** a 4-layer chain, and CTA-pipelining combined with tensor parallelism (two pipelined pairs plus an all-reduce), compared against TP4.

`PLAN.md` has the detailed plan and the per-phase results. The `results/` directory has the raw tables and plots.

## Phase 4: TP4 down-proj -> RMSNorm -> QKV
CTA-pipelining at the TP4 boundary of Llama-70B: down-proj (K=28672 sharded, N=8192) -> all-reduce -> residual add -> RMSNorm -> QKV (N=10240, column-sharded), one process per GPU on node1 (8x H100 NVSwitch). The down-proj GEMM signals finished tiles; a small reducer kernel (R=12 SMs for M <= 2048, else 8) does the NVLS `multimem.ld_reduce`, adds the residual, writes the replicated x with `multimem.st` and signals per 128-row panel; the QKV GEMM waits per panel and applies RMSNorm in its epilogue. The epoch is device-side so the whole forward is CUDA-graph capturable. Full tables (b2b and per-step, exposed %) are in `results/tp4_chain.md`. The gains below are against torch-level baselines and overstate what is available in an engine; see [Status and conclusion](#status-and-conclusion-2026-10-02).

Files: `csrc/ctapp_tp4.cuh`, `csrc/gemm_tp4.cu`, `csrc/sm90_gemm_coop_ctapp.hpp` (generated by `scripts/gen_coop_ctapp.py`), `ctapp/tp4.py`, `tests/test_tp4.py`, `bench/tp4_bench.py`, `results/tp4_chain.md`.

```python
from ctapp.tp4 import CtappBoundary
b = CtappBoundary(M, Kr, N2r, rank, world, group_name, device, resid=True, use_graph=False)
b.set_weights(Wdown_r, Wqkv_r_folded)   # (N1, Kr) and (N2r, N1), gamma folded into Wqkv
out = b.forward(h_r, resid)             # (M, N2r) RMSNorm'd QKV output shard
x = b.x()                               # replicated residual stream (M, N1)
```

Back-to-back, world 4, 50 iterations (gain = 1 - ctapp / best baseline):

| M | ideal (ms) | ctapp (ms) | best baseline (ms) | gain |
|---|---|---|---|---|
| 1024 | 0.246 | 0.344 (ctapp) | 0.344 (flashinfer) | 0 % |
| 2048 | 0.512 | 0.590 (ctapp) | 0.658 (flashinfer) | 10 % |
| 4096 | 1.028 | 1.053 (ctapp) | 1.285 (asynctp) | 18 % |
| 8192 | 2.046 | 2.236 (ctapp) | 2.531 (asynctp) | 12 % |
| 16384 | 4.248 | 4.754 (ctapp) | 5.022 (asynctp) | 5 % |

Caveat: vLLM's eager NCCL path does the 64 MB all-reduce ~2x faster than the `nccl` baseline here (see Status and conclusion). Baselines: multimem all-reduce, FlashInfer fused all-reduce + residual + RMSNorm (`trtllm_allreduce_fusion`), async-TP, NCCL.

Known limitations:
- Epoch parity needs two graphs: the QKV epilogue's `rowss` pointer is baked in and double-buffered, so `use_graph=True` captures once per parity.
- M must be a multiple of 128 and N1 = 8192 is fixed.
- Requires sm_90a, NVLS multicast support and torch symmetric memory.
- The QKV GEMM runs on SMs - R SMs even though the reducer is idle then.

### vLLM plugin
`vllm_plugin/` is a vLLM 0.30.0 plugin that swaps both TP all-reduce boundaries of every Llama layer (o_proj -> norm -> gate_up and down_proj -> norm -> next QKV) for `CtappBoundary`, falling back to the stock path for M < 512 or M not a multiple of 128. Install with `uv pip install -e vllm_plugin` into the vLLM venv and enable with `CTAPP_VLLM=1` (`CTAPP_BOUNDARY=both|down|oproj`, `CTAPP_R`, `CTAPP_A_RASTER/SWIZZLE`, `CTAPP_B_RASTER/SWIZZLE`, `CTAPP_CHECK`). Details, ablation and profiles: `results/vllm_prefill_4k.md`. Result: on Llama-3.1-70B TP4 4K prefill the down boundary gives 3 % over stock eager (280.7 vs 287.5 ms; 289.9 in the same session as the ablation) but trails vLLM's compiled async-TP (273.1 ms) by 3 %, because the reduction is hidden but our GEMMs run 1.3-1.6x cuBLAS time in situ. Overall a negative-to-marginal production result; see Status and conclusion.

## Layout
```
csrc/          CUDA/C++: CUTLASS kernel wrappers and the pipelining hooks (ctapp.cuh)
ctapp/         Python: JIT build (ext.py), pipeline setup, methods, timing harness
bench/         benchmark scripts (one per experiment)
tests/         correctness tests (bit-exact vs sequential, negative control)
results/       markdown/CSV results and plots
scripts/       helpers (pick_gpus.sh picks idle GPUs)
third_party/   CUTLASS 4.8.0 submodule (unmodified)
```

## Getting started
**Requirements:**
- H100 (SM90a) GPUs with peer access over NVLink.
- CUDA matching the PyTorch build: 12.8 with torch 2.9 (cu128) or 12.9 with torch 2.12 (cu129). Set `CUDA_HOME`
  (default `/usr/local/cuda-12.8`); `ninja` must be on `PATH` (it is in the venv's `bin/`).
- On a Slurm cluster the GPUs are only visible inside an allocation: hold one with an `sbatch ... sleep infinity`
  job and run commands via `srun --jobid=<id> --overlap bash -c 'export CUDA_VISIBLE_DEVICES=...; python ...'`
  (set `CUDA_VISIBLE_DEVICES` inside the `bash -c` string, Slurm overrides it otherwise).

```bash
git clone --recursive https://github.com/garv901/cta-pipelining.git
cd cta-pipelining
export TMPDIR=$PWD/build            # build scratch space

PAIR=$(bash scripts/pick_gpus.sh 2 60)   # an idle GPU pair, e.g. "3,2" (producer, consumer)
CUDA_VISIBLE_DEVICES=$PAIR timeout 600 python3 tests/test_ctapp.py     # correctness
CUDA_VISIBLE_DEVICES=$PAIR timeout 3000 python3 -u bench/fig5.py       # latency sweep vs baselines
```

- The torch extension is JIT-compiled on first use into `build/ext`, which takes about 2 minutes.
- Wrap GPU runs in `timeout`: a protocol bug shows up as a kernel that polls forever.
