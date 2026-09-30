# CTA-Pipelining on H100

This repository explores **CTA-pipelining** ([arXiv:2607.07862](https://arxiv.org/abs/2607.07862)) on a 4x H100 SXM node. The technique runs dependent GEMMs on different GPUs at the same time:
- The producer GEMM writes each finished output tile straight into the next GPU's memory over NVLink, then raises a flag.
- A consumer CTA on the next GPU starts a tile as soon as every tile it depends on has arrived. It does not wait for the whole first GEMM to finish, as a plain layer-per-GPU pipeline would.

The paper's end-to-end results come from B200 GPUs with NVLink 5 and an NVSwitch. This repo asks how much of the benefit carries over to Hopper (H100), with NVLink 4 point-to-point links and no switch. It compares against the two usual ways of splitting a model across GPUs: micro-batching and tensor parallelism.

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
- CUDA 12.8, matching the PyTorch build. Set `CUDA_HOME` if it isn't at `/usr/local/cuda-12.8`.
- PyTorch 2.9 (cu128).

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
