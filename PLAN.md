# CTA-pipelining on 4x H100: plan

## Context
The paper (arXiv 2607.07862) runs dependent GEMMs on different GPUs at the same time. A consumer CTA on GPU B starts a tile as soon as the producer CTAs it depends on (on GPU A) have finished. The paper reports 31.8% lower latency than micro-batching and 29.6% lower than TP, but all of those numbers are from B200.
- **Hopper evidence in the paper:** only an overhead study of the classical SM90 `KernelTma` kernel on H200, where the protocol cost is exposed on every wave.
- **Goal:** reproduce the 2-layer MLP result (Fig 5) on H100 against micro-batching and TP, then combine CTA-pipelining with TP (Fig 7) on 4 GPUs.

Decisions already made:
- Kernel path: `KernelTma` first to get the protocol correct, then the SM90 warp-specialized cooperative persistent kernel.
- Harness: Python with a torch C++/CUDA extension.
- Signalling: the paper's exact scheme (dependency array, scoreboard, workqueue).
- GPUs: 2-GPU work now; 4-GPU phases wait until GPUs 1 and 2 are free.

Implementation is done by Sonnet subagents. Opus is used only for Phase 3.

## Fixed parameters
- **Workload:** Y1 = X·W1ᵀ, then Y2 = Y1·W2ᵀ.
  - X is M×8192; W1 and W2 are 8192×8192.
  - Weights are stored N×K (torch `Linear` layout), so both GEMMs are TN (K-major) GEMMs.
  - BF16 in and out, FP32 accumulate, no activation between the GEMMs.
  - M ∈ {1024, 2048, 4096, 8192, 16384, 32768}.
- **GPU pair for 2-GPU runs:** producer = GPU 3, consumer = GPU 0. Check `nvidia-smi` before every run, record its output next to each result, and never run on GPUs 1 or 2 while other users' jobs are on them.
- **Toolchain:** `/usr/local/cuda-12.8`, matching torch 2.9.1+cu128. CUTLASS lives in `third_party/cutlass` and is never edited.
- **Cluster shape:** 1×1×1. Tile IDs are handed out dynamically, so CTAs in a cluster would get unrelated tiles and TMA multicast would break.

## Layout
```
PLAN.md
third_party/cutlass/     pinned clone, commit recorded in third_party/CUTLASS_COMMIT
csrc/                    CUDA sources for the torch extension
ctapp/                   python: ext.py (JIT build), timing, methods
bench/                   benchmark scripts
tests/                   correctness tests
results/                 csv + md for each experiment
```

## Phase 0: setup and gate benchmark (~1 h)
1. Clone CUTLASS. Build a JIT torch extension containing the unmodified `KernelTma` BF16 TN GEMM in a few tile shapes. Its epilogue stores directly from registers to global memory, with no TMA store.
2. `bench/gate.py`, on GPU 3:
   - Measure `KernelTma` TFLOP/s against cuBLAS (`F.linear`) across M.
   - Time the best `KernelTma` config writing D into peer memory (GPU 0) against local memory.
   - Measure P2P bandwidth for SM stores and for the copy engine.
3. Write `results/gate.md`.
- **Gate:** the `KernelTma` / cuBLAS ratio decides how much Phase 3 matters. Phase 1 continues regardless.

**Phase 0 result** (`results/gate.md`, 2026-09-29):
- Best `KernelTma` config is 64x256x64. It reaches 0.62–0.68x of cuBLAS speed (455–537 vs 700–800 TFLOP/s).
- Writing D into peer memory through the register (NoSmem) epilogue makes the kernel **3.1–3.6x slower**: only about 17 GB/s reaches the other GPU.
- 16-byte coalesced SM stores reach 124 GB/s peer-to-peer. The copy engine reaches 132 GB/s.
- This adds Phase 1a below: stage the output tile through shared memory and store it with wide stores.

## Phase 1: protocol in `KernelTma`, 2 GPUs (~3 h)
**1a (added after Phase 0).** Swap in the smem-staged vectorized epilogue from `epilogue/collective/sm70_epilogue_vectorized.hpp`.
- Target: peer store time ≤ 1.3x local.
- If the target is missed, stop before 1b.
Our kernel copies `KernelTma::operator()` from `sm90_gemm_tma.hpp` into `csrc/`. It inherits the CUTLASS kernel's types and static functions, following the pattern of `experimental/distributed/kernel/dist_gemm_kernel_wrapper.hpp`, and adds two hooks. Every kernel is launched as a 1-D grid with one CTA per output tile. The tile ID is row-major: `t = m * tiles_n + n`.

**Data structures** (int32). Each lives on the GPU that reads it most:

| Structure | Lives on | Contents |
|---|---|---|
| Workqueue `{entries[cap], head, tail}` | GPU of the kernel that consumes it | `entries` start as EMPTY = -1. `cap` = number of tiles of that kernel, so a single run never wraps. The slot is `idx % cap`. |
| Source queue of the first kernel | Producer GPU | Pre-filled with tile IDs in row-major order, `tail = T`. This is how the paper fixes the producer's CTA order. |
| Dependency array | Producer GPU | CSR: `dep_offsets[P+1]` and `dep_consumers[]`. For each producer tile, the consumer tiles whose input row panel overlaps it. This covers the general case where BM1 ≠ BM2. |
| Scoreboard `counters[C]` | Producer GPU | Initial value = number of producer tiles each consumer tile needs. |

**Prologue** (every kernel, before the mainloop):
1. Thread 0 claims a slot: `idx = atomicAdd(src.head, 1)`.
2. Thread 0 spins until the slot is filled: `ld.acquire.sys(entries[idx % cap]) != EMPTY`. It then writes the tile ID to `__shared__ s_tile`.
3. `__syncthreads()`.
4. `fence.proxy.async.global`, so the TMA loads that follow see data another GPU wrote through ordinary stores.
5. Decode `(m, n)` and run the unmodified body.

**Epilogue** (kernels that have a downstream consumer; D points into the consumer GPU's memory):
1. The unmodified epilogue stores the tile.
2. Every thread calls `__threadfence_system()`.
3. `__syncthreads()`.
4. Threads loop over `i` in `[0, deg)`: `c = dep_consumers[off + i]`, then `old = atomic_ref<device>(scoreboard[c]).fetch_sub(1, acq_rel)`.
5. If `old == 1`, the thread pushes `c`:
   - `slot = atomic_ref<system>(*dst.tail).fetch_add(1)`
   - `atomic_ref<system>(dst.entries[slot % cap]).store(c, release)`

**Why this ordering is correct:**
- Each writer thread fences its own data at system scope, then the CTA barrier.
- The acq_rel updates on the scoreboard chain together the fences of every producer CTA that fed consumer tile `c`.
- The system-scope release store of `c` pairs with the consumer's system-scope acquire, which is followed by the barrier and the proxy fence.

**Host side:**
- Build the dependency array and scoreboard in torch on the CPU.
- Before every run, reset the heads, the non-first queues, and the scoreboard from a pristine copy.
- Launch the producer on GPU 3's stream and the consumer on GPU 0's stream, one after the other with no wait between them.

**Tests** (`tests/test_ctapp.py`):
- Fill the Y1 buffer with NaN before each run.
- 50 runs per shape. Y2 must be bit-identical to the same kernel run sequentially, and also close to torch.
- Include one config where BM1 ≠ BM2.
- Negative control: a `skip_wait` debug flag makes the consumer skip waiting. The test must then fail, which proves it can catch races.

**Phase 1 result** (2026-09-29):
- **1a:** the smem-staged epilogue brings the peer write cost down to 1.12x local at M=16384 (was 3.3x). cfg4 = 64x256.
- **1b:** the protocol is correct.
  - All cases are bit-identical, including the case where BM1 ≠ BM2.
  - In the negative control every element mismatches.
- **But no speedup yet:** the pipelined run takes 11.1 ms against 11.35 ms sequential.
  - The producer alone with signalling takes 10.8 ms, against 5.06 ms with no protocol. That is about 90 µs per wave across 62 waves.
  - Cause: with 1 CTA per SM, each CTA's system fence waits for its 32 KB remote write to drain, and nothing hides that wait. This is the paper's Fig 2a classical-kernel effect, reproduced.

**1c result:** the fence diagnosis above was WRONG.
- **What actually caused the slowdown:** the row-major producer tile order. It slows the producer 2.2x even with local output and no signalling. With cfg4:
  - row-major: 10.3 ms local, 10.6 ms peer
  - m-fastest: 4.6 / 5.1 ms
  - grouped (8 tile-rows at a time, n-major within each group): 4.9 / 5.2 ms
- **Actual signalling cost:** 0–15 µs per wave. `fence.acq_rel.sys` from the signalling threads only (F1) costs about the same as no fence, so F1 is the default.
- **Best safe variant:** cfg7 (128x128x64, 3 stages, 2 CTAs/SM), grouped8 order, F1.
  - Host-timed at M=16384: **4.69 ms**, against 11.1 ms for the same kernels run sequentially and 7.63 ms for cuBLAS sequential.
  - At M=4096: 1.53 ms, against 1.91 ms for cuBLAS sequential.
- **Tests:** 13 configurations are bit-identical, and the negative control stays sensitive.

## Phase 2: harness and baselines, 2 GPUs (~3 h)
**Timing:**
- Every stream first blocks on `cuStreamWaitValue32(gate)`, where `gate` is in pinned host memory.
- Then: record the start event on GPU 3 → enqueue all of the method's work → GPU 3 waits on each other GPU's done event → record the end event.
- The host then sets the gate. All launches happen before the gate opens, so launch overhead is hidden without CUDA graphs.
- Report the median of 20 runs after 5 warmup runs. Cross-check M = 16384 with Nsight Systems, as the paper does.

**Methods** (paper baselines use cuBLAS via torch):

| Method | What it does |
|---|---|
| (a) 1 GPU | Two GEMMs back to back on one GPU |
| (b) Sequential 2-GPU | GEMM1, then copy, then GEMM2 |
| (c) Micro-batching | Chunks along M; GEMM1 chunk on GPU 3, `cudaMemcpyPeerAsync` to GPU 0 with an event, GEMM2 chunk on GPU 0. Sweep chunk size 256 to 8192 and keep the best |
| (d) TP2 | Megatron-style: W1 split by columns, W2 split by rows, then `torch.cuda.nccl.all_reduce` |
| (e) CTA-pipelining | Our `KernelTma` kernels |
| (f) Micro-batching, same kernel | (c) using the same `KernelTma` kernel, for a like-for-like comparison |

**Output:** `results/fig5_h100.csv` plus a table in the same shape as the paper's Fig 5. If CTA-pipelining shows no gain, add per-CTA `%globaltimer` stamps to measure T1 (epilogue) and T3 (prologue), as the paper's Fig 2a does.

**Phase 2 result** (`results/fig5_h100.md`, 2026-09-29):
- **Harness:** the `cuStreamWaitValue32` gate works. Its floor is 4.6 µs, and it agrees with nsys within 1%.
- **Latency (µs):**

  | M | 1k | 2k | 4k | 8k | 16k | 32k |
  |---|---|---|---|---|---|---|
  | CTA-pipelining (`KernelTma`) | 498 | 854 | 1457 | 2523 | 4613 | 8850 |
  | Micro-batching (cuBLAS, best chunk) | 336 | 548 | 907 | 1627 | 3086 | 5968 |
  | TP2 | 369 | 702 | 1349 | 2652 | 5191 | 10204 |
  | 1-GPU cuBLAS ÷ 2 (the ideal: 2 layers in one GEMM's time) | 178 | 347 | 688 | 1460 | 3220 | 6333 |

- **CTA-pipelining vs TP2:** it beats TP2 from M=8192 up, by 5–13%.
- **CTA-pipelining vs micro-batching:** it loses to cuBLAS micro-batching at every M, and also to micro-batching built on the same kernel, by 28–43%.
- **Key insight:** on H100, cuBLAS micro-batching (chunk 512) is already at the ideal for M ≥ 16k.
  - Micro-batching / ideal = 1.89 / 1.58 / 1.32 / 1.11 / 0.96 / 0.94 for M = 1k → 32k.
  - So the only room left against micro-batching is at M ≤ 8k, which is the opposite of the paper's trend on B200.
- **Against TP2 there is room at every M:** TP2 / ideal = 1.6–2.1.
- **Anomaly:** at small M, CTA-pipelining is slower than running both GEMMs back to back on one GPU. At M=1024 it takes 498 µs against 416 µs. This needs a per-CTA timeline.

## Phase 2b: small-M timeline diagnosis (proposed, Sonnet, ~1 h)
- Record `%globaltimer` per CTA at: prologue start, tile acquired, mainloop done, store done, signal done. Record on both GPUs.
- Use this to find where the extra time goes at M=1024 and 4096. The candidates are consumer ramp, start skew between the GPUs, polling, and remote atomics.
- Any overhead found here would carry into Phase 3 as well.

**Phase 2b result** (`results/timeline.md`, `results/timeline_M1024.png`, `results/timeline_M4096.png`):
- **Clocks and link:** globaltimer resolves 32 ns, and the NVLink flag round trip is 3.7 µs. The start skew between GPUs is under 5 µs, so it isn't a factor.
- **Main cause: the producer is slowed by the protocol.** At M=1024 its span is 341 µs, against 232 µs for the same kernel with no protocol. Two parts:
  - **(a) Signalling contention.** Each producer tile does 64 `fetch_sub` operations, and all 64 tiles of a row panel hit the same 64 counters. The signal phase takes 28–44 µs per CTA when many signal at once, against 4–8 µs when a CTA signals alone. The fence costs only 7–8 µs of that.
  - **(b) Remote store burst.** At small M the waves run in lock-step, so each wave's 8.6 MB write drains while no CTA is computing. That costs about 50 µs at M=1024.
- **Second cause: the tail.** The last producer wave completes about 4 row panels at once, which leaves one full consumer wave (114 µs at M=1024) after the producer finishes.
- **Ruled out:** consumer slowdown from overlap (within ±5%), start skew, and the start-up delay. The consumer is starved, not backlogged.

## Phase 2c: row-panel scoreboard (done)
- One counter per consumer row panel instead of one per consumer tile, following the paper's own "row-to-row" simplification (Sec III.A).
- The last producer tile of a row pushes all of that row's consumer tiles. Each counter sits on its own 128-byte line.
- This cuts the atomics per producer tile from 64 to 1.

**Phase 2c result** (`results/rowpanel.md`, `results/timeline.md` Phase 2c section):
- **The Phase 2b prediction failed.** Per-CTA signal time barely moved (29.3 → 29.6 µs at M=1024, 18.2 → 15.0 µs at M=4096 g8), and the producer span did not shrink.
- **Correction to Phase 2b (a):** the signal time is not counter contention. Weakening fences and atomics (`fence=2`, unsafe `fence=3`) only moves time between the store phase and the signal phase; their sum stays the same. The real cost is draining each wave's ~8.6 MB remote-store burst.
- **End-to-end:** small gains, −17 µs at M=1024, −40 to −77 µs at 2k–4k, −43 to −108 µs at ≥ 8k. Row-panel stays the default.
- **Standing (µs):** CTAPP 476 / 786 / 1415 / 2433 / 4519 / 8668 for M = 1k → 32k. It beats TP2 by 8–15% from M=8192 up, and loses to micro-batching at every M.

## Phase 3: SM90 cooperative persistent kernel (Opus, ~1 day)
Priorities, from the 2b/2c diagnosis:
1. **Overlap the remote-store drain** with the next tile's mainloop. A persistent kernel keeps computing while the previous tile's stores fly; signal only after the stores are complete (TMA store + `tma_store_wait<0>` / `cp.async.bulk.wait_group`, or a fence by the storing threads, then signal).
2. **Close the kernel gap to cuBLAS.** KernelTma is 0.65x cuBLAS; the cooperative kernel with 2 consumer warpgroups and a TMA-store epilogue should be much closer. Measure 1-GPU speed first.
3. **Tile order that completes rows one at a time** to shrink the tail (~one consumer wave today).
4. Row-panel scoreboard (already done in 2c; port as-is).

Design:
- The producer warp (TMA-load warp) pops each tile ID from the source queue, polls for it, and hands it to the consumer warpgroups through a small shared-memory pipeline. This replaces the static scheduler.
- Consumer warpgroups signal after each tile's store completes.
- Same bit-exact tests, then rerun Phase 2's table.

## Phase 4: 4 GPUs, when GPUs 1 and 2 are free (~half day)
1. A 4-layer chain with one GEMM per GPU (Table I, 4-GPU point), against micro-batching and TP4.
2. CTA-pipelining combined with TP (Fig 7):
   - Pairs (0→1) and (2→3), weights sharded across the two pairs.
   - An NCCL AllReduce between the two consumer GPUs.
   - Compared against TP4 at M ∈ {4096, 8192, 16384}.
