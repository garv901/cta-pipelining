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
- **GPU pair for 2-GPU runs:** pick whichever GPUs are free at run time (any of 0–3; all pairs are NV6). Phases 0–2c used producer = GPU 3, consumer = GPU 0. Check `nvidia-smi` before every run, record it and the physical pair next to each result, and re-measure P2P bandwidth when the pair changes.
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

## Phase 3: SM90 cooperative persistent kernel (in progress)
Full plan: ~/.claude/plans/iterative-orbiting-graham.md. Performance first; the bit-exact suite (3.4) runs once the implementation is stable.

**Goals:**
- G1: coop kernel ≥ 0.90x cuBLAS on 1 GPU at M ≥ 4k.
- G2: producer span with protocol within ~5% of the standalone coop kernel writing to peer D (drain hidden).
- G3: beat TP2 at every M (≥ 25% at M ≥ 8k); beat cuBLAS MB at M ≤ 4k (≥ 10% at 1k–2k); parity at M ≥ 16k is expected.
- G4: tail ≤ ~1 consumer tile time.

**Design:** the CUTLASS kernel stays untouched; both hooks are template parameters.
- `CtappScheduler` (`csrc/ctapp_sched_sm90.hpp`): uses the cooperative kernel's dynamic-persistent path with a custom SM90 scheduler. Warp1 pops and polls the queue (instead of the CLC query) and writes the tile ID into the scheduler smem pipeline. `fetch_next_work` adds `fence.proxy.async.global`. `head` is pre-seeded to gridDim.x, so every role reads the first tile itself.
- `CtappEpilogue<Base>` (`csrc/ctapp_epi_sm90.hpp`): wraps the TMA warp-specialized epilogue. The storing warp (warp 8) runs non-`.read` `cp.async.bulk.wait_group 0` on all lanes, then `fence.proxy.async.global` + `fence.acq_rel.sys`, then the row-panel signal. `signal_defer=1` signals tile i at the start of tile i+1's `store()`, so the drain overlaps the mainloop.
- `csrc/gemm_coop.cu` is a second translation unit in the same extension. Python reuses `Pipeline` (`kernel="coop"`), `build_deps`, the order function and `measure()`.

**3.0 result** (`results/coop_gate.md`, 2026-09-30):
- **G1 passes.**
  - The stock coop kernel runs at 0.90–1.07x cuBLAS in the event-timed sweep (670–700 TFLOP/s) and 0.985x in the gated check at M=16k; cuBLAS is noisy (±10%). `KernelTma` was 0.62–0.68x.
  - Best config: 128x256 for M ≤ 8k, 256x128 from 16k. There are no spills.
- **The peer-D gate fails.** Measured on pair 3→2, with link bandwidth 124.1 GB/s measured on that pair, peer/local is:

  | M | 1k | 2k | 4k | 8k | 16k | 32k |
  |---|---|---|---|---|---|---|
  | peer/local | 1.71 | 1.52 | 1.42 | 1.22 | 1.24 | 1.26–1.28 |

  - The link bound is always below local time, so the link bandwidth is not the limit.
  - Diagnosis: the persistent CTAs run in lock-step, so all 132 epilogues burst ~8.6 MB at once. With the builder's StagesD = 2, the storing warp's `wait_group.read` holds both MMA warpgroups until the data has left over NVLink.
  - Fix under test (3.0b): a full-tile smem D buffer (custom `Sm90TmaWarpSpecialized<…, StagesD = EpiTiles, …>` policy), so the drain overlaps the next mainloop.

**Steps:**
- 3.0 stock coop speed plus TMA store to peer (Sonnet).
- 3.1 scheduler (Opus).
- 3.2 epilogue signal and the 2-GPU pipeline (Opus).
- 3.3 `fig5_coop` and `timeline_coop` (Sonnet).
- 3.4 correctness (later).

## Phase 4: 4 GPUs, when GPUs 1 and 2 are free (~half day)
1. A 4-layer chain with one GEMM per GPU (Table I, 4-GPU point), against micro-batching and TP4.
2. CTA-pipelining combined with TP (Fig 7):
   - Pairs (0→1) and (2→3), weights sharded across the two pairs.
   - An NCCL AllReduce between the two consumer GPUs.
   - Compared against TP4 at M ∈ {4096, 8192, 16384}.

## Step 0 (2026-10-01, node1): measure first — results and decision

Run on Slurm node1 (8x H100 80GB HBM3, NVSwitch, NV18 between every pair; hold jobs 11248 (4 GPUs) and 11250 (2 GPUs),
`srun --overlap`), torch 2.12.1+cu129, CUDA 12.9. Scripts: `bench/node1_link.py`, `bench/tp_strong.py` (+ `tp_strong_report.py`),
`bench/fig5.py --shape/--methods`. Full tables: `results/node1_link.md`, `results/tp_strong.md`, `results/fig5_node1.md`.

**S0.1 link.** SM-store P2P 369 GB/s per GPU egress to any peer and the *same* 368 GB/s aggregate when fanning out to 2 or 3
peers (egress-capped, not per-link); copy engine 392 GB/s; flag round trip 4.5 us; NVLS multicast supported on all GPUs.
Last-wave drain floor 23 us (was 69 at 124 GB/s). Stock coop cfg0 with D in peer memory: 1.20/1.14/1.02x local at M=1k/4k/16k,
bit-equal. Fixed-cost model, paper shape: CTAPP/ideal 1.32/1.16/1.08/1.04/1.02 at M=1k..16k. 70B FFN per-tile-reduce link
ratio 0.07 (TP2) / 0.20 (TP4) / 0.46 (TP8).

**S0.2 strong TP (back-to-back, best variant, exposed = share of the step above 1-GPU/world).**

| shape | TP | M=1k | 2k | 4k | 8k | 16k | best variant (typ.) |
|---|---|---|---|---|---|---|---|
| 70B FFN up->down | 2 | 6 % | 1 % | 4 % | 2 % | 4 % | asynctp |
| 70B FFN up->down | 4 | 16 % | 0 % | 2 % | 6 % | 10 % | asynctp (multimem at 1k) |
| 70B down->norm->QKV | 2 | 18 % | 12 % | 11 % | 8 % | 9 % | asynctp |
| 70B down->norm->QKV | 4 | 45 % | 42 % | 22 % | 18 % | 17 % | multimem at <=2k, asynctp above |
| paper 8192 square | 2 | 24 % | 10 % | 5 % | 5 % | 9 % | asynctp |
| paper 8192 square | 4 | 49 % | 46 % | 31 % | 17 % | 31 % | multimem / asynctp |

Plain NCCL all-reduce exposes 10-16 % (FFN TP2/TP4), 21-23 % (QKV TP2), 38-48 % (QKV TP4). `asynctp` = PyTorch
fused_matmul_reduce_scatter (sharded output); on the QKV chain it needs a separate NCCL all-gather before the QKV GEMM, which is
the un-overlapped part. (Outlier: l70b TP4 M=16k asynctp 21 ms, a queueing glitch also seen by the porting agent; asynctp_ag
is used there.)

**S0.3 recalibration.** Micro-batching (cuBLAS, best chunk) 1.82/1.43/1.26/1.10/0.99x ideal on the paper shape and 1.3-1.9x on
the 70B shapes; TP2 with plain NCCL beats it everywhere except paper M=16k. The Phase-1 KernelTma CTAPP is 1.4-2.3x ideal
(paper) / 2.0-2.9x (down->QKV) because the KernelTma kernels run at 0.68-0.74x cuBLAS and the pipeline is still 24 % slower
than micro-batching the same kernel. Confirms: all further work sits on the cooperative kernel.

**Decision (rule 1 of the Step-0 checkpoint, applied to the measured numbers).**
- The 70B FFN is closed: strong TP2 and TP4 are within 0-6 % of ideal for M >= 2k. Role B on the FFN (per-tile reduce) cannot
  pay for itself; drop it.
- The 2-GPU pipeline-parallel comparison (paper Fig 5, role A) is against micro-batching, which strong TP2 already beats by
  20-40 %; a CTAPP that reaches the model's 1.08x at M=4k would merely tie asynctp (1.12x on the QKV chain). Not worth the
  scheduler/epilogue work on its own; keep only as a by-product.
- **Target: Llama-70B down-proj -> RMSNorm -> QKV at TP4** (8192-wide crossing between a 28672-deep GEMM and a 10240-wide one).
  Strong baseline exposes 17-22 % at M = 4k-16k (asynctp) and 42-45 % at M <= 2k (multimem). Mechanism ("Phase 4"): the
  down-proj epilogue reduces each 128x256 partial tile across the 4 ranks with `multimem.red.add` into a multicast buffer
  (replicated output, P bytes egress per GPU, link ratio 0.27 at 369 GB/s), one `multimem.red` counter per 128-row panel;
  when a panel is complete on a GPU it is RMS-normed and pushed into that GPU's QKV-GEMM workqueue (existing row-panel
  scoreboard/queue protocol), so the all-gather disappears and the QKV GEMM starts ~1 us after the first panel lands instead
  of after the full reduce + norm + all-gather. Fixed costs: 128-row fill of the down-proj (2*128*28672*8192/4/700T = 21 us
  per GPU), drain 23 us, protocol ~15 us, against a 1k-token ideal of 242 us: model 1.24x at M=1k (multimem 1.82x), 1.06x at
  4k (asynctp 1.28x), 1.03x at 8k (asynctp 1.22x). TP2 bring-up rung uses the same code with peer stores instead of multimem.
- **[Superseded 2026-10-02 by the Phase 5 evidence, see "Conclusion and incorrect assumptions" at the end of this file: the claim that NVLS is the right baseline and multimem the matching mechanism at t >= 4 did not hold.]**
- Gate for Phase 4: beat asynctp at M = 4k and 8k by >= 10 % and multimem at M = 1k-2k by >= 20 %, bit-exact vs a reference
  reduce (bf16 multimem accumulation order permitting; else rel err <= the NVLS baseline's).

## Phase 4 (2026-10-02): TP4 down-proj -> RMSNorm -> QKV with a CTA-pipelined NVLS reduction — design

### What problem the kernel addresses (why the QKV chain has a gap and the FFN does not)

The gap is not "communication is slow" but "communication sits in a dependency chain nobody overlaps". With Megatron sharding
the down-proj is K-sharded, so every GPU holds a partial sum of the full output, and the next operations need the complete sum
on every GPU twice over: RMSNorm needs each row's full sum of squares, the N-sharded QKV GEMM needs the full normalised
activation as its A operand, and the residual stream must be complete everywhere for the next layer. Dependency:
partial-sum -> full-sum replicated -> norm -> GEMM.

- FFN (up-proj -> SwiGLU -> down-proj): SwiGLU is elementwise and row-local, so a reduce-scatter that leaves each GPU a quarter
  of the rows is enough and the following all-gather fuses into the next GEMM's loads. Async-TP does exactly that and measured
  0-6 % from ideal (Step 0). Nothing left to recover; path closed.
- QKV chain: both strong baselines pay a serialised stretch. Async-TP overlaps the reduce-scatter with the down-proj, then runs
  the norm on its row quarter, then an all-gather with nothing to hide behind, then the QKV GEMM: the all-gather and the norm are
  exposed. The in-switch multimem all-reduce has no overlap at all: a separate kernel between the GEMMs plus a separate residual
  add and norm. Measured at M=4096/TP4: the two GEMMs alone take 0.92 ms; async-TP 1.27 ms, multimem 1.65 ms. The exposed
  0.35-0.73 ms is the target (17-45 % of the step depending on M).

What the kernel does differently: the reduction becomes a per-tile stream that runs during the producing GEMM. As each 128x256
partial tile lands, a small reducer on reserved SMs pulls the fp32 sum through the switch (multimem.ld_reduce), adds the
residual, broadcasts the finished rows to every GPU (multimem.st), and accumulates the per-row sum of squares. When the down-proj
finishes, the replicated residual stream and the norm statistics already exist everywhere: the all-gather disappears because the
output was broadcast as it was produced, and the norm disappears as a pass because the QKV epilogue scales rows by the
statistic with gamma folded into the weights. The only exposed work is the last wave's reduction and the SMs lent to the reducer.
Gains so far: 1.06 vs 1.27 ms (async-TP) at M=4096; 0.36 vs 0.44 ms (multimem) at M=1024; the distance to the 0.92 ms floor is
mostly the 16 reducer SMs, which the current sweep tries to cut to 6-8.

Scope caveat: this pays only where the consumer of a reduction needs replicated rows, i.e. the attention-side boundary of every
transformer layer (once per layer). It is one boundary, not a general replacement for tensor-parallel collectives.

Production constraints: one process per GPU (torch.distributed, NCCL group for setup only), symmetric memory via
`torch.distributed._symmetric_memory` (multicast pointers), no host sync inside a forward, counters monotonic across forwards
(epoch-scaled targets, never reset), gamma folded into W_qkv, residual add fused into the reduction, every GPU ends with the
full reduced residual stream `x` (replicated, like the all-reduce it replaces).

Files: `csrc/ctapp_tp4.cuh` (protocol hooks), `scripts/gen_coop_ctapp.py` -> `csrc/sm90_gemm_coop_ctapp.hpp` (stock CUTLASS 4.8
cooperative kernel with 5 splice points), `csrc/gemm_tp4.cu` (two kernels + multimem probes, module `ctapp_tp4_ext`),
`ctapp/ext.py::load_tp4`, `ctapp/tp4.py` (per-rank boundary class + baselines), `tests/test_tp4.py`, `bench/tp4_bench.py`.

Kernel 1, down-proj (K-sharded, 128x256x64 cooperative, identical persistent grid and tile order on every rank). After a CTA's
TMA store of partial tile (m, n) lands (`cp.async.bulk.wait_group 0` + `fence.proxy.async` on the issuing warp, named barrier,
`fence.acq_rel.sys`), thread 0 does `multimem.red.release.add` on `tile_cnt[m*tn+n]` (multicast, so every rank's copy counts
all ranks). It then reduces ITS RANK'S 32-row slice of the tile it stored one step earlier: spin on local
`tile_cnt >= epoch*world`, `multimem.ld_reduce.add.acc::f32.v4.bf16x2` over the 4 copies of `partials`, + residual,
`multimem.st.v4.bf16x2` into every rank's `x`, per-row sum of squares via shuffle + `multimem.red.add.f32` into `rowss[parity]`,
then `multimem.red.release.add` on `panel_cnt[m]`. The one-tile deferral means the wait is normally already satisfied (the
other ranks stored the same tile in the same wave); deadlock-free by induction because a wait at step k only depends on
stores at step k-1 of a grid that is identical on every rank. The last tile is reduced after the work loop. No separate
reducer kernel: the cooperative kernel occupies all 132 SMs.

Kernel 2, QKV (N-sharded, same tile config, RMS epilogue). The TMA-load warp spins on local `panel_cnt[m] >= epoch*tn*world`
before loading the A tiles of row panel m (then `fence.proxy.async.global` so the TMA reads see the multimem stores); the
consumer warp groups acquire the same counter before the epilogue, whose EVT scales each row by `rsqrt(rowss[row]/8192 + 1e-5)`
(`Sm90ColBroadcast` of rowss + `Sm90Compute<RmsScaleFn>` on the accumulator). Replicated A means no all-gather.

Host per forward: `epoch += 1; rowss[epoch&1].zero_()` -> kernel 1 (mode 1) -> kernel 2 (mode 2), all on one stream; the
two kernels overlap only through the counters (kernel 2 is queued behind kernel 1 on the stream, so on H100 it starts when
kernel 1's CTAs retire — the overlap is the reduction + norm + all-gather, not the two GEMMs; see results for whether a second
stream / PDL is needed).

Correctness levers: fp32 accumulation inside the switch (`acc::f32`), rounding to bf16 once per element (same as NVLS
all-reduce); rowss accumulated in fp32 from the bf16-rounded x (matches an eager RMSNorm on the bf16 residual stream).

### Phase 4 bring-up log (2026-10-02, node1, TP4, M = 4096 unless noted; b2b ms, max over ranks)

Correctness: first build passed the fp32 reference on 4 GPUs (x rel err 4.5e-3 = the multimem all-reduce baseline's own
error, out 4.4e-3), 23 epochs back to back, 0 ordering violations in the multimem stress probe, mode-0 GEMM bit-exact vs cuBLAS.

Measured primitives (probes in `gemm_tp4.cu`): SM-issued multimem throughput is capped at **~90 GB/s per GPU** for every op
type (ld_reduce 88, multicast st 93, red.add bf16x2 84, with all 4 ranks active; a local copy runs at 1.4 TB/s). Latency:
dependent ld_reduce chain 1.4 us, multicast st + fence.acq_rel.sys 3.0 us, local ld.acquire.sys 0.15 us. Consequence: the
reduction of P/4 = 16 MB per GPU costs 32 MB of multimem ops = 0.36 ms, i.e. 58 % of the 0.63 ms down-proj it must hide under.

| variant | down-proj b2b | note |
|---|---|---|
| mode 0 (plain cooperative GEMM) | 0.63 | floor; cuBLAS 0.90 |
| consumers signal only (no reduce) | 0.69 | store drain + fence + multimem.red per tile: +0.06 |
| mode 3: reduce in the consumer warp groups, deferred one tile | 1.27 | MMA idles during the multimem round trips |
| mode 1: reduce in the 2 idle producer warps (40-register budget) | 0.95 | **not hidden**: +0.32 = exactly the multimem time |
| mode 1 ablations: no ld_reduce / no st / neither | 0.81 / 0.89 / 0.68 | each op type costs its own transfer time |
| mode 1 with 4 instead of 2 x 16 B in flight per thread | 0.95 | not latency-bound |

Two bring-up bugs worth remembering: (1) `NamedBarrier::sync(n, uint32_t id)` adds the 8 reserved ids, so passing
`FirstUserBarrier` (= 8) wrapped to hardware barrier 0/1 and collided with `__syncthreads`/the epilogue barrier -> sporadic
"illegal instruction"; user ids must be 0..7. (2) Changing the setmaxnreg split to 64/224 (sum exactly 65536) hangs the kernel
even in mode 0; stock 40/232 kept, reducer path verified at <= 40 registers via a stand-alone probe kernel.

Open question being measured (mode 4): a separate reducer kernel on R reserved SMs (GEMM launched with sm_count = 132 - R) —
does multimem traffic from *other* SMs slow the GEMM, or only traffic issued from the GEMM's own SMs?

Mode 4 answer (stand-alone reducer kernel on R reserved SMs, GEMM on 132 - R; b2b ms):

| M | floor (2 GEMMs) | down on 116 SMs | down + reducer R=16 | chain (ours) | best baseline |
|---|---|---|---|---|---|
| 1024 | 0.237 | 0.219 | 0.271 | 0.355 | multimem 0.439 (-19 %) |
| 4096 | 0.921 | 0.723 | 0.797 | 1.058 | asynctp 1.265 (-16 %) |
| 8192 | 1.912 | 1.500 | 1.631 | 2.303 | asynctp 2.483 (-7 %) |

So the reducer *does* overlap when it is not inside the GEMM's SMs; the remaining cost is (a) the 16 SMs taken from the GEMM
(12 %), (b) ~10 % GEMM slowdown from the traffic landing in the GEMM GPU's memory system (probe with no-wait reducer), (c) the
reducer's per-tile serialisation (R=8 -> 1.14, R=12 -> 0.91, R=16 -> 0.80 at M=4k: each block does spin, 2 x 16 B per thread,
3 us sys fence, signal per tile, so it needs 16 blocks to reach the 90 GB/s cap). Dead ends measured: programmatic dependent
launch of the QKV GEMM (no gain: the persistent GEMM has no tail to fill), a unicast reducer reading the 4 copies directly
(slower, and its traffic slows the GEMM *more* than multimem's: interference scales with bytes moved, so fewest-bytes multimem
stays), consumer-side and producer-warp reducers (above). Raster 2 (panel order) costs the down GEMM 8-13 % so the down GEMM and
the reducer stay at raster 1; the QKV GEMM's order is independent and raster 2 is ~15 % faster for it.

Reducer v2 (2 tiles per iteration, signal deferred one iteration; QKV at raster 2) and the R sweep (b2b ms, M=4096):
down on 132-R SMs 0.63/0.70/0.71/0.71 for R=4/6/8/12; down + reducer v2 1.49/1.12/0.95/0.80. The reducer is throughput-bound
at ~4-5 GB/s per block (32 KB per ~6 us iteration), so it still needs R=12. Chains: v2 R=12 1.087 (4k), 2.265 (8k), 4.918 (16k)
vs v1 R=16 1.108 / 2.343 / 5.147; at 1k-2k v1 R=16 is marginally better (0.350 / 0.606). Why the deferral did not help: the
`fence.acq_rel.sys` issued after the block barrier is cumulative over *all* threads' stores of the current iteration, so the
block still stalls ~3 us per iteration. Next: v3 = dedicated signaller warp with named-barrier backpressure (workers never wait
on the fence) and 2-4 tiles per iteration, aiming at R=4-6.

Reducer v3/v4 (`tp4_reduce3_kernel<T>`, 544 threads = 16 worker warps + 1 signaller warp; workers hand finished tiles to the
signaller through named barriers 1/2/3 and never wait on the sys fence; T = 2 (v3) or 4 (v4) tiles per iteration). The fence
itself costs 0.18 ms at R=8 (measured by skipping the signal, dbg=16), and decoupling it helps the reducer stage (down + reducer
at R=8: v2 0.95 -> v3 0.79 ms) but the chain only moves from 1.108 (v1 R=16) / 1.087 (v2 R=12) to 1.079 (v3 R=8) at M=4k:
the saved 8 SMs are worth ~0.05 ms and the residual is the traffic interference, which no reducer organisation removes.
Correctness unchanged (x 4.5e-3 = multimem error, out 4.4e-3, 30-forward stress and 0 ordering violations).

Final chain numbers (b2b ms, best R per M, QKV raster 2, down raster 1 swizzle 1):

| M | floor (2 GEMMs) | chain | config | best baseline | gain | gate |
|---|---|---|---|---|---|---|
| 1024 | 0.236 | 0.344 | v3 R=12 | multimem 0.439 | 22 % | >= 20 % vs multimem: met |
| 2048 | 0.477 | 0.595 | v3 R=12 | multimem 0.845 (asynctp 0.867) | 30 % | met |
| 4096 | 0.943 | 1.080 | v4 R=8 (v3 1.079) | asynctp 1.265 | 15 % | >= 10 % vs asynctp: met |
| 8192 | 1.973 | 2.248 | v3 R=8 | asynctp 2.483 | 9 % | marginal |
| 16384 | 4.289 | 4.793 | v4 R=8 | asynctp 4.929 | 3 % | not a target (link ratio favours async-TP) |

Decision: stop kernel iteration here. Production configuration = mode 4 GEMM (raster 1, swizzle 1, sm_count = SMs - R), reducer
v3 with R = 12 for M <= 2048 and 8 above, QKV mode 2 at raster 2, both GEMMs on SMs - R. Remaining work is productisation:
device-side epoch so the forward is CUDA-graph capturable (removes ~0.05-0.1 ms of host launch overhead per step, which is the
largest remaining item at M <= 2k), cached workspace, `CtappBoundary` on this path, tests and the final benchmark table in
`results/tp4_chain.md`.

### Phase 4 result (2026-10-02)

Productionised path (`CtappBoundary` in `ctapp/tp4.py`): the epoch lives on the device (bumped by `add_(1)` at the start of each forward), so the forward is CUDA-graph capturable (two graphs, one per epoch parity, because the `rowss` pointer is baked into the QKV epilogue); `set_weights` runs a warm-up launch of both GEMMs because CUDA lazy module loading deadlocks (spin-limit trap) against an already-resident spinning reducer. TP4, l70b_qkv, b2b ms, best ctapp vs best baseline (FlashInfer `trtllm_allreduce_fusion` added as a baseline, one self-consistent rerun), gain = 1 - ctapp / best baseline: M=1024 0.344 vs 0.344 (flashinfer), 0 % (24 % vs multimem); 2048 0.590 vs 0.658 (flashinfer), 10 % (11 % vs async-TP 0.662, 33 % vs multimem); 4096 1.053 vs 1.285 (asynctp), 18 %; 8192 2.236 vs 2.531 (asynctp), 12 %; 16384 4.754 vs 5.022 (asynctp), 5 %. The gate (>= 10 % over async-TP at 4k/8k, >= 20 % over multimem at 1k-2k) is met at 1k (24 % vs multimem), 2k, 4k (18 %) and 8k (12 %), but FlashInfer's fused all-reduce+residual+RMSNorm, the production path, ties ctapp at 1k and trails it by only 10 % at 2k. Full tables in `results/tp4_chain.md`.

Known limitation (found 2026-10-02 while generalising the boundary): the standalone reducer kernels (`tp4_reduce*_kernel` in
`csrc/ctapp_tp4.cuh`) hard-code 16 columns per thread (two bf16x8 vectors), which covers a 128x256 tile only when rows-per-rank
= 128 / world = 32, i.e. world = 4. At world 2 half the columns of x are never reduced (x rel err ~1.0 in `tests/test_tp4.py
--world 2`); at world 8 threads would overlap. Fix when needed: loop over `cols / 16` vector pairs per thread (or derive the
thread->(row, col) map from `cols`). TP4 is the production target, so this is deferred.

## Phase 5 (2026-10-02): vLLM port and 4K prefill ablation

Built `vllm_plugin/` (package `ctapp_vllm`, enable with `CTAPP_VLLM=1`): a vLLM 0.30.0 general plugin that registers `CtappLlamaForCausalLM`, which replaces the two TP all-reduce boundaries per layer (A: o_proj -> add -> post_attention_layernorm -> gate_up; B: down_proj -> add -> next input_layernorm -> next qkv) with `CtappBoundary` from `ctapp/tp4.py`. Knobs: `CTAPP_BOUNDARY=both|down|oproj`, `CTAPP_R`, `CTAPP_A_RASTER/SWIZZLE`, `CTAPP_B_RASTER/SWIZZLE`, `CTAPP_CHECK`, `CTAPP_LOG_M`. Gammas are folded in place into gate_up/qkv (stock-path norms use a ones-weight RMSNorm); one pool instance per M; stock fallback for M not a multiple of 128 or M < 512 (decode). Correctness on a 4-layer 70B config: boundary rel err 2.6e-3 (A) and 3.2-3.8e-3 (B), identical top-1 token and top-20 logprobs vs stock.
Benchmarks: `bench/vllm_prefill.py`, `bench/vllm_ctapp_check.py`, `bench/vllm_prof.py`; full tables in `results/vllm_prefill_4k.md`.

Numbers (80 layers, TP4, b=1, M=4096, median ms of `llm.generate`): stock eager 287.5 (289.9 same-session), compiled 287.9 (FlashInfer fusion, 2 MB threshold) / 299.9 (256 MB threshold), compiled async-TP 273.1, ctapp down-only R=8 280.7, ctapp both R=8 331.4 (old A config) -> 290.6-292.6 after the A swizzle fix (R=12: 288.1-289.5), oproj-only 297.9. Down-only saves ~0.06-0.12 ms/layer at b=1 (0.4-0.5 at b=4), 3 % over eager, and loses to compiled async-TP by 3 %.

Findings:
1. Baseline: the micro-benchmark gain (1.053 vs 1.285 ms asynctp) does not translate, because the stock eager chain it replaces is cheap (boundary B 1.32 ms vs ours 1.12 ms) and the compiled async-TP also fuses everything else.
2. GEMM efficiency: the reduction is hidden at both boundaries (reducer 0.72-0.84 ms under a 0.78 ms producer / 1.67 ms consumer), but our consumer GEMMs run 1.3-1.6x cuBLAS in situ (120-124 SMs, 128x256 tile, panel waits, reducer traffic), producer 1.25x. The gate_up consumer was 2x cuBLAS with raster 2 / swizzle 1 and is 1.06-1.13x with raster 1 / swizzle 2-8 on 1 GPU; in situ 1.37x.

Next levers: (a) unicast reduce-scatter/all-gather reducer with a light flag protocol (multimem path is SM-issued, ~90 GB/s, ~0.8 ms for 64 MB vs NCCL ring 0.35 ms at 369 GB/s) so the reducer is shorter than the GEMM and R can shrink; (b) run the consumer on 132 SMs after the reducer retires and a better tile for N=14336; (c) CUDA-graph / compiled integration to remove eager overhead. Decode is out of scope (fewer than 2 row panels).

Known issue: a one-off illegal-memory-access crash at pool-instance creation mid-run; worked around with a device sync + TP barrier before creation; root cause unproven.

### Conclusion and incorrect assumptions

Verdict: negative-to-marginal for production at TP4 (3 % over stock eager, 3 % behind compiled async-TP at 4K prefill). Three assumptions made in Step 0 / Phase 4 did not hold.

1. **Baseline.** Assumption: the Step-0 "strong baselines" (torch symmetric-memory variants, torch.distributed NCCL; 64 MB all-reduce ~0.7 ms) represent what an engine pays. Evidence: vLLM's PyNccl path does the same all-reduce in 0.35-0.38 ms (`ncclDevKernel_AllReduce_Sum_bf16_RING_LL`, ~275 GB/s effective), and stock boundary B is 1.32 ms vs ours 1.12 ms. Consequence: the 18 % stand-alone gain (1.053 vs 1.285 ms) shrinks to 3 % vs eager and goes negative vs compiled async-TP.
2. **Mechanism.** Assumption: NVLS multimem is the matching mechanism because it moves P bytes once vs 1.5 P for a ring. Evidence: SM-issued multimem reduction runs at ~90 GB/s per GPU vs 369 GB/s unicast; the reducer takes 0.72-0.84 ms for 64 MB (NCCL ring 0.35 ms), outlasts the 0.78 ms producer GEMM and occupies 8-12 SMs. Consequence: the reduction is hidden only because the GEMM is slow, R cannot shrink, and the SMs taken by the reducer slow the GEMMs.
3. **GEMM efficiency.** Assumption: our cooperative GEMMs are close to cuBLAS when run as producer/consumer. Evidence: in situ 1.25x (producer) and 1.3-1.6x (consumers) cuBLAS time (120-124 SMs, one 128x256 tile, panel waits, reducer traffic; gate_up consumer 1.37x after the swizzle fix). Consequence: the hidden communication is spent on slower compute; boundary A is break-even at best (~2.01 ms vs 1.83 ms stock in the profile; 288-293 vs 289.9 ms end to end).

**Open discrepancy (not explained).** Our bench's `nccl` variant (torch.distributed all_reduce, bf16, 64 MB) measures ~0.75 ms in BOTH venvs (torch 2.12 NCCL, and torch 2.13 with NCCL 2.29.7), while vLLM's PyNccl path shows 0.35-0.38 ms in the profiler trace (RING_LL kernel, 24 blocks). The cause (algorithm/protocol selection, communicator setup, or measurement method) is NOT established. Measure it before making any further baseline claims.

**Step-0 decision superseded.** "NVLS is the right baseline and multimem the matching mechanism at t >= 4" (Step 0 section above) is superseded by the Phase 5 evidence (findings 1 and 2). It is kept in place for the record.

**What would change the outcome.** (i) Go/no-go: a unicast reduce-scatter/all-gather reducer with NCCL-class bandwidth and a light flag protocol; if it cannot reach ~0.35 ms for 64 MB there is no case. (ii) Then consumer-GEMM efficiency (132 SMs after the reducer retires, better tile for N=14336).
