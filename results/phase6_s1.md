# Phase 6 S1: go/no-go measurements (node1, 4x H100 SXM, TP4)

Run 2026-10-02 on Slurm job 11383 (node1, GPUs 0-3). Raw data: `build/p6_s1.json` (probes 1-5), `build/p6_nccl.json` (probe 6, vLLM venv),
`build/p6_nccl_mstar.json` (probe 6, bench venv); logs in `build/logs/p6_s1.log`, `p6_s1_p2.log`, `p6_nccl.log`, `p6_nccl_mstar.log`,
`p6_nccl_debug.log`, `p6_test_tp4.log`. Code: `p6_*` kernels/bindings in `csrc/gemm_tp4.cu`, `bench/push_bw.py`, `bench/nccl_ar.py`.
Timing: 5 warm-up + 20 timed calls, barrier + device sync before each call, median per rank, max over ranks ("per-call"); "b2b" =
`tp4.time_step` back-to-back mean. GB/s use 1 GB = 1e9 B; the reducer's counted bytes at M=4096 are 128 MiB = 134,217,728 B, so the
0.71 ms bar corresponds to 189 GB/s.

## Verdict

**(a) GO with R=4.** On the 4 SMs the 128-SM producer leaves free, 8 reducer CTAs (544 threads, 2 per SM) move the 128 MiB/rank of
reducer traffic in 0.442 ms alone (304 GB/s). Under the concurrent mode-0 down GEMM they take 0.582 ms (231 GB/s, 100 % of the probe
overlapped). The GEMM slowdown was between -0.7 % and +2.9 % over five valid runs, which is within run-to-run noise (multimem: ~10 %).
4 CTAs at 1 per SM also clear the bar, but only just: 0.565 ms alone (238 GB/s) and 0.698 ms under the GEMM (192 GB/s).
**Caveat for D2:** one 384-thread CTA per free SM (the role-CTA shape in D2) takes 0.735-0.748 ms under the GEMM (179-183 GB/s) with
only 78-79 % of the probe overlapped, so with full overlap it would be roughly 170 GB/s. D2's role preamble needs more bytes in flight
per thread (it has 168 registers, the probe 80), or R=8.
**(b) D1 primary.** With 100 % of D stored to one peer, the remote epilogue costs 1.117x per call / 1.110x b2b at M=4096 (sms=128,
raster 1, swizzle 1), which is within 1.15. At M=1024 it is 1.18-1.22x: a risk at small M, to be re-measured with the 4-way epilogue in S2.
**(c) D2 via PDL.** `griddepcontrol.launch_dependents` works both eagerly and in a CUDA graph. Dependent CTAs start 2.5-3.7 µs (eager)
or 0.4-0.5 µs (graph) after the primary starts, exactly on the 4 SMs it leaves free. Without the attribute they start only after the
primary ends.
**(d) Last-arriver protocol allowed.** The two-hop chain (A remote store, `fence.acq_rel.sys`, gpu-scope atomic, B
`fence.acq_rel.sys` + flag, remote checker) showed 0 violations across 4 ranks: 16, 8 and 1 slots with 5000 iterations of 16 KB,
and 16 slots with 2500 iterations of 256 KB. The one-hop no-fence control produced 1760 stale words, so the checker does detect
missing fences. The two-hop no-fence control produced none, so the probe cannot show that A's fence is what makes the chain correct;
keep it, since the PTX model requires it.
**Probe 6:** the "0.75 vs 0.35 ms" NCCL discrepancy is explained, and it is not NCCL. The 64 MiB all-reduce takes 0.33 ms b2b /
0.39-0.42 ms per call in both venvs, through both torch.distributed and PyNccl, alone or right after the down GEMM. NCCL chooses Ring
with the Simple protocol on 24 channels. The bench's `nccl` row (1.767 ms) is down mm 0.622 + AR 0.331 + residual add 0.070 + eager
fp32 RMSNorm 0.538 + QKV mm 0.220. The ~0.75 ms figure includes 0.61 ms of unfused add + RMSNorm; only `NCCL_PROTO=LL` makes the AR
itself ~0.74 ms.

## Probe 1: reducer-shaped push throughput (M=4096, 128 MiB counted per rank)

Per unit (owned 128-row panel, 256-column n-tile, 32-row quarter), each worker does: 4 local inbox loads + residual load, fp32 add,
`st.relaxed.sys.v4` to 3 peers and to local x, and per row a `red.add.f32` of the sum of squares. A signaller warp issues
`fence.acq_rel.sys` + a value flag per T units. "spread" = 1 CTA per SM. "packed" = 2 CTAs per SM on blocks/2 SMs (a 200 KB-smem
blocker kernel occupies the rest). Correctness check (8 CTAs): 0 mismatches in x on all ranks; rowss relative error 2.9e-7 (544 and
384 threads).

Variant `full`, per-call ms (GB/s):

| placement | threads | T | 2 CTAs | 4 CTAs | 8 CTAs | 16 CTAs |
|---|---|---|---|---|---|---|
| spread | 544 | 2 | 1.096 (122) | 0.563 (238) | 0.312 (430) | 0.177 (760) |
| spread | 544 | 4 | 1.224 (110) | 0.628 (214) | 0.342 (393) | 0.188 (713) |
| spread | 384 | 2 | 1.196 (112) | 0.613 (219) | 0.330 (407) | 0.186 (721) |
| spread | 384 | 4 | 1.167 (115) | 0.599 (224) | 0.326 (412) | 0.181 (742) |
| packed (blocks/2 SMs) | 544 | 2 | 1.715 (78) | 0.867 (155) | **0.442 (304)** | 0.231 (580) |
| packed | 544 | 4 | 1.848 (73) | 0.933 (144) | 0.476 (282) | 0.249 (538) |
| packed | 384 | 2 | 1.755 (77) | 0.888 (151) | 0.454 (296) | 0.238 (563) |
| packed | 384 | 4 | 1.696 (79) | 0.857 (157) | 0.439 (306) | 0.233 (577) |

Ablations, 544 threads, T=2, per-call ms:

| config | full | no_fence | no_remote (local x only) | no_remote_no_fence |
|---|---|---|---|---|
| spread 4 CTAs | 0.563 | 0.556 (-1.2 %) | 0.490 (-13 %) | 0.480 |
| spread 8 CTAs | 0.312 | 0.310 | 0.258 (-17 %) | 0.253 |
| packed 8 CTAs / 4 SMs | 0.442 | 0.436 (-1.4 %) | 0.319 (-28 %) | 0.310 |
| spread 16 CTAs | 0.177 | 0.175 | 0.146 (-18 %) | 0.141 |

M=1024 (32 MiB counted), `full`, T=2: spread 4 / 8 CTAs: 0.149 / 0.089 ms (544 threads), 0.154 / 0.088 ms (384 threads). Packed 4 CTAs
on 2 SMs / 8 CTAs on 4 SMs: 0.229 / 0.124 ms (544 threads), 0.228 / 0.123 ms (384 threads).

Reading:
- The kernel is limited by memory-level parallelism per SM, not by NVLink. At 4 spread CTAs the 96 MiB of NVLink egress run at about
  60 GB/s per link (the link capacity is 369 GB/s), and dropping the remote stores saves only 13-28 %.
- A second CTA on the same SM gives +27 % at 4 SMs (0.563 to 0.442 ms).
- The per-batch system fence costs 1-2 %. T=4 gives nothing over T=2. 544 threads are slightly better than 384.
- The full 136-row table is in the appendix.

## Probe 2: interference with the producer GEMM

Setup: `tp4_down` mode 0, M=4096, K=7168, N=8192, r1 s1, sms=128 on the current stream; the probe on a second stream, on every rank.
Both streams start with a delay kernel (300 µs and 320 µs), so the GEMM's 128 CTAs are resident first and the probe lands on the 4 free
SMs. Per-call distinct-SM counts confirm this: 0 of 80 calls (4 ranks x 20) used more than 4 SMs.

GEMM alone: sms=128 0.637 ms per call (0.625 b2b); sms=132 0.650 ms (0.634 b2b). 128 SMs is faster: 1024 tiles make exactly 8 waves.

| threads | T | CTAs | GEMM alone (128) | GEMM with probe | slowdown | probe alone, CTAs on own SMs | probe alone, 4 free SMs (blocker) | probe span under GEMM | GB/s under GEMM | probe overlapped by GEMM |
|---|---|---|---|---|---|---|---|---|---|---|
| 544 | 2 | 4 | 0.637 | 0.641 | +0.6 % | 0.565 | 0.558 | 0.698 | 192 | 84 % |
| 544 | 2 | 8 | 0.637 | 0.641 | +0.6 % | 0.314 | 0.445 | **0.582** | **231** | 100 % |
| 384 | 2 | 4 | 0.637 | 0.641 | +0.6 % | 0.605 | 0.603 | 0.748 | 179 | 78 % |
| 384 | 4 | 4 | 0.637 | 0.641 | +0.6 % | 0.598 | 0.593 | 0.735 | 183 | 79 % |

Run-to-run spread of the slowdown over the valid runs: full sweep -0.7 % / -0.7 % (544 threads, 4 / 8 CTAs); quick runs +2.9 / +2.7,
+1.4 / +1.7 and +1.5 / +1.8 %. The probe starts 50-65 µs after the GEMM (host enqueue lag on the second stream). As a result, about
10 % of the GEMM's runtime is not overlapped, which biases the slowdown low by at most about a tenth of its value. Under the GEMM the
probe slows down by ~25-30 % (HBM/L2 contention).

## Probe 3: remote-epilogue producer (D stored to the right neighbour's symmetric buffer)

All 4 ranks run concurrently. Every rank stores 100 % of D to one peer, `tp4_down(..., d_ptr=peer)`. Results are bit-exact against the
local GEMM computed on the neighbour.

| M | sms | swizzle | local ms | peer ms | peer/local | local b2b | peer b2b | b2b ratio |
|---|---|---|---|---|---|---|---|---|
| 1024 | 128 | 1 | 0.171 | 0.205 | 1.193 | 0.157 | 0.191 | 1.219 |
| 1024 | 128 | 2 | 0.174 | 0.208 | 1.196 | 0.159 | 0.191 | 1.202 |
| 1024 | 132 | 1 | 0.175 | 0.209 | 1.198 | 0.161 | 0.190 | 1.185 |
| 1024 | 132 | 2 | 0.175 | 0.207 | 1.183 | 0.162 | 0.191 | 1.180 |
| 4096 | 128 | 1 | 0.632 | 0.706 | **1.117** | 0.625 | 0.694 | **1.110** |
| 4096 | 128 | 2 | 0.668 | 0.747 | 1.118 | 0.645 | 0.722 | 1.121 |
| 4096 | 132 | 1 | 0.728 | 0.760 | 1.045 | 0.670 | 0.738 | 1.101 |
| 4096 | 132 | 2 | 0.731 | 0.766 | 1.048 | 0.679 | 0.736 | 1.083 |

This matches S0's single-peer figures (1.196 / 1.136). D1 spreads 75 % of D over 3 links, so the per-link burst is lower than here; S2
measures that version.

## Probe 4: flag chain ordering and latency

Each "slot" is an independent A -> (B) -> checker chain per rank pair. A waits for the checker's ack of iteration i-1 before writing
iteration i, so the cycle is one full round trip including the payload write and the check. Violations are stale words seen by the
checker, summed over 4 ranks.

| mode | slots | payload | iterations | violations | cycle µs |
|---|---|---|---|---|---|
| two-hop chain (A fence.sys + gpu atomic -> B fence.sys + flag) | 16 | 16 KB | 5000 | **0** | 8.92 |
| two-hop chain | 8 | 16 KB | 5000 | **0** | 8.57 |
| two-hop chain | 1 | 16 KB | 5000 | **0** | 7.38 |
| two-hop chain | 16 | 256 KB | 2500 | **0** | 23.88 |
| one-hop (A fence.sys + flag) | 16 | 16 KB | 5000 | 0 | 6.41 |
| one-hop | 1 | 16 KB | 5000 | 0 | 5.37 |
| one-hop | 16 | 256 KB | 2500 | 0 | 20.77 |
| control: two-hop, no fence in A | 16 | 16 KB | 5000 | 0 (insensitive) | 6.90 |
| control: two-hop, no fence in A | 16 | 256 KB | 2500 | 0 (insensitive) | 20.86 |
| control: one-hop, no fence | 16 | 16 KB | 5000 | **1760** | 4.54 |
| control: one-hop, no fence | 16 | 256 KB | 2500 | 0 | 17.17 |

One-hop round trip with a 16 KB payload: 5.4 µs, so about 2.7 µs one way. The extra gpu-scope hop (atomic, then B's sys fence + flag)
adds about 2 µs.

## Probe 5: PDL early launch

Rank 0 only. Primary: 128 CTAs x 384 threads x 214,016 B smem, `griddepcontrol.launch_dependents` at start, spins 500 µs. Dependent:
132 CTAs launched with programmatic stream serialization, stamping `%globaltimer` and `%smid`. 5 repetitions per row.

| dependent spin | PDL attribute | mode | early CTAs per rep | early SMs | = free SMs | first dependent - primary start (µs) | first late dependent - primary end (µs) |
|---|---|---|---|---|---|---|---|
| 0 | 1 | eager | 132 x5 | 117, 119, 121, 123 | yes | 2.5-3.7 | - |
| 0 | 1 | graph | 132 x5 | 117, 119, 121, 123 | yes | 0.4-0.5 | - |
| 600 µs | 1 | eager | 4 x5 | 117, 119, 121, 123 | yes | 2.7-3.2 | 0.1-0.2 |
| 600 µs | 1 | graph | 4 x5 | 117, 119, 121, 123 | yes | 0.4 | 0.1-0.2 |
| 0 / 600 µs | 0 | eager | 0 | - | - | 501.7-502.0 | 1.7-2.0 |
| 0 / 600 µs | 0 | graph | 0 | - | - | 500.9 | 0.8-0.9 |

With a short dependent, all 132 dependent CTAs cycle through the 4 free SMs while the primary still runs. The probe used raw
`griddepcontrol` PTX. The CUTLASS producer still needs `-DCUTLASS_ENABLE_GDC_FOR_SM90` in `ctapp/ext.py` (S0 fact) before D2 can use it.

## Probe 6: the NCCL discrepancy (64 MiB bf16 all-reduce, 4 ranks)

vLLM venv (torch 2.13.0+cu130, NCCL 2.29.7, `CUDA_MODULE_LOADING=LAZY`). Per call / b2b ms, max over ranks. torch.distributed and
PyNccl results are identical (max |diff| 0).

| setting | torch.distributed | PyNccl | down mm | mm + AR | AR in chain = (mm + AR) - mm |
|---|---|---|---|---|---|
| default | 0.396 / 0.327 | 0.422 / 0.330 | 0.650 / 0.608 | 1.033 / 0.951 | 0.383 / 0.343 |
| NCCL_PROTO=LL | 0.840 / 0.740 | 0.845 / 0.742 | 0.647 / 0.609 | 1.462 / 1.369 | 0.814 / 0.761 |
| NCCL_PROTO=LL128 | 0.426 / 0.354 | 0.433 / 0.358 | 0.673 / 0.609 | 1.095 / 0.975 | 0.422 / 0.366 |
| NCCL_PROTO=Simple | 0.414 / 0.325 | 0.428 / 0.330 | 0.665 / 0.611 | 1.066 / 0.968 | 0.401 / 0.357 |
| NCCL_ALGO=Ring | 0.404 / 0.325 | 0.439 / 0.332 | 0.649 / 0.607 | 1.040 / 0.990 | 0.391 / 0.383 |
| NCCL_ALGO=Tree | 0.598 / 0.553 | 0.597 / 0.550 | 0.662 / 0.609 | 1.241 / 1.170 | 0.579 / 0.561 |
| NCCL_ALGO=NVLS | 0.386 / 0.326 | 0.417 / 0.327 | 0.645 / 0.609 | 1.026 / 0.945 | 0.382 / 0.336 |

Bench venv (torch 2.12.1+cu129, NCCL 2.29.7; torch.distributed only): default 0.405 / 0.328; LL 0.830 / 0.741; Simple 0.420 / 0.325;
NVLS 0.409 / 0.327. The in-chain AR for default / LL / Simple / NVLS is 0.357 / 0.339, 0.820 / 0.771, 0.352 / 0.338 and 0.402 / 0.329.

Decomposition of the `nccl` row of `bench/tp4_bench.py` (`tp4.NcclBoundary.forward`, M=4096, bench venv, default env):

| piece | per call | b2b |
|---|---|---|
| whole row | 1.841 | 1.767 |
| down mm (torch.mm) | 0.661 | 0.622 |
| dist.all_reduce | 0.394 | 0.331 |
| residual add | 0.089 | 0.070 |
| `rmsnorm_gamma` (eager, fp32 temporaries) | 0.561 | 0.538 |
| QKV mm | 0.278 | 0.220 |
| post = add + RMSNorm + QKV | 0.829 | 0.799 |

**Explanation.**
- **What NCCL actually runs.** `NCCL_DEBUG=INFO` reports `AllReduce: 67108864 Bytes -> Algo RING proto SIMPLE channel{Lo..Hi}={0..23}`
  by default, and `Algo NVLS proto SIMPLE` (16 channels) under `NCCL_ALGO=NVLS`, which is no faster.
- **The kernel name is misleading.** It is `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` for every protocol and for Ring and NVLS
  (`TREE_LL` for Tree). It is NCCL 2.29's entry kernel, not the protocol. The earlier reading "vLLM uses Ring LL" was wrong.
- **The AR runs at ring speed everywhere.** It is the same in both venvs, through both APIs, and in or out of the chain: 0.33 ms b2b,
  i.e. 205 GB/s algorithm bandwidth and 308 GB/s bus bandwidth.
- **Where the ~0.75 ms comes from.** It is reproduced only by forcing `NCCL_PROTO=LL`, or by counting the `nccl` row's unfused eager
  add + RMSNorm (0.61 ms) as part of the reduction. The decomposition shows the second is what the bench measured.
- **Consequence.** The torch-level `nccl` row overstates the stock boundary by ~0.45 ms at 4k (vLLM's `fused_add_rms_norm` is
  0.09 ms), and NCCL ring at 0.33 ms b2b is the honest reduction bar.
- **Profiler durations.** The per-kernel durations of the torch.distributed calls (in `build/p6_nccl*.json`) include cross-rank skew,
  because no barrier precedes the profiled calls, so they are unreliable. Use the timed columns.

## SHARED CODE TOUCHED

- `csrc/gemm_tp4.cu`:
  - `GemmTp4<...>::run` gained a trailing `void* y_override = nullptr`. When non-null, D goes to that raw address and Y only gives the
    shape. It affects all instances; with the default the behaviour is unchanged.
  - `tp4_down` gained a trailing `int64_t d_ptr`, bound as `py::arg("d_ptr") = 0`; 0 keeps the old behaviour.
  - New probe section: `p6_reduce_probe`, `p6_blocker`, `p6_wait`, `p6_delay`, `p6_stamp`, `p6_chain_stress`, `p6_pdl_probe`, with their
    kernels and pybind entries.
  - Build: no spills in any p6 kernel (reduce 56 / 80 regs, chain 37). The GEMM instances keep their pre-existing 168 regs / 40 B spill
    stores / 52 B spill loads.
- New: `bench/push_bw.py`, `bench/nccl_ar.py`, `results/phase6_s1.md`.
- Untouched: `csrc/ctapp_tp4.cuh`, `scripts/gen_coop_ctapp.py`, `csrc/sm90_gemm_coop_ctapp.hpp`, `ctapp/*.py`, tests.
- The bench-venv extension (`build/ext_tp4_2.12.1_cu129`) is rebuilt. The vLLM-venv build (`build/ext_tp4_2.13.0_cu130`) is stale,
  because `nccl_ar.py` never loads the extension; it rebuilds on the next `load_tp4()` in that venv.

## PERF ASSUMPTIONS

S1 probe choices:
- **Data path.** The reducer probe moves data with 16 B SM loads and `st.relaxed.sys.v4` stores.
  - Not taken: TMA-bulk (`cp.async.bulk`) remote pushes from the reducer CTA, or copy-engine pushes.
  - Why: S1 tests the planned design.
  - Cost: the kernel is limited by per-SM memory-level parallelism (about 60 GB/s per 1-CTA SM, 76 GB/s per 2-CTA SM). A bulk-copy
    path could raise per-SM throughput by an unknown factor.
- **No cross-unit pipelining.** Each worker issues all 5 loads of a unit before its stores (K = 2 or 3 vectors per thread), but the
  next unit's loads are not issued before the current unit's stores.
  - Not taken: double-buffered units / more bytes in flight per thread.
  - Why: simplicity; it mirrors the v3 skeleton.
  - Cost: probably most of the 1-CTA-per-SM gap (0.563 vs 0.442 ms at 4 SMs, about 25 %). This matters for the D2 role CTA.
- **Register cap.** `__launch_bounds__(THREADS, 2)` caps registers at 56 (544 threads) / 80 (384 threads) so that 2 CTAs fit per SM.
  - Not taken: the 168-register budget a D2 role CTA would have.
  - Why: the 3-kernel form needs 2 CTAs per SM.
  - Cost: the 384-thread numbers are pessimistic for D2 by an unmeasured amount.
- **Row sum of squares.** One gpu-scope `atomicAdd` per row segment per warp.
  - Not taken: CTA-exclusive panels with register accumulation.
  - Why: matches D1.
  - Cost: not isolated; at most one atomic per 512 B stored locally.
- **Probe 2 overlap.** The probe starts 50-65 µs after the GEMM (host lag), so about 10 % of the GEMM is not overlapped.
  - Not taken: a device-side start gate (an extra spin kernel triggered at GEMM start).
  - Why: the slowdown is ±2 % noise either way.
  - Cost: the slowdown is under-stated by at most ~0.3 pp. For the 4-CTA rows the probe span is optimistic, because 16-22 % of the
    probe runs after the GEMM ends.
- **Packed-placement emulation.** In probe 1 and the "4 free SMs" column of probe 2, a 200 KB-smem spinning blocker stands in for the
  GEMM.
  - Not taken: measuring under a real GEMM.
  - Why: it isolates the placement effect.
  - Cost: none on the decisive numbers, which come from probe 2 under a real GEMM.
- **Remote-epilogue worst case.** Probe 3 sends 100 % of D to one peer.
  - Not taken: the D1 4-way epilogue.
  - Why: it needs S2's epilogue patch.
  - Cost: the real D1 cost should be lower at 4k, but is unproven at 1k (1.2x here).
- **PDL probe.** A hand-written kernel with raw `griddepcontrol` PTX.
  - Not taken: the CUTLASS GDC path.
  - Why: GDC is compiled out today.
  - Cost: S3 must re-verify with `CUTLASS_ENABLE_GDC_FOR_SM90`.
- **Chain probe.** Plain polling without backoff, 16 KB / 256 KB payloads, 16 slots, rather than the real reducer pattern.
  - Cost: none on performance; it is an ordering probe only.
- **Spin limits and carveout.** The probe kernels trap after 2^24 polls (~9 s). Helper kernels use the max-shared carveout, and all
  helper kernels are launched once at start-up, to avoid lazy-loading deadlocks.
  - Cost: none on measured kernels.

Plan-level, carried forward with S1 evidence:
- Scatter by the producer's TMA epilogue: measured 1.11-1.12x at 4k for the one-peer worst case.
- 128 MiB/rank through R=4 with 8 CTAs (3-kernel form): 0.58 ms under the GEMM, shorter than the 0.64 ms producer. Not taken:
  bulk-copy or copy-engine pushes.
- Local rowss through gpu-scope red + last arriver: ordering probe passed.
- Value flags polled with `ld.acquire.sys`: one-way latency ~2.7 µs.
- Producer on 128 SMs: faster than 132 (0.637 vs 0.650 ms), so taking 4 SMs is free.
- Static tile schedule; one `tp4_step` kernel per boundary; consumer tile chosen from the 1-GPU sweep (unchanged; not exercised in S1).
- Engine baseline: NCCL ring at 0.33 ms b2b for 64 MiB is the bar. The torch-level `nccl` row is inflated by 0.61 ms of eager
  add + RMSNorm.

## What did not work (exact errors) and test status

1. **Packed placement crash.** The first and second quick runs of `push_bw.py` died at the first packed configuration (probe 1,
   inside `timed()` -> `torch.cuda.synchronize`) on all 4 ranks with
   `torch.AcceleratorError: CUDA error: unspecified launch failure`.
   - The first fix (max-shared carveout on the helper kernels) did not cure it.
   - **Root cause:** CUDA lazy module loading. The first-ever launch of `p6_wait_kernel` happened while the 131-CTA blocker was already
     spinning. Loading the module needs a context synchronisation, so wait kernel, blocker and probe deadlocked until the 2^24-poll
     spin-limit `__trap` fired (8.9 s, measured).
   - Isolated repro: `build/p6/dbg_spin.py wait131` traps; `build/p6/dbg_count.py` passes with the wait kernel launched first.
   - **Fix:** `push_bw.py` `worker()` launches every helper kernel once while the GPU is idle. The carveout attributes are kept.
2. **Probe 2 ordering artefact.** In the first probe-2 run the probe started before the GEMM (stamps: probe at -9 / -11 µs), because
   host enqueue of the probe took more than the 20 µs delay.
   - At 8 CTAs the probe spread over 8 SMs and pushed GEMM CTAs into a second wave: "+56.3 % slowdown", an artefact.
   - Fixed by starting both streams with delay kernels (300 / 320 µs), verified per call (0 of 80 calls on more than 4 SMs).
3. **Insensitive control.** The two-hop no-fence control never produced a violation (16 KB and 256 KB payloads), so the probe
   cannot show that A's fence is necessary. The one-hop control did (1760).
4. **Unreliable profiler durations.** The NCCL kernel durations from the torch profiler in `nccl_ar.py` are not usable for
   torch.distributed calls (values of 4-41 ms caused by cross-rank skew). Only `time_step` numbers are reported.

**Tests:** `tests/test_tp4.py --world 4` -> **ALL PASS** after all changes (`build/logs/p6_test_tp4.log`).

## Appendix: probe 1, all configurations

| M | placement | threads | T | blocks | variant | ms | kernel span ms | b2b ms | GB/s | SMs |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096 | spread | 544 | 2 | 2 | full | 1.096 | 1.076 | 1.081 | 122.4 | 2 |
| 4096 | spread | 544 | 2 | 2 | no_fence | 1.085 | 1.063 | 1.065 | 123.7 | 2 |
| 4096 | spread | 544 | 2 | 2 | no_remote | 0.958 | 0.940 | 0.944 | 87.6 | 2 |
| 4096 | spread | 544 | 2 | 2 | no_remote_no_fence | 0.939 | 0.923 | 0.928 | 89.3 | 2 |
| 4096 | spread | 544 | 2 | 4 | full | 0.563 | 0.545 | 0.550 | 238.3 | 4 |
| 4096 | spread | 544 | 2 | 4 | no_fence | 0.556 | 0.538 | 0.541 | 241.5 | 4 |
| 4096 | spread | 544 | 2 | 4 | no_remote | 0.490 | 0.472 | 0.476 | 171.1 | 4 |
| 4096 | spread | 544 | 2 | 4 | no_remote_no_fence | 0.480 | 0.463 | 0.467 | 174.7 | 4 |
| 4096 | spread | 544 | 2 | 8 | full | 0.312 | 0.295 | 0.300 | 429.9 | 8 |
| 4096 | spread | 544 | 2 | 8 | no_fence | 0.310 | 0.291 | 0.297 | 433.4 | 8 |
| 4096 | spread | 544 | 2 | 8 | no_remote | 0.258 | 0.241 | 0.245 | 325.4 | 8 |
| 4096 | spread | 544 | 2 | 8 | no_remote_no_fence | 0.253 | 0.236 | 0.239 | 332.2 | 8 |
| 4096 | spread | 544 | 2 | 16 | full | 0.177 | 0.159 | 0.161 | 759.7 | 16 |
| 4096 | spread | 544 | 2 | 16 | no_fence | 0.175 | 0.156 | 0.159 | 768.0 | 16 |
| 4096 | spread | 544 | 2 | 16 | no_remote | 0.146 | 0.127 | 0.131 | 576.0 | 16 |
| 4096 | spread | 544 | 2 | 16 | no_remote_no_fence | 0.141 | 0.124 | 0.128 | 593.2 | 16 |
| 4096 | spread | 544 | 4 | 2 | full | 1.224 | 1.205 | 1.209 | 109.7 | 2 |
| 4096 | spread | 544 | 4 | 2 | no_fence | 1.218 | 1.200 | 1.204 | 110.2 | 2 |
| 4096 | spread | 544 | 4 | 2 | no_remote | 1.077 | 1.058 | 1.061 | 77.9 | 2 |
| 4096 | spread | 544 | 4 | 2 | no_remote_no_fence | 1.070 | 1.053 | 1.057 | 78.4 | 2 |
| 4096 | spread | 544 | 4 | 4 | full | 0.628 | 0.608 | 0.612 | 213.8 | 4 |
| 4096 | spread | 544 | 4 | 4 | no_fence | 0.622 | 0.604 | 0.608 | 215.6 | 4 |
| 4096 | spread | 544 | 4 | 4 | no_remote | 0.548 | 0.531 | 0.535 | 153.0 | 4 |
| 4096 | spread | 544 | 4 | 4 | no_remote_no_fence | 0.545 | 0.528 | 0.532 | 153.8 | 4 |
| 4096 | spread | 544 | 4 | 8 | full | 0.342 | 0.325 | 0.329 | 392.6 | 8 |
| 4096 | spread | 544 | 4 | 8 | no_fence | 0.336 | 0.322 | 0.327 | 399.8 | 8 |
| 4096 | spread | 544 | 4 | 8 | no_remote | 0.288 | 0.271 | 0.275 | 291.2 | 8 |
| 4096 | spread | 544 | 4 | 8 | no_remote_no_fence | 0.286 | 0.269 | 0.273 | 293.5 | 8 |
| 4096 | spread | 544 | 4 | 16 | full | 0.188 | 0.174 | 0.177 | 713.4 | 16 |
| 4096 | spread | 544 | 4 | 16 | no_fence | 0.187 | 0.172 | 0.176 | 718.6 | 16 |
| 4096 | spread | 544 | 4 | 16 | no_remote | 0.161 | 0.143 | 0.147 | 522.0 | 16 |
| 4096 | spread | 544 | 4 | 16 | no_remote_no_fence | 0.158 | 0.141 | 0.145 | 530.9 | 16 |
| 4096 | spread | 384 | 2 | 2 | full | 1.196 | 1.175 | 1.179 | 112.2 | 2 |
| 4096 | spread | 384 | 2 | 2 | no_fence | 1.182 | 1.160 | 1.164 | 113.6 | 2 |
| 4096 | spread | 384 | 2 | 2 | no_remote | 1.052 | 1.034 | 1.038 | 79.7 | 2 |
| 4096 | spread | 384 | 2 | 2 | no_remote_no_fence | 1.042 | 1.023 | 1.027 | 80.5 | 2 |
| 4096 | spread | 384 | 2 | 4 | full | 0.613 | 0.596 | 0.599 | 218.9 | 4 |
| 4096 | spread | 384 | 2 | 4 | no_fence | 0.604 | 0.586 | 0.591 | 222.2 | 4 |
| 4096 | spread | 384 | 2 | 4 | no_remote | 0.537 | 0.520 | 0.524 | 156.1 | 4 |
| 4096 | spread | 384 | 2 | 4 | no_remote_no_fence | 0.535 | 0.514 | 0.518 | 156.8 | 4 |
| 4096 | spread | 384 | 2 | 8 | full | 0.330 | 0.309 | 0.313 | 406.7 | 8 |
| 4096 | spread | 384 | 2 | 8 | no_fence | 0.326 | 0.305 | 0.310 | 412.2 | 8 |
| 4096 | spread | 384 | 2 | 8 | no_remote | 0.284 | 0.265 | 0.269 | 295.1 | 8 |
| 4096 | spread | 384 | 2 | 8 | no_remote_no_fence | 0.281 | 0.261 | 0.265 | 298.2 | 8 |
| 4096 | spread | 384 | 2 | 16 | full | 0.186 | 0.165 | 0.170 | 721.4 | 16 |
| 4096 | spread | 384 | 2 | 16 | no_fence | 0.183 | 0.162 | 0.166 | 732.4 | 16 |
| 4096 | spread | 384 | 2 | 16 | no_remote | 0.159 | 0.138 | 0.143 | 528.0 | 16 |
| 4096 | spread | 384 | 2 | 16 | no_remote_no_fence | 0.157 | 0.136 | 0.140 | 532.7 | 16 |
| 4096 | spread | 384 | 4 | 2 | full | 1.167 | 1.148 | 1.152 | 115.0 | 2 |
| 4096 | spread | 384 | 4 | 2 | no_fence | 1.155 | 1.137 | 1.141 | 116.2 | 2 |
| 4096 | spread | 384 | 4 | 2 | no_remote | 1.031 | 1.013 | 1.017 | 81.4 | 2 |
| 4096 | spread | 384 | 4 | 2 | no_remote_no_fence | 1.025 | 1.007 | 1.011 | 81.8 | 2 |
| 4096 | spread | 384 | 4 | 4 | full | 0.599 | 0.581 | 0.585 | 224.0 | 4 |
| 4096 | spread | 384 | 4 | 4 | no_fence | 0.593 | 0.574 | 0.579 | 226.5 | 4 |
| 4096 | spread | 384 | 4 | 4 | no_remote | 0.529 | 0.509 | 0.512 | 158.5 | 4 |
| 4096 | spread | 384 | 4 | 4 | no_remote_no_fence | 0.522 | 0.505 | 0.509 | 160.6 | 4 |
| 4096 | spread | 384 | 4 | 8 | full | 0.326 | 0.307 | 0.311 | 411.5 | 8 |
| 4096 | spread | 384 | 4 | 8 | no_fence | 0.321 | 0.304 | 0.308 | 418.0 | 8 |
| 4096 | spread | 384 | 4 | 8 | no_remote | 0.279 | 0.260 | 0.264 | 300.3 | 8 |
| 4096 | spread | 384 | 4 | 8 | no_remote_no_fence | 0.275 | 0.257 | 0.261 | 305.2 | 8 |
| 4096 | spread | 384 | 4 | 16 | full | 0.181 | 0.164 | 0.168 | 742.3 | 16 |
| 4096 | spread | 384 | 4 | 16 | no_fence | 0.181 | 0.161 | 0.166 | 741.1 | 16 |
| 4096 | spread | 384 | 4 | 16 | no_remote | 0.155 | 0.136 | 0.140 | 541.3 | 16 |
| 4096 | spread | 384 | 4 | 16 | no_remote_no_fence | 0.153 | 0.134 | 0.138 | 548.2 | 16 |
| 4096 | packed | 544 | 2 | 2 | full | 1.715 | 1.703 | n/a | 78.3 | 1 |
| 4096 | packed | 544 | 2 | 2 | no_fence | 1.692 | 1.679 | n/a | 79.3 | 1 |
| 4096 | packed | 544 | 2 | 2 | no_remote | 1.225 | 1.212 | n/a | 68.5 | 1 |
| 4096 | packed | 544 | 2 | 2 | no_remote_no_fence | 1.197 | 1.184 | n/a | 70.1 | 1 |
| 4096 | packed | 544 | 2 | 4 | full | 0.867 | 0.855 | n/a | 154.8 | 2 |
| 4096 | packed | 544 | 2 | 4 | no_fence | 0.856 | 0.843 | n/a | 156.9 | 2 |
| 4096 | packed | 544 | 2 | 4 | no_remote | 0.620 | 0.607 | n/a | 135.3 | 2 |
| 4096 | packed | 544 | 2 | 4 | no_remote_no_fence | 0.606 | 0.593 | n/a | 138.5 | 2 |
| 4096 | packed | 544 | 2 | 8 | full | 0.442 | 0.431 | n/a | 303.5 | 4 |
| 4096 | packed | 544 | 2 | 8 | no_fence | 0.436 | 0.424 | n/a | 307.8 | 4 |
| 4096 | packed | 544 | 2 | 8 | no_remote | 0.319 | 0.307 | n/a | 263.0 | 4 |
| 4096 | packed | 544 | 2 | 8 | no_remote_no_fence | 0.310 | 0.298 | n/a | 270.2 | 4 |
| 4096 | packed | 544 | 2 | 16 | full | 0.231 | 0.220 | n/a | 579.9 | 8 |
| 4096 | packed | 544 | 2 | 16 | no_fence | 0.227 | 0.215 | n/a | 590.3 | 8 |
| 4096 | packed | 544 | 2 | 16 | no_remote | 0.168 | 0.157 | n/a | 499.0 | 8 |
| 4096 | packed | 544 | 2 | 16 | no_remote_no_fence | 0.164 | 0.152 | n/a | 512.3 | 8 |
| 4096 | packed | 544 | 4 | 2 | full | 1.848 | 1.835 | n/a | 72.6 | 1 |
| 4096 | packed | 544 | 4 | 2 | no_fence | 1.846 | 1.832 | n/a | 72.7 | 1 |
| 4096 | packed | 544 | 4 | 2 | no_remote | 1.274 | 1.261 | n/a | 65.8 | 1 |
| 4096 | packed | 544 | 4 | 2 | no_remote_no_fence | 1.264 | 1.251 | n/a | 66.4 | 1 |
| 4096 | packed | 544 | 4 | 4 | full | 0.933 | 0.920 | n/a | 143.8 | 2 |
| 4096 | packed | 544 | 4 | 4 | no_fence | 0.931 | 0.918 | n/a | 144.1 | 2 |
| 4096 | packed | 544 | 4 | 4 | no_remote | 0.645 | 0.632 | n/a | 130.0 | 2 |
| 4096 | packed | 544 | 4 | 4 | no_remote_no_fence | 0.639 | 0.627 | n/a | 131.3 | 2 |
| 4096 | packed | 544 | 4 | 8 | full | 0.476 | 0.465 | n/a | 281.8 | 4 |
| 4096 | packed | 544 | 4 | 8 | no_fence | 0.474 | 0.461 | n/a | 283.3 | 4 |
| 4096 | packed | 544 | 4 | 8 | no_remote | 0.330 | 0.318 | n/a | 254.2 | 4 |
| 4096 | packed | 544 | 4 | 8 | no_remote_no_fence | 0.327 | 0.315 | n/a | 256.3 | 4 |
| 4096 | packed | 544 | 4 | 16 | full | 0.249 | 0.238 | n/a | 538.3 | 8 |
| 4096 | packed | 544 | 4 | 16 | no_fence | 0.246 | 0.235 | n/a | 544.8 | 8 |
| 4096 | packed | 544 | 4 | 16 | no_remote | 0.175 | 0.163 | n/a | 479.0 | 8 |
| 4096 | packed | 544 | 4 | 16 | no_remote_no_fence | 0.172 | 0.161 | n/a | 487.3 | 8 |
| 4096 | packed | 384 | 2 | 2 | full | 1.755 | 1.742 | n/a | 76.5 | 1 |
| 4096 | packed | 384 | 2 | 2 | no_fence | 1.731 | 1.718 | n/a | 77.5 | 1 |
| 4096 | packed | 384 | 2 | 2 | no_remote | 1.326 | 1.314 | n/a | 63.3 | 1 |
| 4096 | packed | 384 | 2 | 2 | no_remote_no_fence | 1.309 | 1.296 | n/a | 64.1 | 1 |
| 4096 | packed | 384 | 2 | 4 | full | 0.888 | 0.875 | n/a | 151.2 | 2 |
| 4096 | packed | 384 | 2 | 4 | no_fence | 0.875 | 0.862 | n/a | 153.4 | 2 |
| 4096 | packed | 384 | 2 | 4 | no_remote | 0.672 | 0.659 | n/a | 124.9 | 2 |
| 4096 | packed | 384 | 2 | 4 | no_remote_no_fence | 0.661 | 0.649 | n/a | 126.9 | 2 |
| 4096 | packed | 384 | 2 | 8 | full | 0.454 | 0.442 | n/a | 295.8 | 4 |
| 4096 | packed | 384 | 2 | 8 | no_fence | 0.447 | 0.434 | n/a | 300.6 | 4 |
| 4096 | packed | 384 | 2 | 8 | no_remote | 0.344 | 0.332 | n/a | 243.8 | 4 |
| 4096 | packed | 384 | 2 | 8 | no_remote_no_fence | 0.338 | 0.326 | n/a | 248.1 | 4 |
| 4096 | packed | 384 | 2 | 16 | full | 0.238 | 0.227 | n/a | 562.9 | 8 |
| 4096 | packed | 384 | 2 | 16 | no_fence | 0.234 | 0.222 | n/a | 573.2 | 8 |
| 4096 | packed | 384 | 2 | 16 | no_remote | 0.181 | 0.169 | n/a | 462.7 | 8 |
| 4096 | packed | 384 | 2 | 16 | no_remote_no_fence | 0.178 | 0.166 | n/a | 471.2 | 8 |
| 4096 | packed | 384 | 4 | 2 | full | 1.696 | 1.683 | n/a | 79.1 | 1 |
| 4096 | packed | 384 | 4 | 2 | no_fence | 1.689 | 1.676 | n/a | 79.5 | 1 |
| 4096 | packed | 384 | 4 | 2 | no_remote | 1.284 | 1.271 | n/a | 65.3 | 1 |
| 4096 | packed | 384 | 4 | 2 | no_remote_no_fence | 1.274 | 1.262 | n/a | 65.9 | 1 |
| 4096 | packed | 384 | 4 | 4 | full | 0.857 | 0.846 | n/a | 156.6 | 2 |
| 4096 | packed | 384 | 4 | 4 | no_fence | 0.851 | 0.840 | n/a | 157.7 | 2 |
| 4096 | packed | 384 | 4 | 4 | no_remote | 0.650 | 0.637 | n/a | 129.1 | 2 |
| 4096 | packed | 384 | 4 | 4 | no_remote_no_fence | 0.643 | 0.631 | n/a | 130.4 | 2 |
| 4096 | packed | 384 | 4 | 8 | full | 0.439 | 0.426 | n/a | 305.9 | 4 |
| 4096 | packed | 384 | 4 | 8 | no_fence | 0.437 | 0.423 | n/a | 307.3 | 4 |
| 4096 | packed | 384 | 4 | 8 | no_remote | 0.335 | 0.321 | n/a | 250.2 | 4 |
| 4096 | packed | 384 | 4 | 8 | no_remote_no_fence | 0.332 | 0.317 | n/a | 252.4 | 4 |
| 4096 | packed | 384 | 4 | 16 | full | 0.233 | 0.218 | n/a | 577.3 | 8 |
| 4096 | packed | 384 | 4 | 16 | no_fence | 0.230 | 0.215 | n/a | 584.0 | 8 |
| 4096 | packed | 384 | 4 | 16 | no_remote | 0.179 | 0.164 | n/a | 469.7 | 8 |
| 4096 | packed | 384 | 4 | 16 | no_remote_no_fence | 0.176 | 0.161 | n/a | 477.7 | 8 |
| 1024 | spread | 544 | 2 | 4 | full | 0.149 | 0.130 | 0.134 | 225.2 | 4 |
| 1024 | spread | 544 | 2 | 8 | full | 0.089 | 0.070 | 0.074 | 378.5 | 8 |
| 1024 | spread | 384 | 2 | 4 | full | 0.154 | 0.136 | 0.141 | 218.5 | 4 |
| 1024 | spread | 384 | 2 | 8 | full | 0.088 | 0.071 | 0.076 | 380.7 | 8 |
| 1024 | packed | 544 | 2 | 4 | full | 0.229 | 0.217 | n/a | 146.8 | 2 |
| 1024 | packed | 544 | 2 | 8 | full | 0.124 | 0.112 | n/a | 270.0 | 4 |
| 1024 | packed | 384 | 2 | 4 | full | 0.228 | 0.216 | n/a | 147.2 | 2 |
| 1024 | packed | 384 | 2 | 8 | full | 0.123 | 0.111 | n/a | 273.0 | 4 |

check M=4096 threads=544: x mismatches 0, rowss max rel err 2.94e-07

check M=4096 threads=384: x mismatches 0, rowss max rel err 2.99e-07
