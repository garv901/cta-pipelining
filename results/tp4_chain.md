# Phase 4: TP4 down-proj -> residual -> RMSNorm -> QKV on node1 (8x H100 NVSwitch)

**Caveat (corrected 2026-10-02, Phase 6 S1).** The baselines in the Phase 4 tables below are torch-level implementations
(torch symmetric memory, torch.distributed NCCL, FlashInfer). The `nccl` row (1.767 ms b2b at 4k) is NOT a slow all-reduce: it
decomposes into down mm 0.622 + `dist.all_reduce` 0.331 + residual add 0.070 + eager fp32 RMSNorm 0.538 + QKV mm 0.220 ms, i.e. the
64 MB all-reduce itself is 0.33 ms (NCCL Ring / Simple, 24 channels, same in both venvs and through vLLM's PyNccl) and ~0.61 ms is
unfused eager add + RMSNorm. The earlier "~0.75 ms NCCL all-reduce" reading came from counting that unfused norm as reduction
(only `NCCL_PROTO=LL` makes the all-reduce itself ~0.74 ms; the vLLM kernel name `..._RING_LL` is NCCL's entry kernel, not the
protocol). vLLM's stock eager boundary B is down 0.62 + NCCL 0.33-0.38 + `fused_add_rms_norm` 0.09 + QKV 0.23 = **1.32 ms** at 4k
(Phase 5 profile), so that, not the `nccl` row, is the honest stock reference; the stand-alone gains below overstate what an
engine can gain. End-to-end numbers: `results/vllm_prefill_4k.md` (Phase 5 and Phase 6 sections).

Chain per layer boundary: Llama-70B down-projection (K=28672 sharded 4 ways, N=8192) -> all-reduce -> residual add -> RMSNorm -> QKV (N=10240, column-sharded). TP4, one process per GPU, bf16 with fp32 accumulate. `ideal` = 1-GPU eager cuBLAS time of the same chain / 4. exposed % = (t - ideal) / t. Back-to-back = calls queued without host sync (production condition); per-step = barrier + sync before each call. 50 iterations, 5 warm-up, max over ranks.

Methods: `ctapp` = our kernel: mode-4 CUTLASS SM90 cooperative GEMM on (SMs - R) SMs signalling per-tile counters via multimem; a standalone reducer kernel v3 (R blocks: 12 for M <= 2048, 8 above; 16 worker warps + 1 signaller warp) that does the NVLS `multimem.ld_reduce` per tile, adds the residual, writes the replicated x via `multimem.st`, accumulates the row sum-of-squares and signals per 128-row panel; then the QKV GEMM (raster 2) whose CTAs wait on the panel counter and apply the RMSNorm in the epilogue. `ctapp_graph` = the same captured in CUDA graphs (two graphs by epoch parity, static input copies included in the timing). `ctapp_nowait` = the same two GEMMs on (SMs - R) SMs with no reduction = floor. `multimem` = torch symm-mem NVLS all-reduce + eager RMSNorm. `flashinfer` = eager cuBLAS down-proj, then FlashInfer `trtllm_allreduce_fusion` (pattern kARResidualRMSNorm; one kernel does all-reduce + residual add + RMSNorm and writes the new residual; fp32_acc, no PDL, oneshot/twoshot chosen by FlashInfer's heuristic), then the QKV GEMM; the reduction is not overlapped with the GEMM. `asynctp` = torch async-TP fused_matmul_reduce_scatter + RMSNorm on own rows + NCCL all-gather. `nccl` = dist.all_reduce + eager RMSNorm.

## back-to-back: ms (exposed %)

| M | 1-GPU | ideal | ctapp | ctapp_graph | ctapp_nowait | multimem | flashinfer | asynctp | nccl | best ctapp vs best baseline (gain = 1 - ctapp / best baseline) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 0.983 | 0.246 | 0.344 (29%) | 0.360 (32%) | 0.300 (18%) | 0.452 (46%) | 0.344 (28%) | 1.023 (76%) | 0.477 (48%) | ctapp 0.344 vs flashinfer 0.344: -0 % |
| 2048 | 2.049 | 0.512 | 0.590 (13%) | 0.637 (20%) | 0.522 (2%) | 0.878 (42%) | 0.658 (22%) | 0.662 (23%) | 0.910 (44%) | ctapp 0.590 vs flashinfer 0.658: 10 % |
| 4096 | 4.113 | 1.028 | 1.053 (2%) | 1.155 (11%) | 0.936 (-10%) | 1.714 (40%) | 1.298 (21%) | 1.285 (20%) | 1.759 (42%) | ctapp 1.053 vs asynctp 1.285: 18 % |
| 8192 | 8.183 | 2.046 | 2.236 (9%) | 2.377 (14%) | 1.996 (-2%) | 3.418 (40%) | 2.615 (22%) | 2.531 (19%) | 3.481 (41%) | ctapp 2.236 vs asynctp 2.531: 12 % |
| 16384 | 16.992 | 4.248 | 4.754 (11%) | 5.049 (16%) | 4.254 (0%) | 6.760 (37%) | 5.240 (19%) | 5.022 (15%) | 6.925 (39%) | ctapp 4.754 vs asynctp 5.022: 5 % |

## per step: ms (exposed %)

| M | 1-GPU | ideal | ctapp | ctapp_graph | ctapp_nowait | multimem | flashinfer | asynctp | nccl | best ctapp vs best baseline (gain = 1 - ctapp / best baseline) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 0.983 | 0.246 | 0.467 (47%) | 0.422 (42%) | 0.325 (24%) | 0.526 (53%) | 0.424 (42%) | 1.077 (77%) | 0.567 (57%) | ctapp_graph 0.422 vs flashinfer 0.424: 0 % |
| 2048 | 2.049 | 0.512 | 0.711 (28%) | 0.673 (24%) | 0.539 (5%) | 0.959 (47%) | 0.734 (30%) | 0.930 (45%) | 0.991 (48%) | ctapp_graph 0.673 vs flashinfer 0.734: 8 % |
| 4096 | 4.113 | 1.028 | 1.191 (14%) | 1.178 (13%) | 0.927 (-11%) | 1.782 (42%) | 1.359 (24%) | 1.588 (35%) | 1.831 (44%) | ctapp_graph 1.178 vs flashinfer 1.359: 13 % |
| 8192 | 8.183 | 2.046 | 2.206 (7%) | 2.282 (10%) | 1.899 (-8%) | 3.465 (41%) | 2.667 (23%) | 2.706 (24%) | 3.521 (42%) | ctapp 2.206 vs flashinfer 2.667: 17 % |
| 16384 | 16.992 | 4.248 | 4.750 (11%) | 5.005 (15%) | 4.234 (-0%) | 6.795 (37%) | 5.328 (20%) | 5.133 (17%) | 6.950 (39%) | ctapp 4.750 vs asynctp 5.133: 7 % |

rel err vs fp32 reference (max over M): ctapp 0.0046, ctapp_graph 0.0046, ctapp_nowait nan (not checked), multimem 0.0051, flashinfer 0.0049, asynctp 0.0051, nccl 0.0053

## Reading the table (Phase 4 tables)

- The gain over the best baseline (gain = 1 - ctapp / best baseline, back-to-back) is 0 % at 1k (tie with FlashInfer, 0.344 ms each), 10 % at 2k, 18 % at 4k, 12 % at 8k and 5 % at 16k. The best baseline is FlashInfer at 1k-2k and async-TP from 4k up; against multimem alone the gain is 24 % at 1k and 33 % at 2k.
- FlashInfer's fused all-reduce + residual + RMSNorm lands between multimem and async-TP: at 1k it is 24 % faster than multimem and matches ctapp (b2b), at 2k it ties async-TP (0.658 vs 0.662 ms), and from 4k up it is 1-4 % slower than async-TP (22 % vs 19-20 % exposed at 4k-8k, 19 % vs 15 % at 16k) but 22-25 % faster than multimem. Its per-step times are the best baseline at 1k-8k (async-TP is 0.93-1.08 ms at 1k-2k per step; async-TP at 1k is also unstable between runs, 0.609 ms b2b in the earlier run vs 1.023 ms here).
- The remaining gap to the `ctapp_nowait` floor is the R SMs lent to the reducer plus a ~10 % GEMM slowdown from multimem traffic landing in the GEMM GPU.
- The graph variant wins per-step at M <= 2k (host launch overhead) and loses b2b everywhere because the timed region includes copying the inputs into static buffers (a production integration would capture against caller-owned buffers).
- The gate from PLAN.md (>= 10 % over async-TP at 4k/8k, >= 20 % over multimem at 1k-2k) is met at 1k (24 % vs multimem), 2k (33 % vs multimem, 11 % vs async-TP), 4k (18 % vs async-TP) and 8k (12 %). The FlashInfer production baseline was added after the gate was written; against it the gain is 0 % at 1k, 10 % at 2k, 19 % at 4k, 14 % at 8k and 9 % at 16k.

## Phase 6: v5 protocol (panel-ownership unicast, value flags, one max-M instance), b2b ms, max over ranks

Same chain and harness (`bench/tp4_bench.py --protocol both --repeat 5`, interleaved median of 5 repeats; node drift up to 5 %,
so compare within a column group of one run only). R = 4 for v5 (producer and consumer on 128 SMs), v3 at its default R (12 for
M <= 2048, 8 above). `v5 none` = 3-kernel form (producer mode 7 || owner reducer on a side stream -> consumer mode 5); `v5 pdl` =
consumer launched as the producer's PDL dependent (the vLLM default for boundary B); both with the panel-major tail tile order
(`down_tail="auto"`) and consumer tile 128x256. Sources: `results/phase6_s3.md` tables 2a (`build/p6_s3_chain_rep.json`, run 1;
`build/p6_s3_chain_steal.json`, run 2) and `results/phase6_s2.md` table 2a (`build/tp4_4_both_R4.json`, 16k, S2 order, no tail).

| M | v3 (run 1) | v5 none (run 1) | v5 pdl (run 1) | v3 (run 2) | v5 none (run 2) | v5 pdl (run 2) | v3 / v5 none at 16k (S2 run) | stock vLLM eager boundary B (profile) |
|---|---|---|---|---|---|---|---|---|
| 1024 | 0.348 | 0.334 | 0.334 | 0.348 | 0.335 | 0.334 | | |
| 2048 | 0.603 | 0.568 | 0.557 | 0.600 | 0.566 | 0.555 | | |
| 4096 | 1.090 | 1.086 | 1.087 | 1.084 | 1.081 | **1.074** | | 1.32 |
| 8192 | 2.306 | 2.217 | 2.287 | 2.305 | 2.243 | **2.227** | | |
| 16384 | | | | | | | 4.770 / 4.599 | |

out rel err vs fp32 reference: v5 4.16-4.40e-3, v3 4.3-4.6e-3. v5 `pdl` at 16k was not measured. The boundary-A shape (o_proj ->
gate_up, Kr 2048, N2r 14336) at 4k: v3 2.152, v5 pdl 2.198, v5 pdl + steal **1.882** (stock in the vLLM profile 1.83).

Reading: v5 is 1-4 % faster than v3 at 1k-8k and 4 % at 16k, much less than planned (target <= 0.95 ms at 4k for boundary B). The
critical path is the mode-7 producer (0.72-0.75 ms vs 0.626 for the local-store mode 0: wave-synchronous NVLink egress of the
remote TMA stores) followed by the consumer at near-peak speed (0.27-0.33 ms); the reduction is off the critical path. Corrected
reference for the old `nccl` row: all-reduce 0.33 ms + unfused add / RMSNorm 0.61 ms (caveat above).

## Reproduce

Needs a Slurm hold job on node1 and `/tmp/run4.sh` (sets PATH/CUDA_HOME/TMPDIR and `CUDA_VISIBLE_DEVICES=$1`, then `exec "$@"`). The extension JIT-builds into `build/ext_tp4` on first use; `csrc/sm90_gemm_coop_ctapp.hpp` is generated by `scripts/gen_coop_ctapp.py`.

```bash
srun --jobid=<hold job> --overlap bash /tmp/run4.sh 0,1,2,3 python -u tests/test_tp4.py --world 4
srun --jobid=<hold job> --overlap bash /tmp/run4.sh 0,1,2,3 python -u bench/tp4_bench.py --world 4 --tokens 1024,2048,4096,8192,16384 --iters 50 --warmup 5 --json build/tp4_w4.json
```
