# Phase 4: TP4 down-proj -> residual -> RMSNorm -> QKV on node1 (8x H100 NVSwitch)

**Caveat (2026-10-02, after the vLLM port).** The baselines in this table are torch-level implementations
(torch symmetric memory, torch.distributed NCCL, FlashInfer). vLLM's eager NCCL path does the 64 MB all-reduce in
0.35-0.38 ms, about 2x faster than the `nccl` row here (~0.7-0.75 ms; the cause of the difference is not yet established),
so the gains in this table overstate what is available in an engine. End to end in vLLM at 4K prefill the down boundary
is only 3 % faster than stock eager and 3 % slower than compiled async-TP: see `results/vllm_prefill_4k.md`.

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

## Reading the table

- The gain over the best baseline (gain = 1 - ctapp / best baseline, back-to-back) is 0 % at 1k (tie with FlashInfer, 0.344 ms each), 10 % at 2k, 18 % at 4k, 12 % at 8k and 5 % at 16k. The best baseline is FlashInfer at 1k-2k and async-TP from 4k up; against multimem alone the gain is 24 % at 1k and 33 % at 2k.
- FlashInfer's fused all-reduce + residual + RMSNorm lands between multimem and async-TP: at 1k it is 24 % faster than multimem and matches ctapp (b2b), at 2k it ties async-TP (0.658 vs 0.662 ms), and from 4k up it is 1-4 % slower than async-TP (22 % vs 19-20 % exposed at 4k-8k, 19 % vs 15 % at 16k) but 22-25 % faster than multimem. Its per-step times are the best baseline at 1k-8k (async-TP is 0.93-1.08 ms at 1k-2k per step; async-TP at 1k is also unstable between runs, 0.609 ms b2b in the earlier run vs 1.023 ms here).
- The remaining gap to the `ctapp_nowait` floor is the R SMs lent to the reducer plus a ~10 % GEMM slowdown from multimem traffic landing in the GEMM GPU.
- The graph variant wins per-step at M <= 2k (host launch overhead) and loses b2b everywhere because the timed region includes copying the inputs into static buffers (a production integration would capture against caller-owned buffers).
- The gate from PLAN.md (>= 10 % over async-TP at 4k/8k, >= 20 % over multimem at 1k-2k) is met at 1k (24 % vs multimem), 2k (33 % vs multimem, 11 % vs async-TP), 4k (18 % vs async-TP) and 8k (12 %). The FlashInfer production baseline was added after the gate was written; against it the gain is 0 % at 1k, 10 % at 2k, 19 % at 4k, 14 % at 8k and 9 % at 16k.

## Reproduce

Needs a Slurm hold job on node1 and `/tmp/run4.sh` (sets PATH/CUDA_HOME/TMPDIR and `CUDA_VISIBLE_DEVICES=$1`, then `exec "$@"`). The extension JIT-builds into `build/ext_tp4` on first use; `csrc/sm90_gemm_coop_ctapp.hpp` is generated by `scripts/gen_coop_ctapp.py`.

```bash
srun --jobid=<hold job> --overlap bash /tmp/run4.sh 0,1,2,3 python -u tests/test_tp4.py --world 4
srun --jobid=<hold job> --overlap bash /tmp/run4.sh 0,1,2,3 python -u bench/tp4_bench.py --world 4 --tokens 1024,2048,4096,8192,16384 --iters 50 --warmup 5 --json build/tp4_w4.json
```
