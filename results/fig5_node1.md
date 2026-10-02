# S0.3: Fig-5 methods recalibrated on node1 (2x H100 NVSwitch, 369 GB/s SM-store link), gated latency us, median of 20 reps

Methods: `OneGpuCublas` (both GEMMs on one GPU; ideal = half of it), `MicroBatchCublas` (chunked rows, GEMM1 on A -> cudaMemcpyPeer -> GEMM2 on B; best chunk shown), `TensorParallel2` (Megatron shards, plain NCCL all-reduce, eager), `Ctapp` (Phase-1 KernelTma kernels with the workqueue/scoreboard protocol, best cfg/order). `l70b` is SwiGLU (micro-batching ships the 2x28672-wide [gate; up] tensor and applies SwiGLU on B); `l70b_qkv` here is the plain GEMM chain without the RMSNorm.

## paper: 8192 -> 8192 -> 8192

| M | 1-GPU | ideal | MB best (chunk) | MB/ideal | TP2 nccl | TP2/ideal | CTAPP best (variant) | CTAPP/ideal |
|---|---|---|---|---|---|---|---|---|
| 1024 | 353 | 177 | 321 (chunk=512) | 1.82 | 266 | 1.51 | 397 (cfg7/7 group1) | 2.25 |
| 2048 | 710 | 355 | 509 (chunk=512) | 1.43 | 499 | 1.41 | 670 (cfg7/7 group1) | 1.89 |
| 4096 | 1413 | 707 | 892 (chunk=512) | 1.26 | 963 | 1.36 | 1236 (cfg7/7 group1) | 1.75 |
| 8192 | 2987 | 1494 | 1650 (chunk=512) | 1.10 | 1917 | 1.28 | 2333 (cfg7/7 group8) | 1.56 |
| 16384 | 6345 | 3172 | 3153 (chunk=1024) | 0.99 | 3845 | 1.21 | 4374 (cfg7/7 group8) | 1.38 |

## l70b: Llama-70B FFN 8192 -> 2x28672 SwiGLU -> 8192

| M | 1-GPU | ideal | MB best (chunk) | MB/ideal | TP2 nccl | TP2/ideal | CTAPP best (variant) | CTAPP/ideal |
|---|---|---|---|---|---|---|---|---|
| 1024 | 1004 | 502 | 938 (chunk=512) | 1.87 | 599 | 1.19 | n/a (SwiGLU not supported by the TMA kernels) | n/a |
| 2048 | 2046 | 1023 | 1605 (chunk=512) | 1.57 | 1160 | 1.13 | n/a (SwiGLU not supported by the TMA kernels) | n/a |
| 4096 | 4266 | 2133 | 2946 (chunk=512) | 1.38 | 2305 | 1.08 | n/a (SwiGLU not supported by the TMA kernels) | n/a |
| 8192 | 8337 | 4168 | 5618 (chunk=512) | 1.35 | 4697 | 1.13 | n/a (SwiGLU not supported by the TMA kernels) | n/a |
| 16384 | 16794 | 8397 | 10716 (chunk=1024) | 1.28 | 9722 | 1.16 | n/a (SwiGLU not supported by the TMA kernels) | n/a |

## l70b_qkv: Llama-70B down -> QKV, 28672 -> 8192 -> 10240

| M | 1-GPU | ideal | MB best (chunk) | MB/ideal | TP2 nccl | TP2/ideal | CTAPP best (variant) | CTAPP/ideal |
|---|---|---|---|---|---|---|---|---|
| 1024 | 829 | 414 | 785 (chunk=512) | 1.89 | 516 | 1.24 | 1196 (cfg4/4 group8) | 2.88 |
| 2048 | 1676 | 838 | 1421 (chunk=512) | 1.70 | 1009 | 1.20 | 2121 (cfg4/4 group8) | 2.53 |
| 4096 | 3664 | 1832 | 2711 (chunk=512) | 1.48 | 2027 | 1.11 | 3980 (cfg7/7 group8) | 2.17 |
| 8192 | 7636 | 3818 | 5270 (chunk=512) | 1.38 | 4353 | 1.14 | 7832 (cfg7/7 group8) | 2.05 |
| 16384 | 14445 | 7223 | 10365 (chunk=512) | 1.44 | 8518 | 1.18 | 16751 (cfg6/6 group8) | 2.32 |

## Like-for-like KernelTma, paper shape (10 reps)

| M | cuBLAS 1-GPU | OneGpuTma cfg4 | OneGpuTma cfg7 | MicroBatchTma best | CTAPP best | CTAPP vs MicroBatchTma |
|---|---|---|---|---|---|---|
| 4096 | 1386 | 1691 | 1873 | 996 (cfg7 chunk=512) | 1233 (cfg7/7 group1) | -24% |
| 16384 | 6220 | 8712 | 9188 | 3560 (cfg7 chunk=512) | 4406 (cfg7/7 group8) | -24% |

## Reading

- The KernelTma kernels run at 0.68-0.74x cuBLAS on node1 (as on the old machine), and the Phase-1 CTAPP pipeline is still 24-25 % slower than micro-batching the same kernel. CTAPP/ideal 1.4-2.3x (paper) and 2.0-2.9x (down->QKV) is therefore a kernel-efficiency result, not a link result; the old 124 GB/s numbers were 1.9/1.6/1.3/1.1/0.96x with a slower cuBLAS as the reference. Any further CTAPP work must sit on the cooperative kernel (Phase 3.0, cuBLAS speed).
- Micro-batching is at best 1.26x ideal at M=4k (paper) and 1.35-1.38x on the 70B shapes; TP2 with plain NCCL is 1.08-1.19x on the 70B FFN. The 2-GPU pipeline-parallel comparison (paper Fig 5) is therefore against a baseline that strong TP2 already beats by 20-40 %; see results/tp_strong.md for the strong TP numbers that gate the decision.
- Numbers were taken while the 4-GPU strong-TP sweep ran on four other GPUs of the same node; 1-GPU cuBLAS at M=16384 = 6.3 ms = 690 TFLOP/s, so no visible contention.
