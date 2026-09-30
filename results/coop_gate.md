# Phase 3.0: stock CUTLASS cooperative persistent GEMM

nvidia-smi before 1-GPU run:
```
index, utilization.gpu [%], memory.used [MiB]
0, 75 %, 77537 MiB
1, 0 %, 5 MiB
2, 100 %, 74289 MiB
3, 0 %, 567 MiB
gpu_uuid, pid, used_gpu_memory [MiB]
GPU-e82ab913-2d6b-1a55-89a8-9ad23a5a09cf, 2069192, 77002 MiB
GPU-e82ab913-2d6b-1a55-89a8-9ad23a5a09cf, 2680904, 518 MiB
GPU-f7c1496c-b7f5-b1d2-78ea-68826f778059, 2198486, 74278 MiB
GPU-cfd9ce0a-b7af-895a-f54c-4f175b610977, 2680904, 518 MiB
```

## Config info

- cfg0: coop-persistent tile 128x256x64 threads=384 smem=214016 stages=4 grid(M=16384)=1x132x1
- cfg1: coop-persistent tile 128x128x64 threads=384 smem=214016 stages=6 grid(M=16384)=1x132x1
- cfg2: coop-persistent tile 256x128x64 threads=384 smem=214016 stages=4 grid(M=16384)=1x132x1

## Sanity (M=4096, raster AlongN, swizzle 1, output pre-filled with NaN; allclose rtol=atol=2e-2 vs torch.mm)

| cfg | max abs diff | NaN | allclose |
|---|---|---|---|
| 128x256x64 | 0 | False | True |
| 128x128x64 | 0 | False | True |
| 256x128x64 | 0 | False | True |

## Throughput, N=K=8192 (median us / TFLOP/s; best over raster x swizzle per config)

| M | cuBLAS | cfg0 | cfg1 | cfg2 | best overall | best/cuBLAS speed ratio |
|---|---|---|---|---|---|---|
| 1024 | 174 / 789 | 174 / 788 (M8) | 183 / 753 (N1) | 209 / 658 (N1) | cfg0 AlongM sw8 | 0.998 |
| 2048 | 412 / 668 | 408 / 674 (N2) | 416 / 661 (N4) | 410 / 670 (M8) | cfg0 AlongN sw2 | 1.010 |
| 4096 | 799 / 688 | 785 / 700 (N4) | 854 / 644 (N2) | 847 / 649 (N8) | cfg0 AlongN sw4 | 1.018 |
| 8192 | 1717 / 640 | 1611 / 683 (N1) | 1724 / 638 (N8) | 1671 / 658 (M8) | cfg0 AlongN sw1 | 1.066 |
| 16384 | 2886 / 762 | 3318 / 663 (N8) | 3361 / 654 (M8) | 3215 / 684 (M8) | cfg2 AlongM sw8 | 0.898 |
| 32768 | 6280 / 700 | 6657 / 661 (N8) | 6730 / 653 (N8) | 6541 / 672 (M4) | cfg2 AlongM sw4 | 0.960 |

Cell suffix: raster (N/M) + swizzle.

## Cross-check M=16384 with ctapp.timing.measure (gated; median, p10, p90 us)

- cuBLAS [3312.1440410614014, 3292.9567098617554, 3340.947198867798]
- coop [3363.744020462036, 3323.151922225952, 3367.875123023987] (cfg2 raster 2 swizzle 8); event-timed: cuBLAS 2886, coop 3215

## Gate G1: best/cuBLAS >= 0.85 at M >= 4096 (target 0.90)

M=4096: 1.018, M=8192: 1.066, M=16384: 0.898, M=32768: 0.960 -> PASS (target 0.90 not met)

nvidia-smi before peer run:
```
index, utilization.gpu [%], memory.used [MiB]
0, 87 %, 78641 MiB
1, 0 %, 601 MiB
2, 0 %, 1125 MiB
3, 0 %, 567 MiB
gpu_uuid, pid, used_gpu_memory [MiB]
GPU-e82ab913-2d6b-1a55-89a8-9ad23a5a09cf, 2069192, 78630 MiB
GPU-4eac38f2-1a70-662a-f6f4-9b62d1778ef4, 4055128, 590 MiB
GPU-f7c1496c-b7f5-b1d2-78ea-68826f778059, 4059665, 590 MiB
GPU-f7c1496c-b7f5-b1d2-78ea-68826f778059, 4065195, 518 MiB
GPU-cfd9ce0a-b7af-895a-f54c-4f175b610977, 4065195, 518 MiB
```


## Peer D (kernel on physical GPU 3, D on physical GPU 2); link bound = M*8192*2 B / 124.1 GB/s (SM-store P2P, measured on this pair)

| M | cfg/raster/swizzle | local us | peer us | link bound us | peer/local | peer / max(local, link) | within 15% | D equal |
|---|---|---|---|---|---|---|---|---|
| 1024 | cfg0 AlongM sw8 | 174 | 297 | 135 | 1.706 | 1.706 | NO | True |
| 1024 | cfg1 AlongN sw1 | 182 | 296 | 135 | 1.629 | 1.629 | NO | True |
| 2048 | cfg0 AlongN sw2 | 346 | 527 | 270 | 1.524 | 1.524 | NO | True |
| 2048 | cfg2 AlongM sw8 | 349 | 540 | 270 | 1.549 | 1.549 | NO | True |
| 4096 | cfg0 AlongN sw4 | 722 | 1026 | 541 | 1.421 | 1.421 | NO | True |
| 4096 | cfg2 AlongN sw8 | 795 | 1040 | 541 | 1.309 | 1.309 | NO | True |
| 8192 | cfg0 AlongN sw1 | 1563 | 1913 | 1081 | 1.224 | 1.224 | NO | True |
| 8192 | cfg2 AlongM sw8 | 1638 | 1952 | 1081 | 1.191 | 1.191 | NO | True |
| 16384 | cfg2 AlongM sw8 | 3250 | 4044 | 2162 | 1.244 | 1.244 | NO | True |
| 16384 | cfg0 AlongN sw8 | 3223 | 3992 | 2162 | 1.238 | 1.238 | NO | True |
| 32768 | cfg2 AlongM sw4 | 6595 | 8445 | 4325 | 1.281 | 1.281 | NO | True |
| 32768 | cfg0 AlongN sw8 | 6301 | 7919 | 4325 | 1.257 | 1.257 | NO | True |
