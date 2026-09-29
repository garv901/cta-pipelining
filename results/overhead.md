# Phase 1c: protocol overhead

nvidia-smi before run:
```
index, utilization.gpu [%], memory.used [MiB]
0, 1 %, 529 MiB
1, 0 %, 5 MiB
2, 100 %, 74709 MiB
3, 0 %, 567 MiB
```
nvidia-smi after run:
```
index, utilization.gpu [%], memory.used [MiB]
0, 87 %, 2679 MiB
1, 0 %, 5 MiB
2, 0 %, 74709 MiB
3, 0 %, 1759 MiB
```

## Occupancy (ctapp kernel)
| cfg | config | CTAs/SM |
|---|---|---|
| 4 | smem-epi tile 64x256x64 threads=128 smem=204880 stages=5 | 1 |
| 6 | smem-epi tile 64x256x64 threads=128 smem=122928 stages=3 | 1 |
| 8 | smem-epi tile 64x256x32 threads=128 smem=102480 stages=5 | 2 |
| 5 | smem-epi tile 128x128x64 threads=128 smem=229488 stages=7 | 1 |
| 7 | smem-epi tile 128x128x64 threads=128 smem=98352 stages=3 | 2 |

## Producer alone, no protocol
| M | cfg | waves | local D us | TFLOP/s | peer D us | TFLOP/s |
|---|---|---|---|---|---|---|
| 4096 | 64x256x64 auto | 15.5 | 972 | 566 | 1260 | 436 |
| 4096 | 64x256x64 s3 | 15.5 | 1208 | 455 | 1343 | 409 |
| 4096 | 64x256x32 s5 | 7.8 | 1367 | 402 | 1417 | 388 |
| 4096 | 128x128x64 auto | 15.5 | 1060 | 519 | 1266 | 434 |
| 4096 | 128x128x64 s3 | 7.8 | 939 | 585 | 1165 | 472 |
| 16384 | 64x256x64 auto | 62.1 | 4751 | 463 | 5063 | 434 |
| 16384 | 64x256x64 s3 | 62.1 | 5680 | 387 | 5416 | 406 |
| 16384 | 64x256x32 s5 | 31.0 | 5463 | 403 | 5526 | 398 |
| 16384 | 128x128x64 auto | 62.1 | 5956 | 369 | 5818 | 378 |
| 16384 | 128x128x64 s3 | 31.0 | 4631 | 475 | 5869 | 375 |

## Producer alone with signalling (consumer not launched). ovh/wave = (t_signal - t_peer_noprotocol) / waves, us; ovh/wave(q) = relative to 'queue only' (same kernel and same tile order, no signalling)
| M | cfg | order | queue-only us | current us | ovh/wave | ovh/wave(q) | F1 us | ovh/wave | ovh/wave(q) | F0 (unsafe) us | ovh/wave | ovh/wave(q) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4096 | 64x256x64 auto | row-major | 2443 | 2390 | 72.9 | -3.4 | 2505 | 80.3 | 4.0 | 2480 | 78.7 | 2.4 |
| 4096 | 64x256x64 auto | grouped8 | 1280 | 1316 | 3.6 | 2.4 | 1306 | 3.0 | 1.7 | 1301 | 2.7 | 1.4 |
| 4096 | 64x256x64 s3 | row-major | 1922 | 2037 | 44.7 | 7.4 | 2101 | 48.9 | 11.6 | 2084 | 47.8 | 10.5 |
| 4096 | 64x256x64 s3 | grouped8 | 1196 | 1419 | 4.9 | 14.3 | 1275 | -4.4 | 5.1 | 1252 | -5.9 | 3.6 |
| 4096 | 64x256x32 s5 | row-major | 2091 | 2091 | 86.9 | -0.0 | 2087 | 86.3 | -0.5 | 2055 | 82.3 | -4.6 |
| 4096 | 64x256x32 s5 | grouped8 | 1470 | 1504 | 11.2 | 4.4 | 1508 | 11.7 | 4.9 | 1504 | 11.1 | 4.3 |
| 4096 | 128x128x64 auto | row-major | 1432 | 1549 | 18.2 | 7.5 | 1457 | 12.3 | 1.6 | 1436 | 10.9 | 0.2 |
| 4096 | 128x128x64 auto | grouped8 | 1062 | 1419 | 9.8 | 23.0 | 1147 | -7.7 | 5.5 | 1122 | -9.3 | 3.9 |
| 4096 | 128x128x64 s3 | row-major | 1315 | 1432 | 34.4 | 15.1 | 1349 | 23.7 | 4.4 | 1248 | 10.7 | -8.6 |
| 4096 | 128x128x64 s3 | grouped8 | 1094 | 1341 | 22.7 | 31.8 | 1218 | 6.8 | 15.9 | 1126 | -5.0 | 4.0 |
| 16384 | 64x256x64 auto | row-major | 10587 | 10659 | 90.2 | 1.2 | 10751 | 91.6 | 2.6 | 10702 | 90.8 | 1.8 |
| 16384 | 64x256x64 auto | grouped8 | 5156 | 5223 | 2.6 | 1.1 | 5216 | 2.5 | 1.0 | 5178 | 1.8 | 0.4 |
| 16384 | 64x256x64 s3 | row-major | 7606 | 8951 | 57.0 | 21.7 | 8550 | 50.5 | 15.2 | 8531 | 50.2 | 14.9 |
| 16384 | 64x256x64 s3 | grouped8 | 5326 | 5320 | -1.5 | -0.1 | 4911 | -8.1 | -6.7 | 4834 | -9.4 | -7.9 |
| 16384 | 64x256x32 s5 | row-major | 10172 | 10207 | 150.9 | 1.1 | 10133 | 148.5 | -1.3 | 10152 | 149.1 | -0.7 |
| 16384 | 64x256x32 s5 | grouped8 | 5746 | 6006 | 15.5 | 8.4 | 5953 | 13.8 | 6.7 | 5721 | 6.3 | -0.8 |
| 16384 | 128x128x64 auto | row-major | 5750 | 5895 | 1.2 | 2.3 | 5783 | -0.6 | 0.5 | 5772 | -0.7 | 0.4 |
| 16384 | 128x128x64 auto | grouped8 | 4192 | 5272 | -8.8 | 17.4 | 4386 | -23.1 | 3.1 | 4321 | -24.1 | 2.1 |
| 16384 | 128x128x64 s3 | row-major | 5668 | 5688 | -5.8 | 0.6 | 5699 | -5.5 | 1.0 | 5549 | -10.3 | -3.9 |
| 16384 | 128x128x64 s3 | grouped8 | 4165 | 4588 | -41.3 | 13.6 | 4398 | -47.4 | 7.5 | 4271 | -51.5 | 3.4 |

## 2-layer pipeline, host-timed median of 10, ms (bit-equal = vs same-kernel sequential)
| M | cfg | order | pipe F-current | bit-equal | pipe F1 | bit-equal | sequential (same kernels) | cuBLAS sequential |
|---|---|---|---|---|---|---|---|---|
| 4096 | 64x256x64 auto | row-major | 2.70 | True | 2.81 | True | 2.67 | 1.91 |
| 4096 | 64x256x64 auto | grouped8 | 2.59 | True | 2.67 | True | 2.67 | 1.91 |
| 4096 | 64x256x64 s3 | row-major | 2.31 | True | 2.38 | True | 2.88 | 1.90 |
| 4096 | 64x256x64 s3 | grouped8 | 1.78 | True | 1.72 | True | 2.88 | 1.90 |
| 4096 | 64x256x32 s5 | row-major | 2.52 | True | 2.51 | True | 3.26 | 1.91 |
| 4096 | 64x256x32 s5 | grouped8 | 2.19 | True | 2.17 | True | 3.26 | 1.91 |
| 4096 | 128x128x64 auto | row-major | 1.64 | True | 1.65 | True | 2.65 | 1.92 |
| 4096 | 128x128x64 auto | grouped8 | 1.84 | True | 1.77 | True | 2.65 | 1.92 |
| 4096 | 128x128x64 s3 | row-major | 1.55 | True | 1.53 | True | 2.39 | 1.91 |
| 4096 | 128x128x64 s3 | grouped8 | 1.63 | True | 1.55 | True | 2.39 | 1.91 |
| 16384 | 64x256x64 auto | row-major | 11.12 | True | 11.26 | True | 11.21 | 7.62 |
| 16384 | 64x256x64 auto | grouped8 | 11.00 | True | 11.13 | True | 11.21 | 7.62 |
| 16384 | 64x256x64 s3 | row-major | 9.97 | True | 8.94 | True | 13.26 | 7.62 |
| 16384 | 64x256x64 s3 | grouped8 | 6.13 | True | 6.06 | True | 13.26 | 7.62 |
| 16384 | 64x256x32 s5 | row-major | 10.89 | True | 10.81 | True | 12.96 | 7.65 |
| 16384 | 64x256x32 s5 | grouped8 | 10.38 | True | 10.37 | True | 12.96 | 7.65 |
| 16384 | 128x128x64 auto | row-major | 5.93 | True | 6.15 | True | 13.97 | 7.65 |
| 16384 | 128x128x64 auto | grouped8 | 6.17 | True | 6.18 | True | 13.97 | 7.65 |
| 16384 | 128x128x64 s3 | row-major | 5.44 | True | 5.89 | True | 11.10 | 7.63 |
| 16384 | 128x128x64 s3 | grouped8 | 4.86 | True | 4.69 | True | 11.10 | 7.63 |

## Notes (hand-written, not regenerated by bench/overhead.py)
- Config ids: 4 = 64x256x64 auto (5 stages), 6 = 64x256x64 3 stages, 8 = 64x256x32 5 stages, 5 = 128x128x64 auto (7 stages), 7 = 128x128x64 3 stages. 64x256x64 with 2 stages is impossible: KernelTma static_asserts stages >= 3 (K_PIPE_MMAS < K_PIPE_MAX - 1); 3 stages = 123 KB, only 1 CTA/SM. 64x256x32 (BK=32) with 5 stages = 102 KB gets 2 CTAs/SM.
- "waves" = tiles / (132 * CTAs per SM).
- Main finding: the Phase 1 "overhead" is the row-major producer tile order, not signalling. The queue-only ctapp kernel (no dependency signalling at all) already takes 10.6 ms vs 5.07 ms for gemm_tma with peer D (M=16384, cfg4). Fences F1/F0 change nothing measurable for cfg 4/5/8 (ovh/wave(q) is within +-5 us of zero).
- Tile-order experiment (build/order_exp.py, queue-only, M=16384, peer D): cfg4 row-major 10580 us, m-fastest 5067 us, groups of 8 tile-rows n-major inside a group 5165 us; cfg8: 10111 / 5487 / 5710 us. Row-major makes each wave (~4 tile rows x all 32 n-tiles) stream all of W1 (128 MB, > L2) from HBM. `group_rows=8` in Pipeline keeps row panels completing in groups of 8 tile rows.
- Remaining ovh/wave(q) for cfg 6 with row-major is ~15-22 us/wave, and ~0 with grouped8, so the drain/fence latency is small once the order is L2-friendly (fence cost itself: F1 vs F0 differences are within noise, 3-10 us per wave at most).
- The sequential baseline uses gemm_tma with CUTLASS's default rasterization (not the grouped order), so for 128x128 it is slower than necessary (5.97 ms local vs 4.19 ms queue-only grouped8 for one GEMM at M=16384).
- The GPUs were shared: the first two runs of this benchmark were discarded because other users' jobs were running on physical GPUs 0/3 (nvidia-smi showed 99-100% util, 64-77 GB used). The run above started with GPUs 0/3 essentially idle, but util on GPU 0 was 87% after the run (someone else started), so a few rows may be noisy (e.g. negative overheads).
