# Phase 0 gate

nvidia-smi before run:
```
index, utilization.gpu [%], memory.used [MiB]
0, 0 %, 529 MiB
1, 100 %, 74289 MiB
2, 98 %, 74709 MiB
3, 0 %, 567 MiB
```
CUTLASS: 0b55a2f691d69981583568fd9eb69687b1f0de8a #define CUTLASS_MAJOR 4 #define CUTLASS_MINOR 8 #define CUTLASS_PATCH 0

## Correctness (M=1024, vs F.linear; rel = |d|/max(|ref|,1e-2))
| config | max abs err | max rel err |
|---|---|---|
| tile 64x128x64 threads=128 smem=221328 stages=9 | 0 | 0 |
| tile 128x128x64 threads=128 smem=229488 stages=7 | 0 | 0 |
| tile 64x256x64 threads=128 smem=204880 stages=5 | 0 | 0 |
| tile 128x256x64 threads=128 smem=196672 stages=4 | 0 | 0 |

## Throughput, N=K=8192 (median us / TFLOP/s); cuBLAS = torch.mm(X, W.t(), out=Y)
| M | cuBLAS | cfg0 64x128x64 | cfg1 128x128x64 | cfg2 64x256x64 | cfg3 128x256x64 | best cfg | best/cuBLAS speed ratio |
|---|---|---|---|---|---|---|---|
| 1024 | 174 / 790 | 361 / 381 | 265 / 519 | 256 / 537 | 3057 / 45 | 2 | 0.680 |
| 2048 | 344 / 800 | 731 / 376 | 529 / 519 | 516 / 532 | 6151 / 45 | 2 | 0.666 |
| 4096 | 685 / 803 | 1507 / 365 | 1175 / 468 | 1040 / 529 | 12188 / 45 | 2 | 0.659 |
| 8192 | 1417 / 776 | 3169 / 347 | 2845 / 386 | 2301 / 478 | 23831 / 46 | 2 | 0.616 |
| 16384 | 3082 / 714 | 6393 / 344 | 5942 / 370 | 4615 / 476 | 47482 / 46 | 2 | 0.668 |
| 32768 | 6272 / 701 | 12761 / 345 | 11909 / 369 | 9667 / 455 | 94005 / 47 | 2 | 0.649 |

## Remote-store cost (best config, kernel on GPU 3)
| M | cfg | local us | peer us | peer/local | local write GB/s | peer write GB/s |
|---|---|---|---|---|---|---|
| 4096 | 2 | 1095 | 3936 | 3.595 | 61 | 17 |
| 16384 | 2 | 4938 | 15553 | 3.150 | 54 | 17 |

## P2P bandwidth, 256 MiB (GB/s)
| direction | copy engine | SM stores (16B) |
|---|---|---|
| 3->0 | 132.2 | 124.2 |
| 0->3 | 132.2 | 124.2 |
## Summary
- KernelTma best config (64x256x64) reaches 0.62-0.68x of cuBLAS speed (455-537 vs 701-803 TFLOP/s); 128x256x64 runs at ~45 TFLOP/s (ptxas: wgmma serialized, 1 consumer warpgroup with 256 acc regs/thread).
- Kernel with D in peer memory is 3.6x (M=4096) and 3.1x (M=16384) slower than local D; implied peer write rate is 17 GB/s vs 54-61 GB/s local.
- P2P bandwidth at 256 MiB: 132 GB/s copy engine, 124 GB/s SM 16-byte stores, same in both directions.
