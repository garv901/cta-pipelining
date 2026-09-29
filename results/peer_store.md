# Part 1a: smem-staged vectorized epilogue

nvidia-smi before run:
```
index, utilization.gpu [%], memory.used [MiB]
0, 0 %, 46109 MiB
1, 0 %, 5 MiB
2, 0 %, 74709 MiB
3, 0 %, 43785 MiB
```

## Local vs peer D (kernel on GPU 3, D on GPU 0 for peer)
| M | cfg | config | local us | local TFLOP/s | peer us | peer/local | peer write GB/s | bit-equal to cfg2 |
|---|---|---|---|---|---|---|---|---|
| 4096 | 2 | nosmem-epi tile 64x256x64 | 1109 | 496 | 3956 | 3.567 | 17 | True |
| 4096 | 1 | nosmem-epi tile 128x128x64 | 1178 | 467 | 3923 | 3.330 | 17 | True |
| 4096 | 4 | smem-epi tile 64x256x64 | 970 | 566 | 1256 | 1.294 | 53 | True |
| 4096 | 5 | smem-epi tile 128x128x64 | 1071 | 513 | 1293 | 1.207 | 52 | True |
| 16384 | 2 | nosmem-epi tile 64x256x64 | 4668 | 471 | 15585 | 3.339 | 17 | True |
| 16384 | 1 | nosmem-epi tile 128x128x64 | 5937 | 370 | 15526 | 2.615 | 17 | True |
| 16384 | 4 | smem-epi tile 64x256x64 | 4538 | 485 | 5060 | 1.115 | 53 | True |
| 16384 | 5 | smem-epi tile 128x128x64 | 5963 | 369 | 5836 | 0.979 | 46 | True |
