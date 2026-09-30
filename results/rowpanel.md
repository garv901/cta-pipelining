# Phase 2c: row-panel scoreboard vs tile scoreboard (2x H100, producer = physical GPU 3, consumer = physical GPU 0)

Same gated harness as results/fig5_h100.md (median of 20 reps after 5 warmup, us). CTAPP = cfg/cfg group variant; 'tile' = old per-consumer-tile scoreboard, 'rowpanel' = new per-consumer-row-panel scoreboard. Best CTAPP variant picked per mode per M. Drift check: same-session baseline values vs results/fig5_h100.csv (old value in parentheses). Ideal = 1-GPU cuBLAS / 2. Reductions = 1 - CTAPP_rowpanel / baseline (positive = CTAPP faster).

| M | CTAPP tile (old) | CTAPP rowpanel | MB cuBLAS best | TP2 | ideal | red. vs MB % | red. vs TP % |
|---|---|---|---|---|---|---|---|
| 1024 | 493 (cfg7/7 group1) | 476 (cfg7/7 group1) | 336 (chunk=256; old 336) | 368 (old 369) | 178 (old 177) | -41.7 | -29.3 |
| 2048 | 839 (cfg7/7 group1) | 786 (cfg7/7 group1) | 548 (chunk=512; old 548) | 700 (old 702) | 346 (old 347) | -43.6 | -12.3 |
| 4096 | 1455 (cfg7/7 group1) | 1415 (cfg7/7 group8) | 908 (chunk=512; old 907) | 1348 (old 1349) | 694 (old 688) | -55.9 | -5.0 |
| 8192 | 2476 (cfg7/7 group8) | 2433 (cfg7/7 group8) | 1626 (chunk=512; old 1627) | 2649 (old 2652) | 1398 (old 1459) | -49.6 | +8.2 |
| 16384 | 4571 (cfg7/7 group8) | 4519 (cfg7/7 group8) | 3086 (chunk=512; old 3086) | 5191 (old 5191) | 3163 (old 3220) | -46.4 | +12.9 |
| 32768 | 8776 (cfg7/7 group8) | 8668 (cfg7/7 group8) | 5959 (chunk=1024; old 5968) | 10200 (old 10204) | 6384 (old 6333) | -45.5 | +15.0 |

## All CTAPP variants (median us)

| M | variant | tile | rowpanel | delta (rowpanel - tile) |
|---|---|---|---|---|
| 1024 | cfg7/7 group8 | 593 | 577 | -17 |
| 1024 | cfg7/7 group1 | 493 | 476 | -17 |
| 1024 | cfg4/4 group8 | 594 | 605 | +11 |
| 1024 | cfg6/6 group8 | 565 | 566 | +1 |
| 2048 | cfg7/7 group8 | 906 | 866 | -40 |
| 2048 | cfg7/7 group1 | 839 | 786 | -53 |
| 2048 | cfg4/4 group8 | 1148 | 1177 | +29 |
| 2048 | cfg6/6 group8 | 910 | 916 | +7 |
| 4096 | cfg7/7 group8 | 1492 | 1415 | -77 |
| 4096 | cfg7/7 group1 | 1455 | 1429 | -26 |
| 4096 | cfg4/4 group8 | 2372 | 2418 | +45 |
| 4096 | cfg6/6 group8 | 1637 | 1635 | -2 |
| 8192 | cfg7/7 group8 | 2476 | 2433 | -43 |
| 8192 | cfg7/7 group1 | 2773 | 2688 | -85 |
| 8192 | cfg4/4 group8 | 4974 | 5061 | +87 |
| 8192 | cfg6/6 group8 | 3077 | 3068 | -8 |
| 16384 | cfg7/7 group8 | 4571 | 4519 | -52 |
| 16384 | cfg7/7 group1 | 5800 | 5590 | -210 |
| 16384 | cfg4/4 group8 | 10437 | 10502 | +65 |
| 16384 | cfg6/6 group8 | 5872 | 5902 | +30 |
| 32768 | cfg7/7 group8 | 8776 | 8668 | -108 |
| 32768 | cfg7/7 group1 | 11632 | 11767 | +134 |
| 32768 | cfg4/4 group8 | 21346 | 21415 | +70 |
| 32768 | cfg6/6 group8 | 11568 | 11700 | +132 |

## Environment

Per-M nvidia-smi before/after and foreign-process check: build/rowpanel_smi.txt (every M clean on the first attempt, no foreign compute process on physical GPU 0/3 after each sweep).
