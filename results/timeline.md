# Phase 2b: small-M timeline of CTA-pipelining (cfg7/7 = 128x128x64, fence=1, 2 CTAs/SM = 264 slots)

Producer = physical GPU 3 (A), consumer = physical GPU 0 (B). Times in us relative to the gate stamp on A (1-thread kernel right after the gate wait); B stamps mapped to A's clock with the measured offset. Median rep of 10 (5 warmup), gated harness identical to fig5.

## Clock and timer

| item | value |
|---|---|
| globaltimer tick (20000 consecutive reads, dev A and B) | dev0: min nonzero delta 32 ns, median 32 ns, mode 32 ns, 66% of consecutive reads identical; dev1: min nonzero delta 32 ns, median 32 ns, mode 32 ns, 66% of consecutive reads identical |
| A-to-B globaltimer offset (tB - tA), start / end of run | -16.070226 s / -16.070201 s (drift 24.9 us over the run; std of the 5 trials 31 / 36 ns) |
| ping-pong RTT via peer flags (200 round trips x 5) | min 3.65 us, median 3.84 us (one-way flag latency about 1.8 us) |
| harness stamps-on vs stamps-off (same session) | M=1024 g1: 470 vs 466, M=1024 g8: 581 vs 574, M=4096 g1: 1417 vs 1403, M=4096 g8: 1444 vs 1430 us |

## Decomposition (pipelined run)

| metric (us) | M=1024 g1 | M=1024 g8 | M=4096 g1 | M=4096 g8 |
|---|---|---|---|---|
| harness latency (stamps on) | 470 | 581 | 1417 | 1444 |
| **1. start skew**: B gate stamp minus A | -1.7 | 4.4 | 4.6 | 2.7 |
| first CTA entry B minus first CTA entry A | -1.7 | 3.6 | 4.8 | 2.2 |
| **2. producer**: first CTA entry | 2.1 | 2.9 | 2.1 | 2.8 |
| last producer k4 (all signalled) | 343.3 | 334.4 | 1253.4 | 1160.3 |
| producer span (first entry to last k4) | 341.2 | 331.5 | 1251.3 | 1157.4 |
| per-CTA mainloop k2-k1 (mean) | 102.9 | 99.6 | 113.0 | 109.3 |
| per-CTA store k3-k2 (mean) | 24.7 | 23.3 | 18.4 | 12.8 |
| per-CTA signal k4-k3 (mean) | 29.3 | 28.6 | 24.2 | 18.2 |
| tiles / slots / waves; max CTAs alive | 512 / 264 / 2; 264 | 512 / 264 / 2; 264 | 2048 / 264 / 8; 264 | 2048 / 264 / 8; 264 |
| **3. ramp**: first row panel complete (min over rows) | 187.2 | 333.5 | 188.0 | 371.6 |
| row 0 complete | 187.2 | 333.6 | 188.4 | 371.6 |
| first consumer tile acquired (min k1 on B) | 188.0 | 339.5 | 194.7 | 379.4 |
| signal latency, row ready to acquire, waiting CTAs (median / max) | 1.0 / 2.2 | 6.1 / 6.3 | 7.3 / 12.8 | 7.6 / 9.5 |
| **4. consumer**: total wait sum(k1-k0), CTA-us | 61121 | 88186 | 128720 | 127840 |
| wait p10/p50/p90/max per CTA | 38/187/190/340 | 1/333/333/333 | 1/59/188/371 | 1/1/374/376 |
| mainloop per tile, pipelined (mean / median) | 105.2 / 104.2 | 106.7 / 107.3 | 108.1 / 108.1 | 113.6 / 113.1 |
| mainloop per tile, standalone (a) (mean / median) | 104.0 / 105.0 | 104.0 / 105.0 | 112.2 / 113.2 | 112.2 / 113.2 |
| consumer store per tile (pipelined / standalone) | 5.9 / 5.9 | 5.9 / 5.9 | 5.7 / 5.8 | 5.8 / 5.8 |
| **5. tail**: last consumer k4 minus last producer k4 | 113.9 | 240.1 | 163.6 | 282.7 |
| consumer work done after producer's last k4, CTA-us / 264 slots | 105.7 | 339.2 | 137.2 | 409.6 |
| **6. end-to-end from stamps** (last consumer k4) | 457.2 | 574.5 | 1417.0 | 1442.9 |
| harness minus stamps e2e (join, event/gate offsets) | 13.3 | 6.7 | -0.2 | 0.9 |

Standalone references: (a) consumer alone on B, pre-filled queue, no deps: span M=1024: 232, M=4096: 965 us (harness 239, 972). (b) producer alone on A with signalling: harness M=1024 g1: 347, M=1024 g8: 336, M=4096 g1: 1251, M=4096 g8: 1160 us.

## Time budget (harness = sum of parts, us)

| component | M=1024 g1 | M=1024 g8 | M=4096 g1 | M=4096 g8 |
|---|---|---|---|---|
| start (gate to first producer CTA) | 2.1 | 2.9 | 2.1 | 2.8 |
| compute-equivalent: standalone consumer span (a) | 232.4 | 232.4 | 965.2 | 965.2 |
| producer: signalling excess (span with local Y1 + signalling, minus (a)) | 59.1 | 47.3 | 253.6 | 177.4 |
| producer: remote-store excess (span minus span with local Y1) | 49.8 | 51.9 | 32.4 | 14.8 |
| tail after producer end | 113.9 | 240.1 | 163.6 | 282.7 |
| join / unexplained (harness - stamps e2e) | 13.3 | 6.7 | -0.2 | 0.9 |
| **sum = harness** | 470.5 | 581.2 | 1416.8 | 1443.8 |

(Local-Y1 spans come from the `localy1` knob run, a separate run of the producer alone, so the split between the two producer excess rows has run-to-run noise of a few us.)

## Knob experiments (harness us; span = producer first entry to last k4)

| run | M=1024 g1 | M=1024 g8 | M=4096 g1 | M=4096 g8 |
|---|---|---|---|---|
| baseline: pipe / producer alone | 470 / 347 | 581 / 336 | 1417 / 1251 | 1444 / 1160 |
| nofence (fence=2, UNSAFE, measurement only): pipe / producer alone | 463 / 338 | 569 / 330 | 1418 / 1241 | 1372 / 1109 |
| nofence: per-CTA signal | 22.1 | 22.6 | 14.9 | 11.8 |
| localy1 (producer alone, Y1 stored locally, signalling kept): harness | 299 | 287 | 1226 | 1150 |
| localy1: per-CTA store / signal | 6.2 / 34.4 | 5.9 / 36.7 | 6.0 / 34.4 | 5.8 / 31.4 |
| g24 (producer groups of 2 and 4 tile-rows): pipe harness, M=1024 g2 / g4; M=4096 g2 / g4 | 492 / 480 | | 1418 / 1382 | |

## Signal time vs number of producer CTAs signalling at the same moment

Per producer CTA: concurrency = number of CTAs whose [k3,k4] window contains the midpoint of this CTA's window. Median k4-k3 (us), producer alone, baseline run and localy1 run (no NVLink data stores).

| case | run | 1-10 | 11-30 | 31-80 | 81+ |
|---|---|---|---|---|---|
| M=1024 g1 | base | 6.1 (n=55) | 7.5 (n=18) | 28.2 (n=52) | 31.3 (n=387) |
| M=1024 g1 | localy1 | 6.4 (n=2) | 4.8 (n=10) | 6.4 (n=2) | 35.8 (n=498) |
| M=1024 g1 | nofence | 3.4 (n=54) | 5.6 (n=23) | 17.8 (n=110) | 23.6 (n=325) |
| M=4096 g1 | base | 7.9 (n=265) | 7.4 (n=245) | 17.4 (n=187) | 28.4 (n=1351) |
| M=4096 g1 | localy1 | 6.1 (n=178) | 7.0 (n=178) | 9.1 (n=56) | 43.6 (n=1636) |
| M=4096 g1 | nofence | 4.7 (n=557) | 5.4 (n=161) | 14.1 (n=659) | 22.6 (n=671) |
| M=4096 g8 | base | 8.0 (n=201) | 8.0 (n=537) | 13.9 (n=448) | 30.4 (n=862) |
| M=4096 g8 | localy1 | 6.0 (n=69) | 4.9 (n=46) | 10.3 (n=304) | 38.6 (n=1629) |
| M=4096 g8 | nofence | 5.2 (n=566) | 5.1 (n=645) | 12.6 (n=359) | 25.2 (n=478) |

## Environment

nvidia-smi before the base run:
```
gpu_uuid, pid, used_gpu_memory [MiB]
GPU-e82ab913-2d6b-1a55-89a8-9ad23a5a09cf, 1912379, 518 MiB
GPU-f7c1496c-b7f5-b1d2-78ea-68826f778059, 3891903, 74698 MiB
GPU-cfd9ce0a-b7af-895a-f54c-4f175b610977, 1912379, 518 MiB
index, uuid, utilization.gpu [%], memory.used [MiB]
0, GPU-e82ab913-2d6b-1a55-89a8-9ad23a5a09cf, 0 %, 529 MiB
1, GPU-4eac38f2-1a70-662a-f6f4-9b62d1778ef4, 0 %, 5 MiB
2, GPU-f7c1496c-b7f5-b1d2-78ea-68826f778059, 98 %, 74709 MiB
3, GPU-cfd9ce0a-b7af-895a-f54c-4f175b610977, 0 %, 567 MiB
```
after:
```
gpu_uuid, pid, used_gpu_memory [MiB]
GPU-e82ab913-2d6b-1a55-89a8-9ad23a5a09cf, 1912379, 1672 MiB
GPU-f7c1496c-b7f5-b1d2-78ea-68826f778059, 3891903, 74698 MiB
GPU-cfd9ce0a-b7af-895a-f54c-4f175b610977, 1912379, 1148 MiB
index, uuid, utilization.gpu [%], memory.used [MiB]
0, GPU-e82ab913-2d6b-1a55-89a8-9ad23a5a09cf, 0 %, 1683 MiB
1, GPU-4eac38f2-1a70-662a-f6f4-9b62d1778ef4, 0 %, 5 MiB
2, GPU-f7c1496c-b7f5-b1d2-78ea-68826f778059, 100 %, 74709 MiB
3, GPU-cfd9ce0a-b7af-895a-f54c-4f175b610977, 0 %, 1197 MiB
```

Notes: the `localy1` knob only changes the producer-alone runs (its pipe rows are unchanged code and not used). Monitoring: a 1 s nvidia-smi sampler during each run showed only this job on GPUs 0/3, except brief pids on GPU 3 at start-up (see build/tl_*.log.mon); one localy1 run with such a pid present gave a producer-alone time of 266 us (M=1024 g1) and 1073 us (M=4096 g1) instead of 299 / 1226 in the three other runs, so treat localy1 as +-10% noise-prone.

## Diagnosis

| hypothesis | verdict | numbers |
|---|---|---|
| H1 start / launch skew | no | B gate stamp vs A: -1.7 to +4.6 us; first CTA entry skew the same; consumer is idle until the first row panel is done (>= 187 us) anyway. Not on the critical path. |
| H2 ramp (first row panel late) | not the cause of the excess | first row panel completes at 187 us (one producer wave incl. store + signal), consumer acquires it 1 us later (median signal latency 1 us at g1). The consumer only needs 232 us of work against a 341 us producer, so B is starved, not backlogged; the ramp does not extend the end time. At g8 the first row is done at 333 us (M=1024) / 372 us (M=4096), which is the reason g8 is worse at small M. |
| H3 consumer slower while overlapped | no | mainloop per tile pipelined vs standalone: 105 vs 104 (M=1024 g1), 107 vs 104 (g8), 108 vs 112 and 114 vs 112 (M=4096). Store 5.9 vs 5.9 us. |
| H4 producer slowed by signalling and remote stores | YES, main cause on the producer | per-CTA store + signal = 54 us at M=1024 (vs 6 us store on the consumer) and 43 us at M=4096. Producer span is 341 us vs 232 us for the same kernel with no protocol at M=1024, and 1251 vs 965 us at M=4096. |
| H5 tail after producer ends | YES, second cause | 114 us (M=1024 g1) = one full consumer wave (mainloop 105 + store 6): all rows of the last producer wave complete within about 2 us of each other, so 256 consumer tiles start together at the very end. 164 us at M=4096 g1; 240 / 283 us with g8. |

Where the producer excess goes:
* Signalling (k3 to k4, mean 29 / 24 us) is the largest part and is a contention cost, not a fence or link cost: an isolated CTA signals in 4 to 8 us, but when 80+ CTAs signal at once (the normal case, because CTAs run in lock-step waves) it takes 28 to 44 us, also with local Y1 (no link traffic). Removing the fence (fence=2) saves only 7 to 8 us per CTA. Each producer tile does tiles_n2 = 64 device-scope acq_rel fetch_sub on the scoreboard counters of its row panel, and all 64 producer tiles of a row hit the same 64 counters (256 contiguous bytes), so a wave issues about 17k atomics on about 8 cache lines. Not verified by a direct A/B change (would need a coalesced counter layout, beyond the 3-knob budget).
* Remote store (k2 to k3, 25 us vs 6 us local) costs about 50 us at M=1024, where both waves run in lock-step, so the 8.6 MB per wave burst (about 125 GB/s = the measured P2P store bandwidth) is not overlapped with any mainloop; only 15-30 us at M=4096 where waves de-synchronise and some drain hides behind other CTAs' mainloops.
* Because the tile count per row panel is 64 and a wave holds 4.1 row panels, dependency completion is bunched at wave boundaries (rows_done at M=1024 g1: 187 x4, 344 x4).


## Phase 2c: row-panel scoreboard (bench/timeline.py default; `--tile` = old scoreboard; PNGs timeline_rowpanel_M1024.png / timeline_rowpanel_M4096.png)

Clean runs (monitor saw only our pid on physical GPU 0/3; nvidia-smi in the pickles build/timeline_rowpanel_*.pkl). Tile-scoreboard numbers are the Phase 2b table above.

| metric (us) | M=1024 g1 tile -> rowpanel | M=4096 g1 | M=4096 g8 |
|---|---|---|---|
| per-CTA signal k4-k3 | 29.3 -> 29.6 | 24.2 -> 22.3 | 18.2 -> 15.0 |
| producer span | 341 -> 343 | 1251 -> 1244 | 1157 -> 1100 |
| tail | 114 -> 119 | 164 -> 168 | 283 -> 270 |
| harness latency (stamps on) | 470 -> 474 | 1417 -> 1429 | 1444 -> 1388 |

Knob runs with the row-panel scoreboard (producer-alone harness, M=1024 g1 / 1024 g8 / 4096 g1 / 4096 g8): baseline 355 / 340 / 1261 / 1130; nofence 343 / 333 / 1274 / 1110 (signal 23.4 / 20.4 / 14.1 / 11.2); relaxed (fence=3: no fence and relaxed counter atomic, UNSAFE) 338 / 333 / 1328 / 1086 (signal 17.8 / 13.7 / 7.4 / 5.9, store 25.5 / 20.5 / 11.0 / 8.7).
Conclusion: cutting atomics from 64 to 1 per producer tile did not shorten the producer. The per-CTA signal time is not counter contention; the store+signal time is conserved when fences/atomics are weakened (store and signal shift into each other), which points at draining the remote store burst.
