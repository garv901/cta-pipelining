# Phase 6 S2: v5 panel-ownership protocol, 3-kernel form (node1, 4x H100 SXM, TP4)

Run 2026-10-02 on Slurm job 11383 (node1, GPUs 0-3). Raw data: `build/tp4_4_both_R4.json` (v3 vs v5 R=4), `build/tp4_4_v5_R8.json`,
`build/tp4_4_v5_R4_var0.json`, `build/p6_s2_stages_{a,b,c,final}.json`; logs in `build/logs/p6_s2_*.log`. Timing: `tp4.time_step`,
"per-step" = barrier + device sync before each call, median; "b2b" = back-to-back mean; both max over ranks; 50 iterations
(stages runs a/b: 30). Per-step includes gloo-barrier rank skew for every kernel that waits on peers; the gate uses b2b.
Run-to-run spread of the same v5 4k chain is 1.060-1.110 ms b2b (5 measurements), i.e. about +-2.5 %.

## 1. Verdict

**Gate FAILS.** At M=4096 the v5 chain takes **1.088 ms b2b** (bench; 1.060-1.110 across 5 runs), against a target of 1.00 ms.
v3 measured 1.061 ms in the same process. Errors are within tolerance: x rel 3.2-4.1e-3 (limit 1e-2) and out rel 4.30e-3 (limit
2e-2). S3 was not started.

- **Where v5 stands against v3.** v5 matches v3 at 4k (+2.5 %, within noise) and at 2k, and is faster at 1k (-3.8 %), 8k (-2.6 %)
  and 16k (-3.6 %). Every M uses one instance (max_M), so v5 needs no per-M pool.
- **Why it misses the gate (structural).** In the 3-kernel form the QKV GEMM cannot start until the producer kernel has ended,
  so chain ≈ (down7 ‖ reducer) + qkv5 + 0.02 ms. With a perfect reducer the floor is already down7 + qkv5 = 0.741 + 0.269 =
  1.01 ms (final stages run).
- **Main cost: the mode-7 producer.** It costs 1.13-1.22x of mode 0 at 4k (0.707-0.759 vs 0.622-0.626 ms). The ablation
  `down7_local` (same scatter epilogue and flags, but every destination local) is 0.651-0.656 ms. So about 0.09 ms of the
  0.10-0.13 ms overhead is NVLink egress of the remote TMA stores, which burst when each 128-tile wave finishes. Dropping the
  per-tile store drain changes nothing (`down7_nodrain` 0.737-0.750).
- **Reducer.** The specified design (31 workers + 1 signaller, batch-synchronous) is clearly slower than S1's packed 8x544:
  0.636-0.656 ms alone at R=4 vs 0.442. It also does not keep pace with the producer: down+reducer = producer + 0.16-0.28 ms.
  - The cause is the lockstep batch. Each batch costs about 5 us whatever its size (1.7 us fixed + 0.053 us/row): the signaller's
    `fence.sys` and serial atomics plus the worker flag/load round trip.
  - I added a **warp-per-unit variant** (`red_variant=1`, now the v5 default): no CTA barrier; each warp owns whole units and does
    its own fence + atomic. It takes 0.44-0.50 ms alone at R=4 (equal to S1's packed shape) and keeps pace: down+reducer =
    producer + 0.04 ms at 4k. The specified variant stays available (`red_variant=0`, `red_threads` 1024 | 544) and passes all tests.
- **R=8 is worse than R=4 at every M.** The GEMMs lose 4 SMs (mode 0 on 124 SMs: 0.687 vs 0.626 ms), and the R=4 reducer already
  keeps up.

## 2. Tables

### 2a. Bench: `bench/tp4_bench.py --protocol both --R 4` (+ `--protocol v5 --R 8`, + `--red-variant 0`), ms per-step / b2b

v3 uses its default R (12 for M <= 2048, 8 above). v5 uses the reducer v1 with 1024 threads, R CTAs and producer raster 1.
"nowait" is the two mode-0 GEMMs on SM_count - R SMs (the raw floor). Out rel err is the max over ranks.

| M | v3 eager | v3 graph | v3 nowait | **v5 R4 eager** | v5 R4 graph | v5 R4 nowait | v5 R8 eager | v5 R8 graph | v5 R4 var0 (spec reducer) eager | err v3 / v5 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 0.474 / 0.345 | 0.427 / 0.360 | 0.324 / 0.298 | 0.508 / **0.332** | 0.426 / 0.355 | 0.261 / 0.235 | 0.489 / 0.366 | 0.452 / 0.388 | 0.552 / 0.371 | 4.32e-3 / 4.29e-3 |
| 2048 | 0.716 / 0.588 | 0.682 / 0.638 | 0.549 / 0.550 | 0.710 / **0.592** | 0.703 / 0.638 | 0.488 / 0.461 | 0.747 / 0.633 | 0.778 / 0.687 | 0.784 / 0.648 | 4.36e-3 / 4.40e-3 |
| 4096 | 1.199 / 1.061 | 1.207 / 1.168 | 0.936 / 0.933 | 1.155 / **1.088** | 1.190 / 1.161 | 0.878 / 0.868 | 1.218 / 1.123 | 1.239 / 1.227 | 1.415 / 1.177 | 4.64e-3 / 4.30e-3 |
| 8192 | 2.240 / 2.248 | 2.309 / 2.387 | 1.874 / 1.991 | 2.196 / **2.190** | 2.264 / 2.333 | 1.746 / 1.852 | 2.256 / 2.342 | 2.401 / 2.507 | 2.392 / 2.428 | 3.98e-3 / 4.16e-3 |
| 16384 | 4.834 / 4.770 | 5.026 / 5.055 | 4.210 / 4.266 | 4.601 / **4.599** | 4.871 / 4.911 | 4.035 / 4.043 | 4.755 / 4.741 | 5.017 / 5.036 | 5.016 / 4.950 | 4.41e-3 / 4.21e-3 |

Gate cell: v5 R4 eager 4096 b2b = 1.088 > 1.00 (FAIL). The v5 R4 protocol overhead over its own nowait floor is 0.22 ms at 4k
(v3: 0.13 ms over its R=8 floor). v5 starts from a 0.065 ms lower floor because it reserves 4 SMs instead of 8.

### 2b. Stage breakdown at M=4096 (`bench/v5_stages.py`, final run, b2b ms; per-step in brackets where informative)

| stage | R4 t1024 **v1** (default) | R4 t1024 v0 (spec) | R4 t544x8 v0 | R8 t1024 v1 |
|---|---|---|---|---|
| producer mode 0, same SMs (raster 1) | 0.626 (s128) | 0.626 | 0.626 | 0.687 (s124) |
| producer mode 7 alone | 0.741 (1.18x) | 0.709 (1.13x) | 0.720 (1.15x) | 0.828 (1.21x) |
| mode 7, no per-tile drain (dbg 32) | 0.737 | 0.750 | 0.745 | 0.830 |
| mode 7, all destinations local | 0.651 (1.04x) | 0.656 | 0.647 | 0.725 |
| reducer alone, tile flags pre-set | 0.440 | 0.656 | 0.460 (8 CTAs spread on 8 SMs) | 0.249 |
| producer ‖ reducer (real flags) | 0.782 (+0.041) | 0.906 (+0.197) | 0.877 [per-step 1.326] | 0.847 (+0.019) |
| consumer mode 5, panel flags pre-set | 0.269 | 0.268 | 0.266 | 0.274 |
| consumer mode 0, same SMs | 0.258 | 0.259 | 0.259 | 0.258 |
| full chain (eager) | **1.073** | 1.194 | 1.192 | 1.143 |

- **Mode 0 baseline.** Mode 0 on 132 SMs: 0.658 ms b2b this run (0.625 in run a). The brief's figure is 0.637.
- **Mode 7 / mode 0 ratio.** Across 7 same-process measurements at 4k it is 1.13-1.22x (S1: 1.11x with 100 % of D to one peer).
- **Producer ‖ reducer.** The expectation was ≤ producer + 0.05 ms. Variant v1 meets it at 4k (+0.041); v0 misses it by 4x.
- **Chain.** Chain ≈ (producer ‖ reducer) + consumer + 0.02: 0.782 + 0.269 + 0.022 = 1.073. The 0.02 ms covers `tp4_step`, the
  stream/event hops and kernel boundaries.
- **Producer raster 2 (n fast), run a, v0.** Producer 0.778 vs 0.707 and chain 1.258 vs 1.135, so raster 1 is kept.

### 2c. M=1024 remote-epilogue ratio with the 4-way split (b2b; per-step in brackets)

| | mode 0 (s128 r1) | mode 7 (3/4 of tiles remote) | mode 7 all-local | mode 7 no drain |
|---|---|---|---|---|
| ms | 0.157 [0.173] | 0.186-0.189 [0.218-0.223] | 0.161 [0.190] | 0.186-0.188 |
| ratio vs mode 0 | 1.00 | **1.19-1.20x b2b, 1.26-1.29x per-step** | 1.03x | 1.19x |

The 4-way split does not reduce S1's 1.19x. Nearly all of it is NVLink store time (the all-local run is 1.03x).

M=1024 v1 R4 chain stages: producer ‖ reducer 0.251 (+0.062 over mode 7), consumer mode 5 0.083 (mode 0: 0.082), chain 0.334.

## 3. SHARED CODE TOUCHED

The uncommitted S1 changes (`p6_*` probes/bindings in `csrc/gemm_tp4.cu`, `bench/push_bw.py`, `bench/nccl_ar.py`,
`results/phase6_s1.md`) are kept unchanged. Nothing is committed.

- **`scripts/gen_coop_ctapp.py`: new patch 8.** The `.hpp` was regenerated with the script, not hand-edited, and its diff contains
  only the patch-8 changes.
  - **Template parameter.** `GemmUniversalCtapp` gets a 5th parameter, `bool CtappScatter_ = false`. The default keeps every
    existing instantiation identical.
  - **Params.** `Params` gains `EpilogueParams epi_dst[4]`. The member is unconditional, so params grow to **3968 B (down) /
    3648 B (QKV)**, close to the legacy 4 KB kernel-parameter limit. This is fine with CUDA 12.9 on SM90, but any further field
    needs large-kernel-params.
  - **`to_underlying_arguments`.** When `CtappScatter && mode == 7` it builds `epi_dst[i]` with problem M = `ctapp.dst_rows` and
    `ptr_D = ctapp.dst_ptr[i]`.
  - **Prefetch.** All 4 descriptors are prefetched in mode 7.
  - **Consumer store.** In mode 7 it goes through `CollectiveEpilogue(params.epi_dst[M_idx % world])` at row block `M_idx / world`.
- **`csrc/sm90_gemm_coop_ctapp.hpp`:** regenerated.
- **`csrc/ctapp_tp4.cuh`:**
  - `CtappTp4Params` gains `unsigned* peer_tile_flags[8]`, `void* dst_ptr[4]`, `int dst_rows`, `int panels_per_owner`.
    Documented modes 5 and 7.
  - New helpers `tp4::st_relaxed_sys_u32` and `tp4::st_relaxed_sys_f32x4`.
  - `tp4_wait_panel` / `tp4_before_epilogue` accept mode 5 (target = epoch).
  - `tp4_after_store` mode 7 does: drain, named barrier, `fence.acq_rel.sys`, then `st.relaxed.sys` of epoch at the owner's
    `tile_flags[rank][m/4][n]`.
  - New `struct Tp4Red5Params`, plus `tp4::red5_unit`, `tp4::red5_publish`, `tp4::sum5_bf16x8`, `tp4::kB5Done`, `tp4::kB5Release`.
  - New kernels:
    - `template<int THREADS> tp4_reduce5_kernel` (v0 = spec signaller design; 1024 → 63 regs, 544 → 55 regs; 0 spills).
    - `template<int THREADS> tp4_reduce5w_kernel` (v1 = warp-per-unit; 1024 / 512 → 59 regs, 0 spills).
    - `tp4_step_kernel(int* epoch, float4* rowss_local4, unsigned* panel_done, int M)`.
- **`csrc/gemm_tp4.cu`:**
  - **`GemmTp4` instantiation.** It is now `GemmUniversalCtapp<..., PersistentScheduler, !Rms>`, so scatter is compiled into the
    down GEMM only. `info()` now prints `params=`.
  - **`tp4_down(...)`.** New trailing args `dst_ptrs=[]`, `dst_rows=0`, `peer_tile_flags=[]`, `panels_per_owner=0`. Mode 7 is
    accepted when: world 4, 4 addresses, `dst_rows == ppo*128`, `ceil(M/512) <= ppo`, N == 8192, and 16 B alignment.
  - **`tp4_reduce(...)`.**
    - It now accepts `version=5`.
    - New kwargs: `inbox_ptrs=[]`, `peer_rowss=[]`, `peer_panel_flag=[]`, `rowss_local=0`, `panel_done=0`, `panels_per_owner=0`,
      `threads=1024`, `variant=0`. Valid (variant, threads) pairs are (0, 1024), (0, 544), (1, 1024), (1, 512).
    - v5 reuses existing arguments: `tile_cnt` = local tile_flags, `peer_x`, `resid`, `epoch_dev`, `M` = rows, `raster` = unit order.
  - **`tp4_qkv(...)`.** Mode 5 is accepted; `panel_cnt` is the panel_flag array.
  - New binding `tp4_step(epoch_dev, rowss_local, panel_done, M)`. `tp4_info(2)` reports the reducer register and spill attributes.
- **`ctapp/tp4.py`:**
  - **`CtappBoundary`.** The constructor is now `CtappBoundary(M, Kr, N2r, rank, world, group_name, device, resid=True,
    protocol=True, use_graph=False, R=None, qkv_raster=2, qkv_swizzle=1, max_M=None, down_raster=None, red_threads=1024,
    red_blocks=None, red_variant=1)`.
    - `protocol` accepts True / False / "v3" / "v5". v5 defaults: R=4, down_raster=1, red_blocks=R.
    - In v5, `forward` takes M from `h_r` (M % 128 == 0, M <= max_M), and `out` / `x()` are `[:M]` views.
    - v5 graphs are keyed by (parity, M).
    - `warmup()` launches every v5 kernel while idle. The mode-0 GEMM writes into local scratch, never into the inbox.
    - New private methods: `_init_v5`, `_warmup_v5`, `_reduce_v5`, `_down_v5`, `_qkv_v5`, `_enqueue_v5`, `_forward_v5`.
  - **`CtappBoundaryPool`.** Gains `protocol="v3"` and `max_M=8192`. In v5 it keeps one instance and returns None for M > max_M.
- **`tests/test_tp4.py`:**
  - New flags `--protocol v3|v5`, `--red-variant`, `--red-threads`.
  - v5 adds the flags == epoch check, the variable-M sequence 4096→1024→8192→512→4096 on one max_M=8192 instance with fresh
    weights, and a 30-forward stress with alternating M.
- **`tests/test_tp4_pool.py`:** new `--protocol` flag; v5 asserts a single instance.
- **`bench/tp4_bench.py`:**
  - New flags `--protocol v3|v5|both`, `--R` (applies to v5 only with `both`), `--methods`, `--red-threads`, `--red-variant`,
    `--down-raster`.
  - v5 methods are named `ctapp_v5`, `ctapp_v5_graph`, `ctapp_v5_nowait`.
  - JSON is written to `build/tp4_{world}_{protocol}[_R{R}].json`. The default v3 invocation and its path `build/tp4_4.json`
    are unchanged.
- **`bench/v5_stages.py`:** new stage-breakdown bench. Configs are `R:threads:blocks:raster[:variant]`; `--nodrain 1` adds the
  `down7_nodrain` / `down7_local` ablations.

## 4. **PERF ASSUMPTIONS**

| choice | faster alternative not taken | why not now | rough cost |
|---|---|---|---|
| **1024-thread reducer CTAs, one per reserved SM** (v1 default; it fills the RF at ≤ 64 regs) | 2 x 512 (or 544) per SM to pack more independent warps per SM | Packing depends on placement. When the reducer is placed before the GEMM, the block scheduler spreads 8 CTAs over 8 SMs and 4 GEMM CTAs cannot become resident (t544x8: per-step producer ‖ reducer 1.33 ms vs 0.88 b2b). 1024 threads give placement-independent residency. | v1 1024 vs v1 512x8 b2b: 0.773 vs 0.783 at 4k, so none measured |
| **No cross-unit pipelining in the reducer** (v1: per unit, flag wait → 16 x [2 rows, 10 x 16 B loads in flight/lane → sum → 4 stores] → fence + atomic; no prefetch across the flag wait or row pairs) | Double-buffered loads (next row pair / next unit issued before this pair's stores), or TMA bulk loads to smem | Needs about 80 regs (cap is 64 for 1024 threads), and v1 already keeps pace with the producer | Reducer at about 75 GB/s/SM moved (0.44 ms alone at R=4); tail after the producer +0.04 ms at 4k, +0.06 ms at 1k |
| **Per-unit fence + atomic** (v1: each warp, after each 32x256 unit, `fence.acq_rel.sys` + gpu-scope `atomicAdd(panel_done)`; 1024 per rank at 4k) | Batching fences across units (v0 signaller) or per-CTA smem counters with one fence per batch | v0 measured worse: batching puts the fence and serial atomics on the batch barrier's critical path | Roughly 1-2 us per unit per warp, i.e. ~2-4 % of reducer time (not ablated) |
| **Value-flag polling** (lanes 0-3 spin `ld.acquire.sys` on the 4 per-source tile flags per unit; QKV mode 5 lane 0 spins per panel) | One counter per tile (4 producer atomics, 1 poll), multimem flags, or nanosleep backoff | Value flags need no reset, so one instance serves any M; the 4 polls run in parallel | Consumer mode 5 vs mode 0 +0.011 ms at 4k, +0.001 at 1k (wait time included) |
| **Host launch order: reducer enqueued before the producer** (on `s_red`, after `ev_start`) | Producer first, so packed reducer CTAs land on the free SMs | Irrelevant for 1024 threads (1 CTA/SM fills the SM either way); required only for the packed variants | 0 for the default; packed variants: per-step hazard above |
| **`tp4_step` as a separate kernel** (epoch += 1, zero rowss_local / panel_done) | Fold it into the previous forward's reducer tail or the producer prologue | Stream order gives the reset → reducer ordering for free; folding needs a cross-forward ordering argument | About 0.005-0.01 ms per forward (part of the 0.022 ms chain-minus-stages residual) |
| **Inbox memory footprint** (symmetric `(4, ppo*128, 8192)` bf16 = max_M x 16 KB per rank: 128 MiB at max_M 8192, 256 MiB at 16384) | Smaller ring of inbox slots with flow control | Needs back-pressure flags producer ← owner | Memory only. v5 total = inbox + 2 x = 3 x max_M x 16 KB (384 MiB at 8k), the same as one v3 instance; the v3 pool holds up to 4 instances |
| **Mode-7 producer stores remote tiles directly from the epilogue** (TMA to the peer inbox, wave-synchronous) | Spread the egress: deeper epilogue store pipeline / ping-pong so stores overlap the next mainloop, staggered rasters, or local store + owner pull | Generator / epilogue restructuring is beyond S2 (and the 128x128 / role work is S3/S4) | +0.09 ms at 4k (down7 0.741 vs all-local 0.651), +0.03 ms at 1k (1.20x vs 1.03x) |
| **Producer raster 1 (m fast)**: every panel completes only at the producer's end | Raster 2 (panels complete in order) | Without PDL the consumer cannot start before the producer kernel ends anyway, and raster 2 makes mode 7 slower | Raster 2: +0.07 ms producer, +0.12 ms chain (v0, run a) |
| **2 rows x 16 B per lane per iteration**, plain `__ldcg` / `__ldg` loads | Wider (32 B) vectors or more rows in flight | Register cap | Included in the reducer row above |
| **Graph mode copies inputs into static buffers before replay** (pre-existing design, extended to max_M) | Capture with caller-owned buffers | Same API as v3 | Graph b2b is 0.07 ms slower than eager at 4k (1.161 vs 1.088) |

## 5. What did not work, errors, test status

**What did not work:**
- **Spec reducer v0 (31 workers + 1 signaller, 1024 threads).** Alone at R=4 it takes 0.636-0.656 ms, slower than S1's packed
  8x544 (0.442), and producer ‖ reducer = producer + 0.16-0.28 ms. The per-batch cost is about 5 us, nearly constant in batch
  size. The 544-thread variant (behind `threads=544`) takes 0.425-0.46 ms alone, but its chain is no better (1.192 ms; placement
  hazard).
- **First build of `tp4_reduce5_kernel<544>`.** It compiled to 69 regs with `minBlocks=1`, so it could not pack 2 per SM.
  Changing the launch bounds to `(THREADS, THREADS <= 544 ? 2 : 1)` gives 55 regs and 0 spills.
- **R=8.** Slower than R=4 at every M (table 2a).
- **Producer raster 2.** Slower (table 2b note).
- **Dropping the per-tile store drain.** No gain (`down7_nodrain`), so the drain is not the mode-7 cost.
- **Unexplained anomaly.** In run a, v0 with raster 2 at M=1024 gave a chain of 0.953 ms b2b (other configs 0.36-0.38). I did not
  investigate because that config is not used.
- **v0 per-step inflation.** Per-step numbers for v0 producer ‖ reducer reached 2.5 ms in run a (b2b 0.99): rank skew plus the
  slow reducer. v1 per-step is 0.86 ms.
- **Errors.** No runtime errors, hangs or traps occurred in S2. The only error text was from calling `tp4_info` on the login host
  (no GPU): `torch.AcceleratorError: CUDA error: no CUDA-capable device is detected`. I re-ran it on node1.

**Tests (all on node1, 4 GPUs, after the final code change):**

| command | result | log |
|---|---|---|
| `tests/test_tp4.py --world 4` (v3) | ALL PASS (31 checks) | `build/logs/p6_s2_test_v3b.log` |
| `tests/test_tp4.py --world 4 --protocol v5` (default reducer v1) | ALL PASS (50 checks; x rel 3.25e-3, out 4.29e-3 at M=1024; variable-M x ≤ 3.46e-3, out ≤ 4.16e-3; 30-forward stress x 4.07e-3, out 3.96e-3; flags == epoch) | `build/logs/p6_s2_test_v5_v1.log` |
| `tests/test_tp4.py --world 4 --protocol v5 --red-variant 0` (spec reducer) | ALL PASS (50 checks) | `build/logs/p6_s2_test_v5_v0.log` |
| `tests/test_tp4_pool.py --world 4` (v3) | ALL PASS (24 checks) | `build/logs/p6_s2_test_pool_v3b.log` |
| `tests/test_tp4_pool.py --world 4 --protocol v5` | ALL PASS (25 checks, 1 distinct instance for every M) | `build/logs/p6_s2_test_pool_v5b.log` |

v3 remains the default protocol everywhere (`CtappBoundary(protocol=True)`, `CtappBoundaryPool(protocol="v3")`, `tp4_bench.py`
without `--protocol`).
