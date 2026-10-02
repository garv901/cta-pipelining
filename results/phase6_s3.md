# Phase 6 S3: 2-kernel boundary via PDL, consumer tile, producer order (node1, 4x H100 SXM, TP4)

Run 2026-10-02 on Slurm job 11383 (node1, GPUs 0-3). Raw data: `build/p6_s3_*.json`, logs `build/logs/p6_s3_*.log`.

**How timings were taken.** All numbers come from `tp4.time_step`, max over ranks:
- "b2b" is the back-to-back mean. "per-step" is the median with a barrier and device sync before each call. The gate uses b2b.
- The formal chain tables use the new `bench/tp4_bench.py --repeat N`. It builds every instance for an M once, then interleaves the
  timings N times over all methods and reports the median. Without this, run-to-run drift is as large as the gaps being measured.

**Node drift, read this before the numbers.** A second user's 2-GPU job shares node1. Over the session the same binary and config
moved by up to 5 %:
- v3 at 4k measured 1.061 in S2 and 1.084-1.090 in every S3 formal run.
- v5 "pdl" with tail 16 measured 1.002-1.012 in the earlier single-instance trace runs and 1.057-1.060 in the re-run an hour later.

Compare variants only within one run (columns of one table). The v3 column is the in-process reference.

## 1. Verdict

**Gate: mostly not met. Only the 1k target is met.**

| gate item | target | best measured (formal, interleaved) | status |
|---|---|---|---|
| boundary B chain, 4k, b2b eager | ≤ 1.02 | **1.064** (pdl, tile 128), 1.074 (pdl, tile 256); in-process v3 1.084-1.090 | **missed** (met only in the earlier, quieter single-instance runs: pdl tail16 1.002 / 1.012, pdl1 tail8 1.004) |
| boundary B chain, 4k, graph | ≤ 1.05 | **1.093** (pdl, graph without input copy); 1.168 with the input copy | **missed** |
| boundary B chain, 1k | ≤ 0.32 | **0.312-0.314** (role+steal); pdl / pdl1 / none 0.333-0.335 | **met** (Variant B with work stealing only) |
| boundary A chain (Kr 2048, N2r 14336), 4k | ≤ 1.70 | **1.882 / 1.890** (pdl+steal / none+steal); v3 2.152 | **missed** (traced single forward 1.79) |

Errors match S2 in every variant: out rel 4.16-4.40e-3 for v5 (limit 2e-2) and 3.98-4.64e-3 for v3. All tests pass (section 5).

**What S3 changed, relative to v3 in the same process:**
- At 4k on B, the best v5 variant went from +2.5 % over v3 (S2) to -1 to -2.4 % (pdl 1.074 vs 1.084; pdl t128 1.064 vs 1.090).
- At 1k, role+steal is -10 % (0.313 vs 0.348).
- At 8k, pdl is -3.4 % (2.227 vs 2.305).
- On shape A at 4k, the steal variants are -12.5 % (1.882 vs 2.152).

Normalised to S2's v3 (1.061), the best 4k B chain is 1.036-1.051, so the 1.02 target is not reached robustly.

**Why the 4k target is missed (from the stamps, section 2b):**
- The critical path is producer end, then the consumer.
- The producer mode 7 alone is 0.72-0.75 ms. In the chain its last CTA ends at 0.74-0.83 ms (rank spread about 60 us).
- The consumer then takes 0.27-0.33 ms on its own panels (alone: 0.254-0.256).
- PDL removed the launch gap: the consumer's first tile starts 0-5 us after the producer's last CTA. The panel-major producer tail
  (patch 10) took the reducer off the critical path: its last panel lands within the consumer's first wave.
- What is left is producer + consumer at near-peak speed (S2's floor estimate was 0.95-1.0). The remaining margin is the mode-7
  remote-store overhead (+0.1 ms over mode 0) plus about 0.05 ms of consumer contention.

**Why Variant B (role) lost:**
- The role reducer runs inside the GEMM CTA, with its 214 KB smem carve-out, leaving about 28 KB of L1. That caps the plain loads
  in flight.
- The standalone warp-per-unit reducer drops from 0.420 to 0.576 ms alone when launched with the same carve-out (`p6_s3_l1probe`).
- So R=4 role CTAs fall 0.2-0.3 ms behind the producer: chain 1.31-1.33 at 4k.
- cp.async staging only partly fixed this (1.150). Work stealing fixes it at 1k and on shape A, where it is the best variant, but
  not at 2k-8k on B.

**Shape A** is bounded by its consumer:
- The consumer GEMM is 1.37-1.47 ms alone, against cuBLAS 1.34. In situ it is 1.45-1.55.
- The 0.25-0.34 ms producer cannot hide the 64 MB reduction: the standalone reducer ends at 0.69-0.76 ms.
- Stealing moves the reduction onto all 128 consumer CTAs: steal ends at 0.44-0.46 ms.
- The floor in this structure is therefore about 0.35 + 0.1 + 1.4 = 1.85 ms. 1.70 would need a faster consumer GEMM, not a
  better boundary.

**Recommendation for S4:**
- **Protocol v5, producer raster 1 / swizzle 1 with the panel-major tail** (`down_tail="auto"`: 16 for M ≥ 2048, 8 below). This
  is now the v5 default.
- **Consumer tile 256.** Tile 128 is not a consistent win. It is 0.01-0.03 faster at 2k/4k for none/pdl, slower at 1k/8k, and
  0.4-1.4 ms slower on shape A.
- **Boundary B (down → norm → QKV): `fusion="pdl"`** (Variant A', standalone reducer on its own stream, PDL consumer grid 128,
  stock producer trigger). It is best or tied at 2k-8k (0.555 / 1.074 / 2.227) and 0.334 at 1k. `role`+`steal` is 0.02 ms better
  at M ≤ 1024 if S4 wants a per-M switch; every mode is a per-call argument, so a per-forward switch is cheap.
  - `pdl1` (single stream: reducer, producer, consumer) is the best at 2k (0.540) and tied at 4k. But it is +8 % at 8k (2.41 vs
    2.23) in two runs, so it is not the default.
- **Boundary A (o_proj → norm → gate_up): `fusion="pdl", steal=True`** (or `none`+steal): 0.500 at 1k and 1.882 at 4k, against
  0.545 / 2.198 for pdl.
- **Graphs:** capture without the input copy (graph_bind-style, or inside vLLM's own static buffers). The copy costs 0.07-0.14
  ms at 4k.

## 2. Tables

### 2a. Chain bench, boundary B (Kr 7168 / N2r 2560), ms b2b (per-step), interleaved median of 5 (`p6_s3_chain_rep`)

R=4. Tile 256 unless "t128". v5 uses tail auto. "role" is Variant B, grid 132. "pdl" / "pdl1" / "none" use consumer grid 128.

| M | v3 | v5 none | none t128 | pdl | pdl t128 | pdl1 | pdl1 t128 | role | role t128 | role+steal | role+steal t128 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 0.348 (0.529) | 0.334 (0.500) | 0.347 | 0.334 (0.470) | 0.353 | 0.333 (0.429) | 0.352 | 0.388 (0.485) | 0.404 | **0.313** (0.408) | 0.334 |
| 2048 | 0.603 (0.753) | 0.568 (0.708) | 0.541 | 0.557 (0.684) | 0.552 | **0.540** (0.656) | 0.558 | 0.713 (0.817) | 0.691 | 0.608 (0.696) | 0.605 |
| 4096 | 1.090 (1.266) | 1.086 (1.181) | 1.073 | 1.087 (1.226) | **1.064** | 1.078 (1.178) | 1.187 | 1.328 (1.449) | 1.330 | 1.146 (1.219) | 1.199 |
| 8192 | 2.306 (2.382) | **2.217** (2.251) | 2.282 | 2.287 (2.296) | 2.266 | 2.329 (2.279) | 2.585 | 2.677 (2.697) | 2.775 | 2.375 (2.326) | 2.619 |
| out rel err | 4.3-4.6e-3 | 4.16-4.40e-3 (identical in every v5 column) | | | | | | | | | |

Second interleaved run (`p6_s3_chain_steal`, median of 5, tile 256, b2b). "+steal" means the mode-8 consumer claims reducer units
before its tiles (grid 128 for none/pdl):

| M | v3 | none | pdl | pdl1 | none+steal | pdl+steal | role+steal |
|---|---|---|---|---|---|---|---|
| 1024 | 0.348 | 0.335 | 0.334 | 0.335 | 0.334 | 0.335 | **0.314** |
| 2048 | 0.600 | 0.566 | 0.555 | **0.541** | 0.576 | 0.576 | 0.610 |
| 4096 | 1.084 | 1.081 | **1.074** | 1.078 | 1.092 | 1.084 | 1.203 |
| 8192 | 2.305 | 2.243 | **2.227** | 2.412 | 2.244 | 2.244 | 2.396 |

Graph (`p6_s3_chain_graph`, median of 3, tile 256, b2b; eager / graph with input copy / graph without copy "graphb"):

| M | v3 eager / graph | none | pdl | pdl1 | role+steal |
|---|---|---|---|---|---|
| 1024 | 0.348 / 0.365 | 0.334 / 0.358 / 0.331 | 0.334 / 0.357 / 0.332 | 0.335 / 0.358 / 0.333 | 0.313 / 0.336 / **0.312** |
| 2048 | 0.596 / 0.644 | 0.570 / 0.615 / 0.573 | 0.556 / 0.603 / 0.561 | 0.540 / 0.582 / **0.549** | 0.610 / 0.652 / 0.612 |
| 4096 | 1.087 / 1.178 | 1.070 / 1.206 / 1.107 | 1.079 / 1.170 / **1.093** | 1.076 / 1.168 / 1.122 | 1.146 / 1.218 / 1.176 |
| 8192 | 2.346 / 2.431 | 2.233 / 2.479 / 2.299 | 2.307 / 2.454 / **2.241** | 2.409 / 2.438 / 2.366 | 2.463 / 2.538 / 2.498 |

R=8 (`p6_s3_chain_R8`, median of 3, tile 256, b2b):

| M | none | pdl | pdl1 | role | role+steal |
|---|---|---|---|---|---|
| 1024 | 0.371 | 0.373 | 0.373 | 0.368 | 0.379 |
| 4096 | 1.150 | 1.149 | 1.111 | 1.199 | 1.213 |

R=8 is worse everywhere except static role, which improves from 1.328 to 1.199 but stays behind A'.

### 2b. Stage stamps at M=4096, boundary B (%globaltimer, us from the first producer CTA entry, max over ranks, median of 5 traced eager forwards)

| config (run) | b2b | producer end min-max | reducer panels first-last | consumer entry min/max | consumer 1st tile (median) | role reduce end / steal end | consumer end |
|---|---|---|---|---|---|---|---|
| none, tail 16 (`trace4k_cmp`, current node state) | 1.038 | 741-795 | 615-880 | 799/799 | 804 | - | 1117 |
| pdl, tail 16 (`trace4k_cmp`) | 1.060 | 742-804 | 617-879 | 799/804 | 803 | - | 1115 |
| pdl1, tail 16 (`trace4k_cmp`) | 1.055 | 761-826 | 646-931 | 763/827 | 803 | - | 1120 |
| role+steal, tail 16 (`trace4k_cmp`) | 1.123 | 744-806 | 759-895 | 34/807 | 815 | 855 / 896 | 1204 |
| pdl, trig0, tail 16 (`trace4k_tail`, quiet period) | 1.002 | 724-783 | 617-871 | 778/784 | 783 | - | 1088 |
| pdl, tail 16 (`trace4k_pdl1`, quiet period) | 1.012 | 677-747 | 579-827 | 741/748 | 746 | - | 1011 |
| pdl1, tail 8 (`trace4k_pdl1`, quiet period) | 1.004 | 694-758 | 753-852 | 696/759 | 760 | - | 1043 |
| pdl, no tail, trig0 (`trace4k`) | 1.046 | 657-728 | 770-796 | 721/728 | 783 | - | 1068 |
| pdl, no tail, trig1 (`trace4k`): inversion | 1.158 | 670-742 | 812-841 | 672/743 | 823 | - | 1112 |
| role (static) R4 rows4 (`trace4k`) | 1.307 | 671-743 | 933-974 | 34/744 | 949 | 975 / - | 1265 |
| role (static) t128 R8 (`trace4k`) | 1.140 | 686-830 | 829-880 | 32/831 | 856 | 881 / - | 1201 |
| role+st3 rows2 (`trace4k_steal`) | 1.150 | 674-745 | 883-929 | 35/745 | 882 | 929 / - | 1220 |
| role+steal, no tail (`trace4k_steal`) | 1.052 | 709-783 | 829-840 | 36/784 | 832 | 823 / 841 | 1126 |

Boundary A at 4k (`traceA`, `traceA_steal`; consumer raster 1 / swizzle 2):

| config | b2b | producer end | reducer panels | consumer 1st tile (median) | steal end | consumer end |
|---|---|---|---|---|---|---|
| none, tail auto | 2.044-2.065 | 237-337 | 472-759 | 568-656 | - | 2076-2282 |
| pdl, tail auto | 2.126 | 253-339 | 509-721 | 631 | - | 2200 |
| none+steal | 1.793 | 248-334 | 456-462 | 447 | 443 | 1968 |
| pdl+steal | 1.867 | 255-341 | 452-457 | 461 | 457 | 1962 |
| role+steal | 2.014 | 262-403 | 414-449 | 432 | 450 | 2033 |

### 2c. Producer swizzle sweep: mode 7 vs mode 0, raster 1, 128 SMs, b2b ms (per-step) (`p6_s3_sweep_down_final`; earlier run `p6_s3_sweep_down`)

| M | s1 | s2 | s4 | s8 | s1 + tail 8 | s1 + tail 16 | mode 0 s1 |
|---|---|---|---|---|---|---|---|
| 1024 | 0.189 (0.222) | 0.188 | 0.190 | **0.187** | **0.187** | 0.191 | 0.157 |
| 2048 | **0.359** (0.401) | 0.367 | 0.367 | 0.361 | 0.370 | 0.375 | 0.311 |
| 4096 | 0.722 (0.765) | **0.714** | 0.737 | 0.721 | 0.753 | 0.751 | 0.626 |
| 8192 | 1.584 (1.552) | 1.553 | **1.541** | 1.569 | 1.674 | 1.676 | 1.392 |

The earlier run gave 4k s1 0.717, s2 0.730, s8 0.727, and 8k s1 1.570, s4 1.473. Spread is about 0.02-0.05, so swizzle 1/2/8 are
equal at ≤ 4k. The tail costs +0.03 alone at 4k and +0.09 at 8k, but at 4k it saves more than that in the chain (stamps). Choice:
swizzle 1 + tail auto. The tail order needs raster 1 / swizzle 1.

### 2d. Consumer tile sweep, 1 GPU, mode 0 (no waits), K = 8192, b2b ms (`p6_s3_sweep_qkv`)

| N (shape) | M | t256 r2 s1 | t256 r1 s2 | t128 r1 s1 | t128 r2 s2 | t128 r2 s4 | t128 r1 s4 | cuBLAS |
|---|---|---|---|---|---|---|---|---|
| 2560 (B), sms 128 | 1024 | 0.082 | **0.081** | 0.089 | 0.089 | 0.089 | 0.089 | 0.060 |
| 2560 (B), sms 128 | 4096 | 0.256 | 0.254 | 0.249 | 0.239 | **0.238** | 0.239 | 0.219 |
| 2560 (B), sms 132 | 4096 | 0.256 | 0.257 | 0.274 | 0.240 | 0.239 | 0.244 | |
| 14336 (A), sms 128 | 1024 | 0.496 | 0.346 | **0.328** | 0.383 | 0.338 | 0.332 | 0.310 |
| 14336 (A), sms 128 | 4096 | 1.896 | **1.374** | 1.504 | 1.676 | 1.464 | 1.480 | 1.342 |
| 14336 (A), sms 132 | 4096 | 1.798 | 1.472 | 1.750 | 1.654 | 1.518 | 1.523 | |

Tile 128 (raster 2 / swizzle 2, now `QKV128_B`) is 0.015 faster alone at 4k on B. In the chain it gains at most 0.02, and loses
at 1k/8k and when combined with pdl1. On A, 128 SMs beat 132: 1792 tiles is exactly 14.0 waves on 128 against 13.6 on 132.

### 2e. Boundary A shape (Kr 2048, N2r 14336), chain b2b ms, consumer raster 1 / swizzle 2

| M | v3 | none | pdl | pdl1 | role | none+steal | pdl+steal | role+steal | (run) |
|---|---|---|---|---|---|---|---|---|---|
| 1024 | 0.566 | 0.532 | 0.545 | - | - | **0.496** | 0.500 | 0.510 | `chain_A_steal`, median of 5 |
| 4096 | 2.152 | 2.151 | 2.198 | - | - | 1.890 | **1.882** | 1.978 | `chain_A_steal`, median of 5 |
| 1024 | 0.561 | 0.533 (t128 0.545) | 0.545 (0.627) | 0.556 (0.641) | 0.628 (0.700) | - | - | 0.510 (0.597) | `chain_A`, median of 3 |
| 4096 | 2.123 | 2.076 (t128 2.465) | 2.143 (3.499) | 2.542 (3.505) | 3.146 (3.611) | - | - | 2.063 (2.929) | `chain_A`, median of 3 |

graphb on A at 4k: none 2.312, pdl 2.156, pdl1 2.436, role 3.146, role+steal 2.032.

Consumer raster 2 on A is noisy (b2b min-max spread 0.2-0.4 ms) and not a reliable gain: at 4k, swizzle 8 gave none 2.060, pdl
1.969, role+steal 2.293; swizzle 4 gave 2.45-2.69.

## 3. SHARED CODE TOUCHED

S1/S2 work is kept. S3 deltas against the S2 snapshot in `build/s2_backup/`:

- **`ctapp/ext.py`** `load_tp4`: `extra_cuda_cflags += -DCUTLASS_ENABLE_GDC_FOR_SM90=1`. This is one shared flag list, so both
  venvs' build dirs pick it up on their next build. The vLLM-venv build was not rebuilt here.
- **`scripts/gen_coop_ctapp.py`**, which regenerates `csrc/sm90_gemm_coop_ctapp.hpp` (960 lines):
  - **9a** (producer trigger): `if (params.ctapp.pdl_trigger) cutlass::arch::launch_dependent_grids();` after `cluster_wait_fn()`
    in the prologue, plus the trace stamp slot 1.
  - **9b**: the 3 `wait_on_dependent_grids()` are skipped for modes 2, 5, **7** and 8. Mode 7 was added for pdl1, where the
    producer is the PDL dependent of the spinning reducer.
  - **9c**: new template arg `bool CtappRole_ = false`. For mode 8 it runs `tp4_role_preamble(params.ctapp, smem_buf)` at the top
    of `operator()`, before the TMA prefetch and pipeline init. Also the entry stamp `tp4_trace_entry`.
  - **10** (new): `tp4_remap_tile(params.ctapp, work_tile_info)` after all 7 `work_tile_info = next_work_tile_info;` and both
    `scheduler.initial_work_tile_info(...)` assignments (9 sites, asserted).
  - Patches 1-8 are unchanged.
- **`csrc/ctapp_tp4.cuh`**:
  - `CtappTp4Params`, new fields:
    - `int tail_cols`, `int tiles_m` (patch 10).
    - `int pdl_trigger` (9a).
    - `int R`, `int* role_ctr`, `unsigned long long* trace` ([grid][8] stamps).
    - `Tp4Red5Params red5` (role reducer params; mode 8).
    - Mode 8 is documented.
  - `Tp4Red5Params`, new fields:
    - `trace` (panel publish stamps), `swz` (producer swizzle order).
    - `rows` (2|4|8 rows per lane in flight), `stages` (0 | 2 | 3 cp.async ring).
    - `pdl` (standalone reducer triggers at entry), `tail` (units follow the tail order), `int* unit_ctr` (work stealing).
  - New device functions:
    - `tp4_cta_id`, `tp4_smid`, `tp4_trace`, `tp4_trace_entry`.
    - `red5_map(p, Pr, u, j, n, q)`: unit to (panel, n-tile, quarter); honours swz and tail.
    - `red5w_loop<ROWS>(p, first, stride)`: warp-per-unit loop, static stride or `claim()` via atomicAdd on `unit_ctr`.
    - `cp_async16`, `cp_async_commit`, `cp_async_wait<N>`, `red5a_warp_bytes`, `kRed5aSmemOffset = 128`,
      `red5a_loop<ROWS, ST>` (cp.async staged).
    - `tp4_role_reduce(p, role, R, smem)`.
    - `tp4_role_preamble(p, scratch)`: arrival atomic; reduces if `role < R || unit_ctr`.
    - `tp4_remap_tile<WorkTileInfo>(p, w)`.
  - New kernel: `tp4_reduce5r_kernel` (the role reducer standalone, 384 threads).
  - Changed kernels:
    - `tp4_reduce5w_kernel` now calls `red5w_loop` and triggers PDL at entry when `p.pdl`.
    - `tp4_step_kernel` also zeroes `role_ctr[0..1]`.
- **`csrc/gemm_tp4.cu`**:
  - `GemmTp4<BM,BN,BK,Rms>` now passes `CtappRole_ = Rms` to `GemmUniversalCtapp` (`CtappScatter_ = !Rms` as in S2), so the QKV
    instances get the mode-8 preamble. New instance `GemmQkv128 = GemmTp4<128,128,64,true>`.
  - `make_red5(..., rows, red_trace, stages=0, unit_ctr=0, tail=0)`.
  - Signatures (pybind defaults in brackets):
    - `tp4_down(..., pdl_trigger [0], trace [0], tail_cols [0], pdl [0])`.
    - `tp4_qkv(..., tile [256], R [0], role_ctr, rank, red_order, red_swz, tile_flags, resid, inbox_ptrs, peer_x, peer_rowss,
      peer_panel_flag, rowss_local, panel_done, panels_per_owner, red_rows [4], trace, red_trace, red_stages [0], red_dyn [0],
      red_tail [0])`. Mode 8 needs `role_ctr` numel ≥ 2.
    - `tp4_reduce(..., variant [0], swz [1], rows [2], red_trace [0], dsmem [0], stages [0], unit_ctr [0], tail_cols [0],
      pdl_trigger [0])`. Variant 2 is the standalone role reducer. `dsmem` is a diagnostic: it sets MaxDynamicSharedMemorySize and
      carve-out 100.
    - `tp4_step(..., role_ctr)`, numel ≥ 2.
    - `tp4_info(3)` returns the GemmQkv128 attributes.
- **`ctapp/tp4.py`** `CtappBoundary`:
  - New kwargs: `fusion="none"|"pdl"|"pdl1"|"role"`, `qkv_tile=256|128`, `role_rows=4`, `down_swizzle=1`, `pdl_trigger=None`
    (default 1 for pdl1/role, else 0), `trace=None` (or env CTAPP_TRACE), `role_stages=0`, `steal=False`, `down_tail="auto"`,
    `graph_bind=False`.
  - `role_ctr` is now int32[2].
  - New `_tail(M)`, `_tr`, `trace_summary()`. `_reduce_v5`, `_down_v5`, `_qkv_v5`, `_enqueue_v5`, `warmup` and the graph path are
    updated. Warmup covers every kernel/mode used by the instance.
  - **Default change:** v5 "none" now uses the tail order (`down_tail="auto"`). `down_tail=0` restores the S2 order.
- **`tests/test_tp4.py`**:
  - New options `--fusion none|pdl|pdl1|role`, `--steal`, `--role-stages`, `--down-tail`, `--qkv-tile`, `--qkv-swizzle`, `--R`,
    `--role-rows`.
  - The v5 extras are a variable-M sequence on one instance, a 30-forward b2b stress, and a 50-forward b2b variable-M
    (4096/1024/8192) hang check, eager and graph.
- **`bench/tp4_bench.py`**:
  - `--fusion` list with `+steal`, `--qkv-tile` list, `--qkv-raster/--qkv-swizzle`, `--role-rows`, `--role-stages`,
    `--pdl-trigger`, `--down-tail`, `--down-swizzle`, `--shape A|B`.
  - New method `ctapp_graphb`.
  - New **`--repeat N`** (interleaved median).
  - `QKV128_B = (2, 2)`. Graph-mode error check is now `"_graph" in name`.
- **`bench/v5_stages.py`** (untracked, S2 file): sweeps `down` (`--tails`), `qkv` and `trace`, with the fusion-config grammar
  `fusion[+steal|+st2|+st3]:tile:R:rows[:down_swizzle[:shape[:pdl_trigger[:down_tail]]]]`.
- Not touched: the vLLM plugin, docs, and the v3 kernels' semantics. v3 shares the regenerated GEMM header; patches 9 and 10 are
  inert for its modes.

## 4. **PERF ASSUMPTIONS**

1. **Role-CTA vectors in flight.**
   - Chosen: 12 warps × warp-per-unit, `rows=4`, plain `ld.global`. That is 20 × 16 B per lane (80 regs) in flight under the
     GEMM's 214 KB smem / ~28 KB L1 carve-out.
   - Faster alternative not taken: an L1-free load path into the CTA's idle 214 KB smem (TMA `cp.async.bulk` per unit, or a
     deeper cp.async ring with more rows), or stealing by default.
   - Why not now: the cp.async ring (stages 3, rows 2) only reached 0.596 alone vs 0.420 for the standalone reducer. The TMA
     variant would need a per-unit tensor map or a 1-D bulk-copy restructure.
   - Rough cost: static role is 0.25-0.3 ms slower at 4k (1.33 vs 1.07-1.09). Stealing recovers most of it (1.15-1.20).
2. **Static tile share for late role CTAs** (Variant B).
   - Chosen: after reducing, role CTAs take their full static-scheduler share of GEMM tiles (CUTLASS static persistent scheduler,
     grid 132).
   - Faster alternative not taken: a dynamic tile counter for the consumer, or fewer tiles for role CTAs.
   - Why not now: it means replacing the CUTLASS scheduler in the generated kernel.
   - Rough cost: role CTAs start tiles at 0.95-1.0 ms against 0.80 for the others, so about 0.03-0.1 ms of consumer tail at 4k.
     The same applies to steal CTAs, which all start tiles about 0.1 ms late at 1k.
3. **Producer trigger placement.**
   - Chosen: early trigger after the prologue (9a) for role / pdl1. The stock last-tile trigger for "pdl" (two streams).
   - Faster alternative not taken: a trigger at "last wave started" (tile-index based), releasing consumer CTAs exactly as
     producer SMs free.
   - Why not now: the early trigger in two-stream "pdl" caused launch-order inversion. Queued consumer CTAs took the free SMs
     before the reducer launched: +0.06-0.11 ms. A persistent producer holds its SMs anyway, so the consumer gains nothing earlier.
   - Rough cost: ≤ 5 us of launch latency (consumer first tile 0-5 us after producer end in the stamps).
4. **Consumer grid 128 in A'** (pdl / pdl1 / none and their +steal).
   - Chosen: grid SM_count - R, so an inversion can only serialise, never deadlock. The 4 reducer SMs idle after the reducer ends.
   - Faster alternative not taken: grid 132.
   - Rough cost: about 0 for B (320 tiles = 3 waves either way; 1-GPU 0.256 vs 0.256). For A, 128 is faster (14.0 vs 13.6 waves:
     1.374 vs 1.472). Up to 3 % for other shapes where 132 divides better.
5. **Single `tp4_step` kernel per forward** (epoch bump, zero rowss_local / panel_done / role_ctr).
   - Faster alternative not taken: fold it into the producer's first CTA, or make the flags parity-epoch so nothing needs zeroing.
   - Rough cost: one extra launch and serialisation, about 3-5 us per boundary (0.3-0.5 % at 4k).
6. **Swizzle picked from a coarse sweep.**
   - Chosen: producer swizzle 1 + tail auto, from single runs of {1,2,4,8} × {1k,2k,4k,8k} with ±0.02-0.05 noise.
   - Faster alternative not taken: at 8k, swizzle 4 is 0.03-0.11 faster alone, and the tail costs +0.09 alone. The 8k chain was
     not measured with tail 0 / swizzle 4. A per-M (swizzle, tail) table, or a tail sized to exactly the last wave, would be
     better.
   - Rough cost: possibly 0.03-0.1 ms at 8k.
7. **No act_and_mul fusion.**
   - Chosen: on boundary A, the gate_up output still goes to a separate SiLU-and-mul kernel in vLLM.
   - Faster alternative not taken: fuse it into the consumer epilogue and write M × 7168 instead of M × 14336.
   - Why not now: out of S3 scope (chain only).
   - Rough cost: about 0.04-0.05 ms per A boundary at 4k (117 MB read+write saved per rank).
8. **Tail order cost.** The panel-major tail costs the producer +0.02-0.03 ms at 4k and +0.09 at 8k alone, accepted for the
   chain-level saving (about 0.04-0.05 at 4k). The "auto" rule (16 for M ≥ 2048, else 8) came from a 3-point sweep.
9. **Steal claim granularity.**
   - Chosen: one `atomicAdd` per 32 × 256 unit per warp, on a single counter, issued synchronously before each unit.
   - Faster alternative not taken: prefetch the next claim while processing the current unit, or claim k units per atomic.
   - Rough cost: about 0.5-1 us claim latency per unit, against roughly 10-13 us of unit work. That is 5-8 % of the reduction's
     time on the warp's critical path, plus serialisation of up to 1584 warps on one L2 line at the producer end.
10. **cp.async ring size.** stages 2 | 3 × rows 2 (≤ 92 KB of the 214 KB smem). Not tuned further once the measurement showed it
    still L1/latency-bound (0.59-0.60 alone).
11. **Graph input copy.** `use_graph` copies h_r / resid into static buffers before replay: +0.07-0.14 ms at 4k. `graph_bind` removes
    it but needs stable caller pointers (bench only). S4 should capture inside vLLM's static buffers.
12. **Two streams + 2 events in "pdl"** (the recommended B variant). Only pdl1 / role are single-stream. Host cost about 10 us, no
    GPU gap measured, but graph capture has to include the fork/join.
13. **Standalone reducer shape unchanged from S2** (R=4 CTAs × 1024 threads, warp-per-unit). On shape A it is the bottleneck
    (ends 0.4 ms after the producer). Only stealing fixes that. A larger standalone grid would compete with the producer.
14. **Measurement method.** Median of 3-5 interleaved repeats, 30-50 iterations. Absolute values drift ±3-5 % with node load (a
    co-tenant job), so gate comparisons near 1.02 are within noise.

## 5. What did not work, errors, hangs, test status

**What did not work:**
- **Static role reducer (Variant B), R=4:** 1.31-1.33 ms at 4k, 0.388 at 1k. The root cause is the L1 carve-out.
  - The standalone v1 reducer took 0.420 alone, and 0.576 when launched with 214016 B dynamic smem.
  - v2 rows4 took 0.482 alone, and 0.629 under the same carve-out (`p6_s3_l1probe`).
  - rows 8 brought no gain (1.286). R=8 helped (1.199) but stayed behind A'.
- **cp.async staged role reducer (role+st3, rows 2):** 1.150 at 4k.
- **Early producer trigger in two-stream "pdl"** (launch-order inversion): 1.104-1.158 vs 1.002-1.046 with the stock trigger.
- **Stealing on B at 2k-8k:** no gain for pdl / none (1.084-1.092 vs 1.074-1.081 at 4k). pdl1+steal measured 1.078 vs 1.021
  (trace run).
- **Tile 128 on shape A** (2.47-3.5 vs 2.08), and **pdl1 at 8k** (2.33-2.41 vs 2.22-2.24).
- **Consumer raster 2 on A:** no reliable gain.

**Errors hit, all fixed:**
- `ValueError: invalid literal for int() with base 10: 'eal'` in `v5_stages.py`: "+steal" matched `startswith("st")`. Fixed with an
  exact match on `st2` / `st3`.
- A duplicated variant-0 launch in `tp4_reduce` after adding the dsmem diagnostic was removed before measuring.
- A `cudaFuncSetAttribute` carve-out set by a `dsmem>0` diagnostic launch persists for that kernel for the rest of the process,
  which contaminated later rows-8 numbers in the same process.
  - Only the diagnostic (`dsmem`), and `stages>0` for `red_variant=2`, set it. No production path does. The affected numbers were
    discarded.

**Hangs:** none observed in any run (benches, sweeps, traces, tests).

**Test status** (final runs after the last code change; logs `build/logs/p6_s3_final_test_*`). Every v5 run includes the 30-forward
stress and the 50-forward b2b variable-M (4096/1024/8192) hang check, eager and graph:

| test | result |
|---|---|
| `tests/test_tp4.py --world 4 --protocol v3` | ALL PASS (32) |
| `--protocol v5` (fusion none, tail auto) | ALL PASS (54) |
| `--protocol v5 --fusion pdl` | ALL PASS (54) |
| `--protocol v5 --fusion pdl1` | ALL PASS (54) |
| `--protocol v5 --fusion pdl1 --qkv-tile 128` | ALL PASS (54) |
| `--protocol v5 --fusion role` | ALL PASS (54) |
| `--protocol v5 --fusion role --steal` | ALL PASS (54) |
| `tests/test_tp4_pool.py --world 4` (v3) | ALL PASS (24) |
| `tests/test_tp4_pool.py --world 4 --protocol v5` | ALL PASS (25) |

Nothing is committed.
