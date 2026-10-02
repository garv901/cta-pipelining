# Phase 6 S4: v5 boundary in vLLM, re-ablation, profile (node1, 4x H100 SXM, TP4, Llama-3.1-70B shape)

Run 2026-10-02, 12:17-13:00 UTC, Slurm job 11383 (node1, GPUs 0-3), vLLM 0.30.0, torch 2.13.0+cu130, dummy weights.
- Raw data: `build/vllm_prefill_*_s4*.json`, `build/ck_s4_*.json`, `build/prof/s4_*`, `build/logs/s4_*` (status
  `build/logs/s4_pf_status.log`, clock / power samples `build/logs/s4_smi_*.csv`).
- Full write-up: `results/vllm_prefill_4k.md` "Phase 6". Design and gates: PLAN.md "Phase 6".

## 1. Verdict

**The Phase 6 success criterion (>= 8 % over stock eager AND faster than compiled async-TP at 4K prefill) is NOT met.**

80 layers, 4K-token prefill, b=1, `llm.generate` median ms. Each entry is the mean of 2-9 runs, interleaved over four rounds in
35 min. Gain = 1 - ours / baseline.

| variant | ms | gain vs stock eager | vs compiled async-TP |
|---|---|---|---|
| stock eager | 291.5 (288.9-293.8, n=9) | - | -4.9 % |
| compiled async-TP (stock) | 277.9 (275.9-281.0, n=4) | 4.7 % | - |
| **ours v5, both boundaries** (B pdl; A pdl + steal; R=4) | **280.8** (279.6-282.6, n=5) | **3.7 %** | -1.0 % |
| ours v5, down-only | 281.4 (280.3-283.4, n=4) | 3.5 % | -1.3 % |
| v3 down-only, R=8 (Phase 5 config; Phase 5 measured 280.7 vs eager 289.9) | 281.0 (280.3-281.6, n=2) | 3.6 % | -1.1 % |

b=2 / b=4: ours both 3.9 / 3.1 %, down-only 4.7 / 4.6 %, async-TP 5.6 / 7.3 %. 8K tokens (b=1): both 3.4 %, down-only 5.0 %,
async-TP 9.4 %.

**Correctness** (4-layer config, `--rescale 1`):
- Top-1 token equals stock at M=4096 (down / both), at M=8192 and at b=2.
- Top-20 overlap is 20/20, except 19/20 at M=8192 and for the second b=2 prompt. Those are ties at rank 20, the same as Phase 5's b=2.
- max |dlogprob| 1.56e-2 (down) / 3.12e-2 (both), i.e. 1-2 bf16 ulps, identical to Phase 5.
- Per-layer boundary rel err: B qkv 3.5-3.7e-3 (x 2.2-2.5e-3), A g 2.6-2.8e-3 (x 7-10e-4).

**What changed from Phase 5:**
- Boundary B end to end: nothing. v5 281.4 vs v3 281.0 in the same rounds, within noise. The micro-bench improvement
  (1.084-1.090 to 1.064-1.074 ms) is only 79 x 0.015 ≈ 1 ms.
- Boundary A: a loss became break-even. oproj-only 290.9 vs eager 290.8-293.8 (Phase 5: 297.9 vs 289.9). Both-boundaries now
  equals down-only (Phase 5: 288-293).
- All of the A improvement is work stealing: A without steal is 300.05 ms.

**The measured ceiling, and why:**
1. **GEMMs at peak.** In situ, boundary B is producer, then consumer. The reducer ends 36-53 us after the producer, inside the
   consumer's first wave, so the reduction is off the critical path.
2. **Producer scatter penalty.** Mode 7 takes 0.73 ms vs cuBLAS down 0.61 ms in the same trace (wave-synchronous NVLink egress of
   the remote TMA stores). The consumer takes 0.27 vs 0.21 ms.
3. **Boundary A is consumer-bound.** gate_up takes 1.48-1.56 ms in situ vs cuBLAS 1.23-1.26, more than the 0.41 ms all-reduce +
   norm it hides.
4. **The 700 W power cap** (new finding).
   - Every variant runs at a median of ~690 W through the 80-layer prefill.
   - The overlapped boundary runs at a median SM clock of 1470 MHz, against 1680 for stock eager and 1515 for async-TP.
   - Per-boundary B saving: 0.25 ms in the 4-layer trace, where the clocks are not capped, but 0.17 ms at 80 layers (1.129 vs
     1.299 ms). A goes from +0.02 to -0.02 ms.
   - 79 x 0.17 - 80 x 0.02 ≈ 11.7 ms GPU time, which matches the 277.0 vs 265.4 ms GPU step and the ~10-11 ms end-to-end gain.

## 2. Tables

### 2a. b=1, M=4096 per round (median ms; JSON tag suffix `_s4<round>`)

| variant | a/b | c/d | e/f | h | i/j |
|---|---|---|---|---|---|
| stock eager (before / after) | 289.56 / 290.23 | 290.76 / 293.78 | 291.15 / 293.22 | 288.94 | 293.55 / 292.65 |
| compiled async-TP | 275.91 | - | 280.95 | 276.32 | 278.35 |
| v5 down-only | 280.29 | 281.22 | 280.69 (max_M 16384) | - | 283.42 |
| v5 both | 279.72 | 281.06 | 281.12 (max_M 16384) | 279.63 | 282.63 |
| v3 down-only R=8 | 280.32 | 281.62 | - | - | - |

One-off, round c/d (b=1):

| variant | median ms |
|---|---|
| v5 oproj-only | 290.91 |
| v5 both, A without steal | 300.05 |
| v5 both, B tile 128 | 281.81 |
| v5 down-only, pdl1 | 282.04 |
| v5 down-only, none (3 kernels) | 282.60 |

### 2b. Batch 1, 2, 4 (round e/f, ctapp with `--max-m 16384`), median ms (min)

| variant | b=1 | b=2 | b=4 |
|---|---|---|---|
| stock eager (mean of 2) | 292.2 | 579.6 | 1146.2 |
| compiled async-TP | 280.95 (274.37) | 547.39 (529.38) | 1062.20 (1033.15) |
| v5 down-only | 280.69 (276.85) | 552.61 (545.91) | 1093.86 (1086.37) |
| v5 both | 281.12 (277.10) | 556.82 (550.02) | 1110.78 (1098.95) |

8K tokens, b=1 (round g), median ms:

| variant | ms |
|---|---|
| eager | 597.77 |
| async-TP | 541.74 |
| v5 down | 567.72 |
| v5 both | 577.62 |

### 2c. Correctness (`bench/vllm_ctapp_check.py --rescale 1`, vs stock; per-layer rel err from `CTAPP_CHECK=1`)

| run | top-1 | top-20 | max abs dlogprob | B qkv rel err | A g rel err |
|---|---|---|---|---|---|
| v5 down, M=4096 | 124749 = stock | 20/20 | 1.56e-2 | 3.73 / 3.60 / 3.51e-3 | - |
| v5 both, M=4096 | 124749 = stock | 20/20 | 3.12e-2 | 3.73 / 3.60 / 3.52e-3 | 2.69-2.78e-3 |
| v5 both, M=8192 (= max_M) | 24006 = stock | 19/20 (tie at rank 20) | 1.58e-2 | 3.73 / 3.60 / 3.52e-3 | 2.64-2.73e-3 |
| v5 both, b=2 (two M=4096 steps) | 124749, 24006 = stock | 20/20, 19/20 | 3.13e-2, 1.57e-2 | 3.51-3.73e-3 | 2.69-2.78e-3 |

### 2d. Profile (torch profiler, rank 0), per-boundary ms

B is measured from the end of act_and_mul to the end of QKV; A from the start of o_proj to the end of gate_up.

| | 4-layer stock | 4-layer v5 | 80-layer stock | 80-layer v5 both | Phase 5 |
|---|---|---|---|---|---|
| B | 1.246 | **0.994** (prod 0.732, reducer +0.053 after prod, consumer 0.270 starting 11 us before prod end) | 1.299 | **1.129** (prod 0.814, cons 0.322) | 1.12 (stock 1.32) |
| A | 1.819 | 1.803 (prod 0.317, cons 1.476) | 1.885 | 1.907 (prod 0.350, cons 1.564) | 2.01 (stock 1.83) |
| layer period | 3.29 | 3.03 (down-only) | 3.434 | 3.309 | |

Kernel sequence per B boundary (all confirmed):
1. `tp4_step_kernel` (1.6 us).
2. Producer `GemmUniversalCtapp` mode 7 (grid 128) and `tp4_reduce5w_kernel<1024>` (grid 4, side stream) in parallel.
3. Consumer mode 5 (grid [1,128], PDL), which starts as the producer's CTAs retire.

There are no host gaps.

### 2e. Power and clocks (`nvidia-smi` every 100 ms, util >= 50 % samples, 4 GPUs, b=1 runs)

| variant | median power | SM clock median / mean / p10 | median ms |
|---|---|---|---|
| stock eager | 689 W | 1680 / 1704 / 1590 MHz | 288.94 |
| v5 both | 690 W | 1470 / 1529 / 1395 MHz | 279.63 |
| compiled async-TP | 685 W | 1515 / 1584 / 1410 MHz | 276.32 |

## 3. **SHARED CODE TOUCHED**

**`vllm_plugin/ctapp_vllm/model.py`** (installed editable):
- New env knobs:
  - `CTAPP_PROTO=v5|v3`, default **v5**. This changes the default behaviour: Phase 5 configs now need `CTAPP_PROTO=v3`.
  - `CTAPP_FUSION=none|pdl|pdl1|role`, default pdl.
  - `CTAPP_STEAL`: unset means A 1 / B 0; if set it applies to both boundaries.
  - `CTAPP_QKV_TILE=256|128` (B only; the default `CTAPP_B_SWIZZLE` becomes 2 for tile 128).
- `CTAPP_MAX_M` default changed from 16384 to **8192**, for v3 too: with v3, M > 8192 now takes the stock path unless the
  variable is set.
- For v5, `CtappBoundaryPool(..., protocol="v5", max_M, fusion, qkv_tile, steal)` creates one instance per boundary.
- The creation guard (sync + TP barrier) fires once for v5, and per new M for v3.
- The "first time M seen" log line uses a `_seen_M` set.
- `ctapp/tp4.py` is unchanged; the pool already supported v5.

**`bench/vllm_prefill.py`:**
- New flags `--proto`, `--fusion`, `--steal`, `--qkv-tile`, `--max-m` (exported as `CTAPP_*`), and `--tag` (a JSON-name suffix).
- `_seq<N>` is appended to the tag when seq != 4096. The JSON gains an `env` dict (`CTAPP_*`).
- For ctapp it sets a default `CUDA_MODULE_LOADING=LAZY`.
- Caveat: `--variant ctapp` without `--proto` now runs v5 but keeps the Phase 5 tag (e.g. `ctapp_down_R8`). Pass `--proto` to
  disambiguate.

**`csrc/gemm_tp4.cu`:** comment and `TORCH_CHECK` message only. The raster mapping now reads 1 = AlongM (m fast), 2 = AlongN; it
was the other way round. Both venv build dirs were rebuilt after this edit (`build/logs/s4_rebuild_*.log`).

**Docs:**
- `results/vllm_prefill_4k.md`: new "Phase 6" section, Phase 5 tables kept, Phase 5 env table annotated, NCCL discrepancy marked
  resolved, reproduce lines now pass `--proto`.
- `results/tp4_chain.md`: corrected caveat with the `nccl` row breakdown, v5 rows, old tables kept.
- `PLAN.md`: "Phase 6" section, plus corrections in "Conclusion and incorrect assumptions".
- `README.md`: status section and plugin paragraph.
- `results/phase6_s4.md`: this file.

## 4. **PERF ASSUMPTIONS**

1. **Eager-only plugin, no CUDA graphs and no torch.compile.**
   - Faster alternative: capture the boundary in vLLM's piecewise CUDA graphs / compile path (custom op with static buffers,
     `graph_bind`-style).
   - Why not now: the plugin overrides the model forward with Python control flow, and v5 graphs need stable caller pointers.
   - Rough cost: about 0 at 4K prefill. In Phase 5, compiled without async-TP was 287.8 vs eager 289.5 ms, so the GPU-bound prefill
     gains nothing from compile alone. It matters for decode and small M, which are out of scope.
2. **One instance per boundary sized `CTAPP_MAX_M`** (default 8192).
   - Memory: B 424 MiB + A 608 MiB, about 1.0 GiB per GPU (2.1 GiB at 16384).
   - Faster / leaner alternatives: an inbox ring with back-pressure, or sharing the x double-buffers between the A and B instances.
   - Cost: memory only. M > max_M takes the stock path, so the b=2/4 runs used `--max-m 16384`. With the default 8192, the
     M = 12288 / 16384 chunked steps would be stock.
3. **Boundary-A steal on by default** (one `atomicAdd` claim per 32x256 unit per warp, issued synchronously).
   - Faster alternative: batched or prefetched claims.
   - Cost: S3 estimated 5-8 % of the reduction's critical path. End to end, steal is required (300.05 ms without, 280.8 with).
4. **No SiLU·mul (act_and_mul) fusion into the gate_up epilogue.** The separate kernel takes 63.5 us per layer in the 80-layer trace
   (5.1 ms of 281, 1.8 %). Not built (out of S4 scope). At best it would close the 1 % gap to async-TP, not reach 8 %.
5. **No per-M fusion switching** (pdl for every M).
   - Faster alternative: S3 found role + steal 0.02 ms better at M <= 1024 and pdl1 best at 2k.
   - Cost: 0 for the 4K benchmark (M = 4096 only), and ≤ 0.02 ms per boundary for small chunked steps.
6. **R = 4, tile 256, producer swizzle 1 + tail auto, consumer grid 128 for "pdl".** All chosen from S2/S3 micro-bench sweeps, not
   re-tuned in vLLM. Tile 128 and pdl1 / none were re-checked end to end and were equal or worse.
7. **Two streams + 2 events per boundary for "pdl".** About 10 us of host work per boundary. It is hidden because prefill is
   GPU-bound and the profile shows no host gaps.
8. **Power cap not addressed.** The overlapped design spends the same energy in less time and is clocked down 12 % more than stock.
   - Alternatives that reduce energy per boundary: fewer bytes moved, i.e. a reduce-scatter-only boundary with sequence-parallel
     norm as async-TP does; avoiding the double x write.
   - Not explored. Cost: roughly half of the short-trace gain (0.08 of 0.25 ms per B boundary).
9. **Measurement.** Medians of 20 iterations with 3 warm-up, single runs per round, rounds interleaved. Run-to-run spread is about
   ±1 % (eager 288.9-293.8). Differences below ~2 ms are not significant.

## 5. What did not work, what was skipped, commit

**Errors and hangs:** none. Every vLLM run, check and profile returned rc=0, and there are no tracebacks in `build/logs/s4_*`. The
extension built cleanly in both venvs (199 s initial vLLM-venv build; 212 s / 191 s vLLM / bench-venv rebuilds after the comment edit). The only tool error was a harness refusal of a chained `sleep`,
which was not a GPU issue.

**Did not help (measured):**
- B tile 128 (281.81), fusion pdl1 (282.04) and none (282.60): equal to or slower than pdl.
- A without steal: 300.05.
- oproj-only: break-even (290.91).

**Skipped:**
- CUDA-graph / compiled integration of the plugin (out of scope, see PERF 1).
- act_and_mul fusion.
- An 80-layer profile of down-only (only "both" was profiled at 80 layers).
- v3 at b=2/4 and at 8K (Phase 5 has b=2/4).
- An energy-per-forward measurement (only power and clock were sampled).
- v5 with R=8 or role fusion in vLLM (S2/S3 showed both worse).

GPU time was not the constraint; these were left out as low value given the verdict.

**Commit:** one commit on main, "Phase 6: panel-ownership unicast protocol (v5), PDL-fused boundary, vLLM re-ablation", with no
Claude attribution. Not pushed. The hash is in `git log`.
