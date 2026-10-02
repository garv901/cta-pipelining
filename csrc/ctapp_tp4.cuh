// Phase 4: TP-t layer boundary down-proj -> RMSNorm -> QKV with a CTA-pipelined in-switch (NVLS multimem) reduction.
//
// Producer (down-proj, K-sharded, one identical persistent grid per rank): every CTA stores its partial tile into the
// rank's copy of a symmetric buffer, signals a multicast per-tile counter, then reduces ITS RANK'S SLICE OF ROWS of the tile
// it stored one step earlier (one wave old, so every rank's partial of it is normally already there): multimem.ld_reduce
// (fp32 accumulate over all copies) + optional residual add -> multimem.st broadcast into every rank's copy of x, per-row
// sum of squares -> multimem.red.add into every rank's copy of rowss, then one multimem.red.release on the panel counter.
// A 128-row panel is complete everywhere when panel_cnt reaches epoch * tiles_n * world.
// Consumer (QKV, N-sharded): the TMA-load warp waits for the panel of each tile before loading A; the epilogue scales each
// row by rsqrt(rowss/N + eps) (RMSNorm with gamma folded into the weights).
// Counters are monotonic (epoch-scaled targets), rowss is double-buffered by epoch parity and zeroed by the host.
#pragma once
#include <cuda_bf16.h>
#include <cstdint>
#include "cutlass/arch/barrier.h"

// Phase-6 v5 owner-reducer parameters (see the v5 section below); also embedded in CtappTp4Params for the mode-8 role CTAs.
struct Tp4Red5Params {
  const uint4* inbox[4] = {};         // local inbox slot per source rank, (ppo * 128, 8192) bf16 each
  const unsigned* tile_flags = nullptr;  // local [4][ppo][32]
  const uint4* resid = nullptr;       // local (M, 8192) bf16, optional
  uint4* x[4] = {};                   // x[parity] of every rank (x[rank] is local)
  float* rowss[4] = {};               // rowss[parity] of every rank
  unsigned* panel_flag[4] = {};       // panel_flag of every rank
  float* rowss_local = nullptr;       // local [max_M], zeroed by tp4_step
  unsigned* panel_done = nullptr;     // local [max_M / 128], zeroed by tp4_step
  const int* epoch_ptr = nullptr;     // device epoch (bumped by tp4_step); overrides `epoch`
  unsigned long long* trace = nullptr;  // optional: %globaltimer at each panel publish, [max_M / 128]
  int epoch = 0;
  int rank = 0, m_valid = 0, ppo = 0;
  int order = 2;                      // the producer's raster: 2 = n fast (units panel-major), 1 = m fast (units n-major)
  int swz = 1;                        // order 1 only: the producer's swizzle (units follow its tile order: n groups of swz)
  int rows = 2;                       // warp-per-unit loops: rows per lane per iteration (16 B x (4 + 1) x rows loads in flight)
  int stages = 0;                     // role reducer (S3): 0 = register loads (ld.global, staged by L1); 2 | 3 = cp.async (LDGSTS,
                                      // L1 bypass) into a per-warp smem ring of `stages` x rows rows (rows == 2), see red5a_loop
  int pdl = 0;                         // S3 "pdl1": stand-alone reducer kernels trigger their PDL dependent (the producer) at entry
  int tail = 0;                       // S3: the producer's panel-major tail (tail_cols, see tp4_remap_tile); units follow it
  int* unit_ctr = nullptr;            // S3 work stealing: non-null = warps claim units dynamically (atomicAdd, zeroed by tp4_step)
                                      // instead of the static stride; every claimant runs the unit to completion
};

struct CtappTp4Params {
  int mode = 0;              // 0 stock; 1 producer: consumers signal, the 2 idle producer warps reduce (default); 2 consumer (wait per panel + acquire);
                             // 3 producer with the reduce done by the consumer warp groups themselves, deferred by one tile (A/B only)
                             // 4 producer: consumers signal only; a separate reducer kernel (tp4_reduce_kernel) on reserved SMs reduces
                             // 5 consumer, v5 protocol: wait per panel on a value flag (panel_cnt[m] >= epoch) + acquire
                             // 7 producer, v5 protocol: TMA-store tile (m, n) into owner (m % world)'s inbox slot, value flag into the owner
                             // 8 consumer, v5 protocol, role-switching (PDL dependent of the mode-7 producer): the first R CTAs run the
                             //   owner reducer (tp4_role_reduce) before their GEMM tiles; otherwise as mode 5
  int dbg = 0;               // ablation bitmask (timing experiments only, breaks correctness): see kDbg* below
  int rank = 0, world = 1, epoch = 0;
  int tiles_n = 0;           // producer: N tiles of this GEMM (tile id = m * tiles_n + n)
  int panel_target = 0;      // consumer: per-epoch completion count of a panel (= producer tiles_n * world)
  int ld = 0;                // leading dimension (elements) of partials / x / resid (= N1)
  int bm = 0, bn = 0;        // producer tile shape
  unsigned* tile_cnt_mc = nullptr;      // multicast [tm*tn]; stored partials, target epoch*world
  const unsigned* tile_cnt = nullptr;   // local copy
  const uint4* partials_mc = nullptr;   // multicast; every rank's partial D (M x ld bf16)
  uint4* x_mc = nullptr;                // multicast; reduced x (M x ld bf16)
  const uint4* resid = nullptr;         // local, optional residual added before the store (M x ld bf16)
  float* rowss_mc = nullptr;            // multicast [M]; sum of squares of the reduced rows (this epoch's parity buffer)
  unsigned* panel_cnt_mc = nullptr;     // multicast [tm]; slice reductions done, target epoch*tiles_n*world
  const unsigned* panel_cnt = nullptr;  // consumer: local copy of the producer's panel_cnt
  const int* epoch_ptr = nullptr;       // optional device-side epoch (CUDA-graph capturable); overrides `epoch`
  float* rowss_mc_base = nullptr;       // reducer kernels: if set, rowss_mc = base + (epoch & 1) * rowss_stride (elements)
  int rowss_stride = 0;
  const uint4* peer_partials[8] = {};   // mode 4u: unicast address of every rank's partials copy (only [0, world) used)
  uint4* peer_x[8] = {};                // mode 4u: unicast address of every rank's x copy
  // mode 7 (v5 producer scatter)
  unsigned* peer_tile_flags[8] = {};    // tile_flags array at every rank (unicast): [src][panels_per_owner][tiles_n], value = epoch
  void* dst_ptr[4] = {};                // TMA destination per owner rank: that rank's inbox slot for THIS (source) rank, dst_rows x ld
  int dst_rows = 0;                     // rows of an inbox slot (= panels_per_owner * bm)
  int panels_per_owner = 0;             // ceil(max_M / bm / world)
  // Phase 6 S3 (2-kernel boundary via programmatic dependent launch)
  int tail_cols = 0;                    // S3 producer tile order (mode 7, raster 1 / swizzle 1 only): 0 = stock m-fast order; C > 0 =
                                        // m-fast over the first tiles_n - C n-tiles, then the last C n-tiles panel-major (tp4_remap_tile)
  int tiles_m = 0;                      // tail_cols: M tiles of this GEMM
  int pdl_trigger = 0;                  // every CTA executes griddepcontrol.launch_dependents right after the prologue (patch 9a)
  int R = 0;                            // mode 8: the first R CTAs to start (role_ctr) run the v5 owner reducer, then their GEMM share
  int* role_ctr = nullptr;              // mode 8: CTA arrival counter, zeroed by tp4_step
  unsigned long long* trace = nullptr;  // optional %globaltimer stamps per CTA, [grid][8] (see tp4_trace)
  Tp4Red5Params red5{};                 // mode 8: role-reducer parameters
};

__device__ __forceinline__ int tp4_epoch(CtappTp4Params const& p) { return p.epoch_ptr ? __ldcg(p.epoch_ptr) : p.epoch; }
// Reducer kernels: read the epoch once and derive the rowss parity slice, so the rest of the kernel uses p.epoch / p.rowss_mc.
__device__ __forceinline__ void tp4_resolve(CtappTp4Params& p) {
  const int e = tp4_epoch(p);
  p.epoch = e; p.epoch_ptr = nullptr;
  if (p.rowss_mc_base) p.rowss_mc = p.rowss_mc_base + (e & 1) * p.rowss_stride;
}

namespace tp4 {

enum : int { kDbgNoTileWait = 1, kDbgNoLdReduce = 2, kDbgNoMcStore = 4, kDbgNoRowss = 8, kDbgNoPanelSignal = 16,
             kDbgNoStoreDrain = 32, kDbgNoReduce = 64, kDbgNoTmaWait = 128, kDbgNoEpiAcquire = 256 };
static constexpr int kThreads = 256;   // the two consumer (MMA/epilogue) warp groups of the cooperative kernel
// Named barrier ids are *user* ids: NamedBarrier::sync(n, uint32_t id) adds ReservedNamedBarrierCount (8), and the hardware
// has 16 barriers, so user ids must be 0..7. (Passing FirstUserBarrier = 8 here wrapped to hardware barrier 0 / 1 and collided
// with __syncthreads and the epilogue barrier: sporadic illegal-instruction faults.)
static constexpr uint32_t kBarrier = 0;         // consumer warp groups (256 threads)
static constexpr int kRedThreads = 64;          // the two idle producer warps (Epilogue-load and MainloopAux) act as the reducer
static constexpr uint32_t kRedBarrier = 1;      // reducer warps (64 threads)

__device__ __forceinline__ unsigned ld_acquire_sys(const unsigned* p) {
  unsigned v; asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ unsigned long long globaltimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
static constexpr unsigned kSpinLimit = 1u << 28;   // ~seconds; a protocol bug traps instead of hanging the job

__device__ __forceinline__ void spin_ge(const unsigned* p, unsigned target) {
  unsigned n = 0;
  while (ld_acquire_sys(p) < target) { if (++n == kSpinLimit) __trap(); }
}
__device__ __forceinline__ void fence_acq_rel_sys() { asm volatile("fence.acq_rel.sys;" ::: "memory"); }
__device__ __forceinline__ void st_relaxed_sys_u32(unsigned* p, unsigned v) { asm volatile("st.relaxed.sys.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void st_relaxed_sys_f32x4(float* p, float4 v) {
  asm volatile("st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};" :: "l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.global;" ::: "memory"); }
__device__ __forceinline__ void bar_sync() { cutlass::arch::NamedBarrier::sync(kThreads, kBarrier); }

__device__ __forceinline__ void mm_red_release_add_u32(unsigned* mc, unsigned v) {
  asm volatile("multimem.red.release.sys.global.add.u32 [%0], %1;" :: "l"(mc), "r"(v) : "memory");
}
__device__ __forceinline__ void mm_red_add_f32(float* mc, float v) {
  asm volatile("multimem.red.relaxed.sys.global.add.f32 [%0], %1;" :: "l"(mc), "f"(v) : "memory");
}
__device__ __forceinline__ uint4 mm_ld_reduce_bf16x8(const uint4* mc) {
  uint4 v;
  asm volatile("multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(mc) : "memory");
  return v;
}
__device__ __forceinline__ void mm_red_add_bf16x8(uint4* mc, uint4 v) {
  asm volatile("multimem.red.relaxed.sys.global.add.v4.bf16x2 [%0], {%1,%2,%3,%4};" :: "l"(mc), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ void mm_st_bf16x8(uint4* mc, uint4 v) {
  asm volatile("multimem.st.relaxed.sys.global.v4.bf16x2 [%0], {%1,%2,%3,%4};" :: "l"(mc), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

__device__ __forceinline__ float2 unpack(unsigned u) {
  __nv_bfloat162 b = *reinterpret_cast<__nv_bfloat162*>(&u); return __bfloat1622float2(b);
}
__device__ __forceinline__ unsigned pack(float lo, float hi) {
  __nv_bfloat162 b = __floats2bfloat162_rn(lo, hi); return *reinterpret_cast<unsigned*>(&b);
}

// Reduce this rank's slice of rows of producer tile (m, n). All kThreads consumer threads must call it.
__device__ __forceinline__ void reduce_slice(CtappTp4Params const& p, int m, int n, int tid) {
  const int tile = m * p.tiles_n + n;
  if (tid == 0 && !(p.dbg & kDbgNoTileWait)) {
    const unsigned target = static_cast<unsigned>(tp4_epoch(p)) * static_cast<unsigned>(p.world);
    spin_ge(p.tile_cnt + tile, target);
  }
  bar_sync();
  const int rpr = p.bm / p.world;                 // rows per rank (32 for 128-row tiles at TP4)
  const int tpr = kThreads / rpr;                 // threads per row
  const int cols = p.bn / tpr;                    // columns per thread (multiple of 8)
  const int row = m * p.bm + p.rank * rpr + tid / tpr;
  const int col = n * p.bn + (tid % tpr) * cols;
  const size_t off = (static_cast<size_t>(row) * p.ld + col) / 8;   // uint4 index
  float ss = 0.f;
  #pragma unroll 4
  for (int c = 0; c < cols / 8; c++) {
    uint4 v = (p.dbg & kDbgNoLdReduce) ? make_uint4(tid, c, 0, 0) : mm_ld_reduce_bf16x8(p.partials_mc + off + c);
    if (p.resid) {
      uint4 r = __ldg(p.resid + off + c);
      float2 a0 = unpack(v.x), a1 = unpack(v.y), a2 = unpack(v.z), a3 = unpack(v.w);
      float2 r0 = unpack(r.x), r1 = unpack(r.y), r2 = unpack(r.z), r3 = unpack(r.w);
      v.x = pack(a0.x + r0.x, a0.y + r0.y); v.y = pack(a1.x + r1.x, a1.y + r1.y);
      v.z = pack(a2.x + r2.x, a2.y + r2.y); v.w = pack(a3.x + r3.x, a3.y + r3.y);
    }
    if (!(p.dbg & kDbgNoMcStore)) mm_st_bf16x8(p.x_mc + off + c, v);
    float2 f;
    f = unpack(v.x); ss += f.x * f.x + f.y * f.y; f = unpack(v.y); ss += f.x * f.x + f.y * f.y;
    f = unpack(v.z); ss += f.x * f.x + f.y * f.y; f = unpack(v.w); ss += f.x * f.x + f.y * f.y;
  }
  for (int o = tpr / 2; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);   // tpr lanes of a row are contiguous
  if (tid % tpr == 0 && !(p.dbg & kDbgNoRowss)) mm_red_add_f32(p.rowss_mc + row, ss);
  bar_sync();
  if (tid == 0 && !(p.dbg & kDbgNoPanelSignal)) { fence_acq_rel_sys(); mm_red_release_add_u32(p.panel_cnt_mc + m, 1u); }
}

// Reducer warps (64 threads, 2 per row): wait until every rank stored tile (m, n), then reduce this rank's 32-row slice.
// Register-lean on purpose: these warps run under the producer warp group's setmaxnreg budget (40 registers); exceeding it
// faults or hangs the kernel. kRedBatch x 16 B in flight per thread.
static constexpr int kRedBatch = 4;

__device__ __forceinline__ float sumsq_add(uint4 v, float ss) {
  float2 f;
  f = unpack(v.x); ss += f.x * f.x + f.y * f.y; f = unpack(v.y); ss += f.x * f.x + f.y * f.y;
  f = unpack(v.z); ss += f.x * f.x + f.y * f.y; f = unpack(v.w); ss += f.x * f.x + f.y * f.y;
  return ss;
}
__device__ __forceinline__ uint4 add_bf16x8(uint4 v, uint4 r) {
  float2 a0 = unpack(v.x), a1 = unpack(v.y), a2 = unpack(v.z), a3 = unpack(v.w);
  float2 r0 = unpack(r.x), r1 = unpack(r.y), r2 = unpack(r.z), r3 = unpack(r.w);
  return make_uint4(pack(a0.x + r0.x, a0.y + r0.y), pack(a1.x + r1.x, a1.y + r1.y), pack(a2.x + r2.x, a2.y + r2.y), pack(a3.x + r3.x, a3.y + r3.y));
}

__device__ __forceinline__ void reduce_slice64(CtappTp4Params const& p, int m, int n, int tid) {
  if ((tid & 31) == 0 && !(p.dbg & kDbgNoTileWait))
    spin_ge(p.tile_cnt + m * p.tiles_n + n, static_cast<unsigned>(tp4_epoch(p)) * static_cast<unsigned>(p.world));
  __syncwarp();
  const int rpr = p.bm / p.world;                 // rows per rank
  const int tpr = kRedThreads / rpr;              // threads per row (2 at TP4, 1 at TP8, 4 at TP2)
  const int cols = p.bn / tpr;                    // columns per thread
  const int row = m * p.bm + p.rank * rpr + tid / tpr;
  const size_t off = (static_cast<size_t>(row) * p.ld + n * p.bn + (tid % tpr) * cols) / 8;
  float ss = 0.f;
  #pragma unroll 1
  for (int c = 0; c < cols / 8; c += kRedBatch) {
    uint4 v[kRedBatch];
    #pragma unroll
    for (int j = 0; j < kRedBatch; j++) v[j] = (p.dbg & kDbgNoLdReduce) ? make_uint4(tid, c, 0, 0) : mm_ld_reduce_bf16x8(p.partials_mc + off + c + j);
    #pragma unroll
    for (int j = 0; j < kRedBatch; j++) {
      if (p.resid) v[j] = add_bf16x8(v[j], __ldg(p.resid + off + c + j));
      if (!(p.dbg & kDbgNoMcStore)) mm_st_bf16x8(p.x_mc + off + c + j, v[j]);
      ss = sumsq_add(v[j], ss);
    }
  }
  for (int o = tpr / 2; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if (tid % tpr == 0 && !(p.dbg & kDbgNoRowss)) mm_red_add_f32(p.rowss_mc + row, ss);
  cutlass::arch::NamedBarrier::sync(kRedThreads, kRedBarrier);
  if (tid == 0 && !(p.dbg & kDbgNoPanelSignal)) { fence_acq_rel_sys(); mm_red_release_add_u32(p.panel_cnt_mc + m, 1u); }
}

}  // namespace tp4

// --- hooks called from the patched cooperative kernel (csrc/sm90_gemm_coop_ctapp.hpp) ---

// CTAPP_TRACE stamps (p.trace != nullptr; buffer zeroed by the host before the traced launch), per CTA (linear block id; the
// AlongN raster launches a (1, G, 1) grid) 8 u64 slots: 0 CTA entry, 1 prologue done (after the pipeline-init __syncthreads),
// 2 first / 3 last mainloop tile start (TMA warp, after the panel wait), 4 / 5 role-reducer start / end (mode 8 role CTAs),
// 6 consumer warp groups done (after the last tile's store + signal), 7 = %smid | (role + 1) << 32 (0 = no role).
__device__ __forceinline__ int tp4_cta_id() { return (int)(blockIdx.x + blockIdx.y * gridDim.x); }
__device__ __forceinline__ unsigned tp4_smid() { unsigned r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r; }
__device__ __forceinline__ void tp4_trace(CtappTp4Params const& p, int k) {   // caller selects the thread
  if (p.trace) p.trace[tp4_cta_id() * 8 + k] = tp4::globaltimer();
}
__device__ __forceinline__ void tp4_trace_entry(CtappTp4Params const& p) {   // thread 0, top of the kernel: slots 0 and 7
  if (p.trace) { p.trace[tp4_cta_id() * 8] = tp4::globaltimer(); p.trace[tp4_cta_id() * 8 + 7] = tp4_smid(); }
}

// Mainloop producer warp, before issuing the TMA loads of tile (m_idx, *): the reduced panel must be complete.
__device__ __forceinline__ void tp4_wait_panel(CtappTp4Params const& p, int m_idx) {
  if (p.mode == 2 || p.mode == 5 || p.mode == 8) {
    const unsigned target = static_cast<unsigned>(tp4_epoch(p)) * static_cast<unsigned>(p.mode != 2 ? 1 : p.panel_target);
    if ((threadIdx.x & 31) == 0 && !(p.dbg & tp4::kDbgNoTmaWait)) tp4::spin_ge(p.panel_cnt + m_idx, target);
    __syncwarp();
    tp4::fence_proxy_async();
  }
  if (p.trace && (threadIdx.x & 31) == 0) {
    unsigned long long* t = p.trace + tp4_cta_id() * 8;
    const unsigned long long now = tp4::globaltimer();
    if (t[2] == 0) t[2] = now;
    t[3] = now;
  }
}

// Consumer warp groups, before the epilogue: every thread acquires the panel so its rowss loads see the remote writes.
__device__ __forceinline__ void tp4_before_epilogue(CtappTp4Params const& p, int m_idx) {
  if (p.mode != 2 && p.mode != 5 && p.mode != 8) return;
  if (p.dbg & tp4::kDbgNoEpiAcquire) return;
  const unsigned target = static_cast<unsigned>(tp4_epoch(p)) * static_cast<unsigned>(p.mode != 2 ? 1 : p.panel_target);
  tp4::spin_ge(p.panel_cnt + m_idx, target);
}

// Consumer warp groups, after collective_epilogue.store() of tile (m, n): signal it, then reduce the previous tile's slice.
__device__ __forceinline__ void tp4_after_store(CtappTp4Params const& p, int m, int n, int& prev_m, int& prev_n, int tid) {
  if (p.mode == 7) {   // v5: drain this tile's TMA stores (owner's inbox, possibly remote), then a unicast value flag at the owner
    if (tid < 32 && !(p.dbg & tp4::kDbgNoStoreDrain)) { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); tp4::fence_proxy_async(); }
    tp4::bar_sync();
    if (tid == 0) {
      const int owner = m % p.world;
      tp4::fence_acq_rel_sys();
      tp4::st_relaxed_sys_u32(p.peer_tile_flags[owner] + (p.rank * p.panels_per_owner + m / p.world) * p.tiles_n + n,
                              static_cast<unsigned>(tp4_epoch(p)));
    }
    return;
  }
  if (p.mode != 1 && p.mode != 3 && p.mode != 4) return;
  if (tid < 32 && !(p.dbg & tp4::kDbgNoStoreDrain)) { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); tp4::fence_proxy_async(); }   // the TMA-issuing warp
  tp4::bar_sync();
  if (tid == 0) { tp4::fence_acq_rel_sys(); tp4::mm_red_release_add_u32(p.tile_cnt_mc + m * p.tiles_n + n, 1u); }
  if (p.mode == 3 && prev_m >= 0 && !(p.dbg & tp4::kDbgNoReduce)) tp4::reduce_slice(p, prev_m, prev_n, tid);
  prev_m = m; prev_n = n;
}

// Consumer warp groups, after the work loop: reduce the slice of the last tile this CTA stored.
__device__ __forceinline__ void tp4_finish(CtappTp4Params const& p, int prev_m, int prev_n, int tid) {
  if (tid == 0) tp4_trace(p, 6);
  if (p.mode != 3 || prev_m < 0 || (p.dbg & tp4::kDbgNoReduce)) return;
  tp4::reduce_slice(p, prev_m, prev_n, tid);
}

// The two idle producer warps (Epilogue-load warp when there is no source C, MainloopAux warp): walk the CTA's tile sequence
// and reduce this rank's slice of every tile as soon as all ranks have stored it. Never on the MMA critical path; the
// consumers never wait on these warps, so the wait here cannot deadlock (every rank's consumers store tile k unconditionally).
template <class Scheduler, class WorkTileInfo>
__device__ __forceinline__ void tp4_reducer(CtappTp4Params const& p, Scheduler& scheduler, WorkTileInfo work_tile_info, int warp_in_pair, int lane) {
  const int tid = warp_in_pair * 32 + lane;
  while (work_tile_info.is_valid()) {
    if (!(p.dbg & tp4::kDbgNoReduce)) tp4::reduce_slice64(p, work_tile_info.M_idx, work_tile_info.N_idx, tid);
    auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info);
    work_tile_info = next_work_tile_info;
  }
}

// Stand-alone reducer kernel (mode 4): runs concurrently with the producer GEMM on SMs the GEMM leaves free (sm_count = SMs - R).
// Block b handles tiles b, b + R, ... in the GEMM's linear scheduler order (raster 1 = AlongN: m fast; 2 = AlongM: n fast; no
// swizzle). 512 threads: 16 per 32-row slice row, 2 x 16 B each in flight.
__global__ void __launch_bounds__(512) tp4_reduce_kernel(CtappTp4Params p, int raster, int tiles_m) {
  tp4_resolve(p);
  const int tiles = tiles_m * p.tiles_n;
  const int rpr = p.bm / p.world, tpr = 512 / rpr, cols = p.bn / tpr;   // 32 rows, 16 threads per row, 16 columns per thread
  const int tid = threadIdx.x;
  for (int L = blockIdx.x; L < tiles; L += gridDim.x) {
    const int m = raster == 2 ? L / p.tiles_n : L % tiles_m;
    const int n = raster == 2 ? L % p.tiles_n : L / tiles_m;
    if (tid == 0 && !(p.dbg & tp4::kDbgNoTileWait))
      tp4::spin_ge(p.tile_cnt + m * p.tiles_n + n, static_cast<unsigned>(tp4_epoch(p)) * static_cast<unsigned>(p.world));
    __syncthreads();
    const int row = m * p.bm + p.rank * rpr + tid / tpr;
    const size_t off = (static_cast<size_t>(row) * p.ld + n * p.bn + (tid % tpr) * cols) / 8;
    uint4 v0 = (p.dbg & tp4::kDbgNoLdReduce) ? make_uint4(tid, 0, 0, 0) : tp4::mm_ld_reduce_bf16x8(p.partials_mc + off);
    uint4 v1 = (p.dbg & tp4::kDbgNoLdReduce) ? make_uint4(tid, 1, 0, 0) : tp4::mm_ld_reduce_bf16x8(p.partials_mc + off + 1);
    if (p.resid) { v0 = tp4::add_bf16x8(v0, __ldg(p.resid + off)); v1 = tp4::add_bf16x8(v1, __ldg(p.resid + off + 1)); }
    if (!(p.dbg & tp4::kDbgNoMcStore)) { tp4::mm_st_bf16x8(p.x_mc + off, v0); tp4::mm_st_bf16x8(p.x_mc + off + 1, v1); }
    float ss = tp4::sumsq_add(v1, tp4::sumsq_add(v0, 0.f));
    for (int o = tpr / 2; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if (tid % tpr == 0 && !(p.dbg & tp4::kDbgNoRowss)) tp4::mm_red_add_f32(p.rowss_mc + row, ss);
    __syncthreads();
    if (tid == 0 && !(p.dbg & tp4::kDbgNoPanelSignal)) { tp4::fence_acq_rel_sys(); tp4::mm_red_release_add_u32(p.panel_cnt_mc + m, 1u); }
  }
}

// Mode 4u: same structure as tp4_reduce_kernel, but the 4 ranks' partials are read with plain (relaxed.sys) loads over NVLink
// unicast and x is written to every rank's copy with plain stores; counters / rowss stay multimem.
__device__ __forceinline__ uint4 ld_relaxed_sys_v4(const uint4* p) {
  uint4 v; asm volatile("ld.relaxed.sys.global.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ void st_relaxed_sys_v4(uint4* p, uint4 v) {
  asm volatile("st.relaxed.sys.global.v4.b32 [%0], {%1,%2,%3,%4};" :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__global__ void __launch_bounds__(512) tp4_reduce_unicast_kernel(CtappTp4Params p, int raster, int tiles_m) {
  tp4_resolve(p);
  const int tiles = tiles_m * p.tiles_n;
  const int rpr = p.bm / p.world, tpr = 512 / rpr, cols = p.bn / tpr;
  const int tid = threadIdx.x;
  for (int L = blockIdx.x; L < tiles; L += gridDim.x) {
    const int m = raster == 2 ? L / p.tiles_n : L % tiles_m;
    const int n = raster == 2 ? L % p.tiles_n : L / tiles_m;
    if (tid == 0 && !(p.dbg & tp4::kDbgNoTileWait))
      tp4::spin_ge(p.tile_cnt + m * p.tiles_n + n, static_cast<unsigned>(tp4_epoch(p)) * static_cast<unsigned>(p.world));
    __syncthreads();
    const int row = m * p.bm + p.rank * rpr + tid / tpr;
    const size_t off = (static_cast<size_t>(row) * p.ld + n * p.bn + (tid % tpr) * cols) / 8;
    float ss = 0.f;
    #pragma unroll
    for (int j = 0; j < 2; j++) {
      float a[8];
      #pragma unroll
      for (int i = 0; i < 8; i++) a[i] = 0.f;
      if (p.dbg & tp4::kDbgNoLdReduce) { a[0] = tid; a[1] = j; }
      else {
        for (int r = 0; r < p.world; r++) {
          uint4 t = ld_relaxed_sys_v4(p.peer_partials[r] + off + j);
          float2 f0 = tp4::unpack(t.x), f1 = tp4::unpack(t.y), f2 = tp4::unpack(t.z), f3 = tp4::unpack(t.w);
          a[0] += f0.x; a[1] += f0.y; a[2] += f1.x; a[3] += f1.y; a[4] += f2.x; a[5] += f2.y; a[6] += f3.x; a[7] += f3.y;
        }
      }
      if (p.resid) {
        uint4 rr = __ldg(p.resid + off + j);
        float2 r0 = tp4::unpack(rr.x), r1 = tp4::unpack(rr.y), r2 = tp4::unpack(rr.z), r3 = tp4::unpack(rr.w);
        a[0] += r0.x; a[1] += r0.y; a[2] += r1.x; a[3] += r1.y; a[4] += r2.x; a[5] += r2.y; a[6] += r3.x; a[7] += r3.y;
      }
      uint4 v = make_uint4(tp4::pack(a[0], a[1]), tp4::pack(a[2], a[3]), tp4::pack(a[4], a[5]), tp4::pack(a[6], a[7]));
      if (!(p.dbg & tp4::kDbgNoMcStore))
        for (int r = 0; r < p.world; r++) st_relaxed_sys_v4(p.peer_x[r] + off + j, v);
      ss = tp4::sumsq_add(v, ss);
    }
    for (int o = tpr / 2; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if (tid % tpr == 0 && !(p.dbg & tp4::kDbgNoRowss)) tp4::mm_red_add_f32(p.rowss_mc + row, ss);
    __syncthreads();
    if (tid == 0 && !(p.dbg & tp4::kDbgNoPanelSignal)) { tp4::fence_acq_rel_sys(); tp4::mm_red_release_add_u32(p.panel_cnt_mc + m, 1u); }
  }
}

// Reducer v2: two tiles per iteration (4 independent multimem.ld_reduce in flight per thread) and the fence + panel signal of
// iteration i is issued at the end of iteration i+1 (after its loads/stores are in flight), plus a final flush after the loop.
__global__ void __launch_bounds__(512) tp4_reduce2_kernel(CtappTp4Params p, int raster, int tiles_m) {
  tp4_resolve(p);
  const int tiles = tiles_m * p.tiles_n;
  const int rpr = p.bm / p.world, tpr = 512 / rpr, cols = p.bn / tpr;
  const int tid = threadIdx.x;
  const unsigned tgt = static_cast<unsigned>(tp4_epoch(p)) * static_cast<unsigned>(p.world);
  int prev_m0 = -1, prev_m1 = -1; bool prev_has1 = false;
  for (int L0 = blockIdx.x * 2; L0 < tiles; L0 += gridDim.x * 2) {
    const int L1 = L0 + 1;
    const bool has1 = L1 < tiles;
    const int m0 = raster == 2 ? L0 / p.tiles_n : L0 % tiles_m, n0 = raster == 2 ? L0 % p.tiles_n : L0 / tiles_m;
    const int m1 = has1 ? (raster == 2 ? L1 / p.tiles_n : L1 % tiles_m) : m0;
    const int n1 = has1 ? (raster == 2 ? L1 % p.tiles_n : L1 / tiles_m) : n0;
    if (tid == 0 && !(p.dbg & tp4::kDbgNoTileWait)) {
      tp4::spin_ge(p.tile_cnt + m0 * p.tiles_n + n0, tgt);
      if (has1) tp4::spin_ge(p.tile_cnt + m1 * p.tiles_n + n1, tgt);
    }
    __syncthreads();
    const int rl = tid / tpr, cl = (tid % tpr) * cols;
    const int row0 = m0 * p.bm + p.rank * rpr + rl, row1 = m1 * p.bm + p.rank * rpr + rl;
    const size_t off0 = (static_cast<size_t>(row0) * p.ld + n0 * p.bn + cl) / 8;
    const size_t off1 = (static_cast<size_t>(row1) * p.ld + n1 * p.bn + cl) / 8;
    uint4 v0, v1, v2 = make_uint4(0, 0, 0, 0), v3 = make_uint4(0, 0, 0, 0);
    if (p.dbg & tp4::kDbgNoLdReduce) { v0 = make_uint4(tid, 0, 0, 0); v1 = make_uint4(tid, 1, 0, 0); v2 = make_uint4(tid, 2, 0, 0); v3 = make_uint4(tid, 3, 0, 0); }
    else {
      v0 = tp4::mm_ld_reduce_bf16x8(p.partials_mc + off0);
      v1 = tp4::mm_ld_reduce_bf16x8(p.partials_mc + off0 + 1);
      if (has1) { v2 = tp4::mm_ld_reduce_bf16x8(p.partials_mc + off1); v3 = tp4::mm_ld_reduce_bf16x8(p.partials_mc + off1 + 1); }
    }
    if (p.resid) {
      v0 = tp4::add_bf16x8(v0, __ldg(p.resid + off0)); v1 = tp4::add_bf16x8(v1, __ldg(p.resid + off0 + 1));
      if (has1) { v2 = tp4::add_bf16x8(v2, __ldg(p.resid + off1)); v3 = tp4::add_bf16x8(v3, __ldg(p.resid + off1 + 1)); }
    }
    if (!(p.dbg & tp4::kDbgNoMcStore)) {
      tp4::mm_st_bf16x8(p.x_mc + off0, v0); tp4::mm_st_bf16x8(p.x_mc + off0 + 1, v1);
      if (has1) { tp4::mm_st_bf16x8(p.x_mc + off1, v2); tp4::mm_st_bf16x8(p.x_mc + off1 + 1, v3); }
    }
    float ssa = tp4::sumsq_add(v1, tp4::sumsq_add(v0, 0.f));
    float ssb = tp4::sumsq_add(v3, tp4::sumsq_add(v2, 0.f));
    for (int o = tpr / 2; o > 0; o >>= 1) { ssa += __shfl_xor_sync(0xffffffffu, ssa, o); ssb += __shfl_xor_sync(0xffffffffu, ssb, o); }
    if (tid % tpr == 0 && !(p.dbg & tp4::kDbgNoRowss)) {
      tp4::mm_red_add_f32(p.rowss_mc + row0, ssa);
      if (has1) tp4::mm_red_add_f32(p.rowss_mc + row1, ssb);
    }
    __syncthreads();
    if (tid == 0 && prev_m0 >= 0 && !(p.dbg & tp4::kDbgNoPanelSignal)) {
      tp4::fence_acq_rel_sys();
      tp4::mm_red_release_add_u32(p.panel_cnt_mc + prev_m0, 1u);
      if (prev_has1) tp4::mm_red_release_add_u32(p.panel_cnt_mc + prev_m1, 1u);
    }
    prev_m0 = m0; prev_m1 = m1; prev_has1 = has1;
  }
  if (tid == 0 && prev_m0 >= 0 && !(p.dbg & tp4::kDbgNoPanelSignal)) {
    tp4::fence_acq_rel_sys();
    tp4::mm_red_release_add_u32(p.panel_cnt_mc + prev_m0, 1u);
    if (prev_has1) tp4::mm_red_release_add_u32(p.panel_cnt_mc + prev_m1, 1u);
  }
}

// Reducer v3: 17 warps. Warps 0..15 are workers (T tiles per iteration, all 2*T multimem.ld_reduce issued up front); warp 16 is a
// signaller that fences + signals panel counters for iteration i while the workers already run iteration i+1.
// Barriers (user ids): 1 = workers only (512); 2 = "iteration done" (512 worker arrive + 32 signaller sync, 544);
// 3 = "iteration released" (32 signaller arrive + 512 worker sync, 544). Workers run at most one iteration ahead.
template <int T>
__global__ void __launch_bounds__(544) tp4_reduce3_kernel(CtappTp4Params p, int raster, int tiles_m) {
  tp4_resolve(p);
  const int tiles = tiles_m * p.tiles_n;
  const int rpr = p.bm / p.world, tpr = 512 / rpr, cols = p.bn / tpr;
  const int tid = threadIdx.x;
  const unsigned tgt = static_cast<unsigned>(tp4_epoch(p)) * static_cast<unsigned>(p.world);
  int iters = 0;
  for (int L0 = blockIdx.x * T; L0 < tiles; L0 += gridDim.x * T) iters++;
  if (tid >= 512) {
    cutlass::arch::NamedBarrier::arrive(544, 3);
    for (int i = 0; i < iters; i++) {
      cutlass::arch::NamedBarrier::sync(544, 2);
      if (i + 1 < iters) cutlass::arch::NamedBarrier::arrive(544, 3);
      if (tid == 512 && !(p.dbg & tp4::kDbgNoPanelSignal)) {
        tp4::fence_acq_rel_sys();
        const int L0 = (i * gridDim.x + blockIdx.x) * T;
        #pragma unroll
        for (int t = 0; t < T; t++) {
          const int L = L0 + t;
          if (L < tiles) tp4::mm_red_release_add_u32(p.panel_cnt_mc + (raster == 2 ? L / p.tiles_n : L % tiles_m), 1u);
        }
      }
      __syncwarp();
    }
    return;
  }
  for (int i = 0; i < iters; i++) {
    cutlass::arch::NamedBarrier::sync(544, 3);
    const int L0 = (i * gridDim.x + blockIdx.x) * T;
    if (tid == 0 && !(p.dbg & tp4::kDbgNoTileWait)) {
      #pragma unroll
      for (int t = 0; t < T; t++) {
        const int L = L0 + t;
        if (L < tiles) {
          const int m = raster == 2 ? L / p.tiles_n : L % tiles_m, n = raster == 2 ? L % p.tiles_n : L / tiles_m;
          tp4::spin_ge(p.tile_cnt + m * p.tiles_n + n, tgt);
        }
      }
    }
    cutlass::arch::NamedBarrier::sync(512, 1);
    const int rl = tid / tpr, cl = (tid % tpr) * cols;
    size_t off[T]; int row[T]; bool ok[T];
    uint4 v[T][2];
    #pragma unroll
    for (int t = 0; t < T; t++) {
      const int L = L0 + t;
      ok[t] = L < tiles;
      const int Lc = ok[t] ? L : L0;
      const int m = raster == 2 ? Lc / p.tiles_n : Lc % tiles_m, n = raster == 2 ? Lc % p.tiles_n : Lc / tiles_m;
      row[t] = m * p.bm + p.rank * rpr + rl;
      off[t] = (static_cast<size_t>(row[t]) * p.ld + n * p.bn + cl) / 8;
      v[t][0] = make_uint4(tid, 2 * t, 0, 0); v[t][1] = make_uint4(tid, 2 * t + 1, 0, 0);
    }
    if (!(p.dbg & tp4::kDbgNoLdReduce)) {
      #pragma unroll
      for (int t = 0; t < T; t++) if (ok[t]) {
        v[t][0] = tp4::mm_ld_reduce_bf16x8(p.partials_mc + off[t]);
        v[t][1] = tp4::mm_ld_reduce_bf16x8(p.partials_mc + off[t] + 1);
      }
    }
    #pragma unroll
    for (int t = 0; t < T; t++) if (ok[t]) {
      if (p.resid) { v[t][0] = tp4::add_bf16x8(v[t][0], __ldg(p.resid + off[t])); v[t][1] = tp4::add_bf16x8(v[t][1], __ldg(p.resid + off[t] + 1)); }
      if (!(p.dbg & tp4::kDbgNoMcStore)) { tp4::mm_st_bf16x8(p.x_mc + off[t], v[t][0]); tp4::mm_st_bf16x8(p.x_mc + off[t] + 1, v[t][1]); }
    }
    float ss[T];
    #pragma unroll
    for (int t = 0; t < T; t++) ss[t] = tp4::sumsq_add(v[t][1], tp4::sumsq_add(v[t][0], 0.f));
    #pragma unroll
    for (int o = tpr / 2; o > 0; o >>= 1)
      #pragma unroll
      for (int t = 0; t < T; t++) ss[t] += __shfl_xor_sync(0xffffffffu, ss[t], o);
    if (tid % tpr == 0 && !(p.dbg & tp4::kDbgNoRowss)) {
      #pragma unroll
      for (int t = 0; t < T; t++) if (ok[t]) tp4::mm_red_add_f32(p.rowss_mc + row[t], ss[t]);
    }
    cutlass::arch::NamedBarrier::arrive(544, 2);
  }
}

// ======================================================================================================================
// Phase 6, protocol v5 ("panel ownership"): rank r owns the 128-row panels m with m % 4 == r. The producer GEMM (mode 7) TMA-stores
// tile (m, n) of rank s into inbox_owner[s] at row block m / 4 and sets tile_flags_owner[s][m / 4][n] = epoch. The owner reducer
// below sums the 4 inbox slices + resid in fp32, stores x to all 4 ranks (unicast st.relaxed.sys), accumulates the row sum of
// squares locally, and the CTA completing a panel (last arriver on panel_done[m]) pushes the panel's 128 rowss to every rank and
// sets panel_flag[m] = epoch everywhere. All flags are values (= epoch), so one instance serves any M <= max_M.
// ======================================================================================================================
namespace tp4 {
static constexpr uint32_t kB5Done = 2, kB5Release = 3;   // user named-barrier ids (as v3)

// Unit u (of the Pr * 128 units of this rank) -> owned panel j (global panel 4 j + rank), n-tile, quarter q, in the producer's tile
// order: order 2 (n fast) = panel-major; order 1 (m fast) = for each group of swz n-tiles, for each owned panel, for each n of the
// group (CUTLASS AlongM raster with swizzle swz; swz = 1 gives n-major).
__device__ __forceinline__ void red5_map(Tp4Red5Params const& p, int Pr, int u, int& j, int& n, int& q) {
  q = u & 3;
  const int t = u >> 2;
  if (p.tail > 0) {                     // producer tail order (tp4_remap_tile): n-major head over 32 - tail n-tiles, then panel-major
    const int head = Pr * (32 - p.tail);
    if (t < head) { n = t / Pr; j = t - n * Pr; }
    else { const int t2 = t - head; j = t2 / p.tail; n = 32 - p.tail + (t2 - j * p.tail); }
    return;
  }
  if (p.order == 1) {
    if (p.swz <= 1) { n = t / Pr; j = t % Pr; }
    else { const int g = t / p.swz; j = g % Pr; n = (g / Pr) * p.swz + t % p.swz; }
  }
  else { j = u >> 7; n = (u >> 2) & 31; }
}
// Unit k of this CTA (global unit u = blockIdx.x + k * gridDim.x) -> owned panel j (global panel 4 j + rank), n-tile, quarter q.
__device__ __forceinline__ void red5_unit(Tp4Red5Params const& p, int Pr, int k, int& j, int& n, int& q) {
  red5_map(p, Pr, blockIdx.x + k * gridDim.x, j, n, q);
}

// Signaller warp, panel m complete (this CTA was the last arriver): push its 128 row sums of squares, then the flag, to every rank.
__device__ __forceinline__ void red5_publish(Tp4Red5Params const& p, int m, unsigned epoch, int lane) {
  if (lane == 0) fence_acq_rel_sys();   // acquire side of the panel_done chain (other CTAs: their stores -> fence -> atomic)
  __syncwarp();
  const float4 v = __ldcg(reinterpret_cast<const float4*>(p.rowss_local + m * 128) + lane);
  #pragma unroll
  for (int d = 0; d < 4; d++) st_relaxed_sys_f32x4(p.rowss[d] + m * 128 + lane * 4, v);
  __syncwarp();
  if (lane == 0) {
    fence_acq_rel_sys();                // x (all CTAs, via the chain) and rowss before the flags
    #pragma unroll
    for (int d = 0; d < 4; d++) st_relaxed_sys_u32(p.panel_flag[d] + m, epoch);
    if (p.trace) p.trace[m] = globaltimer();
  }
  __syncwarp();
}

// out = bf16(a0 + a1 + a2 + a3 + r), fp32 accumulation
__device__ __forceinline__ uint4 sum5_bf16x8(uint4 a0, uint4 a1, uint4 a2, uint4 a3, uint4 r) {
  unsigned o[4];
  const unsigned* x0 = &a0.x; const unsigned* x1 = &a1.x; const unsigned* x2 = &a2.x; const unsigned* x3 = &a3.x; const unsigned* xr = &r.x;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    float2 f0 = unpack(x0[i]), f1 = unpack(x1[i]), f2 = unpack(x2[i]), f3 = unpack(x3[i]), fr = unpack(xr[i]);
    o[i] = pack(f0.x + f1.x + f2.x + f3.x + fr.x, f0.y + f1.y + f2.y + f3.y + fr.y);
  }
  return make_uint4(o[0], o[1], o[2], o[3]);
}
}  // namespace tp4

// Owner reducer, R CTAs of THREADS threads: (THREADS - 32) / 32 worker warps + 1 signaller warp (v3 barrier pattern: 2 = batch
// done, 3 = batch released; workers run at most one batch ahead of the signaller). The CTA's units (32 rows x 256 cols each) are
// flattened into 32 row segments per unit; batch i gives every worker warp 2 consecutive row segments (same unit; one 16 B vector
// per lane per row -> 10 loads in flight per thread). Each warp waits for its unit's 4 tile flags itself (lanes 0-3, then
// __syncwarp). The signaller, per batch: one fence.acq_rel.sys if a unit completed, then atomicAdd(panel_done[m]) per completed
// unit; the last arriver of a panel publishes it (red5_publish).
template <int THREADS>
__global__ void __launch_bounds__(THREADS, THREADS <= 544 ? 2 : 1) tp4_reduce5_kernel(Tp4Red5Params p) {   // 544: 2 CTAs / SM (<= 56 regs)
  constexpr int NW = (THREADS - 32) / 32;   // worker warps
  constexpr int SEG = 2 * NW;               // row segments per batch
  const unsigned epoch = p.epoch_ptr ? static_cast<unsigned>(__ldcg(p.epoch_ptr)) : static_cast<unsigned>(p.epoch);
  const int P = p.m_valid / 128;
  const int Pr = P > p.rank ? (P - 1 - p.rank) / 4 + 1 : 0;                                   // panels owned by this rank
  const int U = Pr * 128;                                                                     // units of this rank
  const int nu = U > (int)blockIdx.x ? (U - 1 - (int)blockIdx.x) / (int)gridDim.x + 1 : 0;   // units of this CTA
  const int nseg = nu * 32;
  const int iters = (nseg + SEG - 1) / SEG;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (iters == 0) return;
  if (warp == NW) {                                     // signaller
    cutlass::arch::NamedBarrier::arrive(THREADS, tp4::kB5Release);
    int k_next = 0;                                     // next unit of this CTA to account
    for (int i = 0; i < iters; i++) {
      cutlass::arch::NamedBarrier::sync(THREADS, tp4::kB5Done);
      if (i + 1 < iters) cutlass::arch::NamedBarrier::arrive(THREADS, tp4::kB5Release);
      const int k_end = min((i + 1) * SEG, nseg) / 32;  // units [0, k_end) are complete
      if (k_next < k_end) {
        if (lane == 0) tp4::fence_acq_rel_sys();        // this CTA's x stores + rowss atomics (bar-ordered) before panel_done
        for (; k_next < k_end; k_next++) {
          int j, n, q;
          tp4::red5_unit(p, Pr, k_next, j, n, q);
          const int m = j * 4 + p.rank;
          unsigned last = 0;
          if (lane == 0) last = atomicAdd(p.panel_done + m, 1u) == 127u;   // 128 units per panel (32 n-tiles x 4 quarters)
          if (__shfl_sync(0xffffffffu, last, 0)) tp4::red5_publish(p, m, epoch, lane);
        }
      }
      __syncwarp();
    }
    return;
  }
  uint4* xo[4];                                         // destination order: rank+1, rank+2, rank+3, local
  #pragma unroll
  for (int d = 0; d < 4; d++) {
    const int r = (p.rank + 1 + d) & 3;
    uint4* ptr = p.x[0];
    #pragma unroll
    for (int i = 1; i < 4; i++) if (r == i) ptr = p.x[i];
    xo[d] = ptr;
  }
  for (int i = 0; i < iters; i++) {
    cutlass::arch::NamedBarrier::sync(THREADS, tp4::kB5Release);
    const int g = i * SEG + 2 * warp;                   // first of this warp's 2 row segments (g even -> same unit)
    if (g < nseg) {
      const int k = g >> 5, r = g & 31;
      int j, n, q;
      tp4::red5_unit(p, Pr, k, j, n, q);
      if (lane < 4) tp4::spin_ge(p.tile_flags + (lane * p.ppo + j) * 32 + n, epoch);
      __syncwarp();
      const int m = j * 4 + p.rank;
      const int xrow = m * 128 + q * 32 + r;
      const unsigned ioff = static_cast<unsigned>(j * 128 + q * 32 + r) * 1024u + n * 32 + lane;   // uint4 index
      const unsigned xoff = static_cast<unsigned>(xrow) * 1024u + n * 32 + lane;
      uint4 a[2][4], rv[2];
      #pragma unroll
      for (int rr = 0; rr < 2; rr++) {
        #pragma unroll
        for (int s = 0; s < 4; s++) a[rr][s] = __ldcg(p.inbox[s] + ioff + rr * 1024);
        rv[rr] = p.resid ? __ldg(p.resid + xoff + rr * 1024) : make_uint4(0, 0, 0, 0);
      }
      #pragma unroll
      for (int rr = 0; rr < 2; rr++) {
        const uint4 o = tp4::sum5_bf16x8(a[rr][0], a[rr][1], a[rr][2], a[rr][3], rv[rr]);
        #pragma unroll
        for (int d = 0; d < 4; d++) st_relaxed_sys_v4(xo[d] + xoff + rr * 1024, o);
        float ss = tp4::sumsq_add(o, 0.f);
        #pragma unroll
        for (int sh = 16; sh > 0; sh >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, sh);
        if (lane == 0) atomicAdd(p.rowss_local + xrow + rr, ss);
      }
    }
    cutlass::arch::NamedBarrier::arrive(THREADS, tp4::kB5Done);
  }
}

// Owner reducer, warp-per-unit loop (no CTA barrier, no signaller): the calling warp owns units u = first, first + stride, ...
// Per unit: lanes 0-3 wait the 4 tile flags, 32 / ROWS x (ROWS rows: 5 ROWS loads of 16 B in flight per lane (4 inbox slices +
// resid), sum, 4 unicast stores, sumsq -> lane 0 RED to rowss_local), then __syncwarp, lane 0 fence.acq_rel.sys + atomicAdd
// (panel_done[m]); the last arriver publishes the panel (red5_publish). Same memory-model chain as the signaller variant, with
// __syncwarp in place of the named barriers.
template <int ROWS>
__device__ __forceinline__ void red5w_loop(Tp4Red5Params const& p, int first, int stride) {
  static_assert(ROWS == 2 || ROWS == 4 || ROWS == 8, "ROWS");
  const unsigned epoch = p.epoch_ptr ? static_cast<unsigned>(__ldcg(p.epoch_ptr)) : static_cast<unsigned>(p.epoch);
  const int P = p.m_valid / 128;
  const int Pr = P > p.rank ? (P - 1 - p.rank) / 4 + 1 : 0;
  const int U = Pr * 128;
  const int lane = threadIdx.x & 31;
  uint4* xo[4];                                         // destination order: rank+1, rank+2, rank+3, local
  #pragma unroll
  for (int d = 0; d < 4; d++) {
    const int r = (p.rank + 1 + d) & 3;
    uint4* ptr = p.x[0];
    #pragma unroll
    for (int i = 1; i < 4; i++) if (r == i) ptr = p.x[i];
    xo[d] = ptr;
  }
  // unit claim: static stride, or (unit_ctr) the next unit of the shared counter (units are claimed in the producer's tile order,
  // so a claimant waits at most for its own unit's tiles; late joiners drain whatever is left)
  auto claim = [&]() -> int {
    int v = 0;
    if (lane == 0) v = atomicAdd(p.unit_ctr, 1);
    return __shfl_sync(0xffffffffu, v, 0);
  };
  for (int u = p.unit_ctr ? claim() : first; u < U; u = p.unit_ctr ? claim() : u + stride) {
    int j, n, q;
    tp4::red5_map(p, Pr, u, j, n, q);
    if (lane < 4) tp4::spin_ge(p.tile_flags + (lane * p.ppo + j) * 32 + n, epoch);
    __syncwarp();
    const int m = j * 4 + p.rank;
    const int xrow0 = m * 128 + q * 32;
    const unsigned ioff0 = static_cast<unsigned>(j * 128 + q * 32) * 1024u + n * 32 + lane;   // uint4 index
    const unsigned xoff0 = static_cast<unsigned>(xrow0) * 1024u + n * 32 + lane;
    #pragma unroll 1
    for (int r = 0; r < 32; r += ROWS) {
      const unsigned ioff = ioff0 + r * 1024u, xoff = xoff0 + r * 1024u;
      uint4 a[ROWS][4], rv[ROWS];
      #pragma unroll
      for (int rr = 0; rr < ROWS; rr++) {
        #pragma unroll
        for (int s = 0; s < 4; s++) a[rr][s] = __ldcg(p.inbox[s] + ioff + rr * 1024);
      }
      #pragma unroll
      for (int rr = 0; rr < ROWS; rr++) rv[rr] = p.resid ? __ldg(p.resid + xoff + rr * 1024) : make_uint4(0, 0, 0, 0);
      #pragma unroll
      for (int rr = 0; rr < ROWS; rr++) {
        const uint4 o = tp4::sum5_bf16x8(a[rr][0], a[rr][1], a[rr][2], a[rr][3], rv[rr]);
        #pragma unroll
        for (int d = 0; d < 4; d++) st_relaxed_sys_v4(xo[d] + xoff + rr * 1024, o);
        float ss = tp4::sumsq_add(o, 0.f);
        #pragma unroll
        for (int sh = 16; sh > 0; sh >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, sh);
        if (lane == 0) atomicAdd(p.rowss_local + xrow0 + r + rr, ss);
      }
    }
    __syncwarp();
    unsigned last = 0;
    if (lane == 0) {
      tp4::fence_acq_rel_sys();                         // this warp's x stores (via __syncwarp) + its rowss REDs before panel_done
      last = atomicAdd(p.panel_done + m, 1u) == 127u;   // 128 units per panel
    }
    if (__shfl_sync(0xffffffffu, last, 0)) tp4::red5_publish(p, m, epoch, lane);
  }
}

// cp.async (LDGSTS, .cg = L1 bypass) helpers for the staged role reducer
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

// Bytes of smem the staged role reducer needs per warp (ring of ST batches x ROWS rows x 5 slices x 512 B).
__host__ __device__ constexpr int red5a_warp_bytes(int rows, int stages) { return stages * rows * 5 * 512; }
constexpr int kRed5aSmemOffset = 128;   // role CTAs: bytes 0..127 of the dynamic smem hold the role broadcast scratch

// Owner reducer, warp-per-unit loop with cp.async staging (S3 role reducer, stages > 0). Why: the consumer GEMM's role CTAs run
// with 214 KB of dynamic smem, i.e. the max-shared carveout leaves ~28 KB of L1, and plain ld.global of the 4 inbox slices +
// resid is staged through L1, which caps the loads in flight per SM (measured: the stand-alone reducer with 214 KB dynamic smem
// is 1.3x slower alone and its tail under the producer 1.5x longer). LDGSTS.BYPASS writes straight into smem (the GEMM's
// pipeline smem, not yet initialised), so the in-flight bytes are bounded by the smem ring instead. Same unit order, flag waits,
// stores, rowss REDs, fence + panel_done + publish as red5w_loop. Flattened pipeline over (unit, ROWS-row batch): issue batch
// k + ST - 1 (waiting the unit's 4 tile flags at its first batch), wait batch k, compute it. Each lane only reads back the slots it
// copied itself, so no warp sync is needed around the ring (WAR: batch k - 1's slots were consumed before batch k + ST - 1 is issued).
template <int ROWS, int ST>
__device__ __forceinline__ void red5a_loop(Tp4Red5Params const& p, int first, int stride, uint4* wsm) {
  static_assert(ROWS == 2 && (ST == 2 || ST == 3), "ROWS / ST");
  constexpr int NB = 32 / ROWS;                         // batches per unit
  constexpr int SLOT = ROWS * 5 * 32;                   // uint4 per ring slot
  const unsigned epoch = p.epoch_ptr ? static_cast<unsigned>(__ldcg(p.epoch_ptr)) : static_cast<unsigned>(p.epoch);
  const int P = p.m_valid / 128;
  const int Pr = P > p.rank ? (P - 1 - p.rank) / 4 + 1 : 0;
  const int U = Pr * 128;
  const int lane = threadIdx.x & 31;
  const int nu = first < U ? (U - 1 - first) / stride + 1 : 0;
  const int K = nu * NB;
  uint4* xo[4];                                         // destination order: rank+1, rank+2, rank+3, local
  #pragma unroll
  for (int d = 0; d < 4; d++) {
    const int r = (p.rank + 1 + d) & 3;
    uint4* ptr = p.x[0];
    #pragma unroll
    for (int i = 1; i < 4; i++) if (r == i) ptr = p.x[i];
    xo[d] = ptr;
  }
  auto issue = [&](int k) {
    if (k < K) {
      const int ui = k / NB, b = k - ui * NB;
      int j, n, q;
      tp4::red5_map(p, Pr, first + ui * stride, j, n, q);
      if (b == 0) {
        if (lane < 4) tp4::spin_ge(p.tile_flags + (lane * p.ppo + j) * 32 + n, epoch);
        __syncwarp();
      }
      const int m = j * 4 + p.rank;
      const unsigned ioff = static_cast<unsigned>(j * 128 + q * 32 + b * ROWS) * 1024u + n * 32 + lane;
      const unsigned xoff = static_cast<unsigned>(m * 128 + q * 32 + b * ROWS) * 1024u + n * 32 + lane;
      uint4* sl = wsm + (k % ST) * SLOT + lane;
      #pragma unroll
      for (int rr = 0; rr < ROWS; rr++) {
        #pragma unroll
        for (int s = 0; s < 4; s++) cp_async16(sl + (rr * 5 + s) * 32, p.inbox[s] + ioff + rr * 1024);
        if (p.resid) cp_async16(sl + (rr * 5 + 4) * 32, p.resid + xoff + rr * 1024);
      }
    }
    cp_async_commit();                                  // empty groups past the end keep the group count uniform
  };
  #pragma unroll
  for (int k = 0; k < ST - 1; k++) issue(k);
  #pragma unroll 1
  for (int k = 0; k < K; k++) {
    issue(k + ST - 1);
    cp_async_wait<ST - 1>();                            // batch k has landed (this lane's copies)
    const int ui = k / NB, b = k - ui * NB;
    int j, n, q;
    tp4::red5_map(p, Pr, first + ui * stride, j, n, q);
    const int m = j * 4 + p.rank;
    const int xrow = m * 128 + q * 32 + b * ROWS;
    const unsigned xoff = static_cast<unsigned>(xrow) * 1024u + n * 32 + lane;
    const uint4* sl = wsm + (k % ST) * SLOT + lane;
    #pragma unroll
    for (int rr = 0; rr < ROWS; rr++) {
      const uint4 rv = p.resid ? sl[(rr * 5 + 4) * 32] : make_uint4(0, 0, 0, 0);
      const uint4 o = tp4::sum5_bf16x8(sl[(rr * 5 + 0) * 32], sl[(rr * 5 + 1) * 32], sl[(rr * 5 + 2) * 32], sl[(rr * 5 + 3) * 32], rv);
      #pragma unroll
      for (int d = 0; d < 4; d++) st_relaxed_sys_v4(xo[d] + xoff + rr * 1024, o);
      float ss = tp4::sumsq_add(o, 0.f);
      #pragma unroll
      for (int sh = 16; sh > 0; sh >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, sh);
      if (lane == 0) atomicAdd(p.rowss_local + xrow + rr, ss);
    }
    if (b == NB - 1) {
      __syncwarp();
      unsigned last = 0;
      if (lane == 0) {
        tp4::fence_acq_rel_sys();
        last = atomicAdd(p.panel_done + m, 1u) == 127u;
      }
      if (__shfl_sync(0xffffffffu, last, 0)) tp4::red5_publish(p, m, epoch, lane);
    }
  }
  cp_async_wait<0>();
}

// Owner reducer, warp-per-unit variant (S2 default): every warp of every CTA owns whole units, unit u = gw + k * total warps with
// gw = warp * gridDim.x + blockIdx.x (a round's units spread over the SMs). 2 rows per lane per iteration (64 regs cap at 1024).
template <int THREADS>
__global__ void __launch_bounds__(THREADS, THREADS <= 544 ? 2 : 1) tp4_reduce5w_kernel(Tp4Red5Params p) {
  constexpr int NW = THREADS / 32;
  if (p.pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");   // "pdl1": the producer may launch now
  red5w_loop<2>(p, (int)(threadIdx.x >> 5) * (int)gridDim.x + (int)blockIdx.x, (int)gridDim.x * NW);
}

// Phase-6 S3 role reducer: the warp-per-unit loop run by the R role CTAs of the mode-8 consumer GEMM (all warps of the CTA, before
// its pipeline init, i.e. with the full __launch_bounds__(384, 1) register budget), units strided over R CTAs x warps
// (gw = warp * R + role). p.rows (2 | 4 | 8) = rows per lane per iteration (5 x rows 16 B loads in flight per lane).
// smem = the CTA's dynamic smem base (stages > 0: per-warp rings from byte kRed5aSmemOffset on).
__device__ __forceinline__ void tp4_role_reduce(Tp4Red5Params const& p, int role, int R, char* smem) {
  const int nw = (int)(blockDim.x >> 5), first = (int)(threadIdx.x >> 5) * R + role, stride = R * nw;
  if (p.stages > 0 && !p.unit_ctr) {
    uint4* wsm = reinterpret_cast<uint4*>(smem + kRed5aSmemOffset + (threadIdx.x >> 5) * red5a_warp_bytes(2, p.stages));
    if (p.stages == 3) red5a_loop<2, 3>(p, first, stride, wsm);
    else red5a_loop<2, 2>(p, first, stride, wsm);
    return;
  }
  if (p.rows == 8) red5w_loop<8>(p, first, stride);
  else if (p.rows == 4) red5w_loop<4>(p, first, stride);
  else red5w_loop<2>(p, first, stride);
}

// The role reducer as a stand-alone kernel (variant 2): `blocks` CTAs of 384 threads, one per SM, 168-register budget like the
// GEMM's role CTAs; for the stage breakdown and the 3-kernel comparison.
__global__ void __launch_bounds__(384, 1) tp4_reduce5r_kernel(Tp4Red5Params p) {
  extern __shared__ __align__(16) char red5r_smem[];
  if (p.pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  tp4_role_reduce(p, (int)blockIdx.x, (int)gridDim.x, red5r_smem);
}

// Patch 10 (S3): producer tile order "panel-major tail". The static persistent scheduler with raster 1 / swizzle 1 hands out
// linear tile L as (m, n) = (L % tiles_m, L / tiles_m); with tail_cols = C > 0 the first tiles_m * (tiles_n - C) tiles keep that
// order and the rest walk the last C n-tiles panel by panel (m slow), so the panels of the second-to-last waves complete (and can
// be reduced and consumed) while the last wave is still being produced. A bijection on the tile grid; every warp role remaps
// every tile it gets, so all of them agree.
template <class WorkTileInfo>
__device__ __forceinline__ void tp4_remap_tile(CtappTp4Params const& p, WorkTileInfo& w) {
  if (p.tail_cols <= 0 || !w.is_valid()) return;
  const int TM = p.tiles_m, TN = p.tiles_n, C = p.tail_cols;
  const int L = static_cast<int>(w.N_idx) * TM + static_cast<int>(w.M_idx);
  const int head = TM * (TN - C);
  int m, n;
  if (L < head) { n = L / TM; m = L - n * TM; }
  else { const int L2 = L - head; m = L2 / C; n = TN - C + (L2 - m * C); }
  w.M_idx = m; w.N_idx = n;
}

// Role-switch preamble of the mode-8 consumer GEMM (patch 9c; every thread of the CTA, before the pipeline init): the first R
// CTAs to arrive (role_ctr, zeroed by tp4_step) run the owner reducer, then everybody falls through into the stock prologue.
// With work stealing (red5.unit_ctr = role_ctr + 1) every CTA runs the claim loop: the early CTAs (on the SMs the producer leaves
// free) keep pace with the producer, the CTAs that start as producer CTAs retire drain the remaining units, then all fall through.
// scratch = the start of the kernel's dynamic smem (not yet in use: the mbarriers are initialised after this).
__device__ __forceinline__ void tp4_role_preamble(CtappTp4Params const& p, int* scratch) {
  if (threadIdx.x == 0) scratch[0] = atomicAdd(p.role_ctr, 1);
  __syncthreads();
  const int role = scratch[0];
  __syncthreads();
  if (role < p.R || p.red5.unit_ctr) {   // work stealing (unit_ctr): every CTA claims units until none are left
    if (threadIdx.x == 0) tp4_trace(p, 4);
    tp4_role_reduce(p.red5, role, p.R, reinterpret_cast<char*>(scratch));
    __syncthreads();
    if (threadIdx.x == 0) tp4_trace(p, 5);
  }
  if (threadIdx.x == 0 && p.trace) p.trace[tp4_cta_id() * 8 + 7] = tp4_smid() | (static_cast<unsigned long long>(role < p.R ? role + 1 : 0) << 32);
}

// Per-step reset for v5 (one launch on the compute stream before the producer): epoch += 1, zero rowss_local[0:M] and
// panel_done[0:M/128] (and the mode-8 role counter). Nothing else needs resetting (all other flags are epoch values;
// rowss[parity] is fully overwritten).
__global__ void tp4_step_kernel(int* epoch, float4* rowss_local4, unsigned* panel_done, int M, int* role_ctr) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
  if (i == 0) { epoch[0] += 1; if (role_ctr) { role_ctr[0] = 0; role_ctr[1] = 0; } }   // mode-8 role counter, unit (steal) counter (S3)
  for (int k = i; k < M / 4; k += stride) rowss_local4[k] = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int k = i; k < M / 128; k += stride) panel_done[k] = 0u;
}
