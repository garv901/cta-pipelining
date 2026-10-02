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

struct CtappTp4Params {
  int mode = 0;              // 0 stock; 1 producer: consumers signal, the 2 idle producer warps reduce (default); 2 consumer (wait per panel + acquire);
                             // 3 producer with the reduce done by the consumer warp groups themselves, deferred by one tile (A/B only)
                             // 4 producer: consumers signal only; a separate reducer kernel (tp4_reduce_kernel) on reserved SMs reduces
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

// Mainloop producer warp, before issuing the TMA loads of tile (m_idx, *): the reduced panel must be complete.
__device__ __forceinline__ void tp4_wait_panel(CtappTp4Params const& p, int m_idx) {
  if (p.mode != 2) return;
  const unsigned target = static_cast<unsigned>(tp4_epoch(p)) * static_cast<unsigned>(p.panel_target);
  if ((threadIdx.x & 31) == 0 && !(p.dbg & tp4::kDbgNoTmaWait)) tp4::spin_ge(p.panel_cnt + m_idx, target);
  __syncwarp();
  tp4::fence_proxy_async();
}

// Consumer warp groups, before the epilogue: every thread acquires the panel so its rowss loads see the remote writes.
__device__ __forceinline__ void tp4_before_epilogue(CtappTp4Params const& p, int m_idx) {
  if (p.mode != 2) return;
  if (p.dbg & tp4::kDbgNoEpiAcquire) return;
  const unsigned target = static_cast<unsigned>(tp4_epoch(p)) * static_cast<unsigned>(p.panel_target);
  tp4::spin_ge(p.panel_cnt + m_idx, target);
}

// Consumer warp groups, after collective_epilogue.store() of tile (m, n): signal it, then reduce the previous tile's slice.
__device__ __forceinline__ void tp4_after_store(CtappTp4Params const& p, int m, int n, int& prev_m, int& prev_n, int tid) {
  if (p.mode != 1 && p.mode != 3 && p.mode != 4) return;
  if (tid < 32 && !(p.dbg & tp4::kDbgNoStoreDrain)) { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); tp4::fence_proxy_async(); }   // the TMA-issuing warp
  tp4::bar_sync();
  if (tid == 0) { tp4::fence_acq_rel_sys(); tp4::mm_red_release_add_u32(p.tile_cnt_mc + m * p.tiles_n + n, 1u); }
  if (p.mode == 3 && prev_m >= 0 && !(p.dbg & tp4::kDbgNoReduce)) tp4::reduce_slice(p, prev_m, prev_n, tid);
  prev_m = m; prev_n = n;
}

// Consumer warp groups, after the work loop: reduce the slice of the last tile this CTA stored.
__device__ __forceinline__ void tp4_finish(CtappTp4Params const& p, int prev_m, int prev_n, int tid) {
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
