// Phase 4 extension (module ctapp_tp4_ext): down-proj -> RMSNorm -> QKV at TP-t with CTA-pipelined NVLS reduction.
// Kernels: the patched SM90 cooperative GEMM (csrc/sm90_gemm_coop_ctapp.hpp + csrc/ctapp_tp4.cuh) in producer / consumer /
// stock modes, plus small multimem probe kernels for tests/test_tp4.py. Symmetric-memory buffers come in as raw device
// addresses (torch.distributed._symmetric_memory handle: buffer_ptrs[rank], multicast_ptr) so one process per GPU works.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <optional>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/kernel_hardware_info.h"
#include "sm90_gemm_coop_ctapp.hpp"

using namespace cute;
using Element = cutlass::bfloat16_t;

// ---- RMSNorm row scale for the QKV epilogue: out = acc * rsqrt(rowss / N + eps); gamma is folded into W. ----
static constexpr float kRmsInvN = 1.f / 8192.f;   // N1 of the 70B-class target; the Python side asserts N1 == 8192
static constexpr float kRmsEps = 1e-5f;

template <class T> struct RmsScaleFn { };
template <class T, int N>
struct RmsScaleFn<cutlass::Array<T, N>> {
  CUTLASS_HOST_DEVICE cutlass::Array<T, N> operator()(cutlass::Array<T, N> const& ss, cutlass::Array<T, N> const& acc) const {
    cutlass::Array<T, N> out;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; i++) out[i] = acc[i] * rsqrtf(ss[i] * kRmsInvN + kRmsEps);
    return out;
  }
};

// Builder-equivalent TMA warp-specialized epilogue with a chosen fusion (see CoopEpilogue in gemm_coop.cu).
template <class TileShape, class Fusion>
struct Tp4Epilogue {
  using Schedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using EpiTile = decltype(cutlass::epilogue::collective::detail::sm90_compute_tile_shape_or_override<
      Element, cutlass::epilogue::collective::EpilogueTileAuto, Schedule, TileShape>());
  static constexpr int EpiTiles = decltype(size(shape_div(take<0, 2>(TileShape{}), EpiTile{})))::value;
  static constexpr int Frag = decltype(size(EpiTile{}))::value / 256;
  using Policy = cutlass::epilogue::Sm90TmaWarpSpecialized<(EpiTiles < 4 ? EpiTiles : 4), (EpiTiles < 2 ? EpiTiles : 2), Frag, false, true>;
  using Op = typename cutlass::epilogue::collective::detail::Sm90TmaBuilderImpl<
      TileShape, EpiTile, float, float, void, cutlass::layout::RowMajor, 8, Element, cutlass::layout::RowMajor, 8, Fusion, Policy>::CollectiveOp;
};

template <class TileShape>
using LinCombFusion = cutlass::epilogue::fusion::LinearCombination<Element, float, void, float>;
template <class TileShape>
using RmsFusion = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<RmsScaleFn, Element, float, cutlass::FloatRoundStyle::round_to_nearest>,
    cutlass::epilogue::fusion::Sm90ColBroadcast<0, TileShape, float, float, Stride<_1, _0, _0>>,
    cutlass::epilogue::fusion::Sm90AccFetch>;

template <int BM, int BN, int BK, bool Rms>
struct GemmTp4 {
  using TileShape = Shape<Int<BM>, Int<BN>, Int<BK>>;
  using ClusterShape = Shape<_1, _1, _1>;
  using Fusion = cute::conditional_t<Rms, RmsFusion<TileShape>, LinCombFusion<TileShape>>;
  using CollectiveEpilogue = typename Tp4Epilogue<TileShape, Fusion>::Op;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Element, cutlass::layout::RowMajor, 8,
      Element, cutlass::layout::ColumnMajor, 8,
      float, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversalCtapp<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  static typename Gemm::Arguments make_args(int M, int N, int K, const void* x, const void* w, void* y, int raster, int swizzle,
                                            CtappTp4Params const& ctapp, const float* rowss, int sms) {
    auto hw = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel>(at::cuda::current_device());
    if (sms > 0) hw.sm_count = sms;   // leave SMs free for a concurrent reducer kernel (mode 4)
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
        {reinterpret_cast<const Element*>(x), cutlass::make_cute_packed_stride(typename GemmKernel::StrideA{}, {M, K, 1}),
         reinterpret_cast<const Element*>(w), cutlass::make_cute_packed_stride(typename GemmKernel::StrideB{}, {N, K, 1})},
        {{}, nullptr, typename GemmKernel::StrideC{}, reinterpret_cast<Element*>(y), cutlass::make_cute_packed_stride(typename GemmKernel::StrideD{}, {M, N, 1})},
        hw};
    if constexpr (Rms) args.epilogue.thread = {{rowss, 0.f, {}}, {}, {}};   // {col broadcast (rowss), acc fetch, compute}
    else args.epilogue.thread = {1.f, 0.f};
    using RO = decltype(args.scheduler.raster_order);
    args.scheduler.raster_order = static_cast<RO>(raster);  // 0 Heuristic, 1 AlongN, 2 AlongM
    args.scheduler.max_swizzle_size = swizzle;
    args.ctapp = ctapp;
    return args;
  }

  // Workspace size is 0 for this scheduler; CUTLASS still wants a non-null pointer, so hand it a static 256 B device buffer (never freed).
  static void* dummy_workspace() {
    static void* p = nullptr;
    if (!p) TORCH_CHECK(cudaMalloc(&p, 256) == cudaSuccess, "cudaMalloc");
    return p;
  }

  static void run(const at::Tensor& X, const at::Tensor& W, const at::Tensor& Y, int raster, int swizzle, CtappTp4Params const& ctapp,
                  const float* rowss, cudaStream_t stream, int sms = 0, bool pdl = false) {
    auto args = make_args(X.size(0), W.size(0), X.size(1), X.data_ptr(), W.data_ptr(), Y.data_ptr(), raster, swizzle, ctapp, rowss, sms);
    Gemm gemm;
    TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "can_implement failed");
    const size_t ws_bytes = Gemm::get_workspace_size(args);   // 0 for this scheduler: no allocation per call
    at::Tensor ws;
    if (ws_bytes) ws = at::empty({(int64_t)ws_bytes}, X.options().dtype(at::kByte));
    TORCH_CHECK(gemm.initialize(args, ws_bytes ? ws.data_ptr() : dummy_workspace(), stream) == cutlass::Status::kSuccess, "initialize failed");
    TORCH_CHECK(gemm.run(stream, nullptr, pdl) == cutlass::Status::kSuccess, "run failed");
  }

  static size_t workspace_probe() {
    CtappTp4Params p;
    auto a = make_args(128, 256, 64, nullptr, nullptr, nullptr, 1, 1, p, nullptr, 0);
    return Gemm::get_workspace_size(a);
  }

  static std::string info() {
    return std::string("coop-ctapp") + (Rms ? "-rms" : "") + " tile " + std::to_string(BM) + "x" + std::to_string(BN) + "x" + std::to_string(BK) +
           " threads=" + std::to_string(GemmKernel::MaxThreadsPerBlock) + " smem=" + std::to_string(GemmKernel::SharedStorageSize) +
           " stages=" + std::to_string(CollectiveMainloop::DispatchPolicy::Stages) +
           " workspace=" + std::to_string(workspace_probe());
  }
};

using GemmDown = GemmTp4<128, 256, 64, false>;
using GemmQkv = GemmTp4<128, 256, 64, true>;

static void check_bf16_2d(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.dtype() == at::kBFloat16 && t.dim() == 2 && t.is_contiguous(), name, " must be a contiguous CUDA bf16 matrix");
}

// Down-proj partial GEMM D = A W^T into this rank's copy of the symmetric partials buffer (D), with the Phase-4 producer protocol.
// mode 0 runs the same kernel without any protocol (A/B baseline for the hook cost).
static int g_tp4_dbg = 0;
void tp4_set_debug(int64_t v) { g_tp4_dbg = (int)v; }   // ablation bitmask (see ctapp_tp4.cuh); timing experiments only

void tp4_down(at::Tensor A, at::Tensor W, at::Tensor D, int64_t raster, int64_t swizzle, int64_t mode, int64_t rank, int64_t world, int64_t epoch,
              int64_t tile_cnt_mc, at::Tensor tile_cnt, int64_t partials_mc, int64_t x_mc, std::optional<at::Tensor> resid,
              int64_t rowss_mc, int64_t panel_cnt_mc, int64_t sms, std::optional<at::Tensor> epoch_dev, int64_t rowss_mc_base, int64_t rowss_stride) {
  check_bf16_2d(A, "A"); check_bf16_2d(W, "W"); check_bf16_2d(D, "D");
  c10::cuda::CUDAGuard guard(A.device());
  const int M = A.size(0), N = W.size(0);
  TORCH_CHECK(D.size(0) == M && D.size(1) == N && M % 128 == 0 && N % 256 == 0, "shape");
  TORCH_CHECK(mode == 0 || mode == 1 || mode == 3 || mode == 4, "mode");
  TORCH_CHECK(128 % world == 0 && (256 * world) % 256 == 0 && 256 / (256 / (128 / world)) % 8 == 0, "world must divide the tile rows into >=8-column thread slices");
  CtappTp4Params p;
  p.mode = (int)mode; p.rank = (int)rank; p.world = (int)world; p.epoch = (int)epoch; p.dbg = g_tp4_dbg;
  p.tiles_n = N / 256; p.ld = N; p.bm = 128; p.bn = 256;
  p.tile_cnt_mc = reinterpret_cast<unsigned*>(tile_cnt_mc);
  p.tile_cnt = reinterpret_cast<const unsigned*>(tile_cnt.data_ptr<int32_t>());
  p.partials_mc = reinterpret_cast<const uint4*>(partials_mc);
  p.x_mc = reinterpret_cast<uint4*>(x_mc);
  p.resid = resid ? reinterpret_cast<const uint4*>(resid->data_ptr()) : nullptr;
  p.rowss_mc = reinterpret_cast<float*>(rowss_mc);
  p.panel_cnt_mc = reinterpret_cast<unsigned*>(panel_cnt_mc);
  p.epoch_ptr = epoch_dev ? epoch_dev->data_ptr<int32_t>() : nullptr;
  p.rowss_mc_base = reinterpret_cast<float*>(rowss_mc_base); p.rowss_stride = (int)rowss_stride;
  GemmDown::run(A, W, D, (int)raster, (int)swizzle, p, nullptr, at::cuda::getCurrentCUDAStream(), (int)sms);
}

// Mode-4 reducer kernel: `blocks` CTAs of 512 threads on the current stream (launch BEFORE the mode-4 GEMM, on another stream,
// with the GEMM's sms = SM count - blocks so both are resident). Arguments mirror tp4_down.
void tp4_reduce(int64_t M, int64_t rank, int64_t world, int64_t epoch, int64_t raster, int64_t blocks,
                at::Tensor tile_cnt, int64_t partials_mc, int64_t x_mc, std::optional<at::Tensor> resid, int64_t rowss_mc, int64_t panel_cnt_mc,
                int64_t unicast = 0, std::vector<int64_t> peer_partials = {}, std::vector<int64_t> peer_x = {}, int64_t version = 1,
                std::optional<at::Tensor> epoch_dev = std::nullopt, int64_t rowss_mc_base = 0, int64_t rowss_stride = 0) {
  c10::cuda::CUDAGuard guard(tile_cnt.device());
  TORCH_CHECK(version >= 1 && version <= 4, "version");
  TORCH_CHECK(version == 1 || unicast == 0, "versions >= 2 have no unicast variant");
  TORCH_CHECK(raster == 1 || raster == 2, "raster must be 1 (AlongN) or 2 (AlongM); the reducer follows the GEMM's tile order");
  CtappTp4Params p;
  p.mode = 4; p.rank = (int)rank; p.world = (int)world; p.epoch = (int)epoch; p.dbg = g_tp4_dbg;
  p.tiles_n = 8192 / 256; p.ld = 8192; p.bm = 128; p.bn = 256;
  p.tile_cnt = reinterpret_cast<const unsigned*>(tile_cnt.data_ptr<int32_t>());
  p.partials_mc = reinterpret_cast<const uint4*>(partials_mc);
  p.x_mc = reinterpret_cast<uint4*>(x_mc);
  p.resid = resid ? reinterpret_cast<const uint4*>(resid->data_ptr()) : nullptr;
  p.rowss_mc = reinterpret_cast<float*>(rowss_mc);
  p.panel_cnt_mc = reinterpret_cast<unsigned*>(panel_cnt_mc);
  p.epoch_ptr = epoch_dev ? epoch_dev->data_ptr<int32_t>() : nullptr;
  p.rowss_mc_base = reinterpret_cast<float*>(rowss_mc_base); p.rowss_stride = (int)rowss_stride;
  if (version == 3) { tp4_reduce3_kernel<2><<<(int)blocks, 544, 0, at::cuda::getCurrentCUDAStream()>>>(p, (int)raster, (int)(M / 128)); return; }
  if (version == 4) { tp4_reduce3_kernel<4><<<(int)blocks, 544, 0, at::cuda::getCurrentCUDAStream()>>>(p, (int)raster, (int)(M / 128)); return; }
  if (version == 2) {
    tp4_reduce2_kernel<<<(int)blocks, 512, 0, at::cuda::getCurrentCUDAStream()>>>(p, (int)raster, (int)(M / 128));
    return;
  }
  if (unicast) {
    TORCH_CHECK(world <= 8 && (int)peer_partials.size() >= world && (int)peer_x.size() >= world, "unicast needs per-rank addresses");
    for (int r = 0; r < world; r++) { p.peer_partials[r] = reinterpret_cast<const uint4*>(peer_partials[r]); p.peer_x[r] = reinterpret_cast<uint4*>(peer_x[r]); }
    tp4_reduce_unicast_kernel<<<(int)blocks, 512, 0, at::cuda::getCurrentCUDAStream()>>>(p, (int)raster, (int)(M / 128));
    return;
  }
  tp4_reduce_kernel<<<(int)blocks, 512, 0, at::cuda::getCurrentCUDAStream()>>>(p, (int)raster, (int)(M / 128));
}

// QKV GEMM Y = rmsnorm_scale(X W^T): X is this rank's copy of the reduced symmetric x buffer; rows are scaled by
// rsqrt(rowss[row]/8192 + 1e-5). mode 2 waits for each 128-row panel (panel_cnt >= epoch * panel_target); mode 0 does not wait.
void tp4_qkv(at::Tensor X, at::Tensor W, at::Tensor Y, int64_t raster, int64_t swizzle, int64_t mode, int64_t epoch,
             at::Tensor panel_cnt, int64_t panel_target, at::Tensor rowss, int64_t sms, int64_t pdl, std::optional<at::Tensor> epoch_dev) {
  check_bf16_2d(X, "X"); check_bf16_2d(W, "W"); check_bf16_2d(Y, "Y");
  c10::cuda::CUDAGuard guard(X.device());
  const int M = X.size(0), N = W.size(0);
  TORCH_CHECK(X.size(1) == 8192, "RMSNorm scale is compiled for N1 = 8192");
  TORCH_CHECK(Y.size(0) == M && Y.size(1) == N && M % 128 == 0 && N % 256 == 0, "shape");
  TORCH_CHECK(mode == 0 || mode == 2, "mode");
  TORCH_CHECK(rowss.dtype() == at::kFloat && rowss.numel() >= M, "rowss");
  CtappTp4Params p;
  p.mode = (int)mode; p.epoch = (int)epoch; p.panel_target = (int)panel_target; p.dbg = g_tp4_dbg;
  p.panel_cnt = reinterpret_cast<const unsigned*>(panel_cnt.data_ptr<int32_t>());
  p.epoch_ptr = epoch_dev ? epoch_dev->data_ptr<int32_t>() : nullptr;
  GemmQkv::run(X, W, Y, (int)raster, (int)swizzle, p, rowss.data_ptr<float>(), at::cuda::getCurrentCUDAStream(), (int)sms, pdl != 0);
}

std::string tp4_info(int64_t which) {
  c10::cuda::CUDAGuard guard(at::cuda::current_device());
  return which == 0 ? GemmDown::info() : GemmQkv::info();
}

// ---- multimem probes (tests/test_tp4.py) ----
__global__ void mm_red_u32_kernel(unsigned* mc, int n, unsigned v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) asm volatile("multimem.red.relaxed.sys.global.add.u32 [%0], %1;" :: "l"(mc + i), "r"(v) : "memory");
}
__global__ void mm_red_f32_kernel(float* mc, const float* src, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) tp4::mm_red_add_f32(mc + i, src[i]);
}
__global__ void mm_ld_reduce_kernel(const uint4* mc, uint4* out, int n4) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) out[i] = tp4::mm_ld_reduce_bf16x8(mc + i);
}
__global__ void mm_red_bf16_kernel(uint4* mc, const uint4* src, int n4) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) tp4::mm_red_add_bf16x8(mc + i, src[i]);
}
__global__ void mm_st_kernel(uint4* mc, const uint4* src, int n4) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) tp4::mm_st_bf16x8(mc + i, src[i]);
}
// Ordering stress: for it in [0, iters): each block writes value (it+1) into its rank's slice of data (multimem.st, all copies),
// then one thread per block does fence + multimem.red.release on flag[block]; then waits until the local flag copy reaches
// world*(it+1) and checks every rank's slice in the local data copy == it+1. Counts violations.
__global__ void mm_order_stress_kernel(uint4* data_mc, const uint4* data_local, unsigned* flag_mc, const unsigned* flag_local,
                                       int rank, int world, int iters, int per_rank4, int* bad) {
  const int b = blockIdx.x;                  // slot: data slice [b * world * per_rank4 .. ), flag[b]
  uint4* slice = data_mc + (static_cast<size_t>(b) * world + rank) * per_rank4;
  for (int it = 1; it <= iters; it++) {
    unsigned v = static_cast<unsigned>(it) | (static_cast<unsigned>(it) << 16);
    for (int i = threadIdx.x; i < per_rank4; i += blockDim.x) tp4::mm_st_bf16x8(slice + i, make_uint4(v, v, v, v));
    __syncthreads();
    if (threadIdx.x == 0) { tp4::fence_acq_rel_sys(); tp4::mm_red_release_add_u32(flag_mc + b, 1u); }
    if (threadIdx.x == 0) { while (tp4::ld_acquire_sys(flag_local + b) < static_cast<unsigned>(world * it)) {} }
    __syncthreads();
    const uint4* all = data_local + static_cast<size_t>(b) * world * per_rank4;
    int nbad = 0;
    for (int i = threadIdx.x; i < world * per_rank4; i += blockDim.x) {
      uint4 x = all[i];   // plain load after the acquire chain (thread 0 acquire + __syncthreads)
      nbad += (x.x != v) + (x.y != v) + (x.z != v) + (x.w != v);
    }
    if (nbad) atomicAdd(bad, nbad);
    __syncthreads();
  }
}

// Latency probe (one thread): ns per dependent multimem.ld_reduce; ns per multimem.st + fence.acq_rel.sys; ns per local ld.acquire.sys.
__global__ void mm_latency_kernel(const uint4* mc_ld, uint4* mc_st, const unsigned* local_flag, int iters, float* out) {
  unsigned long long t0 = tp4::globaltimer();
  unsigned idx = 0;
  for (int i = 0; i < iters; i++) { uint4 v = tp4::mm_ld_reduce_bf16x8(mc_ld + (idx & 63)); idx = v.x & 1u; }
  unsigned long long t1 = tp4::globaltimer();
  for (int i = 0; i < iters; i++) { tp4::mm_st_bf16x8(mc_st + (i & 63), make_uint4(idx, 0, 0, 0)); tp4::fence_acq_rel_sys(); }
  unsigned long long t2 = tp4::globaltimer();
  for (int i = 0; i < iters; i++) { idx += tp4::ld_acquire_sys(local_flag + (idx & 1)); }
  unsigned long long t3 = tp4::globaltimer();
  for (int i = 0; i < iters; i++) tp4::mm_st_bf16x8(mc_st + (i & 63), make_uint4(idx, 0, 0, 0));
  tp4::fence_acq_rel_sys();
  unsigned long long t4 = tp4::globaltimer();
  out[0] = float(t1 - t0) / iters; out[1] = float(t2 - t1) / iters; out[2] = float(t3 - t2) / iters; out[3] = float(t4 - t3) / iters; out[4] = float(idx & 0);
}
std::vector<double> mm_latency(int64_t mc_ld, int64_t mc_st, at::Tensor local_flag, int64_t iters) {
  c10::cuda::CUDAGuard guard(local_flag.device());
  auto out = at::zeros({5}, local_flag.options().dtype(at::kFloat));
  mm_latency_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const uint4*>(mc_ld), reinterpret_cast<uint4*>(mc_st),
      reinterpret_cast<const unsigned*>(local_flag.data_ptr<int32_t>()), (int)iters, out.data_ptr<float>());
  auto h = out.cpu();
  return {h[0].item<double>(), h[1].item<double>(), h[2].item<double>(), h[3].item<double>()};
}

// Register-footprint probe for the reducer path (see "ptxas info" for mm_reduce_probe_kernel in the build log): must stay <= 64.
__global__ void __launch_bounds__(64) mm_reduce_probe_kernel(CtappTp4Params p, int m, int n) { tp4::reduce_slice64(p, m, n, threadIdx.x); }
void mm_reduce_probe(int64_t rank, int64_t world) {
  CtappTp4Params p; p.rank = (int)rank; p.world = (int)world; p.dbg = 0;
  if (p.tiles_n < 0) mm_reduce_probe_kernel<<<1, 64>>>(p, 0, 0);   // never launched; keeps the kernel instantiated
}

static int blocks(int n, int t = 256) { return (n + t - 1) / t; }

void mm_red_u32(int64_t mc, int64_t n, int64_t v) {
  mm_red_u32_kernel<<<blocks(n), 256, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<unsigned*>(mc), (int)n, (unsigned)v);
}
void mm_red_f32(int64_t mc, at::Tensor src) {
  c10::cuda::CUDAGuard guard(src.device());
  mm_red_f32_kernel<<<blocks(src.numel()), 256, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<float*>(mc), src.data_ptr<float>(), (int)src.numel());
}
void mm_ld_reduce_bf16(int64_t mc, at::Tensor out) {
  c10::cuda::CUDAGuard guard(out.device());
  int n4 = out.numel() / 8;
  mm_ld_reduce_kernel<<<blocks(n4), 256, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const uint4*>(mc), reinterpret_cast<uint4*>(out.data_ptr()), n4);
}
void mm_red_bf16(int64_t mc, at::Tensor src) {   // multimem.red.add bf16x2 of src into every copy
  c10::cuda::CUDAGuard guard(src.device());
  int n4 = src.numel() / 8;
  mm_red_bf16_kernel<<<blocks(n4), 256, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<uint4*>(mc), reinterpret_cast<const uint4*>(src.data_ptr()), n4);
}
void mm_st_bf16(int64_t mc, at::Tensor src) {
  c10::cuda::CUDAGuard guard(src.device());
  int n4 = src.numel() / 8;
  mm_st_kernel<<<blocks(n4), 256, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<uint4*>(mc), reinterpret_cast<const uint4*>(src.data_ptr()), n4);
}
int64_t mm_order_stress(int64_t data_mc, at::Tensor data_local, int64_t flag_mc, at::Tensor flag_local, int64_t rank, int64_t world, int64_t iters, int64_t slots) {
  c10::cuda::CUDAGuard guard(data_local.device());
  int per_rank4 = data_local.numel() / 8 / (int)(slots * world);
  TORCH_CHECK(per_rank4 > 0 && flag_local.numel() >= slots, "stress buffers too small");
  auto bad = at::zeros({1}, data_local.options().dtype(at::kInt));
  mm_order_stress_kernel<<<(int)slots, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<uint4*>(data_mc), reinterpret_cast<const uint4*>(data_local.data_ptr()), reinterpret_cast<unsigned*>(flag_mc),
      reinterpret_cast<const unsigned*>(flag_local.data_ptr<int32_t>()), (int)rank, (int)world, (int)iters, per_rank4, bad.data_ptr<int>());
  return bad.item<int>();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tp4_down", &tp4_down, py::arg("A"), py::arg("W"), py::arg("D"), py::arg("raster"), py::arg("swizzle"), py::arg("mode"), py::arg("rank"), py::arg("world"),
        py::arg("epoch"), py::arg("tile_cnt_mc"), py::arg("tile_cnt"), py::arg("partials_mc"), py::arg("x_mc"), py::arg("resid"), py::arg("rowss_mc"), py::arg("panel_cnt_mc"), py::arg("sms") = 0,
        py::arg("epoch_dev") = std::nullopt, py::arg("rowss_mc_base") = 0, py::arg("rowss_stride") = 0);
  m.def("tp4_qkv", &tp4_qkv, py::arg("X"), py::arg("W"), py::arg("Y"), py::arg("raster"), py::arg("swizzle"), py::arg("mode"), py::arg("epoch"),
        py::arg("panel_cnt"), py::arg("panel_target"), py::arg("rowss"), py::arg("sms") = 0, py::arg("pdl") = 0, py::arg("epoch_dev") = std::nullopt);
  m.def("tp4_reduce", &tp4_reduce, py::arg("M"), py::arg("rank"), py::arg("world"), py::arg("epoch"), py::arg("raster"), py::arg("blocks"), py::arg("tile_cnt"), py::arg("partials_mc"), py::arg("x_mc"), py::arg("resid"), py::arg("rowss_mc"), py::arg("panel_cnt_mc"), py::arg("unicast") = 0, py::arg("peer_partials") = std::vector<int64_t>{}, py::arg("peer_x") = std::vector<int64_t>{}, py::arg("version") = 1,
        py::arg("epoch_dev") = std::nullopt, py::arg("rowss_mc_base") = 0, py::arg("rowss_stride") = 0);
  m.def("tp4_info", &tp4_info); m.def("tp4_set_debug", &tp4_set_debug);
  m.def("mm_red_u32", &mm_red_u32); m.def("mm_red_f32", &mm_red_f32); m.def("mm_ld_reduce_bf16", &mm_ld_reduce_bf16); m.def("mm_st_bf16", &mm_st_bf16); m.def("mm_red_bf16", &mm_red_bf16);
  m.def("mm_order_stress", &mm_order_stress); m.def("mm_latency", &mm_latency);
}
