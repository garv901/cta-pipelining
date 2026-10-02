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
  // CtappScatter_ = !Rms: the mode-7 producer scatter (4 destination epilogues) is compiled into the down-proj instance only;
  // CtappRole_ = Rms: the mode-8 role-switch preamble (role reducer) into the QKV consumer instances only
  using GemmKernel = cutlass::gemm::kernel::GemmUniversalCtapp<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, cutlass::gemm::PersistentScheduler, !Rms, Rms>;
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
    args.scheduler.raster_order = static_cast<RO>(raster);  // 0 Heuristic, 1 AlongM (m fast), 2 AlongN (n fast)  (cutlass RasterOrderOptions)
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

  // y_override: if non-null, D is written to this raw device address instead of Y.data_ptr() (Y still gives the shape); e.g. a
  // peer's symmetric-memory buffer (Phase-6 S1 remote-epilogue probe).
  static void run(const at::Tensor& X, const at::Tensor& W, const at::Tensor& Y, int raster, int swizzle, CtappTp4Params const& ctapp,
                  const float* rowss, cudaStream_t stream, int sms = 0, bool pdl = false, void* y_override = nullptr) {
    auto args = make_args(X.size(0), W.size(0), X.size(1), X.data_ptr(), W.data_ptr(), y_override ? y_override : Y.data_ptr(), raster, swizzle, ctapp, rowss, sms);
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
           " workspace=" + std::to_string(workspace_probe()) + " params=" + std::to_string(sizeof(typename GemmKernel::Params));
  }
};

using GemmDown = GemmTp4<128, 256, 64, false>;
using GemmQkv = GemmTp4<128, 256, 64, true>;
using GemmQkv128 = GemmTp4<128, 128, 64, true>;   // Phase-6 S3: QKV consumer tile for small N (N=2560: 640 tiles vs 320)

// Effective CUTLASS swizzle of the persistent scheduler (tile_scheduler_params.h get_log_swizzle_size) for a tiles_m x tiles_n grid.
static int eff_swizzle(int tiles_m, int tiles_n, int max_swizzle) {
  const int d = std::min(tiles_m, tiles_n);
  if (max_swizzle >= 8 && d >= 6) return 8;
  if (max_swizzle >= 4 && d >= 3) return 4;
  if (max_swizzle >= 2 && d >= 2) return 2;
  return 1;
}

static void check_bf16_2d(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.dtype() == at::kBFloat16 && t.dim() == 2 && t.is_contiguous(), name, " must be a contiguous CUDA bf16 matrix");
}

// Down-proj partial GEMM D = A W^T into this rank's copy of the symmetric partials buffer (D), with the Phase-4 producer protocol.
// mode 0 runs the same kernel without any protocol (A/B baseline for the hook cost).
static int g_tp4_dbg = 0;
void tp4_set_debug(int64_t v) { g_tp4_dbg = (int)v; }   // ablation bitmask (see ctapp_tp4.cuh); timing experiments only

void tp4_down(at::Tensor A, at::Tensor W, at::Tensor D, int64_t raster, int64_t swizzle, int64_t mode, int64_t rank, int64_t world, int64_t epoch,
              int64_t tile_cnt_mc, at::Tensor tile_cnt, int64_t partials_mc, int64_t x_mc, std::optional<at::Tensor> resid,
              int64_t rowss_mc, int64_t panel_cnt_mc, int64_t sms, std::optional<at::Tensor> epoch_dev, int64_t rowss_mc_base, int64_t rowss_stride,
              int64_t d_ptr, std::vector<int64_t> dst_ptrs, int64_t dst_rows, std::vector<int64_t> peer_tile_flags, int64_t panels_per_owner,
              int64_t pdl_trigger, int64_t trace, int64_t tail_cols, int64_t pdl) {
  check_bf16_2d(A, "A"); check_bf16_2d(W, "W"); check_bf16_2d(D, "D");
  c10::cuda::CUDAGuard guard(A.device());
  const int M = A.size(0), N = W.size(0);
  TORCH_CHECK(D.size(0) == M && D.size(1) == N && M % 128 == 0 && N % 256 == 0, "shape");
  TORCH_CHECK(mode == 0 || mode == 1 || mode == 3 || mode == 4 || mode == 7, "mode");
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
  p.pdl_trigger = (int)pdl_trigger;                                   // S3: trigger the PDL dependent right after the prologue
  p.trace = reinterpret_cast<unsigned long long*>(trace);             // S3: CTAPP_TRACE per-CTA stamps ([grid][8] u64, zeroed)
  if (tail_cols > 0) {   // S3 panel-major tail tile order (patch 10 / tp4_remap_tile): stock raster 1, swizzle 1 underneath
    TORCH_CHECK((mode == 0 || mode == 7) && raster == 1 && swizzle == 1 && tail_cols <= N / 256, "tail_cols: mode 0 | 7, raster 1, swizzle 1, <= tiles_n");
    p.tail_cols = (int)tail_cols; p.tiles_m = M / 128;
  }
  if (mode == 7) {   // v5 producer scatter: tile (m, n) -> dst_ptrs[m % 4] at row block m / 4 (dst_rows x N), flag peer_tile_flags[m % 4]
    TORCH_CHECK(world == 4 && dst_ptrs.size() == 4 && peer_tile_flags.size() == 4, "mode 7: world == 4 and 4 destination / flag addresses");
    TORCH_CHECK(panels_per_owner > 0 && dst_rows == panels_per_owner * 128 && (M / 128 + 3) / 4 <= panels_per_owner && N == 8192,
                "mode 7: dst_rows == panels_per_owner * 128 >= rows of the owned panels, N == 8192");
    for (int i = 0; i < 4; i++) {
      TORCH_CHECK(dst_ptrs[i] != 0 && dst_ptrs[i] % 16 == 0 && peer_tile_flags[i] != 0, "mode 7: null / misaligned address");
      p.dst_ptr[i] = reinterpret_cast<void*>(dst_ptrs[i]);
      p.peer_tile_flags[i] = reinterpret_cast<unsigned*>(peer_tile_flags[i]);
    }
    p.dst_rows = (int)dst_rows; p.panels_per_owner = (int)panels_per_owner;
  }
  // pdl (S3 "pdl1"): launch as the programmatic dependent of the previous kernel on the stream (the stand-alone reducer)
  TORCH_CHECK(pdl == 0 || mode == 7 || mode == 0, "pdl: mode 7 (or mode 0 for warm-up)");
  GemmDown::run(A, W, D, (int)raster, (int)swizzle, p, nullptr, at::cuda::getCurrentCUDAStream(), (int)sms, pdl != 0, reinterpret_cast<void*>(d_ptr));
}

// v5 owner-reducer parameters (tp4_reduce version 5 and the mode-8 role CTAs of tp4_qkv). tile_flags = this rank's tile_flags
// [4][ppo][32] (int32), M = rows of this step (m_valid), order / swz = the producer's raster / swizzle (unit order), peer_x /
// peer_rowss / peer_panel_flag = every rank's x[parity] / rowss[parity] / panel_flag.
static Tp4Red5Params make_red5(int64_t M, int64_t rank, int64_t epoch, int64_t order, int64_t swz, const at::Tensor& tile_flags,
                               std::optional<at::Tensor> const& resid, std::vector<int64_t> const& inbox_ptrs, std::vector<int64_t> const& peer_x,
                               std::vector<int64_t> const& peer_rowss, std::vector<int64_t> const& peer_panel_flag, int64_t rowss_local,
                               int64_t panel_done, int64_t panels_per_owner, std::optional<at::Tensor> const& epoch_dev, int64_t rows,
                               int64_t red_trace, int64_t stages = 0, int64_t unit_ctr = 0, int64_t tail = 0) {
  TORCH_CHECK(inbox_ptrs.size() == 4 && peer_x.size() == 4 && peer_rowss.size() == 4 && peer_panel_flag.size() == 4,
              "v5: world == 4 with 4 inbox / x / rowss / panel_flag addresses");
  TORCH_CHECK(M % 128 == 0 && M >= 0 && panels_per_owner > 0 && (M / 128 + 3) / 4 <= panels_per_owner, "v5: M % 128 == 0 and M <= max_M");
  TORCH_CHECK(tile_flags.dtype() == at::kInt && tile_flags.numel() >= 4 * panels_per_owner * 32, "v5: tile_flags shape");
  TORCH_CHECK(rowss_local != 0 && panel_done != 0, "v5: rowss_local / panel_done");
  TORCH_CHECK(order == 1 || order == 2, "v5: order (producer raster) 1 | 2");
  TORCH_CHECK(rows == 2 || rows == 4 || rows == 8, "v5: rows 2 | 4 | 8");
  TORCH_CHECK(stages == 0 || ((stages == 2 || stages == 3) && rows == 2), "v5: stages 0 | 2 | 3 (cp.async staging needs rows == 2)");
  Tp4Red5Params q;
  for (int i = 0; i < 4; i++) {
    q.inbox[i] = reinterpret_cast<const uint4*>(inbox_ptrs[i]);
    q.x[i] = reinterpret_cast<uint4*>(peer_x[i]);
    q.rowss[i] = reinterpret_cast<float*>(peer_rowss[i]);
    q.panel_flag[i] = reinterpret_cast<unsigned*>(peer_panel_flag[i]);
  }
  q.tile_flags = reinterpret_cast<const unsigned*>(tile_flags.data_ptr<int32_t>());
  q.resid = resid ? reinterpret_cast<const uint4*>(resid->data_ptr()) : nullptr;
  q.rowss_local = reinterpret_cast<float*>(rowss_local);
  q.panel_done = reinterpret_cast<unsigned*>(panel_done);
  q.epoch_ptr = epoch_dev ? epoch_dev->data_ptr<int32_t>() : nullptr;
  q.trace = reinterpret_cast<unsigned long long*>(red_trace);
  q.epoch = (int)epoch; q.rank = (int)rank; q.m_valid = (int)M; q.ppo = (int)panels_per_owner; q.order = (int)order;
  // the producer's effective swizzle (n groups) for the unit order; only m-fast order 1 follows it
  q.swz = order == 1 ? eff_swizzle((int)((M + 127) / 128), 32, (int)swz) : 1;
  q.rows = (int)rows;
  q.stages = (int)stages;
  q.unit_ctr = reinterpret_cast<int*>(unit_ctr);
  TORCH_CHECK(tail == 0 || (order == 1 && swz == 1 && tail <= 32), "v5: tail (producer tail_cols) needs order 1, swz 1");
  q.tail = (int)tail;
  return q;
}

// Mode-4 reducer kernel: `blocks` CTAs of 512 threads on the current stream (launch BEFORE the mode-4 GEMM, on another stream,
// with the GEMM's sms = SM count - blocks so both are resident). Arguments mirror tp4_down.
void tp4_reduce(int64_t M, int64_t rank, int64_t world, int64_t epoch, int64_t raster, int64_t blocks,
                at::Tensor tile_cnt, int64_t partials_mc, int64_t x_mc, std::optional<at::Tensor> resid, int64_t rowss_mc, int64_t panel_cnt_mc,
                int64_t unicast = 0, std::vector<int64_t> peer_partials = {}, std::vector<int64_t> peer_x = {}, int64_t version = 1,
                std::optional<at::Tensor> epoch_dev = std::nullopt, int64_t rowss_mc_base = 0, int64_t rowss_stride = 0,
                std::vector<int64_t> inbox_ptrs = {}, std::vector<int64_t> peer_rowss = {}, std::vector<int64_t> peer_panel_flag = {},
                int64_t rowss_local = 0, int64_t panel_done = 0, int64_t panels_per_owner = 0, int64_t threads = 1024,
                int64_t variant = 0, int64_t swz = 1, int64_t rows = 2, int64_t red_trace = 0, int64_t dsmem = 0, int64_t stages = 0,
                int64_t unit_ctr = 0, int64_t tail_cols = 0, int64_t pdl_trigger = 0) {
  c10::cuda::CUDAGuard guard(tile_cnt.device());
  TORCH_CHECK(version >= 1 && version <= 5, "version");
  if (version == 5) {
    // v5 owner reducer (see make_red5). variant 0 = signaller design, 1 = warp-per-unit (2 rows / lane), 2 = the S3 role
    // reducer as a stand-alone kernel (384 threads, 168-register budget, `rows` rows / lane).
    TORCH_CHECK(world == 4, "v5: world == 4");
    Tp4Red5Params q = make_red5(M, rank, epoch, raster, swz, tile_cnt, resid, inbox_ptrs, peer_x, peer_rowss, peer_panel_flag,
                                      rowss_local, panel_done, panels_per_owner, epoch_dev, variant == 2 ? rows : 2, red_trace,
                                      variant == 2 ? stages : 0, unit_ctr, tail_cols);
    TORCH_CHECK(stages == 0 || variant == 2, "v5: stages needs variant 2");
    TORCH_CHECK(unit_ctr == 0 || variant >= 1, "v5: unit_ctr (work stealing) needs a warp-per-unit variant (1 | 2)");
    TORCH_CHECK(pdl_trigger == 0 || variant >= 1, "v5: pdl_trigger needs variant 1 | 2");
    q.pdl = (int)pdl_trigger;   // "pdl1": trigger the PDL dependent (the producer) at kernel entry
    if (variant == 2 && stages > 0) dsmem = std::max<int64_t>(dsmem, kRed5aSmemOffset + 12 * red5a_warp_bytes(2, (int)stages));
    auto st = at::cuda::getCurrentCUDAStream();
    // dsmem > 0 (variants 1 / 2, S3 diagnostic): launch with that much dynamic smem and the max-shared carveout, i.e. the L1 size
    // the role CTAs of the consumer GEMM see (214016 B smem -> ~28 KB L1)
    if (dsmem > 0 && variant == 1 && threads == 1024) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(tp4_reduce5w_kernel<1024>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dsmem));
      C10_CUDA_CHECK(cudaFuncSetAttribute(tp4_reduce5w_kernel<1024>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    }
    if (dsmem > 0 && variant == 2 && threads == 384) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(tp4_reduce5r_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dsmem));
      C10_CUDA_CHECK(cudaFuncSetAttribute(tp4_reduce5r_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    }
    TORCH_CHECK(dsmem == 0 || variant >= 1, "v5: dsmem needs variant 1 or 2");
    if (variant == 0 && threads == 1024) tp4_reduce5_kernel<1024><<<(int)blocks, 1024, 0, st>>>(q);
    else if (variant == 0 && threads == 544) tp4_reduce5_kernel<544><<<(int)blocks, 544, 0, st>>>(q);
    else if (variant == 1 && threads == 1024) tp4_reduce5w_kernel<1024><<<(int)blocks, 1024, (size_t)dsmem, st>>>(q);
    else if (variant == 1 && threads == 512) tp4_reduce5w_kernel<512><<<(int)blocks, 512, 0, st>>>(q);
    else if (variant == 2 && threads == 384) tp4_reduce5r_kernel<<<(int)blocks, 384, (size_t)dsmem, st>>>(q);
    else TORCH_CHECK(false, "v5: (variant, threads) in {(0, 1024), (0, 544), (1, 1024), (1, 512), (2, 384)}");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  TORCH_CHECK(version == 1 || unicast == 0, "versions >= 2 have no unicast variant");
  TORCH_CHECK(raster == 1 || raster == 2, "raster must be 1 (AlongM, m fast) or 2 (AlongN, n fast); the reducer follows the GEMM's tile order");
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
// mode 5: panel_cnt is the v5 panel_flag array (value flag). mode 8 (S3): as mode 5, plus the first R CTAs to start run the v5
// owner reducer first (role_ctr: arrival counter zeroed by tp4_step; the red5 arguments as tp4_reduce version 5, M = X rows).
// tile: 256 (128x256x64, default) | 128 (128x128x64). pdl: launch as a programmatic dependent of the previous kernel on the stream.
void tp4_qkv(at::Tensor X, at::Tensor W, at::Tensor Y, int64_t raster, int64_t swizzle, int64_t mode, int64_t epoch,
             at::Tensor panel_cnt, int64_t panel_target, at::Tensor rowss, int64_t sms, int64_t pdl, std::optional<at::Tensor> epoch_dev,
             int64_t tile, int64_t R, std::optional<at::Tensor> role_ctr, int64_t rank, int64_t red_order, int64_t red_swz,
             std::optional<at::Tensor> tile_flags, std::optional<at::Tensor> resid, std::vector<int64_t> inbox_ptrs,
             std::vector<int64_t> peer_x, std::vector<int64_t> peer_rowss, std::vector<int64_t> peer_panel_flag, int64_t rowss_local,
             int64_t panel_done, int64_t panels_per_owner, int64_t red_rows, int64_t trace, int64_t red_trace, int64_t red_stages,
             int64_t red_dyn, int64_t red_tail) {
  check_bf16_2d(X, "X"); check_bf16_2d(W, "W"); check_bf16_2d(Y, "Y");
  c10::cuda::CUDAGuard guard(X.device());
  const int M = X.size(0), N = W.size(0);
  TORCH_CHECK(tile == 256 || tile == 128, "tile 256 | 128");
  TORCH_CHECK(X.size(1) == 8192, "RMSNorm scale is compiled for N1 = 8192");
  TORCH_CHECK(Y.size(0) == M && Y.size(1) == N && M % 128 == 0 && N % tile == 0, "shape");
  TORCH_CHECK(mode == 0 || mode == 2 || mode == 5 || mode == 8, "mode");
  TORCH_CHECK(rowss.dtype() == at::kFloat && rowss.numel() >= M, "rowss");
  TORCH_CHECK((mode != 5 && mode != 8) || panel_cnt.numel() >= M / 128, "panel_flag");
  CtappTp4Params p;
  p.mode = (int)mode; p.epoch = (int)epoch; p.panel_target = (int)panel_target; p.dbg = g_tp4_dbg;
  p.panel_cnt = reinterpret_cast<const unsigned*>(panel_cnt.data_ptr<int32_t>());
  p.epoch_ptr = epoch_dev ? epoch_dev->data_ptr<int32_t>() : nullptr;
  p.trace = reinterpret_cast<unsigned long long*>(trace);
  if (mode == 8) {
    int dev_sms = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&dev_sms, cudaDevAttrMultiProcessorCount, X.get_device()));
    const int64_t grid = std::min<int64_t>(sms > 0 ? sms : dev_sms, (int64_t)(M / 128) * (N / tile));   // >= the real grid (swizzle rounds up)
    TORCH_CHECK(R >= (red_dyn ? 0 : 1) && R <= std::min<int64_t>(grid, 32), "mode 8: 1 <= R <= min(grid, 32) (0 allowed with red_dyn)");
    TORCH_CHECK(role_ctr && role_ctr->dtype() == at::kInt && role_ctr->numel() >= 2, "mode 8: role_ctr (int32 [2], zeroed by tp4_step)");
    TORCH_CHECK(red_dyn == 0 || red_stages == 0, "mode 8: red_dyn uses the register loop (red_stages 0)");
    TORCH_CHECK(tile_flags.has_value(), "mode 8: tile_flags");
    TORCH_CHECK(!epoch_dev || epoch_dev.has_value(), "epoch");
    p.R = (int)R;
    p.role_ctr = role_ctr->data_ptr<int32_t>();
    p.red5 = make_red5(M, rank, epoch, red_order, red_swz, *tile_flags, resid, inbox_ptrs, peer_x, peer_rowss, peer_panel_flag,
                       rowss_local, panel_done, panels_per_owner, epoch_dev, red_rows, red_trace, red_stages,
                       red_dyn ? reinterpret_cast<int64_t>(role_ctr->data_ptr<int32_t>() + 1) : 0, red_tail);
    const int smem = tile == 128 ? (int)GemmQkv128::GemmKernel::SharedStorageSize : (int)GemmQkv::GemmKernel::SharedStorageSize;
    TORCH_CHECK(red_stages == 0 || kRed5aSmemOffset + 12 * red5a_warp_bytes(2, (int)red_stages) <= smem, "mode 8: red_stages ring > smem");
  }
  if (tile == 128) GemmQkv128::run(X, W, Y, (int)raster, (int)swizzle, p, rowss.data_ptr<float>(), at::cuda::getCurrentCUDAStream(), (int)sms, pdl != 0);
  else GemmQkv::run(X, W, Y, (int)raster, (int)swizzle, p, rowss.data_ptr<float>(), at::cuda::getCurrentCUDAStream(), (int)sms, pdl != 0);
}

// v5 per-step reset (see tp4_step_kernel): epoch_dev += 1, rowss_local[0:M] = 0, panel_done[0:M/128] = 0, on the current stream.
void tp4_step(at::Tensor epoch_dev, at::Tensor rowss_local, at::Tensor panel_done, int64_t M, std::optional<at::Tensor> role_ctr) {
  c10::cuda::CUDAGuard guard(epoch_dev.device());
  TORCH_CHECK(epoch_dev.dtype() == at::kInt && rowss_local.dtype() == at::kFloat && panel_done.dtype() == at::kInt, "tp4_step: dtypes");
  TORCH_CHECK(M % 128 == 0 && rowss_local.numel() >= M && panel_done.numel() >= M / 128, "tp4_step: sizes");
  TORCH_CHECK(!role_ctr || (role_ctr->dtype() == at::kInt && role_ctr->numel() >= 2), "tp4_step: role_ctr (int32 [2]: role, unit counters)");
  const int grid = std::max(1, std::min(64, (int)((M / 4 + 255) / 256)));
  tp4_step_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(epoch_dev.data_ptr<int32_t>(), reinterpret_cast<float4*>(rowss_local.data_ptr<float>()),
                                                                     reinterpret_cast<unsigned*>(panel_done.data_ptr<int32_t>()), (int)M,
                                                                     role_ctr ? role_ctr->data_ptr<int32_t>() : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <class F>
static std::string func_attrs(const char* name, F f) {
  cudaFuncAttributes a;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a, f));
  return std::string(name) + ": regs=" + std::to_string(a.numRegs) + " local(spill)=" + std::to_string(a.localSizeBytes) + "B maxthreads=" +
         std::to_string(a.maxThreadsPerBlock) + " smem=" + std::to_string(a.sharedSizeBytes);
}

std::string tp4_info(int64_t which) {
  c10::cuda::CUDAGuard guard(at::cuda::current_device());
  if (which == 2)
    return func_attrs("tp4_reduce5_kernel<1024>", tp4_reduce5_kernel<1024>) + "; " + func_attrs("tp4_reduce5_kernel<544>", tp4_reduce5_kernel<544>) +
           "; " + func_attrs("tp4_reduce5w_kernel<1024>", tp4_reduce5w_kernel<1024>) + "; " + func_attrs("tp4_reduce5w_kernel<512>", tp4_reduce5w_kernel<512>) +
           "; " + func_attrs("tp4_reduce3_kernel<2>", tp4_reduce3_kernel<2>) + "; " + func_attrs("tp4_reduce5r_kernel", tp4_reduce5r_kernel);
  if (which == 3) return GemmQkv128::info();
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

// ======================================================================================================================
// Phase-6 S1 go/no-go probes (bench/push_bw.py). Nothing here is used by the production protocol paths.
// ======================================================================================================================
namespace p6 {
static constexpr unsigned kSpin = 1u << 24;   // probe spin limit (polls); a hang traps instead of blocking the job
__device__ __forceinline__ unsigned smid() { unsigned s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); return s; }
__device__ __forceinline__ unsigned ld_acquire_gpu(const unsigned* p) {
  unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ void st_relaxed_sys_u32(unsigned* p, unsigned v) { asm volatile("st.relaxed.sys.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void st_release_gpu_u32(unsigned* p, unsigned v) { asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void spin_ge_sys(const unsigned* p, unsigned t) { unsigned n = 0; while (tp4::ld_acquire_sys(p) < t) { if (++n == kSpin) __trap(); } }
__device__ __forceinline__ void spin_ge_gpu(const unsigned* p, unsigned t) { unsigned n = 0; while (ld_acquire_gpu(p) < t) { if (++n == kSpin) __trap(); } }
__device__ __forceinline__ void spin_ge_gpu_sleep(const unsigned* p, unsigned t) {
  unsigned n = 0; while (ld_acquire_gpu(p) < t) { __nanosleep(500); if (++n == kSpin) __trap(); }
}
// out = bf16(a0 + a1 + a2 + a3 + r), fp32 accumulation
__device__ __forceinline__ uint4 sum5_bf16x8(uint4 a0, uint4 a1, uint4 a2, uint4 a3, uint4 r) {
  unsigned o[4];
  const unsigned* x0 = &a0.x; const unsigned* x1 = &a1.x; const unsigned* x2 = &a2.x; const unsigned* x3 = &a3.x; const unsigned* xr = &r.x;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    float2 f0 = tp4::unpack(x0[i]), f1 = tp4::unpack(x1[i]), f2 = tp4::unpack(x2[i]), f3 = tp4::unpack(x3[i]), fr = tp4::unpack(xr[i]);
    o[i] = tp4::pack(f0.x + f1.x + f2.x + f3.x + fr.x, f0.y + f1.y + f2.y + f3.y + fr.y);
  }
  return make_uint4(o[0], o[1], o[2], o[3]);
}
}  // namespace p6

// ---- probe 1: reducer-shaped owner-side throughput (no GEMM, no flag waits) ----
// Rank `rank` owns the 128-row panels m with m % 4 == rank. Work unit u = (owned panel j, n-tile of 256 cols, 32-row quarter q):
// q = u & 3, n = (u >> 2) & 31, j = u >> 7, global panel m = 4 j + rank. Per unit each worker thread handles K 16 B vectors
// (vector e = tid + k * W of the 32 x 32-vector unit; a warp covers one row per k): 4 inbox loads (all issued first) + resid load,
// fp32 add, st.relaxed.sys.v4 to 4 destinations (peers rank+1, rank+2, rank+3, then local; no_remote: local only), and one
// red.add.f32 (atomicAdd, gpu scope) per row into the local rowss. The last warp is a signaller (v3 barrier pattern: 2 = batch done,
// 3 = batch released; workers <= 1 batch ahead): per batch of T units one fence.acq_rel.sys (unless no_fence) + st.relaxed.sys flag.
struct P6ReduceArgs {
  const uint4* inbox[4];        // local, (M/4, 8192) bf16 each
  const uint4* resid;           // local, (M, 8192) bf16
  uint4* x[4];                  // rank r's x (M, 8192) bf16; x[rank] is local
  float* rowss;                 // local [M]
  unsigned* flags;              // local [nbatch]
  unsigned* ctl;                // [0] done counter (monotone, +1 per CTA), [1] blocker release flag
  int* smid;                    // [gridDim.x]
  unsigned long long* stamps;   // [gridDim.x][2] %globaltimer CTA start / end
  int rank, units, nbatch;
  unsigned flagval, done_target;
  int no_remote, no_fence;
};

template <int THREADS, int T>
__global__ void __launch_bounds__(THREADS, 2) p6_reduce_probe_kernel(P6ReduceArgs a) {
  constexpr int W = THREADS - 32;            // worker threads
  constexpr int K = (1024 + W - 1) / W;      // vectors per worker thread per unit (1024 x 16 B per unit)
  const int tid = threadIdx.x;
  if (tid == 0) { a.stamps[2 * blockIdx.x] = tp4::globaltimer(); a.smid[blockIdx.x] = (int)p6::smid(); }
  int iters = 0;
  for (int b = blockIdx.x; b < a.nbatch; b += gridDim.x) iters++;
  if (tid >= W) {                            // signaller warp
    cutlass::arch::NamedBarrier::arrive(THREADS, 3);
    for (int i = 0; i < iters; i++) {
      cutlass::arch::NamedBarrier::sync(THREADS, 2);
      if (i + 1 < iters) cutlass::arch::NamedBarrier::arrive(THREADS, 3);
      if (tid == W) {
        if (!a.no_fence) tp4::fence_acq_rel_sys();
        p6::st_relaxed_sys_u32(a.flags + blockIdx.x + i * gridDim.x, a.flagval);
      }
      __syncwarp();
    }
  } else {
    uint4* xo[4];                            // destination order: rank+1, rank+2, rank+3, local
    #pragma unroll
    for (int d = 0; d < 4; d++) {
      const int r = (a.rank + 1 + d) & 3;
      uint4* p = a.x[0];
      #pragma unroll
      for (int i = 1; i < 4; i++) if (r == i) p = a.x[i];
      xo[d] = p;
    }
    for (int i = 0; i < iters; i++) {
      cutlass::arch::NamedBarrier::sync(THREADS, 3);
      const int b = blockIdx.x + i * gridDim.x;
      #pragma unroll 1
      for (int t = 0; t < T; t++) {
        const int u = b * T + t;
        if (u >= a.units) break;             // CTA-uniform
        const int q = u & 3, n = (u >> 2) & 31, j = u >> 7, m = 4 * j + a.rank;
        unsigned oi[K], ox[K];               // uint4 indices (< 2^32 for M <= 16k)
        uint4 v[K][4], r[K];
        #pragma unroll
        for (int k = 0; k < K; k++) {
          const int e = tid + k * W, rr = e >> 5, c = e & 31;
          oi[k] = (unsigned)(j * 128 + q * 32 + rr) * 1024u + n * 32 + c;
          ox[k] = (unsigned)(m * 128 + q * 32 + rr) * 1024u + n * 32 + c;
        }
        #pragma unroll
        for (int k = 0; k < K; k++) if (tid + k * W < 1024) {
          #pragma unroll
          for (int s = 0; s < 4; s++) v[k][s] = __ldcg(a.inbox[s] + oi[k]);
          r[k] = __ldg(a.resid + ox[k]);
        }
        #pragma unroll
        for (int k = 0; k < K; k++) if (tid + k * W < 1024) {   // warp-uniform (W % 32 == 0)
          const uint4 o = p6::sum5_bf16x8(v[k][0], v[k][1], v[k][2], v[k][3], r[k]);
          if (a.no_remote) ::st_relaxed_sys_v4(xo[3] + ox[k], o);
          else {
            #pragma unroll
            for (int d = 0; d < 4; d++) ::st_relaxed_sys_v4(xo[d] + ox[k], o);
          }
          float ss = tp4::sumsq_add(o, 0.f);
          #pragma unroll
          for (int sh = 16; sh > 0; sh >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, sh);
          if ((tid & 31) == 0) atomicAdd(a.rowss + m * 128 + q * 32 + ((tid + k * W) >> 5), ss);
        }
      }
      cutlass::arch::NamedBarrier::arrive(THREADS, 2);
    }
  }
  __syncthreads();
  if (tid == 0) {
    a.stamps[2 * blockIdx.x + 1] = tp4::globaltimer();
    __threadfence();
    if (atomicAdd(a.ctl, 1u) + 1u == a.done_target) p6::st_release_gpu_u32(a.ctl + 1, a.flagval);
  }
}

// Blocker: grid = SMs to occupy, 32 threads, large dynamic smem (one CTA per SM; a probe CTA with >= 32 KB smem cannot co-reside).
// Spins (nanosleep) until ctl[1] >= target. started is a monotone counter of resident blocker CTAs.
__global__ void p6_blocker_kernel(const unsigned* release, unsigned target, unsigned* started) {
  if (threadIdx.x == 0) { atomicAdd(started, 1u); p6::spin_ge_gpu_sleep(release, target); }
}
__global__ void p6_wait_kernel(const unsigned* started, unsigned target) { p6::spin_ge_gpu_sleep(started, target); }
__global__ void p6_delay_kernel(long long ns) {
  const unsigned long long t0 = tp4::globaltimer();
  while ((long long)(tp4::globaltimer() - t0) < ns) {}
}
__global__ void p6_stamp_kernel(unsigned long long* out) { out[0] = tp4::globaltimer(); }

// Small helper kernels get the max-shared carveout: an SM running a 0-smem kernel configured with a small carveout cannot take a
// 200 KB blocker CTA (or a GEMM CTA) until it exits.
static void p6_max_carveout(const void* f) {
  C10_CUDA_CHECK(cudaFuncSetAttribute(f, cudaFuncAttributePreferredSharedMemoryCarveout, (int)cudaSharedmemCarveoutMaxShared));
}
void p6_reduce_probe(at::Tensor inbox, at::Tensor resid, std::vector<int64_t> x_ptrs, at::Tensor rowss, at::Tensor flags, at::Tensor ctl,
                     at::Tensor smid, at::Tensor stamps, int64_t rank, int64_t M, int64_t blocks, int64_t threads, int64_t T,
                     int64_t no_remote, int64_t no_fence, int64_t flagval, int64_t done_target, int64_t smem_pad) {
  c10::cuda::CUDAGuard guard(inbox.device());
  TORCH_CHECK(M % 512 == 0 && inbox.numel() == M * 8192 && resid.numel() >= M * 8192 && x_ptrs.size() == 4, "p6_reduce_probe: shapes");
  TORCH_CHECK(rowss.numel() >= M && smid.numel() >= blocks && stamps.numel() >= 2 * blocks, "p6_reduce_probe: buffers");
  P6ReduceArgs a;
  const size_t q4 = (size_t)(M / 4) * 8192 / 8;
  for (int s = 0; s < 4; s++) a.inbox[s] = reinterpret_cast<const uint4*>(inbox.data_ptr()) + s * q4;
  a.resid = reinterpret_cast<const uint4*>(resid.data_ptr());
  for (int r = 0; r < 4; r++) a.x[r] = reinterpret_cast<uint4*>(x_ptrs[r]);
  a.rowss = rowss.data_ptr<float>();
  a.ctl = reinterpret_cast<unsigned*>(ctl.data_ptr<int32_t>());
  a.smid = smid.data_ptr<int32_t>();
  a.stamps = reinterpret_cast<unsigned long long*>(stamps.data_ptr<int64_t>());
  a.rank = (int)rank; a.units = (int)(M / 4); a.nbatch = (int)((a.units + T - 1) / T);
  TORCH_CHECK(flags.numel() >= a.nbatch, "p6_reduce_probe: flags");
  a.flags = reinterpret_cast<unsigned*>(flags.data_ptr<int32_t>());
  a.flagval = (unsigned)flagval; a.done_target = (unsigned)done_target; a.no_remote = (int)no_remote; a.no_fence = (int)no_fence;
  auto st = at::cuda::getCurrentCUDAStream();
  const size_t sm = (size_t)smem_pad;
  TORCH_CHECK(sm <= 48 * 1024, "smem_pad <= 48 KB");
#define P6R(TH, TT) if (threads == TH && T == TT) { p6_reduce_probe_kernel<TH, TT><<<(int)blocks, TH, sm, st>>>(a); C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
  P6R(544, 2) P6R(544, 4) P6R(384, 2) P6R(384, 4)
#undef P6R
  TORCH_CHECK(false, "p6_reduce_probe: threads in {384, 544}, T in {2, 4}");
}
void p6_blocker(at::Tensor ctl, int64_t target, int64_t grid, int64_t smem) {   // ctl[1] = release flag, ctl[2] = started counter
  c10::cuda::CUDAGuard guard(ctl.device());
  static bool init = false;
  if (!init) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(p6_blocker_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024));
    p6_max_carveout((const void*)p6_blocker_kernel);
    init = true;
  }
  unsigned* c = reinterpret_cast<unsigned*>(ctl.data_ptr<int32_t>());
  p6_blocker_kernel<<<(int)grid, 32, (size_t)smem, at::cuda::getCurrentCUDAStream()>>>(c + 1, (unsigned)target, c + 2);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void p6_wait(at::Tensor ctl, int64_t target) {   // spin until ctl[2] (blocker started counter) >= target
  c10::cuda::CUDAGuard guard(ctl.device());
  static bool init = false;
  if (!init) { p6_max_carveout((const void*)p6_wait_kernel); init = true; }
  p6_wait_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const unsigned*>(ctl.data_ptr<int32_t>()) + 2, (unsigned)target);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void p6_delay(int64_t ns) {
  static bool init = false;
  if (!init) { p6_max_carveout((const void*)p6_delay_kernel); init = true; }
  p6_delay_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>((long long)ns); C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void p6_stamp(at::Tensor out, int64_t idx) {
  c10::cuda::CUDAGuard guard(out.device());
  static bool init = false;
  if (!init) { p6_max_carveout((const void*)p6_stamp_kernel); init = true; }
  p6_stamp_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<unsigned long long*>(out.data_ptr<int64_t>()) + idx);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- probe 4: flag-chain ordering stress and latency ----
// Roles by blockIdx: [0, S) A (worker), [S, 2S) B (signaller), [2S, 3S) C (checker); 256 threads; slot k = blockIdx % S.
// Rank s sends to d = s + 1 and checks what s - 1 sent. Per iteration it = 1..iters (value = it):
//   A: wait own ack[k] >= it - 1 (ld.acquire.sys); n4 x 16 B (default 16 KB) of value it -> d's data[k] (st.relaxed.sys.v4); bar;
//      thread 0: fence.acq_rel.sys (unless mode & 2); then two-hop: atomicAdd(local counter[k], 1) / one-hop (mode & 1): st.relaxed.sys
//      flag[k] = it into d.
//   B (two-hop only, thread 0): spin local counter[k] >= it (ld.acquire.gpu); fence.acq_rel.sys; st.relaxed.sys flag[k] = it into d.
//   C: thread 0 spin own flag[k] >= it (ld.acquire.sys); bar; all threads plain-load the 16 KB and count words != it; bar;
//      thread 0 fence.acq_rel.sys + st.relaxed.sys ack[k] = it into s - 1.
// ns[k] = A's wall time for all iterations (incl. the final ack) -> cycle time per iteration.
struct P6ChainArgs {
  uint4* peer_data; const uint4* my_data;
  unsigned* peer_flag; const unsigned* my_flag;
  unsigned* peer_ack; const unsigned* my_ack;
  unsigned* counter;
  int* bad; unsigned long long* ns;
  int iters, slots, mode, n4;   // n4: payload per slot in 16 B vectors (multiple of 256)
};
__global__ void __launch_bounds__(256) p6_chain_stress_kernel(P6ChainArgs a) {
  const int role = blockIdx.x / a.slots, k = blockIdx.x % a.slots, tid = threadIdx.x;
  const bool onehop = a.mode & 1, nofence = a.mode & 2;
  if (role == 0) {
    uint4* dst = a.peer_data + (size_t)k * a.n4 + tid;
    const unsigned* ack = a.my_ack + k;
    unsigned* sig = onehop ? a.peer_flag + k : a.counter + k;
    unsigned long long* nsp = a.ns + k;
    unsigned long long t0 = tp4::globaltimer();
    for (int it = 1; it <= a.iters; it++) {
      if (tid == 0) p6::spin_ge_sys(ack, (unsigned)(it - 1));
      __syncthreads();
      const unsigned v = (unsigned)it;
      #pragma unroll 4
      for (int i = 0; i < a.n4; i += 256) ::st_relaxed_sys_v4(dst + i, make_uint4(v, v, v, v));
      __syncthreads();
      if (tid == 0) {
        if (!nofence) tp4::fence_acq_rel_sys();
        if (onehop) p6::st_relaxed_sys_u32(sig, v);
        else atomicAdd(sig, 1u);
      }
    }
    if (tid == 0) { p6::spin_ge_sys(ack, (unsigned)a.iters); *nsp = tp4::globaltimer() - t0; }
  } else if (role == 1) {
    if (onehop || tid != 0) return;
    for (int it = 1; it <= a.iters; it++) {
      p6::spin_ge_gpu(a.counter + k, (unsigned)it);
      tp4::fence_acq_rel_sys();
      p6::st_relaxed_sys_u32(a.peer_flag + k, (unsigned)it);
    }
  } else {
    const uint4* src = a.my_data + (size_t)k * a.n4 + tid;
    int nbad = 0;
    for (int it = 1; it <= a.iters; it++) {
      if (tid == 0) p6::spin_ge_sys(a.my_flag + k, (unsigned)it);
      __syncthreads();
      const unsigned v = (unsigned)it;
      #pragma unroll 4
      for (int i = 0; i < a.n4; i += 256) {
        const uint4 x = src[i];
        nbad += (x.x != v) + (x.y != v) + (x.z != v) + (x.w != v);
      }
      __syncthreads();
      if (tid == 0) { tp4::fence_acq_rel_sys(); p6::st_relaxed_sys_u32(a.peer_ack + k, v); }
    }
    if (nbad) atomicAdd(a.bad, nbad);
  }
}
// Returns [violations, ns_slot0, ..., ns_slot{S-1}].
std::vector<int64_t> p6_chain_stress(int64_t peer_data, at::Tensor my_data, int64_t peer_flag, at::Tensor my_flag, int64_t peer_ack, at::Tensor my_ack,
                                     at::Tensor counter, int64_t mode, int64_t iters, int64_t slots, int64_t n4) {
  c10::cuda::CUDAGuard guard(my_data.device());
  TORCH_CHECK(n4 > 0 && n4 % 256 == 0, "p6_chain_stress: n4 must be a positive multiple of 256");
  TORCH_CHECK(my_data.nbytes() >= (size_t)slots * n4 * 16 && my_flag.numel() >= slots && my_ack.numel() >= slots && counter.numel() >= slots, "p6_chain_stress: buffers");
  auto bad = at::zeros({1}, my_data.options().dtype(at::kInt));
  auto ns = at::zeros({slots}, my_data.options().dtype(at::kLong));
  P6ChainArgs a;
  a.peer_data = reinterpret_cast<uint4*>(peer_data); a.my_data = reinterpret_cast<const uint4*>(my_data.data_ptr());
  a.peer_flag = reinterpret_cast<unsigned*>(peer_flag); a.my_flag = reinterpret_cast<const unsigned*>(my_flag.data_ptr<int32_t>());
  a.peer_ack = reinterpret_cast<unsigned*>(peer_ack); a.my_ack = reinterpret_cast<const unsigned*>(my_ack.data_ptr<int32_t>());
  a.counter = reinterpret_cast<unsigned*>(counter.data_ptr<int32_t>());
  a.bad = bad.data_ptr<int>(); a.ns = reinterpret_cast<unsigned long long*>(ns.data_ptr<int64_t>());
  a.iters = (int)iters; a.slots = (int)slots; a.mode = (int)mode; a.n4 = (int)n4;
  p6_chain_stress_kernel<<<(int)(3 * slots), 256, 0, at::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  std::vector<int64_t> out{(int64_t)bad.item<int>()};
  auto h = ns.cpu();
  for (int i = 0; i < slots; i++) out.push_back(h[i].item<int64_t>());
  return out;
}

// ---- probe 5: PDL early launch ----
// Primary: every thread executes griddepcontrol.launch_dependents first; thread 0 stamps start, spins spin_ns, stamps end.
// Dependent (launched with programmatic stream serialization when pdl != 0): thread 0 stamps start + %smid; no griddepcontrol.wait.
__global__ void __launch_bounds__(384, 1) p6_pdl_primary_kernel(unsigned long long* t, int* sm, long long spin_ns) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (threadIdx.x == 0) {
    const unsigned long long t0 = tp4::globaltimer();
    t[2 * blockIdx.x] = t0; sm[blockIdx.x] = (int)p6::smid();
    while ((long long)(tp4::globaltimer() - t0) < spin_ns) {}
    t[2 * blockIdx.x + 1] = tp4::globaltimer();
  }
}
__global__ void __launch_bounds__(384, 1) p6_pdl_dependent_kernel(unsigned long long* t, int* sm, long long spin_ns) {
  if (threadIdx.x == 0) {
    const unsigned long long t0 = tp4::globaltimer();
    t[blockIdx.x] = t0; sm[blockIdx.x] = (int)p6::smid();
    while ((long long)(tp4::globaltimer() - t0) < spin_ns) {}   // optional: hold the SM (0 = stamp and exit)
  }
}
void p6_pdl_probe(at::Tensor tp, at::Tensor sp, at::Tensor td, at::Tensor sd, int64_t spin_ns, int64_t pdl, int64_t smem, int64_t grid_p, int64_t grid_d,
                  int64_t dep_spin_ns) {
  c10::cuda::CUDAGuard guard(tp.device());
  TORCH_CHECK(tp.numel() >= 2 * grid_p && sp.numel() >= grid_p && td.numel() >= grid_d && sd.numel() >= grid_d, "p6_pdl_probe: buffers");
  static bool init = false;
  if (!init) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(p6_pdl_primary_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024));
    C10_CUDA_CHECK(cudaFuncSetAttribute(p6_pdl_dependent_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024));
    init = true;
  }
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  cudaLaunchConfig_t c1 = {};
  c1.gridDim = dim3((unsigned)grid_p); c1.blockDim = dim3(384); c1.dynamicSmemBytes = (size_t)smem; c1.stream = st;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&c1, p6_pdl_primary_kernel, reinterpret_cast<unsigned long long*>(tp.data_ptr<int64_t>()), sp.data_ptr<int32_t>(), (long long)spin_ns));
  cudaLaunchAttribute at1[1];
  at1[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at1[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cudaLaunchConfig_t c2 = {};
  c2.gridDim = dim3((unsigned)grid_d); c2.blockDim = dim3(384); c2.dynamicSmemBytes = (size_t)smem; c2.stream = st;
  c2.attrs = at1; c2.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&c2, p6_pdl_dependent_kernel, reinterpret_cast<unsigned long long*>(td.data_ptr<int64_t>()), sd.data_ptr<int32_t>(), (long long)dep_spin_ns));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tp4_down", &tp4_down, py::arg("A"), py::arg("W"), py::arg("D"), py::arg("raster"), py::arg("swizzle"), py::arg("mode"), py::arg("rank"), py::arg("world"),
        py::arg("epoch"), py::arg("tile_cnt_mc"), py::arg("tile_cnt"), py::arg("partials_mc"), py::arg("x_mc"), py::arg("resid"), py::arg("rowss_mc"), py::arg("panel_cnt_mc"), py::arg("sms") = 0,
        py::arg("epoch_dev") = std::nullopt, py::arg("rowss_mc_base") = 0, py::arg("rowss_stride") = 0, py::arg("d_ptr") = 0,
        py::arg("dst_ptrs") = std::vector<int64_t>{}, py::arg("dst_rows") = 0, py::arg("peer_tile_flags") = std::vector<int64_t>{}, py::arg("panels_per_owner") = 0,
        py::arg("pdl_trigger") = 0, py::arg("trace") = 0, py::arg("tail_cols") = 0, py::arg("pdl") = 0);
  m.def("tp4_qkv", &tp4_qkv, py::arg("X"), py::arg("W"), py::arg("Y"), py::arg("raster"), py::arg("swizzle"), py::arg("mode"), py::arg("epoch"),
        py::arg("panel_cnt"), py::arg("panel_target"), py::arg("rowss"), py::arg("sms") = 0, py::arg("pdl") = 0, py::arg("epoch_dev") = std::nullopt,
        py::arg("tile") = 256, py::arg("R") = 0, py::arg("role_ctr") = std::nullopt, py::arg("rank") = 0, py::arg("red_order") = 1,
        py::arg("red_swz") = 1, py::arg("tile_flags") = std::nullopt, py::arg("resid") = std::nullopt,
        py::arg("inbox_ptrs") = std::vector<int64_t>{}, py::arg("peer_x") = std::vector<int64_t>{}, py::arg("peer_rowss") = std::vector<int64_t>{},
        py::arg("peer_panel_flag") = std::vector<int64_t>{}, py::arg("rowss_local") = 0, py::arg("panel_done") = 0, py::arg("panels_per_owner") = 0,
        py::arg("red_rows") = 4, py::arg("trace") = 0, py::arg("red_trace") = 0, py::arg("red_stages") = 0, py::arg("red_dyn") = 0, py::arg("red_tail") = 0);
  m.def("tp4_reduce", &tp4_reduce, py::arg("M"), py::arg("rank"), py::arg("world"), py::arg("epoch"), py::arg("raster"), py::arg("blocks"), py::arg("tile_cnt"), py::arg("partials_mc"), py::arg("x_mc"), py::arg("resid"), py::arg("rowss_mc"), py::arg("panel_cnt_mc"), py::arg("unicast") = 0, py::arg("peer_partials") = std::vector<int64_t>{}, py::arg("peer_x") = std::vector<int64_t>{}, py::arg("version") = 1,
        py::arg("epoch_dev") = std::nullopt, py::arg("rowss_mc_base") = 0, py::arg("rowss_stride") = 0,
        py::arg("inbox_ptrs") = std::vector<int64_t>{}, py::arg("peer_rowss") = std::vector<int64_t>{}, py::arg("peer_panel_flag") = std::vector<int64_t>{},
        py::arg("rowss_local") = 0, py::arg("panel_done") = 0, py::arg("panels_per_owner") = 0, py::arg("threads") = 1024,
        py::arg("variant") = 0, py::arg("swz") = 1, py::arg("rows") = 2, py::arg("red_trace") = 0, py::arg("dsmem") = 0, py::arg("stages") = 0, py::arg("unit_ctr") = 0, py::arg("tail_cols") = 0, py::arg("pdl_trigger") = 0);
  m.def("tp4_step", &tp4_step, py::arg("epoch_dev"), py::arg("rowss_local"), py::arg("panel_done"), py::arg("M"), py::arg("role_ctr") = std::nullopt);
  m.def("tp4_info", &tp4_info); m.def("tp4_set_debug", &tp4_set_debug);
  m.def("mm_red_u32", &mm_red_u32); m.def("mm_red_f32", &mm_red_f32); m.def("mm_ld_reduce_bf16", &mm_ld_reduce_bf16); m.def("mm_st_bf16", &mm_st_bf16); m.def("mm_red_bf16", &mm_red_bf16);
  m.def("mm_order_stress", &mm_order_stress); m.def("mm_latency", &mm_latency);
  // Phase-6 S1 probes (bench/push_bw.py)
  m.def("p6_reduce_probe", &p6_reduce_probe, py::arg("inbox"), py::arg("resid"), py::arg("x_ptrs"), py::arg("rowss"), py::arg("flags"), py::arg("ctl"),
        py::arg("smid"), py::arg("stamps"), py::arg("rank"), py::arg("M"), py::arg("blocks"), py::arg("threads"), py::arg("T"), py::arg("no_remote") = 0,
        py::arg("no_fence") = 0, py::arg("flagval") = 1, py::arg("done_target") = 0, py::arg("smem_pad") = 0);
  m.def("p6_blocker", &p6_blocker); m.def("p6_wait", &p6_wait); m.def("p6_delay", &p6_delay); m.def("p6_stamp", &p6_stamp);
  m.def("p6_chain_stress", &p6_chain_stress); m.def("p6_pdl_probe", &p6_pdl_probe);
}
