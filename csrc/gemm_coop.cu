// Stock CUTLASS SM90 persistent cooperative BF16 TN GEMM (TMA warp-specialized epilogue), Y[M,N] = X[M,K] * W[N,K]^T.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/kernel_hardware_info.h"

using namespace cute;

// Tile scheduler flavours: a later step adds a tag with the same shape (Type, name).
struct SchedStock { using Type = cutlass::gemm::PersistentScheduler; static constexpr const char* name = "persistent"; };

// TMA epilogue. StagesD = 0: builder default (StagesD = min(subtiles, 2)). Otherwise the builder's own type computation
// (builders/sm90_builder.inl) with StagesD smem D buffers (-1 = whole tile) and an EPI_N-wide subtile (0 = auto, 32).
template <class TileShape, int StagesD, int EpiN>
struct CoopEpilogue {
  using Element = cutlass::bfloat16_t;
  using Schedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using Builder = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, Shape<_1, _1, _1>,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float,
      void, cutlass::layout::RowMajor, 8,
      Element, cutlass::layout::RowMajor, 8, Schedule>::CollectiveOp;
  using EpiTile = decltype(cutlass::epilogue::collective::detail::sm90_compute_tile_shape_or_override<
      Element, cute::conditional_t<EpiN == 0, cutlass::epilogue::collective::EpilogueTileAuto, Shape<_128, Int<EpiN>>>, Schedule, TileShape>());
  static constexpr int EpiTiles = decltype(size(shape_div(take<0, 2>(TileShape{}), EpiTile{})))::value;
  static constexpr int Frag = decltype(size(EpiTile{}))::value / 256;
  using Policy = cutlass::epilogue::Sm90TmaWarpSpecialized<(EpiTiles < 4 ? EpiTiles : 4), (StagesD < 0 ? EpiTiles : StagesD), Frag, false, true>;
  using Custom = typename cutlass::epilogue::collective::detail::Sm90TmaBuilderImpl<
      TileShape, EpiTile, float, float, void, cutlass::layout::RowMajor, 8, Element, cutlass::layout::RowMajor, 8,
      cutlass::epilogue::fusion::LinearCombination<Element, float, void, float>, Policy>::CollectiveOp;
  using Op = cute::conditional_t<StagesD == 0, Builder, Custom>;
};
static_assert(std::is_same_v<CoopEpilogue<Shape<_128, _256, _64>, 2, 0>::Custom, CoopEpilogue<Shape<_128, _256, _64>, 2, 0>::Builder>,
              "custom epilogue type computation must match the builder");

template <int BM, int BN, int BK, class Sched = SchedStock, int StagesD = 0, int EpiN = 0>
struct GemmCoop {
  using Element = cutlass::bfloat16_t;
  using TileShape = Shape<Int<BM>, Int<BN>, Int<BK>>;
  using ClusterShape = Shape<_1, _1, _1>;

  using CollectiveEpilogue = typename CoopEpilogue<TileShape, StagesD, EpiN>::Op;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Element, cutlass::layout::RowMajor, 8,
      Element, cutlass::layout::ColumnMajor, 8,
      float, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, typename Sched::Type>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  static typename Gemm::Arguments make_args(int M, int N, int K, const void* x, const void* w, void* y, int raster, int swizzle) {
    auto hw = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel>(at::cuda::current_device());
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
        {reinterpret_cast<const Element*>(x), cutlass::make_cute_packed_stride(typename GemmKernel::StrideA{}, {M, K, 1}),
         reinterpret_cast<const Element*>(w), cutlass::make_cute_packed_stride(typename GemmKernel::StrideB{}, {N, K, 1})},
        {{1.f, 0.f}, nullptr, typename GemmKernel::StrideC{},
         reinterpret_cast<Element*>(y), cutlass::make_cute_packed_stride(typename GemmKernel::StrideD{}, {M, N, 1})},
        hw};
    using RO = decltype(args.scheduler.raster_order);
    args.scheduler.raster_order = static_cast<RO>(raster);  // 0 Heuristic, 1 AlongN, 2 AlongM
    args.scheduler.max_swizzle_size = swizzle;
    return args;
  }

  static void run(const at::Tensor& X, const at::Tensor& W, const at::Tensor& Y, int raster, int swizzle, cudaStream_t stream) {
    auto args = make_args(X.size(0), W.size(0), X.size(1), X.data_ptr(), W.data_ptr(), Y.data_ptr(), raster, swizzle);
    Gemm gemm;
    TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "can_implement failed");
    auto ws = at::empty({(int64_t)Gemm::get_workspace_size(args)}, X.options().dtype(at::kByte));
    TORCH_CHECK(gemm.initialize(args, ws.data_ptr(), stream) == cutlass::Status::kSuccess, "initialize failed");
    TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess, "run failed");
  }

  static std::string info() {
    auto args = make_args(16384, 8192, 8192, nullptr, nullptr, nullptr, 0, 1);
    auto grid = Gemm::get_grid_shape(args);
    return std::string("coop-") + Sched::name + " tile " + std::to_string(BM) + "x" + std::to_string(BN) + "x" + std::to_string(BK) +
           " threads=" + std::to_string(GemmKernel::MaxThreadsPerBlock) +
           " smem=" + std::to_string(GemmKernel::SharedStorageSize) +
           " stages=" + std::to_string(CollectiveMainloop::DispatchPolicy::Stages) +
           " grid(M=16384)=" + std::to_string(grid.x) + "x" + std::to_string(grid.y) + "x" + std::to_string(grid.z);
  }
};

#define FOR_COOP_CONFIGS(F)                   \
  case 0: F(GemmCoop<128, 256, 64>); break;   \
  case 1: F(GemmCoop<128, 128, 64>); break;   \
  case 2: F(GemmCoop<256, 128, 64>); break;

void coop_gemm(at::Tensor X, at::Tensor W, at::Tensor Y, int64_t config_id, int64_t raster, int64_t swizzle) {
  c10::cuda::CUDAGuard guard(X.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#define RUN(...) __VA_ARGS__::run(X, W, Y, (int)raster, (int)swizzle, stream)
  switch (config_id) { FOR_COOP_CONFIGS(RUN) default: TORCH_CHECK(false, "bad coop config_id"); }
#undef RUN
}

std::string coop_info(int64_t config_id) {
  c10::cuda::CUDAGuard guard(at::cuda::current_device());
#define INFO(...) return __VA_ARGS__::info()
  switch (config_id) { FOR_COOP_CONFIGS(INFO) default: TORCH_CHECK(false, "bad coop config_id"); }
#undef INFO
}

void register_coop(pybind11::module& m) {
  m.def("coop_gemm", &coop_gemm, py::arg("X"), py::arg("W"), py::arg("Y"), py::arg("cfg"), py::arg("raster") = 0, py::arg("swizzle") = 1);
  m.def("coop_info", &coop_info);
}
