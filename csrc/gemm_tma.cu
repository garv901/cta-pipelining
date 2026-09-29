// Plain CUTLASS SM90 KernelTma BF16 GEMM, Y[M,N] = X[M,K] * W[N,K]^T, register-to-gmem epilogue (NoSmem).
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/sm70_epilogue_vectorized.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/device_kernel.h"
#include "cutlass/util/packed_stride.hpp"
#include "sm90_gemm_tma_ctapp.hpp"
#include <cuda.h>

using namespace cute;

// Register -> smem (fp32 tile, swizzled) -> 128-bit row-contiguous global stores, 16 threads along N per row.
template <int BM, int BN>
struct SmemEpilogue {
  using Element = cutlass::bfloat16_t;
  using Collective = cutlass::epilogue::collective::Epilogue<
      Stride<int64_t, _1, int64_t>, Stride<int64_t, _1, int64_t>,
      cutlass::epilogue::thread::LinearCombination<Element, 1, float, float>,
      ComposedLayout<Swizzle<3, 3, 4>, smem_ptr_flag_bits<32>, Layout<Shape<Int<BM>, Int<BN>>, Stride<Int<BN>, _1>>>,
      Copy_Atom<DefaultCopy, float>,
      decltype(make_tiled_copy(Copy_Atom<DefaultCopy, float>{}, Layout<Shape<_8, _16>, Stride<_16, _1>>{}, Layout<Shape<_1, _8>>{})),
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>>;
};

template <int BM, int BN, int BK, bool SmemEpi = false, int Stages = 0>  // Stages = 0: auto carveout
struct GemmTma {
  using Element = cutlass::bfloat16_t;
  using TileShape = Shape<Int<BM>, Int<BN>, Int<BK>>;
  using ClusterShape = Shape<_1, _1, _1>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Element, cutlass::layout::RowMajor, 8,
      Element, cutlass::layout::ColumnMajor, 8,
      float, TileShape, ClusterShape,
      cute::conditional_t<Stages == 0, cutlass::gemm::collective::StageCountAutoCarveout<0>, cutlass::gemm::collective::StageCount<Stages>>,
      cutlass::gemm::KernelTma>::CollectiveOp;

  using NoSmemEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      Element, cutlass::layout::RowMajor, 8,
      Element, cutlass::layout::RowMajor, 8,
      cutlass::epilogue::NoSmemWarpSpecialized>::CollectiveOp;
  using CollectiveEpilogue = cute::conditional_t<SmemEpi, typename SmemEpilogue<BM, BN>::Collective, NoSmemEpilogue>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  static typename Gemm::Arguments make_args(const at::Tensor& X, const at::Tensor& W, const at::Tensor& Y) {
    int M = X.size(0), K = X.size(1), N = W.size(0);
    auto* x = reinterpret_cast<Element*>(X.data_ptr());
    auto* w = reinterpret_cast<Element*>(W.data_ptr());
    auto* y = reinterpret_cast<Element*>(Y.data_ptr());
    return {
        cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
        {x, cutlass::make_cute_packed_stride(typename GemmKernel::StrideA{}, {M, K, 1}),
         w, cutlass::make_cute_packed_stride(typename GemmKernel::StrideB{}, {N, K, 1})},
        {{1.f, 0.f}, y, cutlass::make_cute_packed_stride(typename GemmKernel::StrideC{}, {M, N, 1}),
                     y, cutlass::make_cute_packed_stride(typename GemmKernel::StrideD{}, {M, N, 1})}};
  }

  static void run(const at::Tensor& X, const at::Tensor& W, const at::Tensor& Y, cudaStream_t stream) {
    auto args = make_args(X, W, Y);
    Gemm gemm;
    TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "can_implement failed");
    auto ws = at::empty({(int64_t)Gemm::get_workspace_size(args)}, X.options().dtype(at::kByte));
    TORCH_CHECK(gemm.initialize(args, ws.data_ptr(), stream) == cutlass::Status::kSuccess, "initialize failed");
    TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess, "run failed");
  }

  // 1-D grid, one CTA per output tile; tile ids come from ctapp.src.
  static void run_ctapp(const at::Tensor& X, const at::Tensor& W, const at::Tensor& Y, CtappParams ctapp, cudaStream_t stream) {
    using Kernel = CtappGemmKernel<GemmKernel>;
    auto args = make_args(X, W, Y);
    TORCH_CHECK(GemmKernel::can_implement(args), "can_implement failed");
    typename Kernel::Params params{GemmKernel::to_underlying_arguments(args, nullptr), ctapp};
    int tiles_m = (X.size(0) + BM - 1) / BM;
    TORCH_CHECK(ctapp.tiles_n == (W.size(0) + BN - 1) / BN, "tiles_n mismatch");
    void const* kernel = (void const*)cutlass::device_kernel<Kernel>;
    int smem = GemmKernel::SharedStorageSize + 16;  // + tile id slot, see ctapp_prologue
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    cutlass::device_kernel<Kernel><<<dim3(tiles_m * ctapp.tiles_n), Kernel::get_block_shape(), smem, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  // Max resident CTAs per SM of the ctapp kernel.
  static int occupancy() {
    using Kernel = CtappGemmKernel<GemmKernel>;
    void const* kernel = (void const*)cutlass::device_kernel<Kernel>;
    int smem = GemmKernel::SharedStorageSize + 16, n = 0;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel, Kernel::MaxThreadsPerBlock, smem));
    return n;
  }

  static std::string info() {
    return std::string(SmemEpi ? "smem-epi " : "nosmem-epi ") + "tile " + std::to_string(BM) + "x" + std::to_string(BN) + "x" + std::to_string(BK) +
           " threads=" + std::to_string(GemmKernel::MaxThreadsPerBlock) +
           " smem=" + std::to_string(GemmKernel::SharedStorageSize) +
           " stages=" + std::to_string(CollectiveMainloop::DispatchPolicy::Stages);
  }
};

#define FOR_CONFIGS(F)                       \
  case 0: F(GemmTma<64, 128, 64>); break;    \
  case 1: F(GemmTma<128, 128, 64>); break;   \
  case 2: F(GemmTma<64, 256, 64>); break;    \
  case 3: F(GemmTma<128, 256, 64>); break;   \
  case 4: F(GemmTma<64, 256, 64, true>); break;   \
  case 5: F(GemmTma<128, 128, 64, true>); break;   \
  case 6: F(GemmTma<64, 256, 64, true, 3>); break;   \
  case 7: F(GemmTma<128, 128, 64, true, 3>); break;   \
  case 8: F(GemmTma<64, 256, 32, true, 5>); break;   \
  case 9: F(GemmTma<64, 256, 32, true, 0>); break;

void gemm_tma(at::Tensor X, at::Tensor W, at::Tensor Y, int64_t config_id) {
  c10::cuda::CUDAGuard guard(X.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#define RUN(...) __VA_ARGS__::run(X, W, Y, stream)
  switch (config_id) { FOR_CONFIGS(RUN) default: TORCH_CHECK(false, "bad config_id"); }
#undef RUN
}

void ctapp_gemm(at::Tensor X, at::Tensor W, at::Tensor Y, int64_t config_id,
                at::Tensor src_entries, at::Tensor src_head, at::Tensor src_tail,
                c10::optional<at::Tensor> dep_offsets, c10::optional<at::Tensor> dep_consumers, c10::optional<at::Tensor> scoreboard,
                c10::optional<at::Tensor> dst_entries, c10::optional<at::Tensor> dst_head, c10::optional<at::Tensor> dst_tail,
                int64_t tiles_n, int64_t skip_wait, int64_t fence) {
  c10::cuda::CUDAGuard guard(X.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto ip = [](const c10::optional<at::Tensor>& t) { return t ? t->data_ptr<int>() : nullptr; };
  CtappParams p{
      {src_entries.data_ptr<int>(), src_head.data_ptr<int>(), src_tail.data_ptr<int>(), (int)src_entries.numel()},
      ip(dep_offsets), ip(dep_consumers), ip(scoreboard),
      {ip(dst_entries), ip(dst_head), ip(dst_tail), dst_entries ? (int)dst_entries->numel() : 1},
      (int)tiles_n, (int)skip_wait, (int)fence};
#define RUN(...) __VA_ARGS__::run_ctapp(X, W, Y, p, stream)
  switch (config_id) { FOR_CONFIGS(RUN) default: TORCH_CHECK(false, "bad config_id"); }
#undef RUN
}

int64_t occupancy(int64_t config_id) {
#define OCC(...) return __VA_ARGS__::occupancy()
  switch (config_id) { FOR_CONFIGS(OCC) default: TORCH_CHECK(false, "bad config_id"); }
#undef OCC
}

std::string config_info(int64_t config_id) {
#define INFO(...) return __VA_ARGS__::info()
  switch (config_id) { FOR_CONFIGS(INFO) default: TORCH_CHECK(false, "bad config_id"); }
#undef INFO
}

void enable_peer_access(int64_t dev, int64_t peer) {
  c10::cuda::CUDAGuard guard(dev);
  cudaError_t e = cudaDeviceEnablePeerAccess(peer, 0);
  if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
  else TORCH_CHECK(e == cudaSuccess, cudaGetErrorString(e));
}

__global__ void copy_kernel(const int4* __restrict__ src, int4* __restrict__ dst, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    dst[i] = src[i];
}

// src is local to the current device, dst may be peer memory. Runs on src's device.
void p2p_store_bw_kernel(at::Tensor src, at::Tensor dst) {
  c10::cuda::CUDAGuard guard(src.device());
  size_t n = src.nbytes() / sizeof(int4);
  int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  copy_kernel<<<sms * 8, 512, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const int4*>(src.data_ptr()), reinterpret_cast<int4*>(dst.data_ptr()), n);
}

// ---- timing gate: int32 in pinned mapped host memory; streams block on it with cuStreamWaitValue32 ----
static volatile int* g_gate = nullptr;
static CUdeviceptr g_gate_dev = 0;

void gate_alloc() {
  if (g_gate) return;
  void* h; C10_CUDA_CHECK(cudaHostAlloc(&h, sizeof(int), cudaHostAllocPortable | cudaHostAllocMapped));
  void* d; C10_CUDA_CHECK(cudaHostGetDevicePointer(&d, h, 0));
  g_gate = (volatile int*)h; g_gate_dev = (CUdeviceptr)d; *g_gate = 0;
}
void gate_set(int64_t v) { *g_gate = (int)v; }
// Blocks the current stream of the current device until gate >= 1.
void gate_wait() {
  CUresult r = cuStreamWaitValue32((CUstream)at::cuda::getCurrentCUDAStream().stream(), g_gate_dev, 1, CU_STREAM_WAIT_VALUE_GEQ);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuStreamWaitValue32 failed: ", (int)r);
}
// dst on another device than src; issued on the current stream of src's device.
void peer_copy(at::Tensor dst, at::Tensor src) {
  c10::cuda::CUDAGuard guard(src.device());
  C10_CUDA_CHECK(cudaMemcpyPeerAsync(dst.data_ptr(), dst.get_device(), src.data_ptr(), src.get_device(), src.nbytes(), at::cuda::getCurrentCUDAStream()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_tma", &gemm_tma);
  m.def("ctapp_gemm", &ctapp_gemm);
  m.def("config_info", &config_info);
  m.def("occupancy", &occupancy);
  m.def("enable_peer_access", &enable_peer_access);
  m.def("p2p_store_bw_kernel", &p2p_store_bw_kernel);
  m.def("gate_alloc", &gate_alloc);
  m.def("gate_set", &gate_set);
  m.def("gate_wait", &gate_wait);
  m.def("peer_copy", &peer_copy);
}
