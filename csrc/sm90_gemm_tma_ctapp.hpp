// KernelTma with CTA-pipelining hooks. operator() is a copy of KernelTma's (sm90_gemm_tma.hpp); changes are marked CTAPP CHANGE.
#pragma once
#include "cutlass/gemm/kernel/sm90_gemm_tma.hpp"
#include "ctapp.cuh"

template <class BaseKernel>
struct CtappGemmKernel : BaseKernel {
  using BaseParams = typename BaseKernel::Params;
  using CollectiveMainloop = typename BaseKernel::CollectiveMainloop;
  using CollectiveEpilogue = typename BaseKernel::CollectiveEpilogue;
  using TileShape = typename BaseKernel::TileShape;
  using TiledMma = typename BaseKernel::TiledMma;
  using StrideA = typename BaseKernel::StrideA;
  using StrideB = typename BaseKernel::StrideB;
  using StrideC = typename BaseKernel::StrideC;
  using StrideD = typename BaseKernel::StrideD;
  struct Params { BaseParams base; CtappParams ctapp; };

  CUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf) {
    using namespace cute;
    using X = Underscore;

// Any Tensor Op MMA Atom in the WGMMA ISA is arch conditional to sm90a.
#if ! defined(__CUDA_ARCH_FEAT_SM90_ALL)
    CUTE_INVALID_CONTROL_PATH("ERROR : Arch conditional MMA instruction used without targeting sm90a compute capability. Aborting.\n");
#else

    // Preconditions
    static_assert(cute::rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    int thread_idx = int(threadIdx.x);
    int warp_idx   = cutlass::canonical_warp_idx_sync();
    int lane_predicate = cute::elect_one_sync();
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();

    // Issue Tma Descriptor Prefetch from a single thread
    if ((warp_idx == 0) && lane_predicate) {
      CollectiveMainloop::prefetch_tma_descriptors(params.base.mainloop);
    }

    // CTAPP CHANGE 1: claim a tile from the workqueue (blocks until its inputs are ready).
    int tile = ctapp_prologue(params.ctapp, reinterpret_cast<int*>(smem_buf + BaseKernel::SharedStorageSize));

    // Separate out problem shape for convenience
    // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(params.base.problem_shape, Int<1>{});
    auto M = get<0>(problem_shape_MNKL);
    auto N = get<1>(problem_shape_MNKL);
    auto K = get<2>(problem_shape_MNKL);
    auto L = get<3>(problem_shape_MNKL);

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = params.base.mainloop.tma_load_a.get_tma_tensor(make_shape(M,K,L));                            // (m,k,l)
    Tensor mB_nkl = params.base.mainloop.tma_load_b.get_tma_tensor(make_shape(N,K,L));                            // (n,k,l)

    // Get the appropriate blocks for this thread block -- potential for thread block locality
    auto blk_shape = TileShape{};                                                                // (BLK_M,BLK_N,BLK_K)
    auto blk_coord = make_coord(_,_,_);                                                   // (m,n,k) -- defer the slice

    // Make tiled views
    Tensor gA_mkl = local_tile(mA_mkl, blk_shape, blk_coord, Step<_1, X,_1>{});                  // (BLK_M,BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(mB_nkl, blk_shape, blk_coord, Step< X,_1,_1>{});                  // (BLK_N,BLK_K,n,k,l)

    // Compute m_coord, n_coord, and l_coord with their post-tiled shapes
    // CTAPP CHANGE 2: tile coordinates come from the queue, not blockIdx.
    int m_coord = tile / params.ctapp.tiles_n;
    int n_coord = tile % params.ctapp.tiles_n;
    int l_coord = 0;
    auto output_tile_coord = make_coord(m_coord, n_coord, _, l_coord);

    // Slice with m_coord and n_coord
    Tensor gA = gA_mkl(_,_,m_coord,_,l_coord);                                                       // (BLK_M,BLK_K,k)
    Tensor gB = gB_nkl(_,_,n_coord,_,l_coord);                                                       // (BLK_N,BLK_K,k)

    // Allocate the tiled_mma and the accumulators for the (M,N) blk_shape
    TiledMma tiled_mma;
    Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape));                   // (MMA,MMA_M,MMA_N)

    auto k_tile_iter  = cute::make_coord_iterator(shape<2>(gA));
    auto k_tile_count = size<2>(gA);

    // Ensure memory ops in this kernel are not done prior to completion of dependent grids.
    cutlass::arch::wait_on_dependent_grids();

    // Perform the collective scoped MMA
    CollectiveMainloop collective_mma;
    collective_mma(
      gA, params.base.mainloop.tma_load_a,
      gB, params.base.mainloop.tma_load_b,
      accumulators,
      k_tile_iter, k_tile_count,
      thread_idx,
      block_rank_in_cluster,
      smem_buf,
      params.base.mainloop
    );

    constexpr int BLK_M_RANK = cute::rank<0>(blk_shape);
    auto m_max_coord = unwrap(cute::transform(make_seq<BLK_M_RANK>{}, [&](auto i) {
        return  get<0,i>(problem_shape_MNKL) - get<0,i>(blk_shape) * get<0,i>(output_tile_coord);
      }));

    constexpr int BLK_N_RANK = cute::rank<1>(blk_shape);
    auto n_max_coord = unwrap(cute::transform(make_seq<BLK_N_RANK>{}, [&](auto i) {
        return  get<1,i>(problem_shape_MNKL) - get<1,i>(blk_shape) * get<1,i>(output_tile_coord);
      }));
    auto residue_mnk = make_tuple(m_max_coord, n_max_coord, Int<0>{});

    // Epilogue and write to gD
    CollectiveEpilogue epilogue{params.base.epilogue};
    epilogue(
      problem_shape_MNKL,
      blk_shape,
      output_tile_coord,
      accumulators,
      tiled_mma,
      residue_mnk,
      thread_idx,
      smem_buf
    );

    // CTAPP CHANGE 3: signal dependent consumer tiles.
    if (params.ctapp.dep_offsets) ctapp_epilogue(params.ctapp, tile);
#endif
  }
};
