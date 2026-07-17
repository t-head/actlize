/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved. 
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/gemm/gemm.h"
#include "ppu/cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cute/tensor.hpp"

namespace cutlass::gemm::kernel {

///////////////////////////////////////////////////////////////////////////////

template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileScheduler_
>
class GemmUniversal<
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileScheduler_,
  cute::enable_if_t<cute::is_base_of_v<KernelAiuMultistageStreamK, typename CollectiveMainloop_::DispatchPolicy::Schedule>>>
{
public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;

  static_assert(rank(ProblemShape{}) == 3 or rank(ProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  // static_assert(cute::is_void_v<TileScheduler_> or cute::is_same_v<TileScheduler_, PersistentScheduler>,
  //   "PPU0010 kernel does not support specializing the tile scheduler.");
  using TileSchedulerTag = TileScheduler_;
  using TileScheduler = typename detail::TileSchedulerSelector<
    TileScheduler_, ArchTag, TileShape,
    cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::Scheduler;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;
  using ClusterShape = typename DispatchPolicy::ClusterShape;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  static_assert(cute::is_same_v<ElementAccumulator, typename CollectiveEpilogue::ElementAccumulator>,
    "Mainloop and epilogue do not agree on accumulator value type.");

  static constexpr int SharedStorageSize = static_cast<int>(cute::max(
      sizeof(typename CollectiveMainloop::SharedStorage),
      sizeof(typename CollectiveEpilogue::SharedStorage)));

  static constexpr uint32_t MaxThreadsPerBlock = cute::size(TiledMma{});
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  static constexpr uint32_t NumMmaWarpGroups = 1;

  // Device side arguments
  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
  };

  // Kernel entry point API
  struct Params {
    GemmUniversalMode mode;
    ProblemShape problem_shape;
    MainloopParams mainloop;
    EpilogueParams epilogue;
    KernelHardwareInfo hw_info;
    TileSchedulerParams scheduler;
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    CUTLASS_TRACE_HOST("to_underlying_arguments():");

    (void) workspace;
    auto problem_shape = args.problem_shape;
    if constexpr (detail::IF_SWAP_AB<CollectiveMainloop>::value) {
      // swap M/N
      get<0>(problem_shape) = get<1>(args.problem_shape);
      get<1>(problem_shape) = get<0>(args.problem_shape);
    }
    auto problem_shape_MNKL = append<4>(problem_shape, 1);

    // Get CU count if needed, otherwise use user supplied CU count
    int cu_count = args.hw_info.cu_count;
    if (cu_count <= 0) {
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid CU count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the CU count.");
      cu_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }

    CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid CU count to " << cu_count);
    KernelHardwareInfo hw_info{args.hw_info.device_id, cu_count};

    // Calculate workspace pointers
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    void* scheduler_workspace = workspace_ptr;
    workspace_offset += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape, args.hw_info, NumMmaWarpGroups);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    void* epilogue_workspace = workspace_ptr + workspace_offset;
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    void* mainloop_workspace = nullptr;

    // printf("args.scheduler.splits = %d\n", args.scheduler.splits);

    return {
      args.mode,
      problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, mainloop_workspace),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, epilogue_workspace),
      hw_info,
      TileScheduler::to_underlying_arguments(problem_shape_MNKL, TileShape{}, ClusterShape{}, hw_info, args.scheduler, scheduler_workspace)
    };
  }

  static bool
  can_implement(Arguments const& args) {
    return args.mode == GemmUniversalMode::kGemm or
          (args.mode == GemmUniversalMode::kBatched && rank(ProblemShape{}) == 4);
  }

  static size_t
  get_workspace_size(Arguments const& args) {
    size_t workspace_size = 0;
    workspace_size += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape, args.hw_info, NumMmaWarpGroups);
    workspace_size = round_nearest(workspace_size,  MinWorkspaceAlignment);

    workspace_size += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_size = round_nearest(workspace_size,  MinWorkspaceAlignment);

    // printf("workspace_size = %d\n", workspace_size);
    return workspace_size;
  }

  static cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, hggcStream_t stream = nullptr) {
    Status status = Status::kSuccess;
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    status = TileScheduler::template initialize_workspace<ProblemShape, ElementAccumulator>(
      args.scheduler, workspace_ptr + workspace_offset, stream, args.problem_shape, args.hw_info, NumMmaWarpGroups);
    workspace_offset += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape, args.hw_info, NumMmaWarpGroups);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    status = CollectiveEpilogue::initialize_workspace(args.problem_shape, args.epilogue, workspace_ptr + workspace_offset, stream);
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    return status;
  }

  // Computes the kernel launch grid shape based on runtime parameters
  static dim3
  get_grid_shape(Params const& params) {
    // Given device CU count, set grid size s.t. we do not launch more thread blocks than we can run concurrently
    TileSchedulerArguments args{};
    if constexpr (!std::is_const_v<decltype(args.max_swizzle_size)>) {
      args.max_swizzle_size = 1 << params.scheduler.log_swizzle_size_;
    }
    args.raster_order = params.scheduler.raster_order_ == TileScheduler::RasterOrder::AlongN ? TileScheduler::RasterOrderOptions::AlongN : TileScheduler::RasterOrderOptions::AlongM;
    return TileScheduler::get_grid_shape(params.problem_shape, TileShape{}, ClusterShape{}, params.hw_info, args);
  }

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  // Kernel helper function to get next work unit
  CUTLASS_DEVICE
  typename TileScheduler::WorkTileInfo
  fetch_next_work(
    typename TileScheduler::WorkTileInfo& work_tile_info,
    TileScheduler& scheduler) const {
    // Check whether we should continue on with the current work unit. If this is the case,
    // the work unit will have been updated in continue_current_work to reflect the new
    // tile to be computed.
    if (scheduler.continue_current_work(work_tile_info)) {
      return work_tile_info;
    }

    // Get next work tile
    scheduler.advance_to_next_work();
    return scheduler.get_current_work();
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf) {
    using namespace cute;
    using X = Underscore;

    TileScheduler scheduler{params.scheduler};
    auto work_tile_info = scheduler.get_current_work();

    // Preconditions
    CUTE_STATIC_ASSERT(is_static<TileShape>::value);

    // Separate out problem shape for convenience
    // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto M = get<0>(problem_shape_MNKL);
    auto N = get<1>(problem_shape_MNKL);
    auto K = get<2>(problem_shape_MNKL);
    auto L = get<3>(problem_shape_MNKL);

    // Preconditions
    static_assert(rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    // Get the appropriate blocks for this thread block -- potential for thread block locality
    int thread_idx = int(threadIdx.x);
    auto blk_shape = TileShape{};                                                                // (BLK_M,BLK_N,BLK_K)
    // auto [m_coord, n_coord, l_coord] = static_cast<uint3>(blockIdx);
    // auto blk_coord_mnkl = make_coord(m_coord, n_coord, _, l_coord);                                        // (m,n,k,l)

    // Represent the full tensors
    Tensor mA_mkl = make_tensor(make_gmem_ptr(params.mainloop.ptr_A), make_shape(M,K,L), params.mainloop.dA); //(m,k,l)
    Tensor mB_nkl = make_tensor(make_gmem_ptr(params.mainloop.ptr_B), make_shape(N,K,L), params.mainloop.dB); //(n,k,l)

    while (work_tile_info.is_valid()) {

      if (!TileScheduler::valid_warpgroup_in_work_tile(work_tile_info)) {
        work_tile_info = fetch_next_work(work_tile_info, scheduler);
        continue;
      }

      auto m_coord = work_tile_info.M_idx;
      auto n_coord = work_tile_info.N_idx;
      auto l_coord = work_tile_info.L_idx;

      auto blk_coord_mnkl = make_coord(m_coord, n_coord, _, l_coord);

      // Get current k slice
      Tensor mA_mk = make_mix_tensor_like(mA_mkl(_,_,l_coord));                                                  // (m,k)
      Tensor mB_nk = make_mix_tensor_like(mB_nkl(_,_,l_coord));                                                  // (n,k)
      // Slice to get the tiles this thread block is responsible for
      Tensor gA = local_tile(mA_mk, blk_shape, take<0,3>(blk_coord_mnkl), Step<_1, X,_1>{});           // (BLK_M,BLK_K,k)
      Tensor gB = local_tile(mB_nk, blk_shape, take<0,3>(blk_coord_mnkl), Step< X,_1,_1>{});           // (BLK_N,BLK_K,k)

      // Compute tile residues for predication
      auto m_max_coord = M - size<0>(gA) * get<0>(blk_coord_mnkl);                             // M - BLK_M * m_coord
      auto n_max_coord = N - size<0>(gB) * get<1>(blk_coord_mnkl);                             // N - BLK_N * n_coord
      auto k_residue   = K - size<1>(gA) * size<2>(gA);                                        // K - BLK_K * k_coord_max
      auto residue_mnk = make_tuple(m_max_coord, n_max_coord, k_residue);

      // Allocate the tiled_mma and the accumulators for the (M,N) blk_shape
      TiledMma tiled_mma;
      Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape)); // (MMA,MMA_M,MMA_N)
      clear(accumulators);

      // Get the number of K tiles to compute for this work as well as the starting K tile offset of the work.
      auto work_k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, blk_shape);
      auto work_k_tile_start = TileScheduler::get_work_k_tile_start(work_tile_info);
      auto k_tile_iter = cute::make_coord_iterator(idx2crd(work_k_tile_start, shape<2>(gA)), shape<2>(gA));

#if 0
      if (thread0()) {
        printf("work_tile_info(m,n,k,l) = (%d,%d,%d,%d), coord(m,n,l) = (%d,%d,%d)\n",
          work_tile_info.M_idx, work_tile_info.N_idx, work_tile_info.K_idx, work_tile_info.L_idx,
          m_coord, n_coord, l_coord);
        print("    gA:"); print(gA); print("\n");
        print("    gB:"); print(gB); print("\n");
        printf("    work_k_tile_(start,count) = (%d,%d)\n", work_k_tile_start, work_k_tile_count);
      }
#endif
      // Perform the collective scoped MMA
      CollectiveMainloop collective_mma(params.mainloop, params.problem_shape);
      collective_mma(
        accumulators,
        gA,
        gB,
        accumulators,
        k_tile_iter, work_k_tile_count,
        residue_mnk,
        thread_idx,
        smem_buf
      );

      // Perform reduction across splits, if needed
      TileScheduler::fixup(
        params.scheduler, work_tile_info, accumulators, NumMmaWarpGroups, 0);

      if (TileScheduler::compute_epilogue(work_tile_info, params.scheduler)) {
        // Epilogue and write to gD
        CollectiveEpilogue epilogue{params.epilogue};
        epilogue(
          problem_shape_MNKL,
          blk_shape,
          blk_coord_mnkl,
          accumulators,
          tiled_mma,
          residue_mnk,
          thread_idx,
          smem_buf
        );
      }

      // Get next work tile
      work_tile_info = fetch_next_work(work_tile_info, scheduler);
    }
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel
