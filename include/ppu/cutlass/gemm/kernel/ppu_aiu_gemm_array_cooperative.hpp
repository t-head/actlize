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
#include "ppu/cute/util.hpp"
#include "ppu/cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cute/tensor.hpp"

namespace cutlass::gemm::kernel {

///////////////////////////////////////////////////////////////////////////////
/*
  This kernel layer implementation can also be used for ptr array batch-gemm(by removing the StrideA check in enable_if)
  When using same tile, has ~10% perf drop, need to investigate in the future
*/
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
  cute::enable_if_t<cute::is_base_of_v<KernelAiuMultistageBatchArray, typename CollectiveMainloop_::DispatchPolicy::Schedule> &&
                    !cute::is_same_v<typename CollectiveMainloop_::UnderlyingStrideA, typename CollectiveMainloop_::StrideA>>>
{
public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;

  static_assert(rank(typename ProblemShape::UnderlyingProblemShape{}) == 3 or rank(typename ProblemShape::UnderlyingProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using UnderlyingStrideA = typename CollectiveMainloop::UnderlyingStrideA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using UnderlyingStrideB = typename CollectiveMainloop::UnderlyingStrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using UnderlyingStrideC = typename CollectiveEpilogue::UnderlyingStrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
  using UnderlyingStrideD = typename CollectiveEpilogue::UnderlyingStrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  static_assert(cute::is_void_v<TileScheduler_> or cute::is_same_v<TileScheduler_, PersistentScheduler>,
    "PPU0010 kernel does not support specializing the tile scheduler.");
  static constexpr bool IsGroupedGemmKernel = !cute::is_same_v<UnderlyingStrideA, StrideA>;

  using TileSchedulerTag = TileScheduler_;
  using TileScheduler = cute::conditional_t<IsGroupedGemmKernel,
    typename detail::TileSchedulerSelector<
      GroupScheduler, ArchTag,
      TileShape, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
      ProblemShape>::Scheduler,
    typename detail::TileSchedulerSelector<
      TileScheduler_, ArchTag,
      TileShape, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>
      >::Scheduler>;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;


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
    void* workspace;
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    CUTLASS_TRACE_HOST("to_underlying_arguments():");

    ProblemShape problem_shapes = args.problem_shape;

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
    workspace_offset += TileScheduler::template get_workspace_size<typename ProblemShape::UnderlyingProblemShape, ElementAccumulator>(
      args.scheduler, typename ProblemShape::UnderlyingProblemShape{}, args.hw_info, NumMmaWarpGroups);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    void* epilogue_workspace = workspace_ptr + workspace_offset;
    workspace_offset += CollectiveEpilogue::get_workspace_size(problem_shapes, args.epilogue);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    void* mainloop_workspace = nullptr;
    // void* mainloop_workspace = workspace_ptr + workspace_offset;
    // workspace_offset += CollectiveMainloop::get_workspace_size(problem_shapes, args.mainloop, args.hw_info.cu_count);
    // workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    // Precompute the sub tiles numbers in epilogue, pass into tile scheduler.  Therefore it will be used
    // in separate reduction scheme for streamk case, NumEpilogueSubTiles default value is 1, which means
    // subtile will not be used, therefore separate reduction will not be enabled.
    constexpr uint32_t NumEpilogueSubTiles = 1; // CollectiveEpilogue::get_store_pipe_increment(TileShape{});
    TileSchedulerParams scheduler;
    if constexpr (IsGroupedGemmKernel) {
      scheduler = TileScheduler::to_underlying_arguments(
      problem_shapes, TileShape{}, ClusterShape{}, hw_info, args.scheduler, scheduler_workspace, NumEpilogueSubTiles);
    }
    else {
      scheduler = TileScheduler::to_underlying_arguments(
      problem_shapes.get_host_problem_shape(), TileShape{}, ClusterShape{}, hw_info, args.scheduler, scheduler_workspace, NumEpilogueSubTiles);
    }

    return {
      args.mode,
      problem_shapes,
      CollectiveMainloop::to_underlying_arguments(problem_shapes, args.mainloop, mainloop_workspace),
      CollectiveEpilogue::to_underlying_arguments(problem_shapes, args.epilogue, epilogue_workspace),
      hw_info,
      scheduler,
      workspace
    };
  }

  static bool
  can_implement(Arguments const& args) {
    bool implementable = true;
    if constexpr (IsGroupedGemmKernel) {
      // Group GEMM currently only supports rank-3 problem shapes
      implementable &= (args.mode == GemmUniversalMode::kGrouped && rank(typename ProblemShape::UnderlyingProblemShape{}) == 3);
    } else {
      implementable &= (args.mode == GemmUniversalMode::kArray && rank(typename ProblemShape::UnderlyingProblemShape{}) == 4);
    }
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Arguments or Problem Shape don't meet the requirements for Ptr Array Gemm or Grouped Gemm.\n");
      return implementable;
    }

    implementable &= TileScheduler::can_implement(args.scheduler);
    return implementable;
  }

  static int
  get_workspace_size(Arguments const& args) {
    return 0;
  }

  static
  cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, hggcStream_t stream = nullptr) {
    return Status::kSuccess;
  }

  static dim3
  get_grid_shape(Params const& params) {
    // Given device CU count, set grid size s.t. we do not launch more thread blocks than we can run concurrently
    TileSchedulerArguments args{};
    if constexpr (!std::is_const_v<decltype(args.max_swizzle_size)>) {
      args.max_swizzle_size = 1 << params.scheduler.log_swizzle_size_;
    }
    args.raster_order = params.scheduler.raster_order_ == TileScheduler::RasterOrder::AlongN ? TileScheduler::RasterOrderOptions::AlongN : TileScheduler::RasterOrderOptions::AlongM;
    dim3 grid_shape;
    if constexpr (IsGroupedGemmKernel) {
      grid_shape = TileScheduler::get_grid_shape(params.problem_shape, TileShape{}, ClusterShape{}, params.hw_info, args);
    }
    else {
      grid_shape = TileScheduler::get_grid_shape(params.problem_shape.get_host_problem_shape(), TileShape{}, ClusterShape{}, params.hw_info, args);
    }
    return grid_shape;
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
    auto problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(work_tile_info.L_idx), Int<1>{});
    auto M = get<0>(problem_shape_MNKL);
    auto N = get<1>(problem_shape_MNKL);
    auto K = get<2>(problem_shape_MNKL);
    auto L = get<3>(problem_shape_MNKL);

    // Preconditions
    static_assert(rank(UnderlyingStrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(rank(UnderlyingStrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(rank(UnderlyingStrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(rank(UnderlyingStrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    // Get the appropriate blocks for this thread block -- potential for thread block locality
    int thread_idx = int(threadIdx.x);
    auto blk_shape = TileShape{};                                                                // (BLK_M,BLK_N,BLK_K)
    // auto [m_coord, n_coord, l_coord] = static_cast<uint3>(blockIdx);
    // auto blk_coord_mnkl = make_coord(m_coord, n_coord, _, l_coord);                                        // (m,n,k)

    while (work_tile_info.is_valid()) {

      if (!TileScheduler::valid_warpgroup_in_work_tile(work_tile_info)) {
        work_tile_info = fetch_next_work(work_tile_info, scheduler);
        continue;
      }

      if constexpr (IsGroupedGemmKernel) {
        problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(work_tile_info.L_idx), Int<1>{});
        M = get<0>(problem_shape_MNKL);
        N = get<1>(problem_shape_MNKL);
        K = get<2>(problem_shape_MNKL);
        L = get<3>(problem_shape_MNKL);
      }
      auto m_coord = work_tile_info.M_idx; // Get M slice
      auto n_coord = work_tile_info.N_idx; // Get N slice
      auto l_coord = work_tile_info.L_idx; // Get batch slice

      auto blk_coord_mnkl = make_coord(m_coord, n_coord, _, l_coord);
      // Represent the full tensors
      auto stride_a = cute::get_ab_stride<IsGroupedGemmKernel>()(params.mainloop.dA, l_coord);
      auto stride_b = cute::get_ab_stride<IsGroupedGemmKernel>()(params.mainloop.dB, l_coord);
      Tensor mA_mkl = make_tensor(make_gmem_ptr(params.mainloop.ptr_A[l_coord]), make_shape(M,K), stride_a); //(m,k,l)
      Tensor mB_nkl = make_tensor(make_gmem_ptr(params.mainloop.ptr_B[l_coord]), make_shape(N,K), stride_b); //(n,k,l)


      Tensor mA_mk = make_mix_tensor_like(mA_mkl);                                                  // (m,k)
      Tensor mB_nk = make_mix_tensor_like(mB_nkl);                                                  // (n,k)

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

      auto k_tile_iter  = cute::make_coord_iterator(shape<2>(gA));
      int  k_tile_count = size<2>(gA);

      // Perform the collective scoped MMA
      CollectiveMainloop collective_mma(params.mainloop, take<0, 3>(problem_shape_MNKL), work_tile_info.L_idx);
      collective_mma(
        accumulators,
        gA,
        gB,
        accumulators,
        k_tile_iter, k_tile_count,
        residue_mnk,
        thread_idx,
        smem_buf
      );

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

      // Get next work tile
      work_tile_info = fetch_next_work(work_tile_info, scheduler);
      if constexpr (IsGroupedGemmKernel) {
        problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(work_tile_info.L_idx), Int<1>{});
      }
    } // Scheduler work fetch loop
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel
