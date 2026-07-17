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

/*! \file
  \brief Functor performing elementwise operations used by epilogues.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"

#include "cutlass/epilogue/fusion/ppu_callbacks_tma_warpspecialized.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies an element wise operation to all elements within the fragment
/// and writes it out to destination storage.
///
/// Ways to generalize this:
/// - CTA tile shape
/// - vectorization requirements (GMEM)
/// - vectoriz(able) transform()
///
template <
  class StrideC_,
  class StrideD_,
  class FusionCallbacks_,
  class SmemLayout_,
  class CopyAtomR2S_,
  class TiledCopyS2R_,
  class CopyAtomR2G_
>
class EpilogueEvt {
public:
  //
  // Type Aliases
  //
  // derived types of output thread level operator

  using SmemLayout   = SmemLayout_;
  using CopyAtomR2S  = CopyAtomR2S_;
  using TiledCopyS2R = TiledCopyS2R_;
  using CopyAtomR2G  = CopyAtomR2G_;

  using ElementAccumulator =  typename CopyAtomR2S::ValType;
  using ElementCompute = ElementAccumulator;
  using ElementScalar = ElementCompute;
  using ElementOutput = typename CopyAtomR2G::ValType;;
  using ElementC = ElementOutput;
  using StrideC = StrideC_;
  using ElementD = ElementOutput;
  using StrideD = StrideD_;
  using FusionCallbacks = FusionCallbacks_;
  using FusionStorage = typename FusionCallbacks::SharedStorage;

  using ThreadEpilogueOp = typename epilogue::fusion::FusionCallbacksTraits<FusionCallbacks>::Operation;

  static_assert(rank(StrideC{}) == 3, "StrideCD must be rank-3: [M, N, L]");
  static_assert(rank(StrideD{}) == 3, "StrideCD must be rank-3: [M, N, L]");

  struct SharedStorage
  {
    cute::array_aligned<ElementAccumulator, cute::cosize_v<SmemLayout>> smem_epilogue;
  };

  // Host side epilogue arguments
  struct Arguments {
    ElementC const* ptr_C = nullptr;
    StrideC dC{};
    ElementD* ptr_D = nullptr;
    StrideD dD{};
  };

  // Device side epilogue params
  // different to Arguments, construct callback_args in kernel
  // gemm_configs will always use EpilogueEvt
  // both aiu/cutlass gemm will force use rtc since evt params is hard to construct in static
  struct Params {
    ElementC const* ptr_C = nullptr;
    StrideC dC{};
    ElementD* ptr_D = nullptr;
    StrideD dD{};
    typename FusionCallbacks::Params callback_args{};
  };

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      [[maybe_unused]] ProblemShape const& _,
      Arguments const& args,
      [[maybe_unused]] void* workspace) {
    Params params;
    params.ptr_C = args.ptr_C;
    params.dC = args.dC;
    params.ptr_D = args.ptr_D;
    params.dD = args.dD;
    return params;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, hggcStream_t stream) {
    return cutlass::Status::kSuccess;
  }

  template <class ProblemShape>
  CUTLASS_HOST_DEVICE static bool
  can_implement(
      [[maybe_unused]] ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  EpilogueEvt(Params const& params_)
      : params(params_), fusion_callbacks(params_.callback_args, FusionStorage{}) {}

  CUTLASS_DEVICE
  bool
  is_source_needed() {
    return false;
  }

  template<
    class ProblemShapeMNKL,
    class BlockShapeMNK,
    class BlockCoordMNKL,
    class FrgEngine, class FrgLayout,
    class TiledMma,
    class ResidueMNK
  >
  CUTLASS_DEVICE void
  operator()(
      ProblemShapeMNKL problem_shape_mnkl,
      BlockShapeMNK blk_shape_MNK,
      BlockCoordMNKL blk_coord_mnkl,
      cute::Tensor<FrgEngine,FrgLayout> const& accumulators,                   // (MMA,MMA_M,MMA_N)
      TiledMma tiled_mma,
      ResidueMNK residue_mnk,
      int thread_idx,
      char* smem_buf)
  {
    using namespace cute;
    using X = Underscore;

    static_assert(rank(ProblemShapeMNKL{}) == 4, "ProblemShapeMNKL must be rank 4");
    static_assert(is_static<BlockShapeMNK>::value, "ThreadBlock tile shape must be static");
    static_assert(rank(BlockShapeMNK{}) == 3, "BlockShapeMNK must be rank 3");
    static_assert(rank(BlockCoordMNKL{}) == 4, "BlockCoordMNKL must be rank 3");

    // synchronizing function for smem reads/writes
#if HGGC_BARRIER_ENABLED
    auto synchronize = [] () { cutlass::arch::NamedBarrier::sync(typename TiledCopyS2R::TiledNumThr{}, 0); };
#else
    auto synchronize = [] () { __syncthreads(); };
#endif

    // Separate out problem shape for convenience
    auto M = get<0>(problem_shape_mnkl);
    auto N = get<1>(problem_shape_mnkl);
    auto L = get<3>(problem_shape_mnkl);

    // Represent the full output tensor
    Tensor mC_mnl = make_tensor(make_gmem_ptr(params.ptr_C), make_shape(M,N,L), params.dC);      //             (m,n,l)
    Tensor mD_mnl = make_tensor(make_gmem_ptr(params.ptr_D), make_shape(M,N,L), params.dD);      //             (m,n,l)
    Tensor gC_mnl = local_tile(mC_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});      // (BLK_M,BLK_N,m,n,l)
    Tensor gD_mnl = local_tile(mD_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});      // (BLK_M,BLK_N,m,n,l)

    // Slice to get the tile this CTA is responsible for
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord_mnkl;
    Tensor gC = gC_mnl(_,_,m_coord,n_coord,l_coord);                                                   // (BLK_M,BLK_N)
    Tensor gD = gD_mnl(_,_,m_coord,n_coord,l_coord);                                                   // (BLK_M,BLK_N)

    // Construct a tensor in SMEM that we can partition for rearranging data
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    Tensor sC = make_tensor(make_smem_ptr(storage.smem_epilogue.data()), SmemLayout{});              // (SMEM_M,SMEM_N)

    // Partition sC to match the accumulator partitioning
    auto tiled_r2s = make_tiled_copy_C(CopyAtomR2S{}, tiled_mma);
    auto tC     = tiled_r2s.get_thread_slice(thread_idx);
    Tensor tCaC = tC.retile_S(accumulators);                                          // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor tCsC = tC.partition_D(sC);                                                 // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // Tile gD and gC by the shape of SmemLayout first
    auto tile  = make_shape(size<0>(sC), size<1>(sC));
    Tensor gCt = flat_divide(gC, tile);                                                // (SMEM_M,SMEM_N,TILE_M,TILE_N)
    Tensor gDt = flat_divide(gD, tile);                                                // (SMEM_M,SMEM_N,TILE_M,TILE_N)

    // Partition sC, gC, and gD for the output
    auto tiled_s2r = TiledCopyS2R{};
    auto tD     = tiled_s2r.get_thread_slice(thread_idx);
    Tensor tDsC = tD.partition_S(sC);                                   //               ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tDgC = tD.partition_D(gCt);                                  // ((Atom,AtomNum),ATOM_M,ATOM_N,TILE_M,TILE_N)
    Tensor tDgD = tD.partition_D(gDt);                                  // ((Atom,AtomNum),ATOM_M,ATOM_N,TILE_M,TILE_N)

    // Allocate intermediate registers on the dst tensors
    Tensor tDrC = make_tensor<ElementAccumulator>(take<0,3>(shape(tDgC)));            // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tDrD = make_tensor<ElementOutput>(shape(tDrC));                            // ((Atom,AtomNum),ATOM_M,ATOM_N)

    // Repeat the D-partitioning for coordinates and predication
    Tensor cD   = make_identity_tensor(make_shape(size<0>(gD),size<1>(gD)));          // (BLK_M,BLK_N) -> (blk_m,blk_n)
    Tensor cDt  = flat_divide(cD, tile);                                //                (SMEM_M,SMEM_N,TILE_M,TILE_N)
    Tensor tDcD = tD.partition_D(cDt);                                  // ((Atom,AtomNum),ATOM_M,ATOM_N,TILE_M,TILE_N)

    CUTE_STATIC_ASSERT(size<1>(tCaC) % size<3>(tDgC) == 0);  // TILE_M divides MMA_M
    CUTE_STATIC_ASSERT(size<2>(tCaC) % size<4>(tDgC) == 0);  // TILE_N divides MMA_N
    // TiledCopyS2R::TiledNumThr is block size, size<0>(typename TiledMma::AtomLayoutC_TV{}) is warp size, why should be equal?????
    // CUTE_STATIC_ASSERT(typename TiledCopyS2R::TiledNumThr{} == size<0>(typename TiledMma::AtomLayoutC_TV{}));

    constexpr int EvtFragSize = size(tDrC(_,0,0));
    // frag for one epilogue outer loop
    // Tensor evt_c_tensor = make_tensor<ElementOutput>(take<0,3>(shape(tDgC)));
    Tensor evt_c_tensor = tDgC(_,_,_,0,0);

    auto cst_args = cutlass::epilogue::fusion::detail::ConsumerStoreArgs{
                      problem_shape_mnkl,
                      blk_shape_MNK,
                      blk_coord_mnkl,
                      residue_mnk,
                      SmemLayout{},
                      tiled_s2r,
                      thread_idx,
                      cD,
                      evt_c_tensor,
                      evt_c_tensor
                    };
    auto cst_callbacks = fusion_callbacks.get_consumer_store_callbacks<true>(cst_args);

    // For each tiling needed for SmemLayout to cover shape(gD)
    CUTLASS_PRAGMA_UNROLL
    for (int step_m = 0; step_m < size<2>(cDt); ++step_m)
    {
      CUTLASS_PRAGMA_UNROLL
      for (int step_n = 0; step_n < size<3>(cDt); ++step_n)
      {
        // Step 1. Copy to SMEM
        CUTLASS_PRAGMA_UNROLL
        for (int pipe_m = 0; pipe_m < size<1>(tCsC); ++pipe_m) {
          CUTLASS_PRAGMA_UNROLL
          for (int pipe_n = 0; pipe_n < size<2>(tCsC); ++pipe_n) {
            int mma_m = step_m * size<1>(tCsC) + pipe_m;
            int mma_n = step_n * size<2>(tCsC) + pipe_n;

            copy(tiled_r2s, tCaC(_,mma_m,mma_n), tCsC(_,pipe_m,pipe_n));
          }
        }

        // Step 2. Wait for SMEM writes to complete
        synchronize();

        // Step 3. Copy from SMEM into a fragment
        copy(tiled_s2r, tDsC, tDrC);

        // Step 4. Wait for SMEM reads to complete
        synchronize();

        Tensor tDgDmn = tDgD(_,_,_,step_m,step_n);
        Tensor tDcDmn = tDcD(_,_,_,step_m,step_n);

        if (fusion_callbacks.is_C_load_needed()) {
          // CopyAtomR2G is Vector Copy which can also used for G2R
          // copy(CopyAtomR2G{}, tDgC(_,_,_,step_m,step_n), evt_c_tensor);

          // follow original epilogue logic, transfer part of global tensor to epilogue_op
          // read to vreg in each sub loop, avoid alloc vreg for the whole subloop
          evt_c_tensor = tDgC(_,_,_,step_m,step_n);
        }

        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < size<1>(tDgDmn); ++m)
        {
          CUTLASS_PRAGMA_UNROLL
          for (int n = 0; n < size<2>(tDgDmn); ++n)
          {
            int sub_loop_idx = m * size<2>(tDgDmn) + n;
            // Predication
            if (get<0>(tDcDmn(0,m,n)) < get<0>(residue_mnk) &&
                get<1>(tDcDmn(0,m,n)) < get<1>(residue_mnk))
            {
              // run evt
              Tensor tDrC_array = recast<Array<ElementAccumulator, EvtFragSize>>(tDrC(_,m,n));
              Tensor tDrD_array = recast<Array<ElementOutput, EvtFragSize>>(tDrD(_,m,n));
              // run output op for all regs in cur epi loop
              // corresponds to multiple vmem.st
              tDrD_array[0] = cst_callbacks.visit(tDrC_array[0], sub_loop_idx, step_m, step_n);
              copy(CopyAtomR2G{}, tDrD(_,m,n), tDgDmn(_,m,n));
            }
          }
        }

      }
    }
  }

private:
  Params params;
  FusionCallbacks fusion_callbacks;
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
