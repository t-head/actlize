/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved. 
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Template for a multistage threadblock-scoped Implicit GEMM Convolution kernel.
*/

#pragma once

#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/cache_operation.h"
#include "cutlass/gemm/threadblock/mma_base.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting alu cores and SIMT math
/// instructions.
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Iterates over tiles of A operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorA_,
    /// Iterates over tiles of A operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorA_,
    /// Cache operation for operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Iterates over tiles of B operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorB_,
    /// Iterates over tiles of B operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorB_,
    /// Cache operation for operand B
    cutlass::arch::CacheOperation::Kind CacheOpB,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// Number of stages,
    int Stages,
    typename WarpTileIteratorA,
    typename WarpTileIteratorB,
    /// loading matrix A&B from tsm to registers in double buffer way
    bool DoubleBufRegLd = true,
    /// Allow global load A/B use differnce warp size.
    bool SkipWarps = false
>
class ImplicitGemmMultistageAiu :
  public gemm::threadblock::MmaBase<Shape_, Policy_, Stages> {
public:
  ///< Base class
  using Base = gemm::threadblock::MmaBase<Shape_, Policy_, Stages>;
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;
  ///< Iterates over tiles of A operand in global memory
  using IteratorA = IteratorA_;
  ///< Iterates over tiles of B operand in global memory
  using IteratorB = IteratorB_;
  ///< Policy describing tuning details
  using Policy = Policy_;

  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;

  static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;

  //
  // Dependent types
  //

  /// Fragment of accumulator tile

  using ElementC = typename Policy::Operator::ElementC;
  using FragmentC = typename Policy::Operator::FragmentC;

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  /// Internal structure exposed for introspection.
  struct Detail {

    static_assert(Base::kWarpGemmIterations > 1,
                  "The pipelined structure requires at least two warp-level "
                  "GEMM operations.");

    /// Number of cp.async instructions to load one stage of operand A
    static int const AsyncCopyIterationsPerStageA =
        IteratorA::ThreadMap::Iterations::kCount;

    /// Number of cp.async instructions to load one stage of operand B
    static int const AsyncCopyIterationsPerStageB =
        IteratorB::ThreadMap::Iterations::kCount;

    /// Number of stages
    static int const kStages = Stages;

    /// Number of cp.async instructions to load on group of operand A
    static int const kAccessesPerGroupA =
        (AsyncCopyIterationsPerStageA + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;

    /// Number of cp.async instructions to load on group of operand B
    static int const kAccessesPerGroupB =
        (AsyncCopyIterationsPerStageB + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;
  };

 private:

  using WarpLoadedFragmentA = typename Operator::FragmentA;
  using WarpLoadedFragmentB = typename Operator::FragmentB;
  using WarpTransformedFragmentA = typename Operator::TransformedFragmentA;
  using WarpTransformedFragmentB = typename Operator::TransformedFragmentB;
  static int const tsm_a_size_ = Base::SharedStorage::ShapeA::kCount / Stages;
  static int const tsm_b_size_ = Base::SharedStorage::ShapeB::kCount / Stages;

 private:

  //
  // Data members
  //

  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA smem_iterator_A_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB smem_iterator_B_;

  typename IteratorA::Element *tsm_w_ptr_a_;
  typename IteratorA::Element *tsm_r_ptr_a_;
  typename IteratorA::Element *tsm_w_ptr_b_;
  typename IteratorA::Element *tsm_r_ptr_b_;

  WarpTileIteratorA warp_tile_iterator_A_;
  WarpTileIteratorB warp_tile_iterator_B_;

public:

  /// Construct from tensor references
  CUTLASS_DEVICE
  ImplicitGemmMultistageAiu(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      typename Base::SharedStorage &shared_storage,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx
    ):
      Base(shared_storage, thread_idx, warp_idx, lane_idx),
      smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
      smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx),
      warp_tile_iterator_A_(),
      warp_tile_iterator_B_()
  {
    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension
    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

    // Add per-warp offsets in units of warp-level tiles
    this->warp_tile_iterator_A_.add_tile_offset(warp_idx_m);
    this->warp_tile_iterator_B_.add_tile_offset(warp_idx_n);

    tsm_w_ptr_a_ = shared_storage.operand_A_ref().data();
    tsm_r_ptr_a_ = shared_storage.operand_A_ref().data();
    tsm_w_ptr_b_ = shared_storage.operand_B_ref().data();
    tsm_r_ptr_b_ = shared_storage.operand_B_ref().data();
  }

  /// Perform a threadblock-scoped matrix multiply-accumulate
  CUTLASS_DEVICE
  void operator()(
      ///< problem size of GEMM
      int gemm_k_iterations,
      ///< destination accumulator tile
      FragmentC &accum,
      ///< iterator over A operand in global memory
      IteratorA iterator_A,
      ///< iterator over B operand in global memory
      IteratorB iterator_B,
      ///< initial value of accumulator
      FragmentC const &src_accum,
      ///< Imaginary strides used for planar-complex only - ignored here
      int64_t imag_stride_A = 0,
      int64_t imag_stride_B = 0) {

    //
    // Prologue
    //

    // Issue several complete stages
    CUTLASS_PRAGMA_UNROLL
    for (int stage = 0; stage < Base::kStages - 1;
         ++stage, --gemm_k_iterations) {
      iterator_A.aiu_load(tsm_w_ptr_a_);


      // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
      //   for (int cube = 0; cube < 1; cube++) {
      //     printf("\ncube\n\n");
      //     for (int row = 0; row < 32; row++) {
      //       printf("\nrow: %d\n", row);
      //       for (int col = 0; col < 32; col++) {
      //         printf("%.4f ", tsm_w_ptr_a_[cube * 32 * 32 + row * 32 + col]);
      //       }
      //     }
      //   }
      // }

      iterator_A.advance();
      // pointer to next stage's tsm
      tsm_w_ptr_a_ += tsm_a_size_;



      iterator_B.aiu_load(tsm_w_ptr_b_);

      // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
      //   printf("tsm_b_size_: %d\n", tsm_b_size_);
      //   for (int cube = 0; cube < 1; cube++) {
      //     printf("\ncube\n\n");
      //     for (int row = 0; row < 32; row++) {
      //       printf("\nrow: %d\n", row);
      //       for (int col = 0; col < 64; col++) {
      //         printf("%2d ", (int)(tsm_w_ptr_b_[cube * 64 * 64 + row * 64 + col]));
      //       }
      //     }
      //   }
      // }

      iterator_B.advance();
      // pointer to next stage's tsm
      tsm_w_ptr_b_ += tsm_b_size_;

      // Inserts a fence to group cp.async instructions into stages.
      cutlass::arch::cp_async_fence();
    }

    // Perform accumulation in the 'd' output operand
    accum = src_accum;

    // Waits until kStages-2 stages have committed.
    cutlass::arch::cp_async_wait<Base::kStages - 2>();
    __syncthreads();

    // Pair of fragments used to overlap shared memory loads and math
    // instructions
    WarpLoadedFragmentA warp_loaded_frag_A[2];
    WarpLoadedFragmentB warp_loaded_frag_B[2];
    WarpTransformedFragmentA warp_transformed_frag_A[2];
    WarpTransformedFragmentB warp_transformed_frag_B[2];

    Operator warp_mma;


    this->warp_tile_iterator_A_.load(warp_loaded_frag_A[0], tsm_r_ptr_a_);
    this->warp_tile_iterator_B_.load(warp_loaded_frag_B[0], tsm_r_ptr_b_);

    // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x < 4) {
    //   float *tmp = reinterpret_cast<float*>(warp_loaded_frag_B);
    //   printf("thread: %d, val: %f, %f, %f, %f\n", threadIdx.x, tmp[0], tmp[1], tmp[2], tmp[3]);
    // }

    // ++this->warp_tile_iterator_A_;
    // ++this->warp_tile_iterator_B_;

    this->warp_tile_iterator_A_.plus();
    this->warp_tile_iterator_B_.plus();

    // start load the third stage
    iterator_A.aiu_load(tsm_w_ptr_a_);
    iterator_B.aiu_load(tsm_w_ptr_b_);

    int smem_write_stage_idx = Base::kStages - 1;
    int smem_read_stage_idx = 0;

    warp_mma.transform(warp_transformed_frag_A[0], warp_transformed_frag_B[0],
                       warp_loaded_frag_A[0], warp_loaded_frag_B[0]);

    //
    // Mainloop
    //

    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations > (-Base::kStages + 1);) {
      //
      // Loop over GEMM K dimension
      //

      // Computes a warp-level GEMM on data held in shared memory
      // Each "warp_mma_k" refers to a warp-level matrix multiply-accumulate
      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations;
           ++warp_mma_k) {

        // Load warp-level tiles from shared memory, wrapping to k offset if
        // this is the last group as the case may be.

        // this->warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
        // this->warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);

        this->warp_tile_iterator_A_.load(warp_loaded_frag_A[(warp_mma_k + 1) % 2], tsm_r_ptr_a_);
        this->warp_tile_iterator_B_.load(warp_loaded_frag_B[(warp_mma_k + 1) % 2], tsm_r_ptr_b_);

        // ++this->warp_tile_iterator_A_;
        // ++this->warp_tile_iterator_B_;

        this->warp_tile_iterator_A_.plus();
        this->warp_tile_iterator_B_.plus();

        if (warp_mma_k > 0)
          warp_mma.transform(warp_transformed_frag_A[warp_mma_k % 2],
                             warp_transformed_frag_B[warp_mma_k % 2],
                             warp_loaded_frag_A[warp_mma_k % 2],
                             warp_loaded_frag_B[warp_mma_k % 2]);

        // Issue global->shared copies for the next stage
        int group_start_iteration_A, group_start_iteration_B;

        if (warp_mma_k + 1 == Base::kWarpGemmIterations) {
          group_start_iteration_A = 0;
          group_start_iteration_B = 0;
        } else {
          group_start_iteration_A =
              (warp_mma_k + 1) * Detail::kAccessesPerGroupA;
          group_start_iteration_B =
              (warp_mma_k + 1) * Detail::kAccessesPerGroupB;
        }

        // }


        // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x >= 0 && threadIdx.x <= 3) {
        //   float *frag_b = reinterpret_cast<float*>(&warp_transformed_frag_B[warp_mma_k % 2]);
        //   printf("thread: %d, val: %f, %f\n", threadIdx.x, frag_b[2], frag_b[3]);
        // }


        warp_mma(
                accum,
                 warp_transformed_frag_A[warp_mma_k % 2],
                 warp_transformed_frag_B[warp_mma_k % 2],
                 accum
                );

        if (warp_mma_k + 1 == Base::kWarpGemmIterations)
          warp_mma.transform(warp_transformed_frag_A[(warp_mma_k + 1) % 2],
                             warp_transformed_frag_B[(warp_mma_k + 1) % 2],
                             warp_loaded_frag_A[(warp_mma_k + 1) % 2],
                             warp_loaded_frag_B[(warp_mma_k + 1) % 2]);

        if (warp_mma_k + 2 == Base::kWarpGemmIterations) {
          // Inserts a fence to group cp.async instructions into stages.
          cutlass::arch::cp_async_fence();

          // Waits until kStages-2 stages of cp.async have committed
          arch::cp_async_wait<Base::kStages - 2>();
          __syncthreads();

          // Move to the next stage
          iterator_A.advance();
          tsm_w_ptr_a_ += tsm_a_size_;
          iterator_B.advance();
          tsm_w_ptr_b_ += tsm_b_size_;

          // add tsm read ptr to next stage
          tsm_r_ptr_a_ += tsm_a_size_;
          tsm_r_ptr_b_ += tsm_b_size_;

          // Add negative offsets to return iterators to the 'start' of the
          // circular buffer in shared memory
          if (smem_write_stage_idx == (Base::kStages - 1)) {
            tsm_w_ptr_a_ -= tsm_a_size_ * Base::kStages;
            tsm_w_ptr_b_ -= tsm_b_size_ * Base::kStages;
            smem_write_stage_idx = 0;
          } else {
            ++smem_write_stage_idx;
          }


          iterator_A.aiu_load(tsm_w_ptr_a_);
          // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
          //   for (int cube = 0; cube < 1; cube++) {
          //     printf("\ncube\n\n");
          //     for (int row = 0; row < 1; row++) {
          //       printf("\nrow: %d\n", row);
          //       for (int col = 0; col < 32; col++) {
          //         printf("%.4f ", tsm_w_ptr_a_[cube * 32 * 32 + row * 32 + col]);
          //       }
          //     }
          //   }
          //   printf("\n");
          // }

          iterator_B.aiu_load(tsm_w_ptr_b_);
          // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
          //   for (int cube = 0; cube < 1; cube++) {
          //     printf("\ncube\n\n");
          //     for (int row = 0; row < 1; row++) {
          //       printf("\nrow: %d\n", row);
          //       for (int col = 0; col < 32; col++) {
          //         printf("%.4f ", tsm_w_ptr_b_[cube * 32 * 32 + row * 32 + col]);
          //       }
          //     }
          //   }
            // printf("\n");
          // }

          if (smem_read_stage_idx == (Base::kStages - 1)) {
            // reset tsm read ptr when loop on multiple stages
            tsm_r_ptr_a_ -= tsm_a_size_ * Base::kStages;
            tsm_r_ptr_b_ -= tsm_b_size_ * Base::kStages;
            smem_read_stage_idx = 0;
          } else {
            ++smem_read_stage_idx;
          }

          --gemm_k_iterations;
        }
      }

    }

    // Insert fence and wait for all outstanding cp.async operations to commit.
    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
