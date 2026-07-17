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
    \brief Templates implementing loading of convolution tiles mapped to GEMM A (output gradient tile)
    matrix from memory.

    This iterator assumes TensorNHWC layout of tensors in Global Memory.

    The iterator is specialized for each of the three convolution operators: forward propagation (Fprop),
    backward data gradient (Dgrad), and backward weight gradient (Wgrad).
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/coord.h"
#include "cutlass/predicate_vector.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/tensor_view.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"

#include "cutlass/arch/memory_aiu.h"

#include "cutlass/conv/threadblock/aiu/conv_aiu_load.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Shape_,
  typename Element_,
  typename ThreadMap_,
  int AccessSize = ThreadMap_::kElementsPerAccess
>
class Conv2dWgradOutputGradientTileAccessIteratorOptimizedAiu {
public:

  //
  // Types
  //
  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::TensorNHWC;
  using ThreadMap = ThreadMap_;
  using AccessType = AlignedArray<Element, AccessSize>;
  using TensorRef = cutlass::TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;
  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  static IteratorAlgorithm const kIteratorAlgorithm = conv::IteratorAlgorithm::kOptimized;
  static StrideSupport const kStrideSupport = conv::StrideSupport::kStrided;
  static int const kConvDim = 2;
  using ConvProblemSize = typename conv::Conv2dProblemSize;

  static_assert(sizeof_bits<Element>::value >= 8,
    "WGRAD requires elements of size 8b or greater.");

  //
  // Parameters structure
  //

  using Params = Conv2dWgradOutputGradientIteratorOptimizedParams;

  static int const kAccessesPerVector = ThreadMap::kElementsPerAccess / AccessType::kElements;

  static_assert((Shape::kRow * sizeof_bits<Element>::value / 8) % 128 == 0,
    "block_m must be divided by 128B for wgrad iterA");

  static int const CubeW = Shape::kRow > 128 / sizeof(Element) ? 128 / sizeof(Element) : Shape::kRow;

  static int const AiuLoadIter = Shape::kRow / CubeW;

  // static int const CubeC = 128 * 8 / sizeof_bits<Element>::value;

  static int const CubeStride = CubeW * Shape::kColumn;

private:

  Conv2dWgradOutputGradientIteratorOptimizedParams const &params_;
  Conv2dProblemSize const &problem_size_;
  LongIndex iteration_contiguous_;
  LongIndex iteration_strided_;
  LongIndex iteration_vector_;
  uint8_t const *global_ptr_;

  uint32_t predicates_[kAccessesPerVector];
  int offset_npq_;
  int offset_k_;

  int warp_idx_;

public:

  CUTLASS_HOST_DEVICE
  Conv2dWgradOutputGradientTileAccessIteratorOptimizedAiu(
    Conv2dWgradOutputGradientIteratorOptimizedParams const &params,
    Conv2dProblemSize const &problem_size,
    Element const *ptr,
    int thread_idx,
    MatrixCoord const &threadblock_offset = MatrixCoord()
  ):
    params_(params),
    problem_size_(problem_size),
    global_ptr_(reinterpret_cast<const uint8_t*>(ptr)),
    predicates_{0},
    offset_k_(0),
    offset_npq_(0) {

    warp_idx_ = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

    layout::PitchLinearCoord thread_coord = ThreadMap::initial_offset(thread_idx);

    offset_k_ = threadblock_offset.row();
    offset_npq_ = threadblock_offset.column();

  }

  CUTLASS_HOST_DEVICE
  static Params getParams(Conv2dProblemSize const &problem_size, Layout const &layout) {
    return Params(problem_size,
                  layout,
                  sizeof_bits<Element>::value,
                  {Shape::kRow, Shape::kColumn},
                  ThreadMap::kThreads,
                  ThreadMap::kElementsPerAccess,
                  {ThreadMap::Iterations::kContiguous, ThreadMap::Iterations::kStrided},
                  {ThreadMap::Delta::kContiguous, ThreadMap::Delta::kStrided});
  }


  CUTLASS_HOST_DEVICE
  void advance() {
    // moves to the next GEMM-K offset (offset_npq_) in GEMM-A by a CTA-K tile
    offset_npq_ += Shape::kColumn * problem_size_.split_k_slices;

  }

  CUTLASS_HOST_DEVICE
  void aiu_load(Element *tsm_ptr) const {

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < AiuLoadIter; i++) {
      // cutlass::arch::tsm_load_cube_sim(
      //   global_ptr_, tsm_ptr + i * CubeStride,
      //   1, 1, 1, problem_size_.N * problem_size_.P * problem_size_.Q, problem_size_.K,
      //   problem_size_.K, 1,
      //   0, 0, 0, offset_npq_, offset_k_ + i * CubeC,
      //   Shape::kColumn, CubeC,
      //   0, 0, 0,
      //   1, 1, 1,
      //   1, 1, 1, 0, 0, 0,
      //   1, 1, 1
      // );

      if (warp_idx_ == 0) {

        // printf("%ld, %ld, %ld, 11111111111\n", params_.layout.stride()[0], params_.layout.stride()[1], params_.layout.stride()[2]);

        // Shape::kColumn is CubeH
        CONV_AIU_LOAD_2D_IMPL<Element, Shape::kColumn, CubeW>()(
          tsm_ptr + i * CubeStride, global_ptr_,
          problem_size_.K, problem_size_.N * problem_size_.P * problem_size_.Q,
          params_.layout.stride()[0] * sizeof(Element), params_.layout.stride()[0] * sizeof(Element),
          offset_k_ + i * CubeW, offset_npq_
        );
      }
    }

  }

public:

  /// Determines whether the Implicit GEMM can execute the given problem.
  CUTLASS_HOST_DEVICE
  static Status can_implement(Conv2dProblemSize const &problem_size) {

    // check alignment constraint on iterator's contiguous dimension
    if (problem_size.K % AccessSize) {
      return Status::kErrorInvalidProblem;
    }

    return Status::kSuccess;
  }

};
/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////


