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
    \brief Templates implementing loading of convolution tiles mapped to GEMM B (filter tile) 
    matrix from memory.

    This iterator assumes TensorNHWC or TensorCxRSKx<Interleave> layout of tensors in Global Memory.
    
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

#include "cutlass/conv/threadblock/conv2d_params.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Shape_,
  typename Element_,
  typename Layout_,
  typename ThreadMap_,
  int FoldNum,
  int AccessSize = ThreadMap_::kElementsPerAccess
>
class Conv2dFpropFilterTileAccessIteratorOptimizedSmallChannelAiu{
public:
  
  //
  // Types
  //

  using Shape = Shape_;
  using Element = Element_;
  using Layout = Layout_;
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

  static int const kAccessesPerVector = ThreadMap::kElementsPerAccess / AccessType::kElements;

  static_assert(ThreadMap::Iterations::kContiguous == 1,
    "fprop first layer's filter iter's iter contiguous must be 1");

  static_assert(kAccessesPerVector == 1,
    "fprop first layer's filter iter's iter vector must be 1");

  static_assert(ThreadMap::Iterations::kStrided < 32,
    "fprop first layer's filter iter's iter stride must < 32");


  //
  // Parameters structure
  //

  struct Params : Conv2dFpropFilterIteratorOptimizedParams<Layout> {

    CUTLASS_HOST_DEVICE
    Params() { }
    
    CUTLASS_HOST_DEVICE
    Params(Conv2dFpropFilterIteratorOptimizedParams<Layout> const &base): 
      Conv2dFpropFilterIteratorOptimizedParams<Layout>(base) { }

    CUTLASS_HOST_DEVICE
    Params(
      Conv2dProblemSize const &problem_size,
      Layout const &layout
    ):
      Conv2dFpropFilterIteratorOptimizedParams<Layout>(
        problem_size,
        layout,
        sizeof_bits<Element>::value,
        {Shape::kRow, Shape::kColumn},
        ThreadMap::kThreads,
        ThreadMap::kElementsPerAccess,
        {ThreadMap::Iterations::kContiguous, ThreadMap::Iterations::kStrided},
        {ThreadMap::Delta::kContiguous, ThreadMap::Delta::kStrided}
      ) {

    }
  };

private:

  Conv2dFpropFilterIteratorOptimizedParams<Layout> const &params_;
  Conv2dProblemSize const &problem_size_;

  Element const *global_ptr_;
  int offset_k_;
  int offset_rsc_;

  int cur_r_;
  int cur_s_;

  //
  // Assertions
  //

public:

  CUTLASS_HOST_DEVICE
  Conv2dFpropFilterTileAccessIteratorOptimizedSmallChannelAiu(
    Conv2dFpropFilterIteratorOptimizedParams<Layout> const &params,
    Conv2dProblemSize const &problem_size,
    Element const *ptr,
    int thread_idx,
    MatrixCoord const &threadblock_offset = MatrixCoord()
  ):
    params_(params), 
    problem_size_(problem_size),
    global_ptr_(ptr),
    offset_k_(0),
    offset_rsc_(0),
    cur_r_(0),
    cur_s_(0) {

    layout::PitchLinearCoord thread_coord = ThreadMap::initial_offset(thread_idx);

    offset_k_ = threadblock_offset.column() + thread_coord.strided();

  }


  CUTLASS_HOST_DEVICE
  void advance() {
    
    cur_s_ += FoldNum;
    if (cur_s_ >= problem_size_.S) {
      cur_s_ = 0;
      ++cur_r_;
      if (cur_r_ >= problem_size_.R) {
        cur_r_ =0;
        // cur_c always 0
      }
    }

    offset_rsc_ = (cur_r_ * problem_size_.S + cur_s_) * problem_size_.C;

  }

  /// Returns a pointer to the vector starting at the current coordinate
  CUTLASS_HOST_DEVICE
  void aiu_load(Element *tsm_ptr) const {

    // dim_c is the end c position of current rs, to avoid load next rs's c
    int tensor_dim_c = (cur_r_ * problem_size_.S + min(problem_size_.S, cur_s_ + FoldNum)) * problem_size_.C;

    // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
    //   printf("cur_r: %d, cur_s: %d\n", cur_r_, cur_s_);
    //   printf("dim_k: %d, dim_c: %d, stride: %d, off_k: %d, off_rsc: %d\n", problem_size_.K, tensor_dim_c, problem_size_.R * problem_size_.S * problem_size_.C, offset_k_, offset_rsc_);
    // }


    cutlass::arch::tsm_load_cube_sim(
      global_ptr_, tsm_ptr,
      1, 1, 1, problem_size_.K, tensor_dim_c,
       // stride is rsc
      problem_size_.R * problem_size_.S * problem_size_.C, 1,
      0, 0, 0, offset_k_, offset_rsc_,
      Shape::kColumn, Shape::kRow,
      0, 0, 0,
      1, 1, 1,
      1, 1, 1, 0, 0, 0,
      1, 1, 1
    );
  }

  /// Determines whether the Implicit GEMM can execute the given problem.
  CUTLASS_HOST_DEVICE
  static Status can_implement(Conv2dProblemSize const &problem_size) {


    return Status::kSuccess;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
