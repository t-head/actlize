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
#include "cutlass/matrix_shape.h"
#include "cutlass/predicate_vector.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/tensor_view.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/threadblock/conv2d_params.h"

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
  conv::StrideSupport StrideSupport_ = conv::StrideSupport::kUnity,
  int AccessSize = ThreadMap_::kElementsPerAccess
>
class Conv2dDgradOutputGradientTileAccessIteratorOptimizedAiu {
public:

  static_assert(StrideSupport_ == conv::StrideSupport::kUnity,
    "Only unit-stride dgrad is supported at this time.");

  //
  // Types
  //

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::TensorNHWC;
  using TensorCoord = typename Layout::TensorCoord;
  using ThreadMap = ThreadMap_;
  using AccessType = AlignedArray<Element, AccessSize>;
  using TensorRef = cutlass::TensorRef<Element, Layout>;
  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  static IteratorAlgorithm const kIteratorAlgorithm = conv::IteratorAlgorithm::kOptimized;
  static StrideSupport const kStrideSupport = conv::StrideSupport::kUnity;
  static int const kConvDim = 2;
  using ConvProblemSize = typename conv::Conv2dProblemSize;

  static constexpr int SwzlMode = Shape::kColumn * sizeof(Element) == 128 ? 0 : 1;

  using Mask = uint64_t;

  //
  // Simplifying assertions
  //
  static_assert(ThreadMap::Iterations::kContiguous == 1,
    "Require Iterations::kContiguous == 1");

  //
  // Parameters structure
  //

  using Params = Conv2dDgradOutputGradientIteratorOptimizedParams;

  static int const kAccessesPerVector = ThreadMap::kElementsPerAccess / AccessType::kElements;

private:

  Conv2dDgradOutputGradientIteratorOptimizedParams const &params_;
  Conv2dProblemSize const &problem_size_;

  uint8_t const *global_ptr_;

  // current filter position (r, s)
  int filter_r_;
  int filter_s_;
  int filter_k_;
  int offset_n_;
  int offset_p_;
  int offset_q_;

  int lower_p_;
  int lower_q_;
  int upper_p_;
  int upper_q_;

  int warp_idx_;

public:

  CUTLASS_HOST_DEVICE
  Conv2dDgradOutputGradientTileAccessIteratorOptimizedAiu(
    Conv2dDgradOutputGradientIteratorOptimizedParams const &params,
    Conv2dProblemSize const &problem_size,
    Element const *ptr,
    int thread_idx,
    MatrixCoord const &threadblock_offset = MatrixCoord()       // tile index - units are threadblock-scoped tiles
  ):
    params_(params),
    problem_size_(problem_size),
    global_ptr_(reinterpret_cast<const uint8_t*>(ptr)),
    offset_n_(0),
    offset_p_(0),
    offset_q_(0),
    filter_k_(0),
    filter_r_(problem_size_.R - 1),
    filter_s_(problem_size_.S - 1) {

    warp_idx_ = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

    filter_k_ = threadblock_offset.column();

    int offset_nhw = threadblock_offset.row();

    int residual, h, w;

    params_.hw_divmod(offset_n_, residual, offset_nhw);
    params_.w_divmod(h, w, residual);

    offset_p_ = (h + problem_size_.pad_h - (problem_size_.R - 1) * problem_size_.dilation_h) / problem_size_.stride_h;
    // filter_s is start from S-1, offset_q_ corresponds to pos with filter_s=S-1
    offset_q_ = (w + problem_size_.pad_w - (problem_size_.S - 1) * problem_size_.dilation_w) / problem_size_.stride_w;


    // lower_q_ is first q corresponds to w=0 and s=S-1
    // this is the leftmost q of the conv window of the first w
    lower_q_ = problem_size_.pad_w - (problem_size_.S - 1) * problem_size_.dilation_w;
    lower_p_ = problem_size_.pad_h - (problem_size_.R - 1) * problem_size_.dilation_h;
    // upper corresponds to w=W-1 and s=S-1
    upper_q_ = - problem_size_.pad_w;
    upper_p_ = - problem_size_.pad_h;

    // offset_p_ = lower_p_;
    // offset_q_ = lower_q_;

    // lower_q_ = 0;
    // lower_p_ = 0;

      // TensorCoord coord = at_(offset_n[s], offset_h[s], offset_w[s], 0, 0);


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

private:


public:

  CUTLASS_HOST_DEVICE
  void advance() {

    int inc_next = 0;

    // moves to the next tile
    --filter_s_;
    if (filter_s_ == -1) {
      filter_s_ = problem_size_.S - 1;
      --filter_r_;
      inc_next = 1;

      if (filter_r_ == -1) {
        filter_r_ = problem_size_.R - 1;
        inc_next = 2;
        filter_k_ += params_.filter_k_delta;
      }
    }

    // stride always 1 in this implementation
    // advance on s
    // if (inc_next == 0) {
    //   offset_q_ -= problem_size_.dilation_w / problem_size_.stride_w;
    // } else if (inc_next == 1) {
    //   offset_q_ += (problem_size_.S - 1) * problem_size_.dilation_w / problem_size_.stride_w;
    //   offset_p_ -= problem_size_.dilation_h / problem_size_.stride_h;
    // } else {
    //   offset_q_ += (problem_size_.S - 1) * problem_size_.dilation_w / problem_size_.stride_w;
    //   offset_p_ += (problem_size_.R - 1) * problem_size_.dilation_h / problem_size_.stride_h;
    // }

  }

  CUTLASS_HOST_DEVICE
  void aiu_load(Element *tsm_ptr) const {

    // cutlass::arch::tsm_load_cube_sim(
    //   global_ptr_, tsm_ptr,
    //   problem_size_.N, 1, problem_size_.H, problem_size_.W, problem_size_.K,
    //   problem_size_.K, 1,
    //   offset_n_, 0, offset_p_, offset_q_, filter_k_,
    //   Shape::kRow, Shape::kColumn,
    //   0, problem_size_.pad_h, problem_size_.pad_w,
    //   1, problem_size_.stride_h, problem_size_.stride_w,
    //   1, problem_size_.R, problem_size_.S, 0, filter_r_, filter_s_,
    //   1, problem_size_.dilation_h, problem_size_.dilation_w,
    //   1
    // );

    if (warp_idx_ == 0) {
      // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
      //   printf("%d, %d, %d, %d, 11111111\n", offset_p_, offset_q_, upper_p_, upper_q_);
      // }
      CONV_AIU_LOAD_5D_IMPL<Element, Shape::kRow, Shape::kColumn>()(
        tsm_ptr, global_ptr_,
        problem_size_.K, problem_size_.Q, problem_size_.P, 1, problem_size_.N,
        params_.layout.stride()[0] * sizeof(Element), params_.layout.stride()[1] * sizeof(Element), params_.layout.stride()[1] * sizeof(Element), params_.layout.stride()[2] * sizeof(Element),
        lower_q_, lower_p_, 0,
        upper_q_, upper_p_, 0,
        1, 1, 1, 1,
        filter_k_, offset_q_, offset_p_, 0, offset_n_,
        // position in conv window should multiply dilation
        filter_s_ * problem_size_.dilation_w, filter_r_ * problem_size_.dilation_h, 0
      );
    }


  }

  /// Determines whether the Implicit GEMM can execute the given problem.
  CUTLASS_HOST_DEVICE
  static Status can_implement(Conv2dProblemSize const &problem_size) {

    // This is specialized for unit stride
    if (problem_size.stride() != MatrixCoord({1, 1})) {
      return Status::kErrorNotSupported;
    }

    // check alignment constraint on iterator's contiguous dimension
    if (problem_size.K % AccessSize) {
      return Status::kErrorNotSupported;
    }

    // Limit on filter size
    if (problem_size.R > 32 || problem_size.S > 32) {
      return Status::kErrorNotSupported;
    }
    return Status::kSuccess;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////


