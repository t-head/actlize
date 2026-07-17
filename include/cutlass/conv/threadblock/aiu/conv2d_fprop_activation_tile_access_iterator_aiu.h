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
    \brief Templates implementing loading of convolution tiles mapped to GEMM A (activation tile)
    matrix from memory.

    This iterator assumes TensorNHWC or TensorNCxHWx<Interleave> layout of tensors in Global Memory.

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
  typename Layout_,
  typename ThreadMap_,
  int AccessSize = ThreadMap_::kElementsPerAccess,
  int KernelSize = 0
>
class Conv2dFpropActivationTileAccessIteratorAiu {
public:

  //
  // Types
  //

  using Shape = Shape_;
  using Element = Element_;
  using Layout = Layout_;
  using TensorCoord = typename Layout::TensorCoord;
  using ThreadMap = ThreadMap_;
  using AccessType = AlignedArray<Element, AccessSize>;
  using TensorRef = cutlass::TensorRef<Element, Layout>;
  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  static IteratorAlgorithm const kIteratorAlgorithm = conv::IteratorAlgorithm::kOptimized;
  static StrideSupport const kStrideSupport = conv::StrideSupport::kStrided;
  static int const kConvDim = 2;
  using ConvProblemSize = typename conv::Conv2dProblemSize;

  static constexpr int SwzlMode = Shape::kColumn * sizeof(Element) == 128 ? 0 : 1;

  using Mask = uint64_t;

  //
  // Simplifying assertions
  //
#ifndef SAIL_CUSTOMIZE_CUTLASS
  static_assert(ThreadMap::Iterations::kContiguous == 1,
    "Require Iterations::kContiguous == 1");
#endif

  //
  // Parameters structure
  //

  using Params = Conv2dFpropActivationIteratorOptimizedParams<Layout>;

  static int const kAccessesPerVector = ThreadMap::kElementsPerAccess / AccessType::kElements;

private:

  Conv2dFpropActivationIteratorOptimizedParams<Layout> const &params_;
  Conv2dProblemSize const &problem_size_;

  // current filter position (r, s)
  int cur_r_;
  int cur_s_;

  int offset_n_;
  int offset_h_;
  int offset_w_;
  int offset_c_;

  int lower_h_;
  int lower_w_;
  int upper_h_;
  int upper_w_;

  int warp_idx_;

  uint8_t const *global_ptr_;

public:

  CUTLASS_HOST_DEVICE
  Conv2dFpropActivationTileAccessIteratorAiu(
    Conv2dFpropActivationIteratorOptimizedParams<Layout> const &params,
    Conv2dProblemSize const &problem_size,
    Element const *ptr,
    int thread_idx,
    MatrixCoord const &threadblock_offset = MatrixCoord()       // tile index - units are threadblock-scoped tiles
  ):
    params_(params),
    problem_size_(problem_size),
    global_ptr_(reinterpret_cast<const uint8_t*>(ptr)),
    offset_c_(0),
    cur_r_(0),
    cur_s_(0) {

    warp_idx_ = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

    int offset_npq = threadblock_offset.row();
    int offset_p, offset_q, residual;
    params.pq_divmod(offset_n_, residual, offset_npq);
    params.q_divmod(offset_p, offset_q, residual);

    offset_h_ = offset_p * problem_size_.stride_h - problem_size_.pad_h;
    offset_w_ = offset_q * problem_size_.stride_w - problem_size_.pad_w;
    offset_c_ = threadblock_offset.column();

    lower_h_ = -problem_size_.pad_h;
    lower_w_ = -problem_size_.pad_w;
    upper_h_ = lower_h_ + (problem_size_.P - 1) * problem_size_.stride_h - problem_size_.H + 1;
    upper_w_ = lower_w_ + (problem_size_.Q - 1) * problem_size_.stride_w - problem_size_.W + 1;
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

public:

  CUTLASS_HOST_DEVICE
  void advance() {

    // moves to the next tile
    ++cur_s_;
    // offset_w_ += problem_size_.dilation_w;
    if (cur_s_ == problem_size_.S) {
      cur_s_ = 0;
      // offset_w_ -= problem_size_.S * problem_size_.dilation_w;
      ++cur_r_;
      // offset_h_ += problem_size_.dilation_h;
      if (cur_r_ == problem_size_.R) {
        cur_r_ = 0;
        // offset_h_ -= problem_size_.R * problem_size_.dilation_h;
        offset_c_ += params_.filter_c_delta;
      }
    }



  }

  /// Returns a pointer to the vector starting at the current coordinate
  CUTLASS_HOST_DEVICE
  void aiu_load(Element *tsm_ptr) const {

    // cutlass::arch::tsm_load_cube_sim(
    //   global_ptr_, tsm_ptr,
    //   problem_size_.N, 1, problem_size_.H, problem_size_.W, problem_size_.C,
    //   problem_size_.C, 1,
    //   offset_n_, 0, offset_h_, offset_w_, offset_c_,
    //   Shape::kRow, Shape::kColumn,
    //   0, problem_size_.pad_h, problem_size_.pad_w,
    //   1, problem_size_.stride_h, problem_size_.stride_w,
    //   1, problem_size_.R, problem_size_.S, 0, cur_r_, cur_s_,
    //   1, problem_size_.dilation_h, problem_size_.dilation_w
    // );

    if (warp_idx_ == 0) {

      CONV_AIU_LOAD_5D_IMPL<Element, Shape::kRow, Shape::kColumn>()(
        tsm_ptr, global_ptr_,
        problem_size_.C, problem_size_.W, problem_size_.H, 1, problem_size_.N,
        params_.layout.stride()[0] * sizeof(Element), params_.layout.stride()[1] * sizeof(Element), params_.layout.stride()[1] * sizeof(Element), params_.layout.stride()[2] * sizeof(Element),
        lower_w_, lower_h_, 0,
        upper_w_, upper_h_, 0,
        problem_size_.stride_w, problem_size_.stride_h, 1, 1,
        offset_c_, offset_w_, offset_h_, 0, offset_n_,
        cur_s_ * problem_size_.dilation_w, cur_r_ * problem_size_.dilation_h, 0
      );

      // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
      //   printf("===========\n");
      //   printf("tsm_ptr: %p\n", tsm_ptr);
      //   printf("dim: %d, %d, %d, %d\n", problem_size_.C, problem_size_.W, problem_size_.H, problem_size_.N);
      //   printf("stride: %ld, %ld, %ld\n", params_.layout.stride()[0], params_.layout.stride()[1], params_.layout.stride()[2]);
      //   printf("lower: %d, %d, upper: %d, %d\n", lower_w_, lower_h_, upper_w_, upper_h_);
      //   printf("cube: %d, %d\n", Shape::kRow, Shape::kColumn);
      //   printf("conv_stride: %d, %d\n", problem_size_.stride_w, problem_size_.stride_h);
      //   printf("start: %d, %d, %d, %d\n", offset_c_, offset_w_, offset_h_, offset_n_);
      //   printf("im2col_offset: %d, %d\n", cur_s_, cur_r_);
      //   printf("SwzlMode: %d\n", SwzlMode);

      // }

    }
  }

  /// Determines whether the Implicit GEMM can execute the given problem.
  CUTLASS_HOST_DEVICE
  static Status can_implement(Conv2dProblemSize const &problem_size) {

    // check alignment constraint on iterator's contiguous dimension
    if (problem_size.C % AccessSize) {
      return Status::kErrorInvalidProblem;
    }

    if (platform::is_same<Layout, layout::TensorNCxHWx<32>>::value) {
      if (problem_size.C % 32) {
        return Status::kErrorInvalidProblem;
      }
    }

    if (platform::is_same<Layout, layout::TensorNCxHWx<64>>::value) {
      if (problem_size.C % 64) {
        return Status::kErrorInvalidProblem;
      }
    }

    // Conv2dFpropActivationTileAccessIteratorOptimized has constraint on filter positions
    // due to the number of mask bits.
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
