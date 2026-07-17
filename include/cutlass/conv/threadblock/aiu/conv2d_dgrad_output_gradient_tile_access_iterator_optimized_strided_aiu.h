/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved. 
 * Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
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
  conv::StrideSupport StrideSupport_ = conv::StrideSupport::kStrided,
  int AccessSize = ThreadMap_::kElementsPerAccess
>
class Conv2dDgradOutputGradientTileAccessIteratorOptimizedStridedAiu {
public:

  static_assert(StrideSupport_ == conv::StrideSupport::kStrided,
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
  static StrideSupport const kStrideSupport = conv::StrideSupport::kStrided;
  static int const kConvDim = 2;
  using ConvProblemSize = typename conv::Conv2dProblemSize;

  using Mask = uint64_t;

  //
  // Simplifying assertions
  //
  static_assert(ThreadMap::Iterations::kContiguous == 1,
    "Require Iterations::kContiguous == 1");

  //
  // Parameters structure
  //

  using Params = Conv2dDgradOutputGradientIteratorOptimizedParamsStrided;

  static int const kAccessesPerVector = ThreadMap::kElementsPerAccess / AccessType::kElements;

private:

  Conv2dDgradOutputGradientIteratorOptimizedParamsStrided const &params_;
  Conv2dProblemSize const &problem_size_;
  LongIndex iteration_contiguous_;
  LongIndex iteration_strided_;
  LongIndex iteration_vector_;
  // can't use param_.inc_next since each block has different rs loop number
  int64_t inc_next[3];

  uint8_t const *global_ptr_;

  // current filter position (r, s)
  int filter_r_;
  int filter_s_;
  int filter_k_;

  int last_r_;
  int last_s_;

  int loop_num_;
  int start_r_;
  int start_s_;
  int rs_idx_;

  int offset_n_;
  int offset_p_;
  int offset_q_;

  int lower_p_;
  int lower_q_;
  int upper_p_;
  int upper_q_;

  int warp_idx_;


  Index masks_[ThreadMap::Iterations::kStrided][kAccessesPerVector][2];

public:

  CUTLASS_HOST_DEVICE
  Conv2dDgradOutputGradientTileAccessIteratorOptimizedStridedAiu(
    Conv2dDgradOutputGradientIteratorOptimizedParamsStrided const &params,
    Conv2dProblemSize const &problem_size,
    Element const *ptr,
    int thread_idx,
    MatrixCoord const &threadblock_offset = MatrixCoord()       // tile index - units are threadblock-scoped tiles
  ):
    params_(params),
    problem_size_(problem_size),
    global_ptr_(reinterpret_cast<const uint8_t*>(ptr)),
    filter_k_(0),
    filter_r_(0),
    filter_s_(0) {

    warp_idx_ = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

    // cur block belongs to which rs position
    // index in blocks with cur rs
    int block_rs_idx = 0;
    // h/w size for current rs position
    // block.y is on NHW direction, can't get this index in iterb, so use blockIdx.y directly
    if (blockIdx.y < problem_size.block_num[0]) {
      rs_idx_ = 0;
      block_rs_idx = blockIdx.y;
    } else if (blockIdx.y < problem_size.block_num[1]) {
      rs_idx_ = 1;
      block_rs_idx = blockIdx.y - problem_size.block_num[0];
    } else if (blockIdx.y < problem_size.block_num[2]) {
      rs_idx_ = 2;
      block_rs_idx = blockIdx.y - problem_size.block_num[1];
    } else {
      rs_idx_ = 3;
      block_rs_idx = blockIdx.y - problem_size.block_num[2];
    }

    // update filter_r/s to last valid position
    filter_r_ = problem_size.loop_r[rs_idx_] - 1;
    filter_s_ = problem_size.loop_s[rs_idx_] - 1;

    // filter_r_ /= problem_size_.stride_h;
    // filter_s_ /= problem_size_.stride_w;

    // should supply on k dimension only, because k loop is independent to rs loop
    // the complementary k loop should run for all rs loop
    loop_num_ = problem_size.loop_r[rs_idx_] * problem_size.loop_s[rs_idx_] * (problem_size.K + Shape::kColumn - 1) / Shape::kColumn;

    if (blockIdx.y >= problem_size.block_num[3]) {
      loop_num_ = 0;
    }

    filter_k_ = threadblock_offset.column();

    // offset in all elements in current rs
    int offset_nhw = block_rs_idx * Shape::kRow;

    offset_n_ = offset_nhw / (problem_size.valid_h[rs_idx_] * problem_size.valid_w[rs_idx_]);
    int residual = offset_nhw % (problem_size.valid_h[rs_idx_] * problem_size.valid_w[rs_idx_]);

    int h, w;
    // index in elements of current valid rs position
    h = residual / problem_size.valid_w[rs_idx_];
    w = residual % problem_size.valid_w[rs_idx_];
    // transfer to ori input coord
    h *= 2;
    w *= 2;
    h += problem_size.start_h[rs_idx_];
    w += problem_size.start_w[rs_idx_];
    // get p cooresponds to h and start_r, than minus loop_r - 1
    // offset_p_ = (h + problem_size_.pad_h - filter_r_ * problem_size_.dilation_h) / problem_size.stride_h;
    // offset_q_ = (w + problem_size_.pad_w - filter_s_ * problem_size_.dilation_w) / problem_size.stride_w;
    offset_p_ = (h + problem_size_.pad_h - problem_size.start_r_[rs_idx_] * problem_size_.dilation_h) / problem_size.stride_h - problem_size.loop_r[rs_idx_] + 1;
    offset_q_ = (w + problem_size_.pad_w - problem_size.start_s_[rs_idx_] * problem_size_.dilation_w) / problem_size.stride_w - problem_size.loop_s[rs_idx_] + 1;

    // chang h to start_h in offset_p_
    lower_p_ = (problem_size.start_h[rs_idx_] + problem_size_.pad_h - problem_size.start_r_[rs_idx_] * problem_size_.dilation_h) / problem_size.stride_h - problem_size.loop_r[rs_idx_] + 1;
    lower_q_ = (problem_size.start_w[rs_idx_] + problem_size_.pad_w - problem_size.start_s_[rs_idx_] * problem_size_.dilation_w) / problem_size.stride_w - problem_size.loop_s[rs_idx_] + 1;
    // chang h to last h and compare to P
    upper_p_ = (problem_size.start_h[rs_idx_] + (problem_size.valid_h[rs_idx_] - 1) * problem_size.stride_h + problem_size_.pad_h - problem_size.start_r_[rs_idx_] * problem_size_.dilation_h) / problem_size.stride_h - problem_size_.P + 1 - problem_size.loop_r[rs_idx_] + 1;
    upper_q_ = (problem_size.start_w[rs_idx_] + (problem_size.valid_w[rs_idx_] - 1) * problem_size.stride_w + problem_size_.pad_w - problem_size.start_s_[rs_idx_] * problem_size_.dilation_w) / problem_size.stride_w - problem_size_.Q + 1 - problem_size.loop_s[rs_idx_] + 1;

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
  int get_loop_num() {
    return loop_num_;
  }


  CUTLASS_HOST_DEVICE
  void advance() {

    int next_idx = 0;

    // moves to the next tile
    // TODO: only support stride = 2 now
    --filter_s_;
    if (filter_s_ < 0) {
      filter_s_ = problem_size_.loop_s[rs_idx_] - 1;
      --filter_r_;
      // next_idx = 1;

      if (filter_r_ < 0) {
        filter_r_ = problem_size_.loop_r[rs_idx_] - 1;
        // inc_next = 2;
        filter_k_ += params_.filter_k_delta;
      }
    }

    // stride always 1 in this implementation
    // advance on s
    // if (next_idx == 0) {
    //   // successive valid r/s positions adjacent dilation dy
    //   offset_q_ -= problem_size_.dilation_w;
    // } else if (next_idx == 1) {
    //   // advance (problem_size_.S - start_s_) / stride times
    //   offset_q_ += (problem_size_.S - start_s_ - 1) / problem_size_.stride_w * problem_size_.dilation_w;
    //   offset_p_ -= problem_size_.dilation_h;
    // } else {
    //   offset_q_ += (problem_size_.S - start_s_ - 1) / problem_size_.stride_w * problem_size_.dilation_w;
    //   offset_p_ += (problem_size_.R - start_r_ - 1) / problem_size_.stride_h * problem_size_.dilation_h;
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
    if (problem_size.stride() != MatrixCoord({2, 2})) {
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


