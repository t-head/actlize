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
    \brief Templates implementing loading of convolution tiles mapped to GEMM B (activation tile)
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
#if SAIL_WGRAD_ITER_OPT
  , bool UnityRS = 0  // kernel_size/stride = 1, padding = 0
#endif
>
class Conv2dWgradActivationTileAccessIteratorOptimizedAiu {
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
  static IteratorAlgorithm const kIteratorAlgorithm = conv::IteratorAlgorithm::kAnalytic;
  static StrideSupport const kStrideSupport = conv::StrideSupport::kStrided;
  static int const kConvDim = 2;
  using ConvProblemSize = typename conv::Conv2dProblemSize;

  static_assert(sizeof_bits<Element>::value >= 8,
    "WGRAD requires elements of size 8b or greater.");

  //
  // Parameters structure
  //

  using Params = Conv2dWgradActivationIteratorOptimizedParams;

  static int const kAccessesPerVector = ThreadMap::kElementsPerAccess / AccessType::kElements;

  static int const CubeW = Shape::kColumn > 128 / sizeof(Element) ? 128 / sizeof(Element) : Shape::kColumn;

  static int const AiuLoadIter = Shape::kColumn / CubeW;

  static int const CubeC = 128 * 8 / sizeof_bits<Element>::value;

  // Shape::kRow is on gemm.k
  static int const CubeStride = CubeW * Shape::kRow;

  // static int const ElemInCube = 128 * 8 / sizeof_bits<Element>::value;

private:

  Conv2dWgradActivationIteratorOptimizedParams const &params_;
  Conv2dProblemSize const &problem_size_;
  LongIndex iteration_contiguous_;
  LongIndex iteration_strided_;
  LongIndex iteration_vector_;
  uint8_t const *global_ptr_;


  int cur_r_[AiuLoadIter];
  int cur_s_[AiuLoadIter];

  // Precomputed effective filter postion (r,s) in contiguous dimension stays constant for each gemm_iteration_k
  // required for npq -> nhw translation
  // int precomputed_filter_r_[AiuLoadIter];
  // int precomputed_filter_s_[AiuLoadIter];

  // Channel dimension in contiguous dimension stays constant for each gemm_iteration_k
  int filter_c_[AiuLoadIter];

  int offset_npq_;
  int offset_n_;
  int offset_p_;
  int offset_q_;

  int lower_h_;
  int lower_w_;
  int upper_h_;
  int upper_w_;


  int warp_idx_;

public:

  CUTLASS_HOST_DEVICE
  Conv2dWgradActivationTileAccessIteratorOptimizedAiu(
    Conv2dWgradActivationIteratorOptimizedParams const &params,
    Conv2dProblemSize const &problem_size,
    Element const *ptr,
    int thread_idx,
    MatrixCoord const &threadblock_offset = MatrixCoord()
  ):
    params_(params),
    problem_size_(problem_size),
    global_ptr_(reinterpret_cast<const uint8_t*>(ptr))
  {

    warp_idx_ = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

    layout::PitchLinearCoord thread_coord = ThreadMap::initial_offset(thread_idx);

    // initialize r,s,c filter position for every contiguous iteration
    CUTLASS_PRAGMA_UNROLL
    for(int c = 0; c < AiuLoadIter; ++c) {

      int rsc_offset = threadblock_offset.column()
                        + c * CubeW;

      // The subseqnet fast_divmod() operations are equivalent to the following logical computation:
      //
      //
      // filter_r_[c] = rsc_offset / (problem_size_.S * problem_size_.C);
      // int residual = rsc_offset % (problem_size_.S * problem_size_.C);
      //
      // filter_s_[c] = residual / problem_size_.C;
      // filter_c_[c] = residual % problem_size_.C;

      int residual;

      params_.sc_divmod(cur_r_[c], residual, rsc_offset);
      params_.c_divmod(cur_s_[c], filter_c_[c], residual);

      int r = cur_r_[c];
      int s = cur_s_[c];

      // if (problem_size_.mode == Mode::kConvolution) {
      //   r = (problem_size_.R - 1 - r);
      //   s = (problem_size_.S - 1 - s);
      // }

      // precomputed_filter_r_[c] =  - problem_size_.pad_h + r * problem_size_.dilation_h;
      // precomputed_filter_s_[c] =  - problem_size_.pad_w + s * problem_size_.dilation_w;

    }

    offset_npq_ = threadblock_offset.row();
    int residual;
    params_.pq_divmod(offset_n_, residual, offset_npq_);
    params_.q_divmod(offset_p_, offset_q_, residual);

    lower_h_ = -problem_size_.pad_h;
    lower_w_ = -problem_size_.pad_w;
    upper_h_ = lower_h_ + (problem_size_.H + 2 * problem_size_.pad_h - problem_size_.dilation_h * (problem_size_.R - 1) - 1) - problem_size_.H + 1;
    upper_w_ = lower_w_ + (problem_size_.W + 2 * problem_size_.pad_w - problem_size_.dilation_w * (problem_size_.S - 1) - 1) - problem_size_.W + 1;

  }


  CUTLASS_HOST_DEVICE
  void advance() {

    offset_npq_ += Shape::kRow * problem_size_.split_k_slices;
    int residual;
    params_.pq_divmod(offset_n_, residual, offset_npq_);
    params_.q_divmod(offset_p_, offset_q_, residual);
  }

  CUTLASS_HOST_DEVICE
  void aiu_load(Element *tsm_ptr) const {

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < AiuLoadIter; i++) {
      int offset_h = offset_p_ * problem_size_.stride_h - problem_size_.pad_h;
      int offset_w = offset_q_ * problem_size_.stride_w - problem_size_.pad_w;
      // int offset_h = offset_p_ * problem_size_.stride_h + precomputed_filter_r_[i];
      // int offset_w = offset_q_ * problem_size_.stride_w + precomputed_filter_s_[i];

      // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
      //   printf("aiu_loop: %d, off_n: %d, off_h: %d, off_w: %d, off_c: %d\n", i, offset_n_, offset_h, offset_w, filter_c_[i]);
      //   printf("cur_r: %d, cur_s: %d\n", cur_r_[i], cur_s_[i]);
      // }

      // cutlass::arch::tsm_load_cube_sim(
      //   global_ptr_, tsm_ptr + i * CubeStride,
      //   problem_size_.N, 1, problem_size_.H, problem_size_.W, problem_size_.C,
      //   problem_size_.C, 1,
      //   offset_n_, 0, offset_h, offset_w, filter_c_[i],
      //   Shape::kRow, CubeC,
      //   0, problem_size_.pad_h, problem_size_.pad_w,
      //   1, problem_size_.stride_h, problem_size_.stride_w,
      //   1, problem_size_.R, problem_size_.S, 0, cur_r_[i], cur_s_[i],
      //   1, problem_size_.dilation_h, problem_size_.dilation_w
      // );

      if (warp_idx_ == 0) {

        CONV_AIU_LOAD_5D_IMPL<Element, Shape::kRow, CubeW>()(
          tsm_ptr + i * Shape::kRow * CubeW, global_ptr_,
          problem_size_.C, problem_size_.W, problem_size_.H, 1, problem_size_.N,
          params_.layout.stride()[0] * sizeof(Element), params_.layout.stride()[1] * sizeof(Element), params_.layout.stride()[1] * sizeof(Element), params_.layout.stride()[2] * sizeof(Element),
          lower_w_, lower_h_, 0,
          upper_w_, upper_h_, 0,
          problem_size_.stride_w, problem_size_.stride_h, 1, 1,
          filter_c_[i], offset_w, offset_h, 0, offset_n_,
          cur_s_[i] * problem_size_.dilation_w, cur_r_[i] * problem_size_.dilation_h, 0
        );
      }
    }

  }

  /// Determines whether the Implicit GEMM can execute the given problem.
  CUTLASS_HOST_DEVICE
  static Status can_implement(Conv2dProblemSize const &problem_size) {

    // check alignment constraint on iterator's contiguous dimension
    if (problem_size.C % AccessSize) {
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
