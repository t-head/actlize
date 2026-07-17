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
class Conv2dWgradActivationTileAccessIteratorOptimizedSmallChannel {
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

private:

  Conv2dWgradActivationIteratorOptimizedParams const &params_;
  Conv2dProblemSize const &problem_size_;
  LongIndex iteration_contiguous_;
  LongIndex iteration_strided_;
  char const *pointer_[ThreadMap::Iterations::kStrided][ThreadMap::Iterations::kContiguous];

  // Precomputed effective filter postion (r,s) in contiguous dimension stays constant for each gemm_iteration_k
  // required for npq -> nhw translation
  int precomputed_filter_r_[ThreadMap::Iterations::kContiguous];
  int precomputed_filter_s_[ThreadMap::Iterations::kContiguous];

  int offset_pq_[ThreadMap::Iterations::kStrided][ThreadMap::Iterations::kContiguous];
  // current q position
  int q_[ThreadMap::Iterations::kStrided][ThreadMap::Iterations::kContiguous];
  // r/s validate judge
  int predicates_;
  int pq_;

  Element const *base;

public:

  CUTLASS_HOST_DEVICE
  Conv2dWgradActivationTileAccessIteratorOptimizedSmallChannel(
    Conv2dWgradActivationIteratorOptimizedParams const &params,
    Conv2dProblemSize const &problem_size,
    Element const *ptr,
    int thread_idx,
    MatrixCoord const &threadblock_offset = MatrixCoord()
  ):
    params_(params),
    base(ptr),
    predicates_(0),
    problem_size_(problem_size)
  {

    pq_ = problem_size_.P * problem_size_.Q;
    // each block process one batch
    int n = threadblock_offset.row() / Shape::kRow;
    int gemm_k_iterations = implicit_gemm_k_iterations(Operator::kWgrad, Shape::kRow, problem_size);
    layout::PitchLinearCoord thread_coord = ThreadMap::initial_offset(thread_idx);

    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 3) {
      // printf("thread: %d, row: %d, col: %d\n", threadIdx.x, thread_coord.strided(), thread_coord.contiguous());
    }
    
    CUTLASS_PRAGMA_UNROLL
    for(int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
      int rsc_offset = threadblock_offset.column() + thread_coord.contiguous()
                          + c * ThreadMap::Delta::kContiguous;

      uint32_t pred = rsc_offset < problem_size_.R * problem_size_.S * problem_size_.C;
      predicates_ |= (pred << c);

      int residual, r_, s_, c_;

      params_.sc_divmod(r_, residual, rsc_offset);
      params_.c_divmod(s_, c_, residual);

      if (problem_size_.mode == Mode::kConvolution) {
        r_ = (problem_size_.R - 1 - r_);
        s_ = (problem_size_.S - 1 - s_);
      }

      // one thread read two s, the input data of successive s are also successive
      // thus can only calc position of first s and read 8 elements
      // needn't minus padding here, since input data already padded
      r_ =  r_ * problem_size_.dilation_h;
      s_ =  s_ * problem_size_.dilation_w;


      // initialize n, p, q offset for every strided iteration
      CUTLASS_PRAGMA_UNROLL
      for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {

        // each block do contiguous npq to ensure advance will not jump multiply rows
        offset_pq_[s][c] = thread_coord.strided() + s * ThreadMap::Delta::kStrided;
        int p;
        params_.q_divmod(p, q_[s][c], offset_pq_[s][c]);
        int h = p * problem_size_.stride_h + r_;
        int w = q_[s][c] * problem_size_.stride_w + s_;
        TensorCoord coord(n, h, w, c_);

        LongIndex offset = params_.layout(coord);
        pointer_[s][c] = reinterpret_cast<char const *>(ptr);
        pointer_[s][c] += offset * sizeof_bits<Element>::value / 8;

        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 6 && s == 0) {
          // printf("pq: %d, r: %d, s: %d, h: %d, w: %d, c: %d, offset: %ld\n", offset_pq_[s][c], r_, s_, h, w, c_, offset);
        }
      }
    }

#if SAIL_TMP_WORKAROUND
    set_iteration_index(0);
#endif
  }

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(Index index) {
    iteration_contiguous_ = index % ThreadMap::Iterations::kContiguous;
    iteration_strided_ = index / ThreadMap::Iterations::kContiguous;
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
  }

  CUTLASS_HOST_DEVICE
  void advance() {
    
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for(int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        // stride_w = 2
        pointer_[s][c] += Shape::kRow * problem_size_.stride_w * params_.layout.stride()[0] * sizeof_bits<Element>::value / 8;
        offset_pq_[s][c] += Shape::kRow;
        q_[s][c] += Shape::kRow;
        // row increased, pointer_ should add one more input line
        if (q_[s][c] >= problem_size_.Q) {
          pointer_[s][c] += (problem_size_.stride_h - 1) * params_.layout.stride()[1] * sizeof_bits<Element>::value / 8;
          // add padding area, invalid pixels after last valid pixel+1
          // for resnet case, last w=229, last valid w=222
          // 222+2=224 which is invalid, should advance 6 more position to next row
          pointer_[s][c] += params_.invalid_num * params_.layout.stride()[0] * sizeof_bits<Element>::value / 8;
          q_[s][c] -= problem_size_.Q;
        }
      }
    }
  }

  /// Returns true if the current coordinate is within the activation tensor x
  CUTLASS_HOST_DEVICE
  bool valid() const {
    // avoid extra async copy
    return offset_pq_[iteration_strided_][iteration_contiguous_] < pq_
            && (predicates_ & (1u << iteration_contiguous_));
  }

  /// Returns a pointer to the vector starting at the current coordinate
  CUTLASS_HOST_DEVICE
  AccessType const *get() const {
    bool valid = offset_pq_[iteration_strided_][iteration_contiguous_] < pq_;
    const Element *ptr = reinterpret_cast<const Element*>(pointer_[iteration_strided_][iteration_contiguous_]);
    if (threadIdx.x == 6 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && iteration_contiguous_ == 0) {
      // printf("thread: %d, iter: %ld, valid: %d, add: %d, val: %f\n", threadIdx.x, iteration_strided_, valid, int(ptr - base) / int(params_.layout.stride()[0]), float(*ptr));
    }

    return reinterpret_cast<AccessType const *>(pointer_[iteration_strided_][iteration_contiguous_]);
  }

  /// Increments to the next memory access
  CUTLASS_HOST_DEVICE
  Conv2dWgradActivationTileAccessIteratorOptimizedSmallChannel &operator++() {

    ++iteration_contiguous_;
    if (iteration_contiguous_ < ThreadMap::Iterations::kContiguous) {
      return *this;
    }
    iteration_contiguous_ = 0;
    ++iteration_strided_;
    if (iteration_strided_ < ThreadMap::Iterations::kStrided) {
      return *this;
    }
    iteration_strided_ = 0;
 
    return *this;
  }

  /// Determines whether the Implicit GEMM can execute the given problem.
  CUTLASS_HOST_DEVICE
  static Status can_implement(Conv2dProblemSize const &problem_size) {

    // check alignment constraint on iterator's contiguous dimension
    // if (problem_size.R != 8 || problem_size.S != 8 || problem_size.C != 4
    //     || problem_size.stride_h != 2 || problem_size.stride_w != 2
    //     // p don't advance in every main loop
    //     || problem_size.Q <= Shape::kRow) {
    //     // || ThreadMap::Iterations::kContiguous != 1) {
    //   return Status::kErrorInvalidProblem;
    // }

    return Status::kSuccess;
  }
  
};
/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
