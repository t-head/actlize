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
  LongIndex iteration_vector_;
  char const *pointer_;

  // Precomputed effective filter postion (r,s) in contiguous dimension stays constant for each gemm_iteration_k
  // required for npq -> nhw translation
  int precomputed_filter_r_[ThreadMap::Iterations::kContiguous][kAccessesPerVector];
  int precomputed_filter_s_[ThreadMap::Iterations::kContiguous][kAccessesPerVector];
  int filter_c_[ThreadMap::Iterations::kContiguous][kAccessesPerVector];

  int offset_pq_[ThreadMap::Iterations::kStrided];
  int offset_p_[ThreadMap::Iterations::kStrided];
  int offset_q_[ThreadMap::Iterations::kStrided];
  // r/s validate judge
  int predicates_[kAccessesPerVector];
  int pq_;
  int n_;

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
    pointer_(reinterpret_cast<char const *>(ptr)),
    problem_size_(problem_size)
  {
    CUTLASS_PRAGMA_UNROLL
    for (int v_idx = 0; v_idx < kAccessesPerVector; ++v_idx) {
      predicates_[v_idx] = 0;
    }

    int block_per_batch = gridDim.z / problem_size_.N;

    pq_ = problem_size_.P * problem_size_.Q;
    // each block process one batch
    n_ = blockIdx.z / block_per_batch;
    int block_in_batch = blockIdx.z % block_per_batch;
    int gemm_k_iterations = implicit_gemm_k_iterations(Operator::kWgrad, Shape::kRow, problem_size);
    layout::PitchLinearCoord thread_coord = ThreadMap::initial_offset(thread_idx);

    
    CUTLASS_PRAGMA_UNROLL
    for(int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
      CUTLASS_PRAGMA_UNROLL
      for (int v_idx = 0; v_idx < kAccessesPerVector; ++v_idx) {

        int rsc_offset = threadblock_offset.column() + thread_coord.contiguous()
                            + c * ThreadMap::Delta::kContiguous + v_idx * AccessType::kElements;

        uint32_t pred = rsc_offset < problem_size_.R * problem_size_.S * problem_size_.C;
        predicates_[v_idx] |= (pred << c);

        int residual, r_, s_;

        params_.sc_divmod(r_, residual, rsc_offset);
        params_.c_divmod(s_, filter_c_[c][v_idx], residual);

        if (problem_size_.mode == Mode::kConvolution) {
          r_ = (problem_size_.R - 1 - r_);
          s_ = (problem_size_.S - 1 - s_);
        }

        // one thread read two s, the input data of successive s are also successive
        // thus can only calc position of first s and read 8 elements
        // needn't minus padding here, since input data already padded
        precomputed_filter_r_[c][v_idx] =  r_ * problem_size_.dilation_h;
        precomputed_filter_s_[c][v_idx] =  s_ * problem_size_.dilation_w;


        // initialize n, p, q offset for every strided iteration
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {

          // each block do contiguous npq to ensure advance will not jump multiply rows
          offset_pq_[s] = block_in_batch * Shape::kRow * gemm_k_iterations + thread_coord.strided() + s * ThreadMap::Delta::kStrided;
          params_.q_divmod(offset_p_[s], offset_q_[s], offset_pq_[s]);
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
    iteration_vector_ = index % kAccessesPerVector;
    int residual_access = index / kAccessesPerVector;

    iteration_contiguous_ = residual_access % ThreadMap::Iterations::kContiguous;
    iteration_strided_ = residual_access / ThreadMap::Iterations::kContiguous;
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    CUTLASS_PRAGMA_UNROLL
    for(int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
      CUTLASS_PRAGMA_UNROLL
      for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
        pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
      }
    }
  }

  CUTLASS_HOST_DEVICE
  void advance() {
    
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      offset_pq_[s] += Shape::kRow;
      params_.q_divmod(offset_p_[s], offset_q_[s], offset_pq_[s]);
    }
  }

  /// Returns true if the current coordinate is within the activation tensor x
  CUTLASS_HOST_DEVICE
  bool valid() const {
    // return false;
    // avoid extra async copy
    return offset_pq_[iteration_strided_] < pq_
            && (predicates_[iteration_vector_] & (1u << iteration_contiguous_));
  }

  CUTLASS_HOST_DEVICE
  TensorCoord at() const {

    int h = offset_p_[iteration_strided_] * problem_size_.stride_h + precomputed_filter_r_[iteration_contiguous_][iteration_vector_];
    int w = offset_q_[iteration_strided_] * problem_size_.stride_w + precomputed_filter_s_[iteration_contiguous_][iteration_vector_];

    return TensorCoord(n_, h, w, filter_c_[iteration_contiguous_][iteration_vector_]);
  }

  /// Returns a pointer to the vector starting at the current coordinate
  CUTLASS_HOST_DEVICE
  AccessType const *get() const {
    TensorCoord coord = at();
    LongIndex offset = params_.layout(coord);

    // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 108 && threadIdx.x == 96) {
    //   printf("%ld, 11111111111111\n", offset);
    // }

    // if (offset % 4 != 0) {
    //   printf("thread: %d, block: %d, %d, %d, off: %ld, 1111111\n", threadIdx.x, blockIdx.x, blockIdx.y, blockIdx.z, offset);
    // }

    return reinterpret_cast<AccessType const *>(pointer_ + offset * sizeof_bits<Element>::value / 8);
  }

  /// Increments to the next memory access
  CUTLASS_HOST_DEVICE
  Conv2dWgradActivationTileAccessIteratorOptimizedSmallChannel &operator++() {
    ++iteration_vector_;
    if (iteration_vector_ < kAccessesPerVector) {
      return *this;
    }

    iteration_vector_ = 0;

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

    return Status::kSuccess;
  }
  
};
/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
