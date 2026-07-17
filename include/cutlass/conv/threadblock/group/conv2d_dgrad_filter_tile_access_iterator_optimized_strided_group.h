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
    \brief Templates implementing loading of convolution tiles mapped to GEMM B (filter tile)
    matrix from memory.

    This iterator assumes TensorNHWC layout of tensors in Global Memory.

    The iterator is specialized for each of the three convolution operators: forward propagation (Fprop),
    backward data gradient (Dgrad), and backward weight gradient (Wgrad).
*/

#if SAIL_DGRAD_STRIDE_OPT

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
  typename ThreadMap_,
  conv::StrideSupport StrideSupport_ = conv::StrideSupport::kStrided,
  int AccessSize = ThreadMap_::kElementsPerAccess
>
class Conv2dDgradFilterTileAccessIteratorOptimizedStridedGroup {
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
  static StrideSupport const kStrideSupport = StrideSupport_;
  static int const kConvDim = 2;
  using ConvProblemSize = typename conv::Conv2dProblemSize;

  static int const kAccessesPerVector = ThreadMap::kElementsPerAccess / AccessType::kElements;

  //
  // Parameters structure
  //

  struct Params : Conv2dDgradFilterIteratorOptimizedParamsStrided {

    //
    // Methods
    //
    CUTLASS_HOST_DEVICE
    Params() { }

    CUTLASS_HOST_DEVICE
    Params(Conv2dDgradFilterIteratorOptimizedParamsStrided const &base):
      Conv2dDgradFilterIteratorOptimizedParamsStrided(base) { }

    CUTLASS_HOST_DEVICE
    Params(
      Conv2dProblemSize const &problem_size,
      Layout const &layout
    ):
      Conv2dDgradFilterIteratorOptimizedParamsStrided(
        problem_size,
        layout,
        sizeof_bits<Element>::value,
        {Shape::kRow, Shape::kColumn},
        ThreadMap::kThreads,
        ThreadMap::kElementsPerAccess,
        {ThreadMap::Iterations::kContiguous, ThreadMap::Iterations::kStrided},
        {ThreadMap::Delta::kContiguous, ThreadMap::Delta::kStrided}
      ) { }

  };

private:

  Conv2dDgradFilterIteratorOptimizedParamsStrided const &params_;
  Conv2dProblemSize const &problem_size_;
  LongIndex iteration_contiguous_;
  LongIndex iteration_strided_;
  LongIndex iteration_vector_;
  char const *pointer_;

  uint32_t predicates_[kAccessesPerVector];
  int filter_s_;
  int filter_r_;
  int filter_k_;

  int rs_idx_;
  int start_r_;
  int start_s_;

  // group conv
  int problem_k_;

  //
  // Assertions
  //

  // We map predicates into bits packed in this uint32_t container
  static_assert(ThreadMap::Iterations::kStrided *
    ThreadMap::Iterations::kContiguous < sizeof(predicates_) * 8,
    "Currently, the number of loads per iteration is limited by the size of the predicates container.");

public:

  CUTLASS_HOST_DEVICE
  Conv2dDgradFilterTileAccessIteratorOptimizedStridedGroup(
    Conv2dDgradFilterIteratorOptimizedParamsStrided const &params,
    Conv2dProblemSize const &problem_size,
    Element const *ptr,
    int thread_idx,
    MatrixCoord const &threadblock_offset = MatrixCoord()
  ):
    params_(params),
    problem_size_(problem_size),
    pointer_(reinterpret_cast<char const *>(ptr)),
    predicates_{0},
    filter_s_(0),
    filter_r_(0),
    filter_k_(0) {

    // blockshape.m must equal to n to guarantee Shape::kColumn identical to Shape::kRow in data iterator
    // block.y is on NHW direction, can't get this index in iterb, so use blockIdx.y directly
    if (blockIdx.y < problem_size.block_num[0]) {
      rs_idx_ = 0;
    } else if (blockIdx.y < problem_size.block_num[1]) {
      rs_idx_ = 1;
    } else if (blockIdx.y < problem_size.block_num[2]) {
      rs_idx_ = 2;
    } else {
      rs_idx_ = 3;
    }
    filter_r_ = problem_size.start_r_[rs_idx_];
    filter_s_ = problem_size.start_s_[rs_idx_];
    start_r_ = filter_r_;
    start_s_ = filter_s_;

    layout::PitchLinearCoord thread_coord = ThreadMap::initial_offset(thread_idx);

    // filter_k_ = threadblock_offset.row() + thread_coord.strided();
    Index column = threadblock_offset.column() + thread_coord.contiguous();

    // add for group conv
    int channels_per_group_ = problem_size_.C / problem_size_.groups;
    int kernels_per_group_ = problem_size_.K / problem_size_.groups;
    int block_offset = threadblock_offset.column() / kernels_per_group_ * channels_per_group_;
    problem_k_ = Shape::kColumn / channels_per_group_ * kernels_per_group_ + block_offset;
    filter_k_ = threadblock_offset.row() + thread_coord.strided() + block_offset;

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {

        int filter_k = filter_k_ + s * ThreadMap::Delta::kStrided;
        int filter_c = column + c * ThreadMap::Delta::kContiguous;

        // add group conv
        int group_k = (filter_c / channels_per_group_ + 1) * kernels_per_group_;
        int group_c = (filter_k / kernels_per_group_ + 1) * channels_per_group_;

        int pred_idx = c + s * ThreadMap::Iterations::kContiguous;

        CUTLASS_PRAGMA_UNROLL
        for (int v_idx = 0; v_idx < kAccessesPerVector; ++v_idx) {
          uint32_t pred = ((filter_k < problem_k_
                        && filter_k < group_k + v_idx * kernels_per_group_
                        && filter_c + v_idx * AccessType::kElements < problem_size_.C
                        && filter_c + v_idx * AccessType::kElements < group_c)
                        && filter_r_ < problem_size_.R
                        && filter_s_ < problem_size_.S ? 1u : 0);

          // one bit for each loop, whether k and c are valid
          predicates_[v_idx] |= (pred << pred_idx);
        }
      }
    }

    column -= (column / channels_per_group_) * channels_per_group_;
    // r and s = 0, k and c is thread first iter's position

    pointer_ += (
      filter_k_ * params.layout.stride()[2] + filter_r_ * params.layout.stride()[1] + filter_s_ * params.layout.stride()[0] + column
    ) * sizeof_bits<Element>::value / 8;

    set_iteration_index(0);
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

    pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
  }

  CUTLASS_HOST_DEVICE
  void advance() {

#if __HGGCCC_RTC__
    LongIndex next;
    if (rs_idx_ == 0) {
      next = params_.inc_next_s_[0];
    } else if (rs_idx_ == 1) {
      next = params_.inc_next_s_[1];
    } else if (rs_idx_ == 2) {
      next = params_.inc_next_s_[2];
    } else {
      next = params_.inc_next_s_[3];
    }
#else
    LongIndex next = params_.inc_next_s_[rs_idx_];
#endif

    // moves to the next tile
    filter_s_ += 2;
    if (filter_s_ >= problem_size_.S) {

      filter_s_ = start_s_;
      filter_r_ += 2;
      if (filter_r_ < problem_size_.R) {
#if __HGGCCC_RTC__
        if (rs_idx_ == 0) {
          next = params_.inc_next_r_[0];
        } else if (rs_idx_ == 1) {
          next = params_.inc_next_r_[1];
        } else if (rs_idx_ == 2) {
          next = params_.inc_next_r_[2];
        } else {
          next = params_.inc_next_r_[3];
        }
#else
        next = params_.inc_next_r_[rs_idx_];
#endif
      } else {
        filter_r_ = start_r_;
#if __HGGCCC_RTC__
        if (rs_idx_ == 0) {
          next = params_.inc_next_k_[0];
        } else if (rs_idx_ == 1) {
          next = params_.inc_next_k_[1];
        } else if (rs_idx_ == 2) {
          next = params_.inc_next_k_[2];
        } else {
          next = params_.inc_next_k_[3];
        }
#else
        next = params_.inc_next_k_[rs_idx_];
#endif
        filter_k_ += params_.filter_k_delta;
      }
    }

    // Clear predicates if needed
    // update predicates if new k is invalid
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      if (filter_k_ + s * ThreadMap::Delta::kStrided >= problem_k_) {
        uint32_t kClearMask = ((1u << ThreadMap::Iterations::kContiguous) - 1) << (s * ThreadMap::Iterations::kContiguous);

        // clear all predicates_ on this strided index
        CUTLASS_PRAGMA_UNROLL
        for (int v_idx = 0; v_idx < kAccessesPerVector; ++v_idx) {
          predicates_[v_idx] = (predicates_[v_idx] & (~kClearMask));
        }
      }
    }

    pointer_ += next;
  }

  /// Returns true if the current coordinate is within the filter tensor W
  CUTLASS_HOST_DEVICE
  bool valid() {
    LongIndex pred_idx = iteration_contiguous_ + iteration_strided_ * ThreadMap::Iterations::kContiguous;
    return (predicates_[iteration_vector_] & (1u << pred_idx));
  }

  /// Returns a pointer to the vector starting at the current coordinate
  CUTLASS_HOST_DEVICE
  AccessType const *get() const {
    return reinterpret_cast<AccessType const *>(pointer_ +
      iteration_contiguous_ * ThreadMap::Delta::kContiguous * sizeof_bits<Element>::value / 8);
  }

  /// Increments to the next memory access
  CUTLASS_HOST_DEVICE
  Conv2dDgradFilterTileAccessIteratorOptimizedStridedGroup &operator++() {
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

      // Move to the next K coordinate within the tile
      pointer_ += params_.inc_next_strided;

      return *this;
    }
    iteration_strided_ = 0;

    return *this;
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

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////


