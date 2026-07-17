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
  bool FoldOnR,
  int AccessSize = ThreadMap_::kElementsPerAccess
>
class Conv2dFpropActivationTileAccessIteratorOptimizedSmallChannel {
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

  //
  // Parameters structure
  //

  using Params = Conv2dFpropActivationIteratorOptimizedParams<Layout>;

  static int const kAccessesPerVector = ThreadMap::kElementsPerAccess / AccessType::kElements;

  static_assert(ThreadMap::Iterations::kContiguous == 1,
    "fprop first layer's activation iter's iter contiguous must be 1");

  static_assert(kAccessesPerVector == 1,
    "fprop first layer's activation iter's iter vector must be 1");

  static_assert(ThreadMap::Iterations::kStrided < 32,
    "fprop first layer's activation iter's iter stride must < 32");

private:

  Conv2dFpropActivationIteratorOptimizedParams<Layout> const &params_;
  Conv2dProblemSize const &problem_size_;
  LongIndex iteration_contiguous_;
  LongIndex iteration_strided_;
  LongIndex iteration_vector_;

  // One pointer per access
  char const *pointer_[ThreadMap::Iterations::kStrided];
  uint32_t mask_;

  // filter_r_'s max is r/FoldNum
  int filter_r_;
  int filter_s_;
  int filter_c_;

public:

  CUTLASS_HOST_DEVICE
  Conv2dFpropActivationTileAccessIteratorOptimizedSmallChannel(
    Conv2dFpropActivationIteratorOptimizedParams<Layout> const &params,
    Conv2dProblemSize const &problem_size,
    Element const *ptr,
    int thread_idx,
    MatrixCoord const &threadblock_offset = MatrixCoord()       // tile index - units are threadblock-scoped tiles
  ):
    params_(params), 
    problem_size_(problem_size),
    filter_c_(0),
    filter_r_(0), 
    filter_s_(0) {

    layout::PitchLinearCoord thread_coord = ThreadMap::initial_offset(thread_idx);

    // filter_c_ will pointer to next w when exceed
    filter_c_ = threadblock_offset.column() + thread_coord.contiguous();

    int offset_n[ThreadMap::Iterations::kStrided];
    int offset_p[ThreadMap::Iterations::kStrided];
    int offset_q[ThreadMap::Iterations::kStrided];

    // on which r position, should jump to start of next row when s exceed
    // here needn't consider dilation, update row/col position based on dilation later
    int row_offset = filter_c_ / problem_size_.C / problem_size_.S;
    // update c in current row, needn't consider s since will move to next s when c exceed
    filter_c_ = filter_c_ % (problem_size_.C * problem_size_.S);

    // update col position based on dilation
    if (problem_size_.dilation_w != 1) {
      int s_index = filter_c_ / problem_size_.C;
      filter_c_ += problem_size_.C * s_index * (problem_size_.dilation_w - 1);
    }

    mask_ = 0u;

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {

      pointer_[s] = reinterpret_cast<char const *>(ptr);
 
      int offset_npq = threadblock_offset.row() + thread_coord.strided() + s * ThreadMap::Delta::kStrided;

      int residual;

      params.pq_divmod(offset_n[s], residual, offset_npq);
      params.q_divmod(offset_p[s], offset_q[s], residual);

      // coord's hw is in padded coordinate
      TensorCoord coord = at_(offset_n[s], offset_p[s], offset_q[s], 0, 0, row_offset);

      pointer_[s] += params_.layout(coord) * sizeof_bits<Element>::value / 8;

      // only consider valid on vertical
      // right and bottom is already padded to fit new kernel size
      uint32_t pred = ((offset_n[s] < problem_size_.N) ? 1u : 0);
      mask_ |= (pred << s);

    }

    set_iteration_index(0);
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

  /// Returns the coordinate in the activations tensor X that is correspoinding to 
  // output npq and filter position r, s
  CUTLASS_HOST_DEVICE
  TensorCoord at_(int n, int p, int q, int r, int s, int row_offset) const {

    if (problem_size_.mode == Mode::kConvolution) {
      r = problem_size_.R - 1 - r;
      s = problem_size_.S - 1 - s;
    }

    // needn't minus padding, since input tensor has added padding already
    int h = p * problem_size_.stride_h + r * problem_size_.dilation_h;
    int w = q * problem_size_.stride_w + s * problem_size_.dilation_w;

    // update row offset on nhw coordinate, kernel window will not cross batch
    h += row_offset * problem_size_.dilation_h;

    return TensorCoord(n, h, w, filter_c_);
  }
  
  /// Adds a pointer offset in units of element
  CUTLASS_HOST_DEVICE
  void add_byte_offset_(LongIndex byte_offset) {

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      pointer_[s] += byte_offset;
    }
  }
  
public:

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(Index index) {
    iteration_vector_ = index % kAccessesPerVector;
    int residual_access = index / kAccessesPerVector;

    iteration_contiguous_ = residual_access % ThreadMap::Iterations::kContiguous;
    iteration_strided_ = residual_access / ThreadMap::Iterations::kContiguous;
  }

  /// Adds a pointer offset in units of element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    add_byte_offset_(pointer_offset * sizeof_bits<Element>::value / 8);
  }

  CUTLASS_HOST_DEVICE
  void advance() { 

    int next_idx = 0;
    // params_.inc_next[0] is already * fold_num
    int inc_next = params_.inc_next[0];

    // moves to the next tile
    ++filter_s_;
    // FoldNum is the step on S
    // FoldNum may bigger than S, this time always advance on R
    if (filter_s_ >= problem_size_.S / FoldNum) {
      filter_s_ = 0;
      ++filter_r_;

      if (FoldOnR) {
        // FoldNum / problem_size_.S is the step on R
        // FoldNum should be multiple of padded_s
        if (filter_r_ < problem_size_.R / (FoldNum / problem_size_.S)) {
          next_idx = 1;
          inc_next = params_.inc_next[1];
        }
        else {
          filter_r_ = 0;
          next_idx = 2;
          inc_next = params_.inc_next[2];
        }
      } else {
        // to run quicker when not fold on R
        if (filter_r_ < problem_size_.R) {
          next_idx = 1;
          inc_next = params_.inc_next[1];
        }
        else {
          filter_r_ = 0;
          next_idx = 2;
          inc_next = params_.inc_next[2];
        }

      }
    }
    
    add_byte_offset_(inc_next);

    // always become invalid when advance on channel in first layer implementation
    if (next_idx == 2) {  
      mask_ = 0u;
    }
    
  }

  CUTLASS_HOST_DEVICE
  bool valid() {
    // decrease performance slightly
    // always pass when return true directly?????
    return (mask_ & (1u << iteration_strided_));
  }

  /// Returns a pointer to the vector starting at the current coordinate
  CUTLASS_HOST_DEVICE
  AccessType const *get() const {
    return reinterpret_cast<AccessType const *>(pointer_[iteration_strided_]) + iteration_vector_;
  }

  /// Increments to the next memory access
  CUTLASS_HOST_DEVICE
  Conv2dFpropActivationTileAccessIteratorOptimizedSmallChannel &operator++() {

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

    // check alignment constraint on iterator's contiguous dimension
    // if (problem_size.C % AccessSize) {
    //   return Status::kErrorInvalidProblem;
    // }

    // advance multiple s/r each time
    if ((problem_size.R * problem_size.S) % FoldNum != 0) {
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
    // if (problem_size.R > 32 || problem_size.S > 32) {
    //   return Status::kErrorNotSupported;
    // }
    return Status::kSuccess;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
