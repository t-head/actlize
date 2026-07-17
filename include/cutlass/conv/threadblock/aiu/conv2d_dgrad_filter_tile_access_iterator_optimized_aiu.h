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
class Conv2dDgradFilterTileAccessIteratorOptimizedAiu {
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

  // shape is K/N for B input
  static constexpr int SwzlMode = Shape::kRow * sizeof(Element) == 128 ? 0 : 1;

  static int const kAccessesPerVector = ThreadMap::kElementsPerAccess / AccessType::kElements;

  // 128B per aiu load on contiguous dimension
  static int const CubeW = Shape::kColumn > 128 / sizeof(Element) ? 128 / sizeof(Element) : Shape::kColumn;

  static int const AiuLoadIter = Shape::kColumn / CubeW;

  // static int const CubeC = 128 * 8 / sizeof_bits<Element>::value;

  // Shape::kRow is on gemm.k
  static int const CubeStride = CubeW * Shape::kRow;

  static int const ElemInCube = 128 * 8 / sizeof_bits<Element>::value;

  //
  // Parameters structure
  //

  struct Params : Conv2dDgradFilterIteratorOptimizedParams {

    //
    // Methods
    //
    CUTLASS_HOST_DEVICE
    Params() { }

    CUTLASS_HOST_DEVICE
    Params(Conv2dDgradFilterIteratorOptimizedParams const &base):
      Conv2dDgradFilterIteratorOptimizedParams(base) { }

    CUTLASS_HOST_DEVICE
    Params(
      Conv2dProblemSize const &problem_size,
      Layout const &layout
    ):
      Conv2dDgradFilterIteratorOptimizedParams(
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

  Conv2dDgradFilterIteratorOptimizedParams const &params_;
  Conv2dProblemSize const &problem_size_;

  int filter_rs_;
  int filter_k_;
  int filter_c_[AiuLoadIter];
  uint8_t const *global_ptr_;

  int warp_idx_;

public:

  CUTLASS_HOST_DEVICE
  Conv2dDgradFilterTileAccessIteratorOptimizedAiu(
    Conv2dDgradFilterIteratorOptimizedParams const &params,
    Conv2dProblemSize const &problem_size,
    Element const *ptr,
    int thread_idx,
    MatrixCoord const &threadblock_offset = MatrixCoord()
  ):
    params_(params),
    problem_size_(problem_size),
    global_ptr_(reinterpret_cast<const uint8_t*>(ptr)),
    filter_rs_(0),
    filter_k_(0) {

    warp_idx_ = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

    filter_k_ = threadblock_offset.row();

    CUTLASS_PRAGMA_UNROLL
    for(int c = 0; c < AiuLoadIter; ++c) {
      filter_c_[c] = threadblock_offset.column() + c * ElemInCube;
    }

  }

  CUTLASS_HOST_DEVICE
  void advance() {


    // moves to the next tile
    ++filter_rs_;
    if (filter_rs_ == params_.RS) {

      filter_rs_ = 0;
      filter_k_ += params_.filter_k_delta;
    }

  }

  CUTLASS_HOST_DEVICE
  void aiu_load(Element *tsm_ptr) const {

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < AiuLoadIter; i++) {
      int offset_rsc = filter_rs_ * problem_size_.C + filter_c_[i];
      int tensor_dim_c = min(problem_size_.R * problem_size_.S, filter_rs_ + 1) * problem_size_.C;

      // cutlass::arch::tsm_load_cube_sim(
      //   global_ptr_, tsm_ptr + i * CubeStride,
      //   1, 1, 1, problem_size_.K, tensor_dim_c,
      //   problem_size_.R * problem_size_.S * problem_size_.C, 1,
      //   0, 0, 0, filter_k_, offset_rsc,
      //   Shape::kRow, CubeC,
      //   0, 0, 0,
      //   1, 1, 1,
      //   1, 1, 1, 0, 0, 0,
      //   1, 1, 1
      // );

      if (warp_idx_ == 0) {
        // Shape::kColumn is CubeH
        CONV_AIU_LOAD_2D_IMPL<Element, Shape::kRow, CubeW>()(
          tsm_ptr + i * CubeStride, global_ptr_,
          tensor_dim_c, problem_size_.K,
          params_.layout.stride()[2] * sizeof(Element), params_.layout.stride()[2] * sizeof(Element),
          offset_rsc, filter_k_
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


