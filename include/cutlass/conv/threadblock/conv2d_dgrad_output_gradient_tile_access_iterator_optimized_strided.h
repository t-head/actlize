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

#if SAIL_DGRAD_STRIDE_OPT
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
  typename ThreadMap_,
  conv::StrideSupport StrideSupport_ = conv::StrideSupport::kStrided,
  int AccessSize = ThreadMap_::kElementsPerAccess,
  conv::KernelType KernelType_ = conv::KernelType::kNormal
>
class Conv2dDgradOutputGradientTileAccessIteratorOptimizedStrided {
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
  static conv::KernelType const kType = KernelType_;

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

  // One pointer per access
  char const *pointer_[ThreadMap::Iterations::kStrided];

  // current filter position (r, s)
  int filter_r_ = 0;
  int filter_s_ = 0;
  int filter_k_ = 0;

  int loop_num_ = 0;
  int start_r_ = 0;
  int start_s_ = 0;
  int rs_idx_ = 0;
  int problem_k_ = 1;

  Index masks_[ThreadMap::Iterations::kStrided][kAccessesPerVector][2];

public:

  CUTLASS_HOST_DEVICE
  Conv2dDgradOutputGradientTileAccessIteratorOptimizedStrided(
    Conv2dDgradOutputGradientIteratorOptimizedParamsStrided const &params,
    Conv2dProblemSize const &problem_size,
    Element const *ptr,
    int thread_idx,
    MatrixCoord const &threadblock_offset = MatrixCoord()       // tile index - units are threadblock-scoped tiles
  ):
    params_(params),
    problem_size_(problem_size),
    filter_k_(0),
    filter_r_(0),
    filter_s_(0) {

    clear_mask();

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
    } else if (blockIdx.y < problem_size.block_num[3]) {
      rs_idx_ = 3;
      block_rs_idx = blockIdx.y - problem_size.block_num[2];
    } else {
      CUTLASS_PRAGMA_UNROLL
      for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
        pointer_[s] = reinterpret_cast<char const *>(ptr);
      }
      // cur block is invalid, return with mask = 0
      // can't return outside since only dgrad strided has this param valid
      return;
    }

    filter_r_ = problem_size.start_r_[rs_idx_];
    filter_s_ = problem_size.start_s_[rs_idx_];
    start_r_ = filter_r_;
    start_s_ = filter_s_;

    // should supply on k dimension only, because k loop is independent to rs loop
    // the complementary k loop should run for all rs loop
    loop_num_ = problem_size.loop_r[rs_idx_] * problem_size.loop_s[rs_idx_] * (problem_size.K + Shape::kColumn - 1) / Shape::kColumn;

    if (blockIdx.y >= problem_size.block_num[3]) {
      loop_num_ = 0;
    }

    layout::PitchLinearCoord thread_coord = ThreadMap::initial_offset(thread_idx);

    filter_k_ = threadblock_offset.column() + thread_coord.contiguous();
    problem_k_ = kType == conv::KernelType::kMultipleGroup ? Shape::kColumn + filter_k_ : problem_size.K;


    int offset_n[ThreadMap::Iterations::kStrided];
    int offset_h[ThreadMap::Iterations::kStrided];
    int offset_w[ThreadMap::Iterations::kStrided];

    int r_pos = filter_r_;
    int s_pos = filter_s_;

    if (problem_size_.mode == Mode::kConvolution) {
      r_pos = (problem_size_.R - 1 - r_pos);
      s_pos = (problem_size_.S - 1 - s_pos);
    }

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {

      pointer_[s] = reinterpret_cast<char const *>(ptr);

      // offset in all elements in current rs
      int offset_nhw = block_rs_idx * Shape::kRow + thread_coord.strided() + s * ThreadMap::Delta::kStrided;

      offset_n[s] = offset_nhw / (problem_size.valid_h[rs_idx_] * problem_size.valid_w[rs_idx_]);
      int residual = offset_nhw % (problem_size.valid_h[rs_idx_] * problem_size.valid_w[rs_idx_]);

      // index in elements of current valid rs position
      offset_h[s] = residual / problem_size.valid_w[rs_idx_];
      offset_w[s] = residual % problem_size.valid_w[rs_idx_];
      // transfer to ori input coord
      offset_h[s] *= 2;
      offset_w[s] *= 2;
      offset_h[s] += problem_size.start_h[rs_idx_];
      offset_w[s] += problem_size.start_w[rs_idx_];
      // always no remainder since rs is already in vaild position
      int cur_p = (offset_h[s] + problem_size_.pad_h - r_pos * problem_size_.dilation_h) / problem_size.stride_h;
      int cur_q = (offset_w[s] + problem_size_.pad_w - s_pos * problem_size_.dilation_w) / problem_size.stride_w;
      TensorCoord coord = TensorCoord(offset_n[s], cur_p, cur_q, filter_k_);

      pointer_[s] += params_.layout(coord) * sizeof_bits<Element>::value / 8;
    }

    // rs add 2 in each loop
    CUTLASS_PRAGMA_NO_UNROLL
    for (int r = problem_size.start_r_[rs_idx_]; r < problem_size_.R; r += 2) {
      CUTLASS_PRAGMA_UNROLL
      for (int s_idx = 0; s_idx < ThreadMap::Iterations::kStrided; ++s_idx) {
        int cur_r = r;
        if (problem_size_.mode == Mode::kConvolution) {
          cur_r = (problem_size_.R - 1 - r);
        }

        int p = (offset_h[s_idx] + problem_size_.pad_h - cur_r * problem_size_.dilation_h) / problem_size.stride_h;

        bool pred = (offset_n[s_idx] < problem_size_.N && p >= 0 && p < problem_size_.P);
        // masks_[loop][r or s], one bit for each r/s position
        CUTLASS_PRAGMA_UNROLL
        for (int v_idx = 0; v_idx < kAccessesPerVector; ++v_idx) {
          masks_[s_idx][v_idx][0] |= (pred << r);
        }
      }
    }

    CUTLASS_PRAGMA_NO_UNROLL
    for (int s = problem_size.start_s_[rs_idx_]; s < problem_size_.S; s += 2) {
      CUTLASS_PRAGMA_UNROLL
      for (int s_idx = 0; s_idx < ThreadMap::Iterations::kStrided; ++s_idx) {
        int cur_s = s;
        if (problem_size_.mode == Mode::kConvolution) {
          cur_s = (problem_size_.S - 1 - s);
        }

        int q = (offset_w[s_idx] + problem_size_.pad_w - cur_s * problem_size_.dilation_w) / problem_size.stride_w;

        bool pred = (q >= 0 && q < problem_size_.Q);
        CUTLASS_PRAGMA_UNROLL
        for (int v_idx = 0; v_idx < kAccessesPerVector; ++v_idx) {
          masks_[s_idx][v_idx][1] |= (pred << s);
        }
      }
    }

    CUTLASS_PRAGMA_UNROLL
    for (int v_idx = 0; v_idx < kAccessesPerVector; ++v_idx) {
      clear_mask_(filter_k_ + v_idx * AccessSize >= problem_k_, v_idx);
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

  /// Returns the coordinate in the output gradient tensor dy that is correspoinding to
  // output nhw and filter position k, r, s
  CUTLASS_HOST_DEVICE
  TensorCoord at_(int n, int h, int w, int r, int s) const {

    if (problem_size_.mode == Mode::kConvolution) {
      r = problem_size_.R - 1 - r;
      s = problem_size_.S - 1 - s;
    }

    int p = h + problem_size_.pad_h - r * problem_size_.dilation_h;
    int q = w + problem_size_.pad_w - s * problem_size_.dilation_w;

    return TensorCoord(n, p, q, filter_k_);
  }

  /// Adds a pointer offset in units of element
  CUTLASS_HOST_DEVICE
  void add_byte_offset_(LongIndex byte_offset) {

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      pointer_[s] += byte_offset;
    }
  }

  /// Clears the predicates
  CUTLASS_HOST_DEVICE
  void clear_mask_(bool clear, int index) {
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {

      // We are using inline device assembly assembly here to avoid an device C++ compilation
      // artifact in which control flow instructions are generated. Instead, our
      // intent is to predicate the mov instructions.
      #if defined(__HGGC_ARCH__)
      asm volatile(
          "{\n"
          "  .reg .pred p;\n"
          "  .reg .u32  m;"
          "  ppu.mov.u32 m, %2;"
          "  ppu.cmpp.ne.b32 p, %1, 0;\n"
          "  @p ppu.mov.u32 m, 0;\n"
          "  ppu.mov.u32 %0, m;\n"
          "}\n"
        :
          "=r"(masks_[s][index][0])
       :
          "r"((int)clear),
          "r"(masks_[s][index][0])
      );
      asm volatile(
          "{\n"
          "  .reg .pred p;\n"
          "  .reg .u32  m;"
          "  ppu.mov.u32 m, %2;"
          "  ppu.cmpp.ne.b32 p, %1, 0;\n"
          "  @p ppu.mov.u32 m, 0;\n"
          "  ppu.mov.u32 %0, m;\n"
          "}\n"
        :
          "=r"(masks_[s][index][1])
       :
          "r"((int)clear),
          "r"(masks_[s][index][1])
      );
      #else
        if (clear) {
          masks_[s][index][0] = 0;
          masks_[s][index][1] = 0;
        }
      #endif
    }
  }

public:
  CUTLASS_HOST_DEVICE
  int get_loop_num() {
    return loop_num_;
  }

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
#if __HGGCCC_RTC__
    int inc_next;
    if (rs_idx_ == 0) {
      inc_next = params_.inc_next[0][0];
    } else if (rs_idx_ == 1) {
      inc_next = params_.inc_next[0][1];
    } else if (rs_idx_ == 2) {
      inc_next = params_.inc_next[0][2];
    } else {
      inc_next = params_.inc_next[0][3];
    }
#elif SAIL_TMP_WORKAROUND
    int inc_next = params_.inc_next[0][rs_idx_];
#endif

    // moves to the next tile
    // TODO: only support stride = 2 now
    filter_s_ += 2;
    if (filter_s_ >= problem_size_.S) {
      filter_s_ = start_s_;
      filter_r_ += 2;

      if (filter_r_ < problem_size_.R) {
        next_idx = 1;
#if __HGGCCC_RTC__
        if (rs_idx_ == 0) {
          inc_next = params_.inc_next[1][0];
        } else if (rs_idx_ == 1) {
          inc_next = params_.inc_next[1][1];
        } else if (rs_idx_ == 2) {
          inc_next = params_.inc_next[1][2];
        } else {
          inc_next = params_.inc_next[1][3];
        }
#elif SAIL_TMP_WORKAROUND
        inc_next = params_.inc_next[1][rs_idx_];
#endif
      }
      else {
        filter_r_ = start_r_;
        next_idx = 2;
#if __HGGCCC_RTC__
        if (rs_idx_ == 0) {
          inc_next = params_.inc_next[2][0];
        } else if (rs_idx_ == 1) {
          inc_next = params_.inc_next[2][1];
        } else if (rs_idx_ == 2) {
          inc_next = params_.inc_next[2][2];
        } else {
          inc_next = params_.inc_next[2][3];
        }
#elif SAIL_TMP_WORKAROUND
        inc_next = params_.inc_next[2][rs_idx_];
#endif
      }
    }

    // inc_next already consider stride
#if SAIL_TMP_WORKAROUND
    add_byte_offset_(inc_next);
#else
    add_byte_offset_(params_.inc_next[next_idx][rs_idx_]);
#endif

    if (next_idx == 2) {
      filter_k_ += params_.filter_k_delta;
    }

    CUTLASS_PRAGMA_UNROLL
    for (int v_idx = 0; v_idx < kAccessesPerVector; ++v_idx) {
      clear_mask_(filter_k_ + v_idx * AccessSize >= problem_k_, v_idx);
    }
  }

  /// Clears the predicates
  CUTLASS_HOST_DEVICE
  void clear_mask() {
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < kAccessesPerVector; ++v) {
        masks_[s][v][0] = Mask(0);
        masks_[s][v][1] = Mask(0);
      }
    }
  }

  CUTLASS_HOST_DEVICE
  bool valid() {
    return
      (masks_[iteration_strided_][iteration_vector_][0] & (Index(1) << filter_r_)) &&
      (masks_[iteration_strided_][iteration_vector_][1] & (Index(1) << filter_s_));
  }

  /// Returns a pointer to the vector starting at the current coordinate
  CUTLASS_HOST_DEVICE
  AccessType const *get() const {

    return reinterpret_cast<AccessType const *>(pointer_[iteration_strided_]) + iteration_vector_;
  }

  /// Increments to the next memory access
  CUTLASS_HOST_DEVICE
  Conv2dDgradOutputGradientTileAccessIteratorOptimizedStrided &operator++() {

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

#endif // #ifdef SAIL_CUSTOMIZE_CUTLASS

/////////////////////////////////////////////////////////////////////////////////////////////////


