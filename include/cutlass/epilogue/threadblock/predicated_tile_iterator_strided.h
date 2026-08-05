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
  \brief Epilogue for threadblock scoped GEMMs using Tensor Ops.

  The epilogue rearranges the result of a matrix product through shared memory to match canonical
  tensor layouts in global memory. Epilogues support conversion and reduction operations.

*/

#if SAIL_DGRAD_STRIDE_OPT

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/epilogue/threadblock/output_tile_thread_map.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/memory.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator_params.h"
#include "cutlass/conv/conv2d_problem_size.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////

namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Tile iterator used to load and store output tile from shared memory in epilogue.
///
/// Satisfies: ReadableTileIterator | PredicatedTileIterator | ForwardTileIterator
///
template <
  typename ThreadMap_,       ///< Thread map (conept: OutputTileThreadMap)
  typename Element_          ///< Element data type
>
class PredicatedTileIteratorStrided {
public:
  using ThreadMap = ThreadMap_;
  using Shape = typename ThreadMap::Shape;

  using Element = Element_; // output data type

  using Layout = layout::RowMajor;
  using TensorRef = TensorRef<Element, Layout>;
  using ConstTensorRef = typename TensorRef::ConstTensorRef;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using TensorCoord = MatrixCoord;
  using ConvProblemSize = typename cutlass::conv::Conv2dProblemSize;

  static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
  static int const kThreads = ThreadMap::kThreads;
  static int const kIterations = ThreadMap::Count::kTile;

  static_assert( ThreadMap::Iterations::kRow > 0,"ThreadMap::Iterations::kRow must be > 0");
  static_assert( ThreadMap::Iterations::kGroup > 0,"ThreadMap::Iterations::kGroup must be > 0");
  static_assert( ThreadMap::Iterations::kCluster > 0,"ThreadMap::Iterations::kCluster must be > 0");
  static_assert( ThreadMap::Iterations::kColumn > 0,"ThreadMap::Iterations::kColumn must be > 0");

  /// Fragment object
  using Fragment = Array<
    Element,
    ThreadMap::Iterations::kColumn *
    ThreadMap::Iterations::kRow *
    ThreadMap::Iterations::kGroup *
    ThreadMap::Iterations::kCluster * ThreadMap::kElementsPerAccess>;

  /// Memory access size
  using AccessType = AlignedArray<Element, ThreadMap::kElementsPerAccess>;

  //
  // Parameters struct
  //

  /// Uses a non-template class
  struct Params : PredicatedTileIteratorParams {

    CUTLASS_HOST_DEVICE
    Params() { }

    CUTLASS_HOST_DEVICE
    Params(Layout const &layout):
      PredicatedTileIteratorParams(
        LongIndex(layout.stride(0)) * int(sizeof(AccessType)) / kElementsPerAccess,
        make_OutputTileThreadMapDesc<ThreadMap>()
      )
    {

    }
  };

  /// Mask object
  struct Mask {

    static int const kCount = ThreadMap::Iterations::kColumn;

    /// Predicate state
    bool predicates[kCount];

    //
    // Mask
    //
    CUTLASS_HOST_DEVICE
    Mask() {
      enable();
    }

    ///< Efficiently disables all accesses guarded by mask
    CUTLASS_HOST_DEVICE void clear() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) {
        predicates[i] = false;
      }
    }

    ///< CUTLASS_HOST_DEVICE enables all accesses guarded by mask
    CUTLASS_DEVICE void enable() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) {
        predicates[i] = true;
      }
    }
  };

private:

  //
  // Data members
  //

  /// Parameters structure containing reference and precomputed state.
  PredicatedTileIteratorParams params_;

  /// Byte-level pointer
  uint8_t *byte_pointer_;

  /// Array of boolean values to contain steady-state predicates
  Mask mask_;

  /// Extent of the matrix tile in rows
  Index extent_row_;

  /// A thread's starting row position (assuming steady-state predicates have been computed)
  Index thread_start_row_;

  /// Internal state counter
  int state_[3];

  int offset_nhw_cur_rs_ = 0;
  int offset_nhw_ori_ = 0;
  int valid_h_ = 1;
  int valid_w_ = 1;
  int start_h_ = 0;
  int start_w_ = 0;
  int h_ = 1;
  int w_ = 1;
  Element *g_ptr_;


  int rs_idx_;

private:

  //
  // Methods
  //

public:

  //
  // Methods
  //

  /// Constructor

  #if SAIL_FUSE_OP_EXT
  CUTLASS_DEVICE
  PredicatedTileIteratorStrided() {
  }
  #endif

  CUTLASS_DEVICE
  PredicatedTileIteratorStrided(
    PredicatedTileIteratorParams const & params,
    Element *pointer,
    TensorCoord extent,
    int thread_idx,
    TensorCoord threadblock_offset = TensorCoord(),
    Coord<3> transpose_shape = Coord<3>(),
    TensorCoord group_shape = TensorCoord(),
    int groups = 1
  ):
    params_(params)
  {
    g_ptr_ = pointer;
    // move constructor into initial to maintain ori api
  }

  CUTLASS_HOST_DEVICE
  void initial(
    PredicatedTileIteratorParams const & params,
    ConvProblemSize const &problem_size,
    Element *pointer,
    TensorCoord extent,
    int thread_idx,
    TensorCoord threadblock_offset = TensorCoord()
  ) {

    clear_mask();

    // Shape is 8x128, use Shape::kColumn which is blockshape.n and assume it is identical to blockshape.m
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
      // cur block is invalid, return with mask = 0
      // can't return outside since only dgrad strided has this param valid
      // set valid_w_ = 1 to avoid /0 in ++
      valid_w_ = 1;
      valid_h_ = 1;
      return;
    }

    // t0-t7: 0/0 -- 0/56
    // t32-t39: 4/0 -- 4/56
    // t64-t71: 64/0 -- 64/56
    // t96-t103: 68/0 -- 68/56
    // block.x=1 starts at row 128
    // position in output matrix, already consider threadblock_offset,
    TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx) + threadblock_offset;

    h_ = problem_size.H;
    w_ = problem_size.W;
    valid_w_ = problem_size.valid_w[rs_idx_];
    valid_h_ = problem_size.valid_h[rs_idx_];
    start_h_ = problem_size.start_h[rs_idx_];
    start_w_ = problem_size.start_w[rs_idx_];
    // get vertical offset of each thread
    offset_nhw_cur_rs_ = block_rs_idx * problem_size.block_shape_m + ThreadMap::initial_offset(thread_idx).row();
    int cur_n = offset_nhw_cur_rs_ / (valid_h_ * valid_w_);
    int residual = offset_nhw_cur_rs_ % (valid_h_ * valid_w_);

    // index in elements of current valid rs position
    int cur_h = residual / valid_w_;
    int cur_w = residual % valid_w_;
    // transfer to ori input coord
    cur_h *= 2;
    cur_w *= 2;
    cur_h += start_h_;
    cur_w += start_w_;
    // offset on nhw, needn't to multiply c
    offset_nhw_ori_ = cur_n * problem_size.H * problem_size.W + cur_h * problem_size.W + cur_w;

    // nhw size
    extent_row_ = extent.row();
    thread_start_row_ = offset_nhw_ori_;

    // Initialize predicates
    CUTLASS_PRAGMA_UNROLL
    // 2
    for (int c = 0; c < ThreadMap::Iterations::kColumn; ++c) {
      // is column out of boundary
      mask_.predicates[c] = ((thread_offset.column()
        + ThreadMap::Delta::kColumn * c) < extent.column());  // ThreadMap::Delta::kColumn is 64
    }

    // Initialize pointer
    // byte_pointer_ is uint8*, thread_offset is element coordinate, * element_bytes
    byte_pointer_ = reinterpret_cast<uint8_t *>(pointer) +
      LongIndex(offset_nhw_ori_) * LongIndex(params_.stride) +
      // sizeof(AccessType) / kElementsPerAccess is bytes of element
      LongIndex(thread_offset.column()) * sizeof(AccessType) / kElementsPerAccess;  // AccessType is 128b, kElementsPerAccess = 8

    // Initialize internal state counter
    state_[0] = state_[1] = state_[2] = 0;
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_byte_offset(Fragment &frag, int64_t byte_offset) {

    uint8_t *byte_pointer = byte_pointer_;
    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {

      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {

          int frag_row_idx =
            (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

          int row_offset = row * ThreadMap::Delta::kRow
            + group * ThreadMap::Delta::kGroup
            + cluster * ThreadMap::Delta::kCluster;

          bool row_guard = ((row_offset + thread_start_row_) < extent_row_);

          AccessType *memory_pointer = reinterpret_cast<AccessType *>(byte_pointer + byte_offset);

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {

            bool guard = row_guard && mask_.predicates[column];

            cutlass::arch::global_load<
              AccessType,
              sizeof(AccessType)
            >(
                frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn +
                         column],
                (void *)&memory_pointer[column * ThreadMap::Delta::kColumn /
                                        kElementsPerAccess],
                guard);
          }

          if (row + 1 < ThreadMap::Iterations::kRow) {
            byte_pointer += params_.increment_row;
          }
        }

        if (group + 1 < ThreadMap::Iterations::kGroup) {
          byte_pointer += params_.increment_group;
        }
      }

      if (cluster + 1 < ThreadMap::Iterations::kCluster) {
        byte_pointer += params_.increment_cluster;
      }
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) {
      load_with_byte_offset(frag, 0);

  }

  /// Stores a fragment to memory
  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const &frag, int64_t byte_offset) {
    uint8_t *byte_pointer = byte_pointer_;
    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);
    int offset_nhw_ori_old = offset_nhw_ori_;

    CUTLASS_PRAGMA_UNROLL
    // 1
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {

      CUTLASS_PRAGMA_UNROLL
      // 1
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {

        CUTLASS_PRAGMA_UNROLL
        // 1
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {

          int frag_row_idx =
            (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

          // can't use original logic since row_offset isn't identical to nhw_offset
          bool row_guard = offset_nhw_ori_old < extent_row_;

          AccessType *memory_pointer = reinterpret_cast<AccessType *>(byte_pointer + byte_offset);

          CUTLASS_PRAGMA_UNROLL
          // 2, loop on hori, 8 elements in right 64
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {

            bool guard = row_guard && mask_.predicates[column];

            if (guard) {
              // column * 64 / 8, move to right 64
              memory_pointer[column * ThreadMap::Delta::kColumn / kElementsPerAccess] =
                // 0 * 2 + column, move to next vector in reg
                frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column];
            }
          }

          // three if is invalid now
          if (row + 1 < ThreadMap::Iterations::kRow) {
            // move 4 row in cur rs position
            int offset_nhw_cur_rs = offset_nhw_cur_rs_ + (row + 1) * ThreadMap::Delta::kRow;
            int cur_n = offset_nhw_cur_rs / (valid_h_ * valid_w_);
            int residual = offset_nhw_cur_rs % (valid_h_ * valid_w_);

            // index in elements of current valid rs position
            int cur_h = residual / valid_w_;
            int cur_w = residual % valid_w_;
            // transfer to ori input coord
            cur_h *= 2;
            cur_w *= 2;
            cur_h += start_h_;
            cur_w += start_w_;
            // offset on nhw, needn't to multiply c
            int offset_nhw_ori_new = cur_n * h_ * w_ + cur_h * w_ + cur_w;
            byte_pointer += LongIndex(offset_nhw_ori_new - offset_nhw_ori_old) * LongIndex(params_.stride);
            offset_nhw_ori_old = offset_nhw_ori_new;
          }
        }

        if (group + 1 < ThreadMap::Iterations::kGroup) {
          byte_pointer += params_.increment_group;
        }
      }

      if (cluster + 1 < ThreadMap::Iterations::kCluster) {
        byte_pointer += params_.increment_cluster;
      }
    }
  }

  /// Stores a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) {
      store_with_byte_offset(frag, 0);

  }

  /// Advances to the next position to load or store
  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorStrided &operator++() {


    ++state_[0];
    // move to next 8 row
    // offset in elements in cur rh is 8, calc offset in ori hw
    offset_nhw_cur_rs_ += ThreadMap::Shape::kRow;
    int cur_n = offset_nhw_cur_rs_ / (valid_h_ * valid_w_);
    int residual = offset_nhw_cur_rs_ % (valid_h_ * valid_w_);

    // index in elements of current valid rs position
    int cur_h = residual / valid_w_;
    int cur_w = residual % valid_w_;
    // transfer to ori input coord
    cur_h *= 2;
    cur_w *= 2;
    cur_h += start_h_;
    cur_w += start_w_;
    // offset on nhw, needn't to multiply c
    int offset_nhw_ori_new = cur_n * h_ * w_ + cur_h * w_ + cur_w;
    byte_pointer_ += LongIndex(offset_nhw_ori_new - offset_nhw_ori_) * LongIndex(params_.stride);
    thread_start_row_ = offset_nhw_ori_new;
    offset_nhw_ori_ = offset_nhw_ori_new;

    if (state_[0] == ThreadMap::Count::kRow) {  // 8

      state_[0] = 0;
      ++state_[1];
      byte_pointer_ += params_.advance_group; // 0

      thread_start_row_ += (ThreadMap::Shape::kGroup - 1) *
        ThreadMap::Shape::kRow * ThreadMap::Count::kRow;

      if (state_[1] == ThreadMap::Count::kGroup) {  // 1

        state_[1] = 0;
        ++state_[2];
        // move to next 64 lines in output
        // byte_pointer_ never use after this, only 1 group now
        byte_pointer_ += params_.advance_cluster; // 8192

        thread_start_row_ += ThreadMap::Count::kGroup *
          ThreadMap::Shape::kGroup * ThreadMap::Count::kRow * ThreadMap::Shape::kRow;

        if (state_[2] == ThreadMap::Count::kCluster) {  // 1
          state_[2] = 0;
          byte_pointer_ += params_.advance_tile;  // 0
        }
      }
    }

    return *this;
  }

  ///< Efficiently disables all accesses guarded by mask
  CUTLASS_DEVICE void clear_mask() {
    mask_.clear();
  }

  ///< Efficiently enables all accesses guarded by mask
  CUTLASS_DEVICE void enable_mask() {
    mask_.enable();
  }

  ///< Sets the mask
  CUTLASS_DEVICE void get_mask(Mask &mask) {
    return mask_;
  }

  ///< Sets the mask
  CUTLASS_DEVICE void set_mask(Mask const &mask) {
    mask_ = mask;
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

#endif

////////////////////////////////////////////////////////////////////////////////
