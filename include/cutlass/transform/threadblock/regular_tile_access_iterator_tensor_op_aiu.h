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
    \brief Templates implementing computing the addresses of storing of tiles
   from pitch-linear rank=2 tensors.
*/

#pragma once

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor_op_multiplicand_ppu.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator.h"

// regular_tile_access_iterator_tensor_op.h will include this file, thus always goes here for fp16
// but can't do this for mma_tensor_op_tile_iterator.h, since the basic class is not declare in another file

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace transform {
namespace threadblock {

#if ACOMPUTE_VERSION == 10000

// aiu swizzle layout, only support fp16 now
template <typename Shape_, int AdvanceRank,
          typename ThreadMap_, int Alignment, int Crosswise>
class RegularTileAccessIterator<Shape_, cutlass::half_t,
                                layout::TensorOpMultiplicandCrosswise<
                                    sizeof_bits<cutlass::half_t>::value, Crosswise>,
                                AdvanceRank, ThreadMap_, Alignment> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = cutlass::half_t;
  using Layout =
      layout::TensorOpMultiplicandCrosswise<sizeof_bits<cutlass::half_t>::value,
                                            Crosswise>;
  static int const kAdvanceRank = AdvanceRank;
  static int const kAlignment = Alignment;
  static int const kCrosswise = Crosswise;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  static_assert(!(ThreadMap::Delta::kContiguous % kCrosswise),
                "kCrosswise is the smallest unit in the contiguous dimension "
                "for shared memory swizzling.");

  /// Internal details made public to facilitate introspection
  struct Detail {
    /// This iterator is specialized for an access size that is 128 bits in
    /// length.
    static int const kAccessSizeInBits = 128;

    static_assert(sizeof_bits<cutlass::half_t>::value *
                          ThreadMap::kElementsPerAccess ==
                      kAccessSizeInBits,
                  "This iterator requires a policy whose access size is 128bs");

    /// Number of pointers
    ///
    /// Note:TN kblock32 layouts only needs 1 pointer, but strangely
    /// reducing pointer count hurts perfomrnace
    static int const kPointerCount =
        (ThreadMap::Iterations::kStrided > 1 ? 2 : 1);
  };

  /// Element type per access
  using AccessType = Array<Element, Layout::kElementsPerAccess>;

  // each slice is 256b
  static int const ElementsPerSlice = 256 / sizeof_bits<Element>::value;

  static int const ElementsPerVector = Detail::kAccessSizeInBits / sizeof_bits<Element>::value;

  // always 4 row to fill one cache line, since one cache line is 4 slices
  static int const RowsPerLine = 4;

  // one matrix is always 4 cache lines
  static int const RowsPerMatrix = 4;

  // 4 rows in one slice will in same cache line, one load will cover WarpThreadArrangement::kStrided / 4 cache lines
  static int const RowsPerLoad = ThreadMap::Detail::WarpThreadArrangement::kStrided / 4;

  // tow successive vector is one section, should change place in even cache lines
  static int const VectorPerSection = 2;

  // tsm size for one stage, number of elements
  static int const StageSize = Shape::kStrided * Shape::kContiguous;

  static int const SliceSize = Shape::kStrided * ElementsPerSlice;

 private:
  //
  // Data members
  //

  /// Total number of sections.  The memory is divided into stages.  One stage
  /// can store one tile.  Stage is divided into sections.  Interleaved layout
  /// can have multiple sections in a stage.  The rest layout only has one section
  /// in a stage.
  int sections_;

  /// Sections that a stage has
  int sections_per_stage_;

  /// Stride value
  Index stride_;

  /// Internal pointer to first access of tile
  AccessType *pointer_[Detail::kPointerCount];

  /// Internal byte offset
  Index vector_offset_;

  /// Iteration in the contiguous dimension
  int iteration_contiguous_;

  /// Iteration in the strided dimension
  int iteration_strided_;

  AccessType *base_;

 public:
  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  RegularTileAccessIterator(TensorRef ref,  ///< Pointer to start of tensor
                            int thread_id   ///< ID of each participating thread
                            )
        // stride is block.k * stage
      : sections_(ref.stride(0) / kCrosswise),
        base_(reinterpret_cast<AccessType*>(ref.data())),
        sections_per_stage_(Shape::kContiguous / kCrosswise),
        // stride_ = kCrosswise x sections_ x kFactor
        stride_(ref.stride(0) * Layout::kFactor / Layout::kElementsPerAccess),
        vector_offset_(0) {

    // in element unit
    layout::PitchLinearCoord thread_offset_base =
        ThreadMap::initial_offset(thread_id);

    int slice_idx = thread_offset_base.contiguous() / ElementsPerSlice;
    int conti_idx = thread_offset_base.contiguous() % ElementsPerSlice;

    // successive B32x4 always interval 4 cache lines
    // each section is 512B, so section_idx always 0 in initial, needn't to get row_idx in section
    int row_idx = thread_offset_base.strided() / RowsPerLine;
    // updata conti_idx, each slice is ElementsPerSlice width
    conti_idx += (thread_offset_base.strided() % RowsPerLine) * ElementsPerSlice;

    // transfer to vector unit
    conti_idx /= ElementsPerVector;

    int section_idx = conti_idx / VectorPerSection;
    int idx_in_section = conti_idx % VectorPerSection;

    // swizzle for each slice
    // slice offset for first section is 0/2/1/3 for 4 slices
    int slice_swizzle_idx = ((slice_idx & 1) << 1) + ((slice_idx & 2) >> 1);
    section_idx = (section_idx + slice_swizzle_idx) % 4;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Detail::kPointerCount; ++i) {
      // each loop is one matrix
      row_idx += i * RowsPerLoad;

      // swizzle inner section for odd cache lines
      // each lane needs swizzle between differenti iteration_strided_ only for 4 slice
      // this time two pointer is enough since only two situation
      int idx_in_section_swizzle = idx_in_section ^ (row_idx % 2);

      // initialize pointer
      pointer_[i] = reinterpret_cast<AccessType *>(ref.data()) +
                    slice_idx * SliceSize / ElementsPerVector
                    // one row always 8 vector
                    + row_idx * 8
                    + (section_idx * VectorPerSection + idx_in_section_swizzle);
    }

    set_iteration_index(0);
  }

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    iteration_contiguous_ = index % ThreadMap::Iterations::kContiguous;
    iteration_strided_ = index / ThreadMap::Iterations::kContiguous;
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    vector_offset_ += pointer_offset / ElementsPerVector;
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType *get() const {
    AccessType *access_ptr = pointer_[iteration_strided_ & 1];
    int stride_idx = (iteration_strided_ & ~1);
    // each row always 8 vector
    int row_offset = stride_idx * RowsPerLoad * 8;

    // TODO: only support iteration_contiguous_ = 1 now
    // one loop always 512B, 32 vector
    return reinterpret_cast<AccessType *>(access_ptr + row_offset + vector_offset_);
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileAccessIterator &operator++() {
    ++iteration_contiguous_;

    if (iteration_contiguous_ < ThreadMap::Iterations::kContiguous)
      return *this;

    // Enter here only if (iteration_contiguous_ ==
    // ThreadMap::Iteration::kContiguous)
    iteration_contiguous_ = 0;
    ++iteration_strided_;

    if (iteration_strided_ < ThreadMap::Iterations::kStrided) {
      return *this;
    }

    // Enter here only if (iteration_strided_ == ThreadMap::Iteration::kStrided)
    // which means we enter the next section.
    iteration_strided_ = 0;

    return *this;
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    add_pointer_offset(coord.contiguous() * StageSize);
  }
};

#endif

////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace transform
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
