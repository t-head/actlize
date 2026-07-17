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
    \brief This defines a "fragment" iterator for visiting the fragments of an accumulator tile
      that participate in one warp-level store operation.

      Typically, the accumulator tile is the largest single block of register-backed storage 
      within the kernel. Storing it to memory is best accomplished by partitioning it into
      smaller tiles and storing these sequentially.

      Round trips through shared memory during the Epilogue phase require partitioning, as
      shared memory capacity is typically insufficient for a threadblock's total accumulator
      size.
*/

#pragma once

#include "cutlass/array.h"
#include "cutlass/layout/matrix.h"

#include "cutlass/epilogue/warp/simt_policy.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Fragment iterator for SIMT accumulator arrangements
template <
  typename WarpShape,             ///< shape of warp-level GEMM (concept: MatrixShape)
  typename Operator,              ///< matrix multiply operation (concept: arch::Mma)
  typename Layout,                ///< target shared memory layout
  typename MmaSimtPolicy          ///< policy defining lane arrangement (concept: MmaSimtPolicy)
>
class FragmentIteratorSimt;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for row-major shared memory
template <
  typename WarpShape_,     ///< shape of the warp-level GEMM tile
  typename Operator_ ,     ///< matrix multiply operator (concept: arch::Mma)
  typename MmaSimtPolicy_  ///< policy defining lane arrangement (concept: MmaSimtPolicy)
>
class FragmentIteratorSimt<WarpShape_, Operator_, layout::RowMajor, MmaSimtPolicy_> {
public:

  using WarpShape = WarpShape_;
  using Operator = Operator_;
  using Layout = layout::RowMajor;

  /// Policy for warp-level epilogue components
  using Policy = SimtPolicy<WarpShape, Operator, Layout, MmaSimtPolicy_>;

  /// This is the fragment size produced by one access of the iterator.
  using Fragment = Array<
    typename Operator::ElementC, 
    Policy::kElementsPerIteration>;

  /// This is the complete warp-level accumulator tile.
  using AccumulatorTile = Array<
    typename Operator::ElementC, 
    Policy::kAccumulatorElementCount>;

  using OutputAccumulatorTile = AccumulatorTile;

  /// Number of times this iterator can be incremented
  static int const kIterations = Policy::kIterations;

private:

  /// Internal access type
  using AccessType = Array<typename Operator::ElementC, Policy::kElementsPerAccess>;

private:

  //
  // Data members
  //

  /// Accumulator tile
  AccessType const *accumulators_;

  /// Internal index
  int index_;

public:

  /// Constructs an iterator
  CUTLASS_HOST_DEVICE
  FragmentIteratorSimt(AccumulatorTile const &accum): 
    accumulators_(reinterpret_cast<AccessType const *>(&accum)), 
    index_(0) {

  }

  /// Increments
  CUTLASS_HOST_DEVICE
  FragmentIteratorSimt &operator++() {
    ++index_;
    return *this;
  }

  /// Decrements
  CUTLASS_HOST_DEVICE
  FragmentIteratorSimt &operator--() {
    --index_;
    return *this;
  }

  /// Loads a fragment from the referenced part of the accumulator tile
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag, int index_offset = 0) const {

    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < Policy::kAccessesPerIteration; ++n) {

      int accumulator_access_offset = index_ * Policy::kAccessesPerIteration + n;

      frag_ptr[n] = accumulators_[accumulator_access_offset];
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Template for reading and writing tiles of accumulators to shared memory
template <typename WarpShape_,               ///< shape of warp-level GEMM (concept: GemmShape)
          typename ThreadOutputShape_,       /// Size of the matrix to load (concept: TensorNHWC)
          typename ThreadBlockOutputShape_,  /// Size of the matrix to load (concept: TensorNHWC)
          typename Operator_,                ///< matrix multi ply operation (concept: arch::Mma)
          typename Element_,                 ///< data type of element to be written
          typename Layout_,                  ///< target shared memory layout
          typename MmaSimtPolicy_            ///< policy defining lane arrangement (concept: MmaSimtPolicy)
          >
class TileIteratorSimtDirect2dConv {
 public:
  using WarpShape = WarpShape_;
  using ThreadOutputShape = ThreadOutputShape_;
  using ThreadBlockOutputShape = ThreadBlockOutputShape_;
  using Operator = Operator_;
  using Element = Element_;
  using Layout = layout::RowMajor;
  using MmaSimtPolicy = MmaSimtPolicy_;

  using TensorRef = TensorRef<Element, Layout>;  ///< Tensor Reference object
  using TensorCoord = MatrixCoord;               ///< Logical coordinate in referenced tensor
  using Index = typename TensorRef::Index;
  using LongIndex = typename TensorRef::LongIndex;

  // Thread-level shape of a fragment
  using ThreadShape = MatrixShape<ThreadOutputShape::kNHW, ThreadOutputShape::kC>;

  static_assert(!(ThreadShape::kColumn % MmaSimtPolicy::LaneMmaShape::kN),
                "Thread-level GEMM must be divisible by Policy::LaneMmaShape.");

  using ThreadTileCount = MatrixShape<ThreadBlockOutputShape::kH / ThreadOutputShape::kH,
                                      ThreadBlockOutputShape::kW / ThreadOutputShape::kW>;

  using Iterations =
      MatrixShape<ThreadShape::kRow, ThreadShape::kColumn / MmaSimtPolicy::LaneMmaShape::kN>;

  /// This is the complete warp-level accumulator tile.
  using AccumulatorTile = typename Operator::FragmentC;

  /// This is the fragment size produced by one access of the iterator.
  using Fragment = AccumulatorTile;

  /// Padding quantity
  using Padding = MatrixShape<0, 0>;

 private:
  // Storage type for accessing memory
  using AccessType = AlignedArray<Element, MmaSimtPolicy::LaneMmaShape::kN>;
  //
  // Data members
  //

  /// Internal pointer to memory
  AccessType *pointer_;

  /// Internal layout object
  Layout layout_;

  /// Base smem offset;
  Index base_smem_address_;

 public:
  /// Default constructor
  CUTLASS_HOST_DEVICE
  TileIteratorSimtDirect2dConv() : pointer_(nullptr) {}

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  TileIteratorSimtDirect2dConv(TensorRef const &ref, unsigned thread_id, unsigned lane_id)
      : pointer_(reinterpret_cast<AccessType *>(ref.data())),
        layout_(ref.stride()[0] / AccessType::kElements) {
  
    auto lane_layout = MmaSimtPolicy::get_lane_layout();

    MatrixCoord lane_offset = lane_layout.inverse(lane_id);

    // Get base HW offset of current threads
    const int threadgroup = thread_id / (ThreadBlockOutputShape::kC / ThreadOutputShape::kC);
    const int base_p = (threadgroup / (ThreadTileCount::kColumn)) * ThreadOutputShape::kH;
    const int base_q = (threadgroup % (ThreadTileCount::kColumn)) * ThreadOutputShape::kW;

    const int row_offset = base_p * ThreadBlockOutputShape::kW + base_q;

    pointer_ += layout_(
        {row_offset,
         lane_offset.column() * MmaSimtPolicy::LaneMmaShape::kN / int(AccessType::kElements)});
  }

  /// Adds a pointer offset
  CUTLASS_HOST_DEVICE
  TileIteratorSimtDirect2dConv &add_pointer_offset(Index pointer_offset) {
    pointer_ += pointer_offset / AccessType::kElements;
    return *this;
  }

  /// Store
  CUTLASS_HOST_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    AccessType *storer_pointer_ =
        reinterpret_cast<AccessType *>(reinterpret_cast<uint8_t *>(pointer_) + base_smem_address_);
    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int h = 0; h < ThreadOutputShape::kH; ++h) {
      CUTLASS_PRAGMA_UNROLL
      for (int w = 0; w < ThreadOutputShape::kW; ++w) {
        CUTLASS_PRAGMA_UNROLL
        for (int col = 0; col < Iterations::kColumn; ++col) {
          int offset = (w + h * ThreadBlockOutputShape::kW) *
                           (ThreadBlockOutputShape::kC / AccessType::kElements) +
                       col;
          storer_pointer_[offset + pointer_offset / int(AccessType::kElements)] =
              frag_ptr[w + h * ThreadOutputShape::kW + col];
        }
      }
    }
  }

  /// Store
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) { store_with_pointer_offset(frag, 0); }

  /// Set smem base address
  CUTLASS_HOST_DEVICE
  void set_smem_base_address(Index address) { base_smem_address_ = address; }
};
/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
