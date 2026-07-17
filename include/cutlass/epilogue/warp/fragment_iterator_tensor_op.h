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

#include "cutlass/epilogue/warp/tensor_op_policy.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace warp {

////////////////////////////////////////////////////////////////////////////////

///
template <
  typename WarpShape,         ///< shape of warp-level GEMM (concept: MatrixShape)
  typename OperatorShape,     ///< matrix multiply operation shape (concept: gemm::GemmShape)
  typename OperatorElementC,  ///< matrix multiply operation data type (concept: data type)
  typename OperatorFragmentC, ///< matrix multiply operation fragment (concept: Array)
  typename Layout,            ///< target shared memory layout
  bool Transpose = false      ///< transpose tensor cell output before shared memory
#if SAIL_EPILOGUE_OPT >= 2
  , typename ElementOutput = OperatorElementC
#endif
>
class FragmentIteratorTensorOp;

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for row-major shared memory
template <
  typename WarpShape_,         ///< shape of the warp-level GEMM tile
  typename OperatorShape_,     ///< matrix multiply operation shape (concept: gemm::GemmShape)
  typename OperatorElementC_,  ///< matrix multiply operation data type (concept: data type)
  typename OperatorFragmentC_  ///< matrix multiply operation fragment (concept: Array)
#if SAIL_EPILOGUE_OPT >= 2
  , typename ElementOutput
#endif
>
class FragmentIteratorTensorOp<WarpShape_, OperatorShape_, OperatorElementC_, OperatorFragmentC_, layout::RowMajor, false
#if SAIL_EPILOGUE_OPT >= 2
  , ElementOutput
#endif
> {
public:

  using WarpShape = WarpShape_;
  using OperatorShape = OperatorShape_;
  using OperatorElementC = OperatorElementC_;
  using OperatorFragmentC = OperatorFragmentC_;
  using Layout = layout::RowMajor;

  using Policy = TensorOpPolicy<WarpShape, OperatorShape, Layout>;

  using Fragment = Array<
#if SAIL_EPILOGUE_OPT >= 2
    ElementOutput,
#else
    OperatorElementC,
#endif
#if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__)) && ACOMPUTE_VERSION == 10000
    Policy::OperatorCount::kColumn * Policy::kElementsPerAccess * Policy::kRegPerFragmentRow
#else
  #if !defined(ACOMPUTE_AIU_CONV)
    Policy::OperatorCount::kColumn * Policy::kElementsPerAccess
  #else
    // 2 core matrix on row of each mma
    Policy::OperatorCount::kColumn * Policy::kElementsPerAccess * 2
  #endif
#endif
  >;

  /// This is the complete warp-level accumulator tile.
  using AccumulatorTile = Array<
    OperatorElementC,
    OperatorFragmentC::kElements * Policy::OperatorCount::kRow * Policy::OperatorCount::kColumn>;

  using OutputAccumulatorTile = AccumulatorTile;

  /// Number of times this iterator can be incremented
  static int const kIterations = Policy::kIterations;
  using TileIterations = MatrixShape<kIterations, 1>;
  static int const kIterationsPerTile = kIterations / TileIterations::kCount;

private:

  /// Internal access type
#if SAIL_EPILOGUE_OPT >= 2
  // TODO: maybe should keep AccessType to be an array when no type conversion
  using AccessType = OperatorElementC;
#else
  using AccessType = Array<OperatorElementC, Policy::kElementsPerAccess>;
#endif

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
  FragmentIteratorTensorOp(AccumulatorTile const &accum):
    accumulators_(reinterpret_cast<AccessType const *>(&accum)),
    index_(0) {
  }

  /// Increments
  CUTLASS_HOST_DEVICE
  FragmentIteratorTensorOp &operator++() {
    ++index_;
    return *this;
  }

  /// Decrements
  CUTLASS_HOST_DEVICE
  FragmentIteratorTensorOp &operator--() {
    --index_;
    return *this;
  }

  /// Loads a fragment from the referenced part of the accumulator tile
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag, int index_offset = 0) const {

    int index = index_ + index_offset;

#if SAIL_EPILOGUE_OPT >= 2
    // transfer to ElemOutput, AccessType is acc type
    ElementOutput *frag_ptr = reinterpret_cast<ElementOutput *>(&frag);
#else
    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);
#endif

    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < Policy::OperatorCount::kColumn; ++n) {

#if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__)) && ACOMPUTE_VERSION == 10000
      // Policy::OperatorCount::kColumn=4, each loop copy four element
      // jump four element each row, ptr type is two element, so jump index*2
      // jump 32 element for next two column, not 16 element for next column
      const int accumulator_access_offset =
        (index + n * Policy::kAccumulatorColumnStride / Policy::kElementsPerAccess) * Policy::kRegPerFragmentRow;
      CUTLASS_PRAGMA_UNROLL
      for (int r = 0; r < Policy::kRegPerFragmentRow; ++r) {
        frag_ptr[Policy::kRegPerFragmentRow * n + r] = accumulators_[accumulator_access_offset + r];
      }
#else
  #if SAIL_EPILOGUE_OPT >= 2
    #if !defined(ACOMPUTE_AIU_CONV)
      // needn't /2 since accumulators_ type is one element now
      // and index needs to *2
      int accumulator_access_offset =
        index * Policy::kElementsPerAccess + n * Policy::kAccumulatorColumnStride;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Policy::kElementsPerAccess; i++) {
        frag_ptr[n * Policy::kElementsPerAccess + i] = accumulators_[accumulator_access_offset + i];
      }
    #else
      // one loop is in one mma, next loop will cross mma
      int accumulator_access_offset =
        (index % 2) * Policy::kElementsPerAccess + (index / 2) * 8 + n * Policy::kAccumulatorColumnStride * 2;
      CUTLASS_PRAGMA_UNROLL
      // acompute 10500 aiu conv will use 16/16 mma, which has 4 elements on each row of one mma
      for (int i = 0; i < Policy::kElementsPerAccess * 2; i++) {
        frag_ptr[n * Policy::kElementsPerAccess * 2 + i] = accumulators_[accumulator_access_offset + i + 2 * (i / 2)];
      }
    #endif
  #else
      int accumulator_access_offset =
        index + n * Policy::kAccumulatorColumnStride / Policy::kElementsPerAccess;
      frag_ptr[n] = accumulators_[accumulator_access_offset];
  #endif
#endif
    }
  }
};

// for conv nchw output
template <
  typename WarpShape_,         ///< shape of the warp-level GEMM tile
  typename OperatorShape_,     ///< matrix multiply operation shape (concept: gemm::GemmShape)
  typename OperatorElementC_,  ///< matrix multiply operation data type (concept: data type)
  typename OperatorFragmentC_  ///< matrix multiply operation fragment (concept: Array)
#if SAIL_EPILOGUE_OPT >= 2
  , typename ElementOutput
#endif
>
class FragmentIteratorTensorOp<WarpShape_, OperatorShape_, OperatorElementC_, OperatorFragmentC_, layout::RowMajor, true
#if SAIL_EPILOGUE_OPT >= 2
  , ElementOutput
#endif
> {
public:

  using WarpShape = WarpShape_;
  using OperatorShape = OperatorShape_;
  using OperatorElementC = OperatorElementC_;
  using OperatorFragmentC = OperatorFragmentC_;
  // used by kernel to decide whether output layout is nchw or nhwc
  using Layout = layout::ColumnMajor;

  // policy is type to read mma output, which is col major, different to layout
  using Policy = TensorOpPolicy<WarpShape, OperatorShape, layout::RowMajor, true>;

  using Fragment = Array<
#if SAIL_EPILOGUE_OPT >= 2
    ElementOutput,
#else
    OperatorElementC,
#endif
    Policy::IterativeCount::kRow * Policy::kElementsPerAccess
  >;

  /// This is the complete warp-level accumulator tile.
  using AccumulatorTile = Array<
    OperatorElementC,
    OperatorFragmentC::kElements * Policy::OperatorCount::kRow * Policy::OperatorCount::kColumn>;

  using OutputAccumulatorTile = AccumulatorTile;

  /// Number of times this iterator can be incremented
  static int const kIterations = Policy::kIterations;

private:

  /// Internal access type
#if SAIL_EPILOGUE_OPT >= 2
  // TODO: maybe should keep AccessType to be an array when no type conversion
  using AccessType = OperatorElementC;
#else
  using AccessType = Array<OperatorElementC, Policy::kElementsPerAccess>;
#endif

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
  FragmentIteratorTensorOp(AccumulatorTile const &accum):
    accumulators_(reinterpret_cast<AccessType const *>(&accum)),
    index_(0) {
  }

  /// Increments
  CUTLASS_HOST_DEVICE
  FragmentIteratorTensorOp &operator++() {
    ++index_;
    return *this;
  }

  /// Decrements
  CUTLASS_HOST_DEVICE
  FragmentIteratorTensorOp &operator--() {
    --index_;
    return *this;
  }

  /// Loads a fragment from the referenced part of the accumulator tile
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag, int index_offset = 0) const {

    int index = index_ + index_offset;

#if SAIL_EPILOGUE_OPT >= 2
    // transfer to ElemOutput, AccessType is acc type
    ElementOutput *frag_ptr = reinterpret_cast<ElementOutput *>(&frag);
#else
    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);
#endif

    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < Policy::IterativeCount::kRow; ++n) {

#if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__)) && ACOMPUTE_VERSION == 10000
      // 0 1 | 2 3
      // 4 5 | 6 7
      // ppu read 0145 in one iter and 2367 in next iter
      // vertical should jump Policy::kElementsPerAccess * 2 elements each time
      // horizontal should advance 2 elements in odd loop and 2 * Policy::kAccumulatorColumnStride in even loop
      const int accumulator_access_offset =
        (index / 2) * Policy::kAccumulatorColumnStride * 2 + (index % 2) * Policy::kElementsPerAccess + n * Policy::kElementsPerAccess * 2;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Policy::kElementsPerAccess; ++i) {
        frag_ptr[n * Policy::kElementsPerAccess + i] = accumulators_[accumulator_access_offset + i];
      }
#else
  #if SAIL_EPILOGUE_OPT >= 2
    #if !defined(ACOMPUTE_AIU_CONV)
      // index needn't /4 since accumulators_ type is one element now
      // and n needs to *4, since each loop will read four elements
      int accumulator_access_offset =
        index * Policy::kAccumulatorColumnStride + n * Policy::kElementsPerAccess;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Policy::kElementsPerAccess; i++) {
        frag_ptr[n * Policy::kElementsPerAccess + i] = accumulators_[accumulator_access_offset + i];
      }
    #else
      // ppu1.5 layout:
      // 0 1 | 4 5
      // 2 3 | 6 7
      // 8 9 | 12 13
      // 10 11 | 14 15
      // ......
      // first iter: [0, 1, 2, 3, 8, 9, 10, 11, ...]
      // next iter: [4, 5, 6, 7, 12, 13, 14, 15, ...]
      const int accumulator_access_offset =
        (index / 2) * Policy::kAccumulatorColumnStride * 2 + (index % 2) * Policy::kElementsPerAccess * 2 +
        (n / 2) * Policy::kElementsPerAccess * 4 + (n % 2) * Policy::kElementsPerAccess;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Policy::kElementsPerAccess; i++) {
        frag_ptr[n * Policy::kElementsPerAccess + i] = accumulators_[accumulator_access_offset + i];
      }
    #endif
  #else
      int accumulator_access_offset =
        index * Policy::kAccumulatorColumnStride / Policy::kElementsPerAccess + n;
      frag_ptr[n] = accumulators_[accumulator_access_offset];
  #endif
#endif
    }
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Dedicated to interleaved layout
template <
    /// shape of the warp-level GEMM tile
    typename WarpShape_,
    /// matrix multiply operator shape (concept: gemm::GemmShape)
    typename OperatorShape_,
    /// matrix multiply operator data type (concept: data type)
    typename OperatorElementC_,
    /// matrix multiply operator fragment (concept: Array)
    typename OperatorFragmentC_,
    /// number of interleaved k
    int InterleavedK>
class FragmentIteratorTensorOp<WarpShape_, OperatorShape_, OperatorElementC_, OperatorFragmentC_,
                               layout::ColumnMajorInterleaved<InterleavedK>> {
 public:
  using WarpShape = WarpShape_;
  using OperatorShape = OperatorShape_;
  using OperatorElementC = OperatorElementC_;
  using OperatorFragmentC = OperatorFragmentC_;
  static int const kInterleavedK = InterleavedK;
  using Layout = layout::ColumnMajorInterleaved<kInterleavedK>;

  using Policy = TensorOpPolicy<WarpShape, OperatorShape, Layout>;

  /// This is the fragment size produced by one access of the iterator.
  using Fragment =
      Array<OperatorElementC,
            Policy::kElementsPerAccess * InterleavedK / OperatorShape::kN>;

  /// This is the complete warp-level accumulator tile.
  using AccumulatorTile =
      Array<OperatorElementC, OperatorFragmentC::kElements *
                                  Policy::OperatorCount::kRow *
                                  Policy::OperatorCount::kColumn>;

  /// Number of times this iterator can be incremented
  static int const kIterations = Policy::kIterations;

 private:
  /// Internal access type
  using AccessType =
      Array<OperatorElementC, Policy::kElementsPerAccess>;

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
  FragmentIteratorTensorOp(AccumulatorTile const &accum)
      : accumulators_(reinterpret_cast<AccessType const *>(&accum)),
        index_(0) {}

  /// Increments
  CUTLASS_HOST_DEVICE
  FragmentIteratorTensorOp &operator++() {
    ++index_;
    return *this;
  }

  /// Decrements
  CUTLASS_HOST_DEVICE
  FragmentIteratorTensorOp &operator--() {
    --index_;
    return *this;
  }

  /// Loads a fragment from the referenced part of the accumulator tile
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag, int index_offset = 0) const {
    int index = index_ + index_offset;

    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < (InterleavedK / OperatorShape::kN); ++n) {
      int index_m = index % (Policy::OperatorCount::kRow *
                             Policy::kIterationsPerInstruction);
      int index_n = index / (Policy::OperatorCount::kRow *
                             Policy::kIterationsPerInstruction);
      int accumulator_access_offset =
          (index_m / Policy::kIterationsPerInstruction) *
              (Policy::OperatorCount::kColumn *
               Policy::kIterationsPerInstruction) +
          (index_m % Policy::kIterationsPerInstruction) +
          index_n * (InterleavedK / OperatorShape::kN) *
              Policy::kIterationsPerInstruction +
          n * Policy::kIterationsPerInstruction;

      frag_ptr[n] = accumulators_[accumulator_access_offset];
    }
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
