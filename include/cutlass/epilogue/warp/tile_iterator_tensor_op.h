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
    \brief
*/

#pragma once

#include "cutlass/array.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"

#include "cutlass/epilogue/warp/tensor_op_policy.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Template for reading and writing tiles of accumulators to shared memory
template <
  typename WarpShape,     ///< shape of warp-level GEMM (concept: MatrixShape)
  typename OperatorShape, ///< matrix multiply operation shape (concept: gemm::GemmShape)
  typename Element,       ///< data type of element to be written
  typename Layout,        ///< target shared memory layout
  bool Transpose = false  ///< transpose tensor cell output before shared memory
>
class TileIteratorTensorOp;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Template for reading and writing tiles of accumulators to shared memory
template <
  typename WarpShape_,     ///< shape of warp-level GEMM (concept: GemmShape)
  typename OperatorShape_, ///< matrix multiply operation shape (concept: gemm::GemmShape)
  typename Element_        ///< data type of element to be written
>
class TileIteratorTensorOp<WarpShape_, OperatorShape_, Element_, layout::RowMajor, false> {
public:

  using WarpShape = WarpShape_;
  using OperatorShape = OperatorShape_;
  using Element = Element_;
  using Layout = layout::RowMajor;
  using TensorLayout = Layout;

  using TensorRef = TensorRef<Element, Layout>;         ///< Tensor Reference object
  using TensorCoord = MatrixCoord;                      ///< Logical coordinate in referenced tensor
  using Index = typename TensorRef::Index;
  using LongIndex = typename TensorRef::LongIndex;

  using Policy = TensorOpPolicy<WarpShape, OperatorShape, Layout>;

  /// Shape of the tile in memory
  using Shape = MatrixShape<
    Policy::kRowsPerIteration,
    WarpShape::kN
  >;

  /// This is the fragment size produced by one access of the iterator.
#if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__)) && ACOMPUTE_VERSION == 10000
  // Policy::OperatorCount::kColumn is half when mma size double, should *2 to maintain register size
  // can't change Policy::kElementsPerAccess because it will affect smem padding
  using Fragment = Array<
    Element,
    Policy::kRegPerFragmentRow * Policy::OperatorCount::kColumn * Policy::kElementsPerAccess>;
#else
  #if !defined(ACOMPUTE_AIU_CONV)
    using Fragment = Array<
      Element,
      Policy::OperatorCount::kColumn * Policy::kElementsPerAccess>;
  #else
    // 2 core matrix on row of each mma
    using Fragment = Array<
      Element,
      Policy::OperatorCount::kColumn * Policy::kElementsPerAccess * 2>;
  #endif
#endif

  /// This is the complete warp-level accumulator tile.
  //using AccumulatorTile = typename Operator::FragmentC;

  /// Number of times this iterator can be incremented
  static int const kIterations = Policy::kIterations;

  /// Number of times this iterator can be incremented
  using TileIterations = MatrixShape<kIterations, 1>;

  // Internal constants
  struct Detail {
    static int const kLanesInQuad = 4;
  };

  /// Padding quantity
  using Padding = MatrixShape<
    0,
    Detail::kLanesInQuad * Policy::kElementsPerAccess>;

private:

  /// Storage type for accessing memory
  using AccessType = AlignedArray<Element, Policy::kElementsPerAccess>;

  //
  // Data members
  //

  /// Internal pointer to memory
  AccessType *pointer_;

  /// Internal layout object
  Layout layout_;

  /// Thread offset
  MatrixCoord thread_offset_;

public:

  /// Default constructor
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOp(): pointer_(nullptr) { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOp(
    TensorRef const &ref,
    unsigned lane_id
  ):
    pointer_(reinterpret_cast<AccessType *>(ref.data())),
    layout_(ref.stride()[0] / Policy::kElementsPerAccess) {

    int quad_id = (lane_id / Detail::kLanesInQuad);
    int lane_in_quad = (lane_id % Detail::kLanesInQuad);

    thread_offset_ = {
      quad_id, lane_in_quad * Policy::kElementsPerAccess
    };

    pointer_ += layout_({thread_offset_.row(), thread_offset_.column() / Policy::kElementsPerAccess});
  }

  /// Adds a pointer offset
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOp & add_pointer_offset(Index pointer_offset) {
    pointer_ += pointer_offset / Policy::kElementsPerAccess;
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOp & add_tile_offset(TensorCoord const &tile_offset) {

    MatrixCoord coord_offset(
      tile_offset.row() * Shape::kRow,
      tile_offset.column() * Shape::kColumn
    );

    thread_offset_ += coord_offset;

    pointer_ += layout_({
      coord_offset.row(),
      coord_offset.column() / Policy::kElementsPerAccess
    });

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOp & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  /// Store
  CUTLASS_HOST_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {

    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    CUTLASS_PRAGMA_UNROLL
#if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__)) && ACOMPUTE_VERSION == 10000
    // Policy::OperatorCount::kColumn is half when mma size is double
    for (int n = 0; n < Policy::OperatorCount::kColumn * Policy::kRegPerFragmentRow; ++n) {
#else
    for (int n = 0; n < Policy::OperatorCount::kColumn; ++n) {
#endif
  #if !defined(ACOMPUTE_AIU_CONV)
      pointer_[n * Detail::kLanesInQuad + pointer_offset / Policy::kElementsPerAccess] = frag_ptr[n];
  #else
      // two core matrix in one mma
      pointer_[n * 2 * Detail::kLanesInQuad + pointer_offset / Policy::kElementsPerAccess] = frag_ptr[n * 2];
      pointer_[(n * 2 + 1) * Detail::kLanesInQuad + pointer_offset / Policy::kElementsPerAccess] = frag_ptr[n * 2 + 1];
  #endif
    }
  }

  /// Store
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }

  /// Load
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) const {

    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
#if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__)) && ACOMPUTE_VERSION == 10000
    // Policy::OperatorCount::kColumn is half when mma size is double
    for (int n = 0; n < Policy::OperatorCount::kColumn * 2; ++n) {
#else
    for (int n = 0; n < Policy::OperatorCount::kColumn; ++n) {
#endif
      frag_ptr[n] = pointer_[n * Detail::kLanesInQuad + pointer_offset / Policy::kElementsPerAccess];
    }
  }

  /// Load
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  CUTLASS_HOST_DEVICE
  TileIteratorTensorOp & operator++() {
    return add_tile_offset({1, 0});
  }
};

// for conv nchw
template <
  typename WarpShape_,     ///< shape of warp-level GEMM (concept: GemmShape)
  typename OperatorShape_, ///< matrix multiply operation shape (concept: gemm::GemmShape)
  typename Element_        ///< data type of element to be written
>
class TileIteratorTensorOp<WarpShape_, OperatorShape_, Element_, layout::RowMajor, true> {
public:

  using WarpShape = WarpShape_;
  using OperatorShape = OperatorShape_;
  using Element = Element_;
  // layout is used to calc tsm add, which is still row major
  using Layout = layout::RowMajor;

  using TensorLayout = Layout;
  using TensorRef = TensorRef<Element, Layout>;         ///< Tensor Reference object
  using TensorCoord = MatrixCoord;                      ///< Logical coordinate in referenced tensor
  using Index = typename TensorRef::Index;
  using LongIndex = typename TensorRef::LongIndex;

  // policy is type to read mma output, which is col major, different to layout
  using Policy = TensorOpPolicy<WarpShape, OperatorShape, layout::RowMajor, true>;

  /// Shape of the tile in memory
  // each warp read one vertical slice now
  using Shape = MatrixShape<
    Policy::kColumnsPerIteration,
    WarpShape::kM
  >;

  /// This is the fragment size produced by one access of the iterator.
  using Fragment = Array<
    Element,
    Policy::IterativeCount::kRow * Policy::kElementsPerAccess>;

  /// This is the complete warp-level accumulator tile.
  //using AccumulatorTile = typename Operator::FragmentC;

  /// Number of times this iterator can be incremented
  static int const kIterations = Policy::kIterations;
  /// Number of times this iterator can be incremented
  using TileIterations = MatrixShape<kIterations, 1>;

  // Internal constants
  struct Detail {
    static int const kLanesInQuad = 4;
    static int const kQuadsInWarp = 8;
  };

  /// Padding quantity
  using Padding = MatrixShape<
    0,
    Detail::kLanesInQuad * Policy::kElementsPerAccess>;

private:

  /// Storage type for accessing memory
  // always write one element once in col major
  using AccessType = AlignedArray<Element, 1>;

  //
  // Data members
  //

  /// Internal pointer to memory
  AccessType *pointer_;

  /// Internal layout object
  Layout layout_;

  /// Thread offset
  MatrixCoord thread_offset_;

public:

  /// Default constructor
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOp(): pointer_(nullptr) { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOp(
    TensorRef const &ref,
    unsigned lane_id
  ):
    pointer_(reinterpret_cast<AccessType *>(ref.data())),
    // needn't to divide kElementsPerAccess, since accessType is one element now
    layout_(ref.stride()[0]) {

    // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
    //   printf("smem base: %ld, srtride: %ld 00000000000000\n", ref.data(), ref.stride()[0]);
    // }

    int quad_id = (lane_id / Detail::kLanesInQuad);
    int lane_in_quad = (lane_id % Detail::kLanesInQuad);

    // hggc one lane write two elements on vertical, ppu one lane's two row are not successive
    thread_offset_ = {
#if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__)) && ACOMPUTE_VERSION == 10000
      lane_in_quad * 1, quad_id
#else
      lane_in_quad * 2, quad_id
#endif
    };

    pointer_ += layout_({thread_offset_.row(), thread_offset_.column()});
  }

  /// Adds a pointer offset
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOp & add_pointer_offset(Index pointer_offset) {
    pointer_ += pointer_offset;
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOp & add_tile_offset(TensorCoord const &tile_offset) {

    // should do transpose between warps, change row/col here
    MatrixCoord coord_offset(
      tile_offset.column() * Shape::kRow,
      tile_offset.row() * Shape::kColumn
    );

    thread_offset_ += coord_offset;

    pointer_ += layout_({
      coord_offset.row(),
      // needn't to divide kElementsPerAccess, since accessType is one element now
      coord_offset.column()
    });

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOp & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  /// Store
  CUTLASS_HOST_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {

    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    // Policy::IterativeCount::kRow is on tsm's horizontal since tsm is col major
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < Policy::IterativeCount::kRow; ++n) {
      // two elements of one lane is in different row in tsm, can't write together
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Policy::kElementsPerAccess; i++) {
#if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__)) && ACOMPUTE_VERSION == 10000
        // successive lanes on vertical write successive rows
        pointer_[n * Detail::kQuadsInWarp + i * 4 * layout_.stride()[0] + pointer_offset] = frag_ptr[n * Policy::kElementsPerAccess + i];
#else
        // on lane write two successive rows
        pointer_[n * Detail::kQuadsInWarp + i * layout_.stride()[0] + pointer_offset] = frag_ptr[n * Policy::kElementsPerAccess + i];
#endif
      }
    }
  }

  /// Store
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }

  /// Load
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) const {
    // assert(0);
  }

  /// Load
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  CUTLASS_HOST_DEVICE
  TileIteratorTensorOp & operator++() {
    return add_tile_offset({1, 0});
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Template for reading and writing tiles of accumulators to shared memory
template <
  typename WarpShape_,     ///< shape of warp-level GEMM (concept: GemmShape)
  typename OperatorShape_, ///< matrix multiply operation shape (concept: gemm::GemmShape)
  typename Element_,       ///< data type of element to be written
  typename Layout_
>
class TileIteratorTensorOpCanonical {
public:

  using WarpShape = WarpShape_;
  using OperatorShape = OperatorShape_;
  using Element = Element_;
  using Layout = Layout_;

  using TensorRef = TensorRef<Element, Layout>;         ///< Tensor Reference object
  using TensorCoord = MatrixCoord;                      ///< Logical coordinate in referenced tensor
  using Index = typename TensorRef::Index;
  using LongIndex = typename TensorRef::LongIndex;

  using Policy = TensorOpPolicy<WarpShape, OperatorShape, Layout>;

  static int const kAccessSize = 1;
  static int const kAccessCount = Policy::kElementsPerAccess / kAccessSize;

  /// Shape of the tile in memory
  using Shape = MatrixShape<
    Policy::kRowsPerIteration,
    WarpShape::kN
  >;

  /// This is the fragment size produced by one access of the iterator.
#if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__)) && ACOMPUTE_VERSION == 10000
  // Policy::OperatorCount::kColumn is half when mma size double, should *2 to maintain register size
  // can't change Policy::kElementsPerAccess because it will affect smem padding
  using Fragment = Array<
    Element,
    Policy::kRegPerFragmentRow * Policy::OperatorCount::kColumn * Policy::kElementsPerAccess>;
#else
  using Fragment = Array<
    Element,
    Policy::OperatorCount::kColumn * Policy::kElementsPerAccess>;
#endif

  /// This is the complete warp-level accumulator tile.
  //using AccumulatorTile = typename Operator::FragmentC;

  /// Number of times this iterator can be incremented
  static int const kIterations = Policy::kIterations;

  // Internal constants
  struct Detail {
    static int const kLanesInQuad = 4;
  };

  /// Padding quantity
  using Padding = MatrixShape<
    0,
    Detail::kLanesInQuad * Policy::kElementsPerAccess>;

private:

  /// Storage type for accessing memory
  using AccessType = AlignedArray<Element, kAccessSize>;

  //
  // Data members
  //

  /// Internal pointer to memory
  AccessType *pointer_;

  /// Internal layout object
  Layout layout_;

  /// Guard to indicate whether the shape is divisible
  bool divisible_;

  /// Extent of the output tensor
  MatrixCoord extent_;

  /// Thread offset
  MatrixCoord thread_offset_;

public:

  /// Default constructor
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpCanonical(): pointer_(nullptr) { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpCanonical(
    TensorRef const &ref,
    unsigned lane_id
  ):
    pointer_(reinterpret_cast<AccessType *>(ref.data())),
    layout_(ref.stride()[0]),
    divisible_(true),
    extent_(WarpShape::kM, WarpShape::kN) {

    int quad_id = (lane_id / Detail::kLanesInQuad);
    int lane_in_quad = (lane_id % Detail::kLanesInQuad);

    thread_offset_ = {
      quad_id, lane_in_quad * Policy::kElementsPerAccess
    };

    pointer_ += layout_({thread_offset_.row(), thread_offset_.column()});
  }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpCanonical(
    TensorRef const &ref,
    TensorCoord const &extent,
    unsigned lane_id
  ):
    pointer_(reinterpret_cast<AccessType *>(ref.data())),
    layout_(ref.stride()[0]),
    divisible_(false),
    extent_(extent) {

    int quad_id = (lane_id / Detail::kLanesInQuad);
    int lane_in_quad = (lane_id % Detail::kLanesInQuad);

    thread_offset_ = {
      quad_id, lane_in_quad * Policy::kElementsPerAccess
    };

    pointer_ += layout_({thread_offset_.row(), thread_offset_.column()});
  }

  /// Adds a pointer offset
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpCanonical & add_pointer_offset(Index pointer_offset) {
    pointer_ += pointer_offset;
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpCanonical & add_tile_offset(TensorCoord const &tile_offset) {

    MatrixCoord coord_offset(
      tile_offset.row() * Shape::kRow,
      tile_offset.column() * Shape::kColumn
    );

    thread_offset_ += coord_offset;

    pointer_ += layout_({
      coord_offset.row(),
      coord_offset.column()
    });

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpCanonical & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  /// Store
  CUTLASS_HOST_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {

    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < Policy::OperatorCount::kColumn; ++n) {
      CUTLASS_PRAGMA_UNROLL
      for (int a = 0; a < kAccessCount; ++a) {

        int ptr_idx = n * Detail::kLanesInQuad * kAccessCount + pointer_offset + a;
        int frag_idx = n * kAccessCount + a;

        int col = thread_offset_.column() + n * Detail::kLanesInQuad * Policy::kElementsPerAccess + a;

        if (divisible_ || (thread_offset_.row() < extent_.row() && col < extent_.column())) {
          pointer_[ptr_idx] = frag_ptr[frag_idx];
        }
      }
    }
  }

  /// Store
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }

  /// Load
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) const {

    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < Policy::OperatorCount::kColumn; ++n) {
      CUTLASS_PRAGMA_UNROLL
      for (int a = 0; a < kAccessCount; ++a) {

        int ptr_idx = n * Detail::kLanesInQuad * kAccessCount + pointer_offset + a;
        int frag_idx = n * kAccessCount + a;

        int col = thread_offset_.column() + n * Detail::kLanesInQuad * Policy::kElementsPerAccess + a;

        if (divisible_ || (thread_offset_.row() < extent_.row() && col < extent_.column())) {
          frag_ptr[frag_idx] = pointer_[ptr_idx];
        }
      }
    }
  }

  /// Load
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpCanonical & operator++() {
    return add_tile_offset({1, 0});
  }
};

template <
  typename WarpShape_,     ///< shape of warp-level GEMM (concept: GemmShape)
  typename OperatorShape_, ///< matrix multiply operation shape (concept: gemm::GemmShape)
  typename Element_        ///< data type of element to be written
>
class TileIteratorTensorOpFP16Acc {
public:

  using WarpShape = WarpShape_;
  using OperatorShape = OperatorShape_;
  using Element = Element_;
  using Layout = layout::RowMajor;
  using TensorLayout = Layout;

  using TensorRef = TensorRef<Element, Layout>;         ///< Tensor Reference object
  using TensorCoord = MatrixCoord;                      ///< Logical coordinate in referenced tensor
  using Index = typename TensorRef::Index;
  using LongIndex = typename TensorRef::LongIndex;

  using Policy = TensorOpPolicy<WarpShape, OperatorShape, Layout>;

  /// Shape of the tile in memory
  using Shape = MatrixShape<
    Policy::kRowsPerIteration,
    WarpShape::kN
  >;

  /// This is the fragment size produced by one access of the iterator.
#if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__)) && ACOMPUTE_VERSION == 10000
  // Policy::OperatorCount::kColumn is half when mma size double, should *2 to maintain register size
  // can't change Policy::kElementsPerAccess because it will affect smem padding
  using Fragment = Array<
    Element,
    Policy::kRegPerFragmentRow * Policy::OperatorCount::kColumn * Policy::kElementsPerAccess>;
#else
  #if !defined(ACOMPUTE_AIU_CONV)
    using Fragment = Array<
      Element,
      Policy::OperatorCount::kColumn * Policy::kElementsPerAccess>;
  #else
    // 2 core matrix on row of each mma
    using Fragment = Array<
      Element,
      Policy::OperatorCount::kColumn * Policy::kElementsPerAccess * 2>;
  #endif
#endif

  /// This is the complete warp-level accumulator tile.
  //using AccumulatorTile = typename Operator::FragmentC;

  /// Number of times this iterator can be incremented
  static int const kIterations = Policy::kIterations;

  /// Number of times this iterator can be incremented
  using TileIterations = MatrixShape<kIterations, 1>;

  // Internal constants
  struct Detail {
    static int const kLanesInQuad = 4;
  };

  /// Padding quantity
  using Padding = MatrixShape<
    0,
    Detail::kLanesInQuad * Policy::kElementsPerAccess>;

private:

  /// Storage type for accessing memory
  using AccessType = AlignedArray<Element, Policy::kElementsPerAccess>;

  //
  // Data members
  //

  /// Internal pointer to memory
  AccessType *pointer_;

  /// Internal layout object
  Layout layout_;

  /// Thread offset
  MatrixCoord thread_offset_;

public:

  /// Default constructor
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpFP16Acc(): pointer_(nullptr) { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpFP16Acc(
    TensorRef const &ref,
    unsigned lane_id
  ):
    pointer_(reinterpret_cast<AccessType *>(ref.data())),
    layout_(ref.stride()[0] / Policy::kElementsPerAccess) {

    int quad_id = (lane_id / Detail::kLanesInQuad);
    int lane_in_quad = (lane_id % Detail::kLanesInQuad);

    thread_offset_ = {
      // quad_id, lane_in_quad * Policy::kElementsPerAccess
      // each lane store 2 successive elems
      quad_id, lane_in_quad * Policy::kElementsPerAccess * 2
    };

    pointer_ += layout_({thread_offset_.row(), thread_offset_.column() / Policy::kElementsPerAccess});
  }

  /// Adds a pointer offset
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpFP16Acc & add_pointer_offset(Index pointer_offset) {
    pointer_ += pointer_offset / Policy::kElementsPerAccess;
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpFP16Acc & add_tile_offset(TensorCoord const &tile_offset) {

    MatrixCoord coord_offset(
      tile_offset.row() * Shape::kRow,
      tile_offset.column() * Shape::kColumn
    );

    thread_offset_ += coord_offset;

    pointer_ += layout_({
      coord_offset.row(),
      coord_offset.column() / Policy::kElementsPerAccess
    });

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpFP16Acc & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  /// Store
  CUTLASS_HOST_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {

    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    CUTLASS_PRAGMA_UNROLL
#if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__)) && ACOMPUTE_VERSION == 10000
    // Policy::OperatorCount::kColumn is half when mma size is double
    for (int n = 0; n < Policy::OperatorCount::kColumn * Policy::kRegPerFragmentRow; ++n) {
#else
    for (int n = 0; n < Policy::OperatorCount::kColumn; ++n) {
#endif
  #if !defined(ACOMPUTE_AIU_CONV)
      // pointer_[n * Detail::kLanesInQuad + pointer_offset / Policy::kElementsPerAccess] = frag_ptr[n];
      // each lane store two successive elems
      pointer_[(n / 2) * 8 + (n % 2) + pointer_offset / Policy::kElementsPerAccess] = frag_ptr[n];
  #else
      // two core matrix in one mma
      pointer_[n * 2 * Detail::kLanesInQuad + pointer_offset / Policy::kElementsPerAccess] = frag_ptr[n * 2];
      pointer_[(n * 2 + 1) * Detail::kLanesInQuad + pointer_offset / Policy::kElementsPerAccess] = frag_ptr[n * 2 + 1];
  #endif
    }
  }

  /// Store
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }

  /// Load
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) const {

    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
#if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__)) && ACOMPUTE_VERSION == 10000
    // Policy::OperatorCount::kColumn is half when mma size is double
    for (int n = 0; n < Policy::OperatorCount::kColumn * 2; ++n) {
#else
    for (int n = 0; n < Policy::OperatorCount::kColumn; ++n) {
#endif
      // frag_ptr[n] = pointer_[n * Detail::kLanesInQuad + pointer_offset / Policy::kElementsPerAccess];
      // each lane store two successive elems
      frag_ptr[n] = pointer_[(n / 2) * 8 + (n % 2) + pointer_offset / Policy::kElementsPerAccess];
    }
  }

  /// Load
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpFP16Acc & operator++() {
    return add_tile_offset({1, 0});
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
