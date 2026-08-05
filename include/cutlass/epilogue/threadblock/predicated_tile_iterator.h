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
  \brief Epilogue for threadblock scoped GEMMs using Tensor Ops.

  The epilogue rearranges the result of a matrix product through shared memory to match canonical
  tensor layouts in global memory. Epilogues support conversion and reduction operations.

*/

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
  typename Element_,         ///< Element data type
  bool Transpose = false,    ///< transpose tensor cell output before shared memory
  bool MultGroup_ = false    ///< Group conv to combine multipe group in ont CTA>
>
class PredicatedTileIterator {
public:
  using ThreadMap = ThreadMap_;
  using Shape = typename ThreadMap::Shape;

  using Element = Element_;

  using Layout = layout::RowMajor;
  using TensorRef = TensorRef<Element, Layout>;
  using ConstTensorRef = typename TensorRef::ConstTensorRef;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using TensorCoord = MatrixCoord;

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

    #if SAIL_FUSE_OP_EXT
    ///< Check if mask is disabled totally
    CUTLASS_DEVICE bool is_all_clear() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) {
        if (predicates[i] == true) {
          return false;
        }
      }
      return true;
    }
    #endif
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

  /// A thread's starting column
  Index thread_start_column_;

  /// Internal state counter
  int state_[3];



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
  PredicatedTileIterator() {
  }
  #endif

  CUTLASS_DEVICE
  PredicatedTileIterator(
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
    // one epilogue loop always output 8 rows which is smallest mma shape
    // but when block.n=16 for fp16, one inst for one warp can write 16 rows
    // can decide which thread's row index is bigger than 7 and set invalid
    // but ThreadMap::initial_offset just return row index in whole block, can't use this to get valid
    // TODO: compare the performance between small alignment and big alignment with invalid for half warp
    TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx) + threadblock_offset;

    extent_row_ = extent.row();
    thread_start_row_ = thread_offset.row();

    // Initialize predicates
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < ThreadMap::Iterations::kColumn; ++c) {

      mask_.predicates[c] = ((thread_offset.column()
        + ThreadMap::Delta::kColumn * c) < extent.column());
    }

    // Null pointer performs no accesses
    if (!pointer) {
      mask_.clear();
    }

    // Initialize pointer
    byte_pointer_ = reinterpret_cast<uint8_t *>(pointer) +
      LongIndex(thread_offset.row()) * LongIndex(params_.stride) +
      LongIndex(thread_offset.column()) * sizeof(AccessType) / kElementsPerAccess;

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

            cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
                frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column],
                (void *)&memory_pointer[column * ThreadMap::Delta::kColumn / kElementsPerAccess],
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


  /// Stores a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) {

    store_with_byte_offset(frag, 0);
  }

  /// Advances to the next position to load or store
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator &operator++() {

    ++state_[0];
    byte_pointer_ += params_.advance_row;
    thread_start_row_ += ThreadMap::Shape::kRow;

    if (state_[0] == ThreadMap::Count::kRow) {

      state_[0] = 0;
      ++state_[1];
      byte_pointer_ += params_.advance_group;

      thread_start_row_ += (ThreadMap::Shape::kGroup - 1) *
        ThreadMap::Shape::kRow * ThreadMap::Count::kRow;

      if (state_[1] == ThreadMap::Count::kGroup) {

        state_[1] = 0;
        ++state_[2];
        byte_pointer_ += params_.advance_cluster;

        thread_start_row_ += ThreadMap::Count::kGroup *
          ThreadMap::Shape::kGroup * ThreadMap::Count::kRow * ThreadMap::Shape::kRow;

        if (state_[2] == ThreadMap::Count::kCluster) {
          state_[2] = 0;
          byte_pointer_ += params_.advance_tile;
        }
      }
    }

    return *this;
  }

  /// Need to get the thread start row from the tile iterator
  CUTLASS_DEVICE
  int32_t thread_start_row() const {
    return thread_start_row_;
  }

  /// Need to get the thread start row from the tile iterator
  CUTLASS_DEVICE
  int32_t thread_start_column() const {
    return thread_start_column_;
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
    mask = mask_;
  }

  ///< Sets the mask
  CUTLASS_DEVICE void set_mask(Mask const &mask) {
    mask_ = mask;
  }

  #if SAIL_FUSE_OP_EXT
  CUTLASS_DEVICE bool is_mask_all_clear() {
    return mask_.is_all_clear();
  }
  #endif
};

// for mgroup conv
template <
  typename ThreadMap_,       ///< Thread map (conept: OutputTileThreadMap)
  typename Element_,         ///< Element data type
  bool Transpose             ///< transpose tensor cell output before shared memory
>
class PredicatedTileIterator <ThreadMap_, Element_, Transpose, true>{
public:
  using ThreadMap = ThreadMap_;
  using Shape = typename ThreadMap::Shape;

  using Element = Element_;

  using Layout = layout::RowMajor;
  using TensorRef = TensorRef<Element, Layout>;
  using ConstTensorRef = typename TensorRef::ConstTensorRef;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using TensorCoord = MatrixCoord;

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

    #if SAIL_FUSE_OP_EXT
    ///< Check if mask is disabled totally
    CUTLASS_DEVICE bool is_all_clear() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) {
        if (predicates[i] == true) {
          return false;
        }
      }
      return true;
    }
    #endif
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

  int row_per_g_;
  Index extent_col_;
  int column;

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
  PredicatedTileIterator() {
  }
  #endif

  CUTLASS_DEVICE
  PredicatedTileIterator(
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
    // one epilogue loop always output 8 rows which is smallest mma shape
    // but when block.n=16 for fp16, one inst for one warp can write 16 rows
    // can decide which thread's row index is bigger than 7 and set invalid
    // but ThreadMap::initial_offset just return row index in whole block, can't use this to get valid
    // TODO: compare the performance between small alignment and big alignment with invalid for half warp

    TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx) + threadblock_offset;

    row_per_g_ = group_shape.row();
    int g_idx_in_row_ = thread_offset.row() / row_per_g_;
    int col_per_g_ = group_shape.column();
    int g_idx_in_col_ = thread_offset.column() / col_per_g_;

    thread_start_row_ = thread_offset.row();
    extent_row_ = Index((g_idx_in_col_ % groups + 1) * row_per_g_);
    extent_col_ = Index(col_per_g_);

    column = thread_offset.column() - (g_idx_in_col_ / groups) * (groups * col_per_g_);

    // Initialize predicates
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < ThreadMap::Iterations::kColumn; ++c) {
      mask_.predicates[c] = ((column
        + ThreadMap::Delta::kColumn * c) < col_per_g_ * (g_idx_in_row_ + 1));
    }

    // Null pointer performs no accesses
    if (!pointer) {
      mask_.clear();
    }

    int threadblock_col_offset = (g_idx_in_col_ / groups) * col_per_g_;
    int col_offset = g_idx_in_col_ * col_per_g_;

    // Initialize pointer
    LongIndex offset = LongIndex(thread_offset.row()) * LongIndex(params_.stride) +
      LongIndex(threadblock_col_offset + thread_offset.column() - col_offset) * sizeof(AccessType) / kElementsPerAccess;

    byte_pointer_ = reinterpret_cast<uint8_t *>(pointer) + offset;

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

          bool row_guard = ((row_offset + thread_start_row_) < extent_row_ );

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

            cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
                frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column],
                (void *)&memory_pointer[column * ThreadMap::Delta::kColumn / kElementsPerAccess],
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


  /// Stores a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) {

    store_with_byte_offset(frag, 0);
  }

  /// Advances to the next position to load or store
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator &operator++() {
    ++state_[0];

    byte_pointer_ += params_.advance_row;
    thread_start_row_ += ThreadMap::Shape::kRow;

    if (state_[0] == ThreadMap::Count::kRow) {
      state_[0] = 0;
      ++state_[1];
      byte_pointer_ += params_.advance_group;

      thread_start_row_ += (ThreadMap::Shape::kGroup - 1) *
        ThreadMap::Shape::kRow * ThreadMap::Count::kRow;

      if (state_[1] == ThreadMap::Count::kGroup) {

        state_[1] = 0;
        ++state_[2];
        byte_pointer_ += params_.advance_cluster;

        thread_start_row_ += ThreadMap::Count::kGroup *
          ThreadMap::Shape::kGroup * ThreadMap::Count::kRow * ThreadMap::Shape::kRow;

        if (state_[2] == ThreadMap::Count::kCluster) {
          state_[2] = 0;
          byte_pointer_ += params_.advance_tile;
        }
      }
    }


    int g_idx_in_row_ = thread_start_row_ / row_per_g_;

    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < ThreadMap::Iterations::kColumn; ++c) {
      mask_.predicates[c] = ((column
        + ThreadMap::Delta::kColumn * c) < extent_col_ * (g_idx_in_row_ + 1));
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
    mask = mask_;
  }

  ///< Sets the mask
  CUTLASS_DEVICE void set_mask(Mask const &mask) {
    mask_ = mask;
  }

  #if SAIL_FUSE_OP_EXT
  CUTLASS_DEVICE bool is_mask_all_clear() {
    return mask_.is_all_clear();
  }
  #endif
};

// for conv nchw
template <
  typename ThreadMap_,       ///< Thread map (conept: OutputTileThreadMap)
  typename Element_         ///< Element data type
>
class PredicatedTileIterator <ThreadMap_, Element_, true, false> {
public:
  using ThreadMap = ThreadMap_;
  using Shape = typename ThreadMap::Shape;

  using Element = Element_;

  using Layout = layout::RowMajor;
  using TensorRef = TensorRef<Element, Layout>;
  using ConstTensorRef = typename TensorRef::ConstTensorRef;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using TensorCoord = MatrixCoord;

  static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
  static int const kThreads = ThreadMap::kThreads;
  static int const kIterations = ThreadMap::Count::kTile;

  static_assert( ThreadMap::Iterations::kRow > 0,"ThreadMap::Iterations::kRow must be > 0");
  static_assert( ThreadMap::Iterations::kGroup == 1,"ThreadMap::Iterations::kGroup must be = 1");
  static_assert( ThreadMap::Iterations::kCluster == 1,"ThreadMap::Iterations::kCluster must be = 1");
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

    #if SAIL_FUSE_OP_EXT
    ///< Check if mask is disabled totally
    CUTLASS_DEVICE bool is_all_clear() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) {
        if (predicates[i] == true) {
          return false;
        }
      }
      return true;
    }
    #endif
  };

private:

  //
  // Data members
  //

  /// Parameters structure containing reference and precomputed state.
  PredicatedTileIteratorParams params_;

  Element *elem_pointer_;

  /// Array of boolean values to contain steady-state predicates
  Mask mask_;

  /// Extent of the matrix tile in rows
  Index extent_row_;

  /// A thread's starting row position (assuming steady-state predicates have been computed)
  Index block_start_row_;
  Index block_start_col_;
  Index thread_start_row_;
  Index thread_start_col_;

  // batch/m_in_batch/n size before transpose
  Coord<3> transpose_shape_;

  int row_remain_;
  bool column_guard_[ThreadMap::Iterations::kColumn];
  int thread_column_offset_[ThreadMap::Iterations::kColumn];
  int thread_row_offset_[ThreadMap::Iterations::kColumn];

  /// Internal state counter
  int state_[3];


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
  PredicatedTileIterator() {
  }
  #endif

  CUTLASS_DEVICE
  PredicatedTileIterator(
    PredicatedTileIteratorParams const & params,
    Element *pointer,
    TensorCoord extent,
    int thread_idx,
    TensorCoord threadblock_offset = TensorCoord(),
    Coord<3> transpose_shape = Coord<3>(),
    TensorCoord group_shape = TensorCoord(),
    int groups = 1
  ):
    // params_.stride is w now, but not used, we use transpose_shape_ to get output ptr
    params_(params),
    transpose_shape_(transpose_shape)
  {
    // element offset
    TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx);

    extent_row_ = extent.row();

    // get transposed block offset
    Index batch_idx = threadblock_offset.row() / transpose_shape_[1];
    block_start_col_ = threadblock_offset.row() % transpose_shape_[1];
    block_start_row_ = transpose_shape_[2] * batch_idx + threadblock_offset.column();

    // already on transposed coordinate
    thread_start_row_ = thread_offset.row();
    thread_start_col_ = thread_offset.column();

    // row_remain is on c, which is on vertical after transpose, should minus thread_offset.row
    row_remain_ = transpose_shape_[2] - threadblock_offset.column() - thread_start_row_;

    // Initialize predicates
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < ThreadMap::Iterations::kColumn; ++c) {
      int cur_col = block_start_col_ + thread_start_col_ + c * ThreadMap::Delta::kColumn;
      int cross_batch = cur_col / transpose_shape_[1];
      thread_column_offset_[c] = cur_col % transpose_shape_[1];
      thread_row_offset_[c] = block_start_row_ + thread_start_row_ + cross_batch * transpose_shape_[2];
    }

    // Initialize internal state counter
    state_[0] = state_[1] = state_[2] = 0;
    elem_pointer_ = pointer;

    // Null pointer performs no accesses
    if (!pointer) {
      mask_.clear();
    }
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    // assert(0);
  }

  /// Loads a fragment from memory
  // not used, but will cause stack if add assert(0)
  CUTLASS_DEVICE
  void load_with_byte_offset(Fragment &frag, int64_t byte_offset) {
    // assert(0);
  }


  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) {
    // assert(0);
  }

  /// Stores a fragment to memory
  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const &frag, int64_t byte_offset) {
    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {

      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {

          // avoid original block exceed on horizontal
          // block start always valid, only block_row + thread_row cross batch means thread invalid
          bool guard_hori = (row_remain_ - row * ThreadMap::Delta::kRow) > 0;

          int frag_row_idx =
            (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
            int cur_row = thread_row_offset_[column] + row * ThreadMap::Delta::kRow;
            int cur_col = thread_column_offset_[column];

            // avoid original block exceed on vertical
            bool guard = guard_hori && (cur_row < transpose_shape_[0] * transpose_shape_[2]);

            cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
                frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column],
                // TODO: here consider transpose_shape_[1] = output stride on n, which means stride on w/h is equal to 1/w
                // but original cutlass also just use stride on n, and consider nhw successive
                (void *)&elem_pointer_[cur_row * transpose_shape_[1] + cur_col],
                guard);

          }
        }
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
  PredicatedTileIterator &operator++() {

    ++state_[0];
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < ThreadMap::Iterations::kColumn; ++c) {
      thread_row_offset_[c] += ThreadMap::Shape::kRow;
    }
    row_remain_ -= ThreadMap::Shape::kRow;

    if (state_[0] == ThreadMap::Count::kRow) {

      state_[0] = 0;
      ++state_[1];

      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kColumn; ++c) {
        thread_row_offset_[c] += (ThreadMap::Shape::kGroup - 1) *
          ThreadMap::Shape::kRow * ThreadMap::Count::kRow;
      }

      if (state_[1] == ThreadMap::Count::kGroup) {

        state_[1] = 0;
        ++state_[2];

        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < ThreadMap::Iterations::kColumn; ++c) {
          thread_row_offset_[c] += ThreadMap::Count::kGroup *
            ThreadMap::Shape::kGroup * ThreadMap::Count::kRow * ThreadMap::Shape::kRow;
        }

        if (state_[2] == ThreadMap::Count::kCluster) {
          state_[2] = 0;
        }
      }
    }

    return *this;
  }

  #if SAIL_FUSE_OP_EXT
  CUTLASS_DEVICE bool is_mask_all_clear() {
    return mask_.is_all_clear();
  }
  #endif

};

////////////////////////////////////////////////////////////////////////////////

/// Tile iterator used to load output tile from shared memory in epilogue.
///
/// Satisfies: ReadableTileIterator | InterleavedPredicatedTileIterator | ForwardTileIterator
///
template <
  typename ThreadMap_,       ///< Thread map (conept: OutputTileThreadMap)
  typename Element_,         ///< Element data type
  int InterleavedN           ///< Number of Interleaved N
>
class InterleavedPredicatedTileIterator {
public:
  using ThreadMap = ThreadMap_;

  using Element = Element_;

  using Layout = layout::ColumnMajorInterleaved<InterleavedN>;
  using TensorRef = TensorRef<Element, Layout>;
  using ConstTensorRef = typename TensorRef::ConstTensorRef;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using TensorCoord = layout::PitchLinearCoord;

  static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
  static int const kThreads = ThreadMap::kThreads;
  static int const kIterations = ThreadMap::Iterations::kCount;

  /// Fragment object
  using Fragment = Array<Element, ThreadMap::kElementsPerAccess>;

  /// Memory access size
  using AccessType = AlignedArray<Element, ThreadMap::kElementsPerAccess>;

  //
  // Parameters struct
  //

  struct Params {

    //
    // Data members
    //

    LongIndex stride;               ///< stride in bytes between columns

    LongIndex advance_row;          ///< amount to add to move to the next 'row' position
    LongIndex advance_column;       ///< amount to add to move to the next 'column' position

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Status initialize(Index stride_) {

      stride = LongIndex(stride_);

      advance_row =
          ThreadMap::Delta::kContiguous * sizeof_bits<Element>::value / 8;

      advance_column = LongIndex(stride_) - ThreadMap::Iterations::kContiguous *
                                                kElementsPerAccess *
                                                sizeof_bits<Element>::value *
                                                ThreadMap::kWarpSize / 8;

      return Status::kSuccess;
    }

    CUTLASS_HOST_DEVICE
    Params() {
      initialize(0);
    }

    CUTLASS_HOST_DEVICE
    Params(Layout const &layout) {

      initialize(LongIndex(layout.stride(0)) * int(sizeof(AccessType)) / kElementsPerAccess);
    }
  };

  /// Mask object
  struct Mask {
    static int const kCount = (ThreadMap::Iterations::kContiguous < 8)
                                  ? 8
                                  : ThreadMap::Iterations::kContiguous;

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
  Params params_;

  /// Byte-level pointer
  uint8_t *byte_pointer_;

  /// Array of boolean values to contain steady-state predicates
  Mask mask_;

  /// Extent of the matrix tile in columns
  Index extent_col_;

  /// A thread's starting column position (assuming steady-state predicates have
  /// been computed)
  Index thread_start_col_;

  /// Internal iteration counter
  int iteration_contiguous_;

  int iteration_strided_;

private:

  //
  // Methods
  //

public:

  //
  // Methods
  //
  #if SAIL_FUSE_OP_EXT
  CUTLASS_DEVICE
  InterleavedPredicatedTileIterator() {
  }
  #endif

  /// Constructor
  CUTLASS_DEVICE
  InterleavedPredicatedTileIterator(
    Params const & params,
    Element *pointer,
    TensorCoord extent,
    int thread_idx,
    TensorCoord threadblock_offset
  ):
    params_(params) {
    TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx) +
                                TensorCoord(threadblock_offset.contiguous() * InterleavedN,
                                 threadblock_offset.strided() / InterleavedN);

    extent_col_ = extent.strided() / InterleavedN;
    thread_start_col_ = thread_offset.strided();

    // Initialize predicates
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
      mask_.predicates[c] =
          ((thread_offset.contiguous() + ThreadMap::Delta::kContiguous * c) <
           (extent.contiguous() * InterleavedN));
    }

    // Initialize pointer
    byte_pointer_ = reinterpret_cast<uint8_t *>(pointer) +
      LongIndex(thread_offset.strided()) * LongIndex(params_.stride) +
      LongIndex(thread_offset.contiguous()) * sizeof(AccessType) / kElementsPerAccess;

    // Initialize internal state counter
    iteration_contiguous_ = iteration_strided_ = 0;
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) {

    uint8_t *byte_pointer = byte_pointer_;
    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);
    AccessType *memory_pointer = reinterpret_cast<AccessType *>(byte_pointer);

    int col_offset = iteration_strided_ * ThreadMap::Delta::kStrided;

    bool col_guard = ((thread_start_col_ + col_offset) < extent_col_);

    bool guard = col_guard && mask_.predicates[iteration_contiguous_];

    cutlass::arch::global_load<
      AccessType,
      sizeof(AccessType)
    >(
        *frag_ptr,
        (void *)memory_pointer,
        guard);
  }

  /// Stores a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) {
    uint8_t *byte_pointer = byte_pointer_;
    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);
    AccessType *memory_pointer = reinterpret_cast<AccessType *>(byte_pointer);

    int col_offset = iteration_strided_ * ThreadMap::Delta::kStrided;

    bool col_guard = ((thread_start_col_ + col_offset) < extent_col_);

    bool guard = col_guard && mask_.predicates[iteration_contiguous_];

    cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
        *frag_ptr, (void *)memory_pointer, guard);
  }

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int iteration) {
    iteration_contiguous_ = iteration % ThreadMap::Iterations::kContiguous;
    iteration_strided_ = iteration / ThreadMap::Iterations::kContiguous;
  }

  /// Advances to the next position to load or store
  CUTLASS_HOST_DEVICE
  InterleavedPredicatedTileIterator &operator++() {

    ++iteration_contiguous_;
    byte_pointer_ += params_.advance_row;

    if (iteration_contiguous_ == ThreadMap::Iterations::kContiguous) {

      iteration_contiguous_ = 0;
      ++iteration_strided_;
      byte_pointer_ += params_.advance_column;

      if (iteration_strided_ == ThreadMap::Iterations::kStrided) {
        iteration_strided_ = 0;
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
    mask = mask_;
  }

  ///< Sets the mask
  CUTLASS_DEVICE void set_mask(Mask const &mask) {
    mask_ = mask;
  }
};

///////////////////////////////////////////////////////////////////////////////

/// Tile iterator used to load output tile from shared memory in epilogue.
///
/// Satisfies: ReadableTileIterator | InterleavedMaskedTileIterator | ForwardTileIterator
///
template <
  typename ThreadMap_,       ///< Thread map (conept: OutputTileThreadMap)
  typename Element_,         ///< Element data type
  int InterleavedN           ///< Number of Interleaved N
>
class InterleavedConvPredicatedTileIterator {
public:
  using ThreadMap = ThreadMap_;

  using Element = Element_;

  using Layout = layout::TensorNCxHWx<InterleavedN>;
  using TensorRef = TensorRef<Element, Layout>;
  using ConstTensorRef = typename TensorRef::ConstTensorRef;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using TensorCoord = Tensor4DCoord;

  static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
  static int const kThreads = ThreadMap::kThreads;
  static int const kIterations = ThreadMap::Iterations::kCount;

  /// Fragment object
  using Fragment = Array<Element, ThreadMap::kElementsPerAccess>;

  /// Memory access size
  using AccessType = AlignedArray<Element, ThreadMap::kElementsPerAccess>;

  //
  // Parameters struct
  //

  struct Params {

    //
    // Data members
    //

    LongIndex stride_col;           ///< stride in bytes between columns
    LongIndex stride_row;           ///< stride in bytes between rows

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Status initialize(typename Layout::Stride stride_) {
      stride_col = stride_[1];
      stride_row = stride_[2];

      return Status::kSuccess;
    }

    CUTLASS_HOST_DEVICE
    Params() {
      initialize(cutlass::make_Coord(0, 0, 0));
    }

    CUTLASS_HOST_DEVICE
    Params(Layout const &layout) {

      initialize(layout.stride());
    }
  };

  /// Mask object
  struct Mask {
    static int const kCount =
        (ThreadMap::Iterations::kRow < 8) ? 8 : ThreadMap::Iterations::kRow;

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
  Params params_;

  /// Byte-level pointer
  uint8_t *byte_pointer_;

  /// Array of boolean values to contain steady-state predicates
  Mask mask_;

  /// Extent of the matrix tile in columns
  Index extent_col_;

  /// Extent of the matrix tile in rows
  Index extent_row_;

  /// Extent of the matrix tile in pq
  Index extent_pq_;

  /// A thread's starting row position (assuming steady-state predicates have
  /// been computed)
  Index thread_start_row_;

  /// A thread's starting column position (assuming steady-state predicates have
  /// been computed)
  Index thread_start_col_;

  /// Internal iteration counter
  LongIndex iteration_row_;
  LongIndex iteration_col_;

  uint32_t pq_mul_;

  uint32_t pq_shr_;

private:

  //
  // Methods
  //

public:

  //
  // Methods
  //

  /// Constructor
  CUTLASS_DEVICE
  InterleavedConvPredicatedTileIterator(
    Params const & params,
    Element *pointer,
    TensorCoord extent,
    int thread_idx,
    MatrixCoord threadblock_offset
  ):
    params_(params) {
    MatrixCoord thread_offset = ThreadMap::initial_offset(thread_idx) + threadblock_offset;

    extent_col_ = extent.c();
    extent_pq_ = extent.h() * extent.w();
    extent_row_ = extent.n() * extent_pq_;

    find_divisor(pq_mul_, pq_shr_, extent_pq_);

    thread_start_row_ = thread_offset.row();
    thread_start_col_ = thread_offset.column();

    // Initialize predicates
    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < ThreadMap::Iterations::kRow; ++r) {
      mask_.predicates[r] =
          ((thread_offset.row() + ThreadMap::Delta::kRow * r) < extent_row_);
    }

    // Initialize pointer
    byte_pointer_ = reinterpret_cast<uint8_t *>(pointer) +
                    ((thread_start_col_ / InterleavedN) * params_.stride_col +
                     (thread_start_col_ % InterleavedN)) *
                        sizeof_bits<Element>::value / 8;

    // Initialize internal state counter
    iteration_row_ = iteration_col_ = 0;
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) {

    int col_offset = iteration_col_ * ThreadMap::Delta::kColumn;
    bool col_guard = ((thread_start_col_ + col_offset) < extent_col_);
    bool guard = col_guard && mask_.predicates[iteration_row_];

    int n, pq_rem;

    fast_divmod(n, pq_rem,
                thread_start_row_ + iteration_row_ * ThreadMap::Delta::kRow,
                extent_pq_, pq_mul_, pq_shr_);

    uint8_t *byte_pointer =
        byte_pointer_ + (n * params_.stride_row + pq_rem * InterleavedN) *
                            sizeof_bits<Element>::value / 8;
    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);
    AccessType const *memory_pointer =
        reinterpret_cast<AccessType const *>(byte_pointer);

    cutlass::arch::global_load<
      AccessType,
      sizeof(AccessType)
    >(
        *frag_ptr,
        (void *)memory_pointer,
        guard);
  }

  /// Stores a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) {

    int col_offset = iteration_col_ * ThreadMap::Delta::kColumn;
    bool col_guard = ((thread_start_col_ + col_offset) < extent_col_);
    bool guard = col_guard && mask_.predicates[iteration_row_];

    int n, pq_rem;

    fast_divmod(n, pq_rem,
                thread_start_row_ + iteration_row_ * ThreadMap::Delta::kRow,
                extent_pq_, pq_mul_, pq_shr_);

    uint8_t *byte_pointer =
        byte_pointer_ + (n * params_.stride_row + pq_rem * InterleavedN) *
                            sizeof_bits<Element>::value / 8;
    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);
    AccessType *memory_pointer = reinterpret_cast<AccessType *>(byte_pointer);

    cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
        *frag_ptr, (void *)memory_pointer, guard);
  }

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int iteration) {
    iteration_row_ = iteration % ThreadMap::Iterations::kRow;
    iteration_col_ = iteration / ThreadMap::Iterations::kRow;
  }

  /// Advances to the next position to load or store
  CUTLASS_HOST_DEVICE
  InterleavedConvPredicatedTileIterator &operator++() {

    ++iteration_row_;

    if (iteration_row_ == ThreadMap::Iterations::kRow) {

      iteration_row_ = 0;
      ++iteration_col_;
      byte_pointer_ += params_.stride_col;

      if (iteration_col_ == ThreadMap::Iterations::kColumn) {
        iteration_col_ = 0;
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
    mask = mask_;
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

////////////////////////////////////////////////////////////////////////////////
