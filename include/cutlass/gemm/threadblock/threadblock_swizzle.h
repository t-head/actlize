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
    \brief Implements several possible threadblock-swizzling functions mapping blockIdx to
      GEMM problems.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/platform/platform.h"
#include "cutlass/gemm/gemm.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeThreadIdxX() {
  return threadIdx.x;
}

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeThreadIdxY() {
  return threadIdx.y;
}

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeThreadIdxZ() {
  return threadIdx.z;
}

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockIdxX() {
  return blockIdx.x;
}

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockIdxY() {
  return blockIdx.y;
}

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockIdxZ() {
  return blockIdx.z;
}

/// Helper to rematerialize block Dim. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockDimX() {
  return blockDim.x;
}

/// Helper to rematerialize block Dim. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockDimY() {
  return blockDim.y;
}

/// Helper to rematerialize block Dim. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockDimZ() {
  return blockDim.z;
}

CUTLASS_HOST_DEVICE
void EncodeGridDim(dim3 &grid, int* pshift = nullptr) {
  int shift = 0;

// disable grid shift for ppu since ppu support >16b gridDim y

  if (pshift) {
    *pshift = shift;
  }
}

CUTLASS_HOST_DEVICE
void DecodeBlockIdx(int &block_idx_x, int &block_idx_y, int shift) {
// disable grid shift for ppu since ppu support >16b gridDim y
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Threadblock swizzling function for GEMMs
template <int N = 1>
struct GemmIdentityThreadblockSwizzle {

  CUTLASS_HOST_DEVICE
  GemmIdentityThreadblockSwizzle() { }

  int const kTile = N;

  /// Returns the shape of the problem in units of logical tiles
  CUTLASS_HOST_DEVICE
  GemmCoord get_tiled_shape(
    GemmCoord problem_size,
    GemmCoord tile_size,
    int split_k_slices,
    GemmCoord stride = GemmCoord{1, 1, 1}) const {

    return GemmCoord(
      (problem_size.m() + tile_size.m() - 1) / tile_size.m() * stride.m(),
      (problem_size.n() + tile_size.n() - 1) / tile_size.n() * stride.n(),
      split_k_slices);
  }

  /// Computes device grid dimensions given a size in units of logical tiles
  CUTLASS_HOST_DEVICE
  dim3 get_grid_shape(GemmCoord tiled_shape, int* pshift = nullptr) const {
    dim3 grid;
    if ((tiled_shape.m() < kTile) || (tiled_shape.n() < kTile)) {
      grid = dim3(tiled_shape.m(), tiled_shape.n(), tiled_shape.k());
    } else {
      grid = dim3(tiled_shape.m() * kTile, (tiled_shape.n() + kTile - 1) / kTile, tiled_shape.k());
    }
    EncodeGridDim(grid, pshift);
    return grid;
  }

  /// Calculates optimal swizzle width
  CUTLASS_HOST_DEVICE
  int get_log_tile(GemmCoord tiled_shape) const {
    auto n = tiled_shape.n();
    // Thresholds picked so that it doesn't cause too many no-op CTAs
    if (N >= 8 && n >= 6)
      return 3;
    else if (N >= 4 && n >= 3)
      return 2;
    else if (N >= 2 && n >= 2)
      return 1;
    else
      return 0;
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  GemmCoord get_tile_offset(int log_tile) const {
    int block_idx_x = RematerializeBlockIdxX();
    int block_idx_y = RematerializeBlockIdxY();
    int block_idx_z = RematerializeBlockIdxZ();

    return GemmCoord{(block_idx_x >> log_tile),  //
                     (block_idx_y << log_tile) + ((block_idx_x) & ((1 << (log_tile)) - 1)),
                     block_idx_z};
  }


  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  GemmCoord get_tile_offset(GemmCoord tiled_shape, int shift = 0) const {

    int block_idx_x = RematerializeBlockIdxX();
    int block_idx_y = RematerializeBlockIdxY();
    DecodeBlockIdx(block_idx_x, block_idx_y, shift);

    if ((tiled_shape.m() < kTile) || (tiled_shape.n() < kTile))
      return GemmCoord{block_idx_x, block_idx_y, RematerializeBlockIdxZ()};

    return GemmCoord{
      (block_idx_x / kTile),
      (block_idx_y * kTile) + (block_idx_x % kTile),
      RematerializeBlockIdxZ()
    };
  }
};

#ifdef SAIL_CUSTOMIZE_CUTLASS
/// Threadblock swizzling function for GEMMs
template <int N = 1>
struct GemmIdentityThreadblockSwizzleOpt {

  CUTLASS_HOST_DEVICE
  GemmIdentityThreadblockSwizzleOpt() { }

  int const kTile = N;

  /// Returns the shape of the problem in units of logical tiles
  CUTLASS_HOST_DEVICE
  GemmCoord get_tiled_shape(
    GemmCoord problem_size,
    GemmCoord tile_size,
    int split_k_slices) const {

    return GemmCoord(
      (problem_size.m() + tile_size.m() - 1) / tile_size.m(),
      (problem_size.n() + tile_size.n() - 1) / tile_size.n(),
      split_k_slices);
  }

  /// Computes device grid dimensions given a size in units of logical tiles
  CUTLASS_HOST_DEVICE
  dim3 get_grid_shape(GemmCoord tiled_shape, int* pshift = nullptr) const {
    dim3 grid;
    if (kTile == 1 || (tiled_shape.m() <= kTile) || (tiled_shape.n() == 1)) {
      grid = dim3(tiled_shape.m(), tiled_shape.n(), tiled_shape.k());
    } else {
      grid = dim3(tiled_shape.m() * tiled_shape.n(), 1, tiled_shape.k());
    }
    EncodeGridDim(grid, pshift);
    return grid;
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  GemmCoord get_tile_offset(GemmCoord tiled_shape, int shift = 0) const {

    int block_idx_x = RematerializeBlockIdxX();
    int block_idx_y = RematerializeBlockIdxY();
    DecodeBlockIdx(block_idx_x, block_idx_y, shift);

    if ((kTile == 1) || (tiled_shape.m() <= kTile) || (tiled_shape.n() == 1))
      return GemmCoord{block_idx_x, block_idx_y, RematerializeBlockIdxZ()};

    const int y_offset = tiled_shape.n() / kTile * kTile;
    // return coord offset in tiled_shape
    if (block_idx_x < y_offset * tiled_shape.m()) {
      return GemmCoord{block_idx_x / kTile % tiled_shape.m(), block_idx_x % kTile + block_idx_x / (tiled_shape.m() * kTile) * kTile, RematerializeBlockIdxZ()};
    } else {
      // n remain block
      const int remain_idx_x = block_idx_x - y_offset * tiled_shape.m();
      const int remain_tile = tiled_shape.n() % kTile;
      return GemmCoord{remain_idx_x / remain_tile, remain_idx_x % remain_tile + y_offset, RematerializeBlockIdxZ()};
    }
  }
};
#endif

#if SAIL_DGRAD_STRIDE_OPT
// TODO: only support strid = 2 now.
template <int N = 1>
struct GemmIdentityThreadblockSwizzleStrided {

  CUTLASS_HOST_DEVICE
  GemmIdentityThreadblockSwizzleStrided() { }

  int const kTile = N;

  /// Returns the shape of the problem in units of logical tiles
  CUTLASS_HOST_DEVICE
  GemmCoord get_tiled_shape(
    GemmCoord problem_size,
    GemmCoord tile_size,
    int split_k_slices,
    GemmCoord stride = GemmCoord{1, 1, 1}) const {

    // for dgrad block shape should be (h+1)/2*(w+1)/2/128 + (h+1)/2*(w)/2/128 + (h)/2*(w+1)/2/128 + (h)/2*(w)/2/128
    // which is equal to hw/128, but each should uprounded, add 4 on block.x directly for convenience
    // will cause redundant blocks when nhw is small, so ori method is faster than a vendor DNN library when batch is small
    // not +3 to avoid m is aligned but each part doesn't
    return GemmCoord(
      (problem_size.m() + tile_size.m() - 1) / tile_size.m() + 4,
      (problem_size.n() + tile_size.n() - 1) / tile_size.n() * stride.n(),
      split_k_slices);
  }

  /// Computes device grid dimensions given a size in units of logical tiles
  CUTLASS_HOST_DEVICE
  dim3 get_grid_shape(GemmCoord tiled_shape, int* pshift = nullptr) const {
    dim3 grid;
    if ((tiled_shape.m() < kTile) || (tiled_shape.n() < kTile)) {
      grid = dim3(tiled_shape.m(), tiled_shape.n(), tiled_shape.k());
    } else {
      grid = dim3(tiled_shape.m() * kTile, (tiled_shape.n() + kTile - 1) / kTile, tiled_shape.k());
    }
    EncodeGridDim(grid, pshift);
    return grid;
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  GemmCoord get_tile_offset(GemmCoord tiled_shape, int shift = 0) const {

    int block_idx_x = RematerializeBlockIdxX();
    int block_idx_y = RematerializeBlockIdxY();
    DecodeBlockIdx(block_idx_x, block_idx_y, shift);

    if ((tiled_shape.m() < kTile) || (tiled_shape.n() < kTile))
      return GemmCoord{block_idx_x, block_idx_y, RematerializeBlockIdxZ()};

    return GemmCoord{
      (block_idx_x / kTile),
      (block_idx_y * kTile) + (block_idx_x % kTile),
      RematerializeBlockIdxZ()
    };
  }
};

struct GemmHorizontalThreadblockSwizzleStrided {

  CUTLASS_HOST_DEVICE
  GemmHorizontalThreadblockSwizzleStrided() { }

  /// Returns the shape of the problem in units of logical tiles
  CUTLASS_HOST_DEVICE
  GemmCoord get_tiled_shape(
    GemmCoord problem_size,
    GemmCoord tile_size,
    int split_k_slices,
    GemmCoord stride = GemmCoord{1, 1, 1}) const {

    return GemmCoord(
      (problem_size.m() + tile_size.m() - 1) / tile_size.m() + 4,
      (problem_size.n() + tile_size.n() - 1) / tile_size.n() * stride.n(),
      split_k_slices);
  }

  /// Computes device grid dimensions given a size in units of logical tiles
  CUTLASS_HOST_DEVICE
  dim3 get_grid_shape(GemmCoord tiled_shape, int* pshift = nullptr) const {
    dim3 grid(tiled_shape.n(), tiled_shape.m(), tiled_shape.k());
    EncodeGridDim(grid, pshift);
    return grid;
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  GemmCoord get_tile_offset(GemmCoord tiled_shape, int shift = 0) const {
    int block_idx_x = RematerializeBlockIdxX();
    int block_idx_y = RematerializeBlockIdxY();
    DecodeBlockIdx(block_idx_x, block_idx_y, shift);

    return GemmCoord{
      block_idx_y,
      block_idx_x,
      RematerializeBlockIdxZ()
    };
  }
};
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Threadblock swizzling function for GEMMs
struct GemmHorizontalThreadblockSwizzle {

  CUTLASS_HOST_DEVICE
  GemmHorizontalThreadblockSwizzle() { }

  /// Returns the shape of the problem in units of logical tiles
  CUTLASS_HOST_DEVICE
  GemmCoord get_tiled_shape(
    GemmCoord problem_size,
    GemmCoord tile_size,
    int split_k_slices,
    GemmCoord stride = GemmCoord{1, 1, 1}) const {

    return GemmCoord(
      (problem_size.m() + tile_size.m() - 1) / tile_size.m() * stride.m(),
      (problem_size.n() + tile_size.n() - 1) / tile_size.n() * stride.n(),
      split_k_slices);
  }

  /// Computes device grid dimensions given a size in units of logical tiles
  CUTLASS_HOST_DEVICE
  dim3 get_grid_shape(GemmCoord tiled_shape, int* pshift = nullptr) const {
    dim3 grid(tiled_shape.n(), tiled_shape.m(), tiled_shape.k());
    EncodeGridDim(grid, pshift);
    return grid;
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  GemmCoord get_tile_offset(GemmCoord tiled_shape, int shift = 0) const {
    int block_idx_x = RematerializeBlockIdxX();
    int block_idx_y = RematerializeBlockIdxY();
    DecodeBlockIdx(block_idx_x, block_idx_y, shift);

    return GemmCoord{
      block_idx_y,
      block_idx_x,
      RematerializeBlockIdxZ()
    };
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Threadblock swizzling function for batched GEMMs
struct GemmBatchedIdentityThreadblockSwizzle {

  /// Returns the shape of the problem in units of logical tiles
  CUTLASS_HOST_DEVICE
  GemmCoord get_tiled_shape(
    GemmCoord problem_size,
    GemmCoord tile_size,
    int batch_count) const {

    return GemmCoord(
      (problem_size.m() + tile_size.m() - 1) / tile_size.m(),
      (problem_size.n() + tile_size.n() - 1) / tile_size.n(),
      batch_count);
  }

  /// Calculates optimal swizzle width
  CUTLASS_HOST_DEVICE
  int get_log_tile(GemmCoord tiled_shape) const {
    return 0;
  }

  /// Computes device grid dimensions given a size in units of logical tiles
  CUTLASS_HOST_DEVICE
  dim3 get_grid_shape(GemmCoord tiled_shape, int* pshift = nullptr) const {
    dim3 grid(tiled_shape.m(), tiled_shape.n(), tiled_shape.k());
    EncodeGridDim(grid, pshift);
    return grid;
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  GemmCoord get_tile_offset(GemmCoord tiled_shape, int shift = 0) const {
    int block_idx_x = RematerializeBlockIdxX();
    int block_idx_y = RematerializeBlockIdxY();
    DecodeBlockIdx(block_idx_x, block_idx_y, shift);

    return GemmCoord{
      block_idx_x,
      block_idx_y,
      RematerializeBlockIdxZ()
    };
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  GemmCoord get_tile_offset(int log_tile) const {
    int block_idx_x = RematerializeBlockIdxX();
    int block_idx_y = RematerializeBlockIdxY();
    int block_idx_z = RematerializeBlockIdxZ();

    return GemmCoord{(block_idx_x >> log_tile),  //
                     (block_idx_y << log_tile) + ((block_idx_x) & ((1 << (log_tile)) - 1)),
                     block_idx_z};
  }

  /// Gets the batch index
  CUTLASS_DEVICE
  int get_batch_idx() const {
    return RematerializeBlockIdxZ();
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Threadblock swizzling function for split-K GEMMs
template <int N = 1>
struct GemmSplitKIdentityThreadblockSwizzle {

  int const kTile = N;

  /// Returns the shape of the problem in units of logical tiles
  CUTLASS_HOST_DEVICE
  GemmCoord get_tiled_shape(
    GemmCoord problem_size,
    GemmCoord tile_size,
    int partitions) const {

    return GemmCoord(
      (problem_size.m() + tile_size.m() - 1) / tile_size.m(),
      (problem_size.n() + tile_size.n() - 1) / tile_size.n(),
      partitions);
  }

  /// Computes device grid dimensions given a size in units of logical tiles
  CUTLASS_HOST_DEVICE
  dim3 get_grid_shape(GemmCoord tiled_shape, int* pshift = nullptr) const {
    dim3 grid;
    if ((tiled_shape.m() < kTile) || (tiled_shape.n() < kTile)) {
      grid = dim3(tiled_shape.m(), tiled_shape.n(), tiled_shape.k());
    } else {
      grid = dim3(tiled_shape.m() * kTile, (tiled_shape.n() + kTile - 1) / kTile, tiled_shape.k());
    }
    EncodeGridDim(grid, pshift);
    return grid;
  }


  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  GemmCoord get_tile_offset(GemmCoord tiled_shape, int shift = 0) const {

    int block_idx_x = RematerializeBlockIdxX();
    int block_idx_y = RematerializeBlockIdxY();
    DecodeBlockIdx(block_idx_x, block_idx_y, shift);

    if ((tiled_shape.m() < kTile) || (tiled_shape.n() < kTile))
      return GemmCoord{block_idx_x, block_idx_y, RematerializeBlockIdxZ()};

    return GemmCoord{
      (block_idx_x / kTile),
      (block_idx_y * kTile) + (block_idx_x % kTile),
      RematerializeBlockIdxZ()
    };
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Threadblock swizzling function for split-K GEMMs
struct GemmSplitKHorizontalThreadblockSwizzle {

  /// Returns the shape of the problem in units of logical tiles
  CUTLASS_HOST_DEVICE
  GemmCoord get_tiled_shape(
    GemmCoord problem_size,
    GemmCoord tile_size,
    int partitions) const {

    return GemmCoord(
      (problem_size.m() + tile_size.m() - 1) / tile_size.m(),
      (problem_size.n() + tile_size.n() - 1) / tile_size.n(),
      partitions);
  }

  /// Computes device grid dimensions given a size in units of logical tiles
  CUTLASS_HOST_DEVICE
  dim3 get_grid_shape(GemmCoord tiled_shape, int* pshift = nullptr) const {
    dim3 grid(tiled_shape.n(), tiled_shape.m(), tiled_shape.k());
    EncodeGridDim(grid, pshift);
    return grid;
  }


  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  GemmCoord get_tile_offset(GemmCoord tiled_shape, int shift = 0) const {
    int block_idx_x = RematerializeBlockIdxX();
    int block_idx_y = RematerializeBlockIdxY();
    DecodeBlockIdx(block_idx_x, block_idx_y, shift);

    return GemmCoord{
      block_idx_y,
      block_idx_x,
      RematerializeBlockIdxZ()
    };
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Threadblock swizzling function for batched GEMVs
struct GemvBatchedStridedThreadblockDefaultSwizzle {

  /// Returns the shape of the problem in units of logical tiles
  CUTLASS_HOST_DEVICE
  BatchedGemmCoord get_tiled_shape(
    BatchedGemmCoord problem_size,
    BatchedGemmCoord tile_size) const {

    return BatchedGemmCoord(
      1, // M is always 1
      (problem_size.n() + tile_size.n() - 1) / tile_size.n(),
      (problem_size.k() + tile_size.k() - 1) / tile_size.k(),
      (problem_size.batch() + tile_size.batch() - 1) / tile_size.batch());
  }

  /// Computes device grid dimensions given a size in units of logical tiles
  CUTLASS_HOST_DEVICE
  dim3 get_grid_shape(BatchedGemmCoord tiled_shape, int* pshift = nullptr) const {
    dim3 grid(tiled_shape.n(), tiled_shape.batch(), tiled_shape.k());
    EncodeGridDim(grid, pshift);
    return grid;
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  BatchedGemmCoord get_tile_offset(int shift = 0) const {
    int block_idx_x = RematerializeBlockIdxX();
    int block_idx_y = RematerializeBlockIdxY();
    DecodeBlockIdx(block_idx_x, block_idx_y, shift);

    return BatchedGemmCoord{
      0, // M is always 1
      block_idx_x,
      RematerializeBlockIdxZ(),
      block_idx_y,
    };
  }

  /// Gets the batch tile index
  CUTLASS_DEVICE
  int get_batch_tile_idx() const {
    return RematerializeBlockIdxY();
  }

  /// Gets the absolute batch index
  CUTLASS_DEVICE
  int get_batch_idx() const {
    return RematerializeBlockDimY()*RematerializeBlockIdxY() + RematerializeThreadIdxY();
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace cutlass

