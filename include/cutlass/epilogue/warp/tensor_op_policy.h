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
    \brief Defines basic structures needed for implementing the warp-scoped phase of the epilogue.
          These quantities assume a 'column-major' arrangement of TensorOp instructions, of which
          a row-oriented slice is visible per iteration.
*/

#pragma once

#include "cutlass/matrix_shape.h"
#include "cutlass/layout/matrix.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace warp {

////////////////////////////////////////////////////////////////////////////////

/// Policy details related to the epilogue
template <
  typename WarpShape,     ///< shape of warp-level GEMM (concept: MatrixShape)
  typename OperatorShape, ///< matrix multiply operation shape (concept: gemm:GemmShape)
  typename Layout,        ///< target shared memory layout
  bool Transpose = false  ///< transpose tensor cell output before shared memory
>
struct TensorOpPolicy;

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for row-major
template <
  typename WarpShape,           ///< shape of warp-level GEMM (concept: MatrixShape)
  typename OperatorShape        ///< matrix multiply operation shape (concept: gemm::GemmShape)
>
struct TensorOpPolicy<WarpShape, OperatorShape, layout::RowMajor, false> {

  /// Number of operations
  using OperatorCount = MatrixShape<
    (WarpShape::kM + OperatorShape::kM - 1) / OperatorShape::kM,
    (WarpShape::kN + OperatorShape::kN - 1) / OperatorShape::kN
  >;

  //
  // Hard-coded constants regarding Tensor Operations
  //

#if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__)) && ACOMPUTE_VERSION == 10000
  static int const kElementsPerAccess = 1;
  static int const kRegPerFragmentRow = 4;
#else
  static int const kElementsPerAccess = 2;
#endif

  static int const kRowsPerIteration = 8;
  static bool const kDivisible =
    !(WarpShape::kM % OperatorShape::kM) && !(WarpShape::kN % OperatorShape::kN);

  //
  // Derived quantities
  //

  // Number of 'externally visible' iterations per actual instruction
  static int const kIterationsPerInstruction = OperatorShape::kM / kRowsPerIteration;

  // Number of externally visible iterations
  static int const kIterations = OperatorCount::kRow * kIterationsPerInstruction;

  static int const kAccumulatorRowStride = kElementsPerAccess;
  static int const kAccumulatorColumnStride = kElementsPerAccess * OperatorCount::kRow * kIterationsPerInstruction;
};

// for conv nchw output
template <
  typename WarpShape,           ///< shape of warp-level GEMM (concept: MatrixShape)
  typename OperatorShape        ///< matrix multiply operation shape (concept: gemm::GemmShape)
>
struct TensorOpPolicy<WarpShape, OperatorShape, layout::RowMajor, true> {

  /// Number of operations
  using OperatorCount = MatrixShape<
    (WarpShape::kM + OperatorShape::kM - 1) / OperatorShape::kM,
    (WarpShape::kN + OperatorShape::kN - 1) / OperatorShape::kN
  >;

  //
  // Hard-coded constants regarding Tensor Operations
  //

  // elements in each iteration
  static int const kElementsPerAccess = 2;

  static int const kRowsPerIteration = 8;
  static int const kColumnsPerIteration = 8;
  using IterativeCount = MatrixShape<
    (WarpShape::kM + kRowsPerIteration - 1) / kRowsPerIteration,
    (WarpShape::kN + kColumnsPerIteration - 1) / kColumnsPerIteration
  >;

  static bool const kDivisible =
    !(WarpShape::kM % OperatorShape::kM) && !(WarpShape::kN % OperatorShape::kN);

  //
  // Derived quantities
  //

  // Number of 'externally visible' iterations per actual instruction
  static int const kColumnIterationsPerInstruction = OperatorShape::kN / kColumnsPerIteration;
  static int const kRowIterationsPerInstruction = OperatorShape::kM / kRowsPerIteration;

  // Number of externally visible iterations
  // number of epilogue's main loop
  static int const kIterations = OperatorCount::kColumn * kColumnIterationsPerInstruction;

  // two elements to next row
  static int const kAccumulatorRowStride = kElementsPerAccess;
  static int const kAccumulatorColumnStride = kElementsPerAccess * OperatorCount::kRow * kRowIterationsPerInstruction;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for column-major-interleaved
template <
    typename WarpShape,  ///< shape of warp-level GEMM (concept: MatrixShape)
    typename OperatorShape,   ///< matrix multiply operation (concept: arch::Mma)
    int InterleavedK     ///< number of interleaved k
    >
struct TensorOpPolicy<WarpShape, OperatorShape,
                      layout::ColumnMajorInterleaved<InterleavedK> > {
  /// Number of operations
  using OperatorCount = MatrixShape<WarpShape::kM / OperatorShape::kM,
                                    WarpShape::kN / OperatorShape::kN>;

  //
  // Hard-coded constants regarding Tensor Operations
  //

#if SAIL_PPU_MMA
  static int const kElementsPerAccess = 4;
#else
  static int const kElementsPerAccess = 2;
#endif
  static int const kRowsPerIteration = 8;

  //
  // Derived quantities
  //

  // Number of 'externally visible' iterations per actual instruction
  static int const kIterationsPerInstruction =
      OperatorShape::kM / kRowsPerIteration;

  // Number of externally visible iterations
  static int const kIterations = WarpShape::kN / InterleavedK *
                                 OperatorCount::kRow *
                                 kIterationsPerInstruction;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
