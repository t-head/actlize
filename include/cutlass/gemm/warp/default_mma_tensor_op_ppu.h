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
    \brief Default warp-level GEMM operators selected by data type, size, and layouts of operands.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/warp/mma_tensor_op.h"
#include "cutlass/gemm/warp/mma_tensor_op_fast_f32.h"
#include "cutlass/gemm/warp/default_mma_tensor_op.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial Specialization - inputs and output types are float - uses BF16 internally
template <
    /// Shape of one matrix production operation (concept: GemmShape)
    typename WarpShape_,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB,
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC,
    /// Number of partitions along K dimension
    int PartitionsK,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor,
    // cube_ndhw size
    int CubeA,
    int CubeB
  >
struct DefaultMmaTensorOp<
  WarpShape_,
  GemmShape<16, 8, 8>,
  float, LayoutA,
  float, LayoutB,
  float, LayoutC,
  arch::OpMultiplyAddFastBF16,
  PartitionsK, AccumulatorsInRowMajor,
  CubeA, CubeB
> {

  // Uses BF16 internally
  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
      cutlass::arch::Mma<
        GemmShape<16, 8, 8>,
        32,
        bfloat16_t, cutlass::layout::RowMajor,
        bfloat16_t, cutlass::layout::ColumnMajor,
        float, cutlass::layout::RowMajor,
        arch::OpMultiplyAdd
      >,
      cutlass::MatrixShape<1, 1> >;

  // Define the warp-level tensor op
  using Type = cutlass::gemm::warp::MmaTensorOp<
      WarpShape_, float, LayoutA, float, LayoutB, float, LayoutC,
      Policy, PartitionsK, AccumulatorsInRowMajor, bool, CubeA, CubeB
    >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial Specialization - inputs and output types are float - uses F16 internally
template <
    /// Shape of one matrix production operation (concept: GemmShape)
    typename WarpShape_,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB,
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC,
    /// Number of partitions along K dimension
    int PartitionsK,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor,
    // cube_ndhw size
    int CubeA,
    int CubeB
  >
struct DefaultMmaTensorOp<
  WarpShape_,
  GemmShape<16, 8, 8>,
  float, LayoutA,
  float, LayoutB,
  float, LayoutC,
  arch::OpMultiplyAddFastF16,
  PartitionsK, AccumulatorsInRowMajor,
  CubeA, CubeB
> {
  // Uses F16 internally
  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
      cutlass::arch::Mma<
        GemmShape<16, 8, 8>,
        32,
        half_t, cutlass::layout::RowMajor,
        half_t, cutlass::layout::ColumnMajor,
        float, cutlass::layout::RowMajor,
        arch::OpMultiplyAdd
      >,
      cutlass::MatrixShape<1, 1> >;

  // Define the warp-level tensor op
  using Type = cutlass::gemm::warp::MmaTensorOp<
      WarpShape_, float, LayoutA, float, LayoutB, float, LayoutC,
      Policy, PartitionsK, AccumulatorsInRowMajor, bool, CubeA, CubeB
    >;
};

#if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__)) && ACOMPUTE_VERSION == 10000
// fp16 realization when instruction shape is 16/16/16
// input data type will be transfered in MmaTensorOp::transfer when stage = 3
template <
    /// Shape of one matrix production operation (concept: GemmShape)
    typename WarpShape_,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB,
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC,
    /// Number of partitions along K dimension
    int PartitionsK,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor,
    // cube_ndhw size
    int CubeA,
    int CubeB
  >
struct DefaultMmaTensorOp<
  WarpShape_,
  GemmShape<16, 16, 16>,
  float, LayoutA,
  float, LayoutB,
  float, LayoutC,
  arch::OpMultiplyAddFastF16,
  PartitionsK, AccumulatorsInRowMajor,
  CubeA, CubeB
> {
  // Uses F16 internally
  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
      cutlass::arch::Mma<
        GemmShape<16, 16, 16>,
        32,
        half_t, cutlass::layout::RowMajor,
        half_t, cutlass::layout::ColumnMajor,
        float, cutlass::layout::RowMajor,
        arch::OpMultiplyAdd
      >,
      cutlass::MatrixShape<1, 1> >;

  // Define the warp-level tensor op
  using Type = cutlass::gemm::warp::MmaTensorOp<
      WarpShape_, float, LayoutA, float, LayoutB, float, LayoutC,
      Policy, PartitionsK, AccumulatorsInRowMajor, bool, CubeA, CubeB
    >;
};
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial Specialization - inputs and output types are float - uses TF32 internally
template <
    /// Shape of one matrix production operation (concept: GemmShape)
    typename WarpShape_,
    /// Shape of target matrix multiply instruction (concept: GemmShape)
    typename InstructionShape_,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB,
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC,
    /// Number of partitions along K dimension
    int PartitionsK,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor,
    // cube_ndhw size
    int CubeA,
    int CubeB
  >
struct DefaultMmaTensorOp<
  WarpShape_,
  InstructionShape_,
  float, LayoutA,
  float, LayoutB,
  float, LayoutC,
  arch::OpMultiplyAdd, PartitionsK, AccumulatorsInRowMajor,
  CubeA, CubeB
> {

  // Uses TF32 internally
  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
      cutlass::arch::Mma<
        InstructionShape_,
        32,
        tfloat32_t, cutlass::layout::RowMajor,
        tfloat32_t, cutlass::layout::ColumnMajor,
        float, cutlass::layout::RowMajor,
        arch::OpMultiplyAdd
      >,
      cutlass::MatrixShape<1, 1> >;

  // Define the warp-level tensor op
  using Type = cutlass::gemm::warp::MmaTensorOp<
      WarpShape_, float, LayoutA, float, LayoutB, float, LayoutC,
      Policy, PartitionsK, AccumulatorsInRowMajor, bool, CubeA, CubeB
    >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial Specialization - inputs and output types are float - uses TF32 for Fast Accurate FP32
template <
    /// Shape of one matrix production operation (concept: GemmShape)
    typename WarpShape_,
    /// Shape of target matrix multiply instruction (concept: GemmShape)
    typename InstructionShape_,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB,
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC,
    /// Number of partitions along K dimension
    int PartitionsK,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor,
    // cube_ndhw size
    int CubeA,
    int CubeB
>
struct DefaultMmaTensorOp<
  WarpShape_,
  InstructionShape_,
  float, LayoutA,
  float, LayoutB,
  float, LayoutC,
  arch::OpMultiplyAddFastF32, PartitionsK, AccumulatorsInRowMajor,
  CubeA, CubeB
> {

  // Uses TF32 internally
  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
      cutlass::arch::Mma<
        InstructionShape_,
        32,
        cutlass::tfloat32_t, cutlass::layout::RowMajor,
        cutlass::tfloat32_t, cutlass::layout::ColumnMajor,
        float, cutlass::layout::RowMajor,
        arch::OpMultiplyAdd
      >,
      cutlass::MatrixShape<1, 1> >;

  // Define the warp-level tensor op
  using Type = cutlass::gemm::warp::MmaTensorOpFastF32<
      WarpShape_, float, LayoutA, float, LayoutB, float, LayoutC,
      Policy, PartitionsK, AccumulatorsInRowMajor
    >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass/gemm/warp/mma_complex_tensor_op_tile_iterator_ppu.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
