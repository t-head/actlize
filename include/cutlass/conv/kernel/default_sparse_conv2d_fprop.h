/*
 * Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

/*! \file
    \brief
    Default kernel-level implicit GEMM sparse convolution definitions combine threadblock-scoped
      matrix multiply-add with the appropriate threadblock-scoped epilogue.
*/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv2d.h"

#include "cutlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_optimized.h"
#include "cutlass/conv/threadblock/conv2d_fprop_filter_tile_access_iterator_optimized.h"
#include "cutlass/gemm/threadblock/default_mma_core_sparse_ppu.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

namespace cutlass {
namespace conv {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Defines a kernel for SparseConv2dFprop
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename OperatorClass,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  int Stages,
  typename MathOperatorTag,
  conv::IteratorAlgorithm IteratorAlgorithm = IteratorAlgorithm::kAnalytic,
  conv::StrideSupport StrideSupport = StrideSupport::kStrided
#ifdef SAIL_CUSTOMIZE_CUTLASS
  /// Sparse is used for compression matrix.
  , gemm::Operand CompressOp_ = gemm::Operand::kA
#endif
> struct DefaultSparseConv2dFprop;

/// Defines a kernel for Sparse Conv2dFprop specialzation for Optimzed IteratorAlgorithm and
/// multistage pipeline for compression A.
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  int Stages,
  typename MathOperatorTag
>
struct DefaultSparseConv2dFprop <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  arch::PPU0010,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  Stages,
  MathOperatorTag,
  IteratorAlgorithm::kOptimized
> {
  using MmaCore = typename cutlass::gemm::threadblock::DefaultSparseMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
      Stages, MathOperatorTag
      >;

  static int const kSparse = MmaCore::kSparse;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using IteratorA =
    cutlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorOptimized<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK / kSparse>,
      ElementA,
      LayoutA,
      ThreadMapA
    >;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using IteratorB =
    cutlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorOptimized<
      cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
      ElementB,
      LayoutB,
      ThreadMapB
    >;

  // Define iterators over tiles from the E operand
  using ElementE = typename MmaCore::ElementE;
  using LayoutE = typename MmaCore::GmemLayoutE;
  using ThreadMapE = typename MmaCore::IteratorThreadMapE;
  using AccessTypeE =
      cutlass::Array<ElementE, MmaCore::VectorSizeE / sizeof_bits<ElementE>::value>;
  using IteratorE =
      cutlass::transform::threadblock::PredicatedTileAccessIterator<
          cutlass::MatrixShape<ThreadblockShape::kM,
                               ThreadblockShape::kK / kSparse /
                                   MmaCore::kElementsPerElementE>,
          ElementE, LayoutE, 1, ThreadMapE, AccessTypeE>;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = threadblock::SparseImplicitGemmMultistage<
    ThreadblockShape,
    IteratorA,
    typename MmaCore::SmemIteratorA,
    arch::CacheOperation::Always,
    IteratorB,
    typename MmaCore::SmemIteratorB,
    arch::CacheOperation::Global,
    IteratorE,
    typename MmaCore::SmemIteratorE,
    MmaCore::kCacheOpE,
    MmaPolicy,
    Stages
  >;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueTensorOp<
    ThreadblockShape,
    WarpMmaTensorOp,
    1,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolutionSparse<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};

#ifdef SAIL_CUSTOMIZE_CUTLASS
/// Defines a kernel for Sparse Conv2dFprop specialzation for Optimzed IteratorAlgorithm and
/// multistage pipeline for compression B.
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  int Stages,
  typename MathOperatorTag
>
struct DefaultSparseConv2dFprop <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  Stages,
  MathOperatorTag,
  IteratorAlgorithm::kOptimized,
  StrideSupport::kStrided,
  gemm::Operand::kB
> {
  using MmaCore = typename cutlass::gemm::threadblock::DefaultSparseMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
      Stages, MathOperatorTag, false, cutlass::arch::CacheOperation::Global, cutlass::arch::CacheOperation::Global, gemm::Operand::kB>;

  static int const kSparse = MmaCore::kSparse;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using IteratorA =
    cutlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorOptimized<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      ElementA,
      LayoutA,
      ThreadMapA
    >;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using IteratorB =
    cutlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorOptimized<
      cutlass::MatrixShape<ThreadblockShape::kK / kSparse, ThreadblockShape::kN>,
      ElementB,
      LayoutB,
      ThreadMapB
    >;

  // Define iterators over tiles from the E operand
  using ElementE = typename MmaCore::ElementE;
  using LayoutE = typename MmaCore::GmemLayoutE;
  using ThreadMapE = typename MmaCore::IteratorThreadMapE;
  using AccessTypeE =
      cutlass::Array<ElementE, MmaCore::VectorSizeE / sizeof_bits<ElementE>::value>;
  using IteratorE =
      cutlass::transform::threadblock::PredicatedTileAccessIterator<
          cutlass::MatrixShape<ThreadblockShape::kK / kSparse / MmaCore::kElementsPerElementE, ThreadblockShape::kN>,
          ElementE, LayoutE, 0, ThreadMapE, AccessTypeE>;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = threadblock::SparseImplicitGemmMultistage<
    ThreadblockShape,
    IteratorA,
    typename MmaCore::SmemIteratorA,
    arch::CacheOperation::Always,
    IteratorB,
    typename MmaCore::SmemIteratorB,
    arch::CacheOperation::Global,
    IteratorE,
    typename MmaCore::SmemIteratorE,
    MmaCore::kCacheOpE,
    MmaPolicy,
    Stages,
    bool,
    gemm::Operand::kB
  >;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueTensorOpArch<
    ThreadblockShape,
    WarpMmaTensorOp,
    1,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount,
    ArchTag
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolutionSparse<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop,
    Conv2dProblemSize,
    gemm::Operand::kB
  >;
};
#endif
/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
