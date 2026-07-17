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
    Default kernel-level implicit GEMM convolution definitions combine threadblock-scoped
      matrix multiply-add with the appropriate threadblock-scoped epilogue.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv2d.h"

#include "cutlass/conv/threadblock/conv2d_dgrad_output_gradient_tile_access_iterator_analytic.h"
#include "cutlass/conv/threadblock/conv2d_dgrad_output_gradient_tile_access_iterator_optimized.h"
#include "cutlass/conv/threadblock/conv2d_dgrad_output_gradient_tile_access_iterator_optimized_small_channel.h"
#if SAIL_DGRAD_STRIDE_OPT
#include "cutlass/conv/threadblock/conv2d_dgrad_output_gradient_tile_access_iterator_optimized_strided.h"
#include "cutlass/conv/threadblock/conv2d_dgrad_filter_tile_access_iterator_optimized_strided.h"
#endif
#include "cutlass/conv/threadblock/conv2d_dgrad_filter_tile_access_iterator_analytic.h"
#include "cutlass/conv/threadblock/conv2d_dgrad_filter_tile_access_iterator_optimized.h"
#include "cutlass/conv/threadblock/conv2d_dgrad_filter_tile_access_iterator_optimized_small_channel.h"
#include "cutlass/conv/threadblock/conv2d_tile_iterator.h"

#if SAIL_PPU_MMA
#include "cutlass/conv/threadblock/aiu/conv2d_dgrad_output_gradient_tile_access_iterator_optimized_aiu.h"
#include "cutlass/conv/threadblock/aiu/conv2d_dgrad_filter_tile_access_iterator_optimized_aiu.h"
#include "cutlass/conv/threadblock/aiu/conv2d_dgrad_output_gradient_tile_access_iterator_optimized_strided_aiu.h"
#include "cutlass/conv/threadblock/aiu/conv2d_dgrad_filter_tile_access_iterator_optimized_strided_aiu.h"
#endif

// group conv
#include "cutlass/conv/threadblock/group/conv2d_dgrad_filter_tile_access_iterator_optimized_group.h"
#include "cutlass/conv/threadblock/group/conv2d_dgrad_filter_tile_access_iterator_optimized_strided_group.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Defines a kernel for Conv2dGroupDgrad
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
  conv::StrideSupport StrideSupport = StrideSupport::kStrided,
  /// Access granularity of A matrix in units of elements
  int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value,
  /// Access granularity of B matrix in units of elements
  int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value,
  bool UseAsync = true,
  bool DoubleBufRegLd = true,
  conv::KernelType kType = KernelType::kGroup,
  int WarpIterations = 0
> struct DefaultConv2dGroupDgrad;

/// group conv kernels.
/// Dgrad Unity Strided
/// and multistage
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
  int AlignmentA,
  int AlignmentB
>
struct DefaultConv2dGroupDgrad <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  OperatorClass,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  Stages,
  MathOperatorTag,
  IteratorAlgorithm::kOptimized,
  StrideSupport::kUnity,
  AlignmentA,
  AlignmentB,
  true,
  true,
  KernelType::kGroup
> {

  static bool const Transpose = platform::is_same<LayoutC, cutlass::layout::TensorNCHW>::value ? true : false;

  static bool const Strided = false;

  // Define the core components from GEMM
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
        ElementB, layout::RowMajor, ElementAccumulator, layout::RowMajor, OperatorClass,
        3, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using IteratorA =
    cutlass::conv::threadblock::Conv2dDgradOutputGradientTileAccessIteratorOptimized<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      ElementA,
      ThreadMapA,
      StrideSupport::kUnity,
      AlignmentA
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using IteratorB =
    cutlass::conv::threadblock::Conv2dDgradFilterTileAccessIteratorOptimized<
      cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
      ElementB,
      ThreadMapB,
      StrideSupport::kUnity,
      AlignmentB
    >;

  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  static cutlass::arch::CacheOperation::Kind const CacheOpA =
      ((sizeof_bits<ElementA>::value * AlignmentA) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;
  static cutlass::arch::CacheOperation::Kind const CacheOpB =
      ((sizeof_bits<ElementB>::value * AlignmentB) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  static bool const SkipWarps =
        ThreadblockShape::kM * ThreadblockShape::kK / MmaCore::kWarpSize / AlignmentA < MmaCore::WarpCount::kCount ||
        ThreadblockShape::kN * ThreadblockShape::kK / MmaCore::kWarpSize / AlignmentB < MmaCore::WarpCount::kCount
        ? true : false;

  // Define the Mma
    using Mma = threadblock::ImplicitGemmMultistage<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    CacheOpA,
    IteratorB,
    SmemIteratorB,
    CacheOpB,
    MmaPolicy,
    Stages,
    true,
    SkipWarps
  >;

  // Define the epilogue
  using Epilogue =
      typename detail::DefaultConvEpilogue<
        ArchTag,
        ThreadblockShape,
        WarpMmaTensorOp,
        1,
        EpilogueOutputOp,
        Transpose
      >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolutionGroup<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kDgrad
  >;
};

/// group conv.
/// Dgrad Strided
/// and multistage.
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
  int AlignmentA,
  int AlignmentB,
  bool DoubleBufRegLd
>
struct DefaultConv2dGroupDgrad <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  OperatorClass,
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
  AlignmentA,
  AlignmentB,
  true,
  DoubleBufRegLd,
  KernelType::kGroup
> {

  static bool const Transpose = platform::is_same<LayoutC, cutlass::layout::TensorNCHW>::value ? true : false;

  static bool const Strided = true;

  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
        ElementB, layout::RowMajor, ElementAccumulator, layout::RowMajor, OperatorClass,
        3, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using IteratorA =
      cutlass::conv::threadblock::Conv2dDgradOutputGradientTileAccessIteratorOptimizedStrided<
      // cutlass::conv::threadblock::Conv2dDgradOutputGradientTileAccessIteratorAnalytic<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA,
        ThreadMapA,
        StrideSupport::kStrided,
        AlignmentA
        // StrideSupport::kUnity
      >;


  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using IteratorB =
      cutlass::conv::threadblock::Conv2dDgradFilterTileAccessIteratorOptimizedStrided<
      // cutlass::conv::threadblock::Conv2dDgradFilterTileAccessIteratorOptimized<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB,
        ThreadMapB,
        StrideSupport::kStrided,
        AlignmentB
      >;

  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  static cutlass::arch::CacheOperation::Kind const CacheOpA =
      ((sizeof_bits<ElementA>::value * AlignmentA) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;
  static cutlass::arch::CacheOperation::Kind const CacheOpB =
      ((sizeof_bits<ElementB>::value * AlignmentB) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  static bool const SkipWarps =
        ThreadblockShape::kM * ThreadblockShape::kK / MmaCore::kWarpSize / AlignmentA < MmaCore::WarpCount::kCount ||
        ThreadblockShape::kN * ThreadblockShape::kK / MmaCore::kWarpSize / AlignmentB < MmaCore::WarpCount::kCount
        ? true : false;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmMultistage<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    CacheOpA,
    IteratorB,
    SmemIteratorB,
    CacheOpB,
    MmaPolicy,
    Stages,
    DoubleBufRegLd,
    SkipWarps
  >;

  // Define the epilogue
  using Epilogue =

      typename detail::DefaultConvEpilogueStrided<
        ArchTag,
        ThreadblockShape,
        WarpMmaTensorOp,
        1,
        EpilogueOutputOp,
        Transpose
      >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolutionGroup<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kDgrad,
    Conv2dProblemSize,
    true,
    true
  >;
};

/// Defines a kernel for Conv2dDgrad specialzation for Optimized IteratorAlgorithm,
/// 2 stage pipeline, and FFMA-based mainloop for PPU, group conv
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
  typename MathOperatorTag,
  int AlignmentA,
  int AlignmentB
>
struct DefaultConv2dGroupDgrad <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassSimt,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  Stages,
  MathOperatorTag,
  IteratorAlgorithm::kOptimized,
  StrideSupport::kUnity,
  AlignmentA,
  AlignmentB,
  false,
  true,
  KernelType::kGroup
> {

  // Define the core components from GEMM
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::RowMajor, ElementAccumulator, layout::RowMajor, arch::OpClassSimt,
      2, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using IteratorA =
    cutlass::conv::threadblock::TileIterator<
      cutlass::conv::threadblock::Conv2dDgradOutputGradientTileAccessIteratorOptimized<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA,
        ThreadMapA,
        StrideSupport::kUnity
      >
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using IteratorB =
    cutlass::conv::threadblock::TileIterator<
      cutlass::conv::threadblock::Conv2dDgradFilterTileAccessIteratorOptimized<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB,
        ThreadMapB
      >
    >;

  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaSimtOp = typename MmaCore::MmaWarpSimt;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmPipelined<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    IteratorB,
    SmemIteratorB,
    ElementC,
    LayoutC,
    MmaPolicy
  >;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueSimt<
    ThreadblockShape,
    WarpMmaSimtOp,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolutionGroup<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kDgrad
  >;
};

/// Defines a kernel for Conv2dDgrad specialzation for Analytic IteratorAlgorithm,
/// 2 stage pipeline, and FFMA-based mainloop for PPU
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
  typename MathOperatorTag,
  int AlignmentA,
  int AlignmentB
>
struct DefaultConv2dGroupDgrad <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassSimt,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  Stages,
  MathOperatorTag,
  IteratorAlgorithm::kAnalytic,
  StrideSupport::kStrided,
  AlignmentA,
  AlignmentB,
  false,
  true,
  KernelType::kGroup
> {

  // Define the core components from GEMM
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::RowMajor, ElementAccumulator, layout::RowMajor, arch::OpClassSimt,
      2, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using IteratorA =
    cutlass::conv::threadblock::TileIterator<
      cutlass::conv::threadblock::Conv2dDgradOutputGradientTileAccessIteratorAnalytic<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA,
        ThreadMapA,
        StrideSupport::kStrided
      >
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using IteratorB =
    cutlass::conv::threadblock::TileIterator<
      cutlass::conv::threadblock::Conv2dDgradFilterTileAccessIteratorAnalytic<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB,
        ThreadMapB
      >
    >;

  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaSimtOp = typename MmaCore::MmaWarpSimt;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmPipelined<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    IteratorB,
    SmemIteratorB,
    ElementC,
    LayoutC,
    MmaPolicy
  >;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueSimt<
    ThreadblockShape,
    WarpMmaSimtOp,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolutionGroup<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kDgrad
  >;
};

/// Defines a kernel for Conv2dDgrad specialzation for Optimized IteratorAlgorithm Dgrad Unity
// 2 stage pipeline
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
  int AlignmentA,
  int AlignmentB
>
struct DefaultConv2dGroupDgrad <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  OperatorClass,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  Stages,
  MathOperatorTag,
  IteratorAlgorithm::kAnalytic,
  StrideSupport::kStrided,
  AlignmentA,
  AlignmentB,
  false,
  true,
  KernelType::kGroup
> {
static bool const Transpose = platform::is_same<LayoutC, cutlass::layout::TensorNCHW>::value ? true : false;

  static bool const Strided = false;

  // Define the core components from GEMM
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::RowMajor, ElementAccumulator, layout::RowMajor, OperatorClass,
      2, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using IteratorA =
    cutlass::conv::threadblock::TileIterator<
      cutlass::conv::threadblock::Conv2dDgradOutputGradientTileAccessIteratorAnalytic<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA,
        ThreadMapA,
        StrideSupport::kStrided,
        AlignmentA
      >
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using IteratorB =
    cutlass::conv::threadblock::TileIterator<
      cutlass::conv::threadblock::Conv2dDgradFilterTileAccessIteratorAnalytic<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB,
        ThreadMapB,
        AlignmentB
      >
    >;

  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  static cutlass::arch::CacheOperation::Kind const CacheOpA =
      ((sizeof_bits<ElementA>::value * AlignmentA) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;
  static cutlass::arch::CacheOperation::Kind const CacheOpB =
      ((sizeof_bits<ElementB>::value * AlignmentB) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmPipelined<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    IteratorB,
    SmemIteratorB,
    ElementC,
    LayoutC,
    MmaPolicy
  >;

  // Define the epilogue
  using Epilogue =
      typename detail::DefaultConvEpilogue<
      ArchTag,
      ThreadblockShape,
      WarpMmaTensorOp,
      1,
      EpilogueOutputOp,
      Transpose
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolutionGroup<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kDgrad,
    Conv2dProblemSize
  >;
};

//mgroup kUnity
////////////////////////////////////
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
  int AlignmentA,
  int AlignmentB,
  bool DoubleBufRegLd
>
struct DefaultConv2dGroupDgrad <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  OperatorClass,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  Stages,
  MathOperatorTag,
  IteratorAlgorithm::kAnalytic,
  StrideSupport::kUnity,
  AlignmentA,
  AlignmentB,
  true,
  DoubleBufRegLd,
  KernelType::kMultipleGroup,
  0
> {

  static bool const Transpose = platform::is_same<LayoutC, cutlass::layout::TensorNCHW>::value ? true : false;

  static bool const Strided = true;

  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
        ElementB, layout::RowMajor, ElementAccumulator, layout::RowMajor, OperatorClass,
        3, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using IteratorA =
      cutlass::conv::threadblock::Conv2dDgradOutputGradientTileAccessIteratorAnalytic<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA,
        ThreadMapA,
        StrideSupport::kUnity,
        AlignmentA,
        KernelType::kMultipleGroup
      >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using IteratorB =
      cutlass::conv::threadblock::Conv2dDgradFilterTileAccessIteratorAnalytic<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB,
        ThreadMapB,
        AlignmentB,
        KernelType::kMultipleGroup
      >;

  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  static cutlass::arch::CacheOperation::Kind const CacheOpA =
      ((sizeof_bits<ElementA>::value * AlignmentA) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;
  static cutlass::arch::CacheOperation::Kind const CacheOpB =
      ((sizeof_bits<ElementB>::value * AlignmentB) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  static bool const SkipWarps =
        ThreadblockShape::kM * ThreadblockShape::kK / MmaCore::kWarpSize / AlignmentA < MmaCore::WarpCount::kCount ||
        ThreadblockShape::kN * ThreadblockShape::kK / MmaCore::kWarpSize / AlignmentB < MmaCore::WarpCount::kCount
        ? true : false;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmMultistageGroup<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    CacheOpA,
    IteratorB,
    SmemIteratorB,
    CacheOpB,
    MmaPolicy,
    Stages,
    false,
    0
  >;

  // Define the epilogue
  using Epilogue =
      typename detail::DefaultConvEpilogue<
        ArchTag,
        ThreadblockShape,
        WarpMmaTensorOp,
        1,
        EpilogueOutputOp,
        Transpose
      >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolutionMultGroup<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kDgrad
  >;
};

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
  int AlignmentA,
  int AlignmentB,
  bool DoubleBufRegLd
>
struct DefaultConv2dGroupDgrad <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  OperatorClass,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  Stages,
  MathOperatorTag,
  IteratorAlgorithm::kAnalytic,
  StrideSupport::kUnity,
  AlignmentA,
  AlignmentB,
  true,
  DoubleBufRegLd,
  KernelType::kMultipleGroup,
  1
> {

  static bool const Transpose = platform::is_same<LayoutC, cutlass::layout::TensorNCHW>::value ? true : false;

  static bool const Strided = true;

  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
        ElementB, layout::RowMajor, ElementAccumulator, layout::RowMajor, OperatorClass,
        3, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using IteratorA =
      cutlass::conv::threadblock::Conv2dDgradOutputGradientTileAccessIteratorAnalytic<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA,
        ThreadMapA,
        StrideSupport::kUnity,
        AlignmentA,
        KernelType::kMultipleGroup
      >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using IteratorB =
      cutlass::conv::threadblock::Conv2dDgradFilterTileAccessIteratorAnalytic<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB,
        ThreadMapB,
        AlignmentB,
        KernelType::kMultipleGroup
      >;

  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  static cutlass::arch::CacheOperation::Kind const CacheOpA =
      ((sizeof_bits<ElementA>::value * AlignmentA) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;
  static cutlass::arch::CacheOperation::Kind const CacheOpB =
      ((sizeof_bits<ElementB>::value * AlignmentB) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  static bool const SkipWarps =
        ThreadblockShape::kM * ThreadblockShape::kK / MmaCore::kWarpSize / AlignmentA < MmaCore::WarpCount::kCount ||
        ThreadblockShape::kN * ThreadblockShape::kK / MmaCore::kWarpSize / AlignmentB < MmaCore::WarpCount::kCount
        ? true : false;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmMultistageGroup<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    CacheOpA,
    IteratorB,
    SmemIteratorB,
    CacheOpB,
    MmaPolicy,
    Stages,
    false,
    1
  >;

  // Define the epilogue
  using Epilogue =
      typename detail::DefaultConvEpilogue<
        ArchTag,
        ThreadblockShape,
        WarpMmaTensorOp,
        1,
        EpilogueOutputOp,
        Transpose
      >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolutionMultGroup<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kDgrad
  >;
};

//mgroup kUnity, Optimized
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
  int AlignmentA,
  int AlignmentB,
  bool DoubleBufRegLd
>
struct DefaultConv2dGroupDgrad <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  OperatorClass,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  Stages,
  MathOperatorTag,
  IteratorAlgorithm::kOptimized,
  StrideSupport::kUnity,
  AlignmentA,
  AlignmentB,
  true,
  DoubleBufRegLd,
  KernelType::kMultipleGroup,
  0
> {

  static bool const Transpose = platform::is_same<LayoutC, cutlass::layout::TensorNCHW>::value ? true : false;

  static bool const Strided = true;

  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
        ElementB, layout::RowMajor, ElementAccumulator, layout::RowMajor, OperatorClass,
        3, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using IteratorA =
    cutlass::conv::threadblock::Conv2dDgradOutputGradientTileAccessIteratorOptimized<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      ElementA,
      ThreadMapA,
      StrideSupport::kUnity,
      AlignmentA,
      KernelType::kMultipleGroup
    >;


  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using IteratorB =
    cutlass::conv::threadblock::Conv2dDgradFilterTileAccessIteratorOptimizedGroup<
      cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
      ElementB,
      ThreadMapB,
      StrideSupport::kUnity,
      AlignmentB
    >;

  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  static cutlass::arch::CacheOperation::Kind const CacheOpA =
      ((sizeof_bits<ElementA>::value * AlignmentA) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;
  static cutlass::arch::CacheOperation::Kind const CacheOpB =
      ((sizeof_bits<ElementB>::value * AlignmentB) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  static bool const SkipWarps =
        ThreadblockShape::kM * ThreadblockShape::kK / MmaCore::kWarpSize / AlignmentA < MmaCore::WarpCount::kCount ||
        ThreadblockShape::kN * ThreadblockShape::kK / MmaCore::kWarpSize / AlignmentB < MmaCore::WarpCount::kCount
        ? true : false;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmMultistageGroup<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    CacheOpA,
    IteratorB,
    SmemIteratorB,
    CacheOpB,
    MmaPolicy,
    Stages,
    false,
    0
  >;

  // Define the epilogue
  using Epilogue =
      typename detail::DefaultConvEpilogue<
        ArchTag,
        ThreadblockShape,
        WarpMmaTensorOp,
        1,
        EpilogueOutputOp,
        Transpose
      >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolutionMultGroup<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kDgrad
  >;
};

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
  int AlignmentA,
  int AlignmentB,
  bool DoubleBufRegLd
>
struct DefaultConv2dGroupDgrad <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  OperatorClass,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  Stages,
  MathOperatorTag,
  IteratorAlgorithm::kOptimized,
  StrideSupport::kUnity,
  AlignmentA,
  AlignmentB,
  true,
  DoubleBufRegLd,
  KernelType::kMultipleGroup,
  1
> {

  static bool const Transpose = platform::is_same<LayoutC, cutlass::layout::TensorNCHW>::value ? true : false;

  static bool const Strided = true;

  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
        ElementB, layout::RowMajor, ElementAccumulator, layout::RowMajor, OperatorClass,
        3, MathOperatorTag>;

  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using IteratorA =
    cutlass::conv::threadblock::Conv2dDgradOutputGradientTileAccessIteratorOptimized<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      ElementA,
      ThreadMapA,
      StrideSupport::kUnity,
      AlignmentA,
      KernelType::kMultipleGroup
    >;


    using SmemIteratorA = typename MmaCore::SmemIteratorA;

    // Define iterators over tiles from the B operand
    using ThreadMapB = typename MmaCore::IteratorThreadMapB;
    using IteratorB =
      cutlass::conv::threadblock::Conv2dDgradFilterTileAccessIteratorOptimizedGroup<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB,
        ThreadMapB,
        StrideSupport::kUnity,
        AlignmentB
    >;

    using SmemIteratorB = typename MmaCore::SmemIteratorB;

    // Warp-level GEMM components
    using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
    using MmaPolicy = typename MmaCore::MmaPolicy;

    static cutlass::arch::CacheOperation::Kind const CacheOpA =
        ((sizeof_bits<ElementA>::value * AlignmentA) == 128)
            ? cutlass::arch::CacheOperation::Global
            : cutlass::arch::CacheOperation::Always;
    static cutlass::arch::CacheOperation::Kind const CacheOpB =
        ((sizeof_bits<ElementB>::value * AlignmentB) == 128)
            ? cutlass::arch::CacheOperation::Global
            : cutlass::arch::CacheOperation::Always;

    static bool const SkipWarps =
          ThreadblockShape::kM * ThreadblockShape::kK / MmaCore::kWarpSize / AlignmentA < MmaCore::WarpCount::kCount ||
          ThreadblockShape::kN * ThreadblockShape::kK / MmaCore::kWarpSize / AlignmentB < MmaCore::WarpCount::kCount
          ? true : false;

    // Define the Mma
    using Mma = threadblock::ImplicitGemmMultistageGroup<
      ThreadblockShape,
      IteratorA,
      SmemIteratorA,
      CacheOpA,
      IteratorB,
      SmemIteratorB,
      CacheOpB,
      MmaPolicy,
      Stages,
      false,
      1
    >;

    // Define the epilogue
    using Epilogue =
      typename detail::DefaultConvEpilogue<
        ArchTag,
        ThreadblockShape,
        WarpMmaTensorOp,
        1,
        EpilogueOutputOp,
        Transpose
      >::Epilogue;

    // Define the kernel
    using Kernel = cutlass::conv::kernel::ImplicitGemmConvolutionMultGroup<
      Mma,
      Epilogue,
      ThreadblockSwizzle,
      conv::Operator::kDgrad
    >;
  };

//mgroup kStrided, Optimized
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
  int AlignmentA,
  int AlignmentB,
  bool DoubleBufRegLd
>
struct DefaultConv2dGroupDgrad <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  OperatorClass,
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
  AlignmentA,
  AlignmentB,
  true,
  DoubleBufRegLd,
  KernelType::kMultipleGroup,
  0
> {

  static bool const Transpose = platform::is_same<LayoutC, cutlass::layout::TensorNCHW>::value ? true : false;

  static bool const Strided = true;

  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
        ElementB, layout::RowMajor, ElementAccumulator, layout::RowMajor, OperatorClass,
        3, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using IteratorA =
      cutlass::conv::threadblock::Conv2dDgradOutputGradientTileAccessIteratorOptimizedStrided<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA,
        ThreadMapA,
        StrideSupport::kStrided,
        AlignmentA,
        KernelType::kMultipleGroup
      >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using IteratorB =
      cutlass::conv::threadblock::Conv2dDgradFilterTileAccessIteratorOptimizedStridedGroup<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB,
        ThreadMapB,
        StrideSupport::kStrided,
        AlignmentB
      >;

  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  static cutlass::arch::CacheOperation::Kind const CacheOpA =
      ((sizeof_bits<ElementA>::value * AlignmentA) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;
  static cutlass::arch::CacheOperation::Kind const CacheOpB =
      ((sizeof_bits<ElementB>::value * AlignmentB) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  static bool const SkipWarps =
        ThreadblockShape::kM * ThreadblockShape::kK / MmaCore::kWarpSize / AlignmentA < MmaCore::WarpCount::kCount ||
        ThreadblockShape::kN * ThreadblockShape::kK / MmaCore::kWarpSize / AlignmentB < MmaCore::WarpCount::kCount
        ? true : false;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmMultistageGroup<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    CacheOpA,
    IteratorB,
    SmemIteratorB,
    CacheOpB,
    MmaPolicy,
    Stages,
    false,
    0
  >;

  // Define the epilogue
  using Epilogue =
      typename detail::DefaultConvEpilogueStrided<
        ArchTag,
        ThreadblockShape,
        WarpMmaTensorOp,
        1,
        EpilogueOutputOp,
        Transpose
      >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolutionMultGroup<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kDgrad,
    Conv2dProblemSize,
    true,
    true
  >;
};

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
  int AlignmentA,
  int AlignmentB,
  bool DoubleBufRegLd
>
struct DefaultConv2dGroupDgrad <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  OperatorClass,
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
  AlignmentA,
  AlignmentB,
  true,
  DoubleBufRegLd,
  KernelType::kMultipleGroup,
  1
> {

  static bool const Transpose = platform::is_same<LayoutC, cutlass::layout::TensorNCHW>::value ? true : false;

  static bool const Strided = true;

  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
        ElementB, layout::RowMajor, ElementAccumulator, layout::RowMajor, OperatorClass,
        3, MathOperatorTag>;

 // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using IteratorA =
      cutlass::conv::threadblock::Conv2dDgradOutputGradientTileAccessIteratorOptimizedStrided<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA,
        ThreadMapA,
        StrideSupport::kStrided,
        AlignmentA,
        KernelType::kMultipleGroup
      >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using IteratorB =
      cutlass::conv::threadblock::Conv2dDgradFilterTileAccessIteratorOptimizedStridedGroup<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB,
        ThreadMapB,
        StrideSupport::kStrided,
        AlignmentB
      >;

  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  static cutlass::arch::CacheOperation::Kind const CacheOpA =
      ((sizeof_bits<ElementA>::value * AlignmentA) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;
  static cutlass::arch::CacheOperation::Kind const CacheOpB =
      ((sizeof_bits<ElementB>::value * AlignmentB) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  static bool const SkipWarps =
        ThreadblockShape::kM * ThreadblockShape::kK / MmaCore::kWarpSize / AlignmentA < MmaCore::WarpCount::kCount ||
        ThreadblockShape::kN * ThreadblockShape::kK / MmaCore::kWarpSize / AlignmentB < MmaCore::WarpCount::kCount
        ? true : false;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmMultistageGroup<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    CacheOpA,
    IteratorB,
    SmemIteratorB,
    CacheOpB,
    MmaPolicy,
    Stages,
    false,
    1
  >;

  using Epilogue =
      typename detail::DefaultConvEpilogueStrided<
        ArchTag,
        ThreadblockShape,
        WarpMmaTensorOp,
        1,
        EpilogueOutputOp,
        Transpose
      >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolutionMultGroup<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kDgrad,
    Conv2dProblemSize,
    true,
    true
  >;
};

} // namespace kernel
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

