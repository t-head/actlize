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
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator_aiu.h"
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

// include here to avoid sparse gemm use aiu swizzle
#include "cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op_aiu.h"

#if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__))
#include "cutlass/conv/threadblock/aiu/conv2d_dgrad_output_gradient_tile_access_iterator_optimized_aiu.h"
#include "cutlass/conv/threadblock/aiu/conv2d_dgrad_filter_tile_access_iterator_optimized_aiu.h"
#include "cutlass/conv/threadblock/aiu/conv2d_dgrad_output_gradient_tile_access_iterator_optimized_strided_aiu.h"
#include "cutlass/conv/threadblock/aiu/conv2d_dgrad_filter_tile_access_iterator_optimized_strided_aiu.h"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Defines a kernel for Conv2dDgrad
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
  conv::KernelType kType = KernelType::kNormal
> struct DefaultConv2dDgrad;

/////////////////////////////////////////////////////////////////////////////////////////////////
//                               OpClassTensorOp convolutions
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv2dDgrad specialzation for Analytic IteratorAlgorithm Dgrad Strided and
// multistage pipeline.
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
struct DefaultConv2dDgrad <
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
  true,
  true
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
    cutlass::conv::threadblock::Conv2dDgradOutputGradientTileAccessIteratorAnalytic<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      ElementA,
      ThreadMapA,
      StrideSupport::kStrided,
      AlignmentA
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using IteratorB =
    cutlass::conv::threadblock::Conv2dDgradFilterTileAccessIteratorAnalytic<
      cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
      ElementB,
      ThreadMapB,
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
          ThreadblockShape::kM * ThreadblockShape::kK / MmaCore::kWarpSize / (16 / sizeof(ElementA)) < MmaCore::WarpCount::kCount ||
          ThreadblockShape::kN * ThreadblockShape::kK / MmaCore::kWarpSize / (16 / sizeof(ElementB)) < MmaCore::WarpCount::kCount
          ? true : false;

  using Mma = typename threadblock::IgemmMultistageMainloop<
    cutlass::conv::Operator::kDgrad,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
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
  >::Type;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueTensorOp<
    ThreadblockShape,
    WarpMmaTensorOp,
    1,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount,
    Strided,
    Transpose
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kDgrad
  >;
};

/// Defines a kernel for Conv2dDgrad specialzation for Analytic IteratorAlgorithm Dgrad Strided
// and 2 stage pipeline.
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
struct DefaultConv2dDgrad <
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
  true
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
  using Epilogue = typename detail::DefaultConvEpilogue<
    ArchTag,
    ThreadblockShape,
    WarpMmaTensorOp,
    1,
    EpilogueOutputOp,
    Transpose
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kDgrad
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv2dDgrad specialzation for Analytic IteratorAlgorithm Dgrad Unity Strided
// and multistage pipeline.
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
struct DefaultConv2dDgrad <
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
  true
> {

  // Define the core components from GEMM
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
      StrideSupport::kUnity
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using IteratorB =
    cutlass::conv::threadblock::Conv2dDgradFilterTileAccessIteratorAnalytic<
      cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
      ElementB,
      ThreadMapB
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
          ThreadblockShape::kM * ThreadblockShape::kK / MmaCore::kWarpSize / (16 / sizeof(ElementA)) < MmaCore::WarpCount::kCount ||
          ThreadblockShape::kN * ThreadblockShape::kK / MmaCore::kWarpSize / (16 / sizeof(ElementB)) < MmaCore::WarpCount::kCount
          ? true : false;

  using Mma = typename threadblock::IgemmMultistageMainloop<
    cutlass::conv::Operator::kDgrad,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
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
  >::Type;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueTensorOp<
    ThreadblockShape,
    WarpMmaTensorOp,
    1,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kDgrad
  >;
};

/// Defines a kernel for Conv2dDgrad specialzation for Analytic IteratorAlgorithm Dgrad Unity
// 2 stage pipeline.
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
struct DefaultConv2dDgrad <
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
  false,
  true
> {

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
        StrideSupport::kUnity
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
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
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
  using Epilogue = typename detail::DefaultConvEpilogue<
    ArchTag,
    ThreadblockShape,
    WarpMmaTensorOp,
    1,
    EpilogueOutputOp
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kDgrad
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv2dDgrad specialzation for optimized IteratorAlgorithm Dgrad Unity Strided
// and multistage pipeline.
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
struct DefaultConv2dDgrad <
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
  DoubleBufRegLd
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
          ThreadblockShape::kM * ThreadblockShape::kK / MmaCore::kWarpSize / (16 / sizeof(ElementA)) < MmaCore::WarpCount::kCount ||
          ThreadblockShape::kN * ThreadblockShape::kK / MmaCore::kWarpSize / (16 / sizeof(ElementB)) < MmaCore::WarpCount::kCount
          ? true : false;

  using Mma = typename threadblock::IgemmMultistageMainloop<
    cutlass::conv::Operator::kDgrad,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
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
  >::Type;

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
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
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
struct DefaultConv2dDgrad <
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
  false,
  true
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
      cutlass::conv::threadblock::Conv2dDgradOutputGradientTileAccessIteratorOptimized<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA,
        ThreadMapA,
        StrideSupport::kUnity,
        AlignmentA
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
        ThreadMapB,
        StrideSupport::kUnity,
        AlignmentB
      >
    >;

  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
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
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kDgrad
  >;
};

#if SAIL_DGRAD_STRIDE_OPT
// strided version
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
struct DefaultConv2dDgrad <
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
  false,
  true
> {

  static bool const Transpose = platform::is_same<LayoutC, cutlass::layout::TensorNCHW>::value ? true : false;

  static bool const Strided = true;

  // Define the core components from GEMM
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
        ElementB, layout::RowMajor, ElementAccumulator, layout::RowMajor, OperatorClass,
        2, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using IteratorA =
    cutlass::conv::threadblock::TileIterator<
      cutlass::conv::threadblock::Conv2dDgradOutputGradientTileAccessIteratorOptimizedStrided<
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
      cutlass::conv::threadblock::Conv2dDgradFilterTileAccessIteratorOptimizedStrided<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB,
        ThreadMapB,
        StrideSupport::kStrided,
        AlignmentB
      >
    >;

  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
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
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kDgrad,
    Conv2dProblemSize,
    false,
    true
  >;
};

// strided version
/// Defines a kernel for Conv2dDgrad specialzation for Optimized IteratorAlgorithm Dgrad Unity
// multiple stages.
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
struct DefaultConv2dDgrad <
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
  DoubleBufRegLd
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
          ThreadblockShape::kM * ThreadblockShape::kK / MmaCore::kWarpSize / (16 / sizeof(ElementA)) < MmaCore::WarpCount::kCount ||
          ThreadblockShape::kN * ThreadblockShape::kK / MmaCore::kWarpSize / (16 / sizeof(ElementB)) < MmaCore::WarpCount::kCount
          ? true : false;

  using Mma = typename threadblock::IgemmMultistageMainloop<
    cutlass::conv::Operator::kDgrad,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
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
  >::Type;

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
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kDgrad,
    Conv2dProblemSize,
    true,
    true
  >;
};
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////
//                            OpClassSimt convolutions
/////////////////////////////////////////////////////////////////////////////////////////////////
/// Defines a kernel for Conv2dDgrad specialzation for Analytic IteratorAlgorithm,
/// multi-stage pipeline, and FFMA-based mainloop for PPU

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
struct DefaultConv2dDgrad <
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
  true,
  true
> {

  // Define the core components from GEMM
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::RowMajor, ElementAccumulator, layout::RowMajor, arch::OpClassSimt,
      Stages, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using IteratorA =
    cutlass::conv::threadblock::Conv2dDgradOutputGradientTileAccessIteratorAnalytic<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      ElementA,
      ThreadMapA,
      StrideSupport::kStrided
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using IteratorB =
    cutlass::conv::threadblock::Conv2dDgradFilterTileAccessIteratorAnalytic<
      cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
      ElementB,
      ThreadMapB
    >;

  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaSimtOp = typename MmaCore::MmaWarpSimt;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmMultistage<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    arch::CacheOperation::Always,
    IteratorB,
    SmemIteratorB,
    arch::CacheOperation::Always,
    MmaPolicy,
    Stages
  >;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueSimt<
    ThreadblockShape,
    WarpMmaSimtOp,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kDgrad
  >;

};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv2dDgrad specialzation for Optimized IteratorAlgorithm,
/// multi-stage pipeline, and FFMA-based mainloop for PPU

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
struct DefaultConv2dDgrad <
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
  true,
  true
> {

  // Define the core components from GEMM
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::RowMajor, ElementAccumulator, layout::RowMajor, arch::OpClassSimt,
      Stages, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using IteratorA =
    cutlass::conv::threadblock::Conv2dDgradOutputGradientTileAccessIteratorOptimized<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      ElementA,
      ThreadMapA,
      StrideSupport::kUnity
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using IteratorB =
    cutlass::conv::threadblock::Conv2dDgradFilterTileAccessIteratorOptimized<
      cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
      ElementB,
      ThreadMapB
    >;

  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaSimtOp = typename MmaCore::MmaWarpSimt;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmMultistage<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    arch::CacheOperation::Always,
    IteratorB,
    SmemIteratorB,
    arch::CacheOperation::Always,
    MmaPolicy,
    Stages
  >;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueSimt<
    ThreadblockShape,
    WarpMmaSimtOp,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kDgrad
  >;

};

/////////////////////////////////////////////////////////////////////////////////////////////////

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
struct DefaultConv2dDgrad <
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
  true
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
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kDgrad
  >;

};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv2dDgrad specialzation for Optimized IteratorAlgorithm,
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
struct DefaultConv2dDgrad <
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
  true
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
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kDgrad
  >;

};
/////////////////////////////////////////////////////////////////////////////////////////////////

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
  bool UseAsync = true
> struct DefaultConv2dDgradSmallChannel;


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
struct DefaultConv2dDgradSmallChannel <
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
  true
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
    cutlass::conv::threadblock::Conv2dDgradOutputGradientTileAccessIteratorOptimizedSmallChannel<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      ElementA,
      ThreadMapA,
      StrideSupport::kStrided,
      AlignmentA
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using IteratorB =
    cutlass::conv::threadblock::Conv2dDgradFilterTileAccessIteratorOptimizedSmallChannel<
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
    true
  >;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueTensorOpArch<
    ThreadblockShape,
    WarpMmaTensorOp,
    1,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount,
    ArchTag,
    Strided,
    Transpose
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kDgrad
  >;
};

#if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__)) && ACOMPUTE_VERSION == 10500

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
  conv::KernelType kType = KernelType::kNormal
> struct DefaultConv2dDgradAiu;

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
struct DefaultConv2dDgradAiu <
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
  DoubleBufRegLd
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
    cutlass::conv::threadblock::Conv2dDgradOutputGradientTileAccessIteratorOptimizedAiu<
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
    cutlass::conv::threadblock::Conv2dDgradFilterTileAccessIteratorOptimizedAiu<
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
          ThreadblockShape::kM * ThreadblockShape::kK / MmaCore::kWarpSize / (16 / sizeof(ElementA)) < MmaCore::WarpCount::kCount ||
          ThreadblockShape::kN * ThreadblockShape::kK / MmaCore::kWarpSize / (16 / sizeof(ElementB)) < MmaCore::WarpCount::kCount
          ? true : false;

  using WarpTileIteratorA = cutlass::gemm::warp::MmaTensorOpMultiplicandTileIteratorAiu<
    cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
    cutlass::MatrixShape<WarpShape::kM, WarpShape::kK>,
    ElementA,
    cutlass::MatrixShape<InstructionShape::kM, InstructionShape::kK>,
    false
  >;

  using WarpTileIteratorB = cutlass::gemm::warp::MmaTensorOpMultiplicandTileIteratorTransAiu<
    cutlass::MatrixShape<ThreadblockShape::kN, ThreadblockShape::kK>,
    cutlass::MatrixShape<WarpShape::kN, WarpShape::kK>,
    ElementB,
    cutlass::MatrixShape<InstructionShape::kN, InstructionShape::kK>,
    true
  >;

  // Define the Mma
    using Mma = threadblock::ImplicitGemmMultistageAiu<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    CacheOpA,
    IteratorB,
    SmemIteratorB,
    CacheOpB,
    MmaPolicy,
    Stages,
    WarpTileIteratorA,
    WarpTileIteratorB
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
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
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
struct DefaultConv2dDgradAiu <
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
  DoubleBufRegLd
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
      cutlass::conv::threadblock::Conv2dDgradOutputGradientTileAccessIteratorOptimizedStridedAiu<
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
      cutlass::conv::threadblock::Conv2dDgradFilterTileAccessIteratorOptimizedStridedAiu<
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
          ThreadblockShape::kM * ThreadblockShape::kK / MmaCore::kWarpSize / (16 / sizeof(ElementA)) < MmaCore::WarpCount::kCount ||
          ThreadblockShape::kN * ThreadblockShape::kK / MmaCore::kWarpSize / (16 / sizeof(ElementB)) < MmaCore::WarpCount::kCount
          ? true : false;

  using WarpTileIteratorA = cutlass::gemm::warp::MmaTensorOpMultiplicandTileIteratorAiu<
    cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
    cutlass::MatrixShape<WarpShape::kM, WarpShape::kK>,
    ElementA,
    cutlass::MatrixShape<InstructionShape::kM, InstructionShape::kK>,
    false
  >;

  using WarpTileIteratorB = cutlass::gemm::warp::MmaTensorOpMultiplicandTileIteratorTransAiu<
    cutlass::MatrixShape<ThreadblockShape::kN, ThreadblockShape::kK>,
    cutlass::MatrixShape<WarpShape::kN, WarpShape::kK>,
    ElementB,
    cutlass::MatrixShape<InstructionShape::kN, InstructionShape::kK>,
    true
  >;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmMultistageAiu<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    CacheOpA,
    IteratorB,
    SmemIteratorB,
    CacheOpB,
    MmaPolicy,
    Stages,
    WarpTileIteratorA,
    WarpTileIteratorB
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
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kDgrad,
    Conv2dProblemSize,
    true,
    true
  >;
};

#endif

} // namespace kernel
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

