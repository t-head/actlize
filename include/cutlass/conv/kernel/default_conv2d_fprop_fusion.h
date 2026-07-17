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

#include "cutlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_analytic.h"
#include "cutlass/conv/threadblock/conv2d_fprop_filter_tile_access_iterator_analytic.h"
#include "cutlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_optimized.h"
#include "cutlass/conv/threadblock/conv2d_fprop_filter_tile_access_iterator_optimized.h"
#include "cutlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_optimized_small_channel.h"
#include "cutlass/conv/threadblock/conv2d_fprop_filter_tile_access_iterator_optimized_small_channel.h"
#include "cutlass/conv/threadblock/predicated_scale_bias_vector_access_iterator.h"
#include "cutlass/transform/threadblock/regular_scale_bias_vector_access_iterator.h"
#include "cutlass/gemm/warp/scale_bias_tile_iterator.h"

#if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__))
// include here to avoid sparse gemm use aiu swizzle
#include "cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op_aiu.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator_aiu.h"
#include "cutlass/conv/threadblock/aiu/conv2d_fprop_activation_tile_access_iterator_aiu.h"
#include "cutlass/conv/threadblock/aiu/conv2d_fprop_filter_tile_access_iterator_aiu.h"
#include "cutlass/conv/threadblock/aiu/conv2d_fprop_activation_tile_access_iterator_optimized_small_channel_aiu.h"
#include "cutlass/conv/threadblock/aiu/conv2d_fprop_filter_tile_access_iterator_optimized_small_channel_aiu.h"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Defines a kernel for fused batch-norm & Relu and Conv2dFprop
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementScaleBias,
  typename LayoutScaleBias,
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
  int KernelSize = 0,
  conv::KernelType kType = KernelType::kNormal
> struct DefaultConv2dFpropFusion;

/////////////////////////////////////////////////////////////////////////////////////////////////
//                         OpClassTensorOp convolutions
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv2dFprop specialzation for Analytic IteratorAlgorithm and multistage
/// pipeline.
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementScaleBias,
  typename LayoutScaleBias,
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
struct DefaultConv2dFpropFusion <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementScaleBias,
  LayoutScaleBias,
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
  IteratorAlgorithm::kAnalytic,
  StrideSupport::kStrided,
  AlignmentA,
  AlignmentB
> {

  // Define the core components from GEMM
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
      Stages, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = cutlass::AlignedArray<ElementA, AlignmentA>;
  using IteratorA =
    cutlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorAnalytic<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      ElementA, LayoutA,
      ThreadMapA,
      AccessTypeA
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = cutlass::AlignedArray<ElementB, AlignmentB>;
  using IteratorB =
    cutlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorAnalytic<
      cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
      ElementB, LayoutB,
      ThreadMapB,
      AccessTypeB
    >;

  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Define iterators over tiles from scale/bias vectors
  using IteratorScaleBias =
      cutlass::conv::threadblock::PredicatedScaleBiasVectorAccessIterator<
          cutlass::MatrixShape<1, ThreadblockShape::kK>, ElementScaleBias,
          LayoutScaleBias>;

  using SmemIteratorScaleBias =
      cutlass::transform::threadblock::RegularScaleBiasVectorAccessIterator<
          cutlass::MatrixShape<1, ThreadblockShape::kK>, ElementScaleBias,
          LayoutScaleBias>;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Warp-level iterators to load scale and bias vectors
  static int const kThreadCount = 32;
  using WarpIteratorScaleBias = cutlass::gemm::warp::ScaleBiasTileIterator<
      MatrixShape<WarpShape::kM, WarpShape::kK>, ElementScaleBias,
      LayoutScaleBias, MatrixShape<InstructionShape::kM, InstructionShape::kK>,
      typename WarpMmaTensorOp::IteratorA::Base::Policy, kThreadCount,
      MmaCore::WarpCount::kK>;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmFpropFusionMultistage<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    arch::CacheOperation::Always,
    IteratorB,
    SmemIteratorB,
    arch::CacheOperation::Always,
    IteratorScaleBias,
    SmemIteratorScaleBias,
    arch::CacheOperation::Always,
    MmaPolicy,
    WarpIteratorScaleBias,
    Stages
  >;

  // Define the epilogue
  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueTensorOp<
    ThreadblockShape,
    WarpMmaTensorOp,
    kPartitionsK,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolutionFusion<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv2dFprop specialzation for Optimzed IteratorAlgorithm and
/// multistage pipeline.
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementScaleBias,
  typename LayoutScaleBias,
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
  int AlignmentB,
  int KernelSize
>
struct DefaultConv2dFpropFusion <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementScaleBias,
  LayoutScaleBias,
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
  AlignmentA,
  AlignmentB,
  KernelSize
> {

  static bool const Transpose = platform::is_same<LayoutC, cutlass::layout::TensorNCHW>::value ? true : false;

  static bool const Strided = false;

  // Define the core components from GEMM
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
    ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
    ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
    Stages, MathOperatorTag
  >;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using IteratorA =
    cutlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorOptimized<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      ElementA,
      LayoutA,
      ThreadMapA,
      AlignmentA,
      KernelSize
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using IteratorB =
    cutlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorOptimized<
      cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
      ElementB,
      LayoutB,
      ThreadMapB,
      AlignmentB,
      KernelSize
    >;

  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Define iterators over tiles from scale/bias vectors
  using IteratorScaleBias =
      cutlass::conv::threadblock::PredicatedScaleBiasVectorAccessIterator<
          cutlass::MatrixShape<1, ThreadblockShape::kK>, ElementScaleBias,
          LayoutScaleBias>;

  using SmemIteratorScaleBias =
      cutlass::transform::threadblock::RegularScaleBiasVectorAccessIterator<
          cutlass::MatrixShape<1, ThreadblockShape::kK>, ElementScaleBias,
          LayoutScaleBias>;

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

  // Warp-level iterators to load scale and bias vectors
  static int const kThreadCount = 32;
  using WarpIteratorScaleBias = cutlass::gemm::warp::ScaleBiasTileIterator<
      MatrixShape<WarpShape::kM, WarpShape::kK>, ElementScaleBias,
      LayoutScaleBias, MatrixShape<InstructionShape::kM, InstructionShape::kK>,
      typename WarpMmaTensorOp::IteratorA::Base::Policy, kThreadCount,
      MmaCore::WarpCount::kK>;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmFpropFusionMultistage<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    CacheOpA,
    IteratorB,
    SmemIteratorB,
    CacheOpB,
    IteratorScaleBias,
    SmemIteratorScaleBias,
    arch::CacheOperation::Always,
    MmaPolicy,
    WarpIteratorScaleBias,
    Stages,
    true,
    SkipWarps
  >;

  // Define the epilogue
  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueTensorOp<
    ThreadblockShape,
    WarpMmaTensorOp,
    kPartitionsK,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount,
    Strided,
    Transpose
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolutionFusion<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
