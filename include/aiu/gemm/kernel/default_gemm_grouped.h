/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved. 
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 **************************************************************************************************/

/*! \file
    \brief
      Default kernel-level GEMM definitions combine threadblock-scoped matrix multiply-add with
      the appropriate threadblock-scoped epilogue.

      Note, CUTLASS epilogues universally target row-major outputs. Column-major outputs are
      accommodated by exchanging A and B operands and assuming transposed layouts. Partial
      specializations here choose 'device::GemmTransposed' to implement this functionality.

*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/complex.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

#include "aiu/gemm/threadblock/default_mma.h"

#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/default_gemm_complex.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"

#include "cutlass/layout/permute.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace aiu {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Whether the schedule of problems to visit has been precomputed
    cutlass::gemm::kernel::GroupScheduleMode GroupScheduleMode_ = cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
    /// Operation performed by GEMM
    typename Operator = typename cutlass::gemm::device::DefaultGemmConfiguration<
        OperatorClass, ArchTag, ElementA_, ElementB_, ElementC_,
        ElementAccumulator>::Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    /// Not used by PPU-cutlass yet
    cutlass::gemm::SharedMemoryClearOption SharedMemoryClear = cutlass::gemm::SharedMemoryClearOption::kNone,
    /// Permute result D
    /// Not used by PPU-cutlass yet
    typename PermuteDLayout = cutlass::layout::NoPermute,
    ///
    typename Enable = void
    >
struct DefaultGemmGrouped;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Real-valued GEMM kernels
//

template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Layout type for C and D matrix operands
    typename LayoutC,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Whether the schedule of problems to visit has been precomputed
    cutlass::gemm::kernel::GroupScheduleMode GroupScheduleMode_,
    /// Operation performed by GEMM
    typename Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    cutlass::gemm::SharedMemoryClearOption SharedMemoryClear,
    /// Permute result D
    typename PermuteDLayout
>
struct DefaultGemmGrouped<
  ElementA,
  LayoutA,
  kAlignmentA,
  ElementB,
  LayoutB,
  kAlignmentB,
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
  GroupScheduleMode_,
  Operator,
  SharedMemoryClear,
  PermuteDLayout,
  typename cutlass::platform::enable_if< ! cutlass::is_complex<ElementAccumulator>::value>::type
> {

  // If true, we must construct a 'transposed-and-exchanged' Mma operator.
  static bool const kInternalTranspose = cutlass::platform::is_same<LayoutC, cutlass::layout::ColumnMajor>::value;

  using MapArguments = cutlass::gemm::kernel::detail::MapArguments<
    ElementA,
    LayoutA,
    kAlignmentA,
    ElementB,
    LayoutB,
    kAlignmentB,
    LayoutC,
    kInternalTranspose
  >;

  // Define the default GEMM kernel
  // using DefaultGemmKernel = typename kernel::DefaultGemm<
  //   typename MapArguments::ElementA,
  //   typename MapArguments::LayoutA,
  //   MapArguments::kAlignmentA,
  //   typename MapArguments::ElementB,
  //   typename MapArguments::LayoutB,
  //   MapArguments::kAlignmentB,
  //   ElementC,
  //   typename MapArguments::LayoutC,
  //   ElementAccumulator,
  //   Class,
  //   ArchTag,
  //   ThreadblockShape,
  //   WarpShape,
  //   InstructionShape,
  //   EpilogueOutputOp,
  //   ThreadblockSwizzle,
  //   Stages,
  //   true,
  //   Operator
  // >::GemmKernel;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
                            WarpShape, InstructionShape,
                            typename MapArguments::ElementA, typename MapArguments::LayoutA,
                            typename MapArguments::ElementB, typename MapArguments::LayoutB,
                            ElementAccumulator, cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>::Type;

  /// Define the epilogue
  using Epilogue =
      typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
          ThreadblockShape, WarpMmaTensorOp, kPartitionsK, EpilogueOutputOp,
          EpilogueOutputOp::kCount>::Epilogue;

  /// Wmma Fragment type for A and B (for FP32 tensorcell on PPU)
  using WmmaFragABType = typename ToTF32<typename MapArguments::ElementA>::type;

  /// Define the threadblock-scoped matrix multiply-accumulate
  using Mma = typename aiu::gemm::threadblock::DefaultMma<
      typename MapArguments::ElementA, typename MapArguments::LayoutA,
      typename MapArguments::ElementB, typename MapArguments::LayoutB,
      ElementAccumulator, cutlass::layout::RowMajor, cutlass::arch::OpClassTensorOp, cutlass::arch::PPU0010,
      ThreadblockShape, WarpShape, InstructionShape, Stages,
      Operator, Epilogue, WmmaFragABType>::ThreadblockMma;

    /// Define the kernel in terms of the default kernel
  using GemmKernel = cutlass::gemm::kernel::GemmGrouped<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    GroupScheduleMode_,
    kInternalTranspose
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

//
// Complex-valued GEMM kernels
//

template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Layout type for C and D matrix operands
    typename LayoutC,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Whether the schedule of problems to visit has been precomputed
    cutlass::gemm::kernel::GroupScheduleMode GroupScheduleMode_,
    /// Operation performed by GEMM
    typename Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    cutlass::gemm::SharedMemoryClearOption SharedMemoryClear
  >
struct DefaultGemmGrouped<
  ElementA,
  LayoutA,
  kAlignmentA,
  ElementB,
  LayoutB,
  kAlignmentB,
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
  GroupScheduleMode_,
  Operator,
  SharedMemoryClear,
  cutlass::layout::NoPermute, /*PermuteDLayout*/
  typename cutlass::platform::enable_if<cutlass::is_complex<ElementAccumulator>::value>::type
> {

  // If true, we must construct a 'transposed-and-exchanged' Mma operator.
  static bool const kInternalTranspose = cutlass::platform::is_same<LayoutC, cutlass::layout::ColumnMajor>::value;

  using MapArguments = cutlass::gemm::kernel::detail::MapArguments<
    ElementA,
    LayoutA,
    kAlignmentA,
    ElementB,
    LayoutB,
    kAlignmentB,
    LayoutC,
    kInternalTranspose
  >;

  // using DefaultGemmKernel = typename kernel::DefaultGemmComplex<
  //   typename MapArguments::ElementA,
  //   typename MapArguments::LayoutA,
  //   typename MapArguments::ElementB,
  //   typename MapArguments::LayoutB,
  //   ElementC,
  //   typename MapArguments::LayoutC,
  //   ElementAccumulator,
  //   OperatorClass,
  //   ArchTag,
  //   ThreadblockShape,
  //   WarpShape,
  //   InstructionShape,
  //   EpilogueOutputOp,
  //   ThreadblockSwizzle,
  //   Stages,
  //   MapArguments::kTransformA,
  //   MapArguments::kTransformB,
  //   Operator,
  //   false
  // >::GemmKernel;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
    WarpShape, InstructionShape, ElementA, LayoutA, ElementB, LayoutB, ElementAccumulator,
    cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>::Type;

  /// Define the epilogue
  using Epilogue =
      typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
          ThreadblockShape, WarpMmaTensorOp, kPartitionsK, EpilogueOutputOp,
          EpilogueOutputOp::kCount>::Epilogue;

  /// Wmma Fragment type for A and B (for FP32 tensorcell on PPU)
  using WmmaFragABType = typename ToTF32<ElementA>::type;

  /// Define the threadblock-scoped matrix multiply-accumulate
  using Mma = typename aiu::gemm::threadblock::DefaultMma<
      ElementA, LayoutA, ElementB, LayoutB,
      ElementAccumulator, cutlass::layout::RowMajor, cutlass::arch::OpClassTensorOp, cutlass::arch::PPU0010,
      ThreadblockShape, WarpShape, InstructionShape, Stages,
      Operator, Epilogue, WmmaFragABType>::ThreadblockMma;

  /// Define the kernel in terms of the default kernel
  using GemmKernel = cutlass::gemm::kernel::GemmGrouped<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    GroupScheduleMode_,
    kInternalTranspose
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
