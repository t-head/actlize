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
    \brief Template for a pipelined GEMM kernel. Does not compute batching or support split-K.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/device_kernel.h"

#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/gemm/kernel/gemm_batched.h"

#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"

#include "cutlass/gemm/kernel/default_gemm_splitk_parallel.h"

#include "cutlass/epilogue/thread/conversion_op.h"
#include "cutlass/reduction/kernel/reduce_split_k.h"
#include "cutlass/reduction/thread/reduction_operators.h"

#include "aiu/gemm/kernel/default_gemm.h"
#include "aiu/gemm/kernel/default_gemm_splitk_parallel.h"
#include "aiu/gemm/device/default_gemm_configuration.h"
#include "aiu/gemm/tool/cutlass_type_convert.h"
//#include "../tool/launch_bound.h"

using GemmCoord = cutlass::gemm::GemmCoord;
////////////////////////////////////////////////////////////////////////////////

namespace aiu {
namespace gemm {
namespace device {

////////////////////////////////////////////////////////////////////////////////

/*! Gemm device-level operator. This is an interface to efficient CUTLASS GEMM kernels that may
  be invoked from host code.

  The contributions of this class are:

    1. At compile time, it maps data types and high-level structural parameters onto
       specific CUTLASS components.

    2. At runtime, it maps logical arguments to GEMM problems to kernel parameters.

    3. At runtime, it launches kernels on the device.

  The intent is to provide a convenient mechanism for interacting with most plausible GEMM
  configurations for each supported architecture. Consequently, not all parameters are exposed
  to the top-level interface. Rather, sensible defaults at each level of the CUTLASS hierarchy
  are selected to tradeoff simplicity of the interface with flexibility. We expect
  most configurations to be specified at this level. Applications with more exotic requirements
  may construct their kernels of interest using CUTLASS components at the threadblock, warp,
  and thread levels of abstraction.

  CUTLASS exposes computations using the functor design pattern in which objects compose some
  internal state with an overloaded function call operator. This enables decoupling of
  initialization from execution, possibly reducing overhead during steady state phases of
  application execution.

  CUTLASS device-level operators expose an Arguments structure encompassing each logical
  input to the computation. This is distinct from the kernel-level Params structure pattern
  which contains application-specific precomputed state needed by the device code.

  Example of a CUTLASS GEMM operator implementing the functionality of acBLAS's SGEMM NN
  is as follows:

    //
    // Instantiate the CUTLASS GEMM operator.
    //

    cutlass::gemm::device::Gemm<
      float,
      cutlass::layout::ColumnMajor,
      float,
      cutlass::layout::ColumnMajor,
      float,
      cutlass::layout::ColumnMajor
    > gemm_op;

    //
    // Launch the GEMM operation on the device
    //

    cutlass::Status status = gemm_op({
      {m, n, k},                          // GemmCoord problem_size,
      {A, lda},                           // cutlass::TensorRef<float, cutlass::layout::ColumnMajor> ref_A,
      {B, ldb},                           // cutlass::TensorRef<float, cutlass::layout::ColumnMajor> ref_B,
      {C, ldc},                           // cutlass::TensorRef<float, cutlass::layout::ColumnMajor> ref_C,
      {D, ldd},                           // cutlass::TensorRef<float, cutlass::layout::ColumnMajor> ref_D,
      {alpha, beta}                       // EpilogueOutputOp::Params epilogue_op_params
    });


  A simplified view of the template is listed below.

    template <
      /// Element type for A matrix operand
      typename ElementA,

      /// Layout type for A matrix operand
      typename LayoutA,

      /// Element type for B matrix operand
      typename ElementB,

      /// Layout type for B matrix operand
      typename LayoutB,

      /// Element type for C and D matrix operands
      typename ElementC,

      /// Layout type for C and D matrix operands
      typename LayoutC,

      /// Element type for internal accumulation
      typename ElementAccumulator,

      /// Operator class tag
      typename OperatorClass,

      /// Tag indicating architecture to tune for.  This is the minimum CU that
      /// supports the intended feature. The device kernel can be built
      /// targeting any CU larger than this number.
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
      int Stages
    >
    class Gemm;
*/
template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Element type for internal accumulation
    typename ElementAccumulator_ = ElementC_,
    /// Operator class tag
    typename OperatorClass_ = cutlass::arch::OpClassSimt,
    /// Tag indicating architecture to tune for
    typename ArchTag_ = cutlass::arch::PPU0010,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle_ = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kStages,
    /// Access granularity of A matrix in units of elements
    int AlignmentA = 1,
    /// Access granularity of B matrix in units of elements
    int AlignmentB = 1,
    /// Operation performed by GEMM
    typename Operator_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::Operator,
    /// Wmma Fragment type for A and B (for FP32 tensorcell on PPU)
    typename WmmaFragABType = typename ToTF32<ElementA_>::type
>
class GemmBatched {
 public:

  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using TensorRefA = cutlass::TensorRef<ElementA const, LayoutA>;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using TensorRefB = cutlass::TensorRef<ElementB const, LayoutB>;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using TensorRefC = cutlass::TensorRef<ElementC const, LayoutC>;
  #if SAIL_FUSE_OP_EXT
  using ElementFuseInExtra = ElementC;
  using LayoutExtra = LayoutC_;
  using TensorRefExtra = cutlass::TensorRef<ElementFuseInExtra const, LayoutExtra>;
  #endif
  using TensorRefD = cutlass::TensorRef<ElementC, LayoutC>;
  using ElementAccumulator = ElementAccumulator_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using EpilogueOutputOp = EpilogueOutputOp_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static int const kStages = Stages;
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;
  static int const kAlignmentC = EpilogueOutputOp::kCount;
  using Operator = Operator_;

  #if SAIL_FUSE_OP_EXT
  static int const kExtraInputNum = EpilogueOutputOp::kExtraEpilogueInputs > 0 ? EpilogueOutputOp::kExtraEpilogueInputs : 1;
  static int const kExtraInputLoopNum = cutlass::epilogue::GetExtraEpilogueBinaryInputs<EpilogueOutputOp>::value;
  #endif

  /// Define the kernel
  using DefaultGemmKernel = typename kernel::DefaultGemm<
    ElementA,
    LayoutA,
    AlignmentA,
    ElementB,
    LayoutB,
    AlignmentB,
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
    kStages,
    false,
    Operator,
    WmmaFragABType
  >::GemmKernel;

  using GemmKernel = cutlass::gemm::kernel::GemmBatched<typename DefaultGemmKernel::Mma, typename DefaultGemmKernel::Epilogue, ThreadblockSwizzle>;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    GemmCoord problem_size;
    cutlass::TensorRef<ElementA const, LayoutA> ref_A;
    int64_t stride_A;
    cutlass::TensorRef<ElementB const, LayoutB> ref_B;
    int64_t stride_B;
    cutlass::TensorRef<ElementC const, LayoutC> ref_C;
    int64_t stride_C;
    #if SAIL_FUSE_OP_EXT
    TensorRefExtra ref_Extra[kExtraInputNum];
    int64_t stride_Epilogues[kExtraInputNum];
    #endif
    cutlass::TensorRef<ElementC, LayoutC> ref_D;
    int64_t stride_D;
    typename EpilogueOutputOp::Params epilogue;
    int batch_count;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments() { }

    /// Constructs an Arguments structure
    CUTLASS_HOST_DEVICE
    Arguments(
      GemmCoord problem_size_,
      cutlass::TensorRef<ElementA const, LayoutA> ref_A_,
      int64_t stride_A_,
      cutlass::TensorRef<ElementB const, LayoutB> ref_B_,
      int64_t stride_B_,
      cutlass::TensorRef<ElementC const, LayoutC> ref_C_,
      int64_t stride_C_,
      cutlass::TensorRef<ElementC, LayoutC> ref_D_,
      int64_t stride_D_,
      typename EpilogueOutputOp::Params epilogue_,
      int batch_count_
      #if SAIL_FUSE_OP_EXT
      , TensorRefExtra *pref_Extra_ = nullptr
      , int64_t *pstride_Epilogues_ = nullptr
      #endif
    ):
      problem_size(problem_size_),
      ref_A(ref_A_),
      stride_A(stride_A_),
      ref_B(ref_B_),
      stride_B(stride_B_),
      ref_C(ref_C_),
      stride_C(stride_C_),
      ref_D(ref_D_),
      stride_D(stride_D_),
      epilogue(epilogue_),
      batch_count(batch_count_) {
        #if SAIL_FUSE_OP_EXT
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kExtraInputLoopNum; i++) {
          if (pref_Extra_) {
            ref_Extra[i] = pref_Extra_[i];
          } else {
            ref_Extra[i] = {nullptr, 0};
          }
          if (pstride_Epilogues_) {
            stride_Epilogues[i] = pstride_Epilogues_[i];
          } else {
            stride_Epilogues[i] = 0;
          }
        }
        #endif
      }
  };

private:

  /// Kernel parameters object
  typename GemmKernel::Params params_;

public:

  /// Constructs the GEMM.
  GemmBatched() { }

  /// Determines whether the GEMM can execute the given problem.
  static cutlass::Status can_implement(Arguments const &args) {

    if (!TensorRef_aligned(args.ref_C, kAlignmentC) || (args.stride_C % kAlignmentC)) {
      return cutlass::Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(args.ref_D, kAlignmentC) || (args.stride_D % kAlignmentC)) {
      return cutlass::Status::kErrorMisalignedOperand;
    }

    if ((args.problem_size.m() % kAlignmentC) || (args.problem_size.n() % kAlignmentC)) {

      return cutlass::Status::kErrorMisalignedOperand;
    }

    #if SAIL_FUSE_OP_EXT
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kExtraInputLoopNum; i++) {
      if (args.ref_Extra[i].good() && (!TensorRef_aligned(args.ref_Extra[i], kAlignmentC) || (args.stride_Epilogues[i] % kAlignmentC))) {
        return cutlass::Status::kErrorMisalignedOperand;
      }
    }
    #endif

    return cutlass::Status::kSuccess;
  }

  /// Gets the workspace size
  static cutlass::CUsize get_workspace_size(Arguments const &args) {
    return 0;
  }

  /// Initializes GEMM state from arguments.
  cutlass::Status initialize(Arguments const &args, void *workspace = nullptr, hggcStream_t stream = nullptr) {

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord grid_shape = threadblock_swizzle.get_tiled_shape(
      args.problem_size,
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
      args.batch_count);

    #if SAIL_FUSE_OP_EXT
    cutlass::TensorRef<ElementFuseInExtra, LayoutExtra> ref_Extra_NC[kExtraInputNum];
    int64_t stride_Epilogues_NC[kExtraInputNum];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kExtraInputLoopNum; i++) {
      ref_Extra_NC[i] = args.ref_Extra[i].non_const_ref();
      stride_Epilogues_NC[i] = args.stride_Epilogues[i];
    }
    #endif

    // Initialize the Params structure
    params_ = typename GemmKernel::Params{
      args.problem_size,
      grid_shape,
      args.ref_A.non_const_ref(),
      args.stride_A,
      args.ref_B.non_const_ref(),
      args.stride_B,
      args.ref_C.non_const_ref(),
      args.stride_C,
      args.ref_D,
      args.stride_D,
      args.epilogue,
      args.batch_count
      #if SAIL_FUSE_OP_EXT
      , ref_Extra_NC
      , stride_Epilogues_NC
      #endif
    };

    return cutlass::Status::kSuccess;
  }

  /// Lightweight update given a subset of arguments
  cutlass::Status update(Arguments const &args, void *workspace = nullptr) {

    params_.ref_A.reset(args.ref_A.non_const_ref().data());
    params_.ref_B.reset(args.ref_B.non_const_ref().data());
    params_.ref_C.reset(args.ref_C.non_const_ref().data());
    params_.ref_D.reset(args.ref_D.data());
    #if SAIL_FUSE_OP_EXT
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kExtraInputLoopNum; i++) {
      params_.ref_Extra[i] = args.ref_Extra[i].non_const_ref();
    }
    #endif
    params_.epilogue = args.epilogue;

    return cutlass::Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  cutlass::Status run(hggcStream_t stream = nullptr) {

    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
    dim3 block(GemmKernel::kThreadCount, 1, 1);

    hggcError_t result;

    int smem_size = int(sizeof(typename GemmKernel::SharedStorage));
    if (smem_size >= (48 << 10)) {
      result = hggcFuncSetAttribute(cutlass::Kernel<GemmKernel>,
                                    hggcFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (result != hggcSuccess) {
        return cutlass::Status::kErrorInternal;
      }

      result = hggcFuncSetAttribute(
          cutlass::Kernel<GemmKernel>,
          hggcFuncAttributePreferredSharedMemoryCarveout, 100);

      if (result != hggcSuccess) {
        return cutlass::Status::kErrorInternal;
      }
    }

    cutlass::Kernel<GemmKernel><<<grid, block, smem_size, stream>>>(params_);

    result = hggcGetLastError();

    return result == hggcSuccess ? cutlass::Status::kSuccess : cutlass::Status::kErrorInternal;
  }

  /// Runs the kernel using initialized state.
  cutlass::Status operator()(hggcStream_t stream = nullptr) {
    return run(stream);
  }

  /// Runs the kernel using initialized state.
  cutlass::Status operator()(
    Arguments const &args,
    void *workspace = nullptr,
    hggcStream_t stream = nullptr) {

    cutlass::Status status = initialize(args, workspace, stream);

    if (status == cutlass::Status::kSuccess) {
      status = run(stream);
    }

    return status;
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Parital specialization for column-major output exchanges problem size and operand.
template <
  /// Element type for A matrix operand
  typename ElementA_,
  /// Layout type for A matrix operand
  typename LayoutA_,
  /// Element type for B matrix operand
  typename ElementB_,
  /// Layout type for B matrix operand
  typename LayoutB_,
  /// Element type for C and D matrix operands
  typename ElementC_,
  /// Element type for internal accumulation
  typename ElementAccumulator_,
  /// Operator class tag
  typename OperatorClass_,
  /// Tag indicating architecture to tune for
  typename ArchTag_,
  /// Threadblock-level tile size (concept: GemmShape)
  typename ThreadblockShape_,
  /// Warp-level tile size (concept: GemmShape)
  typename WarpShape_,
  /// Warp-level tile size (concept: GemmShape)
  typename InstructionShape_,
  /// Epilogue output operator
  typename EpilogueOutputOp_,
  /// Threadblock-level swizzling operator
  typename ThreadblockSwizzle_,
  /// Number of stages used in the pipelined mainloop
  int Stages,
  /// Access granularity of A matrix in units of elements
  int AlignmentA,
  /// Access granularity of B matrix in units of elements
  int AlignmentB,
  typename Operator_,
  /// Wmma Fragment type for A and B (for FP32 tensorcell on PPU)
  typename WmmaFragABType
>
class GemmBatched<
  ElementA_,
  LayoutA_,
  ElementB_,
  LayoutB_,
  ElementC_,
  cutlass::layout::ColumnMajor,
  ElementAccumulator_,
  OperatorClass_,
  ArchTag_,
  ThreadblockShape_,
  WarpShape_,
  InstructionShape_,
  EpilogueOutputOp_,
  ThreadblockSwizzle_,
  Stages,
  AlignmentA,
  AlignmentB,
  Operator_,
  WmmaFragABType
> {
public:

  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using TensorRefA = cutlass::TensorRef<ElementA const, LayoutA>;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using TensorRefB = cutlass::TensorRef<ElementB const, LayoutB>;
  using ElementC = ElementC_;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TensorRefC = cutlass::TensorRef<ElementC const, LayoutC>;
  #if SAIL_FUSE_OP_EXT
  using ElementFuseInExtra = ElementC;
  using LayoutExtra = LayoutC;
  using TensorRefExtra = cutlass::TensorRef<ElementFuseInExtra const, LayoutExtra>;
  #endif
  using TensorRefD = cutlass::TensorRef<ElementC, LayoutC>;
  using ElementAccumulator = ElementAccumulator_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using EpilogueOutputOp = EpilogueOutputOp_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static int const kStages = Stages;

  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;
  static int const kAlignmentC = EpilogueOutputOp::kCount;
  static bool const kSplitKSerial = false;
  using Operator = Operator_;

  #if SAIL_FUSE_OP_EXT
  static int const kExtraInputNum = EpilogueOutputOp::kExtraEpilogueInputs > 0 ? EpilogueOutputOp::kExtraEpilogueInputs : 1;
  static int const kExtraInputLoopNum = cutlass::epilogue::GetExtraEpilogueBinaryInputs<EpilogueOutputOp>::value;
  #endif

  //
  using UnderlyingOperator = GemmBatched<
    ElementB,
    typename cutlass::layout::LayoutTranspose<LayoutB>::type,
    ElementA,
    typename cutlass::layout::LayoutTranspose<LayoutA>::type,
    ElementC,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    kAlignmentB,
    kAlignmentA,
    Operator_,
    WmmaFragABType
  >;

  using UnderlyingArguments = typename UnderlyingOperator::Arguments;
  using GemmKernel = typename UnderlyingOperator::GemmKernel;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    GemmCoord problem_size;
    cutlass::TensorRef<ElementA const, LayoutA> ref_A;
    int64_t stride_A;
    cutlass::TensorRef<ElementB const, LayoutB> ref_B;
    int64_t stride_B;
    cutlass::TensorRef<ElementC const, LayoutC> ref_C;
    int64_t stride_C;
    #if SAIL_FUSE_OP_EXT
    TensorRefExtra ref_Extra[kExtraInputNum];
    int64_t stride_Epilogues[kExtraInputNum];
    #endif
    cutlass::TensorRef<ElementC, LayoutC> ref_D;
    int64_t stride_D;
    typename EpilogueOutputOp::Params epilogue;
    int batch_count;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments() { }

    /// Constructs an Arguments structure
    CUTLASS_HOST_DEVICE
    Arguments(
      GemmCoord problem_size_,
      cutlass::TensorRef<ElementA const, LayoutA> ref_A_,
      int64_t stride_A_,
      cutlass::TensorRef<ElementB const, LayoutB> ref_B_,
      int64_t stride_B_,
      cutlass::TensorRef<ElementC const, LayoutC> ref_C_,
      int64_t stride_C_,
      cutlass::TensorRef<ElementC, LayoutC> ref_D_,
      int64_t stride_D_,
      typename EpilogueOutputOp::Params epilogue_,
      int batch_count_
      #if SAIL_FUSE_OP_EXT
      , TensorRefExtra *pref_Extra_ = nullptr
      , int64_t *pstride_Epilogues = nullptr
      #endif
    ):
      problem_size(problem_size_),
      ref_A(ref_A_),
      stride_A(stride_A_),
      ref_B(ref_B_),
      stride_B(stride_B_),
      ref_C(ref_C_),
      stride_C(stride_C_),
      ref_D(ref_D_),
      stride_D(stride_D_),
      epilogue(epilogue_),
      batch_count(batch_count_) {
        #if SAIL_FUSE_OP_EXT
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < kExtraInputLoopNum; i++) {
            if (pref_Extra_) {
              ref_Extra[i] = pref_Extra_[i];
            } else {
              ref_Extra[i] = {nullptr, 0};
            }
            if (pstride_Epilogues) {
              stride_Epilogues[i] = pstride_Epilogues[i];
            } else {
              stride_Epilogues[i] = 0;
            }
          }
        #endif
      }
  };

private:

  UnderlyingOperator underlying_operator_;

public:

  /// Constructs the GEMM.
  GemmBatched() { }

  /// Helper to construct a transposed equivalent for the underying GEMM operator
  static UnderlyingArguments to_underlying_arguments(Arguments const &args) {

    #if SAIL_FUSE_OP_EXT
    cutlass::TensorRef<typename UnderlyingOperator::ElementFuseInExtra const, typename UnderlyingOperator::LayoutExtra> ref_Extra_NC[kExtraInputNum];
    int64_t stride_Epilogues_NC[kExtraInputNum];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kExtraInputLoopNum; i++) {
      ref_Extra_NC[i] = {args.ref_Extra[i].non_const_ref().data(), args.ref_Extra[i].non_const_ref().stride(0)};
      stride_Epilogues_NC[i] = args.stride_Epilogues[i];
    }
    #endif

    return UnderlyingArguments(
      {args.problem_size.n(), args.problem_size.m(), args.problem_size.k()},
      {args.ref_B.data(), args.ref_B.stride(0)},
      args.stride_B,
      {args.ref_A.data(), args.ref_A.stride(0)},
      args.stride_A,
      {args.ref_C.data(), args.ref_C.stride(0)},
      args.stride_C,
      {args.ref_D.data(), args.ref_D.stride(0)},
      args.stride_D,
      args.epilogue,
      args.batch_count
      #if SAIL_FUSE_OP_EXT
      ,ref_Extra_NC
      ,stride_Epilogues_NC
      #endif
    );
  }

  /// Determines whether the GEMM can execute the given problem.
  static cutlass::Status can_implement(Arguments const &args) {

    return UnderlyingOperator::can_implement(to_underlying_arguments(args));
  }

  /// Gets the workspace size
  static cutlass::CUsize get_workspace_size(Arguments const &args) {

    return UnderlyingOperator::get_workspace_size(to_underlying_arguments(args));
  }

  /// Initializes GEMM state from arguments.
  cutlass::Status initialize(Arguments const &args, void *workspace = nullptr, hggcStream_t stream = nullptr) {

    return underlying_operator_.initialize(to_underlying_arguments(args), workspace, stream);
  }

  /// Lightweight update given a subset of arguments
  cutlass::Status update(Arguments const &args, void *workspace = nullptr) {

    return underlying_operator_.update(to_underlying_arguments(args), workspace);
  }

  /// Runs the kernel using initialized state.
  cutlass::Status run(hggcStream_t stream = nullptr) {

    return underlying_operator_.run(stream);
  }

  /// Runs the kernel using initialized state.
  cutlass::Status operator()(hggcStream_t stream = nullptr) {
    return run(stream);
  }

  /// Runs the kernel using initialized state.
  cutlass::Status operator()(
    Arguments const &args,
    void *workspace = nullptr,
    hggcStream_t stream = nullptr) {

    cutlass::Status status = initialize(args, workspace, stream);

    if (status == cutlass::Status::kSuccess) {
      status = run(stream);
    }

    return status;
  }

};

template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Element type for internal accumulation
    typename ElementAccumulator_ = ElementC_,
    /// Operator class tag
    typename OperatorClass_ = cutlass::arch::OpClassSimt,
    /// Tag indicating architecture to tune for
    typename ArchTag_ = cutlass::arch::PPU0010,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::EpilogueOutputOp,
    typename ConvertScaledOp_ = cutlass::epilogue::thread::Convert<
        ElementAccumulator_,
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementAccumulator_,
                                 ElementAccumulator_>::EpilogueOutputOp::kCount,
        ElementAccumulator_>,
    typename ReductionOp_ = cutlass::reduction::thread::ReduceAdd<
        ElementAccumulator_, typename EpilogueOutputOp_::ElementAccumulator,
        EpilogueOutputOp_::kCount>,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle_ = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kStages,
    /// Access granularity of A matrix in units of elements
    int AlignmentA =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kAlignmentA,
    /// Access granularity of B matrix in units of elements
    int AlignmentB =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kAlignmentB,
    /// Operation performed by GEMM
    typename Operator_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::Operator,
    /// Wmma Fragment type for A and B (for FP32 tensorcell on PPU)
    typename WmmaFragABType = typename ToTF32<ElementA_>::type
>
class GemmBatchedReduceParallel {
 public:

  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using TensorRefA = cutlass::TensorRef<ElementA const, LayoutA>;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using TensorRefB = cutlass::TensorRef<ElementB const, LayoutB>;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using TensorRefC = cutlass::TensorRef<ElementC const, LayoutC>;
  using TensorRefD = cutlass::TensorRef<ElementC, LayoutC>;
  using ElementAccumulator = ElementAccumulator_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using EpilogueOutputOp = EpilogueOutputOp_;
  using ConvertScaledOp = ConvertScaledOp_;
  using ReductionOp = ReductionOp_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static int const kStages = Stages;
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;
  static int const kAlignmentC = EpilogueOutputOp::kCount;
  using Operator = Operator_;


  /// Define the kernel
  using DefaultGemmKernel = typename kernel::DefaultGemmSplitKParallel<
    ElementA,
    LayoutA,
    kAlignmentA,
    ElementB,
    LayoutB,
    kAlignmentB,
    ElementAccumulator,
    LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    ConvertScaledOp,
    ThreadblockSwizzle,
    kStages,
    Operator,
    WmmaFragABType
  >::GemmKernel;

  using GemmKernel = typename cutlass::gemm::kernel::GemmBatched<typename DefaultGemmKernel::Mma, typename DefaultGemmKernel::Epilogue, ThreadblockSwizzle>;

  using ReductionKernel = cutlass::reduction::kernel::ReduceSplitK<
    cutlass::MatrixShape<4, 32 * EpilogueOutputOp::kCount>,
    EpilogueOutputOp,
    ReductionOp
  >;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    GemmCoord problem_size;
    cutlass::TensorRef<ElementA const, LayoutA> ref_A;
    int64_t stride_A;
    cutlass::TensorRef<ElementB const, LayoutB> ref_B;
    int64_t stride_B;
    cutlass::TensorRef<ElementC const, LayoutC> ref_C;
    int64_t stride_C;
    cutlass::TensorRef<ElementC, LayoutC> ref_D;
    int64_t stride_D;
    typename EpilogueOutputOp::Params epilogue;
    int batch_count;
    typename ConvertScaledOp::Params convert;
    typename ReductionOp::Params reduction;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments() { }

    /// Constructs an Arguments structure
    CUTLASS_HOST_DEVICE
    Arguments(
      GemmCoord problem_size_,
      cutlass::TensorRef<ElementA const, LayoutA> ref_A_,
      int64_t stride_A_,
      cutlass::TensorRef<ElementB const, LayoutB> ref_B_,
      int64_t stride_B_,
      cutlass::TensorRef<ElementC const, LayoutC> ref_C_,
      int64_t stride_C_,
      cutlass::TensorRef<ElementC, LayoutC> ref_D_,
      int64_t stride_D_,
      typename EpilogueOutputOp::Params epilogue_,
      int batch_count_,
      typename ConvertScaledOp::Params convert_ =
        typename ConvertScaledOp::Params(),
      typename ReductionOp::Params reduction_ =
        typename ReductionOp::Params()
    ):
      problem_size(problem_size_),
      ref_A(ref_A_),
      stride_A(stride_A_),
      ref_B(ref_B_),
      stride_B(stride_B_),
      ref_C(ref_C_),
      stride_C(stride_C_),
      ref_D(ref_D_),
      stride_D(stride_D_),
      epilogue(epilogue_),
      batch_count(batch_count_),
      convert(convert_),
      reduction(reduction_) {
      }
  };

private:

  /// Kernel parameters object
  typename GemmKernel::Params gemm_params_;

  typename ReductionKernel::Params reduction_params_;

public:

  /// Constructs the GEMM.
  GemmBatchedReduceParallel() { }

  /// Determines whether the GEMM can execute the given problem.
  static cutlass::Status can_implement(Arguments const &args) {
    if (args.stride_D != 0) {
      return cutlass::Status::kErrorInvalidProblem;
    }

    if (!TensorRef_aligned(args.ref_A, kAlignmentA) || (args.stride_A % kAlignmentA)) {
      return cutlass::Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(args.ref_B, kAlignmentB) || (args.stride_B % kAlignmentB)) {
      return cutlass::Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(args.ref_C, kAlignmentC) || (args.stride_C % kAlignmentC)) {
      return cutlass::Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(args.ref_D, kAlignmentC) || (args.stride_D % kAlignmentC)) {
      return cutlass::Status::kErrorMisalignedOperand;
    }

    if ((args.problem_size.m() % kAlignmentA) || (args.problem_size.k() % kAlignmentA) ||
      (args.problem_size.n() % kAlignmentB) || (args.problem_size.k() % kAlignmentB) ||
      (args.problem_size.m() % kAlignmentC) || (args.problem_size.n() % kAlignmentC)) {

      return cutlass::Status::kErrorMisalignedOperand;
    }


    return cutlass::Status::kSuccess;
  }

  /// Gets the workspace size
  static cutlass::CUsize get_workspace_size(Arguments const &args) {
    if (args.batch_count > 1) {
      // should reduce output, ensure batch not exceed in conv
      return sizeof(ElementAccumulator_) * size_t(args.problem_size.m()) * size_t(args.problem_size.n()) * args.batch_count;
    } else {
      return 0;
    }
  }

  /// Initializes GEMM state from arguments.
  cutlass::Status initialize(Arguments const &args, void *workspace = nullptr, hggcStream_t stream = nullptr) {

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord grid_shape = threadblock_swizzle.get_tiled_shape(
      args.problem_size,
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
      args.batch_count);

    cutlass::TensorRef<ElementAccumulator_, cutlass::layout::RowMajor> ref_workspace(
      static_cast<ElementAccumulator_ *>(workspace),
      args.problem_size.n());

    int64_t partition_stride = int64_t(args.problem_size.m()) * int64_t(args.problem_size.n());

    // Initialize the Params structure
    gemm_params_ = typename GemmKernel::Params{
      args.problem_size,
      grid_shape,
      args.ref_A.non_const_ref(),
      args.stride_A,
      args.ref_B.non_const_ref(),
      args.stride_B,
      ref_workspace,
      args.stride_C,
      ref_workspace,
      partition_stride,
      args.convert,
      args.batch_count
    };


    reduction_params_ = typename ReductionKernel::Params(
      args.problem_size.mn(),
      grid_shape.k(),
      partition_stride,
      0,
      0,
      ref_workspace,
      args.ref_D,
      args.ref_C.non_const_ref(),
      args.epilogue,
      args.reduction
    );

    return cutlass::Status::kSuccess;
  }

  /// Lightweight update given a subset of arguments
  cutlass::Status update(Arguments const &args, void *workspace = nullptr) {

    gemm_params_.ref_A.reset(args.ref_A.non_const_ref().data());
    gemm_params_.ref_B.reset(args.ref_B.non_const_ref().data());
    gemm_params_.ref_D.reset(static_cast<ElementAccumulator_ *>(workspace));

    reduction_params_.workspace.reset(static_cast<ElementAccumulator_ *>(workspace));
    reduction_params_.destination.reset(args.ref_D.data());
    reduction_params_.source.reset(args.ref_C.non_const_ref().data());
    reduction_params_.output = args.epilogue;
    reduction_params_.reduction = args.reduction;

    return cutlass::Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  cutlass::Status run(hggcStream_t stream = nullptr) {

    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(gemm_params_.grid_tiled_shape);
    dim3 block(GemmKernel::kThreadCount, 1, 1);

    hggcError_t result;

    int smem_size = int(sizeof(typename GemmKernel::SharedStorage));
    if (smem_size >= (48 << 10)) {
      result = hggcFuncSetAttribute(cutlass::Kernel<GemmKernel>,
                                    hggcFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (result != hggcSuccess) {
        return cutlass::Status::kErrorInternal;
      }

      result = hggcFuncSetAttribute(
          cutlass::Kernel<GemmKernel>,
          hggcFuncAttributePreferredSharedMemoryCarveout, 100);

      if (result != hggcSuccess) {
        return cutlass::Status::kErrorInternal;
      }
    }

    cutlass::Kernel<GemmKernel><<<grid, block, smem_size, stream>>>(gemm_params_);

    //
    // Launch reduction kernel
    //

    block = ReductionKernel::block_shape();
    grid = ReductionKernel::grid_shape(gemm_params_.problem_size.mn());

    cutlass::Kernel<ReductionKernel><<< grid, block, 0, stream >>>(reduction_params_);

    result = hggcGetLastError();

    if (result != hggcSuccess) {
      return cutlass::Status::kErrorInternal;
    }

    return result == hggcSuccess ? cutlass::Status::kSuccess : cutlass::Status::kErrorInternal;
  }

  /// Runs the kernel using initialized state.
  cutlass::Status operator()(hggcStream_t stream = nullptr) {
    return run(stream);
  }

  /// Runs the kernel using initialized state.
  cutlass::Status operator()(
    Arguments const &args,
    void *workspace = nullptr,
    hggcStream_t stream = nullptr) {

    cutlass::Status status = initialize(args, workspace, stream);

    if (status == cutlass::Status::kSuccess) {
      status = run(stream);
    }

    return status;
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Parital specialization for column-major output exchanges problem size and operand.
template <
  /// Element type for A matrix operand
  typename ElementA_,
  /// Layout type for A matrix operand
  typename LayoutA_,
  /// Element type for B matrix operand
  typename ElementB_,
  /// Layout type for B matrix operand
  typename LayoutB_,
  /// Element type for C and D matrix operands
  typename ElementC_,
  /// Element type for internal accumulation
  typename ElementAccumulator_,
  /// Operator class tag
  typename OperatorClass_,
  /// Tag indicating architecture to tune for
  typename ArchTag_,
  /// Threadblock-level tile size (concept: GemmShape)
  typename ThreadblockShape_,
  /// Warp-level tile size (concept: GemmShape)
  typename WarpShape_,
  /// Warp-level tile size (concept: GemmShape)
  typename InstructionShape_,
  /// Epilogue output operator
  typename EpilogueOutputOp_,
  typename ConvertScaledOp_,
  typename ReductionOp_,
  /// Threadblock-level swizzling operator
  typename ThreadblockSwizzle_,
  /// Number of stages used in the pipelined mainloop
  int Stages,
  /// Access granularity of A matrix in units of elements
  int AlignmentA,
  /// Access granularity of B matrix in units of elements
  int AlignmentB,
  typename Operator_,
  typename WmmaFragABType
>
class GemmBatchedReduceParallel<
  ElementA_,
  LayoutA_,
  ElementB_,
  LayoutB_,
  ElementC_,
  cutlass::layout::ColumnMajor,
  ElementAccumulator_,
  OperatorClass_,
  ArchTag_,
  ThreadblockShape_,
  WarpShape_,
  InstructionShape_,
  EpilogueOutputOp_,
  ConvertScaledOp_,
  ReductionOp_,
  ThreadblockSwizzle_,
  Stages,
  AlignmentA,
  AlignmentB,
  Operator_,
  WmmaFragABType
> {
public:

  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using TensorRefA = cutlass::TensorRef<ElementA const, LayoutA>;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using TensorRefB = cutlass::TensorRef<ElementB const, LayoutB>;
  using ElementC = ElementC_;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TensorRefC = cutlass::TensorRef<ElementC const, LayoutC>;
  using TensorRefD = cutlass::TensorRef<ElementC, LayoutC>;
  using ElementAccumulator = ElementAccumulator_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using EpilogueOutputOp = EpilogueOutputOp_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static int const kStages = Stages;

  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;
  static int const kAlignmentC = EpilogueOutputOp::kCount;
  static bool const kSplitKSerial = false;
  using Operator = Operator_;


  //
  using UnderlyingOperator = GemmBatchedReduceParallel<
    ElementB,
    typename cutlass::layout::LayoutTranspose<LayoutB>::type,
    ElementA,
    typename cutlass::layout::LayoutTranspose<LayoutA>::type,
    ElementC,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ConvertScaledOp_,
    ReductionOp_,
    ThreadblockSwizzle,
    Stages,
    kAlignmentB,
    kAlignmentA,
    WmmaFragABType
  >;

  using UnderlyingArguments = typename UnderlyingOperator::Arguments;
  using GemmKernel = typename UnderlyingOperator::GemmKernel;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    GemmCoord problem_size;
    cutlass::TensorRef<ElementA const, LayoutA> ref_A;
    int64_t stride_A;
    cutlass::TensorRef<ElementB const, LayoutB> ref_B;
    int64_t stride_B;
    cutlass::TensorRef<ElementC const, LayoutC> ref_C;
    int64_t stride_C;
    cutlass::TensorRef<ElementC, LayoutC> ref_D;
    int64_t stride_D;
    typename EpilogueOutputOp::Params epilogue;
    int batch_count;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments() { }

    /// Constructs an Arguments structure
    CUTLASS_HOST_DEVICE
    Arguments(
      GemmCoord problem_size_,
      cutlass::TensorRef<ElementA const, LayoutA> ref_A_,
      int64_t stride_A_,
      cutlass::TensorRef<ElementB const, LayoutB> ref_B_,
      int64_t stride_B_,
      cutlass::TensorRef<ElementC const, LayoutC> ref_C_,
      int64_t stride_C_,
      cutlass::TensorRef<ElementC, LayoutC> ref_D_,
      int64_t stride_D_,
      typename EpilogueOutputOp::Params epilogue_,
      int batch_count_
    ):
      problem_size(problem_size_),
      ref_A(ref_A_),
      stride_A(stride_A_),
      ref_B(ref_B_),
      stride_B(stride_B_),
      ref_C(ref_C_),
      stride_C(stride_C_),
      ref_D(ref_D_),
      stride_D(stride_D_),
      epilogue(epilogue_),
      batch_count(batch_count_) {
      }
  };

private:

  UnderlyingOperator underlying_operator_;

public:

  /// Constructs the GEMM.
  GemmBatchedReduceParallel() { }

  /// Helper to construct a transposed equivalent for the underying GEMM operator
  static UnderlyingArguments to_underlying_arguments(Arguments const &args) {

    return UnderlyingArguments(
      {args.problem_size.n(), args.problem_size.m(), args.problem_size.k()},
      {args.ref_B.data(), args.ref_B.stride(0)},
      args.stride_B,
      {args.ref_A.data(), args.ref_A.stride(0)},
      args.stride_A,
      {args.ref_C.data(), args.ref_C.stride(0)},
      args.stride_C,
      {args.ref_D.data(), args.ref_D.stride(0)},
      args.stride_D,
      args.epilogue,
      args.batch_count
    );
  }

  /// Determines whether the GEMM can execute the given problem.
  static cutlass::Status can_implement(Arguments const &args) {

    return UnderlyingOperator::can_implement(to_underlying_arguments(args));
  }

  /// Gets the workspace size
  static cutlass::CUsize get_workspace_size(Arguments const &args) {

    return UnderlyingOperator::get_workspace_size(to_underlying_arguments(args));
  }

  /// Initializes GEMM state from arguments.
  cutlass::Status initialize(Arguments const &args, void *workspace = nullptr, hggcStream_t stream = nullptr) {

    return underlying_operator_.initialize(to_underlying_arguments(args), workspace, stream);
  }

  /// Lightweight update given a subset of arguments
  cutlass::Status update(Arguments const &args, void *workspace = nullptr) {

    return underlying_operator_.update(to_underlying_arguments(args), workspace);
  }

  /// Runs the kernel using initialized state.
  cutlass::Status run(hggcStream_t stream = nullptr) {

    return underlying_operator_.run(stream);
  }

  /// Runs the kernel using initialized state.
  cutlass::Status operator()(hggcStream_t stream = nullptr) {
    return run(stream);
  }

  /// Runs the kernel using initialized state.
  cutlass::Status operator()(
    Arguments const &args,
    void *workspace = nullptr,
    hggcStream_t stream = nullptr) {

    cutlass::Status status = initialize(args, workspace, stream);

    if (status == cutlass::Status::kSuccess) {
      status = run(stream);
    }

    return status;
  }

};

////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace aiu

////////////////////////////////////////////////////////////////////////////////
