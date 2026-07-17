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

#pragma once

#include "accutlass.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/device_kernel.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

#include "aiu/gemm/kernel/default_gemm.h"
#include "aiu/gemm/device/default_gemm_configuration.h"
#include "aiu/gemm/tool/cutlass_type_convert.h"
//#include "../tool/launch_bound.h"

////////////////////////////////////////////////////////////////////////////////

namespace aiu {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

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
    typename OperatorClass_ = cutlass::arch::OpClassTensorOp,
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
    typename ThreadblockSwizzle_ =
        typename cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    /// Number of stages used in the pipelined mainloop
    int Stages =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kStages,
    /// Access granularity of A matrix in units of elements
    int AlignmentA = 1,
    /// Access granularity of B matrix in units of elements
    int AlignmentB = 1,
    /// If true, kernel supports split-K with serial reduction
    bool SplitKSerial = false,
    /// Operation performed by GEMM
    typename Operator_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::Operator,
    /// Wmma Fragment type for A and B (for FP32 tensorcell on PPU)
    typename WmmaFragABType = typename ToTF32<ElementA_>::type,
    /// Option to enable or disable cutlass epilogue for flexibility
    bool EnableCutlassEpilogue_ = true
    >
class Gemm {
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
  using Operator = Operator_;
  using WmmaFragABType_ = typename FromCutlassType<WmmaFragABType>::type;
  static int const kStages = Stages;
  static int const kAlignmentC = EpilogueOutputOp::kCount;
  static bool const kSplitKSerial = SplitKSerial;
  static cutlass::ComplexTransform const kTransformA = cutlass::ComplexTransform::kNone;
  static cutlass::ComplexTransform const kTransformB = cutlass::ComplexTransform::kNone;

  #if SAIL_FUSE_OP_EXT
  static int const kExtraInputNum = EpilogueOutputOp::kExtraEpilogueInputs > 0 ? EpilogueOutputOp::kExtraEpilogueInputs : 1;
  static int const kExtraInputLoopNum = cutlass::epilogue::GetExtraEpilogueBinaryInputs<EpilogueOutputOp>::value;
  #endif

  /// Define the kernel
  using GemmKernel = typename kernel::DefaultGemm<
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
    kSplitKSerial,
    Operator,
    WmmaFragABType_,
    EnableCutlassEpilogue_
  >::GemmKernel;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    cutlass::gemm::GemmCoord problem_size;
    cutlass::TensorRef<ElementA const, LayoutA> ref_A;
    cutlass::TensorRef<ElementB const, LayoutB> ref_B;
    cutlass::TensorRef<ElementC const, LayoutC> ref_C;
    cutlass::TensorRef<ElementC, LayoutC> ref_D;
    #if SAIL_FUSE_OP_EXT
    cutlass::TensorRef<ElementFuseInExtra const, LayoutExtra> ref_Extra[kExtraInputNum];
    #endif
    typename EpilogueOutputOp::Params epilogue;
    int split_k_slices;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments(): problem_size(0, 0, 0), split_k_slices(1) {

    }

    /// Constructs an Arguments structure
    CUTLASS_HOST_DEVICE
    Arguments(
      cutlass::gemm::GemmCoord problem_size_,
      cutlass::TensorRef<ElementA const, LayoutA> ref_A_,
      cutlass::TensorRef<ElementB const, LayoutB> ref_B_,
      cutlass::TensorRef<ElementC const, LayoutC> ref_C_,
      cutlass::TensorRef<ElementC, LayoutC> ref_D_,
      typename EpilogueOutputOp::Params epilogue_ =
        typename EpilogueOutputOp::Params(),
      int split_k_slices = 1
      #if SAIL_FUSE_OP_EXT
      , cutlass::TensorRef<ElementFuseInExtra const, LayoutExtra> *pref_Extra_ = nullptr
      #endif
    ):
      problem_size(problem_size_),
      ref_A(ref_A_),
      ref_B(ref_B_),
      ref_C(ref_C_),
      ref_D(ref_D_),
      epilogue(epilogue_),
      split_k_slices(split_k_slices) {
        #if SAIL_FUSE_OP_EXT
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kExtraInputLoopNum; i++) {
          if (pref_Extra_) {
            ref_Extra[i] = pref_Extra_[i];
          } else {
            ref_Extra[i] = {nullptr, 0};
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
  Gemm() { }

  /// Determines whether the GEMM can execute the given problem.
  static cutlass::Status can_implement(Arguments const &args) {

    if (!kSplitKSerial && args.split_k_slices > 1) {
      return cutlass::Status::kErrorInvalidProblem;
    }

    if (!TensorRef_aligned(args.ref_C.non_const_ref(), kAlignmentC)) {
      return cutlass::Status::kErrorMisalignedOperand;
    }
    if (!TensorRef_aligned(args.ref_D.non_const_ref(), kAlignmentC)) {
      return cutlass::Status::kErrorMisalignedOperand;
    }

    #if SAIL_FUSE_OP_EXT
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kExtraInputLoopNum; i++) {
      auto ref_Extra_NC = args.ref_Extra[i].non_const_ref();
      if (ref_Extra_NC.good() && !TensorRef_aligned(ref_Extra_NC, kAlignmentC)) {
          return cutlass::Status::kErrorMisalignedOperand;
      }
    }
    #endif

    int len_C = cutlass::platform::is_same<LayoutC, cutlass::layout::RowMajor>::value ? args.problem_size.n() : args.problem_size.m();

    if (len_C % kAlignmentC) {
      return cutlass::Status::kErrorMisalignedOperand;
    }
    return cutlass::Status::kSuccess;
  }

  /// Gets the workspace size
  static cutlass::CUsize get_workspace_size(Arguments const &args) {

    cutlass::CUsize bytes = 0;

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord tiled_shape = threadblock_swizzle.get_tiled_shape(
      args.problem_size,
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
      args.split_k_slices);

    if (kSplitKSerial && args.split_k_slices > 1) {

      bytes += sizeof(int) * size_t(tiled_shape.m()) * size_t(tiled_shape.n());
    }

    return bytes;
  }

  /// Initializes GEMM state from arguments.
  cutlass::Status initialize(Arguments const &args, void *workspace = nullptr, hggcStream_t stream = nullptr) {

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord grid_shape = threadblock_swizzle.get_tiled_shape(
      args.problem_size,
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
      args.split_k_slices);

    if (kSplitKSerial) {
      if (args.split_k_slices > 1) {
        if (!workspace) {
          return cutlass::Status::kErrorWorkspaceNull;
        }

        cutlass::CUsize bytes = get_workspace_size(args);

        hggcError_t result = hggcMemsetAsync(workspace, 0, bytes, stream);

        if (result != hggcSuccess) {
          return cutlass::Status::kErrorInternal;
        }
      }
    }
    else {

      if (args.split_k_slices > 1) {
        return cutlass::Status::kErrorInvalidProblem;
      }
    }

    #if SAIL_FUSE_OP_EXT
    cutlass::TensorRef<ElementFuseInExtra, LayoutExtra> ref_Extra_NC[kExtraInputNum];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kExtraInputLoopNum; i++) {
      ref_Extra_NC[i] = args.ref_Extra[i].non_const_ref();
    }
    #endif

    // Initialize the Params structure
    params_ = typename GemmKernel::Params{
      args.problem_size,
      grid_shape,
      args.ref_A.non_const_ref(),
      args.ref_B.non_const_ref(),
      args.ref_C.non_const_ref(),
      #if SAIL_FUSE_OP_EXT
      ref_Extra_NC,
      #endif
      args.ref_D,
      args.epilogue,
      static_cast<int *>(workspace)
    };

    return cutlass::Status::kSuccess;
  }

  /// Lightweight update given a subset of arguments
  cutlass::Status update(Arguments const &args, void *workspace = nullptr) {

    if (kSplitKSerial && args.split_k_slices > 1) {
      if (!workspace) {
        return cutlass::Status::kErrorWorkspaceNull;
      }
    }

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
    params_.output_op = args.epilogue;
    params_.semaphore = static_cast<int *>(workspace);

    return cutlass::Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  cutlass::Status run(hggcStream_t stream = nullptr) {

    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
    dim3 block(GemmKernel::kThreadCount, 1, 1);

    hggcError_t result;

    //constexpr int LAUNCH_BOUND = LaunchBound<ElementA_, ElementAccumulator_, ThreadblockShape_,
    //                                         WarpShape_, InstructionShape_, Stages>::value;

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
    /// Instruction-level tile size (concept: GemmShape)
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
    /// If true, kernel supports split-K as a serial reduction
    bool SplitKSerial,
    /// Operation performed by GEMM
    typename Operator_,
    /// Wmma Fragment type for A and B (for FP32 tensorcell on PPU)
    typename WmmaFragABType,
    /// Option to enable or disable cutlass epilogue for flexibility
    bool EnableCutlassEpilogue
    >
class Gemm<ElementA_, LayoutA_, ElementB_, LayoutB_, ElementC_,
           cutlass::layout::ColumnMajor,  // partially specialized on LayoutC
           ElementAccumulator_, OperatorClass_, ArchTag_, ThreadblockShape_,
           WarpShape_, InstructionShape_, EpilogueOutputOp_,
           ThreadblockSwizzle_, Stages, AlignmentA, AlignmentB, SplitKSerial,
           Operator_, WmmaFragABType, EnableCutlassEpilogue
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
  using Operator = Operator_;
  static int const kStages = Stages;
  static cutlass::ComplexTransform const kTransformA = cutlass::ComplexTransform::kNone;
  static cutlass::ComplexTransform const kTransformB = cutlass::ComplexTransform::kNone;
  static bool const kSplitKSerial = SplitKSerial;

  #if SAIL_FUSE_OP_EXT
  static int const kExtraInputNum = EpilogueOutputOp::kExtraEpilogueInputs > 0 ? EpilogueOutputOp::kExtraEpilogueInputs : 1;
  static int const kExtraInputLoopNum = cutlass::epilogue::GetExtraEpilogueBinaryInputs<EpilogueOutputOp>::value;
  #endif

  using UnderlyingOperator = Gemm<
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
    AlignmentA,
    AlignmentB,
    SplitKSerial,
    Operator
  >;

  using UnderlyingArguments = typename UnderlyingOperator::Arguments;
  using GemmKernel = typename UnderlyingOperator::GemmKernel;
  static int const kAlignmentC = UnderlyingOperator::kAlignmentC;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    cutlass::gemm::GemmCoord problem_size;
    cutlass::TensorRef<ElementA const, LayoutA> ref_A;
    cutlass::TensorRef<ElementB const, LayoutB> ref_B;
    cutlass::TensorRef<ElementC const, LayoutC> ref_C;
    cutlass::TensorRef<ElementC, LayoutC> ref_D;
    #if SAIL_FUSE_OP_EXT
    cutlass::TensorRef<ElementFuseInExtra const, LayoutExtra> ref_Extra[kExtraInputNum];
    #endif

    typename EpilogueOutputOp::Params epilogue;
    int split_k_slices;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments() { }

    /// Constructs an Arguments structure
    CUTLASS_HOST_DEVICE
    Arguments(
      cutlass::gemm::GemmCoord problem_size_,
      cutlass::TensorRef<ElementA const, LayoutA> ref_A_,
      cutlass::TensorRef<ElementB const, LayoutB> ref_B_,
      cutlass::TensorRef<ElementC const, LayoutC> ref_C_,
      cutlass::TensorRef<ElementC, LayoutC> ref_D_,
      typename EpilogueOutputOp::Params epilogue_ =
        typename EpilogueOutputOp::Params(),
      int split_k_slices = 1
      #if SAIL_FUSE_OP_EXT
      , cutlass::TensorRef<ElementFuseInExtra const, LayoutExtra> *pref_Extra_ = nullptr
      #endif
    ):
      problem_size(problem_size_),
      ref_A(ref_A_),
      ref_B(ref_B_),
      ref_C(ref_C_),
      ref_D(ref_D_),
      epilogue(epilogue_),
      split_k_slices(split_k_slices) {
        #if SAIL_FUSE_OP_EXT
        CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < kExtraInputLoopNum; i++) {
            if (pref_Extra_) {
              ref_Extra[i] = pref_Extra_[i];
            } else {
              ref_Extra[i] = {nullptr, 0};
            }
        }
        #endif
      }
  };

private:

  UnderlyingOperator underlying_operator_;

public:

  /// Constructs the GEMM.
  Gemm() { }

  /// Helper to construct a transposed equivalent for the underying GEMM operator
  static UnderlyingArguments to_underlying_arguments(Arguments const &args) {

    #if SAIL_FUSE_OP_EXT
    cutlass::TensorRef<typename UnderlyingOperator::ElementFuseInExtra const, typename UnderlyingOperator::LayoutExtra> ref_Extra_NC[kExtraInputNum];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kExtraInputLoopNum; i++) {
      ref_Extra_NC[i] = {args.ref_Extra[i].non_const_ref().data(), args.ref_Extra[i].non_const_ref().stride(0)};
    }
    #endif

    return UnderlyingArguments(
      {args.problem_size.n(), args.problem_size.m(), args.problem_size.k()},
      {args.ref_B.data(), args.ref_B.stride(0)},
      {args.ref_A.data(), args.ref_A.stride(0)},
      {args.ref_C.data(), args.ref_C.stride(0)},
      {args.ref_D.data(), args.ref_D.stride(0)},
      args.epilogue,
      args.split_k_slices
      #if SAIL_FUSE_OP_EXT
      , ref_Extra_NC
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

////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace aiu

////////////////////////////////////////////////////////////////////////////////
