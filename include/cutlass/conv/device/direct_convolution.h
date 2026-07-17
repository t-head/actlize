/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved. 
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/* \file
   \brief Template for device-level Depthwise Convolution
*/

#pragma once

#include <limits>

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/conv/convolution.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

template<typename DirectConvolutionKernel_>
class DirectConvolution {
public:

  using ImplicitGemmKernel = DirectConvolutionKernel_;

  using ElementA = typename ImplicitGemmKernel::ElementA;
  using LayoutA = typename ImplicitGemmKernel::LayoutA;
  using ElementB = typename ImplicitGemmKernel::ElementB;
  using LayoutB = typename ImplicitGemmKernel::LayoutB;
  using ElementC = typename ImplicitGemmKernel::ElementC;
  using LayoutC = typename ImplicitGemmKernel::LayoutC;
  using ElementAccumulator = typename ImplicitGemmKernel::ElementAccumulator;
  using ElementCompute = typename ImplicitGemmKernel::ElementCompute;
  using OperatorClass = typename ImplicitGemmKernel::OperatorClass;
  using ArchTag = typename ImplicitGemmKernel::ArchTag;
  using ThreadblockShape = typename ImplicitGemmKernel::ThreadblockShape;
  using WarpShape = typename ImplicitGemmKernel::WarpShape;
  using InstructionShape = typename ImplicitGemmKernel::InstructionShape;
  using ThreadblockSwizzle = typename ImplicitGemmKernel::ThreadblockSwizzle;
  using EpilogueOutputOp = typename ImplicitGemmKernel::EpilogueOutputOp;
  static int const kStages = ImplicitGemmKernel::kStages;
  static int const kConvDim = ImplicitGemmKernel::kConvDim;
  using WarpMmaOperator = typename ImplicitGemmKernel::WarpMmaOperator;
  using ArchMmaOperator = typename ImplicitGemmKernel::ArchMmaOperator;
  using MathOperator = typename ImplicitGemmKernel::MathOperator; 

  static cutlass::conv::Operator const kConvolutionalOperator = ImplicitGemmKernel::kConvolutionalOperator;
  static cutlass::conv::IteratorAlgorithm const kIteratorAlgorithm = ImplicitGemmKernel::kIteratorAlgorithm;
  static cutlass::conv::StrideSupport const kStrideSupport = ImplicitGemmKernel::kStrideSupport;
  static cutlass:: conv::KernelType const KType = ImplicitGemmKernel::KType;

  static int const kWarpCount = 
    (ThreadblockShape::kM / WarpShape::kM) * 
    (ThreadblockShape::kN / WarpShape::kN) *
    (ThreadblockShape::kK / WarpShape::kK);

  #if SAIL_FUSE_OP_EXT
  static int const kExtraInputNum = EpilogueOutputOp::kExtraEpilogueInputs > 0 ? EpilogueOutputOp::kExtraEpilogueInputs : 1;
  static int const kExtraInputLoopNum = cutlass::epilogue::GetExtraEpilogueBinaryInputs<EpilogueOutputOp>::value;
  constexpr static bool const kSupportExtraInput = EpilogueOutputOp::kExtraEpilogueOpsNum > 0;
  #endif

  /// Argument structure
  using Arguments = typename ImplicitGemmKernel::Arguments;

  using ReorderKernel = typename ImplicitGemmKernel::ReorderKernel;

 private:

  /// Kernel parameters object
  typename ImplicitGemmKernel::Params params_;

public:

  /// Constructs Implicit GEMM
  DirectConvolution() { }

  /// Determines whether the Implicit GEMM can execute the given problem.
  static Status can_implement(Arguments const &args) {

    // dispatch to iterators
    Status status = ImplicitGemmKernel::Mma::IteratorA::can_implement(args.problem_size);
    if (Status::kSuccess != status) {
      return status;
    }

    status = ImplicitGemmKernel::Mma::IteratorB::can_implement(args.problem_size);
    if (Status::kSuccess != status) {
      return status;
    }

    #if SAIL_FUSE_OP_EXT
    typename ImplicitGemmKernel::TensorRefExtra ref_Extra_NC[kExtraInputNum];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kExtraInputLoopNum; i++) {
      ref_Extra_NC[i] = args.ref_Extra[i].non_const_ref();
    }
    #endif

    if (KType != conv::KernelType::kDepthwise) {
      return Status::kErrorInvalidProblem;
    }

    // C and K should be multiple of groups
    if (args.problem_size.K != args.problem_size.groups &&
      args.problem_size.C != args.problem_size.groups) {
      return Status::kErrorInvalidProblem;
    }
    

    static int const kAlignmentC = ImplicitGemmKernel::Epilogue::OutputTileIterator::kElementsPerAccess;
    if (kConvolutionalOperator == conv::Operator::kFprop) {
      if (args.problem_size.K % kAlignmentC)
        return Status::kErrorMisalignedOperand;
    } else if (kConvolutionalOperator == conv::Operator::kDgrad) {
       if (args.problem_size.C % kAlignmentC)
        return Status::kErrorMisalignedOperand;
    } else if (kConvolutionalOperator == conv::Operator::kWgrad) {
       if (args.problem_size.C % kAlignmentC)
        return Status::kErrorMisalignedOperand;
    }

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(
      threadblock_swizzle.get_tiled_shape(
        cutlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, args.problem_size),
        {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
        args.problem_size.split_k_slices));

    if (!(grid.y <= std::numeric_limits<uint16_t>::max() &&
          grid.z <= std::numeric_limits<uint16_t>::max())) {

      return Status::kErrorInvalidProblem;
    }

    return Status::kSuccess;
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {  
    return 0;
  }

  /// Initializes GEMM state from arguments.
  Status initialize(
    Arguments const &args, 
    void *workspace = nullptr, 
    hggcStream_t stream = nullptr) {
    
    // initialize the params structure from the arguments
    params_ = typename ImplicitGemmKernel::Params(
    	args,
    	static_cast<int *>(workspace)
    );

    #if 0
    // Depthwise-Conv would set Max Shm Size twice with different values.
    // This would bring race condition when launching same kernel in different threads.
    // So set this mandatory attribute in acompute uniformly.
    int smem_size = int(sizeof(typename ImplicitGemmKernel::SharedStorage));

    if (smem_size >= (48 << 10)) {
      hggcError_t result = hggcFuncSetAttribute(cutlass::Kernel<ImplicitGemmKernel>,
                                    hggcFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (result != hggcSuccess) {
        return Status::kErrorInternal;
      }
    }
    #endif

    return Status::kSuccess;
  }

  /// Initializes GEMM state from arguments.
  Status update(Arguments const &args, void *workspace = nullptr) {

    // update the params structure from the arguments
    params_.ptr_A = args.ref_A.data();
    params_.ptr_B = args.ref_B.data();
    params_.ptr_C = args.ref_C.data();
    params_.ptr_D = args.ref_D.data();
    #if SAIL_FUSE_OP_EXT
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kExtraInputLoopNum; i++) {
      params_.ptr_Extra[i] = args.ref_Extra[i].data();
    }
    #endif
    params_.output_op = args.output_op;
    params_.ptr_reordered_B = args.ref_reordered_B.data();
    params_.semaphore = static_cast<int *>(workspace);

    return Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  Status run(hggcStream_t stream = nullptr) {
    
    // Launch reorder kernel
    if (params_.ptr_reordered_B != nullptr) {
      dim3 grid = ReorderKernel::get_grid_shape(params_);
      dim3 block = ReorderKernel::get_block_shape();

      cutlass::Kernel<ReorderKernel><<<grid, block, 0, stream>>>(params_);
    } else {
      params_.ptr_reordered_B = const_cast<typename platform::remove_const<typename ImplicitGemmKernel::Mma::IteratorB::Element >::type *>(params_.ptr_B);
    }

    // Launch main kernel
    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
    dim3 block(32 * kWarpCount, 1, 1);

    // Dynamic SMEM size based on input params.
    int smem_size = int(params_.get_runtime_smem_size());
    
    #if 0
    // Depthwise-Conv would set Max Shm Size twice with different values.
    // This would bring race condition when launching same kernel in different threads.
    // So set this mandatory attribute in acompute uniformly.

    // Make sure we can use that much shared memory.
    hggcError_t status = 
        hggcFuncSetAttribute(cutlass::Kernel<ImplicitGemmKernel>, hggcFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (status != hggcSuccess)
      return Status::kErrorInternal;
    #endif

    cutlass::Kernel<ImplicitGemmKernel><<<grid, block, smem_size, stream>>>(params_);

    hggcError_t result = hggcGetLastError();

    return result == hggcSuccess ? Status::kSuccess : Status::kErrorInternal;
  }

  /// Runs the kernel using initialized state.
  Status operator()(hggcStream_t stream = nullptr) {
    return run(stream);
  }

  /// Runs the kernel using initialized state.
  Status operator()(
    Arguments const &args, 
    void *workspace = nullptr, 
    hggcStream_t stream = nullptr) {
    
    Status status = initialize(args, workspace, stream);
    
    if (status == Status::kSuccess) {
      status = run(stream);
    }

    return status;
  }

  int get_runtime_smem_size() { return int(params_.get_runtime_smem_size()); }

  static int get_smem_size(bool update_smem_size = false) {
    // int smem_size = int(sizeof(typename ImplicitGemmKernel::SharedStorage));
    int smem_size = int(sizeof(typename ImplicitGemmKernel::SharedStorage)) * kStages;

    if (update_smem_size) {
      // compiler will use 128B static smem?
      int block_per_cu = 255 * 1024 / smem_size;
      int warp_per_block = (ThreadblockShape::kM / WarpShape::kM) * (ThreadblockShape::kN / WarpShape::kN);
      int block_align_num = 8 / warp_per_block;
      // should only increase smem since already use a small stage
      // do nothing when we is not full
      if (block_per_cu % block_align_num && block_per_cu >= block_align_num) {
        block_per_cu -= block_per_cu % block_align_num;
        smem_size = (255 / block_per_cu) * 1024;
      }
    }
    return smem_size;
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

}
}
}

/////////////////////////////////////////////////////////////////////////////////////////////////
