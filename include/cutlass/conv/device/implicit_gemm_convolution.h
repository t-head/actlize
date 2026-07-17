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

/* \file
   \brief Template for device-level Implicit GEMM Convolution
*/

#pragma once

#include <limits>

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/reduction/kernel/reduce_split_k.h"
#include "cutlass/reduction/thread/reduction_operators.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  typename ImplicitGemmKernel_,
  // should assign for splitk parallel, when ImplicitGemmKernel_::EpilogueOutputOp is Convert
  typename EpilogueOutputOp = typename ImplicitGemmKernel_::EpilogueOutputOp
>
class ImplicitGemmConvolution {
public:

  using ImplicitGemmKernel = ImplicitGemmKernel_;

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
  // using EpilogueOutputOp = typename ImplicitGemmKernel::EpilogueOutputOp;
  static int const kStages = ImplicitGemmKernel::kStages;
  static int const kConvDim = ImplicitGemmKernel::kConvDim;
  using WarpMmaOperator = typename ImplicitGemmKernel::WarpMmaOperator;
#if !SAIL_ENABLE_CUTLASS_SIMT
  using ArchMmaOperator = typename ImplicitGemmKernel::ArchMmaOperator;
  using MathOperator = typename ImplicitGemmKernel::MathOperator;
#endif

  static cutlass::conv::Operator const kConvolutionalOperator = ImplicitGemmKernel::kConvolutionalOperator;
  static cutlass::conv::IteratorAlgorithm const kIteratorAlgorithm = ImplicitGemmKernel::kIteratorAlgorithm;

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

  using ReductionOp = typename cutlass::reduction::thread::ReduceAdd<
    ElementAccumulator, typename EpilogueOutputOp::ElementAccumulator,
    EpilogueOutputOp::kCount>;

  using ReductionKernel = cutlass::reduction::kernel::ReduceSplitK<
    cutlass::MatrixShape<4, 32 * EpilogueOutputOp::kCount>,
    EpilogueOutputOp,
    ReductionOp
  >;

  static cutlass::conv::KernelType const KType = ImplicitGemmKernel::KType;

  static auto constexpr ImplicitGemmDevFunc = cutlass::Kernel<ImplicitGemmKernel, 32 * kWarpCount>;

private:

  /// Kernel parameters object
  typename ImplicitGemmKernel::Params params_;
  typename ReductionKernel::Params reduction_params_;

  // start_dim is lowerset dim of stride
  // stride and shape are on same order, for example stride is stride_w/stride_h/stride_n, shape is c/w/h/n
  template <int Rank>
  inline bool is_continuous(const Coord<Rank> &stride, const Coord<Rank + 1> &shape, int start_dim) {
    bool all_zero = true;
    // all strides are zeros means broadcast
    for (int i = 0; i < Rank; ++i) {
      if (stride.at(i) != 0) {
        all_zero &= false;
      }
    }
    if (all_zero) {
      return true;
    }

    for (int i = start_dim; i < Rank; ++i) {
      auto last_stride = i == 0 ? 1 : stride.at(i - 1);
      auto cur_stride = stride.at(i);
      // cur dim has extra stride
      if (cur_stride / last_stride != shape[i]) {
        return false;
      }
    }
    return true;
  }

  // this function is just used to help compile pass
  template <int Rank>
  inline bool is_continuous(const Coord<Rank> &stride, const Coord<Rank> &shape) {
    printf("[Error] run in illegal is_continuous function! Given shape is too short!\n");
    return false;
  }

  template <typename Layout, typename ConvProblemSize>
  inline bool layout_supported(const Layout& layoutC, const Layout& layoutD, MatrixCoord& reduce_add, MatrixCoord& reduce_out, const ConvProblemSize& problem, float beta) {
    auto c_extent = implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem);
    bool support = true;
    bool is_nhwc = std::is_same<Layout, layout::TensorNHWC>::value || std::is_same<Layout, layout::TensorNDHWC>::value;

    // only fprop support stride on c_input
    if (beta != 0) {
      // only fprop nhwc support stride on c_input
      if (kConvolutionalOperator == Operator::kFprop && is_nhwc) {
        support &= is_continuous(layoutC.stride(), c_extent.get_ordered_shape(is_nhwc), 1);
        // RowMajor::packed only use column size to get stride
        reduce_add.column() = layoutC.stride().at(0);
      } else {
        support &= is_continuous(layoutC.stride(), c_extent.get_ordered_shape(is_nhwc), 0);
      }
    }

    // splitK parallel for NCHW and wgrad only support output as continuous
    if (std::is_same<Layout, layout::TensorNCHW>::value || kConvolutionalOperator == Operator::kWgrad) {
      support &= is_continuous(layoutD.stride(), c_extent.get_ordered_shape(is_nhwc), 0);
    }

    // stride_c can't smaller than dim
    if (layoutD.stride().at(0) < c_extent.c()) {
      return false;
    }
    // update coord
    // only fprop nhwc support stride on c_input, only fprop ref.stride[0] = out.stride[0]
    if (kConvolutionalOperator == Operator::kFprop && is_nhwc) {
      reduce_out.column() = layoutD.stride().at(0);
    }
    // we support stride on dim C for NDHWC
    support &= is_continuous(layoutD.stride(), c_extent.get_ordered_shape(is_nhwc), 1);
    return support;
  }

public:

  /// Constructs Implicit GEMM
  ImplicitGemmConvolution() { }

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

    status = ImplicitGemmKernel::can_implement(
      args.problem_size,
      args.ref_A.non_const_ref(),
      args.ref_B.non_const_ref()
      #if SAIL_FUSE_OP_EXT
      , ref_Extra_NC
      #endif
    );
    if (Status::kSuccess != status) {
      return status;
    }

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(
      threadblock_swizzle.get_tiled_shape(
        cutlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, args.problem_size),
        {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
        args.problem_size.split_k_slices,
        cutlass::conv::implicit_gemm_problem_stride_coord(kConvolutionalOperator, args.problem_size.group_stride,
        args.problem_size, KType)));

    if (!(grid.y <= std::numeric_limits<uint16_t>::max() &&
          grid.z <= std::numeric_limits<uint16_t>::max())) {
      return Status::kErrorInvalidProblem;
    }

    return Status::kSuccess;
  }

  static int get_smem_size(bool update_smem_size = false) {
    int smem_size = int(sizeof(typename ImplicitGemmKernel::SharedStorage));
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

  /// Gets the workspace size
  static CUsize get_workspace_size(Arguments const &args) {

    CUsize workspace_bytes = 0;

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
        cutlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, args.problem_size),
        {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
        args.problem_size.split_k_slices,
        cutlass::conv::implicit_gemm_problem_stride_coord(kConvolutionalOperator,
        args.problem_size.group_stride, args.problem_size, KType));

    if(args.split_k_mode == SplitKMode::kParallel) {

      // Split-K parallel: CTAs in k-dimension write the partial results in a temporary workspace.
      // The user needs to call a reduction operator to optain the final output tensor
      workspace_bytes =
        sizeof(ElementAccumulator) *
        size_t(cutlass::conv::implicit_gemm_tensor_c_size(kConvolutionalOperator, args.problem_size)) *
        size_t(grid_tiled_shape.k());
    }

    else if(args.split_k_mode == SplitKMode::kSerial && args.problem_size.split_k_slices > 1) {

      // Split-K serial: The user workspace is used to store semaphore and serialize writing the
      // final reduced output to user's output tensor
      workspace_bytes = sizeof(int) * size_t(grid_tiled_shape.m()) * size_t(grid_tiled_shape.n()) * size_t(args.problem_size.groups);
    }

    return workspace_bytes;
  }

  /// Initializes GEMM state from arguments.
  Status initialize(
    Arguments const &args,
    void *workspace = nullptr,
    hggcStream_t stream = nullptr) {

    if (args.split_k_mode == SplitKMode::kSerial && args.problem_size.split_k_slices > 1) {

      if (!workspace) {
        printf("no workspace for splitK!\n");
        return Status::kErrorWorkspaceNull;
      }

      hggcError_t status = hggcMemsetAsync(workspace, 0, get_workspace_size(args), stream);

      if (status != hggcSuccess) {
        printf("cutlass workspace initial fail!\n");
        return Status::kErrorInternal;
      }
    }

    // initialize the params structure from the arguments
    params_ = typename ImplicitGemmKernel::Params(
    	args,
    	static_cast<int *>(workspace)
    );

    if(params_.split_k_mode == SplitKMode::kParallel && params_.grid_tiled_shape.k() > 1) {
      auto out_size = cutlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, args.problem_size);
      out_size.n() = args.problem_size.groups > 1 && KType == KernelType::kMultipleGroup ?
        out_size.n() / args.problem_size.groups : out_size.n();
      int64_t partition_stride = int64_t(out_size.m()) * int64_t(out_size.n());

      TensorRef<ElementAccumulator, layout::RowMajor> ref_workspace(
        static_cast<ElementAccumulator *>(workspace),
        out_size.n());

      bool no_fold = args.problem_size.fold_num == 1 && args.problem_size.fold_h == 1 &&
                      args.problem_size.fold_w == 1;
      MatrixCoord reduce_add(out_size.m(), out_size.n());
      MatrixCoord reduce_out(out_size.m(), out_size.n());
      // to know wheher c is needed in layout_supported
      float beta;
      typename EpilogueOutputOp::Params output_op = load_epilogue_params_(args, beta);
      // TODO: only do this check and change reduce_out coord when conv problem no fold now,
      // because I don't know the effect of fold on conv.
      // And conv with fold shouldn't run in splitK?
      if (no_fold) {
        if (!layout_supported(args.ref_C.layout(), args.ref_D.layout(), reduce_add, reduce_out, args.problem_size, beta)) {
          printf("Unsupport layout for conv splitK parallel!\n");
          return Status::kErrorInternal;
        }
      }
      // construct ref_c with layout = RowMajor, for recudtion op only accept this layout
      auto output_layout = cutlass::layout::RowMajor::packed(reduce_out);
      auto add_layout = cutlass::layout::RowMajor::packed(reduce_add);

      TensorRef<typename EpilogueOutputOp::ElementOutput, layout::RowMajor> ref_c((typename EpilogueOutputOp::ElementOutput*)args.ref_C.data(), add_layout);
      TensorRef<typename EpilogueOutputOp::ElementOutput, layout::RowMajor> ref_d((typename EpilogueOutputOp::ElementOutput*)args.ref_D.data(), output_layout);

      #if SAIL_FUSE_OP_EXT
      TensorRef<typename EpilogueOutputOp::ElementOutput, layout::RowMajor> ref_Extra_NC[kExtraInputNum];
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kExtraInputLoopNum; i++) {
        auto epilogue_layout = args.ref_Extra[i].layout();
        using LayoutType = decltype(epilogue_layout);

        bool same_shape_as_output = true;
        for (typename LayoutType::Index stride_idx = 0; stride_idx < LayoutType::kStrideRank; stride_idx++) {
          if (epilogue_layout.stride()[stride_idx] == 0) {
            // If any dimension has zero stride, we deem it with broadcast
            same_shape_as_output = false;
            break;
          }
        }

        if (same_shape_as_output) {
          ref_Extra_NC[i] = TensorRef<typename EpilogueOutputOp::ElementOutput, layout::RowMajor>((typename EpilogueOutputOp::ElementOutput*)args.ref_Extra[i].data(), output_layout);
        } else {
          auto output_broadcast_layout = layout::RowMajor::Stride(0);
          ref_Extra_NC[i] = TensorRef<typename EpilogueOutputOp::ElementOutput, layout::RowMajor>((typename EpilogueOutputOp::ElementOutput*)args.ref_Extra[i].data(), output_broadcast_layout);
        }
      }
      #endif

      reduction_params_ = typename ReductionKernel::Params(
        MatrixCoord(out_size.m(), out_size.n()),
        args.problem_size.split_k_slices,
        partition_stride,
        0,
        0,
        ref_workspace,
        ref_d,
        ref_c,
        output_op
        #if SAIL_FUSE_OP_EXT
        ,typename ReductionOp::Params()
        ,ref_Extra_NC
        #endif
      );

      // NCHW splitK parallel only support output as continuous;
      // NHWC splitK parallel can support strideW > C for output,
      // and we need reset conv kernel output D to continuous iteratorD.
      if (no_fold &&
          !(std::is_same<LayoutC, layout::TensorNCHW>::value || kConvolutionalOperator == Operator::kWgrad)) {
        TensorRef<ElementC, LayoutC> ref_conv_out(
          static_cast<ElementC *>(workspace),
          LayoutC::packed(implicit_gemm_tensor_c_extent(kConvolutionalOperator, args.problem_size)));
        params_.iterator_D = ImplicitGemmKernel::ConvOutputIteratorParameter::layout(ref_conv_out);
        params_.iterator_C = ImplicitGemmKernel::ConvOutputIteratorParameter::layout(ref_conv_out);
      }
    }

    int smem_size = get_smem_size();

    if (smem_size >= (48 << 10)) {
      hggcError_t result = hggcFuncSetAttribute(ImplicitGemmDevFunc,
                                    hggcFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (result != hggcSuccess) {
        printf("cutlass set kernel smem size fail!\n");
        return Status::kErrorInternal;
      }

      result = hggcFuncSetAttribute(
          ImplicitGemmDevFunc,
          hggcFuncAttributePreferredSharedMemoryCarveout, 100);

      if (result != hggcSuccess) {
        printf("cutlass set kernel smem ratio fail!\n");
        return Status::kErrorInternal;
      }
    }

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
    params_.semaphore = static_cast<int *>(workspace);

    if(params_.split_k_mode == SplitKMode::kParallel && params_.grid_tiled_shape.k() > 1) {
      reduction_params_.destination.reset(args.ref_D.data());
      reduction_params_.source.reset(args.ref_C.data());
      reduction_params_.workspace.reset(static_cast<ElementAccumulator *>(workspace));
      #if SAIL_FUSE_OP_EXT
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kExtraInputLoopNum; i++) {
        reduction_params_.ref_Extra[i].reset(args.ref_Extra[i].non_const_ref().data());
      }
      #endif
    }

    return Status::kSuccess;
  }

#if SAIL_CUSTOMIZE_CUTLASS
  using LaunchConfigs = typename ImplicitGemmKernel::LaunchConfigs;

  LaunchConfigs get_launch_configs() {
    ThreadblockSwizzle threadblock_swizzle;
    dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
    dim3 block(32 * kWarpCount, 1, 1);
    int smem_bytes = get_smem_size();
    hggcFuncAttributes attr;
    checkHggcErrors(hggcFuncGetAttributes(&attr, ImplicitGemmDevFunc));

    return LaunchConfigs(grid, block, smem_bytes, attr.numRegs, attr.localSizeBytes);
  }
#endif

  /// Runs the kernel using initialized state.
  Status run(hggcStream_t stream = nullptr) {

    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape, &params_.grid_shift);
    dim3 block(32 * kWarpCount, 1, 1);

    int smem_size = get_smem_size();

#if CUTLASS_ENABLE_RTC
    const std::string prog_src = user_pre_headers
                               + std::string("#include \"cutlass/conv/kernel/default_conv2d_fprop.h\"\n")
                               + std::string(cutlass::hgrtc::cutlass_device_kernel_h);

    std::string kernel_traits;
    checkHgrtcErrors(hgrtcGetTypeName<ImplicitGemmKernel>(&kernel_traits));
    std::string kernel_instantiation = "cutlass::Kernel<" + kernel_traits + ">";
    HGfunction kernel = rtCompile(prog_src, kernel_instantiation);

    void* args[] = { reinterpret_cast<void*>(&params_) };
    checkHggcDrvErrors(hgLaunchKernel(kernel, grid.x, grid.y, grid.z, block.x, block.y, block.z, smem_size, stream, args, 0));
#else // CUTLASS_ENABLE_RTC
    ImplicitGemmDevFunc<<<grid, block, smem_size, stream>>>(params_);
#endif // CUTLASS_ENABLE_RTC

    hggcError_t result = hggcGetLastError();

    if (result != hggcSuccess) {
      printf("cutlass kernel run fail!\n");
      return Status::kErrorInternal;
    }

    if(params_.split_k_mode == SplitKMode::kParallel && params_.grid_tiled_shape.k() > 1) {
      block = ReductionKernel::block_shape();
      grid = ReductionKernel::grid_shape(reduction_params_.problem_size);

      Kernel<ReductionKernel><<< grid, block, 0, stream >>>(reduction_params_);
    }

    result = hggcGetLastError();

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

  template <bool Support = kSupportExtraInput>
  typename platform::enable_if<Support, typename EpilogueOutputOp::Params>::type
  load_epilogue_params_(Arguments const &args, float &beta_) {
    // under fprop with epilogue fusion case, args.output_op is still for LinearCombinationEpilogueFuse,
    // so use it for second reduce kernel's output op params, which is good for epilogue fusion.

    typename EpilogueOutputOp::Params epi_params;
    epi_params.alpha = (typename EpilogueOutputOp::ElementCompute)args.output_op.alpha;
    epi_params.beta_c = (typename EpilogueOutputOp::ElementCompute)args.output_op.beta_c;
    epi_params.beta_e = (typename EpilogueOutputOp::ElementCompute)args.output_op.beta_e;
    epi_params.act_param0 = (typename EpilogueOutputOp::ElementCompute)args.output_op.act_param0;
    epi_params.act_param1 = (typename EpilogueOutputOp::ElementCompute)args.output_op.act_param1;
    beta_ = float(epi_params.beta_c);
    return epi_params;
  }

  template <bool Support = kSupportExtraInput>
  typename platform::enable_if<!Support, typename EpilogueOutputOp::Params>::type
  load_epilogue_params_(Arguments const &args, float &beta_) {
    // under wgrad with splitk-paralel case, args.output_op is for Convert as first conv kernel's epilogue output op,
    // so CANNOT reuse it for reduce kernel, here compose "params" for second reduce kernel's LinearCombination params.
    // This behavior is still the same as Mengchi did it.
    typename EpilogueOutputOp::ElementCompute alpha(1.0);
    typename EpilogueOutputOp::ElementCompute beta(0.0);
    beta_ = 0.0;
    return typename EpilogueOutputOp::Params(alpha, beta);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}
}
}

/////////////////////////////////////////////////////////////////////////////////////////////////
