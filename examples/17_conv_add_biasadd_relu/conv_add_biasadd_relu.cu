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

/**
The convolution version of 12_gemm_bias_relu.  Similarly, we put bias vector in Operand C and the
rest is the same as normal convolution.
*/

#include <iostream>
#include <sstream>

// PTG cutlass propriatary configs
#include "accutlass.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/epilogue/thread/linear_combination_epilogue_fuse.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/host_reorder.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/device/convolution.h"
#include "cutlass/util/tensor_view_io.h"

#include "helper.h"

// The code section below describes datatype for input, output tensors and computation between
// elements 
using ElementAccumulator = float;                  // Data type of accumulator
using ElementComputeEpilogue = ElementAccumulator; // Data type of epilogue computation
using ElementInputA = cutlass::half_t;             // Data type of elements in input tensor
using ElementInputB = cutlass::half_t;             // Data type of elements in input tensor
using ElementOutput = cutlass::half_t;                       // Data type of elements in output tensor

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

// This code section describes whether you want to use tensor cells or regular SIMT cores on PPU CU
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes device CU architecture number
using CuArch = cutlass::arch::PPU0010;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;

// This code section describes tile size a warp will compute
using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>; 

// This code section describes the size of MMA op
using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;
// This code section describes how threadblocks are scheduled on PPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmHorizontalThreadblockSwizzle;

// Number of pipelines you want to use
constexpr int NumStages = 3;

// This code section describe iterator algorithm selected is Analytic or Optimized
static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm = cutlass::conv::IteratorAlgorithm::kOptimized;

// This code section describes the epilogue part of the kernel, we use default value
constexpr int kCount = 128 / cutlass::sizeof_bits<ElementOutput>::value;
#if SAIL_FUSE_ALIGN_CUDNN
using ComputeEpilogueFragment = cutlass::Array<ElementOutput, kCount>;
#else
using ComputeEpilogueFragment = cutlass::Array<ElementComputeEpilogue, kCount>;
#endif
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationEpilogueFuse<
    ElementOutput,                                        // Data type of output matrix.
    kCount,     // The number of elements per vectorized.
                                                          // memory access. This becomes the vector width of
                                                          // math instructions in the epilogue too.
#if SAIL_EPILOGUE_OPT >= 2
    ElementOutput,
#else
    ElementAccumulator,                                   // Data type of accumulator
#endif
    ElementComputeEpilogue,                               // Data type for alpha in linear combination
    cutlass::epilogue::thread::ScaleType::Default,  // <- No scaling for input C
    cutlass::FloatRoundStyle::round_to_nearest,           // <- alpha x C + beta_e x input_E
    cutlass::epilogue::PrefetchStrategyType::Kind::kNoPrefetchNeeded,
    0,
    cutlass::plus<ComputeEpilogueFragment>,
    cutlass::epilogue::thread::ReLu<ComputeEpilogueFragment>>;

using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
  ElementInputA, LayoutInputA,
  ElementInputB, LayoutInputB,
  ElementOutput, LayoutOutput,
  ElementAccumulator,
  MMAOp,
  CuArch,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOp,
  SwizzleThreadBlock,
  NumStages,
  cutlass::arch::OpMultiplyAdd,
  IteratorAlgorithm,
  cutlass::conv::StrideSupport::kStrided,
  8,
  8,
  1
>::Kernel;

using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

/////////////////////////////////////////////////////////////////////////////////////////////////

int run() {
  // Print kernel info
  hggcFuncAttributes attr;
  hggcFuncGetAttributes(&attr, cutlass::Kernel<Conv2dFpropKernel>);
  printf("numRegs=%d, localSizeBytes=%zu\n", attr.numRegs, attr.localSizeBytes);

  // Construct Conv2dProblemSize with user defined output size
  int N = 8;
  int H = 56;
  int W = 56;
  int C = 64;
  int K = 256;
  int R = 1;
  int S = 1;
  cutlass::conv::Conv2dProblemSize problem_size(      
    {N, H, W, C},                               // activation 
    {K, R, S, C},                               // filter
    {0, 0, 0, 0},                                 // padding
    {1, 1},                                       // striding
    {1, 1},                                       // dilation
    cutlass::conv::Mode::kCrossCorrelation,       // mode (convolution or cross-correlation)
    1                                             // split-k slices
  );

  // Initialize tensors using CUTLASS helper functions
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(problem_size.activation_extent());
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(problem_size.filter_extent());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(problem_size.output_extent());

  // Create tensor C with dimensions 1x1x1xk which is the bias vector
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_e_bias({1, 1, 1, problem_size.K});

  // Create tensor D used to store output from CUTLASS kernel
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(problem_size.output_extent());
  // Create matrix D with dimensions M x N used to store output from reference
  // kernel
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(problem_size.output_extent());

  // Fill input and output matrices on host using CUTLASS helper functions
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(),
      1,
      ElementInputA(2),
      ElementInputA(-2),
      0);  // <- Fill tensor A on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_b.host_view(),
      1,
      ElementInputB(2),
      ElementInputB(-2),
      0);  // <- Fill tensor B on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_c.host_view(),
      1,
      ElementInputA(4),
      ElementInputA(-4),
      0);  // <- Fill tensor C on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_e_bias.host_view(),
      1,
      ElementOutput(4),
      ElementOutput(-4),
      0);  // <- Fill matrix E on host with uniform-distribution random data
  cutlass::reference::host::TensorFill(
      tensor_d.host_view());  // <- fill matrix D on host with zeros
  cutlass::reference::host::TensorFill(
      tensor_ref_d.host_view());  // <- fill matrix D for reference on host with zeros

  // Copy data from host to PPU
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_e_bias.sync_device();
  tensor_d.sync_device();
  tensor_ref_d.sync_device();

  // Initialize alpha for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta_c = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta_e = ElementComputeEpilogue(1);
  ElementComputeEpilogue act_param0 = ElementComputeEpilogue(0);
  ElementComputeEpilogue act_param1 = ElementComputeEpilogue(0);

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  Conv2dFpropKernel::TensorRefExtra extra_input_tensor_ref[EpilogueOp::kExtraEpilogueInputs]
      = { cutlass::TensorRef<ElementOutput, LayoutOutput>(tensor_e_bias.device_data(), LayoutOutput::Stride(0))};
  typename ImplicitGemm::Arguments arguments{
    problem_size,
    tensor_a.device_ref(),              // <- reference to tensor A on device
    tensor_b.device_ref(),              // <- reference to tensor B on device
    tensor_c.device_ref(),              // <- reference to tensor C on device
    tensor_d.device_ref(),              // <- reference to tensor D on device
    {alpha, beta_c, beta_e, act_param0, act_param1},
    cutlass::conv::SplitKMode::kSerial,
    extra_input_tensor_ref };                    

  // Instantiate CUTLASS kernel depending on templates
  ImplicitGemm implicit_gemm_op;

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check the problem size is supported or not
  cutlass::Status status = implicit_gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = implicit_gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  // Launch initialized CUTLASS kernel
  status = implicit_gemm_op();

  CUTLASS_CHECK(status);

  //
  // Create instantiation for device reference conv kernel
  //

  // Launch device reference to compute strictly the product A * B
  cutlass::reference::device::Conv2d<
      ElementInputA, 
      LayoutInputA, 
      ElementInputB, 
      LayoutInputB, 
      ElementOutput,
      LayoutOutput, 
      ElementComputeEpilogue, 
      ElementAccumulator,
      cutlass::NumericConverter<ElementOutput, ElementComputeEpilogue>>
    (
      cutlass::conv::Operator::kFprop, 
      problem_size, 
      tensor_a.device_ref(),
      tensor_b.device_ref(), 
      tensor_c.device_ref(), 
      tensor_ref_d.device_ref(),
      alpha, 0
    );

  // Wait for kernels to finish
  hggcDeviceSynchronize();

  // Copy output data from CUTLASS and reference kernel to host for comparison
  tensor_d.sync_host();
  tensor_ref_d.sync_host();

  // Compute bias + relu in host code
  for (int n = 0; n < problem_size.N; ++n) {
    for (int p = 0; p < problem_size.P; ++p) {
      for (int q = 0; q < problem_size.Q; ++q) {
        for (int k = 0; k < problem_size.K; ++k) {
          #if SAIL_FUSE_ALIGN_CUDNN
          tensor_ref_d.at({n, p, q, k}) = ElementOutput(std::max(ElementOutput(0),
                                     tensor_ref_d.at({n, p, q, k}) +
                                     tensor_c.at({n, p, q, k}) +
                                     tensor_e_bias.at({0, 0, 0, k})));
          #else
          tensor_ref_d.at({n, p, q, k}) = ElementOutput(std::max(ElementComputeEpilogue(0),
                                     ElementComputeEpilogue(tensor_ref_d.at({n, p, q, k})) +
                                     ElementComputeEpilogue(tensor_c.at({n, p, q, k})) +
                                     ElementComputeEpilogue(tensor_e_bias.at({0, 0, 0, k}))));
          #endif
        }
      }
    }
  }

  // Check if output from CUTLASS kernel and reference kernel are equal or not

  bool checkFail = false;
  for (int n = 0; n < problem_size.N; ++n) {
    for (int p = 0; p < problem_size.P; ++p) {
      for (int q = 0; q < problem_size.Q; ++q) {
        for (int k = 0; k < problem_size.K; ++k) {
          
          if(tensor_ref_d.at({n, p, q, k}) != tensor_d.at({n, p, q, k})) {
            std::cout << "ref: " << float(tensor_ref_d.at({n, p, q, k})) << ", now: " << float(tensor_d.at({n, p, q, k})) 
                      << " n=" << n << ", p=" << p << " q=" << q << ", k=" << k << std::endl;
            checkFail = true;
          }
        }
      }
    }
  }

  std::cout << (!checkFail ? "Passed" : "Failed") << std::endl;

  CUTLASS_CHECK(status);
  return 0;
}

int main(int argc, char const **args) {

  bool notSupported = false;

  hggcDeviceProp props;
  CUTLASS_PPU_CHECK(hggcGetDeviceProperties(&props, 0));

  if (notSupported) {
    return 0;
  }

  return run();
}

/////////////////////////////////////////////////////////////////////////////////////////////////
