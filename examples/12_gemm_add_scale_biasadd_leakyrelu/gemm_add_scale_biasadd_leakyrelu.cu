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
*/

#include <algorithm>
#include <iostream>

// PTG cutlass propriatary configs
#include "accutlass.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination_epilogue_fuse.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/numeric_conversion.h"
#include "helper.h"



// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = cutlass::half_t;         // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
using ElementOutput = cutlass::half_t;              // <- data type of elements in output matrix D
using ElementFuseInputE = ElementOutput;          // <- data type of elements in input matrix E

// The code section below describes matrix layout of input and output matrices.
// Column Major for Matrix A, B and C.
//
// Note this example only works for ColumnMajor output because
//   1) we only have row major epilogue.
//   2) we swap A and B if the output is column major then we can still use the
//      row major epilogue.
//   3) Mx1 bias vector becomes 1xM after the swapping/transposing.
//   4) we can use the existing OutputIterator to load 1xM bias vector.

using LayoutInputA = cutlass::layout::ColumnMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::ColumnMajor;

// This code section describes whether you want to use tensor cells or regular SIMT cores on PPU CU
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes device CU architecture number
using CuArch = cutlass::arch::PPU0010;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 128, 32>;  // <- threadblock tile M = 128, N = 128, K = 32
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 32 
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 16, 16>;  // <- MMA Op tile M = 16, N = 16, K = 16

// This code section describes how threadblocks are scheduled on PPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;  // <- ??

// Define the epilogue operation as LinearCombinationLeakyRelu. Assumes leaky_alpha <= 1, this is approximately equal to
//    temp_ij = alpha * sum_k(a_ik * b_kj) + c_ij + e_j
//    d_ij = max(temp_ij, leaky_alpha * temp_ij)
//
constexpr int kCount = 128 / cutlass::sizeof_bits<ElementOutput>::value;
using ComputeEpilogueFragment = cutlass::Array<ElementComputeEpilogue, kCount>;
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationEpilogueFuse<
    ElementOutput,                                        // <- data type of output matrix
    kCount,                                               // <- this is the number of elements per
                                                          // vectorized memory access. For half
                                                          // precision, it's 8 elements. This becomes
                                                          // the vector width of math instructions in
                                                          // epilogue too
    ElementAccumulator,                                   // <- data type of accumulator
    ElementComputeEpilogue,                               // <- data type for alpha in linear combination function
    cutlass::epilogue::thread::ScaleType::Default,  // <- No scaling for input C
    cutlass::FloatRoundStyle::round_to_nearest,           // <- alpha x C + beta_e x input_E
    cutlass::epilogue::PrefetchStrategyType::Kind::kNoPrefetchNeeded,
    0,
    cutlass::multiplies<ComputeEpilogueFragment>,
    cutlass::plus<ComputeEpilogueFragment>,
    cutlass::epilogue::thread::LeakyReLu<ComputeEpilogueFragment>>;

// Number of pipelines you want to use
constexpr int NumStages = 2;

using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         CuArch,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp,
                                         EpilogueOp,
                                         SwizzleThreadBlock,
                                         NumStages,
                                         cutlass::gemm::device::DefaultGemmConfiguration<MMAOp, CuArch, ElementInputA, ElementInputB,
                                            ElementOutput, ElementAccumulator>::kAlignmentA,
                                         cutlass::gemm::device::DefaultGemmConfiguration<MMAOp, CuArch, ElementInputA, ElementInputB,
                                            ElementOutput, ElementAccumulator>::kAlignmentB,
                                        false,  // <- split-K with serial reduction
                                        typename cutlass::gemm::device::DefaultGemmConfiguration<
                                            MMAOp, CuArch, ElementInputA, ElementInputB, ElementOutput,
                                            ElementAccumulator>::Operator>;

int run() {

  const int length_m = 5120;
  const int length_n = 4096;
  const int length_k = 4096;
  ElementOutput leaky_alpha = ElementOutput(0.1);

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);

  // Initialize tensors using CUTLASS helper functions
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
      problem_size.mk());  // <- Create matrix A with dimensions M x K
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
      problem_size.kn());  // <- Create matrix B with dimensions K x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
      problem_size.mn());  // <- Create matrix C with dimensions M x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_e(
      {problem_size.m(), 1});  // <- Create vector E with dimensions M x 1
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_f(
      {problem_size.m(), 1});  // <- Create vector F with dimensions M x 1
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_g(
      {problem_size.m(), 1});  // <- Create vector G with dimensions M x 1

  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // CUTLASS kernel
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // reference kernel

  // Fill input and output matrices on host using CUTLASS helper functions
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(),
      1,
      ElementInputA(2),
      ElementInputA(-2),
      0);  // <- Fill matrix A on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_b.host_view(),
      1,
      ElementInputB(2),
      ElementInputB(-2),
      0);  // <- Fill matrix B on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_c.host_view(),
      1,
      ElementOutput(2),
      ElementOutput(-2),
      0);  // <- Fill matrix C on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_e.host_view(),
      1,
      ElementOutput(4),
      ElementOutput(-4),
      0);  // <- Fill vector E on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_f.host_view(),
      1,
      ElementOutput(4),
      ElementOutput(-4),
      0);  // <- Fill vector F on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_g.host_view(),
      1,
      ElementOutput(4),
      ElementOutput(-4),
      0);  // <- Fill vector G on host with uniform-distribution random data
  cutlass::reference::host::TensorFill(
      tensor_d.host_view());  // <- fill matrix D on host with zeros
  cutlass::reference::host::TensorFill(
      tensor_ref_d.host_view());  // <- fill matrix D for reference on host with zeros

  // Copy data from host to PPU
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_e.sync_device();
  tensor_f.sync_device();
  tensor_g.sync_device();
  tensor_d.sync_device();
  tensor_ref_d.sync_device();

  // Initialize alpha for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta_c = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta_e = ElementComputeEpilogue(1);

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  cutlass::TensorRef<ElementFuseInputE const, LayoutOutput> extra_input_tensor_ref[EpilogueOp::kExtraEpilogueInputs + 1]
      = { cutlass::TensorRef<ElementFuseInputE, LayoutOutput>(tensor_f.device_data(), 0),
          cutlass::TensorRef<ElementFuseInputE, LayoutOutput>(tensor_g.device_data(), 0) };
  typename Gemm::Arguments arguments{
    problem_size,                       // <- problem size of matrix multiplication
    tensor_a.device_ref(),              // <- reference to matrix A on device
    tensor_b.device_ref(),              // <- reference to matrix B on device
    tensor_c.device_ref(),              // <- reference to matrix C on device
    tensor_d.device_ref(),              // <- reference to matrix D on device
    {alpha, beta_c, beta_e, ElementComputeEpilogue(0), ElementComputeEpilogue(leaky_alpha)},                              // <- alpha
    split_k_slices,                     // <- k-dimension split factor
    extra_input_tensor_ref              // <- the E matrix is treated as the first bias vector, call the TensorRef constructor with pointer & Layout arguments.
                                        // <- the F matrix is treated as the scaling vector, call the TensorRef constructor with pointer & Layout arguments.
                                        // <- the G matrix is treated as the second bias vector, call the TensorRef constructor with pointer & Layout arguments.
    };

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Check the problem size is supported or not 
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  // Launch initialized CUTLASS kernel
  status = gemm_op();
  CUTLASS_CHECK(status);

  //
  // Create instantiation for device reference gemm kernel
  //

  cutlass::reference::device::Gemm<ElementInputA,
                                   LayoutInputA,
                                   ElementInputB,
                                   LayoutInputB,
                                   ElementOutput,
                                   LayoutOutput,
                                   ElementComputeEpilogue,
                                   ElementComputeEpilogue>
      gemm_device_reference;

  // Launch device reference to compute strictly the product A * B
  gemm_device_reference(
    problem_size,
    alpha,
    tensor_a.device_ref(),
    tensor_b.device_ref(),
    beta_c,
    tensor_c.device_ref(),
    tensor_ref_d.device_ref());

  // Wait for kernels to finish
  hggcDeviceSynchronize();

  // Copy output data from CUTLASS and reference kernel to host for comparison
  tensor_d.sync_host();
  tensor_ref_d.sync_host();

  // Compute bias + relu in host code
  for (int i = 0; i < problem_size.m(); ++i) {
    for (int j = 0; j < problem_size.n(); ++j) {
      ElementOutput temp = ElementOutput(tensor_ref_d.at({i, j}));
      temp = temp * tensor_f.at({i, 0}) + tensor_g.at({i, 0});
      tensor_ref_d.at({i, j}) = std::max(
        ElementOutput(temp * leaky_alpha), 
        temp
      );
    }
  }

  bool checkFail = false;
  for (int i = 0; i < problem_size.m() && !checkFail; ++i) {
    for (int j = 0; j < problem_size.n() && !checkFail; ++j) {
      if(tensor_ref_d.at({i,j}) != tensor_d.at({i,j})) {
        std::cout << "ref: " << float(tensor_ref_d.at({i,j})) << ", now: " << float(tensor_d.at({i,j})) << " i=" << i << ", j=" << j << std::endl;
        checkFail = true;
      }
    }
  }
  // Check if output from CUTLASS kernel and reference kernel are equal or not
  std::cout << (checkFail == false
                    ? "Passed"
                    : "Failed")
            << std::endl;

  CUTLASS_CHECK(status);
  return 0;
}

int main() {

  bool notSupported = false;

  // CUTLASS must be compiled with PPU SDK to run these examples.
  hggcDeviceProp props;

  hggcError_t error = hggcGetDeviceProperties(&props, 0);
  if (error != hggcSuccess) {
    std::cerr << "hggcGetDeviceProperties() returned an error: " << hggcGetErrorString(error) << std::endl;
    return -1;
  }

  if (notSupported) {
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  return run();
}
  
