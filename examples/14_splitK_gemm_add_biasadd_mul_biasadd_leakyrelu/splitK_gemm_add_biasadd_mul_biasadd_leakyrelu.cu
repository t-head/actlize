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
This example shows how to use split-k version of matrix multiplication using functions and data
structures provided by CUTLASS; which we run on a PPU.

What is split-k?
Consider a problem size of M = 128, N = 128, K = 4096. In this case, if my thread-block tile size (a
tile can be viewed as a 2d matrix) is 128x128x4096, then we launch a singled a thread-block taking
up a single CU of 84 CUs present on V100. Hence the efficiency of computation is really low. So, how
to solve it? This is where split-k comes in. It is a way of partitioning K-dimension of matrix
multiplication and distribute across multiple CUs and get better efficiency than single CU. In the
above example, we can partition K-dimension with split-k factor of 16 i.e., thread-block tile size
will be 128x128x256 and will be launching on 16 CUs. Once each thread-block computes their partial
inner product (1/16th of output), they accumulate to single output matrix.

Writing a single high performance matrix multiplication kernel is hard but do-able. Whereas writing
high performance kernels at scale which works for multiple problem sizes with good abstractions is
really hard. CUTLASS solves this problem by providing simplified abstractions to compose
multiple sections of gemm kernel. When used properly, the kernels can hit peak performance of PPU
easily.

CUTLASS divides a kernel into hierarchical composable sections. Which means, at each thread, warp
and thread-block level, they compute on their own tile-size with higher level of tile sizes being
composed from lower level ones. Multiple thread-tiles (tile size each thread computes) can be used
to form warp-tiles (tile size each warp computes) and multiple warp tiles can be used to compute
threadblock-tile (tile size computed by a threadblock).

In thie example, we split variable initialization into
1. Setting up data properties : describes how matrices are laid out in the memory and how the kernel
can view them (logical to physical mapping)
2. Setting up computation properties : describes how the above set matrices will be used to compute
output of matrix multiplication.

First, we setup the data types of matrices A, B, C and D along with alpha, beta as the equation for
GEMM is D = alpha * A * B + beta * C. In CUTLASS, the kernels first compute A * B and leaves the
rest of the computation to end of the kernel as alpha * X + beta * C is a simple element-wise
operation on X (A * B) and C. We call this as epilogue of kernel. Hence, we setup data types for
alpha and beta to be equal to ElementComputeEpilogue = float. As we want to MMA instructions on
PPU and they support only half-precision floating point (fp16 or half), we use data type for
elements in input matrix A and B as cutlass::half_t. PPU also supports accumulation of partial dot
product to fp32, which can store wider range of numbers, we use it as data type of output matrix
elements and accumulation. We convey this to CUTLASS kernel by initializing template variables
ElementAccumulator (float), ElementComputeEpilogue (float), ElementInputA (cutlass::half_t),
ElementInputB (cutlass::half_t), ElementOutput (float). Communicating just the data type is not
enough. As the data is laid out linearly in memory, we have to convey the layout of matrices. We do
that by initializing template variable LayoutInputA to column major cutlass variable, LayoutInputB
to row major and LayoutOutput to row major. Next, we setup rules to comptue alpha * X + beta * C
which is called epilogue of the kernel. We initialize template variable EpilogueOp, which takes the
data type of output ElementOutput (int32_t), the number of elements per vector memory access (16),
data type of accumulator (int32_t) and data type of computation of linear combination (alpha * X +
beta * C).

Now that we setup the properties of data, we have to setup properties of computation.

Second, we create template variables of tile sizes for thread-block, warp and mma-op to 128x128x32,
64x64x4, 8x8x4 (MxNxK) respectively. When passed to instantiate CUTLASS GEMM kernel, it internally
deduce the amount of threads needed per thread-block, amount of shared memory, storing data in
bank-conflict free manner, and ton of other variables required to compose, intialize and launch a
high performance GEMM kernel. This is the beauty of CUTLASS, it relieves developer from
understanding and coding complicated hardware optimizations which can easily go wrong.

There are few more template variables initialized such as, which threadblock tile of output matrix
is done which threadblock launched on an CU, device CU architecture of PPU you want to run on.

These are all put together to create a template variable which describes CUTLASS GEMM kernel using
cutlass::gemm::device::GemmSplitKParallel template.

The next step is to intialize physical data, instantiate and initialize CUTLASS kernel and run it.
We use CUTLASS utilities to initialize, fill, compare matrices as they are simple and doesn't come
in the way of learning CUTLASS.

Once all the matrices are initialized and filled with data, create arguments tuple to launch CUTLASS
kernel which takes problem size (M = 5120, N = 4096 and K = 4096), matrices, alpha, beta and the
important one, split k-dimension factor. Along with that, we query CUTLASS if any scratch-space
memory required by the kernel we instantiated. If yes, we create it and pass it along with other
arguments created to intialize CUTLASS kernel then, the kernel is launched.

In this example, we later on launch a reference gemm kernel (from CUTLASS utilities) to compare if
the output from CUTLASS kernel is same as reference GEMM kernel.
*/

#include <iostream>

// PTG cutlass propriatary configs
#include "accutlass.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_splitk_parallel.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination_epilogue_fuse.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "helper.h"

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = float;                   // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
using ElementOutput = float;                        // <- data type of elements in output matrix D
using ElementFuseInputE = ElementOutput;          // <- data type of elements in input matrix E

// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

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

// This code section describes ?
constexpr int kCount = 128 / cutlass::sizeof_bits<ElementOutput>::value;
using ComputeEpilogueFragment = cutlass::Array<ElementComputeEpilogue, kCount>;
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationEpilogueFuse<
    ElementOutput,                                     // <- data type of output matrix
    kCount,                                            // <- This is the number of elements per
                                                       // vectorized memory access. For half
                                                       // precision, it's 8 elements. This becomes
                                                       // the vector width of math instructions in
                                                       // epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue,                               // <- data type for alpha in linear combination function
    cutlass::epilogue::thread::ScaleType::Default,  // <- No scaling for input C
    cutlass::FloatRoundStyle::round_to_nearest,           // <- alpha x C + beta_e x input_E
    cutlass::epilogue::PrefetchStrategyType::Kind::kNoPrefetchNeeded,
    0,
    cutlass::plus<ComputeEpilogueFragment>,
    cutlass::multiplies<ComputeEpilogueFragment>,
    cutlass::plus<ComputeEpilogueFragment>,
    cutlass::epilogue::thread::LeakyReLu<ComputeEpilogueFragment>>;

// Put all the created template variables to create GemmSplitKParallel template variable
using Gemm = cutlass::gemm::device::GemmSplitKParallel<ElementInputA,
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
                                                       EpilogueOp>;

int run() {

  hggcDeviceProp props;

  hggcError_t error = hggcGetDeviceProperties(&props, 0);
  if (error != hggcSuccess) {
    std::cerr << "hggcGetDeviceProperties() returned an error: " << hggcGetErrorString(error) << std::endl;
    return -1;
  }

  //
  // Define problem size
  //

  const int length_m = 5120;
  const int length_n = 4096;
  const int length_k = 4096;

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
      {1, problem_size.n()});  // <- Create vector E with dimensions M x 1
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_f(
      {1, problem_size.n()});  // <- Create vector F with dimensions M x 1
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_g(
      {1, problem_size.n()});  // <- Create vector G with dimensions M x 1

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
      ElementInputA(4),
      ElementInputA(-4),
      0);  // <- Fill matrix A on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_b.host_view(),
      1,
      ElementInputB(4),
      ElementInputB(-4),
      0);  // <- Fill matrix B on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_c.host_view(),
      1,
      ElementOutput(4),
      ElementOutput(-4),
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
  tensor_d.sync_device();
  tensor_ref_d.sync_device();

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta_c = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta_e = ElementComputeEpilogue(1);
  ElementOutput leaky_alpha = ElementOutput(1);

  // Split K dimension into 16 partitions
  int split_k_slices = 16;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  cutlass::TensorRef<ElementFuseInputE const, LayoutOutput> extra_input_tensor_ref[EpilogueOp::kExtraEpilogueInputs]
      = { cutlass::TensorRef<ElementFuseInputE, LayoutOutput>(tensor_e.device_data(), 0),
          cutlass::TensorRef<ElementFuseInputE, LayoutOutput>(tensor_f.device_data(), 0),
          cutlass::TensorRef<ElementFuseInputE, LayoutOutput>(tensor_g.device_data(), 0) };

  //printf("tensor_e.host_data()=%f\n", *(float*)tensor_e.host_data());
  typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                     tensor_a.device_ref(),  // <- reference to matrix A on device
                                     tensor_b.device_ref(),  // <- reference to matrix B on device
                                     tensor_c.device_ref(),  // <- reference to matrix C on device
                                     tensor_d.device_ref(),  // <- reference to matrix D on device
                                     {alpha, beta_c, beta_e, ElementComputeEpilogue(0), ElementComputeEpilogue(leaky_alpha)},   // <- alpha
                                     split_k_slices,                    // <- k-dimension split factor
                                     typename cutlass::epilogue::thread::Convert<
                                        ElementAccumulator,
                                        cutlass::gemm::device::DefaultGemmConfiguration< MMAOp, CuArch, ElementInputA, ElementInputB,
                                                                  ElementAccumulator,
                                                                  ElementAccumulator>::EpilogueOutputOp::kCount,
                                                                  ElementAccumulator >::Params(),
                                     typename cutlass::reduction::thread::ReduceAdd< ElementAccumulator, typename EpilogueOp::ElementAccumulator,
                                                                                     EpilogueOp::kCount >::Params(),
                                     extra_input_tensor_ref             // <- the E matrix is treated as the first bias vector, call the TensorRef constructor with pointer & Layout arguments.
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

  // Create instantiation for device reference gemm kernel
  cutlass::reference::device::Gemm<ElementInputA,
                                   LayoutInputA,
                                   ElementInputB,
                                   LayoutInputB,
                                   ElementOutput,
                                   LayoutOutput,
                                   ElementComputeEpilogue,
                                   ElementComputeEpilogue>
      gemm_device;

  // Launch device reference gemm kernel
  gemm_device(problem_size,
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
      ElementOutput temp = ElementOutput(tensor_ref_d.at({i, j}) + tensor_e.at({0, j}));
      temp = temp * tensor_f.at({0, j}) + tensor_g.at({0, j});
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

  std::cout << (!checkFail ? "Passed" : "Failed") << std::endl;

  return (!checkFail ? 0  : -1);
}

int main() {
  return run();
}

