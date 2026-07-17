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

#include <iostream>
#include <vector>

// PTG cutlass propriatary configs
#include "accutlass.h"

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm_batched.h"

#pragma warning( disable : 4503)

/*
This example demonstrates how to use cutlass to compute a batched strided gemm.
In this example, both A and B matrix are non-transpose and column major matrix
batched_C = batched_A x batched_B
As an example, matrix C can be seen as
-----------------------------------------------------------
(0,0,0) | (0,0,1) | (0,0,2) | (1,0,0) | (1,0,1) | (1,0,2) |
-----------------------------------------------------------
(0,1,0) | (0,1,1) | (0,1,2) | (1,1,0) | (1,1,1) | (1,1,2) |
-----------------------------------------------------------
(0,2,0) | (0,2,1) | (0,2,2) | (1,2,0) | (1,2,1) | (1,2,2) |
-----------------------------------------------------------
(0,3,0) | (0,3,1) | (0,3,2) | (1,3,0) | (1,3,1) | (1,3,2) |
-----------------------------------------------------------
(0,4,0) | (0,4,1) | (0,4,2) | (1,4,0) | (1,4,1) | (1,4,2) |
-----------------------------------------------------------
(0,5,0) | (0,5,1) | (0,5,2) | (1,5,0) | (1,5,1) | (1,5,2) |
-----------------------------------------------------------
           batch 0          |           batch 1
where we denote each element with (batch_idx, row_idx, column_idx)
In this example, batch size is 2, M is 6 and N is 3
The stride (batch_stride_C) between the first element of two batches is ldc * n

matrix A can be seen as
---------------------------------------
(0,0,0) | (0,0,1) | (1,0,0) | (1,0,1) |
---------------------------------------
(0,1,0) | (0,1,1) | (1,1,0) | (1,1,1) |
---------------------------------------
(0,2,0) | (0,2,1) | (1,2,0) | (1,2,1) |
---------------------------------------
(0,3,0) | (0,3,1) | (1,3,0) | (1,3,1) |
---------------------------------------
(0,4,0) | (0,4,1) | (1,4,0) | (1,4,1) |
---------------------------------------
(0,5,0) | (0,5,1) | (1,5,0) | (1,5,1) |
---------------------------------------
     batch 0      |      batch 1
, where batch size is 2, M is 6 and K is 2
The stride (batch_stride_B) between the first element of two batches is lda * k

matrix B can be seen as
-----------------------------
(0,0,0) | (0,0,1) | (0,0,2) |
----------------------------- batch 0
(0,1,0) | (0,1,1) | (0,1,2) |
-------------------------------------
(1,0,0) | (1,0,1) | (1,0,2) |
----------------------------- batch 1
(1,1,0) | (1,1,1) | (1,1,2) |
-----------------------------
, where the batch size is 2, N is 3 and K is 2
The stride (batch_stride_C) between the first element of two batches is k


*/

hggcError_t cutlass_strided_batched_sgemm(
  int m, 
  int n,
  int k,
  float alpha,
  float const *A,
  int lda,
  long long int batch_stride_A,
  float const *B,
  int ldb,
  long long int batch_stride_B,
  float *C,
  int ldc,
  long long int batch_stride_C,
  float *E,
  long long int batch_stride_E,
  float beta,
  int batch_count) {

  using ElementAccumulator = float;         // <- data type of accumulator
  using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
  using ElementInputA = float;              // <- data type of elements in input matrix A
  using ElementInputB = float;              // <- data type of elements in input matrix B
  using ElementOutput = float;              // <- data type of elements in output matrix D
  using ElementFuseInputE = ElementOutput;          // <- data type of elements in input matrix E

  using LayoutInputA = cutlass::layout::ColumnMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  
  // This code section describes whether you want to use tensor cells or regular SIMT cores on PPU CU
  using MMAOp = cutlass::arch::OpClassTensorOp;
  
  // This code section describes device CU architecture number
  using CuArch = cutlass::arch::PPU0010;
  
  // This code section describes the tile size a thread block will compute
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 256, 32>;
  // This code section describes tile size a warp will compute
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 32
  // This code section describes the size of MMA op
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 16, 16>;  // <- MMA Op tile M = 8, N = 8, K = 4
  
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
      ElementAccumulator,                                   // Data type of accumulator
      ElementComputeEpilogue,                               // <- data type for alpha in linear combination function
      cutlass::epilogue::thread::ScaleType::Default,  // <- No scaling for input C
      cutlass::FloatRoundStyle::round_to_nearest,           // <- alpha x C + beta_e x input_E
      cutlass::epilogue::PrefetchStrategyType::Kind::kNoPrefetchNeeded,
      0,
      cutlass::plus<ComputeEpilogueFragment>,
      cutlass::epilogue::thread::ReLu<ComputeEpilogueFragment>>;

    using Gemm = cutlass::gemm::device::GemmBatched<
      ElementInputA, LayoutInputA,
      ElementInputB, LayoutInputB,
      ElementOutput, LayoutOutput,
      ElementAccumulator, MMAOp,
      CuArch,
      ShapeMMAThreadBlock,
      ShapeMMAWarp,
      ShapeMMAOp,
      EpilogueOp,
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
      2,
      4,
      4
    >;

   // Print kernel info
  // hggcFuncAttributes attr;
  // hggcFuncGetAttributes(&attr, cutlass::Kernel<Gemm::GemmKernel>);
  // printf("numRegs=%u, localSizeBytes=%zu\n", attr.numRegs, attr.localSizeBytes);

  ElementComputeEpilogue beta_e = ElementComputeEpilogue(1);
  cutlass::TensorRef<ElementFuseInputE const, LayoutOutput> extra_input_tensor_ref[EpilogueOp::kExtraEpilogueInputs]
      = { cutlass::TensorRef<ElementFuseInputE, LayoutOutput>(E, 0) };
  int64_t extra_input_strides[EpilogueOp::kExtraEpilogueInputs + 1] = {batch_stride_E};
  Gemm gemm_op;

  // Print kernel info
  hggcFuncAttributes attr;
  hggcFuncGetAttributes(&attr, cutlass::Kernel<Gemm::GemmKernel>);
  printf("numRegs=%u, localSizeBytes=%zu\n", attr.numRegs, attr.localSizeBytes);

  cutlass::Status status = gemm_op({
    {m, n, k},
    {A, lda}, 
    batch_stride_A,
    {B, ldb}, 
    batch_stride_B,
    {C, ldc}, 
    batch_stride_C,
    {C, ldc}, 
    batch_stride_C,
    {alpha, beta, beta_e, ElementComputeEpilogue(0), ElementComputeEpilogue(0)},
    batch_count,
    extra_input_tensor_ref,
    extra_input_strides
  });

  if (status != cutlass::Status::kSuccess) {
    return hggcErrorUnknown;
  }

  return hggcSuccess;
}

template<typename T> 
hggcError_t strided_batched_gemm_nn_reference(
  int m,
  int n,
  int k,
  T alpha,
  std::vector<T> const &A, 
  int lda,
  long long int batch_stride_A,
  std::vector<T> const &B, 
  int ldb,
  long long int batch_stride_B,
  std::vector<T> &C, 
  int ldc,
  long long int batch_stride_C,
  std::vector<T> &E,
  long long int batch_stride_E,
  T beta,
  int batch_count) {
  /*
  strided batched gemm NN
  */
  
  hggcError_t result = hggcSuccess;

  if (A.size() < lda * k * batch_count) {
    std::cout << "the size of A is too small" << std::endl;
    return hggcErrorInvalidValue;
  }
  if (B.size() < ldb * n) {
    std::cout << "the size of B is too small" << std::endl;
    return hggcErrorInvalidValue;
  }
  if (C.size() < ldc * n * batch_count) {
    std::cout << "the size of C is too small" << std::endl;
    return hggcErrorInvalidValue;
  }
  if (E.size() < batch_stride_E * batch_count) {
    std::cout << "the size of E is too small" << std::endl;
    return hggcErrorInvalidValue;
  }
  
  for (int batch_idx = 0; batch_idx < batch_count; batch_idx++) {
    for (int n_idx = 0; n_idx < n; n_idx++) {
      for (int m_idx = 0; m_idx < m; m_idx++) {
        T accum = beta * C[batch_idx * batch_stride_C + n_idx * ldc + m_idx];
        for (int k_idx = 0; k_idx < k; k_idx++) {
          accum += alpha 
            * A[batch_idx * batch_stride_A + k_idx * lda + m_idx]
            * B[batch_idx * batch_stride_B + n_idx * ldb + k_idx];
        }
        accum += E[batch_idx * batch_stride_E + m_idx];
        if (accum < 0)
          accum = 0;
        C[batch_idx * batch_stride_C + n_idx * ldc + m_idx] = accum;
      }
    }
  }

  return result;
}

int main() {

  // Arbitrary problem size
  int const m = 512;
  int const n = 256;
  int const k = 1024;
  int const batch_count = 16;

  // A, B are non-transpose, column major
  int const lda = m;
  int const ldb = k * batch_count;
  int const ldc = m;

  int const count_A = batch_count * lda * k;
  int const count_B = ldb * n;
  int const count_C = batch_count * ldc * n;
  int const count_E = batch_count * m;

  // the memory is batched along K dimension
  long long int batch_stride_A = static_cast<long long int>(lda) * static_cast<long long int>(k);
  long long int batch_stride_B = static_cast<long long int>(k);
  long long int batch_stride_C = static_cast<long long int>(ldc) * static_cast<long long int>(n);
  long long int batch_stride_bias = static_cast<long long int>(m);

  // alpha and beta
  float alpha = 1.0f;
  float beta = 2.0f;

  hggcError_t result = hggcSuccess;

  // allocate the host memory
  std::vector<float> host_A(count_A);
  std::vector<float> host_B(count_B);
  std::vector<float> host_C(count_C);
  std::vector<float> result_C(count_C);
  std::vector<float> host_E(count_E);

  // allocate the device memory
  float *A;
  float *B;
  float *C;
  float *E;

  result = hggcMalloc(&A, count_A * sizeof(float));
  if (result != hggcSuccess) {
    std::cerr << "hggcMalloc result = " << result << std::endl;
    return result;
  }
  result = hggcMalloc(&B, count_B * sizeof(float));
  if (result != hggcSuccess) {
    std::cerr << "hggcMalloc result = " << result << std::endl;
    return result;
  }
  result = hggcMalloc(&C, count_C * sizeof(float));
  if (result != hggcSuccess) {
    std::cerr << "hggcMalloc result = " << result << std::endl;
    return result;
  }
  result = hggcMalloc(&E, count_E * sizeof(float));
  if (result != hggcSuccess) {
    std::cerr << "hggcMalloc result = " << result << std::endl;
    return result;
  }

  // Limit range to avoid floating-point errors
  int const kRange = 8;

  // fill A
  for (int b_idx = 0; b_idx < batch_count; b_idx++) {
    for (int col_idx = 0; col_idx < k; col_idx++) {
      for (int row_idx = 0; row_idx < m; row_idx++) {
        host_A[row_idx + col_idx * lda + b_idx * lda * k] = static_cast<float>((row_idx + col_idx * lda + b_idx * lda * k) % kRange);
      }
    }
  }
  // fill B
  for (int b_idx = 0; b_idx < batch_count; b_idx++) {
    for (int col_idx = 0; col_idx < n; col_idx++) {
      for (int row_idx = 0; row_idx < k; row_idx++) {
        host_B[row_idx + col_idx * ldb + b_idx * k] = static_cast<float>(((n + k * ldb + batch_count * k) - (row_idx + col_idx * ldb + b_idx * k)) % kRange) * (-1 ^(row_idx&1));
      }
    }
  }
  // fill C
  for (int b_idx = 0; b_idx < batch_count; b_idx++) {
    for (int col_idx = 0; col_idx < n; col_idx++) {
      for (int row_idx = 0; row_idx < m; row_idx++) {
        host_C[row_idx + col_idx * ldc + b_idx * ldc * n] = 1.f;
      }
    }
  }
  // fill E
  for (int b_idx = 0; b_idx < batch_count; b_idx++) {
    for (int row_idx = 0; row_idx < m; row_idx++) {
      host_E[row_idx + b_idx * m] = b_idx * row_idx;
    }
  }

  // ref memory
  std::vector<float> ref_A(host_A);
  std::vector<float> ref_B(host_B);
  std::vector<float> ref_C(host_C);
  std::vector<float> ref_E(host_E);

  // copy host memory to device
  result = hggcMemcpy(A, host_A.data(), count_A * sizeof(float), hggcMemcpyHostToDevice);
  if (result != hggcSuccess) {
    std::cerr << "hggcMemcpy result = " << result << std::endl;
    return result;
  }
  result = hggcMemcpy(B, host_B.data(), count_B * sizeof(float), hggcMemcpyHostToDevice);
  if (result != hggcSuccess) {
    std::cerr << "hggcMemcpy result = " << result << std::endl;
    return result;
  }
  result = hggcMemcpy(C, host_C.data(), count_C * sizeof(float), hggcMemcpyHostToDevice);
  if (result != hggcSuccess) {
    std::cerr << "hggcMemcpy result = " << result << std::endl;
    return result;
  }
  result = hggcMemcpy(E, host_E.data(), count_E * sizeof(float), hggcMemcpyHostToDevice);
  if (result != hggcSuccess) {
    std::cerr << "hggcMemcpy result = " << result << std::endl;
    return result;
  }

  // run cutlass
  result = cutlass_strided_batched_sgemm(
    m, n, k, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, E, batch_stride_bias,
    beta, batch_count);
  if (result != hggcSuccess)
    return result;

  // copy device memory to host
  result = hggcMemcpy(result_C.data(), C, count_C * sizeof(float), hggcMemcpyDeviceToHost);
  if (result != hggcSuccess) {
    std::cerr << "hggcMemcpy result = " << result << std::endl;
    return result;
  }

  //compare with reference code
  result = strided_batched_gemm_nn_reference(m, n, k, alpha, ref_A, lda, batch_stride_A, ref_B, ldb, batch_stride_B, ref_C, ldc, batch_stride_C, ref_E, batch_stride_bias,
    beta, batch_count);
  if (result != 0)
    return result;

  // Expect bit-level accuracy for this simple example
  if (ref_C != result_C) {
    std::cout << "CUTLASS strided batched gemm does not run correctly" << std::endl;
    return hggcErrorUnknown;
  }

  // free memory
  result = hggcFree(A);
  if (result != hggcSuccess) {
    std::cerr << "hggcFree result = " << result << std::endl;
    return result;
  }
  result = hggcFree(B);
  if (result != hggcSuccess) {
    std::cerr << "hggcFree result = " << result << std::endl;
    return result;
  }
  result = hggcFree(C);
  if (result != hggcSuccess) {
    std::cerr << "hggcFree result = " << result << std::endl;
    return result;
  }


  if (result == hggcSuccess) {
    std::cout << "Passed." << std::endl;
  }

  // Exit.
  return result == hggcSuccess ? 0 : -1;
}
