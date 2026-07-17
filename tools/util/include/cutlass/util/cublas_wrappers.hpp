/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved. 
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <hggc_runtime.h>
#include <acblas_v2.h>

//-- BLAM_DEBUG_OUT ---------------------------------------------------------
#ifdef BLAM_DEBUG
# include <iostream>
# ifndef BLAM_DEBUG_OUT
#  define BLAM_DEBUG_OUT(msg)    std::cerr << "BLAM: " << msg << std::endl
#  define BLAM_DEBUG_OUT_2(msg)  std::cerr << msg << std::endl
# endif // BLAM_DEBUG_OUT
#else
# ifndef BLAM_DEBUG_OUT
#  define BLAM_DEBUG_OUT(msg)
#  define BLAM_DEBUG_OUT_2(msg)
# endif // BLAM_DEBUG_OUT
#endif // BLAM_DEBUG

// User could potentially define ComplexFloat/ComplexDouble instead of std::
#ifndef BLAM_COMPLEX_TYPES
#define BLAM_COMPLEX_TYPES 1
#include <hggc/std/complex>
namespace blam {
template <typename T>
using Complex       = hggc::std::complex<T>;
using ComplexFloat  = hggc::std::complex<float>;
using ComplexDouble = hggc::std::complex<double>;
}
#endif // BLAM_COMPLEX_TYPES

// User could potentially define Half instead of cute::
#ifndef BLAM_HALF_TYPE
#define BLAM_HALF_TYPE 1
#include <cute/numeric/half.hpp>
namespace blam {
using Half = cute::half_t;
}
#endif // BLAM_HALF_TYPE

namespace blam
{
namespace acblas
{

inline const char*
acblas_get_error(acblasStatus_t status)
{
  switch (status) {
    case ACBLAS_STATUS_SUCCESS:
      return "ACBLAS_STATUS_SUCCESS";
    case ACBLAS_STATUS_NOT_INITIALIZED:
      return "ACBLAS_STATUS_NOT_INITIALIZED -- The acBLAS library was not initialized.";
    case ACBLAS_STATUS_ALLOC_FAILED:
      return "ACBLAS_STATUS_ALLOC_FAILED -- Resource allocation failed inside the acBLAS library.";
    case ACBLAS_STATUS_INVALID_VALUE:
      return "ACBLAS_STATUS_INVALID_VALUE -- An unsupported value or parameter was passed to the function.";
    case ACBLAS_STATUS_ARCH_MISMATCH:
      return "ACBLAS_STATUS_ARCH_MISMATCH -- The function requires a feature absent from the device architecture.";
    case ACBLAS_STATUS_MAPPING_ERROR:
      return "ACBLAS_STATUS_MAPPING_ERROR -- An access to GPU memory space failed.";
    case ACBLAS_STATUS_EXECUTION_FAILED:
      return "ACBLAS_STATUS_EXECUTION_FAILED -- The GPU program failed to execute.";
    case ACBLAS_STATUS_INTERNAL_ERROR:
      return "ACBLAS_STATUS_INTERNAL_ERROR -- An internal acBLAS operation failed.";
    case ACBLAS_STATUS_NOT_SUPPORTED:
      return "ACBLAS_STATUS_NOT_SUPPORTED -- The functionality requested is not supported.";
    case ACBLAS_STATUS_LICENSE_ERROR:
      return "ACBLAS_STATUS_LICENSE_ERROR -- An error was detected when checking the current licensing.";
    default:
      return "ACBLAS_ERROR -- <unknown>";
  }
}

inline bool
acblas_is_error(acblasStatus_t status)
{
  return status != ACBLAS_STATUS_SUCCESS;
}


// hgemm
inline acblasStatus_t
gemm(acblasHandle_t handle,
     acblasOperation_t transA, acblasOperation_t transB,
     int m, int n, int k,
     const Half* alpha,
     const Half* A, int ldA,
     const Half* B, int ldB,
     const Half* beta,
     Half* C, int ldC)
{
  BLAM_DEBUG_OUT("acblasHgemm");

  return acblasGemmEx(handle, transA, transB,
                      m, n, k,
                      reinterpret_cast<const __half*>(alpha),
                      reinterpret_cast<const __half*>(A), HGGC_R_16F, ldA,
                      reinterpret_cast<const __half*>(B), HGGC_R_16F, ldB,
                      reinterpret_cast<const __half*>(beta),
                      reinterpret_cast<      __half*>(C), HGGC_R_16F, ldC,
                      HGGC_R_16F, ACBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// mixed hf gemm
inline acblasStatus_t
gemm(acblasHandle_t handle,
     acblasOperation_t transA, acblasOperation_t transB,
     int m, int n, int k,
     const float* alpha,
     const Half* A, int ldA,
     const Half* B, int ldB,
     const float* beta,
     float* C, int ldC)
{
  BLAM_DEBUG_OUT("acblasGemmEx mixed half-float");

  return acblasGemmEx(handle, transA, transB,
                      m, n, k,
                      alpha,
                      reinterpret_cast<const __half*>(A), HGGC_R_16F, ldA,
                      reinterpret_cast<const __half*>(B), HGGC_R_16F, ldB,
                      beta,
                      C, HGGC_R_32F, ldC,
                      HGGC_R_32F, ACBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// igemm
inline acblasStatus_t
gemm(acblasHandle_t handle,
     acblasOperation_t transA, acblasOperation_t transB,
     int m, int n, int k,
     const int32_t* alpha,
     const int8_t* A, int ldA,
     const int8_t* B, int ldB,
     const int32_t* beta,
     int32_t* C, int ldC)
{
  BLAM_DEBUG_OUT("acblasIgemm");

  return acblasGemmEx(handle, transA, transB,
                      m, n, k,
                      alpha,
                      A, HGGC_R_8I, ldA,
                      B, HGGC_R_8I, ldB,
                      beta,
                      C, HGGC_R_32I, ldC,
                      HGGC_R_32I, ACBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// sgemm
inline acblasStatus_t
gemm(acblasHandle_t handle,
     acblasOperation_t transA, acblasOperation_t transB,
     int m, int n, int k,
     const float* alpha,
     const float* A, int ldA,
     const float* B, int ldB,
     const float* beta,
     float* C, int ldC)
{
  BLAM_DEBUG_OUT("acblasSgemm");

  return acblasSgemm(handle, transA, transB,
                     m, n, k,
                     alpha,
                     A, ldA,
                     B, ldB,
                     beta,
                     C, ldC);
}

// dgemm
inline acblasStatus_t
gemm(acblasHandle_t handle,
     acblasOperation_t transA, acblasOperation_t transB,
     int m, int n, int k,
     const double* alpha,
     const double* A, int ldA,
     const double* B, int ldB,
     const double* beta,
     double* C, int ldC)
{
  BLAM_DEBUG_OUT("acblasDgemm");

  return acblasDgemm(handle, transA, transB,
                     m, n, k,
                     alpha,
                     A, ldA,
                     B, ldB,
                     beta,
                     C, ldC);
}

// cgemm
inline acblasStatus_t
gemm(acblasHandle_t handle,
     acblasOperation_t transA, acblasOperation_t transB,
     int m, int n, int k,
     const ComplexFloat* alpha,
     const ComplexFloat* A, int ldA,
     const ComplexFloat* B, int ldB,
     const ComplexFloat* beta,
     ComplexFloat* C, int ldC)
{
  BLAM_DEBUG_OUT("acblasCgemm");

  return acblasCgemm(handle, transA, transB,
                     m, n, k,
                     reinterpret_cast<const acFloatComplex*>(alpha),
                     reinterpret_cast<const acFloatComplex*>(A), ldA,
                     reinterpret_cast<const acFloatComplex*>(B), ldB,
                     reinterpret_cast<const acFloatComplex*>(beta),
                     reinterpret_cast<acFloatComplex*>(C), ldC);
}

// zgemm
inline acblasStatus_t
gemm(acblasHandle_t handle,
     acblasOperation_t transA, acblasOperation_t transB,
     int m, int n, int k,
     const ComplexDouble* alpha,
     const ComplexDouble* A, int ldA,
     const ComplexDouble* B, int ldB,
     const ComplexDouble* beta,
     ComplexDouble* C, int ldC)
{
  BLAM_DEBUG_OUT("acblasZgemm");

  return acblasZgemm(handle, transA, transB,
                     m, n, k,
                     reinterpret_cast<const acDoubleComplex*>(alpha),
                     reinterpret_cast<const acDoubleComplex*>(A), ldA,
                     reinterpret_cast<const acDoubleComplex*>(B), ldB,
                     reinterpret_cast<const acDoubleComplex*>(beta),
                     reinterpret_cast<acDoubleComplex*>(C), ldC);
}

// hgemm
inline acblasStatus_t
gemm_batch(acblasHandle_t handle,
           acblasOperation_t transA, acblasOperation_t transB,
           int m, int n, int k,
           const Half* alpha,
           const Half* A, int ldA, int loA,
           const Half* B, int ldB, int loB,
           const Half* beta,
           Half* C, int ldC, int loC,
           int batch_size)
{
  BLAM_DEBUG_OUT("acblasHgemmStridedBatched");

  return acblasHgemmStridedBatched(handle, transA, transB,
                                   m, n, k,
                                   reinterpret_cast<const __half*>(alpha),
                                   reinterpret_cast<const __half*>(A), ldA, loA,
                                   reinterpret_cast<const __half*>(B), ldB, loB,
                                   reinterpret_cast<const __half*>(beta),
                                   reinterpret_cast<__half*>(C), ldC, loC,
                                   batch_size);
}

// sgemm
inline acblasStatus_t
gemm_batch(acblasHandle_t handle,
           acblasOperation_t transA, acblasOperation_t transB,
           int m, int n, int k,
           const float* alpha,
           const float* A, int ldA, int loA,
           const float* B, int ldB, int loB,
           const float* beta,
           float* C, int ldC, int loC,
           int batch_size)
{
  BLAM_DEBUG_OUT("acblasSgemmStridedBatched");

  return acblasSgemmStridedBatched(handle, transA, transB,
                                   m, n, k,
                                   alpha,
                                   A, ldA, loA,
                                   B, ldB, loB,
                                   beta,
                                   C, ldC, loC,
                                   batch_size);
}

// dgemm
inline acblasStatus_t
gemm_batch(acblasHandle_t handle,
           acblasOperation_t transA, acblasOperation_t transB,
           int m, int n, int k,
           const double* alpha,
           const double* A, int ldA, int loA,
           const double* B, int ldB, int loB,
           const double* beta,
           double* C, int ldC, int loC,
           int batch_size)
{
  BLAM_DEBUG_OUT("acblasDgemmStridedBatched");

  return acblasDgemmStridedBatched(handle, transA, transB,
                                   m, n, k,
                                   alpha,
                                   A, ldA, loA,
                                   B, ldB, loB,
                                   beta,
                                   C, ldC, loC,
                                   batch_size);
}

// cgemm
inline acblasStatus_t
gemm_batch(acblasHandle_t handle,
           acblasOperation_t transA, acblasOperation_t transB,
           int m, int n, int k,
           const ComplexFloat* alpha,
           const ComplexFloat* A, int ldA, int loA,
           const ComplexFloat* B, int ldB, int loB,
           const ComplexFloat* beta,
           ComplexFloat* C, int ldC, int loC,
           int batch_size)
{
  BLAM_DEBUG_OUT("acblasCgemmStridedBatched");

  return acblasCgemmStridedBatched(handle, transA, transB,
                                   m, n, k,
                                   reinterpret_cast<const acFloatComplex*>(alpha),
                                   reinterpret_cast<const acFloatComplex*>(A), ldA, loA,
                                   reinterpret_cast<const acFloatComplex*>(B), ldB, loB,
                                   reinterpret_cast<const acFloatComplex*>(beta),
                                   reinterpret_cast<acFloatComplex*>(C), ldC, loC,
                                   batch_size);
}

// zgemm
inline acblasStatus_t
gemm_batch(acblasHandle_t handle,
           acblasOperation_t transA, acblasOperation_t transB,
           int m, int n, int k,
           const ComplexDouble* alpha,
           const ComplexDouble* A, int ldA, int loA,
           const ComplexDouble* B, int ldB, int loB,
           const ComplexDouble* beta,
           ComplexDouble* C, int ldC, int loC,
           int batch_size)
{
  BLAM_DEBUG_OUT("acblasZgemmStridedBatched");

  return acblasZgemmStridedBatched(handle, transA, transB,
                                   m, n, k,
                                   reinterpret_cast<const acDoubleComplex*>(alpha),
                                   reinterpret_cast<const acDoubleComplex*>(A), ldA, loA,
                                   reinterpret_cast<const acDoubleComplex*>(B), ldB, loB,
                                   reinterpret_cast<const acDoubleComplex*>(beta),
                                   reinterpret_cast<acDoubleComplex*>(C), ldC, loC,
                                   batch_size);
}

// hgemm
inline acblasStatus_t
gemm_batch(acblasHandle_t handle,
           acblasOperation_t transA, acblasOperation_t transB,
           int m, int n, int k,
           const Half* alpha,
           const Half* const A[], int ldA,
           const Half* const B[], int ldB,
           const Half* beta,
           Half* const C[], int ldC,
           int batch_size)
{
  BLAM_DEBUG_OUT("acblasHgemmBatched");

  return acblasHgemmBatched(handle, transA, transB,
                            m, n, k,
                            reinterpret_cast<const __half*>(alpha),
                            reinterpret_cast<const __half**>(const_cast<const Half**>(A)), ldA,
                            // A, ldA,   // acBLAS 9.2
                            reinterpret_cast<const __half**>(const_cast<const Half**>(B)), ldB,
                            // B, ldB,   // acBLAS 9.2
                            reinterpret_cast<const __half*>(beta),
                            reinterpret_cast<__half**>(const_cast<Half**>(C)), ldC,
                            // C, ldC,   // acBLAS 9.2
                            batch_size);
}

// sgemm
inline acblasStatus_t
gemm_batch(acblasHandle_t handle,
           acblasOperation_t transA, acblasOperation_t transB,
           int m, int n, int k,
           const float* alpha,
           const float* const A[], int ldA,
           const float* const B[], int ldB,
           const float* beta,
           float* const C[], int ldC,
           int batch_size)
{
  BLAM_DEBUG_OUT("acblasSgemmBatched");

  return acblasSgemmBatched(handle, transA, transB,
                            m, n, k,
                            alpha,
                            const_cast<const float**>(A), ldA,
                            // A, ldA,   // acBLAS 9.2
                            const_cast<const float**>(B), ldB,
                            // B, ldB,   // acBLAS 9.2
                            beta,
                            const_cast<float**>(C), ldC,
                            // C, ldC,   // acBLAS 9.2
                            batch_size);
}

// dgemm
inline acblasStatus_t
gemm_batch(acblasHandle_t handle,
           acblasOperation_t transA, acblasOperation_t transB,
           int m, int n, int k,
           const double* alpha,
           const double* const A[], int ldA,
           const double* const B[], int ldB,
           const double* beta,
           double* const C[], int ldC,
           int batch_size)
{
  BLAM_DEBUG_OUT("acblasDgemmBatched");

  return acblasDgemmBatched(handle, transA, transB,
                            m, n, k,
                            alpha,
                            const_cast<const double**>(A), ldA,
                            // A, ldA,   // acBLAS 9.2
                            const_cast<const double**>(B), ldB,
                            // B, ldB,   // acBLAS 9.2
                            beta,
                            const_cast<double**>(C), ldC,
                            // C, ldC,   // acBLAS 9.2
                            batch_size);
}

// cgemm
inline acblasStatus_t
gemm_batch(acblasHandle_t handle,
           acblasOperation_t transA, acblasOperation_t transB,
           int m, int n, int k,
           const ComplexFloat* alpha,
           const ComplexFloat* const A[], int ldA,
           const ComplexFloat* const B[], int ldB,
           const ComplexFloat* beta,
           ComplexFloat* const C[], int ldC,
           int batch_size)
{
  BLAM_DEBUG_OUT("acblasCgemmBatched");

  return acblasCgemmBatched(handle, transA, transB,
                            m, n, k,
                            reinterpret_cast<const acFloatComplex*>(alpha),
                            const_cast<const acFloatComplex**>(reinterpret_cast<const acFloatComplex* const *>(A)), ldA,
                            //reinterpret_cast<const acFloatComplex* const *>(A), ldA,  // acBLAS 9.2
                            const_cast<const acFloatComplex**>(reinterpret_cast<const acFloatComplex* const *>(B)), ldB,
                            //reinterpret_cast<const acFloatComplex* const *>(B), ldB,  // acBLAS 9.2
                            reinterpret_cast<const acFloatComplex*>(beta),
                            const_cast<acFloatComplex**>(reinterpret_cast<acFloatComplex* const *>(C)), ldC,
                            //reinterpret_cast<acFloatComplex* const *>(C), ldC,        // acBLAS 9.2
                            batch_size);
}

// zgemm
inline acblasStatus_t
gemm_batch(acblasHandle_t handle,
           acblasOperation_t transA, acblasOperation_t transB,
           int m, int n, int k,
           const ComplexDouble* alpha,
           const ComplexDouble* const A[], int ldA,
           const ComplexDouble* const B[], int ldB,
           const ComplexDouble* beta,
           ComplexDouble* const C[], int ldC,
           int batch_size)
{
  BLAM_DEBUG_OUT("acblasZgemmBatched");

  return acblasZgemmBatched(handle, transA, transB,
                            m, n, k,
                            reinterpret_cast<const acDoubleComplex*>(alpha),
                            const_cast<const acDoubleComplex**>(reinterpret_cast<const acDoubleComplex* const *>(A)), ldA,
                            //reinterpret_cast<const acDoubleComplex* const *>(A), ldA,  // acBLAS 9.2
                            const_cast<const acDoubleComplex**>(reinterpret_cast<const acDoubleComplex* const *>(B)), ldB,
                            //reinterpret_cast<const acDoubleComplex* const *>(B), ldB,  // acBLAS 9.2
                            reinterpret_cast<const acDoubleComplex*>(beta),
                            const_cast<acDoubleComplex**>(reinterpret_cast<acDoubleComplex* const *>(C)), ldC,
                            //reinterpret_cast<acDoubleComplex* const *>(C), ldC,        // acBLAS 9.2
                            batch_size);
}

} // end namespace acblas
} // end namespace blam
