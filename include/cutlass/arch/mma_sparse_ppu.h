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
    \brief Sparse matrix multiply accumulate for PPU
*/

#pragma once

#if defined(__HGGCCC_RTC__)
#include <hggc/std/cassert>
#else
#include <assert.h>
#endif

#include "mma.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

// PPU SDK provides awmma sparse MMA types natively (hggc_mma.h).
// Include the SDK header so awmma::psel0 etc. are available.
#include <hggc_mma.h>
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace arch {

/////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Sparse Matrix Multiply 16832
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F16 = F16 * F16 + F16
template <>
struct SparseMma<
  gemm::GemmShape<16, 8, 32>,
  32,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  half_t,
  layout::RowMajor,
  OpMultiplyAdd,
  SPFormatType::Thread
> {

  using Shape = gemm::GemmShape<16, 8, 32>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 8>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 8>;

  using ElementC = half_t;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<half_t, 4>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  static int const kSparse = 2;

  static int const kMetaSizeInBits = 2;

  static int const kMaxID2 = 2;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c, uint32_t const &E, int const id2) const {


  uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
  uint32_t const *C = reinterpret_cast<uint32_t const *>(&c);
  uint32_t *D = reinterpret_cast<uint32_t *>(&d);

  if (id2 == 0) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sp.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16 {%0,%1}, "
        "{%2,%3,%4,%5}, {%6,%7,%8,%9}, {%10,%11}, %12, 0x0;\n"
        : "=r"(D[0]), "=r"(D[1])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(B[2]), "r"(B[3]), "r"(C[0]), "r"(C[1]), "r"(E));
#endif
  }
  else if (id2 == 1) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sp.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16 {%0,%1}, "
        "{%2,%3,%4,%5}, {%6,%7,%8,%9}, {%10,%11}, %12, 0x1;\n"
        : "=r"(D[0]), "=r"(D[1])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(B[2]), "r"(B[3]), "r"(C[0]), "r"(C[1]), "r"(E));
#endif
  }
  else {
    assert(0);
  }
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F32 = F16 * F16 + F32
template <>
struct SparseMma<
  gemm::GemmShape<16, 8, 32>,
  32,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd,
  SPFormatType::Thread
  > {

  using Shape = gemm::GemmShape<16, 8, 32>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 8>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 8>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  static int const kSparse = 2;

  static int const kMetaSizeInBits = 2;

  static int const kMaxID2 = 2;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c, uint32_t const &E, int const id2) const {


  uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
  float const *C = reinterpret_cast<float const *>(&c);
  float *D = reinterpret_cast<float *>(&d);

  if (id2 == 0) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(B[2]), "r"(B[3]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]),
          "r"(E));
#endif
  }
  else if (id2 == 1) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x1;\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(B[2]), "r"(B[3]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]),
          "r"(E));
#endif
  }
  else {
    assert(0);
  }

  }
};


////////////////////////////////////////////////////////////////////////////////
/// Matrix multiply-add operation: F32 = F16 * F16 + F32
template <>
struct SparseMma<
  gemm::GemmShape<16, 16, 32>,
  32,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd,
  SPFormatType::Thread
  > {
  using Shape = gemm::GemmShape<16, 16, 32>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 16>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 8>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 8>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  static int const kSparse = 2;

  static int const kMetaSizeInBits = 2;

  static int const kMaxID2 = 2;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c, uint32_t const &E, int const id2) const {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    int psel = (id2 == 0 ? awmma::psel0 : awmma::psel1);
    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
    float const *C = reinterpret_cast<float const *>(&c);
    float *D = reinterpret_cast<float *>(&d);
asm volatile(
    "ppu.tc01.mma.sp.sync.aligned.m16n16k32.row.col.f32.f16.f16.f32 {%0, %1, %2, %3, %4, %5, %6, %7}, "
    " {%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, "
    " {%20, %21, %22, %23, %24, %25, %26, %27}, %28, %29, %30;\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3]), "=f"(D[4]), "=f"(D[5]), "=f"(D[6]), "=f"(D[7])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(A[4]), "r"(A[5]), "r"(A[6]), "r"(A[7]),
        "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
        "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]),
        "r"((int)E), "r"(psel), "r"(0));
// #else
//   const uint16_t *frag_a = reinterpret_cast<const uint16_t *>(&a);
//   const uint16_t *frag_b = reinterpret_cast<const uint16_t *>(&b);
//   const uint16_t *frag_e = reinterpret_cast<const uint16_t *>(&E);
//   const ElementC *frag_c = reinterpret_cast<const ElementC *>(&c);
//   ElementC *frag_d = reinterpret_cast<ElementC *>(&d);

//   int lane_idx = threadIdx.x % 32;
//   unsigned mask = 0xffffffff;

//   for (int loop = 0; loop < FragmentC::kElements; loop++) {
//     ElementC acc = 0;

//     // sp mma k.32 meta data 4 bytes, id0(T0,4...28; T1,5..29), id1(T2,6..39; T3,7..31)
//     // lo part(2 bytes) id0(id1):T0(T2)->n0k0.15; T1(T3)->n0k16.31; ...
//     // hi part(2 bytes) id0(id1):T0(T2)->n8k0.15; T1(T3)->n8k16.31; ...
//     int thread_e_lo = (lane_idx % 4) * 4 + ((loop / 4) % 2) * 16 + id2 * 2;
//     int thread_e_hi = thread_e_lo + 1;
//     int reg_e = (loop % 4) / 2;
//     uint16_t meta_lo = __shfl_sync(mask, frag_e[reg_e], thread_e_lo);
//     uint16_t meta_hi = __shfl_sync(mask, frag_e[reg_e], thread_e_hi);
//     for (int k = 0; k < Shape::kK / 4; k++) {
//       uint16_t meta = k < 4 ? meta_lo : meta_hi;
//       int e = (meta >> ((k % 4) * 4)) & 0xf;
//       ElementA cur_a_half[4];
//       ElementA cur_b_half[2];

//       // find 4 A elements.
//       for (int ka = 0; ka < 4; ka++) {
//         int real_k = k * 4 + ka;
//         // part0(M0..8) k.32 T0T1T2T3(8 elements), same as part1(M8..15)
//         int thread_a = lane_idx - (lane_idx % 4) + ((real_k % 8) / 2);
//         // reg(M0..8) R0(2 elements)R1R4R5, reg(M8..15)R2R3R6R7
//         int reg_a = ((real_k / 16) * 4) + ((real_k / 8) * 2) + real_k % 2 + (loop / 4) * 4;
//         // reg order to R1.R3, R4,R7 layout.
//         // int reg_a = ((real_k / 8) * 2) + real_k % 2 + (loop / 4) * 8;
//         uint16_t cur_a = __shfl_sync(mask, frag_a[reg_a], thread_a);
//         memcpy(&cur_a_half[ka], &cur_a, sizeof(uint16_t));
//       }

//       // find 2 B elements and do accumulate.
//       for (int kb = 0; kb < 2; kb++) {
//         int real_k = k * 2 + kb;
//         int thread_b = (lane_idx % 4) * 4 + (loop % 2) * 16 + (real_k % 8) / 2;
//         int reg_b = (((loop % 4) / 2) * 4) + (real_k / 8) * 2 + real_k % 2;
//         uint16_t cur_b = __shfl_sync(mask, frag_b[reg_b], thread_b);
//         memcpy(&cur_b_half[kb], &cur_b, sizeof(uint16_t));
//         int idx = (e >> (kb * 2)) & 0x3;
//         acc += ElementC(cur_a_half[idx]) * ElementC(cur_b_half[kb]);
//       }
//     }
//     frag_d[loop] = acc + frag_c[loop];
//   }
#endif
  }
};

template <>
struct SparseMma<
  gemm::GemmShape<8, 16, 32>,
  32,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd,
  SPFormatType::Thread
  > {
  using Shape = gemm::GemmShape<8, 16, 32>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 8>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 8>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  static int const kSparse = 2;

  static int const kMetaSizeInBits = 2;

  static int const kMaxID2 = 2;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c, uint32_t const &E, int const id2) const {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
  uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
  float const *C = reinterpret_cast<float const *>(&c);
  float *D = reinterpret_cast<float *>(&d);
  int psel = (id2 == 0 ? awmma::psel0 : awmma::psel1);
asm volatile(
    "ppu.tc01.mma.sp.sync.aligned.m8n16k32.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, "
  "{%12, %13, %14, %15}, %16, %17, %18;\n"
    : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
    : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
      "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
      "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]),
      "r"((int)E), "r"(psel), "r"(0));
// #else
//   const uint16_t *frag_a = reinterpret_cast<const uint16_t *>(&a);
//   const uint16_t *frag_b = reinterpret_cast<const uint16_t *>(&b);
//   const uint16_t *frag_e = reinterpret_cast<const uint16_t *>(&E);
//   const ElementC *frag_c = reinterpret_cast<const ElementC *>(&c);
//   ElementC *frag_d = reinterpret_cast<ElementC *>(&d);

//   int lane_idx = threadIdx.x % 32;
//   unsigned mask = 0xffffffff;

//   for (int loop = 0; loop < FragmentC::kElements; loop++) {
//     ElementC acc = 0;

//     int thread_e_lo = (lane_idx % 4) * 4 + (loop % 2) * 16 + id2 * 2;
//     int thread_e_hi = thread_e_lo + 1;
//     int reg_e = loop / 2;
//     uint16_t meta_lo = __shfl_sync(mask, frag_e[reg_e], thread_e_lo);
//     uint16_t meta_hi = __shfl_sync(mask, frag_e[reg_e], thread_e_hi);

//     for (int k = 0; k < Shape::kK / 4; k++) {
//       uint16_t meta = k < 4 ? meta_lo : meta_hi;
//       int e = (meta >> ((k % 4) * 4)) & 0xf;
//       ElementA cur_a_half[4];
//       ElementA cur_b_half[2];

//       // find 4 A elements.
//       for (int ka = 0; ka < 4; ka++) {
//         int real_k = k * 4 + ka;
//         int thread_a = lane_idx - (lane_idx % 4) + ((real_k % 8) / 2);
//         int reg_a = ((real_k / 8) * 2) + (real_k % 2);
//         uint16_t cur_a = __shfl_sync(mask, frag_a[reg_a], thread_a);
//         memcpy(&cur_a_half[ka], &cur_a, sizeof(uint16_t));
//       }

//       // find 2 B elements and do accumulate.
//       for (int kb = 0; kb < 2; kb++) {
//         int real_k = k * 2 + kb;

//         int thread_b = (lane_idx % 4) * 4 + (loop % 2) * 16 + (real_k % 8) / 2;
//         int reg_b = (((loop % 4) / 2) * 4) + (real_k / 8) * 2 + real_k % 2;
//         uint16_t cur_b = __shfl_sync(mask, frag_b[reg_b], thread_b);
//         memcpy(&cur_b_half[kb], &cur_b, sizeof(uint16_t));
//         int idx = (e >> (kb * 2)) & 0x3;
//         // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.x == 0 && loop == 0)
//         //   printf("k:%d, kb:%d, real_k:%d, idx:(%d)%d, elementA:%d, elementB:%d\n", k, kb, real_k, e, idx, (int)cur_a_half[idx], (int)cur_b_half[kb]);
//         acc += ElementC(cur_a_half[idx]) * ElementC(cur_b_half[kb]);
//       }
//     }
//     frag_d[loop] = acc + frag_c[loop];
//   }
#endif
  }
};

template <>
struct SparseMma<
  gemm::GemmShape<16, 16, 16>,
  32,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd,
  SPFormatType::Thread
  > {
  using Shape = gemm::GemmShape<16, 16, 16>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 8>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 4>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 8>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  static int const kSparse = 2;

  static int const kMetaSizeInBits = 2;

  static int const kMaxID2 = 4;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c, uint32_t const &E, int const id2) const {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
  int psel = awmma::mma_sparsity_psel(id2 & 0x03);
  uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
  float const *C = reinterpret_cast<float const *>(&c);
  float *D = reinterpret_cast<float *>(&d);
asm volatile(
    "ppu.tc01.mma.sp.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3, %4, %5, %6, %7},"
    " {%8, %9, %10, %11}, {%12, %13}, {%14, %15, %16, %17, %18, %19, %20, %21}, %22, %23, %24;\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3]), "=f"(D[4]), "=f"(D[5]), "=f"(D[6]), "=f"(D[7])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
        "r"(B[0]), "r"(B[1]),
        "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]),
        "r"((int)E), "r"(psel), "r"(0));

#else
  assert(0);
#endif
  }
};

template <>
struct SparseMma<
  gemm::GemmShape<8, 16, 16>,
  32,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd,
  SPFormatType::Thread
  > {
  using Shape = gemm::GemmShape<8, 16, 16>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 4>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 4>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  static int const kSparse = 2;

  static int const kMetaSizeInBits = 2;

  static int const kMaxID2 = 4;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c, uint32_t const &E, int const id2) const {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
  int psel = awmma::mma_sparsity_psel(id2 & 0x03);
  uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
  float const *C = reinterpret_cast<float const *>(&c);
  float *D = reinterpret_cast<float *>(&d);
asm volatile(
    "ppu.tc01.mma.sp.sync.aligned.m8n16k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6, %7},"
    " {%8, %9, %10, %11}, %12, %13, %14;\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      : "r"(A[0]), "r"(A[1]),
        "r"(B[0]), "r"(B[1]),
        "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]),
        "r"((int)E), "r"(psel), "r"(0));
#else
  assert(0);
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Sparse Matrix Multiply 16832 - Float BF16, FP32 accumulation
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F32 = bf16 * bf16 + F32
template <>
struct SparseMma<gemm::GemmShape<16, 8, 32>, 32, bfloat16_t, layout::RowMajor,
           bfloat16_t, layout::ColumnMajor, float, layout::RowMajor,
           OpMultiplyAdd, SPFormatType::Thread> {
  using Shape = gemm::GemmShape<16, 8, 32>;

  using ElementA = bfloat16_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<bfloat16_t, 8>;

  using ElementB = bfloat16_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<bfloat16_t, 8>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  static int const kSparse = 2;

  static int const kMetaSizeInBits = 2;

  static int const kMaxID2 = 2;

  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c, uint32_t const &E, int const id2) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
    float const *C = reinterpret_cast<float const *>(&c);
    float *D = reinterpret_cast<float *>(&d);

    if (id2 == 0) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "r"(E));
#endif
    } else if (id2 == 1) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x1;\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "r"(E));
#endif
    } else {
    assert(0);
    }

  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Sparse Matrix Multiply 16816 - Float TF32
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F32 = tf32 * tf32 + F32
template <>
struct SparseMma<gemm::GemmShape<16, 8, 16>, 32, tfloat32_t, layout::RowMajor,
           tfloat32_t, layout::ColumnMajor, float, layout::RowMajor,
           OpMultiplyAdd, SPFormatType::Thread> {
  using Shape = gemm::GemmShape<16, 8, 16>;

  using ElementA = tfloat32_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<tfloat32_t, 4>;

  using ElementB = tfloat32_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<tfloat32_t, 4>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  static int const kSparse = 2;

  static int const kMetaSizeInBits = 4;

  static int const kMaxID2 = 2;

  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c, uint32_t const &E, int const id2) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
    float const *C = reinterpret_cast<float const *>(&c);
    float *D = reinterpret_cast<float *>(&d);

    if (id2 == 0) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sp.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "r"(E));
#endif
    } else if (id2 == 1) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sp.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x1;\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "r"(E));
#endif
    } else {
    assert(0);
    }

  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Sparse Matrix Multiply 16864 - S8 input, S32 accumulation
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = S8 * S8 + S32
template <>
struct SparseMma<
  gemm::GemmShape<16,8,64>,
  32,
  int8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd,
  SPFormatType::Thread> {

  using Shape = gemm::GemmShape<16,8,64>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 16>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  static int const kSparse = 2;

  static int const kMetaSizeInBits = 2;

  static int const kMaxID2 = 1;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c,
    uint32_t const &E,
    int const id2
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    if (id2 == 0) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sp.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(E));
#endif
    }
    else
    assert(0);

  }
};

/// Matrix multiply-add operation: S32 = S8 * U8 + S32
template <>
struct SparseMma<
  gemm::GemmShape<16,8,64>,
  32,
  int8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd,
  SPFormatType::Thread> {

  using Shape = gemm::GemmShape<16,8,64>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 16>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  static int const kSparse = 2;

  static int const kMetaSizeInBits = 2;

  static int const kMaxID2 = 1;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c,
    uint32_t const &E,
    int const id2
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    if (id2 == 0) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sp.sync.aligned.m16n8k64.row.col.s32.s8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(E));
#endif
    }
    else
    assert(0);

  }
};

/// Matrix multiply-add operation: S32 = U8 * S8 + S32
template <>
struct SparseMma<
  gemm::GemmShape<16,8,64>,
  32,
  uint8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd,
  SPFormatType::Thread> {

  using Shape = gemm::GemmShape<16,8,64>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 16>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  static int const kSparse = 2;

  static int const kMetaSizeInBits = 2;

  static int const kMaxID2 = 1;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c,
    uint32_t const &E,
    int const id2
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    if (id2 == 0) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sp.sync.aligned.m16n8k64.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(E));
#endif
    }
    else
    assert(0);

  }
};

/// Matrix multiply-add operation: S32 = U8 * U8 + S32
template <>
struct SparseMma<
  gemm::GemmShape<16,8,64>,
  32,
  uint8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd,
  SPFormatType::Thread> {

  using Shape = gemm::GemmShape<16,8,64>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 16>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  static int const kSparse = 2;

  static int const kMetaSizeInBits = 2;

  static int const kMaxID2 = 1;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c,
    uint32_t const &E,
    int const id2
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    if (id2 == 0) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sp.sync.aligned.m16n8k64.row.col.s32.u8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(E));
#endif
    }
    else
    assert(0);

  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Sparse Matrix Multiply 16864 - S8 input, S32 accumulation - SATURATE
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = S8 * S8 + S32
template <>
struct SparseMma<
  gemm::GemmShape<16,8,64>,
  32,
  int8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate,
  SPFormatType::Thread> {

  using Shape = gemm::GemmShape<16,8,64>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 16>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  static int const kSparse = 2;

  static int const kMetaSizeInBits = 2;

  static int const kMaxID2 = 1;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c,
    uint32_t const &E,
    int const id2
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    if (id2 == 0) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sp.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32.satfinite {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(E));
#endif
    }
    else
    assert(0);

  }
};

/// Matrix multiply-add operation: S32 = S8 * U8 + S32
template <>
struct SparseMma<
  gemm::GemmShape<16,8,64>,
  32,
  int8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate,
  SPFormatType::Thread> {

  using Shape = gemm::GemmShape<16,8,64>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 16>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  static int const kSparse = 2;

  static int const kMetaSizeInBits = 2;

  static int const kMaxID2 = 1;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c,
    uint32_t const &E,
    int const id2
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    if (id2 == 0) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sp.sync.aligned.m16n8k64.row.col.s32.s8.u8.s32.satfinite {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(E));
#endif
    }
    else
    assert(0);

  }
};

/// Matrix multiply-add operation: S32 = U8 * S8 + S32
template <>
struct SparseMma<
  gemm::GemmShape<16,8,64>,
  32,
  uint8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate,
  SPFormatType::Thread> {

  using Shape = gemm::GemmShape<16,8,64>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 16>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  static int const kSparse = 2;

  static int const kMetaSizeInBits = 2;

  static int const kMaxID2 = 1;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c,
    uint32_t const &E,
    int const id2
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    if (id2 == 0) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sp.sync.aligned.m16n8k64.row.col.s32.u8.s8.s32.satfinite {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(E));
#endif
    }
    else
    assert(0);

  }
};

/// Matrix multiply-add operation: S32 = U8 * U8 + S32
template <>
struct SparseMma<
  gemm::GemmShape<16,8,64>,
  32,
  uint8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate,
  SPFormatType::Thread> {

  using Shape = gemm::GemmShape<16,8,64>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 16>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  static int const kSparse = 2;

  static int const kMetaSizeInBits = 2;

  static int const kMaxID2 = 1;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c,
    uint32_t const &E,
    int const id2
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    if (id2 == 0) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sp.sync.aligned.m16n8k64.row.col.s32.u8.u8.s32.satfinite {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(E));
#endif
    }
    else
    assert(0);

  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Sparse Matrix Multiply 168128 - S4 input, S32 accumulation
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = S4 * S4 + S32
template <>
struct SparseMma<
  gemm::GemmShape<16,8,128>,
  32,
  cutlass::int4b_t,
  layout::RowMajor,
  cutlass::int4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd,
  SPFormatType::Thread> {

  using Shape = gemm::GemmShape<16,8,128>;

  using ElementA = cutlass::int4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<cutlass::int4b_t, 32>;

  using ElementB = cutlass::int4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<cutlass::int4b_t, 32>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  static int const kSparse = 2;

  static int const kMetaSizeInBits = 2;

  static int const kMaxID2 = 1;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c,
    uint32_t const &E,
    int const id2
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    if (id2 == 0) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sp.sync.aligned.m16n8k128.row.col.s32.s4.s4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(E));
#endif
    }
    else
    assert(0);

  }
};

/// Matrix multiply-add operation: S32 = S4 * U4 + S32
template <>
struct SparseMma<
  gemm::GemmShape<16,8,128>,
  32,
  cutlass::int4b_t,
  layout::RowMajor,
  cutlass::uint4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd,
  SPFormatType::Thread> {

  using Shape = gemm::GemmShape<16,8,128>;

  using ElementA = cutlass::int4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<cutlass::int4b_t, 32>;

  using ElementB = cutlass::uint4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<cutlass::uint4b_t, 32>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  static int const kSparse = 2;

  static int const kMetaSizeInBits = 2;

  static int const kMaxID2 = 1;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c,
    uint32_t const &E,
    int const id2
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    if (id2 == 0) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sp.sync.aligned.m16n8k128.row.col.s32.s4.u4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(E));
#endif
    }
    else
    assert(0);

  }
};

/// Matrix multiply-add operation: S32 = U4 * S4 + S32
template <>
struct SparseMma<
  gemm::GemmShape<16,8,128>,
  32,
  cutlass::uint4b_t,
  layout::RowMajor,
  cutlass::int4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd,
  SPFormatType::Thread> {

  using Shape = gemm::GemmShape<16,8,128>;

  using ElementA = cutlass::uint4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<cutlass::uint4b_t, 32>;

  using ElementB = cutlass::int4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<cutlass::int4b_t, 32>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  static int const kSparse = 2;

  static int const kMetaSizeInBits = 2;

  static int const kMaxID2 = 1;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c,
    uint32_t const &E,
    int const id2
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    if (id2 == 0) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sp.sync.aligned.m16n8k128.row.col.s32.u4.s4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(E));
#endif
    }
    else
    assert(0);

  }
};

/// Matrix multiply-add operation: S32 = U4 * U4 + S32
template <>
struct SparseMma<
  gemm::GemmShape<16,8,128>,
  32,
  cutlass::uint4b_t,
  layout::RowMajor,
  cutlass::uint4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd,
  SPFormatType::Thread> {

  using Shape = gemm::GemmShape<16,8,128>;

  using ElementA = cutlass::uint4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<cutlass::uint4b_t, 32>;

  using ElementB = cutlass::uint4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<cutlass::uint4b_t, 32>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  static int const kSparse = 2;

  static int const kMetaSizeInBits = 2;

  static int const kMaxID2 = 1;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c,
    uint32_t const &E,
    int const id2
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    if (id2 == 0) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sp.sync.aligned.m16n8k128.row.col.s32.u4.u4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(E));
#endif
    }
    else
    assert(0);

  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Sparse Matrix Multiply 168128 - S4 input, S32 accumulation - SATURATE
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = S4 * S4 + S32
template <>
struct SparseMma<
  gemm::GemmShape<16,8,128>,
  32,
  cutlass::int4b_t,
  layout::RowMajor,
  cutlass::int4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate,
  SPFormatType::Thread> {

  using Shape = gemm::GemmShape<16,8,128>;

  using ElementA = cutlass::int4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<cutlass::int4b_t, 32>;

  using ElementB = cutlass::int4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<cutlass::int4b_t, 32>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  static int const kSparse = 2;

  static int const kMetaSizeInBits = 2;

  static int const kMaxID2 = 1;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c,
    uint32_t const &E,
    int const id2
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    if (id2 == 0) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sp.sync.aligned.m16n8k128.row.col.s32.s4.s4.s32.satfinite {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(E));
#endif
    }
    else
    assert(0);

  }
};

/// Matrix multiply-add operation: S32 = S4 * U4 + S32
template <>
struct SparseMma<
  gemm::GemmShape<16,8,128>,
  32,
  cutlass::int4b_t,
  layout::RowMajor,
  cutlass::uint4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate,
  SPFormatType::Thread> {

  using Shape = gemm::GemmShape<16,8,128>;

  using ElementA = cutlass::int4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<cutlass::int4b_t, 32>;

  using ElementB = cutlass::uint4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<cutlass::uint4b_t, 32>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  static int const kSparse = 2;

  static int const kMetaSizeInBits = 2;

  static int const kMaxID2 = 1;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c,
    uint32_t const &E,
    int const id2
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    if (id2 == 0) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sp.sync.aligned.m16n8k128.row.col.s32.s4.u4.s32.satfinite {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(E));
#endif
    }
    else
    assert(0);

  }
};

/// Matrix multiply-add operation: S32 = U4 * S4 + S32
template <>
struct SparseMma<
  gemm::GemmShape<16,8,128>,
  32,
  cutlass::uint4b_t,
  layout::RowMajor,
  cutlass::int4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate,
  SPFormatType::Thread> {

  using Shape = gemm::GemmShape<16,8,128>;

  using ElementA = cutlass::uint4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<cutlass::uint4b_t, 32>;

  using ElementB = cutlass::int4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<cutlass::int4b_t, 32>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  static int const kSparse = 2;

  static int const kMetaSizeInBits = 2;

  static int const kMaxID2 = 1;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c,
    uint32_t const &E,
    int const id2
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    if (id2 == 0) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sp.sync.aligned.m16n8k128.row.col.s32.u4.s4.s32.satfinite {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(E));
#endif
    }
    else
    assert(0);

  }
};

/// Matrix multiply-add operation: S32 = U4 * U4 + S32
template <>
struct SparseMma<
  gemm::GemmShape<16,8,128>,
  32,
  cutlass::uint4b_t,
  layout::RowMajor,
  cutlass::uint4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate,
  SPFormatType::Thread> {

  using Shape = gemm::GemmShape<16,8,128>;

  using ElementA = cutlass::uint4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<cutlass::uint4b_t, 32>;

  using ElementB = cutlass::uint4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<cutlass::uint4b_t, 32>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  static int const kSparse = 2;

  static int const kMetaSizeInBits = 2;

  static int const kMaxID2 = 1;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c,
    uint32_t const &E,
    int const id2
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    if (id2 == 0) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sp.sync.aligned.m16n8k128.row.col.s32.u4.u4.s32.satfinite {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(E));
#endif
    }
    else
    assert(0);

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace arch
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
