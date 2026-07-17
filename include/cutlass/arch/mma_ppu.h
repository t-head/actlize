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
    \brief Matrix multiply
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

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace arch {

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 1688 - Float BF16, FP32 accumulation
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation - F32 = bf16 * bf16 + F32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 8>,
  32,
  bfloat16_t,
  layout::RowMajor,
  bfloat16_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 8, 8>;

  using ElementA = bfloat16_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<bfloat16_t, 4>;

  using ElementB = bfloat16_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<bfloat16_t, 2>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {


  uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
  float const *C = reinterpret_cast<float const *>(&c);
  float *D = reinterpret_cast<float *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "    
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      :
        "r"(A[0]), "r"(A[1]),
        "r"(B[0]),
        "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])
  );
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "    
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      :
        "r"(A[0]), "r"(A[1]),
        "r"(B[0]),
        "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])
  );
#endif

  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 1684 - Float TF32
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F32 = tf32 * tf32 + F32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 4>,
  32,
  tfloat32_t,
  layout::RowMajor,
  tfloat32_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 8, 4>;

  using ElementA = tfloat32_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<tfloat32_t, 2>;

  using ElementB = tfloat32_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<tfloat32_t, 1>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


  uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
  float const *C = reinterpret_cast<float const *>(&c);
  float *D = reinterpret_cast<float *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"    
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      :
        "r"(A[0]), "r"(A[1]),
        "r"(B[0]),
        "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])
  );
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"    
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      :
        "r"(A[0]), "r"(A[1]),
        "r"(B[0]),
        "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])
  );
#endif

  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 1688 - Float TF32
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F32 = tf32 * tf32 + F32
template <>
struct Mma<gemm::GemmShape<16, 8, 8>, 32, tfloat32_t, layout::RowMajor,
           tfloat32_t, layout::ColumnMajor, float, layout::RowMajor,
           OpMultiplyAdd> {
  using Shape = gemm::GemmShape<16, 8, 8>;

  using ElementA = tfloat32_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<tfloat32_t, 4>;

  using ElementB = tfloat32_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<tfloat32_t, 2>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
    float const *C = reinterpret_cast<float const *>(&c);
    float *D = reinterpret_cast<float *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "    
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "    
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
#endif

  }
};

template <>
struct Mma<gemm::GemmShape<16, 16, 8>, 32, tfloat32_t, layout::RowMajor,
           tfloat32_t, layout::ColumnMajor, float, layout::RowMajor,
           OpMultiplyAdd> {
  using Shape = gemm::GemmShape<16, 16, 8>;

  using ElementA = tfloat32_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<tfloat32_t, 4>;

  using ElementB = tfloat32_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<tfloat32_t, 4>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 8>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {

    #if !SAIL_SIMULATE_CUTLASS_MMA
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      float const *C = reinterpret_cast<float const *>(&c);
      float *D = reinterpret_cast<float *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.mma.sync.aligned.m16n16k8.row.col.f32.tf32.tf32.f32 "    
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, {%16,%17,%18,%19,%20,%21,%22,%23};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3]), "=f"(D[4]), "=f"(D[5]), "=f"(D[6]), "=f"(D[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n16k8.row.col.f32.tf32.tf32.f32 "    
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, {%16,%17,%18,%19,%20,%21,%22,%23};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3]), "=f"(D[4]), "=f"(D[5]), "=f"(D[6]), "=f"(D[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]));
#endif
    #else
      const float *frag_a = reinterpret_cast<const float *>(&a);
      const float *frag_b = reinterpret_cast<const float *>(&b);
      const ElementC *frag_c = reinterpret_cast<const ElementC *>(&c);
      ElementC *frag_d = reinterpret_cast<ElementC *>(&d);
      int lane_idx = threadIdx.x % 32;
      unsigned mask = 0xffffffff;
      // two elem in each 8x8
      for (int loop = 0; loop < 8; loop++) {
        ElementC acc = 0;
        for (int k = 0; k < Shape::kK; k++) {
          // threadIdx of elem a, will use two elem each time, right 8 k's thread is identical to left 8 k
          int thread_a = lane_idx - (lane_idx % 4) + (k % 4);
          // (loop / 4) * 4 corresponds to top or down part, ((k / 8) * 2) corresponds to left or right part
          // k % 2 corresponds to which elem in each part, inner loop will use two elem in each thread
          int reg_a = ((loop / 4) * 2) + (k / 4);
          float cur_a = __shfl_sync(mask, frag_a[reg_a], thread_a);
          ElementA cur_a_tf32;
          memcpy(&cur_a_tf32, &cur_a, sizeof(float));
          // first two part is which col in matrix b, last part is which row
          int thread_b = (lane_idx % 4) * 4 + (loop % 2) * 16 + (k % 4);
          // first is hori part, second is vert part
          int reg_b = ((loop % 4) / 2) * 2 + (k / 4);
          float cur_b = __shfl_sync(mask, frag_b[reg_b], thread_b);
          ElementB cur_b_tf32;
          memcpy(&cur_b_tf32, &cur_b, sizeof(float));
          acc += ElementC(cur_a_tf32 * cur_b_tf32);
        }
        frag_d[loop] = acc + frag_c[loop];
      }
    #endif
  }
};

template <>
struct Mma<gemm::GemmShape<16, 16, 4>, 32, tfloat32_t, layout::RowMajor,
           tfloat32_t, layout::ColumnMajor, float, layout::RowMajor,
           OpMultiplyAdd> {
  using Shape = gemm::GemmShape<16, 16, 4>;

  using ElementA = tfloat32_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<tfloat32_t, 2>;

  using ElementB = tfloat32_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<tfloat32_t, 2>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 8>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      float const *C = reinterpret_cast<float const *>(&c);
      float *D = reinterpret_cast<float *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.mma.sync.aligned.m16n16k4.row.col.f32.tf32.tf32.f32 "    
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3]), "=f"(D[4]), "=f"(D[5]), "=f"(D[6]), "=f"(D[7])
        : "r"(A[0]), "r"(A[1]),
          "r"(B[0]), "r"(B[1]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n16k4.row.col.f32.tf32.tf32.f32 "    
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3]), "=f"(D[4]), "=f"(D[5]), "=f"(D[6]), "=f"(D[7])
        : "r"(A[0]), "r"(A[1]),
          "r"(B[0]), "r"(B[1]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]));
#endif
  }
};

template <>
struct Mma<gemm::GemmShape<8, 16, 8>, 32, tfloat32_t, layout::RowMajor,
           tfloat32_t, layout::ColumnMajor, float, layout::RowMajor,
           OpMultiplyAdd> {
  using Shape = gemm::GemmShape<8, 16, 8>;

  using ElementA = tfloat32_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<tfloat32_t, 2>;

  using ElementB = tfloat32_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<tfloat32_t, 4>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      float const *C = reinterpret_cast<float const *>(&c);
      float *D = reinterpret_cast<float *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.mma.sync.aligned.m8n16k8.row.col.f32.tf32.tf32.f32 "    
        "{%0,%1,%2,%3}, {%4,%5}, {%6,%7,%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]),
          "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
#endif
  }
};

template <>
struct Mma<gemm::GemmShape<8, 16, 4>, 32, tfloat32_t, layout::RowMajor,
           tfloat32_t, layout::ColumnMajor, float, layout::RowMajor,
           OpMultiplyAdd> {
  using Shape = gemm::GemmShape<8, 16, 4>;

  using ElementA = tfloat32_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<tfloat32_t, 1>;

  using ElementB = tfloat32_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<tfloat32_t, 2>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      float const *C = reinterpret_cast<float const *>(&c);
      float *D = reinterpret_cast<float *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.mma.sync.aligned.m8n16k4.row.col.f32.tf32.tf32.f32 "    
        "{%0,%1,%2,%3}, {%4}, {%5,%6}, {%7,%8,%9,%10};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]),
          "r"(B[0]), "r"(B[1]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
#endif
  }
};

// MMA 16168, F32 = F32 * F32 + F32
template <>
struct Mma<gemm::GemmShape<16, 16, 8>, 32, float, layout::RowMajor,
           float, layout::ColumnMajor, float, layout::RowMajor,
           OpMultiplyAdd> {
  using Shape = gemm::GemmShape<16, 16, 8>;
  using ElementA = float;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<float, 4>;
  using ElementB = float;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<float, 4>;
  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 8>;
  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;
  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {
      float const *A = reinterpret_cast<float const *>(&a);
      float const *B = reinterpret_cast<float const *>(&b);
      float const *C = reinterpret_cast<float const *>(&c);
      float *D = reinterpret_cast<float *>(&d);
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.mma.sync.aligned.m16n16k8.row.col.f32.f32.f32.f32 "    
          "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, {%16,%17,%18,%19,%20,%21,%22,%23};\n"
          : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3]), "=f"(D[4]), "=f"(D[5]), "=f"(D[6]), "=f"(D[7])
          : "f"(A[0]), "f"(A[1]), "f"(A[2]), "f"(A[3]),
            "f"(B[0]), "f"(B[1]), "f"(B[2]), "f"(B[3]),
            "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n16k8.row.col.f32.f32.f32.f32 "    
          "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, {%16,%17,%18,%19,%20,%21,%22,%23};\n"
          : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3]), "=f"(D[4]), "=f"(D[5]), "=f"(D[6]), "=f"(D[7])
          : "f"(A[0]), "f"(A[1]), "f"(A[2]), "f"(A[3]),
            "f"(B[0]), "f"(B[1]), "f"(B[2]), "f"(B[3]),
            "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]));
#endif
  }
};


// MMA 16168, F32 = F16 * F16 + F32
template <>
struct Mma<
  gemm::GemmShape<16, 16, 8>,
  32,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 16, 8>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 4>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 4>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 8>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      float const *C = reinterpret_cast<float const *>(&c);
      float *D = reinterpret_cast<float *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.mma.sync.aligned.m16n16k8.row.col.f32.f16.f16.f32  {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "    
        "{%12,%13,%14,%15,%16,%17,%18,%19};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3]), "=f"(D[4]), "=f"(D[5]), "=f"(D[6]), "=f"(D[7])
        : "r"(A[0]), "r"(A[1]),
          "r"(B[0]), "r"(B[1]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n16k8.row.col.f32.f16.f16.f32  {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "    
        "{%12,%13,%14,%15,%16,%17,%18,%19};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3]), "=f"(D[4]), "=f"(D[5]), "=f"(D[6]), "=f"(D[7])
        : "r"(A[0]), "r"(A[1]),
          "r"(B[0]), "r"(B[1]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]));
#endif
  }
};

// MMA 81616, F32 = F16 * F16 + F32
template <>
struct Mma<
  gemm::GemmShape<8, 16, 16>,
  32,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8, 16, 16>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 4>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 8>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      float const *C = reinterpret_cast<float const *>(&c);
      float *D = reinterpret_cast<float *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.mma.sync.aligned.m8n16k16.row.col.f32.f16.f16.f32  {%0,%1,%2,%3}, {%4,%5}, {%6,%7,%8,%9}, "    
        "{%10,%11,%12,%13};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]),
          "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
#endif
  }
};

// MMA 8168, F32 = F16 * F16 + F32
template <>
struct Mma<
  gemm::GemmShape<8, 16, 8>,
  32,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8, 16, 8>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 2>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 4>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      float const *C = reinterpret_cast<float const *>(&c);
      float *D = reinterpret_cast<float *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.mma.sync.aligned.m8n16k8.row.col.f32.f16.f16.f32  {%0,%1,%2,%3}, {%4}, {%5,%6}, "    
        "{%7,%8,%9,%10};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]),
          "r"(B[0]), "r"(B[1]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
#endif
  }
};

template <>
struct Mma<
  gemm::GemmShape<16, 16, 16>,
  32,
  bfloat16_t,
  layout::RowMajor,
  bfloat16_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 16, 16>;

  using ElementA = bfloat16_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<bfloat16_t, 8>;

  using ElementB = bfloat16_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<bfloat16_t, 8>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 8>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      float const *C = reinterpret_cast<float const *>(&c);
      float *D = reinterpret_cast<float *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.mma.sync.aligned.m16n16k16.row.col.f32.bf16.bf16.f32  {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "    
        "{%16,%17,%18,%19,%20,%21,%22,%23};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3]), "=f"(D[4]), "=f"(D[5]), "=f"(D[6]), "=f"(D[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n16k16.row.col.f32.bf16.bf16.f32  {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "    
        "{%16,%17,%18,%19,%20,%21,%22,%23};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3]), "=f"(D[4]), "=f"(D[5]), "=f"(D[6]), "=f"(D[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]));
#endif
  }
};


////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 16816
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F16 = F16 * F16 + F16
template <>
struct Mma<
  gemm::GemmShape<16, 8, 16>,
  32,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  half_t,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 8, 16>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 8>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 4>;

  using ElementC = half_t;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<half_t, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {


  uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
  uint32_t const *C = reinterpret_cast<uint32_t const *>(&c);
  uint32_t *D = reinterpret_cast<uint32_t *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"    
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
        "r"(B[0]), "r"(B[1]),
        "r"(C[0]), "r"(C[1])
  );
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"    
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
        "r"(B[0]), "r"(B[1]),
        "r"(C[0]), "r"(C[1])
  );
#endif

  }
};

////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F32 = bf16 * bf16 + F32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 16>,
  32,
  bfloat16_t,
  layout::RowMajor,
  bfloat16_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 8, 16>;

  using ElementA = bfloat16_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<bfloat16_t, 8>;

  using ElementB = bfloat16_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<bfloat16_t, 4>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
    float const *C = reinterpret_cast<float const *>(&c);
    float *D = reinterpret_cast<float *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "    
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "    
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
#endif

  }
};

////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F32 = F16 * F16 + F32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 16>,
  32,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 8, 16>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 8>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 4>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
    float const *C = reinterpret_cast<float const *>(&c);
    float *D = reinterpret_cast<float *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32  {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "    
        "{%10,%11,%12,%13};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32  {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "    
        "{%10,%11,%12,%13};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
#endif

  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 884 - F64
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F64 = F64 * F64 + F64
template <>
struct Mma<
  gemm::GemmShape<8,8,4>,
  32,
  double,
  layout::RowMajor,
  double,
  layout::ColumnMajor,
  double,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8,8,4>;

  using ElementA = double;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<double, 1>;

  using ElementB = double;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<double, 1>;

  using ElementC = double;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<double, 2>;

  using Operator = OpMultiplyAdd;

  using ArchTag = arch::PPU0010;

  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {


  double const & A = reinterpret_cast<double const &>(a);
  double const & B = reinterpret_cast<double const &>(b);

  double const *C = reinterpret_cast<double const *>(&c);
  double *D = reinterpret_cast<double *>(&d);

  assert(0);

  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 16816 - S8 input, S32 accumulation
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = S8 * S8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,16>,
  32,
  int8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16,8,16>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 8>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;

  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const &B = reinterpret_cast<uint32_t const &>(b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, "    
        "{%7,%8,%9,%10};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(B), "r"(C[0]), "r"(C[1]), "r"(C[2]),
          "r"(C[3]));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, "    
        "{%7,%8,%9,%10};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(B), "r"(C[0]), "r"(C[1]), "r"(C[2]),
          "r"(C[3]));
#endif

  }
};

/// Matrix multiply-add operation: S32 = U8 * S8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,16>,
  32,
  uint8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16,8,16>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 8>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const &B = reinterpret_cast<uint32_t const &>(b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k16.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, "    
        "{%7,%8,%9,%10};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(B), "r"(C[0]), "r"(C[1]), "r"(C[2]),
          "r"(C[3]));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k16.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, "    
        "{%7,%8,%9,%10};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(B), "r"(C[0]), "r"(C[1]), "r"(C[2]),
          "r"(C[3]));
#endif

  }
};

/// Matrix multiply-add operation: S32 = S8 * U8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,16>,
  32,
  int8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16,8,16>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 8>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const &B = reinterpret_cast<uint32_t const &>(b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k16.row.col.s32.s8.u8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, "    
        "{%7,%8,%9,%10};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(B), "r"(C[0]), "r"(C[1]), "r"(C[2]),
          "r"(C[3]));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k16.row.col.s32.s8.u8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, "    
        "{%7,%8,%9,%10};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(B), "r"(C[0]), "r"(C[1]), "r"(C[2]),
          "r"(C[3]));
#endif

  }
};

/// Matrix multiply-add operation: S32 = U8 * U8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,16>,
  32,
  uint8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16,8,16>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 8>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const &B = reinterpret_cast<uint32_t const &>(b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, "    
        "{%7,%8,%9,%10};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(B), "r"(C[0]), "r"(C[1]), "r"(C[2]),
          "r"(C[3]));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, "    
        "{%7,%8,%9,%10};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(B), "r"(C[0]), "r"(C[1]), "r"(C[2]),
          "r"(C[3]));
#endif


  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 16816 - S8 input, S32 accumulation - SATURATE
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = S8 * S8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,16>,
  32,
  int8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16,8,16>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 8>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const &B = reinterpret_cast<uint32_t const &>(b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32.satfinite {%0,%1,%2,%3}, {%4,%5}, "    
        "{%6}, {%7,%8,%9,%10};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(B), "r"(C[0]), "r"(C[1]), "r"(C[2]),
          "r"(C[3]));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32.satfinite {%0,%1,%2,%3}, {%4,%5}, "    
        "{%6}, {%7,%8,%9,%10};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(B), "r"(C[0]), "r"(C[1]), "r"(C[2]),
          "r"(C[3]));
#endif

  }
};

/// Matrix multiply-add operation: S32 = U8 * S8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,16>,
  32,
  uint8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16,8,16>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 8>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const &B = reinterpret_cast<uint32_t const &>(b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k16.row.col.s32.u8.s8.s32.satfinite {%0,%1,%2,%3}, {%4,%5}, "    
        "{%6}, {%7,%8,%9,%10};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(B), "r"(C[0]), "r"(C[1]), "r"(C[2]),
          "r"(C[3]));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k16.row.col.s32.u8.s8.s32.satfinite {%0,%1,%2,%3}, {%4,%5}, "    
        "{%6}, {%7,%8,%9,%10};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(B), "r"(C[0]), "r"(C[1]), "r"(C[2]),
          "r"(C[3]));
#endif

  }
};

/// Matrix multiply-add operation: S32 = S8 * U8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,16>,
  32,
  int8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16,8,16>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 8>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const &B = reinterpret_cast<uint32_t const &>(b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k16.row.col.s32.s8.u8.s32.satfinite {%0,%1,%2,%3}, {%4,%5}, "    
        "{%6}, {%7,%8,%9,%10};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(B), "r"(C[0]), "r"(C[1]), "r"(C[2]),
          "r"(C[3]));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k16.row.col.s32.s8.u8.s32.satfinite {%0,%1,%2,%3}, {%4,%5}, "    
        "{%6}, {%7,%8,%9,%10};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(B), "r"(C[0]), "r"(C[1]), "r"(C[2]),
          "r"(C[3]));
#endif

  }
};

/// Matrix multiply-add operation: S32 = U8 * U8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,16>,
  32,
  uint8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16,8,16>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 8>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const &B = reinterpret_cast<uint32_t const &>(b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32.satfinite {%0,%1,%2,%3}, {%4,%5}, "    
        "{%6}, {%7,%8,%9,%10};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(B), "r"(C[0]), "r"(C[1]), "r"(C[2]),
          "r"(C[3]));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32.satfinite {%0,%1,%2,%3}, {%4,%5}, "    
        "{%6}, {%7,%8,%9,%10};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(B), "r"(C[0]), "r"(C[1]), "r"(C[2]),
          "r"(C[3]));
#endif

  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 16832 - S8 input, S32 accumulation
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = S8 * S8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,32>,
  32,
  int8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16,8,32>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 16>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {
#if SAIL_SIMULATE_CUTLASS_MMA
  const int8_t *frag_a = reinterpret_cast<const int8_t *>(&a);
  const int8_t *frag_b = reinterpret_cast<const int8_t *>(&b);
  const ElementC *frag_c = reinterpret_cast<const ElementC *>(&c);
  ElementC *frag_d = reinterpret_cast<ElementC *>(&d);
  int lane_idx = threadIdx.x % 32;
  unsigned mask = 0xffffffff;

  for (int loop = 0; loop < 4; loop++) {
    ElementC acc = 0;
    for (int k = 0; k < Shape::kK; k++) {
      int thread_a = lane_idx - (lane_idx % 4) + ((k - (k / 16) * 16) / 4);
      int reg_a = ((loop / 2) * 4) + (k / 16) * 8 + (k % 4);
      int8_t cur_a = __shfl_sync(mask, frag_a[reg_a], thread_a);
      ElementA cur_a_int;
      memcpy(&cur_a_int, &cur_a, sizeof(int8_t));
      int thread_b = (lane_idx % 4) * 8 + (loop % 2) * 4 + ((k - (k / 16) * 16) / 4);
      int reg_b = (k / 16) * 4 + (k % 4);
      int8_t cur_b = __shfl_sync(mask, frag_b[reg_b], thread_b);
      ElementB cur_b_int;
      memcpy(&cur_b_int, &cur_b, sizeof(int8_t));
      acc += ElementC(cur_a_int * cur_b_int);
    }
    frag_d[loop] = acc + frag_c[loop];
  }
#else
    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "    
        "{%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "    
        "{%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#endif

#endif
  }
};

/// Simulate 161632 Matrix multiply-add operation: S32 = S8 * S8 + S32
template <>
struct Mma<
  gemm::GemmShape<16, 16, 32>,
  32,
  int8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {
  using Shape = gemm::GemmShape<16, 16, 32>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 16>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 8>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {

    #if !SAIL_SIMULATE_CUTLASS_MMA
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      int const *C = reinterpret_cast<int const *>(&c);
      int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.mma.sync.aligned.m16n16k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "    
        "{%16,%17,%18,%19,%20,%21,%22,%23};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3]), "=r"(D[4]), "=r"(D[5]), "=r"(D[6]), "=r"(D[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(C[4]), "r"(C[5]), "r"(C[6]), "r"(C[7]));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n16k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "    
        "{%16,%17,%18,%19,%20,%21,%22,%23};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3]), "=r"(D[4]), "=r"(D[5]), "=r"(D[6]), "=r"(D[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(C[4]), "r"(C[5]), "r"(C[6]), "r"(C[7]));
#endif
    #else
    const int8_t *frag_a = reinterpret_cast<const int8_t *>(&a);
    const int8_t *frag_b = reinterpret_cast<const int8_t *>(&b);
    const ElementC *frag_c = reinterpret_cast<const ElementC *>(&c);
    ElementC *frag_d = reinterpret_cast<ElementC *>(&d);
    int lane_idx = threadIdx.x % 32;
    unsigned mask = 0xffffffff;

    for (int loop = 0; loop < 8; loop++) {
      ElementC acc = 0;
      for (int k = 0; k < Shape::kK; k++) {
        // lane idx A row base[0,4,8,12,16,20,24,28] + loop k for one k dim to find shfl thread who store continue
        int thread_a = lane_idx - (lane_idx % 4) + ((k % 16) / 4);
        // loop id to find base k dim(first 4 loop in same row) + k to find k dim across 16 elements.
        int reg_a = ((loop / 4) * 8) + (k / 16) * 4 + (k % 4);
        int8_t cur_a = __shfl_sync(mask, frag_a[reg_a], thread_a);
        ElementA cur_a_int;
        memcpy(&cur_a_int, &cur_a, sizeof(int8_t));
        // lane_idx B matrix n dim base + loop for one thread reg output is acrosss 128 bits(elements) + k to loop one colom thread(group 4)
        int thread_b = (lane_idx % 4) * 4 + (loop % 2) * 16 + ((k % 16) / 4);
        // loop to calcate row(k dim) base + k dim reg.
        int reg_b = (((loop % 4) / 2) * 8) + (k / 16) * 4 + (k % 4);
        int8_t cur_b = __shfl_sync(mask, frag_b[reg_b], thread_b);
        ElementB cur_b_int;
        memcpy(&cur_b_int, &cur_b, sizeof(int8_t));
        acc += ElementC(cur_a_int * cur_b_int);
      }
      frag_d[loop] = acc + frag_c[loop];
    }
    #endif
  }
};

template <>
struct Mma<
  gemm::GemmShape<16, 16, 32>,
  32,
  int8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {
  using Shape = gemm::GemmShape<16, 16, 32>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 16>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 8>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::PPU0010;

  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {

    #if !SAIL_SIMULATE_CUTLASS_MMA
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      int const *C = reinterpret_cast<int const *>(&c);
      int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.mma.sync.aligned.m16n16k32.row.col.satfinite.s32.s8.s8.s32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "    
        "{%16,%17,%18,%19,%20,%21,%22,%23};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3]), "=r"(D[4]), "=r"(D[5]), "=r"(D[6]), "=r"(D[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(C[4]), "r"(C[5]), "r"(C[6]), "r"(C[7]));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n16k32.row.col.satfinite.s32.s8.s8.s32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "    
        "{%16,%17,%18,%19,%20,%21,%22,%23};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3]), "=r"(D[4]), "=r"(D[5]), "=r"(D[6]), "=r"(D[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(C[4]), "r"(C[5]), "r"(C[6]), "r"(C[7]));
#endif
    #else
    const int8_t *frag_a = reinterpret_cast<const int8_t *>(&a);
    const int8_t *frag_b = reinterpret_cast<const int8_t *>(&b);
    const ElementC *frag_c = reinterpret_cast<const ElementC *>(&c);
    ElementC *frag_d = reinterpret_cast<ElementC *>(&d);
    int lane_idx = threadIdx.x % 32;
    unsigned mask = 0xffffffff;
    const ElementC c_max = (1u << (cutlass::sizeof_bits<ElementC>::value-1)) - 1;
    const ElementC c_min = -(1u << (cutlass::sizeof_bits<ElementC>::value-1));

    for (int loop = 0; loop < 8; loop++) {
      ElementC acc = 0;
      for (int k = 0; k < Shape::kK; k++) {
        // lane idx A row base[0,4,8,12,16,20,24,28] + loop k for one k dim to find shfl thread who store continue
        int thread_a = lane_idx - (lane_idx % 4) + ((k % 16) / 4);
        // loop id to find base k dim(first 4 loop in same row) + k to find k dim across 16 elements.
        int reg_a = ((loop / 4) * 8) + (k / 16) * 4 + (k % 4);
        int8_t cur_a = __shfl_sync(mask, frag_a[reg_a], thread_a);
        ElementA cur_a_int;
        memcpy(&cur_a_int, &cur_a, sizeof(int8_t));
        // lane_idx B matrix n dim base + loop for one thread reg output is acrosss 128 bits(elements) + k to loop one colom thread(group 4)
        int thread_b = (lane_idx % 4) * 4 + (loop % 2) * 16 + ((k % 16) / 4);
        // loop to calcate row(k dim) base + k dim reg.
        int reg_b = (((loop % 4) / 2) * 8) + (k / 16) * 4 + (k % 4);
        int8_t cur_b = __shfl_sync(mask, frag_b[reg_b], thread_b);
        ElementB cur_b_int;
        memcpy(&cur_b_int, &cur_b, sizeof(int8_t));
        int64_t temp_sum = int64_t(acc) + int64_t(ElementC(cur_a_int * cur_b_int));
        if (temp_sum > c_max) {
          temp_sum = c_max;
        } else if (temp_sum < c_min) {
          temp_sum = c_min;
        }
        acc = ElementC(temp_sum);
      }

      int64_t temp_sum = int64_t(acc) + int64_t(frag_c[loop]);
      if (temp_sum > c_max) {
        temp_sum = c_max;
      } else if (temp_sum < c_min) {
        temp_sum = c_min;
      }
      frag_d[loop] = ElementC(temp_sum);
    }
    #endif
  }
};

/// Simulate 161632 Matrix multiply-add operation: S32 = U8 * U8 + S32
template <>
struct Mma<
  gemm::GemmShape<16, 16, 32>,
  32,
  uint8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {
  using Shape = gemm::GemmShape<16, 16, 32>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 16>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 8>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {

    #if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__)) && !SAIL_SIMULATE_CUTLASS_MMA
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      int const *C = reinterpret_cast<int const *>(&c);
      int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.mma.sync.aligned.m16n16k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "    
        "{%16,%17,%18,%19,%20,%21,%22,%23};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3]), "=r"(D[4]), "=r"(D[5]), "=r"(D[6]), "=r"(D[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(C[4]), "r"(C[5]), "r"(C[6]), "r"(C[7]));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n16k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "    
        "{%16,%17,%18,%19,%20,%21,%22,%23};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3]), "=r"(D[4]), "=r"(D[5]), "=r"(D[6]), "=r"(D[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(C[4]), "r"(C[5]), "r"(C[6]), "r"(C[7]));
#endif
    #else
    const uint8_t *frag_a = reinterpret_cast<const uint8_t *>(&a);
    const uint8_t *frag_b = reinterpret_cast<const uint8_t *>(&b);
    const ElementC *frag_c = reinterpret_cast<const ElementC *>(&c);
    ElementC *frag_d = reinterpret_cast<ElementC *>(&d);
    int lane_idx = threadIdx.x % 32;
    unsigned mask = 0xffffffff;

    for (int loop = 0; loop < 8; loop++) {
      ElementC acc = 0;
      for (int k = 0; k < Shape::kK; k++) {
        // lane idx A row base[0,4,8,12,16,20,24,28] + loop k for one k dim to find shfl thread who store continue
        int thread_a = lane_idx - (lane_idx % 4) + ((k % 16) / 4);
        // loop id to find base k dim(first 4 loop in same row) + k to find k dim across 16 elements.
        int reg_a = ((loop / 4) * 8) + (k / 16) * 4 + (k % 4);
        uint8_t cur_a = __shfl_sync(mask, frag_a[reg_a], thread_a);
        ElementA cur_a_int;
        memcpy(&cur_a_int, &cur_a, sizeof(uint8_t));
        // lane_idx B matrix n dim base + loop for one thread reg output is acrosss 128 bits(elements) + k to loop one colom thread(group 4)
        int thread_b = (lane_idx % 4) * 4 + (loop % 2) * 16 + ((k % 16) / 4);
        // loop to calcate row(k dim) base + k dim regs.
        int reg_b = (((loop % 4) / 2) * 8) + (k / 16) * 4 + (k % 4);
        uint8_t cur_b = __shfl_sync(mask, frag_b[reg_b], thread_b);
        ElementB cur_b_int;
        memcpy(&cur_b_int, &cur_b, sizeof(uint8_t));
        acc += ElementC(cur_a_int * cur_b_int);
      }
      frag_d[loop] = acc + frag_c[loop];
    }
    #endif
  }
};

template <>
struct Mma<
  gemm::GemmShape<16, 16, 32>,
  32,
  uint8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {
  using Shape = gemm::GemmShape<16, 16, 32>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 16>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 8>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::PPU0010;

  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {

    #if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__)) && !SAIL_SIMULATE_CUTLASS_MMA
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      int const *C = reinterpret_cast<int const *>(&c);
      int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.mma.sync.aligned.m16n16k32.row.col.satfinite.s32.u8.u8.s32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "    
        "{%16,%17,%18,%19,%20,%21,%22,%23};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3]), "=r"(D[4]), "=r"(D[5]), "=r"(D[6]), "=r"(D[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(C[4]), "r"(C[5]), "r"(C[6]), "r"(C[7]));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n16k32.row.col.satfinite.s32.u8.u8.s32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "    
        "{%16,%17,%18,%19,%20,%21,%22,%23};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3]), "=r"(D[4]), "=r"(D[5]), "=r"(D[6]), "=r"(D[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(C[4]), "r"(C[5]), "r"(C[6]), "r"(C[7]));
#endif
    #else
    const uint8_t *frag_a = reinterpret_cast<const uint8_t *>(&a);
    const uint8_t *frag_b = reinterpret_cast<const uint8_t *>(&b);
    const ElementC *frag_c = reinterpret_cast<const ElementC *>(&c);
    ElementC *frag_d = reinterpret_cast<ElementC *>(&d);
    int lane_idx = threadIdx.x % 32;
    unsigned mask = 0xffffffff;
    const ElementC c_max = (1u << (cutlass::sizeof_bits<ElementC>::value-1)) - 1;
    const ElementC c_min = -(1u << (cutlass::sizeof_bits<ElementC>::value-1));

    for (int loop = 0; loop < 8; loop++) {
      ElementC acc = 0;
      for (int k = 0; k < Shape::kK; k++) {
        // lane idx A row base[0,4,8,12,16,20,24,28] + loop k for one k dim to find shfl thread who store continue
        int thread_a = lane_idx - (lane_idx % 4) + ((k % 16) / 4);
        // loop id to find base k dim(first 4 loop in same row) + k to find k dim across 16 elements.
        int reg_a = ((loop / 4) * 8) + (k / 16) * 4 + (k % 4);
        uint8_t cur_a = __shfl_sync(mask, frag_a[reg_a], thread_a);
        ElementA cur_a_int;
        memcpy(&cur_a_int, &cur_a, sizeof(uint8_t));
        // lane_idx B matrix n dim base + loop for one thread reg output is acrosss 128 bits(elements) + k to loop one colom thread(group 4)
        int thread_b = (lane_idx % 4) * 4 + (loop % 2) * 16 + ((k % 16) / 4);
        // loop to calcate row(k dim) base + k dim regs.
        int reg_b = (((loop % 4) / 2) * 8) + (k / 16) * 4 + (k % 4);
        uint8_t cur_b = __shfl_sync(mask, frag_b[reg_b], thread_b);
        ElementB cur_b_int;
        memcpy(&cur_b_int, &cur_b, sizeof(uint8_t));
        int64_t temp_sum = int64_t(acc) + int64_t(ElementC(cur_a_int * cur_b_int));
        if (temp_sum > c_max) {
          temp_sum = c_max;
        } else if (temp_sum < c_min) {
          temp_sum = c_min;
        }
        acc = ElementC(temp_sum);
      }
      int64_t temp_sum = int64_t(acc) + int64_t(frag_c[loop]);
      if (temp_sum > c_max) {
        temp_sum = c_max;
      } else if (temp_sum < c_min) {
        temp_sum = c_min;
      }
      frag_d[loop] = ElementC(temp_sum);
    }
    #endif
  }
};

template <>
struct Mma<
  gemm::GemmShape<16, 16, 16>,
  32,
  int8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {
  using Shape = gemm::GemmShape<16, 16, 16>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 8>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 8>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {
    #if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__))
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      int const *C = reinterpret_cast<int const *>(&c);
      int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.mma.sync.aligned.m16n16k16.row.col.s32.s8.s8.s32 "    
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3]), "=r"(D[4]), "=r"(D[5]), "=r"(D[6]), "=r"(D[7])
        : "r"(A[0]), "r"(A[1]),
          "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(C[4]), "r"(C[5]), "r"(C[6]), "r"(C[7]));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n16k16.row.col.s32.s8.s8.s32 "    
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3]), "=r"(D[4]), "=r"(D[5]), "=r"(D[6]), "=r"(D[7])
        : "r"(A[0]), "r"(A[1]),
          "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(C[4]), "r"(C[5]), "r"(C[6]), "r"(C[7]));
#endif
    #endif
  }
};

template <>
struct Mma<
  gemm::GemmShape<16, 16, 16>,
  32,
  uint8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {
  using Shape = gemm::GemmShape<16, 16, 16>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 8>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 8>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {
    #if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__))
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      int const *C = reinterpret_cast<int const *>(&c);
      int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.mma.sync.aligned.m16n16k16.row.col.s32.u8.u8.s32 "    
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3]), "=r"(D[4]), "=r"(D[5]), "=r"(D[6]), "=r"(D[7])
        : "r"(A[0]), "r"(A[1]),
          "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(C[4]), "r"(C[5]), "r"(C[6]), "r"(C[7]));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n16k16.row.col.s32.u8.u8.s32 "    
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3]), "=r"(D[4]), "=r"(D[5]), "=r"(D[6]), "=r"(D[7])
        : "r"(A[0]), "r"(A[1]),
          "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(C[4]), "r"(C[5]), "r"(C[6]), "r"(C[7]));
#endif
    #endif
  }
};

template <>
struct Mma<
  gemm::GemmShape<8, 16, 32>,
  32,
  int8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {
  using Shape = gemm::GemmShape<8, 16, 32>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 8>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {
    #if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__))
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      int const *C = reinterpret_cast<int const *>(&c);
      int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.mma.sync.aligned.m8n16k32.row.col.s32.s8.s8.s32 "    
        "{%0,%1,%2,%3}, {%4,%5}, {%6,%7,%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]),
          "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#endif
    #endif
  }
};

template <>
struct Mma<
  gemm::GemmShape<8, 16, 32>,
  32,
  uint8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {
  using Shape = gemm::GemmShape<8, 16, 32>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 8>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {
    #if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__))
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      int const *C = reinterpret_cast<int const *>(&c);
      int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.mma.sync.aligned.m8n16k32.row.col.s32.u8.u8.s32 "    
        "{%0,%1,%2,%3}, {%4,%5}, {%6,%7,%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]),
          "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#endif
    #endif
  }
};

template <>
struct Mma<
  gemm::GemmShape<8, 16, 16>,
  32,
  int8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {
  using Shape = gemm::GemmShape<8, 16, 16>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 4>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {
    #if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__))
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      int const *C = reinterpret_cast<int const *>(&c);
      int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.mma.sync.aligned.m8n16k16.row.col.s32.s8.s8.s32 "    
        "{%0,%1,%2,%3}, {%4}, {%5,%6}, {%7,%8,%9,%10};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]),
          "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#endif
    #endif
  }
};

template <>
struct Mma<
  gemm::GemmShape<8, 16, 16>,
  32,
  uint8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {
  using Shape = gemm::GemmShape<8, 16, 16>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 4>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {
    #if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__))
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      int const *C = reinterpret_cast<int const *>(&c);
      int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.mma.sync.aligned.m8n16k16.row.col.s32.u8.u8.s32 "    
        "{%0,%1,%2,%3}, {%4}, {%5,%6}, {%7,%8,%9,%10};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]),
          "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#endif
    #endif
  }
};

/// Matrix multiply-add operation: S32 = U8 * S8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,32>,
  32,
  uint8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16,8,32>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 16>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "    
        "{%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "    
        "{%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#endif

  }
};

/// Matrix multiply-add operation: S32 = S8 * U8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,32>,
  32,
  int8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16,8,32>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 16>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "    
        "{%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "    
        "{%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#endif

  }
};

/// Matrix multiply-add operation: S32 = U8 * U8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,32>,
  32,
  uint8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16,8,32>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 16>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "    
        "{%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "    
        "{%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#endif

  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 16832 - S8 input, S32 accumulation - SATURATE
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = S8 * S8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,32>,
  32,
  int8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16,8,32>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 16>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


  uint32_t const * A = reinterpret_cast<uint32_t const *>(&a);
  uint32_t const * B = reinterpret_cast<uint32_t const *>(&b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite {%0,%1,%2,%3}, "    
      "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
        "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite {%0,%1,%2,%3}, "    
      "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
        "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#endif

  }
};

/// Matrix multiply-add operation: S32 = U8 * S8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,32>,
  32,
  uint8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16,8,32>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 16>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32.satfinite {%0,%1,%2,%3}, "    
        "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32.satfinite {%0,%1,%2,%3}, "    
        "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#endif

  }
};

/// Matrix multiply-add operation: S32 = S8 * U8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,32>,
  32,
  int8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16,8,32>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 16>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32.satfinite {%0,%1,%2,%3}, "    
        "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32.satfinite {%0,%1,%2,%3}, "    
        "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#endif

  }
};

/// Matrix multiply-add operation: S32 = U8 * U8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,32>,
  32,
  uint8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16,8,32>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 16>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32.satfinite {%0,%1,%2,%3}, "    
        "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32.satfinite {%0,%1,%2,%3}, "    
        "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#endif

  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 16864 - S4 input, S32 accumulation
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = S4 * S4 + S32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 64>,
  32,
  cutlass::int4b_t,
  layout::RowMajor,
  cutlass::int4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 8, 64>;

  using ElementA = cutlass::int4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<cutlass::int4b_t, 32>;

  using ElementB = cutlass::int4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<cutlass::int4b_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
    assert(0);
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "    
        "{%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#endif

  }
};

/// Matrix multiply-add operation: S32 = U4 * S4 + S32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 64>,
  32,
  cutlass::uint4b_t,
  layout::RowMajor,
  cutlass::int4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 8, 64>;

  using ElementA = cutlass::uint4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<cutlass::uint4b_t, 32>;

  using ElementB = cutlass::int4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<cutlass::int4b_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
    assert(0);
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "    
        "{%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#endif

  }
};

/// Matrix multiply-add operation: S32 = S4 * U4 + S32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 64>,
  32,
  cutlass::int4b_t,
  layout::RowMajor,
  cutlass::uint4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 8, 64>;

  using ElementA = cutlass::int4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<cutlass::int4b_t, 32>;

  using ElementB = cutlass::uint4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<cutlass::uint4b_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
    assert(0);
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k64.row.col.s32.s4.u4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "    
        "{%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#endif

  }
};

/// Matrix multiply-add operation: S32 = U4 * U4 + S32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 64>,
  32,
  cutlass::uint4b_t,
  layout::RowMajor,
  cutlass::uint4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 8, 64>;

  using ElementA = cutlass::uint4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<cutlass::uint4b_t, 32>;

  using ElementB = cutlass::uint4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<cutlass::uint4b_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
    assert(0);
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k64.row.col.s32.u4.u4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "    
        "{%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#endif

  }
};


////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 16864 - S4 input, S32 accumulation - SATURATE
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = S4 * S4 + S32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 64>,
  32,
  cutlass::int4b_t,
  layout::RowMajor,
  cutlass::int4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16, 8, 64>;

  using ElementA = cutlass::int4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<cutlass::int4b_t, 32>;

  using ElementB = cutlass::int4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<cutlass::int4b_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


  uint32_t const * A = reinterpret_cast<uint32_t const *>(&a);
  uint32_t const * B = reinterpret_cast<uint32_t const *>(&b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
    assert(0);
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32.satfinite {%0,%1,%2,%3}, "    
      "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
        "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#endif

  }
};

/// Matrix multiply-add operation: S32 = U4 * S4 + S32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 64>,
  32,
  cutlass::uint4b_t,
  layout::RowMajor,
  cutlass::int4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16, 8, 64>;

  using ElementA = cutlass::uint4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<cutlass::uint4b_t, 32>;

  using ElementB = cutlass::int4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<cutlass::int4b_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
    assert(0);
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32.satfinite {%0,%1,%2,%3}, "    
        "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#endif

  }
};

/// Matrix multiply-add operation: S32 = S4 * U4 + S32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 64>,
  32,
  cutlass::int4b_t,
  layout::RowMajor,
  cutlass::uint4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16, 8, 64>;

  using ElementA = cutlass::int4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<cutlass::int4b_t, 32>;

  using ElementB = cutlass::uint4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<cutlass::uint4b_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
    assert(0);
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k64.row.col.s32.s4.u4.s32.satfinite {%0,%1,%2,%3}, "    
        "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#endif

  }
};

/// Matrix multiply-add operation: S32 = U4 * U4 + S32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 64>,
  32,
  cutlass::uint4b_t,
  layout::RowMajor,
  cutlass::uint4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16, 8, 64>;

  using ElementA = cutlass::uint4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<cutlass::uint4b_t, 32>;

  using ElementB = cutlass::uint4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<cutlass::uint4b_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
    assert(0);
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k64.row.col.s32.u4.u4.s32.satfinite {%0,%1,%2,%3}, "    
        "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#endif

  }
};

/// Matrix multiply-add operation: S32 = B1 & B1 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,256>,
  32,
  cutlass::uint1b_t,
  layout::RowMajor,
  cutlass::uint1b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16,8,256>;

  using ElementA = cutlass::uint1b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<cutlass::uint1b_t, 128>;

  using ElementB = cutlass::uint1b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<cutlass::uint1b_t, 64>;

  using ElementC = int32_t;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int32_t, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
    assert(0);
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc {%0,%1,%2,%3}, "    
        "{%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#endif

  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 168256 - B1 input, S32 accumulation - XOR,POPC
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = B1 & B1 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,256>,
  32,
  cutlass::uint1b_t,
  layout::RowMajor,
  cutlass::uint1b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpXorPopc> {

  using Shape = gemm::GemmShape<16,8,256>;

  using ElementA = cutlass::uint1b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<cutlass::uint1b_t, 128>;

  using ElementB = cutlass::uint1b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<cutlass::uint1b_t, 64>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpXorPopc;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
    assert(0);
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc {%0,%1,%2,%3}, "    
        "{%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#endif

  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace arch
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////



#if defined(__HGGCCC_RTC__)
#include <hggc/std/cassert>
#else
#include <assert.h>
#endif

// CUTLASS includes
#include "cutlass/arch/mma.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace arch {

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 1688 - FP16 accumulation
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation - F16 = F16 * F16 + F16
template <>
struct Mma<
  gemm::GemmShape<16, 8, 8>,
  32,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  half_t,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 8, 8>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 4>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 2>;

  using ElementC = half_t;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<half_t, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


  unsigned const *A = reinterpret_cast<unsigned const *>(&a);
  unsigned const *B = reinterpret_cast<unsigned const *>(&b);
  unsigned const *C = reinterpret_cast<unsigned const *>(&c);
  unsigned *D = reinterpret_cast<unsigned *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%5,%6};\n"    
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(C[0]), "r"(C[1]));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%5,%6};\n"    
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(C[0]), "r"(C[1]));
#endif

  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 1688 - FP32 accumulation
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F32 = F16 * F16 + F32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 8>,
  32,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 8, 8>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 4>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 2>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {
#if SAIL_SIMULATE_CUTLASS_MMA
  const uint16_t *frag_a = reinterpret_cast<const uint16_t *>(&a);
  const uint16_t *frag_b = reinterpret_cast<const uint16_t *>(&b);
  const ElementC *frag_c = reinterpret_cast<const ElementC *>(&c);
  ElementC *frag_d = reinterpret_cast<ElementC *>(&d);
  int lane_idx = threadIdx.x % 32;
  unsigned mask = 0xffffffff;
  // two elem in each 8x8
  for (int loop = 0; loop < 4; loop++) {
    ElementC acc = 0;
    for (int k = 0; k < Shape::kK; k++) {
      // threadIdx of elem a, will use two elem each time
      int thread_a = lane_idx - (lane_idx % 4) + (k / 2);
      // (loop / 2) * 2 corresponds to top part or down part, top part start from reg[0], down part start from reg[2]
      // k % 2 corresponds which elem in each part, inner loop will use two elem in each thread
      int reg_a = ((loop / 2) * 2) + (k % 2);
      uint16_t cur_a = __shfl_sync(mask, frag_a[reg_a], thread_a);
      ElementA cur_a_half;
      memcpy(&cur_a_half, &cur_a, sizeof(uint16_t));
      // first two part is which col in matrix b, last part is which row
      int thread_b = (lane_idx % 4) * 8 + (loop % 2) * 4 + k / 2;
      int reg_b = k % 2;
      uint16_t cur_b = __shfl_sync(mask, frag_b[reg_b], thread_b);
      ElementB cur_b_half;
      memcpy(&cur_b_half, &cur_b, sizeof(uint16_t));
      acc += ElementC(cur_a_half * cur_b_half);
    }
    frag_d[loop] = acc + frag_c[loop];
  }
#else
  unsigned const *A = reinterpret_cast<unsigned const *>(&a);
  unsigned const *B = reinterpret_cast<unsigned const *>(&b);
  float const *C = reinterpret_cast<float const *>(&c);
  float *D = reinterpret_cast<float *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"    
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      :
        "r"(A[0]), "r"(A[1]),
        "r"(B[0]),
        "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])
  );
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"    
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      :
        "r"(A[0]), "r"(A[1]),
        "r"(B[0]),
        "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])
  );
#endif
#endif
  }
};

#if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__))
// two ppu mma implementation, half->half, half->float
template <>
struct Mma<
  gemm::GemmShape<16, 16, 16>,
  32,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  half_t,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 16, 16>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 8>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 8>;

  using ElementC = half_t;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<half_t, 8>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  CUTLASS_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {
    // only fake mma now
    // assert(0);
    #if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__)) && !SAIL_SIMULATE_CUTLASS_MMA
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      uint32_t const *C = reinterpret_cast<uint32_t const *>(&c);
      uint32_t *D = reinterpret_cast<uint32_t *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.mma.sync.aligned.m16n16k16.row.col.f16.f16.f16.f16  {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, "    
      "{%12,%13,%14,%15};\n"
      : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
        "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
        "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n16k16.row.col.f16.f16.f16.f16  {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, "    
      "{%12,%13,%14,%15};\n"
      : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
        "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
        "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
#endif
    #else
      assert(0);
    #endif
  }
};

template <>
struct Mma<
  gemm::GemmShape<16, 16, 16>,
  32,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 16, 16>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 8>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 8>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 8>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {

    #if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__)) && !SAIL_SIMULATE_CUTLASS_MMA
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      float const *C = reinterpret_cast<float const *>(&c);
      float *D = reinterpret_cast<float *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32  {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "    
      "{%16,%17,%18,%19,%20,%21,%22,%23};\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3]), "=f"(D[4]), "=f"(D[5]), "=f"(D[6]), "=f"(D[7])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
        "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
        "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32  {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "    
      "{%16,%17,%18,%19,%20,%21,%22,%23};\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3]), "=f"(D[4]), "=f"(D[5]), "=f"(D[6]), "=f"(D[7])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
        "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
        "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]));
#endif

    #else
      // TSM_LD_NCOM_B32X4, similar to nvidia
      const uint16_t *frag_a = reinterpret_cast<const uint16_t *>(&a);
      const uint16_t *frag_b = reinterpret_cast<const uint16_t *>(&b);
      const ElementC *frag_c = reinterpret_cast<const ElementC *>(&c);
      ElementC *frag_d = reinterpret_cast<ElementC *>(&d);
      int lane_idx = threadIdx.x % 32;
      unsigned mask = 0xffffffff;
      // two elem in each 8x8
      for (int loop = 0; loop < 8; loop++) {
        ElementC acc = 0;
        for (int k = 0; k < Shape::kK; k++) {
          // threadIdx of elem a, will use two elem each time, right 8 k's thread is identical to left 8 k
          int thread_a = lane_idx - (lane_idx % 4) + ((k % 8) / 2);
          // (loop / 4) * 4 corresponds to top or down part, ((k / 8) * 2) corresponds to left or right part
          // k % 2 corresponds to which elem in each part, inner loop will use two elem in each thread
          int reg_a = ((loop / 4) * 4) + ((k / 8) * 2) + (k % 2);
          uint16_t cur_a = __shfl_sync(mask, frag_a[reg_a], thread_a);
          ElementA cur_a_half;
          memcpy(&cur_a_half, &cur_a, sizeof(uint16_t));
          // first two part is which col in matrix b, last part is which row
          // int thread_b = (lane_idx % 4) * 8 + (loop % 2) * 4 + (k % 8) / 2;
          int thread_b = (lane_idx % 4) * 4 + (loop % 2) * 16 + (k % 8) / 2;
          // first is hori part, second is vert part, third is index in cur part
          int reg_b = (((loop % 4) / 2) * 4) + (k / 8) * 2 + k % 2;
          uint16_t cur_b = __shfl_sync(mask, frag_b[reg_b], thread_b);
          ElementB cur_b_half;
          memcpy(&cur_b_half, &cur_b, sizeof(uint16_t));
          acc += ElementC(cur_a_half * cur_b_half);
        }
        frag_d[loop] = acc + frag_c[loop];
      }

      // VMEM_LD_B32X4 version
      // const uint16_t *frag_a = reinterpret_cast<const uint16_t *>(&a);
      // const uint16_t *frag_b = reinterpret_cast<const uint16_t *>(&b);
      // const ElementC *frag_c = reinterpret_cast<const ElementC *>(&c);
      // ElementC *frag_d = reinterpret_cast<ElementC *>(&d);
      // int lane_idx = threadIdx.x % 32;
      // unsigned mask = 0xffffffff;
      // ElementC acc = 0;
      // // each thread get 8 outputs of half row
      // for (int loop = 0; loop < 8; loop++) {
      //   for (int k = 0; k < Shape::kK; k++) {
      //     int thread_a = lane_idx - (lane_idx % 2) + (k / 8);
      //     int reg_a = k % 8;
      //     // shfl_sync can't shuffle half
      //     uint16_t cur_a = __shfl_sync(mask, frag_a[reg_a], thread_a);
      //     ElementA cur_a_half;
      //     memcpy(&cur_a_half, &cur_a, sizeof(uint16_t));
      //     // first two part is which col in matrix b, last part is which row
      //     int thread_b = (lane_idx % 2) * 16 + loop * 2 + k / 8;
      //     int reg_b = k % 8;
      //     uint16_t cur_b = __shfl_sync(mask, frag_b[reg_b], thread_b);
      //     ElementB cur_b_half;
      //     memcpy(&cur_b_half, &cur_b, sizeof(uint16_t));
      //     acc += ElementC(cur_a_half * cur_b_half);
      //     // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 11) {
      //     //   printf("%d, %d, %d, %d, %d, %d, 111111111\n", loop, k, thread_a, reg_a, thread_b, reg_b);
      //     // }
      //   }
      //   frag_d[loop] = acc + frag_c[loop];
      // }
    #endif
    return;

  }
};
#endif

////////////////////////////////////////////////////////////////////////////////
//
// Integer matrix multiply .8816 (8b)
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = S8 * S8 + S32
template <>
struct Mma<
  gemm::GemmShape<8, 8, 16>,
  32,
  int8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8, 8, 16>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 4>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
  assert(0);
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"    
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A), "r"(B), "r"(C[0]), "r"(C[1]));
#endif

  }
};

/// Matrix multiply-add operation: S32 = U8 * S8 + S32
template <>
struct Mma<
  gemm::GemmShape<8, 8, 16>,
  32,
  uint8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8, 8, 16>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 4>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
  assert(0);
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m8n8k16.row.col.s32.u8.s8.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"    
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A), "r"(B), "r"(C[0]), "r"(C[1]));
#endif

  }
};

/// Matrix multiply-add operation: S32 = S8 * U8 + S32
template <>
struct Mma<
  gemm::GemmShape<8, 8, 16>,
  32,
  int8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8, 8, 16>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 4>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
  assert(0);
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m8n8k16.row.col.s8.u8 {%0,%1}, {%2}, {%3}, {%4,%5};\n"    
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A), "r"(B), "r"(C[0]), "r"(C[1]));
#endif


  }
};

/// Matrix multiply-add operation: S32 = U8 * U8 + S32
template <>
struct Mma<
  gemm::GemmShape<8, 8, 16>,
  32,
  uint8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8, 8, 16>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 4>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
  assert(0);
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m8n8k16.row.col.s32.u8.u8.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"    
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A), "r"(B), "r"(C[0]), "r"(C[1]));
#endif

  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Integer matrix multiply  (8b) with SATURATE
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = S8 * S8 + S32
template <>
struct Mma<
  gemm::GemmShape<8,8,16>,
  32,
  int8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<8,8,16>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 4>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
  assert(0);
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"    
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A), "r"(B), "r"(C[0]), "r"(C[1]));
#endif

  }
};

/// Matrix multiply-add operation: S32 = U8 * S8 + S32
template <>
struct Mma<
  gemm::GemmShape<8,8,16>,
  32,
  uint8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<8,8,16>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 4>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
  assert(0);
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.s8.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"    
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A), "r"(B), "r"(C[0]), "r"(C[1]));
#endif

  }
};

/// Matrix multiply-add operation: S32 = S8 * U8 + S32
template <>
struct Mma<
  gemm::GemmShape<8,8,16>,
  32,
  int8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<8,8,16>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 4>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
  assert(0);
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.u8.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"    
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A), "r"(B), "r"(C[0]), "r"(C[1]));
#endif

  }
};

/// Matrix multiply-add operation: S32 = U8 * U8 + S32
template <>
struct Mma<
  gemm::GemmShape<8,8,16>,
  32,
  uint8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<8,8,16>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 4>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
  assert(0);
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"    
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A), "r"(B), "r"(C[0]), "r"(C[1]));
#endif

  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Integer matrix multiply  (4b)
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = S4 * S4 + S32
template <>
struct Mma<
  gemm::GemmShape<8,8,32>,
  32,
  int4b_t,
  layout::RowMajor,
  int4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8,8,32>;

  using ElementA = int4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int4b_t, 8>;

  using ElementB = int4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int4b_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

assert(0);

  }
};

/// Matrix multiply-add operation: S32 = U4 * S4 + S32
template <>
struct Mma<
  gemm::GemmShape<8,8,32>,
  32,
  uint4b_t,
  layout::RowMajor,
  int4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8,8,32>;

  using ElementA = uint4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint4b_t, 8>;

  using ElementB = int4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int4b_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

assert(0);

  }
};

/// Matrix multiply-add operation: S32 = S4 * U4 + S32
template <>
struct Mma<
  gemm::GemmShape<8,8,32>,
  32,
  int4b_t,
  layout::RowMajor,
  uint4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8,8,32>;

  using ElementA = int4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int4b_t, 8>;

  using ElementB = uint4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint4b_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

assert(0);

  }
};

/// Matrix multiply-add operation: S32 = U4 * U4 + S32
template <>
struct Mma<
  gemm::GemmShape<8,8,32>,
  32,
  uint4b_t,
  layout::RowMajor,
  uint4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8,8,32>;

  using ElementA = uint4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint4b_t, 8>;

  using ElementB = uint4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint4b_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

assert(0);

  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Integer matrix multiply  (4b) - SATURATE
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = S4 * S4 + S32
template <>
struct Mma<
  gemm::GemmShape<8,8,32>,
  32,
  int4b_t,
  layout::RowMajor,
  int4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<8,8,32>;

  using ElementA = int4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int4b_t, 8>;

  using ElementB = int4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int4b_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

assert(0);

  }
};

/// Matrix multiply-add operation: S32 = U4 * S4 + S32
template <>
struct Mma<
  gemm::GemmShape<8,8,32>,
  32,
  uint4b_t,
  layout::RowMajor,
  int4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<8,8,32>;

  using ElementA = uint4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint4b_t, 8>;

  using ElementB = int4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int4b_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

assert(0);

  }
};

/// Matrix multiply-add operation: S32 = S4 * U4 + S32
template <>
struct Mma<
  gemm::GemmShape<8,8,32>,
  32,
  int4b_t,
  layout::RowMajor,
  uint4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<8,8,32>;

  using ElementA = int4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int4b_t, 8>;

  using ElementB = uint4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint4b_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

assert(0);

  }
};

/// Matrix multiply-add operation: S32 = U4 * U4 + S32
template <>
struct Mma<
  gemm::GemmShape<8,8,32>,
  32,
  uint4b_t,
  layout::RowMajor,
  uint4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<8,8,32>;

  using ElementA = uint4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint4b_t, 8>;

  using ElementB = uint4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint4b_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

assert(0);

  }
};

////////////////////////////////////////////////////////////////////////////////
//
// b1 ^ b1 + s32 => s32
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <>
struct Mma<
  gemm::GemmShape<8,8,128>,
  32,
  uint1b_t,
  layout::RowMajor,
  uint1b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpXorPopc> {

  using Shape = gemm::GemmShape<8,8,128>;

  using ElementA = uint1b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint1b_t, 32>;

  using ElementB = uint1b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint1b_t, 32>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpXorPopc;
  using ArchTag = arch::PPU0010;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {


  assert(0); // Binary matrix multiply not supported on PPU.


  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace arch
} // namespace cutlass

#if defined(__HGGCCC_RTC__)
#include <hggc/std/cassert>
#else
#include <assert.h>
#endif

#include "mma.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

#if ((__HGGCCC_VER_MAJOR__ > 10) || (__HGGCCC_VER_MAJOR__ == 10 && __HGGCCC_VER_MINOR__ >= 1))
#define CUTLASS_ARCH_MMA_PPU0010_SUPPORTED
#endif

#if (defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 100))

#if ((__HGGCCC_VER_MAJOR__ > 10) || (__HGGCCC_VER_MAJOR__ == 10 &&__HGGCCC_VER_MINOR__ >= 1))
#define CUTLASS_ARCH_MMA_PPU0010_ENABLED
#endif

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace arch {

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Matrix multiply accumulate 884 - FP16 accumulation
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F16 = F16 * F16 + F16
template <>
struct Mma<
  gemm::GemmShape<8,8,4>,
  8,
  half_t,
  layout::ColumnMajor,
  half_t,
  layout::ColumnMajor,
  half_t,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8, 8, 4>;

  using ElementA = half_t;
  using LayoutA = layout::ColumnMajor;
  using FragmentA = Array<half_t, 4>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 4>;

  using ElementC = half_t;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<half_t, 8>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) {

#if defined(CUTLASS_ARCH_MMA_PPU0010_ENABLED)

    unsigned const *A = reinterpret_cast<unsigned const *>(&a);
    unsigned const *B = reinterpret_cast<unsigned const *>(&b);
    unsigned const *C = reinterpret_cast<unsigned const *>(&c);
    unsigned *D = reinterpret_cast<unsigned *>(&d);

assert(0);

#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: F16 = F16 * F16 + F16
template <>
struct Mma<
  gemm::GemmShape<8, 8, 4>,
  8,
  half_t,
  layout::ColumnMajor,
  half_t,
  layout::RowMajor,
  half_t,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8, 8, 4>;

  using ElementA = half_t;
  using LayoutA = layout::ColumnMajor;
  using FragmentA = Array<half_t, 4>;

  using ElementB = half_t;
  using LayoutB = layout::RowMajor;
  using FragmentB = Array<half_t, 4>;

  using ElementC = half_t;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<half_t, 8>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) {

#if defined(CUTLASS_ARCH_MMA_PPU0010_ENABLED)

    unsigned const *A = reinterpret_cast<unsigned const *>(&a);
    unsigned const *B = reinterpret_cast<unsigned const *>(&b);
    unsigned const *C = reinterpret_cast<unsigned const *>(&c);
    unsigned *D = reinterpret_cast<unsigned *>(&d);

    assert(0);

#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: F16 = F16 * F16 + F16
template <>
struct Mma<
  gemm::GemmShape<8, 8, 4>,
  8,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  half_t,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8, 8, 4>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 4>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 4>;

  using ElementC = half_t;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<half_t, 8>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) {

#if defined(CUTLASS_ARCH_MMA_PPU0010_ENABLED)

    unsigned const *A = reinterpret_cast<unsigned const *>(&a);
    unsigned const *B = reinterpret_cast<unsigned const *>(&b);
    unsigned const *C = reinterpret_cast<unsigned const *>(&c);
    unsigned *D = reinterpret_cast<unsigned *>(&d);

    assert(0);

#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: F16 = F16 * F16 + F16
template <>
struct Mma<
  gemm::GemmShape<8, 8, 4>,
  8,
  half_t,
  layout::RowMajor,
  half_t,
  layout::RowMajor,
  half_t,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8, 8, 4>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 4>;

  using ElementB = half_t;
  using LayoutB = layout::RowMajor;
  using FragmentB = Array<half_t, 4>;

  using ElementC = half_t;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<half_t, 8>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) {

#if defined(CUTLASS_ARCH_MMA_PPU0010_ENABLED)

    unsigned const *A = reinterpret_cast<unsigned const *>(&a);
    unsigned const *B = reinterpret_cast<unsigned const *>(&b);
    unsigned const *C = reinterpret_cast<unsigned const *>(&c);
    unsigned *D = reinterpret_cast<unsigned *>(&d);

    assert(0);

#else
    assert(0);
#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Matrix multiply accumulate 884 - FP32 accumulation
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F32 = F16 * F16 + F32
template <>
struct Mma<
  gemm::GemmShape<8, 8, 4>,
  8,
  half_t,
  layout::ColumnMajor,
  half_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8, 8, 4>;

  using ElementA = half_t;
  using LayoutA = layout::ColumnMajor;
  using FragmentA = Array<half_t, 4>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 4>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 8>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  /// Multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) {

#if defined(CUTLASS_ARCH_MMA_PPU0010_ENABLED)

  unsigned const *A = reinterpret_cast<unsigned const *>(&a);
  unsigned const *B = reinterpret_cast<unsigned const *>(&b);
  float const *C = reinterpret_cast<float const *>(&c);
  float *D = reinterpret_cast<float *>(&d);

  assert(0);

#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: F32 = F16 * F16 + F32
template <>
struct Mma<
  gemm::GemmShape<8, 8, 4>,
  8,
  half_t,
  layout::ColumnMajor,
  half_t,
  layout::RowMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8, 8, 4>;

  using ElementA = half_t;
  using LayoutA = layout::ColumnMajor;
  using FragmentA = Array<half_t, 4>;

  using ElementB = half_t;
  using LayoutB = layout::RowMajor;
  using FragmentB = Array<half_t, 4>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 8>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  /// Multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) {

#if defined(CUTLASS_ARCH_MMA_PPU0010_ENABLED)

  unsigned const *A = reinterpret_cast<unsigned const *>(&a);
  unsigned const *B = reinterpret_cast<unsigned const *>(&b);
  float const *C = reinterpret_cast<float const *>(&c);
  float *D = reinterpret_cast<float *>(&d);

  assert(0);

#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: F32 = F16 * F16 + F32
template <>
struct Mma<
  gemm::GemmShape<8, 8, 4>,
  8,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8, 8, 4>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 4>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 4>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 8>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  /// Multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) {

#if defined(CUTLASS_ARCH_MMA_PPU0010_ENABLED)

  unsigned const *A = reinterpret_cast<unsigned const *>(&a);
  unsigned const *B = reinterpret_cast<unsigned const *>(&b);
  float const *C = reinterpret_cast<float const *>(&c);
  float *D = reinterpret_cast<float *>(&d);

  assert(0);

#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: F32 = F16 * F16 + F32
template <>
struct Mma<
  gemm::GemmShape<8, 8, 4>,
  8,
  half_t,
  layout::RowMajor,
  half_t,
  layout::RowMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8, 8, 4>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 4>;

  using ElementB = half_t;
  using LayoutB = layout::RowMajor;
  using FragmentB = Array<half_t, 4>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 8>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::PPU0010;

  /// Multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) {

#if defined(CUTLASS_ARCH_MMA_PPU0010_ENABLED)

  unsigned const *A = reinterpret_cast<unsigned const *>(&a);
  unsigned const *B = reinterpret_cast<unsigned const *>(&b);
  float const *C = reinterpret_cast<float const *>(&c);
  float *D = reinterpret_cast<float *>(&d);

  assert(0);

#else
    assert(0);
#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation specialized for the entire warp
template <
  typename LayoutA,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename Operator
>
struct Mma<
  gemm::GemmShape<16, 16, 4>,
  32,
  half_t,
  LayoutA,
  half_t,
  LayoutB,
  ElementC,
  LayoutC,
  Operator
> : 
  public Mma<
    gemm::GemmShape<8, 8, 4>, 
    8, 
    half_t, 
    LayoutA, 
    half_t, 
    LayoutB,
    ElementC, 
    LayoutC, 
    Operator> {

  using Shape = gemm::GemmShape<16, 16, 4>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace arch
} // namespace cutlass

#include <hggc_fp16.h>

#include "cutlass/arch/mma.h"

#include "cutlass/layout/matrix.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace arch {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <typename LayoutA, typename LayoutB, typename LayoutC>
struct Mma<
  gemm::GemmShape<2,1,1>,
  1,
  half_t,
  LayoutA,
  half_t,
  LayoutB,
  half_t,
  LayoutC,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<2, 1, 1>;
  using Operator = OpMultiplyAdd;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<half_t, 2> &d,
    Array<half_t, 2> const &a,
    Array<half_t, 1> const &b,
    Array<half_t, 2> const &c
  ) {

#if (defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 100))

    __half2 const & A = reinterpret_cast<__half2 const &>(a);
    __half2 B = __half2half2(reinterpret_cast<__half const &>(b));
    __half2 const & C = reinterpret_cast<__half2 const &>(c);

    __half2 D = __hfma2(A, B, C);

    d = reinterpret_cast<Array<half_t, 2> &>(D);

#else
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      d[i] = a[i] * b[0] + c[i];
    }
#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <typename LayoutA, typename LayoutB>
struct Mma<
  gemm::GemmShape<1,2,1>,
  1,
  half_t,
  LayoutA,
  half_t,
  LayoutB,
  half_t,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<1, 2, 1>;
  using Operator = OpMultiplyAdd;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<half_t, 2> &d,
    Array<half_t, 1> const &a,
    Array<half_t, 2> const &b,
    Array<half_t, 2> const &c
  ) {

#if (defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 100))

    __half2 const & A = __half2half2(reinterpret_cast<__half const &>(a));
    __half2 B = reinterpret_cast<__half2 const &>(b);
    __half2 const & C = reinterpret_cast<__half2 const &>(c);

    __half2 D = __hfma2(A, B, C);

    d = reinterpret_cast<Array<half_t, 2> &>(D);

#else
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      d[i] = a[0] * b[i] + c[i];
    }
#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <>
struct Mma <
  gemm::GemmShape<2, 2, 1>,
  1,
  half_t,
  layout::ColumnMajor,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<2, 2, 1>;
  using Operator = OpMultiplyAdd;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<half_t, 4> &d,
    Array<half_t, 2> const &a,
    Array<half_t, 2> const &b,
    Array<half_t, 4> const &c
  ) {

#if (defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 100))

    __half2 const & A = reinterpret_cast<__half2 const &>(a);
    __half2 Blo = __low2half2(reinterpret_cast<__half2 const &>(b));
    __half2 Bhi = __high2half2(reinterpret_cast<__half2 const &>(b));

    __half2 const *C = reinterpret_cast<__half2 const *>(&c);

    __half2 Dlo = __hfma2(A, Blo, C[0]);
    __half2 Dhi = __hfma2(A, Bhi, C[1]);

    Array<half_t, 2> * D = reinterpret_cast<Array<half_t, 2> *>(&d);

    D[0] = reinterpret_cast<Array<half_t, 2> const &>(Dlo);
    D[1] = reinterpret_cast<Array<half_t, 2> const &>(Dhi);

#else
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < 2; ++j) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < 2; ++i) {
        d[i + 2 * j] = a[i] * b[j] + c[i + 2 * j];
      }
    }
#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <>
struct Mma<
  gemm::GemmShape<2, 2, 1>,
  1,
  half_t,
  layout::ColumnMajor,
  half_t,
  layout::RowMajor,
  half_t,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<2, 2, 1>;
  using Operator = OpMultiplyAdd;
  
  CUTLASS_HOST_DEVICE
  void operator()(
    Array<half_t, 4> &d,
    Array<half_t, 2> const &a,
    Array<half_t, 2> const &b,
    Array<half_t, 4> const &c
  ) {

#if (defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 100))

    __half2 Alo = __low2half2(reinterpret_cast<__half2 const &>(a));
    __half2 Ahi = __high2half2(reinterpret_cast<__half2 const &>(a));
    __half2 const & B = reinterpret_cast<__half2 const &>(b);
    
    __half2 const *C = reinterpret_cast<__half2 const *>(&c);

    __half2 Dlo = __hfma2(Alo, B, C[0]);
    __half2 Dhi = __hfma2(Ahi, B, C[0]);
    
    Array<half_t, 2> * D = reinterpret_cast<Array<half_t, 2> *>(&d);

    D[0] = reinterpret_cast<Array<half_t, 2> &>(Dlo);
    D[1] = reinterpret_cast<Array<half_t, 2> &>(Dhi);
#else
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < 2; ++j) {
        d[i * 2 + j] = a[i] * b[j] + c[i * 2 + j];
      }
    }
#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}
}

#include "cutlass/layout/matrix.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace arch {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <typename LayoutA, typename LayoutB, typename LayoutC>
struct Mma<
  gemm::GemmShape<1,1,4>,
  1,
  int8_t,
  LayoutA,
  int8_t,
  LayoutB,
  int,
  LayoutC,
  OpMultiplyAdd> {
  
  using Shape = gemm::GemmShape<1, 1, 4>;
  using Operator = OpMultiplyAdd;
  
  CUTLASS_HOST_DEVICE
  void operator()(
    Array<int, 1> &d,
    Array<int8_t, 4> const &a,
    Array<int8_t, 4> const &b,
    Array<int, 1> const &c
  ) {

#if (defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 100))

    unsigned const &A = reinterpret_cast<unsigned const &>(a);
    unsigned const &B = reinterpret_cast<unsigned const &>(b);

    asm volatile("ppu.dp4a.s32.s32 %0, %1, %2, %3;"
                 : "=r"(d[0])
                 : "r"(A), "r"(B), "r"(c[0]));

#else

    d[0] = c[0];

    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < 4; ++k) {
      d[0] += a[k] * b[k];
    }

#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <typename LayoutC>
struct Mma<
  gemm::GemmShape<1, 1, 2>,
  1,
  int16_t,
  layout::RowMajor,
  int16_t,
  layout::ColumnMajor,
  int,
  LayoutC,
  OpMultiplyAdd> {
  
  using Shape = gemm::GemmShape<1, 1, 2>;
  using Operator = OpMultiplyAdd;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<int, 1> &d,
    Array<int16_t, 2> const &a,
    Array<int16_t, 2> const &b,
    Array<int, 1> const &c
  ) {

    d[0] = c[0];

    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < 2; ++k) {
      d[0] += a[k] * b[k];
    }

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}
}

#include "cutlass/arch/mma.h"
#include "cutlass/complex.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/gemm.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace arch {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <
  /// Layout of A matrix
  typename LayoutA,
  /// Layout of B matrix
  typename LayoutB,
  /// Layout of C matrix
  typename LayoutC
>
struct Mma<gemm::GemmShape<1, 1, 1>, 1, float, LayoutA, float, LayoutB, float, LayoutC, OpMultiplyAdd> {

  using Shape = gemm::GemmShape<1, 1, 1>;
  using Operator = OpMultiplyAdd;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<float, 1> &d,
    Array<float, 1> const &a,
    Array<float, 1> const &b,
    Array<float, 1> const &c
  ) {
    d[0] = a[0] * b[0] + c[0];
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <
  /// Layout of A matrix
  typename LayoutA,
  /// Layout of B matrix
  typename LayoutB,
  /// Layout of C matrix
  typename LayoutC
>
struct Mma<gemm::GemmShape<1, 1, 1>, 1, double, LayoutA, double, LayoutB, double, LayoutC, OpMultiplyAdd> {

  using Shape = gemm::GemmShape<1, 1, 1>;
  using Operator = OpMultiplyAdd;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<double, 1> &d,
    Array<double, 1> const &a,
    Array<double, 1> const &b,
    Array<double, 1> const &c
  ) {

    d[0] = a[0] * b[0] + c[0];
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <
  /// Layout of A matrix
  typename LayoutA,
  /// Layout of B matrix
  typename LayoutB,
  /// Layout of C matrix
  typename LayoutC
>
struct Mma<gemm::GemmShape<1, 1, 1>, 1, int, LayoutA, int, LayoutB, int, LayoutC, OpMultiplyAdd> {

  using Shape = gemm::GemmShape<1, 1, 1>;
  using Operator = OpMultiplyAdd;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<int, 1> &d,
    Array<int, 1> const &a,
    Array<int, 1> const &b,
    Array<int, 1> const &c
  ) {

    d[0] = a[0] * b[0] + c[0];
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <
  /// Layout of A matrix
  typename LayoutA,
  /// Layout of B matrix
  typename LayoutB,
  /// Layout of C matrix
  typename LayoutC
>
struct Mma<
  gemm::GemmShape<1, 1, 1>,
  1,
  complex<float>, 
  LayoutA, 
  complex<float>, 
  LayoutB, 
  complex<float>, 
  LayoutC, 
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<1, 1, 1>;
  using Operator = OpMultiplyAddComplex;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<complex<float>, 1> &d,
    Array<complex<float>, 1> const &a,
    Array<complex<float>, 1> const &b,
    Array<complex<float>, 1> const &c
  ) {

    d[0].real() = a[0].real() * b[0].real() + c[0].real();
    d[0].imag() = a[0].imag() * b[0].real() + c[0].imag();
    d[0].real() = -a[0].imag() * b[0].imag() + d[0].real();
    d[0].imag() = a[0].real() * b[0].imag() + d[0].imag();
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <
  /// Layout of A matrix
  typename LayoutA,
  /// Layout of B matrix
  typename LayoutB,
  /// Layout of C matrix
  typename LayoutC
>
struct Mma<
  gemm::GemmShape<1, 1, 1>,
  1,
  complex<float>, 
  LayoutA, 
  float, 
  LayoutB, 
  complex<float>, 
  LayoutC, 
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<1, 1, 1>;
  using Operator = OpMultiplyAddComplex;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<complex<float>, 1> &d,
    Array<complex<float>, 1> const &a,
    Array<float, 1> const &b,
    Array<complex<float>, 1> const &c
  ) {

    d[0].real() = a[0].real() * b[0] + c[0].real();
    d[0].imag() = a[0].imag() * b[0] + c[0].imag();
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <
  /// Layout of A matrix
  typename LayoutA,
  /// Layout of B matrix
  typename LayoutB,
  /// Layout of C matrix
  typename LayoutC
>
struct Mma<
  gemm::GemmShape<1, 1, 1>,
  1,
  float, 
  LayoutA, 
  complex<float>, 
  LayoutB, 
  complex<float>, 
  LayoutC, 
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<1, 1, 1>;
  using Operator = OpMultiplyAddComplex;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<complex<float>, 1> &d,
    Array<float, 1> const &a,
    Array<complex<float>, 1> const &b,
    Array<complex<float>, 1> const &c
  ) {

    d[0].real() = a[0] * b[0].real() + c[0].real();
    d[0].imag() = a[0] * b[0].imag() + d[0].imag();
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <
  /// Layout of A matrix
  typename LayoutA,
  /// Layout of B matrix
  typename LayoutB,
  /// Layout of C matrix
  typename LayoutC
>
struct Mma<
  gemm::GemmShape<1, 1, 1>,
  1,
  complex<double>, 
  LayoutA, 
  complex<double>, 
  LayoutB, 
  complex<double>, 
  LayoutC, 
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<1, 1, 1>;
  using Operator = OpMultiplyAddComplex;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<complex<double>, 1> &d,
    Array<complex<double>, 1> const &a,
    Array<complex<double>, 1> const &b,
    Array<complex<double>, 1> const &c
  ) {

    d[0].real() = a[0].real() * b[0].real() + c[0].real();
    d[0].imag() = a[0].imag() * b[0].real() + c[0].imag();
    d[0].real() = -a[0].imag() * b[0].imag() + d[0].real();
    d[0].imag() = a[0].real() * b[0].imag() + d[0].imag();
  }
};

/// Matrix multiply-add operation
template <
  /// Layout of A matrix
  typename LayoutA,
  /// Layout of B matrix
  typename LayoutB,
  /// Layout of C matrix
  typename LayoutC
>
struct Mma<
  gemm::GemmShape<1, 1, 1>,
  1,
  complex<double>, 
  LayoutA, 
  double, 
  LayoutB, 
  complex<double>, 
  LayoutC, 
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<1, 1, 1>;
  using Operator = OpMultiplyAddComplex;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<complex<double>, 1> &d,
    Array<complex<double>, 1> const &a,
    Array<double, 1> const &b,
    Array<complex<double>, 1> const &c
  ) {

    d[0].real() = a[0].real() * b[0] + c[0].real();
    d[0].imag() = a[0].imag() * b[0] + c[0].imag();
  }
};

/// Matrix multiply-add operation
template <
  /// Layout of A matrix
  typename LayoutA,
  /// Layout of B matrix
  typename LayoutB,
  /// Layout of C matrix
  typename LayoutC
>
struct Mma<
  gemm::GemmShape<1, 1, 1>,
  1,
  double, 
  LayoutA, 
  complex<double>, 
  LayoutB, 
  complex<double>, 
  LayoutC, 
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<1, 1, 1>;
  using Operator = OpMultiplyAddComplex;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<complex<double>, 1> &d,
    Array<double, 1> const &a,
    Array<complex<double>, 1> const &b,
    Array<complex<double>, 1> const &c
  ) {

    d[0].real() = a[0] * b[0].real() + c[0].real();
    d[0].imag() = a[0] * b[0].imag() + d[0].imag();
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <
  /// Layout of A matrix
  typename LayoutA,
  /// Layout of B matrix
  typename LayoutB,
  /// Layout of C matrix
  typename LayoutC
>
struct Mma<gemm::GemmShape<1, 1, 1>, 1, half_t, LayoutA, half_t, LayoutB, float, LayoutC, OpMultiplyAdd> {

  using Shape = gemm::GemmShape<1, 1, 1>;
  using Operator = OpMultiplyAdd;
  
  CUTLASS_HOST_DEVICE
  void operator()(
    Array<float, 1> &d,
    Array<half_t, 1> const &a,
    Array<half_t, 1> const &b,
    Array<float, 1> const &c
  ) {
    d[0] = float(a[0]) * float(b[0]) + c[0];
  }
};

/// Matrix multiply-add operation
template <
  /// Layout of A matrix
  typename LayoutA,
  /// Layout of B matrix
  typename LayoutB,
  /// Layout of C matrix
  typename LayoutC
>
struct Mma<gemm::GemmShape<1, 1, 1>, 1, bfloat16_t, LayoutA, bfloat16_t, LayoutB, float, LayoutC, OpMultiplyAdd> {

  using Shape = gemm::GemmShape<1, 1, 1>;
  using Operator = OpMultiplyAdd;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<float, 1> &d,
    Array<bfloat16_t, 1> const &a,
    Array<bfloat16_t, 1> const &b,
    Array<float, 1> const &c
  ) {
    d[0] = float(a[0]) * float(b[0]) + c[0];
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}
}
