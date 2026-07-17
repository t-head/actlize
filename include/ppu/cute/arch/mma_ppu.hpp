/*
 * Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
 * All rights reserved.
 *
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
 */

#pragma once

#include <cute/config.hpp>
#include <cute/arch/mma.hpp>

// Config
#if (defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 100))
#  define CUTE_ARCH_MMA_PPU0010_ENABLED

#if (__HGGC_ARCH__ <= 150)
#define CUTE_ARCH_MMA_B1_AND_PPU0010_ENABLED
#endif

#if (__HGGC_ARCH__ <= 150)
#define CUTE_ARCH_MMA_B1_XOR_PPU0010_ENABLED
#endif

#endif

namespace cute {

struct PPU_16x16x16_F32F16F16F32_TN
{
  using DRegisters = float[8];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = float[8];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      float         & d4, float         & d5, float         & d6, float         & d7,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1, uint32_t const& b2, uint32_t const& b3,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3,
      float const   & c4, float const   & c5, float const   & c6, float const   & c7)
  {
#ifdef __HGGC_ARCH__
#if 0
    uint32_t a[4] = {a0, a1, a2, a3};
    uint32_t b[4] = {b0, b1, b2, b3};
    float c[8] = {c0, c1, c2, c3, c4, c5, c6, c7};
    float *d[8] = {&d0, &d1, &d2, &d3, &d4, &d5, &d6, &d7};

    const uint16_t *frag_a = reinterpret_cast<const uint16_t *>(&a);
    const uint16_t *frag_b = reinterpret_cast<const uint16_t *>(&b);
    const float *frag_c = reinterpret_cast<const float *>(&c);
    int lane_idx = threadIdx.x % 32;
    unsigned mask = 0xffffffff;
    // two elem in each 8x8
    CUTE_NO_UNROLL
    for (int loop = 0; loop < 8; loop++) {
      float acc = 0;
      CUTE_NO_UNROLL
      for (int k = 0; k < 16; k++) {
        // threadIdx of elem a, will use two elem each time, right 8 k's thread is identical to left 8 k
        int thread_a = lane_idx - (lane_idx % 4) + ((k % 8) / 2);
        // (loop / 4) * 4 corresponds to top or down part, ((k / 8) * 2) corresponds to left or right part
        // k % 2 corresponds to which elem in each part, inner loop will use two elem in each thread
        int reg_a = ((loop / 4) * 4) + ((k / 8) * 2) + (k % 2);
        uint16_t cur_a = __shfl_sync(mask, frag_a[reg_a], thread_a);
        cutlass::half_t cur_a_half;
        memcpy(&cur_a_half, &cur_a, sizeof(uint16_t));
        // first two part is which col in matrix b, last part is which row
        // int thread_b = (lane_idx % 4) * 8 + (loop % 2) * 4 + (k % 8) / 2;
        int thread_b = (lane_idx % 4) * 4 + (loop % 2) * 16 + (k % 8) / 2;
        // first is hori part, second is vert part, third is index in cur part
        int reg_b = (((loop % 4) / 2) * 4) + (k / 8) * 2 + k % 2;
        uint16_t cur_b = __shfl_sync(mask, frag_b[reg_b], thread_b);
        cutlass::half_t cur_b_half;
        memcpy(&cur_b_half, &cur_b, sizeof(uint16_t));
        acc += float(cur_a_half * cur_b_half);
      }
      *d[loop] = acc + frag_c[loop];
    }


  #else
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32  {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "    
        "{%16,%17,%18,%19,%20,%21,%22,%23};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4), "=f"(d5), "=f"(d6), "=f"(d7)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3), "f"(c4), "f"(c5), "f"(c6), "f"(c7));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32  {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "    
        "{%16,%17,%18,%19,%20,%21,%22,%23};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4), "=f"(d5), "=f"(d6), "=f"(d7)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3), "f"(c4), "f"(c5), "f"(c6), "f"(c7));
#endif
  #endif
#endif
  }
};


struct PPU_16x16x16_F32BF16BF16F32_TN
{
  using DRegisters = float[8];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = float[8];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      float         & d4, float         & d5, float         & d6, float         & d7,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1, uint32_t const& b2, uint32_t const& b3,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3,
      float const   & c4, float const   & c5, float const   & c6, float const   & c7)
  {

#ifdef __HGGC_ARCH__
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.mma.sync.aligned.m16n16k16.row.col.f32.bf16.bf16.f32  {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "    
        "{%16,%17,%18,%19,%20,%21,%22,%23};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4), "=f"(d5), "=f"(d6), "=f"(d7)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3), "f"(c4), "f"(c5), "f"(c6), "f"(c7));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n16k16.row.col.f32.bf16.bf16.f32  {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "    
        "{%16,%17,%18,%19,%20,%21,%22,%23};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4), "=f"(d5), "=f"(d6), "=f"(d7)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3), "f"(c4), "f"(c5), "f"(c6), "f"(c7));
#endif
#else
      CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x16x16_F32BF16BF16F32_TN without device ARCH");
#endif
  }
};

struct PPU_16x16x16_F16F16F16F16_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1, uint32_t const& b2, uint32_t const& b3,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#ifdef __HGGC_ARCH__
    assert(0);
#endif
  }
};

struct PPU_16x16x8_F32TF32TF32F32_TN
{
  using DRegisters = float[8];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = float[8];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      float         & d4, float         & d5, float         & d6, float         & d7,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1, uint32_t const& b2, uint32_t const& b3,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3,
      float const   & c4, float const   & c5, float const   & c6, float const   & c7)
  {
#ifdef __HGGC_ARCH__
    uint32_t a[4] = {a0, a1, a2, a3};
    uint32_t b[4] = {b0, b1, b2, b3};
    float c[8] = {c0, c1, c2, c3, c4, c5, c6, c7};
    float *d[8] = {&d0, &d1, &d2, &d3, &d4, &d5, &d6, &d7};

    const float *frag_a = reinterpret_cast<const float *>(&a);
    const float *frag_b = reinterpret_cast<const float *>(&b);
    const float *frag_c = reinterpret_cast<const float *>(&c);
    int lane_idx = threadIdx.x % 32;
    unsigned mask = 0xffffffff;
    // two elem in each 8x8
    CUTE_NO_UNROLL
    for (int loop = 0; loop < 8; loop++) {
      float acc = 0;
      CUTE_NO_UNROLL
      for (int k = 0; k < 8; k++) {
        // threadIdx of elem a, will use two elem each time, right 8 k's thread is identical to left 8 k
        int thread_a = lane_idx - (lane_idx % 4) + (k % 4);
        // (loop / 4) * 4 corresponds to top or down part, ((k / 8) * 2) corresponds to left or right part
        // k % 2 corresponds to which elem in each part, inner loop will use two elem in each thread
        int reg_a = ((loop / 4) * 2) + (k / 4);
        float cur_a = __shfl_sync(mask, frag_a[reg_a], thread_a);
        cutlass::tfloat32_t cur_a_tf32;
        memcpy(&cur_a_tf32, &cur_a, sizeof(float));
        // first two part is which col in matrix b, last part is which row
        int thread_b = (lane_idx % 4) * 4 + (loop % 2) * 16 + (k % 4);
        // first is hori part, second is vert part
        int reg_b = ((loop % 4) / 2) * 2 + (k / 4);
        float cur_b = __shfl_sync(mask, frag_b[reg_b], thread_b);
        cutlass::tfloat32_t cur_b_tf32;
        memcpy(&cur_b_tf32, &cur_b, sizeof(float));
        acc += float(cur_a_tf32 * cur_b_tf32);
      }
      *d[loop] = acc + frag_c[loop];
    }

#endif
  }
};

struct PPU_16x16x32_S32S8S8S32_TN
{
  using DRegisters = uint32_t[8];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = uint32_t[8];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t      & d4, uint32_t      & d5, uint32_t      & d6, uint32_t      & d7,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1, uint32_t const& b2, uint32_t const& b3,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3,
      uint32_t const& c4, uint32_t const& c5, uint32_t const& c6, uint32_t const& c7)
  {

#ifdef __HGGC_ARCH__
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.mma.sync.aligned.m16n16k32.row.col.s32.s8.s8.s32  {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "    
        "{%16,%17,%18,%19,%20,%21,%22,%23};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4), "=f"(d5), "=f"(d6), "=f"(d7)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3), "f"(c4), "f"(c5), "f"(c6), "f"(c7));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n16k32.row.col.s32.s8.s8.s32  {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "    
        "{%16,%17,%18,%19,%20,%21,%22,%23};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4), "=f"(d5), "=f"(d6), "=f"(d7)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3), "f"(c4), "f"(c5), "f"(c6), "f"(c7));
#endif
#else
      CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x16x32_S32S8S8S32_TN without device ARCH");
#endif
  }
};

struct PPU_16x16x32_S32S8U8S32_TN
{
  using DRegisters = uint32_t[8];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = uint32_t[8];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t      & d4, uint32_t      & d5, uint32_t      & d6, uint32_t      & d7,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1, uint32_t const& b2, uint32_t const& b3,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3,
      uint32_t const& c4, uint32_t const& c5, uint32_t const& c6, uint32_t const& c7)
  {

#ifdef __HGGC_ARCH__
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.mma.sync.aligned.m16n16k32.row.col.s32.s8.u8.s32  {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "    
        "{%16,%17,%18,%19,%20,%21,%22,%23};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4), "=f"(d5), "=f"(d6), "=f"(d7)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3), "f"(c4), "f"(c5), "f"(c6), "f"(c7));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n16k32.row.col.s32.s8.u8.s32  {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "    
        "{%16,%17,%18,%19,%20,%21,%22,%23};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4), "=f"(d5), "=f"(d6), "=f"(d7)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3), "f"(c4), "f"(c5), "f"(c6), "f"(c7));
#endif
#else
      CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x16x32_S32S8U8S32_TN without device ARCH");
#endif
  }
};

struct PPU_16x16x32_S32U8S8S32_TN
{
  using DRegisters = uint32_t[8];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = uint32_t[8];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t      & d4, uint32_t      & d5, uint32_t      & d6, uint32_t      & d7,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1, uint32_t const& b2, uint32_t const& b3,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3,
      uint32_t const& c4, uint32_t const& c5, uint32_t const& c6, uint32_t const& c7)
  {

#ifdef __HGGC_ARCH__
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.mma.sync.aligned.m16n16k32.row.col.s32.u8.s8.s32  {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "    
        "{%16,%17,%18,%19,%20,%21,%22,%23};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4), "=f"(d5), "=f"(d6), "=f"(d7)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3), "f"(c4), "f"(c5), "f"(c6), "f"(c7));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n16k32.row.col.s32.u8.s8.s32  {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "    
        "{%16,%17,%18,%19,%20,%21,%22,%23};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4), "=f"(d5), "=f"(d6), "=f"(d7)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3), "f"(c4), "f"(c5), "f"(c6), "f"(c7));
#endif
#else
      CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x16x32_S32U8S8S32_TN without device ARCH");
#endif
  }
};

struct PPU_16x16x32_S32U8U8S32_TN
{
  using DRegisters = uint32_t[8];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = uint32_t[8];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t      & d4, uint32_t      & d5, uint32_t      & d6, uint32_t      & d7,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1, uint32_t const& b2, uint32_t const& b3,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3,
      uint32_t const& c4, uint32_t const& c5, uint32_t const& c6, uint32_t const& c7)
  {

#ifdef __HGGC_ARCH__
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.mma.sync.aligned.m16n16k32.row.col.s32.u8.u8.s32  {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "    
        "{%16,%17,%18,%19,%20,%21,%22,%23};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4), "=f"(d5), "=f"(d6), "=f"(d7)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3), "f"(c4), "f"(c5), "f"(c6), "f"(c7));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n16k32.row.col.s32.u8.u8.s32  {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "    
        "{%16,%17,%18,%19,%20,%21,%22,%23};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4), "=f"(d5), "=f"(d6), "=f"(d7)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3), "f"(c4), "f"(c5), "f"(c6), "f"(c7));
#endif
#else
      CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x16x32_S32U8U8S32_TN without device ARCH");
#endif
  }
};

// PPU1.0 81616 mma
struct PPU_8x16x16_F32F16F16F32_TN
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[4];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0,               uint32_t const& a1,
      uint32_t const& b0, uint32_t const& b1, uint32_t const& b2, uint32_t const& b3,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#ifdef __HGGC_ARCH__
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.mma.sync.aligned.m8n16k16.row.col.f32.f16.f16.f32  {%0,%1,%2,%3}, {%4,%5}, {%6,%7,%8,%9}, "    
        "{%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#endif
#endif
  }
};


struct PPU_8x16x16_F32BF16BF16F32_TN
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[4];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0,               uint32_t const& a1,
      uint32_t const& b0, uint32_t const& b1, uint32_t const& b2, uint32_t const& b3,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {

#ifdef __HGGC_ARCH__
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.mma.sync.aligned.m8n16k16.row.col.f32.bf16.bf16.f32  {%0,%1,%2,%3}, {%4,%5}, {%6,%7,%8,%9}, "    
        "{%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#endif
#else
      CUTE_RUNTIME_ASSERT("Attempting to use PPU_8x16x16_F32BF16BF16F32_TN without device ARCH");
#endif
  }
};
} // namespace cute
