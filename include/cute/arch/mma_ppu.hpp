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

#include <cute/config.hpp>
#include <cute/arch/mma.hpp>

// Config
#if (defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 100))
#  define CUTE_ARCH_MMA_PPU0010_ENABLED
#endif

namespace cute
{

struct PPU_DP4A
{
  using DRegisters = int32_t[1];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = int32_t[1];

  // Register asm fma
  CUTE_HOST_DEVICE static void
  fma(int32_t& d, uint32_t const& a, uint32_t const& b, int32_t const& c)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
    asm volatile("ppu.dp4a.s32.s32 %0, %1, %2, %3;"
                 : "=r"(d)
                 : "r"(a), "r"(b), "r"(c));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_DP4A without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};


} // namespace cute



// Config
#define CUTE_ARCH_MMA_PPU0010_SUPPORTED
#if (defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 100))
#  define CUTE_ARCH_MMA_PPU0010_ENABLED
#endif

namespace cute
{

//
// PPU MMA 884 F16F16F16
//

struct PPU_8x8x4_F16F16F16F16_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  // Register asm fma
  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
assert(0);
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_8x8x4_F16F16F16F16_TN without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct PPU_8x8x4_F16F16F16F16_NT
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  // Register asm fma
  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
assert(0);
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_8x8x4_F16F16F16F16_NT without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct PPU_8x8x4_F16F16F16F16_NN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  // Register asm fma
  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
assert(0);
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_8x8x4_F16F16F16F16_NN without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct PPU_8x8x4_F16F16F16F16_TT
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  // Register asm fma
  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
assert(0);
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_8x8x4_F16F16F16F16_TT without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

//
// PPU0010 MMA 884 F16F16F32
//

struct PPU_8x8x4_F32F16F16F32_TN
{
  using DRegisters = float[8];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = float[8];

  // Register asm fma
  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float      & d2, float      & d3,
      float         & d4, float         & d5, float      & d6, float      & d7,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0, uint32_t const& b1,
      float    const& c0, float    const& c1, float const& c2, float const& c3,
      float    const& c4, float    const& c5, float const& c6, float const& c7)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
assert(0);
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_8x8x4_F32F16F16F32_TN without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct PPU_8x8x4_F32F16F16F32_NT
{
  using DRegisters = float[8];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = float[8];

  // Register asm fma
  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float      & d2, float      & d3,
      float         & d4, float         & d5, float      & d6, float      & d7,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0, uint32_t const& b1,
      float    const& c0, float    const& c1, float const& c2, float const& c3,
      float    const& c4, float    const& c5, float const& c6, float const& c7)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
assert(0);
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_8x8x4_F32F16F16F32_NT without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct PPU_8x8x4_F32F16F16F32_NN
{
  using DRegisters = float[8];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = float[8];

  // Register asm fma
  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float      & d2, float      & d3,
      float         & d4, float         & d5, float      & d6, float      & d7,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0, uint32_t const& b1,
      float    const& c0, float    const& c1, float const& c2, float const& c3,
      float    const& c4, float    const& c5, float const& c6, float const& c7)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
assert(0);
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_8x8x4_F32F16F16F32_NN without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct PPU_8x8x4_F32F16F16F32_TT
{
  using DRegisters = float[8];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = float[8];

  // Register asm fma
  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float      & d2, float      & d3,
      float         & d4, float         & d5, float      & d6, float      & d7,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0, uint32_t const& b1,
      float    const& c0, float    const& c1, float const& c2, float const& c3,
      float    const& c4, float    const& c5, float const& c6, float const& c7)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
assert(0);
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_8x8x4_F32F16F16F32_TT without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }

};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // end namespace cute



// Config
#define CUTE_ARCH_MMA_PPU0010_SUPPORTED
#if (defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 100))
#  define CUTE_ARCH_MMA_PPU0010_ENABLED
#endif

namespace cute
{

//
// PPU MMA 1688 F16F16F32
//

struct PPU_16x8x8_F32F16F16F32_TN
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = float[4];

  // Register asm fma
  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float      & d2, float      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      float    const& c0, float    const& c1, float const& c2, float const& c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"    
                 "{%0, %1, %2, %3},"
                 "{%4, %5},"
                 "{%6},"
                 "{%7, %8, %9, %10};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        :  "r"(a0),  "r"(a1),
           "r"(b0),
           "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"    
                 "{%0, %1, %2, %3},"
                 "{%4, %5},"
                 "{%6},"
                 "{%7, %8, %9, %10};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        :  "r"(a0),  "r"(a1),
           "r"(b0),
           "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x8_F32F16F16F32_TN without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

//
// PPU0010 MMA 8816 S8S8S32
//

struct PPU_8x8x16_S32S8S8S32_TN
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  // Register asm fma
  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
    assert(0);
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32"    
                 "{%0, %1},"
                 "{%2},"
                 "{%3},"
                 "{%4, %5};\n"
        : "=r"(d0), "=r"(d1)
        :  "r"(a0),
           "r"(b0),
           "r"(c0),  "r"(c1));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_8x8x16_S32S8S8S32_TN without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // end namespace cute



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

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x8 TN
struct PPU_16x8x8_F16F16F16F16_TN
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "    
      "{%0, %1},"
      "{%2, %3},"
      "{%4},"
      "{%5, %6};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "    
      "{%0, %1},"
      "{%2, %3},"
      "{%4},"
      "{%5, %6};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x8_F16F16F16F16_TN without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x16 TN
struct PPU_16x8x16_F16F16F16F16_TN
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "    
      "{%0,  %1},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  %9};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "    
      "{%0,  %1},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  %9};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x16_F16F16F16F16_TN without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x8 TN
struct PPU_16x8x8_F32F16F16F32_TNV2
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x8_F32F16F16F32_TNV2 without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x16 TN
struct PPU_16x8x16_F32F16F16F32_TN
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x16_F32F16F16F32_TN without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x8 TN
struct PPU_16x8x8_F32BF16BF16F32_TN
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x8_F32BF16BF16F32_TN without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x16 TN
struct PPU_16x8x16_F32BF16BF16F32_TN
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x16_F32BF16BF16F32_TN without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x4 TN
struct PPU_16x8x4_F32TF32TF32F32_TN
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x4_F32TF32TF32F32_TN without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x8 TN
struct PPU_16x8x8_F32TF32TF32F32_TN
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x8_F32TF32TF32F32_TN without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 8x8x16 TN
struct PPU_8x8x16_S32S8S8S32_TNV2
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
    assert(0);
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "    
      "{%0, %1},"
      "{%2},"
      "{%3},"
      "{%4, %5};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),
         "r"(b0),
         "r"(c0),  "r"(c1));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_8x8x16_S32S8S8S32_TNV2 without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 8x8x16 TN
struct PPU_8x8x16_S32S8S8S32_TN_SATURATE
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
    assert(0);
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32.satfinite "    
      "{%0, %1},"
      "{%2},"
      "{%3},"
      "{%4, %5};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),
         "r"(b0),
         "r"(c0),  "r"(c1));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_8x8x16_S32S8S8S32_TN_SATURATE without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x16 TN
struct PPU_16x8x16_S32S8S8S32_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x16_S32S8S8S32_TN without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x16 TN
struct PPU_16x8x16_S32S8S8S32_TN_SATURATE
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32.satfinite "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32.satfinite "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x16_S32S8S8S32_TN_SATURATE without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN
struct PPU_16x8x32_S32S8S8S32_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x32_S32S8S8S32_TN without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN
struct PPU_16x8x32_S32S8S8S32_TN_SATURATE
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x32_S32S8S8S32_TN_SATURATE without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 8x8x16 TN
struct PPU_8x8x16_S32S8U8S32_TN
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
    assert(0);
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m8n8k16.row.col.s32.s8.u8.s32 "    
      "{%0, %1},"
      "{%2},"
      "{%3},"
      "{%4, %5};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),
         "r"(b0),
         "r"(c0),  "r"(c1));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_8x8x16_S32S8U8S32_TN without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 8x8x16 TN
struct PPU_8x8x16_S32S8U8S32_TN_SATURATE
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
    assert(0);
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m8n8k16.row.col.s32.s8.u8.s32.satfinite "    
      "{%0, %1},"
      "{%2},"
      "{%3},"
      "{%4, %5};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),
         "r"(b0),
         "r"(c0),  "r"(c1));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_8x8x16_S32S8U8S32_TN_SATURATE without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x16 TN
struct PPU_16x8x16_S32S8U8S32_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k16.row.col.s32.s8.u8.s32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k16.row.col.s32.s8.u8.s32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x16_S32S8U8S32_TN without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x16 TN
struct PPU_16x8x16_S32S8U8S32_TN_SATURATE
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k16.row.col.s32.s8.u8.s32.satfinite "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k16.row.col.s32.s8.u8.s32.satfinite "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x16_S32S8U8S32_TN_SATURATE without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN
struct PPU_16x8x32_S32S8U8S32_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x32_S32S8U8S32_TN without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN
struct PPU_16x8x32_S32S8U8S32_TN_SATURATE
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32.satfinite "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32.satfinite "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x32_S32S8U8S32_TN_SATURATE without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 8x8x16 TN
struct PPU_8x8x16_S32U8S8S32_TN
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
    assert(0);
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m8n8k16.row.col.s32.u8.s8.s32 "    
      "{%0, %1},"
      "{%2},"
      "{%3},"
      "{%4, %5};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),
         "r"(b0),
         "r"(c0),  "r"(c1));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_8x8x16_S32U8S8S32_TN without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 8x8x16 TN
struct PPU_8x8x16_S32U8S8S32_TN_SATURATE
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
    assert(0);
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m8n8k16.row.col.s32.u8.s8.s32.satfinite "    
      "{%0, %1},"
      "{%2},"
      "{%3},"
      "{%4, %5};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),
         "r"(b0),
         "r"(c0),  "r"(c1));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_8x8x16_S32U8S8S32_TN_SATURATE without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x16 TN
struct PPU_16x8x16_S32U8S8S32_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k16.row.col.s32.u8.s8.s32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k16.row.col.s32.u8.s8.s32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x16_S32U8S8S32_TN without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x16 TN
struct PPU_16x8x16_S32U8S8S32_TN_SATURATE
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k16.row.col.s32.u8.s8.s32.satfinite "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k16.row.col.s32.u8.s8.s32.satfinite "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x16_S32U8S8S32_TN_SATURATE without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN
struct PPU_16x8x32_S32U8S8S32_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x32_S32U8S8S32_TN without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN
struct PPU_16x8x32_S32U8S8S32_TN_SATURATE
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32.satfinite "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32.satfinite "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x32_S32U8S8S32_TN_SATURATE without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 8x8x16 TN
struct PPU_8x8x16_S32U8U8S32_TN
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
    assert(0);
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m8n8k16.row.col.s32.u8.u8.s32 "    
      "{%0, %1},"
      "{%2},"
      "{%3},"
      "{%4, %5};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),
         "r"(b0),
         "r"(c0),  "r"(c1));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_8x8x16_S32U8U8S32_TN without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 8x8x16 TN
struct PPU_8x8x16_S32U8U8S32_TN_SATURATE
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
    assert(0);
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m8n8k16.row.col.s32.u8.u8.s32.satfinite "    
      "{%0, %1},"
      "{%2},"
      "{%3},"
      "{%4, %5};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),
         "r"(b0),
         "r"(c0),  "r"(c1));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_8x8x16_S32U8U8S32_TN_SATURATE without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x16 TN
struct PPU_16x8x16_S32U8U8S32_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x16_S32U8U8S32_TN without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x16 TN
struct PPU_16x8x16_S32U8U8S32_TN_SATURATE
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32.satfinite "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32.satfinite "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x16_S32U8U8S32_TN_SATURATE without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN
struct PPU_16x8x32_S32U8U8S32_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x32_S32U8U8S32_TN without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN
struct PPU_16x8x32_S32U8U8S32_TN_SATURATE
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32.satfinite "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32.satfinite "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x32_S32U8U8S32_TN_SATURATE without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 8x8x32 TN
struct PPU_8x8x32_S32S4S4S32_TN
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
    assert(0);
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_8x8x32_S32S4S4S32_TN without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 8x8x32 TN
struct PPU_8x8x32_S32S4S4S32_TN_SATURATE
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
    assert(0);
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_8x8x32_S32S4S4S32_TN_SATURATE without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN
struct PPU_16x8x32_S32S4S4S32_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x32_S32S4S4S32_TN without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN
struct PPU_16x8x32_S32S4S4S32_TN_SATURATE
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32.satfinite "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32.satfinite "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x32_S32S4S4S32_TN_SATURATE without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x64 TN
struct PPU_16x8x64_S32S4S4S32_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x64_S32S4S4S32_TN without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x64 TN
struct PPU_16x8x64_S32S4S4S32_TN_SATURATE
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32.satfinite "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32.satfinite "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x64_S32S4S4S32_TN_SATURATE without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 8x8x32 TN
struct PPU_8x8x32_S32S4U4S32_TN
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
    assert(0);
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_8x8x32_S32S4U4S32_TN without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 8x8x32 TN
struct PPU_8x8x32_S32S4U4S32_TN_SATURATE
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
    assert(0);
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_8x8x32_S32S4U4S32_TN_SATURATE without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN
struct PPU_16x8x32_S32S4U4S32_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k32.row.col.s32.s4.u4.s32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k32.row.col.s32.s4.u4.s32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x32_S32S4U4S32_TN without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN
struct PPU_16x8x32_S32S4U4S32_TN_SATURATE
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k32.row.col.s32.s4.u4.s32.satfinite "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k32.row.col.s32.s4.u4.s32.satfinite "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x32_S32S4U4S32_TN_SATURATE without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x64 TN
struct PPU_16x8x64_S32S4U4S32_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k64.row.col.s32.s4.u4.s32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k64.row.col.s32.s4.u4.s32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x64_S32S4U4S32_TN without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x64 TN
struct PPU_16x8x64_S32S4U4S32_TN_SATURATE
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k64.row.col.s32.s4.u4.s32.satfinite "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k64.row.col.s32.s4.u4.s32.satfinite "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x64_S32S4U4S32_TN_SATURATE without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 8x8x32 TN
struct PPU_8x8x32_S32U4S4S32_TN
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
    assert(0);
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_8x8x32_S32U4S4S32_TN without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 8x8x32 TN
struct PPU_8x8x32_S32U4S4S32_TN_SATURATE
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
    assert(0);
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_8x8x32_S32U4S4S32_TN_SATURATE without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN
struct PPU_16x8x32_S32U4S4S32_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k32.row.col.s32.u4.s4.s32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k32.row.col.s32.u4.s4.s32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x32_S32U4S4S32_TN without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN
struct PPU_16x8x32_S32U4S4S32_TN_SATURATE
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k32.row.col.s32.u4.s4.s32.satfinite "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k32.row.col.s32.u4.s4.s32.satfinite "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x32_S32U4S4S32_TN_SATURATE without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x64 TN
struct PPU_16x8x64_S32U4S4S32_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x64_S32U4S4S32_TN without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x64 TN
struct PPU_16x8x64_S32U4S4S32_TN_SATURATE
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32.satfinite "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32.satfinite "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x64_S32U4S4S32_TN_SATURATE without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 8x8x32 TN
struct PPU_8x8x32_S32U4U4S32_TN
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
    assert(0);
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_8x8x32_S32U4U4S32_TN without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 8x8x32 TN
struct PPU_8x8x32_S32U4U4S32_TN_SATURATE
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
    assert(0);
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_8x8x32_S32U4U4S32_TN_SATURATE without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN
struct PPU_16x8x32_S32U4U4S32_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k32.row.col.s32.u4.u4.s32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k32.row.col.s32.u4.u4.s32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x32_S32U4U4S32_TN without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN
struct PPU_16x8x32_S32U4U4S32_TN_SATURATE
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k32.row.col.s32.u4.u4.s32.satfinite "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k32.row.col.s32.u4.u4.s32.satfinite "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x32_S32U4U4S32_TN_SATURATE without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x64 TN
struct PPU_16x8x64_S32U4U4S32_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k64.row.col.s32.u4.u4.s32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k64.row.col.s32.u4.u4.s32 "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x64_S32U4U4S32_TN without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x64 TN
struct PPU_16x8x64_S32U4U4S32_TN_SATURATE
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k64.row.col.s32.u4.u4.s32.satfinite "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k64.row.col.s32.u4.u4.s32.satfinite "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x64_S32U4U4S32_TN_SATURATE without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 8x8x128 TN
struct PPU_8x8x128_S32U1U1S32_TN_XORPOPC
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_B1_XOR_PPU0010_ENABLED)
    assert(0);
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_8x8x128_S32U1U1S32_TN_XORPOPC without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x128 TN
struct PPU_16x8x128_S32U1U1S32_TN_XORPOPC
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_B1_XOR_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.xor.popc "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.xor.popc "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x128_S32U1U1S32_TN_XORPOPC without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x256 TN
struct PPU_16x8x256_S32U1U1S32_TN_XORPOPC
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_B1_XOR_PPU0010_ENABLED)
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ex.mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc "    
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#endif
#else
    CUTE_RUNTIME_ASSERT("Attempting to use PPU_16x8x256_S32U1U1S32_TN_XORPOPC without CUTE_ARCH_MMA_PPU0010_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////


} // namespace cute

