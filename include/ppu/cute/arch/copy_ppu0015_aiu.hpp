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

#include <cute/arch/copy.hpp>
// definition of AiuDesc
#include "ppu/cute/arch/copy_ppu0010_aiu.hpp"

namespace cute
{

// struct AiuDesc {
//   const uint8_t* gmem_ptr;
//   int dim_h;
//   int dim_w;
//   // stride in GemmArgs is int now
//   int stride_w;

//   CUTE_HOST_DEVICE void
//   print() {
//     printf("dim: %d, %d, stride: %d, ptr: %p\n", dim_h, dim_w, stride_w, gmem_ptr);
//   }
// };

template <class NumBitsPerTMA, int ElemSize, int CUBE_H, int CUBE_W>
struct PPU0015_AIU_LOAD_IMPL;

template <class NumBitsPerTMA, int CUBE_H, int CUBE_W>
struct PPU0015_AIU_LOAD_IMPL<NumBitsPerTMA, 8, CUBE_H, CUBE_W> {

  static constexpr int swzl_mode = CUBE_W * 1 == 128 ? 0 : 1;

  CUTE_HOST_DEVICE void operator()(void *smem_ptr, const void* gmem_ptr, AiuDesc desc, int coord_w, int coord_h, int coord_n) {
    #if defined(__HGGC_ARCH__) && ACOMPUTE_VERSION >= 10500
      uint64_t tensor_stride_w = desc.stride_w * 1;
      uint64_t tensor_stride_n = tensor_stride_w * desc.dim_h;
      asm volatile(
        "ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.padz.swzl.b8"
        "[%0], [%1], {%2, %3, %4}, {%5, %6, %7}, {%8, %9}, {%10, %11, %12}, %13;\n"
        :: "r"(smem_ptr), "l"(gmem_ptr),
          "r"(desc.dim_w), "r"(desc.dim_h), "r"(1),
          "r"(CUBE_W), "r"(CUBE_H), "r"(1),
          "l"(tensor_stride_w), "l"(tensor_stride_n),
          "r"(coord_w), "r"(coord_h), "r"(0),
          "r"(swzl_mode)
      );

      // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
      //   printf("==========\n");
      //   printf("smem_ptr: %ld\n", smem_ptr);
      //   printf("dim: %d, %d\n", desc.dim_h, desc.dim_w);
      //   printf("cube: %d, %d\n", CUBE_H, CUBE_W);
      //   printf("stride: %ld\n", tensor_stride_w);
      //   printf("coord: %d, %d\n", coord_h, coord_w);
      //   printf("swzl_mode: %d\n", swzl_mode);
      // }
    #else
        CUTE_RUNTIME_ASSERT("Support for AIU_LOAD has not been enabled for PPU0015_AIU_LOAD_IMPL");
    #endif
  }
};

template <class NumBitsPerTMA, int CUBE_H, int CUBE_W>
struct PPU0015_AIU_LOAD_IMPL<NumBitsPerTMA, 16, CUBE_H, CUBE_W> {

  static constexpr int swzl_mode = CUBE_W * 2 == 128 ? 0 : 1;

  CUTE_HOST_DEVICE void operator()(void *smem_ptr, const void* gmem_ptr, AiuDesc desc, int coord_w, int coord_h, int coord_n) {
    #if defined(__HGGC_ARCH__) && ACOMPUTE_VERSION >= 10500
      uint64_t tensor_stride_w = desc.stride_w * 2;
      uint64_t tensor_stride_n = tensor_stride_w * desc.dim_h;
      asm volatile(
        "ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.padz.swzl.b16"
        "[%0], [%1], {%2, %3, %4}, {%5, %6, %7}, {%8, %9}, {%10, %11, %12}, %13;\n"
        :: "r"(smem_ptr), "l"(gmem_ptr),
          "r"(desc.dim_w), "r"(desc.dim_h), "r"(1),
          "r"(CUBE_W), "r"(CUBE_H), "r"(1),
          "l"(tensor_stride_w), "l"(tensor_stride_n),
          "r"(coord_w), "r"(coord_h), "r"(0),
          "r"(swzl_mode)
      );
    #else
        CUTE_RUNTIME_ASSERT("Support for AIU_LOAD has not been enabled for PPU0015_AIU_LOAD_IMPL");
    #endif
  }
};

template <class NumBitsPerTMA, int CUBE_H, int CUBE_W>
struct PPU0015_AIU_LOAD_IMPL<NumBitsPerTMA, 32, CUBE_H, CUBE_W> {

  static constexpr int swzl_mode = CUBE_W * 4 == 128 ? 0 : 1;

  CUTE_HOST_DEVICE void operator()(void *smem_ptr, const void* gmem_ptr, AiuDesc desc, int coord_w, int coord_h, int coord_n) {
    #if defined(__HGGC_ARCH__) && ACOMPUTE_VERSION >= 10500
      uint64_t tensor_stride_w = desc.stride_w * 4;
      uint64_t tensor_stride_n = tensor_stride_w * desc.dim_h;
      asm volatile(
        "ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.padz.swzl.b32"
        "[%0], [%1], {%2, %3, %4}, {%5, %6, %7}, {%8, %9}, {%10, %11, %12}, %13;\n"
        :: "r"(smem_ptr), "l"(gmem_ptr),
          "r"(desc.dim_w), "r"(desc.dim_h), "r"(1),
          "r"(CUBE_W), "r"(CUBE_H), "r"(1),
          "l"(tensor_stride_w), "l"(tensor_stride_n),
          "r"(coord_w), "r"(coord_h), "r"(0),
          "r"(swzl_mode)
      );
    #else
        CUTE_RUNTIME_ASSERT("Support for AIU_LOAD has not been enabled for PPU0015_AIU_LOAD_IMPL");
    #endif
  }
};

template <class NumBitsPerTMA, typename Element, bool Trans, int CUBE_H, int CUBE_W>
struct PPU0015_AIU_LOAD;

template <class NumBitsPerTMA, typename Element, int CUBE_H, int CUBE_W>
struct PPU0015_AIU_LOAD<NumBitsPerTMA, Element, false, CUBE_H, CUBE_W>
{
  CUTE_HOST_DEVICE static void
  copy(void *smem_ptr, const void* gmem_ptr, AiuDesc desc, int coord_w, int coord_h, int coord_n = 0)
  {
    // swap coord_h and coord_w, to algin ppu1.0
    PPU0015_AIU_LOAD_IMPL<NumBitsPerTMA, sizeof_bits<Element>::value, CUBE_H, CUBE_W>()(smem_ptr, gmem_ptr, desc, coord_w, coord_h, coord_n);
  }
};

template <class NumBitsPerTMA, typename Element, int CUBE_H, int CUBE_W>
struct PPU0015_AIU_LOAD<NumBitsPerTMA, Element, true, CUBE_H, CUBE_W>
{
  CUTE_HOST_DEVICE static void
  copy(void *smem_ptr, const void* gmem_ptr, AiuDesc desc, int coord_h, int coord_w, int coord_n = 0)
  {
    // swap coord_h and coord_w, to algin ppu1.0.
    PPU0015_AIU_LOAD_IMPL<NumBitsPerTMA, sizeof_bits<Element>::value, CUBE_H, CUBE_W>()(smem_ptr, gmem_ptr, desc, coord_w, coord_h, coord_n);
  }
};

// only 32b use Swap32 to represent swap vregs, 16b use lbo/sbo
template <typename Element, bool Trans, bool Swap32 = false>
struct PPU0015_TSM_LD_SWZL_IMPL;

template <typename Element>
struct PPU0015_TSM_LD_SWZL_IMPL<Element, false> {
  CUTE_HOST_DEVICE void operator()(int *vreg, int tsm_add, int lbo, int sbo, int swzl_mode) {
#if defined(__HGGC_ARCH__) && ACOMPUTE_VERSION >= 10500
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 {%0, %1, %2, %3}, [%4], %5, %6, %7;"
      : "=r"(vreg[0]), "=r"(vreg[1]), "=r"(vreg[2]), "=r"(vreg[3])
      : "l"(tsm_add), "r"(lbo), "r"(sbo), "r"(swzl_mode)
    );
#endif
#else
    CUTE_RUNTIME_ASSERT("Support for TSM_LD_SWZL has not been enabled for Trans=false");
#endif
  }
};

template <>
struct PPU0015_TSM_LD_SWZL_IMPL<half_t, true> {
  CUTE_HOST_DEVICE void operator()(int *vreg, int tsm_add, int lbo, int sbo, int swzl_mode) {
#if defined(__HGGC_ARCH__) && ACOMPUTE_VERSION >= 10500
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16.trans {%0, %1, %2, %3}, [%4], %5, %6, %7;"
      : "=r"(vreg[0]), "=r"(vreg[1]), "=r"(vreg[2]), "=r"(vreg[3])
      : "l"(tsm_add), "r"(lbo), "r"(sbo), "r"(swzl_mode)
    );
#endif
#else
    CUTE_RUNTIME_ASSERT("Support for TSM_LD_SWZL has not been enabled for Trans=true");
#endif
  }
};

template <>
struct PPU0015_TSM_LD_SWZL_IMPL<bfloat16_t, true> {
  CUTE_HOST_DEVICE void operator()(int *vreg, int tsm_add, int lbo, int sbo, int swzl_mode) {
#if defined(__HGGC_ARCH__) && ACOMPUTE_VERSION >= 10500
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16.trans {%0, %1, %2, %3}, [%4], %5, %6, %7;"
      : "=r"(vreg[0]), "=r"(vreg[1]), "=r"(vreg[2]), "=r"(vreg[3])
      : "l"(tsm_add), "r"(lbo), "r"(sbo), "r"(swzl_mode)
    );
#endif
#else
    CUTE_RUNTIME_ASSERT("Support for TSM_LD_SWZL has not been enabled for Trans=true");
#endif
  }
};

template <>
struct PPU0015_TSM_LD_SWZL_IMPL<float, true> {
  CUTE_HOST_DEVICE void operator()(int *vreg, int tsm_add, int lbo, int sbo, int swzl_mode) {
#if defined(__HGGC_ARCH__) && ACOMPUTE_VERSION >= 10500
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x2.b32.trans {%0, %1, %2, %3}, [%4], %5;"
      : "=r"(vreg[0]), "=r"(vreg[1]), "=r"(vreg[2]), "=r"(vreg[3])
      : "l"(tsm_add), "r"(swzl_mode)
    );
#endif
#else
    CUTE_RUNTIME_ASSERT("Support for TSM_LD_SWZL has not been enabled for Trans=true");
#endif
  }
};

template <>
struct PPU0015_TSM_LD_SWZL_IMPL<float, true, true> {
  CUTE_HOST_DEVICE void operator()(int *vreg, int tsm_add, int lbo, int sbo, int swzl_mode) {
#if defined(__HGGC_ARCH__) && ACOMPUTE_VERSION >= 10500
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x2.b32.trans.diagswap {%0, %1, %2, %3}, [%4], %5;"
      : "=r"(vreg[0]), "=r"(vreg[1]), "=r"(vreg[2]), "=r"(vreg[3])
      : "l"(tsm_add), "r"(swzl_mode)
    );
#endif
#else
    CUTE_RUNTIME_ASSERT("Support for TSM_LD_SWZL has not been enabled for Trans=true");
#endif
  }
};

template <typename Element, int CUBE_H, int CUBE_W, int LBO, int SBO, bool trans, int InstNum, int SwzlMode, bool Swap32>
struct PPU0015_TSM_LD_SWZL_DISPATCH;

template <typename Element, int CUBE_H, int CUBE_W, int LBO, int SBO, int InstNum, int SwzlMode, bool Swap32>
struct PPU0015_TSM_LD_SWZL_DISPATCH<Element, CUBE_H, CUBE_W, LBO, SBO, false, InstNum, SwzlMode, Swap32> {
  CUTE_HOST_DEVICE void
  operator()(void *frag_ptr, void *smem_base, int coord_w, int coord_h, int stage, int cube_in_stage)
  {
#if defined(__HGGC_ARCH__) && ACOMPUTE_VERSION >= 10500
    Element *stage_base = reinterpret_cast<Element*>(smem_base);
    stage_base += CUBE_H * CUBE_W * (cube_in_stage + stage * InstNum);
    // static constexpr int alignment = 16 / sizeof(Element);
    stage_base += coord_h * CUBE_W + coord_w;
    int tsm_add = reinterpret_cast<uintptr_t>(stage_base) / 16;
    int *vreg = reinterpret_cast<int *>(frag_ptr);
    PPU0015_TSM_LD_SWZL_IMPL<Element, false, Swap32>()(vreg, tsm_add, LBO, SBO, SwzlMode);
#else
    CUTE_RUNTIME_ASSERT("Support for TSM_LD_SWZL has not been enabled for PPU0015_TSM_LD_SWZL_IMPL");
#endif
  }
};


template <typename Element, int CUBE_H, int CUBE_W, int LBO, int SBO, int InstNum, int SwzlMode, bool Swap32>
struct PPU0015_TSM_LD_SWZL_DISPATCH<Element, CUBE_H, CUBE_W, LBO, SBO, true, InstNum, SwzlMode, Swap32> {
  CUTE_HOST_DEVICE void
  operator()(void *frag_ptr, void *smem_base, int coord_h, int coord_w, int stage, int cube_in_stage)
  {
#if defined(__HGGC_ARCH__) && ACOMPUTE_VERSION >= 10500
    Element *stage_base = reinterpret_cast<Element*>(smem_base);
    stage_base += CUBE_H * CUBE_W * (cube_in_stage + stage * InstNum);
    stage_base += coord_h * CUBE_W + coord_w;
    int tsm_add = reinterpret_cast<uintptr_t>(stage_base) / 16;
    int *vreg = reinterpret_cast<int *>(frag_ptr);
    PPU0015_TSM_LD_SWZL_IMPL<Element, true, Swap32>()(vreg, tsm_add, LBO, SBO, SwzlMode);
#else
    CUTE_RUNTIME_ASSERT("Support for TSM_LD_SWZL has not been enabled for PPU0015_TSM_LD_SWZL_IMPL");
#endif
  }
};

template <typename Element, int CUBE_H, int CUBE_W, bool Swap, bool Trans, int InstNum>
struct PPU0015_TSM_LD_SWZL {
// struct PPU0015_TSM_LD_SWZL<Element, CUBE_H, CUBE_W, Swap, Trans, InstNum> {

  static constexpr int row_stride = 8 * CUBE_W * sizeof(Element) / 16;
  static constexpr int lbo = (Swap != Trans) ? row_stride : 1;
  static constexpr int sbo = (Swap != Trans) ? 1 : row_stride;
  static constexpr int swzl_mode = CUBE_W * sizeof(Element) == 128 ? 0 : 1;
  // 32 trans only use lbo to describe swap or not
  static constexpr bool Swap32 = Swap && Trans && sizeof(Element) == 4;

  // will not transfer cube_in_stage when only one inst
  CUTE_HOST_DEVICE static void
  copy(void *frag_ptr, void *smem_base, int coord_w, int coord_h, int cube_in_stage = 0, int stage = 0)
  {
    PPU0015_TSM_LD_SWZL_DISPATCH<Element, CUBE_H, CUBE_W, lbo, sbo, Trans, InstNum, swzl_mode, Swap32>()(
      frag_ptr, smem_base, coord_w, coord_h, stage, cube_in_stage
    );

    // if (threadIdx.x == 0) {
    //   printf("==========\n");
    //   // printf("tsm_add: %p, val: %f, vreg_val: %f\n", tmp, float(*tmp), float(*vreg_tmp));
    //   printf("coord_h: %d, coord_w: %d, cube_in_stage: %d, stage: %d\n", coord_h, coord_w, cube_in_stage, stage);
    //   printf("smem_base: %d, lbo: %d, sbo: %d, swzl_mode: %d\n", smem_base, lbo, sbo, swzl_mode);
    // }
  }
};

// CVT variant: pair-interleaved 16x64-cube layout for TSM group conflict elimination.
// Data is stored as cube pairs (cube 2i / 2i+1 interleaved); cube_in_stage addresses the pair-interleaved slot.
template <typename Element, int CUBE_H, int CUBE_W, int BlockH, int BlockW, bool Swap, bool Trans, int InstNum, bool Cvt, int OddTile>
struct PPU0015_TSM_LD_SWZL_CVT;

// Trans=false specialization
template <typename Element, int CUBE_H, int CUBE_W, int BlockH, int BlockW, bool Swap, int InstNum, bool CvtQ, int OddTile>
struct PPU0015_TSM_LD_SWZL_CVT<Element, CUBE_H, CUBE_W, BlockH, BlockW, Swap, false, InstNum, CvtQ, OddTile> {

  static_assert(sizeof(Element) == 2 && CUBE_H == 16 && CUBE_W == 64);
  static_assert(BlockH % CUBE_H == 0 && BlockW % (2*CUBE_W) == 0);

  static constexpr int row_stride = 8 * CUBE_W * sizeof(Element) / 16;
  static constexpr int col_stride = CUBE_H * CUBE_W * sizeof(Element) / 16;
  static constexpr int swzl_mode = 0;

  CUTE_HOST_DEVICE static void
  copy(void *frag_ptr, void *smem_base, int coord_w, int coord_h, int cube_in_stage = 0, int stage = 0)
  {
#if defined(__HGGC_ARCH__) && ACOMPUTE_VERSION >= 10500

    Element *stage_base = reinterpret_cast<Element*>(smem_base);
    int stg = cube_in_stage + stage * InstNum;
    stage_base += BlockH * CUBE_W * (stg & ~1);

    int lbo, sbo;
    if (stg == OddTile) {
      lbo = Swap ? row_stride : 1;
      sbo = Swap ? 1 : row_stride;
      stage_base += (stg & 1) * CUBE_H * CUBE_W;
      stage_base += coord_h * CUBE_W + coord_w;
    } else {
      if constexpr (CvtQ) {
        lbo = row_stride;
        sbo = col_stride;
        if constexpr (!Swap) { // Q
          lbo += - (stg & 1) | 1; // same as: stg %2 ? 1 : -1;
        } else { // K
          sbo += - (stg & 1) | 1;
        }
      } else {
        lbo = Swap ? row_stride : col_stride;
        sbo = Swap ? col_stride : row_stride;
      }
      stage_base += (stg & 1) << 3;
      stage_base += coord_h * BlockW + coord_w;
    }
    int tsm_add = reinterpret_cast<uintptr_t>(stage_base) / 16;
    int *vreg = reinterpret_cast<int *>(frag_ptr);

    PPU0015_TSM_LD_SWZL_IMPL<Element, false, false>()(vreg, tsm_add, lbo, sbo, swzl_mode);
#else
    CUTE_RUNTIME_ASSERT("Support for TSM_LD_SWZL has not been enabled for PPU0015_TSM_LD_SWZL_CVT");
#endif
  }
};

// Trans=false and。OddTile = -1 specialization
template <typename Element, int CUBE_H, int CUBE_W, int BlockH, int BlockW, bool Swap, int InstNum, bool CvtQ>
struct PPU0015_TSM_LD_SWZL_CVT<Element, CUBE_H, CUBE_W, BlockH, BlockW, Swap, false, InstNum, CvtQ, -1> {

  static_assert(sizeof(Element) == 2 && CUBE_H == 16 && CUBE_W == 64);
  static_assert(BlockH % CUBE_H == 0 && BlockW % (2*CUBE_W) == 0);

  static constexpr int row_stride = 8 * CUBE_W * sizeof(Element) / 16;
  static constexpr int col_stride = CUBE_H * CUBE_W * sizeof(Element) / 16;
  static constexpr int swzl_mode = 0;

  CUTE_HOST_DEVICE static void
  copy(void *frag_ptr, void *smem_base, int coord_w, int coord_h, int cube_in_stage = 0, int stage = 0)
  {
#if defined(__HGGC_ARCH__) && ACOMPUTE_VERSION >= 10500

    Element *stage_base = reinterpret_cast<Element*>(smem_base);
    int stg = cube_in_stage + stage * InstNum;
    stage_base += BlockH * CUBE_W * (stg & ~1);
    int lbo, sbo;
    if constexpr (CvtQ) {
      lbo = row_stride;
      sbo = col_stride;
      if constexpr (!Swap) { // Q
        lbo += - (stg & 1) | 1; // same as: stg %2 ? 1 : -1;
      } else { // K
        sbo += - (stg & 1) | 1;
      }
      stage_base += (stg & 1) << 3;
      stage_base += coord_h * BlockW + coord_w;
    } else {
      lbo = Swap ? row_stride : col_stride;
      sbo = Swap ? col_stride : row_stride;
      stage_base += (stg & 1) << 3;
      stage_base += coord_h * BlockW + coord_w;
    }

    int tsm_add = reinterpret_cast<uintptr_t>(stage_base) / 16;
    int *vreg = reinterpret_cast<int *>(frag_ptr);

    PPU0015_TSM_LD_SWZL_IMPL<Element, false, false>()(vreg, tsm_add, lbo, sbo, swzl_mode);
#else
    CUTE_RUNTIME_ASSERT("Support for TSM_LD_SWZL has not been enabled for PPU0015_TSM_LD_SWZL_CVT");
#endif
  }
};

// Trans=true specialization
template <typename Element, int CUBE_H, int CUBE_W, int BlockH, int BlockW, bool Swap, int InstNum, bool Cvt, int OddTile>
struct PPU0015_TSM_LD_SWZL_CVT<Element, CUBE_H, CUBE_W, BlockH, BlockW, Swap, true, InstNum, Cvt, OddTile> {

  static_assert(sizeof(Element) == 2 && CUBE_H == 16 && CUBE_W == 64);
  static_assert(BlockH % CUBE_H == 0);
  static_assert(Swap);
  static_assert(OddTile == -1);

  static constexpr int row_stride = 8 * CUBE_W * sizeof(Element) / 16;
  static constexpr int LBO = (Cvt) ? (CUBE_H * CUBE_W >> 3) : 1;
  static constexpr int SBO = row_stride;
  static constexpr int swzl_mode = 0;

  CUTE_HOST_DEVICE static void
  copy(void *frag_ptr, void *smem_base, int coord_h, int coord_w, int cube_in_stage = 0, int stage = 0)
  {
#if defined(__HGGC_ARCH__) && ACOMPUTE_VERSION >= 10500

    Element *stage_base = reinterpret_cast<Element*>(smem_base);
    int stg = cube_in_stage + stage * InstNum;

    if constexpr (Cvt) {
      stage_base += BlockH * CUBE_W * (stg & ~1);
      stage_base += (stg & 1) << 3;
    } else {
      stage_base += BlockH * CUBE_W * stg;
    }
    stage_base += coord_h * BlockW + coord_w;

    int tsm_add = reinterpret_cast<uintptr_t>(stage_base) / 16;
    int *vreg = reinterpret_cast<int *>(frag_ptr);

    PPU0015_TSM_LD_SWZL_IMPL<Element, true, false>()(vreg, tsm_add, LBO, SBO, swzl_mode);
#else
    CUTE_RUNTIME_ASSERT("Support for TSM_LD_SWZL has not been enabled for PPU0015_TSM_LD_SWZL_CVT");
#endif
  }
};

} // namespace cute
