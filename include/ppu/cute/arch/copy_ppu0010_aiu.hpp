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

#include "ppu/cute/arch/copy_aiu_base.hpp"

#define DEBUG_PRINT 0
#if DEBUG_PRINT
#include <cute/util/debug.hpp>
#endif


namespace cute
{

struct AiuDesc {
  const uint8_t* gmem_ptr;  /*this ptr is invalid when transfer mix tensor to aiu*/
  int dim_h;
  int dim_w;
  int cube_h = 0; /* only used in ppu 10000*/
  int cube_w = 0; /* only used in ppu 10000*/
  int offset_w = 0; /* only used in ppu 10000*/
  int stride_w = 0; /* only used in ppu 10500*/

  template<typename Element, bool Trans, int Block_MN, int Block_K, typename Stride>
  CUTE_HOST_DEVICE
  void init(const uint8_t* gmem_ptr_, int MN, int K, Stride stride) {

    gmem_ptr = gmem_ptr_;

#if ACOMPUTE_VERSION == 10000
    static constexpr int BlockContSize = Block_K * sizeof_bits<Element>::value / 8;
    static constexpr int AiuContByteSize = BlockContSize > 128 ? 128 : BlockContSize;
    static constexpr int AiuContElemSize = AiuContByteSize / sizeof_bits<Element>::value * 8;

    if (!Trans) {
      dim_h = MN;
      dim_w = get<0>(stride);
      cube_h = Block_MN;
      cube_w = AiuContElemSize;
      offset_w = 0;
    } else {
      int block_cont_size = Block_MN * sizeof(Element);
      int aiu_cont_elem_size = cute::min(128, block_cont_size) / sizeof(Element);

      dim_h = K;
      dim_w = get<1>(stride);
      cube_h = Block_K;
      cube_w = aiu_cont_elem_size;
      offset_w = 0;
    }
    if constexpr (sizeof_bits<Element>::value == 4) {
      dim_w /= 2;
      cube_w /= 2;
    }
#else
    if (!Trans) {
      dim_h = MN;
      dim_w = K;
      stride_w = get<0>(stride);
    } else {
      dim_h = K;
      dim_w = MN;
      stride_w = get<1>(stride);
    }
#endif
  }

  // for 1.5fa usage, transfer no_cont_dim/cont_dim/ld directly
  CUTE_HOST_DEVICE
  void init(const uint8_t* gmem_ptr_, int h, int w, int stride) {
    gmem_ptr = gmem_ptr_;
    dim_h = h;
    dim_w = w;
    stride_w = stride; /* only used in ppu 10500*/
  }


#if DEBUG_PRINT
  CUTE_HOST_DEVICE void
  print(void *smem_ptr, const void* gmem_ptr, int coord_w, int coord_h, int coord_n) {
    if (cute::thread0()) {
      printf("    AIU_LOAD, gmem_ptr = %p, smem_ptr = %p, dim_[h,w] = [%d,%d], cube_[h,w] = [%d,%d], coord_[h,w,n] = [%d,%d,%d], offset_w = %d\n",
              gmem_ptr, smem_ptr, dim_h, dim_w, cube_h, cube_w, coord_h, coord_w, coord_n, offset_w);
    }
  }
#endif
};

template <class NumBitsPerTMA, typename Element, bool Trans, typename Enable = void>
struct PPU0010_AIU_LOAD : PPU_AIU_LOAD_BASE {};

template <class NumBitsPerTMA, typename Element>
struct PPU0010_AIU_LOAD<NumBitsPerTMA, Element, false, cute::enable_if_t<sizeof_bits<Element>::value == 16>>
: PPU_AIU_LOAD_BASE
{
  CUTE_HOST_DEVICE static void
  copy(void *smem_ptr, const void* gmem_ptr, AiuDesc desc, int coord_w, int coord_h, int coord_n = 0)
  {
    // desc.print(smem_ptr, gmem_ptr, coord_w, coord_h, coord_n);
#if defined(__HGGC_ARCH__) && ACOMPUTE_VERSION == 10000
    asm volatile(
      "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.swzl.2d.b16 [%0], [%1], "
      "{%2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11};\n"
      :: "r"(smem_ptr), "l"(gmem_ptr - desc.offset_w * 2),
        "r"(1), "r"(desc.dim_h), "r"(desc.dim_w),
        "r"(0), "r"(coord_h), "r"(coord_w + desc.offset_w),
        "r"(1), "r"(desc.cube_h), "r"(desc.cube_w), "r"(1)
    );
#else
    CUTE_RUNTIME_ASSERT("Support for AIU_LOAD has not been enabled for Trans=false,b16");
#endif
  }
};

template <class NumBitsPerTMA, typename Element>
struct PPU0010_AIU_LOAD<NumBitsPerTMA, Element, true, cute::enable_if_t<sizeof_bits<Element>::value == 16>>
: PPU_AIU_LOAD_BASE
{
  CUTE_HOST_DEVICE static void
  // coord_h and coord_w are reverse to no trans version, because aiu kernel always use identity stride (1,0,2) for input tensor
  // which will move coord_k on first and coord_m/n on second, and coord_mn when trans is coord_w for aiu
  copy(void *smem_ptr, const void* gmem_ptr, AiuDesc desc, int coord_h, int coord_w, int coord_n = 0)
  {
#if defined(__HGGC_ARCH__) && ACOMPUTE_VERSION == 10000
    asm volatile(
      "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.swzl.2d.b16 [%0], [%1], "
      "{%2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11};\n"
      :: "r"(smem_ptr), "l"(gmem_ptr - desc.offset_w * 2),
        "r"(1), "r"(desc.dim_h), "r"(desc.dim_w),
        "r"(0), "r"(coord_h), "r"(coord_w + desc.offset_w),
        "r"(1), "r"(desc.cube_h), "r"(desc.cube_w), "r"(1)
    );
#else
    CUTE_RUNTIME_ASSERT("Support for AIU_LOAD has not been enabled for Trans=true,b16");
#endif
  }
};

template <class NumBitsPerTMA, typename Element>
struct PPU0010_AIU_LOAD<NumBitsPerTMA, Element, false, cute::enable_if_t<sizeof_bits<Element>::value == 32>>
: PPU_AIU_LOAD_BASE
{
  CUTE_HOST_DEVICE static void
  copy(void *smem_ptr, const void *gmem_ptr, AiuDesc desc, int coord_w, int coord_h, int coord_n = 0)
  {
#if defined(__HGGC_ARCH__) && ACOMPUTE_VERSION == 10000
    asm volatile(
      "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.swzl.2d.b32 [%0], [%1], "
      "{%2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11};\n"
      :: "r"(smem_ptr), "l"(gmem_ptr - desc.offset_w * 4),
        "r"(1), "r"(desc.dim_h), "r"(desc.dim_w),
        "r"(0), "r"(coord_h), "r"(coord_w + desc.offset_w),
        "r"(1), "r"(desc.cube_h), "r"(desc.cube_w), "r"(1)
    );
#else
    CUTE_RUNTIME_ASSERT("Support for AIU_LOAD has not been enabled for Trans=false,b32");
#endif
  }
};

template <class NumBitsPerTMA, typename Element>
struct PPU0010_AIU_LOAD<NumBitsPerTMA, Element, true, cute::enable_if_t<sizeof_bits<Element>::value == 32>>
: PPU_AIU_LOAD_BASE
{
  CUTE_HOST_DEVICE static void
  // coord_h and coord_w are reverse to no trans version, because aiu kernel always use identity stride (1,0,2) for input tensor
  // which will move coord_k on first and coord_m/n on second, and coord_mn when trans is coord_w for aiu
  copy(void *smem_ptr, const void *gmem_ptr, AiuDesc desc, int coord_h, int coord_w, int coord_n = 0)
  {
#if defined(__HGGC_ARCH__) && ACOMPUTE_VERSION == 10000
    asm volatile(
      "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.swzl.2d.b32 [%0], [%1], "
      "{%2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11};\n"
      :: "r"(smem_ptr), "l"(gmem_ptr - desc.offset_w * sizeof(float)),
        "r"(1), "r"(desc.dim_h), "r"(desc.dim_w),
        "r"(0), "r"(coord_h), "r"(coord_w + desc.offset_w),
        "r"(1), "r"(desc.cube_h), "r"(desc.cube_w), "r"(1)
    );
#else
    CUTE_RUNTIME_ASSERT("Support for AIU_LOAD has not been enabled for Trans=true,b32");
#endif
  }
};

template <class NumBitsPerTMA, typename Element>
struct PPU0010_AIU_LOAD<NumBitsPerTMA, Element, false, cute::enable_if_t<sizeof_bits<Element>::value == 8>>
: PPU_AIU_LOAD_BASE
{
  CUTE_HOST_DEVICE static void
  copy(void *smem_ptr, const void *gmem_ptr, AiuDesc desc, int coord_w, int coord_h, int coord_n = 0)
  {
#if defined(__HGGC_ARCH__) && ACOMPUTE_VERSION == 10000
    asm volatile(
      "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.swzl.2d.b8 [%0], [%1], "
      "{%2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11};\n"
      :: "r"(smem_ptr), "l"(gmem_ptr - desc.offset_w * 1),
        "r"(1), "r"(desc.dim_h), "r"(desc.dim_w),
        "r"(0), "r"(coord_h), "r"(coord_w + desc.offset_w),
        "r"(1), "r"(desc.cube_h), "r"(desc.cube_w), "r"(1)
    );
#else
    CUTE_RUNTIME_ASSERT("Support for AIU_LOAD has not been enabled for Trans=false,b8");
#endif
  }
};

template <class NumBitsPerTMA, typename Element>
struct PPU0010_AIU_LOAD<NumBitsPerTMA, Element, true, cute::enable_if_t<sizeof_bits<Element>::value == 8>>
: PPU_AIU_LOAD_BASE
{
  CUTE_HOST_DEVICE static void
  // coord_h and coord_w are reverse to no trans version, because aiu kernel always use identity stride (1,0,2) for input tensor
  // which will move coord_k on first and coord_m/n on second, and coord_mn when trans is coord_w for aiu
  copy(void *smem_ptr, const void *gmem_ptr, AiuDesc desc, int coord_h, int coord_w, int coord_n = 0)
  {
#if defined(__HGGC_ARCH__) && ACOMPUTE_VERSION == 10000
    asm volatile(
      "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.swzl.2d.b8 [%0], [%1], "
      "{%2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11};\n"
      :: "r"(smem_ptr), "l"(gmem_ptr - desc.offset_w * 1),
        "r"(1), "r"(desc.dim_h), "r"(desc.dim_w),
        "r"(0), "r"(coord_h), "r"(coord_w + desc.offset_w),
        "r"(1), "r"(desc.cube_h), "r"(desc.cube_w), "r"(1)
    );
#else
    CUTE_RUNTIME_ASSERT("Support for AIU_LOAD has not been enabled for Trans=true,b8");
#endif
  }
};

template <class NumBitsPerTMA, typename Element>
struct PPU0010_AIU_LOAD<NumBitsPerTMA, Element, false, cute::enable_if_t<sizeof_bits<Element>::value == 4>>
: PPU_AIU_LOAD_BASE
{
  CUTE_HOST_DEVICE static void
  copy(void *smem_ptr, const void *gmem_ptr, AiuDesc desc, int coord_w, int coord_h, int coord_n = 0)
  {
    coord_w /= 2;
    // desc.print(smem_ptr, gmem_ptr, coord_w, coord_h, coord_n);
#if defined(__HGGC_ARCH__) && ACOMPUTE_VERSION == 10000
    asm volatile(
      "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.swzl.2d.b8 [%0], [%1], "
      "{%2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11};\n"
      :: "r"(smem_ptr), "l"(gmem_ptr - desc.offset_w * 1),
        "r"(1), "r"(desc.dim_h), "r"(desc.dim_w),
        "r"(0), "r"(coord_h), "r"(coord_w + desc.offset_w),
        "r"(1), "r"(desc.cube_h), "r"(desc.cube_w), "r"(1)
    );
#else
    CUTE_RUNTIME_ASSERT("Support for AIU_LOAD has not been enabled for Trans=false,b4");
#endif
  }
};

template<typename Element, int CUBE_H, int CUBE_W, bool SWAP>
CUTE_HOST_DEVICE static void
ppu_tsm_ld_swzl_sim(void *frag_ptr, void *smem_base, int coord_w, int coord_h, int stage) {
  Element *stage_base = reinterpret_cast<Element*>(smem_base);
  stage_base += CUBE_H * CUBE_W * stage;

  int slice_idx = coord_w / (32 / sizeof(Element));

  // no trans can always copy 32b each time
  uint32_t *slice_base = reinterpret_cast<uint32_t*>(stage_base);
  slice_base += CUBE_H * 8 * slice_idx;
  uint32_t *frag = reinterpret_cast<uint32_t*>(frag_ptr);

  int slice_start_vec = (((slice_idx & 1) << 1) + ((slice_idx & 2) >> 1)) * 2;

  int lane_idx = threadIdx.x % 32;
  // each warp is arrange in 8x4 for each vreg
  int lane_row_idx = lane_idx / 4 + coord_h;
  int lane_col_idx = lane_idx % 4;
  // four vreg
  CUTE_NO_UNROLL
  for (int v = 0; v < 4; v++) {
    // cur lane's cur vreg is on which row of original tsm layout
    int vreg_row_idx;
    if (!SWAP) {
      // vreg 1/3 is on lower 8 rows
      vreg_row_idx = (v % 2) * 8 + lane_row_idx;
    } else {
      // vreg 2/3 is on lower 8 rows
      vreg_row_idx = (v / 2) * 8 + lane_row_idx;
    }

    // cache line index, four rows in one cache line
    int vreg_line_idx = vreg_row_idx / 4;
    int vreg_vec_idx;
    if (!SWAP) {
      // vreg 2/3 on right of two vectors
      vreg_vec_idx = (vreg_row_idx % 4) * 2 + (v / 2);
    } else {
      // vreg 1/3 on right of two vectors
      vreg_vec_idx = (vreg_row_idx % 4) * 2 + (v % 2);
    }
    // swizzle between two vector in odd cache line
    int vreg_vec_idx_swz1 = vreg_vec_idx ^ (vreg_line_idx % 2);
    // swizzle for each slice
    int vreg_vec_idx_swz2 = (vreg_vec_idx_swz1 + slice_start_vec) % 8;

    // int32 offset of cur reg
    int lane_32b_off = vreg_line_idx * 32 + vreg_vec_idx_swz2 * 4 + lane_col_idx;
    frag[v] = slice_base[lane_32b_off];
  }

}

template <typename Element, bool Trans, int Num = 4>
struct PPU0010_TSM_LD_SWZL_IMPL;

template <typename Element>
struct PPU0010_TSM_LD_SWZL_IMPL<Element, false, 4> {
  CUTE_HOST_DEVICE void operator()(int *vreg, Element *stage_base, int coord_h, int CUBE_H, int channel_bytes_offset) {
#if defined(__HGGC_ARCH__) && ACOMPUTE_VERSION == 10000
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ldmatrix.sync.aligned.m8n8.x4.swzl.shared.b16 {%0, %1, %2, %3}, [%4], {%5, %6, %7, %8, %9, %10};"
      : "=r"(vreg[0]), "=r"(vreg[1]), "=r"(vreg[2]), "=r"(vreg[3])
      : "l"(stage_base), "r"(0), "r"(coord_h),
        "r"(1), "r"(CUBE_H), "r"(1), "r"(channel_bytes_offset)
    );
#endif
#else
    CUTE_RUNTIME_ASSERT("Support for TSM_LD_SWZL has not been enabled for Trans=false");
#endif
  }
};

template <typename Element>
struct PPU0010_TSM_LD_SWZL_IMPL<Element, false, 2> {
  CUTE_HOST_DEVICE void operator()(int *vreg, Element *stage_base, int coord_h, int CUBE_H, int channel_bytes_offset) {
#if defined(__HGGC_ARCH__) && ACOMPUTE_VERSION == 10000
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ldmatrix.sync.aligned.m8n8.x2.swzl.shared.b16 {%0, %1, %2, %3}, [%4], {%5, %6, %7, %8, %9, %10};"
      : "=r"(vreg[0]), "=r"(vreg[1]), "=r"(vreg[2]), "=r"(vreg[3])
      : "l"(stage_base), "r"(0), "r"(coord_h),
        "r"(1), "r"(CUBE_H), "r"(1), "r"(channel_bytes_offset)
    );
#endif
#else
    CUTE_RUNTIME_ASSERT("Support for TSM_LD_SWZL has not been enabled for Trans=false");
#endif
  }
};

template <>
struct PPU0010_TSM_LD_SWZL_IMPL<half_t, true, 4> {
  CUTE_HOST_DEVICE void operator()(int *vreg, half_t *stage_base, int coord_h, int CUBE_H, int channel_bytes_offset) {
#if defined(__HGGC_ARCH__) && ACOMPUTE_VERSION == 10000
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ldmatrix.sync.aligned.m16n16.x1.swzl.trans.shared.b16 {%0, %1, %2, %3}, [%4], {%5, %6, %7, %8, %9, %10};"
      : "=r"(vreg[0]), "=r"(vreg[1]), "=r"(vreg[2]), "=r"(vreg[3])
      : "l"(stage_base), "r"(0), "r"(coord_h),
        "r"(1), "r"(CUBE_H), "r"(1), "r"(channel_bytes_offset)
    );
#endif
#else
    CUTE_RUNTIME_ASSERT("Support for TSM_LD_SWZL has not been enabled for Trans=true");
#endif
  }
};

template <>
struct PPU0010_TSM_LD_SWZL_IMPL<bfloat16_t, true, 4> {
  CUTE_HOST_DEVICE void operator()(int *vreg, bfloat16_t *stage_base, int coord_h, int CUBE_H, int channel_bytes_offset) {
#if defined(__HGGC_ARCH__) && ACOMPUTE_VERSION == 10000
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ldmatrix.sync.aligned.m16n16.x1.swzl.trans.shared.b16 {%0, %1, %2, %3}, [%4], {%5, %6, %7, %8, %9, %10};"
      : "=r"(vreg[0]), "=r"(vreg[1]), "=r"(vreg[2]), "=r"(vreg[3])
      : "l"(stage_base), "r"(0), "r"(coord_h),
        "r"(1), "r"(CUBE_H), "r"(1), "r"(channel_bytes_offset)
    );
#endif
#else
    CUTE_RUNTIME_ASSERT("Support for TSM_LD_SWZL has not been enabled for Trans=true");
#endif
  }
};

template <>
struct PPU0010_TSM_LD_SWZL_IMPL<float, true, 4> {
  CUTE_HOST_DEVICE void operator()(int *vreg, float *stage_base, int coord_h, int CUBE_H, int channel_bytes_offset) {
#if defined(__HGGC_ARCH__) && ACOMPUTE_VERSION == 10000
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ldmatrix.sync.aligned.m8n16.x1.swzl.trans.shared.b32 {%0, %1, %2, %3}, [%4], {%5, %6, %7, %8, %9, %10};"
      : "=r"(vreg[0]), "=r"(vreg[1]), "=r"(vreg[2]), "=r"(vreg[3])
      : "l"(stage_base), "r"(0), "r"(coord_h),
        "r"(1), "r"(CUBE_H), "r"(1), "r"(channel_bytes_offset)
    );
#endif
#else
    CUTE_RUNTIME_ASSERT("Support for TSM_LD_SWZL has not been enabled for Trans=true");
#endif
  }
};


template <typename Element, int CUBE_H, int CUBE_W, bool Swap, bool Trans, int InstNum = 1, int Num = 4>
struct PPU0010_TSM_LD_SWZL;

template <typename Element, int CUBE_H, int CUBE_W, bool Swap, int InstNum, int Num>
struct PPU0010_TSM_LD_SWZL<Element, CUBE_H, CUBE_W, Swap, false, InstNum, Num> {

  CUTE_HOST_DEVICE static void
  copy(void *frag_ptr, void *smem_base, int coord_w, int coord_h, int cube_in_stage = 0, int stage = 0)
  {
    // if (cute::thread(0)) {
    //   print("smem_base = 0x%x, CUBE_W = %d, CUBE_H = %d, coord_w = %d, coord_h = %d, cube_in_stage = %d, stage = %d\n",
    //         smem_base, CUBE_W, CUBE_H, coord_w, coord_h, cube_in_stage, stage);
    // }
#if defined(__HGGC_ARCH__) && ACOMPUTE_VERSION == 10000
  Element *stage_base = reinterpret_cast<Element*>(smem_base);
  stage_base += CUBE_H * CUBE_W * (cube_in_stage + stage * InstNum);
  int *vreg = reinterpret_cast<int *>(frag_ptr);
  int channel_bytes_offset = coord_w * sizeof(Element);


  PPU0010_TSM_LD_SWZL_IMPL<Element, false, Num>()(vreg, stage_base, coord_h, CUBE_H, channel_bytes_offset);
#else
  CUTE_RUNTIME_ASSERT("Support for TSM_LD_SWZL has not been enabled for Trans=false");
  // ppu_tsm_ld_swzl_sim<Element, CUBE_H, CUBE_W, false>(frag_ptr, smem_base, coord_w, coord_h, stage);
#endif
  }
};

template <typename Element, int CUBE_H, int CUBE_W, bool Swap, int InstNum, int Num>
struct PPU0010_TSM_LD_SWZL<Element, CUBE_H, CUBE_W, Swap, true, InstNum, Num> {

  CUTE_HOST_DEVICE static void
  // similar to aiu_load with trans, first coord is coord on K, which is coord_h when trans
  copy(void *frag_ptr, void *smem_base, int coord_h, int coord_w, int cube_in_stage = 0, int stage = 0)
  {
    // if (cute::thread(0)) {
    //   print("smem_base = 0x%x, CUBE_W = %d, CUBE_H = %d, coord_w = %d, coord_h = %d, cube_in_stage = %d, stage = %d\n",
    //         smem_base, CUBE_W, CUBE_H, coord_w, coord_h, cube_in_stage, stage);
    // }
#if defined(__HGGC_ARCH__) && ACOMPUTE_VERSION == 10000
  Element *stage_base = reinterpret_cast<Element*>(smem_base);
  stage_base += CUBE_H * CUBE_W * (cube_in_stage + stage * InstNum);
  int *vreg = reinterpret_cast<int *>(frag_ptr);
  int channel_bytes_offset = coord_w * sizeof(Element);
  PPU0010_TSM_LD_SWZL_IMPL<Element, true, Num>()(vreg, stage_base, coord_h, CUBE_H, channel_bytes_offset);
#else
  CUTE_RUNTIME_ASSERT("Support for TSM_LD_SWZL has not been enabled for Trans=true");
#endif
  }

};

} // namespace cute
