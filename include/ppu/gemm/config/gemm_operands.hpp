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

#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_atom.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/layout/layout.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"


#include "ppu/cute/arch/mma_ppu.hpp"
#include "ppu/cute/arch/mma_ppu0015.hpp"
#include "ppu/cute/arch/copy_ppu0015_aiu.hpp"
#include "cute/arch/mma_ppu.hpp"
#include "ppu/cutlass/float4.h"

namespace cutlass {
namespace gemm {
namespace config {
using namespace cute;

template <class StrideIntT>
cute::Stride<StrideIntT, cute::Int<1>, int64_t>
make_cute_packed_stride(cute::Stride<StrideIntT, cute::Int<1>, int64_t> s, cute::Shape<int,int,int> shape_MKL, long long int batch_stride) {
  static_assert(std::is_integral_v<StrideIntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");
  auto s_copy = s;
  cute::get<0>(s_copy) = static_cast<StrideIntT>(cute::get<1>(shape_MKL));
  int batch_count =  cute::get<2>(shape_MKL);
  cute::get<2>(s_copy) = static_cast<StrideIntT>(batch_stride);
  return s_copy;
}

template <class StrideIntT>
cute::Stride<cute::Int<1>, StrideIntT, int64_t>
make_cute_packed_stride(cute::Stride<cute::Int<1>, StrideIntT, int64_t> s, cute::Shape<int,int,int> shape_MKL, long long int batch_stride) {
  static_assert(std::is_integral_v<StrideIntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");
  auto s_copy = s;
  cute::get<1>(s_copy) = static_cast<StrideIntT>(cute::get<0>(shape_MKL));
  int batch_count =  cute::get<2>(shape_MKL);
  cute::get<2>(s_copy) = static_cast<StrideIntT>(batch_stride);
  return s_copy;
}


// #define PPU_MMA_ON_PPU

template <typename T>
struct GetMmaInst;

template <>
struct GetMmaInst<float> {
#if ACOMPUTE_VERSION == 10000
#else
  using type = PPU0015_16x16x8_F32F32F32F32_TN;
#endif
};

template <>
struct GetMmaInst<cutlass::tfloat32_t> {
#if ACOMPUTE_VERSION == 10000
  using type = PPU_16x16x8_F32TF32TF32F32_TN;
#else
  // using type = PPU_16x8x8_F32TF32TF32F32_TN;
  using type = PPU0015_16x16x8_F32TF32TF32F32_TN;
#endif
};

template <>
struct GetMmaInst<cutlass::half_t> {
#if ACOMPUTE_VERSION == 10000
  using type = PPU_16x16x16_F32F16F16F32_TN;
#else
  // using type = PPU_16x8x16_F32F16F16F32_TN;
  using type = PPU0015_16x16x16_F32F16F16F32_TN;
#endif
};

template <>
struct GetMmaInst<cutlass::bfloat16_t> {
#if ACOMPUTE_VERSION == 10000
  using type = PPU_16x16x16_F32F16F16F32_TN;
#else
  // using type = PPU_16x8x16_F32F16F16F32_TN;
  using type = PPU0015_16x16x16_F32BF16BF16F32_TN;
#endif
};

template <>
struct GetMmaInst<cutlass::float_e4m3_t> {
#if ACOMPUTE_VERSION == 10000
#else
  using type = PPU0015_16x16x32_F32E4M3E4M3F32_TN;
#endif
};


template <>
struct GetMmaInst<cutlass::float_e5m2_t> {
#if ACOMPUTE_VERSION == 10000
#else
  using type = PPU0015_16x16x32_F32E5M2E4M3F32_TN;
#endif
};

template <typename T>
struct GetAiuMmaInst;

template <>
struct GetAiuMmaInst<float> {
#if ACOMPUTE_VERSION == 10000
#else
  using type = PPU0015_16x16x8_F32F32F32F32_TN;
#endif
};

template <>
struct GetAiuMmaInst<cutlass::tfloat32_t> {
#if ACOMPUTE_VERSION == 10000
  using type = PPU_16x16x8_F32TF32TF32F32_TN;
#else
  using type = PPU0015_16x16x8_F32TF32TF32F32_TN;
#endif
};

template <>
struct GetAiuMmaInst<cutlass::half_t> {
#if ACOMPUTE_VERSION == 10000
  using type = PPU_16x16x16_F32F16F16F32_TN;
#else
  using type = PPU0015_16x16x16_F32F16F16F32_TN;
#endif
};

template <>
struct GetAiuMmaInst<cutlass::bfloat16_t> {
#if ACOMPUTE_VERSION == 10000
#else
  using type = PPU0015_16x16x16_F32BF16BF16F32_TN;
#endif
};

template <>
struct GetAiuMmaInst<cutlass::float_e4m3_t> {
#if ACOMPUTE_VERSION == 10000
#else
  using type = PPU0015_16x16x32_F32E4M3E4M3F32_TN;
#endif
};


template <>
struct GetAiuMmaInst<cutlass::float_e5m2_t> {
#if ACOMPUTE_VERSION == 10000
#else
  using type = PPU0015_16x16x32_F32E5M2E4M3F32_TN;
#endif
};

template <>
struct GetAiuMmaInst<cutlass::float4_t> {
#if ACOMPUTE_VERSION == 10000
#else
  using type = PPU0015_16x16x64_F32F4F4F32_TN;
#endif
};

// ========== normal gemm ==========
template <
  typename Element,
  bool Trans,
  int Alignment,
  typename SizeCont,
  int ThreadNum,
  typename CopyInst
> struct DefaultGemm_TensorOpPPU0010_Operand;

template <
  typename Element,
  int Alignment,
  typename SizeCont,
  int ThreadNum,
  typename CopyInst
> struct DefaultGemm_TensorOpPPU0010_Operand<
  Element,
  false,
  Alignment,
  SizeCont,
  ThreadNum,
  CopyInst
> {
  static constexpr int ElemInLine = min(SizeCont(), 128 / sizeof(Element));
  static constexpr int kFactor = 128 / ElemInLine / sizeof(Element);
  static constexpr int SwizzleIdx = kFactor == 1 ? 3 : (kFactor == 2 ? 2 : 1);
  static constexpr int SwizzleAtom = sizeof(Element) == 4 ? 2 : (sizeof(Element) == 2 ? 3 : 4);

  using SmemLayoutAtom = decltype(
    composition(Swizzle<SwizzleIdx,SwizzleAtom,3>{},
                Layout<Shape < _8,Int<ElemInLine>>,
                       Stride<Int<ElemInLine>, _1>>{}));
  using SmemCopyAtom = Copy_Atom<PPU_U32x4_LDSM_N, Element>;

  static constexpr int ThreadK = platform::min(ThreadNum, ElemInLine / Alignment);
  static constexpr int ThreadM = ThreadNum / ThreadK;
  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<CopyInst, Element>{},
                    Layout<Shape <Int<ThreadM>,Int<ThreadK>>,
                           Stride< Int<ThreadK>,_1>>{},
                    Layout<Shape < _1,Int<Alignment>>>{}));
};

template <
  typename Element,
  int Alignment,
  typename SizeCont,
  int ThreadNum,
  typename CopyInst
> struct DefaultGemm_TensorOpPPU0010_Operand<
  Element,
  true,
  Alignment,
  SizeCont,
  ThreadNum,
  CopyInst
> {
  static constexpr int ElemInLine = min(SizeCont(), 128 / sizeof(Element));
  static constexpr int kFactor = 128 / ElemInLine / sizeof(Element);
  static constexpr int SwizzleIdx = kFactor == 1 ? 3 : (kFactor == 2 ? 2 : 1);
  static constexpr int SwizzleAtom = sizeof(Element) == 4 ? 2 : (sizeof(Element) == 2 ? 3 : 4);

  using SmemLayoutAtom = decltype(
    composition(Swizzle<SwizzleIdx,SwizzleAtom,3>{},
                Layout<Shape <Int<ElemInLine>, _8>,
                       Stride<_1, Int<ElemInLine>>>{}));
  using SmemCopyOp = typename platform::conditional<
                        sizeof(Element) == 4,
                        UniversalCopy<float>,
                        PPU_U16x8_LDSM_T
                     >::type;
  using SmemCopyAtom = Copy_Atom<SmemCopyOp, Element>;

  static constexpr int ThreadM = platform::min(ThreadNum, ElemInLine / Alignment);
  static constexpr int ThreadK = ThreadNum / ThreadM;
  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<CopyInst, Element>{},
                    Layout<Shape <Int<ThreadM>,Int<ThreadK>>,
                           Stride< _1, Int<ThreadM>>>{},
                    Layout<Shape < Int<Alignment>, _1>>{}));
};

template <
  typename EpilogueCopyInst,
  typename ElementC,
  typename ElementAcc,
  int Alignment,
  typename BlockM,
  typename BlockN,
  typename WarpOnM,
  int ThreadNum
> struct DefaultGemm_Epilogue_Configuration {
  static constexpr int ElemInLine = min(BlockN(), 128 / sizeof(ElementAcc));
  static constexpr bool kFactor = 128 / ElemInLine / sizeof(ElementAcc);
  static constexpr int SwizzleIdx = kFactor == 1 ? 3 : (kFactor == 2 ? 2 : 1);
  static constexpr int SwizzleAtom = sizeof(ElementAcc) == 4 ? 2 : 3;

  using SmemLayoutAtomO = decltype(
    composition(Swizzle<SwizzleIdx, SwizzleAtom, 3>{},
                Layout<Shape<Int<8>, Int<ElemInLine>>, Stride<Int<ElemInLine>, _1>>{}));

  using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<WarpOnM() * 16>, BlockN>{}));

  static constexpr int EpiThreadN = platform::min(ThreadNum, BlockN() / Alignment);
  static constexpr int EpiThreadM = ThreadNum / EpiThreadN;
  using GmemLayoutAtom = Layout<Shape <Int<EpiThreadM>, Int<EpiThreadN>>,
                                  Stride<Int<EpiThreadN>, _1>>;

  using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<EpilogueCopyInst, ElementC>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<Alignment>>>{}));

};


// ========== aiu gemm ==========
template <
  typename Element,
  bool Trans,
  typename Block_MN,
  typename Block_K,
  bool Swap
> struct DefaultGemm_AIU_Operand;

template <
  typename Element,
  typename Block_MN,
  typename Block_K,
  bool Swap
> struct DefaultGemm_AIU_Operand<
  Element,
  false,
  Block_MN,
  Block_K,
  Swap
> {

  static_assert(Block_K{} * sizeof_bits<Element>::value * 8 % 32 == 0, "aiu_no_trans: block_k must be multiple of 32B");
  static constexpr int BlockContSize = Block_K{} * sizeof_bits<Element>::value / 8;
  static_assert(BlockContSize > 128 ? (BlockContSize % 128 == 0) : (BlockContSize % 32 == 0), "aiu_trans: block contiguous size should be multiple of 128B or 32B");
  static constexpr int AiuContByteSize = BlockContSize > 128 ? 128 : BlockContSize;
  static constexpr int AiuContElemSize = AiuContByteSize / sizeof_bits<Element>::value * 8;
  static constexpr bool split_on_k = AiuContElemSize < Block_K{};
  // each aiu_load should at least cover one mma
  static constexpr int inst_num = split_on_k ? (Block_K{} / AiuContElemSize) : (Block_MN{} > 16 ? 2 : 1);

#if ACOMPUTE_VERSION == 10000
  static constexpr int CUBE_H = Block_MN{};
  static constexpr int CUBE_W = AiuContElemSize;
  static constexpr int bits_per_aiu = CUBE_H * CUBE_W * sizeof_bits<Element>::value;
  using CopyInst = PPU0010_AIU_LOAD<cute::C<bits_per_aiu>, Element, false>;
  using SmemCopyOp = PPU0010_TSM_LD_SWZL<Element, CUBE_H, CUBE_W, Swap, false, 1>;
#else
  // will only split on one dimension
  static constexpr int CUBE_H = split_on_k ? Block_MN{} : (Block_MN{} / inst_num);
  static constexpr int CUBE_W = AiuContElemSize;
  static constexpr int bits_per_aiu = CUBE_H * CUBE_W * sizeof_bits<Element>::value;
  using CopyInst = PPU0015_AIU_LOAD<cute::C<bits_per_aiu>, Element, false, CUBE_H, CUBE_W>;
  using SmemCopyOp = PPU0015_TSM_LD_SWZL<Element, CUBE_H, CUBE_W, Swap, false, inst_num>;
#endif


  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<CopyInst, Element>{},
                    Layout<Shape <_1,_1>,
                           Stride<_1,_1>>{},
                    Layout<Shape <Int<CUBE_H>, Int<CUBE_W>>>{}));

  using SmemCopyAtom = Copy_Atom<SmemCopyOp, Element>;
  // to make smem_layout[0] or smem_layout[1] to be two dimensional, SmemLayoutAtom.stride is not used in aiu gemm
  using SmemAtomStride = typename platform::conditional<
    split_on_k,
    Stride<Int<CUBE_W>, _1>,
    Stride<_1, Int<CUBE_H>>
  >::type;
  using SmemLayoutAtom = Layout<Shape<Int<CUBE_H>, Int<CUBE_W>>, SmemAtomStride>;
};

template <
  typename Element,
  typename Block_MN,
  typename Block_K,
  bool Swap
> struct DefaultGemm_AIU_Operand<
  Element,
  true,
  Block_MN,
  Block_K,
  Swap
> {

  static constexpr int BlockContSize = Block_MN{} * sizeof(Element);
  static_assert(BlockContSize > 128 ? (BlockContSize % 128 == 0) : (BlockContSize % 32 == 0), "aiu_trans: block contiguous size should be multiple of 128B or 32B");
  static constexpr int AiuContByteSize = BlockContSize > 128 ? 128 : BlockContSize;
  using AiuContElemSize = Int<AiuContByteSize / sizeof(Element)>;
  static constexpr int InstNum = Block_MN{} / AiuContElemSize{};

  static constexpr int bits_per_aiu = AiuContByteSize * 8 * Block_K{};
  static constexpr int CUBE_H = Block_K{};
  static constexpr int CUBE_W = AiuContElemSize{};

#if ACOMPUTE_VERSION == 10000
  using CopyInst = PPU0010_AIU_LOAD<cute::C<bits_per_aiu>, Element, true>;
  using SmemCopyOp = PPU0010_TSM_LD_SWZL<Element, Block_K{}, AiuContElemSize{}, Swap, true, InstNum>;
#else
  using CopyInst = PPU0015_AIU_LOAD<cute::C<bits_per_aiu>, Element, true, CUBE_H, CUBE_W>;
  using SmemCopyOp = PPU0015_TSM_LD_SWZL<Element, Block_K{}, AiuContElemSize{}, Swap, true, InstNum>;
#endif

  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<CopyInst, Element>{},
                    Layout<Shape <_1,_1>,
                           Stride<_1,_1>>{},
                    Layout<Shape <Int<CUBE_W>, Int<CUBE_H>>>{}));

  using SmemCopyAtom = Copy_Atom<SmemCopyOp, Element>;
  // using SmemLayoutAtom = Layout<Shape<_8, Block_K>, Stride<Block_K, _1>>;
  using SmemLayoutAtom = Layout<Shape<Int<CUBE_W>, Int<CUBE_H>>, Stride<_1, Int<CUBE_W>>>;
};




} // namespace config
} // namespace gemm
} // namespace cutlass


