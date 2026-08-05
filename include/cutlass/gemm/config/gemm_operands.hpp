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


#include "cute/arch/mma_ppu0010.hpp"
#include "cute/arch/mma_ppu0015.hpp"
#include "cute/arch/copy_ppu0010_aiu.hpp"
#include "cute/arch/copy_ppu0015_aiu.hpp"
#include "cutlass/float4.h"

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
  auto batch_count =  cute::get<2>(shape_MKL);
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
  auto batch_count =  cute::get<2>(shape_MKL);
  cute::get<2>(s_copy) = static_cast<StrideIntT>(batch_stride);
  return s_copy;
}


template <typename Arch, typename TypeA, typename TypeB = TypeA, typename TypeAcc=float>
struct GetMmaInst;

template <typename Arch>
struct GetMmaInst<Arch, float, float, float> {
  // PPU_ARCH == 1.5, only for build on PPU1.0, should never be run on PPU1.0
  using type = cute::conditional_t<
    cute::is_same_v<Arch, cutlass::arch::PPU0015>,
    PPU0015_16x16x8_F32F32F32F32_TN,
    void
  >;
};

template <typename Arch>
struct GetMmaInst<Arch, cutlass::tfloat32_t, cutlass::tfloat32_t, float> {
  using type = cute::conditional_t<
    cute::is_same_v<Arch, cutlass::arch::PPU0015>,
    PPU0015_16x16x8_F32TF32TF32F32_TN,
    PPU0010_16x16x8_F32TF32TF32F32_TN
  >;
};

template <typename Arch>
struct GetMmaInst<Arch, cutlass::half_t, cutlass::half_t, float> {
  using type = cute::conditional_t<
    cute::is_same_v<Arch, cutlass::arch::PPU0015>,
    PPU0015_16x16x16_F32F16F16F32_TN,
    PPU0010_16x16x16_F32F16F16F32_TN
  >;
};

template <typename Arch>
struct GetMmaInst<Arch, cutlass::half_t, cutlass::half_t, cutlass::half_t> {
  using type = cute::conditional_t<
    cute::is_same_v<Arch, cutlass::arch::PPU0015>,
    PPU0015_16x16x16_F16F16F16F16_TN,
    PPU0010_16x16x16_F16F16F16F16_TN
  >;
};

template <typename Arch>
struct GetMmaInst<Arch, cutlass::bfloat16_t, cutlass::bfloat16_t, float> {
  using type = cute::conditional_t<
    cute::is_same_v<Arch, cutlass::arch::PPU0015>,
    PPU0015_16x16x16_F32BF16BF16F32_TN,
    PPU0010_16x16x16_F32BF16BF16F32_TN
  >;
};

template <typename Arch>
struct GetMmaInst<Arch, cutlass::float_e4m3_t, cutlass::float_e4m3_t, float> {
  // PPU_ARCH == 1.5, only for build on PPU1.0, should never be run on PPU1.0
  using type = cute::conditional_t<
    cute::is_same_v<Arch, cutlass::arch::PPU0015>,
    PPU0015_16x16x32_F32E4M3E4M3F32_TN,
    void
  >;
};

template <typename Arch>
struct GetMmaInst<Arch, cutlass::float_e4m3_t, cutlass::float_e5m2_t, float> {
  // PPU_ARCH == 1.5, only for build on PPU1.0, should never be run on PPU1.0
  using type = cute::conditional_t<
    cute::is_same_v<Arch, cutlass::arch::PPU0015>,
    PPU0015_16x16x32_F32E4M3E5M2F32_TN,
    void
  >;
};

template <typename Arch>
struct GetMmaInst<Arch, cutlass::float_e5m2_t, cutlass::float_e4m3_t, float> {
  // PPU_ARCH == 1.5, only for build on PPU1.0, should never be run on PPU1.0
  using type = cute::conditional_t<
    cute::is_same_v<Arch, cutlass::arch::PPU0015>,
    PPU0015_16x16x32_F32E5M2E4M3F32_TN,
    void
  >;
};

template <typename Arch>
struct GetMmaInst<Arch, cutlass::float_e5m2_t, cutlass::float_e5m2_t, float> {
  // PPU_ARCH == 1.5, only for build on PPU1.0, should never be run on PPU1.0
  using type = cute::conditional_t<
    cute::is_same_v<Arch, cutlass::arch::PPU0015>,
    PPU0015_16x16x32_F32E5M2E5M2F32_TN,
    void
  >;
};

template <typename Arch>
struct GetMmaInst<Arch, cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::half_t> {
  // PPU_ARCH == 1.5, only for build on PPU1.0, should never be run on PPU1.0
  using type = cute::conditional_t<
    cute::is_same_v<Arch, cutlass::arch::PPU0015>,
    PPU0015_16x16x32_F16E4M3E4M3F16_TN,
    void
  >;
};

template <typename Arch>
struct GetMmaInst<Arch, cutlass::float_e4m3_t, cutlass::float_e5m2_t, cutlass::half_t> {
  // PPU_ARCH == 1.5, only for build on PPU1.0, should never be run on PPU1.0
  using type = cute::conditional_t<
    cute::is_same_v<Arch, cutlass::arch::PPU0015>,
    PPU0015_16x16x32_F16E4M3E5M2F16_TN,
    void
  >;
};

template <typename Arch>
struct GetMmaInst<Arch, cutlass::float_e5m2_t, cutlass::float_e4m3_t, cutlass::half_t> {
  // PPU_ARCH == 1.5, only for build on PPU1.0, should never be run on PPU1.0
  using cute = cute::conditional_t<
    cute::is_same_v<Arch, cutlass::arch::PPU0015>,
    PPU0015_16x16x32_F16E5M2E4M3F16_TN,
    void
  >;
};

template <typename Arch>
struct GetMmaInst<Arch, cutlass::float_e5m2_t, cutlass::float_e5m2_t, cutlass::half_t> {
  // PPU_ARCH == 1.5, only for build on PPU1.0, should never be run on PPU1.0
  using type = cute::conditional_t<
    cute::is_same_v<Arch, cutlass::arch::PPU0015>,
    PPU0015_16x16x32_F16E5M2E5M2F16_TN,
    void
  >;
};

template <typename Arch>
struct GetMmaInst<Arch, int8_t, int8_t, int32_t> {
  using type = cute::conditional_t<
    cute::is_same_v<Arch, cutlass::arch::PPU0015>,
    PPU0015_16x16x32_S32S8S8S32_TN,
    PPU0010_16x16x32_S32S8S8S32_TN
  >;
};

template <typename Arch, typename TypeA, typename TypeB = TypeA, typename TypeAcc=float>
struct GetAiuMmaInst;

template <typename Arch>
struct GetAiuMmaInst<Arch, float, float, float> {
  // PPU_ARCH == 1.5, only for build on PPU1.0, should never be run on PPU1.0
  using type = cute::conditional_t<
    cute::is_same_v<Arch, cutlass::arch::PPU0015>,
    PPU0015_16x16x8_F32F32F32F32_TN,
    void
  >;
};

template <typename Arch>
struct GetAiuMmaInst<Arch, cutlass::tfloat32_t, cutlass::tfloat32_t, float> {
  using type = cute::conditional_t<
    cute::is_same_v<Arch, cutlass::arch::PPU0015>,
    PPU0015_16x16x8_F32TF32TF32F32_TN,
    PPU0010_16x16x8_F32TF32TF32F32_TN
  >;
};

template <typename Arch>
struct GetAiuMmaInst<Arch, cutlass::half_t, cutlass::half_t, float> {
  using type = cute::conditional_t<
    cute::is_same_v<Arch, cutlass::arch::PPU0015>,
    PPU0015_16x16x16_F32F16F16F32_TN,
    PPU0010_16x16x16_F32F16F16F32_TN
  >;
};

template <typename Arch>
struct GetAiuMmaInst<Arch, cutlass::half_t, cutlass::half_t, cutlass::half_t> {
  using type = cute::conditional_t<
    cute::is_same_v<Arch, cutlass::arch::PPU0015>,
    PPU0015_16x16x16_F16F16F16F16_TN,
    PPU0010_16x16x16_F16F16F16F16_TN
  >;
};

template <typename Arch>
struct GetAiuMmaInst<Arch, cutlass::bfloat16_t, cutlass::bfloat16_t, float> {
  using type = cute::conditional_t<
    cute::is_same_v<Arch, cutlass::arch::PPU0015>,
    PPU0015_16x16x16_F32BF16BF16F32_TN,
    PPU0010_16x16x16_F32BF16BF16F32_TN
  >;
};

template <typename Arch>
struct GetAiuMmaInst<Arch, cutlass::float_e4m3_t, cutlass::float_e4m3_t, float> {
  // PPU_ARCH == 1.5, only for build on PPU1.0, should never be run on PPU1.0
  using type = cute::conditional_t<
    cute::is_same_v<Arch, cutlass::arch::PPU0015>,
    PPU0015_16x16x32_F32E4M3E4M3F32_TN,
    void
  >;
};

template <typename Arch>
struct GetAiuMmaInst<Arch, cutlass::float_e4m3_t, cutlass::float_e5m2_t, float> {
  // PPU_ARCH == 1.5, only for build on PPU1.0, should never be run on PPU1.0
  using type = cute::conditional_t<
    cute::is_same_v<Arch, cutlass::arch::PPU0015>,
    PPU0015_16x16x32_F32E4M3E5M2F32_TN,
    void
  >;
};

template <typename Arch>
struct GetAiuMmaInst<Arch, cutlass::float_e5m2_t, cutlass::float_e4m3_t, float> {
  // PPU_ARCH == 1.5, only for build on PPU1.0, should never be run on PPU1.0
  using type = cute::conditional_t<
    cute::is_same_v<Arch, cutlass::arch::PPU0015>,
    PPU0015_16x16x32_F32E5M2E4M3F32_TN,
    void
  >;
};

template <typename Arch>
struct GetAiuMmaInst<Arch, cutlass::float_e5m2_t, cutlass::float_e5m2_t, float> {
  // PPU_ARCH == 1.5, only for build on PPU1.0, should never be run on PPU1.0
  using type = cute::conditional_t<
    cute::is_same_v<Arch, cutlass::arch::PPU0015>,
    PPU0015_16x16x32_F32E5M2E5M2F32_TN,
    void
  >;
};

template <typename Arch>
struct GetAiuMmaInst<Arch, cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::half_t> {
  // PPU_ARCH == 1.5, only for build on PPU1.0, should never be run on PPU1.0
  using type = cute::conditional_t<
    cute::is_same_v<Arch, cutlass::arch::PPU0015>,
    PPU0015_16x16x32_F16E4M3E4M3F16_TN,
    void
  >;
};

template <typename Arch>
struct GetAiuMmaInst<Arch, cutlass::float_e4m3_t, cutlass::float_e5m2_t, cutlass::half_t> {
  // PPU_ARCH == 1.5, only for build on PPU1.0, should never be run on PPU1.0
  using type = cute::conditional_t<
    cute::is_same_v<Arch, cutlass::arch::PPU0015>,
    PPU0015_16x16x32_F16E4M3E5M2F16_TN,
    void
  >;
};

template <typename Arch>
struct GetAiuMmaInst<Arch, cutlass::float_e5m2_t, cutlass::float_e4m3_t, cutlass::half_t> {
  // PPU_ARCH == 1.5, only for build on PPU1.0, should never be run on PPU1.0
  using type = cute::conditional_t<
    cute::is_same_v<Arch, cutlass::arch::PPU0015>,
    PPU0015_16x16x32_F16E5M2E4M3F16_TN,
    void
  >;
};

template <typename Arch>
struct GetAiuMmaInst<Arch, cutlass::float_e5m2_t, cutlass::float_e5m2_t, cutlass::half_t> {
  // PPU_ARCH == 1.5, only for build on PPU1.0, should never be run on PPU1.0
  using type = cute::conditional_t<
    cute::is_same_v<Arch, cutlass::arch::PPU0015>,
    PPU0015_16x16x32_F16E5M2E5M2F16_TN,
    void
  >;
};

template <typename Arch>
struct GetAiuMmaInst<Arch, int8_t, int8_t, int32_t> {
  using type = cute::conditional_t<
    cute::is_same_v<Arch, cutlass::arch::PPU0015>,
    PPU0015_16x16x32_S32S8S8S32_TN,
    PPU0010_16x16x32_S32S8S8S32_TN
  >;
};

template <typename Arch>
struct GetAiuMmaInst<Arch, cutlass::float4_t, cutlass::float4_t, float> {
  // PPU_ARCH == 1.5, only for build on PPU1.0, should never be run on PPU1.0
  using type = cute::conditional_t<
    cute::is_same_v<Arch, cutlass::arch::PPU0015>,
    PPU0015_16x16x64_F32F4F4F32_TN,
    void
  >;
};

// ========== normal gemm ==========
template <
  typename Element,
  bool Trans,
  int Alignment,
  typename SizeCont,
  int ThreadNum,
  typename CopyInst
> struct DefaultGemm_TensorOpPPU_Operand;

template <
  typename Element,
  int Alignment,
  typename SizeCont,
  int ThreadNum,
  typename CopyInst
> struct DefaultGemm_TensorOpPPU_Operand<
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
> struct DefaultGemm_TensorOpPPU_Operand<
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

  using EpilogueTile = decltype(shape(coalesce(make_layout(shape(SmemLayoutO{})), Step<_1, _1>{})));

  static constexpr int EpiThreadN = platform::min(ThreadNum, BlockN() / Alignment);
  static constexpr int EpiThreadM = ThreadNum / EpiThreadN;
  using GmemLayoutAtom = Layout<Shape <Int<EpiThreadM>, Int<EpiThreadN>>,
                                  Stride<Int<EpiThreadN>, _1>>;

  using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<EpilogueCopyInst, ElementAcc>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<Alignment>>>{}));

};


// ========== aiu gemm ==========
template <
  typename Arch,
  typename Element,
  bool Trans,
  typename Block_MN,
  typename Block_K,
  bool Swap,
  int StageStride = 0,
  bool Swzl = true
> struct DefaultGemm_AIU_Operand;

template <
  typename Arch,
  typename Element,
  typename Block_MN,
  typename Block_K,
  bool Swap,
  int StageStride,
  bool Swzl
> struct DefaultGemm_AIU_Operand<
  Arch,
  Element,
  false,
  Block_MN,
  Block_K,
  Swap,
  StageStride,
  Swzl
> {
  static constexpr int BlockContSize = Block_K{} * sizeof_bits<Element>::value / 8;
  static_assert(BlockContSize > 128 ? (BlockContSize % 128 == 0) : (BlockContSize % 32 == 0), "aiu_trans: block contiguous size should be multiple of 128B or 32B");
  static constexpr int AiuContByteSize = BlockContSize > 128 ? 128 : BlockContSize;
  static constexpr int AiuContElemSize = AiuContByteSize / sizeof_bits<Element>::value * 8;

  static_assert(BlockContSize <= 128 || (BlockContSize % 128 == 0));
  static constexpr int align_bytes = cute::is_same_v<Arch, cutlass::arch::PPU0015> ? 64 : 32;
#if defined(ENABLE_AIU)
  static_assert(BlockContSize % align_bytes == 0, "aiu_no_trans: block contiguous size doesn't meet alignment");
#else
  static_assert(BlockContSize % 32 == 0, "aiu_no_trans: block contiguous size should be multiple 32B");
#endif

  static constexpr bool split_on_k_10500 = AiuContElemSize < Block_K{};
  static constexpr int inst_num = cute::is_same_v<Arch, cutlass::arch::PPU0015> ?
                                  (split_on_k_10500 ? (Block_K{} / AiuContElemSize) : 1) : Block_K{} / AiuContElemSize;
  static constexpr int CUBE_H = cute::is_same_v<Arch, cutlass::arch::PPU0015> ?
                                  (split_on_k_10500 ? Block_MN{} : (Block_MN{} / inst_num)) : Block_MN{};
  static constexpr int CUBE_W = AiuContElemSize;
  static constexpr int bits_per_aiu = CUBE_H * CUBE_W * sizeof_bits<Element>::value;
  using CopyInst = cute::conditional_t<
    cute::is_same_v<Arch, cutlass::arch::PPU0015>,
    PPU0015_AIU_LOAD<cute::C<bits_per_aiu>, Element, false, CUBE_H, CUBE_W, Swzl>,
    PPU0010_AIU_LOAD<cute::C<bits_per_aiu>, Element, false, Swzl>
  >;
  using SmemCopyOp = cute::conditional_t<
    cute::is_same_v<Arch, cutlass::arch::PPU0015>,
    cute::conditional_t<StageStride == 0,
                        PPU0015_TSM_LD_SWZL<Element, CUBE_H, CUBE_W, Swap, false, inst_num>,
                        PPU0015_TSM_LD_SWZL_SmemStageStride<Element, CUBE_H, CUBE_W, Swap, false, StageStride>>,
    PPU0010_TSM_LD_SWZL<Element, CUBE_H, CUBE_W, Swap, false, inst_num>
  >;

  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<CopyInst, Element>{},
                    Layout<Shape <_1,_1>,
                           Stride<_1,_1>>{},
                    Layout<Shape <Int<CUBE_H>, Int<CUBE_W>>>{}));

  using SmemCopyAtom = Copy_Atom<SmemCopyOp, Element>;
  using SmemAtomStride = typename platform::conditional<
    split_on_k_10500 or not Swzl,
    Stride<Int<CUBE_W>, _1>,
    Stride<_1, Int<CUBE_H>>
  >::type;
  using SmemLayoutAtom = cute::conditional_t<
    cute::is_same_v<Arch, cutlass::arch::PPU0015>,
    Layout<Shape<Int<CUBE_H>, Int<CUBE_W>>, SmemAtomStride>,
    Layout<Shape<Int<CUBE_H>, Int<CUBE_W>>, Stride<Int<CUBE_W>, _1>>
  >;
};

template <
  typename Arch,
  typename Element,
  typename Block_MN,
  typename Block_K,
  bool Swap,
  int StageStride,
  bool Swzl
> struct DefaultGemm_AIU_Operand<
  Arch,
  Element,
  true,
  Block_MN,
  Block_K,
  Swap,
  StageStride,
  Swzl
> {

  static constexpr int BlockContSize = Block_MN{} * sizeof(Element);
  static constexpr int AiuContByteSize = BlockContSize > 128 ? 128 : BlockContSize;
  using AiuContElemSize = Int<AiuContByteSize / sizeof(Element)>;
  static constexpr int InstNum = Block_MN{} / AiuContElemSize{};

  static constexpr int bits_per_aiu = AiuContByteSize * 8 * Block_K{};
  static constexpr int CUBE_H = Block_K{};
  static constexpr int CUBE_W = AiuContElemSize{};

  static_assert(BlockContSize <= 128 || (BlockContSize % 128 == 0));
  static constexpr int align_bytes = cute::is_same_v<Arch, cutlass::arch::PPU0015> ? 64 : 32;
#if defined(ENABLE_AIU)
  static_assert(BlockContSize % align_bytes == 0, "aiu_trans: block contiguous size doesn't meet alignment");
#else
  static_assert(BlockContSize % 32 == 0, "aiu_trans: block contiguous size should be multiple 32B");
#endif

  using CopyInst = cute::conditional_t<
    cute::is_same_v<Arch, cutlass::arch::PPU0015>,
    PPU0015_AIU_LOAD<cute::C<bits_per_aiu>, Element, true, CUBE_H, CUBE_W, Swzl>,
    PPU0010_AIU_LOAD<cute::C<bits_per_aiu>, Element, true, Swzl>
  >;
  using SmemCopyOp = cute::conditional_t<
    cute::is_same_v<Arch, cutlass::arch::PPU0015>,
    cute::conditional_t<StageStride == 0,
                        PPU0015_TSM_LD_SWZL<Element, Block_K{}, AiuContElemSize{}, Swap, true, InstNum>,
                        PPU0015_TSM_LD_SWZL_SmemStageStride<Element, Block_K{}, AiuContElemSize{}, Swap, true, StageStride>>,
    PPU0010_TSM_LD_SWZL<Element, Block_K{}, AiuContElemSize{}, Swap, true, InstNum>
  >;

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


