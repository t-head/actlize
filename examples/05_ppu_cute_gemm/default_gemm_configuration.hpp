/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved. 
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#define CUTE_LOG_DEBUG 1

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

#define CUTE_PPU 1
#define ACOMPUTE_VERSION 10000

#ifndef ENABLE_AIU
#define ENABLE_AIU 1
#endif

#if CUTE_PPU
#include "ppu/ppu_include.hpp"
#endif

// #define USE_M8_MMA

namespace cutlass {
namespace gemm {
namespace device {
using namespace cute;

// This type is only intended to demonstrate porting 2.x kernels to 3.0
template<
  class OperatorClass, class ArchTag,
  class ElementA, class LayoutA,
  class ElementB, class LayoutB,
  class ElementC, class LayoutC,
  class ElementAccumulator>
struct DefaultGemmConfigurationToCutlass3Types {
  static_assert(sizeof(ElementA) == 0, "No valid DefaultGemmConfigurationToCutlass3Types configuration exists.");
};

///////////////////////////////////////////////////////////////////////////////

namespace detail {

template <typename Element, typename Layout, int Alignment, int SizeK>
struct DefaultGemm_TensorOpPPU0010_OperandA;

template <typename Element, typename Layout, int Alignment, int SizeK>
struct DefaultGemm_TensorOpPPU0010_OperandB;

//
// F16: 128-by-128-by-64
//

/// Operand A - Row-major (K-Major)
template <>
struct DefaultGemm_TensorOpPPU0010_OperandA<half_t, layout::RowMajor, 8, 64>
{
  // Smem
  using SmemLayoutAtom = decltype(
    composition(Swizzle<3,3,3>{},
                Layout<Shape < _8,_64>,
                       Stride<_64, _1>>{}));
  using SmemCopyAtom = Copy_Atom<PPU_U32x4_LDSM_N, half_t>;

  // Gmem
  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, half_t>{},
                    Layout<Shape <_16,_8>,
                           Stride< _8,_1>>{},
                    Layout<Shape < _1,_8>>{}));
};

/// Operand A - Column-major (M-major)
template <int SizeK>
struct DefaultGemm_TensorOpPPU0010_OperandA<half_t, layout::ColumnMajor, 8, SizeK>
{
  // Smem
  using SmemLayoutAtom = decltype(
    composition(Swizzle<3,3,3>{},
                Layout<Shape <_64, _8>,
                       Stride< _1,_64>>{}));
  using SmemCopyAtom = Copy_Atom<PPU_U16x8_LDSM_T, half_t>;

  // Gmem
  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, half_t>{},
                    Layout<Shape <_16, _8>,
                           Stride< _1,_16>>{},
                    Layout<Shape < _8, _1>>{}));
};

// Because the F32F16 TiledMMA is A-B symmetric, we can reuse the DefaultOperands

// Operand B - Column-Major (K-major)
template <int Alignment, int SizeK>
struct DefaultGemm_TensorOpPPU0010_OperandB<half_t, layout::ColumnMajor, Alignment, SizeK>
     : DefaultGemm_TensorOpPPU0010_OperandA<half_t, layout::RowMajor,    Alignment, SizeK>
{};

// Operand B - Row-Major (N-major)
template <int Alignment, int SizeK>
struct DefaultGemm_TensorOpPPU0010_OperandB<half_t, layout::RowMajor,    Alignment, SizeK>
     : DefaultGemm_TensorOpPPU0010_OperandA<half_t, layout::ColumnMajor, Alignment, SizeK>
{};

//
// F16: 128-by-128-by-32 (small k-block)
//

/// Operand A - Row-major (K-Major)
template <>
struct DefaultGemm_TensorOpPPU0010_OperandA<half_t, layout::RowMajor, 8, 32>
{
  // Smem
  using SmemLayoutAtom = decltype(
    composition(Swizzle<2,3,3>{},
                Layout<Shape < _8,_32>,
                       Stride<_32, _1>>{}));
  using SmemCopyAtom = Copy_Atom<PPU_U32x4_LDSM_N, half_t>;

  // Gmem
  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, half_t>{},
                    Layout<Shape <_32,_4>,
                           Stride< _4,_1>>{},
                    Layout<Shape < _1,_8>>{}));
};

///////////////////////////////////////////////////////////////////////////////
#if ENABLE_AIU
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

  static_assert(Block_K{} * sizeof(Element) % 32 == 0, "aiu_no_trans: block_k must be multiple of 32B");
  static constexpr int kBlockKSmem = Block_K{} % 64 == 0 ? 64 : 32;
  static constexpr int bits_per_aiu = Block_MN{} * kBlockKSmem * sizeof(Element) * 8;
  using CopyInst = PPU0010_AIU_LOAD<cute::C<bits_per_aiu>, Element, false>;

  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<CopyInst, Element>{},
                    Layout<Shape <_1,_1>,
                           Stride<_1,_1>>{},
                    Layout<Shape <Block_MN, Int<kBlockKSmem>>>{}));

#ifdef USE_M8_MMA
  static constexpr int Num = 2;
#else
  static constexpr int Num = 4;
#endif
  using SmemCopyOp = PPU0010_TSM_LD_SWZL<Element, Block_MN{}, kBlockKSmem, Swap, false, Block_K{}/kBlockKSmem, Num>;
  using SmemCopyAtom = Copy_Atom<SmemCopyOp, Element>;
  using SmemLayoutAtom = Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>;
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
  using CopyInst = PPU0010_AIU_LOAD<cute::C<bits_per_aiu>, Element, true>;

  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<CopyInst, Element>{},
                    Layout<Shape <_1,_1>,
                           Stride<_1,_1>>{},
                    Layout<Shape <AiuContElemSize,Block_K>>{}));

  using SmemCopyOp = PPU0010_TSM_LD_SWZL<Element, Block_K{}, AiuContElemSize{}, Swap, true, InstNum>;
  using SmemCopyAtom = Copy_Atom<SmemCopyOp, Element>;
  // using SmemLayoutAtom = Layout<Shape<_8, Block_K>, Stride<Block_K, _1>>;
  using SmemLayoutAtom = Layout<Shape<AiuContElemSize, Block_K>, Stride<_1, AiuContElemSize>>;
};

#endif


}

///////////////////////////////////////////////////////////////////////////////

// PPU0010 MMA F32F16
template <typename LayoutA, typename LayoutB, typename LayoutC>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassTensorOp, arch::PPU0010,
    half_t, LayoutA,
    half_t, LayoutB,
    float, LayoutC,
    float>
{
  static constexpr int blockK = 32; // support 32,64,96,128,160,192,224,256 on AIU
#if USE_M8_MMA
  static constexpr int blockM = 32;
#else
  static constexpr int blockM = 64;
#endif
  using TileShape = Shape<Int<blockM>, _64, Int<blockK>>;
  static constexpr int ThreadCount = 128;

  using TiledMma = TiledMMA<
#if CUTE_PPU
#if USE_M8_MMA
      MMA_Atom<PPU_8x16x16_F32F16F16F32_TN>,
#else
      MMA_Atom<PPU_16x16x16_F32F16F16F32_TN>,
#endif
#else
      MMA_Atom<PPU_16x8x16_F32F16F16F32_TN>,
#endif
      Layout<Shape<_2,_2,_1>>,  // 2x2x1 thread group
#if USE_M8_MMA
      Tile<_16,_32,_16>>;       // 32x32x16 MMA for LDSM, 1x2x1 value group
#else
      Tile<_32,_32,_16>>;
#endif

#if ENABLE_AIU
  using DispatchPolicy = MainloopPPUAiu<3>;
  static constexpr bool TransA = platform::is_same<LayoutA, cutlass::layout::RowMajor>::value ? false : true;
  static constexpr bool TransB = platform::is_same<LayoutB, cutlass::layout::ColumnMajor>::value ? false : true;
  using DefaultOperandA = detail::DefaultGemm_AIU_Operand<half_t, TransA, Int<blockM>, Int<blockK>, true>;
  using DefaultOperandB = detail::DefaultGemm_AIU_Operand<half_t, TransB, _64, Int<blockK>, true>;
#else
  static constexpr int kAlignmentA = 8;
  static constexpr int kAlignmentB = 8;
  using DispatchPolicy = MainloopPPU0010CpAsync<3>;
  using DefaultOperandA = detail::DefaultGemm_TensorOpPPU0010_OperandA<
    half_t, LayoutA, kAlignmentA, 32>;
  using DefaultOperandB = detail::DefaultGemm_TensorOpPPU0010_OperandB<
    half_t, LayoutB, kAlignmentB, 32>;
#endif

  // A
  using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom; // M, K
  using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;

  // B

  using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom; // N, K
  using SmemCopyAtomB = typename DefaultOperandB::SmemCopyAtom;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    half_t, TagToStrideA_t<LayoutA>,
    half_t, TagToStrideB_t<LayoutB>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::EpilogueDefault>;
};

///////////////////////////////////////////////////////////////////////////////

namespace detail {

//
// TF32: 128-by-128-by-kblock (kBlock = 16, 32)
//

/// Operand A - Row-major  (K-major) (kBlock = 32)
template <>
struct DefaultGemm_TensorOpPPU0010_OperandA<tfloat32_t, layout::RowMajor, 4, 32>
{
  // Smem
  using SmemLayoutAtom = decltype(
    composition(Swizzle<3,2,3>{},
                Layout<Shape < _8,_32>,
                       Stride<_32, _1>>{}));
  using SmemCopyAtom = Copy_Atom<PPU_U32x4_LDSM_N, tfloat32_t>;

  // Gmem
  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, tfloat32_t>{},
                    Layout<Shape <_16,_8>,
                           Stride< _8,_1>>{},
                    Layout<Shape < _1,_4>>{}));
};

/// Operand A - Row-major  (K-major) (kBlock = 16)
template <>
struct DefaultGemm_TensorOpPPU0010_OperandA<tfloat32_t, layout::RowMajor, 4, 16>
{
  // Smem
  using SmemLayoutAtom = decltype(
    composition(Swizzle<2,2,3>{},
                Layout<Shape < _8,_16>,
                       Stride<_16, _1>>{}));
  using SmemCopyAtom = Copy_Atom<PPU_U32x4_LDSM_N, tfloat32_t>;
  // Gmem
  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, tfloat32_t>{},
                    Layout<Shape <_32,_4>,
                           Stride< _4,_1>>{},
                    Layout<Shape < _1,_4>>{}));
};

/// Operand A - Column-major  (M-major)
template <int SizeK>
struct DefaultGemm_TensorOpPPU0010_OperandA<tfloat32_t, layout::ColumnMajor, 4, SizeK>
{
  // Smem
  using SmemLayoutAtom = decltype(
    composition(Swizzle<3,2,3>{},
                Layout<Shape <_32, _8>,
                       Stride< _1,_32>>{}));
  using SmemCopyAtom = Copy_Atom<UniversalCopy<tfloat32_t>, tfloat32_t>;
  // Gmem
  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, tfloat32_t>{},
                    Layout<Shape <_16, _8>,
                           Stride< _1,_16>>{},
                    Layout<Shape < _4, _1>>{}));
};

// Because the TF32 TiledMMA is A-B symmetric, we can reuse the DefaultOperands

// Operand B - Column-Major  (K-major)
template <int Alignment, int SizeK>
struct DefaultGemm_TensorOpPPU0010_OperandB<tfloat32_t, layout::ColumnMajor, Alignment, SizeK>
     : DefaultGemm_TensorOpPPU0010_OperandA<tfloat32_t, layout::RowMajor,    Alignment, SizeK>
{};

// Operand B - Row-Major  (N-major)
template <int Alignment, int SizeK>
struct DefaultGemm_TensorOpPPU0010_OperandB<tfloat32_t, layout::RowMajor,    Alignment, SizeK>
     : DefaultGemm_TensorOpPPU0010_OperandA<tfloat32_t, layout::ColumnMajor, Alignment, SizeK>
{};

}

///////////////////////////////////////////////////////////////////////////////

// PPU0010 MMA F32TF32
template <typename LayoutA, typename LayoutB, typename LayoutC>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassTensorOp, arch::PPU0010,
    tfloat32_t, LayoutA,
    tfloat32_t, LayoutB,
    float, LayoutC,
    float>
{
  static constexpr int blockK = 32;
  using TileShape = Shape<_128, _128, Int<blockK>>;
  static constexpr int ThreadCount = 128;
  using TiledMma = TiledMMA<
#if CUTE_PPU
      MMA_Atom<PPU_16x16x8_F32TF32TF32F32_TN>,
#else
      MMA_Atom<PPU_16x8x8_F32TF32TF32F32_TN>,
#endif
      Layout<Shape<_2,_2,_1>, Stride<_2, _1, _1>>, // 2x2x1 thread group
      Tile<_32,_32,_8>>;                           // 32x32x8 MMA for LDSM, 1x2x1 value group


#if ENABLE_AIU
  using DispatchPolicy = MainloopPPUAiu<3>;
  static constexpr bool TransA = platform::is_same<LayoutA, cutlass::layout::RowMajor>::value ? false : true;
  static constexpr bool TransB = platform::is_same<LayoutB, cutlass::layout::ColumnMajor>::value ? false : true;
  using DefaultOperandA = detail::DefaultGemm_AIU_Operand<float, TransA, _128, Int<blockK>, true>;
  using DefaultOperandB = detail::DefaultGemm_AIU_Operand<float, TransB, _128, Int<blockK>, true>;
#else
  static constexpr int kAlignmentA = 4;
  static constexpr int kAlignmentB = 4;
  using DispatchPolicy = MainloopPPU0010CpAsync<3>;
  using DefaultOperandA = detail::DefaultGemm_TensorOpPPU0010_OperandA<
    tfloat32_t, LayoutA, kAlignmentA, 32>;
  using DefaultOperandB = detail::DefaultGemm_TensorOpPPU0010_OperandB<
    tfloat32_t, LayoutB, kAlignmentB, 32>;
#endif

  // A
  using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom; // M, K
  using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;

  // B
  using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom; // N, K
  using SmemCopyAtomB = typename DefaultOperandB::SmemCopyAtom;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    tfloat32_t, TagToStrideA_t<LayoutA>,
    tfloat32_t, TagToStrideB_t<LayoutB>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::EpilogueDefault>;
};

///////////////////////////////////////////////////////////////////////////////
template <typename LayoutC>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassTensorOp, arch::PPU0010,
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, LayoutC,
    int32_t>
{
  static constexpr int blockK = 64;
  using TileShape = Shape<_128, _128, Int<blockK>>;
  static constexpr int ThreadCount = 128;
  using TiledMma = TiledMMA<
#if CUTE_PPU
      MMA_Atom<PPU_16x16x32_S32S8S8S32_TN>,
#else
      MMA_Atom<PPU_16x8x32_S32S8S8S32_TN>,
#endif
      Layout<Shape<_2,_2,_1>>,   // 2x2x1 thread group
      Tile<_32,_32,_32>>;        // 16x16x32 MMA for LDSM, 1x2x1 value group

#if ENABLE_AIU
  using DispatchPolicy = MainloopPPUAiu<3>;
  static constexpr bool TransA = false;
  static constexpr bool TransB = false;
  using DefaultOperandA = detail::DefaultGemm_AIU_Operand<int8_t, TransA, _128, Int<blockK>, true>;
  using DefaultOperandB = detail::DefaultGemm_AIU_Operand<int8_t, TransB, _128, Int<blockK>, true>;

  // A
  using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom; // M, K
  using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;

  // B
  using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom; // N, K
  using SmemCopyAtomB = typename DefaultOperandB::SmemCopyAtom;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;
#else
  using DispatchPolicy = MainloopPPU0010CpAsync<3>;
  // A (M,K)  K-major
  using SmemLayoutAtomA = decltype(
    composition(
      Swizzle<2,4,3>{},
      Layout<Shape <_16,_64>,
             Stride<_64, _1>>{}));
  static constexpr int kAlignmentA = 16;
  using GmemTiledCopyA = decltype(
    make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, int8_t>{},
                    Layout<Shape <_32,_4>,
                           Stride< _4,_1>>{},
                    Layout<Shape<_1,Int<kAlignmentA>>>{}));
  // LDS.32- or LDSM-based copy atom
  // using SmemCopyAtomA = Copy_Atom<DefaultCopy, uint8_t>;
  using SmemCopyAtomA = Copy_Atom<PPU_U32x4_LDSM_N, uint8_t>;  // LDSM works

  // B (N,K)  K-major
  using SmemLayoutAtomB = decltype(
    composition(
      Swizzle<2,4,3>{},
      Layout<Shape <_16,_64>,
             Stride<_64, _1>>{}));
  static constexpr int kAlignmentB = 16;
  using GmemTiledCopyB = decltype(
    make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, int8_t>{},
                    Layout<Shape <_32,_4>,
                           Stride< _4,_1>>{},
                    Layout<Shape<_1,Int<kAlignmentB>>>{}));

  // LDS.32- or LDSM-based copy atom
  // using SmemCopyAtomB = Copy_Atom<DefaultCopy, uint32_t>;
  using SmemCopyAtomB = Copy_Atom<PPU_U32x4_LDSM_N, uint8_t>;  // LDSM works
#endif

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    int8_t, TagToStrideA_t<cutlass::layout::RowMajor>,
    int8_t, TagToStrideB_t<cutlass::layout::ColumnMajor>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<int32_t, 1, int32_t, int32_t>,
    cutlass::gemm::EpilogueDefault>;
};

///////////////////////////////////////////////////////////////////////////////
//////////////////////////// SIMT TWO STAGE ///////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

namespace detail {

template <typename Element, typename Layout, int ThreadCount, int ShapeM, int ShapeK>
struct DefaultGemm_Simt_OperandA;

///////////////////////////////////////////////////////////////////////////////

template <typename Element>
struct DefaultGemm_Simt_OperandA<Element, layout::ColumnMajor, 256, 128, 8>
{
  using SmemLayoutAtom = Layout<Shape <_128,  _8>,
                                Stride<  _1,_128>>;

  using SmemCopyAtom = Copy_Atom<DefaultCopy, Element>;

  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<Element>, Element>{},
                    Layout<Shape <_32, _8>,
                           Stride< _1,_32>>{},
                    Layout<Shape<_1,_1>>{}));
};

template <typename Element>
struct DefaultGemm_Simt_OperandA<Element, layout::RowMajor, 256, 128, 8>
{
  using SmemLayoutAtom = Layout<Shape <_128,          _8>,
                                Stride<  _1,Int<128 + 4>>>;   // Padded

  using SmemCopyAtom = Copy_Atom<DefaultCopy, Element>;

  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<Element>, Element>{},
                    Layout<Shape <_32, _8>,
                           Stride< _8, _1>>{},
                    Layout<Shape<_1,_1>>{}));

};

template <typename Element, typename Layout, int ThreadCount, int ShapeN, int ShapeK>
struct DefaultGemm_Simt_OperandB;

template <typename Element, int ThreadCount, int ShapeN, int ShapeK>
struct DefaultGemm_Simt_OperandB<Element, layout::ColumnMajor, ThreadCount, ShapeN, ShapeK>
     : DefaultGemm_Simt_OperandA<Element, layout::RowMajor,    ThreadCount, ShapeN, ShapeK> {};

template <typename Element, int ThreadCount, int ShapeN, int ShapeK>
struct DefaultGemm_Simt_OperandB<Element, layout::RowMajor,    ThreadCount, ShapeN, ShapeK>
     : DefaultGemm_Simt_OperandA<Element, layout::ColumnMajor, ThreadCount, ShapeN, ShapeK> {};

} // end namespace detail

// SIMT Two Stage
template <
  class ArchTag,
  class ElementA, class LayoutA,
  class ElementB, class LayoutB,
  class ElementC, class LayoutC,
  class ElementAccumulator>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassSimt, ArchTag,
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator>
{
  using TileShape = Shape<_128, _128, _8>;
  static constexpr int ThreadCount = 256;
  using DispatchPolicy = MainloopPPU0010TwoStage;
  using TiledMma = TiledMMA<
      MMA_Atom<UniversalFMA<ElementAccumulator, ElementA, ElementB, ElementC>>,
      Layout<Shape<_16, _16, _1>>>;

  // A
  static constexpr int kAlignmentA = 1;
  using DefaultOperandA = detail::DefaultGemm_Simt_OperandA<ElementA, LayoutA, ThreadCount, 128, 8>;
  using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom;
  using SmemCopyAtomA   = typename DefaultOperandA::SmemCopyAtom;
  using GmemTiledCopyA  = typename DefaultOperandA::GmemTiledCopy;

  // B
  static constexpr int kAlignmentB = 1;
  using DefaultOperandB = detail::DefaultGemm_Simt_OperandB<ElementB, LayoutB, ThreadCount, 128, 8>;
  using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom;
  using SmemCopyAtomB   = typename DefaultOperandB::SmemCopyAtom;
  using GmemTiledCopyB  = typename DefaultOperandB::GmemTiledCopy;

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    ElementA, TagToStrideA_t<LayoutA>,
    ElementB, TagToStrideB_t<LayoutB>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::EpilogueDefault>;
};


//
// DP4A - int8    Proof-of-concept
//

// SIMT Two Stage TN - idp4a
template <
  class ArchTag,
  class ElementC, class LayoutC>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassSimt, ArchTag,
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    ElementC, LayoutC,
    int32_t>
{
  using TileShape = Shape<_128, _128, _32>;
  static constexpr int ThreadCount = 256;
  using DispatchPolicy = MainloopPPU0010TwoStage;
  // NOTE: permuting MMA M mode lets us generate 128b smem loads (LDS.128) but has worst case bank conflicts
  using TiledMma = TiledMMA<
      MMA_Atom<PPU_DP4A>,
      Layout<Shape<_16,_16,_1>>>;  // Tile of atoms (threads)

  // A (M,K)  K-major
  using ElementA = int8_t;
  // 40% from regular M and N major layout
  // using SmemLayoutAtomA = Layout<Shape <_128,_32>,
  //                                Stride<  _1,_128>>;
  // 80% from interleaved layouts
  using SmemLayoutAtomA = Layout<Shape <_128, Shape <_4,  _8>>,
                                 Stride<  _4, Stride<_1,_512>>>;

  using SmemCopyAtomA = Copy_Atom<DefaultCopy, ElementA>;
  static constexpr int kAlignmentA = 4;
  using GmemTiledCopyA = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint32_t>, ElementA>{},
                    Layout<Shape <_32,_8>,
                           Stride< _8,_1>>{},
                    Layout<Shape < _1,_4>>{}));

  // B (N,K)  K-major
  using ElementB = int8_t;
  // 40% from regular M and N major layout
  // using SmemLayoutAtomB = Layout<Shape <_128,_32>,
  //                                Stride<  _1,_128>>;
  // 80% from interleaved layouts
  using SmemLayoutAtomB = Layout<Shape <_128, Shape <_4,  _8>>,
                                 Stride<  _4, Stride<_1,_512>>>;

  using SmemCopyAtomB = Copy_Atom<DefaultCopy, ElementB>;
  static constexpr int kAlignmentB = 4;
  using GmemTiledCopyB = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint32_t>, ElementB>{},
                    Layout<Shape <_32,_8>,
                           Stride< _8,_1>>{},
                    Layout<Shape < _1,_4>>{}));

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    ElementA, TagToStrideA_t<cutlass::layout::RowMajor>,
    ElementB, TagToStrideB_t<cutlass::layout::ColumnMajor>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<ElementC, 1, int32_t, int32_t>,
    cutlass::gemm::EpilogueDefault>;
};

///////////////////////////////////////////////////////////////////////////////

// SIMT Two Stage NN - idp4a
template <
  class ArchTag,
  class ElementC, class LayoutC>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassSimt, ArchTag,
    int8_t, cutlass::layout::ColumnMajor,
    int8_t, cutlass::layout::ColumnMajor,
    ElementC, LayoutC,
    int32_t>
{
  using TileShape = Shape<_128, _128, _32>;
  static constexpr int ThreadCount = 256;

  using DispatchPolicy = MainloopPPU0010TwoStage;

  using TiledMma = TiledMMA<
      MMA_Atom<PPU_DP4A>,
      Layout<Shape<_16, _16, _1>>>;

  // A (M,K)  M-major
  using ElementA = int8_t;
  using SmemLayoutAtomA = Layout<Shape <_128, Shape <_4,  _8>>,
                                 Stride<  _4, Stride<_1,_512>>>;
  using SmemCopyAtomA = Copy_Atom<DefaultCopy, ElementA>;
  static constexpr int kAlignmentA = 1;
  using GmemTiledCopyA = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint8_t>, ElementA>{},
                    Layout<Shape <_32, _8>,
                           Stride< _1,_32>>{},
                    Layout<Shape < _1, _1>>{}));

  // B (N,K)  K-major
  using ElementB = int8_t;
  using SmemLayoutAtomB = Layout<Shape <_128, Shape <_4,  _8>>,
                                 Stride<  _4, Stride<_1,_512>>>;
  using SmemCopyAtomB = Copy_Atom<DefaultCopy, ElementB>;
  static constexpr int kAlignmentB = 4;
  using GmemTiledCopyB = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint32_t>, ElementB>{},
                    Layout<Shape <_32,_8>,
                           Stride< _8,_1>>{},
                    Layout<Shape < _1,_4>>{}));

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    ElementA, TagToStrideA_t<cutlass::layout::ColumnMajor>,
    ElementB, TagToStrideB_t<cutlass::layout::ColumnMajor>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<ElementC, 1, int32_t, int32_t>,
    cutlass::gemm::EpilogueDefault>;
};

///////////////////////////////////////////////////////////////////////////////

// SIMT Two Stage NT - idp4a
template <
  class ArchTag,
  class ElementC, class LayoutC>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassSimt, ArchTag,
    int8_t, cutlass::layout::ColumnMajor,
    int8_t, cutlass::layout::RowMajor,
    ElementC, LayoutC,
    int32_t>
{
  using TileShape = Shape<_128, _128, _32>;
  static constexpr int ThreadCount = 256;
  using DispatchPolicy = MainloopPPU0010TwoStage;
  using TiledMma = TiledMMA<
      MMA_Atom<PPU_DP4A>,
      Layout<Shape<_16, _16, _1>>>;

  // A (M,K)  M-major
  using ElementA = int8_t;
  using SmemLayoutAtomA = Layout<Shape <_128, Shape <_4,  _8>>,
                                 Stride<  _4, Stride<_1,_512>>>;
  using SmemCopyAtomA = Copy_Atom<DefaultCopy, ElementA>;
  static constexpr int kAlignmentA = 1;
  using GmemTiledCopyA = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint8_t>, ElementA>{},
                    Layout<Shape <_32, _8>,
                           Stride< _1,_32>>{},
                    Layout<Shape < _1, _1>>{}));

  // B (N,K)  N-major
  using ElementB = int8_t;
  using SmemLayoutAtomB = Layout<Shape <_128, Shape <_4,  _8>>,
                                 Stride<  _4, Stride<_1,_512>>>;
  using SmemCopyAtomB = Copy_Atom<DefaultCopy, ElementB>;
  static constexpr int kAlignmentB = 1;
  using GmemTiledCopyB = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint8_t>, ElementB>{},
                    Layout<Shape <_32, _8>,
                           Stride< _1,_32>>{},
                    Layout<Shape < _1, _1>>{}));

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    ElementA, TagToStrideA_t<cutlass::layout::ColumnMajor>,
    ElementB, TagToStrideB_t<cutlass::layout::RowMajor>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<ElementC, 1, int32_t, int32_t>,
    cutlass::gemm::EpilogueDefault>;
};

///////////////////////////////////////////////////////////////////////////////

// SIMT Two Stage TT - idp4a
template <
  class ArchTag,
  class ElementC, class LayoutC>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassSimt, ArchTag,
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::RowMajor,
    ElementC, LayoutC,
    int32_t>
{
  using TileShape = Shape<_128, _128, _32>;
  static constexpr int ThreadCount = 256;
  using DispatchPolicy = MainloopPPU0010TwoStage;
  using TiledMma = TiledMMA<
      MMA_Atom<PPU_DP4A>,
      Layout<Shape<_16, _16, _1>>>;

  // A (M,K)  K-major
  using ElementA = int8_t;
  using SmemLayoutAtomA = Layout<Shape <_128, Shape <_4,  _8>>,
                                 Stride<  _4, Stride<_1,_512>>>;
  using SmemCopyAtomA = Copy_Atom<DefaultCopy, ElementA>;
  static constexpr int kAlignmentA = 4;
  using GmemTiledCopyA = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint32_t>, ElementA>{},
                    Layout<Shape <_32,_8>,
                           Stride< _8,_1>>{},
                    Layout<Shape < _1,_4>>{}));

  // B (N,K)  N-major
  using ElementB = int8_t;
  using SmemLayoutAtomB = Layout<Shape <_128, Shape <_4,  _8>>,
                                 Stride<  _4, Stride<_1,_512>>>;
  using SmemCopyAtomB = Copy_Atom<DefaultCopy, ElementB>;
  static constexpr int kAlignmentB = 1;
  using GmemTiledCopyB = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint8_t>, ElementB>{},
                    Layout<Shape <_32, _8>,
                           Stride< _1,_32>>{},
                    Layout<Shape < _1, _1>>{}));

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    ElementA, TagToStrideA_t<cutlass::layout::RowMajor>,
    ElementB, TagToStrideB_t<cutlass::layout::RowMajor>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<ElementC, 1, int32_t, int32_t>,
    cutlass::gemm::EpilogueDefault>;
};

///////////////////////////////////////////////////////////////////////////////
/////////////////////////// SIMT MULTI STAGE //////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// SIMT Multi Stage NT
template <
  class ElementA,
  class ElementB,
  class ElementC, class LayoutC,
  class ElementAccumulator>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassSimt, arch::PPU0010,
    ElementA, cutlass::layout::ColumnMajor,
    ElementB, cutlass::layout::RowMajor,
    ElementC, LayoutC,
    ElementAccumulator>
{
  using TileShape = Shape<_128, _128, _16>;
  static constexpr int ThreadCount = 256;
  using DispatchPolicy = MainloopPPU0010CpAsync<3>;
  using TiledMma = TiledMMA<
      MMA_Atom<UniversalFMA<ElementAccumulator, ElementA, ElementB, ElementC>>,
      Layout<Shape<_16, _16, _1>>,                            // 16x16x1 thread group
      Tile<Layout<Shape<_16,_2>,Stride<_2,_1>>,               // 32x32x1 MMA with perm for load vectorization
           Layout<Shape<_16,_2>,Stride<_2,_1>>,Underscore>>;

  // A (M,K)  M-major
  using SmemLayoutAtomA = Layout<Shape<_128,_16>>;
  using SmemCopyAtomA = Copy_Atom<DefaultCopy, ElementA>;
  static constexpr int kAlignmentA = 2;
  using AlignmentTypeA = cute::uint_byte_t<static_cast<int>(sizeof(ElementA)) * kAlignmentA>;
  using GmemTiledCopyA = decltype(
    make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEALWAYS<AlignmentTypeA>, ElementA>{},
                    Layout<Shape<_32,_8>>{},
                    Layout<Shape< _2,_1>>{}));

  // B (N,K)  N-major
  using SmemLayoutAtomB = Layout<Shape<_128,_16>>;
  using SmemCopyAtomB = Copy_Atom<DefaultCopy, ElementB>;
  static constexpr int kAlignmentB = 2;
  using AlignmentTypeB = cute::uint_byte_t<static_cast<int>(sizeof(ElementB)) * kAlignmentB>;
  using GmemTiledCopyB = decltype(
    make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEALWAYS<AlignmentTypeB>, ElementB>{},
                    Layout<Shape<_32,_8>>{},
                    Layout<Shape< _2,_1>>{}));

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    ElementA, TagToStrideA_t<cutlass::layout::ColumnMajor>,
    ElementB, TagToStrideB_t<cutlass::layout::RowMajor>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::EpilogueDefault>;
};

///////////////////////////////////////////////////////////////////////////////

// SIMT Multi Stage TN
template <
  class ElementA,
  class ElementB,
  class ElementC, class LayoutC,
  class ElementAccumulator>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassSimt, arch::PPU0010,
    ElementA, cutlass::layout::RowMajor,
    ElementB, cutlass::layout::ColumnMajor,
    ElementC, LayoutC,
    ElementAccumulator>
{
  using TileShape = Shape<_128, _128, _16>;
  static constexpr int ThreadCount = 256;
  using DispatchPolicy = MainloopPPU0010CpAsync<3>;
  using TiledMma = TiledMMA<
      MMA_Atom<UniversalFMA<ElementAccumulator, ElementA, ElementB, ElementC>>,
      Layout<Shape<_16, _16, _1>>>;

  // A (M,K)  K-major
  using SmemLayoutAtomA = Layout<Shape <_128,          _16>,
                                 Stride<  _1, Int<128 + 1>>>;  // Padded by kAlignmentA
  using SmemCopyAtomA = Copy_Atom<DefaultCopy, ElementA>;
  static constexpr int kAlignmentA = 1;
  using GmemTiledCopyA = decltype(
    make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEALWAYS<ElementA>, ElementA>{},
                    Layout<Shape <_16,_16>,
                           Stride<_16, _1>>{}));

  // B (N,K)  K-major
  using SmemLayoutAtomB = Layout<Shape <_128,          _16>,
                                 Stride<  _1, Int<128 + 1>>>;  // Padded by kAlignmentB
  using SmemCopyAtomB = Copy_Atom<DefaultCopy, ElementB>;
  static constexpr int kAlignmentB = 1;
  using GmemTiledCopyB = decltype(
    make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEALWAYS<ElementB>, ElementB>{},
                    Layout<Shape <_16,_16>,
                           Stride<_16, _1>>{}));

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    ElementA, TagToStrideA_t<cutlass::layout::RowMajor>,
    ElementB, TagToStrideB_t<cutlass::layout::ColumnMajor>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::EpilogueDefault>;
};

///////////////////////////////////////////////////////////////////////////////

// SIMT Multi Stage NN
template <
  class ElementA,
  class ElementB,
  class ElementC, class LayoutC,
  class ElementAccumulator>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassSimt, arch::PPU0010,
    ElementA, cutlass::layout::ColumnMajor,
    ElementB, cutlass::layout::ColumnMajor,
    ElementC, LayoutC,
    ElementAccumulator>
{
  using TileShape = Shape<_128, _128, _16>;
  static constexpr int ThreadCount = 256;
  using DispatchPolicy = MainloopPPU0010CpAsync<3>;
  using TiledMma = TiledMMA<
      MMA_Atom<UniversalFMA<ElementAccumulator, ElementA, ElementB, ElementC>>,
      Layout<Shape<_16, _16, _1>>,                                      // 16x16x1 thread group
      Tile<Layout<Shape<_16,_2>,Stride<_2,_1>>,Underscore,Underscore>>; // 32x16x1 MMA with perm for load vectorization

  // A (M,K)  M-major
  using SmemLayoutAtomA = Layout<Shape<_128,_16>>;
  using SmemCopyAtomA = Copy_Atom<DefaultCopy, ElementA>;
  static constexpr int kAlignmentA = 2;
  using AlignmentTypeA = cute::uint_byte_t<static_cast<int>(sizeof(ElementA)) * kAlignmentA>;
  using GmemTiledCopyA = decltype(
    make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEALWAYS<AlignmentTypeA>, ElementA>{},
                    Layout<Shape<_32,_8>>{},
                    Layout<Shape< _2,_1>>{}));

  // B (N,K)  K-major
  using SmemLayoutAtomB = Layout<Shape <_128,          _16>,
                                 Stride<  _1, Int<128 + 1>>>;  // Padded by kAlignmentB
  using SmemCopyAtomB = Copy_Atom<DefaultCopy, ElementB>;
  static constexpr int kAlignmentB = 1;
  using GmemTiledCopyB = decltype(
    make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEALWAYS<ElementB>, ElementB>{},
                    Layout<Shape <_16,_16>,
                           Stride<_16, _1>>{}));

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    ElementA, TagToStrideA_t<cutlass::layout::ColumnMajor>,
    ElementB, TagToStrideB_t<cutlass::layout::ColumnMajor>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::EpilogueDefault>;
};

///////////////////////////////////////////////////////////////////////////////

// SIMT Multi Stage TT
template <
  class ElementA,
  class ElementB,
  class ElementC, class LayoutC,
  class ElementAccumulator>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassSimt, arch::PPU0010,
    ElementA, cutlass::layout::RowMajor,
    ElementB, cutlass::layout::RowMajor,
    ElementC, LayoutC,
    ElementAccumulator>
{
  using TileShape = Shape<_128, _128, _16>;
  static constexpr int ThreadCount = 256;
  using DispatchPolicy = MainloopPPU0010CpAsync<3>;
  using TiledMma = TiledMMA<
      MMA_Atom<UniversalFMA<ElementAccumulator, ElementA, ElementB, ElementC>>,
      Layout<Shape<_16, _16, _1>>,                                      // 16x16x1 thread group
      Tile<Underscore,Layout<Shape<_16,_2>,Stride<_2,_1>>,Underscore>>; // 16x32x1 MMA with perm for load vectorization

  // A (M,K)  K-major
  using SmemLayoutAtomA = Layout<Shape <_128,          _16>,
                                 Stride<  _1, Int<128 + 1>>>;  // Padded by kAlignmentA
  using SmemCopyAtomA = Copy_Atom<DefaultCopy, ElementA>;
  static constexpr int kAlignmentA = 1;
  using GmemTiledCopyA = decltype(
    make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEALWAYS<ElementA>, ElementA>{},
                    Layout<Shape <_16,_16>,
                           Stride<_16, _1>>{}));

  // B (N,K)  N-major
  using SmemLayoutAtomB = Layout<Shape <_128,_16>>;
  using SmemCopyAtomB = Copy_Atom<DefaultCopy, ElementB>;
  static constexpr int kAlignmentB = 2;
  using AlignmentTypeB = cute::uint_byte_t<static_cast<int>(sizeof(ElementB)) * kAlignmentB>;
  using GmemTiledCopyB = decltype(
    make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEALWAYS<AlignmentTypeB>, ElementB>{},
                    Layout<Shape<_32,_8>>{},
                    Layout<Shape< _2,_1>>{}));

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    ElementA, TagToStrideA_t<cutlass::layout::RowMajor>,
    ElementB, TagToStrideB_t<cutlass::layout::RowMajor>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::EpilogueDefault>;
};

} // namespace device
} // namespace gemm
} // namespace cutlass

