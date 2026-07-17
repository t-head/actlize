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

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm.h"

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"

#include "cutlass/numeric_types.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "cute/atom/copy_atom.hpp"

#include "cutlass/gemm/gemm.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/layout/layout.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "ppu/cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "ppu/cutlass/gemm/kernel/ppu_aiu_gemm_parallel.hpp"
#include "ppu/cutlass/epilogue/thread/conversion_op.h"
#include "cutlass/epilogue/collective/ppu_epilogue_vectorized.hpp"
#include "ppu/cutlass/epilogue/collective/ppu_epilogue_vectorized_parallel.hpp"
#include "ppu/cutlass/epilogue/collective/ppu_epilogue_vectorized_evt.hpp"

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"


#include "ppu/cute/arch/mma_ppu.hpp"
#include "ppu/cute/arch/mma_ppu0015.hpp"

#include "ppu/cute/arch/copy_ppu0015_aiu.hpp"
#include "cute/arch/mma_ppu.hpp"
#include "ppu/gemm/config/gemm_configs.hpp"

#include "ppu/cute/algorithm/functional.hpp"

#include "ppu/ppu_include.hpp"
namespace cutlass {
namespace gemm {
namespace config {
using namespace cute;

template<
 typename ElementA,
 typename LayoutA,
 int AlignmentA,
 typename ElementB,
 typename LayoutB,
 int AlignmentB,
 typename ElementAcc,
 typename ElementC,
 int AlignmentC,
 typename ElementCompute,
 int BlockM,
 int BlockN,
 int BlockK,
 int WarpM,
 int WarpN,
 int WarpK,
 int Stage,
 typename EpilogueOp,
 bool UseAiu,
 bool ParallelSplitk = false,
 bool UseEpilogueEvt = true /*cutlass gemm static will not use evt to avoid change device api*/
> struct GemmKernelConfig;

// normal gemm
template<
 typename ElementA,
 typename LayoutA,
 int AlignmentA,
 typename ElementB,
 typename LayoutB,
 int AlignmentB,
 typename ElementAcc,
 typename ElementC,
 int AlignmentC,
 typename ElementCompute,
 int BlockM,
 int BlockN,
 int BlockK,
 int WarpM,
 int WarpN,
 int WarpK,
 int Stage,
 typename EpilogueOp,
 bool UseEpilogueEvt
> struct GemmKernelConfig <
  ElementA,
  LayoutA,
  AlignmentA,
  ElementB,
  LayoutB,
  AlignmentB,
  ElementAcc,
  ElementC,
  AlignmentC,
  ElementCompute,
  BlockM,
  BlockN,
  BlockK,
  WarpM,
  WarpN,
  WarpK,
  Stage,
  EpilogueOp,
  false,
  false,
  UseEpilogueEvt
> {

  using WarpOnM = Int<BlockM / WarpM>;
  using WarpOnN = Int<BlockN / WarpN>;
  static constexpr int ThreadNum = WarpOnM() * WarpOnN() * 32;
  static constexpr bool TransA = platform::is_same<LayoutA, cutlass::layout::RowMajor>::value ? false : true;
  static constexpr bool TransB = platform::is_same<LayoutB, cutlass::layout::ColumnMajor>::value ? false : true;

  static constexpr int MaxAlignmentA = 16 / sizeof(ElementA);
  static constexpr int MaxAlignmentB = 16 / sizeof(ElementB);

  using TransformA = typename platform::conditional<
    platform::is_same<ElementA, float>::value && platform::is_same<ElementCompute, cutlass::tfloat32_t>::value,
    cute::convert<cutlass::tfloat32_t>,
    cute::identity
  >::type;

  using TransformB = typename platform::conditional<
    platform::is_same<ElementB, float>::value && platform::is_same<ElementCompute, cutlass::tfloat32_t>::value,
    cute::convert<cutlass::tfloat32_t>,
    cute::identity
  >::type;

  using DispatchPolicy = typename platform::conditional<
    (AlignmentA < MaxAlignmentA || AlignmentB < MaxAlignmentB) && platform::min(sizeof(ElementA), sizeof(ElementB)) < 4,
    cutlass::gemm::MainloopPPU0010TwoStageLdmatrix,
    cutlass::gemm::MainloopPPU0010CpAsync<Stage>
  >::type;

  using CopyInstA = typename platform::conditional<
    AlignmentA < MaxAlignmentA,
    UniversalCopy<ElementA>,
    PPU_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>
  >::type;

  using CopyInstB = typename platform::conditional<
    AlignmentB < MaxAlignmentB,
    UniversalCopy<ElementB>,
    PPU_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>
  >::type;

  using GemmOperandA = DefaultGemm_TensorOpPPU0010_Operand<ElementA, TransA, AlignmentA, typename platform::conditional<TransA, Int<BlockM>, Int<BlockK>>::type, ThreadNum, CopyInstA>;
  using GemmOperandB = DefaultGemm_TensorOpPPU0010_Operand<ElementB, TransB, AlignmentB, typename platform::conditional<TransB, Int<BlockN>, Int<BlockK>>::type, ThreadNum, CopyInstB>;

  using MmaInst = typename GetMmaInst<ElementCompute>::type;

  static constexpr int MmaInstOnN = 16 / size<1>(typename MMA_Traits<MmaInst>::Shape_MNK{});

  using TiledMma = cute::TiledMMA<
      cute::MMA_Atom<MmaInst>,
      cute::Layout<Shape<WarpOnM, WarpOnN, _1>>,
      cute::Layout<Shape<_1, Int<MmaInstOnN>, _1>>>;

  // ElemA/B and LayoutA/B is already transfered
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveMma<
    DispatchPolicy, Shape<Int<BlockM>, Int<BlockN>, Int<BlockK>>,
    ElementA, cutlass::detail::TagToStrideA_t<LayoutA>,
    ElementB, cutlass::detail::TagToStrideB_t<LayoutB>,
    TiledMma,
    typename GemmOperandA::GmemTiledCopy, typename GemmOperandA::SmemLayoutAtom, typename GemmOperandA::SmemCopyAtom, TransformA,
    typename GemmOperandB::GmemTiledCopy, typename GemmOperandB::SmemLayoutAtom, typename GemmOperandB::SmemCopyAtom, TransformB
  >;

  using EpilogueCopyInst = AutoVectorizingCopyWithAssumedAlignment<AlignmentC * sizeof(ElementC) * 8>;
  using GemmEpilogueConfiguration = DefaultGemm_Epilogue_Configuration<EpilogueCopyInst, ElementC, ElementAcc, AlignmentC, Int<BlockM>, Int<BlockN>, WarpOnM, ThreadNum>;

  template <typename... T>
  using ChooseEpilogue = typename platform::conditional<
    UseEpilogueEvt,
    cutlass::epilogue::collective::EpilogueEvt<T...>,
    cutlass::epilogue::collective::Epilogue<T...>,
  >::type;

  using CollectiveEpilogue = ChooseEpilogue<
    cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>,
    cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>,
    EpilogueOp,
    typename GemmEpilogueConfiguration::SmemLayoutO,
    Copy_Atom<EpilogueCopyInst,ElementAcc>,
    typename GemmEpilogueConfiguration::GmemTiledCopyO,
    Copy_Atom<EpilogueCopyInst,ElementC>
  >;

  using GemmKernel = typename cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue
  >;
};


// aiu gemm
template<
 typename ElementA,
 typename LayoutA,
 typename ElementB,
 typename LayoutB,
 typename ElementAcc,
 typename ElementC,
 int AlignmentC,
 typename ElementCompute,
 int BlockM,
 int BlockN,
 int BlockK,
 int WarpM,
 int WarpN,
 int WarpK,
 int Stage,
 typename EpilogueOp
> struct GemmKernelConfig <
  ElementA,
  LayoutA,
  1,
  ElementB,
  LayoutB,
  1,
  ElementAcc,
  ElementC,
  AlignmentC,
  ElementCompute,
  BlockM,
  BlockN,
  BlockK,
  WarpM,
  WarpN,
  WarpK,
  Stage,
  EpilogueOp,
  true,
  false
> {

  using WarpOnM = Int<BlockM / WarpM>;
  using WarpOnN = Int<BlockN / WarpN>;
  static constexpr int ThreadNum = WarpOnM() * WarpOnN() * 32;
  static constexpr bool TransA = platform::is_same<LayoutA, cutlass::layout::RowMajor>::value ? false : true;
  static constexpr bool TransB = platform::is_same<LayoutB, cutlass::layout::ColumnMajor>::value ? false : true;

  using DispatchPolicy = cutlass::gemm::MainloopPPUAiu<Stage>;

  using GemmOperandA = DefaultGemm_AIU_Operand<ElementA, TransA, Int<BlockM>, Int<BlockK>, false>;
  using GemmOperandB = DefaultGemm_AIU_Operand<ElementB, TransB, Int<BlockN>, Int<BlockK>, true>;

  using TransformA = typename platform::conditional<
    platform::is_same<ElementA, float>::value && platform::is_same<ElementCompute, cutlass::tfloat32_t>::value,
    cute::convert<cutlass::tfloat32_t>,
    cute::identity
  >::type;

  using TransformB = typename platform::conditional<
    platform::is_same<ElementB, float>::value && platform::is_same<ElementCompute, cutlass::tfloat32_t>::value,
    cute::convert<cutlass::tfloat32_t>,
    cute::identity
  >::type;

  using MmaInst = typename GetAiuMmaInst<ElementCompute>::type;

  static constexpr int MmaInstOnN = 16 / size<1>(typename MMA_Traits<MmaInst>::Shape_MNK{});

  using TiledMma = cute::TiledMMA<
      cute::MMA_Atom<MmaInst>,
      cute::Layout<Shape<WarpOnM, WarpOnN, _1>>,
      cute::Layout<Shape<_1, Int<MmaInstOnN>, _1>>>;

  // ElemA/B and LayoutA/B is already transfered
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveMma<
    DispatchPolicy, Shape<Int<BlockM>, Int<BlockN>, Int<BlockK>>,
    ElementA, cutlass::detail::TagToStrideA_t<LayoutA>,
    ElementB, cutlass::detail::TagToStrideB_t<LayoutB>,
    TiledMma,
    typename GemmOperandA::GmemTiledCopy, typename GemmOperandA::SmemLayoutAtom, typename GemmOperandA::SmemCopyAtom, TransformA,
    typename GemmOperandB::GmemTiledCopy, typename GemmOperandB::SmemLayoutAtom, typename GemmOperandB::SmemCopyAtom, TransformB
  >;

  using EpilogueCopyInst = AutoVectorizingCopyWithAssumedAlignment<AlignmentC * sizeof(ElementC) * 8>;
  using GemmEpilogueConfiguration = DefaultGemm_Epilogue_Configuration<EpilogueCopyInst, ElementC, ElementAcc, AlignmentC, Int<BlockM>, Int<BlockN>, WarpOnM, ThreadNum>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::EpilogueEvt<
    cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>,
    cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>,
    EpilogueOp,
    typename GemmEpilogueConfiguration::SmemLayoutO,
    Copy_Atom<EpilogueCopyInst,ElementAcc>,
    typename GemmEpilogueConfiguration::GmemTiledCopyO,
    Copy_Atom<EpilogueCopyInst,ElementC>
  >;

  using GemmKernel = typename cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue
  >;
};

// aiu gemm parallel
template<
 typename ElementA,
 typename LayoutA,
 typename ElementB,
 typename LayoutB,
 typename ElementAcc,
 typename ElementC,
 int AlignmentC,
 typename ElementCompute,
 int BlockM,
 int BlockN,
 int BlockK,
 int WarpM,
 int WarpN,
 int WarpK,
 int Stage,
 typename EpilogueOp
> struct GemmKernelConfig <
  ElementA,
  LayoutA,
  1,
  ElementB,
  LayoutB,
  1,
  ElementAcc,
  ElementC,
  AlignmentC,
  ElementCompute,
  BlockM,
  BlockN,
  BlockK,
  WarpM,
  WarpN,
  WarpK,
  Stage,
  EpilogueOp,
  true,
  true
> {

  using WarpOnM = Int<BlockM / WarpM>;
  using WarpOnN = Int<BlockN / WarpN>;
  static constexpr int ThreadNum = WarpOnM() * WarpOnN() * 32;
  static constexpr bool TransA = platform::is_same<LayoutA, cutlass::layout::RowMajor>::value ? false : true;
  static constexpr bool TransB = platform::is_same<LayoutB, cutlass::layout::ColumnMajor>::value ? false : true;

  using DispatchPolicy = cutlass::gemm::MainloopPPUAiu<Stage>;

  using GemmOperandA = DefaultGemm_AIU_Operand<ElementA, TransA, Int<BlockM>, Int<BlockK>, false>;
  using GemmOperandB = DefaultGemm_AIU_Operand<ElementB, TransB, Int<BlockN>, Int<BlockK>, true>;

  using TransformA = typename platform::conditional<
    platform::is_same<ElementA, float>::value && platform::is_same<ElementCompute, cutlass::tfloat32_t>::value,
    cute::convert<cutlass::tfloat32_t>,
    cute::identity
  >::type;

  using TransformB = typename platform::conditional<
    platform::is_same<ElementB, float>::value && platform::is_same<ElementCompute, cutlass::tfloat32_t>::value,
    cute::convert<cutlass::tfloat32_t>,
    cute::identity
  >::type;

  using MmaInst = typename GetAiuMmaInst<ElementCompute>::type;

  static constexpr int MmaInstOnN = 16 / size<1>(typename MMA_Traits<MmaInst>::Shape_MNK{});

  using TiledMma = cute::TiledMMA<
      cute::MMA_Atom<MmaInst>,
      cute::Layout<Shape<WarpOnM, WarpOnN, _1>>,
      cute::Layout<Shape<_1, Int<MmaInstOnN>, _1>>>;

  // ElemA/B and LayoutA/B is already transfered
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveMma<
    DispatchPolicy, Shape<Int<BlockM>, Int<BlockN>, Int<BlockK>>,
    ElementA, cutlass::detail::TagToStrideA_t<LayoutA>,
    ElementB, cutlass::detail::TagToStrideB_t<LayoutB>,
    TiledMma,
    typename GemmOperandA::GmemTiledCopy, typename GemmOperandA::SmemLayoutAtom, typename GemmOperandA::SmemCopyAtom, TransformA,
    typename GemmOperandB::GmemTiledCopy, typename GemmOperandB::SmemLayoutAtom, typename GemmOperandB::SmemCopyAtom, TransformB
  >;

  using EpilogueCopyInst = AutoVectorizingCopyWithAssumedAlignment<AlignmentC * sizeof(ElementC) * 8>;
  using GemmEpilogueConfiguration = DefaultGemm_Epilogue_Configuration<EpilogueCopyInst, ElementC, ElementAcc, AlignmentC, Int<BlockM>, Int<BlockN>, WarpOnM, ThreadNum>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::EpilogueParallel<
    cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>,
    EpilogueOp,
    typename GemmEpilogueConfiguration::SmemLayoutO,
    Copy_Atom<EpilogueCopyInst,ElementAcc>,
    typename GemmEpilogueConfiguration::GmemTiledCopyO,
    Copy_Atom<EpilogueCopyInst,ElementC>
  >;

  using GemmKernel = typename cutlass::gemm::kernel::GemmUniversalParallel<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue
  >;
};

} // namespace config
} // namespace gemm
} // namespace cutlass


