/***************************************************************************************************
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

#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/detail/layout.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_generic.h"
// cutlass3 change
#ifndef __HGGCCC_RTC__
#include "cutlass/epilogue/thread/linear_combination_bias_elementwise.h"
#endif
#include "cutlass/epilogue/fusion/callbacks.hpp"
#include "cutlass/epilogue/fusion/ppu_callbacks.hpp"

#if defined(__HGGCCC_RTC__)
#include <hggc/std/type_traits>
#else
#include <type_traits>
#endif


#include "cutlass/epilogue/collective/ppu_epilogue_vectorized.hpp"
#include "cutlass/epilogue/collective/ppu_epilogue_vectorized_evt.hpp"
#include "cutlass/gemm/config/gemm_operands.hpp"
///////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::collective {

namespace detail {

template <bool UseEpilogueEvt, typename... T>
using ChooseEpilogue = typename platform::conditional<
  UseEpilogueEvt,
  cutlass::epilogue::collective::EpilogueEvt<T...>,
  cutlass::epilogue::collective::Epilogue<T...>,
>::type;

// callbacks builder with aux out
template <
  class FusionOp,
  class TileShape_MNK,
  class EpilogueTile_MN,
  class ElementAccumulator
>
struct CallbacksBuilder<
  cutlass::epilogue::EpilogueSimtVectorized,
  FusionOp,
  TileShape_MNK,
  EpilogueTile_MN,
  ElementAccumulator,
  cute::enable_if_t<(FusionOp::IsAuxOutSupported ^ FusionOp::IsAuxInSupported) // only one aux tensor
              && not cute::is_subbyte_v<typename FusionOp::ElementAux>>
> {
  using SmemLayoutAtomAux = void;
  using SmemCopyOpAux = void;

  using Callbacks = fusion::FusionCallbacks<
    cutlass::epilogue::EpilogueSimtVectorized,
    FusionOp, TileShape_MNK, EpilogueTile_MN,
    SmemLayoutAtomAux, SmemCopyOpAux
  >;
};
} // namespace detail

///////////////////////////////////////////////////////////////////////////////

// No-smem builder
template <
  typename Arch,
  class OpClass,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC_,
  class GmemLayoutTagC_,
  int AlignmentC,
  class ElementD,
  class GmemLayoutTagD,
  int AlignmentD,
  class Schedule,
  FloatRoundStyle RoundStyle
>
struct CollectiveBuilder<
    Arch,
    OpClass,
    TileShape_MNK,
    ClusterShape_MNK,
    EpilogueTileType,
    ElementAccumulator,
    ElementCompute,
    ElementC_,
    GmemLayoutTagC_,
    AlignmentC,
    ElementD,
    GmemLayoutTagD,
    AlignmentD,
    Schedule,
    fusion::LinearCombination<ElementD,ElementCompute,ElementC_,ElementCompute,RoundStyle>,
    cute::enable_if_t<cute::is_same_v<Schedule, NoSmemWarpSpecialized> ||
                      cute::is_same_v<Schedule, PtrArrayNoSmemWarpSpecialized> >> {

  // Passing void C disables source load
  using ElementC = cute::conditional_t<cute::is_void_v<ElementC_>,
      ElementD, ElementC_>; // prevents cute breakages
  using GmemLayoutTagC = cute::conditional_t<cute::is_void_v<ElementC_>,
      GmemLayoutTagD, GmemLayoutTagC_>;
  static constexpr thread::ScaleType::Kind ScaleType = cute::is_void_v<ElementC_> ?
      thread::ScaleType::OnlyAlphaScaling : thread::ScaleType::Default;

  static constexpr int FragmentSize = 1;
  using ThreadOp = thread::LinearCombination<
    ElementD, FragmentSize, ElementAccumulator, ElementCompute,
    ScaleType, RoundStyle, ElementC>;

  using CollectiveOp = cute::conditional_t<
    cute::is_same_v<Schedule, NoSmemWarpSpecialized>,
      cutlass::epilogue::collective::DefaultEpilogue<
        cutlass::detail::TagToStrideC_t<GmemLayoutTagC>,
        cutlass::detail::TagToStrideC_t<GmemLayoutTagD>,
        ThreadOp,
        cutlass::gemm::EpilogueDefault>,
    // Epilogue for Ptr-Array and Grouped Gemm
      cutlass::epilogue::collective::DefaultEpilogueArray<
        cutlass::detail::TagToStrideC_t<GmemLayoutTagC>,
        cutlass::detail::TagToStrideC_t<GmemLayoutTagD>,
        ThreadOp,
        Schedule>
    >;
};

// Auto builder
template <
  typename Arch,
  class OpClass,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC,
  class GmemLayoutTagC,
  int AlignmentC,
  class ElementD,
  class GmemLayoutTagD,
  int AlignmentD,
  class FusionOperation
>
struct CollectiveBuilder<
    Arch,
    OpClass,
    TileShape_MNK,
    ClusterShape_MNK,
    EpilogueTileType,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    GmemLayoutTagC,
    AlignmentC,
    ElementD,
    GmemLayoutTagD,
    AlignmentD,
    EpilogueScheduleAuto,
    FusionOperation,
    void> {
private:
  static_assert(cute::is_same_v<FusionOperation, fusion::LinearCombination<ElementD,ElementCompute,ElementC,ElementCompute>>,
                "Auto schedule doesn't support fusion. Use one of the TmaWarpSpecialized schedules instead.");

  // Pick No-Smem epilogue as the Auto Epilogue Schedule (Auto schedules do not guarantee best performance)
  // since TMA epilogues are not compatible with non-TMA non-WS mainloops
  using EpilogueSchedule = NoSmemWarpSpecialized;
  using _CollectiveBuilder = CollectiveBuilder<
    Arch,
    OpClass,
    TileShape_MNK,
    ClusterShape_MNK,
    EpilogueTileType,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    GmemLayoutTagC,
    AlignmentC,
    ElementD,
    GmemLayoutTagD,
    AlignmentD,
    EpilogueSchedule,
    FusionOperation
  >;

public:
  using CollectiveOp = typename _CollectiveBuilder::CollectiveOp;
};


// EpilogueSimtVectorized && Tma warp-specialized builder
template <
  typename Arch,
  class OpClass,
  class TileShape_MNK,
  class WarpShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator_,
  class ElementCompute,
  class ElementC,
  class GmemLayoutTagC,
  int AlignmentC,
  class ElementD_,
  class GmemLayoutTagD,
  int AlignmentD,
  class Schedule,
  class EpilogueOp
>
struct CollectiveBuilder<
    Arch,
    OpClass,
    TileShape_MNK,
    WarpShape_MNK, //original placeholder of ClusterShape_MNK
    EpilogueTileType,
    ElementAccumulator_,
    ElementCompute,
    ElementC,
    GmemLayoutTagC,
    AlignmentC,
    ElementD_,
    GmemLayoutTagD,
    AlignmentD,
    Schedule,
    EpilogueOp,
    cute::enable_if_t<cute::is_same_v<Schedule, EpilogueSimtVectorized> ||
                      cute::is_same_v<Schedule, EpilogueSimtVectorizedWithoutEvt> ||
                      cute::is_same_v<Schedule, EpiloguePtrArraySimtVectorized> ||
                      cute::is_same_v<Schedule, TmaWarpSpecialized> ||
                      cute::is_same_v<Schedule, TmaWarpSpecializedCooperative> ||
                      detail::ppu_is_ptr_array_tma_v<Schedule>>> {
private:
  static_assert(cute::is_same_v<EpilogueTileType, cutlass::epilogue::collective::EpilogueTileAuto>, "PPU can only support auto epilogue tile");

  using ElementAccumulator = ElementAccumulator_;
  using ElementD = cute::conditional_t<cute::is_void_v<ElementD_>,
                     fusion::get_element_aux_t<EpilogueOp>, ElementD_>;

  // User should configure custom warp tile shape through WarpShape_MNK
  static_assert(cute::get<0>(WarpShape_MNK{}) != 1 ||
                cute::get<1>(WarpShape_MNK{}) != 1 ||
                cute::get<2>(WarpShape_MNK{}) != 1, "ClusterShape_MNK needs to be configured as Warp Tile Shape for PPU, 1-1-1 cannot be accepted");

  constexpr static int BlockM = cute::get<0>(TileShape_MNK{});
  constexpr static int BlockN = cute::get<1>(TileShape_MNK{});
  constexpr static int WarpM = cute::get<0>(WarpShape_MNK{});
  constexpr static int WarpN = cute::get<1>(WarpShape_MNK{});
  using WarpOnM = Int<BlockM / WarpM>;
  using WarpOnN = Int<BlockN / WarpN>;
  static constexpr int ThreadNum = WarpOnM() * WarpOnN() * 32;
  static constexpr int FragmentSize = BlockM * BlockN / ThreadNum;
  static constexpr int Alignment = platform::min(AlignmentC, AlignmentD);

  using EpilogueCopyInst = AutoVectorizingCopyWithAssumedAlignment<128>;
  using GemmEpilogueConfiguration = gemm::config::DefaultGemm_Epilogue_Configuration<EpilogueCopyInst, ElementAccumulator, Alignment, Int<BlockM>, Int<BlockN>, WarpOnM, ThreadNum>;
  using EpilogueDispatchPolicy = cute::conditional_t<(cute::is_same_v<Schedule, EpiloguePtrArraySimtVectorized> || detail::ppu_is_ptr_array_tma_v<Schedule>),
                                                     EpiloguePtrArraySimtVectorized,
                                                     cutlass::epilogue::EpilogueSimtVectorized>;
  using EpilogueTile_MN = decltype(shape(coalesce(make_layout(shape(typename GemmEpilogueConfiguration::SmemLayoutO{})), Step<_1, _1>{})));

  using DefaultOperation = thread::LinearCombination<ElementD, FragmentSize, ElementAccumulator, ElementAccumulator>;
  using FusionCallbacks =
    typename epilogue::collective::detail::CallbacksBuilder<
      EpilogueDispatchPolicy,
      EpilogueOp,
      TileShape_MNK,
      EpilogueTile_MN,
      ElementAccumulator
    >::Callbacks;
  static constexpr bool use_evt = !cute::is_same_v<Schedule, EpilogueSimtVectorizedWithoutEvt>;
  using ThreadEpilogueOp = cute::conditional_t<use_evt, FusionCallbacks, DefaultOperation>;
public:
  using CollectiveOp = detail::ChooseEpilogue<
    use_evt,
    cutlass::detail::TagToStrideC_t<GmemLayoutTagC>,
    cutlass::detail::TagToStrideC_t<GmemLayoutTagD>,
    ThreadEpilogueOp,
    typename GemmEpilogueConfiguration::SmemLayoutO,
    Copy_Atom<EpilogueCopyInst,ElementAccumulator>,
    typename GemmEpilogueConfiguration::GmemTiledCopyO,
    Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<sizeof(ElementD) * AlignmentD * 8>,ElementD>,
    EpilogueDispatchPolicy
  >;
};

} // namespace cutlass::epilogue::collective: