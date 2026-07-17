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

#include "cutlass/arch/mma.h"

#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/detail/layout.hpp"
#include "cutlass/detail/collective.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_generic.h"
#include "cutlass/epilogue/thread/linear_combination_bias_elementwise.h"
#include "cutlass/epilogue/fusion/callbacks.hpp"
#include "cutlass/epilogue/fusion/ppu_callbacks_tma_warpspecialized.hpp"

#if defined(__HGGCCC_RTC__)
#include <hggc/std/type_traits>
#else
#include <type_traits>
#endif

///////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::collective {

///////////////////////////////////////////////////////////////////////////////

namespace detail {

// Returns the parameterized dispatch policy for epilogue
template<class TileShapeMNK, class EpilogueTileMN, class ElementC, class ElementD, class Schedule>
constexpr auto
ppu_get_dispatch_policy() {
  using namespace cute;

  constexpr int EpiTiles = size(shape_div(take<0,2>(TileShapeMNK{}), EpilogueTileMN{}));
  constexpr int FragmentSize = size(EpilogueTileMN{}) / (detail::ppu0015_is_cooperative_v<Schedule> ? 256 : 128);
  constexpr int ReuseSmemC = (sizeof_bits_v<ElementC> == sizeof_bits_v<ElementD>) && (sizeof_bits_v<ElementD> > 8);
  constexpr int StagesD = 2;
  constexpr int StagesC = ReuseSmemC ? cute::max(EpiTiles, StagesD + 1) : EpiTiles;

  return PPU0015TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC>{};
}

// Attempts to compute a reasonable epilogue tile based on block tile shape or allows the user to provide one.
template <class ElementD, class EpilogueTileType, class Schedule, class TileShape_MNK>
constexpr auto
ppu0015_compute_tile_shape_or_override() {
  if constexpr (cute::is_same_v<EpilogueTileType, EpilogueTileAuto>) {

    if constexpr (detail::ppu0015_is_cooperative_v<Schedule>) {
      using N_tile = decltype(cute::min(_32{}, get<1>(TileShape_MNK{})));
      if constexpr (size<0>(TileShape_MNK{}) >= 128) {
        return Shape<_128, N_tile>{};
      }
      else {
        return Shape<_64, N_tile>{};
      }
    }
    else if constexpr (detail::ppu0015_is_warp_specialized_v<Schedule>) {
      if constexpr (sizeof_bits_v<ElementD> == 8) {
        using N_tile = decltype(cute::min(_64{}, get<1>(TileShape_MNK{})));
        return Shape<_64, N_tile>{};
      }
      else {
        using N_tile = decltype(cute::min(_32{}, get<1>(TileShape_MNK{})));
        return Shape<_64,N_tile>{};
      }
    }
    else {
      static_assert(cutlass::detail::dependent_false<Schedule>, "Unsupported schedule.");
    }
  }
  else if constexpr (cute::is_tuple<EpilogueTileType>::value) {
    EpilogueTileType epi_tile;
    constexpr int M = size<0>(shape(epi_tile));
    constexpr int N = size<1>(shape(epi_tile));

    static_assert(!is_layout<EpilogueTileType>::value, "EpilogueTile must be a cute::Tile or cute::Shape");
    static_assert(M ==  64 && detail::ppu0015_is_warp_specialized_v<Schedule> ||
                  M == 128 && detail::ppu0015_is_cooperative_v<Schedule>, "Unsupported tile shape");
    static_assert(N % 16 == 0, "Unsupported tile shape");

    return epi_tile;
  }
  else {
    static_assert(cutlass::detail::dependent_false<EpilogueTileType>, "Invalid type for EpilogueTileType.");
  }
}

///////////////////////////////////////////////////////////////////////////////
// Descriptor classes for defining EVT nodes
// Some of the epilogue visitor nodes require non-intuitive template arguments
// such as CopyOpS2R for AuxLoad node. Traditionaly, these are resolved by the
// builder classes. Here we provide a set of descriptor classes that resolve
// these template arguments from more intuitive types such as Stride, Layout

// Get TileShape, EpilogueTile, Dispatch Policy, StagesC, and STagesD
template<
  typename TileShape_MNK,
  typename EpilogueTileType, 
  typename ElementC,
  typename ElementD,
  typename Schedule
>
struct EpilogueDescriptor {
  using TileShape = TileShape_MNK;
  using EpilogueTile = 
    decltype(
      detail::ppu0015_compute_tile_shape_or_override<
        ElementD, EpilogueTileType, Schedule, TileShape_MNK
      >()
    );
  using DispatchPolicy = 
    decltype(
      detail::ppu_get_dispatch_policy<
        TileShape_MNK, EpilogueTile, 
        ElementC, ElementD, Schedule
      >()
    );
  constexpr static int StagesC = DispatchPolicy::StagesC;
  constexpr static int StagesD = DispatchPolicy::StagesD;
};

template<
  typename EpilogueDescriptor,
  typename ElementVector
>
struct RowBroadcastDescriptor {
  constexpr static int Stages = ceil_div(
    EpilogueDescriptor::StagesC, 
    size(shape_div(take<0, 2>(typename EpilogueDescriptor::TileShape{}), typename EpilogueDescriptor::EpilogueTile{}))
  ) + 1;

  using Element = ElementVector;
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////

// No-smem builder
template <
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
    arch::PPU0010,
    arch::OpClassTensorOp,
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
    arch::PPU0010,
    arch::OpClassTensorOp,
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
    arch::PPU0010,
    arch::OpClassTensorOp,
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

template <
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
  FloatRoundStyle RoundStyle
>
struct CollectiveBuilder<
    arch::PPU0010,
    arch::OpClassTensorOp,
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
    cutlass::gemm::EpilogueTransposed,
    fusion::LinearCombination<ElementD,ElementCompute,ElementC_,ElementCompute,RoundStyle>,
    void> {
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

  using CollectiveOp =
    cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::detail::TagToStrideC_t<GmemLayoutTagC>,
      cutlass::detail::TagToStrideC_t<GmemLayoutTagD>,
      ThreadOp,
      cutlass::gemm::EpilogueTransposed>;
};


///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::collective
