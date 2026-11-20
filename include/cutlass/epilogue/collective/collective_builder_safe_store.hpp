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

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/builders/ppu_builder.inl"
#include "cutlass/epilogue/collective/ppu_epilogue_vectorized_evt_safe_store.hpp"

namespace cutlass::epilogue::collective {
namespace detail {

template <class LegacyEpilogue>
struct PpuSafeStoreEpilogueAdapter {
  static_assert(cutlass::detail::dependent_false<LegacyEpilogue>,
                "Safe store builder requires the PPU EVT epilogue.");
};

template <class StrideC, class StrideD, class Callbacks, class SmemLayout,
          class CopyR2S, class CopyS2R, class CopyR2G, class Schedule, class CopyG2R>
struct PpuSafeStoreEpilogueAdapter<EpilogueEvt<StrideC, StrideD, Callbacks,
    SmemLayout, CopyR2S, CopyS2R, CopyR2G, Schedule, CopyG2R>> {
  using CollectiveOp = EpilogueEvtSafeStore<StrideC, StrideD, Callbacks,
      SmemLayout, CopyR2S, CopyS2R, CopyR2G, Schedule, CopyG2R>;
};

// Keep invalid configurations out of the legacy builder's layout arithmetic.
template <bool Valid, class ArchTag, class OpClass, class TileShape, class WarpShape,
          class EpilogueTile, class Accumulator, class Compute,
          class ElementC, class LayoutC, int AlignmentC,
          class ElementD, class LayoutD, int AlignmentD,
          class Schedule, class Fusion, class Enable>
struct PpuSafeStoreBuilderImpl {
  static_assert(cute::is_same_v<ArchTag, cutlass::arch::PPU0010>,
                "Safe store builder requires PPU0010.");
  static_assert(cute::is_same_v<Schedule, cutlass::epilogue::EpilogueSimtVectorized>,
                "Safe store builder requires EpilogueSimtVectorized.");
  static_assert(AlignmentC > 0 && AlignmentD > 0,
                "Safe store builder alignments must be positive.");
};

template <class ArchTag, class OpClass, class TileShape, class WarpShape,
          class EpilogueTile, class Accumulator, class Compute,
          class ElementC, class LayoutC, int AlignmentC,
          class ElementD, class LayoutD, int AlignmentD,
          class Schedule, class Fusion, class Enable>
struct PpuSafeStoreBuilderImpl<true, ArchTag, OpClass, TileShape, WarpShape,
    EpilogueTile, Accumulator, Compute, ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD, Schedule, Fusion, Enable> {
  using LegacyEpilogue = typename CollectiveBuilder<ArchTag, OpClass,
      TileShape, WarpShape, EpilogueTile, Accumulator, Compute,
      ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
      Schedule, Fusion, Enable>::CollectiveOp;
  using CollectiveOp = typename PpuSafeStoreEpilogueAdapter<LegacyEpilogue>::CollectiveOp;
};

} // namespace detail

// A separate entry point: existing CollectiveBuilder specializations stay unchanged.
template <class ArchTag, class OpClass, class TileShape_MNK, class ClusterShape_MNK,
          class EpilogueTileType, class ElementAccumulator, class ElementCompute,
          class ElementC, class GmemLayoutTagC, int AlignmentC,
          class ElementD, class GmemLayoutTagD, int AlignmentD,
          class EpilogueScheduleType,
          class FusionOpOrCallbacks = fusion::LinearCombination<ElementD, ElementCompute, ElementC, ElementCompute>,
          class Enable = void>
struct CollectiveBuilderSafeStore : detail::PpuSafeStoreBuilderImpl<
    cute::is_same_v<ArchTag, cutlass::arch::PPU0010> &&
    cute::is_same_v<EpilogueScheduleType, cutlass::epilogue::EpilogueSimtVectorized> &&
    (AlignmentC > 0 && AlignmentD > 0),
    ArchTag, OpClass, TileShape_MNK, ClusterShape_MNK, EpilogueTileType,
    ElementAccumulator, ElementCompute, ElementC, GmemLayoutTagC, AlignmentC,
    ElementD, GmemLayoutTagD, AlignmentD, EpilogueScheduleType, FusionOpOrCallbacks, Enable> {};

} // namespace cutlass::epilogue::collective
