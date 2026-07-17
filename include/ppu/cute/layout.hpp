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

#include <cute/layout.hpp>

namespace cute{

using _D0 = ScaledBasis<decltype(_1{}), 0>;
using _D1 = ScaledBasis<decltype(_1{}), 1>;
using _D2 = ScaledBasis<decltype(_1{}), 2>;
using _D3 = ScaledBasis<decltype(_1{}), 3>;

template<typename SmemLayout, bool SplitOnK>
struct MakeTsmLdTensorImpl;

template<typename SmemLayout>
struct MakeTsmLdTensorImpl<SmemLayout, false> {
  CUTE_HOST_DEVICE constexpr
  auto
  operator()() {
    constexpr int Split_Rank = decltype(rank(shape<0>(SmemLayout{})))::value;
    static_assert(Split_Rank <= 2, "dim0 of tsm layout should <= 2");
    constexpr int Smem_Rank = decltype(rank(shape(SmemLayout{})))::value;
    static_assert(Smem_Rank <= 3, "dim of tsm layout should <= 3");
    // split dim
    auto identity_stride_split = append<Split_Rank>(_D1{}, _D2{});
    auto identity_stride_without_stage = make_stride(identity_stride_split, _D0{});
    auto identity_stride = append<Smem_Rank>(identity_stride_without_stage, _D3{});

    Tensor csA = make_counting_tensor(make_layout(SmemLayout{}.shape(), identity_stride));

    return csA;
  }
};

template<typename SmemLayout>
struct MakeTsmLdTensorImpl<SmemLayout, true> {
  CUTE_HOST_DEVICE constexpr
  auto
  operator()() {

    constexpr int Split_Rank = decltype(rank(shape<1>(SmemLayout{})))::value;
    static_assert(Split_Rank <= 2, "dim0 of tsm layout should <= 2");
    constexpr int Smem_Rank = decltype(rank(shape(SmemLayout{})))::value;
    static_assert(Smem_Rank <= 3, "dim of tsm layout should <= 3");
    // split dim
    auto identity_stride_split = append<Split_Rank>(_D0{}, _D2{});
    auto identity_stride_without_stage = make_stride(_D1{}, identity_stride_split);
    auto identity_stride = append<Smem_Rank>(identity_stride_without_stage, _D3{});

    Tensor csA = make_counting_tensor(make_layout(SmemLayout{}.shape(), identity_stride));

    return csA;
  }
};

template<typename SmemLayout>
CUTE_HOST_DEVICE constexpr
auto make_tsm_ld_tensor() {
  constexpr int Rank0 = decltype(rank(shape<0>(SmemLayout{})))::value;
  constexpr int Rank1 = decltype(rank(shape<1>(SmemLayout{})))::value;
  // only support split on one dim
  static_assert((Rank0 <= 2 && Rank1 == 1) || (Rank1 <= 2 && Rank0 == 1), "dim0 of tsm layout should <= 2");
  constexpr bool split_on_k = Rank1 > 1;

  return MakeTsmLdTensorImpl<SmemLayout, split_on_k>()();
}

} // namespace cute
