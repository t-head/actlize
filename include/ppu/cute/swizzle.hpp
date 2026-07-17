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

#include <cute/swizzle.hpp>

namespace cute{

template <int BBits, int MBase, int SShift = BBits>

struct PPU_Swizzle
{
  static constexpr int num_bits = BBits;
  static constexpr int num_base = MBase;
  static constexpr int num_shft = SShift;

  static_assert(num_base >= 0,             "MBase must be positive.");
  static_assert(num_bits >= 0,             "BBits must be positive.");
  static_assert(abs(num_shft) >= num_bits, "abs(SShift) must be more than BBits.");


  // using 'int' type here to avoid unintentially casting to unsigned... unsure.
  using bit_msk = cute::constant<int, (1 << num_bits) - 1>;
  using yyy_msk = cute::constant<int, bit_msk{} << (num_base + max(0,num_shft))>;
  using zzz_msk = cute::constant<int, bit_msk{} << (num_base - min(0,num_shft))>;
  using msk_sft = cute::constant<int, num_shft>;
  using zero_yyy_msk = cute::constant<int, ~(bit_msk{} << (num_base + max(0,num_shft)))>;

  static constexpr uint32_t swizzle_code = uint32_t(yyy_msk{} | zzz_msk{});

  template <class Offset>
  CUTE_HOST_DEVICE constexpr static
  auto
  apply(Offset const& offset)
  {
    auto mask_z = shiftr(offset & yyy_msk{}, msk_sft{});
    auto mask_y = shiftl(offset & zzz_msk{}, msk_sft{});
    auto tmp_offset = (offset & zero_yyy_msk{});
    auto mask_swizzle = mask_z | mask_y;
    return tmp_offset ^ mask_swizzle;
  }

  template <class Offset>
  CUTE_HOST_DEVICE constexpr
  auto
  operator()(Offset const& offset) const
  {
    return apply(offset);
  }
};


template <uint32_t Y, uint32_t Z>
CUTE_HOST_DEVICE constexpr
auto
make_ppu_swizzle()
{
  constexpr uint32_t BZ = popcount(Y);                    // Number of swizzle bits
  constexpr uint32_t BY = popcount(Z);                    // Number of swizzle bits
  static_assert(BZ == BY, "Number of bits in Y and Z don't match");
  constexpr uint32_t TZ_Y = countr_zero(Y);               // Number of trailing zeros in Y
  constexpr uint32_t TZ_Z = countr_zero(Z);               // Number of trailing zeros in Z
  constexpr uint32_t M = cute::min(TZ_Y, TZ_Z) % 32;
  constexpr  int32_t S = int32_t(TZ_Y) - int32_t(TZ_Z);   // Difference in trailing zeros
  static_assert((Y | Z) == PPU_Swizzle<BZ,M,S>::swizzle_code, "Something went wrong.");
  return PPU_Swizzle<BZ,M,S>{};
}

template <int B0, int M0, int S0,
          int B1, int M1, int S1>
CUTE_HOST_DEVICE constexpr
auto
composition(PPU_Swizzle<B0,M0,S0>, PPU_Swizzle<B1,M1,S1>)
{
  static_assert(S0 == S1, "Can only merge swizzles of the same shift.");
  constexpr uint32_t Y = PPU_Swizzle<B0,M0,S0>::yyy_msk::value ^ PPU_Swizzle<B1,M1,S1>::yyy_msk::value;
  constexpr uint32_t Z = PPU_Swizzle<B0,M0,S0>::zzz_msk::value ^ PPU_Swizzle<B1,M1,S1>::zzz_msk::value;
  return make_ppu_swizzle<Y,Z>();
}

} // end namespace cute
