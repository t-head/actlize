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

#include <cute/layout.hpp>
#include <cute/layout_composed.hpp>

#include "ppu/cute/swizzle.hpp"

namespace cute
{

namespace detail {

template <class IntZ, class IntY, class Offset, int... I>
CUTE_HOST_DEVICE constexpr
auto
make_ppu_swizzle_strides(true_type,
                     IntZ   const& Z,
                     IntY   const& Y,
                     Offset const& offset,
                     int_sequence<I...>)
{
  return cute::make_tuple(conditional_return((offset & (Y << Int<I>{})) == Int<0>{}, (Y+Z) << Int<I>{}, (Y-Z) << Int<I>{})...);
  //return cute::make_tuple(conditional_return((offset & (Y << Int<I>{})) == Int<0>{}, Z << Int<I>{}, -(Z << Int<I>{}))...);

}

template <class IntZ, class IntY, class Offset, int... I>
CUTE_HOST_DEVICE constexpr
auto
make_ppu_swizzle_strides(false_type,
                     IntZ   const& Z,
                     IntY   const& Y,
                     Offset const& offset,
                     int_sequence<I...>)
{
  return cute::make_tuple(conditional_return((offset & (Z << Int<I>{})) == Int<0>{}, Z << Int<I>{}, -(Z << Int<I>{}))...);
  // return cute::make_tuple(conditional_return((offset & (Z << Int<I>{})) == Int<0>{}, (Y+Z) << Int<I>{}, (Y-Z) << Int<I>{})...);
}

// Get the "non-swizzle" part of a composed layout,
// which is the underlying (non-composed) Layout.
template <int B, int M, int S, class Offset, class LayoutB>
CUTE_HOST_DEVICE constexpr
auto
get_nonswizzle_portion(ComposedLayout<PPU_Swizzle<B,M,S>,Offset,LayoutB> const& slayout)
{
  return slayout.layout_b();
}

} // end namespace detail

template <class Coord, int B, int M, int S, class Offset, class Layout>
CUTE_HOST_DEVICE constexpr
auto
slice_and_offset(Coord const& coord, ComposedLayout<PPU_Swizzle<B,M,S>,Offset,Layout> const& layout)
{
  if constexpr (all_underscore<Coord>::value) {
    // Skip the expensive/complicated attempt to decay to a normal layout and just reshape
    return cute::make_tuple(composition(layout.layout_a(), layout.offset(), slice(coord, layout.layout_b())), Int<0>{});
  } else {

    // Projections of the swizzle layout for composition
    auto sw = make_layout(make_shape(Int<(1 << M)>{}, Int<(1 << B)>{}, Int<(1 << (abs(S)-B))>{}, Int<(1 << B)>{}, Int<1>{}));

    auto swizzle_anti_zy = make_layout(shape(sw),
                                       make_stride(stride<0>(sw),      Int<0>{}, stride<2>(sw),      Int<0>{}, size(sw)));
    auto swizzle_only_zy = make_layout(shape(sw),
                                       make_stride(     Int<0>{}, stride<1>(sw),      Int<0>{}, stride<3>(sw), Int<0>{}));

    // The portion of the layout that is not yet consumed
    auto sliced_layout = slice(coord, layout.layout_b());

    // If the sliced_layout hits two bits that are swizzled together, then don't attempt to decay

    // Compose with the layout to get the swizzle projection, P o L  [The Z and Y contributing portions of L]
    //   (this also tests that shape/stride of layout compose with swizzle)
    auto sliced_layout_only_zy = composition(swizzle_only_zy, sliced_layout);
    // Transform the end coordinate to get the active bits of the swizzle, (P o L)(c*)
    auto swizzle_active_bits = sliced_layout_only_zy(size(sliced_layout_only_zy)-Int<1>{});
    // Determine if any active bits collide under the swizzle
    auto hit_ZandY = !(swizzle_active_bits & ~layout.layout_a()(swizzle_active_bits));

    // The portion of the layout that we are consuming now
    auto diced_layout = dice(coord, layout.layout_b());
    auto diced_coord  = dice(coord, coord);

    auto diced_layout_anti_zy = composition(swizzle_anti_zy, diced_layout);
    auto diced_layout_only_zy = composition(swizzle_only_zy, diced_layout);

    // New swizzle and offset
    auto swizzle = layout.layout_a();
    // offset_only_zy interacts with swizzle and gets accumulated with layout.offset()
    //   being careful about the static/dynamic contributions from diced_layout and diced_coord
    auto offset_only_zy = layout.offset() ^ to_mixed_bits(diced_layout_only_zy, diced_coord);
    // offset_anti_zy always gets passed through, no interaction with swizzle
    auto offset_anti_zy = diced_layout_anti_zy(diced_coord);

    // If Layout's codomain hits on         Y AND Z, then it's not reducible
    // If Layout's codomain hits on         Y XOR Z, then it's dynamic-normal
    // If Layout's codomain hits on neither Y NOR Z, then it's static-normal

    // Test the sliced layout for hit_X & hit_Y for potential decay
    if constexpr (is_constant<false, decltype(hit_ZandY)>::value)
    { // Hits on Y AND Z, so it's not reducible
      return cute::make_tuple(composition(swizzle, offset_only_zy, sliced_layout), offset_anti_zy);
    } else
    { // Misses on Y or Z, so it's static-normal or dynamic-normal

      // Lowest bit of the Z and Y masks
      auto Z = typename PPU_Swizzle<B,M,S>::zzz_msk{} & -typename PPU_Swizzle<B,M,S>::zzz_msk{};
      auto Y = typename PPU_Swizzle<B,M,S>::yyy_msk{} & -typename PPU_Swizzle<B,M,S>::yyy_msk{};
      auto stride_lo = detail::make_ppu_swizzle_strides(Z < Y, Z, Y, offset_only_zy, make_int_sequence<B>{});
      auto stride_hi = detail::make_ppu_swizzle_strides(Z > Y, Z, Y, offset_only_zy, make_int_sequence<B>{});

      // Construct a (dynamic) layout that we can perform the composition with
      auto swizzle_layout = make_layout(make_shape (Int<(1 << M)>{}, repeat<B>(Int<2>{}), Int<(1 << (abs(S)-B))>{}, repeat<B>(Int<2>{}), Int<                  1>{}),
                                        make_stride(Int<       1>{},           stride_lo, Int<(1 <<      (M+B))>{},          stride_hi , Int<(1 << (M+B+abs(S)))>{}));

      // Decay to a normal layout with offset
      return cute::make_tuple(composition(swizzle_layout, sliced_layout),
                              swizzle(offset_only_zy) + offset_anti_zy);
    }
  }

  CUTE_GCC_UNREACHABLE;
}

//
// composition
//

// Ignore identity case
template <int M, int S,
          class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
composition(PPU_Swizzle<0,M,S> const&,
            Int<0> const&,
            Layout<Shape,Stride> const& layout)
{
  return layout;
}

template <int B, int M, int S,
          class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
composition(PPU_Swizzle<B,M,S> const& sxor,
            Layout<Shape,Stride> const& layout)
{
  return composition(sxor, Int<0>{}, layout);
}

template <class ShapeA, class StrideA,
          int B, int M, int S>
CUTE_HOST_DEVICE constexpr
auto
composition(Layout<ShapeA,StrideA> const& a,
            PPU_Swizzle<B,M,S>         const& b)
{
  // Get the Z bits and the Y bits
  auto active_Y = a(typename PPU_Swizzle<B,M,S>::yyy_msk{});
  auto active_Z = a(typename PPU_Swizzle<B,M,S>::zzz_msk{});

  // Works in simple cases... but could be greatly generalized

  return composition(make_ppu_swizzle<active_Y,active_Z>(), a);
}




} // end namespace cute
