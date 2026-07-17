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

#include <cute/container/tuple.hpp>
#include <cute/container/array.hpp>
#include <cute/algorithm/tuple_algorithms.hpp>
#include <cute/numeric/integral_constant.hpp>


namespace cute{

struct SplitkCoordIteratorSentinal
{};

template <class Coord, class Shape>
struct SplitkCoordIterator
{
  static_assert(is_congruent<Coord, Shape>::value);

  CUTE_HOST_DEVICE constexpr
  Coord const& operator*() const { return coord; }

  CUTE_HOST_DEVICE constexpr
  SplitkCoordIterator& operator++() { coord += step; return *this; }

  // Sentinel for the end of the implied range
  CUTE_HOST_DEVICE constexpr
  bool operator< (SplitkCoordIteratorSentinal const&) const { return back(coord) <  back(shape); }
  CUTE_HOST_DEVICE constexpr
  bool operator==(SplitkCoordIteratorSentinal const&) const { return back(coord) == back(shape); }
  CUTE_HOST_DEVICE constexpr
  bool operator!=(SplitkCoordIteratorSentinal const&) const { return back(coord) != back(shape); }
  // NOTE: These are expensive, avoid use
  CUTE_HOST_DEVICE constexpr
  bool operator< (SplitkCoordIterator const& other) const { return colex_less(coord, other.coord); }
  CUTE_HOST_DEVICE constexpr
  bool operator==(SplitkCoordIterator const& other) const { return coord == other.coord; }
  CUTE_HOST_DEVICE constexpr
  bool operator!=(SplitkCoordIterator const& other) const { return coord != other.coord; }

  Coord coord;
  Shape const& shape;
  int step;
};


// A forward iterator for a coordinate that starts from zero
template <class Shape>
CUTE_HOST_DEVICE constexpr
auto
make_splitk_coord_iterator(Shape const& shape, int splitk_idx, int splitk_size)
{
  auto coord = repeat_like(shape, int(splitk_idx));
  return SplitkCoordIterator<int, Shape>{coord, shape, splitk_size};
}

} // end namespace cute
