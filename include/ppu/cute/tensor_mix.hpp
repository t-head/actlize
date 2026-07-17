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

#include <cute/tensor.hpp>

namespace cute
{

//
// PtrArithmeticTupleMixIterator
//

template <class ArithTuple, class Ptr>
struct PtrArithmeticTupleMixIterator
{
  using value_type   = ArithTuple;
  using element_type = ArithTuple;
  using reference    = ArithTuple;

  ArithTuple coord_;
  Ptr ptr_;

  CUTE_HOST_DEVICE constexpr
  PtrArithmeticTupleMixIterator(ArithTuple const& coord = {},
                                Ptr ptr = {}) : coord_(coord), ptr_(ptr) {}

  CUTE_HOST_DEVICE constexpr
  ArithTuple const& operator*() const { return coord_; }

  template <class Coord>
  CUTE_HOST_DEVICE constexpr
  auto operator[](Coord const& c) const { return *(*this + c); }

  // when counting tensor has only one dimension or only one dimension not zero,
  // ScaledBasis add int is still ScaledBasis, offset will be ScaledBasis but not ArithmeticTuple
  // add coord when offset is tuple or ScaledBasis, otherwise add ptr
  template <class Coord,
            typename = cute::enable_if_t<cute::is_tuple<Coord>::value || cute::is_scaled_basis<Coord>::value>>
  CUTE_HOST_DEVICE constexpr
  auto operator+(Coord const& c) const {
    return PtrArithmeticTupleMixIterator<decltype(coord_ + c), decltype(ptr_)>(coord_ + c, ptr_);
  }

  template <typename TInt,
            typename = cute::enable_if_t<!cute::is_tuple<TInt>::value && !cute::is_scaled_basis<TInt>::value>>
  CUTE_HOST_DEVICE constexpr
  auto operator+(const TInt i) const {
    return PtrArithmeticTupleMixIterator<decltype(coord_), decltype(ptr_)>(coord_, ptr_ + i);
  }

};

//
// Display utilities
//

template <class ArithTuple, class Ptr>
CUTE_HOST_DEVICE void print(PtrArithmeticTupleMixIterator<ArithTuple, Ptr> const& iter)
{
  printf("PtrArithTuple"); print(iter.coord_);
}

#if !defined(__HGGCCC_RTC__)
template <class ArithTuple, class Ptr>
CUTE_HOST std::ostream& operator<<(std::ostream& os, PtrArithmeticTupleMixIterator<ArithTuple, Ptr> const& iter)
{
  return os << "PtrArithTuple" << iter.coord_;
}
#endif

template <class Tuple, class Ptr>
CUTE_HOST_DEVICE constexpr
auto
make_ptr_inttuple_mix_iter(Tuple const& t, Ptr const& ptr) {
  return PtrArithmeticTupleMixIterator(as_arithmetic_tuple(t), ptr);
}

template <class T>
struct is_mix_iterator : false_type {};
template <class ArithTuple, class Ptr>
struct is_mix_iterator<PtrArithmeticTupleMixIterator<ArithTuple,Ptr>> : true_type {};

using _D0 = ScaledBasis<decltype(_1{}), 0>;
using _D1 = ScaledBasis<decltype(_1{}), 1>;
using _D2 = ScaledBasis<decltype(_1{}), 2>;
using _D3 = ScaledBasis<decltype(_1{}), 3>;

template <int Rank, int Rank0, int Rank1>
CUTE_HOST_DEVICE
auto make_counting_stride() {}

template <>
CUTE_HOST_DEVICE
auto make_counting_stride<2, 1, 1>() {
#if 1 // ACOMPUTE_VERSION == 10000
  return make_stride(_D1{}, _D0{});
#else
  // before swapping coord_h and coord_w in aiu_load, 10500 use make_identity_tensor to get global tensor
  // ensure stride is (_0,_1,_2) which is identical to dim order since cute always put row before column in dim
  // can unify with 1.0 when refactor fa code
  // if constexpr (is_smem<T>::value) {
  //   return make_stride(_D1{}, _D0{});
  // } else {
  //   return make_stride(_D0{}, _D1{});
  // }

  // TODO: make mix tensor for global tensor will only use shape of one batch which is two dimensional now
  // if use 3D aiu copy with 3D global tensor will use <3,1,1> realization
  // which may conflict to tsm load stride
  // and my conflict now if only one stage because tsm stride will use this realization
  // maybe decouple stride of global and tsm is better
  return make_stride(_D0{}, _D1{}); // Deprecated.
#endif
}

template<>
CUTE_HOST_DEVICE
auto make_counting_stride<2, 1, 2>() {
  return make_stride(_D1{}, make_stride(_D0{}, _D2{}));
}

template<>
CUTE_HOST_DEVICE
auto make_counting_stride<2, 2, 1>() {
  return make_stride(make_stride(_D1{}, _D2{}), _D0{});
}

template<>
CUTE_HOST_DEVICE
auto make_counting_stride<3, 1, 1>() {
  return make_stride(_D1{}, _D0{}, _D3{});
}

template<>
CUTE_HOST_DEVICE
auto make_counting_stride<3, 2, 1>() {
  return make_stride(make_stride(_D1{}, _D2{}), _D0{}, _D3{});
}

template<>
CUTE_HOST_DEVICE
auto make_counting_stride<3, 1, 2>() {
  return make_stride(_D1{}, make_stride(_D0{}, _D2{}), _D3{});
}

template <class T>
struct MakeMixTensor
{

  template <class Layout,
            __CUTE_REQUIRES(has_dereference<T>::value &&
                            is_layout<Layout>::value)>
  CUTE_HOST_DEVICE constexpr auto
  operator()(T const& iter, Layout const& layout)
  {
    constexpr int Rank = decltype(rank(Layout{}))::value;
    constexpr int Rank0 = decltype(rank(shape<0>(Layout{})))::value;
    constexpr int Rank1 = decltype(rank(shape<1>(Layout{})))::value;
    auto stride = make_counting_stride<Rank, Rank0, Rank1>();
    auto layout_counting = make_layout(layout.shape(), stride);
    auto iter_mix = make_ptr_inttuple_mix_iter(repeat_like(coshape(layout_counting), Int<0>{}), iter);
    using Engine = ViewEngine<decltype(iter_mix)>;
    using LayoutCounting = decltype(layout_counting);
    return Tensor<Engine, LayoutCounting>(iter_mix, layout_counting);
  }

  template <class LayoutArg, class... LayoutArgs,
            __CUTE_REQUIRES(not is_layout<LayoutArg>::value)>
  CUTE_HOST_DEVICE constexpr auto
  operator()(LayoutArg const& arg, LayoutArgs const&... args) const
  {
    return operator()(make_layout(arg, args...));
  }

  template <class LayoutArg, class... LayoutArgs,
            __CUTE_REQUIRES(not is_layout<LayoutArg>::value)>
  CUTE_HOST_DEVICE constexpr auto
  operator()(T const& iter, LayoutArg const& arg, LayoutArgs const&... args)
  {
    return operator()(iter, make_layout(arg, args...));
  }
};

// Make a non-owning Tensor that will use a pointer (view)
// e.g. make_tensor(vec.data(), 12)
template <class Iterator, class... Args>
CUTE_HOST_DEVICE constexpr
auto
make_mix_tensor(Iterator const& iter, Args const&... args)
{
  return MakeMixTensor<Iterator>{}(iter, args...);
}

template <class Engine, class Layout>
CUTE_HOST_DEVICE constexpr
auto
make_mix_tensor_like(Tensor<Engine,Layout> const& tensor)
{
  return make_mix_tensor(tensor.data(), tensor.layout());
}

} // end namespace cute
