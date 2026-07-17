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

#include <cute/arch/copy_ppu.hpp>
#include <cute/atom/copy_traits.hpp>
#include <cute/layout.hpp>

namespace cute
{

template <>
struct Copy_Traits<PPU_U32x1_LDSM_N>
{
  // Logical thread id to thread idx (warp)
  using ThrID = Layout<_32>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <Shape <  _8,_4>,_128>,
                           Stride<Stride<_128,_0>,  _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_32,_32>,
                           Stride<_32, _1>>;

  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
};

template <>
struct Copy_Traits<PPU_U32x2_LDSM_N>
{
  // Logical thread id to thread idx (warp)
  using ThrID = Layout<_32>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <Shape < _16,_2>,_128>,
                           Stride<Stride<_128,_0>,  _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_32,Shape <_32,   _2>>,
                           Stride<_32,Stride< _1,_1024>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
};

template <>
struct Copy_Traits<PPU_U32x4_LDSM_N>
{
  // Logical thread id to thread idx (warp)
  using ThrID = Layout<_32>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape < _32,_128>,
                           Stride<_128,  _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_32,Shape <_32,   _4>>,
                           Stride<_32,Stride< _1,_1024>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
};

template <>
struct Copy_Traits<PPU_U16x2_LDSM_T>
{
  // Logical thread id to thread idx (warp)
  using ThrID = Layout<_32>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <Shape <  _8,_4>,_128>,
                           Stride<Stride<_128,_0>,  _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape <  _4, _8>,Shape <_16,  _2>>,
                           Stride<Stride<_256,_16>,Stride< _1,_128>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
};

template <>
struct Copy_Traits<PPU_U16x4_LDSM_T>
{
  // Logical thread id to thread idx (warp)
  using ThrID = Layout<_32>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <Shape < _16,_2>,_128>,
                           Stride<Stride<_128,_0>,  _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape <  _4, _8>,Shape <_16,  _2,   _2>>,
                           Stride<Stride<_256,_16>,Stride< _1,_128,_1024>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
};

template <>
struct Copy_Traits<PPU_U16x8_LDSM_T>
{
  // Logical thread id to thread idx (warp)
  using ThrID = Layout<_32>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape < _32,_128>,
                           Stride<_128,  _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape <  _4, _8>,Shape <_16,  _2,   _4>>,
                           Stride<Stride<_256,_16>,Stride< _1,_128,_1024>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
};

} // end namespace cute



namespace cute
{

template <class S, class D>
struct Copy_Traits<PPU_CP_ASYNC_CACHEALWAYS<S,D>>
{
  // Logical thread id to thread idx (one-thread)
  using ThrID = Layout<_1>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,Int<sizeof_bits<S>::value>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,Int<sizeof_bits<D>::value>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // Construct a zfill variant with a given predicate value
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<PPU_CP_ASYNC_CACHEALWAYS_ZFILL<S,D>>
  with(bool pred) const {
    return {pred};
  }
};

template <class S, class D>
struct Copy_Traits<PPU_CP_ASYNC_CACHEGLOBAL<S,D>>
{
  // Logical thread id to thread idx (one-thread)
  using ThrID = Layout<_1>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,Int<sizeof_bits<S>::value>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,Int<sizeof_bits<D>::value>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // Construct a zfill variant with a given predicate value
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<PPU_CP_ASYNC_CACHEGLOBAL_ZFILL<S,D>>
  with(bool pred) const {
    return {pred};
  }
};

template <class S, class D>
struct Copy_Traits<PPU_CP_ASYNC_CACHEALWAYS_ZFILL<S,D>>
{
  // Logical thread id to thread idx (one-thread)
  using ThrID = Layout<_1>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,Int<sizeof_bits<S>::value>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,Int<sizeof_bits<D>::value>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // Predicate value that determines whether to load or zfill
  bool pred = false;

  // Overload copy_unpack for zfill variant to pass the predicate into the op
  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr
  void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst)
  {
    static_assert(is_gmem<TS>::value, "Expected gmem source for ppu.cp.async.");
    static_assert(is_smem<TD>::value, "Expected smem destination for ppu.cp.async.");

    Tensor rS = recast<S>(src);
    Tensor rD = recast<D>(dst);

    CUTE_STATIC_ASSERT_V(size(rS) == Int<1>{},
      "In CopyAtom, src layout doesn't vectorize into registers. This src layout is incompatible with this tiled copy.");
    CUTE_STATIC_ASSERT_V(size(rD) == Int<1>{},
      "In CopyAtom, dst layout doesn't vectorize into registers. This dst layout is incompatible with this tiled copy.");

    PPU_CP_ASYNC_CACHEALWAYS_ZFILL<S,D>::copy(rS[0], rD[0], traits.pred);
  }
};

template <class S, class D>
struct Copy_Traits<PPU_CP_ASYNC_CACHEGLOBAL_ZFILL<S,D>>
{
  // Logical thread id to thread idx (one-thread)
  using ThrID = Layout<_1>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,Int<sizeof_bits<S>::value>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,Int<sizeof_bits<D>::value>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // Predicate value that determines whether to load or zfill
  bool pred = false;

  // Overload copy_unpack for zfill variant to pass the predicate into the op
  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr
  void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst)
  {
    static_assert(is_gmem<TS>::value, "Expected gmem source for ppu.cp.async.");
    static_assert(is_smem<TD>::value, "Expected smem destination for ppu.cp.async.");

    Tensor rS = recast<S>(src);
    Tensor rD = recast<D>(dst);

    CUTE_STATIC_ASSERT_V(size(rS) == Int<1>{},
      "In CopyAtom, src layout doesn't vectorize into registers. This src layout is incompatible with this tiled copy.");
    CUTE_STATIC_ASSERT_V(size(rD) == Int<1>{},
      "In CopyAtom, dst layout doesn't vectorize into registers. This dst layout is incompatible with this tiled copy.");

    PPU_CP_ASYNC_CACHEGLOBAL_ZFILL<S,D>::copy(rS[0], rD[0], traits.pred);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Element copy selector
template <class SrcTensor, class DstTensor>
CUTE_HOST_DEVICE constexpr
auto
select_elementwise_copy(SrcTensor const&, DstTensor const&)
{
  using SrcType = typename SrcTensor::value_type;
  using DstType = typename DstTensor::value_type;

#if defined(CUTE_ARCH_CP_ASYNC_PPU_ENABLED)
  if constexpr (is_gmem<SrcTensor>::value && is_smem<DstTensor>::value &&
                sizeof(SrcType) == sizeof(DstType) &&
               (sizeof(SrcType) == 4 || sizeof(SrcType) == 8 || sizeof(SrcType) == 16))
  {
    return PPU_CP_ASYNC_CACHEALWAYS<SrcType,DstType>{};
  } else {
    return UniversalCopy<SrcType,DstType>{};
  }

  CUTE_GCC_UNREACHABLE;
#else
  return UniversalCopy<SrcType,DstType>{};
#endif
}

}
