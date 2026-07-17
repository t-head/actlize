/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved. 
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

template <class NumBitsPerAIU, typename Element, bool Trans, int CUBE_H, int CUBE_W>
struct Copy_Traits<PPU_AIU_LOAD<NumBitsPerAIU, Element, Trans, CUBE_H, CUBE_W>>
{
  using ThrID   = Layout<_1>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerAIU>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerAIU>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // can't use reference since CopyAtom has no constructor
  AiuDesc desc_;

  template <class Coord, int... Is>
  CUTE_HOST_DEVICE constexpr
  void
  copy_unpack_(void *dst_ptr, const void *src_ptr,
               Coord const& src_coord, seq<Is...>) const
  {
    PPU_AIU_LOAD<NumBitsPerAIU, Element, Trans, CUBE_H, CUBE_W>::copy(dst_ptr, src_ptr, desc_, get<Is>(src_coord)...);
  }

  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr
  void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst)
  {
    static_assert(is_smem<TD>::value, "Expected smem dst for AIU_LOAD");

    // if (cute::thread0()) {
    //   print("        copy_unpack, src = "); print(src); print("\n");
    //   print("                     dst = "); print(dst); print("\n");
    // }

    if constexpr (is_mix_iterator<typename TS::iterator>::value) {
      traits.copy_unpack_(cute::raw_pointer_cast(dst.data()), src.data().ptr_.get(), src.data().coord_, tuple_seq<decltype(src.data().coord_)>{});
    } else {
      traits.copy_unpack_(cute::raw_pointer_cast(dst.data()), traits.desc_.gmem_ptr, src.data().coord_, tuple_seq<decltype(src.data().coord_)>{});
    }
  }

};


template <typename Element, int CUBE_H, int CUBE_W, bool Swap, bool Trans, int InstNum>
struct Copy_Traits<PPU_TSM_LD_SWZL<Element, CUBE_H, CUBE_W, Swap, Trans, InstNum>>
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

  void *smem_base_;

  template <class Coord, int... Is>
  CUTE_HOST_DEVICE constexpr
  void
  copy_unpack_(void *dst_ptr, void* src_ptr,
               Coord const& src_coord, seq<Is...>) const
  {
    // PPU0015_TSM_LD_SWZL<Element, CUBE_H, CUBE_W, Swap, Trans, InstNum>::copy(dst_ptr, smem_base_, get<Is>(src_coord)...);
    PPU_TSM_LD_SWZL<Element, CUBE_H, CUBE_W, Swap, Trans, InstNum>::copy(dst_ptr, src_ptr, get<Is>(src_coord)...);
  }

  template <class TS, class SLayout,
          class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr
  void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst)
  {
    if constexpr (is_mix_iterator<typename TS::iterator>::value) {
      traits.copy_unpack_(cute::raw_pointer_cast(dst.data()), src.data().ptr_.get(), src.data().coord_, tuple_seq<decltype(src.data().coord_)>{});
    } else {
      traits.copy_unpack_(cute::raw_pointer_cast(dst.data()), traits.smem_base_, src.data().coord_, tuple_seq<decltype(src.data().coord_)>{});
    }
    // traits.copy_unpack_(cute::raw_pointer_cast(dst.data()), src.data().coord_, tuple_seq<decltype(src.data().coord_)>{});
  }
};


} // namespace cute
