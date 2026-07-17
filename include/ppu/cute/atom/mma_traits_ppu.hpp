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

#include "ppu/cute/arch/mma_ppu.hpp"
#include <cute/atom/mma_traits.hpp>

#include <cute/layout.hpp>

#include <cute/numeric/integer_subbyte.hpp>

#include <cutlass/numeric_types.h>

namespace cute
{

namespace {

using PPU_16x16_Row = Layout<Shape <Shape < _4,_8>,Shape < _4,_2>>,
                              Stride<Stride<_16,_1>,Stride<_64,_8>>>;

}

template <>
struct MMA_Traits<PPU_16x16x16_F16F16F16F16_TN>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using Shape_MNK = Shape<_16,_16,_16>;
  using ThrID   = Layout<_32>;
  using ALayout = Layout<Shape <Shape < _4,_8>,Shape < _2,_2,  _2>>,
                         Stride<Stride<_32,_1>,Stride<_16,_128,_8>>>;
  using BLayout = Layout<Shape <Shape < _4,_8>,Shape < _2,_2,_2>>,
                         Stride<Stride<_32,_1>,Stride<_16,_128,_8>>>;
  using CLayout = PPU_16x16_Row;
};

template <>
struct MMA_Traits<PPU_16x16x16_F32F16F16F32_TN>
     : MMA_Traits<PPU_16x16x16_F16F16F16F16_TN>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;
};

template <>
struct MMA_Traits<PPU_16x16x16_F32BF16BF16F32_TN>
     : MMA_Traits<PPU_16x16x16_F16F16F16F16_TN>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;
};

template <>
struct MMA_Traits<PPU_16x16x8_F32TF32TF32F32_TN>
{
  using ValTypeD = float;
  using ValTypeA = cutlass::tfloat32_t;
  using ValTypeB = cutlass::tfloat32_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16,_16,_8>;
  using ThrID   = Layout<_32>;
  using ALayout = Layout<Shape <Shape < _4,_8>,Shape <_2, _2>>,
                         Stride<Stride<_16,_1>,Stride<_64,_8>>>;
  using BLayout = Layout<Shape <Shape < _4,_8>,Shape <_2, _2>>,
                         Stride<Stride<_16,_1>,Stride<_64,_8>>>;
  using CLayout = PPU_16x16_Row;
};

template <>
struct MMA_Traits<PPU_16x16x32_S32S8S8S32_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using Shape_MNK = Shape<_16,_16,_32>;
  using ThrID   = Layout<_32>;
  using ALayout = Layout<Shape <Shape < _4,_8>,Shape < _4,_2,  _2>>,
                         Stride<Stride<_64,_1>,Stride<_16,_256,_8>>>;
  using BLayout = Layout<Shape <Shape < _4,_8>,Shape < _4,_2,  _2>>,
                         Stride<Stride<_64,_1>,Stride<_16,_256,_8>>>;
  using CLayout = PPU_16x16_Row;
};

template <>
struct MMA_Traits<PPU_16x16x32_S32S8U8S32_TN>
     : MMA_Traits<PPU_16x16x32_S32S8S8S32_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;
};

template <>
struct MMA_Traits<PPU_16x16x32_S32U8S8S32_TN>
     : MMA_Traits<PPU_16x16x32_S32S8S8S32_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;
};

template <>
struct MMA_Traits<PPU_16x16x32_S32U8U8S32_TN>
     : MMA_Traits<PPU_16x16x32_S32S8S8S32_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;
};

namespace {

using PPU_8x16_Row = Layout<Shape <Shape < _4,_8>,Shape <_4>>,
                              Stride<Stride<_8,_1>,Stride<_32>>>;

}

template <>
struct MMA_Traits<PPU_8x16x16_F32F16F16F32_TN>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_8,_16,_16>;
  using ThrID   = Layout<_32>;
  using ALayout = Layout<Shape <Shape < _4,_8>,Shape < _2,_2>>,
                        Stride<Stride<_16,_1>,Stride<_8,_64>>>;
  using BLayout = Layout<Shape <Shape < _4,_8>,Shape < _2,_2,_2>>,
                         Stride<Stride<_32,_1>,Stride<_16,_128,_8>>>;
  using CLayout = PPU_8x16_Row;
};

template <>
struct MMA_Traits<PPU_8x16x16_F32BF16BF16F32_TN>
     : MMA_Traits<PPU_8x16x16_F32F16F16F32_TN>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;
};

} // namespace cute

