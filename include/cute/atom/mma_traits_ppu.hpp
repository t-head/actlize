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

#include <cute/arch/mma_ppu.hpp>
#include <cute/atom/mma_traits.hpp>
#include <cute/layout.hpp>
#include <cute/numeric/integer_subbyte.hpp>
#include <cutlass/numeric_types.h>

namespace cute
{

template <>
struct MMA_Traits<PPU_DP4A>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using Shape_MNK = Shape<_1,_1,_4>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape<_1,_4>>;
  using BLayout = Layout<Shape<_1,_4>>;
  using CLayout = Layout<Shape<_1,_1>>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cute



namespace cute
{

namespace {

// Logical thread id to thread idx (quadpair)
using PPU_QuadPair = Layout<Shape <_4, _2>,
                             Stride<_1,_16>>;
// (T8,V4) -> (M8,K4)
using PPU_8x4_Row  = Layout<Shape <_8,_4>,
                             Stride<_1,_8>>;
// (T8,V4) -> (M8,K4)
using PPU_8x4_Col  = Layout<Shape <Shape <_4,_2>,_4>,
                             Stride<Stride<_8,_4>,_1>>;
// (T8,V8) -> (M8,N8)
using PPU_8x8_16b  = Layout<Shape <_8,_8>,
                             Stride<_1,_8>>;
// (T8,V8) -> (M8,N8)
using PPU_8x8_32b  = Layout<Shape <Shape <_2, _2,_2>,Shape <_2,_2, _2>>,
                             Stride<Stride<_1,_16,_4>,Stride<_8,_2,_32>>>;

} 

///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<PPU_8x8x4_F16F16F16F16_TN>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using Shape_MNK = Shape<_8,_8,_4>;
  using ThrID   = PPU_QuadPair;
  using ALayout = PPU_8x4_Row;
  using BLayout = PPU_8x4_Row;
  using CLayout = PPU_8x8_16b;
};

///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<PPU_8x8x4_F16F16F16F16_NT>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using Shape_MNK = Shape<_8,_8,_4>;
  using ThrID   = PPU_QuadPair;
  using ALayout = PPU_8x4_Col;
  using BLayout = PPU_8x4_Col;
  using CLayout = PPU_8x8_16b;
};

///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<PPU_8x8x4_F16F16F16F16_NN>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using Shape_MNK = Shape<_8,_8,_4>;
  using ThrID   = PPU_QuadPair;
  using ALayout = PPU_8x4_Col;
  using BLayout = PPU_8x4_Row;
  using CLayout = PPU_8x8_16b;
};

///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<PPU_8x8x4_F16F16F16F16_TT>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using Shape_MNK = Shape<_8,_8,_4>;
  using ThrID   = PPU_QuadPair;
  using ALayout = PPU_8x4_Row;
  using BLayout = PPU_8x4_Col;
  using CLayout = PPU_8x8_16b;
};

///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<PPU_8x8x4_F32F16F16F32_TN>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_8,_8,_4>;
  using ThrID   = PPU_QuadPair;
  using ALayout = PPU_8x4_Row;
  using BLayout = PPU_8x4_Row;
  using CLayout = PPU_8x8_32b;
};

///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<PPU_8x8x4_F32F16F16F32_NT>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_8,_8,_4>;
  using ThrID   = PPU_QuadPair;
  using ALayout = PPU_8x4_Col;
  using BLayout = PPU_8x4_Col;
  using CLayout = PPU_8x8_32b;
};

///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<PPU_8x8x4_F32F16F16F32_NN>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_8,_8,_4>;
  using ThrID   = PPU_QuadPair;
  using ALayout = PPU_8x4_Col;
  using BLayout = PPU_8x4_Row;
  using CLayout = PPU_8x8_32b;
};

///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<PPU_8x8x4_F32F16F16F32_TT>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_8,_8,_4>;
  using ThrID   = PPU_QuadPair;
  using ALayout = PPU_8x4_Row;
  using BLayout = PPU_8x4_Col;
  using CLayout = PPU_8x8_32b;
};

///////////////////////////////////////////////////////////////////////////////
} // namespace cute



namespace cute
{

template <>
struct MMA_Traits<PPU_16x8x8_F32F16F16F32_TN>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16,_8,_8>;
  using ThrID   = Layout<_32>;
  using ALayout = Layout<Shape <Shape < _4,_8>,Shape < _2,_2>>,
                         Stride<Stride<_32,_1>,Stride<_16,_8>>>;
  using BLayout = Layout<Shape <Shape < _4,_8>,_2>,
                         Stride<Stride<_16,_1>,_8>>;
  using CLayout = Layout<Shape <Shape < _4,_8>,Shape < _2,_2>>,
                         Stride<Stride<_32,_1>,Stride<_16,_8>>>;
};

///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<PPU_8x8x16_S32S8S8S32_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using Shape_MNK = Shape<_8,_8,_16>;
  using ThrID   = Layout<_32>;
  using ALayout = Layout<Shape <Shape < _4,_8>,_4>,
                         Stride<Stride<_32,_1>,_8>>;
  using BLayout = Layout<Shape <Shape < _4,_8>,_4>,
                         Stride<Stride<_32,_1>,_8>>;
  using CLayout = Layout<Shape <Shape < _4,_8>,_2>,
                         Stride<Stride<_16,_1>,_8>>;
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cute



namespace cute
{

namespace {

// (T32,V1) -> (M8,N8)
using PPU_8x4      = Layout<Shape <Shape < _4,_8>,_1>,
                             Stride<Stride< _8,_1>,_0>>;
// (T32,V2) -> (M8,N8)
using PPU_8x8_Row  = Layout<Shape <Shape < _4,_8>,_2>,
                             Stride<Stride<_16,_1>,_8>>;
// (T32,V4) -> (M16,N8)
using PPU_16x8_Row = Layout<Shape <Shape < _4,_8>,Shape < _2,_2>>,
                             Stride<Stride<_32,_1>,Stride<_16,_8>>>;

}

///////////////////////////////////////////////////////////////////////////////
//////////////////////// fp16 = fp16 * fp16 + fp16 ////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<PPU_16x8x8_F16F16F16F16_TN>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using Shape_MNK = Shape<_16,_8,_8>;
  using ThrID   = Layout<_32>;
  using ALayout = PPU_16x8_Row;
  using BLayout = PPU_8x8_Row;
  using CLayout = PPU_16x8_Row;
};

template <>
struct MMA_Traits<PPU_16x8x16_F16F16F16F16_TN>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using Shape_MNK = Shape<_16,_8,_16>;
  using ThrID   = Layout<_32>;
  using ALayout = Layout<Shape <Shape < _4,_8>,Shape < _2,_2,  _2>>,
                         Stride<Stride<_32,_1>,Stride<_16,_8,_128>>>;
  using BLayout = Layout<Shape <Shape < _4,_8>,Shape <_2, _2>>,
                         Stride<Stride<_16,_1>,Stride<_8,_64>>>;
  using CLayout = PPU_16x8_Row;
};

///////////////////////////////////////////////////////////////////////////////
//////////////////////// fp32 = fp16 * fp16 + fp32 ////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<PPU_16x8x8_F32F16F16F32_TNV2>
     : MMA_Traits<PPU_16x8x8_F16F16F16F16_TN>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;
};

template <>
struct MMA_Traits<PPU_16x8x16_F32F16F16F32_TN>
     : MMA_Traits<PPU_16x8x16_F16F16F16F16_TN>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;
};

///////////////////////////////////////////////////////////////////////////////
//////////////////////// fp32 = bf16 * bf16 + fp32 ////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<PPU_16x8x8_F32BF16BF16F32_TN>
     : MMA_Traits<PPU_16x8x8_F16F16F16F16_TN>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;
};

template <>
struct MMA_Traits<PPU_16x8x16_F32BF16BF16F32_TN>
     : MMA_Traits<PPU_16x8x16_F16F16F16F16_TN>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;
};

///////////////////////////////////////////////////////////////////////////////
//////////////////////// fp32 = tf32 * tf32 + fp32 ////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<PPU_16x8x4_F32TF32TF32F32_TN>
{
  using ValTypeD = float;
  using ValTypeA = cutlass::tfloat32_t;
  using ValTypeB = cutlass::tfloat32_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16,_8,_4>;
  using ThrID   = Layout<_32>;
  using ALayout = Layout<Shape <Shape < _4,_8>,_2>,
                         Stride<Stride<_16,_1>,_8>>;
  using BLayout = PPU_8x4;
  using CLayout = PPU_16x8_Row;
};

template <>
struct MMA_Traits<PPU_16x8x8_F32TF32TF32F32_TN>
{
  using ValTypeD = float;
  using ValTypeA = cutlass::tfloat32_t;
  using ValTypeB = cutlass::tfloat32_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16,_8,_8>;
  using ThrID   = Layout<_32>;
  using ALayout = Layout<Shape <Shape < _4,_8>,Shape <_2, _2>>,
                         Stride<Stride<_16,_1>,Stride<_8,_64>>>;
  using BLayout = Layout<Shape <Shape <_4,_8>, _2>,
                         Stride<Stride<_8,_1>,_32>>;
  using CLayout = PPU_16x8_Row;
};

///////////////////////////////////////////////////////////////////////////////
/////////////////////////// s32 = s8 * s8 + s32 ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<PPU_8x8x16_S32S8S8S32_TNV2>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using Shape_MNK = Shape<_8,_8,_16>;
  using ThrID   = Layout<_32>;
  using ALayout = Layout<Shape <Shape < _4,_8>,Shape <_4>>,
                         Stride<Stride<_8,_1>,Stride<_32>>>;
  using BLayout = Layout<Shape <Shape < _4,_8>,Shape <_4>>,
                         Stride<Stride<_8,_1>,Stride<_32>>>;
  using CLayout = PPU_8x8_Row;
};

template <>
struct MMA_Traits<PPU_8x8x16_S32S8S8S32_TN_SATURATE>
     : MMA_Traits<PPU_8x8x16_S32S8S8S32_TNV2> {};

template <>
struct MMA_Traits<PPU_16x8x16_S32S8S8S32_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using Shape_MNK = Shape<_16,_8,_16>;
  using ThrID   = Layout<_32>;
  using ALayout = Layout<Shape <Shape < _4,_8>,Shape < _4,_2>>,
                         Stride<Stride<_64,_1>,Stride<_16,_8>>>;
  using BLayout = Layout<Shape <Shape < _4,_8>,Shape <_4>>,
                         Stride<Stride<_8,_1>,Stride<_32>>>;
  using CLayout = PPU_16x8_Row;
};

template <>
struct MMA_Traits<PPU_16x8x16_S32S8S8S32_TN_SATURATE>
     : MMA_Traits<PPU_16x8x16_S32S8S8S32_TN> {};

template <>
struct MMA_Traits<PPU_16x8x32_S32S8S8S32_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using Shape_MNK = Shape<_16,_8,_32>;
  using ThrID   = Layout<_32>;
  using ALayout = Layout<Shape <Shape < _4,_8>,Shape < _4,_2,  _2>>,
                         Stride<Stride<_64,_1>,Stride<_16,_8,_256>>>;
  using BLayout = Layout<Shape <Shape < _4,_8>, Shape <_4,  _2>>,
                         Stride<Stride<_32,_1>, Stride<_8,_128>>>;
  using CLayout = PPU_16x8_Row;
};

template <>
struct MMA_Traits<PPU_16x8x32_S32S8S8S32_TN_SATURATE>
     : MMA_Traits<PPU_16x8x32_S32S8S8S32_TN> {};

///////////////////////////////////////////////////////////////////////////////
/////////////////////////// s32 = s8 * u8 + s32 ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<PPU_8x8x16_S32S8U8S32_TN>
     : MMA_Traits<PPU_8x8x16_S32S8S8S32_TNV2>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;
};

template <>
struct MMA_Traits<PPU_8x8x16_S32S8U8S32_TN_SATURATE>
     : MMA_Traits<PPU_8x8x16_S32S8U8S32_TN> {};

template <>
struct MMA_Traits<PPU_16x8x16_S32S8U8S32_TN>
     : MMA_Traits<PPU_16x8x16_S32S8S8S32_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;
};

template <>
struct MMA_Traits<PPU_16x8x16_S32S8U8S32_TN_SATURATE>
     : MMA_Traits<PPU_16x8x16_S32S8U8S32_TN> {};

template <>
struct MMA_Traits<PPU_16x8x32_S32S8U8S32_TN>
     : MMA_Traits<PPU_16x8x32_S32S8S8S32_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;
};

template <>
struct MMA_Traits<PPU_16x8x32_S32S8U8S32_TN_SATURATE>
     : MMA_Traits<PPU_16x8x32_S32S8U8S32_TN> {};

///////////////////////////////////////////////////////////////////////////////
/////////////////////////// s32 = u8 * s8 + s32 ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<PPU_8x8x16_S32U8S8S32_TN>
     : MMA_Traits<PPU_8x8x16_S32S8S8S32_TNV2>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;
};

template <>
struct MMA_Traits<PPU_8x8x16_S32U8S8S32_TN_SATURATE>
     : MMA_Traits<PPU_8x8x16_S32U8S8S32_TN> {};

template <>
struct MMA_Traits<PPU_16x8x16_S32U8S8S32_TN>
     : MMA_Traits<PPU_16x8x16_S32S8S8S32_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;
};

template <>
struct MMA_Traits<PPU_16x8x16_S32U8S8S32_TN_SATURATE>
     : MMA_Traits<PPU_16x8x16_S32U8S8S32_TN> {};

template <>
struct MMA_Traits<PPU_16x8x32_S32U8S8S32_TN>
     : MMA_Traits<PPU_16x8x32_S32S8S8S32_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;
};

template <>
struct MMA_Traits<PPU_16x8x32_S32U8S8S32_TN_SATURATE>
     : MMA_Traits<PPU_16x8x32_S32U8S8S32_TN> {};

///////////////////////////////////////////////////////////////////////////////
/////////////////////////// s32 = u8 * u8 + s32 ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<PPU_8x8x16_S32U8U8S32_TN>
     : MMA_Traits<PPU_8x8x16_S32S8S8S32_TNV2>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;
};

template <>
struct MMA_Traits<PPU_8x8x16_S32U8U8S32_TN_SATURATE>
     : MMA_Traits<PPU_8x8x16_S32U8U8S32_TN> {};

template <>
struct MMA_Traits<PPU_16x8x16_S32U8U8S32_TN>
     : MMA_Traits<PPU_16x8x16_S32S8S8S32_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;
};

template <>
struct MMA_Traits<PPU_16x8x16_S32U8U8S32_TN_SATURATE>
     : MMA_Traits<PPU_16x8x16_S32U8U8S32_TN> {};

template <>
struct MMA_Traits<PPU_16x8x32_S32U8U8S32_TN>
     : MMA_Traits<PPU_16x8x32_S32S8S8S32_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;
};

template <>
struct MMA_Traits<PPU_16x8x32_S32U8U8S32_TN_SATURATE>
     : MMA_Traits<PPU_16x8x32_S32U8U8S32_TN> {};


///////////////////////////////////////////////////////////////////////////////
/////////////////////////// s32 = s4 * s4 + s32 ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////
template <>
struct MMA_Traits<PPU_16x8x64_S32S4S4S32_TN>
{
//   using ValTypeD = int32_t;
//   using ValTypeA = int8_t;
//   using ValTypeB = int8_t;
//   using ValTypeC = int32_t;

//   using Shape_MNK = Shape<_16,_8,_32>;
//   using ThrID   = Layout<_32>;
//   using ALayout = Layout<Shape <Shape < _4,_8>,Shape < _4,_2,  _2>>,
//                          Stride<Stride<_64,_1>,Stride<_16,_8,_256>>>;
//   using BLayout = Layout<Shape <Shape < _4,_8>, Shape <_4,  _2>>,
//                          Stride<Stride<_32,_1>, Stride<_8,_128>>>;
//   using CLayout = PPU_16x8_Row;

  using ElementDVal = int32_t;
  using ElementAVal = int8_t;
  using ElementBVal = int8_t;
  using ElementCVal = int32_t;

  using Shape_MNK = Shape<_16,_8,_32>;
  using ThrID   = Layout<_32>;
  using ALayout = Layout<Shape <Shape < _4,_8>,Shape < _4,_2,  _2>>,
                         Stride<Stride<_64,_1>,Stride<_16,_8,_256>>>;
  using BLayout = Layout<Shape <Shape < _4,_8>, Shape <_4,  _2>>,
                         Stride<Stride<_32,_1>, Stride<_8,_128>>>;
  using CLayout = PPU_16x8_Row;


//   using ElementDVal = int32_t;
//   using ElementAVal = int4b_t;
//   using ElementBVal = int4b_t;
//   using ElementCVal = int32_t;

//   using Shape_MNK = Shape<_16,_8,_64>;
//   using ThrID   = Layout<_32>;
//   using ALayout = Layout<Shape <Shape < _4,_8>,Shape < _8,_2,  _2>>,
//                          Stride<Stride<_128,_1>,Stride<_16,_8,_512>>>;
//   using BLayout = Layout<Shape <Shape < _4,_8>, Shape <_8,  _2>>,
//                          Stride<Stride<_64,_1>, Stride<_8,_256>>>;
//   using CLayout = PPU_16x8_Row;
};


///////////////////////////////////////////////////////////////////////////////
/////////////////////////// s32 = b1 ^ b1 + s32 ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<PPU_16x8x256_S32U1U1S32_TN_XORPOPC>
{
  using ValTypeD = int32_t;
  using ValTypeA = cute::uint1b_t;
  using ValTypeB = cute::uint1b_t;
  using ValTypeC = int32_t;

  using Shape_MNK = Shape<_16,_8,_256>;
  using ThrID   = Layout<_32>;
  using ALayout = Layout<Shape <_32,Shape < _8, _4,_2,   _2>>,
                         Stride<_64,Stride<_64,_16,_8,_2048>>>;
  using BLayout = Layout<Shape <_32,Shape <_32,   _2>>,
                         Stride<_32,Stride< _1,_1024>>>;
  using CLayout = PPU_16x8_Row;
};
} // end namespace cute

