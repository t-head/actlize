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

#include "cutlass/arch/arch.h"
#include "cutlass/gemm/gemm.h"

#include "cute/layout.hpp"
#include "cute/numeric/integral_constant.hpp"
//////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm {
using namespace cute;

//////////////////////////////////////////////////////////////////////////////

//
// Kernel schedule policies (the base class tags, one for each kernel layer file)
//
struct KernelMultistage { };
struct KernelCpAsyncWarpSpecialized { };
struct KernelCpAsyncWarpSpecializedPingpong { };
struct KernelCpAsyncWarpSpecializedCooperative { };
struct KernelTma { };
struct KernelTmaWarpSpecialized { };
struct KernelTmaWarpSpecializedPingpong { };
struct KernelTmaWarpSpecializedCooperative { };
struct KernelPtrArrayTmaWarpSpecializedCooperative { };

//////////////////////////////////////////////////////////////////////////////

//
// Builder dispatch policies (not a part of the main CUTLASS layers, simply used to opt into 
// specific collective builder dispatches)
//

// FP8 related policies (including Fast Accumulation)
struct KernelTmaWarpSpecializedFP8FastAccum : KernelTmaWarpSpecialized { };
struct KernelTmaWarpSpecializedPingpongFP8FastAccum : KernelTmaWarpSpecializedPingpong { };
struct KernelTmaWarpSpecializedCooperativeFP8FastAccum: KernelTmaWarpSpecializedCooperative { };
struct KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum : KernelPtrArrayTmaWarpSpecializedCooperative { };

// Policies to opt into mixed type GEMMs
struct KernelTmaWarpSpecializedMixedInput : KernelTmaWarpSpecialized { };
struct KernelTmaWarpSpecializedPingpongMixedInput : KernelTmaWarpSpecializedPingpong { };
struct KernelTmaWarpSpecializedCooperativeMixedInput: KernelTmaWarpSpecializedCooperative { };

//////////////////////////////////////////////////////////////////////////////

// Policies for dispatch of epilogue
struct EpilogueDefault { };
struct EpilogueTransposed { };

//////////////////////////////////////////////////////////////////////////////

//
// Collective Mainloop Policies
//

// 2 stage pipeline through 1 stage in smem, 1 in rmem, WITHOUT predicated gmem loads
struct MainloopPPU0010TwoStageUnpredicated {
  constexpr static int Stages = 2;
  using ArchTag = arch::PPU0010;
  using Schedule = KernelMultistage;
  using ClusterShape = Shape<_1,_1,_1>;
};

// 2 stage pipeline through 1 stage in smem, 1 in rmem, with predicated gmem loads
struct MainloopPPU0010TwoStage {
  constexpr static int Stages = 2;
  using ArchTag = arch::PPU0010;
  using Schedule = KernelMultistage;
  using ClusterShape = Shape<_1,_1,_1>;
};

// n-buffer in smem (cp.async), pipelined with registers, WITHOUT predicated gmem loads
template<int Stages_>
struct MainloopPPU0010CpAsyncUnpredicated {
  constexpr static int Stages = Stages_;
  using ArchTag = arch::PPU0010;
  using Schedule = KernelMultistage;
  using ClusterShape = Shape<_1,_1,_1>;
};

// n-buffer in smem (cp.async), pipelined with registers, with predicated gmem loads
template<int Stages_>
struct MainloopPPU0010CpAsync {
  constexpr static int Stages = Stages_;
  using ArchTag = arch::PPU0010;
  using Schedule = KernelMultistage;
  using ClusterShape = Shape<_1,_1,_1>;
};

//////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm

