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


#include "ppu/accutlass.hpp"
#include "ppu/cute/arch/util.hpp"
#include "ppu/cute/algorithm/functional.hpp"
#include "ppu/cute/algorithm/gemm.hpp"

#include "ppu/cute/tensor_mix.hpp"

#if ACOMPUTE_VERSION == 10000
// keep this 2 legacy definition for FA
#include "ppu/cute/arch/mma_ppu.hpp"
#include "ppu/cute/atom/mma_traits_ppu.hpp"

#include "ppu/cute/arch/mma_ppu.hpp"
#include "ppu/cute/arch/copy_ppu0010_aiu.hpp"
#include "ppu/cute/atom/mma_traits_ppu.hpp"
#include "ppu/cute/atom/copy_traits_ppu0010_aiu.hpp"
#include "ppu/cute/swizzle_layout.hpp"
#else
#include "ppu/cute/arch/mma_ppu0015.hpp"
#include "ppu/cute/arch/copy_ppu0015_aiu.hpp"
#include "ppu/cute/atom/mma_traits_ppu0015.hpp"
#include "ppu/cute/atom/copy_traits_ppu0015_aiu.hpp"
#endif

#include "ppu/cute/arch/copy_ppu_aiu.hpp"
#include "ppu/cute/atom/copy_traits_ppu_aiu.hpp"

#include "ppu/cutlass/gemm/dispatch_policy.hpp"

#include "ppu/cutlass/gemm/collective/ppu_mma_aiu_multistage.hpp"
#include "ppu/cutlass/gemm/collective/ppu_mma_aiu_multistage_batch_array.hpp"

#include "ppu/cutlass/gemm/kernel/tile_scheduler.hpp"
#include "ppu/cutlass/gemm/kernel/ppu_aiu_gemm.hpp"
#include "ppu/cutlass/gemm/kernel/ppu_aiu_gemm_batch_array.hpp"
#include "ppu/cutlass/gemm/kernel/ppu_aiu_gemm_array_cooperative.hpp"
#include "ppu/cutlass/gemm/kernel/ppu_aiu_gemm_streamk.hpp"

#if ACOMPUTE_VERSION == 10000
#include "ppu/cutlass/gemm/collective/ppu_mma_aiu_multistage_mixed_input.hpp"
#include "ppu/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input.hpp"
#else
#include "ppu/cutlass/gemm/collective/ppu0015_mma_aiu_multistage_with_scale.hpp"
#include "ppu/cutlass/gemm/collective/ppu0010_mma_multistage_fp4.hpp"
#include "ppu/cutlass/gemm/kernel/ppu0015_gemm.hpp" // fp4 gemm
#endif
#include "ppu/cute/int_tuple.hpp"
#include "ppu/cute/algorithm/copy.hpp"
#include "ppu/cute/layout.hpp"
#include "ppu/cutlass/gemm/collective/ppu0010_mma_twostage_ldmatrix.hpp"
#include "ppu/cutlass/epilogue/thread/conversion_op.h"
#include "ppu/cutlass/epilogue/collective/ppu_epilogue_vectorized_parallel.hpp"
#include "ppu/cutlass/epilogue/collective/ppu_epilogue_vectorized_evt.hpp"
#include "ppu/cutlass/float4.h"
#include "ppu/cute/algorithm/gemm.hpp"
#include "ppu/cutlass/epilogue/fusion/ppu_visitor_load_tma_warpspecialized.hpp"
#include "ppu/cutlass/epilogue/thread/activation.h"


