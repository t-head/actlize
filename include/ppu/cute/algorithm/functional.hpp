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

#include <cute/util/type_traits.hpp>
#include <cutlass/tfloat32.h>

namespace cute {

CUTLASS_HOST_DEVICE
static tfloat32_t fp32_to_tf32(uint32_t x) {

  #if defined(__HGGC_ARCH__) || defined(__HGGC_ARCH__)
    asm volatile("ppu.cvt.rtte.tf32.f32 %0, %1;" : "=r"(x) : "r"(x));
  #else
    float s = *(reinterpret_cast<float*>(&x));
    if (std::isfinite(s)) {
      x += 0x1000u;
    }
  #endif

  return tfloat32_t::bitcast(x);
}

// cutlass3 use unary op for dtype convert, only now mma type
// gemm config will use this converter to convert to tf32 when input is fp32
template <class T>
struct convert;

template <>
struct convert<cutlass::tfloat32_t> {
  CUTE_HOST_DEVICE
  decltype(auto) operator()(cutlass::tfloat32_t &arg) const {
    arg = fp32_to_tf32(arg.storage);
    return arg;
  }
};

} // namespace cut
