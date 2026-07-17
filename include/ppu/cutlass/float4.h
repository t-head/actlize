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

#include <cutlass/numeric_size.h>

namespace cutlass {

static float fp4_fp32_map[16] = {
  0.0,
  0.5,
  1.0,
  1.5,
  2.0,
  3.0,
  4.0,
  6.0,
  -0.0,
  -0.5,
  -1.0,
  -1.5,
  -2.0,
  -3.0,
  -4.0,
  -6.0
};

struct float4_t {
  uint8_t storage = 0;

  CUTLASS_HOST_DEVICE
  float to_float() {
    uint8_t input = storage;
    uint32_t sign_bit = ((input >> 3) & 0x1);
    uint32_t exp_bits = ((input >> 1) & 0x3);
    uint32_t mant_bit = (input & 0x1);
    if (exp_bits != 0) {
      mant_bit = (mant_bit | 1u << 1);
      exp_bits = exp_bits - 1;
    }
    float sign_value  = (sign_bit == 0 ? 1.0 : -1.0);
    float exp_value   = exp2(float(exp_bits));
    float mant_value  = mant_bit * 0.5;
    float float_value = sign_value * exp_value * mant_value;
    return float_value;
  }

  // set two fp4 values
  void from_float(float in) {
    int start_idx, end_idx;
    if (in >= 0) {
      start_idx = 7;
      end_idx = 0;
    } else {
      start_idx = 15;
      end_idx = 8;
    }

    in = fabs(in);
    for (int i = start_idx; i >= end_idx; --i) {
      if (in >= fabs(fp4_fp32_map[i])) {
        storage = i;
        return;
      }
    }
  }

  CUTLASS_HOST_DEVICE
  static float convert_fp4_to_float(uint8_t const& x) {
    uint8_t const &input = x;

    uint32_t sign_bit = ((input >> 3) & 0x1);
    uint32_t exp_bits = ((input >> 1) & 0x3);
    uint32_t mant_bit = (input & 0x1);
    if (exp_bits != 0) {
      mant_bit = (mant_bit | 1u << 1);
      exp_bits = exp_bits - 1;
    }
    float sign_value  = (sign_bit == 0 ? 1.0 : -1.0);
    float exp_value   = exp2(float(exp_bits));
    float mant_value  = mant_bit * 0.5;
    float float_value = sign_value * exp_value * mant_value;
    return float_value;
  }

};

} // namespace cutlass
