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

namespace cutlass {
namespace conv {
namespace threadblock {

template <typename Element, int CUBE_H, int CUBE_W>
struct CONV_AIU_LOAD_2D_IMPL;

template <int CUBE_H, int CUBE_W>
struct CONV_AIU_LOAD_2D_IMPL<half_t, CUBE_H, CUBE_W>
{
  static constexpr int swzl_mode = CUBE_W * 2 == 128 ? 0 : 1;

  CUTLASS_HOST_DEVICE void
  operator()(void *tsm_ptr, const uint8_t* global_ptr, int dim_w, int dim_h, uint64_t stride_w, uint64_t stride_h, int coord_w, int coord_h) {
    asm volatile(
      "ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.padz.swzl.b16 [%0], [%1], "
      "{%2, %3, %4}, {%5, %6, %7}, {%8, %9}, {%10, %11, %12}, %13;\n"
      :: "r"(tsm_ptr), "l"(global_ptr),
        "r"(dim_w), "r"(dim_h), "r"(1),
        "r"(CUBE_W), "r"(CUBE_H), "r"(1),
        "l"(stride_h), "l"(coord_w),
        "r"(coord_w), "r"(coord_h), "r"(0),
        "r"(swzl_mode)
    );

    // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
    //   printf("===========\n");
    //   half_t *tmp = reinterpret_cast<half_t*>(tsm_ptr);
    //   printf("%f, %f, 111111111\n", float(tmp[0]), float(tmp[0]));
    //   printf("tsm_ptr: %p\n", tsm_ptr);
    //   printf("dim: %d, %d\n", dim_w, dim_h);
    //   printf("stride: %ld, %ld\n", stride_w, stride_h);
    //   printf("cube: %d, %d\n", CUBE_H, CUBE_W);
    //   printf("start: %d, %d\n", coord_w, coord_h);
    //   printf("SwzlMode: %d\n", swzl_mode);
    // }
  }
};

template <int CUBE_H, int CUBE_W>
struct CONV_AIU_LOAD_2D_IMPL<bfloat16_t, CUBE_H, CUBE_W>
     : CONV_AIU_LOAD_2D_IMPL<half_t, CUBE_H, CUBE_W>
{};

template <int CUBE_H, int CUBE_W>
struct CONV_AIU_LOAD_2D_IMPL<float, CUBE_H, CUBE_W>
{
  static constexpr int swzl_mode = CUBE_W * 4 == 128 ? 0 : 1;

  CUTLASS_HOST_DEVICE void
  operator()(void *tsm_ptr, const uint8_t* global_ptr, int dim_w, int dim_h, uint64_t stride_w, uint64_t stride_h, int coord_w, int coord_h) {
    asm volatile(
      "ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.padz.swzl.b32 [%0], [%1], "
      "{%2, %3, %4}, {%5, %6, %7}, {%8, %9}, {%10, %11, %12}, %13;\n"
      :: "r"(tsm_ptr), "l"(global_ptr),
        "r"(dim_w), "r"(dim_h), "r"(1),
        "r"(CUBE_W), "r"(CUBE_H), "r"(1),
        "l"(stride_w), "l"(stride_h),
        "r"(coord_w), "r"(coord_h), "r"(0),
        "r"(swzl_mode)
    );


    // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
    //   printf("===========\n");
    //   printf("tsm_ptr: %p\n", tsm_ptr);
    //   printf("dim: %d, %d\n", dim_w, dim_h);
    //   printf("stride: %ld, %ld\n", stride_w, stride_h);
    //   printf("cube: %d, %d\n", CUBE_H, CUBE_W);
    //   printf("start: %d, %d\n", coord_w, coord_h);
    //   printf("SwzlMode: %d\n", swzl_mode);
    // }
  }
};

template <int CUBE_H, int CUBE_W>
struct CONV_AIU_LOAD_2D_IMPL<tfloat32_t, CUBE_H, CUBE_W>
     : CONV_AIU_LOAD_2D_IMPL<float, CUBE_H, CUBE_W>
{};


template <typename Element, int CUBE_H, int CUBE_W>
struct CONV_AIU_LOAD_5D_IMPL;

template <int CUBE_H, int CUBE_W>
struct CONV_AIU_LOAD_5D_IMPL<half_t, CUBE_H, CUBE_W>
{
  static constexpr int swzl_mode = CUBE_W * 2 == 128 ? 0 : 1;

  CUTLASS_HOST_DEVICE void
  operator()(
    void *tsm_ptr, const uint8_t* global_ptr,
    int dim_c, int dim_w, int dim_h, int dim_d, int dim_n,
    uint64_t stride_w, uint64_t stride_h, uint64_t stride_d, uint64_t stride_n,
    int lower_w, int lower_h, int lower_d,
    int upper_w, int upper_h, int upper_d,
    int conv_stride_w, int conv_stride_h, int conv_stride_d, int conv_stride_n,
    int coord_c, int coord_w, int coord_h, int coord_d, int coord_n,
    int cur_s, int cur_r, int cur_t
  ) {
    asm volatile(
      "ppu.cp.async.aiu.bulk.tensor.shared.global.5d.im2col.padz.swzl.b16 [%0], [%1], "
      "{%2, %3, %4, %5, %6},"
      "{%7, %8, %9, %10},"
      "{%11, %12, %13},"
      "{%14, %15, %16},"
      "{%17, %18},"
      "{%19, %20, %21, %22},"
      "{%23, %24, %25, %26, %27},"
      "{%28, %29, %30},"
      "%31;\n"
      :: "r"(tsm_ptr), "l"(global_ptr),
        "r"(dim_c), "r"(dim_w), "r"(dim_h), "r"(dim_d), "r"(dim_n),
        "r"(stride_w), "r"(stride_h), "r"(stride_d), "r"(stride_n),
        "r"(lower_w), "r"(lower_h), "r"(lower_d),
        "r"(upper_w), "r"(upper_h), "r"(upper_d),
        "r"(CUBE_W), "r"(CUBE_H),
        "r"(conv_stride_w), "r"(conv_stride_h), "r"(conv_stride_d), "r"(conv_stride_n),
        "r"(coord_c), "r"(coord_w), "r"(coord_h), "r"(coord_d), "r"(coord_n),
        "r"(cur_s), "r"(cur_r), "r"(cur_t),
        "r"(swzl_mode)
    );

    // if (blockIdx.x == 3 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
    //   printf("===========\n");
    //   half_t *tmp = reinterpret_cast<half_t*>(tsm_ptr);
    //   printf("%f, %f, %f, %f, %f, 111111\n", float(tmp[0]), float(tmp[1*64]), float(tmp[2*64]), float(tmp[3*64]), float(tmp[16*64]));
    //   printf("tsm_ptr: %p\n", tsm_ptr);
    //   printf("dim: %d, %d, %d, %d\n", dim_c, dim_w, dim_h, dim_n);
    //   printf("stride: %ld, %ld, %ld\n", stride_w, stride_h, stride_n);
    //   printf("lower: %d, %d, upper: %d, %d\n", lower_w, lower_h, upper_w, upper_h);
    //   printf("cube: %d, %d\n", CUBE_H, CUBE_W);
    //   printf("conv_stride: %d, %d\n", conv_stride_w, conv_stride_h);
    //   printf("start: %d, %d, %d, %d\n", coord_c, coord_w, coord_h, coord_n);
    //   printf("im2col_offset: %d, %d\n", cur_s, cur_r);
    //   printf("SwzlMode: %d\n", swzl_mode);
    // }
  }
};

template <int CUBE_H, int CUBE_W>
struct CONV_AIU_LOAD_5D_IMPL<bfloat16_t, CUBE_H, CUBE_W>
     : CONV_AIU_LOAD_5D_IMPL<half_t, CUBE_H, CUBE_W>
{};

template <int CUBE_H, int CUBE_W>
struct CONV_AIU_LOAD_5D_IMPL<float, CUBE_H, CUBE_W>
{
  static constexpr int swzl_mode = CUBE_W * 4 == 128 ? 0 : 1;

  CUTLASS_HOST_DEVICE void
  operator()(
    void *tsm_ptr, const uint8_t* global_ptr,
    int dim_c, int dim_w, int dim_h, int dim_d, int dim_n,
    uint64_t stride_w, uint64_t stride_h, uint64_t stride_d, uint64_t stride_n,
    int lower_w, int lower_h, int lower_d,
    int upper_w, int upper_h, int upper_d,
    int conv_stride_w, int conv_stride_h, int conv_stride_d, int conv_stride_n,
    int coord_c, int coord_w, int coord_h, int coord_d, int coord_n,
    int cur_s, int cur_r, int cur_t
  ) {
    // printf("%d, %d, %d, 1111111\n", CUBE_W, CUBE_H, swzl_mode);
    asm volatile(
      "ppu.cp.async.aiu.bulk.tensor.shared.global.5d.im2col.padz.swzl.b32 [%0], [%1], "
      "{%2, %3, %4, %5, %6},"
      "{%7, %8, %9, %10},"
      "{%11, %12, %13},"
      "{%14, %15, %16},"
      "{%17, %18},"
      "{%19, %20, %21, %22},"
      "{%23, %24, %25, %26, %27},"
      "{%28, %29, %30},"
      "%31;\n"
      :: "r"(tsm_ptr), "l"(global_ptr),
        "r"(dim_c), "r"(dim_w), "r"(dim_h), "r"(dim_d), "r"(dim_n),
        "r"(stride_w), "r"(stride_h), "r"(stride_d), "r"(stride_n),
        "r"(lower_w), "r"(lower_h), "r"(lower_d),
        "r"(upper_w), "r"(upper_h), "r"(upper_d),
        "r"(CUBE_W), "r"(CUBE_H),
        "r"(conv_stride_w), "r"(conv_stride_h), "r"(conv_stride_d), "r"(conv_stride_n),
        "r"(coord_c), "r"(coord_w), "r"(coord_h), "r"(coord_d), "r"(coord_n),
        "r"(cur_s), "r"(cur_r), "r"(cur_t),
        "r"(swzl_mode)
    );

    // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
    //   printf("===========\n");
    //   printf("tsm_ptr: %p\n", tsm_ptr);
    //   printf("dim: %d, %d, %d, %d\n", dim_c, dim_w, dim_h, dim_n);
    //   printf("stride: %ld, %ld, %ld\n", stride_w, stride_h, stride_n);
    //   printf("lower: %d, %d, upper: %d, %d\n", lower_w, lower_h, upper_w, upper_h);
    //   printf("cube: %d, %d\n", CUBE_H, CUBE_W);
    //   printf("conv_stride: %d, %d\n", conv_stride_w, conv_stride_h);
    //   printf("start: %d, %d, %d, %d\n", coord_c, coord_w, coord_h, coord_n);
    //   printf("im2col_offset: %d, %d\n", cur_s, cur_r);
    //   printf("SwzlMode: %d\n", swzl_mode);
    // }

  }
};

template <int CUBE_H, int CUBE_W>
struct CONV_AIU_LOAD_5D_IMPL<tfloat32_t, CUBE_H, CUBE_W>
     : CONV_AIU_LOAD_5D_IMPL<float, CUBE_H, CUBE_W>
{};

} // namespace threadblock
} // namespace conv
} // namespace cutlass
