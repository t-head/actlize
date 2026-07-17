/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved. 
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Architecture-specific operators on memory added for PPU
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory_ppu.h"
#include "cutlass/arch/cache_operation.h"
#include "hggc_pipeline.h"

#ifndef PPU_CP_ASYNC_ACTIVATED
#if defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 100)
  #define PPU_CP_ASYNC_ACTIVATED 1
#else
  #define PPU_CP_ASYNC_ACTIVATED 0
#endif
#endif


namespace cutlass {
namespace arch {

////////////////////////////////////////////////////////////////////////////////////////////////////

// get swizzle offset with element size, only support tf32
template<typename Element>
CUTLASS_DEVICE
int get_tsm_swiz_off(int slice_stride, int slice_size, int row, int c) {
  int slice_id = c / slice_size;
  // element offset in slice
  int slice_off = c % slice_size;

  // how many elements in one vector
  int elem_per_vec = 16 / sizeof(Element);

  // which 128b in slice, 4 fp32 is 128b
  int vector_idx = slice_off / elem_per_vec;
  // elem offset in vector
  int vector_off = slice_off % elem_per_vec;

  // which cache line in the slice, always 4 row one cache line
  int line_id = row / 4;
  // slice offset in cur cache line
  int line_off = row % 4;
  int slice_in_line = (((slice_id & 1) << 1) + ((slice_id & 2) >> 1) + line_off) % 4;

  // swizzle between two vector in odd cache line
  int vector_idx_swiz = vector_idx ^ (line_id % 2);

  // elements in one cache line
  int elem_per_line = 128 / sizeof(Element);

  // one line is 32 fp32, one slice is 8 fp32
  return slice_id * slice_stride + line_id * elem_per_line + slice_in_line * slice_size + vector_idx_swiz * elem_per_vec + vector_off;
}

template<typename Element>
CUTLASS_DEVICE
void tsm_load_cube_sim(Element const *global_ptr, Element *tsm_ptr,
                       int tensor_dim_n, int tensor_dim_d, int tensor_dim_h, int tensor_dim_w, int tensor_dim_c,
                       int tensor_stride_w, int c_merge_cnt,
                       int offset_n, int offset_d, int offset_h, int offset_w, int offset_c,
                       int cube_ndhw, int cube_c,
                       int pad_d, int pad_h, int pad_w,
                       int kernel_stride_d, int kernel_stride_h, int kernel_stride_w,
                       int dim_t, int dim_r, int dim_s, int cur_t, int cur_r, int cur_s,
                       int dilation_d, int dilation_h, int dilation_w,
                       // 0: fprop_x/wgrad_x
                       // 1: dgrad_dy
                       int conv_type = 0) {
  if (threadIdx.x != 0) {
    return;
  }


  // element size of each slice
  int slice_size = 32 / sizeof(Element);
  int slice_stride = cube_ndhw * slice_size;

  // valid length in each rs position
  int valid_c_length = max(0, min(cube_c, tensor_dim_c - offset_c));

  int valid_start_d, valid_start_h, valid_start_w, valid_end_d, valid_end_h, valid_end_w;

  int step_w = kernel_stride_w;
  int step_h = kernel_stride_h;
  int step_d = kernel_stride_d;

  // in dgrad_dy, tensor_dim_w is size of dx, should get dim_q to decide validity and get offset
  // if transfer dy dim directly, the calculated dx dim is not correct, and result in wrong w_steps/valid_end_w
  int tensor_actual_dim_w = tensor_dim_w;
  int tensor_actual_dim_h = tensor_dim_h;
  int tensor_actual_dim_d = tensor_dim_d;


  if (conv_type == 0) {
    valid_start_d = cur_t * dilation_d - pad_d;
    valid_start_h = cur_r * dilation_h - pad_h;
    valid_start_w = cur_s * dilation_w - pad_w;
    // get output tensor size to get valid_start/end position
    // int tensor_dim_z = (tensor_dim_d + 2 * pad_d - dilation_d * (dim_t - 1) - 1) / kernel_stride_d + 1;
    // int tensor_dim_p = (tensor_dim_h + 2 * pad_h - dilation_h * (dim_r - 1) - 1) / kernel_stride_h + 1;
    // int tensor_dim_q = (tensor_dim_w + 2 * pad_w - dilation_w * (dim_s - 1) - 1) / kernel_stride_w + 1;
    // int valid_end_d = valid_start_d + (tensor_dim_z - 1) * kernel_stride_d;
    // int valid_end_h = valid_start_h + (tensor_dim_p - 1) * kernel_stride_h;
    // int valid_end_w = valid_start_w + (tensor_dim_q - 1) * kernel_stride_w;
    // remove /stride*stride, when stride=3, valid_end may be 1-2 bigger than actual value
    // but offset adds stride each time, step number will not change
    valid_end_d = valid_start_d + (tensor_dim_d + 2 * pad_d - dilation_d * (dim_t - 1) - 1);
    valid_end_h = valid_start_h + (tensor_dim_h + 2 * pad_h - dilation_h * (dim_r - 1) - 1);
    valid_end_w = valid_start_w + (tensor_dim_w + 2 * pad_w - dilation_w * (dim_s - 1) - 1);
  } else if (conv_type == 1) {
    // dy in dgrad always load successively, since nhw in dx are in same rs position
    step_w = 1;
    step_h = 1;
    step_d = 1;

    // input offset_w is actually offset_q, get real offset_w
    int actual_offset_w = -pad_w + offset_w * kernel_stride_w + cur_s * dilation_w;
    // first w in next h in current block, always with same rs position, means neighbour w interval stride positions
    int next_start_w = actual_offset_w % kernel_stride_w;
    // first valid w with valid dy in cur_s
    int next_valid_w = -pad_w + cur_s * dilation_w;
    // how many dy between start_w and valid_w
    valid_start_w = (next_start_w - next_valid_w) / kernel_stride_w;
    int tensor_dim_q = (tensor_dim_w + 2 * pad_w - dilation_w * (dim_s - 1) - 1) / kernel_stride_w + 1;
    // w advance steps in cur rs space
    int w_steps = (tensor_dim_w - 1 - next_start_w) / kernel_stride_w;
    valid_end_w = valid_start_w + w_steps;
    tensor_actual_dim_w = tensor_dim_q;

    int actual_offset_h = -pad_h + offset_h * kernel_stride_h + cur_r * dilation_h;
    int next_start_h = actual_offset_h % kernel_stride_h;
    int next_valid_h = -pad_h + cur_r * dilation_h;
    valid_start_h = (next_start_h - next_valid_h) / kernel_stride_h;
    int tensor_dim_p = (tensor_dim_h + 2 * pad_h - dilation_h * (dim_r - 1) - 1) / kernel_stride_h + 1;
    int h_steps = (tensor_dim_h - 1 - next_start_h) / kernel_stride_h;
    valid_end_h = valid_start_h + h_steps;
    tensor_actual_dim_h = tensor_dim_p;

    int actual_offset_d = -pad_d + offset_d * kernel_stride_d + cur_t * dilation_d;
    int next_start_d = actual_offset_d % kernel_stride_d;
    int next_valid_d = -pad_d + cur_t * dilation_d;
    valid_start_d = (next_start_d - next_valid_d) / kernel_stride_d;
    int tensor_dim_z = (tensor_dim_d + 2 * pad_d - dilation_d * (dim_t - 1) - 1) / kernel_stride_d + 1;
    int d_steps = (tensor_dim_d - 1 - next_start_d) / kernel_stride_d;
    valid_end_d = valid_start_d + d_steps;
    tensor_actual_dim_d = tensor_dim_z;

    // if (blockIdx.x == 0 && blockIdx.y == 1 && threadIdx.x == 0) {
    //   printf("next_start_w: %d, next_valid_w: %d, ori_w: %d, w_steps: %d\n", next_start_w, next_valid_w, tensor_dim_w, w_steps);
    // }
  }


  // if (blockIdx.x == 0 && blockIdx.y == 1 && threadIdx.x == 0) {
  //   printf("cur_r: %d, cur_s: %d, pad_h: %d, pad_w: %d\n", cur_r, cur_s, pad_h, pad_w);
  //   printf("start_h: %d, start_w: %d, end_h: %d, end_w: %d\n", valid_start_h, valid_start_w, valid_end_h, valid_end_w);
  // }

  for (int h = 0; h < cube_ndhw; h++) {
    // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && h <= 8) {
    //   printf("h: %d, off_n: %d, off_h: %d, off_w: %d, dim_h: %d, dim_w: %d\n", h, offset_n, offset_h, offset_w, tensor_dim_h, tensor_dim_w);
    // }


    Element const *global_offset = global_ptr + (((offset_n * tensor_actual_dim_d + offset_d) * tensor_actual_dim_h + offset_h) * tensor_actual_dim_w + offset_w) * tensor_stride_w + offset_c;

    // write c for each merge cnt
    for (int merge_idx = 0; merge_idx < c_merge_cnt; merge_idx++) {
      int cur_offset_w = offset_w + dilation_w * merge_idx;

      bool data_valid =
        offset_d >= 0 && offset_d < tensor_actual_dim_d
        && offset_h >= 0 && offset_h < tensor_actual_dim_h
        && cur_offset_w >= 0 && cur_offset_w < tensor_actual_dim_w
        && offset_n < tensor_dim_n;

      // add offset for cur merge_cnt
      Element const *cur_rs_offset = global_offset + tensor_stride_w * dilation_w * merge_idx;
      for (int c = 0; c < valid_c_length; c++) {
        // Element *tsm_offset = tsm_ptr + get_tsm_swiz_off<Element>(slice_stride, slice_size, h, c);
        Element *tsm_offset = tsm_ptr + get_tsm_swiz_off<Element>(slice_stride, slice_size, h, c + merge_idx * valid_c_length);

        // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && h >= 0 && h < 1  && merge_idx == 3 && c == 0) {
        //   printf("h: %d, c: %d, off: %d, valid: %d, val: %f\n", h, c, get_tsm_swiz_off<Element>(slice_stride, slice_size, h, c + merge_idx * valid_c_length), data_valid, cur_rs_offset[c]);
        //   printf("off_n: %d, off_h: %d, off_w: %d, valid: %d, valid_c_length: %d, c_merge_cnt: %d\n", offset_n, offset_h, cur_offset_w, data_valid, valid_c_length, c_merge_cnt);
        // }

        if (data_valid) {
          *tsm_offset = cur_rs_offset[c];

          // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && h >= 0 && h < 1  && c_merge_cnt == 8) {
          //   printf("merge_idx: %d, c: %d, off_n: %d, off_h: %d, off_w: %d, valid: %d, value: %f\n", merge_idx, c, offset_n, offset_h, cur_offset_w, data_valid, float(*tsm_offset));
          // }
        } else {
          *tsm_offset = 0;
          // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && h >= 0 && h < 1  && c_merge_cnt == 8) {
          //   printf("merge_idx: %d, c: %d, off_n: %d, off_h: %d, off_w: %d, valid: %d, value: %f\n", merge_idx, c, offset_n, offset_h, cur_offset_w, data_valid, float(*tsm_offset));
          // }
        }
      }
    }

    // supply 0 till cube_c
    for (int c = valid_c_length * c_merge_cnt; c < cube_c; c++) {
      Element *tsm_offset = tsm_ptr + get_tsm_swiz_off<Element>(slice_stride, slice_size, h, c);
      *tsm_offset = 0;
    }

    // update offset
    offset_w += step_w;
    // move to next h
    if (offset_w > valid_end_w) {
      offset_w = valid_start_w;
      offset_h += step_h;
      if (offset_h > valid_end_h) {
        offset_h = valid_start_h;
        offset_d += step_d;
        if (offset_d > valid_end_d) {
          offset_d = valid_start_d;
          offset_n++;
        }
      }
    }

    // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && h == 0) {
    //   for (int s = 0; s < 4; s++) {
    //     for (int i = 0; i < 8; i++) {
    //       printf("%f ", tsm_ptr[s * slice_stride + i]);
    //     }
    //     printf("\nslice\n");
    //   }
    //   printf("\nstage\n");
    // }

  }
}

// only support start_y=0/cube_h=1
// always consider dtype as 32 bit, since when fp16 two elements are in one vreg, which is identical to on fp32
CUTLASS_DEVICE
void load_matrix_sync_sim(float *frag, float const*tsm_ptr, int start_x, int cube_w, int slice_id) {
  // start vector idx of the slice
  int slice_start_vec = (((slice_id & 1) << 1) + ((slice_id & 2) >> 1)) * 2;

  int lane_idx = threadIdx.x % 32;
  // each warp is arrange in 8x4 for each vreg
  int lane_row_idx = lane_idx / 4 + start_x;
  int lane_col_idx = lane_idx % 4;
  // four vreg
  CUTLASS_PRAGMA_UNROLL
  for (int v = 0; v < 4; v++) {
    // cur lane's cur vreg is on which row of original tsm layout
    // vreg 2/3 is on lower 8 rows
    int vreg_row_idx = (v / 2) * 8 + lane_row_idx;

    // cache line index, four rows in one cache line
    int vreg_line_idx = vreg_row_idx / 4;
    int vreg_vec_idx = (vreg_row_idx % 4) * 2 + (v % 2);
    // swizzle between two vector in odd cache line
    int vreg_vec_idx_swz1 = vreg_vec_idx ^ (vreg_line_idx % 2);
    // swizzle for each slice
    int vreg_vec_idx_swz2 = (vreg_vec_idx_swz1 + slice_start_vec) % 8;

    // 32 fp32 in one cache line,
    int lane_elem_off = vreg_line_idx * 32 + vreg_vec_idx_swz2 * 4 + lane_col_idx;
    frag[v] = tsm_ptr[lane_elem_off];
    // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x < 32) {
    //   printf("thread: %d, slice: %d, v: %d, vreg_row_idx: %d, vreg_line_idx: %d, vreg_vec_idx: %d, vreg_vec_idx_swz2: %d, lane_elem_off: %d\n", threadIdx.x, slice_id, v, vreg_row_idx, vreg_line_idx, vreg_vec_idx, vreg_vec_idx_swz2, lane_elem_off);
    // }
  }
}


template<typename Element>
CUTLASS_DEVICE
void load_matrix_sync_trans_sim(Element *frag, Element const*tsm_ptr, int start_x, int cube_w, int slice_id) {
  // rows per vreg
  int elem_per_vreg = 4 / sizeof(Element);
  int elem_per_vec = 16 / sizeof(Element);

  int lane_idx = threadIdx.x % 32;
  int lane_row_idx = (lane_idx % 4) * elem_per_vreg + start_x;
  // elem idx
  int lane_col_idx = lane_idx / 4;

  int lane_slice = slice_id;
  int slice_off = 0;

  // four vreg
  CUTLASS_PRAGMA_UNROLL
  for (int v = 0; v < 4; v++) {
    CUTLASS_PRAGMA_UNROLL
    for (int e = 0; e < elem_per_vreg; e++) {
      // each vreg cover 8 rows for fp16
      // lane_row_idx had multiply elem_per_vreg already
      int vreg_row_idx = (v % 2) * 4 * elem_per_vreg + lane_row_idx + e;

      int vreg_line_idx = vreg_row_idx / 4;
      // different col in one warp may in different vector when fp32
      int vreg_vec_idx = (vreg_row_idx % 4) * 2 + (lane_col_idx / elem_per_vec);
      if (elem_per_vreg == 2) {
        // for fp16, vreg 2/3 is on right vector of each slice
        vreg_vec_idx += v / 2;
        slice_off = 0;
      } else {
        // for fp32, vreg 2/3 is on next slice
        lane_slice = slice_id + v / 2;
        slice_off = v / 2;
      }


      int slice_start_vec = (((lane_slice & 1) << 1) + ((lane_slice & 2) >> 1)) * 2;
      // swizzle between two vector in odd cache line
      int vreg_vec_idx_swz1 = vreg_vec_idx ^ (vreg_line_idx % 2);
      // swizzle for each slice
      int vreg_vec_idx_swz2 = (vreg_vec_idx_swz1 + slice_start_vec) % 8;

      // should add slice offset for vreg 2/3 fp32
      // elem_offset should % elem_per_vec, when fp32, right half of the warp has already add on vector idx
      int lane_elem_off = (vreg_line_idx * 8 + vreg_vec_idx_swz2) * elem_per_vec + (lane_col_idx % elem_per_vec) + slice_off * cube_w * 2 * elem_per_vec;


      // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x < 32) {
      // if (lane_elem_off >= 32 * 16) {
      //   printf("thread: %d, v: %d, e: %d, vreg_row_idx: %d, vreg_line_idx: %d, vreg_vec_idx_swz2: %d, vreg_vec_idx: %d, lane_col_idx: %d, lane_elem_off: %d\n", threadIdx.x, v, e, vreg_row_idx, vreg_line_idx, vreg_vec_idx_swz2, vreg_vec_idx, lane_col_idx, lane_elem_off);
      //   // printf("thread: %d, slice: %d, v: %d, e: %d, vreg_row_idx: %d, vreg_line_idx: %d, vreg_vec_idx: %d, vreg_vec_idx_swz2: %d, lane_elem_off: %d\n", threadIdx.x, lane_slice, v, e, vreg_row_idx, vreg_line_idx, vreg_vec_idx, vreg_vec_idx_swz2, elem_off);
      // }

      // one row has 2 vectors in one slice
      frag[v * elem_per_vreg + e] = tsm_ptr[lane_elem_off];

    }

  }
}



// simulator tsm load without swizzle, for A100
#if 0
template<typename Element>
CUTLASS_DEVICE
void load_matrix_sync_sima(Element *frag, Element *tsm_ptr, int start_y, int start_x, int cube_h, int cube_w, int slice_id) {
  // element size of each slice
  int slice_size = 32 / sizeof(Element);
  // offset of current slice and current inst
  Element *tsm_offset = tsm_ptr + (slice_id * cube_h * cube_w + start_y * cube_w + start_x) * slice_size;

  int thread_idx = threadIdx.x % 32;
  int lane_row_off = thread_idx / 4;
  int lane_col_off = thread_idx % 4;

  // write 4 vreg
  for (int k = 0; k < 4; k++) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    int col_offset = k % 2;
    int row_offset = k / 2;
#else
    int row_offset = k % 2;
    int col_offset = k / 2;
#endif

    // int row_offset = k / 2;
    // int col_offset = k % 2;

    // for fp32 one row in one slice is 8 element, only support fp32 now
    // and add lane offset in current 8x4
    Element *cur_tsm_offset = tsm_offset + (row_offset * 8 + lane_row_off) * 8 + col_offset * 4 + lane_col_off;
    frag[k] = *cur_tsm_offset;
  }

}


// for a100 8x8 matrix b
template<typename Element>
CUTLASS_DEVICE
void load_matrix_sync_simb(Element *frag, Element *tsm_ptr, int start_y, int start_x, int cube_h, int cube_w, int slice_id) {
  // element size of each slice
  int slice_size = 32 / sizeof(Element);
  // offset of current slice and current inst
  Element *tsm_offset = tsm_ptr + (slice_id * cube_h * cube_w + start_y * cube_w + start_x) * slice_size;

  int thread_idx = threadIdx.x % 32;
  int lane_row_off = thread_idx / 4;
  int lane_col_off = thread_idx % 4;

  for (int k = 0; k < 2; k++) {
    // for fp32 one row in one slice is 8 element, only support fp32 now
    // and add lane offset in current 8x4
    Element *cur_tsm_offset = tsm_offset + lane_row_off * 8 + k * 4 + lane_col_off;
    frag[k] = *cur_tsm_offset;
  }

}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace arch
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
