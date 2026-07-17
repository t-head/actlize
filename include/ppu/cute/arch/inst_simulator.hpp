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

aiu_load {
  if (threadIdx.x % 32 != 0) {
    return;
  }
  // printf("blockIdx: %d, %d, coord: %d, %d, %d, smem_ptr: %p\n", blockIdx.x, blockIdx.y, coord_n, coord_h, coord_w, smem_ptr);
  // desc.print();

  int cube_h = desc.cube_h;
  int cube_w = desc.cube_w;
  int dim_h = desc.dim_h;
  int dim_w = desc.dim_w;
  int stride_w = desc.stride_w;

  const Element *global_ptr = reinterpret_cast<const Element*>(desc.gmem_ptr);
  Element *tsm_ptr = reinterpret_cast<Element*>(smem_ptr);

  // element size of each slice
  int slice_size = 32 / sizeof(Element);
  int slice_stride = cube_h * slice_size;

  auto get_swzl_off = [=](int h, int w) {
    int slice_id = w / slice_size;
    // element offset in slice
    int slice_off = w % slice_size;

    // how many elements in one vector
    int elem_per_vec = 16 / sizeof(Element);

    // which vector in slice
    int vector_idx = slice_off / elem_per_vec;
    // elem offset in vector
    int vector_off = slice_off % elem_per_vec;

    // which cache line in the slice, always 4 row one cache line
    int line_id = h / 4;
    // slice offset in cur cache line
    int line_off = h % 4;
    int slice_in_line = (((slice_id & 1) << 1) + ((slice_id & 2) >> 1) + line_off) % 4;

    // swizzle between two vector in odd cache line
    int vector_idx_swiz = vector_idx ^ (line_id % 2);

    // elements in one cache line
    int elem_per_line = 128 / sizeof(Element);

    // one line is 32 fp32, one slice is 8 fp32
    return slice_id * slice_stride + line_id * elem_per_line + slice_in_line * slice_size + vector_idx_swiz * elem_per_vec + vector_off;
  };

  CUTE_NO_UNROLL
  for (int h = 0; h < cube_h; ++h) {
    CUTE_NO_UNROLL
    for (int w = 0; w < cube_w; ++w) {
      int cur_h = coord_h + h;
      int cur_w = coord_w + w;
      Element const *global_offset = global_ptr + cur_h * stride_w + cur_w;
      Element *tsm_offset = tsm_ptr + get_swzl_off(h, w);
      if (cur_h < dim_h && cur_w < dim_w) {
        *tsm_offset = *global_offset;
      } else {
        *tsm_offset = 0;
      }
    }
  }
}


template<typename Element, int CUBE_H, int CUBE_W, bool SWAP>
CUTE_HOST_DEVICE static void
ppu_tsm_ld_swzl_sim(void *frag_ptr, void *smem_base, int coord_w, int coord_h, int stage) {
  Element *stage_base = reinterpret_cast<Element*>(smem_base);
  stage_base += CUBE_H * CUBE_W * stage;

  int slice_idx = coord_w / (32 / sizeof(Element));

  // no trans can always copy 32b each time
  uint32_t *slice_base = reinterpret_cast<uint32_t*>(stage_base);
  slice_base += CUBE_H * 8 * slice_idx;
  uint32_t *frag = reinterpret_cast<uint32_t*>(frag_ptr);

  int slice_start_vec = (((slice_idx & 1) << 1) + ((slice_idx & 2) >> 1)) * 2;

  int lane_idx = threadIdx.x % 32;
  // each warp is arrange in 8x4 for each vreg
  int lane_row_idx = lane_idx / 4 + coord_h;
  int lane_col_idx = lane_idx % 4;
  // four vreg
  CUTE_NO_UNROLL
  for (int v = 0; v < 4; v++) {
    // cur lane's cur vreg is on which row of original tsm layout
    int vreg_row_idx;
    if (!SWAP) {
      // vreg 1/3 is on lower 8 rows
      vreg_row_idx = (v % 2) * 8 + lane_row_idx;
    } else {
      // vreg 2/3 is on lower 8 rows
      vreg_row_idx = (v / 2) * 8 + lane_row_idx;
    }

    // cache line index, four rows in one cache line
    int vreg_line_idx = vreg_row_idx / 4;
    int vreg_vec_idx;
    if (!SWAP) {
      // vreg 2/3 on right of two vectors
      vreg_vec_idx = (vreg_row_idx % 4) * 2 + (v / 2);
    } else {
      // vreg 1/3 on right of two vectors
      vreg_vec_idx = (vreg_row_idx % 4) * 2 + (v % 2);
    }
    // swizzle between two vector in odd cache line
    int vreg_vec_idx_swz1 = vreg_vec_idx ^ (vreg_line_idx % 2);
    // swizzle for each slice
    int vreg_vec_idx_swz2 = (vreg_vec_idx_swz1 + slice_start_vec) % 8;

    // int32 offset of cur reg
    int lane_32b_off = vreg_line_idx * 32 + vreg_vec_idx_swz2 * 4 + lane_col_idx;
    frag[v] = slice_base[lane_32b_off];
  }

}
