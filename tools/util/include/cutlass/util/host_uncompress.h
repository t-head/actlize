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
    \brief uncompress sparse matrix from the host side 
*/
#pragma once

#include "cutlass/coord.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/tensor_view.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/gemm.h"

#include <ctype.h>
#include <stdio.h>

namespace cutlass {

template <typename Element, typename Layout, typename ElementE,
          typename LayoutE>
void uncompress(TensorRef<Element, Layout> uncompressed_tensor_a,
                TensorRef<Element, Layout> tensor_a,
                TensorRef<ElementE, LayoutE> tensor_e, int row, int col);

template <typename Element, typename ElementE,
          typename LayoutE>
void uncompress(TensorRef<Element, layout::RowMajor> uncompressed_tensor_a,
                TensorRef<Element, layout::RowMajor> tensor_a,
                TensorRef<ElementE, LayoutE> tensor_e, int row, int col) {
  // How many uncompressed data we can get with ElementE meta data
  int DecompressedElementsPerElementE =
      256 / cutlass::sizeof_bits<Element>::value;

  // Process 4bit meta data a time 
  int step;

  // 1:2 or 2:4 or 4:8
  int a, b;

  if (cutlass::sizeof_bits<Element>::value == 4) {
    step = 8;
    a = 4;
    b = 8;
  } else if (cutlass::sizeof_bits<Element>::value == 8) {
    step = 4;
    a = 2;
    b = 4;
  } else if (cutlass::sizeof_bits<Element>::value == 16) {
    step = 4;
    a = 2;
    b = 4;
  } else if (cutlass::sizeof_bits<Element>::value == 32) {
    step = 2;
    a = 1;
    b = 2;
  }

  int ElementsPerE = (cutlass::sizeof_bits<Element>::value == 4) ? 2 : 1;

  for (int r = 0; r < row; ++r) {
    for (int c = 0; c < (col / DecompressedElementsPerElementE); ++c) {

      ElementE meta = tensor_e.at(MatrixCoord(r, c));

      for (int i = 0; i < DecompressedElementsPerElementE; i += step) {
        int e = (meta >> (i / step * 4)) & 0xf;
        int idx0 = e & 0x3;
        int idx1 = e >> 2;

        if (a == 1) idx0 = idx0 / 2;

        for (int ii = 0; ii < step; ii += ElementsPerE) {
          int real_col =
              c * DecompressedElementsPerElementE + i + ii;
          int compressed_col = (real_col / b) * a;

          if (ii == (idx0 * ElementsPerE)) {
            uncompressed_tensor_a.at(MatrixCoord(r, real_col)) =
                tensor_a.at(MatrixCoord(r, compressed_col));
            if (ElementsPerE == 2)
              uncompressed_tensor_a.at(MatrixCoord(r, real_col + 1)) =
                  tensor_a.at(MatrixCoord(r, compressed_col + 1));
          } else if ((ii == (idx1 * ElementsPerE)) && (a != 1)) {
            uncompressed_tensor_a.at(MatrixCoord(r, real_col)) =
                tensor_a.at(MatrixCoord(r, compressed_col + ElementsPerE));
            if (ElementsPerE == 2)
              uncompressed_tensor_a.at(MatrixCoord(r, real_col + 1)) =
                  tensor_a.at(
                      MatrixCoord(r, compressed_col + ElementsPerE + 1));
          } else {
            uncompressed_tensor_a.at(MatrixCoord(r, real_col)) =
                Element(0);
            if (ElementsPerE == 2)
              uncompressed_tensor_a.at(MatrixCoord(r, real_col + 1)) =
                  Element(0);
          }
        }
      }
    }
  }
}

template <typename Element, typename ElementE,
          typename LayoutE>
void uncompress(TensorRef<Element, layout::ColumnMajor> uncompressed_tensor_a,
                TensorRef<Element, layout::ColumnMajor> tensor_a,
                TensorRef<ElementE, LayoutE> tensor_e, int row, int col) {
  // How many uncompressed data we can get with ElementE meta data
  int DecompressedElementsPerElementE =
      256 / cutlass::sizeof_bits<Element>::value;

  // Process 4bit meta data a time
  int step;

  // 1:2 or 2:4 or 4:8
  int a, b;

  if (cutlass::sizeof_bits<Element>::value == 4) {
    step = 8;
    a = 4;
    b = 8;
  } else if (cutlass::sizeof_bits<Element>::value == 8) {
    step = 4;
    a = 2;
    b = 4;
  } else if (cutlass::sizeof_bits<Element>::value == 16) {
    step = 4;
    a = 2;
    b = 4;
  } else if (cutlass::sizeof_bits<Element>::value == 32) {
    step = 2;
    a = 1;
    b = 2;
  }

  int ElementsPerE = (cutlass::sizeof_bits<Element>::value == 4) ? 2 : 1;
  for (int r = 0; r < col; ++r) {
    for (int c = 0; c < (row / DecompressedElementsPerElementE); ++c) {

      ElementE meta = tensor_e.at(MatrixCoord(c, r));

      for (int i = 0; i < DecompressedElementsPerElementE; i += step) {
        int e = (meta >> (i / step * 4)) & 0xf;
        int idx0 = e & 0x3;
        int idx1 = e >> 2;

        if (a == 1) idx0 = idx0 / 2;

        for (int ii = 0; ii < step; ii += ElementsPerE) {
          int real_row =
              c * DecompressedElementsPerElementE + i + ii;
          int compressed_row = (real_row / b) * a;

          if (ii == (idx0 * ElementsPerE)) {
            uncompressed_tensor_a.at(MatrixCoord(real_row, r)) =
                tensor_a.at(MatrixCoord(compressed_row, r));
            if (ElementsPerE == 2)
              uncompressed_tensor_a.at(MatrixCoord(real_row + 1, r)) =
                  tensor_a.at(MatrixCoord(real_row + 1, r));
          } else if ((ii == (idx1 * ElementsPerE)) && (a != 1)) {
            uncompressed_tensor_a.at(MatrixCoord(real_row, r)) =
                tensor_a.at(MatrixCoord(compressed_row + ElementsPerE, r));
            if (ElementsPerE == 2)
              uncompressed_tensor_a.at(MatrixCoord(real_row + 1, r)) =
                  tensor_a.at(
                      MatrixCoord(compressed_row + ElementsPerE + 1, r));
          } else {
            uncompressed_tensor_a.at(MatrixCoord(real_row, r)) =
                Element(0);
            if (ElementsPerE == 2)
              uncompressed_tensor_a.at(MatrixCoord(real_row + 1, r)) =
                  Element(0);
          }
        }
      }
    }
  }
}

template <typename Element, typename Layout, typename ElementE,
          typename LayoutE>
void uncompress_col2img(TensorRef<Element, Layout> uncompressed_tensor_a,
                TensorRef<Element, Layout> tensor_a,
                TensorRef<ElementE, LayoutE> tensor_e,
                cutlass::Tensor4DCoord input, cutlass::Tensor4DCoord output,
                cutlass::Tensor4DCoord filter, cutlass::Tensor4DCoord padding, cutlass::MatrixCoord stride);

template <typename Element, typename Layout, typename ElementE>
void uncompress_col2img(TensorRef<Element, Layout> uncompressed_tensor_a,
                TensorRef<Element, Layout> tensor_a,
                TensorRef<ElementE, cutlass::layout::RowMajor> tensor_e,
                cutlass::Tensor4DCoord input, cutlass::Tensor4DCoord output,
                cutlass::Tensor4DCoord filter, cutlass::Tensor4DCoord padding, cutlass::MatrixCoord stride) {

  int row = input.n() * output.h() * output.w();
  // int col = filter.k;
  int col = input.c() * filter.h() * filter.w();
  // How many uncompressed data we can get with ElementE meta data
  int DecompressedElementsPerElementE =
      256 / cutlass::sizeof_bits<Element>::value;

  // Process 4bit meta data a time
  int step;

  // 1:2 or 2:4 or 4:8
  int a, b;

  if (cutlass::sizeof_bits<Element>::value == 4) {
    step = 8;
    a = 4;
    b = 8;
  } else if (cutlass::sizeof_bits<Element>::value == 8) {
    step = 4;
    a = 2;
    b = 4;
  } else if (cutlass::sizeof_bits<Element>::value == 16) {
    step = 4;
    a = 2;
    b = 4;
  } else if (cutlass::sizeof_bits<Element>::value == 32) {
    step = 2;
    a = 1;
    b = 2;
  }

  int ElementsPerE = (cutlass::sizeof_bits<Element>::value == 4) ? 2 : 1;

  for (int r = 0; r < row; ++r) {
      // col2img: transform matrix row to conv[n,p,q]
      int n  = r / (output.h() * output.w());
      int npq_residual = r % (output.h() * output.w());
      int p = npq_residual / output.w();
      int q = npq_residual % output.w();

    for (int c = 0; c < (col / DecompressedElementsPerElementE); ++c) {
      ElementE meta = tensor_e.at(MatrixCoord(r, c));
      for (int i = 0; i < DecompressedElementsPerElementE; i += step) {
        int e = (meta >> (i / step * 4)) & 0xf;
        int idx0 = e & 0x3;
        int idx1 = e >> 2;
        if (a == 1) idx0 = idx0 / 2;
        for (int ii = 0; ii < step; ii += ElementsPerE) {
          int real_col =
              c * DecompressedElementsPerElementE + i + ii;

          // gemm_c layout is C->R->S
          int c = real_col % input.c();
          int crs_residual = real_col / input.c();
          int r = crs_residual / filter.w();
          int s = crs_residual % filter.w();

          int h = p * stride.column() - padding.h() + r;
          int w = q * stride.row() - padding.w() + s;

          if (h < 0 || w < 0 || w >= input.w() || h >= input.h())
            continue;

          int compressed_c = (c / b) * a;

          if (ii == (idx0 * ElementsPerE)) {
            uncompressed_tensor_a.at(Tensor4DCoord(n, h, w, c)) =
              tensor_a.at(Tensor4DCoord(n, h, w, compressed_c));
            if (ElementsPerE == 2)
              uncompressed_tensor_a.at(Tensor4DCoord(n, h, w, c + 1)) =
                tensor_a.at(Tensor4DCoord(n, h, w, compressed_c + 1));
          } else if ((ii == (idx1 * ElementsPerE)) && (a != 1)) {
            uncompressed_tensor_a.at(Tensor4DCoord(n, h, w, c)) =
              tensor_a.at(Tensor4DCoord(n, h, w, compressed_c + ElementsPerE));
            if (ElementsPerE == 2)
              uncompressed_tensor_a.at(Tensor4DCoord(n, h, w, c + 1)) =
                tensor_a.at(Tensor4DCoord(n, h, w, compressed_c + ElementsPerE + 1));
          } else {
            uncompressed_tensor_a.at(Tensor4DCoord(n, h, w, c)) = Element(0);
            if (ElementsPerE == 2)
              uncompressed_tensor_a.at(Tensor4DCoord(n, h, w, c + 1)) = Element(0);
          }
        }
      }
    }
  }
}

#ifdef SAIL_CUSTOMIZE_CUTLASS
template <typename Element, typename Layout, typename ElementE>
void uncompress_col2img(TensorRef<Element, Layout> uncompressed_tensor_a,
                TensorRef<Element, Layout> tensor_a,
                TensorRef<ElementE, cutlass::layout::ColumnMajor> tensor_e,
                cutlass::Tensor4DCoord input, cutlass::Tensor4DCoord output,
                cutlass::Tensor4DCoord filter, cutlass::Tensor4DCoord padding, cutlass::MatrixCoord stride) {
  int row = filter.c() * filter.h() * filter.w();
  int col = filter.n();

  // How many uncompressed data we can get with ElementE meta data
  int DecompressedElementsPerElementE =
      256 / cutlass::sizeof_bits<Element>::value;

  // Process 4bit meta data a time
  int step;

  // 1:2 or 2:4 or 4:8
  int a, b;

  if (cutlass::sizeof_bits<Element>::value == 4) {
    step = 8;
    a = 4;
    b = 8;
  } else if (cutlass::sizeof_bits<Element>::value == 8) {
    step = 4;
    a = 2;
    b = 4;
  } else if (cutlass::sizeof_bits<Element>::value == 16) {
    step = 4;
    a = 2;
    b = 4;
  } else if (cutlass::sizeof_bits<Element>::value == 32) {
    step = 2;
    a = 1;
    b = 2;
  }

  // block gemm_k in c group, 128 bytes per channel.
  int channel_per_load = 128 / sizeof(Element);
  int group = min(channel_per_load, filter.c());

  int ElementsPerE = (cutlass::sizeof_bits<Element>::value == 4) ? 2 : 1;

  for (int k = 0; k < col; ++k) {
    for (int r = 0; r < (row / DecompressedElementsPerElementE); ++r) {
      ElementE meta = tensor_e.at(MatrixCoord(r, k));
      for (int i = 0; i < DecompressedElementsPerElementE; i += step) {
        int e = (meta >> (i / step * 4)) & 0xf;
        int idx0 = e & 0x3;
        int idx1 = e >> 2;
        if (a == 1) idx0 = idx0 / 2;
        for (int ii = 0; ii < step; ii += ElementsPerE) {
          int real_row =
              r * DecompressedElementsPerElementE + i + ii;

          // gemm_c layout is C_GROUP->R->S
          int c_group_residual = real_row % (group * filter.w() * filter.h());
          int c_group = real_row / (group * filter.w() * filter.h());
          int c = c_group * group + c_group_residual % group;
          int crs_residual = c_group_residual / group;
          int r = crs_residual / filter.w();
          int s = crs_residual % filter.w();

          if (r < 0 || s < 0 || r >= filter.w() || s >= filter.h())
            continue;

          int compressed_c = (c / b) * a;

          if (ii == (idx0 * ElementsPerE)) {
            uncompressed_tensor_a.at(Tensor4DCoord(k, r, s, c)) =
              tensor_a.at(Tensor4DCoord(k, r, s, compressed_c));
            if (ElementsPerE == 2)
              uncompressed_tensor_a.at(Tensor4DCoord(k, r, s, c + 1)) =
                tensor_a.at(Tensor4DCoord(k, r, s, compressed_c + 1));
          } else if ((ii == (idx1 * ElementsPerE)) && (a != 1)) {
            uncompressed_tensor_a.at(Tensor4DCoord(k, r, s, c)) =
              tensor_a.at(Tensor4DCoord(k, r, s, compressed_c + ElementsPerE));
            if (ElementsPerE == 2)
              uncompressed_tensor_a.at(Tensor4DCoord(k, r, s, c + 1)) =
                tensor_a.at(Tensor4DCoord(k, r, s, compressed_c + ElementsPerE + 1));
          } else {
            uncompressed_tensor_a.at(Tensor4DCoord(k, r, s, c)) = Element(0);
            if (ElementsPerE == 2)
              uncompressed_tensor_a.at(Tensor4DCoord(k, r, s, c + 1)) = Element(0);
          }
        }
      }
    }
  }
}
#endif

} // namespace cutlass
