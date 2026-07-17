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

/*!
  \file
  \brief Extracts the host-params objects into non-template code.
*/

#pragma once

#define TRACE_CONV_PARAMS_INITIALIZERS_ENABLED 0

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"

#if TRACE_CONV_PARAMS_INITIALIZERS_ENABLED
#include <fstream>
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Params structure used for all Conv2d analytic tile iterators
template< typename Layout_ = layout::TensorNHWC >
struct Conv2dAnalyticParams {

  using Layout = Layout_;

  Layout layout;

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  Conv2dAnalyticParams() { }

  CUTLASS_HOST_DEVICE
  Conv2dAnalyticParams(
    Conv2dProblemSize const &,  // unused; placeholder to match other Params interfaces.
    Layout const &layout
  ): layout(layout) {

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

#if TRACE_CONV_PARAMS_INITIALIZERS_ENABLED

CUTLASS_HOST_DEVICE
void TraceIteratorParams(
  char const *conv_operator,
  char const *operand,
  int element_size_bits,
  MatrixCoord threadblock_shape,
  int thread_count,
  int access_size,
  layout::PitchLinearCoord threadmap_iterations,
  layout::PitchLinearCoord threadmap_delta
) {

#if !defined(__HGGC_ARCH__)

  char const *fname = "conv_iterator_params.csv";

  std::ifstream test(fname);
  bool file_exists = test.is_open();

  if (file_exists) {
    test.close();
  }

  std::ofstream trace("conv_iterator_params.csv", std::ofstream::app);

  if (!file_exists) {
    trace
      << "Operator,Operand,ElementSize,CtaRows,CtaColumns,ThreadCount,AccessSize,"
      << "IterationsContiguous,IterationsStrided,DeltaContiguous,DeltaStrided\n";
  }

  trace << conv_operator << "," << operand << "," << element_size_bits << ","
    << threadblock_shape.row() << "," << threadblock_shape.column()
    << "," << thread_count << "," << access_size
    << "," << threadmap_iterations.contiguous() << "," << threadmap_iterations.strided()
    << "," << threadmap_delta.contiguous() << "," << threadmap_delta.strided() << "\n";
#endif
}

#define TRACE_CONV_INITIALIZERS(conv_op, operand, element_size, cta_shape, thread_count, access_size, iterations, delta) \
  TraceIteratorParams(conv_op, operand, element_size, cta_shape, thread_count, access_size, iterations, delta);

#else

#define TRACE_CONV_INITIALIZERS(conv_op, operand, element_size, cta_shape, thread_count, access_size, iterations, delta) {}

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Parameters structure used for Conv2dFpropActivationTileIteratorOptimized
template< typename Layout_ = layout::TensorNHWC >
struct Conv2dFpropActivationIteratorOptimizedParams;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Parameters structure used for Conv2dFpropActivationTileIteratorOptimized
template<>
struct Conv2dFpropActivationIteratorOptimizedParams<layout::TensorNHWC> {

  using Layout = layout::TensorNHWC;

  Layout layout;

  int64_t inc_next[3];    // {next S, next R, next C}
  int filter_c_delta;     // number of logical elements to add to filter_c_
  int PQ;                 // product of P*Q

  FastDivmod pq_divmod;
  FastDivmod q_divmod;

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  Conv2dFpropActivationIteratorOptimizedParams() { }

  CUTLASS_HOST_DEVICE
  Conv2dFpropActivationIteratorOptimizedParams(
    Conv2dProblemSize const &problem_size,
    Layout const &layout,                             ///< layout object
    int element_size_bits,                            ///< size of each element in bits
    MatrixCoord threadblock_shape,
    int thread_count,
    int access_size,
    layout::PitchLinearCoord threadmap_iterations,
    layout::PitchLinearCoord threadmap_delta
  ):
    layout(layout),
    PQ(problem_size.P * problem_size.Q),
    pq_divmod(PQ),
    q_divmod(problem_size.Q) {

    TRACE_CONV_INITIALIZERS("conv2d_fprop", "activation",
      element_size_bits, threadblock_shape, thread_count, access_size, threadmap_iterations, threadmap_delta);

    param_init(problem_size, layout, element_size_bits, threadblock_shape);
  }

  // for rtc usage
  CUTLASS_HOST_DEVICE
  Conv2dFpropActivationIteratorOptimizedParams(
    Conv2dProblemSize const &problem_size,
    Layout const &layout,                             ///< layout object
    int element_size_bits,                            ///< size of each element in bits
    MatrixCoord threadblock_shape
  ):
    layout(layout),
    PQ(problem_size.P * problem_size.Q),
    pq_divmod(PQ),
    q_divmod(problem_size.Q) {

    param_init(problem_size, layout, element_size_bits, threadblock_shape);
  }

  CUTLASS_HOST_DEVICE
  void param_init(
    Conv2dProblemSize const &problem_size,
    Layout const &layout,
    int element_size_bits,
    MatrixCoord threadblock_shape
  ) {

    int conv_sign = (problem_size.mode == Mode::kConvolution ? -1 : 1);

    int fold_num = problem_size.fold_num;

    // next S
    // multiple fold_num to advance multiple s once
    inc_next[0] = fold_num * conv_sign * (
      int64_t(layout.stride()[0]) * problem_size.dilation_w
    ) * element_size_bits / 8;

    // next R
    inc_next[1] = conv_sign * (
        // advance fold_num / problem_size.S rows when fold_num > S, and * dilation
        int64_t(layout.stride()[1]) * problem_size.dilation_h * max(1, (fold_num / problem_size.S))
        // advance (problem_size.s / fold_num - 1) times on s, each fold_num * stride[0]
        // when fold_num >= S, advance 0 times on s
        - max(0, (problem_size.S / fold_num - 1)) * fold_num * layout.stride()[0] * problem_size.dilation_w
      ) * element_size_bits / 8;

    // next C
    inc_next[2] = (
        threadblock_shape.column() * problem_size.split_k_slices
        // advance max(1, (fold_num / problem_size.S)) each times on R, R advance problem_size.R / max(1, (fold_num / problem_size.S)) - 1 times
        // fold_num / problem_size.S should not bigger than R
        - conv_sign * int64_t(problem_size.R / max(1, (fold_num / problem_size.S)) - 1) * max(1, (fold_num / problem_size.S)) * layout.stride()[1] * problem_size.dilation_h
        - conv_sign * int64_t(max(0, problem_size.S / fold_num - 1)) * fold_num * layout.stride()[0] * problem_size.dilation_w
      ) * element_size_bits / 8;

    // logical offset added to internal channel counter - units are elements, not bytes
    filter_c_delta = threadblock_shape.column() * problem_size.split_k_slices;

  }

  CUTLASS_HOST_DEVICE
  void print() const {
    layout.print();

    printf("inc_next[0]: %ld, inc_next[1]: %ld, inc_next[2]: %ld, filter_c_delta: %d, PQ: %d\n", inc_next[0], inc_next[1], inc_next[2], filter_c_delta, PQ);
  }
};

/// Parameters structure used for Conv2dFpropActivationTileIteratorOptimized
template <int Interleaved_>
struct Conv2dFpropActivationIteratorOptimizedParams<layout::TensorNCxHWx<Interleaved_>> {
  static int const kInterleaved = Interleaved_;

  using Layout = layout::TensorNCxHWx<kInterleaved>;

  Layout layout;

  int64_t inc_next[3];    // {next S, next R, next C}
  int filter_c_delta;     // number of logical elements to add to filter_c_
  int PQ;                 // product of P*Q

  FastDivmod pq_divmod;
  FastDivmod q_divmod;

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  Conv2dFpropActivationIteratorOptimizedParams() { }

  CUTLASS_HOST_DEVICE
  Conv2dFpropActivationIteratorOptimizedParams(
    Conv2dProblemSize const &problem_size,
    Layout const &layout,                             ///< layout object
    int element_size_bits,                            ///< size of each element in bits
    MatrixCoord threadblock_shape,
    int thread_count,
    int access_size,
    layout::PitchLinearCoord threadmap_iterations,
    layout::PitchLinearCoord threadmap_delta
  ):
    layout(layout), PQ(problem_size.P * problem_size.Q), pq_divmod(PQ), q_divmod(problem_size.Q) {

    TRACE_CONV_INITIALIZERS("conv2d_fprop", "activation",
      element_size_bits, threadblock_shape, thread_count, access_size, threadmap_iterations, threadmap_delta);

    int conv_sign = (problem_size.mode == Mode::kConvolution ? -1 : 1);

    // next S
    inc_next[0] = conv_sign * (kInterleaved * problem_size.dilation_w) * element_size_bits / 8;

    // next R
    inc_next[1] = conv_sign * (
        int64_t(layout.stride()[0]) * problem_size.dilation_h
        - (problem_size.S - 1) * kInterleaved * problem_size.dilation_w
      ) * element_size_bits / 8;

    // next C
    inc_next[2] = (
        threadblock_shape.column() * problem_size.split_k_slices / kInterleaved * int64_t(layout.stride()[1])
        - conv_sign * int64_t(problem_size.R - 1) * layout.stride()[0] * problem_size.dilation_h
        - conv_sign * int64_t(problem_size.S - 1) * kInterleaved * problem_size.dilation_w
      ) * element_size_bits / 8;

    // logical offset added to internal channel counter - units are elements, not bytes
    filter_c_delta = threadblock_shape.column() * problem_size.split_k_slices;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout_ = layout::TensorNHWC >
struct Conv2dFpropFilterIteratorOptimizedParams;

/////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Conv2dFpropFilterIteratorOptimizedParams<layout::TensorNHWC>
{

  using Layout = layout::TensorNHWC;

  Layout layout;
  int RS;
  int filter_c_delta;

  int64_t inc_next_k;         // offset in units of bytes to next K position
  int64_t inc_next_rs;        // offset in units of bytes to next RS position
  int64_t inc_next_c;         // offset in units of bytes to next C position

  //
  // Methods
  //
  CUTLASS_HOST_DEVICE
  Conv2dFpropFilterIteratorOptimizedParams() { }

  CUTLASS_HOST_DEVICE
  Conv2dFpropFilterIteratorOptimizedParams(
    Conv2dProblemSize const &problem_size,
    Layout const &layout,
    int element_size_bits,                        ///< size of each element in bits
    MatrixCoord threadblock_shape,
    int thread_count,
    int access_size,
    layout::PitchLinearCoord threadmap_iterations,
    layout::PitchLinearCoord threadmap_delta
  ):
    layout(layout) {

    TRACE_CONV_INITIALIZERS("conv2d_fprop", "filter",
      element_size_bits, threadblock_shape, thread_count, access_size, threadmap_iterations, threadmap_delta);

    RS = problem_size.R * problem_size.S;

    int fold_num = problem_size.fold_num;

    inc_next_k = (int64_t(layout.stride()[2]) * threadmap_delta.strided() * element_size_bits) / 8;

    // rs advance fold_num position each time
    inc_next_rs =
      ( int64_t(layout.stride()[0]) * fold_num
        - int64_t(layout.stride()[2]) * (threadmap_iterations.strided() - 1) * threadmap_delta.strided()
      ) * element_size_bits / 8;

    inc_next_c =
      (
        threadblock_shape.row() * problem_size.split_k_slices
        - int64_t(RS - 1) * layout.stride()[0]
        - int64_t(threadmap_iterations.strided() - 1) * threadmap_delta.strided() * layout.stride()[2]
      ) * element_size_bits / 8;

    filter_c_delta = threadblock_shape.row() * problem_size.split_k_slices;
  }

  CUTLASS_HOST_DEVICE
  void print() const {
    layout.print();
    printf("RS: %d, filter_c_delta: %d, inc_next_k: %ld, inc_next_rs: %ld, inc_next_c: %ld\n", RS, filter_c_delta, inc_next_k, inc_next_rs, inc_next_c);
  }
};

template<int Interleaved_>
struct Conv2dFpropFilterIteratorOptimizedParams<layout::TensorCxRSKx<Interleaved_>>
{
  static int const kInterleaved = Interleaved_;
  using Layout = layout::TensorCxRSKx<kInterleaved>;

  Layout layout;
  int RS;
  int filter_c_delta;

  int64_t inc_next_k;         // offset in units of bytes to next K position
  int64_t inc_next_rs;        // offset in units of bytes to next RS position
  int64_t inc_next_c;         // offset in units of bytes to next C position

  //
  // Methods
  //
  CUTLASS_HOST_DEVICE
  Conv2dFpropFilterIteratorOptimizedParams() { }

  CUTLASS_HOST_DEVICE
  Conv2dFpropFilterIteratorOptimizedParams(
    Conv2dProblemSize const &problem_size,
    Layout const &layout,
    int element_size_bits,                        ///< size of each element in bits
    MatrixCoord threadblock_shape,
    int thread_count,
    int access_size,
    layout::PitchLinearCoord threadmap_iterations,
    layout::PitchLinearCoord threadmap_delta
  ):
    layout(layout) {

    TRACE_CONV_INITIALIZERS("conv2d_fprop", "filter",
      element_size_bits, threadblock_shape, thread_count, access_size, threadmap_iterations, threadmap_delta);

    RS = problem_size.R * problem_size.S;

    inc_next_k = (kInterleaved * threadmap_delta.strided() * element_size_bits) / 8;

    inc_next_rs =
      (  int64_t(layout.stride()[0])
        - kInterleaved * (threadmap_iterations.strided() - 1) * threadmap_delta.strided()
      ) * element_size_bits / 8;

    inc_next_c =
      (
        threadblock_shape.row() * problem_size.split_k_slices / kInterleaved * int64_t(layout.stride()[2])
        - int64_t(RS - 1) * layout.stride()[0]
        - int64_t(threadmap_iterations.strided() - 1) * threadmap_delta.strided() * kInterleaved
      ) * element_size_bits / 8;

    filter_c_delta = threadblock_shape.row() * problem_size.split_k_slices;
  }
};

/// Parameters object for Conv2d DGRAD OutputGradient (dy) iterator
struct Conv2dDgradOutputGradientIteratorOptimizedParams {

  using Layout = layout::TensorNHWC;

  Layout layout;

  int64_t inc_next[3];    // {next S, next R, next K}

  int filter_k_delta;     // number of logical elements to add to filter_k_

  int HW;                  // product of H*W

  int step_h = 1;
  int step_w = 1;

  FastDivmod hw_divmod;
  FastDivmod w_divmod;

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  void print() const {
    layout.print();

    printf("inc_next[0]: %ld, inc_next[1]: %ld, inc_next[2]: %ld, filter_c_delta: %d, HW: %d, step_h: %d, step_w: %d\n", inc_next[0], inc_next[1], inc_next[2], filter_k_delta, HW, step_h, step_w);
  }

  CUTLASS_HOST_DEVICE
  Conv2dDgradOutputGradientIteratorOptimizedParams() { }

  CUTLASS_HOST_DEVICE
  Conv2dDgradOutputGradientIteratorOptimizedParams(
    Conv2dProblemSize const &problem_size,
    Layout const &layout,
    int element_size_bits,                        ///< size of each element in bits
    MatrixCoord threadblock_shape,
    int thread_count,
    int access_size,
    layout::PitchLinearCoord threadmap_iterations,
    layout::PitchLinearCoord threadmap_delta
  ):
    layout(layout),
    HW(problem_size.H *problem_size.W),
    hw_divmod(HW),
    w_divmod(problem_size.W) {

    TRACE_CONV_INITIALIZERS("conv2d_dgrad", "output_gradient",
      element_size_bits, threadblock_shape, thread_count, access_size, threadmap_iterations, threadmap_delta);

    param_init(problem_size, layout, element_size_bits, threadblock_shape);

  }

  // for rtc usage
  CUTLASS_HOST_DEVICE
  Conv2dDgradOutputGradientIteratorOptimizedParams(
    Conv2dProblemSize const &problem_size,
    Layout const &layout,
    int element_size_bits,
    MatrixCoord threadblock_shape
  ):
    layout(layout),
    HW(problem_size.H *problem_size.W),
    hw_divmod(HW),
    w_divmod(problem_size.W) {

    param_init(problem_size, layout, element_size_bits, threadblock_shape);
  }

  CUTLASS_HOST_DEVICE
  void param_init (
    Conv2dProblemSize const &problem_size,
    Layout const &layout,
    int element_size_bits,
    MatrixCoord threadblock_shape
  ) {
    int conv_sign = (problem_size.mode == Mode::kConvolution ? 1 : -1);

    int fold_r = (problem_size.fold_num != 1) ? problem_size.fold_h : 1;
    int fold_s = (problem_size.fold_num != 1) ? problem_size.fold_w : 1;

    // original cutlass's optimize iterator only support stride=1, one dx's different s corresponds to dy with dilation_w distance
    step_h = problem_size.dilation_h;
    step_w = problem_size.dilation_w;
    // when consider stride it's different, normal stride=2 support has consider this in iterator
    if (problem_size.fold_num != 1) {
      if (problem_size.stride_h >= problem_size.dilation_h && (problem_size.stride_h % problem_size.dilation_h) == 0) {
        step_h = 1;
      } else if (problem_size.stride_h < problem_size.dilation_h && (problem_size.dilation_h % problem_size.stride_h) == 0) {
        step_h = problem_size.dilation_h / problem_size.stride_h;
      }

      if (problem_size.stride_w >= problem_size.dilation_w && (problem_size.stride_w % problem_size.dilation_w) == 0) {
        step_w = 1;
      } else if (problem_size.stride_w < problem_size.dilation_w && (problem_size.dilation_w % problem_size.stride_w) == 0) {
        step_w = problem_size.dilation_w / problem_size.stride_w;
      }
    }

    // to get hw position in current stride part, M is folded by stride not fold
    if (problem_size.fold_num != 1) {
      int folded_h = (problem_size.H + problem_size.stride_h - 1) / problem_size.stride_h;
      int folded_w = (problem_size.W + problem_size.stride_w - 1) / problem_size.stride_w;
      hw_divmod = FastDivmod(folded_h * folded_w);
      w_divmod = FastDivmod(folded_w);
    }

    // next S
    // needn't *stride since small channel always advance stride each time, which corresponds to adjacent dy
    inc_next[0] = conv_sign * (
      layout.stride()[0] * step_w
    ) * element_size_bits / 8;

    // next R
    inc_next[1] = conv_sign * (
        layout.stride()[1] * step_h
        // advance times should divide fold_s, move fold_s steps each time
        - (problem_size.S - 1) / fold_s * layout.stride()[0] * step_w
      ) * element_size_bits / 8;

    // next K
    inc_next[2] = (
        threadblock_shape.column() * problem_size.split_k_slices
        - conv_sign * (problem_size.R - 1) / fold_r * layout.stride()[1] * step_h
        - conv_sign * (problem_size.S - 1) / fold_s * layout.stride()[0] * step_w
      ) * element_size_bits / 8;

    // logical offset added to internal channel counter - units are elements, not bytes
    filter_k_delta = threadblock_shape.column() * problem_size.split_k_slices;
  }

};

#if SAIL_DGRAD_STRIDE_OPT
struct Conv2dDgradOutputGradientIteratorOptimizedParamsStrided {

  using Layout = layout::TensorNHWC;

  Layout layout;

  int64_t inc_next[3][4];    // {next S, next R, next K}

  int filter_k_delta;     // number of logical elements to add to filter_k_

  int HW;                   // product of H*W

  FastDivmod hw_divmod;
  FastDivmod w_divmod;

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  void print() const {
    layout.print();

    printf("inc_next[0]: %ld, %ld, %ld, %ld\n", inc_next[0][0], inc_next[0][1], inc_next[0][2], inc_next[0][3]);
    printf("inc_next[1]: %ld, %ld, %ld, %ld\n", inc_next[1][0], inc_next[1][1], inc_next[1][2], inc_next[1][3]);
    printf("inc_next[2]: %ld, %ld, %ld, %ld\n", inc_next[2][0], inc_next[2][1], inc_next[2][2], inc_next[2][3]);
    printf("filter_c_delta: %d, HW: %d\n", filter_k_delta, HW);
  }

  CUTLASS_HOST_DEVICE
  Conv2dDgradOutputGradientIteratorOptimizedParamsStrided() { }

  CUTLASS_HOST_DEVICE
  Conv2dDgradOutputGradientIteratorOptimizedParamsStrided(
    Conv2dProblemSize const &problem_size,
    Layout const &layout,
    int element_size_bits,                        ///< size of each element in bits
    MatrixCoord threadblock_shape,
    int thread_count,
    int access_size,
    layout::PitchLinearCoord threadmap_iterations,
    layout::PitchLinearCoord threadmap_delta
  ):
    layout(layout), HW(problem_size.H *problem_size.W), hw_divmod(HW), w_divmod(problem_size.W) {

    TRACE_CONV_INITIALIZERS("conv2d_dgrad", "output_gradient",
      element_size_bits, threadblock_shape, thread_count, access_size, threadmap_iterations, threadmap_delta);

    param_init(problem_size, layout, element_size_bits, threadblock_shape);
  }

  // for rtc usage
  CUTLASS_HOST_DEVICE
  Conv2dDgradOutputGradientIteratorOptimizedParamsStrided(
    Conv2dProblemSize const &problem_size,
    Layout const &layout,
    int element_size_bits,                        ///< size of each element in bits
    MatrixCoord threadblock_shape
  ):
    layout(layout), HW(problem_size.H *problem_size.W), hw_divmod(HW), w_divmod(problem_size.W) {

    param_init(problem_size, layout, element_size_bits, threadblock_shape);
  }

  CUTLASS_HOST_DEVICE
  void param_init(
    Conv2dProblemSize const &problem_size,
    Layout const &layout,
    int element_size_bits,                        ///< size of each element in bits
    MatrixCoord threadblock_shape
  ) {

    int conv_sign = (problem_size.mode == Mode::kConvolution ? 1 : -1);


    for (int rs_idx = 0; rs_idx < 4; rs_idx++) {
      inc_next[0][rs_idx] = conv_sign * (layout.stride()[0] * problem_size.dilation_w) * element_size_bits / 8;
      // next R, s move back one position since stride is 2
      inc_next[1][rs_idx] = conv_sign * (
          layout.stride()[1] * problem_size.dilation_h
          - (problem_size.loop_s[rs_idx] - 1) * layout.stride()[0] * problem_size.dilation_w
        ) * element_size_bits / 8;
      // next K
      inc_next[2][rs_idx] = (
          threadblock_shape.column() * problem_size.split_k_slices
          - conv_sign * (problem_size.loop_r[rs_idx] - 1) * layout.stride()[1] * problem_size.dilation_h
          - conv_sign * (problem_size.loop_s[rs_idx] - 1) * layout.stride()[0] * problem_size.dilation_w
        ) * element_size_bits / 8;
    }

    // logical offset added to internal channel counter - units are elements, not bytes
    filter_k_delta = threadblock_shape.column() * problem_size.split_k_slices;

  }
};

#endif

/// Parameters object for Conv2d DGRAD Filter (w) iterator
struct Conv2dDgradFilterIteratorOptimizedParams {

  using Layout = layout::TensorNHWC;

  Layout layout;
  int RS;
  int filter_k_delta;

  int64_t inc_next_strided;   // offset in units of bytes to next K coordinate within tile
  int64_t inc_next_rs;        // offset in units of bytes to next RS position
  int64_t inc_next_k;         // offset in units of bytes to next K position in subsequent tile

  CUTLASS_HOST_DEVICE
  void print() const {
    layout.print();

    printf("inc_next_strided: %ld, inc_next_rs: %ld, inc_next_k: %ld, RS: %d, filter_k_delta: %d\n", inc_next_strided, inc_next_rs, inc_next_k, RS, filter_k_delta);
  }

  //
  // Methods
  //
  CUTLASS_HOST_DEVICE
  Conv2dDgradFilterIteratorOptimizedParams() { }

  CUTLASS_HOST_DEVICE
  Conv2dDgradFilterIteratorOptimizedParams(
    Conv2dProblemSize const &problem_size,
    Layout const &layout,
    int element_size_bits,                        ///< size of each element in bits
    MatrixCoord threadblock_shape,
    int thread_count,
    int access_size,
    layout::PitchLinearCoord threadmap_iterations,
    layout::PitchLinearCoord threadmap_delta
  ):
    layout(layout), RS(problem_size.R * problem_size.S) {

    TRACE_CONV_INITIALIZERS("conv2d_dgrad", "filter",
      element_size_bits, threadblock_shape, thread_count, access_size, threadmap_iterations, threadmap_delta);

    int fold_r = (problem_size.fold_num != 1) ? problem_size.fold_h : 1;
    int fold_s = (problem_size.fold_num != 1) ? problem_size.fold_w : 1;

    inc_next_strided = (layout.stride()[2] * threadmap_delta.strided() * element_size_bits) / 8;

    // next_rs position should jump stride to get next valid r/s
    inc_next_rs =
      ( layout.stride()[0] * fold_r * fold_s
        - (threadmap_iterations.strided() - 1) * threadmap_delta.strided() * layout.stride()[2]
      ) * element_size_bits / 8;

    inc_next_k =
      (
        threadblock_shape.row() * problem_size.split_k_slices * layout.stride()[2]
        // advance c*fold_r*fold_s elements each time, advance nums should / fold_rs
        - (problem_size.R / fold_r * problem_size.S / fold_s - 1) * layout.stride()[0] * fold_r * fold_s
        - (threadmap_iterations.strided() - 1) * threadmap_delta.strided() * layout.stride()[2]
      ) * element_size_bits / 8;

    filter_k_delta = threadblock_shape.row() * problem_size.split_k_slices;
  }
};

#if SAIL_DGRAD_STRIDE_OPT
struct Conv2dDgradFilterIteratorOptimizedParamsStrided {

  using Layout = layout::TensorNHWC;

  Layout layout;
  int RS;
  int filter_k_delta;

  int64_t inc_next_strided;   // offset in units of bytes to next K coordinate within tile
  int64_t inc_next_s_[4];        // offset in units of bytes to next S position
  int64_t inc_next_r_[4];        // offset in units of bytes to next R position
  int64_t inc_next_k_[4];         // offset in units of bytes to next K position in subsequent tile
  CUTLASS_HOST_DEVICE
  void print() const {
    layout.print();

    printf("inc_next_s_: %ld, %ld, %ld, %ld\n", inc_next_s_[0], inc_next_s_[1], inc_next_s_[2], inc_next_s_[3]);
    printf("inc_next_r_: %ld, %ld, %ld, %ld\n", inc_next_r_[0], inc_next_r_[1], inc_next_r_[2], inc_next_r_[3]);
    printf("inc_next_k_: %ld, %ld, %ld, %ld\n", inc_next_k_[0], inc_next_k_[1], inc_next_k_[2], inc_next_k_[3]);
    printf("filter_c_delta: %d, RS: %d\n", filter_k_delta, RS);
  }

  //
  // Methods
  //
  CUTLASS_HOST_DEVICE
  Conv2dDgradFilterIteratorOptimizedParamsStrided() { }

  CUTLASS_HOST_DEVICE
  Conv2dDgradFilterIteratorOptimizedParamsStrided(
    Conv2dProblemSize const &problem_size,
    Layout const &layout,
    int element_size_bits,                        ///< size of each element in bits
    MatrixCoord threadblock_shape,
    int thread_count,
    int access_size,
    layout::PitchLinearCoord threadmap_iterations,
    layout::PitchLinearCoord threadmap_delta
  ):
    layout(layout), RS(problem_size.R * problem_size.S) {

    TRACE_CONV_INITIALIZERS("conv2d_dgrad", "filter",
      element_size_bits, threadblock_shape, thread_count, access_size, threadmap_iterations, threadmap_delta);

    inc_next_strided = (layout.stride()[2] * threadmap_delta.strided() * element_size_bits) / 8;

    for (int rs_idx = 0; rs_idx < 4; rs_idx++) {
      // s add stride, k move back for loops in cur load
      inc_next_s_[rs_idx] =
        ( layout.stride()[0] * problem_size.stride_w
          - (threadmap_iterations.strided() - 1) * threadmap_delta.strided() * layout.stride()[2]
        ) * element_size_bits / 8;

      // r add stride, s move back (loop-1) * stride k move back for loops in cur load
      inc_next_r_[rs_idx] =
        ( layout.stride()[1] * problem_size.stride_h
          - (problem_size.loop_s[rs_idx] - 1) * layout.stride()[0] * problem_size.stride_w
          - (threadmap_iterations.strided() - 1) * threadmap_delta.strided() * layout.stride()[2]
        ) * element_size_bits / 8;

      // k add 32, rs move back (loop-1) * stride, k move back for loops in cur load
      inc_next_k_[rs_idx] =
        (
          threadblock_shape.row() * problem_size.split_k_slices * layout.stride()[2]
          - (problem_size.loop_s[rs_idx] - 1) * layout.stride()[0] * problem_size.stride_w
          - (problem_size.loop_r[rs_idx] - 1) * layout.stride()[1] * problem_size.stride_h
          - (threadmap_iterations.strided() - 1) * threadmap_delta.strided() * layout.stride()[2]
        ) * element_size_bits / 8;
    }

    filter_k_delta = threadblock_shape.row() * problem_size.split_k_slices;
  }
};
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

// host part of Conv2dWgradOutputGradientIteratorOptimizedParams in rtc
struct Conv2dWgradOutputGradientIteratorOptimizedRtcParams {

  using Layout = layout::TensorNHWC;

  Layout layout;

  int NPQ;                      // precomputd product of N*P*Q for clearing predicates
  int PQ;

  FastDivmod pq_divmod;
  FastDivmod q_divmod;

  int split_k_slices;

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  Conv2dWgradOutputGradientIteratorOptimizedRtcParams() { }

  CUTLASS_HOST_DEVICE
  Conv2dWgradOutputGradientIteratorOptimizedRtcParams(
    Conv2dProblemSize const &problem_size,
    Layout const &layout
  ):
    layout(layout),
    NPQ(problem_size.N * problem_size.P * problem_size.Q),
    PQ(problem_size.P * problem_size.Q),
    pq_divmod(problem_size.P * problem_size.Q),
    q_divmod(problem_size.Q),
    split_k_slices(problem_size.split_k_slices) {
  }
};

/// Parameters object for Conv2d WGRAD Output Gradient (dy) iterator
struct Conv2dWgradOutputGradientIteratorOptimizedParams {

  using Layout = layout::TensorNHWC;

  Layout layout;

  int NPQ;                      // precomputd product of N*P*Q for clearing predicates
  int PQ;

  FastDivmod pq_divmod;
  FastDivmod q_divmod;

  int64_t offset_next_strided;    // offset in units of bytes to next npq coordinate within tile
  int64_t offset_next_contiguous; // offset in units of bytes to next k coordinate within tile
  int64_t inc_next_npq;           // offset in units of bytes to next npq position in subsequent tile

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  Conv2dWgradOutputGradientIteratorOptimizedParams() { }

  CUTLASS_HOST_DEVICE
  Conv2dWgradOutputGradientIteratorOptimizedParams(
    Conv2dProblemSize const &problem_size,
    Layout const &layout,
    int element_size_bits,                        ///< size of each element in bits
    MatrixCoord threadblock_shape,
    int thread_count,
    int access_size,
    layout::PitchLinearCoord threadmap_iterations,
    layout::PitchLinearCoord threadmap_delta
  ):
    layout(layout),
    NPQ(problem_size.N * problem_size.P * problem_size.Q),
    PQ(problem_size.P * problem_size.Q),
    pq_divmod(problem_size.P * problem_size.Q),
    q_divmod(problem_size.Q) {

    TRACE_CONV_INITIALIZERS("conv2d_wgrad", "output_gradient",
      element_size_bits, threadblock_shape, thread_count, access_size, threadmap_iterations, threadmap_delta);

    // Incremental offsets in unites of bytes (number of elements) * sizeof_bits<Element>::value / 8
    offset_next_strided = (threadmap_delta.strided() * layout.stride()[0])
                        * element_size_bits / 8;

    offset_next_contiguous = (threadmap_delta.contiguous())
                            * element_size_bits / 8;

    inc_next_npq = (int64_t(threadblock_shape.column()) * problem_size.split_k_slices * layout.stride()[0])
                      * element_size_bits / 8;
  }

  // for rtc usage
  CUTLASS_HOST_DEVICE
  Conv2dWgradOutputGradientIteratorOptimizedParams(
    Conv2dWgradOutputGradientIteratorOptimizedRtcParams rtc_params,
    int element_size_bits,                        ///< size of each element in bits
    MatrixCoord threadblock_shape,
    layout::PitchLinearCoord threadmap_delta
  ):
    layout(rtc_params.layout),
    NPQ(rtc_params.NPQ),
    PQ(rtc_params.PQ),
    pq_divmod(rtc_params.pq_divmod),
    q_divmod(rtc_params.q_divmod) {

    // Incremental offsets in unites of bytes (number of elements) * sizeof_bits<Element>::value / 8
    offset_next_strided = (threadmap_delta.strided() * layout.stride()[0])
                        * element_size_bits / 8;

    offset_next_contiguous = (threadmap_delta.contiguous())
                            * element_size_bits / 8;

    inc_next_npq = (threadblock_shape.column() * rtc_params.split_k_slices * layout.stride()[0])
                      * element_size_bits / 8;
  }
};

struct Conv2dWgradActivationIteratorOptimizedParams {

  using Layout = layout::TensorNHWC;

  Layout layout;

  FastDivmod sc_divmod;
  FastDivmod pq_divmod;
  FastDivmod q_divmod;
  FastDivmod c_divmod;
  // invalid pixels on end of each line
  int invalid_num;

  //
  // Methods
  //
  CUTLASS_HOST_DEVICE
  Conv2dWgradActivationIteratorOptimizedParams() { }

  CUTLASS_HOST_DEVICE
  Conv2dWgradActivationIteratorOptimizedParams(
    Conv2dProblemSize const &problem_size,
    Layout const &layout
  ):
    layout(layout),
    sc_divmod(problem_size.S * problem_size.C),
    pq_divmod(problem_size.P * problem_size.Q),
    q_divmod(problem_size.Q),
    c_divmod(problem_size.C) {
    invalid_num = problem_size.W + 2 * problem_size.pad_w - ((problem_size.Q - 1) * problem_size.stride_w + problem_size.stride_w);

  }

  CUTLASS_HOST_DEVICE
  Conv2dWgradActivationIteratorOptimizedParams(
    Conv2dProblemSize const &problem_size,
    Layout const &layout,
    int element_size_bits,                        ///< size of each element in bits
    MatrixCoord threadblock_shape,
    int thread_count,
    int access_size,
    layout::PitchLinearCoord threadmap_iterations,
    layout::PitchLinearCoord threadmap_delta
  ):
    Conv2dWgradActivationIteratorOptimizedParams(
      problem_size,
      layout
    ) {

      TRACE_CONV_INITIALIZERS("conv2d_wgrad", "activation",
        element_size_bits, threadblock_shape, thread_count, access_size, threadmap_iterations, threadmap_delta);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

