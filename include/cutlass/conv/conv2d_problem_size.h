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
    \brief This file contains definitions and utility functions for describing convolution problem sizes.

  Conv2dProblem desciption:
    activation (NHWC),
    filter (KRSC),
    output (NPQK),
    pading (pad_pre_h, pad_pre_w, pad_post_h, pad_post_w),
    stride (stride_h, stride_w),
    dilation (dilation_h, dilation_w).

  Free functions to map:
    Map tensor extents (Conv2d -> ImplicitGemm)      : implicit_gemm_tensor_[a|b|c]_extent(ConvolutionOperator)
    Map tensor sizes (Conv2d -> ImplicitGemm)        : implicit_gemm_tensor_[a|b|c]_size(ConvolutionOperator)
    Map tensor problem sizes (Conv2d -> ImplicitGemm): implicit_gemm_problem_size(ConvolutionOperator)
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/conv/convolution.h"

namespace cutlass {
namespace conv {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Problem size structure
struct Conv2dProblemSize {

  // Conv2d strictly problem size parameters
  int N, H, W, C, P, Q, K, R, S;
  int pad_h, pad_w, pad_post_h, pad_post_w;
  int stride_h, stride_w;
  int dilation_h, dilation_w;
  Mode mode;

  // Conv2d implementation-related parameters
  int split_k_slices;
  int groups;
  int group_stride {1};

#if SAIL_DGRAD_STRIDE_OPT
  // used in dgrad when stride>1
  // TODO: only support stride = 2 now
  int block_num[4];
  // start rs position
  int start_r_[4];
  int start_s_[4];
  // h value of elements valid on cur rs position
  int valid_h[4];
  int valid_w[4];
  // start position on ori coordinate
  int start_h[4];
  int start_w[4];
  // loop times on r
  int loop_r[4];
  int loop_s[4];

  int block_shape_m;

#endif

  // used in fprop small channel, and used to mark this is a small channel kernel to bypass stride check
  int fold_num = 1;
  // used in dgrad small channel
  int fold_h = 1;
  int fold_w = 1;

  //
  // Methods
  //

public:
  CUTLASS_HOST_DEVICE
  Conv2dProblemSize():
    N(0), H(0), W(0), C(0), P(0), Q(0), K(0), R(0), S(0),
    pad_h(0), pad_w(0), pad_post_h(0), pad_post_w(0), stride_h(1), stride_w(1), dilation_h(1), dilation_w(1),
    mode(Mode::kConvolution), split_k_slices(1), groups(1), fold_num(1), fold_h(1), fold_w(1) {}

  /// Constructor for default padding, stride, dilation, and split-K
  CUTLASS_HOST_DEVICE
  Conv2dProblemSize(
    int N,
    int H,
    int W,
    int C,
    int P,
    int Q,
    int K,
    int R,
    int S,
    Mode mode,
    int fold_num = 1,
    int fold_h = 1,
    int fold_w = 1
  ):
    N(N), H(H), W(W), C(C), P(P), Q(Q), K(K), R(R), S(S),
    pad_h(R / 2), pad_w(S / 2), pad_post_h(R / 2), pad_post_w(S / 2), stride_h(1), stride_w(1), dilation_h(1), dilation_w(1),
    mode(mode), split_k_slices(1), groups(1), fold_num(fold_num), fold_h(fold_h), fold_w(fold_w) {}

  /// Constructor
  CUTLASS_HOST_DEVICE
  Conv2dProblemSize(
    int N,
    int H,
    int W,
    int C,
    int K,
    int R,
    int S,
    int P,
    int Q,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    Mode mode,
    int split_k_slices = 1,
    int groups = 1,
    int fold_num = 1,
    int fold_h = 1,
    int fold_w = 1
  ):
    N(N), H(H), W(W), C(C), K(K), R(R), S(S), P(P), Q(Q),
    pad_h(pad_h), pad_w(pad_w), pad_post_h(pad_h), pad_post_w(pad_w), stride_h(stride_h), stride_w(stride_w),
    dilation_h(dilation_h), dilation_w(dilation_w),
    mode(mode), split_k_slices(split_k_slices), groups(groups), fold_num(fold_num), fold_h(fold_h), fold_w(fold_w) {}

  /// Constructs convolution problem size from cutlass Tensor4DCoord and MatrixCoord
  // set user-defined output size and sets P and Q (include all data members in ctor)
  CUTLASS_HOST_DEVICE
  Conv2dProblemSize(
    cutlass::Tensor4DCoord input_size,    // NHWC
    cutlass::Tensor4DCoord filter_size,   // KRSC
    cutlass::Tensor4DCoord padding,       // pad_pre_h, pad_post_h, pad_pre_w, pad_post_w
    cutlass::MatrixCoord stride,          // stride_h, stride_w
    cutlass::MatrixCoord dilation,        // dilation_h, dilation_w
    cutlass::Tensor4DCoord output_size,   // NPQK
    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation,
    int split_k_slices = 1,
    int groups = 1,
    int fold_num = 1,
    int fold_h = 1,
    int fold_w = 1
  ):
    N(input_size.n()), H(input_size.h()), W(input_size.w()), C(input_size.c()),
    K(filter_size.n()), R(filter_size.h()), S(filter_size.w()),
    pad_h(padding[0]), pad_w(padding[2]), pad_post_h(padding[1]), pad_post_w(padding[3]),
    stride_h(stride.row()), stride_w(stride.column()),
    dilation_h(dilation.row()), dilation_w(dilation.column()),
    P(output_size.h()), Q(output_size.w()),
    mode(mode), split_k_slices(split_k_slices), groups(groups), fold_num(fold_num), fold_h(fold_h), fold_w(fold_w) {}

  /// Constructs convolution problem size from cutlass Tensor4DCoord and MatrixCoord
  // computes output size and sets P and Q (skip output from ctor arguments)
  CUTLASS_HOST_DEVICE
  Conv2dProblemSize(
    cutlass::Tensor4DCoord input_size,   // NHWC
    cutlass::Tensor4DCoord filter_size,  // KRSC
    cutlass::Tensor4DCoord padding,      // pad_pre_h, pad_post_h, pad_pre_w, pad_post_w
    cutlass::MatrixCoord stride,         // stride_h, stride_w
    cutlass::MatrixCoord dilation,       // dilation_h, dilation_w
    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation,
    int split_k_slices = 1,
    int groups = 1,
    int fold_num = 1,
    int fold_h = 1,
    int fold_w = 1
  ):
    N(input_size.n()), H(input_size.h()), W(input_size.w()), C(input_size.c()),
    K(filter_size.n()), R(filter_size.h()), S(filter_size.w()),
    pad_h(padding[0]), pad_w(padding[2]), pad_post_h(padding[1]), pad_post_w(padding[3]),
    stride_h(stride.row()), stride_w(stride.column()),
    dilation_h(dilation.row()), dilation_w(dilation.column()),
    mode(mode), split_k_slices(split_k_slices), groups(groups), fold_num(fold_num), fold_h(fold_h), fold_w(fold_w) {
      // set output P and Q
      P = ((H + pad_h + pad_post_h - R * dilation_h) / stride_h) + 1;
      Q = ((W + pad_w + pad_post_w - S * dilation_w) / stride_w) + 1;
    }

  /// Constructs convolution problem size from cutlass Tensor4DCoord and MatrixCoord
  // set user-defined output size and sets P and Q (skip padding, striding, and dilation)
  CUTLASS_HOST_DEVICE
  Conv2dProblemSize(
    cutlass::Tensor4DCoord input_size,    // NHWC
    cutlass::Tensor4DCoord filter_size,   // KRSC
    cutlass::Tensor4DCoord output_size,   // NPQK
    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation,
    int split_k_slices = 1,
    int groups = 1,
    int fold_num = 1,
    int fold_h = 1,
    int fold_w = 1
  ):
    N(input_size.n()), H(input_size.h()), W(input_size.w()), C(input_size.c()),
    K(filter_size.n()), R(filter_size.h()), S(filter_size.w()),
    P(output_size.h()), Q(output_size.w()),
    pad_h(R / 2), pad_w(S / 2), pad_post_h(R / 2), pad_post_w(S / 2), stride_h(1), stride_w(1),
    dilation_h(1), dilation_w(1),
    mode(mode), split_k_slices(split_k_slices), groups(groups), fold_num(fold_num), fold_h(fold_h), fold_w(fold_w) {}

#if SAIL_DGRAD_STRIDE_OPT
  CUTLASS_HOST_DEVICE
  void init_for_dgrad(int blockshape_m) {
    block_shape_m = blockshape_m;
    block_num[0] = (N * ((H + 1) / 2) * ((W + 1) / 2) + blockshape_m - 1) / blockshape_m;
    block_num[1] = block_num[0] + (N * ((H + 1) / 2) * ((W) / 2) + blockshape_m - 1) / blockshape_m;
    block_num[2] = block_num[1] + (N * ((H) / 2) * ((W + 1) / 2) + blockshape_m - 1) / blockshape_m;
    block_num[3] = block_num[2] + (N * ((H) / 2) * ((W) / 2) + blockshape_m - 1) / blockshape_m;

    start_r_[0] = pad_h % stride_h;
    start_s_[0] = pad_w % stride_w;
    valid_h[0] = (H + 1) / 2;
    valid_w[0] = (W + 1) / 2;
    start_h[0] = 0;
    start_w[0] = 0;

    start_r_[1] = pad_h % stride_h;
    start_s_[1] = (pad_w + 1) % stride_w;
    valid_h[1] = (H + 1) / 2;
    valid_w[1] = (W) / 2;
    start_h[1] = 0;
    start_w[1] = 1;

    start_r_[2] = (pad_h + 1) % stride_h;
    start_s_[2] = pad_w % stride_w;
    valid_h[2] = (H) / 2;
    valid_w[2] = (W + 1) / 2;
    start_h[2] = 1;
    start_w[2] = 0;

    start_r_[3] = (pad_h + 1) % stride_h;
    start_s_[3] = (pad_w + 1) % stride_w;
    valid_h[3] = (H) / 2;
    valid_w[3] = (W) / 2;
    start_h[3] = 1;
    start_w[3] = 1;

    loop_r[0] = (R - start_r_[0] + stride_h - 1) / stride_h;
    loop_s[0] = (S - start_s_[0] + stride_w - 1) / stride_w;
    loop_r[1] = (R - start_r_[1] + stride_h - 1) / stride_h;
    loop_s[1] = (S - start_s_[1] + stride_w - 1) / stride_w;
    loop_r[2] = (R - start_r_[2] + stride_h - 1) / stride_h;
    loop_s[2] = (S - start_s_[2] + stride_w - 1) / stride_w;
    loop_r[3] = (R - start_r_[3] + stride_h - 1) / stride_h;
    loop_s[3] = (S - start_s_[3] + stride_w - 1) / stride_w;

  }
#endif

  CUTLASS_HOST_DEVICE
  void print() const {
    printf("N: %d, H: %d, W: %d, C: %d, P: %d, Q: %d, K: %d, R: %d, S: %d\n", N, H, W, C, P, Q, K, R, S);
    printf("pad_h: %d, pad_w: %d, pad_post_h: %d, pad_post_w: %d\n", pad_h, pad_w, pad_post_h, pad_post_w);
    printf("stride_h: %d, stride_w: %d, dilation_h: %d, dilation_w: %d\n", stride_h, stride_w, dilation_h, dilation_w);
    printf("mode: %d, split_k_slices: %d, groups: %d, group_stride: %d, fold_num: %d, fold_h: %d, fold_w: %d\n", int(mode), split_k_slices, groups, group_stride, fold_num, fold_h, fold_w);
  }

  // Reset covolution mode in the problem
  CUTLASS_HOST_DEVICE
  Conv2dProblemSize reset_mode(cutlass::conv::Mode mode_) {
    Conv2dProblemSize tmp(*this);
    tmp.mode = mode_;
    return tmp;
  }

  // Reset covolution mode in the problem
  CUTLASS_HOST_DEVICE
  Conv2dProblemSize reset_split_k_slices(int split_k_slices_) {
    Conv2dProblemSize tmp(*this);
    tmp.split_k_slices = split_k_slices_;
    return tmp;
  }

  // Set group covolution stride in the problem
  CUTLASS_HOST_DEVICE
  Conv2dProblemSize set_group_stride(int g_stride_) {
    Conv2dProblemSize tmp(*this);
    tmp.group_stride = g_stride_;
    tmp.C = tmp.C / g_stride_;
    tmp.K = tmp.K / g_stride_;
    return tmp;
  }

  /// Equality operator (ignores mode and split_k_slice)
  CUTLASS_HOST_DEVICE
  bool operator==(Conv2dProblemSize const &conv) const {
    return (
      (N == conv.N) && (W == conv.H) && (W == conv.W) && (C == conv.C) &&
      (K == conv.K) && (R == conv.R) && (S == conv.S) &&
      (P == conv.P) && (Q == conv.Q) &&
      (pad_h == conv.pad_h) && (pad_w == conv.pad_w) && (pad_post_w == conv.pad_post_w) && (pad_post_h == conv.pad_post_h) &
      (stride_h == conv.stride_h) && (stride_w == conv.stride_w) &&
      (dilation_h == conv.dilation_h) && (dilation_h == conv.dilation_h)
    );
  }

  /// Inequality operator
  CUTLASS_HOST_DEVICE
  bool operator!=(Conv2dProblemSize const &rhs) const {
    return !(*this == rhs);
  }

  /// Returns activation extent as Tensor4DCoord
  CUTLASS_HOST_DEVICE
  cutlass::Tensor4DCoord activation_extent() const {
    // C/K already / group in problem_size, stride should use origin value
    return cutlass::Tensor4DCoord ({N, H, W, C * group_stride});
  }

  /// Returns filter extent as Tensor4DCoord
  CUTLASS_HOST_DEVICE
  cutlass::Tensor4DCoord filter_extent() const {
    // weight's stride_c is c in group
    return cutlass::Tensor4DCoord ({K * group_stride, R, S, C * group_stride / groups});
  }

  /// Returns output extent as Tensor4DCoord
  CUTLASS_HOST_DEVICE
  cutlass::Tensor4DCoord output_extent() const {

    return cutlass::Tensor4DCoord ({N, P, Q, K * group_stride});
  }

  /// Returns activation size in number of elements
  CUTLASS_HOST_DEVICE
  int64_t activation_size() const {

    return (N * H * W * C);
  }

  /// Returns filter size in number of elements
  CUTLASS_HOST_DEVICE
  int64_t filter_size() const {

    return (K * R * S * C * group_stride / groups);
  }

  /// Returns output size in number of elements
  CUTLASS_HOST_DEVICE
  int64_t output_size() const {

    return (N * P * Q * K);
  }

  /// Returns output extent as Tensor4DCoord
  CUTLASS_HOST_DEVICE
  cutlass::Tensor4DCoord padding() const {

    return cutlass::Tensor4DCoord ({pad_h, pad_post_h, pad_w, pad_post_w});
  }

  /// Returns stride as MatrixCoord
  CUTLASS_HOST_DEVICE
  cutlass::MatrixCoord stride() const {

    return cutlass::MatrixCoord ({stride_h, stride_w});
  }

  /// Returns dilation as MatrixCoord
  CUTLASS_HOST_DEVICE
  cutlass::MatrixCoord dilation() const {

    return cutlass::MatrixCoord ({dilation_h, dilation_w});
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//                                  ImplicitGemm helper functions                                 //
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Determine the problem size of the implicit GEMM operation
CUTLASS_HOST_DEVICE
cutlass::gemm::GemmCoord implicit_gemm_problem_size(
  Operator conv_operator,
  Conv2dProblemSize const &problem_size) {
  int fold_r = problem_size.fold_h;
  int fold_s = problem_size.fold_w;

  // Compute problem size
  switch (conv_operator) {
  case Operator::kFprop:
    return gemm::GemmCoord(
      problem_size.N * problem_size.P * problem_size.Q,
      problem_size.K,
      problem_size.R * problem_size.S * problem_size.C
    );
  case Operator::kDgrad:
    // when stride=dilation, fold_r/s=1, use fold_num to mark this is a small channel implementation
    if (problem_size.fold_num != 1) {
      int folded_h = (problem_size.H + problem_size.stride_h - 1) / problem_size.stride_h;
      int folded_w = (problem_size.W + problem_size.stride_w - 1) / problem_size.stride_w;
      return gemm::GemmCoord(
        problem_size.N * folded_h * folded_w,
        problem_size.C * fold_r * fold_s,
        problem_size.R * problem_size.S * problem_size.K / fold_r / fold_s
      );
    } else {
      return gemm::GemmCoord(
        problem_size.N * problem_size.H * problem_size.W,
        problem_size.C,
        problem_size.R * problem_size.S * problem_size.K
      );
    }
  case Operator::kWgrad:
    return gemm::GemmCoord(
      problem_size.K,
      problem_size.R * problem_size.S * problem_size.C,
      problem_size.N * problem_size.P * problem_size.Q
    );
  default:
    break;
  }
  return gemm::GemmCoord();
}

CUTLASS_HOST_DEVICE
cutlass::Coord<3> implicit_gemm_problem_transpose_size(
  Operator conv_operator,
  Conv2dProblemSize const &problem_size) {
  switch (conv_operator) {
  case Operator::kFprop:
    return Coord<3>(
      {
        problem_size.N,
        problem_size.P * problem_size.Q,
        problem_size.K
      }
    );
  case Operator::kDgrad:
    return Coord<3>(
      {
        problem_size.N,
        problem_size.H * problem_size.W,
        problem_size.C
      }
    );
  case Operator::kWgrad:
    return Coord<3>(
      {
        problem_size.K,
        problem_size.R * problem_size.S,
        problem_size.C
      }
    );
  default:
    break;
  }
  return Coord<3>();
}

// Determine the number of gemm_k iterations for conv2d problem using implicit gemm algorithm
CUTLASS_HOST_DEVICE
int implicit_gemm_k_iterations(
  Operator conv_operator,
  int threadblock_K,
  Conv2dProblemSize const &problem_size,
  IteratorAlgorithm algorithm  = IteratorAlgorithm::kOptimized,
  KernelType KernelType_ = KernelType::kNormal,
  int threadblock_N = 0) {
  int iterations = 0;
  int elements_per_split_k_slice = 0;

  if (KernelType_ == KernelType::kMultipleGroup) {
    int c_per_group = problem_size.C / problem_size.groups;
    int k_per_group = problem_size.K / problem_size.groups;

    switch (conv_operator) {
      case Operator::kFprop:
        elements_per_split_k_slice = (problem_size.C + problem_size.split_k_slices - 1) / problem_size.split_k_slices;
        iterations = problem_size.R * problem_size.S * ((elements_per_split_k_slice + threadblock_K - 1) / threadblock_K);
        // In group conv, if k_per_group < threadblock_N, one Threadblock will calculate multiple groups
        if (problem_size.groups != 1) {
          if (k_per_group <= threadblock_N) {
            iterations = iterations / (problem_size.groups / (threadblock_N / k_per_group));
          }
        }
        break;

      case Operator::kDgrad:
        elements_per_split_k_slice = (problem_size.K + problem_size.split_k_slices - 1) / problem_size.split_k_slices;
        iterations = problem_size.R * problem_size.S * ((elements_per_split_k_slice + threadblock_K - 1) / threadblock_K);
        // In group conv, if k_per_group < threadblock_N, one Threadblock will calculate multiple groups
        if (problem_size.groups != 1) {
          if (c_per_group <= threadblock_N) {
            iterations = iterations / (problem_size.groups / (threadblock_N / c_per_group));
          }
        }
        break;

      case Operator::kWgrad:
        elements_per_split_k_slice = (problem_size.N * problem_size.P * problem_size.Q + problem_size.split_k_slices - 1) / problem_size.split_k_slices;
        iterations = (elements_per_split_k_slice + threadblock_K - 1) / threadblock_K;
        break;

      default:
        break;
    }
  } else {
    switch (conv_operator) {
    case Operator::kFprop:
      if (problem_size.fold_num != 1) {
        assert(problem_size.split_k_slices == 1);
#ifdef AIU_CONV
        // aiu needn't padding, rsc is original value
        iterations = problem_size.R * ((problem_size.S * problem_size.C + threadblock_K - 1) / threadblock_K);
#else
        iterations = (problem_size.R * problem_size.S * problem_size.C + threadblock_K - 1) / threadblock_K;
#endif
      } else {
        elements_per_split_k_slice = (problem_size.C + problem_size.split_k_slices - 1) / problem_size.split_k_slices;
        iterations = problem_size.R * problem_size.S * ((elements_per_split_k_slice + threadblock_K - 1) / threadblock_K);
      }
      break;

    case Operator::kDgrad:
      if (problem_size.fold_num != 1) {
        assert(problem_size.split_k_slices == 1);
        elements_per_split_k_slice = (problem_size.K + problem_size.split_k_slices - 1) / problem_size.split_k_slices;
        iterations = problem_size.R / problem_size.fold_h * problem_size.S / problem_size.fold_w * ((elements_per_split_k_slice + threadblock_K - 1) / threadblock_K);
      } else {
        elements_per_split_k_slice = (problem_size.K + problem_size.split_k_slices - 1) / problem_size.split_k_slices;
        iterations = problem_size.R * problem_size.S * ((elements_per_split_k_slice + threadblock_K - 1) / threadblock_K);
      }
      break;

    case Operator::kWgrad:
      elements_per_split_k_slice = (problem_size.N * problem_size.P * problem_size.Q + problem_size.split_k_slices - 1) / problem_size.split_k_slices;
      iterations = (elements_per_split_k_slice + threadblock_K - 1) / threadblock_K;
      break;

    default:
      break;
    }
  }
  return iterations;
}

// helper function to get conv stride coord.
CUTLASS_HOST_DEVICE
cutlass::gemm::GemmCoord implicit_gemm_problem_stride_coord(
  Operator conv_operator,
  int const stride,
  Conv2dProblemSize const &problem_size,
  KernelType KernelType_ = KernelType::kGroup) {
  switch (conv_operator) {
    case Operator::kFprop:
    case Operator::kDgrad:
      return gemm::GemmCoord({1, stride, 1});
    case Operator::kWgrad:
      if (KernelType_ == KernelType::kMultipleGroup)
        return gemm::GemmCoord({stride, problem_size.R * problem_size.S, 1});
      return gemm::GemmCoord({stride, 1, 1});
  default:
      break;
  }
  return gemm::GemmCoord({1, 1, 1});
}

CUTLASS_DEVICE
cutlass::gemm::GemmCoord implicit_gemm_problem_stride_group(
  Operator conv_operator,
  Conv2dProblemSize const &problem_size) {
  switch (conv_operator) {
    case Operator::kFprop:
       return gemm::GemmCoord({problem_size.C, int(problem_size.filter_size()), problem_size.K});
    case Operator::kDgrad:
      return gemm::GemmCoord({problem_size.K, int(problem_size.filter_size()), problem_size.C});
    case Operator::kWgrad:
      return gemm::GemmCoord({problem_size.K, problem_size.C, int(problem_size.filter_size())});
  default:
      break;
  }
  return gemm::GemmCoord({1, 1, 1});
}

template <int N = 1, int Output_P = 1, int Output_Q = 1>
CUTLASS_HOST_DEVICE
int depthwise_gemm_k_iterations(
  Operator conv_operator,
  int threadblock_K,
  Conv2dProblemSize const &problem_size,
  IteratorAlgorithm algorithm = IteratorAlgorithm::kAnalytic,
  KernelType group_mode = KernelType::kNormal,
  int threadblock_N = 0) {

    int n =  problem_size.N;
    int p = (problem_size.P + Output_P - 1) /  Output_P;
    int q = (problem_size.Q + Output_Q - 1) /  Output_Q;

    int iterations = (n * p * q + problem_size.split_k_slices - 1) / problem_size.split_k_slices;
    return iterations;
}


CUTLASS_HOST_DEVICE
int implicit_gemm_k_iterations_per_channel(
    Operator conv_operator,
    int threadblock_K,
    Conv2dProblemSize const &problem_size,
    IteratorAlgorithm algorithm = IteratorAlgorithm::kAnalytic) {

  int iterations = 0; //0 means not applicable
  if (algorithm == IteratorAlgorithm::kAnalytic || algorithm == IteratorAlgorithm::kOptimized) {
    switch (conv_operator) {
      case Operator::kFprop:
        iterations = problem_size.R * problem_size.S;
        break;

      case Operator::kDgrad:
        iterations = problem_size.R * problem_size.S;
        break;

      default:
        break;
    }
  }
  return iterations;
}

////////////////////////////////////////////////////////////////////////////////
//  Mapping function (ImplicitGemm A, B, C -> Conv Activation, Filter, Output)
////////////////////////////////////////////////////////////////////////////////
/// Returns ImplicitGemm tensor A extent as Tensor4DCoord
CUTLASS_HOST_DEVICE
cutlass::Tensor4DCoord implicit_gemm_tensor_a_extent(
  Operator conv_operator,
  Conv2dProblemSize const &problem_size) {
  switch (conv_operator) {
    case cutlass::conv::Operator::kFprop: return problem_size.activation_extent();
    case cutlass::conv::Operator::kDgrad: return problem_size.output_extent();
    case cutlass::conv::Operator::kWgrad: return problem_size.output_extent();
    default : break;
  }
  return cutlass::Tensor4DCoord();
}

/// Returns ImplicitGemm tensor B extent as Tensor4DCoord
CUTLASS_HOST_DEVICE
cutlass::Tensor4DCoord implicit_gemm_tensor_b_extent(
  Operator conv_operator,
  Conv2dProblemSize const &problem_size) {
  switch (conv_operator) {
    case cutlass::conv::Operator::kFprop: return problem_size.filter_extent();
    case cutlass::conv::Operator::kDgrad: return problem_size.filter_extent();
    case cutlass::conv::Operator::kWgrad: return problem_size.activation_extent();
    default : break;
  }
  return cutlass::Tensor4DCoord();
}

/// Returns ImplicitGemm tensor C extent as Tensor4DCoord
CUTLASS_HOST_DEVICE
cutlass::Tensor4DCoord implicit_gemm_tensor_c_extent(
  Operator conv_operator,
  Conv2dProblemSize const &problem_size) {
  switch (conv_operator) {
    case cutlass::conv::Operator::kFprop: return problem_size.output_extent();
    case cutlass::conv::Operator::kDgrad: return problem_size.activation_extent();
    case cutlass::conv::Operator::kWgrad: return problem_size.filter_extent();
    default : break;
  }
  return cutlass::Tensor4DCoord();
}

/// Returns ImplicitGemm tensor A size in number of elements
CUTLASS_HOST_DEVICE
int64_t implicit_gemm_tensor_a_size(
  Operator conv_operator,
  Conv2dProblemSize const &problem_size) {
  switch (conv_operator) {
    case cutlass::conv::Operator::kFprop: return problem_size.activation_size();
    case cutlass::conv::Operator::kDgrad: return problem_size.output_size();
    case cutlass::conv::Operator::kWgrad: return problem_size.output_size();
    default : break;
  }
  return 0;
}

/// Returns ImplicitGemm tensor B size in number of elements
CUTLASS_HOST_DEVICE
int64_t implicit_gemm_tensor_b_size(
  Operator conv_operator,
  Conv2dProblemSize const &problem_size) {
  switch (conv_operator) {
    case cutlass::conv::Operator::kFprop: return problem_size.filter_size();
    case cutlass::conv::Operator::kDgrad: return problem_size.filter_size();
    case cutlass::conv::Operator::kWgrad: return problem_size.activation_size();
    default : break;
  }
  return 0;
}

/// Returns ImplicitGemm tensor C size in number of elements
CUTLASS_HOST_DEVICE
int64_t implicit_gemm_tensor_c_size(
  Operator conv_operator,
  Conv2dProblemSize const &problem_size) {
  switch (conv_operator) {
    case cutlass::conv::Operator::kFprop: return problem_size.output_size();
    case cutlass::conv::Operator::kDgrad: return problem_size.activation_size();
    case cutlass::conv::Operator::kWgrad: return problem_size.filter_size();
    default : break;
  }
  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace conv
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////////////////////////
