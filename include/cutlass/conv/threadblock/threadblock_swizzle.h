/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved. 
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/*! \file
    \brief Implements several possible threadblock-swizzling functions mapping blockIdx to 
      Convolution problems.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/platform/platform.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace threadblock {

/// Threadblock swizzling function for wgrad group convolution
struct WgradGroupHorizontalThreadblockSwizzle :
  public gemm::threadblock::GemmHorizontalThreadblockSwizzle {

  using Base = gemm::threadblock::GemmHorizontalThreadblockSwizzle;

  CUTLASS_HOST_DEVICE
  WgradGroupHorizontalThreadblockSwizzle() { }

  /// Returns the shape of the problem in units of logical tiles
  /// For ImplicitGemmConvolution Conv2d problem size: conv_operator(NPQK, NHWC, KRSC)
  CUTLASS_HOST_DEVICE
  gemm::GemmCoord get_tiled_shape(
    gemm::GemmCoord problem_size,
    gemm::GemmCoord tile_size,
    int split_k_slices,
    gemm::GemmCoord stride = gemm::GemmCoord{1, 1, 1}) const {

    // compute number of tiles in m dimension
    int tile_m = (problem_size.m() + tile_size.m() - 1) / tile_size.m();

    // n dimension always use 1 block to combine block_tile_n / groups x kernel size data.
    int group_n = 1;
    int tile_n = group_n * stride.n();

    return gemm::GemmCoord(
      tile_m,
      tile_n,
      split_k_slices);
  }

  /// Returns the shape of the problem in units of logical tiles
  /// For GEMM problem size (MxNxK) (Do not use base class get_tiled_shape())
  private:
    using Base::get_tiled_shape;

  // public:
  //   cutlass::conv::Conv2dProblemSize conv_problem_size;
};

/// Threadblock swizzling function for GEMMs
template <int N = 1, int Output_N = 1, int Output_P = 1, int Output_Q = 1>
struct DepthwiseDirect2dConvIdentityThreadblockSwizzle
    : public gemm::threadblock::GemmIdentityThreadblockSwizzle<N> {
  CUTLASS_HOST_DEVICE
  DepthwiseDirect2dConvIdentityThreadblockSwizzle() {}

  /// Returns the shape of the problem in units of logical tiles
  CUTLASS_HOST_DEVICE
  gemm::GemmCoord get_tiled_shape(
    gemm::GemmCoord problem_size,
    gemm::GemmCoord tile_size,
    int split_k_slices,
    gemm::GemmCoord stride = gemm::GemmCoord{1, 1, 1}) const {

    return gemm::GemmCoord(1,
                     (problem_size.n() + tile_size.n() - 1) / tile_size.n(),
                     split_k_slices);
  }
};
/////////////////////////////////////////////////////////////////////////////////////////////////


} // namespace threadblock
} // namespace gemm
} // namespace cutlass
