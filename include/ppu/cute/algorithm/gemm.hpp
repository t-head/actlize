/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved. 
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <cute/config.hpp>

#include <cute/util/type_traits.hpp>
#include <cute/algorithm/functional.hpp>

#include <cute/tensor.hpp>

#include <cute/atom/mma_atom.hpp>

namespace cute
{

// Dispatch [4]: (V,M) x (V,N) => (V,M,N) with scale
template <class MMA,
          class TD,  class DLayout,
          class TA,  class ALayout,
          class TSA, class SALayout,
          class TB,  class BLayout,
          class TSB, class SBLayout,
          class TC, class CLayout,
          __CUTE_REQUIRES(DLayout::rank == 3 && is_rmem<TD>::value &&
                          ALayout::rank == 2 && is_rmem<TA>::value &&
                          BLayout::rank == 2 && is_rmem<TB>::value &&
                          CLayout::rank == 3 && is_rmem<TC>::value)>
CUTE_HOST_DEVICE
void
gemm(MMA_Atom<MMA>          const& mma,
     Tensor<TD,  DLayout>        & D,  // (V,M,N) Logical data
     Tensor<TA,  ALayout>   const& A,  // (V,M)   Logical data
     Tensor<TSA, SALayout>  const& SA,  // (V,M)   Logical data
     Tensor<TB,  BLayout>   const& B,  // (V,N)   Logical data
     Tensor<TSB, SBLayout>  const& SB,  // (V,N)   Logical data
     Tensor<TC,  CLayout>   const& C)  // (V,M,N) Logical data
{
  CUTE_STATIC_ASSERT_V(size<1>(A) == size<1>(C));  // AM == CM
  CUTE_STATIC_ASSERT_V(size<1>(B) == size<2>(C));  // BN == CN
  CUTE_STATIC_ASSERT_V(size<0>(C) == size<0>(D) && size<1>(C) == size<1>(D) && size<2>(C) == size<2>(D));
  auto M = size<1>(A);
  auto N = size<1>(B);
  auto sM = size<1>(SA);
  auto sN = size<1>(SB);

  Tensor s = make_tensor<uint32_t>(Int<4>{});

  constexpr int m_serpentine = int(M) / int(sM);
  constexpr int n_serpentine = int(N) / int(sN);

  // Col-major serpentine iteration
  CUTE_UNROLL
  for (int n = 0; n < N; ++n) {
    CUTE_UNROLL
    for (int m = 0; m < M; ++m) {
      int ms = (n & 1) ? M-1-m : m;  // Serpentine coordinate
      Tensor d = D(_,ms,n);
      Tensor c = C(_,ms,n);
      Tensor a = A(_,ms);
      Tensor b = B(_,n);
      s[0] = SA[ms / m_serpentine];
      s[1] = SB[n / n_serpentine];
      s[2] = ms % m_serpentine;
      s[3] = n % n_serpentine;

      // mma.call(d, a, b, c, s);
      mma_unpack(mma, d, a, b, c, s);
    }
  }
}

} // end namespace cute
