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

#include <cute/algorithm/copy.hpp>

namespace cute{

template <
  class CopyPolicyA,
  class SrcEngineA, class SrcLayoutA,
  class DstEngineA, class DstLayoutA,
  class CopyPolicyB,
  class SrcEngineB, class SrcLayoutB,
  class DstEngineB, class DstLayoutB
>
CUTE_HOST_DEVICE
void
copy_aiu(
  CopyPolicyA                  const& copy_policy_a,
  Tensor<SrcEngineA, SrcLayoutA> const& src_a,
  Tensor<DstEngineA, DstLayoutA>     && dst_a,
  CopyPolicyB                  const& copy_policy_b,
  Tensor<SrcEngineB, SrcLayoutB> const& src_b,
  Tensor<DstEngineB, DstLayoutB>     && dst_b,
  int &warp_idx
) {
#if ACOMPUTE_VERSION == 10000
  if (warp_idx == 0) {
    copy(copy_policy_a, src_a, dst_a);
    copy(copy_policy_b, src_b, dst_b);
  }
#else
  if (warp_idx == 0) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<1>(src_a) / 2; ++i) {
      copy(copy_policy_a, src_a(_,i,_), dst_a(_,i,_));
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<1>(src_b) / 2; ++i) {
      copy(copy_policy_b, src_b(_,i,_), dst_b(_,i,_));
    }
  }
  else if (warp_idx == 1) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = size<1>(src_a) / 2; i < size<1>(src_a); ++i) {
      copy(copy_policy_a, src_a(_,i,_), dst_a(_,i,_));
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = size<1>(src_b) / 2; i < size<1>(src_b); ++i) {
      copy(copy_policy_b, src_b(_,i,_), dst_b(_,i,_));
    }
  }
#endif
}

template <
  class CopyPolicyA,
  class SrcEngineA, class SrcLayoutA,
  class DstEngineA, class DstLayoutA
>
CUTE_HOST_DEVICE
void
copy_aiu(
  CopyPolicyA                  const& copy_policy_a,
  Tensor<SrcEngineA, SrcLayoutA> const& src_a,
  Tensor<DstEngineA, DstLayoutA>      & dst_a,
  int &warp_idx
) {
#if ACOMPUTE_VERSION == 10000
  if (warp_idx == 0) {
    copy(copy_policy_a, src_a, dst_a);
  }
#else
  // used by fa, will only split on last dim now
  static constexpr int rank  = DstLayoutA::rank;
  if (warp_idx == 0) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<rank-1>(src_a) / 2; ++i) {
      copy(copy_policy_a, src_a(_,_,i), dst_a(_,_,i));
    }
  }
  else if (warp_idx == 1) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = size<rank-1>(src_a) / 2; i < size<rank-1>(src_a); ++i) {
      copy(copy_policy_a, src_a(_,_,i), dst_a(_,_,i));
    }
  }
#endif
}

template <
  class CopyPolicyA,
  class SrcEngineA, class SrcLayoutA,
  class DstEngineA, class DstLayoutA
>
CUTE_HOST_DEVICE
void
copy_aiu(
  CopyPolicyA                  const& copy_policy_a,
  Tensor<SrcEngineA, SrcLayoutA> const& src_a,
  Tensor<DstEngineA, DstLayoutA>     && dst_a,
  int &warp_idx
) {
  copy(copy_policy_a, src_a, dst_a);
}

} // namespace cute
