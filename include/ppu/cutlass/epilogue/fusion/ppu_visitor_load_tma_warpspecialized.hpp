/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved. 
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cutlass/cutlass.h"
#include "cutlass/arch/barrier.h"

#include "cute/tensor.hpp"
#include "cutlass/epilogue/fusion/ppu_visitor_tma_warpspecialized.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::fusion {

using namespace cute;
using namespace detail;

template <
  typename Element,
  int InputIdx
>
struct PPU0015ExtraFetch : PPU0015VisitorImpl<> {

  using Stride = cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>;

  struct Arguments {
    const Element* ptr = nullptr;
    Stride stride = {};
  };

  using Params = Arguments;

  CUTLASS_HOST_DEVICE
  PPU0015ExtraFetch() { }

  CUTLASS_HOST_DEVICE
  PPU0015ExtraFetch(Params const& params, SharedStorage const& shared_storage)
      : params_ptr(&params) {

  }

  Params const* params_ptr;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return args;
  }

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return true;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return true;
  }

  using PPU0015VisitorImpl<>::PPU0015VisitorImpl;

  // Tensor is (vector, subloop_m, subloop_n, loop_m, loop_n);
  template<class Tensor>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {

    Tensor const& extra_tensor;

    static constexpr int SubLoopM = size<1>(Tensor{});
    static constexpr int SubLoopN = size<2>(Tensor{});

    CUTLASS_DEVICE
    ConsumerStoreCallbacks(Tensor const& extra_tensor)
      : extra_tensor(extra_tensor) {}

    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE Array<typename Tensor::value_type, FragmentSize>
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n) {
      auto cur_loop_tensor = extra_tensor(_,_,_,epi_m, epi_n);
      int n = epi_v % SubLoopN;
      int m = epi_v / SubLoopN;
      return recast<Array<typename Tensor::value_type, FragmentSize>>(cur_loop_tensor(_,m,n))(0);
    }

  };

  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {

    auto [m_coord, n_coord, k_coord, l_coord] = args.tile_coord_mnkl;
    // smem size for one epilogue outer loop
    auto tile  = make_shape(size<0>(args.epi_tile), size<1>(args.epi_tile));
    auto M = get<0>(args.problem_shape_mnkl);
    auto N = get<1>(args.problem_shape_mnkl);
    auto L = get<3>(args.problem_shape_mnkl);

    auto tD     = args.tiled_copy.get_thread_slice(args.thread_idx);

    auto mE_mnl = make_tensor(make_gmem_ptr(params_ptr->ptr), make_shape(M,N,L), params_ptr->stride);
    auto gE_mnl = local_tile(mE_mnl, args.tile_shape_mnk, make_coord(_,_,_), Step<_1,_1, Underscore>{});
    auto gE = gE_mnl(_,_,m_coord,n_coord,l_coord);
    auto gEt = flat_divide(gE, tile);
    auto tDgE = tD.partition_D(gEt);

    return ConsumerStoreCallbacks(tDgE);
  }
};


} // namespace cutlass::epilogue::fusion
