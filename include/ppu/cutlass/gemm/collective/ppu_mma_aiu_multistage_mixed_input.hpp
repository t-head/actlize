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

#include "cutlass/cutlass.h"
#include "ppu/cutlass/gemm/dispatch_policy.hpp"
#include "ppu/cutlass/fast_numeric_conversion_for_mix_gemm.h"

#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor_predicate.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/detail/collective.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int Stages,
  class TileShape_,
  class ElementAOptionalTuple,
  class StrideA_,
  class ElementBOptionalTuple,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMma<
    MainloopPPUAiuMixedInput<Stages>,
    TileShape_,
    ElementAOptionalTuple,
    StrideA_,
    ElementBOptionalTuple,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_>
{
private:
  enum class ConversionMode {
    DirectConvert,
    ConvertAndScale,
    ConvertAndScaleWithZero
  };
  using ScaleA = detail::deduce_mixed_width_dtype_t<1, ElementAOptionalTuple>;
  using ScaleB = detail::deduce_mixed_width_dtype_t<1, ElementBOptionalTuple>;
  using ZeroA = detail::deduce_mixed_width_dtype_t<2, ElementAOptionalTuple>;
  using ZeroB = detail::deduce_mixed_width_dtype_t<2, ElementBOptionalTuple>;
public:
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopPPUAiuMixedInput<Stages>;
  using TileShape = TileShape_;
  using ElementA = detail::deduce_mixed_width_dtype_t<0, ElementAOptionalTuple>;
  using ElementB = detail::deduce_mixed_width_dtype_t<0, ElementBOptionalTuple>;
  static constexpr bool IsATransformed = cute::is_tuple<ElementAOptionalTuple>::value;
  using ElementScale = cute::conditional_t<IsATransformed, ScaleA, ScaleB>;
  using ElementZero = cute::conditional_t<IsATransformed, ZeroA, ZeroB>;
  // For cases where we can't have a void type, we can use this to allow the code to compile when the scale / zero is void.
  using NonVoidElementScale = cute::conditional_t<cute::is_void_v<ElementScale>, float, ElementScale>;
  using NonVoidElementZero = cute::conditional_t<cute::is_void_v<ElementZero>, float, ElementZero>;

  using StrideA = StrideA_;
  using StrideB = StrideB_;
  // These are always MN major
  using StrideScale = cute::Stride<cute::Int<1>, int64_t, int64_t>;
  // For cases where we can't have a void scale, we can use this to allow the code to compile when the scale is void.
  using NonVoidStrideScale = cute::conditional_t<cute::is_void_v<StrideScale>, cute::Stride<_1, int64_t, int64_t>, StrideScale>;

  static_assert((IsATransformed && cutlass::gemm::detail::is_k_major<StrideA>()) ||
                (!IsATransformed && cutlass::gemm::detail::is_k_major<StrideB>()),
                "The transformed type must be K-major.");

  static_assert(( IsATransformed && (sizeof(ElementB) == 2)) ||
                (!IsATransformed && (sizeof(ElementA) == 2)) ||
                (cutlass::gemm::detail::is_k_major<StrideA>() &&
                 cutlass::gemm::detail::is_k_major<StrideB>()),
                "The unscaled element must be 2 bytes OR both inputs must be K-major");

  static_assert(cutlass::gemm::detail::is_mn_major<NonVoidStrideScale>(),
    "Scale must be MN major [Col Major if A is scaled, Row Major if B is scaled].");

  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;

  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using GmemTiledCopyScale = decltype(
    make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, NonVoidElementScale>{},
                    Layout<Shape <_16,_8>,
                           Stride< _8,_1>>{},
                    Layout<Shape < _8,_1>>{}));
  using GmemTiledCopyZero = decltype(
    make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, NonVoidElementZero>{},
                    Layout<Shape <_16,_8>,
                           Stride< _8,_1>>{},
                    Layout<Shape < _8,_1>>{}));

  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;

  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using SmemCopyAtomScale = Copy_Atom<cute::DefaultCopy, NonVoidElementScale>;

  // We must ensure the type to be scaled goes to RF
  static constexpr bool SwapAB = !IsATransformed;
  using InternalSmemLayoutAtomA = cute::conditional_t<!SwapAB, SmemLayoutAtomA, SmemLayoutAtomB>;
  using InternalSmemLayoutAtomB = cute::conditional_t<!SwapAB, SmemLayoutAtomB, SmemLayoutAtomA>;
  using InternalSmemCopyAtomA   = cute::conditional_t<!SwapAB, SmemCopyAtomA, SmemCopyAtomB>;
  using InternalSmemCopyAtomB   = cute::conditional_t<!SwapAB, SmemCopyAtomB, SmemCopyAtomA>;
  // TMA converts f32 input to tf32 when copying from GMEM to SMEM
  // For all other types, cast to size equivalent uint type to avoid any rounding by TMA.
  // static constexpr bool ConvertF32toTF32A = cute::is_same_v<float, ElementA>;
  // static constexpr bool ConvertF32toTF32B = cute::is_same_v<float, ElementB>;
  // using ConvertedElementA = cute::conditional_t<ConvertF32toTF32A, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementA>>>;
  // using ConvertedElementB = cute::conditional_t<ConvertF32toTF32B, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementB>>>;
  // using InternalElementA = cute::conditional_t<!SwapAB, ConvertedElementA, ConvertedElementB>;
  // using InternalElementB = cute::conditional_t<!SwapAB, ConvertedElementB, ConvertedElementA>;

  using RealInternalElementA = cute::conditional_t<!SwapAB, ElementA, ElementB>;
  using RealInternalElementB = cute::conditional_t<!SwapAB, ElementB, ElementA>;
  using InternalStrideA  = cute::conditional_t<!SwapAB, StrideA, StrideB>;
  using InternalStrideB  = cute::conditional_t<!SwapAB, StrideB, StrideA>;

  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using InternalTransformA  = cute::conditional_t<!SwapAB, TransformA, TransformB>;
  using InternalTransformB  = cute::conditional_t<!SwapAB, TransformB, TransformA>;
  using ArchTag = typename DispatchPolicy::ArchTag;

  using SmemLayoutAtomScale = Layout<Shape<decltype(cute::shape<0>(InternalSmemLayoutAtomA{})), cute::Int<1>>>;
  using ScaleTileShape = decltype(make_shape(shape<0>(TileShape{}), shape<1>(SmemLayoutAtomScale{})));

  static_assert(rank(InternalSmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(InternalSmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(InternalSmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(InternalSmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(InternalSmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(InternalSmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomScale{}) == 2, "SmemLayoutAtomScale must be rank 2");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomScale{})) == 0, "SmemLayoutAtomScale must equal the tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomScale{})) == 0, "SmemLayoutAtomScale must evenly divide tile k shape.");

  using SmemLayoutA = decltype(tile_to_shape(
      InternalSmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      InternalSmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));

  // It is assumed that the scales and zero-points share the same smem layout
  using SmemLayoutScale = decltype(tile_to_shape(
    SmemLayoutAtomScale{},
    make_shape(shape<0>(ScaleTileShape{}), shape<1>(ScaleTileShape{}), Int<Stages>{})));

  static_assert(DispatchPolicy::Stages >= 2, "CpAsync mainloop must have at least 2 stages in the pipeline.");

private:
  static constexpr ConversionMode
  get_conversion_mode() {
    if constexpr (cute::is_void_v<ElementScale>) {
      return ConversionMode::DirectConvert;
    }
    else if constexpr (cute::is_void_v<ElementZero>) {
      return ConversionMode::ConvertAndScale;
    }
    else {
      return ConversionMode::ConvertAndScaleWithZero;
    }
  }

  static constexpr ConversionMode KernelConversionMode = get_conversion_mode();
  static constexpr bool ModeHasScales = KernelConversionMode == ConversionMode::ConvertAndScale ||
                                        KernelConversionMode == ConversionMode::ConvertAndScaleWithZero;

  static constexpr auto
  elements_per_smem_scale() {
    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return 0;
    }
    else if constexpr (ModeHasScales) {
      return cute::cosize_v<SmemLayoutScale>;
    }
    else {
      // static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Type not handled in scale smem allocation.");
      assert(false);
    }
  }

  static constexpr auto
  elements_per_smem_zero() {
    if constexpr (KernelConversionMode == ConversionMode::DirectConvert ||
                  KernelConversionMode == ConversionMode::ConvertAndScale ) {
      return 0;
    }
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
      return cute::cosize_v<SmemLayoutScale>;
    }
    else {
      // static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Type not handled in scale smem allocation.");
      assert(false);
    }
  }

public:
  struct SharedStorage
  {
    static constexpr int scale_elements = elements_per_smem_scale();
    static constexpr int zero_elements = elements_per_smem_zero();
    cute::ArrayEngine<RealInternalElementA, cute::cosize_v<SmemLayoutA>> smem_a;
    cute::ArrayEngine<RealInternalElementB, cute::cosize_v<SmemLayoutB>> smem_b;
    cute::ArrayEngine<NonVoidElementScale, scale_elements> smem_scale;
    cute::ArrayEngine<NonVoidElementZero, zero_elements> smem_zero;
  };
  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A = nullptr;
    StrideA dA{};
    ElementB const* ptr_B = nullptr;
    StrideB dB{};
    ElementScale const* ptr_S = nullptr;
    NonVoidStrideScale dS{};
    int group_size = 0;
    ElementZero const* ptr_Z = nullptr;
  };

  // Device side kernel params
  struct Params {
    GmemTiledCopyScale gmem_tiled_copy_scale;
    GmemTiledCopyZero gmem_tiled_copy_zero;

    RealInternalElementA const* ptr_A = nullptr;
    InternalStrideA dA{};
    RealInternalElementB const* ptr_B = nullptr;
    InternalStrideB dB{};

    NonVoidElementScale const* ptr_S = nullptr;
    NonVoidElementZero const* ptr_Z = nullptr;

    int64_t scale_k = 0;
    int group_size = 0;
  };

  GmemTiledCopyA gmem_tiled_copy_A;
  GmemTiledCopyB gmem_tiled_copy_B;
  //
  // Methods
  //


  template <class ProblemShape>
  static Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    (void) workspace;

    // Optionally append 1s until problem shape is rank-4 (MNKL), in case it is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    if constexpr (SwapAB) {
      M = get<1>(problem_shape_MNKL);
      N = get<0>(problem_shape_MNKL);
    }

    Params p;
    if constexpr (not SwapAB) {
      p.ptr_A = reinterpret_cast<RealInternalElementA const*>(args.ptr_A);
      p.ptr_B = reinterpret_cast<RealInternalElementB const*>(args.ptr_B);
      p.dA = args.dA;
      p.dB = args.dB;
    }
    else {
      p.ptr_A = reinterpret_cast<RealInternalElementA const*>(args.ptr_B);
      p.ptr_B = reinterpret_cast<RealInternalElementB const*>(args.ptr_A);
      p.dA = args.dB;
      p.dB = args.dA;
    }

    if constexpr (ModeHasScales) {
      p.gmem_tiled_copy_scale = make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, NonVoidElementScale>{},
                    Layout<Shape <_16,_8>,
                           Stride< _8,_1>>{},
                    Layout<Shape < _8,_1>>{});
      p.ptr_S = reinterpret_cast<NonVoidElementScale const*>(args.ptr_S);
      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        p.gmem_tiled_copy_zero = make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, NonVoidElementZero>{},
                    Layout<Shape <_16,_8>,
                           Stride< _8,_1>>{},
                    Layout<Shape < _8,_1>>{});
        p.ptr_Z = reinterpret_cast<NonVoidElementZero const*>(args.ptr_Z);
      }
      p.group_size = args.group_size;
      p.scale_k = (K + args.group_size - 1) / args.group_size;
    }
    return p;
  }

  /// Set up the data needed by this collective for load and mma.
  /// Returns a tuple of tensors. The collective and the kernel layer have the contract.
  /// Returned tuple must contain at least two elements, with the first two elements being gA & gB.
  /// The rest of the tensors can be specified as needed by this collective.
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, Params const& mainloop_params) {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;

    if constexpr (SwapAB) {
      M = get<1>(problem_shape_MNKL);
      N = get<0>(problem_shape_MNKL);
    }
    // init aiu desc
    static constexpr bool TransA = is_static<decltype(get<1>(mainloop_params.dA))>::value ? false : true;
    static constexpr bool TransB = is_static<decltype(get<1>(mainloop_params.dB))>::value ? false : true;

    using TilerA = typename GmemTiledCopyA::Tiler_MN;
    using TilerB = typename GmemTiledCopyB::Tiler_MN;

    gmem_tiled_copy_A.desc_.init<RealInternalElementA, TransA, get<0>(TilerA{}), get<1>(TilerA{})>(nullptr, M, K, mainloop_params.dA);
    gmem_tiled_copy_B.desc_.init<RealInternalElementB, TransB, get<0>(TilerB{}), get<1>(TilerB{})>(nullptr, N, K, mainloop_params.dB);

    // Represent the full tensors
    Tensor mA_mkl = make_tensor(make_gmem_ptr(mainloop_params.ptr_A), make_shape(M,K,L), mainloop_params.dA); //(m,k,l)
    Tensor mB_nkl = make_tensor(make_gmem_ptr(mainloop_params.ptr_B), make_shape(N,K,L), mainloop_params.dB); //(n,k,l)

    auto [m_coord, n_coord, l_coord] = static_cast<uint3>(blockIdx);
    auto blk_coord_mnkl = make_coord(m_coord, n_coord, _, l_coord);                                        // (m,n,k,l)

    // Get batch slice
    Tensor mA_mk = make_mix_tensor_like(mA_mkl(_,_,l_coord));                                                  // (m,k)
    Tensor mB_nk = make_mix_tensor_like(mB_nkl(_,_,l_coord));                                                  // (n,k)

    // Slice to get the tiles this thread block is responsible for
    Tensor gA = local_tile(mA_mk, TileShape{}, take<0,3>(blk_coord_mnkl), Step<_1, X,_1>{});           // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB_nk, TileShape{}, take<0,3>(blk_coord_mnkl), Step< X,_1,_1>{});           // (BLK_N,BLK_K,k)

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return cute::make_tuple(gA, gB);
    }
    else if constexpr (ModeHasScales) {
      auto scale_k = mainloop_params.scale_k;
      Tensor mS_mkl = make_tensor(make_gmem_ptr(mainloop_params.ptr_S), make_shape(M,scale_k,L));         // (m,scale_k,l)
      Tensor gS_mkl = local_tile(mS_mkl, ScaleTileShape{}, make_coord(_,_));       // (BLK_M,BLK_Scale_K,m,scale_k,l)                                                              // (m,scale_k
      Tensor gS = gS_mkl(_,_,m_coord,_,l_coord);                                   // (BLK_M,BLK_K,k)
      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        return cute::make_tuple(gA, gB, gS);
      }
      else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        Tensor mZ_mkl = make_tensor(make_gmem_ptr(mainloop_params.ptr_Z), make_shape(M,scale_k,L));      // (m,scale_k,l)
        Tensor gZ_mkl = local_tile(mZ_mkl, ScaleTileShape{}, make_coord(_,_));      // (BLK_M,BLK_Scale_K,m,scale_k,l)
        Tensor gZ = gZ_mkl(_,_,m_coord,_,l_coord);
        return cute::make_tuple(gA, gB, gS, gZ);
      }
      else {
        // static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in load_init.");
        assert(false);
      }
    }
    else {
      // static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in load_init.");
      assert(false);
    }
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  template <
    class... Ts,
    class FrgTensorC,
    class KTileIterator
  >
  CUTLASS_DEVICE void
  operator() (
      Params const& mainloop_params,
      cute::tuple<Ts...> const& load_inputs,
      FrgTensorC &accum,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      char *smem_buf)
  {
    using namespace cute;

    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(rank(SmemLayoutA{}) == 3,
      "MainloopPPU0010CpAsync must have a pipeline mode in the smem layout.");
    static_assert(rank(SmemLayoutB{}) == 3,
      "MainloopPPU0010CpAsync must have a pipeline mode in the smem layout.");

    int warp_idx = canonical_warp_idx_sync();
    [[maybe_unused]] int warp_group_thread_idx = thread_idx % 128;
    int aiu_warp_group_thread_idx = warp_idx * 32;

    Tensor gA = get<0>(load_inputs);
    Tensor gB = get<1>(load_inputs);

    // Construct shared memory tiles
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.begin()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.begin()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

    // get extra inputs
    auto extra_input_partitions = partition_extra_inputs(mainloop_params, load_inputs, storage, thread_idx);

    CUTE_STATIC_ASSERT_V(size<0>(gA) == size<0>(sA));                          // BLK_M
    CUTE_STATIC_ASSERT_V(size<1>(gA) == size<1>(sA));                          // BLK_K
    CUTE_STATIC_ASSERT_V(size<0>(gB) == size<0>(sB));                          // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(gB) == size<1>(sB));                          // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(sA) == size<1>(sB));                          // BLK_K
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));        // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));        // PIPE

    // Partition the copying of A and B tiles across the threads
    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);

    Tensor tAgA = gmem_thr_copy_A.partition_S(gA);                             // (ACPY,ACPY_M,ACPY_K,k)
    Tensor tAsA = gmem_thr_copy_A.partition_D(sA);                             // (ACPY,ACPY_M,ACPY_K,PIPE)
    Tensor tBgB = gmem_thr_copy_B.partition_S(gB);                             // (BCPY,BCPY_N,BCPY_K,k)
    Tensor tBsB = gmem_thr_copy_B.partition_D(sB);                             // (BCPY,BCPY_N,BCPY_K,PIPE)

    // todo: ReloadFactor isn't constexpr, maybe cause perf issue.
    const int ReloadFactor = (mainloop_params.group_size + size<2>(TileShape{}) - 1) / size<2>(TileShape{});
    // Start async loads for all pipes but the last
    CUTLASS_PRAGMA_UNROLL
    for (int k_pipe = 0; k_pipe < DispatchPolicy::Stages-1; ++k_pipe) {
      copy_aiu(
        gmem_tiled_copy_A, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,k_pipe),
        gmem_tiled_copy_B, tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,k_pipe),
        warp_idx
      );
      if constexpr (ModeHasScales) {
        auto tSgS = get<0>(extra_input_partitions);
        auto tSsS = get<1>(extra_input_partitions);
        const int scale_load_k = *k_tile_iter / ReloadFactor; // This will always be 0 when group_size == K.
        copy(mainloop_params.gmem_tiled_copy_scale, tSgS(_,_,_,scale_load_k), tSsS(_,_,_,k_pipe));
        if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
          auto tZgZ = get<2>(extra_input_partitions);
          auto tZsZ = get<3>(extra_input_partitions);
          copy(mainloop_params.gmem_tiled_copy_zero, tZgZ(_,_,_,scale_load_k), tZsZ(_,_,_,k_pipe));
        }
      }
      cp_async_fence();
      --k_tile_count;
      if (k_tile_count > 0) { ++k_tile_iter; }
    }

    //
    // MMA Atom partitioning
    //

    // Tile MMA compute thread partitions and allocate accumulators
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCrA_mma = thr_mma.partition_fragment_A(sA(_,_,0));                 // (MMA,MMA_M,MMA_K)

    auto lshape = tCrA_mma.layout().shape();
    auto lstride = tCrA_mma.layout().stride();
    auto lshape0 = get<0>(lshape);
    auto lstride0 = get<0>(lstride);
    auto load_to_mma_layout = make_layout(make_shape(make_shape(get<0>(lshape0) * get<1>(lshape0), get<2>(lshape0)), get<1>(lshape), get<2>(lshape) / _2{}, _2{}),
                                          make_stride(make_stride(get<0>(lstride0), get<2>(lstride0)), get<1>(lstride), get<2>(lstride) * _2{}, get<2>(lstride)));
    Tensor tCrA_mma_copy = make_tensor(tCrA_mma.data(), load_to_mma_layout);

    Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));                     // (MMA,MMA_N,MMA_K)

    CUTE_STATIC_ASSERT_V(size<1>(tCrA_mma) == size<1>(accum));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA_mma) == size<2>(tCrB));                      // MMA_K

    //
    // Copy Atom retiling
    //

    using TiledMma_S8 = TiledMMA<
        MMA_Atom<PPU_16x16x32_S32S8S8S32_TN>,
        Layout<Shape<_2,_2,_1>>,  // 2x2x1 thread group
        Tile<_32, _32, _32>>; // 2x1x1 value group for 16x16x32 MMA and LDSM
    TiledMma_S8 tiled_mma_s8;
    auto thr_mma_s8 = tiled_mma_s8.get_thread_slice(thread_idx);
    auto sA_s8 = recast<int8_t>(sA);
    Tensor tCrA_load =thr_mma_s8.partition_fragment_A(sA_s8(_,_,0));

    static_assert(size<1>(tCrA_load) == size<1>(tCrA_mma_copy));
    static_assert(size<0,1>(tCrA_load) == size<3>(tCrA_mma_copy));
    static_assert(size<0,2>(tCrA_load) == size<0,1>(tCrA_mma_copy));

    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma_s8);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(aiu_warp_group_thread_idx);
    Tensor tCsA            = smem_thr_copy_A.partition_S(make_mix_tensor_like(sA_s8));                  // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA_load);              // (CPY,CPY_M,CPY_K)

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));            // CPY_K

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(aiu_warp_group_thread_idx);
    Tensor tCsB            = smem_thr_copy_B.partition_S(make_mix_tensor_like(sB));                  // (CPY,CPY_N,CPY_K,PIPE)
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);                   // (CPY,CPY_N,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));            // CPY_K

    // Compute the max vector length that can be used to copy A. This will match the vector width of the
    // conversions used. It helps by allowing the compiler to convert using the same register that was used
    // to load the data from smem. This significantly reduces the need to move data among registers.
    // Note that this is correct even if copy fails to vectorize, since the granularity at which we perform
    // the conversion does not impact correctness.

    // extra inputs partition and retile
    auto partitioned_extra_info = partition_extra_mma_info(tiled_mma, storage, warp_group_thread_idx);
    auto copy_partitions_extra_info = retile_extra_mma_info(tiled_mma, partitioned_extra_info, warp_group_thread_idx);

    //
    // PIPELINED MAIN LOOP
    //

    // Current pipe index in smem to read from
    int smem_pipe_read  = 0;
    // Current pipe index in smem to write to
    int smem_pipe_write = DispatchPolicy::Stages-1;

    Tensor tCsA_p = tCsA(_,_,_,smem_pipe_read);
    Tensor tCsB_p = tCsB(_,_,_,smem_pipe_read);

    // Size of the register pipeline
    auto K_BLOCK_MAX = size<2>(tCrA_copy_view);
    auto K_ATOM_PER_COPY = size<2>(tCrA_mma) / size<2>(tCrA_copy_view);

    // if (thread0()) {
    //   print("tCrA_copy_view:");          print(tCrA_copy_view);          print("\n");
    //   print("tCrA_mma:");                print(tCrA_mma);          print("\n");
    // }

    // PREFETCH register pipeline
    if (K_BLOCK_MAX > 1) {
      // Wait until our first prefetched tile is loaded in
      cp_async_wait<DispatchPolicy::Stages-2>();
      __syncthreads();
      // Prefetch the first rmem from the first k-tile
      copy_A_and_extra_info(smem_tiled_copy_A, tCsA, tCrA_copy_view,
          partitioned_extra_info, copy_partitions_extra_info, 0, smem_pipe_read);
      copy(smem_tiled_copy_B, tCsB_p(_,_,Int<0>{}), tCrB_copy_view(_,_,Int<0>{}));
      transform_A_kblock<RealInternalElementA>(tCrA_copy_view, tCrA_mma, partitioned_extra_info, 0, K_ATOM_PER_COPY);
    }

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > -(DispatchPolicy::Stages-1); --k_tile_count)
    {
      // Pipeline the outer products with a static for loop.
      //
      // Note, the for_each() function is required here to ensure `k_block` is of type Int<x>.
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block)
      {
        if (k_block == K_BLOCK_MAX - 1)
        {
          // Slice the smem_pipe_read smem
          tCsA_p = tCsA(_,_,_,smem_pipe_read);
          tCsB_p = tCsB(_,_,_,smem_pipe_read);

          // Commit the smem for smem_pipe_read
          cp_async_wait<DispatchPolicy::Stages-2>();
          __syncthreads();
        }

        // Load A, B shmem->regs for k_block+1
        auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;  // static
        copy_A_and_extra_info(smem_tiled_copy_A, tCsA, tCrA_copy_view,
          partitioned_extra_info, copy_partitions_extra_info, k_block_next, smem_pipe_read);
        copy(smem_tiled_copy_B, tCsB_p(_,_,k_block_next), tCrB_copy_view(_,_,k_block_next));
        transform_A_kblock<RealInternalElementA>(tCrA_copy_view, tCrA_mma, partitioned_extra_info, k_block_next, K_ATOM_PER_COPY);

        // Copy gmem to smem before computing gemm on each k-pipe
        if (k_block == 0)
        {
          copy_aiu(
            gmem_tiled_copy_A, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,smem_pipe_write),
            gmem_tiled_copy_B, tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,smem_pipe_write),
            warp_idx
          );
          if constexpr (ModeHasScales) {
            auto tSgS = get<0>(extra_input_partitions);
            auto tSsS = get<1>(extra_input_partitions);
            const int scale_load_k = *k_tile_iter / ReloadFactor; // This will always be 0 when group_size == K.
            // const int scale_load_k = 0;
            copy(mainloop_params.gmem_tiled_copy_scale, tSgS(_,_,_,scale_load_k), tSsS(_,_,_,smem_pipe_write));
            if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
              auto tZgZ = get<2>(extra_input_partitions);
              auto tZsZ = get<3>(extra_input_partitions);
              copy(mainloop_params.gmem_tiled_copy_zero, tZgZ(_,_,_,scale_load_k), tZsZ(_,_,_,smem_pipe_write));
            }
          }
          cp_async_fence();
          if (k_tile_count > 1) { ++k_tile_iter; }
          // Advance the pipe -- Doing it here accounts for K_BLOCK_MAX = 1 (no rmem pipe)
          smem_pipe_write = smem_pipe_read;
          ++smem_pipe_read;
          smem_pipe_read = (smem_pipe_read == DispatchPolicy::Stages) ? 0 : smem_pipe_read;
        }

        CUTLASS_PRAGMA_UNROLL
        for (int k_loop = 0; k_loop < K_ATOM_PER_COPY; k_loop++) {
          auto atom_idx = k_block * K_ATOM_PER_COPY + k_loop;
          // Transform before compute
          cute::transform(tCrA_mma(_,_,atom_idx), TransformA{});
          cute::transform(tCrB(_,_,atom_idx), TransformB{});
          // if (thread0()) {
          //   print("tCrA_mma(_,_,%d):", (int)atom_idx);          print_tensor(tCrA_mma(_,_,atom_idx));          print("\n");
          // }
          // gemm for one tiled_mma atom on K
          cute::gemm(tiled_mma, tCrA_mma(_,_,atom_idx), tCrB(_,_,atom_idx), accum);
        }
      });

    }

    // TODO: original cutlass3 miss this sync
    cp_async_wait<0>();
    __syncthreads();
  }

private:
  template <class... Ts>
  CUTLASS_DEVICE
  auto partition_extra_inputs(
        Params const& mainloop_params,
        cute::tuple<Ts...> const& load_inputs,
        SharedStorage& shared_tensors,
        int const thread_idx) {
    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return cute::tuple{};
    }
    else if constexpr (ModeHasScales) {
      Tensor sS = make_tensor(make_smem_ptr(shared_tensors.smem_scale.begin()), SmemLayoutScale{}); // (BLK_M,BLK_K,PIPE)
      Tensor gS = get<2>(load_inputs);

      auto gmem_thr_copy_scale = mainloop_params.gmem_tiled_copy_scale.get_slice(thread_idx);

      Tensor tSgS = gmem_thr_copy_scale.partition_S(gS);                         // (SCPY,SCPY_M,SCPY_K,k)
      Tensor tSsS = gmem_thr_copy_scale.partition_D(sS);                         // (SCPY,SCPY_M,SCPY_K,PIPE)

      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        return cute::make_tuple(tSgS, tSsS);
      }
      else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        Tensor sZ  = make_tensor(make_smem_ptr(shared_tensors.smem_zero.begin()), SmemLayoutScale{}); // (BLK_M,BLK_K,PIPE)
        Tensor gZ = get<3>(load_inputs);

        auto gmem_thr_copy_zero = mainloop_params.gmem_tiled_copy_zero.get_slice(thread_idx);

        Tensor tZgZ = gmem_thr_copy_zero.partition_S(gZ);                         // (ZCPY,ZCPY_M,ZCPY_K,k)
        Tensor tZsZ = gmem_thr_copy_zero.partition_D(sZ);                         // (ZCPY,ZCPY_M,ZCPY_K,PIPE)

        return cute::make_tuple(tSgS, tSsS, tZgZ, tZsZ);
      }
      else {
        // static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled for input partitioning.");
        assert(false);
      }
    }
    else {
      // static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled for input partitioning.");
      assert(false);
    }
  }

  /// Utilities for partitioning extra inputs for loading from smem in the mainloop.
  template <class TiledMma>
  CUTLASS_DEVICE
  auto partition_extra_mma_info(
    TiledMma const& tiled_mma,
    SharedStorage& storage,
    int thread_idx) {

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      // noting to do
      return cute::tuple{};
    }
    else if constexpr (ModeHasScales) {
      auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
      auto smem_tiled_copy_S   = make_tiled_copy_A(SmemCopyAtomScale{}, tiled_mma);
      auto smem_thr_copy_S     = smem_tiled_copy_S.get_thread_slice(thread_idx);

      Tensor sS   = make_tensor(make_smem_ptr(storage.smem_scale.begin()), SmemLayoutScale{});    // (BLK_M,BLK_SCALE_K,PIPE)
      Tensor tCsS = smem_thr_copy_S.partition_S(sS);                                              // (CPY,CPY_M,CPY_K,PIPE)
      Tensor tCrS = make_fragment_like<ElementScale>(thr_mma.partition_fragment_A(sS(_,_,Int<0>{})));

      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        return cute::make_tuple(tCsS, tCrS);
      }
      else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        Tensor sZ   = make_tensor(make_smem_ptr(storage.smem_zero.begin()), SmemLayoutScale{});  // (BLK_M,BLK_SCALE_K,PIPE)
        Tensor tCsZ = smem_thr_copy_S.partition_S(sZ);                                            // (CPY,CPY_M,CPY_K,PIPE)
        Tensor tCrZ = make_fragment_like<ElementScale>(thr_mma.partition_fragment_A(sZ(_,_,Int<0>{})));
        return cute::make_tuple(tCsS, tCrS, tCsZ, tCrZ);
      }
      else {
        // static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in A -> RF path.");
        assert(false);
      }
    }
    else {
      // static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in A -> RF path.");
      assert(false);
    }
  }

  /// Returns the tiled copy and copy views for the extra inputs.
  template <class TiledMma, class... Ts>
  CUTLASS_DEVICE
  auto retile_extra_mma_info(
    TiledMma const& tiled_mma,
    cute::tuple<Ts...>& partitioned_extra_info,
    int const warp_group_thread_idx) {

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      // noting to do
      return cute::tuple{};
    }
    else if constexpr (ModeHasScales) {
      auto smem_tiled_copy_S = make_tiled_copy_A(SmemCopyAtomScale{}, tiled_mma);
      auto smem_thr_copy_S   = smem_tiled_copy_S.get_thread_slice(warp_group_thread_idx);
      Tensor tCrS_copy_view  = smem_thr_copy_S.retile_D(cute::get<1>(partitioned_extra_info));        // (CPY,CPY_M,CPY_K)

      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        return cute::make_tuple(smem_tiled_copy_S, tCrS_copy_view);
      }
      else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        Tensor tCrZ_copy_view  = smem_thr_copy_S.retile_D(cute::get<3>(partitioned_extra_info));      // (CPY,CPY_M,CPY_K)
        return cute::make_tuple(smem_tiled_copy_S, tCrS_copy_view, tCrZ_copy_view);
      }
      else {
        // static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in A -> RF path.");
        assert(false);
      }
    }
    else {
      // static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in A -> RF path.");
      assert(false);
    }
  }

  /// Utilities to copy A and extra inputs from smem to RF
  template <class SmemTiledCopyA,
            class TensorASmemView,
            class TensorACopyView,
            class... Ts,
            class... Us
            >
  CUTLASS_DEVICE
  void copy_A_and_extra_info(
    SmemTiledCopyA const& smem_tiled_copy_A,
    TensorASmemView const& tCsA,
    TensorACopyView& tCrA_copy_view,
    cute::tuple<Ts...> const& partitioned_mma_extra_info,
    cute::tuple<Us...> const& tiled_copy_and_views,
    int k_block,
    int read_stage) {

    copy(smem_tiled_copy_A, tCsA(_,_,k_block,read_stage), tCrA_copy_view(_,_,k_block));
    if (k_block == 0) {
      // We are starting a new k-tile so copy the scale
      if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
        // nothing to do
      }
      else if constexpr (ModeHasScales) {
        auto smem_tiled_copy_S = cute::get<0>(tiled_copy_and_views);
        auto tCsS              = cute::get<0>(partitioned_mma_extra_info);
        auto tCrS_copy_view    = cute::get<1>(tiled_copy_and_views);
        copy(smem_tiled_copy_S, tCsS(_,_,k_block,read_stage), tCrS_copy_view(_,_,k_block));
        if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
          // Nothing extra to do
        } else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
          auto tCsZ              = cute::get<2>(partitioned_mma_extra_info);
          auto tCrZ_copy_view    = cute::get<2>(tiled_copy_and_views);
          copy(smem_tiled_copy_S, tCsZ(_,_,k_block,read_stage), tCrZ_copy_view(_,_,k_block));
        } else {
          // static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in A -> RF path.");
          assert(false);
        }
      }
      else {
        // static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in A -> RF path.");
        assert(false);
      }
    }
  }
  /// Utilities to transform A.
  template <typename RealInternalElementA,
            class TCrA_load,
            class TCrA_mma,
            class... Ts>
  CUTLASS_DEVICE
  void transform_A_kblock(
    TCrA_load const& tCrA_load,
    TCrA_mma& tCrA_mma,
    cute::tuple<Ts...> const& partitioned_extra_info,
    int const k_block,
    int const K_ATOM_PER_COPY) {

    Tensor cvt_in  = recast<RealInternalElementA>(tCrA_load(_, _, k_block));
    Tensor cvt_out = make_tensor(tCrA_mma(_, _, k_block * K_ATOM_PER_COPY).data(), cvt_in.layout());

    using A_CPY_VEC = Int<4 * 32 / sizeof_bits<RealInternalElementA>::value>;
    convert_tensor(cvt_in, cvt_out, A_CPY_VEC{});

    // if (thread0()) {
    //   print("tCrA_load(_, _, k_block):");            print_tensor(tCrA_load(_, _, k_block));  print("\n");
    //   print("tCrA_mma:");   print_tensor(tCrA_mma);  print("\n");
    // }

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      // do nothing
    }
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
      auto tCrS = cute::get<1>(partitioned_extra_info);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < K_ATOM_PER_COPY; ++i) {
        int atom_idx = k_block * K_ATOM_PER_COPY + i;
        cute::transform(tCrA_mma(_, _, atom_idx), tCrS(_, _, 0), tCrA_mma(_, _, atom_idx), cute::multiplies{});
      }
    }
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
      auto tCrS = cute::get<1>(partitioned_extra_info);
      auto tCrZ = cute::get<3>(partitioned_extra_info);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < K_ATOM_PER_COPY; ++i) {
        int atom_idx = k_block * K_ATOM_PER_COPY + i;
        cute::transform(tCrA_mma(_, _, atom_idx), tCrS(_, _, 0), tCrA_mma(_, _, atom_idx), cute::multiplies{});
        cute::transform(tCrA_mma(_, _, atom_idx), tCrZ(_, _, 0), tCrA_mma(_, _, atom_idx), cute::plus{});
      }
    }
    else {
      assert(false);
    //   static_assert(cutlass::detail::dependent_false<KernelSchedule>, "No A data is loaded.");
    }
  }

  /// Utilities for transforming the A operand prior to issuing tensorcore math.
  template <class EngineIn,
            class EngineOut,
            class TensorLayout,
            int ConversionVectorWidth = cosize_v<TensorLayout>>
  CUTLASS_DEVICE void
  convert_tensor(
    Tensor<EngineIn,TensorLayout> const& in,
    Tensor<EngineOut,TensorLayout>& out,
    cute::Int<ConversionVectorWidth> width = {}) {

    /// This is an element-wise conversion where we expect both tensors to have the same layout.
    /// As a result, we can cast as a cutlass array to use the fast numeric converters without
    /// worrying about indexing into the layout.
    constexpr int N = size(TensorLayout{});
    // constexpr int N = cosize_v<TensorLayout>;

    /// The inputs must be backed by registers & be statically sized.
    static_assert(is_rmem<EngineIn>::value, "Input tensor for A conversion must come from registers");
    static_assert(is_rmem<EngineOut>::value, "Output tensor for A conversion must come from registers");
    static_assert(is_static_v<TensorLayout>, "Tensor layout for the conversion must be static");
    // static_assert(cosize_v<TensorLayout> == size(TensorLayout{}), "Cosize and size of the layout must be equal.");
    static_assert(N % ConversionVectorWidth == 0, "Conversion vector width must divide cosize of the tensor layout.");

    using SrcType = typename EngineIn::value_type;
    using DstType = typename EngineOut::value_type;

    using SrcArray = cutlass::Array<SrcType, ConversionVectorWidth>;
    using DstArray = cutlass::Array<DstType, ConversionVectorWidth>;

    // constexpr cutlass::FloatRoundStyle RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
    // using Converter = cutlass::NumericArrayConverter<DstType, SrcType, ConversionVectorWidth, RoundStyle>;

    // SrcType int8_t consider as uint8_t
    using Converter = cutlass::MixGemmNumericArrayConverter<DstType, SrcType, ConversionVectorWidth>;

    constexpr int NumIterations = N / ConversionVectorWidth;

    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < NumIterations; ++ii) {
      SrcArray const* src_array_ptr = reinterpret_cast<SrcArray const*>(raw_pointer_cast(in(_, ii).data()));
      DstArray* dst_array_ptr = reinterpret_cast<DstArray*>(raw_pointer_cast(out(_, ii).data()));
      *dst_array_ptr = Converter::convert(*src_array_ptr);
    }
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
