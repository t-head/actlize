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

/*! \file
  \brief Functor performing elementwise operations used by epilogues.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"

#include "cutlass/epilogue/fusion/ppu_callbacks.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <class CopyAtom>
struct PpuEvtCopyAlignment : cute::Int<CopyAtom::NumValDst> {};

template <int Bits, class Element>
struct PpuEvtCopyAlignment<
    cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<Bits>, Element>>
    : cute::Int<Bits / cute::sizeof_bits_v<Element>> {
  static_assert(Bits > 0 && Bits % cute::sizeof_bits_v<Element> == 0,
                "Legacy EVT copy alignment must contain whole output elements.");
};

} // namespace detail

/// Applies an element wise operation to all elements within the fragment
/// and writes it out to destination storage.
///
/// Ways to generalize this:
/// - CTA tile shape
/// - vectorization requirements (GMEM)
/// - vectoriz(able) transform()
///
template <
  class StrideC_,
  class StrideD_,
  class FusionCallbacks_,
  class SmemLayout_,
  class CopyAtomR2S_,
  class TiledCopyS2R_,
  class CopyAtomR2G_,
  class EpilogueScheduleType_,
  class CopyAtomG2R_ = CopyAtomR2G_
>
class EpilogueEvt {
public:
  //
  // Type Aliases
  //
  // derived types of output thread level operator

  using SmemLayout   = SmemLayout_;
  using CopyAtomR2S  = CopyAtomR2S_;
  using TiledCopyS2R = TiledCopyS2R_;
  using CopyAtomR2G  = CopyAtomR2G_;
  using CopyAtomG2R  = CopyAtomG2R_;

  using ElementAccumulator =  typename CopyAtomR2S::ValType;
  using ElementCompute = ElementAccumulator;
  using ElementScalar = ElementCompute;
  using ElementOutput = typename CopyAtomR2G::ValType;
  using ElementC = typename CopyAtomG2R::ValType;
  static constexpr int kAlignmentD = detail::PpuEvtCopyAlignment<CopyAtomR2G>::value;
  static_assert(kAlignmentD > 0, "Legacy EVT output alignment must be positive.");
  using StrideC = StrideC_;
  using InternalStrideC = cute::remove_pointer_t<StrideC>;
  using ElementD = ElementOutput;
  using StrideD = StrideD_;
  using InternalStrideD = cute::remove_pointer_t<StrideD>;
  using FusionCallbacks = FusionCallbacks_;
  using FusionStorage = typename FusionCallbacks::SharedStorage;

  using ThreadEpilogueOp = typename epilogue::fusion::FusionCallbacksTraits<FusionCallbacks>::Operation;

  // only for build if GmemTiledCopyC or kAlignmentC is used
  using GmemTiledCopyC = void;
  using GmemTiledCopyD = void;

  static constexpr bool IsUsePtrArray = cute::is_same_v<EpilogueScheduleType_, EpiloguePtrArraySimtVectorized>;
  static constexpr bool IsGroupGemm = IsUsePtrArray and !cute::is_same_v<InternalStrideC, StrideC>;
  using PtrCType = cute::conditional_t<!IsUsePtrArray, ElementC const*, ElementC const**>;
  using PtrDType = cute::conditional_t<!IsUsePtrArray, ElementD*, ElementD**>;
  static_assert(rank(InternalStrideC{}) == 3, "StrideCD must be rank-3: [M, N, L]");
  static_assert(rank(InternalStrideD{}) == 3, "StrideCD must be rank-3: [M, N, L]");

  struct SharedStorage
  {
    cute::array_aligned<ElementAccumulator, cute::cosize_v<SmemLayout>> smem_epilogue;
    FusionStorage thread;
  };

  // Inspect the same S2R partition used by compute_epilogue. A row-major
  // scalar coordinate avoids repeated tuple evaluation in constexpr loops.
  // These helpers emit no device instructions and preserve Arguments/Params.
  static constexpr auto store_partition_layout() {
    using namespace cute;
    constexpr auto tile = make_shape(Int<size<0>(SmemLayout{})>{},
                                     Int<size<1>(SmemLayout{})>{});
    constexpr auto coordinates = make_counting_tensor(
        make_layout(tile, make_stride(Int<size<1>(SmemLayout{})>{}, Int<1>{})));
    constexpr auto divided = flat_divide(coordinates, tile);
    return TiledCopyS2R::tidfrg_D(divided.layout());
  }

  static constexpr bool is_store_shape_safe(int rows, int cols) {
    using namespace cute;
    if (rows <= 0 || cols <= 0) return false;
    constexpr int sm = size<0>(SmemLayout{}), sn = size<1>(SmemLayout{});
    constexpr auto partition = store_partition_layout();
    constexpr auto layout = coalesce(partition);
    constexpr int threads = size<0>(partition), fragment = size<1>(partition);
    constexpr int groups = size(partition) / (threads * fragment);
    // A proven aligned row vector cannot cross an aligned column boundary.
    // Keep the exact partition fallback for other fragment layouts and tails.
    if (is_row_vector_partition() && cols % kAlignmentD == 0) return true;
    const int rm = rows % sm == 0 ? sm : rows % sm;
    const int rn = cols % sn == 0 ? sn : cols % sn;
    for (int group = 0; group < groups; ++group)
      for (int tid = 0; tid < threads; ++tid) {
        const int base = tid + threads * fragment * group;
        const int first = layout(base);
        if (first / sn < rm && first % sn < rn)
          for (int v = 1; v < fragment; ++v) {
            const int current = layout(base + threads * v);
            if (current / sn >= rm || current % sn >= rn) return false;
          }
      }
    return true;
  }

  static constexpr bool is_row_vector_partition() {
    using namespace cute;
    constexpr int sm = size<0>(SmemLayout{}), sn = size<1>(SmemLayout{});
    constexpr auto partition = store_partition_layout();
    constexpr auto vector = coalesce(get<1>(partition));
    if constexpr (size(vector) != kAlignmentD || rank(vector) != 1) {
      return false;
    } else {
      if (sn % kAlignmentD || (kAlignmentD > 1 && stride<0>(vector) != 1)) return false;
      // Every non-vector layout basis must preserve vector alignment. Together
      // with the unit-stride vector mode and the tile bound, this proves all
      // fragments without constexpr work proportional to the tile area.
      constexpr auto bases = flatten(make_layout(remove<1>(partition.shape()),
                                                  remove<1>(partition.stride())));
      bool aligned = true;
      int maximum = 0;
      for_each(make_seq<rank(bases)>{}, [&](auto mode) {
        constexpr int i = decltype(mode)::value;
        constexpr int extent = size<i>(bases), step = stride<i>(bases);
        if (extent <= 0 || step < 0 || (extent > 1 && step % kAlignmentD)) aligned = false;
        maximum += (extent - 1) * step;
      });
      return aligned && maximum + kAlignmentD - 1 < sm * sn;
    }
  }

  // Host side epilogue arguments
  struct Arguments {
#ifndef __HGGCCC_RTC__
    typename FusionCallbacks::Arguments thread{};
#endif
    PtrCType ptr_C = nullptr;
    StrideC dC{};
    PtrDType ptr_D = nullptr;
    StrideD dD{};
  };

  // Device side epilogue params
  // different to Arguments, construct thread params in kernel
  // gemm_configs will always use EpilogueEvt
  // both aiu/cutlass gemm will force use rtc since evt params is hard to construct in static
  struct Params {
    typename FusionCallbacks::Params thread{};
    PtrCType ptr_C = nullptr;
    StrideC dC{};
    PtrDType ptr_D = nullptr;
    StrideD dD{};
  };

  //
  // Methods
  //

  template <class IndexType>
  static constexpr ElementC const*
  get_c_ptr(const Params& params, IndexType l, bool isSrcNeeded) {
    if (!isSrcNeeded) {
      return nullptr;
    }
    if constexpr (IsUsePtrArray) {
      return params.ptr_C[l];
    } else {
      return params.ptr_C;
    }
  }

  template <class IndexType>
  static constexpr ElementD*
  get_d_ptr(const Params& params, IndexType l) {
    if constexpr (IsUsePtrArray) {
      return params.ptr_D[l];
    } else {
      return params.ptr_D;
    }
  }

  template <class IndexType>
  static constexpr InternalStrideC
  get_stride_c(const Params& params, IndexType l) {
    if constexpr (IsGroupGemm) {
      return params.dC[l];
    } else {
      return params.dC;
    }
  }

  template <class IndexType>
  static constexpr InternalStrideD
  get_stride_d(const Params& params, IndexType l) {
    if constexpr (IsGroupGemm) {
      return params.dD[l];
    } else {
      return params.dD;
    }
  }

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      ProblemShape const& problem_shape,
      Arguments const& args,
      [[maybe_unused]] void* workspace) {
    Params params;
    params.ptr_C = args.ptr_C;
    params.dC = args.dC;
    params.ptr_D = args.ptr_D;
    params.dD = args.dD;

#ifndef __HGGCCC_RTC__
    params.thread = FusionCallbacks::to_underlying_arguments(problem_shape, args.thread, workspace);
#endif
    return params;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args, int cu_count=0) {
#ifndef __HGGCCC_RTC__
    return FusionCallbacks::get_workspace_size(problem_shape, args.thread);
#else
    return 0;
#endif
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, hggcStream_t stream,
    HostAdapter* host_adapter = nullptr) {
    return FusionCallbacks::initialize_workspace(problem_shape, args.thread, workspace, stream, host_adapter);
  }

  template <class ProblemShape>
  CUTLASS_HOST_DEVICE static bool
  can_implement(
      [[maybe_unused]] ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  EpilogueEvt(Params const& params_, SharedStorage& shared_tensors)
      : params(params_), fusion_callbacks(params_.thread, shared_tensors.thread) {}

  CUTLASS_DEVICE
  bool
  is_source_needed() {
    return fusion_callbacks.is_C_load_needed();
  }

  template<
    class ProblemShapeMNKL,
    class BlockShapeMNK,
    class BlockCoordMNKL,
    class FrgEngine, class FrgLayout,
    class TiledMma,
    class ResidueMNK
  >
  CUTLASS_DEVICE void
  operator()(
      ProblemShapeMNKL problem_shape_mnkl,
      BlockShapeMNK blk_shape_MNK,
      BlockCoordMNKL blk_coord_mnkl,
      cute::Tensor<FrgEngine,FrgLayout> const& accumulators,                   // (MMA,MMA_M,MMA_N)
      TiledMma tiled_mma,
      ResidueMNK residue_mnk,
      int thread_idx,
      char* smem_buf)
  {
    if (__builtin_expect(is_source_needed(), 0)) {
      compute_epilogue<ProblemShapeMNKL, BlockShapeMNK, BlockCoordMNKL, FrgEngine, FrgLayout, TiledMma, ResidueMNK, true>(
        problem_shape_mnkl, blk_shape_MNK, blk_coord_mnkl, accumulators, tiled_mma, residue_mnk, thread_idx, smem_buf);
    } else {
      compute_epilogue<ProblemShapeMNKL, BlockShapeMNK, BlockCoordMNKL, FrgEngine, FrgLayout, TiledMma, ResidueMNK, false>(
        problem_shape_mnkl, blk_shape_MNK, blk_coord_mnkl, accumulators, tiled_mma, residue_mnk, thread_idx, smem_buf);
    }
  }

  template<
    class ProblemShapeMNKL,
    class BlockShapeMNK,
    class BlockCoordMNKL,
    class FrgEngine, class FrgLayout,
    class TiledMma,
    class ResidueMNK,
    bool isSrcNeeded
  >
  CUTLASS_DEVICE void
  compute_epilogue(
      ProblemShapeMNKL problem_shape_mnkl,
      BlockShapeMNK blk_shape_MNK,
      BlockCoordMNKL blk_coord_mnkl,
      cute::Tensor<FrgEngine,FrgLayout> const& accumulators,                   // (MMA,MMA_M,MMA_N)
      TiledMma tiled_mma,
      ResidueMNK residue_mnk,
      int thread_idx,
      char* smem_buf)
  {
    using namespace cute;
    using X = Underscore;

    static_assert(rank(ProblemShapeMNKL{}) == 4, "ProblemShapeMNKL must be rank 4");
    static_assert(is_static<BlockShapeMNK>::value, "ThreadBlock tile shape must be static");
    static_assert(rank(BlockShapeMNK{}) == 3, "BlockShapeMNK must be rank 3");
    static_assert(rank(BlockCoordMNKL{}) == 4, "BlockCoordMNKL must be rank 3");

    // synchronizing function for smem reads/writes
#if PPU_BARRIER_ENABLED
    auto synchronize = [] () { cutlass::arch::NamedBarrier::sync(typename TiledCopyS2R::TiledNumThr{}, 0); };
#else
    auto synchronize = [] () { __syncthreads(); };
#endif
    // Kernel level shared memory storage
    SharedStorage& shared_tensors = *reinterpret_cast<SharedStorage*>(smem_buf);

    // Separate out problem shape for convenience
    auto M = get<0>(problem_shape_mnkl);
    auto N = get<1>(problem_shape_mnkl);
    auto L = get<3>(problem_shape_mnkl);
    // Batches are managed by using appropriate pointers to C and D matrices
    const int32_t mock_L = IsUsePtrArray ? 1 : L;

    // Slice to get the tile this CTA is responsible for
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord_mnkl;
    const int32_t mock_l_coord = IsUsePtrArray ? 0 : l_coord;
    // Represent the full output tensor
    Tensor mC_mnl = make_tensor(make_gmem_ptr(get_c_ptr(params, l_coord, isSrcNeeded)), make_shape(M,N,mock_L), get_stride_c(params, l_coord));      //             (m,n,l)
    Tensor mD_mnl = make_tensor(make_gmem_ptr(get_d_ptr(params, l_coord)), make_shape(M,N,mock_L), get_stride_d(params, l_coord));      //             (m,n,l)
    Tensor gC_mnl = local_tile(mC_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});      // (BLK_M,BLK_N,m,n,l)
    Tensor gD_mnl = local_tile(mD_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});      // (BLK_M,BLK_N,m,n,l)
    Tensor gC = gC_mnl(_,_,m_coord,n_coord,mock_l_coord);                                                   // (BLK_M,BLK_N)
    Tensor gD = gD_mnl(_,_,m_coord,n_coord,mock_l_coord);                                                   // (BLK_M,BLK_N)

    // Construct a tensor in SMEM that we can partition for rearranging data
    Tensor sC = make_tensor(make_smem_ptr(shared_tensors.smem_epilogue.data()), SmemLayout{});              // (SMEM_M,SMEM_N)

    // Partition sC to match the accumulator partitioning
    auto tiled_r2s = make_tiled_copy_C(CopyAtomR2S{}, tiled_mma);
    auto thread_r2s = tiled_r2s.get_slice(thread_idx);
    auto tC     = tiled_r2s.get_thread_slice(thread_idx);
    Tensor tCaC = tC.retile_S(accumulators);                                          // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor tCsC = tC.partition_D(sC);                                                 // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // Tile gD and gC by the shape of SmemLayout first
    auto tile  = make_shape(size<0>(sC), size<1>(sC));
    Tensor gCt = flat_divide(gC, tile);                                                // (SMEM_M,SMEM_N,TILE_M,TILE_N)
    Tensor gDt = flat_divide(gD, tile);                                                // (SMEM_M,SMEM_N,TILE_M,TILE_N)

    // Partition sC, gC, and gD for the output
    auto tiled_s2r = TiledCopyS2R{};
    auto tD     = tiled_s2r.get_thread_slice(thread_idx);
    Tensor tDsC = tD.partition_S(sC);                                   //               ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tDgC = tD.partition_D(gCt);                                  // ((Atom,AtomNum),ATOM_M,ATOM_N,TILE_M,TILE_N)
    Tensor tDgD = tD.partition_D(gDt);                                  // ((Atom,AtomNum),ATOM_M,ATOM_N,TILE_M,TILE_N)

    // Allocate intermediate registers on the dst tensors
    Tensor tDrAcc = make_tensor<ElementAccumulator>(take<0,3>(shape(tDgC)));            // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tDrD = make_tensor<ElementOutput>(shape(tDrAcc));                            // ((Atom,AtomNum),ATOM_M,ATOM_N)

    // Repeat the D-partitioning for coordinates and predication
    Tensor cD   = make_identity_tensor(make_shape(size<0>(gD),size<1>(gD)));          // (BLK_M,BLK_N) -> (blk_m,blk_n)
    Tensor cDt  = flat_divide(cD, tile);                                //                (SMEM_M,SMEM_N,TILE_M,TILE_N)
    Tensor tDcD = tD.partition_D(cDt);                                  // ((Atom,AtomNum),ATOM_M,ATOM_N,TILE_M,TILE_N)

    CUTE_STATIC_ASSERT(size<1>(tCaC) % size<3>(tDgC) == 0);  // TILE_M divides MMA_M
    CUTE_STATIC_ASSERT(size<2>(tCaC) % size<4>(tDgC) == 0);  // TILE_N divides MMA_N
    // TiledCopyS2R::TiledNumThr is block size, size<0>(typename TiledMma::AtomLayoutC_TV{}) is warp size, why should be equal?????
    // CUTE_STATIC_ASSERT(typename TiledCopyS2R::TiledNumThr{} == size<0>(typename TiledMma::AtomLayoutC_TV{}));

    constexpr int EvtFragSize = size(tDrAcc(_,0,0));
    static_assert(EvtFragSize > 0 && EvtFragSize % kAlignmentD == 0,
                  "Legacy EVT fragment width must match output alignment.");
    // Dynamic dimensions still require a caller-side eligibility check.
    constexpr int StaticM = is_static<decltype(M)>::value ? decltype(M){} : size<0>(SmemLayout{});
    constexpr int StaticN = is_static<decltype(N)>::value ? decltype(N){} : size<1>(SmemLayout{});
    static_assert((!is_static<decltype(M)>::value && !is_static<decltype(N)>::value) ||
                  is_store_shape_safe(StaticM, StaticN),
                  "Legacy EVT static tail does not match its writeback fragment alignment.");
    // frag for one epilogue outer loop
    Tensor evt_c_tensor = make_tensor<ElementC>(take<0,3>(shape(tDgC)));

    // OOB predication for tile quantization "residue"
    // Absolute coordinate tensors (dynamic)
    Tensor mD_crd = make_identity_tensor(make_shape(M,N));                                                     // (M,N)
    Tensor cD_mn = local_tile(mD_crd, take<0,2>(blk_shape_MNK), make_coord(m_coord, n_coord));          // (CTA_M,CTA_N)
    using EpilogueTile = decltype(shape(coalesce(make_layout(shape(SmemLayout{})), Step<_1, _1>{})));
    // The visitor partitions its gmem operands with `tiled_s2r`, 
    // so the coordinate tensor handed to it must use the same thread/value map. 
    Tensor tCcD_mn = tD.partition_S(flat_divide(cD_mn, EpilogueTile{}));              // (S2R,S2R_M,S2R_N,EPI_M,EPI_N)

    // Relative coordinate tensors (static)
    Tensor cD_ = make_counting_tensor(cD_mn.layout());                                                  // (CTA_M,CTA_N)
    Tensor tCcD = make_counting_tensor(tCcD_mn.layout());                              // (S2R,S2R_M,S2R_N,EPI_M,EPI_N)
    // Subtract the global "bottom right" corner from the local "top left" corner to get the max relative coordinate
    auto residue_cD = make_coord(M,N) - cD_mn(_0{});                                                           // (m,n)
    auto residue_tCcD = make_coord(M,N) - tCcD_mn(_0{});

#if SAIL_EPILOGUE_OPT >= 2
    using MmaAccType = typename FrgEngine::value_type;
    static const bool do_epilogue_opt = !cute::is_same_v<MmaAccType, ElementAccumulator>;
    using ArrayType = cutlass::Array<MmaAccType, decltype(cute::size<0>(tCsC))::value>;
    cutlass::NumericArrayConverter<ElementD, MmaAccType, decltype(size<0>(tCsC))::value, FloatRoundStyle::round_to_nearest> converter;
#endif

    auto cst_args = cutlass::epilogue::fusion::detail::ConsumerStoreArgs{
                      problem_shape_mnkl,
                      blk_shape_MNK,
                      blk_coord_mnkl,
                      tiled_mma,
                      EpilogueTile{},
                      tiled_s2r,
                      cD,
                      residue_cD,
                      tCcD,
                      residue_tCcD,
                      evt_c_tensor,
                      thread_idx
                    };
    auto cst_callbacks = fusion_callbacks.template get_consumer_store_callbacks<true>(cst_args);

    // Pre-loop fusion callback entry point
    cst_callbacks.begin();

    // Check https://github.com/ColfaxResearch/cfx-article-src/blob/master/evt/node_types.md for various nodes(very helpful)
    // For each tiling needed for SmemLayout to cover shape(gD)
    CUTLASS_PRAGMA_UNROLL
    for (int step_m = 0; step_m < size<2>(cDt); ++step_m)
    {
      CUTLASS_PRAGMA_UNROLL
      for (int step_n = 0; step_n < size<3>(cDt); ++step_n)
      {
        bool is_last_iteration = step_m == size<2>(cDt)-1 && step_n == size<3>(cDt)-1;
        // Step 1. Copy to SMEM
        CUTLASS_PRAGMA_UNROLL
        for (int pipe_m = 0; pipe_m < size<1>(tCsC); ++pipe_m) {
          CUTLASS_PRAGMA_UNROLL
          for (int pipe_n = 0; pipe_n < size<2>(tCsC); ++pipe_n) {
            int mma_m = step_m * size<1>(tCsC) + pipe_m;
            int mma_n = step_n * size<2>(tCsC) + pipe_n;
#if SAIL_EPILOGUE_OPT >= 2
            if (do_epilogue_opt) {
              const auto result = converter(*reinterpret_cast<const ArrayType *>(tCaC(_,mma_m,mma_n).data()));
              Tensor tCaC_convert = make_tensor(result.data(), tCaC(_,mma_m,mma_n).layout());
              copy(tiled_r2s, tCaC_convert, tCsC(_,pipe_m,pipe_n));
            } else {
              copy(tiled_r2s, tCaC(_,mma_m,mma_n), tCsC(_,pipe_m,pipe_n));
            }
#else
            copy(tiled_r2s, tCaC(_,mma_m,mma_n), tCsC(_,pipe_m,pipe_n));
#endif
          }
        }

        // Step 2. Wait for SMEM writes to complete
        synchronize();

        // Step 3. Copy from SMEM into a fragment
        copy(tiled_s2r, tDsC, tDrAcc);

        // Step 4. Wait for SMEM reads to complete
        synchronize();

        Tensor tDgDmn = tDgD(_,_,_,step_m,step_n);
        Tensor tDcDmn = tDcD(_,_,_,step_m,step_n);

        if (isSrcNeeded) {
          CUTLASS_PRAGMA_UNROLL
          for (int m = 0; m < size<1>(tDgDmn); ++m)
          {
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < size<2>(tDgDmn); ++n)
            {
              if (elem_less(tDcDmn(0,m,n), take<0,2>(residue_mnk))) {
                // CopyAtomR2G is Vector Copy which can also used for G2R
                // copy(CopyAtomR2G{}, tDgC(_,m,n,step_m,step_n), evt_c_tensor(_,m,n));
                copy_g2r_ldcs(CopyAtomG2R{}, tDgC(_,m,n,step_m,step_n), evt_c_tensor(_,m,n));
              }
            }
          }

          // follow original epilogue logic, transfer part of global tensor to epilogue_op
          // read to vreg in each sub loop, avoid alloc vreg for the whole subloop
          // evt_c_tensor = tDgC(_,_,_,step_m,step_n);
        }

        cst_callbacks.begin_loop(step_m, step_n);

        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < size<1>(tDgDmn); ++m)
        {
          CUTLASS_PRAGMA_UNROLL
          for (int n = 0; n < size<2>(tDgDmn); ++n)
          {
            int sub_loop_idx = m * size<2>(tDgDmn) + n;
            // Predication
            if (get<0>(tDcDmn(0,m,n)) < get<0>(residue_mnk) &&
                get<1>(tDcDmn(0,m,n)) < get<1>(residue_mnk))
            {
              // run evt
              Tensor tDrAcc_array = recast<Array<ElementAccumulator, EvtFragSize>>(tDrAcc(_,m,n));
              Tensor tDrD_array = recast<Array<ElementOutput, EvtFragSize>>(tDrD(_,m,n));
              // run output op for all regs in cur epi loop
              // corresponds to multiple vmem.st
              tDrD_array[0] = cst_callbacks.visit(tDrAcc_array[0], sub_loop_idx, step_m, step_n);
              copy(CopyAtomR2G{}, tDrD(_,m,n), tDgDmn(_,m,n));
            }
          }
        }

        // Smem reduction callback entry point using shared memory for workspace
        // cst_callbacks.reduce(tDsC, synchronize, step_m, step_n, is_last_iteration, tDrD);
        cst_callbacks.reduce(sC, synchronize, step_m, step_n, is_last_iteration, tDrD);


        cst_callbacks.end_loop(step_m, step_n);
      }
    }

    cst_callbacks.end();
  }

private:
  Params params;
  FusionCallbacks fusion_callbacks;
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
