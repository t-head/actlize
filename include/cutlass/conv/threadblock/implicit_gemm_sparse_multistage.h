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
    \brief Template for a multistage threadblock-scoped Implicit GEMM Convolution kernel.
*/

#pragma once

#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/cache_operation.h"
#include "cutlass/gemm/threadblock/mma_sparse_base.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting alu cores and SIMT math
/// instructions.
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Iterates over tiles of A operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorA_,
    /// Iterates over tiles of A operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorA_,
    /// Cache operation for operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Iterates over tiles of B operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorB_,
    /// Iterates over tiles of B operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorB_,
    /// Cache operation for operand B
    cutlass::arch::CacheOperation::Kind CacheOpB,
    /// Iterates over tiles of E operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorE_,
    /// Iterates over tiles of E operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorE_,
    /// Cache operation for operand E
    cutlass::arch::CacheOperation::Kind CacheOpE,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// Number of stages,
    int Stages,
    /// Used for partial specialization
    typename Enable = bool
#ifdef SAIL_CUSTOMIZE_CUTLASS
  /// Sparse is used for compression matrix.
  , gemm::Operand CompressOp_ = gemm::Operand::kA
#endif
>
class SparseImplicitGemmMultistage :
#ifdef SAIL_CUSTOMIZE_CUTLASS
  public gemm::threadblock::SparseMmaBase<Shape_, Policy_, Stages, Enable, CompressOp_> {
public:
  ///< Base class
  using Base = gemm::threadblock::SparseMmaBase<Shape_, Policy_, Stages, Enable, CompressOp_>;
#else
  public gemm::threadblock::SparseMmaBase<Shape_, Policy_, Stages, Enable> {
public:
  ///< Base class
  using Base = gemm::threadblock::SparseMmaBase<Shape_, Policy_, Stages, Enable>;
#endif
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;
  ///< Iterates over tiles of A operand in global memory
  using IteratorA = IteratorA_;
  ///< Iterates over tiles of B operand in global memory
  using IteratorB = IteratorB_;
  ///< Iterates over tiles of E operand in global memory
  using IteratorE = IteratorE_;
  ///< Policy describing tuning details
  using Policy = Policy_;

  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;
  using SmemIteratorE = SmemIteratorE_;

  static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;
  static cutlass::arch::CacheOperation::Kind const kCacheOpE = CacheOpE;

  static int const kSparse = Policy::Operator::kSparse;
  static int const kMetaSizeInBits = Policy::Operator::kMetaSizeInBits;
  static int const kMaxID2 = Policy::Operator::kMaxID2;
  static int const kElementsPerElementE =
      Policy::Operator::kElementsPerElementE;

  //
  // Dependent types
  //

  /// Fragment of accumulator tile

  using ElementC = typename Policy::Operator::ElementC;
  using FragmentC = typename Policy::Operator::FragmentC;

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  /// ElementE
  using ElementE = typename IteratorE::Element;

  /// LayoutE
  using LayoutE = typename IteratorE::Layout;

  /// Minimum architecture is PPU1.0 to support cp.async
  using ArchTag = arch::PPU0010;

  /// Internal structure exposed for introspection.
  struct Detail {

    static_assert(Base::kWarpGemmIterations > 1,
                  "The pipelined structure requires at least two warp-level "
                  "GEMM operations.");

    /// Number of cp.async instructions to load one stage of operand A
    static int const AsyncCopyIterationsPerStageA =
        IteratorA::ThreadMap::Iterations::kCount;

    /// Number of cp.async instructions to load one stage of operand B
    static int const AsyncCopyIterationsPerStageB =
        IteratorB::ThreadMap::Iterations::kCount;

    /// Number of cp.async instructions to load one stage of operand E
    static int const AsyncCopyIterationsPerStageE =
        IteratorE::ThreadMap::Iterations::kCount;

    /// Number of stages
    static int const kStages = Stages;

    /// Number of cp.async instructions to load on group of operand A
    static int const kAccessesPerGroupA =
        (AsyncCopyIterationsPerStageA + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;

    /// Number of cp.async instructions to load on group of operand B
    static int const kAccessesPerGroupB =
        (AsyncCopyIterationsPerStageB + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;

    /// Number of cp.async instructions to load on group of operand E
    static int const kAccessesPerGroupE =
        (AsyncCopyIterationsPerStageE + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;

    /// B is compressed, some block/warp shape may need smaller warp to load B from global.
    static int const kBValidWarps = IteratorB::ThreadMap::kThreads / 32;

    /// E operand is tiny.  For the most of time, not all the warps are needed
    /// to load it from the global memory.
    static int const kValidWarps = IteratorE::ThreadMap::kThreads / 32;

    /// A operand is twice as big as B which brings very high register pressure.
    /// We have to sacrifice the double buffer when the warp tile size is big.
    static int const kABufferSize =
    ((sizeof(typename Operator::ElementC) == 4) &&
      ((platform::is_same<typename Operator::Policy::Operator::ElementA,
                          typename Operator::ElementA>::value &&
        platform::is_same<typename Operator::Policy::Operator::ElementB,
                          typename Operator::ElementB>::value)) &&
      (Operator::Shape::kM >= 64 && Operator::Shape::kN >= 64 &&
        Operator::ArchMmaOperator::Shape::kK == 32))
        ? 1
        : 2;

    static int const kBBufferSize =
    ((sizeof(typename Operator::ElementC) == 4) &&
      ((platform::is_same<typename Operator::Policy::Operator::ElementA,
                          typename Operator::ElementA>::value &&
        platform::is_same<typename Operator::Policy::Operator::ElementB,
                          typename Operator::ElementB>::value)) &&
      (Operator::Shape::kM == 32 && Operator::Shape::kN >= 64 && Operator::Shape::kK == 32))
        ? 1
        : 2;
  };

  private:

    using WarpLoadedFragmentA = typename Operator::FragmentA;
    using WarpLoadedFragmentB = typename Operator::FragmentB;
    using WarpTransformedFragmentA = typename Operator::TransformedFragmentA;
    using WarpTransformedFragmentB = typename Operator::TransformedFragmentB;
    using WarpFragmentE = typename Operator::FragmentE;

 private:

  //
  // Data members
  //

  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA smem_iterator_A_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB smem_iterator_B_;

  /// Iterator to write threadblock-scoped tile of E operand to shared memory
  SmemIteratorE smem_iterator_E_;

  /// Warp id
  bool is_warp_valid_;

  bool is_warp_valid_b_;

public:

  /// Construct from tensor references
  CUTLASS_DEVICE
  SparseImplicitGemmMultistage(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      typename Base::SharedStorage &shared_storage,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx
    ):
      Base(shared_storage, thread_idx, warp_idx, lane_idx),
      smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
      smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx),
      smem_iterator_E_(shared_storage.operand_E_ref(), thread_idx) {

    is_warp_valid_ = warp_idx < Detail::kValidWarps;
    is_warp_valid_b_ = warp_idx < Detail::kBValidWarps;

    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension

    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

    // Add per-warp offsets in units of warp-level tiles
    this->warp_tile_iterator_A_.add_tile_offset(
        {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B_.add_tile_offset(
        {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
#ifdef SAIL_CUSTOMIZE_CUTLASS
    this->warp_tile_iterator_E_.add_tile_offset(
        (CompressOp_ == gemm::Operand::kB)
        ? make_Coord(Base::kWarpGemmIterations * warp_idx_k, warp_idx_n)
        : make_Coord(warp_idx_m, Base::kWarpGemmIterations * warp_idx_k));
#else
    this->warp_tile_iterator_E_.add_tile_offset(
        {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
#endif
  }

  CUTLASS_DEVICE
  void copy_tiles_and_advance(
    IteratorA &iterator_A, IteratorB &iterator_B,
    IteratorE &iterator_E, int group_start_A = 0,
    int group_start_B = 0, int group_start_E = 0) {

    iterator_A.set_iteration_index(group_start_A);
    this->smem_iterator_A_.set_iteration_index(group_start_A);

    // Async Copy for operand A
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupA; ++j) {
      if (group_start_A + j < Detail::AsyncCopyIterationsPerStageA) {
        typename IteratorA::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType *>(
                this->smem_iterator_A_.get());

        int const kSrcBytes = sizeof_bits<typename IteratorA::Element>::value *
                              IteratorA::ThreadMap::kElementsPerAccess / 8;

        cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(
                dst_ptr, iterator_A.get(), iterator_A.valid());

        ++iterator_A;

        ++this->smem_iterator_A_;
      }
    }

    iterator_B.set_iteration_index(group_start_B);

    this->smem_iterator_B_.set_iteration_index(group_start_B);

    // Async Copy for operand B
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupB; ++j) {
      if (group_start_B + j < Detail::AsyncCopyIterationsPerStageB) {
        typename IteratorB::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType *>(
                this->smem_iterator_B_.get());

        int const kSrcBytes = sizeof_bits<typename IteratorB::Element>::value *
                              IteratorB::ThreadMap::kElementsPerAccess / 8;
        if (is_warp_valid_b_)
          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(
                  dst_ptr, iterator_B.get(), iterator_B.valid());

        ++iterator_B;
        ++this->smem_iterator_B_;
      }
    }

    iterator_E.set_iteration_index(group_start_E);
    this->smem_iterator_E_.set_iteration_index(group_start_E);

    // async copy for operand E
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupE; ++j) {
      if (group_start_E + j < Detail::AsyncCopyIterationsPerStageE) {
        typename IteratorE::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorE::AccessType *>(
                this->smem_iterator_E_.get());

        int const kSrcBytes = sizeof_bits<typename IteratorE::Element>::value *
                              IteratorE::ThreadMap::kElementsPerAccess / 8;

        auto gmem_ptr = iterator_E.get();

        if (is_warp_valid_)
          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpE>(
              dst_ptr, gmem_ptr, iterator_E.valid() && is_warp_valid_);

        ++iterator_E;
        ++this->smem_iterator_E_;
      }
    }
  }

  /// Perform a threadblock-scoped matrix multiply-accumulate
  CUTLASS_DEVICE
  void operator()(
      ///< problem size of GEMM
      int gemm_k_iterations,
      ///< destination accumulator tile
      FragmentC &accum,
      ///< iterator over A operand in global memory
      IteratorA iterator_A,
      ///< iterator over B operand in global memory
      IteratorB iterator_B,
      ///< iterator over E operand in global memory
      IteratorE iterator_E,
      ///< initial value of accumulator
      FragmentC const &src_accum,
      ///< Imaginary strides used for planar-complex only - ignored here
      int64_t imag_stride_A = 0,
      int64_t imag_stride_B = 0) {

    //
    // Prologue
    //
#ifdef SAIL_CUSTOMIZE_CUTLASS
      auto k_sparse_tile = ((CompressOp_ == gemm::Operand::kB)) ? make_Coord(1, 0) : make_Coord(0, 1);
#endif

    // Issue several complete stages
    CUTLASS_PRAGMA_UNROLL
    for (int stage = 0; stage < Base::kStages - 1;
         ++stage, --gemm_k_iterations) {

      iterator_A.set_iteration_index(0);
      this->smem_iterator_A_.set_iteration_index(0);

      // Async Copy for operand A
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
        typename IteratorA::AccessType *dst_ptr =
          reinterpret_cast<typename IteratorA::AccessType *>(
            this->smem_iterator_A_.get());

        int const kSrcBytes =
            sizeof_bits<typename IteratorA::Element>::value *
            IteratorA::ThreadMap::kElementsPerAccess / 8;

        cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(
          dst_ptr, iterator_A.get(), iterator_A.valid());

        ++iterator_A;
        ++this->smem_iterator_A_;
      }

      iterator_B.set_iteration_index(0);
      this->smem_iterator_B_.set_iteration_index(0);

      // Async Copy for operand B
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
        typename IteratorB::AccessType *dst_ptr =
          reinterpret_cast<typename IteratorB::AccessType *>(
              this->smem_iterator_B_.get());

        int const kSrcBytes =
            sizeof_bits<typename IteratorB::Element>::value *
            IteratorB::ThreadMap::kElementsPerAccess / 8;

        if (is_warp_valid_b_)
          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(
              dst_ptr, iterator_B.get(), iterator_B.valid());

        ++iterator_B;
        ++this->smem_iterator_B_;
      }

      iterator_E.set_iteration_index(0);
      this->smem_iterator_E_.set_iteration_index(0);

      // async copy for operand E
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageE; ++j) {
        typename IteratorE::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorE::AccessType *>(
                this->smem_iterator_E_.get());

        int const kSrcBytes = sizeof_bits<typename IteratorE::Element>::value *
                              IteratorE::ThreadMap::kElementsPerAccess / 8;

        if (is_warp_valid_)
          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpE>(
              dst_ptr, iterator_E.get(), iterator_E.valid() && is_warp_valid_);

        ++iterator_E;

        ++this->smem_iterator_E_;
      }
      // Move to the next stage
      iterator_A.advance();
      iterator_B.advance();
#ifdef SAIL_CUSTOMIZE_CUTLASS
      iterator_E.add_tile_offset(k_sparse_tile);
#else
      iterator_E.add_tile_offset({0, 1});
#endif

      this->smem_iterator_A_.add_tile_offset({0, 1});
      this->smem_iterator_B_.add_tile_offset({1, 0});
#ifdef SAIL_CUSTOMIZE_CUTLASS
      this->smem_iterator_E_.add_tile_offset(k_sparse_tile);
#else
      this->smem_iterator_E_.add_tile_offset({0, 1});
#endif

      // Inserts a fence to group cp.async instructions into stages.
      cutlass::arch::cp_async_fence();
    }

    // Perform accumulation in the 'd' output operand
    accum = src_accum;
    // Waits until kStages-2 stages have committed.
    cutlass::arch::cp_async_wait<Base::kStages - 2>();
    __syncthreads();

    // Pair of fragments used to overlap shared memory loads and math
    // instructions
    WarpLoadedFragmentA warp_loaded_frag_A[Detail::kABufferSize];
    WarpLoadedFragmentB warp_loaded_frag_B[Detail::kBBufferSize];
    WarpTransformedFragmentA warp_transformed_frag_A[Detail::kABufferSize];
    WarpTransformedFragmentB warp_transformed_frag_B[Detail::kBBufferSize];
    WarpFragmentE warp_frag_E[2];

    Operator warp_mma;

    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.set_kgroup_index(0);
    this->warp_tile_iterator_E_.set_kgroup_index(0);

    this->warp_tile_iterator_A_.load(warp_loaded_frag_A[0]);
    this->warp_tile_iterator_B_.load(warp_loaded_frag_B[0]);
    this->warp_tile_iterator_E_.load(warp_frag_E[0]);

    ++this->warp_tile_iterator_A_;
    ++this->warp_tile_iterator_B_;
    ++this->warp_tile_iterator_E_;

    if (gemm_k_iterations == 0) {
      iterator_E.clear_mask();
    }

    // Start issuing the first group of the next stage outside of the mainloop
    copy_tiles_and_advance(iterator_A, iterator_B, iterator_E);

    int smem_write_stage_idx = Base::kStages - 1;
    int smem_read_stage_idx = 0;

    warp_mma.transform(warp_transformed_frag_A[0], warp_transformed_frag_B[0],
                       warp_loaded_frag_A[0], warp_loaded_frag_B[0]);

    //
    // Mainloop
    //

    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations > (-Base::kStages + 1);) {
      //
      // Loop over GEMM K dimension
      //
      // Computes a warp-level GEMM on data held in shared memory
      // Each "warp_mma_k" refers to a warp-level matrix multiply-accumulate
      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations;
           ++warp_mma_k) {

        // Load warp-level tiles from shared memory, wrapping to k offset if
        // this is the last group as the case may be.
        this->warp_tile_iterator_E_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
        this->warp_tile_iterator_E_.load(warp_frag_E[(warp_mma_k + 1) % 2]);
        ++this->warp_tile_iterator_E_;

        if (Detail::kABufferSize == 2) {
          this->warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
          this->warp_tile_iterator_A_.load(
              warp_loaded_frag_A[(warp_mma_k + 1) % Detail::kABufferSize]);
          ++this->warp_tile_iterator_A_;
        }

        if (Detail::kBBufferSize == 2) {
          this->warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
          this->warp_tile_iterator_B_.load(
              warp_loaded_frag_B[(warp_mma_k + 1) % Detail::kBBufferSize]);
          ++this->warp_tile_iterator_B_;
        }


        if (warp_mma_k > 0)
          warp_mma.transform(warp_transformed_frag_A[warp_mma_k % Detail::kABufferSize],
                             warp_transformed_frag_B[warp_mma_k % Detail::kBBufferSize],
                             warp_loaded_frag_A[warp_mma_k % Detail::kABufferSize],
                             warp_loaded_frag_B[warp_mma_k % Detail::kBBufferSize]);

        // Issue global->shared copies for the next stage
        int group_start_iteration_A, group_start_iteration_B, group_start_iteration_E;

        if (warp_mma_k + 1 == Base::kWarpGemmIterations) {
          group_start_iteration_A = 0;
          group_start_iteration_B = 0;
          group_start_iteration_E = 0;
        } else {
          group_start_iteration_A =
              (warp_mma_k + 1) * Detail::kAccessesPerGroupA;
          group_start_iteration_B =
              (warp_mma_k + 1) * Detail::kAccessesPerGroupB;
          group_start_iteration_E =
              (warp_mma_k + 1) * Detail::kAccessesPerGroupE;
        }

        copy_tiles_and_advance(iterator_A, iterator_B, iterator_E, group_start_iteration_A,
                              group_start_iteration_B, group_start_iteration_E);
        warp_mma(
                accum,
                warp_transformed_frag_A[warp_mma_k % Detail::kABufferSize],
                warp_transformed_frag_B[warp_mma_k % Detail::kBBufferSize],
                accum,
                warp_frag_E[warp_mma_k % 2]
              );

        if (Detail::kABufferSize == 1) {
          this->warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
          this->warp_tile_iterator_A_.load(warp_loaded_frag_A[0]);
          ++this->warp_tile_iterator_A_;
        }

        if (Detail::kBBufferSize == 1) {
          this->warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
          this->warp_tile_iterator_B_.load(warp_loaded_frag_B[0]);
          ++this->warp_tile_iterator_B_;
        }

        if (warp_mma_k + 1 == Base::kWarpGemmIterations)
          warp_mma.transform(warp_transformed_frag_A[(warp_mma_k + 1) % Detail::kABufferSize],
                             warp_transformed_frag_B[(warp_mma_k + 1) % Detail::kBBufferSize],
                             warp_loaded_frag_A[(warp_mma_k + 1) % Detail::kABufferSize],
                             warp_loaded_frag_B[(warp_mma_k + 1) % Detail::kBBufferSize]);

        if (warp_mma_k + 2 == Base::kWarpGemmIterations) {
          __syncthreads();
          // Inserts a fence to group cp.async instructions into stages.
          cutlass::arch::cp_async_fence();

          // Waits until kStages-2 stages of cp.async have committed
          arch::cp_async_wait<Base::kStages - 2>();
          __syncthreads();

          // Move to the next stage
          iterator_A.advance();
          iterator_B.advance();
#ifdef SAIL_CUSTOMIZE_CUTLASS
          iterator_E.add_tile_offset(k_sparse_tile);
#else
          iterator_E.add_tile_offset({0, 1});
#endif
          this->smem_iterator_A_.add_tile_offset({0, 1});
          this->smem_iterator_B_.add_tile_offset({1, 0});
#ifdef SAIL_CUSTOMIZE_CUTLASS
          this->smem_iterator_E_.add_tile_offset(k_sparse_tile);
#else
          this->smem_iterator_E_.add_tile_offset({0, 1});
#endif

          // Add negative offsets to return iterators to the 'start' of the
          // circular buffer in shared memory
          if (smem_write_stage_idx == (Base::kStages - 1)) {
            this->smem_iterator_A_.add_tile_offset({0, -Base::kStages});
            this->smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
#ifdef SAIL_CUSTOMIZE_CUTLASS
            this->smem_iterator_E_.add_tile_offset((CompressOp_ == gemm::Operand::kB) ? make_Coord(-Base::kStages, 0) : make_Coord(0, -Base::kStages));
#else
            this->smem_iterator_E_.add_tile_offset({0, -Base::kStages});
#endif
            smem_write_stage_idx = 0;
          } else {
            ++smem_write_stage_idx;
          }

          if (smem_read_stage_idx == (Base::kStages - 1)) {
            this->warp_tile_iterator_A_.add_tile_offset(
                {0, -Base::kStages * Policy::kPartitionsK *
                        Base::kWarpGemmIterations});
            this->warp_tile_iterator_B_.add_tile_offset(
                {-Base::kStages * Policy::kPartitionsK *
                     Base::kWarpGemmIterations,
                 0});
#ifdef SAIL_CUSTOMIZE_CUTLASS
            this->warp_tile_iterator_E_.add_tile_offset(
                (CompressOp_ == gemm::Operand::kB)
                ? make_Coord(-Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0)
                : make_Coord(0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations));
#else
            this->warp_tile_iterator_E_.add_tile_offset(
                   {0, -Base::kStages * Policy::kPartitionsK *
                         Base::kWarpGemmIterations});
#endif
            smem_read_stage_idx = 0;
          } else {
            ++smem_read_stage_idx;
          }

          --gemm_k_iterations;
          if (gemm_k_iterations == 0) {
            iterator_E.clear_mask();
          }
        }
      }

    }

    // Insert fence and wait for all outstanding cp.async operations to commit.
    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
