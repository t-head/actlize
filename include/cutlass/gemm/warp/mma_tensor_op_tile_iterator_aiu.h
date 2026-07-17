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
    \brief Defines iterators used by warp-level matrix multiply operations targeting Tensor Cells.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/arch/memory_ppu.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor_op_multiplicand_ppu.h"

#include "cutlass/platform/platform.h"
#include "cutlass/fast_math.h"

#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator.h"
#include "cutlass/arch/memory_aiu.h"

#include <hggc_mma.h>

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace warp {

////////////////////////////////////////////////////////////////////////////////


#if ACOMPUTE_VERSION == 10000
// to use aiu layout in ppu
// only support fp16 now
// must specify instruction shape, change sizeof_bits<Element_>::value to 16 is still one params for MmaTensorOpMultiplicandTileIterator, will ambiguous to original realization
template <
    /// Size of the matrix to load (concept: PitchLinearShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Shape of one matrix product operation (concept: PitchLinearShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Element number when the layout crosses (in units of elements)
    int Crosswise,
    /// Number of partitions along K dimension
    int PartitionsK_,
    // aiu cube size
    int CubeSize_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, cutlass::half_t,
    cutlass::layout::TensorOpMultiplicandCrosswise<sizeof_bits<cutlass::half_t>::value,
                                                   Crosswise>,
    InstructionShape_, OpDelta_, 32, PartitionsK_, CubeSize_> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Element type
  using Element = cutlass::half_t;

  // aiu pointer type
  using AiuElement = typename platform::conditional<
                        platform::is_same<Element, cutlass::half_t>::value,
                        half,
                        Element
                      >::type;

  static int const CubeSize = CubeSize_;

  static int const kCrosswise = Crosswise;

  static int const kThreads = 32;

  /// Layout of source tile
  using Layout = cutlass::layout::TensorOpMultiplicandCrosswise<
      sizeof_bits<Element>::value, kCrosswise>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = layout::PitchLinearShape<16, 16>;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kContiguous % InstructionShape::kContiguous),
        "Shape of warp-level Mma must be divisible by operator shape.");

    // Determine number of elements along outer dimension per individual LDSM op
#if ACOMPUTE_VERSION == 10000
    // Special for FP16 16x16x32 8 x Q cases, PPU contiguous dimension is not continuous.
    // 4 x 8(128B) ldsm not enough for one mma.
    static int const kLdsmOpOuter = (((InstructionShape::kCount * sizeof(Element)) / 128 > 4) && Layout::kFactor) ?
                                      Layout::kElementsPerAccess * 2 : Layout::kElementsPerAccess;
#else
    static int const kLdsmOpOuter = Layout::kElementsPerAccess;
#endif
    static int const kLdsmOpInner = 8;

    static_assert(!(Shape::kContiguous % kLdsmOpOuter),
                  "Shape of warp-level mma must be divisible by LDSM's "
                  "fundamental tile size.");

    static_assert(!(Shape::kStrided % kLdsmOpInner),
                  "Shape of warp-level mma must be divisible by LDSM's "
                  "fundamental tile size.");

    /// Shape of one individual LDSM instruction
    static int const LdsmShapeContiguous =
        InstructionShape::kContiguous / kLdsmOpOuter;
    static int const LdsmShapeStrided =
        ((4 / LdsmShapeContiguous * kLdsmOpInner) > Shape::kStrided)
            ? (Shape::kStrided / kLdsmOpInner)
            : (4 / LdsmShapeContiguous);

    using LdsmShape =
        layout::PitchLinearShape<LdsmShapeContiguous, LdsmShapeStrided>;

    /// Number and arrangement of LDSM instructions
#if ACOMPUTE_VERSION == 10000
    using LdsmIterations =
        layout::PitchLinearShape<(((InstructionShape::kCount * sizeof(Element)) / 128 > 4) && Layout::kFactor == 1) ?
                                  Shape::kContiguous / kLdsmOpOuter / LdsmShape::kContiguous : 1,
                                  Shape::kStrided / kLdsmOpInner / LdsmShape::kStrided>;
    ///
    static int const kGroupsPerTile = (((InstructionShape::kCount * sizeof(Element)) / 128 > 4) && Layout::kFactor == 1) ?
            Layout::TileShape::kContiguous / Layout::kFactor / LdsmShape::kContiguous / LdsmIterations::kContiguous :
            Layout::TileShape::kContiguous / Layout::kFactor / LdsmShape::kContiguous;
#else
    using LdsmIterations =
      layout::PitchLinearShape<1, Shape::kStrided / kLdsmOpInner / LdsmShape::kStrided>;
    ///
    static int const kGroupsPerTile =
            Layout::TileShape::kContiguous / Layout::kFactor / LdsmShape::kContiguous;
#endif
  };
 private:

  /// Pointer type used for accesses
  using AccessType = Array<Element, Layout::kElementsPerAccess>;

 public:
  //
  // Derived quantities
  //

  static_assert(!(CubeSize_ % 16),
                  "avoid goes here with default param");

  static_assert(!(Shape::kStrided % InstructionShape::kStrided),
                  "warp shape must be divisible by inst shape");

  static_assert(
      !(Shape::kContiguous % InstructionShape::kContiguous),
      "Shape of warp-level Mma must be divisible by operator shape.");

  static int const LdIterCnt = Shape::kStrided / InstructionShape::kStrided;

  // elements in one slice
  static int const ElementPerSlice = 32 * 8 / sizeof_bits<Element>::value;

  // slice in one stage
  static int const SlicePerStage = Shape::kContiguous / InstructionShape::kContiguous;

  // element size for one slice
  static int const SliceSize = CubeSize * ElementPerSlice;

  // element size for one stage
  static int const StageSize = SliceSize * SlicePerStage;

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kStrided *
                                      InstructionShape::kContiguous / kThreads>;

  // always no transpose and only support half now
  // Fragment is whole vreg for one sub loop, AwmmaFragment is for one mma
  using AwmmaFragment = awmma::fragment<awmma::matrix_a, 16, 16, 16, half, awmma::row_major>;

  // used for none double vreg
  using MmaFragment = Array<Element, InstructionShape::kContiguous *
                                      InstructionShape::kStrided / kThreads>;

 private:

  /// Layout object storing stride values
  Index stride_;

  /// Shared memory base pointers - not advanced
  Element const *pointer_;
  half const *base_;

  int slice_idx_;

  // warp start offset in cube
  int warp_offset_;

 public:
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator()
      : pointer_(nullptr),
        warp_offset_(0),
        slice_idx_(0) {}

  /// Constructor from TensorRef
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator(TensorRef const &ref, int lane_id)
      : pointer_(reinterpret_cast<Element const *>(ref.data())),
        warp_offset_(0),
        slice_idx_(0) {
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    // in initial, stride is warp index, contiguous is mma matrix index
    // in ++, jump to next stage when loop of current stage is complete
    // in main loop, reset to first stage, strided is 0, contiguous is -stage * mma_matrix_on_k
    // thus offset.stride is warp offset, offset.contiguous is matrix index, cont / matrix_on_k means move how many stage

    int stage_offset = tile_offset.contiguous() / SlicePerStage;
    pointer_ += stage_offset * StageSize;
    // pointer_ += stage_offset * 128 * 16 * 3;

    // set warp_offset to 64 for example
    warp_offset_ += tile_offset.strided() * Shape::kStrided;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator++() {
    ++slice_idx_;
    if (slice_idx_ == SlicePerStage) {
      slice_idx_ = 0;
      // must use SlicePerStage here since main loop use -Stage * inst_on_k
      add_tile_offset({SlicePerStage, 0});
    }

    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const { load_with_byte_offset(frag, 0); }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {

    // each tsm load is always 4 vreg for any dtype, also for each mma
    Array<unsigned, 4> *fetch_ptr = reinterpret_cast<Array<unsigned, 4> *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < LdIterCnt; ++s) {

      AwmmaFragment *frag_a = reinterpret_cast<AwmmaFragment *>(&fetch_ptr[s]);
      awmma::matrix_load_params tsmLdParams(0, warp_offset_ + s * InstructionShape::kStrided, 1, CubeSize, slice_idx_);
      AiuElement const *slice_ptr = reinterpret_cast<AiuElement const*>(pointer_ + slice_idx_ * SliceSize);
      awmma::load_matrix_sync<awmma::tile_16x1, awmma::swizzle_default>(*frag_a, slice_ptr, tsmLdParams);

    }
  }

  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    slice_idx_ = k_group % SlicePerStage;
  }
};

#elif ACOMPUTE_VERSION == 10500

// test for aiu in PPUI
template <
    typename BlockShape_,
    /// Size of the matrix to load (concept: PitchLinearShape)
    typename Shape_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: PitchLinearShape)
    typename InstructionShape_,
    bool Swap_>
class MmaTensorOpMultiplicandTileIteratorAiu {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;
  using BlockShape = BlockShape_;

  /// Element type
  using Element = Element_;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;
  static constexpr bool Swap = Swap_;

  static int const kThreads = 32;

  /// Fragment object holding a thread's part of a tile
  using Fragment =
     Array<Element, Shape::kRow * InstructionShape::kColumn / kThreads>;

  static int const loops_per_stage_ = Shape::kColumn / InstructionShape::kColumn;

  // elements in one slice
  static int const ElementPerSlice = 32 * 8 / sizeof_bits<Element>::value;

  // element size for one slice
  static int const SliceSize = BlockShape::kRow * ElementPerSlice;

  using AiuElement = typename platform::conditional<
                      platform::is_same<Element, cutlass::half_t>::value,
                      half,
                      Element
                      >::type;

  using AwmmaFragment = awmma::fragment<awmma::matrix_a, 16, 16, InstructionShape::kColumn, AiuElement, awmma::row_major>;

  static constexpr int lbo = Swap ? (8 * BlockShape::kColumn * sizeof(Element) / 16) : 1;
  static constexpr int sbo = Swap ? 1 : (8 * BlockShape::kColumn * sizeof(Element) / 16);
  static constexpr int SwzlMode = Shape::kColumn * sizeof(Element) == 128 ? 0 : 1;

  // offset in tsm of current stage
  int64_t tsm_offset_;
  // mma numbers on each iter
  static int const x_iter_ = Shape::kRow / InstructionShape::kRow;
  int slice_id_;

 public:

  /// Constructor from TensorRef
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIteratorAiu() {
    slice_id_ = 0;
  }

  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIteratorAiu &add_tile_offset(int offset) {

    // row offset, only consider warp_k = block_k
    tsm_offset_ = offset * Shape::kRow;
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag, Element *tsm_ptr) const {
    Array<unsigned, 4> *fetch_ptr = reinterpret_cast<Array<unsigned, 4> *>(&frag);

    int vreg_num_per_inst = InstructionShape::kRow / 8 * 2;
    CUTLASS_PRAGMA_UNROLL
    for (int x = 0; x < x_iter_; x++) {
      // actual tsm load instruction
      int *frag_a = reinterpret_cast<int *>(&fetch_ptr[x]);
      awmma::matrix_load_params tsmLdParams(0, tsm_offset_ + x * InstructionShape::kRow, 1, BlockShape::kRow, slice_id_);
      AiuElement const *slice_ptr = reinterpret_cast<AiuElement const*>(tsm_ptr + slice_id_ * SliceSize);
      // awmma::load_matrix_sync<awmma::tile_16x1, awmma::swizzle_default>(*frag_a, slice_ptr, tsmLdParams);

      Element *stage_base = reinterpret_cast<Element*>(tsm_ptr);
      stage_base += (tsm_offset_ + x * InstructionShape::kRow) * BlockShape::kColumn + slice_id_ * 32 / sizeof(Element);
      int tsm_add = reinterpret_cast<uintptr_t>(stage_base) / 16;

      // no trans always use b16 device assembly
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 {%0, %1, %2, %3}, [%4], %5, %6, %7;"    
        : "=r"(frag_a[0]), "=r"(frag_a[1]), "=r"(frag_a[2]), "=r"(frag_a[3])
        : "l"(tsm_add), "r"(lbo), "r"(sbo), "r"(SwzlMode)
      );
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 {%0, %1, %2, %3}, [%4], %5, %6, %7;"    
        : "=r"(frag_a[0]), "=r"(frag_a[1]), "=r"(frag_a[2]), "=r"(frag_a[3])
        : "l"(tsm_add), "r"(lbo), "r"(sbo), "r"(SwzlMode)
      );
#endif

      // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
      //   printf("=========\n");
      //   printf("tsm_add: %d\n", tsm_add);
      //   printf("lbo: %d, sbo: %d\n", lbo, sbo);
      //   printf("SwzlMode: %d\n", SwzlMode);
      // }

      // // simulator tsm load
      // // always consider dtype is 32 bit
      // float *frag_a = reinterpret_cast<float *>(&fetch_ptr[x]);
      // float const *slice_ptr = reinterpret_cast<float const*>(tsm_ptr + slice_id_ * SliceSize);
      // cutlass::arch::load_matrix_sync_sim(frag_a, slice_ptr, tsm_offset_ + x * InstructionShape::kRow, BlockShape::kRow, slice_id_);
    }

  }

  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIteratorAiu &operator++() {
    // tsm offset will set outside, only needs to store slice id here
    slice_id_++;
    if (slice_id_ == loops_per_stage_) {
      slice_id_ = 0;
    }
  }

  // will have no exit on PPU if use ++???
  CUTLASS_DEVICE
  void plus() {
    // tsm offset will set outside, only needs to store slice id here
    slice_id_++;
    if (slice_id_ == loops_per_stage_) {
      slice_id_ = 0;
    }
  }

};

template <typename Element, bool Swap32 = false>
struct Acompute10500_TSM_LD_SWZL_TRANS;

template <>
struct Acompute10500_TSM_LD_SWZL_TRANS<half_t> {
  CUTLASS_HOST_DEVICE void operator()(int *vreg, int tsm_add, int lbo, int sbo, int swzl_mode) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16.trans {%0, %1, %2, %3}, [%4], %5, %6, %7;"    
      : "=r"(vreg[0]), "=r"(vreg[1]), "=r"(vreg[2]), "=r"(vreg[3])
      : "l"(tsm_add), "r"(lbo), "r"(sbo), "r"(swzl_mode)
    );
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16.trans {%0, %1, %2, %3}, [%4], %5, %6, %7;"    
      : "=r"(vreg[0]), "=r"(vreg[1]), "=r"(vreg[2]), "=r"(vreg[3])
      : "l"(tsm_add), "r"(lbo), "r"(sbo), "r"(swzl_mode)
    );
#endif
  }
};

template <>
struct Acompute10500_TSM_LD_SWZL_TRANS<bfloat16_t> {
  CUTLASS_HOST_DEVICE void operator()(int *vreg, int tsm_add, int lbo, int sbo, int swzl_mode) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16.trans {%0, %1, %2, %3}, [%4], %5, %6, %7;"    
      : "=r"(vreg[0]), "=r"(vreg[1]), "=r"(vreg[2]), "=r"(vreg[3])
      : "l"(tsm_add), "r"(lbo), "r"(sbo), "r"(swzl_mode)
    );
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16.trans {%0, %1, %2, %3}, [%4], %5, %6, %7;"    
      : "=r"(vreg[0]), "=r"(vreg[1]), "=r"(vreg[2]), "=r"(vreg[3])
      : "l"(tsm_add), "r"(lbo), "r"(sbo), "r"(swzl_mode)
    );
#endif
  }
};

template <>
struct Acompute10500_TSM_LD_SWZL_TRANS<float> {
  CUTLASS_HOST_DEVICE void operator()(int *vreg, int tsm_add, int lbo, int sbo, int swzl_mode) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ldmatrix.swzl.sync.bulk.tensor.m8n8.x2.b32.trans {%0, %1, %2, %3}, [%4], %5;"    
      : "=r"(vreg[0]), "=r"(vreg[1]), "=r"(vreg[2]), "=r"(vreg[3])
      : "l"(tsm_add), "r"(swzl_mode)
    );
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x2.b32.trans {%0, %1, %2, %3}, [%4], %5;"    
      : "=r"(vreg[0]), "=r"(vreg[1]), "=r"(vreg[2]), "=r"(vreg[3])
      : "l"(tsm_add), "r"(swzl_mode)
    );
#endif
  }
};

template <>
struct Acompute10500_TSM_LD_SWZL_TRANS<float, true> {
  CUTLASS_HOST_DEVICE void operator()(int *vreg, int tsm_add, int lbo, int sbo, int swzl_mode) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ldmatrix.swzl.sync.bulk.tensor.m8n8.x2.b32.trans.diagswap {%0, %1, %2, %3}, [%4], %5;"    
      : "=r"(vreg[0]), "=r"(vreg[1]), "=r"(vreg[2]), "=r"(vreg[3])
      : "l"(tsm_add), "r"(swzl_mode)
    );
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x2.b32.trans.diagswap {%0, %1, %2, %3}, [%4], %5;"    
      : "=r"(vreg[0]), "=r"(vreg[1]), "=r"(vreg[2]), "=r"(vreg[3])
      : "l"(tsm_add), "r"(swzl_mode)
    );
#endif
  }
};

template <>
struct Acompute10500_TSM_LD_SWZL_TRANS<tfloat32_t> {
  CUTLASS_HOST_DEVICE void operator()(int *vreg, int tsm_add, int lbo, int sbo, int swzl_mode) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ldmatrix.swzl.sync.bulk.tensor.m8n8.x2.b32.trans {%0, %1, %2, %3}, [%4], %5;"    
      : "=r"(vreg[0]), "=r"(vreg[1]), "=r"(vreg[2]), "=r"(vreg[3])
      : "l"(tsm_add), "r"(swzl_mode)
    );
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x2.b32.trans {%0, %1, %2, %3}, [%4], %5;"    
      : "=r"(vreg[0]), "=r"(vreg[1]), "=r"(vreg[2]), "=r"(vreg[3])
      : "l"(tsm_add), "r"(swzl_mode)
    );
#endif
  }
};

template <>
struct Acompute10500_TSM_LD_SWZL_TRANS<tfloat32_t, true> {
  CUTLASS_HOST_DEVICE void operator()(int *vreg, int tsm_add, int lbo, int sbo, int swzl_mode) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.ldmatrix.swzl.sync.bulk.tensor.m8n8.x2.b32.trans.diagswap {%0, %1, %2, %3}, [%4], %5;"    
      : "=r"(vreg[0]), "=r"(vreg[1]), "=r"(vreg[2]), "=r"(vreg[3])
      : "l"(tsm_add), "r"(swzl_mode)
    );
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x2.b32.trans.diagswap {%0, %1, %2, %3}, [%4], %5;"    
      : "=r"(vreg[0]), "=r"(vreg[1]), "=r"(vreg[2]), "=r"(vreg[3])
      : "l"(tsm_add), "r"(swzl_mode)
    );
#endif
  }
};


// block/warp/inst_shape's column always on gemm k direction
// test for aiu in PPUI
template <
    typename BlockShape_,
    /// Size of the matrix to load (concept: PitchLinearShape)
    typename Shape_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: PitchLinearShape)
    typename InstructionShape_,
    bool Swap_>
class MmaTensorOpMultiplicandTileIteratorTransAiu {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;
  using BlockShape = BlockShape_;

  /// Element type
  using Element = Element_;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;
  static constexpr bool Swap = Swap_;

  static int const kThreads = 32;

  /// Fragment object holding a thread's part of a tile
  using Fragment =
     Array<Element, Shape::kRow * InstructionShape::kColumn / kThreads>;

  // elements in one slice
  static int const ElementPerSlice = 32 * 8 / sizeof_bits<Element>::value;

  // element size for one slice
  static int const SliceSize = BlockShape::kColumn * ElementPerSlice;

  using AiuElement = typename platform::conditional<
                      platform::is_same<Element, cutlass::half_t>::value,
                      half,
                      Element
                      >::type;

  using AwmmaFragment = awmma::fragment<awmma::matrix_b, 16, 16, InstructionShape::kColumn, AiuElement, awmma::row_major>;

  static constexpr int CUBE_W = BlockShape::kRow > 128 / sizeof(Element) ? 128 / sizeof(Element) : BlockShape::kRow;
  static constexpr int lbo = Swap ? 1 : (8 * CUBE_W * sizeof(Element) / 16);
  static constexpr int sbo = Swap ? (8 * CUBE_W * sizeof(Element) / 16) : 1;
  static constexpr int SwzlMode = CUBE_W * sizeof(Element) == 128 ? 0 : 1;

  static constexpr bool Swap32 = Swap && (sizeof(Element) == 4);

  // inst.m is on cube_c, each mma will jump different slices for different dtype
  static int const SlicePerMma = 16 * sizeof_bits<Element>::value / 256;

  // offset in tsm of current sta
  // mma numbers on each iter
  static int const x_iter_ = Shape::kRow / InstructionShape::kRow;

  // one mma covers how many slice, 16 tf32 is 2 slice
  static int const SlicePerInst = 2;

  static int const CubeH = BlockShape::kColumn;
  static int const CubeW = BlockShape::kRow < 128 / sizeof(Element) ? BlockShape::kRow : (128 / sizeof(Element));

  // int warp_slice_offset_;
  // int x_offset_;
  int warp_mn_base_;
  int k_offset_;

 public:

  /// Constructor from TensorRef
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIteratorTransAiu() {
    // warp_slice_offset_ = 0;
    // x_offset_ = 0;
    warp_mn_base_ = 0;
    k_offset_ = 0;
  }

  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIteratorTransAiu &add_tile_offset(int offset) {

    // warp start on which slice, slices in different aiu_load store successively
    // warp_slice_offset_ = offset * Shape::kRow / ElementPerSlice;
    warp_mn_base_ = offset * Shape::kRow;
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag, Element *tsm_ptr) const {
    Array<unsigned, 4> *fetch_ptr = reinterpret_cast<Array<unsigned, 4> *>(&frag);

    // int vreg_num_per_inst = InstructionShape::kRow / 8 * 2;
    CUTLASS_PRAGMA_UNROLL
    for (int x = 0; x < x_iter_; x++) {
      // actual tsm load instruction
      int *frag_a = reinterpret_cast<int *>(&fetch_ptr[x]);
      // int slice_idx = warp_slice_offset_ + x * SlicePerMma;
      // awmma::matrix_load_params tsmLdParams(0, x_offset_, 1, BlockShape::kColumn, slice_idx & 0x3);
      // AiuElement const *slice_ptr = reinterpret_cast<AiuElement const*>(tsm_ptr + slice_idx * SliceSize);
      // awmma::load_matrix_sync<awmma::tile_16x1, awmma::swizzle_default>(*frag_a, slice_ptr, tsmLdParams);

      Element *stage_base = reinterpret_cast<Element*>(tsm_ptr);
      int mn_offset = warp_mn_base_ + x * InstructionShape::kRow;
      int cube_idx = mn_offset / CubeW;
      int idx_in_cube = mn_offset % CubeW;
      stage_base += k_offset_ * CubeW + idx_in_cube + cube_idx * CubeH * CubeW;
      int tsm_add = reinterpret_cast<uintptr_t>(stage_base) / 16;

      Acompute10500_TSM_LD_SWZL_TRANS<Element, Swap32>()(frag_a, tsm_add, lbo, sbo, SwzlMode);

      // asm volatile("ppu.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16.trans {%0, %1, %2, %3}, [%4], %5, %6, %7;"
      //   : "=r"(frag_a[0]), "=r"(frag_a[1]), "=r"(frag_a[2]), "=r"(frag_a[3])
      //   : "l"(tsm_add), "r"(lbo), "r"(sbo), "r"(SwzlMode)
      // );


      // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
      //   printf("=========\n");
      //   const Element *tmp = reinterpret_cast<const Element*>(stage_base);
      //   printf("tsm_add: %d, val: %f\n", tsm_add, float(tmp[0]));
      //   printf("lbo: %d, sbo: %d\n", lbo, sbo);
      //   printf("SwzlMode: %d\n", SwzlMode);
      //   printf("cube_idx: %d, idx_in_cube: %d\n", cube_idx, idx_in_cube);
      //   printf("k_offset_: %d, mn_offset: %d\n", k_offset_, mn_offset);
      // }


      // // simulator tsm load
      // Element *frag_a = reinterpret_cast<Element *>(&fetch_ptr[x]);
      // int slice_idx = warp_slice_offset_ + x * SlicePerMma;
      // Element const *slice_ptr = reinterpret_cast<Element const*>(tsm_ptr + slice_idx * SliceSize);
      // cutlass::arch::load_matrix_sync_trans_sim(frag_a, slice_ptr, x_offset_, BlockShape::kColumn, slice_idx & 0x3);

    }

  }

  // will have no exit on PPU if use ++???
  CUTLASS_DEVICE
  void plus() {
    // // mma.k always one slice size
    // x_offset_ += ElementPerSlice;
    // if (x_offset_ == BlockShape::kColumn) {
    //   x_offset_ = 0;
    // }

    k_offset_ += InstructionShape::kColumn;
    if (k_offset_ == BlockShape::kColumn) {
      k_offset_ = 0;
    }
  }
};
#endif


////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
