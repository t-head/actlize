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

// #include "cutlass/gemm/warp/mma_tensor_op_tile_iterator.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace warp {

////////////////////////////////////////////////////////////////////////////////
// actual realization is different for each crosswise/dtype, each needs independent implementation
// needn't transfer crosswise into MmaTensorOp, it is included in Layout, and MmaTensorOpMultiplicandTileIterator uses Layout to find implementation
////////////////////////////////////////////////////////////////////////////////

template <
  typename AccessType_,
  typename TensorRef_,
  int kPointerCount_,
  typename Layout_,
  typename Policy_,
  Operand Operand_>
class MmaTensorOpMultiplicandTileIteratorInitializer;

template <
  typename AccessType_,
  typename TensorRef_,
  int kPointerCount_,
  typename Policy_,
  Operand Operand_>
class MmaTensorOpMultiplicandTileIteratorInitializer<
  AccessType_,
  TensorRef_,
  kPointerCount_,
  cutlass::layout::TensorOpMultiplicandCongruous<16, 64>,
  Policy_,
  Operand_> {
public:

  using AccessType = AccessType_;
  using TensorRef = TensorRef_;
  using Policy = Policy_;
  static int const kPointerCount = kPointerCount_;
  static Operand const kOperand = Operand_;
  using Layout = cutlass::layout::TensorOpMultiplicandCongruous<16, 64>;
  using Index = typename TensorRef::Index;

  // static const AccessType *base;

  CUTLASS_DEVICE
  static void initialize (
    TensorRef const &ref,
    AccessType const *pointer_[kPointerCount],
    int lane_id
  ) {
    Index stride_ = ref.stride(0) / Layout::kElementsPerAccess;

    // base = reinterpret_cast<AccessType*>(ref.data());

    int quad_pair = (lane_id >> 3);
    int quad_quad = (lane_id >> 4);
    int lane_in_quad = (lane_id & 3);
    int lane_in_quad_pair = (lane_id & 7);
    int lane_in_quad_quad = (lane_id & 15);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPointerCount; ++i) {
      int partition_contiguous_idx = -1;
      int access_contiguous_idx = -1;
      int access_strided_idx = -1;

#if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__)) && ACOMPUTE_VERSION == 10000
      if (Policy::LdsmShape::kContiguous == 4) {
        // Q0 Q1 Q2 Q3 (Q stands for 1 8x128bit block).
        // Four blocks are next to each other in the contiguous dimension.
        // PPU MMA 16168.F16/8168.F16, B 8x16
        // PPU ldmatrix address Q0 Q2 Q1 Q3
        // after tsm_ld_ncom_nt1616_b32x4 --> Q0 Q1 Q2 Q3
        partition_contiguous_idx = ((lane_in_quad_pair >> 2) ^ i);
        // quad_pair b1b0->b0b1
        access_contiguous_idx = (((quad_pair & 0x01) << 1) + (quad_pair >> 1)) ^ lane_in_quad;
        access_strided_idx = lane_in_quad_pair;
      } else if (Policy::LdsmShape::kContiguous == 2) {
        // PPU MMA 161616.F16 A/B
        // K*M/K*N shared mem layout
        // after ldmatrix
        // Q0 Q2
        // Q1 Q3
        // Q1<->Q2 swap and 8x8 transpose is needed
        partition_contiguous_idx = ((lane_in_quad_pair >> 2) ^ (i >> 1));
        access_contiguous_idx =
            (((quad_pair & 1) + ((i & 1) << 1)) ^ lane_in_quad);
        access_strided_idx = lane_in_quad_pair + (lane_id >> 4 << 3);
      } else {
        // avoid compiler warning
        access_contiguous_idx = quad_pair;
        assert(0);
      }
#else
      if (Policy::LdsmShape::kContiguous == 4) {
        // Matrix multiply 1688 A/B
        // Q0 Q1 Q2 Q3 (Q stands for 1 8x128bit block).
        // Four blocks are next to each other in the contiguous dimension.
        partition_contiguous_idx = ((lane_in_quad_pair >> 2) ^ i);  // bottom 4 rows in right partition
        access_contiguous_idx = (quad_pair ^ lane_in_quad);  // (/8) ^ (%4)
        access_strided_idx = lane_in_quad_pair;   // % 8
      }
      else if (Policy::LdsmShape::kContiguous == 2 &&
                 kOperand == Operand::kA) {
        // Matrix multiply 16816 A
        // Q0 Q2
        // Q1 Q3
        partition_contiguous_idx = ((lane_in_quad_pair >> 2) ^ (i >> 1));
        access_contiguous_idx =
            (((quad_pair & 1) + ((i & 1) << 1)) ^ lane_in_quad);
        access_strided_idx = lane_in_quad_pair + (lane_id >> 4 << 3);
      } else if (Policy::LdsmShape::kContiguous == 2 &&
                 kOperand == Operand::kB) {
        // Matrix multiply 16816 B
        // Q0 Q1
        // Q2 Q3
        // seems Q1 means on successive dim of tensor cell B input which is column major
        // actual layout is
        // T0-7    T16-23
        // T8-15   T24-31
        partition_contiguous_idx = ((lane_in_quad_pair >> 2) ^ (i >> 1));
        access_contiguous_idx = ((quad_quad + ((i & 1) << 1)) ^ lane_in_quad);
        access_strided_idx = lane_in_quad_quad;
      } else if (Policy::LdsmShape::kContiguous == 1) {
        // Matrix multiply 16832.SP B
        // Q0
        // Q1
        // Q2
        // Q3
        partition_contiguous_idx = ((lane_in_quad_pair >> 2) ^ (i >> 2));
        access_contiguous_idx = ((i & 3) ^ lane_in_quad);
        access_strided_idx = lane_id;
      }
#endif

      int access_contiguous =
          // PartitionShape always <4, 4>
          partition_contiguous_idx * Layout::PartitionShape::kContiguous +
          access_contiguous_idx;

      int access_strided = access_strided_idx;

      pointer_[i] = reinterpret_cast<AccessType const *>(ref.data()) +
                    access_contiguous + access_strided * stride_;
    }
  }

};

template <
  typename AccessType_,
  typename TensorRef_,
  int kPointerCount_,
  typename Policy_,
  Operand Operand_>
class MmaTensorOpMultiplicandTileIteratorInitializer<
  AccessType_,
  TensorRef_,
  kPointerCount_,
  cutlass::layout::TensorOpMultiplicandCongruous<16, 32>,
  Policy_,
  Operand_> {
public:

  using AccessType = AccessType_;
  using TensorRef = TensorRef_;
  using Policy = Policy_;
  static int const kPointerCount = kPointerCount_;
  static Operand const kOperand = Operand_;
  using Layout = cutlass::layout::TensorOpMultiplicandCongruous<16, 32>;
  using Index = typename TensorRef::Index;

  // static const AccessType *base;

  CUTLASS_DEVICE
  static void initialize (
    TensorRef const &ref,
    AccessType const *pointer_[kPointerCount],
    int lane_id
  ) {
    Index stride_ = ref.stride(0) / Layout::kElementsPerAccess;
    int quad_quad = (lane_id >> 4);
    int lane_in_pair = (lane_id & 1);
    int lane_in_quad_pair = (lane_id & 7);
    int lane_in_quad_quad = (lane_id & 15);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPointerCount; ++i) {
      int partition_contiguous_idx = -1;
      int access_contiguous_idx = -1;
      int access_strided_idx = -1;

#if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__)) && ACOMPUTE_VERSION == 10000
      if (Policy::LdsmShape::kContiguous == 2) {
        // assert(0);
        // Matrix multiply 16/16/16 A/B
        /*
               ori                     smem store                             smem load
          0   1   2   3       0   1   2   3   4   5   6   7         0   8   ptr1    1   9
          4   5   6   7       9   8   11  10  13  12  15  14        10  2           11  3
          8   9   10  11      18  19  16  17  22  23  20  21                4   12          5   13
          12  13  14  15      27  26  25  24  31  30  29  28                14  6           15  7
          16  17  18  19                                            16  24          17  25
          20  21  22  23                                            26  18          27  19
          24  25  26  27                                                    20  28          21  29
          28  29  30  31                                                    30  22          31  23
        */

        partition_contiguous_idx = lane_in_pair;
        access_contiguous_idx = ((lane_in_quad_quad >> 3) + (i << 1)) ^ (lane_in_quad_pair >> 1);  // hori/vert
        access_strided_idx = lane_in_quad_pair / 2 + (lane_id >> 4 << 2);
      } else {
        printf("unsupport inst shape for congruous iterator\n");
        assert(0);
      }
#else
      int quad_pair = (lane_id >> 3);
      // Policy::LdsmShape::kContiguous == 2 && Policy::LdsmShape::kStrided == 1 corresponds to warp_shape=16/block_shape=32
      // this case right two columns are for warp1, and point[1] is useless for warp0
      // warp1 will change point[0]/[1] in add_tile_offset to start from right two columns
      if (Policy::LdsmShape::kContiguous == 4 || (Policy::LdsmShape::kContiguous == 2 && Policy::LdsmShape::kStrided == 1)) {
        // must have this
        // Matrix multiply 16/8/8 FP16 A/B
        /*
               ori                     smem store                             smem load
          0   1   2   3       0   1   2   3   4   5   6   7         0   8   16  24  1   9   17  25
          4   5   6   7       9   8   11  10  13  12  15  14        10  2   26  18  11  3   27  19
          8   9   10  11      18  19  16  17  22  23  20  21        20  28  4   12  21  29  5   13
          12  13  14  15      27  26  25  24  31  30  29  28        30  22  14  6   31  23  15  7
          16  17  18  19
          20  21  22  23
          24  25  26  27
          28  29  30  31
        */

        // needn't consider i=1, since one loop will read whole columns on N dimension
        // not only stride difference to ori func, cont_idx also different since T1 in ori func should swizzle to different col of T0
        partition_contiguous_idx = lane_in_pair;  // % 2
        access_contiguous_idx = ((quad_pair + (i << 1)) ^ (lane_in_quad_pair >> 1));  // (/8) ^ (%8 / 2), hori/vert
        access_strided_idx = lane_in_quad_pair >> 1;   // % 8

      } else if (Policy::LdsmShape::kContiguous == 2 && Policy::LdsmShape::kStrided == 2 && kOperand == Operand::kB) {
        // Matrix multiply 16/8/16 FP16 B
        // ori is same to 16/8/8
        // smem load
        // 0   16  ptr1    1   17
        // 18  2           19  3
        //         4   20          5   21
        //         22  6           23  7
        // 8   24  ptr1    9   25
        // 26  10          27  11
        //         12  28          13  29
        //         30  14          31  15

        partition_contiguous_idx = lane_in_pair;
        access_contiguous_idx = (quad_quad + (i << 1)) ^ (lane_in_quad_pair >> 1);
        access_strided_idx = lane_in_quad_quad >> 1;

      } else if (Policy::LdsmShape::kContiguous == 2 && Policy::LdsmShape::kStrided == 2 && kOperand == Operand::kA) {
        // Matrix multiply 16/8/16 FP16 A
        // Policy::LdsmShape::kContiguous=2 when block.m=32/warp.m=32
        // Policy::LdsmShape::kContiguous=1 when block.m=32/warp.m=16
        // smem load
        // 0   8   ptr1    1   9
        // 10  2           11  3
        //         4   12          5   13
        //         14  6           15  7
        // 16  24  ptr1    17  25
        // 26  18          27  19
        //         20  28          21  29
        //         30  22          31  23
        // when warp shape is 16, ptr1 is not used
        // simply not change kPointerCount which use swizzle window / ldsm_shape to get pointer count
        // when warp shape is smaller than swizzle window, this pointer count is to huge
        // but additional pointer initialize has small effect to performance
        partition_contiguous_idx = lane_in_pair;
        access_contiguous_idx = ((lane_in_quad_quad >> 3) + (i << 1)) ^ (lane_in_quad_pair >> 1);
        access_strided_idx = (lane_in_quad_pair >> 1) + (quad_quad << 2);
      } else {
        printf("unsupport inst shape for congruous iterator\n");
        assert(0);
      }
#endif

      int access_contiguous =
          // PartitionShape always <4, 4>
          partition_contiguous_idx * Layout::PartitionShape::kContiguous +
          access_contiguous_idx;

      int access_strided = access_strided_idx;

      pointer_[i] = reinterpret_cast<AccessType const *>(ref.data()) +
                    // this case smem stride is smaller than cache line and access_strided is on cache line coord
                    // stride_ * Layout::kFactor = one cache line, which is the actual stride
                    access_contiguous + access_strided * stride_ * Layout::Base::kFactor;
    }

  }

};

template <
  typename AccessType_,
  typename TensorRef_,
  int kPointerCount_,
  typename Policy_,
  Operand Operand_>
class MmaTensorOpMultiplicandTileIteratorInitializer<
  AccessType_,
  TensorRef_,
  kPointerCount_,
  cutlass::layout::TensorOpMultiplicandCongruous<16, 16>,
  Policy_,
  Operand_> {
public:

  using AccessType = AccessType_;
  using TensorRef = TensorRef_;
  using Policy = Policy_;
  static int const kPointerCount = kPointerCount_;
  static Operand const kOperand = Operand_;
  using Layout = cutlass::layout::TensorOpMultiplicandCongruous<16, 16>;
  using Index = typename TensorRef::Index;

  // static const AccessType *base;

  CUTLASS_DEVICE
  static void initialize (
    TensorRef const &ref,
    AccessType const *pointer_[kPointerCount],
    int lane_id
  ) {
    Index stride_ = ref.stride(0) / Layout::kElementsPerAccess;

    int quad_quad = (lane_id >> 4);
    int lane_in_pair = (lane_id & 1);
    int lane_in_quad = (lane_id & 3);
    int lane_in_quad_pair = (lane_id & 7);
    int lane_in_quad_quad = (lane_id & 15);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPointerCount; ++i) {
      int partition_contiguous_idx = -1;
      int access_contiguous_idx = -1;
      int access_strided_idx = -1;

#if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__)) && ACOMPUTE_VERSION == 10000
      if (Policy::LdsmShape::kContiguous == 2) {
        // 16/16/16 FP16
        //           smem load
        // 0   8   1   9   2   10  3   11
        // 12  4   13  5   14  6   15  7
        // 16  24  17  25  18  16  19  27
        // 28  20  29  21  30  22  31  23
        partition_contiguous_idx = lane_in_quad >> 1;
        access_strided_idx = (lane_in_quad_pair >> 2) + (quad_quad << 1);
        access_contiguous_idx = ((lane_in_pair << 1) + (lane_in_quad_quad >> 3)) ^ (access_strided_idx & 1);
      } else {
        printf("unsupport inst shape for congruous iterator\n");
        assert(0);
      }
#else
      if (Policy::LdsmShape::kContiguous == 2 && Policy::LdsmShape::kStrided == 1) {
        // Matrix multiply 16/8/8 FP16 A/B
        // T16-31's data is unused, can use TSM_LD_SWIZZLE_B32x2
        // should use TileShape = <8, 2>, in each tsm ld last 16 thread's data is for next inner loop
        // but its to complex to make matrix A/B has different inner loop size
        // so latter 16 thread's data is unused,
        /*
           ori                 smem store                             smem load
          0   1       0   1   2   3   4   5   6   7         0   8   1   9   2   10  3   11
          2   3       9   8   11  10  13  12  15  14        12  4   13  5   14  6   15  7
          4   5       16  17  18  19  20  21  22  23        next loop, T16-31's data is unused, just needs to ensure conflict free
          6   7       25  24  27  26  29  28  31  30        so T16-31 use same add to T0-15
          8   9
          10  11
          12  13
          14  15
          16  17
          18  19
          20  21
          22  23
          24  25
          26  27
          28  29
          30  31
        */
        partition_contiguous_idx = lane_in_quad >> 1;
        access_strided_idx = lane_in_quad_pair >> 2;
        access_contiguous_idx = ((lane_in_pair << 1) + (lane_in_quad_quad >> 3)) ^ access_strided_idx;
      } else if (Policy::LdsmShape::kContiguous == 2 && Policy::LdsmShape::kStrided == 2 && kOperand == Operand::kB) {
        // Matrix multiply 16/8/16 FP16 B
        //           smem load
        // 0   16  1   17  2   18  3   19
        // 20  4   21  5   22  6   23  7
        // 8   24  9   25  10  16  11  27
        // 28  12  29  13  30  14  31  15
        partition_contiguous_idx = lane_in_quad >> 1;
        access_strided_idx = ((lane_in_quad_quad >> 3) << 1) + (lane_in_quad_pair >> 2);
        access_contiguous_idx = ((lane_in_pair << 1) + quad_quad) ^ (access_strided_idx & 1);
      } else if (Policy::LdsmShape::kContiguous == 2 && Policy::LdsmShape::kStrided == 2 && kOperand == Operand::kA) {
        // Matrix multiply 16/8/16 FP16 A
        //           smem load
        // 0   8   1   9   2   10  3   11
        // 12  4   13  5   14  6   15  7
        // 16  24  17  25  18  16  19  27
        // 28  20  29  21  30  22  31  23
        partition_contiguous_idx = lane_in_quad >> 1;
        access_strided_idx = (lane_in_quad_pair >> 2) + (quad_quad << 1);
        access_contiguous_idx = ((lane_in_pair << 1) + (lane_in_quad_quad >> 3)) ^ (access_strided_idx & 1);
      } else {
        printf("unsupport inst shape for congruous iterator\n");
        assert(0);
      }
#endif

      int access_contiguous =
          partition_contiguous_idx * Layout::PartitionShape::kContiguous +
          access_contiguous_idx;

      int access_strided = access_strided_idx;

      pointer_[i] = reinterpret_cast<AccessType const *>(ref.data()) +
                    // this case smem stride is smaller than cache line and access_strided is on cache line coord
                    // stride_ * Layout::kFactor = one cache line, which is the actual stride
                    access_contiguous + access_strided * stride_ * Layout::Base::kFactor;
    }

  }

};

template <
  typename AccessType_,
  typename TensorRef_,
  int kPointerCount_,
  typename Policy_,
  Operand Operand_>
class MmaTensorOpMultiplicandTileIteratorInitializer<
  AccessType_,
  TensorRef_,
  kPointerCount_,
  cutlass::layout::TensorOpMultiplicandCongruous<32, 32>,
  Policy_,
  Operand_> {
public:

  using AccessType = AccessType_;
  using TensorRef = TensorRef_;
  using Policy = Policy_;
  static int const kPointerCount = kPointerCount_;
  static Operand const kOperand = Operand_;
  using Layout = cutlass::layout::TensorOpMultiplicandCongruous<32, 32>;
  using Index = typename TensorRef::Index;

  // static const AccessType *base;

  CUTLASS_DEVICE
  static void initialize (
    TensorRef const &ref,
    AccessType const *pointer_[kPointerCount],
    int lane_id
  ) {
    Index stride_ = ref.stride(0);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPointerCount; ++i) {
      int access_strided = lane_id % Policy::kLdsOpInner;
      int access_contiguous = (lane_id / Policy::kLdsOpInner) +
                              // swizzle Policy::kLdsOpOuter elements on each row
                              (access_strided ^ i) * Policy::kLdsOpOuter;

      pointer_[i] = reinterpret_cast<AccessType const *>(ref.data()) +
                    access_contiguous + access_strided * stride_;
    }
  }

};

template <
  typename AccessType_,
  typename TensorRef_,
  int kPointerCount_,
  typename Policy_,
  Operand Operand_>
class MmaTensorOpMultiplicandTileIteratorInitializer<
  AccessType_,
  TensorRef_,
  kPointerCount_,
  cutlass::layout::TensorOpMultiplicandCongruous<32, 16>,
  Policy_,
  Operand_> {
public:

  using AccessType = AccessType_;
  using TensorRef = TensorRef_;
  using Policy = Policy_;
  static int const kPointerCount = kPointerCount_;
  static Operand const kOperand = Operand_;
  using Layout = cutlass::layout::TensorOpMultiplicandCongruous<32, 16>;
  using Index = typename TensorRef::Index;

  // static const AccessType *base;

  CUTLASS_DEVICE
  static void initialize (
    TensorRef const &ref,
    AccessType const *pointer_[kPointerCount],
    int lane_id
  ) {
    Index stride_ = ref.stride(0);

    /*
           ori                      smem store
      0   1   2   3        0   1   2   3   4   5   6   7
      4   5   6   7        10  11  8   9   14  15  12  13
      8   9   10  11       16  17  18  19  20  21  22  23
      12  13  14  15       26  27  24  25  30  31  28  29
      16  17  18  19
      20  21  22  23
      24  25  26  27
      28  29  30  31

                   smem load
      0.0  4.0  8.0  12.0 16.0 20.0 24.0 28.0 | 0.1  4.1  8.1  12.1 16.1 20.1 24.1 28.1 | 1.0  5.0  9.0  13.0 17.0 21.0 25.0 29.0 | 1.1  5.1  9.1  13.1 17.1 21.1 25.1 29.1
      2.1  6.1  10.1 14.1 18.1 22.1 26.1 30.1 | 2.0  6.0  10.0 14.0 18.0 22.0 26.0 30.0 | 3.1  7.1  11.1 15.1 19.1 23.1 27.1 31.1 | 3.0  7.0  11.0 15.0 19.0 23.0 27.0 31.0
      for next loop on s
    */

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPointerCount; ++i) {
      int access_strided = lane_id % Policy::kLdsOpInner;
      // swizzle every kFactor rows
      int residual_strided = access_strided / Layout::kFactor;
      // access_strided /= Layout::kFactor;
      int access_contiguous = (lane_id / Policy::kLdsOpInner) +
                              (residual_strided ^ i) * Policy::kLdsOpOuter;

      pointer_[i] = reinterpret_cast<AccessType const *>(ref.data()) +
                    access_contiguous + access_strided * stride_;
    }
  }

};

} // namespace warp
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
