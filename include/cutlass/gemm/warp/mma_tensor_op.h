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
    \brief Templates implementing warp-level matrix multiply-accumulate operations targeting
      Tensor Cells.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/platform/platform.h"

#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/arch/memory_ppu.h"
#include "cutlass/arch/mma_ppu.h"
#include "cutlass/arch/mma_ppu.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma.h"

#include "cutlass/gemm/warp/mma_tensor_op_policy.h"

#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator.h"
// #include "cutlass/gemm/warp/mma_tensor_op_tile_iterator_initialize.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator_ppu.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <typename T, typename S, int N, FloatRoundStyle Round>
struct ConvertAndPack {

  using Converter = NumericArrayConverter<T, S, N, Round>;

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<S, N> const &source) {
    Converter converter;

    return converter(source);
  }
};

template <typename T, int N, FloatRoundStyle Round>
struct ConvertAndPack<T, T, N, Round> {

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &source) {
		return source;
  }
};

template <int N, FloatRoundStyle Round>
struct ConvertAndPack<bfloat16_t, float, N, Round> {

  using Converter = NumericArrayConverter<bfloat16_t, float, N, Round>;

  CUTLASS_HOST_DEVICE
  Array<bfloat16_t, N> operator()(Array<float, N> const &source) {
    Converter converter;

    Array<float, N> tmp;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      int idx = (((i << 1) & 2) | ((i >> 1) & 1) | (i & 0xfffffffc));
      tmp[i] = source[idx];
    }

    return converter(tmp);
  }
};

template <int N, FloatRoundStyle Round>
struct ConvertAndPack<half_t, float, N, Round> {

  using Converter = NumericArrayConverter<half_t, float, N, Round>;

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(Array<float, N> const &source) {
    Converter converter;

    Array<float, N> tmp;

#if ACOMPUTE_VERSION == 10000
    // fp32 --> fp16 swizzle
    if (N == 8) {
      tmp[0] = source[0];
      tmp[1] = source[2];
      tmp[2] = source[4];
      tmp[3] = source[6];
      tmp[4] = source[1];
      tmp[5] = source[3];
      tmp[6] = source[5];
      tmp[7] = source[7];
    } else
#endif
    {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < N; ++i) {
        int idx = (((i << 1) & 2) | ((i >> 1) & 1) | (i & 0xfffffffc));
        tmp[i] = source[idx];
      }
    }

    return converter(tmp);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting alu cores and SIMT math instructions.
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
  /// Data type of A elements
  typename ElementA_,
  /// Layout of A matrix (concept: MatrixLayout)
  typename LayoutA_,
  /// Data type of B elements
  typename ElementB_,
  /// Layout of B matrix (concept: MatrixLayout)
  typename LayoutB_,
  /// Element type of C matrix
  typename ElementC_,
  /// Layout of C matrix (concept: MatrixLayout)
  typename LayoutC_,
  /// Policy describing warp-level MmaTensorOp (concept: MmaTensorOp policy)
  typename Policy_,
  /// Number of partitions along K dimension
  int PartitionsK_ = 1,
  /// Store the accumulators in row major or column major.  Row major is used
  /// when output layout is interleaved.
  bool AccumulatorsInRowMajor = false,
  /// Used for partial specialization
  typename Enable = bool,
  // aiu cube size
  int CubeA = 1,
  int CubeB = 1
>
class MmaTensorOp {
public:
  /// Shape of warp-level matrix operation (concept: GemmShape)
  using Shape = Shape_;

  /// Data type of multiplicand A
  using ElementA = ElementA_;

  /// Layout of multiplicand A
  using LayoutA = LayoutA_;

  /// Data type of multiplicand B
  using ElementB = ElementB_;

  /// Layout of multiplicand B
  using LayoutB = LayoutB_;

  /// Data type of accumulator matrix C
  using ElementC = ElementC_;

  /// Layout of accumulator matrix C
  using LayoutC = LayoutC_;

  /// Shape of the warp in units of thread (concept: MmaLanePolicySimt)
  using Policy = Policy_;

  /// Underlying matrix multiply operator (concept: arch::Mma)
  using ArchMmaOperator = typename Policy::Operator;

  /// Indicates math operator
  using MathOperator = typename ArchMmaOperator::Operator;

  /// Architecture tag from underlying instruction
  using ArchTag = typename ArchMmaOperator::ArchTag;

  /// Indicates class of matrix operator
  using OperatorClass = arch::OpClassTensorOp;

  /// Shape of underlying instruction
  using InstructionShape = typename ArchMmaOperator::Shape;

  /// Complex transform on A operand
  static ComplexTransform const kTransformA = ComplexTransform::kNone;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = ComplexTransform::kNone;

  /// Number of threads participating in warp-level matrix product
  static int const kThreadCount = 32;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK_;

public:

  /// Iterates over the A operand in memory
  using IteratorA = MmaTensorOpMultiplicandTileIterator<
     MatrixShape<Shape::kM, Shape::kK>, Operand::kA, ElementA, LayoutA,
     MatrixShape<ArchMmaOperator::Shape::kM, ArchMmaOperator::Shape::kK>,
     Policy::OpDelta::kRow, kThreadCount, kPartitionsK, CubeA>;

  /// Storage for A tile
  using FragmentA = typename IteratorA::Fragment;

  /// Storage for transformed A tile
  using TransformedFragmentA =
      Array<typename ArchMmaOperator::ElementA, FragmentA::kElements>;

  using MmaLoadedFragmentA =
      Array<ElementA, ArchMmaOperator::FragmentA::kElements>;

  using TransformedMmaFragmentA = typename ArchMmaOperator::FragmentA;

  /// Iterates over the B operand in memory
  using IteratorB = MmaTensorOpMultiplicandTileIterator<
      MatrixShape<Shape::kK, Shape::kN>, Operand::kB, ElementB, LayoutB,
      MatrixShape<ArchMmaOperator::Shape::kK, ArchMmaOperator::Shape::kN>,
      Policy::OpDelta::kRow, kThreadCount, kPartitionsK, CubeB>;

  /// Storage for B tile
  using FragmentB = typename IteratorB::Fragment;

  /// Storage for transformed B tile
  using TransformedFragmentB =
      Array<typename ArchMmaOperator::ElementB, FragmentB::kElements>;

  using MmaLoadedFragmentB =
      Array<ElementB, ArchMmaOperator::FragmentB::kElements>;

  using TransformedMmaFragmentB = typename ArchMmaOperator::FragmentB;

  /// Iterates over the C operand in memory
  using IteratorC = MmaTensorOpAccumulatorTileIterator<
     MatrixShape<Shape::kM, Shape::kN>, ElementC, LayoutC,
     typename ArchMmaOperator::Shape, typename Policy::OpDelta>;

  /// Storage for C tile
  using FragmentC = typename IteratorC::Fragment;

  /// Number of mma operations performed
  using MmaIterations = MatrixShape<
      (Shape::kM + ArchMmaOperator::Shape::kM - 1) / ArchMmaOperator::Shape::kM,
      (Shape::kN + ArchMmaOperator::Shape::kN - 1) / ArchMmaOperator::Shape::kN
      >;

  // thread load times in each instruction shape
  static constexpr int MmaLoadLoopA = InstructionShape::kM * InstructionShape::kK / 32;
  static constexpr int MmaLoadLoopB = InstructionShape::kN * InstructionShape::kK / 32;

public:

  /// Underlying matrix multiply operator (concept: arch::Mma)
  ArchMmaOperator mma;

public:

  //
  // Methods
  //

  /// Ctor
  CUTLASS_DEVICE
  MmaTensorOp() {}

  /// Performs a warp-level matrix multiply-accumulate operation
  CUTLASS_DEVICE
  void operator()(
    FragmentC &D,
    TransformedFragmentA const &A,
    TransformedFragmentB const &B,
    FragmentC const &C
  ) const {

    using MmaOperandA = typename ArchMmaOperator::FragmentA;
    using MmaOperandB = typename ArchMmaOperator::FragmentB;
    using MmaOperandC = typename ArchMmaOperator::FragmentC;

    D = C;

    MmaOperandA const *ptr_A = reinterpret_cast<MmaOperandA const *>(&A);
    MmaOperandB const *ptr_B = reinterpret_cast<MmaOperandB const *>(&B);
    MmaOperandC *ptr_D = reinterpret_cast<MmaOperandC *>(&D);

    #if defined(__HGGC_ARCH__) && (__HGGC_ARCH__ < 100)
      // Serpentine visitation order maximizing reuse of Rb
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < MmaIterations::kColumn; ++n) {

        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < MmaIterations::kRow; ++m) {

          int m_serpentine = ((n % 2) ? (MmaIterations::kRow - 1 - m) : m);

          if (AccumulatorsInRowMajor) {  // matrix B is reordered
            mma(
              ptr_D[n + m_serpentine * MmaIterations::kColumn],
              ptr_A[m_serpentine],
              ptr_B[n],
              ptr_D[n + m_serpentine * MmaIterations::kColumn]);
          } else {
            mma(
              ptr_D[m_serpentine + n * MmaIterations::kRow],
              ptr_A[m_serpentine],
              ptr_B[n],
              ptr_D[m_serpentine + n * MmaIterations::kRow]);
          }
        }
      }
    #elif defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 100)
      // Serpentine visitation order maximizing reuse of Ra
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < MmaIterations::kRow; ++m) {
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < MmaIterations::kColumn; ++n) {
          int n_serpentine = ((m % 2) ? (MmaIterations::kColumn - 1 - n) : n);
          if (AccumulatorsInRowMajor) {  // matrix B is reordered
            mma(
              ptr_D[n_serpentine + m * MmaIterations::kColumn],
              ptr_A[m],
              ptr_B[n_serpentine],
              ptr_D[n_serpentine + m * MmaIterations::kColumn]);
          } else {
            mma(ptr_D[m + n_serpentine * MmaIterations::kRow],
                ptr_A[m],
                ptr_B[n_serpentine],
                ptr_D[m + n_serpentine * MmaIterations::kRow]);

            // if (m == 0 && n_serpentine == 0 && blockIdx.x == 0 && blockIdx.y == 1 && blockIdx.z == 0 && threadIdx.x >= 0 && threadIdx.x <= 3) {
            //   const ElementA *tmp = reinterpret_cast<const ElementA *>(&ptr_A[m]);
            //   printf("A, thread: %d, val: %f, %f\n", threadIdx.x, float(*tmp), float(*(tmp + 1)));
            // }

            // if (m == 0 && n_serpentine == 0 && blockIdx.x == 0 && blockIdx.y == 1 && blockIdx.z == 0 && threadIdx.x >= 0 && threadIdx.x <= 3) {
            //   const ElementB *tmp = reinterpret_cast<const ElementB *>(&ptr_B[n_serpentine]);
            //   printf("B, thread: %d, val: %f, %f\n", threadIdx.x, float(*tmp), float(*(tmp + 1)));
            // }

            // if (m == 0 && n_serpentine == 0 && blockIdx.x == 0 && blockIdx.y == 1 && blockIdx.z == 0 && threadIdx.x == 0) {
            //   float *tmp = reinterpret_cast<float *>(&ptr_D[m + n_serpentine * MmaIterations::kRow]);
            //   printf("\nD, thread: %d, val: %f\n\n", threadIdx.x, float(*(tmp + 0)));
            // }

          }
        }
      }
    #else

      // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x >= 0 && threadIdx.x <= 3) {
      //   const float *frag_b = reinterpret_cast<const float*>(&ptr_B[0]);
      //   printf("thread: %d, val: %f, %f\n", threadIdx.x, frag_b[2], frag_b[3]);
      // }

      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < MmaIterations::kColumn; ++n) {

        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < MmaIterations::kRow; ++m) {

          int m_serpentine = ((n % 2) ? (MmaIterations::kRow - 1 - m) : m);

          if (AccumulatorsInRowMajor) {  // matrix B is reordered
            mma(
              ptr_D[n + m_serpentine * MmaIterations::kColumn],
              ptr_A[m_serpentine],
              ptr_B[n],
              ptr_D[n + m_serpentine * MmaIterations::kColumn]);
          } else {
            mma(
              ptr_D[m_serpentine + n * MmaIterations::kRow],
              ptr_A[m_serpentine],
              ptr_B[n],
              ptr_D[m_serpentine + n * MmaIterations::kRow]);


            // if (m_serpentine == 0 && n == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x >= 0 && threadIdx.x <= 3) {
            //   const ElementA *tmp = reinterpret_cast<const ElementA *>(&ptr_A[m_serpentine]);
            //   printf("A, thread: %d, val: %f, %f\n", threadIdx.x, float(*tmp), float(*(tmp + 1)));
            // }

            // if (m_serpentine == 0 && n == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x >= 0 && threadIdx.x <= 3) {
            //   const ElementB *tmp = reinterpret_cast<const ElementB *>(&ptr_B[n]);
            //   printf("B, thread: %d, val: %f, %f\n", threadIdx.x, float(*(tmp + 0)), float(*(tmp + 1)));
            // }

            // if (m_serpentine == 0 && n == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
            //   float *tmp = reinterpret_cast<float *>(&ptr_D[m_serpentine + n * MmaIterations::kRow]);
            //   printf("\nD, thread: %d, val: %f\n\n", threadIdx.x, float(*(tmp + 0)));
            // }
          }
        }
      }
    #endif
  }

  CUTLASS_DEVICE
  void operator()(
    FragmentC &D,
    TransformedFragmentA const &A,
    TransformedMmaFragmentB const &B,
    FragmentC const &C,
    int n, int m
  ) const {
    using MmaOperandA = typename ArchMmaOperator::FragmentA;
    using MmaOperandC = typename ArchMmaOperator::FragmentC;

    D = C;

    MmaOperandA const *ptr_A = reinterpret_cast<MmaOperandA const *>(&A);
    MmaOperandC *ptr_D = reinterpret_cast<MmaOperandC *>(&D);

    if (AccumulatorsInRowMajor) {  // matrix B is reordered
        mma(
          ptr_D[n + m * MmaIterations::kColumn],
          ptr_A[m],
          B,
          ptr_D[n + m * MmaIterations::kColumn]);
      } else {
        mma(
          ptr_D[m + n * MmaIterations::kRow],
          ptr_A[m],
          B,
          ptr_D[m + n * MmaIterations::kRow]);
      }
  }

 CUTLASS_DEVICE
  void operator()(
    FragmentC &D,
    TransformedMmaFragmentA const &A,
    TransformedMmaFragmentB const &B,
    FragmentC &C,
    int m, int n, bool AccInRow
  ) const {
    using MmaOperandC = typename ArchMmaOperator::FragmentC;

    D = C;

    MmaOperandC *ptr_D = reinterpret_cast<MmaOperandC *>(&D);

    if (AccumulatorsInRowMajor) {  // matrix B is reordered
        mma(
          ptr_D[n + m * MmaIterations::kColumn],
          A,
          B,
          ptr_D[n + m * MmaIterations::kColumn]);
      } else {
        mma(
          ptr_D[m + n * MmaIterations::kRow],
          A,
          B,
          ptr_D[m + n * MmaIterations::kRow]
          );
      }
  }

  /// Transform the mma operands to the required types
  CUTLASS_DEVICE
  void transform(TransformedFragmentA &dst_A, TransformedFragmentB &dst_B,
                 FragmentA const &A, FragmentB const &B) const {

    //
    // Define conversions from source type to instruction type
    //
    FloatRoundStyle const kRoundA =
        PreferredRoundingMode<typename ArchMmaOperator::ElementA,
                              ElementA>::kRound;
    FloatRoundStyle const kRoundB =
        PreferredRoundingMode<typename ArchMmaOperator::ElementB,
                              ElementB>::kRound;
      // should always transfer between 8 elements when instruction shape is 16/16/16
      detail::ConvertAndPack<typename ArchMmaOperator::ElementA, ElementA,
                            MmaLoadLoopA, kRoundA>
          convert_A;
      detail::ConvertAndPack<typename ArchMmaOperator::ElementB, ElementB,
                            MmaLoadLoopB, kRoundB>
          convert_B;

      Array<ElementA, MmaLoadLoopA> const *ptr_A =
          reinterpret_cast<Array<ElementA, MmaLoadLoopA> const *>(&A);
      Array<typename ArchMmaOperator::ElementA, MmaLoadLoopA> *
          ptr_dst_A = reinterpret_cast<Array<typename ArchMmaOperator::ElementA,
                                             MmaLoadLoopA> *>(&dst_A);
      Array<ElementB, MmaLoadLoopB> const *ptr_B =
          reinterpret_cast<Array<ElementB, MmaLoadLoopB> const *>(&B);
      Array<typename ArchMmaOperator::ElementB, MmaLoadLoopB> *
          ptr_dst_B = reinterpret_cast<Array<typename ArchMmaOperator::ElementB,
                                             MmaLoadLoopB> *>(&dst_B);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentA::kElements / MmaLoadLoopA; i++) {
        ptr_dst_A[i] = convert_A(ptr_A[i]);
      }
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentB::kElements / MmaLoadLoopB; i++) {
        ptr_dst_B[i] = convert_B(ptr_B[i]);
      }
  }


  CUTLASS_DEVICE
  void transformA(TransformedFragmentA &dst_A,FragmentA &A) const {
    //
    // Define conversions from source type to instruction type
    //
    static_assert(sizeof_bits<ElementA>::value >= sizeof_bits<typename ArchMmaOperator::ElementA>::value,
                  "Can only transform from higher bits to lower.");
    FloatRoundStyle const kRoundA =
        PreferredRoundingMode<typename ArchMmaOperator::ElementA,
                              ElementA>::kRound;
    // should always transfer between 8 elements when instruction shape is 16/16/16
    detail::ConvertAndPack<typename ArchMmaOperator::ElementA, ElementA,
                          MmaLoadLoopA, kRoundA>
        convert_A;

    Array<ElementA, MmaLoadLoopA> const *ptr_A =
        reinterpret_cast<Array<ElementA, MmaLoadLoopA> const *>(&A);
    Array<typename ArchMmaOperator::ElementA, MmaLoadLoopA> *
        ptr_dst_A = reinterpret_cast<Array<typename ArchMmaOperator::ElementA,
                                            MmaLoadLoopA> *>(&dst_A);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < FragmentA::kElements / MmaLoadLoopA; i++) {
      ptr_dst_A[i] = convert_A(ptr_A[i]);
    }
  }

CUTLASS_DEVICE
  void transformB(TransformedFragmentB &dst_B,FragmentB &B) const {
    //
    // Define conversions from source type to instruction type
    //
    static_assert(sizeof_bits<ElementB>::value >= sizeof_bits<typename ArchMmaOperator::ElementB>::value,
                  "Can only transform from higher bits to lower.");
    FloatRoundStyle const kRoundB =
        PreferredRoundingMode<typename ArchMmaOperator::ElementB,
                              ElementB>::kRound;
    // should always transfer between 8 elements when instruction shape is 16/16/16
    detail::ConvertAndPack<typename ArchMmaOperator::ElementB, ElementB,
                          MmaLoadLoopB, kRoundB>
        convert_B;

    Array<ElementB, MmaLoadLoopB> const *ptr_B =
        reinterpret_cast<Array<ElementB, MmaLoadLoopB> const *>(&B);
    Array<typename ArchMmaOperator::ElementB, MmaLoadLoopB> *
        ptr_dst_B = reinterpret_cast<Array<typename ArchMmaOperator::ElementB,
                                            MmaLoadLoopB> *>(&dst_B);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < FragmentA::kElements / MmaLoadLoopB; i++) {
      ptr_dst_B[i] = convert_B(ptr_B[i]);
    }
  }

  /// Transform the mma operands to the required types
  CUTLASS_DEVICE
  void transformMmaA(TransformedMmaFragmentA &dst_A, MmaLoadedFragmentA &A) const {
    //
    // Define conversions from source type to instruction type
    //
    static_assert(sizeof_bits<ElementA>::value >= sizeof_bits<typename ArchMmaOperator::ElementA>::value,
                  "Can only transform from higher bits to lower.");

    FloatRoundStyle const kRoundA =
        PreferredRoundingMode<typename ArchMmaOperator::ElementA,
                              ElementA>::kRound;

      // should always transfer between 8 elements when instruction shape is 16/16/16
      detail::ConvertAndPack<typename ArchMmaOperator::ElementA, ElementA,
                            ArchMmaOperator::FragmentA::kElements, kRoundA>
          convert_A;

      dst_A = convert_A(A);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass/gemm/warp/mma_tensor_op_fast_f32.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
