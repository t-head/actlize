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
  \brief Epilogue for threadblock scoped GEMMs using Tensor Ops.

  The epilogue rearranges the result of a matrix product through shared memory to match canonical
  tensor layouts in global memory. Epilogues support conversion and reduction operations.

*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"

#include "cutlass/gemm/gemm.h"

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_clamp.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/epilogue/thread/linear_combination_gelu.h"
#include "cutlass/epilogue/thread/linear_combination_sigmoid.h"
#include "cutlass/epilogue/thread/linear_combination_planar_complex.h"

#include "cutlass/epilogue/thread/conversion_op.h"
#include "cutlass/epilogue/thread/reduction_op.h"

#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear.h"

#include "cutlass/epilogue/warp/fragment_iterator_tensor_op.h"
#include "cutlass/epilogue/warp/fragment_iterator_complex_tensor_op.h"
#include "cutlass/epilogue/warp/tile_iterator_tensor_op.h"
#include "cutlass/epilogue/warp/tile_iterator_tensor_op_mixed.h"
#include "cutlass/epilogue/threadblock/default_thread_map_tensor_op.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"
#include "cutlass/epilogue/threadblock/prefetch_tile_iterator.h"
#if SAIL_DGRAD_STRIDE_OPT
#include "cutlass/epilogue/threadblock/predicated_tile_iterator_strided.h"
#endif
#include "cutlass/epilogue/threadblock/shared_load_iterator.h"
#include "cutlass/epilogue/threadblock/shared_load_iterator_mixed.h"

#include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/epilogue/threadblock/interleaved_epilogue.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <
  typename ElementOutput,
  typename ElementAccumulator,
  int ElementsPerAccess,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename ThreadMap,
  bool Transpose = false,
  // conv transfer to output before store to smem, acc type is output type
  // use this to transfer actual acc type
  typename MmaElementOutput = ElementAccumulator
>
struct DefaultIteratorsTensorOp {

  using WarpTileIterator = typename platform::conditional<
    // transpose's fragment size is different, should add another TileIteratorTensorOpFP16Acc
    sizeof(MmaElementOutput) == 2 && Transpose == false,
    // FP16's acc is different to FP32, each lane store two successive elems
    cutlass::epilogue::warp::TileIteratorTensorOpFP16Acc<
      WarpShape,
      InstructionShape,
      ElementAccumulator
    >,
    cutlass::epilogue::warp::TileIteratorTensorOp<
      WarpShape,
      InstructionShape,
      ElementAccumulator,
      layout::RowMajor,
      Transpose
    >
  >::type;

  using SharedLoadIterator = cutlass::epilogue::threadblock::SharedLoadIterator<
    ThreadMap,
    ElementAccumulator
  >;

  static int const kFragmentsPerIteration = 1;
};

/// Partial specialization for float <= float x 4
template <
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename ThreadMap,
  bool Transpose
>
struct DefaultIteratorsTensorOp<float, float, 4, ThreadblockShape, WarpShape, InstructionShape, ThreadMap, Transpose> {

  using WarpTileIterator = cutlass::epilogue::warp::TileIteratorTensorOp<
    WarpShape,
    InstructionShape,
    float,
    layout::RowMajor,
    Transpose
  >;

  using SharedLoadIterator = cutlass::epilogue::threadblock::SharedLoadIterator<
    ThreadMap,
    float
  >;

  static int const kFragmentsPerIteration = 2;
};

/// Partial specialization for half <= float x 8 epilogues avoids shared memory bank conflicts.
template <
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename ThreadMap
>
struct DefaultIteratorsTensorOp<
  half_t,
  float,
  8,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  ThreadMap> {

  using WarpTileIterator = cutlass::epilogue::warp::TileIteratorTensorOpMixed<
    WarpShape,
    InstructionShape,
    float,
    32,
    16,
    8,
    8
  >;

  using SharedLoadIterator = cutlass::epilogue::threadblock::SharedLoadIteratorMixed<
    ThreadMap,
    float,
    32,
    16,
    8,
    8
  >;

  static int const kFragmentsPerIteration = 2;
};

/// Partial specialization for bf16 <= float x 8 epilogues avoids shared memory bank conflicts.
template <
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename ThreadMap
>
struct DefaultIteratorsTensorOp<
  bfloat16_t,
  float,
  8,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  ThreadMap> {

  using WarpTileIterator = cutlass::epilogue::warp::TileIteratorTensorOpMixed<
    WarpShape,
    InstructionShape,
    float,
    32,
    16,
    8,
    8
  >;

  using SharedLoadIterator = cutlass::epilogue::threadblock::SharedLoadIteratorMixed<
    ThreadMap,
    float,
    32,
    16,
    8,
    8
  >;

  static int const kFragmentsPerIteration = 2;
};

/// Partial specialization for int8_t x 16 <= int32_t x 16 epilogues avoids shared memory bank conflicts.
template <
  int K,
  typename InstructionShape,
  typename ThreadMap
>
struct DefaultIteratorsTensorOp<
  int8_t,
  int32_t,
  16,
  gemm::GemmShape<128, 128, K>,
  gemm::GemmShape<64, 64, K>,
  InstructionShape,
  ThreadMap> {

  using WarpTileIterator = cutlass::epilogue::warp::TileIteratorTensorOpMixed<
    gemm::GemmShape<64, 64, K>,
    InstructionShape,
    int32_t,
    32,
    8,
    16,
    8
  >;

  using SharedLoadIterator = cutlass::epilogue::threadblock::SharedLoadIteratorMixed<
    ThreadMap,
    int32_t,
    32,
    8,
    16,
    8
  >;

  static int const kFragmentsPerIteration = 1;
};

/// Partial specialization for int8_t x 8 <= int32_t x 8 epilogues avoids shared memory bank conflicts.
template <
  int K,
  typename InstructionShape,
  typename ThreadMap
>
struct DefaultIteratorsTensorOp<
  int8_t,
  int32_t,
  8,
  gemm::GemmShape<128, 64, K>,
  gemm::GemmShape<64, 32, K>,
  InstructionShape,
  ThreadMap> {

  using WarpTileIterator = cutlass::epilogue::warp::TileIteratorTensorOpMixed<
    gemm::GemmShape<64, 32, K>,
    InstructionShape,
    int32_t,
    32,
    8,
    8,
    8
  >;

  using SharedLoadIterator = cutlass::epilogue::threadblock::SharedLoadIteratorMixed<
    ThreadMap,
    int32_t,
    32,
    8,
    8,
    8
  >;

  static int const kFragmentsPerIteration = 1;
};

/// Partial specialization for int8_t x 8 <= int32_t x 8 epilogues avoids shared memory bank conflicts.
template <
  int K,
  typename InstructionShape,
  typename ThreadMap
>
struct DefaultIteratorsTensorOp<
  int8_t,
  int32_t,
  8,
  gemm::GemmShape<64, 64, K>,
  gemm::GemmShape<32, 32, K>,
  InstructionShape,
  ThreadMap> {

  using WarpTileIterator = cutlass::epilogue::warp::TileIteratorTensorOpMixed<
    gemm::GemmShape<32, 32, K>,
    InstructionShape,
    int32_t,
    32,
    8,
    8,
    8
  >;

  using SharedLoadIterator = cutlass::epilogue::threadblock::SharedLoadIteratorMixed<
    ThreadMap,
    int32_t,
    32,
    8,
    8,
    8
  >;

  static int const kFragmentsPerIteration = 1;
};

} // namespace detail

////////////////////////////////////////////////////////////////////////////////

/// Defines sensible defaults for epilogues for TensorOps.
template <
  typename Shape_,
  typename WarpMmaTensorOp_,
  int PartitionsK,
  typename OutputOp_,
  int ElementsPerAccess,
#if SAIL_DGRAD_STRIDE_OPT
// for conv dgrad with stride!=1
  int Strided_ = 0,
#endif
  // whether to do transpose in epilogue
  bool Transpose = false,
  // wgrad group conv output multiple groups combined flag
  bool MultGroup = false
>
struct DefaultEpilogueTensorOp {

  using Shape = Shape_;
  using WarpMmaTensorOp = WarpMmaTensorOp_;
  static int const kPartitionsK = PartitionsK;
  using OutputOp = OutputOp_;
  static int const kElementsPerAccess = ElementsPerAccess;

  using ElementOutput = typename OutputOp::ElementOutput;
  using LayoutC = typename WarpMmaTensorOp::LayoutC;
#if SAIL_EPILOGUE_OPT >= 2
  using ElementAccumulator = typename OutputOp::ElementAccumulator;
#else
  using ElementAccumulator = typename WarpMmaTensorOp::ElementC;
#endif

  //
  // Thread map
  //

  using OutputTileThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapTensorOp<
    Shape,
    typename WarpMmaTensorOp::Shape,
    kPartitionsK,
    ElementOutput,
    kElementsPerAccess,
    Transpose
  >::Type;

#if SAIL_DGRAD_STRIDE_OPT
  using OutputTileIterator = typename platform::conditional<
                              Strided_ == 1,
                              cutlass::epilogue::threadblock::PredicatedTileIteratorStrided<
                                OutputTileThreadMap,
                                ElementOutput
                              >,
                              cutlass::epilogue::threadblock::PredicatedTileIterator<
                                OutputTileThreadMap,
                                ElementOutput,
                                Transpose,
                                MultGroup
                              >
                            >::type;
#else
  using OutputTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
    OutputTileThreadMap,
    ElementOutput
  >;
#endif

  using AccumulatorFragmentIterator = typename platform::conditional<is_complex<ElementOutput>::value,
                                    cutlass::epilogue::warp::FragmentIteratorComplexTensorOp<
                                        typename WarpMmaTensorOp::Shape,
                                        typename WarpMmaTensorOp::Policy::Operator::Shape,
                                        typename WarpMmaTensorOp::Policy::Operator::ElementC,
                                        typename WarpMmaTensorOp::Policy::Operator::FragmentC,
                                        LayoutC>,
                                    cutlass::epilogue::warp::FragmentIteratorTensorOp<
                                        typename WarpMmaTensorOp::Shape,
                                        typename WarpMmaTensorOp::Policy::Operator::Shape,
                                        typename WarpMmaTensorOp::Policy::Operator::ElementC,
                                        typename WarpMmaTensorOp::Policy::Operator::FragmentC,
                                        LayoutC,
                                        Transpose
#if SAIL_EPILOGUE_OPT >= 2
                                        , ElementAccumulator
#endif
                                        > >::type;

  /// Support several implementations depending on structure of epilogue
  using DefaultIterators = detail::DefaultIteratorsTensorOp<
    ElementOutput,
// #if SAIL_EPILOGUE_OPT >= 2
//     // already converted
//     ElementOutput,
// #else
    ElementAccumulator,
// #endif
    kElementsPerAccess,
    Shape,
    typename WarpMmaTensorOp::Shape,
    typename WarpMmaTensorOp::Policy::Operator::Shape,
    typename OutputTileThreadMap::CompactedThreadMap,
    Transpose,
    // conv transfer to output before store to smem, ElementAccumulator is output type
    // use this to transfer actual acc type to choose fp16 tsm iter when acc is fp16
    typename WarpMmaTensorOp::Policy::Operator::ElementC
  >;

  using WarpTileIterator = typename DefaultIterators::WarpTileIterator;
  using SharedLoadIterator = typename DefaultIterators::SharedLoadIterator;

  /// Hard-coded padding elements added
#if ACOMPUTE_VERSION == 10000
  // always padding 4 for col major
  using Padding = typename platform::conditional<
                    Transpose == true,
                    // 0/4/8/12/16/20/24/48 store 8 elements, 1/5/9/13/17/21/25/29 should interleave 8 elements
                    cutlass::MatrixShape<0, 8>,
                    // ppu four thread store four elements once
                    // ppu's tsm_load_B32x4 needn't tsm address aligned?
                    cutlass::MatrixShape<0, 4>
                  >::type;
#else
  // always padding 4 for col major
  using Padding = typename platform::conditional<
                    Transpose == true,
                    typename platform::conditional<
                      sizeof_bits<ElementAccumulator>::value == 16,
                      // fp16 can't padding 8, otherwise start add of second row will be misaligned
                      // still have no bank conflict, since each lane only write 16b, 32 lane will only use 16 bank each loop
                      cutlass::MatrixShape<0, 8>,
                      // each lane store successive two rows, t0 and t1 only interleave 8/2=4 elements
                      cutlass::MatrixShape<0, 4>
                      >::type,
                    // each lane store 2 elements once, 4 lane store 8 elements
                    cutlass::MatrixShape<0, 8>
                  >::type;
#endif

  static int const kFragmentsPerIteration = (kPartitionsK == 1 ? DefaultIterators::kFragmentsPerIteration : 1);

  //
  // Define the epilogue
  //
  using Epilogue = cutlass::epilogue::threadblock::Epilogue<
    Shape,
    WarpMmaTensorOp,
    kPartitionsK,
    OutputTileIterator,
    AccumulatorFragmentIterator,
    WarpTileIterator,
    SharedLoadIterator,
    OutputOp,
    Padding,
    kFragmentsPerIteration,
    Transpose
  >;
};

// Add arch for acompute 1.0/1.5 build, host no ACOMPUTE_VERSION
template <
  typename Shape_,
  typename WarpMmaTensorOp_,
  int PartitionsK,
  typename OutputOp_,
  int ElementsPerAccess,
  typename ArchTag,
#if SAIL_DGRAD_STRIDE_OPT
// for conv dgrad with stride!=1
  int Strided_ = 0,
#endif
  // whether to do transpose in epilogue
  bool Transpose = false,
  // wgrad group conv output multiple groups combined flag
  bool MultGroup = false
>
struct DefaultEpilogueTensorOpArch {

  using Shape = Shape_;
  using WarpMmaTensorOp = WarpMmaTensorOp_;
  static int const kPartitionsK = PartitionsK;
  using OutputOp = OutputOp_;
  static int const kElementsPerAccess = ElementsPerAccess;

  using ElementOutput = typename OutputOp::ElementOutput;
  using LayoutC = typename WarpMmaTensorOp::LayoutC;
#if SAIL_EPILOGUE_OPT >= 2
  using ElementAccumulator = typename OutputOp::ElementAccumulator;
#else
  using ElementAccumulator = typename WarpMmaTensorOp::ElementC;
#endif

  //
  // Thread map
  //

  using OutputTileThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapTensorOp<
    Shape,
    typename WarpMmaTensorOp::Shape,
    kPartitionsK,
    ElementOutput,
    kElementsPerAccess,
    Transpose
  >::Type;

#if SAIL_DGRAD_STRIDE_OPT
  using OutputTileIterator = typename platform::conditional<
                              Strided_ == 1,
                              cutlass::epilogue::threadblock::PredicatedTileIteratorStrided<
                                OutputTileThreadMap,
                                ElementOutput
                              >,
                              cutlass::epilogue::threadblock::PredicatedTileIterator<
                                OutputTileThreadMap,
                                ElementOutput,
                                Transpose,
                                MultGroup
                              >
                            >::type;
#else
  using OutputTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
    OutputTileThreadMap,
    ElementOutput
  >;
#endif

  using AccumulatorFragmentIterator = typename platform::conditional<is_complex<ElementOutput>::value,
                                    cutlass::epilogue::warp::FragmentIteratorComplexTensorOp<
                                        typename WarpMmaTensorOp::Shape,
                                        typename WarpMmaTensorOp::Policy::Operator::Shape,
                                        typename WarpMmaTensorOp::Policy::Operator::ElementC,
                                        typename WarpMmaTensorOp::Policy::Operator::FragmentC,
                                        LayoutC>,
                                    cutlass::epilogue::warp::FragmentIteratorTensorOp<
                                        typename WarpMmaTensorOp::Shape,
                                        typename WarpMmaTensorOp::Policy::Operator::Shape,
                                        typename WarpMmaTensorOp::Policy::Operator::ElementC,
                                        typename WarpMmaTensorOp::Policy::Operator::FragmentC,
                                        LayoutC,
                                        Transpose
#if SAIL_EPILOGUE_OPT >= 2
                                        , ElementAccumulator
#endif
                                        > >::type;

  /// Support several implementations depending on structure of epilogue
  using DefaultIterators = detail::DefaultIteratorsTensorOp<
    ElementOutput,
// #if SAIL_EPILOGUE_OPT >= 2
//     // already converted
//     ElementOutput,
// #else
    ElementAccumulator,
// #endif
    kElementsPerAccess,
    Shape,
    typename WarpMmaTensorOp::Shape,
    typename WarpMmaTensorOp::Policy::Operator::Shape,
    typename OutputTileThreadMap::CompactedThreadMap,
    Transpose,
    // conv transfer to output before store to smem, ElementAccumulator is output type
    // use this to transfer actual acc type to choose fp16 tsm iter when acc is fp16
    typename WarpMmaTensorOp::Policy::Operator::ElementC
  >;

  using WarpTileIterator = typename DefaultIterators::WarpTileIterator;
  using SharedLoadIterator = typename DefaultIterators::SharedLoadIterator;

  
  using Padding_PPU0010 = typename platform::conditional<
                    Transpose == true,
                    // 0/4/8/12/16/20/24/48 store 8 elements, 1/5/9/13/17/21/25/29 should interleave 8 elements
                    cutlass::MatrixShape<0, 8>,
                    // ppu four thread store four elements once
                    // ppu's tsm_load_B32x4 needn't tsm address aligned?
                    cutlass::MatrixShape<0, 4>
                  >::type;
  using Padding_Default = typename platform::conditional<
                    Transpose == true,
                    typename platform::conditional<
                      sizeof_bits<ElementAccumulator>::value == 16,
                      // fp16 can't padding 8, otherwise start add of second row will be misaligned
                      // still have no bank conflict, since each lane only write 16b, 32 lane will only use 16 bank each loop
                      cutlass::MatrixShape<0, 8>,
                      // each lane store successive two rows, t0 and t1 only interleave 8/2=4 elements
                      cutlass::MatrixShape<0, 4>
                      >::type,
                    // each lane store 2 elements once, 4 lane store 8 elements
                    cutlass::MatrixShape<0, 8>
                  >::type;

#if defined(__HGGCCC_RTC__)
#if ACOMPUTE_VERSION == 10000 // ppu1.0 rtc
  using Padding = Padding_PPU0010;
#else // ppu1.5 rtc
  using Padding = Padding_Default;
#endif
#else
  // host no ACOMPUTE_VERSION: use ArchTag to judge
  using Padding = typename platform::conditional<ArchTag::kMinComputeCapability <= 80, Padding_PPU0010, Padding_Default>::type;
#endif

  static int const kFragmentsPerIteration = (kPartitionsK == 1 ? DefaultIterators::kFragmentsPerIteration : 1);

  //
  // Define the epilogue
  //
  using Epilogue = cutlass::epilogue::threadblock::Epilogue<
    Shape,
    WarpMmaTensorOp,
    kPartitionsK,
    OutputTileIterator,
    AccumulatorFragmentIterator,
    WarpTileIterator,
    SharedLoadIterator,
    OutputOp,
    Padding,
    kFragmentsPerIteration,
    Transpose
  >;
};

////////////////////////////////////////////////////////////////////////////////

/// Defines sensible defaults for epilogues for TensorOps which uses
/// intereleaved output layout. For this case, shared memory is not needed.
template <typename Shape_, typename WarpMmaTensorOp_, int PartitionsK,
          typename OutputOp_, int ElementsPerAccess, int InterleavedK,
          bool isSplitK = false>
struct DefaultInterleavedEpilogueTensorOp {
  using Shape = Shape_;
  using WarpMmaTensorOp = WarpMmaTensorOp_;
  static int const kPartitionsK = PartitionsK;
  using OutputOp = OutputOp_;
  static int const kElementsPerAccess = ElementsPerAccess;

  using ElementOutput = typename OutputOp::ElementOutput;
  using LayoutC = typename WarpMmaTensorOp::LayoutC;
  using ElementAccumulator = typename WarpMmaTensorOp::ElementC;

  //
  // Thread map
  //
  using OutputTileThreadMap = typename cutlass::epilogue::threadblock::
      DefaultInterleavedThreadMapTensorOp<
          Shape, typename WarpMmaTensorOp::Shape, kPartitionsK, ElementOutput,
          kElementsPerAccess, InterleavedK>::Type;

  using OutputTileIterator =
      cutlass::epilogue::threadblock::InterleavedPredicatedTileIterator<
          OutputTileThreadMap, ElementOutput, InterleavedK>;

  using AccumulatorFragmentIterator =
      cutlass::epilogue::warp::FragmentIteratorTensorOp<
          typename WarpMmaTensorOp::Shape,
          typename WarpMmaTensorOp::Policy::Operator::Shape,
          typename WarpMmaTensorOp::Policy::Operator::ElementC,
          typename WarpMmaTensorOp::Policy::Operator::FragmentC,
          LayoutC>;

  //
  // Define the epilogue
  //
  using Epilogue = cutlass::epilogue::threadblock::InterleavedEpilogue<
      Shape, WarpMmaTensorOp, kPartitionsK, OutputTileIterator,
      AccumulatorFragmentIterator, OutputOp, InterleavedK>;
};

////////////////////////////////////////////////////////////////////////////////

/// Defines sensible defaults for epilogues for TensorOps which uses
/// intereleaved output layout. For this case, shared memory is not needed.
template <typename Shape_, typename WarpMmaTensorOp_, int PartitionsK,
          typename OutputOp_, int ElementsPerAccess, int InterleavedK,
          bool isSplitK = false>
struct DefaultInterleavedConvEpilogue {
  using Shape = Shape_;
  using WarpMmaTensorOp = WarpMmaTensorOp_;
  static int const kPartitionsK = PartitionsK;
  using OutputOp = OutputOp_;
  static int const kElementsPerAccess = ElementsPerAccess;

  using ElementOutput = typename OutputOp::ElementOutput;
  using ElementAccumulator = typename WarpMmaTensorOp::ElementC;

  //
  // Thread map
  //
  using OutputTileThreadMap = typename cutlass::epilogue::threadblock::
      DefaultInterleavedConvThreadMapTensorOp<
          Shape, typename WarpMmaTensorOp::Shape, kPartitionsK, ElementOutput,
          kElementsPerAccess, InterleavedK>::Type;

  using OutputTileIterator =
      cutlass::epilogue::threadblock::InterleavedConvPredicatedTileIterator<
          OutputTileThreadMap, ElementOutput, InterleavedK>;

  using AccumulatorFragmentIterator =
      cutlass::epilogue::warp::FragmentIteratorTensorOp<
          typename WarpMmaTensorOp::Shape,
          typename WarpMmaTensorOp::Policy::Operator::Shape,
          typename WarpMmaTensorOp::Policy::Operator::ElementC,
          typename WarpMmaTensorOp::Policy::Operator::FragmentC,
          // can reuse the gemm version here to do element selection
          layout::ColumnMajorInterleaved<InterleavedK>>;

  //
  // Define the epilogue
  //
  using Epilogue = cutlass::epilogue::threadblock::InterleavedEpilogue<
      Shape, WarpMmaTensorOp, kPartitionsK, OutputTileIterator,
      AccumulatorFragmentIterator, OutputOp, InterleavedK>;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
