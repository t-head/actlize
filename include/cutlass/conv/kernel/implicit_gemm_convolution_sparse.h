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
    \brief Template for a pipelined Implicit GEMM kernel.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/aligned_buffer.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/semaphore.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/conv3d_problem_size.h"
#include "cutlass/epilogue/threadblock/output_iterator_parameter.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Mma_,                                  ///! Threadblock-scoped matrix multiply-accumulate
  typename Epilogue_,                             ///! Epilogue
  typename ThreadblockSwizzle_,                   ///! Threadblock swizzling function
  conv::Operator ConvOperator,                    ///! Convolutional operator (Fprop, Dgrad, Wgrad)
  typename ConvProblemSize_ = Conv2dProblemSize   ///! Convolutional operator on 2D or 3D problem
#ifdef SAIL_CUSTOMIZE_CUTLASS
  , gemm::Operand CompressOp_ = gemm::Operand::kA             ///! Sparse is used for compression matrix.
#endif
  , bool SplitKSerial = false
>
struct ImplicitGemmConvolutionSparse {
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueOutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static Operator const kConvolutionalOperator = ConvOperator;

  using ElementA = typename Mma::IteratorA::Element;
  using LayoutA = typename Mma::IteratorA::Layout;
  using ElementB = typename Mma::IteratorB::Element;
  using LayoutB = typename Mma::IteratorB::Layout;
  using ElementC = typename EpilogueOutputOp::ElementOutput;

  /// Set output tensor C layout
  using LayoutC = LayoutA;

  /// Set sparse meta data infomation.
  static int const kSparse = Mma::kSparse;
  static int const kMetaSizeInBits = Mma::kMetaSizeInBits;
  static int const kMaxID2 = Mma::kMaxID2;
  static int const kElementsPerElementE = Mma::kElementsPerElementE;

  using ElementE = typename Mma::ElementE;
  using LayoutE = typename Mma::LayoutE;

  #if SAIL_FUSE_OP_EXT
  using ElementFuseInExtra = ElementC;
  using LayoutExtra = LayoutC;
  #endif

  using ElementAccumulator = typename EpilogueOutputOp::ElementAccumulator;
  using ElementCompute = typename EpilogueOutputOp::ElementCompute;

  using WarpMmaOperator = typename Mma::Policy::Operator;

  using ArchMmaOperator = typename WarpMmaOperator::ArchMmaOperator;
  using MathOperator = typename ArchMmaOperator::Operator;

  using OperatorClass = typename WarpMmaOperator::OperatorClass;
  using ArchTag = typename WarpMmaOperator::ArchTag;

  using ThreadblockShape = typename Mma::Shape;
  using WarpShape = typename WarpMmaOperator::Shape;
  using InstructionShape = typename ArchMmaOperator::Shape;

  static int const kStages = Mma::kStages;
  static IteratorAlgorithm const kIteratorAlgorithm = Mma::IteratorA::kIteratorAlgorithm;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  using TensorRefA = typename Mma::IteratorA::TensorRef;
  using TensorRefB = typename Mma::IteratorB::TensorRef;
  using TensorRefE = typename Mma::IteratorE::TensorRef;
  using TensorRefC = cutlass::TensorRef<ElementC, LayoutC>;
  #if SAIL_FUSE_OP_EXT
  using TensorRefExtra = cutlass::TensorRef<ElementFuseInExtra, LayoutExtra>;
  #endif

  /// Check iterator A and B convolution dimension are the same and
  // set device::ImplicitGemmConvolution::kConvDim
  static_assert(Mma::IteratorA::kConvDim == Mma::IteratorB::kConvDim,
    "Convolution on different different dimensions is not supported");
  static int const kConvDim = Mma::IteratorA::kConvDim;

  /// Conv dimension and problem size structure (Conv2d or Conv3d)
  using ConvProblemSize = ConvProblemSize_;

  /// Wgrad C stride idx for implicit gemm algorithm
  // Conv2d row-major matrix C (KxRSC)
  // Conv3d row-major matrix C (KxTRSC)
  static int const kWgradCStrideIdx =
    cutlass::platform::is_same<LayoutC, cutlass::layout::TensorNHWC>::value ? 2 : 3;

  /// This chooses the appropriate stride element of the C tensor.
  static int const kTensorCStrideIdx =
    (kConvolutionalOperator == conv::Operator::kWgrad ? kWgradCStrideIdx : 0);

  #if SAIL_FUSE_OP_EXT
  static int const kExtraInputNum = EpilogueOutputOp::kExtraEpilogueInputs > 0 ? EpilogueOutputOp::kExtraEpilogueInputs : 1;
  static int const kExtraInputLoopNum = cutlass::epilogue::GetExtraEpilogueBinaryInputs<EpilogueOutputOp>::value;
  #endif
  //
  //
  //
  using ConvOutputIteratorParameter = epilogue::threadblock::ConvOutputIteratorParameter<
    LayoutC,
    typename Epilogue::OutputTileIterator::Layout,
    TensorRefC,
    ConvOperator,
    ConvProblemSize
    >;

  static KernelType const KType = KernelType::kSparse;

  static bool const kSplitKSerial = SplitKSerial;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    ConvProblemSize problem_size;
    TensorRefA ref_A;
    TensorRefB ref_B;
    TensorRefC ref_C;
    TensorRefC ref_D;
    TensorRefE ref_E;
    #if SAIL_FUSE_OP_EXT
    TensorRefExtra ref_Extra[kExtraInputNum];
    #endif
    typename EpilogueOutputOp::Params output_op;
    SplitKMode split_k_mode;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments() { }

    CUTLASS_HOST_DEVICE
    Arguments(
      ConvProblemSize const & problem_size
    ):
      problem_size(problem_size) { }

    CUTLASS_HOST_DEVICE
    Arguments(
      ConvProblemSize const & problem_size,
      TensorRefA const & ref_A,
      TensorRefB const & ref_B,
      TensorRefC const & ref_C,
      TensorRefC const & ref_D,
      TensorRefE const & ref_E,
      typename EpilogueOutputOp::Params const & output_op,
      SplitKMode const & split_k_mode = SplitKMode::kSerial
      #if SAIL_FUSE_OP_EXT
      , TensorRefExtra *pref_Extra = nullptr
      #endif
    ):
      problem_size(problem_size),
      ref_A(ref_A),
      ref_B(ref_B),
      ref_C(ref_C),
      ref_D(ref_D),
      ref_E(ref_E),
      output_op(output_op),
      split_k_mode(split_k_mode)
    {
      #if SAIL_FUSE_OP_EXT
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kExtraInputLoopNum; i++) {
        if (pref_Extra) {
          ref_Extra[i] = pref_Extra[i];
        } else {
          ref_Extra[i] = TensorRefExtra(nullptr);
        }
      }
      #endif
    }

  };

  /// Parameters structure
  struct Params {
    ConvProblemSize problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    gemm::GemmCoord implicit_gemm_problem_size;
    int grid_shift; // grid dim shifts from y to x;
    int gemm_k_iterations;
    typename Mma::IteratorA::Params iterator_A;
    typename Mma::IteratorA::Element const *ptr_A;
    typename Mma::IteratorB::Params iterator_B;
    typename Mma::IteratorB::Element const *ptr_B;
    typename Mma::IteratorE::Params iterator_E;
    typename Mma::IteratorE::Element *ptr_E;
    typename Epilogue::OutputTileIterator::Params iterator_C;
    typename Epilogue::OutputTileIterator::Element *ptr_C;
    typename Epilogue::OutputTileIterator::Params iterator_D;
    typename Epilogue::OutputTileIterator::Element *ptr_D;
    #if SAIL_FUSE_OP_EXT
    typename Epilogue::OutputTileIterator::Params iterator_Extra[kExtraInputNum];
    ElementFuseInExtra *ptr_Extra[kExtraInputNum];
    #endif
    typename EpilogueOutputOp::Params output_op;
    int *semaphore;
    SplitKMode split_k_mode;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params(): grid_shift(0), gemm_k_iterations(0) { }

    ///
    CUTLASS_HOST_DEVICE
    Params(
      Arguments const &args,
      int *semaphore = nullptr
    ):
      problem_size(args.problem_size),
      implicit_gemm_problem_size(cutlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, args.problem_size)),
      grid_shift(0),
      iterator_A(Mma::IteratorA::getParams(args.problem_size, args.ref_A.layout())),
      ptr_A(args.ref_A.data()),
      iterator_B(args.problem_size, args.ref_B.layout()),
      ptr_B(args.ref_B.data()),
      iterator_C(ConvOutputIteratorParameter::layout(args.ref_C)),
      ptr_C(args.ref_C.data()),
      iterator_D(ConvOutputIteratorParameter::layout(args.ref_D)),
      ptr_D(args.ref_D.data()),
      iterator_E(args.ref_E.layout()),
      ptr_E(args.ref_E.data()),
      output_op(args.output_op),
      semaphore(semaphore),
      split_k_mode(args.split_k_mode)
    {
      gemm_k_iterations = implicit_gemm_k_iterations(kConvolutionalOperator, ThreadblockShape::kK, args.problem_size);

      ThreadblockSwizzle threadblock_swizzle;

      grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
        implicit_gemm_problem_size,
        {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
        args.problem_size.split_k_slices);

      #if SAIL_FUSE_OP_EXT
      CUTLASS_PRAGMA_UNROLL
      for (int i =0; i < kExtraInputLoopNum; i++) {
        iterator_Extra[i] = ConvOutputIteratorParameter::layout(args.ref_Extra[i]);
        ptr_Extra[i] = args.ref_Extra[i].data();
      }
      #endif
    }
  };

#if SAIL_CUSTOMIZE_CUTLASS && !defined(__HGGCCC_RTC__)
  /// Argument structure
  struct LaunchConfigs {
    dim3  grid;
    dim3  block;
    int   smem_bytes;
    int   regs_per_thread;  // in b32
    int   stack_per_thread; // in bytes

    LaunchConfigs(
      dim3 &grid,
      dim3 &block,
      int   smem_bytes,
      int   regs_per_thread,
      int   stack_per_thread
    ) :
      grid(grid),
      block(block),
      smem_bytes(smem_bytes),
      regs_per_thread(regs_per_thread),
      stack_per_thread(stack_per_thread)
    {

    }
  };
#endif

  /// Shared memory storage structure
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  ImplicitGemmConvolutionSparse() { }

  /// Determines whether kernel satisfies alignment
  CUTLASS_HOST_DEVICE
  static Status can_implement(
      ConvProblemSize const & problem_size,
      typename Mma::IteratorA::TensorRef ref_A,
      typename Mma::IteratorB::TensorRef ref_B
      #if SAIL_FUSE_OP_EXT
      , TensorRefExtra (&ref_Extra_NC)[kExtraInputNum]
      #endif
      ) {

    static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
    static int const kAlignmentMajor = (CompressOp_ == gemm::Operand::kB) ? kAlignmentB : kAlignmentA;

    if (!TensorRef_aligned(ref_A, max(1, kAlignmentA / problem_size.fold_num))) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_B, max(1, kAlignmentB / problem_size.fold_num))) {
      return Status::kErrorMisalignedOperand;
    }

    auto gemm_size = cutlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, problem_size);

    if (((gemm_size.k() / kSparse) % kAlignmentB) || (gemm_size.k() % kAlignmentB)) {
      return Status::kErrorMisalignedOperand;
    }

    // The k dimension has to be the multiple of the Threadblock k because out
    // of bound meta data would be initialized to 0 by acync.zfill but 0 is not
    // a valid meta data.
    if (gemm_size.k() % Mma::Shape::kK) {
      return Status::kErrorMisalignedOperand;
    }

    // M dimension has to be multiple of 32 (sparse float) or 16 (sparse int)
    // because of the row reordering of operand E
    static int const kAlignmentM = (sizeof(ElementE) == 2) ? 32 : 16;

#ifdef SAIL_CUSTOMIZE_CUTLASS
    if (((CompressOp_ == gemm::Operand::kB) ? gemm_size.n() : gemm_size.m()) % kAlignmentM) {
      return Status::kErrorMisalignedOperand;
    }
#else
    if (gemm_size.m() % kAlignmentM) {
      return Status::kErrorMisalignedOperand;
    }
#endif

#if SAIL_FUSE_OP_EXT
    static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kExtraInputLoopNum; i++) {
      if (ref_Extra_NC[i].good() && !TensorRef_aligned(ref_Extra_NC[i], kAlignmentC)) {
          return Status::kErrorMisalignedOperand;
      }
    }
#endif

    return Status::kSuccess;
  }

  /// Executes one ImplicitGEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    // Compute threadblock location
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_idx =
        threadblock_swizzle.get_tile_offset(params.grid_tiled_shape, params.grid_shift);

    // Early exit if CTA is out of range
    if (params.grid_tiled_shape.m() <= threadblock_tile_idx.m() ||
      params.grid_tiled_shape.n() <= threadblock_tile_idx.n()) {

      return;
    }

    // Compute position within threadblock
    int thread_idx = threadIdx.x;

#ifdef SAIL_CUSTOMIZE_CUTLASS
    // Construct iterators to A and B operands
    typename Mma::IteratorA iterator_A(
      params.iterator_A,
      params.problem_size,
      params.ptr_A,
      thread_idx,
      MatrixCoord(
        threadblock_tile_idx.m() * Mma::Shape::kM,
      (CompressOp_ == gemm::Operand::kB)
      ? threadblock_tile_idx.k() * Mma::Shape::kK
      : threadblock_tile_idx.k() * Mma::Shape::kK / kSparse
      )
    );

    typename Mma::IteratorB iterator_B(
      params.iterator_B,
      params.problem_size,
      params.ptr_B,
      thread_idx,
      MatrixCoord(
      (CompressOp_ == gemm::Operand::kB)
      ? threadblock_tile_idx.k() * Mma::Shape::kK / kSparse
      : threadblock_tile_idx.k() * Mma::Shape::kK,
        threadblock_tile_idx.n() * Mma::Shape::kN
      ),
      kSparse
    );

    // Problem size is a function of threadblock index in the K dimension
    int problem_size_k = min(
      params.implicit_gemm_problem_size.k(),
      (threadblock_tile_idx.k() + 1) * params.gemm_k_iterations * Mma::Shape::kK);

    cutlass::MatrixCoord tb_offset_E{
      (CompressOp_ == gemm::Operand::kB) ? threadblock_tile_idx.k() * problem_size_k / kSparse : threadblock_tile_idx.m() * Mma::Shape::kM,
      (CompressOp_ == gemm::Operand::kB) ? threadblock_tile_idx.n() * Mma::Shape::kN : threadblock_tile_idx.k() * problem_size_k / kSparse,
    };

    typename Mma::IteratorE iterator_E(
      params.iterator_E,
      params.ptr_E,
      {(CompressOp_ == gemm::Operand::kB) ? problem_size_k / kSparse / kElementsPerElementE : params.implicit_gemm_problem_size.m(),
       (CompressOp_ == gemm::Operand::kB) ? params.implicit_gemm_problem_size.n() : problem_size_k / kSparse / kElementsPerElementE},
      thread_idx, tb_offset_E);
#else
    // Construct iterators to A and B operands
    typename Mma::IteratorA iterator_A(
      params.iterator_A,
      params.problem_size,
      params.ptr_A,
      thread_idx,
      MatrixCoord(
        threadblock_tile_idx.m() * Mma::Shape::kM,
        threadblock_tile_idx.k() * Mma::Shape::kK / kSparse
      )
    );

    typename Mma::IteratorB iterator_B(
      params.iterator_B,
      params.problem_size,
      params.ptr_B,
      thread_idx,
      MatrixCoord(
        threadblock_tile_idx.k() * Mma::Shape::kK,
        threadblock_tile_idx.n() * Mma::Shape::kN
      )
    );

    // Problem size is a function of threadblock index in the K dimension
    int problem_size_k = min(
      params.implicit_gemm_problem_size.k(),
      (threadblock_tile_idx.k() + 1) * params.gemm_k_iterations * Mma::Shape::kK);

    typename Mma::IteratorE iterator_E(
      params.iterator_E,
      params.ptr_E,
      {params.implicit_gemm_problem_size.m(),
       problem_size_k / kSparse / kElementsPerElementE},
      thread_idx,
      MatrixCoord(
        threadblock_tile_idx.m() * Mma::Shape::kM,
        threadblock_tile_idx.k() * Mma::Shape::kK / kSparse
      )
    );
#endif

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    int lane_idx = threadIdx.x % 32;

    //
    // Main loop
    //

    // Construct thread-scoped matrix multiply
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    typename Mma::FragmentC accumulators;

    accumulators.clear();

    // Compute threadblock-scoped matrix multiply-add
    mma(params.gemm_k_iterations, accumulators, iterator_A, iterator_B, iterator_E, accumulators);

    //
    // Epilogue
    //

    EpilogueOutputOp output_op(params.output_op);

    // Construct the semaphore.
    int block_idx = threadblock_tile_idx.m() + threadblock_tile_idx.n() * params.grid_tiled_shape.m();

    Semaphore semaphore(params.semaphore + block_idx, thread_idx);

    // Compute logical position within grid
    threadblock_tile_idx =
        threadblock_swizzle.get_tile_offset(params.grid_tiled_shape, params.grid_shift);

    // If performing a reduction via split-K, fetch the initial synchronization
    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {

      // Fetch the synchronization lock initially but do not block.
      semaphore.fetch();

      // Indicate which position in a serial reduction the output operator is currently updating
      output_op.set_k_partition(threadblock_tile_idx.k(), params.grid_tiled_shape.k());
    }

    MatrixCoord threadblock_offset(
      threadblock_tile_idx.m() * Mma::Shape::kM,
      threadblock_tile_idx.n() * Mma::Shape::kN
    );

    // Tile iterator writing to destination tensor
    typename Epilogue::OutputTileIterator iterator_D(
      params.iterator_D,
      params.ptr_D,
      ConvOutputIteratorParameter::extent(params.problem_size),
      thread_idx,
      threadblock_offset
    );

    // Tile iterator reading from source accumulator tensor
    typename Epilogue::OutputTileIterator iterator_C(
      params.iterator_C,
      params.ptr_C,
      ConvOutputIteratorParameter::extent(params.problem_size),
      thread_idx,
      threadblock_offset
    );


    // Construct the epilogue
    Epilogue epilogue(
      shared_storage.epilogue,
      thread_idx,
      warp_idx,
      lane_idx);

    // Wait on the semaphore - this latency may have been covered by iterator construction
    if (params.split_k_mode == SplitKMode::kSerial && params.grid_tiled_shape.k() > 1) {

      // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
      if (threadblock_tile_idx.k()) {
        iterator_C = iterator_D;
      }

      semaphore.wait(threadblock_tile_idx.k());

      __threadfence();
    }
    // Each split-k-slice writes to a unique tensor location
    else if (params.split_k_mode == SplitKMode::kParallel) {
      iterator_D.add_pointer_offset(threadblock_tile_idx.k() *
        cutlass::conv::implicit_gemm_tensor_c_size(ConvOperator, params.problem_size));
    }

    // Run efficient epilogue
    #if SAIL_FUSE_OP_EXT
    typename Epilogue::OutputTileIterator iterator_Extra[kExtraInputNum];

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kExtraInputLoopNum; i++) {
      iterator_Extra[i] = { params.iterator_Extra[i],
                            reinterpret_cast<typename Epilogue::OutputTileIterator::Element*>(params.ptr_Extra[i]),
                            ConvOutputIteratorParameter::extent(params.problem_size),
                            thread_idx,
                            threadblock_offset
                          };
    }
    epilogue.runEpilogue(output_op, iterator_D, accumulators, iterator_C, iterator_Extra);
    #else
    epilogue(output_op, iterator_D, accumulators, iterator_C);
    #endif

    //
    // Release the semaphore
    //

    if (params.split_k_mode == SplitKMode::kSerial && params.grid_tiled_shape.k() > 1) {

      int lock = 0;
      if (params.grid_tiled_shape.k() == threadblock_tile_idx.k() + 1) {

        // The final threadblock resets the semaphore for subsequent grids.
        lock = 0;
      }
      else {
        // Otherwise, the semaphore is incremented
        lock = threadblock_tile_idx.k() + 1;
      }

      semaphore.release(lock);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

