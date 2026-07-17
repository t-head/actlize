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
#if SAIL_DGRAD_STRIDE_OPT
  , bool UseAsync = true
  , bool DgradStrided_ = false
#endif
  , typename ElementC_ = typename Epilogue_::OutputOp::ElementOutput,
  bool SplitKSerial = false
>
struct ImplicitGemmConvolutionFusion {

  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueOutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static Operator const kConvolutionalOperator = ConvOperator;

  using ElementA = typename Mma::IteratorA::Element;
  using LayoutA = typename Mma::IteratorA::Layout;
  using ElementB = typename Mma::IteratorB::Element;
  using LayoutB = typename Mma::IteratorB::Layout;

  using ElementScaleBias = typename Mma::IteratorScaleBias::Element;
  using LayoutScaleBias = typename Mma::IteratorScaleBias::Layout;

  // using ElementC = typename EpilogueOutputOp::ElementOutput;
  using ElementC = ElementC_;

  /// Set output tensor C layout
  // using LayoutC = LayoutA;
  // TODO: in/out layout may not same now, since only output support transpose
  using LayoutC = typename platform::conditional<
                    platform::is_same<typename Epilogue::AccumulatorFragmentIterator::Layout, cutlass::layout::ColumnMajor>::value,
                    // TODO: only 2d will use transpose now, 3d should separate nchw/ncdhw
                    layout::TensorNCHW,
                    LayoutA
                  >::type;
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
  using TensorRefScaleBias = typename Mma::IteratorScaleBias::TensorRef;
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

  static KernelType const KType = KernelType::kNormal;

  static bool const kSplitKSerial = SplitKSerial;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    ConvProblemSize problem_size;
    TensorRefA ref_A;
    TensorRefB ref_B;
    TensorRefScaleBias ref_scale;
    TensorRefScaleBias ref_bias;
    TensorRefC ref_C;
    TensorRefC ref_D;
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
      TensorRefScaleBias const & ref_scale,
      TensorRefScaleBias const & ref_bias,
      TensorRefC const & ref_C,
      TensorRefC const & ref_D,
      typename EpilogueOutputOp::Params const & output_op,
      SplitKMode const & split_k_mode = SplitKMode::kSerial
      #if SAIL_FUSE_OP_EXT
      , TensorRefExtra *pref_Extra = nullptr
      #endif
    ):
      problem_size(problem_size),
      ref_A(ref_A),
      ref_B(ref_B),
      ref_scale(ref_scale),
      ref_bias(ref_bias),
      ref_C(ref_C),
      ref_D(ref_D),
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
    typename Mma::IteratorScaleBias::Params iterator_scale_bias;
    typename Mma::IteratorScaleBias::Element const *ptr_scale;
    typename Mma::IteratorScaleBias::Element const *ptr_bias;
    typename Epilogue::OutputTileIterator::Params iterator_C;
    ElementC *ptr_C;
    typename Epilogue::OutputTileIterator::Params iterator_D;
    ElementC *ptr_D;
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
      iterator_scale_bias(args.problem_size, args.ref_scale.layout()),
      ptr_scale(args.ref_scale.data()),
      ptr_bias(args.ref_bias.data()),
      iterator_C(ConvOutputIteratorParameter::layout(args.ref_C)),
      ptr_C(args.ref_C.data()),
      iterator_D(ConvOutputIteratorParameter::layout(args.ref_D)),
      ptr_D(args.ref_D.data()),
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
  ImplicitGemmConvolutionFusion() { } 

#if SAIL_CUSTOMIZE_CUTLASS && !defined(__HGGCCC_RTC__)
  /// Determines whether kernel satisfies alignment
  static Status can_implement(
      ConvProblemSize const & problem_size,
      typename Mma::IteratorA::TensorRef ref_A,
      typename Mma::IteratorB::TensorRef ref_B,
      typename Mma::IteratorScaleBias::TensorRef ref_ScaleBias
      #if SAIL_FUSE_OP_EXT
      , TensorRefExtra (&ref_Extra_NC)[kExtraInputNum]
      #endif
  ) {

    static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
    static int const kAlignmentScaleBias = Mma::IteratorScaleBias::AccessType::kElements;

    // small channel implementation will read multiple channel once, thus can input unaligned tensor
    if (!TensorRef_aligned(ref_A, kAlignmentA) && problem_size.fold_num == 1) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_B, kAlignmentB) && problem_size.fold_num == 1) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_ScaleBias, kAlignmentScaleBias) && problem_size.fold_num == 1) {
      return Status::kErrorMisalignedOperand;
    }

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
#endif

  /// Executes one ImplicitGEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    // int block_idx1 = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    // int warp_offset = (block_idx1 * blockDim.x + threadIdx.x) / 32;
    // long Pc0;
    // asm("s.savepc %0, 0x00" : "=s"(Pc0));
    // if (warp_offset < 50) {
    //   __ppu_prefetch_KI((void*)(Pc0 * 8 + warp_offset * 128));
    // }

    // only use block.y to prefetch corresponding A matrix
    // void const *global_ptr = reinterpret_cast<void const*>(params.ptr_A);
    // int block_idx2 = blockIdx.y;
    // int wave_idx = blockIdx.y / 128;
    // int idx_in_wave = (blockIdx.y % 128) * blockDim.x + threadIdx.x;
    // int wave_offset = 668 * 14 * 14; // 668 batch per wave
    // int channel_size = params.problem_size.C * sizeof(ElementA);
    // if (blockIdx.x == 0 && threadIdx.x < Mma::Shape::kM) {
    //   __ppu_prefetch_nonebulk_LLC(const_cast<void*>(static_cast<const void*>(static_cast<const char*>(global_ptr) + (wave_idx * wave_offset + idx_in_wave) * channel_size)));

    // }

    // int thread_off = ((blockIdx.y * gridDim.x) + blockIdx.x) * 256;
    // __ppu_prefetch_nonebulk_LLC(const_cast<void*>(static_cast<const void*>(static_cast<const char*>(global_ptr) + thread_off * 1024)));

    // void const *global_ptr = reinterpret_cast<void const*>(params.ptr_A);
    // int thread_offset = (blockIdx.y % 288) * 128 + threadIdx.x;
    // int wave_idx = blockIdx.y / 288;
    // __ppu_prefetch_nonebulk_LLC(const_cast<void*>(static_cast<const void*>(static_cast<const char*>(global_ptr) + wave_idx * 147440 * 256 + thread_offset * 256)));


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

    // Construct iterators to A and B operands
    typename Mma::IteratorA iterator_A(
      params.iterator_A,
      params.problem_size,
      params.ptr_A,
      thread_idx,
      MatrixCoord(
        threadblock_tile_idx.m() * Mma::Shape::kM,
        threadblock_tile_idx.k() * Mma::Shape::kK
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

    // Construct iterators to A scale/bias vector
    typename Mma::IteratorScaleBias iterator_scale_bias(
      params.iterator_scale_bias,
      params.problem_size,
      params.ptr_scale,
      params.ptr_bias,
      thread_idx,
      MatrixCoord(
        0, (kConvolutionalOperator == conv::Operator::kFprop) ?
                  (threadblock_tile_idx.k() * Mma::Shape::kK) :
                  // Wgrad
                  (threadblock_tile_idx.n() * Mma::Shape::kN)
      )
    );    

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
    mma(params.gemm_k_iterations, accumulators, iterator_A, iterator_B, iterator_scale_bias, accumulators);

    //
    // Epilogue
    //

    EpilogueOutputOp output_op(params.output_op);
    auto split_k_mode = params.split_k_mode;
    auto grid_tiled_shape = params.grid_tiled_shape;

    // Construct the semaphore.
    int block_idx = threadblock_tile_idx.m() + threadblock_tile_idx.n() * grid_tiled_shape.m();

    Semaphore semaphore(params.semaphore + block_idx, thread_idx);
    bool splitk_serial = kSplitKSerial && grid_tiled_shape.k() > 1;

    bool splitk_parallel = split_k_mode == SplitKMode::kParallel && grid_tiled_shape.k() > 1;
    bool last_k = grid_tiled_shape.k() == threadblock_tile_idx.k() + 1;

    // Compute logical position within grid
    threadblock_tile_idx =
        threadblock_swizzle.get_tile_offset(grid_tiled_shape, params.grid_shift);

    // If performing a reduction via split-K, fetch the initial synchronization
    if (splitk_serial) {
        
      // Fetch the synchronization lock initially but do not block.
      semaphore.fetch();

      // Indicate which position in a serial reduction the output operator is currently updating
      output_op.set_k_partition(threadblock_tile_idx.k(), grid_tiled_shape.k());
    }

    MatrixCoord threadblock_offset(
      threadblock_tile_idx.m() * Mma::Shape::kM,
      threadblock_tile_idx.n() * Mma::Shape::kN
    );

    // write to workspace address when splitk mode is parallel
    typename Epilogue::OutputTileIterator::Element *pointer_c;
    typename Epilogue::OutputTileIterator::Element *pointer_d;

    if (splitk_parallel) {
      pointer_c = reinterpret_cast<typename Epilogue::OutputTileIterator::Element*>(params.semaphore);
      pointer_d = reinterpret_cast<typename Epilogue::OutputTileIterator::Element*>(params.semaphore);
    } else {
      pointer_c = reinterpret_cast<typename Epilogue::OutputTileIterator::Element*>(params.ptr_C);
      pointer_d = reinterpret_cast<typename Epilogue::OutputTileIterator::Element*>(params.ptr_D);
    }

    // Tile iterator writing to destination tensor
    typename Epilogue::OutputTileIterator iterator_D(
      params.iterator_D,
      pointer_d,
      ConvOutputIteratorParameter::extent(params.problem_size),
      thread_idx,
      threadblock_offset,
      ConvOutputIteratorParameter::get_transpose_size(params.problem_size)
    );
    
    // Tile iterator reading from source accumulator tensor
    typename Epilogue::OutputTileIterator iterator_C(
      params.iterator_C,
      pointer_c,
      ConvOutputIteratorParameter::extent(params.problem_size),
      thread_idx,
      threadblock_offset,
      ConvOutputIteratorParameter::get_transpose_size(params.problem_size)
    );

    // Construct the epilogue
    Epilogue epilogue(
      shared_storage.epilogue, 
      thread_idx, 
      warp_idx, 
      lane_idx);

    // Wait on the semaphore - this latency may have been covered by iterator construction
    if (splitk_serial) {
        
      // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
      if (threadblock_tile_idx.k()) {
        iterator_C = iterator_D;
      }

      semaphore.wait(threadblock_tile_idx.k());

      __threadfence();
    }
    // Each split-k-slice writes to a unique tensor location
    else if (splitk_parallel) {
      iterator_D.add_pointer_offset(threadblock_tile_idx.k() * 
        cutlass::conv::implicit_gemm_tensor_c_size(ConvOperator, params.problem_size));
    }

    // Run efficient epilogue
    #if SAIL_FUSE_OP_EXT
    typename Epilogue::OutputTileIterator iterator_Extra[kExtraInputNum];

    bool serial_and_not_last_k = (splitk_serial&& !last_k);
    if (serial_and_not_last_k ) {
      // If SplitK-Serial but not lask K, then only need to add accumulator
      epilogue(output_op, iterator_D, accumulators, iterator_C);
    } else {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kExtraInputLoopNum; i++) {
        iterator_Extra[i] = { params.iterator_Extra[i],
                              reinterpret_cast<typename Epilogue::OutputTileIterator::Element*>(params.ptr_Extra[i]),
                              ConvOutputIteratorParameter::extent(params.problem_size),
                              thread_idx,
                              threadblock_offset,
                              ConvOutputIteratorParameter::get_transpose_size(params.problem_size) };
      }
      epilogue.runEpilogue(output_op, iterator_D, accumulators, iterator_C, iterator_Extra);
    }
    #else
    epilogue(output_op, iterator_D, accumulators, iterator_C);
    #endif
  
    //
    // Release the semaphore
    //

    if (splitk_serial) { 

      int lock = 0;
      if (last_k) {

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

