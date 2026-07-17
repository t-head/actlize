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
  \brief Kernel performing a reduction over densely packed tensors in global memory
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_conversion.h"

#include "cutlass/layout/matrix.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace reduction {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Shape_,              ///< shape of CTA        (concept: MatrixShape)
  typename OutputOp_ ,          ///< output operator     (concept: epilogue::thread operator)
  typename ReductionOp_,        ///< reduction operator  (concept: ReductionOperator)
  int PartitionsPerStage = 4    ///< number of partitions to issue
>
class ReduceArraySplitK {
public:

  using Shape = Shape_;
  using ReductionOp = ReductionOp_;
  using OutputOp = OutputOp_;
  static int const kElementsPerAccess = OutputOp::kCount;
#if SAIL_REDUCE_SPLITK_OPT
  static int const kPartitionsPerStage = PartitionsPerStage * 4 / kElementsPerAccess;
#else
  static int const kPartitionsPerStage = PartitionsPerStage;
#endif

  using ElementWorkspace = typename ReductionOp::Element;
  using ElementAccumulator = typename ReductionOp::ElementAccumulator;
  using ElementOutput = typename OutputOp::ElementOutput;

  using WorkspaceTensorRef = TensorRef<ElementWorkspace, layout::RowMajor>;
  using OutputTensorRef = TensorRef<ElementOutput, layout::RowMajor>;

  using FragmentWorkspace = AlignedArray<ElementWorkspace, kElementsPerAccess>;
  using FragmentAccumulator = Array<ElementAccumulator, kElementsPerAccess>;
  using FragmentOutput = AlignedArray<ElementOutput, kElementsPerAccess>;

#if SAIL_FUSE_OP_EXT
  using FragmentExtraInput = Array<ElementOutput, kElementsPerAccess>;
  using TensorRefExtra = OutputTensorRef;
  static int const kExtraInputNum = OutputOp::kExtraEpilogueInputs > 0 ? OutputOp::kExtraEpilogueInputs : 1;
  static int const kExtraInputLoopNum = cutlass::epilogue::GetExtraEpilogueBinaryInputs<OutputOp>::value;
  static bool const kSupportExtraInput = OutputOp::kExtraEpilogueOpsNum > 0;
#endif
  //
  // Types
  //

  /// Params structure
  struct Params {

    MatrixCoord problem_size;
    int partitions;
    CUsize partition_stride;
    WorkspaceTensorRef workspace;
    const ElementOutput* const* ptr_C;
    ElementOutput* const* ptr_D;
    layout::RowMajor layout_C;
    layout::RowMajor layout_D;
    #if SAIL_FUSE_OP_EXT
    TensorRefExtra ref_Extra[kExtraInputNum];
    #endif
    typename OutputOp::Params output;
    typename ReductionOp::Params reduction;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params() { }

    CUTLASS_HOST_DEVICE
    Params(
      MatrixCoord problem_size_,
      int partitions_,
      CUsize partition_stride_,
      WorkspaceTensorRef workspace_,
      const void* const* ptr_C_,
      void* const* ptr_D_,
      layout::RowMajor layout_C_,
      layout::RowMajor layout_D_,
      typename OutputOp::Params output_ = typename OutputOp::Params(),
      typename ReductionOp::Params reduction_ = typename ReductionOp::Params()
      #if SAIL_FUSE_OP_EXT
      , const TensorRefExtra *pRef_Extra_NC = nullptr
      #endif
    ):
      problem_size(problem_size_),
      partitions(partitions_),
      partition_stride(sizeof(FragmentWorkspace) * partition_stride_ / kElementsPerAccess),
      workspace(workspace_),
      ptr_C(reinterpret_cast<const ElementOutput* const*>(ptr_C_)),
      ptr_D(reinterpret_cast<ElementOutput* const*>(ptr_D_)),
      layout_C(layout_C_),
      layout_D(layout_D_),
      output(output_),
      reduction(reduction_) {
        #if SAIL_FUSE_OP_EXT
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kExtraInputLoopNum; i++) {
          if (pRef_Extra_NC) {
            ref_Extra[i] = pRef_Extra_NC[i];
          } else {
            ref_Extra[i] = {nullptr, 0};
          }
        }
        #endif
    }
  };

  struct SharedStorage { };


public:

  /// Computes the grid size given a chosen threadblock shape
  CUTLASS_HOST_DEVICE
  static dim3 grid_shape(
    cutlass::MatrixCoord problem_size) {

    return dim3(
#if SAIL_REDUCE_SPLITK_OPT
      (problem_size.row() * problem_size.column() + Shape::kCount - 1) / Shape::kCount);
#else
      (problem_size.row() + Shape::kRow - 1) / Shape::kRow,
      (problem_size.column() + Shape::kColumn - 1) / Shape::kColumn);
#endif
  }

  /// Determines the threadblock shape
  CUTLASS_HOST_DEVICE
  static dim3 block_shape() {
#if SAIL_REDUCE_SPLITK_OPT
    return dim3(Shape::kCount / kElementsPerAccess);
#else
    return dim3(Shape::kColumn / kElementsPerAccess, Shape::kRow);
#endif
  }

#if SAIL_FUSE_OP_EXT
  template <bool Support = kSupportExtraInput>
  CUTLASS_DEVICE
  typename platform::enable_if<Support, void>::type
  runEpilogueOutput(OutputOp const& output_op,
                    FragmentAccumulator const& accumulator,
                    FragmentOutput const& source_frag,
                    typename OutputOp::FragmentOutput& output_frag,
                    Params const& params,
                    uint32_t const& thread_offset
                   ) {
    FragmentExtraInput extra_frags[kExtraInputNum];

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kExtraInputLoopNum; ++i) {
      FragmentExtraInput const *extra_ptr = reinterpret_cast<FragmentExtraInput const *>(
      #if SAIL_REDUCE_SPLITK_OPT
            params.ref_Extra[i].data() + params.ref_Extra[i].offset(
              MatrixCoord(thread_offset / params.problem_size.column(), thread_offset % params.problem_size.column())));
      #else
            params.ref_Extra[i].data() + params.ref_Extra[i].offset(thread_offset));
      #endif
      reinterpret_cast<FragmentExtraInput &>(extra_frags[i]) = *extra_ptr;
    }

    if (output_op.is_source_needed()) {
      output_frag = output_op(accumulator, source_frag, extra_frags);
    } else {
      output_frag = output_op(accumulator, extra_frags);
    }
  }

  template <bool Support = kSupportExtraInput>
  CUTLASS_DEVICE
  typename platform::enable_if<!Support, void>::type
  runEpilogueOutput(OutputOp const& output_op,
                    FragmentAccumulator const& accumulator,
                    FragmentOutput const& source_frag,
                    typename OutputOp::FragmentOutput& output_frag,
                    Params const& params,
                    uint32_t const& thread_offset
                   ) {
    if (output_op.is_source_needed()) {
      output_frag = output_op(accumulator, source_frag);
    } else {
      output_frag = output_op(accumulator);
    }
  }
#endif

  /// Perform a reduction
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &storage) {
#if SAIL_REDUCE_SPLITK_OPT
    // because workspace, source and destination are all RowMajor,
    // so we don't need MatrixCoord thread_offset to compute offset index.
    uint32_t thread_offset = blockIdx.x * Shape::kCount + threadIdx.x * kElementsPerAccess;
    MatrixCoord thread_mn(thread_offset / params.problem_size.column(), thread_offset % params.problem_size.column());

    if (!(thread_offset < params.problem_size.row() * params.problem_size.column())) {
      return;
    }
#else
    // Determine CTA position
    MatrixCoord thread_offset(
      int(blockIdx.x) * Shape::kRow + threadIdx.y,
      int(blockIdx.y) * Shape::kColumn + threadIdx.x * kElementsPerAccess
    );

    // One guard conditional
    if (!(thread_offset.row() < params.problem_size.row() &&
          thread_offset.column() < params.problem_size.column())) {

      return;
    }
#endif

    ReductionOp reduction_op(params.reduction);

    FragmentAccumulator accumulator;

    accumulator.clear();

    //
    // Load the first slice
    //

#if SAIL_REDUCE_SPLITK_OPT
    char const * __restrict__ workspace_ptr =
      reinterpret_cast<char const *>(
        params.workspace.data() + thread_offset);
#else
    char const *workspace_ptr =
      reinterpret_cast<char const *>(
        params.workspace.data() + params.workspace.offset(thread_offset));
#endif
    int batch_id = blockIdx.z;
    workspace_ptr += batch_id * params.partitions * params.partition_stride;

    FragmentWorkspace workspace_frag[kPartitionsPerStage];

    //
    // Construct the output operator
    //

    OutputOp output_op(params.output);

    //
    // Load and accumulate with a simple batched loading sequence.
    //

#if SAIL_REDUCE_SPLITK_OPT
    {
      int k = 0;
      CUTLASS_PRAGMA_NO_UNROLL
      for (; k <= params.partitions - kPartitionsPerStage; k += kPartitionsPerStage) {

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPartitionsPerStage; ++i) {
          // if (k + i < params.partitions) {
          workspace_frag[i] = *reinterpret_cast<FragmentWorkspace const *>(workspace_ptr);
          workspace_ptr += params.partition_stride;
          // }
        }

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPartitionsPerStage; ++i) {
          // if (k + i < params.partitions) {
            accumulator = reduction_op(accumulator, workspace_frag[i]);
          //}
        }
      }

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kPartitionsPerStage; ++i) {
        if (k + i < params.partitions) {
          workspace_frag[i] = *reinterpret_cast<FragmentWorkspace const *>(workspace_ptr);
          workspace_ptr += params.partition_stride;
        }
      }
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kPartitionsPerStage; ++i) {
        if (k + i < params.partitions) {
          accumulator = reduction_op(accumulator, workspace_frag[i]);
        }
      }
    }
#else
    CUTLASS_PRAGMA_NO_UNROLL
    for (int k = 0; k < params.partitions; k += kPartitionsPerStage) {

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kPartitionsPerStage; ++i) {
        if (k + i < params.partitions) {
          workspace_frag[i] = *reinterpret_cast<FragmentWorkspace const *>(workspace_ptr);
          workspace_ptr += params.partition_stride;
        }
      }

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kPartitionsPerStage; ++i) {
        if (k + i < params.partitions) {
          accumulator = reduction_op(accumulator, workspace_frag[i]);
        }
      }
    }
#endif

    //
    // Conditionally load the source
    //
    FragmentOutput source_frag;
    source_frag.clear();
    if (output_op.is_source_needed()) {
      FragmentOutput const *source_ptr = reinterpret_cast<FragmentOutput const *>(
      #if SAIL_REDUCE_SPLITK_OPT
            // source may be not contiguous, (ldc > col)
            params.ptr_C[batch_id] + params.layout_C(thread_mn)
      #else
            params.ptr_C[batch_id] + params.layout_C(thread_offset)
      #endif
      );
      reinterpret_cast<FragmentOutput &>(source_frag) = *source_ptr;
    }

    #if SAIL_FUSE_OP_EXT
    typename OutputOp::FragmentOutput output_frag;
    runEpilogueOutput(output_op, accumulator, source_frag, output_frag, params, thread_offset);
    #else
    //
    // Compute the output
    //

    typename OutputOp::FragmentOutput output_frag = output_op(accumulator, source_frag);
    #endif

    //
    // Store
    //

    FragmentOutput *dest_ptr = reinterpret_cast<FragmentOutput *>(
#if SAIL_REDUCE_SPLITK_OPT
      // destination may be not contiguous, (ldd > col)
      params.ptr_D[batch_id] + params.layout_D(thread_mn)
#else
      params.ptr_D[batch_id] + params.layout_D(thread_offset)
#endif
    );

    *dest_ptr = reinterpret_cast<FragmentOutput const &>(output_frag);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace reduction
} // namespace cutlass
