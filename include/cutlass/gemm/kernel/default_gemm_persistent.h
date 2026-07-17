/*
 * Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
 * All rights reserved.
 *
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
 */

/*! \file
    \brief
      Default kernel-level GEMM definitions combine threadblock-scoped matrix multiply-add with
      the appropriate threadblock-scoped epilogue.

      Note, CUTLASS epilogues universally target row-major outputs. Column-major outputs are
      accommodated by exchanging A and B operands and assuming transposed layouts. Partial
      specializations here choose 'device::GemmTransposed' to implement this functionality.
*/
#pragma once
#include "cutlass/gemm/kernel/gemm_persistent.h"
#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/default_gemm_complex.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/layout/permute.h"
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    typename ElementA_,
    typename LayoutA_,
    int kAlignmentA,
    typename ElementB_,
    typename LayoutB_,
    int kAlignmentB,
    typename ElementC_,
    typename LayoutC_,
    typename ElementAccumulator,
    typename OperatorClass,         /* Operator class tag */
    typename ArchTag,               /* Tag indicating architecture to tune for */
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    typename EpilogueOutputOp,
    typename ThreadblockSwizzle,
    int Stages,
    PersistentGemmMode PersistentGemmMode_ = PersistentGemmMode::kNonAtomic,
    AtomicDirection Direction_ = AtomicDirection::kRow, 
    // Operation performed by GEMM
    typename Operator = typename device::DefaultGemmConfiguration<OperatorClass,ArchTag,ElementA_,ElementB_,ElementC_,ElementAccumulator>::Operator>
struct DefaultGemmPersistent
{
    static bool const kInternalTranspose = platform::is_same<LayoutC_, layout::ColumnMajor>::value;

    using MapArguments = kernel::detail::MapArguments<
        ElementA_, LayoutA_, kAlignmentA,
        ElementB_, LayoutB_, kAlignmentB,
        LayoutC_, kInternalTranspose>;

      // Define the default GEMM kernel
    using DefaultGemmKernel = typename kernel::DefaultGemm<
        typename MapArguments::ElementA,
        typename MapArguments::LayoutA,
        MapArguments::kAlignmentA,
        typename MapArguments::ElementB,
        typename MapArguments::LayoutB,
        MapArguments::kAlignmentB,
        ElementC_,
        typename MapArguments::LayoutC,
        ElementAccumulator,
        OperatorClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EpilogueOutputOp,
        ThreadblockSwizzle,
        Stages,
        true,
        Operator
    >::GemmKernel;

    using GemmKernel = kernel::GemmPersistent<
        typename DefaultGemmKernel::Mma,
        typename DefaultGemmKernel::Epilogue,
        ThreadblockShape,
        kInternalTranspose,
        PersistentGemmMode_, 
        Direction_>;
};





/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
