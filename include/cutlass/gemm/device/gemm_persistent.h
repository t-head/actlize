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

/*!
  \file
  \brief Device-level persistent GEMM.
*/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/device_kernel.h"
#include "cutlass/gemm/kernel/default_gemm_persistent.h"
#include "cutlass/trace.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Gemm Atomic
template <typename BaseKernel_>
class GemmPersistent {
public:
    using BaseKernel = BaseKernel_;
    using Arguments = typename BaseKernel::Arguments;
    using EpilogueOutputOp = typename BaseKernel::EpilogueOutputOp;
    using AtomicTileVisitor = typename BaseKernel::PersistentTileVisitor;
    typename BaseKernel::Params params_;

    static int maximum_active_blocks(int smem_capacity = -1) {
        CUTLASS_TRACE_HOST("GemmPersistent::maximum_active_blocks()");
        int smem_size = int(sizeof(typename BaseKernel::SharedStorage));
        CUTLASS_TRACE_HOST("  smem_size: " << smem_size << " bytes");
        hggcError_t result;
        if (smem_size > (48 << 10)) {
            result = hggcFuncSetAttribute(Kernel<BaseKernel>, hggcFuncAttributeMaxDynamicSharedMemorySize, smem_size);
            if (result != hggcSuccess) {
                result = hggcGetLastError();
                CUTLASS_TRACE_HOST("  hggcFuncSetAttribute() returned error " << hggcGetErrorString(result));
                return -1;
            }
        }

        int max_active_blocks = -1;
        result = hggcOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, Kernel<BaseKernel>, BaseKernel::kThreadCount, smem_size);
        if (result != hggcSuccess) {
            result = hggcGetLastError();
            CUTLASS_TRACE_HOST("  hggcOccupancyMaxActiveBlocksPerMultiprocessor() returned error " << hggcGetErrorString(result));
            return -1;
        }

        CUTLASS_TRACE_HOST("  max_active_blocks: " << max_active_blocks);
        return max_active_blocks;
    }


    static int sufficient(cutlass::gemm::GemmCoord problem_size, int available_cu_count=-1) {
        // Determine the number of blocks that would be launched to fill up a single
        // wave on the PPU with each CU having maximum occupancy.
        hggcDeviceProp properties;
        int device_idx;
        hggcError_t result = hggcGetDevice(&device_idx);
        if (result != hggcSuccess) {
            result = hggcGetLastError();
            CUTLASS_TRACE_HOST("  hggcGetDevice() returned error " << hggcGetErrorString(result));
            return 0;
        }

        result = hggcGetDeviceProperties(&properties, device_idx);
        if (result != hggcSuccess) {
            result = hggcGetLastError();
            CUTLASS_TRACE_HOST("  hggcGetDeviceProperties() returned error " << hggcGetErrorString(result));
            return 0;
        }
        bool override_cu_count = (available_cu_count < 0 || available_cu_count > properties.multiProcessorCount);
        if (override_cu_count) {
            available_cu_count = properties.multiProcessorCount;
        }
        int max_active_blocks = maximum_active_blocks();
        if (max_active_blocks <= 0) {
            return 0;
        }
        int occupancy_based_block_count = available_cu_count * max_active_blocks;
        int total_tiles = BaseKernel::PersistentTileVisitor::tile_count(problem_size);
        return min(total_tiles, occupancy_based_block_count);
    }

    static cutlass::gemm::GemmCoord grid_shape(cutlass::gemm::GemmCoord problem_size) {
        return BaseKernel::PersistentTileVisitor::grid_shape(problem_size);
    }

    static uint64_t convert_timeout_from_sec(int sec_timeout) {
        int device_clock = 0;
        int current_device = 0;
        hggcGetDevice(&current_device);
        hggcDeviceGetAttribute(&device_clock, hggcDevAttrClockRate, current_device);
        uint64_t ub_timeout = 1000ull * device_clock * sec_timeout;
        return ub_timeout;
    }

    GemmPersistent() { }

    static Status can_implement(Arguments const &args) {
        return BaseKernel::can_implement(args);
    }

    static size_t get_workspace_size(Arguments const &args) {
        return sizeof(typename BaseKernel::TileVisitorLayout);
    }

    Status initialize(Arguments const &args, void *workspace = nullptr, hggcStream_t stream = nullptr, int sec_timeout = 60) {
        CUTLASS_TRACE_HOST("GemmPersistent::initialize() - workspace " << workspace << ", stream: " << (stream ? "non-null" : "null"));
        size_t workspace_bytes = get_workspace_size(args);
        if (workspace_bytes && !workspace) {
            return Status::kErrorWorkspaceNull;
        }

        params_ = typename BaseKernel::Params(args, workspace);

        int smem_size = int(sizeof(typename BaseKernel::SharedStorage));
        if (smem_size >= (48 << 10)) {
            hggcError_t result = hggcFuncSetAttribute(Kernel<BaseKernel>, hggcFuncAttributeMaxDynamicSharedMemorySize, smem_size);
            if (result != hggcSuccess) {
                return Status::kErrorInternal;
            }
        }
        return Status::kSuccess;
    }


    Status run(hggcStream_t stream = nullptr) {
        dim3 grid(params_.threadblock_count, 1, 1);
        dim3 block(BaseKernel::kThreadCount, 1, 1);
        int smem_size = int(sizeof(typename BaseKernel::SharedStorage));
        Kernel<BaseKernel><<<grid, block, smem_size, stream>>>(params_);
        hggcError_t result = hggcGetLastError();
        if (result != hggcSuccess) {
            std::cout << "run: " << hggcGetErrorString(result) << std::endl;
            CUTLASS_TRACE_HOST("  grid launch failed with error " << hggcGetErrorString(result));
            return Status::kErrorInternal;
        }
        return Status::kSuccess;
    }

    /// @brief Initialize and runs the kernel.
    Status operator()(Arguments const &args, void *workspace, hggcStream_t stream=nullptr) {
        Status status = initialize(args, workspace, stream);
        if(Status::kSuccess == status) {
            status = run(stream);
        }
        return status;
    }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
