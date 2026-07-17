/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved. 
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 **************************************************************************************************/

/*!
  \file
  \brief The universal GEMM accommodates streamk, batched strided, and batched array variants.
*/


#pragma once

#if defined(__HGGCCC_RTC__)
#include <hggc/std/limits>
#else
#include <limits>
#endif

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/device_kernel.h"
#include "cutlass/hggc_host_adapter.hpp"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_universal.h"

#include "cutlass/gemm/kernel/default_gemm_universal.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"

#include "cutlass/trace.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////


template <typename GemmKernel_>
class GemmUniversalBase {
public:

  using GemmKernel = GemmKernel_;

  /// Boolean indicating whether the HggcHostAdapter is enabled
  static bool const kEnableHggcHostAdapter = CUTLASS_ENABLE_HGGC_HOST_ADAPTER;

  using ThreadblockShape = typename GemmKernel::Mma::Shape;

  using ElementA = typename GemmKernel::ElementA;
  using LayoutA = typename GemmKernel::LayoutA;
  using TensorRefA = TensorRef<ElementA const, LayoutA>;
  static ComplexTransform const kTransformA = GemmKernel::kTransformA;

  using ElementB = typename GemmKernel::ElementB;
  using LayoutB = typename GemmKernel::LayoutB;
  using TensorRefB = TensorRef<ElementB const, LayoutB>;
  static ComplexTransform const kTransformB = GemmKernel::kTransformB;

  using ElementC = typename GemmKernel::ElementC;
  using LayoutC = typename GemmKernel::LayoutC;
  using TensorRefC = TensorRef<ElementC const, LayoutC>;
  using TensorRefD = TensorRef<ElementC, LayoutC>;

  /// Numerical accumulation element type
  using ElementAccumulator = typename GemmKernel::Mma::ElementC;

  using EpilogueOutputOp = typename GemmKernel::EpilogueOutputOp;
  using ThreadblockSwizzle = typename GemmKernel::ThreadblockSwizzle;
  using Operator = typename GemmKernel::Operator;

  /// Argument structure
  using Arguments = typename GemmKernel::Arguments;


  /// Index of the GEMM Kernel within the HggcHostAdapter
  static int32_t const kGemmKernelIndex = 0;

  /// Kernel dynamic shared memory allocation requirement
  /// Update the kernel function's shared memory configuration for the current device
  static constexpr size_t kSharedStorageSize = sizeof(typename GemmKernel::SharedStorage);

protected:

  //
  // Device properties (uniform across all instances of the current thread)
  //

  // Device ordinal
  CUTLASS_THREAD_LOCAL static int device_ordinal_;

  /// Device CU count
  CUTLASS_THREAD_LOCAL static int device_cus_;

  /// Kernel CU occupancy (in thread blocks)
  CUTLASS_THREAD_LOCAL static int cu_occupancy_;

protected:

  /// Initialize static thread-local members for the thread's current device,
  /// if necessary.
  static Status init_device_props()
  {
    CUTLASS_TRACE_HOST("GemmUniversalBase::init_device_props()");

    hggcError_t hggcrt_result;

    // Get current device ordinal
    int current_ordinal;
    hggcrt_result = hggcGetDevice(&current_ordinal);
    if (hggcrt_result != hggcSuccess) {
      CUTLASS_TRACE_HOST("  hggcGetDevice() returned error " << hggcGetErrorString(hggcrt_result));
      return Status::kErrorInternal;
    }

    // Done if matches the current static member
    if (current_ordinal == device_ordinal_) {
      // Already initialized
      return Status::kSuccess;
    }

    // Update CU count member
    hggcrt_result = hggcDeviceGetAttribute (&device_cus_, hggcDevAttrMultiProcessorCount, current_ordinal);
    if (hggcrt_result != hggcSuccess) {
      CUTLASS_TRACE_HOST("  hggcDeviceGetAttribute() returned error " << hggcGetErrorString(hggcrt_result));
      return Status::kErrorInternal;
    }

    // If requires more than 48KB: configure for extended, dynamic shared memory
    if constexpr (kSharedStorageSize >= (48 << 10))
    {
      hggcrt_result = hggcFuncSetAttribute(
        Kernel2<GemmKernel>,
        hggcFuncAttributeMaxDynamicSharedMemorySize,
        kSharedStorageSize);
      if (hggcrt_result != hggcSuccess) {
        CUTLASS_TRACE_HOST("  hggcFuncSetAttribute() returned error " << hggcGetErrorString(hggcrt_result));
        return Status::kErrorInternal;
      }
    }

    // Update CU occupancy member
    hggcrt_result = hggcOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
      &cu_occupancy_,
      Kernel2<GemmKernel>,
      GemmKernel::kThreadCount,
      kSharedStorageSize,
      hggcOccupancyDisableCachingOverride);
    if (hggcrt_result != hggcSuccess) {
      CUTLASS_TRACE_HOST("  hggcOccupancyMaxActiveBlocksPerMultiprocessorWithFlags() returned error " << hggcGetErrorString(hggcrt_result));
      return Status::kErrorInternal;
    }

    // Update device ordinal member on success
    device_ordinal_ = current_ordinal;

    CUTLASS_TRACE_HOST("  "
      "device_ordinal: (" << device_ordinal_ << "), "
      "device_cus: (" << device_cus_ << "), "
      "cu_occupancy: (" << cu_occupancy_ << ") "
      "smem_size: (" << kSharedStorageSize << ") "
      "GemmKernel::kThreadCount: (" << GemmKernel::kThreadCount << ")");

    return Status::kSuccess;
  }


protected:

  //
  // Instance data members
  //

  /// Kernel parameters
  typename GemmKernel::Params params_;


  /// Initialize params member
  Status init_params(Arguments const &args, HggcHostAdapter *hggc_adapter = nullptr)
  {
    int32_t device_cus = 0;
    int32_t cu_occupancy = 0;

    if constexpr (kEnableHggcHostAdapter) {
      CUTLASS_ASSERT(hggc_adapter);

      //
      // Occupancy query using HggcHostAdapter::query_occupancy().
      //

      if (hggc_adapter) {

        Status status = hggc_adapter->query_occupancy(
          &device_cus,
          &cu_occupancy,
          kGemmKernelIndex,
          GemmKernel::kThreadCount,
          kSharedStorageSize);

        CUTLASS_ASSERT(status == Status::kSuccess);

        if (status != Status::kSuccess) {
          return status;
        }
      }
      else {
        return Status::kErrorInternal;
      }
    }
    else {
      CUTLASS_ASSERT(hggc_adapter == nullptr);

      // Initialize static device properties, if necessary
      Status result = init_device_props();

      if (result != Status::kSuccess) {
        return result;
      }

      //
      // Use thread-local static members for occupancy query initialized by call to
      // `init_device_props()`
      //

      device_cus   = device_cus_;
      cu_occupancy = cu_occupancy_;
    }

    // Initialize params member
    params_ = typename GemmKernel::Params(args, device_cus, cu_occupancy);
    return Status::kSuccess;
  }

public:

  //---------------------------------------------------------------------------------------------
  // Stateless API
  //---------------------------------------------------------------------------------------------

  /// Determines whether the GEMM can execute the given problem.
  static Status can_implement(Arguments const &args, HggcHostAdapter *hggc_adapter = nullptr)
  {
    CUTLASS_TRACE_HOST("GemmUniversalBase::can_implement()");

    dim3 grid = get_grid_shape(args, hggc_adapter);

    if (!(grid.y <= std::numeric_limits<uint16_t>::max() &&
          grid.z <= std::numeric_limits<uint16_t>::max()))
    {
      return Status::kErrorInvalidProblem;
    }

    return GemmKernel::can_implement(args);
  }


  /// Returns the workspace size (in bytes) needed for the problem
  /// geometry expressed by these arguments
  static size_t get_workspace_size(Arguments const &args, HggcHostAdapter *hggc_adapter = nullptr)
  {
    CUTLASS_TRACE_HOST("GemmUniversalBase::get_workspace_size()");

    // Initialize parameters from args
    GemmUniversalBase base;
    if (base.init_params(args, hggc_adapter) != Status::kSuccess) {
      return 0;
    }

    // Get size from parameters
    size_t workspace_bytes = base.params_.get_workspace_size();

    CUTLASS_TRACE_HOST("  workspace_bytes: " << workspace_bytes);
    return workspace_bytes;
  }


  /// Returns the grid extents in thread blocks to launch
  static dim3 get_grid_shape(Arguments const &args, HggcHostAdapter *hggc_adapter = nullptr)
  {
    CUTLASS_TRACE_HOST("GemmUniversalBase::get_grid_shape()");

    // Initialize parameters from args
    GemmUniversalBase base;
    if (base.init_params(args, hggc_adapter) != Status::kSuccess) {
      return dim3(0,0,0);
    }

    // Get dims from parameters
    dim3 grid_dims = base.params_.get_grid_dims();

    CUTLASS_TRACE_HOST(
         "  tiled_shape: " << base.params_.get_tiled_shape()  << "\n"
      << "  grid_dims: {" << grid_dims << "}");

    return grid_dims;
  }


  /// Returns the maximum number of active thread blocks per multiprocessor
  static int maximum_active_blocks(HggcHostAdapter *hggc_adapter = nullptr)
  {
    CUTLASS_TRACE_HOST("GemmUniversalBase::maximum_active_blocks()");

    int32_t device_cus   = 0;
    int32_t cu_occupancy = 0;


    if constexpr (kEnableHggcHostAdapter) {
      CUTLASS_ASSERT(hggc_adapter);

      if (hggc_adapter) {

        Status status = hggc_adapter->query_occupancy(
          &device_cus,
          &cu_occupancy,
          kGemmKernelIndex,
          GemmKernel::kThreadCount,
          kSharedStorageSize);

        CUTLASS_ASSERT(status == Status::kSuccess);

        if (status != Status::kSuccess) {
        return -1;
        }
      }
      else {
        return -1;
      }
    }
    else {
      CUTLASS_ASSERT(hggc_adapter == nullptr);
      // Initialize static device properties, if necessary
      if (init_device_props() != Status::kSuccess) {
        return -1;
      }

      cu_occupancy = cu_occupancy_;
    }

    CUTLASS_TRACE_HOST("  max_active_blocks: " << cu_occupancy_);
    return cu_occupancy;
  }


  //---------------------------------------------------------------------------------------------
  // Stateful API
  //---------------------------------------------------------------------------------------------

  /// Initializes GEMM state from arguments and workspace memory
  Status initialize(
    Arguments const &args,
    void *workspace = nullptr,
    hggcStream_t stream = nullptr,
    HggcHostAdapter *hggc_adapter = nullptr)
  {
    CUTLASS_TRACE_HOST("GemmUniversalBase::initialize() - workspace "
      << workspace << ", stream: " << (stream ? "non-null" : "null"));

    // Initialize parameters from args
    Status result = init_params(args, hggc_adapter);
    if (result != Status::kSuccess) {
      return result;
    }

    // Assign and prepare workspace memory
    if (args.mode == GemmUniversalMode::kGemm) {
      return params_.init_workspace(workspace, stream);
    }

    return Status::kSuccess;
  }


  /// Lightweight update given a subset of arguments.
  Status update(Arguments const &args)
  {
    CUTLASS_TRACE_HOST("GemmUniversalBase()::update()");
    params_.update(args);
    return Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  Status run(hggcStream_t stream = nullptr, HggcHostAdapter *hggc_adapter = nullptr)
  {
    CUTLASS_TRACE_HOST("GemmUniversalBase::run()");

    // Configure grid and block dimensions
    dim3 block(GemmKernel::kThreadCount, 1, 1);
    dim3 grid = params_.get_grid_dims();

    // Launch kernel
    CUTLASS_TRACE_HOST("  "
      "grid: (" << grid << "), "
      "block: (" << block << "), "
      "SMEM: (" << kSharedStorageSize << ")");

    if constexpr (kEnableHggcHostAdapter) {
      CUTLASS_ASSERT(hggc_adapter);
      if (hggc_adapter) {
        void* kernel_params[] = {&params_};
        return hggc_adapter->launch(grid, block, kSharedStorageSize, stream, kernel_params, 0);
      }
      else {
        return Status::kErrorInternal;
      }
    }
    else {
      CUTLASS_ASSERT(hggc_adapter == nullptr);

      Kernel2<GemmKernel><<<grid, block, kSharedStorageSize, stream>>>(params_);

      // Query for errors
      hggcError_t result = hggcGetLastError();
      if (result != hggcSuccess) {
        CUTLASS_TRACE_HOST("  grid launch failed with error " << hggcGetErrorString(result));
        return Status::kErrorInternal;
      }
    }

    return Status::kSuccess;
  }


  /// Runs the kernel using initialized state.
  Status operator()(hggcStream_t stream = nullptr, HggcHostAdapter *hggc_adapter = nullptr)
  {
    return run(stream, hggc_adapter);
  }


  /// Runs the kernel using initialized state.
  Status operator()(
    Arguments const &args, 
    void *workspace = nullptr, 
    hggcStream_t stream = nullptr,
    HggcHostAdapter *hggc_adapter = nullptr)
  {
    Status status = initialize(args, workspace, stream, hggc_adapter);

    if (status == Status::kSuccess) {
      status = run(stream, hggc_adapter);
    }

    return status;
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////
/// Static initializers
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Device ordinal
template <typename GemmKernel_>
CUTLASS_THREAD_LOCAL int GemmUniversalBase<GemmKernel_>::device_ordinal_ = -1;

/// Device CU count
template <typename GemmKernel_>
CUTLASS_THREAD_LOCAL int GemmUniversalBase<GemmKernel_>::device_cus_ = -1;

/// Kernel CU occupancy (in thread blocks)
template <typename GemmKernel_>
CUTLASS_THREAD_LOCAL int GemmUniversalBase<GemmKernel_>::cu_occupancy_ = -1;

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
