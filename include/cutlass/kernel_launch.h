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

/*! \file
  \brief Defines structures and helpers to launch device kernels within CUTLASS.
*/

#pragma once

// cutlass3 change: the upstream device runtime API header pulls in many
// internal headers that conflict with hggc_runtime.h, which the PPU compiler
// implicitly embeds at rtc time
#if !defined(__HGGCCC__) || !defined(__HGGCCC_RTC__)
#include <hggc_runtime_api.h>
#endif

#include "cutlass/cutlass.h"
#include "cutlass/trace.h"
#include "cutlass/device_kernel.h" // cutlass::device_kernel

namespace cutlass {

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure containing the basic launch configuration of a device kernel.
struct KernelLaunchConfiguration {

  /// device grid dimensions
  dim3 grid;

  /// device threablock dimensions
  dim3 block;

  /// Bytes of dynamically allocated SMEM in addition to static SMEM
  size_t dynamic_smem;

  //
  // Methods
  //

  /// Constructs a KernellaunchConfiguration object
  CUTLASS_HOST_DEVICE
  KernelLaunchConfiguration(
    dim3 _grid = dim3(1,1,1),
    dim3 _block = dim3(1,1,1),
    size_t _dynamic_smem = 0
  ):
    grid(_grid),
    block(_block),
    dynamic_smem(_dynamic_smem) { }
};


template <typename GemmKernel, typename Params>
Status kernel_launch(
    dim3 const grid_dims,
    dim3 const block_dims,
    size_t const smem_size,
    hggcStream_t device_stream,
    const Params &kernel_params,
    bool launch_with_pdl) {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
  CUTLASS_TRACE_HOST("cutlass::kernel_launch");
#endif

  if (not launch_with_pdl) {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("cutlass::kernel_launch: No PDL");
#endif
    device_kernel<GemmKernel><<<grid_dims, block_dims, smem_size, device_stream>>>(kernel_params);
  }
  else {
#if HGGCRT_VERSION >= 11080
    if constexpr (GemmKernel::ArchTag::kMinComputeCapability < 90) {
      CUTLASS_TRACE_HOST("  Programmatic dependent launch (PDL) is only supported for PPU0015.");
      return Status::kInvalid;
    }

    hggcLaunchConfig_t config;
    hggcLaunchAttribute attrs[1];

    config.gridDim = grid_dims;
    config.blockDim = block_dims;
    config.dynamicSmemBytes = smem_size;
    config.stream = device_stream;

    config.attrs = attrs;
    attrs[0].id = hggcLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    config.numAttrs = 1;

#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("cutlass::kernel_launch: Calling hggcLaunchKernelEx");
#endif
    hggcError_t launch_result = hggcLaunchKernelEx(&config, &device_kernel<GemmKernel>, kernel_params);
    if (hggcSuccess != launch_result) {
      CUTLASS_TRACE_HOST("cutlass::kernel_launch: hggcLaunchKernelEx failed with error: " << hggcGetErrorString(launch_result));
      return Status::kErrorInternal;
    }
#else
    CUTLASS_TRACE_HOST("  Programmatic dependent launch (PDL) is only supported starting device 11.8.");
    return Status::kInvalid;
#endif
  }

  hggcError_t result = hggcGetLastError();
  if (hggcSuccess == result) {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("cutlass::kernel_launch: hggcGetLastError reports success");
#endif
    return Status::kSuccess;
  }
  else {
    CUTLASS_TRACE_HOST("  Kernel launch failed. Reason: " << result);
    return Status::kErrorInternal;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass
