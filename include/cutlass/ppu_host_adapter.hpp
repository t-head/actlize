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
    \brief Interface betweeen a CUTLASS device-wide operator and device.
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

#include "cutlass/platform/platform.h"
#if ! defined(__HGGCCC_RTC__)
#include <cstdio>
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

// RTC doesn't need definitions for these host classes

#if HGGCRT_VERSION >= 11080 && !defined(__HGGCCC_RTC__)
#define PPU_HOST_ADAPTER_LAUNCH_ATTRIBUTES_ENABLED
#endif

#if HGGCRT_VERSION >= 12000 && !defined(__HGGCCC_RTC__)
#define PPU_HOST_ADAPTER_TENSORMAP_ENABLED
#endif

// Include <device.h> for device Driver API calls if any of these capabilities are enabled.
#if defined(PPU_HOST_ADAPTER_LAUNCH_ATTRIBUTES_ENABLED) ||        \
    defined(PPU_HOST_ADAPTER_TENSORMAP_ENABLED)

#include <hggc.h>
// hggcTypedefs.h provides PFN_<func>_v<ver> driver entry-point
// function-pointer typedefs that the dlsym wrapper macros below rely on.
#include <hggcTypedefs.h>

#endif // defined(PPU_HOST_ADAPTER_LAUNCH_ATTRIBUTES_ENABLED) ||
       // defined(PPU_HOST_ADAPTER_TENSORMAP_ENABLED)

/////////////////////////////////////////////////////////////////////////////////////////////////

//
// Macro-level guard for device Host Adapter
//
#if !defined(CUTLASS_ENABLE_HOST_ADAPTER)
#define CUTLASS_ENABLE_HOST_ADAPTER false
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////


#if !defined(__HGGCCC_RTC__)

#if HGGCRT_VERSION >= 11060
#include <hggc_runtime_api.h>
#endif // in open_source: (__HGGCCC_VERSION__ >= 11.8)

#include <driver_types.h>

#define CUTLASS_PPU_DRIVER_STRINGIFY(tok) #tok

#if defined(CUTLASS_ENABLE_DIRECT_PPU_DRIVER_CALL)

#define CUTLASS_PPU_DRIVER_WRAPPER_DECL(func, ver) \
  template <typename... Args>                       \
  HGresult call_##func(Args... args) {              \
    return func(args...);                           \
  }

#else // defined(CUTLASS_ENABLE_DIRECT_PPU_DRIVER_CALL)

#if HGGCRT_VERSION >= 12050

#define CUTLASS_PPU_DRIVER_WRAPPER_DECL(func, ver)             \
  template <typename... Args>                                   \
  HGresult call_##func(Args... args) {                          \
    hggcDriverEntryPointQueryResult device_status;                \
    void* pfn = nullptr;                                        \
    hggcError_t device_err = hggcGetDriverEntryPointByVersion(    \
        CUTLASS_PPU_DRIVER_STRINGIFY(func),                    \
        &pfn, ver,                                              \
        hggcEnableDefault,                                      \
        &device_status);                                          \
    if (device_status != hggcDriverEntryPointSuccess ||           \
        device_err != hggcSuccess) {                              \
      return HGGC_ERROR_UNKNOWN;                                \
    }                                                           \
    return reinterpret_cast<PFN_##func##_v##ver>(pfn)(args...); \
  }

#else

#define CUTLASS_PPU_DRIVER_WRAPPER_DECL(func, ver)             \
  template <typename... Args>                                   \
  HGresult call_##func(Args... args) {                          \
    hggcDriverEntryPointQueryResult device_status;                \
    void* pfn = nullptr;                                        \
    hggcError_t device_err = hggcGetDriverEntryPoint(             \
        CUTLASS_PPU_DRIVER_STRINGIFY(func),                    \
        &pfn,                                                   \
        hggcEnableDefault,                                      \
        &device_status);                                          \
    if (device_status != hggcDriverEntryPointSuccess ||           \
        device_err != hggcSuccess) {                              \
      return HGGC_ERROR_UNKNOWN;                                \
    }                                                           \
    return reinterpret_cast<PFN_##func>(pfn)(args...);          \
  }

#endif // (__HGGCCC_VERSION__ >= 12.5)

#endif // defined(CUTLASS_ENABLE_DIRECT_PPU_DRIVER_CALL)

#if HGGCRT_VERSION >= 12000
CUTLASS_PPU_DRIVER_WRAPPER_DECL(hgTensorMapEncodeTiled, 12000);
CUTLASS_PPU_DRIVER_WRAPPER_DECL(hgTensorMapEncodeIm2col, 12000);
#endif

#undef CUTLASS_PPU_DRIVER_STRINGIFY

#define CUTLASS_PPU_DRIVER_WRAPPER_CALL(func) cutlass::call_##func

#endif // !defined(__HGGCCC_RTC__)


/////////////////////////////////////////////////////////////////////////////////////////////////

/// This class manages runtime HGlaunchAttribute that can be supplied to HostAdapter
/// HostLaunchAttributes will be an empty struct in earlier toolchains where HGlaunchAttribute
/// is not introduced.
struct HostLaunchAttributes {

#if defined(PPU_HOST_ADAPTER_LAUNCH_ATTRIBUTES_ENABLED)

  /// Reasonable maximum launch attributes that are commonly applied
  static constexpr int32_t kMaximumAttributeCount = 5;

  /// Launch attributes
  HGlaunchAttribute launch_attributes[kMaximumAttributeCount];
  int32_t      attribute_count = 0;

  CUTLASS_HOST_DEVICE
  HostLaunchAttributes(HGlaunchAttribute *launch_attributes_ = nullptr,
                           int32_t attribute_count_ = 0) {
    CUTLASS_ASSERT(attribute_count_ >= 0 && attribute_count_ < kMaximumAttributeCount);
    for (int32_t i = 0; i < attribute_count_ && i < kMaximumAttributeCount; ++i) {
      launch_attributes[i] = launch_attributes_[i];
    }
    attribute_count = attribute_count_;
  }

  CUTLASS_HOST_DEVICE
  HGlaunchAttribute const* data() const {
    return launch_attributes;
  }

  CUTLASS_HOST_DEVICE
  size_t size() const {
    return attribute_count;
  }

#endif // (PPU_HOST_ADAPTER_LAUNCH_ATTRIBUTES_ENABLED)

};


/// This class defines an object which abstracts interactions between the CUTLASS device-wide GEMM and
/// device. The intention is to enable CUTLASS to be used with both the device Runtime API and device Driver API.
struct HostAdapter {

  /// Limit the number of kernels
  static constexpr int32_t kMaximumKernelCount = 4;

  /// Maximum cluster size
  static constexpr int MaxClusterSize = 32;

  //
  // Data members
  //

  /// Handles
  void        *kernel_handles[kMaximumKernelCount];
  int32_t      kernel_count = 0;

  HostLaunchAttributes launch_attributes;

  //
  // Methods
  //

  /// Ctor
  HostAdapter() = default;

  /// Dtor
  virtual ~HostAdapter() = default;

  /// Copy Ctor
  CUTLASS_HOST_DEVICE
  HostAdapter(const HostAdapter & rhs)
      : kernel_count(rhs.kernel_count),
        launch_attributes(rhs.launch_attributes) {
    CUTLASS_ASSERT(rhs.kernel_count >= 0 && rhs.kernel_count < kMaximumKernelCount);

    for (int32_t i = 0; i < rhs.kernel_count && i < kMaximumKernelCount; ++i) {
      kernel_handles[i] = rhs.kernel_handles[i];
    }
  }

  /// Copy Assignment
  CUTLASS_HOST_DEVICE
  HostAdapter& operator=(const HostAdapter & rhs) {
    CUTLASS_ASSERT(rhs.kernel_count >= 0 && rhs.kernel_count < kMaximumKernelCount);
    for (int32_t i = 0; i < rhs.kernel_count && i < kMaximumKernelCount; ++i) {
      kernel_handles[i] = rhs.kernel_handles[i];
    }
    kernel_count = rhs.kernel_count;

    launch_attributes = rhs.launch_attributes;

    return *this;
  }


  /// Move ctor
  CUTLASS_HOST_DEVICE
  HostAdapter(HostAdapter && rhs)
      : kernel_count(rhs.kernel_count),
        launch_attributes(std::move(rhs.launch_attributes)) {
    CUTLASS_ASSERT(rhs.kernel_count >= 0 && rhs.kernel_count < kMaximumKernelCount);

    for (int32_t i = 0; i < rhs.kernel_count && i < kMaximumKernelCount; ++i) {
      kernel_handles[i] = rhs.kernel_handles[i];
    }
  }

  // / Move assignment
  CUTLASS_HOST_DEVICE
  HostAdapter& operator=(HostAdapter && rhs) {
    CUTLASS_ASSERT(rhs.kernel_count >= 0 && rhs.kernel_count < kMaximumKernelCount);
    for (int32_t i = 0; i < rhs.kernel_count && i < kMaximumKernelCount; ++i) {
      kernel_handles[i] = rhs.kernel_handles[i];
    }
    kernel_count = rhs.kernel_count;
    launch_attributes = std::move(rhs.launch_attributes);
    return *this;
  }

  /// Ctor
  CUTLASS_HOST_DEVICE
  HostAdapter(void **kernel_handles_,
                  int32_t kernel_count_,
                  HostLaunchAttributes const &launch_attributes_ = { })
      : kernel_count(kernel_count_),
        launch_attributes(launch_attributes_) {
    CUTLASS_ASSERT(kernel_count >= 0 && kernel_count < kMaximumKernelCount);

    for (int32_t i = 0; i < kernel_count && i < kMaximumKernelCount; ++i) {
      kernel_handles[i] = kernel_handles_[i];
    }
  }

  /// Returns true if the HostAdapter is empty (kernel_count == 0)
  CUTLASS_HOST_DEVICE
  bool empty() const { return !kernel_count; }

  /// Returns kernel_count
  CUTLASS_HOST_DEVICE
  size_t size() const { return static_cast<size_t>(kernel_count); }

  /// Queries the occupancy of a kernel
  virtual Status query_occupancy(
    int32_t *device_cus,
    int32_t *cu_occupancy,
    int32_t kernel_index,
    int32_t thread_count,
    int32_t smem_size) const = 0;

  /// Launches a kernel without using Threadblock Clusters.
  virtual Status launch(
    dim3 const grid_dims,
    dim3 const block_dims,
    size_t const smem_size,
    hggcStream_t device_stream,
    void** kernel_params,
    int32_t kernel_index) const = 0;

  /// Launches a kernel using the device Extensible Launch API and Threadblock Clusters.
  virtual Status launch(
    dim3 const grid_dims,
    dim3 const cluster_dims,
    dim3 const block_dims,
    size_t const smem_size,
    hggcStream_t device_stream,
    void** kernel_params,
    int32_t kernel_index) const = 0;

#if defined(PPU_HOST_ADAPTER_TENSORMAP_ENABLED)

  /// Create a tensor map descriptor object representing im2col memory region.
  virtual HGresult tensorMapEncodeIm2col (
    HGtensorMap* tensorMap,
    HGtensorMapDataType tensorDataType,
    uint32_t tensorRank,
    void* globalAddress,
    const uint64_t* globalDim,
    const uint64_t* globalStrides,
    const int* pixelBoxLowerCorner,
    const int* pixelBoxUpperCorner,
    uint32_t channelsPerPixel,
    uint32_t pixelsPerColumn,
    const uint32_t* elementStrides,
    HGtensorMapInterleave interleave,
    HGtensorMapSwizzle swizzle,
    HGtensorMapL2promotion l2Promotion,
    HGtensorMapFloatOOBfill oobFill) const = 0;

  /// Create a tensor map descriptor object representing tiled memory region.
  virtual HGresult tensorMapEncodeTiled (
    HGtensorMap* tensorMap,
    HGtensorMapDataType tensorDataType,
    uint32_t tensorRank,
    void* globalAddress,
    const uint64_t* globalDim,
    const uint64_t* globalStrides,
    const uint32_t* boxDim,
    const uint32_t* elementStrides,
    HGtensorMapInterleave interleave,
    HGtensorMapSwizzle swizzle,
    HGtensorMapL2promotion l2Promotion,
    HGtensorMapFloatOOBfill oobFill) const = 0;

  /// Modify an existing tensor map descriptor with an updated global address.
  virtual HGresult tensorMapReplaceAddress(
    HGtensorMap* tensorMap,
    void* globalAddress)  const = 0;

#endif // defined(PPU_HOST_ADAPTER_TENSORMAP_ENABLED)

protected:

  /**
   * Fills a buffer in Global Memory with a byte sequence copied from host memory.
   * This function can be overriden to dispatch to the appropriate memsetD*Async API
  */
  virtual Status memsetDeviceImpl(
    void* destination, ///< Device memory pointer to be filled
    void const* fill_value, ///< Value to be filled in the buffer
    size_t fill_size, ///< Size of the data type to be used for filling the buffer
    size_t count, ///< Number of elements of size fill_size
    hggcStream_t stream) const = 0;

public:

  /// Fills a buffer in Global Memory with a byte sequence copied from host memory
  template<class FillValueType>
  CUTLASS_HOST_DEVICE
  Status memsetDevice(
      void* destination,
      FillValueType fill_value,
      size_t count,
      hggcStream_t stream) const {
    return this->memsetDeviceImpl(
      destination,
      &fill_value,
      sizeof(FillValueType),
      count,
      stream);
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
