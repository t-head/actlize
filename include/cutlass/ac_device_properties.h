/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Licensed under the BSD 3-Clause License.
 **************************************************************************************************/

/*! \file
    \brief Extended device property queries for the ac runtime.
*/

#pragma once

#include <hggc_runtime.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

struct acDeviceProp : public hggcDeviceProp {
  int clockRateKHz;
  int memClockRateKHz;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

inline hggcError_t acGetDeviceProperties(acDeviceProp *properties, int device) {
  hggcError_t result = hggcGetDeviceProperties(
    static_cast<hggcDeviceProp *>(properties), device);

  if (result != hggcSuccess) {
    return result;
  }

#if defined(HGGCRT_VERSION) && HGGCRT_VERSION >= 13000
  result = hggcDeviceGetAttribute(
    &properties->clockRateKHz, hggcDevAttrClockRate, device);
  if (result != hggcSuccess) {
    return result;
  }

  result = hggcDeviceGetAttribute(
    &properties->memClockRateKHz, hggcDevAttrMemoryClockRate, device);
  if (result != hggcSuccess) {
    return result;
  }
#else
  properties->clockRateKHz = properties->clockRate;
  properties->memClockRateKHz = properties->memoryClockRate;
#endif

  return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
