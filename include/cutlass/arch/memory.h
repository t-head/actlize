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
    \brief Architecture-specific operators on memory
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/cache_operation.h"

namespace cutlass {
namespace arch {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// Fragment type to store loaded data
    typename AccessType,
    /// The bytes of loading
    int LoadBytes,
    /// Cache operation
    CacheOperation::Kind cache_op = CacheOperation::Always
    >
struct global_load;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Specializations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////

#if (((__HGGCCC_VER_MAJOR__ == 11) && (__HGGCCC_VER_MINOR__ >= 4)) || \
     (__HGGCCC_VER_MAJOR__ > 11)) &&                                  \
    defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 100)
  #define CUTLASS_ENABLE_L2_PREFETCH 1
#else
  #define CUTLASS_ENABLE_L2_PREFETCH 0
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////
// The redundant mov device assembly instruction is used to enforce the compiler to
// keep the initializing code before ld.global
template <typename AccessType>
struct global_load<AccessType,
                   64,
                   CacheOperation::Always
                  > {
  CUTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
  uint4 *data = reinterpret_cast<uint4 *>(&D);

  if (pred_guard != 0) {
    data[0] = __ppu_global_load_b32x4(ptr);
    data[1] = __ppu_global_load_b32x4(((uint8_t*)ptr) + 16);
    data[2] = __ppu_global_load_b32x4(((uint8_t*)ptr) + 32);
    data[3] = __ppu_global_load_b32x4(((uint8_t*)ptr) + 48);
  } else {
    data[0].x = data[0].x;
    data[0].y = data[0].y;
    data[0].z = data[0].z;
    data[0].w = data[0].w;
    data[1].x = data[1].x;
    data[1].y = data[1].y;
    data[1].z = data[1].z;
    data[1].w = data[1].w;
    data[2].x = data[2].x;
    data[2].y = data[2].y;
    data[2].z = data[2].z;
    data[2].w = data[2].w;
    data[3].x = data[3].x;
    data[3].y = data[3].y;
    data[3].z = data[3].z;
    data[3].w = data[3].w;
  }
  }
};


// The redundant mov device assembly instruction is used to enforce the compiler to
// keep the initializing code before ld.global
template <typename AccessType>
struct global_load<AccessType,
                   32,
                   CacheOperation::Always
                  > {
  CUTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
  uint4 *data = reinterpret_cast<uint4 *>(&D);

  if (pred_guard != 0) {
    data[0] = __ppu_global_load_b32x4(ptr);
    data[1] = __ppu_global_load_b32x4(((uint8_t*)ptr) + 16);
  } else {
    data[0].x = data[0].x;
    data[0].y = data[0].y;
    data[0].z = data[0].z;
    data[0].w = data[0].w;
    data[1].x = data[1].x;
    data[1].y = data[1].y;
    data[1].z = data[1].z;
    data[1].w = data[1].w;
  }
  }
};

template <typename AccessType>
struct global_load<AccessType,
                   32,
                   CacheOperation::LastUse
                  > {
  CUTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
  uint4 *data = reinterpret_cast<uint4 *>(&D);

    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  ppu.cmpp.ne.b32 p, %9, 0;\n"
        "  ppu.mov.b32 %0, %10;\n"
        "  ppu.mov.b32 %1, %11;\n"
        "  ppu.mov.b32 %2, %12;\n"
        "  ppu.mov.b32 %3, %13;\n"
        "  ppu.mov.b32 %4, %14;\n"
        "  ppu.mov.b32 %5, %15;\n"
        "  ppu.mov.b32 %6, %16;\n"
        "  ppu.mov.b32 %7, %17;\n"
        "  @p ppu.ld.global.lu.v4.u32 {%0, %1, %2, %3}, [%8];\n"
        "  @p ppu.ld.global.lu.v4.u32 {%4, %5, %6, %7}, [%18];\n"
        "}\n"
        : "=r"(data[0].x), "=r"(data[0].y), "=r"(data[0].z), "=r"(data[0].w),
          "=r"(data[1].x), "=r"(data[1].y), "=r"(data[1].z), "=r"(data[1].w)
        : "l"(ptr), "r"((int)pred_guard), "r"(data[0].x), "r"(data[0].y),
          "r"(data[0].z), "r"(data[0].w), "r"(data[1].x), "r"(data[1].y),
          "r"(data[1].z), "r"(data[1].w), "l"(((uint8_t *)ptr) + 16));
  }
};

template <typename AccessType>
struct global_load<AccessType,
                   16,
                   CacheOperation::Always
                  > {
  CUTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
  if (pred_guard != 0) {
    uint4 data = __ppu_global_load_b32x4(ptr);
    D = *(reinterpret_cast<AccessType*>(&data));
  }
  }
};

template <typename AccessType>
struct global_load<AccessType,
                   16,
                   CacheOperation::LastUse
                  > {
  CUTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
  uint4 &data = reinterpret_cast<uint4 &>(D);
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  ppu.cmpp.ne.b32 p, %5, 0;\n"
        "  ppu.mov.b32 %0, %6;\n"
        "  ppu.mov.b32 %1, %7;\n"
        "  ppu.mov.b32 %2, %8;\n"
        "  ppu.mov.b32 %3, %9;\n"
        "  @p ppu.ld.global.lu.v4.u32 {%0, %1, %2, %3}, [%4];\n"
        "}\n"
        : "=r"(data.x), "=r"(data.y), "=r"(data.z), "=r"(data.w)
        : "l"(ptr), "r"((int)pred_guard), "r"(data.x), "r"(data.y), "r"(data.z), "r"(data.w));
  }
};

template <typename AccessType>
struct global_load<AccessType,
                   8,
                   CacheOperation::Always
                  > {
  CUTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
  uint2 &data = reinterpret_cast<uint2 &>(D);

  if (pred_guard != 0) {
    data = __ppu_global_load_b32x2(ptr);
  } else {
    data.x = data.x;
    data.y = data.y;
  }
  }
};

template <typename AccessType>
struct global_load<AccessType,
                   8,
                   CacheOperation::LastUse
                  > {
  CUTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
  uint2 &data = reinterpret_cast<uint2 &>(D);

    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  ppu.cmpp.ne.b32 p, %3, 0;\n"
        "  ppu.mov.b32 %0, %4;\n"
        "  ppu.mov.b32 %1, %5;\n"
        "  @p ppu.ld.global.lu.v2.u32 {%0, %1}, [%2];\n"
        "}\n"
        : "=r"(data.x), "=r"(data.y)
        : "l"(ptr), "r"((int)pred_guard), "r"(data.x), "r"(data.y));
  }
};

template <typename AccessType>
struct global_load<AccessType,
                   4,
                   CacheOperation::Always
                  > {
  CUTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
  unsigned &data = reinterpret_cast<unsigned &>(D);

  if (pred_guard != 0) {
    data = __ppu_global_load_b32(ptr);
  } else {
    data = data;
  }
  }
};

template <typename AccessType>
struct global_load<AccessType,
                   4,
                   CacheOperation::LastUse
                  > {
  CUTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
  unsigned &data = reinterpret_cast<unsigned &>(D);

    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  ppu.cmpp.ne.b32 p, %2, 0;\n"
        "  ppu.mov.b32 %0, %3;\n"
        "  @p ppu.ld.global.lu.u32 %0, [%1];\n"
        "}\n"
        : "=r"(data)
        : "l"(ptr), "r"((int)pred_guard), "r"(data));
  }
};

template <typename AccessType>
struct global_load<AccessType,
                   2,
                   CacheOperation::Always
                  > {
  CUTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
  uint16_t &data = reinterpret_cast<uint16_t &>(D);

  if (pred_guard != 0) {
    data = __ppu_global_load_b16(ptr);
  } else {
    data = data;
  }
  }
};

template <typename AccessType>
struct global_load<AccessType,
                   2,
                   CacheOperation::LastUse
                  > {
  CUTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
  uint16_t &data = reinterpret_cast<uint16_t &>(D);

    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  ppu.cmpp.ne.b32 p, %2, 0;\n"
        "  ppu.mov.b16 %0, %3;\n"
        "  @p ppu.ld.global.lu.u16 %0, [%1];\n"
        "}\n"
        : "=h"(data)
        : "l"(ptr), "r"((int)pred_guard), "h"(data));
  }
};

template <typename AccessType>
struct global_load<AccessType,
                   1,
                   CacheOperation::Always
                  > {
  CUTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
    if (pred_guard) D = *(reinterpret_cast<AccessType const *>(ptr));
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// Fragment type to store data
    typename AccessType,
    /// The bytes of storing
    int StoreBytes
    >
struct global_store;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Specializations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType>
struct global_store<AccessType, 32> {
  CUTLASS_DEVICE
  global_store(AccessType const &D, void *ptr, bool pred_guard) {
  uint4 const *data = reinterpret_cast<uint4 const *>(&D);

  if (pred_guard != 0) {
    __ppu_global_store_b32x4(ptr, data[0]);
    __ppu_global_store_b32x4(((uint8_t*)ptr) + 16, data[1]);
  }
  }
};

template <typename AccessType>
struct global_store<AccessType, 16> {
  CUTLASS_DEVICE
  global_store(AccessType const &D, void *ptr, bool pred_guard) {
  uint4 const &data = reinterpret_cast<uint4 const &>(D);
  if (pred_guard != 0) {
    __ppu_global_store_b32x4(ptr, data);
  }
  }
};

template <typename AccessType>
struct global_store<AccessType, 8> {
  CUTLASS_DEVICE
  global_store(AccessType const &D, void *ptr, bool pred_guard) {
  uint2 const &data = reinterpret_cast<uint2 const &>(D);
  if (pred_guard != 0) {
    __ppu_global_store_b32x2(ptr, data);
  }
  }
};

template <typename AccessType>
struct global_store<AccessType, 4> {
  CUTLASS_DEVICE
  global_store(AccessType const &D, void *ptr, bool pred_guard) {
  uint32_t const &data = reinterpret_cast<uint32_t const &>(D);
  if (pred_guard != 0) {
    __ppu_global_store_b32(ptr, data);
  }
  }
};

template <typename AccessType>
struct global_store<AccessType, 2> {
  CUTLASS_DEVICE
  global_store(AccessType const &D, void *ptr, bool pred_guard) {
  uint16_t const &data = reinterpret_cast<uint16_t const &>(D);
  if (pred_guard != 0) {
    __ppu_global_store_b16(ptr, data);
  }
  }
};

template <typename AccessType>
struct global_store<AccessType, 1> {
  CUTLASS_DEVICE
  global_store(AccessType const &D, void *ptr, bool pred_guard) {
    if (pred_guard) *(reinterpret_cast<AccessType *>(ptr)) = D;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// ld.shared
template <int Bytes>
CUTLASS_DEVICE
void shared_load(void *dst, uint32_t ptr);

/// ld.shared - 16b
template <>
CUTLASS_DEVICE
void shared_load<2>(void *dst, uint32_t ptr) {
  asm volatile("ppu.ld.shared.u16 %0, [%1];\n"
    : "=h"(*reinterpret_cast<uint16_t *>(dst))
    : "r"(ptr));
}

/// ld.shared - 32b
template <>
CUTLASS_DEVICE
void shared_load<4>(void *dst, uint32_t ptr) {
  asm volatile("ppu.ld.shared.u32 %0, [%1];\n"
    : "=r"(*reinterpret_cast<uint32_t *>(dst))
    : "r"(ptr));
}

/// ld.shared - 64b
template <>
CUTLASS_DEVICE
void shared_load<8>(void *dst, uint32_t ptr) {
  uint2 *dst_u64 = reinterpret_cast<uint2 *>(dst);
  asm volatile("ppu.ld.shared.v2.u32 {%0, %1}, [%2];\n"
    :
      "=r"(dst_u64->x),
      "=r"(dst_u64->y)
    : "r"(ptr));
}

/// ld.shared - 128b
template <>
CUTLASS_DEVICE
void shared_load<16>(void *dst, uint32_t ptr) {
  uint4 *dst_u128 = reinterpret_cast<uint4 *>(dst);
  asm volatile("ppu.ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n"
    :
      "=r"(dst_u128->x),
      "=r"(dst_u128->y),
      "=r"(dst_u128->z),
      "=r"(dst_u128->w)
    : "r"(ptr));
}
/////////////////////////////////////////////////////////////////////////////////////////////////


} // namespace arch
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass/arch/memory_ppu.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
