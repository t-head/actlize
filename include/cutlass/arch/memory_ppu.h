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
    \brief Architecture-specific operators on memory added for PPU
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/cache_operation.h"
#include "hggc_pipeline.h"

#if defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 100)
  #define PPU_CP_ASYNC_ACTIVATED 1
#else
  #define PPU_CP_ASYNC_ACTIVATED 0
#endif

namespace cutlass {
namespace arch {

////////////////////////////////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////////////////////////////

    template <
      /// Layout of destination matrix (column-major implies transpose)
      typename Layout,
      /// .x1, .x2, or .x4
      int MatrixCount
    >
    inline __device__ void ldsm(Array<unsigned, MatrixCount> & D, void const* ptr);

    template <
      /// Layout of destination matrix (column-major implies transpose)
      typename Layout,
      /// .x1, .x2, or .x4
      int MatrixCount,
      /// Element data type
      typename Element
    >
    struct ppu_ldsm {
      public:
        CUTLASS_DEVICE
        ppu_ldsm() {}

        CUTLASS_DEVICE
        void operator() (Array<unsigned, MatrixCount> &_D, void const* _ptr) {}
    };

    template <
      /// Layout of destination matrix (column-major implies transpose)
      typename Layout,
      /// .x1, .x2, or .x4
      int MatrixCount
    >
    inline __device__ void tsm_ld_ncom(Array<unsigned, MatrixCount> & D, void const* ptr);

    template <
      /// Layout of destination matrix (column-major implies transpose)
      typename Layout,
      /// .x1, .x2, or .x4
      int MatrixCount
    >
    inline __device__ void vmem_ld(Array<unsigned, MatrixCount> & D, void const* ptr);

    /////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Determine the appropriate way to target device assembly's "ldmatrix" instruction.
    //
    /////////////////////////////////////////////////////////////////////////////////////////////////

    // PPU ldmatrix
    #define CUTLASS_LDMATRIX_ACTIVATED 1
    #define CUTLASS_LDMATRIX_SUPPORTED 1

    /////////////////////////////////////////////////////////////////////////////////////////////////

    /// CUTLASS helper to get SMEM pointer
    inline __device__ unsigned cutlass_get_smem_pointer(void *ptr) {

      /// PPU use CUTLASS helper to get SMEM pointer
      return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
    }

    /// CUTLASS helper to get SMEM pointer
    inline __device__ unsigned cutlass_get_smem_pointer(void const *ptr) {
      return cutlass_get_smem_pointer(const_cast<void *>(ptr));
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    inline __device__ void ldsm<layout::RowMajor, 1>(
        Array<unsigned, 1> & D,
        void const* ptr) {
      #if defined(CUTLASS_LDMATRIX_ACTIVATED)

        unsigned addr = cutlass_get_smem_pointer(ptr);

        int x;
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile (
    "ppu.tc01.ex.ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];"     : "=r"(x) : "r"(addr));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];"     : "=r"(x) : "r"(addr));
#endif
        reinterpret_cast<int &>(D) = x;

      #else

        CUTLASS_UNUSED(D);
        CUTLASS_UNUSED(ptr);
        CUTLASS_NOT_IMPLEMENTED();

      #endif
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    inline __device__ void ldsm<layout::RowMajor, 2>(
        Array<unsigned, 2> & D,
        void const* ptr) {

      #if defined(CUTLASS_LDMATRIX_ACTIVATED)

        unsigned addr = cutlass_get_smem_pointer(ptr);

        int x, y;
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile (
    "ppu.tc01.ex.ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];"     : "=r"(x), "=r"(y) : "r"(addr));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];"     : "=r"(x), "=r"(y) : "r"(addr));
#endif
        reinterpret_cast<int2 &>(D) = make_int2(x, y);

      #else

        CUTLASS_UNUSED(D);
        CUTLASS_UNUSED(ptr);
        CUTLASS_NOT_IMPLEMENTED();

      #endif
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    inline __device__ void ldsm<layout::RowMajor, 4>(
        Array<unsigned, 4> & D,
        void const* ptr) {

      #if defined(CUTLASS_LDMATRIX_ACTIVATED)
        #if SAIL_SIMULATE_CUTLASS_MMA

          int lane_idx = threadIdx.x % 32;
          unsigned mask = 0xffffffff;
          // 128b of cur row
          int4 temp_reg;
          half *reg_ptr = reinterpret_cast<half *>(&D);
          const int4 *smem_ptr = reinterpret_cast<const int4 *>(ptr);
          temp_reg = *smem_ptr;
          __syncthreads();
          for (int loop = 0; loop < 8; loop++) {
            // 8 threads per group to load one matrix(128B)
            int thread_idx = (loop / 2) * 8 + (lane_idx / 4);
            // each thread store 32bit elem, 4 thread in each row
            int reg_idx = (lane_idx % 4) * 2 + (loop % 2);

            // must shuffle whole 128b reg, since shuffle can't use threadIdx as index, such as the annotated line below
            // this will shuffle the element which reg_idx is calculated in the target thread
            // reg_ptr[loop] = __shfl_sync(mask, temp_reg_half[reg_idx], thread_idx);

            int4 shuffle_reg;
            // can't shuffle int4 directly
            int *temp_reg_int = reinterpret_cast<int *>(&temp_reg);
            int *shuffle_reg_int = reinterpret_cast<int *>(&shuffle_reg);
            for (int i = 0; i < 4; i++) {
              shuffle_reg_int[i] = __shfl_sync(mask, temp_reg_int[i], thread_idx);
            }
            half *shuffle_reg_half = reinterpret_cast<half *>(&shuffle_reg);
            reg_ptr[loop] = shuffle_reg_half[reg_idx];
          }

        #else

          unsigned addr = cutlass_get_smem_pointer(ptr);

          int x, y, z, w;
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile (
    "ppu.tc01.ex.ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"     : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(addr));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"     : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(addr));
#endif
          reinterpret_cast<int4 &>(D) = make_int4(x, y, z, w);
        #endif

      #else

        CUTLASS_UNUSED(D);
        CUTLASS_UNUSED(ptr);
        CUTLASS_NOT_IMPLEMENTED();

      #endif
    }

    template <typename Element>
    struct ppu_ldsm<layout::RowMajor, 1, Element> {
      public:
        CUTLASS_DEVICE
        ppu_ldsm() {}

        CUTLASS_DEVICE
        void operator() (Array<unsigned, 1> &D, void const* ptr) {
          #if CUTLASS_LDMATRIX_ACTIVATED
            #if !SAIL_SIMULATE_CUTLASS_MMA
              // PPU Hardware
              /* awmma::ldmatrix<awmma::no_trans, 1>(&D[0],
                reinterpret_cast<void *>(cutlass_get_smem_pointer(ptr))); */
              unsigned addr = cutlass_get_smem_pointer(ptr);

              int x;
    #if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
    asm volatile (
        "ppu.tc01.ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];"     : "=r"(x) : "r"(addr));
    #elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
    asm volatile(
        "ppu.tc02.ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];"     : "=r"(x) : "r"(addr));
    #endif
              reinterpret_cast<int &>(D) = x;

            #else
              unsigned *smem_ptr = const_cast<unsigned *>(reinterpret_cast<const unsigned *>(ptr));
              unsigned *reg_ptr = reinterpret_cast<unsigned *>(&D);
              *reg_ptr = *smem_ptr;
            #endif
          #else
            assert(0);
          #endif
      }
    };

    template <typename Element>
    struct ppu_ldsm<layout::RowMajor, 2, Element> {
      public:
        CUTLASS_DEVICE
        ppu_ldsm() {}

        CUTLASS_DEVICE
        void operator() (Array<unsigned, 2> &D, void const* ptr) {
          #if CUTLASS_LDMATRIX_ACTIVATED
          #if !SAIL_SIMULATE_CUTLASS_MMA
            // PPU Hardware
            /* awmma::ldmatrix<awmma::no_trans, 2>(&D[0],
              reinterpret_cast<void *>(cutlass_get_smem_pointer(ptr))); */

            unsigned addr = cutlass_get_smem_pointer(ptr);

            int x, y;
    #if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
    asm volatile (
        "ppu.tc01.ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"     : "=r"(x), "=r"(y) : "r"(addr));
    #elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
    asm volatile(
        "ppu.tc02.ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"     : "=r"(x), "=r"(y) : "r"(addr));
    #endif
            reinterpret_cast<int2 &>(D) = make_int2(x, y);

          #else
            // PPU ldsm simulate
            int lane_idx = threadIdx.x % 32;
            unsigned mask = 0xffffffff;
            // 32b per thread.
            int element_per_access = 32 / sizeof_bits<Element>::value;
            // 128b of cur row
            int4 temp_reg;
            Element *reg_ptr = reinterpret_cast<Element *>(&D);
            const int4 *smem_ptr = reinterpret_cast<const int4 *>(ptr);
            temp_reg = *smem_ptr;
            __syncthreads();
            // thread per elements * matrix counts
            for (int loop = 0; loop < element_per_access * 4; loop++) {
              // 8 threads per group to load one matrix(128B)
              int thread_idx = (loop / element_per_access) * 8 + (lane_idx / 4);
              // each thread store 32b elems, 128b/32b 4 groups for one shared memory thread pointers.
              int reg_idx = (lane_idx % 4) * element_per_access + (loop % element_per_access);
              // must shuffle whole 128b reg, since shuffle can't use threadIdx as index, such as the annotated line below
              // this will shuffle the element which reg_idx is calculated in the target thread

              int4 shuffle_reg;
              // can't shuffle int4 directly
              int *temp_reg_int = reinterpret_cast<int *>(&temp_reg);
              int *shuffle_reg_int = reinterpret_cast<int *>(&shuffle_reg);
              // each shared memory thread pointers load one row 128b data, 32b per thread store group for mma.
              for (int i = 0; i < 4; i++) {
                shuffle_reg_int[i] = __shfl_sync(mask, temp_reg_int[i], thread_idx);
              }
              Element *shuffle_reg_element = reinterpret_cast<Element *>(&shuffle_reg);
              reg_ptr[loop] = shuffle_reg_element[reg_idx];
            }
          #endif
        #else
          assert(0);
        #endif
      }
    };
    template <typename Element>
    struct ppu_ldsm<layout::RowMajor, 4, Element> {
      public:
        CUTLASS_DEVICE
        ppu_ldsm() {}

        CUTLASS_DEVICE
        void operator() (Array<unsigned, 4> &D, void const* ptr) {
          #if CUTLASS_LDMATRIX_ACTIVATED
          #if !SAIL_SIMULATE_CUTLASS_MMA
            // PPU Hardware
            /* awmma::ldmatrix<awmma::no_trans, 4>(&D[0],
              reinterpret_cast<void *>(cutlass_get_smem_pointer(ptr))); */

            unsigned addr = cutlass_get_smem_pointer(ptr);

            int x, y, z, w;
            //reinterpret_cast<int4 &>(D) = make_int4(x, y, z, w);
    #if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
    asm volatile (
        "ppu.tc01.ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"     : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(addr));
    #elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
    asm volatile(
        "ppu.tc02.ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"     : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(addr));
    #endif
            reinterpret_cast<int4 &>(D) = make_int4(x, y, z, w);
          #else
            // PPU ldsm simulate
            int lane_idx = threadIdx.x % 32;
            unsigned mask = 0xffffffff;
            // 32b per thread.
            int element_per_access = 32 / sizeof_bits<Element>::value;
            // 128b of cur row
            int4 temp_reg;
            Element *reg_ptr = reinterpret_cast<Element *>(&D);
            const int4 *smem_ptr = reinterpret_cast<const int4 *>(ptr);
            temp_reg = *smem_ptr;
            __syncthreads();
            // thread per elements * matrix counts
            for (int loop = 0; loop < element_per_access * 4; loop++) {
              // 8 threads per group to load one matrix(128B)
              int thread_idx = (loop / element_per_access) * 8 + (lane_idx / 4);
              // each thread store 32b elems, 128b/32b 4 groups for one shared memory thread pointers.
              int reg_idx = (lane_idx % 4) * element_per_access + (loop % element_per_access);
              // must shuffle whole 128b reg, since shuffle can't use threadIdx as index, such as the annotated line below
              // this will shuffle the element which reg_idx is calculated in the target thread

              int4 shuffle_reg;
              // can't shuffle int4 directly
              int *temp_reg_int = reinterpret_cast<int *>(&temp_reg);
              int *shuffle_reg_int = reinterpret_cast<int *>(&shuffle_reg);
              // each shared memory thread pointers load one row 128b data, 32b per thread store group for mma.
              for (int i = 0; i < 4; i++) {
                shuffle_reg_int[i] = __shfl_sync(mask, temp_reg_int[i], thread_idx);
              }
              Element *shuffle_reg_element = reinterpret_cast<Element *>(&shuffle_reg);
              reg_ptr[loop] = shuffle_reg_element[reg_idx];
            }
          #endif
        #else
          assert(0);
        #endif
      }
    };

    template <typename Element>
    struct ppu_ldsm<layout::ColumnMajor, 4, Element> {
      public:
        CUTLASS_DEVICE
        ppu_ldsm() {}

        CUTLASS_DEVICE
        void operator() (Array<unsigned, 4> &D, void const* ptr) {
        #if CUTLASS_LDMATRIX_ACTIVATED
          #if !SAIL_SIMULATE_CUTLASS_MMA
            // PPU Hardware
            /* awmma::ldmatrix<awmma::trans_16x16b16, 4>(&D[0],
              reinterpret_cast<void *>(cutlass_get_smem_pointer(ptr))); */

            unsigned addr = cutlass_get_smem_pointer(ptr);

            int x, y, z, w;
    #if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
    asm volatile (
        "ppu.tc01.ldmatrix.sync.aligned.m16n16.x1.trans.shared.b16 {%0, %1, %2, %3}, [%4];"     : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(addr));
    #elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
    asm volatile(
        "ppu.tc02.ldmatrix.sync.aligned.m16n16.x1.trans.shared.b16 {%0, %1, %2, %3}, [%4];"     : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(addr));
    #endif
            reinterpret_cast<int4 &>(D) = make_int4(x, y, z, w);
          #else
            // PPU ldsm simulate
            int lane_idx = threadIdx.x % 32;
            unsigned mask = 0xffffffff;
            // 128b of cur row
            int4 temp_reg;
            half *reg_ptr = reinterpret_cast<half *>(&D);
            const int4 *smem_ptr = reinterpret_cast<const int4 *>(ptr);
            temp_reg = *smem_ptr;
            __syncthreads();
            for (int loop = 0; loop < 8; loop++) {
              // in which 8x8 and which row
              int thread_idx = (loop / 2) * 8 + (lane_idx % 4) * 2 + (loop % 2);
              // T0/4/8/12/16/20/24/28 in one row
              int reg_idx = (lane_idx / 4);
              // must shuffle whole 128b reg, since shuffle can't use threadIdx as index, such as the annotated line below
              // this will shuffle the element which reg_idx is calculated in the target thread
              // reg_ptr[loop] = __shfl_sync(mask, temp_reg_half[reg_idx], thread_idx);

              int4 shuffle_reg;
              // can't shuffle int4 directly
              int *temp_reg_int = reinterpret_cast<int *>(&temp_reg);
              int *shuffle_reg_int = reinterpret_cast<int *>(&shuffle_reg);
              for (int i = 0; i < 4; i++) {
                shuffle_reg_int[i] = __shfl_sync(mask, temp_reg_int[i], thread_idx);
              }
              half *shuffle_reg_half = reinterpret_cast<half *>(&shuffle_reg);
              reg_ptr[loop] = shuffle_reg_half[reg_idx];
            }
          #endif
        #else
          assert(0);
        #endif
        }
    };

    #if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    // TSM_LD_NCOM_B32X4, similar to nvidia
    template <>
    inline __device__ void tsm_ld_ncom<layout::RowMajor, 4>(
        Array<unsigned, 4> & D,
        void const* ptr) {
      int lane_idx = threadIdx.x % 32;
      unsigned mask = 0xffffffff;
      // 128b of cur row
      int4 temp_reg;
      half *reg_ptr = reinterpret_cast<half *>(&D);
      const int4 *smem_ptr = reinterpret_cast<const int4 *>(ptr);
      temp_reg = *smem_ptr;
      __syncthreads();
      for (int loop = 0; loop < 8; loop++) {
        // in which 8x8 and which row
        int thread_idx = (loop / 2) * 8 + (lane_idx / 4);
        // each thread store 2 elem, 4 thread in each row
        int reg_idx = (lane_idx % 4) * 2 + (loop % 2);
        // must shuffle whole 128b reg, since shuffle can't use threadIdx as index, such as the annotated line below
        // this will shuffle the element which reg_idx is calculated in the target thread
        // reg_ptr[loop] = __shfl_sync(mask, temp_reg_half[reg_idx], thread_idx);

        int4 shuffle_reg;
        // can't shuffle int4 directly
        int *temp_reg_int = reinterpret_cast<int *>(&temp_reg);
        int *shuffle_reg_int = reinterpret_cast<int *>(&shuffle_reg);
        for (int i = 0; i < 4; i++) {
          shuffle_reg_int[i] = __shfl_sync(mask, temp_reg_int[i], thread_idx);
        }
        half *shuffle_reg_half = reinterpret_cast<half *>(&shuffle_reg);
        reg_ptr[loop] = shuffle_reg_half[reg_idx];
      }
    }

    // TSM_LD_NCOM_B32X4 with transpose
    template <>
    inline __device__ void tsm_ld_ncom<layout::ColumnMajor, 4>(
        Array<unsigned, 4> & D,
        void const* ptr) {
      int lane_idx = threadIdx.x % 32;
      unsigned mask = 0xffffffff;
      // 128b of cur row
      int4 temp_reg;
      half *reg_ptr = reinterpret_cast<half *>(&D);
      const int4 *smem_ptr = reinterpret_cast<const int4 *>(ptr);
      temp_reg = *smem_ptr;
      __syncthreads();
      for (int loop = 0; loop < 8; loop++) {
        // in which 8x8 and which row
        int thread_idx = (loop / 2) * 8 + (lane_idx % 4) * 2 + (loop % 2);
        // T0/4/8/12/16/20/24/28 in one row
        int reg_idx = (lane_idx / 4);
        // must shuffle whole 128b reg, since shuffle can't use threadIdx as index, such as the annotated line below
        // this will shuffle the element which reg_idx is calculated in the target thread
        // reg_ptr[loop] = __shfl_sync(mask, temp_reg_half[reg_idx], thread_idx);

        int4 shuffle_reg;
        // can't shuffle int4 directly
        int *temp_reg_int = reinterpret_cast<int *>(&temp_reg);
        int *shuffle_reg_int = reinterpret_cast<int *>(&shuffle_reg);
        for (int i = 0; i < 4; i++) {
          shuffle_reg_int[i] = __shfl_sync(mask, temp_reg_int[i], thread_idx);
        }
        half *shuffle_reg_half = reinterpret_cast<half *>(&shuffle_reg);
        reg_ptr[loop] = shuffle_reg_half[reg_idx];
      }
    }

    // VMEM_LD_B32X4
    template <>
    inline __device__ void vmem_ld<layout::RowMajor, 4>(
        Array<unsigned, 4> & D,
        void const* ptr) {
      int4 *reg_ptr = reinterpret_cast<int4 *>(&D);
      const int4 *smem_ptr = reinterpret_cast<const int4 *>(ptr);
      *reg_ptr = *smem_ptr;
    }

    #endif

    /////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Transpose on 16b granularity
    //
    /////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    inline __device__ void ldsm<layout::ColumnMajor, 1>(
        Array<unsigned, 1> & D,
        void const* ptr) {

      #if CUTLASS_LDMATRIX_ACTIVATED

        unsigned addr = cutlass_get_smem_pointer(ptr);

        int x;
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile (
    "ppu.tc01.ex.ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];"     : "=r"(x) : "r"(addr));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];"     : "=r"(x) : "r"(addr));
#endif
        reinterpret_cast<int &>(D) = x;

      #else

        CUTLASS_UNUSED(D);
        CUTLASS_UNUSED(ptr);
        CUTLASS_NOT_IMPLEMENTED();

      #endif
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    inline __device__ void ldsm<layout::ColumnMajor, 2>(
        Array<unsigned, 2> & D,
        void const* ptr) {

      #if defined(CUTLASS_LDMATRIX_ACTIVATED)

        unsigned addr = cutlass_get_smem_pointer(ptr);

        int x, y;
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile (
    "ppu.tc01.ex.ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];"     : "=r"(x), "=r"(y) : "r"(addr));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];"     : "=r"(x), "=r"(y) : "r"(addr));
#endif
        reinterpret_cast<int2 &>(D) = make_int2(x, y);

      #else

        CUTLASS_UNUSED(D);
        CUTLASS_UNUSED(ptr);
        CUTLASS_NOT_IMPLEMENTED();

      #endif
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    inline __device__ void ldsm<layout::ColumnMajor, 4>(
        Array<unsigned, 4> & D,
        void const* ptr) {

      #if defined(CUTLASS_LDMATRIX_ACTIVATED)
        // reg layout is
        // t0.0  t4.0  t8.0  t12.0...
        // t0.1  t4.1  t8.1  t12.1...
        // t1.0  t5.0  t9.0  t13.0...
        // t1.1  t5.1  t9.1  t13.1...
        // ......
        #if SAIL_SIMULATE_CUTLASS_MMA
          int lane_idx = threadIdx.x % 32;
          unsigned mask = 0xffffffff;
          // 128b of cur row
          int4 temp_reg;
          half *reg_ptr = reinterpret_cast<half *>(&D);
          const int4 *smem_ptr = reinterpret_cast<const int4 *>(ptr);
          temp_reg = *smem_ptr;
          __syncthreads();
          for (int loop = 0; loop < 8; loop++) {
            // in which 8x8 and which row
            int thread_idx = (loop / 2) * 8 + (lane_idx % 4) * 2 + (loop % 2);
            // each thread store 1 elem, 8 thread in each row
            int reg_idx = (lane_idx / 4);
            // must shuffle whole 128b reg, since shuffle can't use threadIdx as index, such as the annotated line below
            // this will shuffle the element which reg_idx is calculated in the target thread
            // reg_ptr[loop] = __shfl_sync(mask, temp_reg_half[reg_idx], thread_idx);

            int4 shuffle_reg;
            // can't shuffle int4 directly
            int *temp_reg_int = reinterpret_cast<int *>(&temp_reg);
            int *shuffle_reg_int = reinterpret_cast<int *>(&shuffle_reg);
            for (int i = 0; i < 4; i++) {
              shuffle_reg_int[i] = __shfl_sync(mask, temp_reg_int[i], thread_idx);
            }
            half *shuffle_reg_half = reinterpret_cast<half *>(&shuffle_reg);
            reg_ptr[loop] = shuffle_reg_half[reg_idx];
          }

        #else

          unsigned addr = cutlass_get_smem_pointer(ptr);

          int x, y, z, w;
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile (
    "ppu.tc01.ex.ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"     : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(addr));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"     : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(addr));
#endif
          reinterpret_cast<int4 &>(D) = make_int4(x, y, z, w);
        #endif

      #else

        CUTLASS_UNUSED(D);
        CUTLASS_UNUSED(ptr);
        CUTLASS_NOT_IMPLEMENTED();

      #endif
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename AccessType, int Bytes>
    struct shared_load_op {
      CUTLASS_DEVICE
      shared_load_op(AccessType &D, void const *ptr) {
        D = *reinterpret_cast<AccessType const *>(ptr);
      }
    };

    template <typename AccessType>
    CUTLASS_DEVICE void shared_load(AccessType &D, void const *ptr) {
      shared_load_op<AccessType, int(sizeof(AccessType))>(D, ptr);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename AccessType>
    struct shared_load_op<AccessType, 16> {
      CUTLASS_DEVICE
      shared_load_op(AccessType &D, void const *ptr) {
        unsigned addr = cutlass_get_smem_pointer(ptr);

        uint4 v;
        asm volatile ("ppu.ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];" :
          "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr));

        D = reinterpret_cast<AccessType const &>(v);
      }
    };

    /////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename AccessType>
    struct shared_load_op<AccessType, 8> {
      CUTLASS_DEVICE
      shared_load_op(AccessType &D, void const *ptr) {
        unsigned addr = cutlass_get_smem_pointer(ptr);

        uint2 v;
        asm volatile ("ppu.ld.shared.v2.b32 {%0, %1}, [%2];" :
          "=r"(v.x), "=r"(v.y) : "r"(addr));

        D = reinterpret_cast<AccessType const &>(v);
      }
    };

    /////////////////////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////////////////////////

/// Initiates an asynchronous copy from global memory to shared memory.
///
/// LDGSTS
///
template <
    /// Size of the access in bytes
    int SizeInBytes,
    /// Cache operation
    CacheOperation::Kind cache_op = CacheOperation::Always>
struct cp_async;

/// Initiates an asynchronous copy from global memory to shared memory. Rather than predicate
/// the entire transfer, zeros are written to SMEM if the guard predicate is false.
///
/// LDGSTS
///
template <
    /// Size of the access in bytes
    int SizeInBytes,
    /// Cache operation
    CacheOperation::Kind cache_op = CacheOperation::Always>
struct cp_async_zfill;

/// Initiates an asynchronous copy from global memory to shared memory. Rather than predicate
/// the entire transfer, nans (0x7eff) are written to SMEM if the guard predicate is false.
///
/// LDGSTS
///
template <
    /// Size of the access in bytes
    int SizeInBytes,
    /// Cache operation
    CacheOperation::Kind cache_op = CacheOperation::Always,
    bool is_elem_fp32 = false>
struct cp_async_nan;

static const uint32_t OOB_NAN_F16 = 0x7eff;
static const uint32_t OOB_NAN_F16x2 = ((OOB_NAN_F16 << 16) | OOB_NAN_F16);
static const uint32_t OOB_NAN_FP32 = 0x7fe00000;

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization
template <
    /// Size of the access in bytes
    int SizeInBytes>
struct cp_async<SizeInBytes, CacheOperation::Always> {

  /// Copy
  CUTLASS_DEVICE
  cp_async(void *smem_ptr, void const *global_ptr, bool pred_guard = true) {
    #if PPU_CP_ASYNC_ACTIVATED

      // Make sure the size is supported.
      static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16),
                "Size is not supported");

      unsigned smem_int_ptr = cutlass_get_smem_pointer(smem_ptr);

      asm volatile(
          "{\n"
          "  .reg .pred p;\n"
          "  ppu.cmpp.ne.b32 p, %0, 0;\n"
          "  @p ppu.cp.async.ca.shared.global [%1], [%2], %3;\n"
          "}\n" ::"r"((int)pred_guard),
          "r"(smem_int_ptr), "l"(global_ptr), "n"(SizeInBytes));

    #else
      if (pred_guard) {
        __pipeline_memcpy_async(smem_ptr, global_ptr, SizeInBytes, 0);
      }
    #endif
  }
};

/// Partial specialization
template <
    /// Size of the access in bytes
    int SizeInBytes>
struct cp_async_zfill<SizeInBytes, CacheOperation::Always> {

  /// Copy with zero fill
  CUTLASS_DEVICE
  cp_async_zfill(void *smem_ptr, void const *global_ptr, bool pred_guard) {
    #if PPU_CP_ASYNC_ACTIVATED

      // Make sure the size is supported.
      static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16),
                "Size is not supported");

      unsigned smem_int_ptr = cutlass_get_smem_pointer(smem_ptr);
      int src_in_bytes = (pred_guard ? SizeInBytes : 0);

      asm volatile(
        // "ppu.cp.async.ca.shared.global.LLC::128B [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
        "ppu.cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
        "l"(global_ptr), "n"(SizeInBytes), "r"(src_in_bytes));
    #else
      using AccessType  = Array<uint8_t, SizeInBytes>;

      int zfill_bytes = (pred_guard ? 0 : SizeInBytes);
      __ppu_pipeline_memcpy_async_zfill(smem_ptr, global_ptr, SizeInBytes, zfill_bytes);
      // __ppu_prefetch_nonebulk_L2(const_cast<void*>(static_cast<const void*>(static_cast<const char*>(global_ptr) + 128)));

    #endif
  }
};

/// Partial specialization for loading 32b/16b and cached at all levels
template <bool is_elem_fp32>
struct cp_async_nan<16, CacheOperation::Always, is_elem_fp32> {
  static int const kSizeInBytes = 16;

  /// Copy with nan fill
  CUTLASS_DEVICE
  cp_async_nan(void *smem_ptr, void const *global_ptr, bool pred_guard) {
    #if PPU_CP_ASYNC_ACTIVATED
      static const uint32_t OOB_NAN_VALUE = is_elem_fp32 ? OOB_NAN_FP32 : OOB_NAN_F16x2;
      static __constant__ uint4 OOB_NAN = {OOB_NAN_VALUE,  OOB_NAN_VALUE,
                                           OOB_NAN_VALUE,  OOB_NAN_VALUE};

      unsigned smem_int_ptr = cutlass_get_smem_pointer(smem_ptr);

      asm volatile(
          "{\n"
          "  .reg .pred p;\n"
          "  ppu.cmpp.ne.b32 p, %0, 0;\n"
#if CUTLASS_ENABLE_L2_PREFETCH
          "  @p ppu.cp.async.ca.shared.global.LLC::128B [%1], [%2], %3;\n"
#else
          "  @p ppu.cp.async.ca.shared.global [%1], [%2], %3;\n"
#endif
          "  @!p ppu.st.shared.v4.u32 [%1], {%4, %5, %6, %7};\n"
          "}\n"
          :
          : "r"((int)pred_guard), "r"(smem_int_ptr), "l"(global_ptr),
            "n"(kSizeInBytes), "r"(OOB_NAN.x), "r"(OOB_NAN.y), "r"(OOB_NAN.z),
            "r"(OOB_NAN.w));

    #else

      CUTLASS_UNUSED(smem_ptr);
      CUTLASS_UNUSED(global_ptr);
      CUTLASS_UNUSED(pred_guard);
      CUTLASS_NOT_IMPLEMENTED();

    #endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization
template <
    /// Size of the access in bytes
    int SizeInBytes>
struct cp_async<SizeInBytes, CacheOperation::Global> {

  /// Copy
  CUTLASS_DEVICE
  cp_async(void *smem_ptr, void const *global_ptr, bool pred_guard = true) {
    #if PPU_CP_ASYNC_ACTIVATED

      static_assert(SizeInBytes == 16,
        "ppu.cp.async only supports CacheOperation::Global when access size is 16B.");

      unsigned smem_int_ptr = cutlass_get_smem_pointer(smem_ptr);

      asm volatile(
          "{\n"
          "  .reg .pred p;\n"
          "  ppu.cmpp.ne.b32 p, %0, 0;\n"
          "  @p ppu.cp.async.cg.shared.global [%1], [%2], %3;\n"
          "}\n" ::"r"((int)pred_guard),
          "r"(smem_int_ptr), "l"(global_ptr), "n"(SizeInBytes));

    #else
      if (pred_guard) {
        __pipeline_memcpy_async(smem_ptr, global_ptr, SizeInBytes, 0);
      }
    #endif
  }
};

template <
    /// Size of the access in bytes
    int SizeInBytes>
struct cp_async_zfill<SizeInBytes, CacheOperation::Global> {

  /// Copy with zero fill
  CUTLASS_DEVICE
  cp_async_zfill(void *smem_ptr, void const *global_ptr, bool pred_guard = true) {
    #if PPU_CP_ASYNC_ACTIVATED

      static_assert(SizeInBytes == 16,
        "ppu.cp.async only supports CacheOperation::Global when access size is 16B.");

      unsigned smem_int_ptr = cutlass_get_smem_pointer(smem_ptr);
      int src_in_bytes = (pred_guard ? SizeInBytes : 0);

      asm volatile(
        // "ppu.cp.async.cg.shared.global.LLC::128B [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
        "ppu.cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
        "l"(global_ptr), "n"(SizeInBytes), "r"(src_in_bytes));

    #else
      using AccessType  = Array<uint8_t, SizeInBytes>;

      int zfill_bytes = (pred_guard ? 0 : SizeInBytes);
      __ppu_pipeline_memcpy_async_zfill(smem_ptr, global_ptr, SizeInBytes, zfill_bytes);
      // __ppu_prefetch_nonebulk_L2(const_cast<void*>(static_cast<const void*>(static_cast<const char*>(global_ptr) + 128)));
    #endif
  }
};

/// Partial specialization for loading 32b/16b and cached at global level
template <bool is_elem_fp32>
struct cp_async_nan<16, CacheOperation::Global, is_elem_fp32> {
  static int const kSizeInBytes = 16;

  /// Copy with nan fill
  CUTLASS_DEVICE
  cp_async_nan(void *smem_ptr, void const *global_ptr, bool pred_guard) {
    #if PPU_CP_ASYNC_ACTIVATED
      static const uint32_t OOB_NAN_VALUE = is_elem_fp32 ? OOB_NAN_FP32 : OOB_NAN_F16x2;
      static __constant__ uint4 OOB_NAN = {OOB_NAN_VALUE,  OOB_NAN_VALUE,
                                           OOB_NAN_VALUE,  OOB_NAN_VALUE};

      unsigned smem_int_ptr = cutlass_get_smem_pointer(smem_ptr);

      asm volatile(
          "{\n"
          "  .reg .pred p;\n"
          "  ppu.cmpp.ne.b32 p, %0, 0;\n"
#if CUTLASS_ENABLE_L2_PREFETCH
          "  @p ppu.cp.async.cg.shared.global.LLC::128B [%1], [%2], %3;\n"
#else
          "  @p ppu.cp.async.cg.shared.global [%1], [%2], %3;\n"
#endif
          "  @!p ppu.st.shared.v4.u32 [%1], {%4, %5, %6, %7};\n"
          "}\n"
          :
          : "r"((int)pred_guard), "r"(smem_int_ptr), "l"(global_ptr),
            "n"(kSizeInBytes), "r"(OOB_NAN.x), "r"(OOB_NAN.y), "r"(OOB_NAN.z),
            "r"(OOB_NAN.w));

    #else

      CUTLASS_UNUSED(smem_ptr);
      CUTLASS_UNUSED(global_ptr);
      CUTLASS_UNUSED(pred_guard);
      CUTLASS_NOT_IMPLEMENTED();

    #endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Establishes an ordering w.r.t previously issued cp.async instructions. Does not block.
CUTLASS_DEVICE
void cp_async_fence() {
  #if PPU_CP_ASYNC_ACTIVATED
  asm volatile("ppu.cp.async.commit_group;\n" ::);
  #else
  __pipeline_commit();
  #endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Blocks until all but <N> previous cp.async.commit_group operations have committed.
template <int N>
CUTLASS_DEVICE void cp_async_wait() {
  #if PPU_CP_ASYNC_ACTIVATED
  asm volatile("ppu.cp.async.wait_group %0;\n" ::"n"(N));
  #else
  __pipeline_wait_prior(N);
  #endif
}

/// Blocks until all previous cp.async.commit_group operations have committed.
template <>
CUTLASS_DEVICE void cp_async_wait<0>() {
  #if PPU_CP_ASYNC_ACTIVATED
  asm volatile("ppu.cp.async.wait_all;\n" ::);
  #else
  __pipeline_wait_prior(0);
  #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace arch
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
