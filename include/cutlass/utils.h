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
    \brief Helpers for printing cutlass/core objects
*/

#pragma once


#include "cutlass/cutlass.h"
#include "cutlass/platform/platform.h"

namespace cutlass {
  namespace gemm {
    namespace kernel {
      template <typename Mma, typename = void>
      struct has_PrefetchIterator : platform::false_type {};

      template <typename Mma>
      struct has_PrefetchIterator<Mma, typename platform::enable_if<platform::is_same<typename Mma::PrefetchIterator, typename Mma::PrefetchIterator>::value, void>::type> : platform::true_type {};
    }
  }

  namespace epilogue {

    struct PrefetchStrategyType {
      enum Kind {
        kNoPrefetchNeeded         = 0,
        kLinearPrefetchLLCSingle  = 1,
        kLinearPrefetchLLCStream  = 2,
        kStridedPrefetchLLCSingle = 4,
        kStridedPrefetchLLCStream = 8,
        kPrefetchTsm              = 16,
      };
    };

    template<class> 
    struct TypeSink {  typedef void type; };

    template<class T> using TypeSinkT = typename TypeSink<T>::type;

    template<class T, class=void> struct IsEpilogueFunctorHeavy {
      static bool const value = false;
    };

    template<class T> struct IsEpilogueFunctorHeavy<T, TypeSinkT< decltype( T::kIsHeavy ) > > {
      static bool const value = T::kIsHeavy;
    };

    template<class T, class=void> struct GetExtraEpilogueBinaryInputs {
      static int const value = 0;
    };

    template<class T> struct GetExtraEpilogueBinaryInputs<T, TypeSinkT< decltype( T::kExtraEpilogueBinaryInputs ) > > {
      static int const value = T::kExtraEpilogueBinaryInputs;
    };

    template <typename EpilogueOutputOp, typename=void>
    struct GetPrefetchStrategy {
      static PrefetchStrategyType::Kind const value = PrefetchStrategyType::Kind::kNoPrefetchNeeded;
    };

    template <typename EpilogueOutputOp>
    struct GetPrefetchStrategy<EpilogueOutputOp, TypeSinkT< decltype( EpilogueOutputOp::PrefetchStrategy ) > > {
      static const PrefetchStrategyType::Kind value = EpilogueOutputOp::PrefetchStrategy;
    };
  }  // namespace epilogue


} // namespace cutlass
///////////////////////////////////////////////////////////////////////////////////////////////////
