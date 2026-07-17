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
  \brief Functor performing conversion operations used by epilogues.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Converts the result without other operations
///
template <
  typename FragmentEpiComputeType,
  typename FragmentOutput,
  int kExtraEpilogueInputs,
  int kCount,
  typename ...EpilogueOps
>
struct SynthesizeEpilogueOps {
  using type = void;

  CUTLASS_HOST_DEVICE
  void operator()(FragmentEpiComputeType &intermediate,
                  const float beta_e,
                  FragmentOutput (&extra_input_vals)[kExtraEpilogueInputs]) {}
};

template <
  typename FragmentOutput,
  int kExtraEpilogueInputs,
  int kCount
>
struct SynthesizeEpilogueOps<Array<float, kCount>,
                             FragmentOutput,
                             kExtraEpilogueInputs,
                             kCount,
                             cutlass::plus<cutlass::Array<float, kCount>>,
                             cutlass::epilogue::thread::ReLu<cutlass::Array<float, kCount>>
                            > {
  using FragmentEpiComputeType = Array<float, kCount>;
  #if defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 100)
  // TODO: check if Holmes UT cases' accuracy is OK under PPU, if so then it's bug of ppu compiler
  using type = void;
  // using type = SynthesizeEpilogueOps<FragmentEpiComputeType,
  //                                    FragmentOutput,
  //                                    kExtraEpilogueInputs,
  //                                    kCount,
  //                                    cutlass::plus<FragmentEpiComputeType>,
  //                                    cutlass::epilogue::thread::ReLu<FragmentEpiComputeType>
  //                                   >;
  #else
  using type = void;
  #endif

  CUTLASS_HOST_DEVICE
  void operator()(FragmentEpiComputeType &intermediate,
                  const float beta_e,
                  FragmentOutput (&extra_input_vals)[kExtraEpilogueInputs]) {
    multiply_add_relu0<FragmentEpiComputeType> mul_add_relu_op;
    NumericArrayConverter<typename FragmentEpiComputeType::Element, typename FragmentOutput::Element, kCount, FloatRoundStyle::round_to_nearest> extra_input_converter;
    intermediate = mul_add_relu_op(beta_e, extra_input_converter(extra_input_vals[0]), intermediate);

  //  FragmentEpiComputeType tmp = extra_input_converter(extra_input_vals[0]);
  //  if (kCount == 1) {
  //   intermediate[0] = intermediate[0] + ( /*beta_e * */extra_input_vals[0][0]);
  //     intermediate[0] =  intermediate[0] < .0f ? .0f : intermediate[0];
  //  } else {
  //   CUTLASS_PRAGMA_UNROLL
  //   for (int i = 0; i < kCount; ++i) {
  //     intermediate[i] = intermediate[i] + ( /*beta_e * */tmp[i]);
  //     intermediate[i] =  intermediate[i] < .0f ? .0f : intermediate[i];
  //   } 
  //  }

  }
};

template <
  typename FragmentOutput,
  int kExtraEpilogueInputs,
  int kCount
>
struct SynthesizeEpilogueOps<Array<cutlass::half_t, kCount>,
                             FragmentOutput,
                             kExtraEpilogueInputs,
                             kCount,
                             cutlass::plus<cutlass::Array<cutlass::half_t, kCount>>,
                             cutlass::epilogue::thread::ReLu<cutlass::Array<cutlass::half_t, kCount>>
                            > {
  using FragmentEpiComputeType = Array<cutlass::half_t, kCount>;
  #if defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 100)
  using type = SynthesizeEpilogueOps<FragmentEpiComputeType,
                                     FragmentOutput,
                                     kExtraEpilogueInputs,
                                     kCount,
                                     cutlass::plus<FragmentEpiComputeType>,
                                     cutlass::epilogue::thread::ReLu<FragmentEpiComputeType>
                                    >;
  #else
  using type = void;
  #endif

  CUTLASS_HOST_DEVICE
  void operator()(FragmentEpiComputeType &intermediate,
                  const float beta_e,
                  FragmentOutput (&extra_input_vals)[kExtraEpilogueInputs]) {
  #if defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 100)
    __half2 *result_ptr = reinterpret_cast<__half2 *>(&intermediate);
    __half2 beta_e2;
    __half *beta_e2_ptr = reinterpret_cast<__half *>(&beta_e2);
    beta_e2_ptr[0] = beta_e2_ptr[1] = (float)beta_e;
    __half2 const *rhs_ptr = reinterpret_cast<__half2 const *>(&extra_input_vals[0]);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount / 2; ++i) {
      result_ptr[i] = __hfma2_relu(result_ptr[i], beta_e2, rhs_ptr[i]);
    }

    if (kCount % 2) {
      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&intermediate);
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&extra_input_vals[0]);
      __half d_residual = __hfma_relu(a_residual_ptr[kCount - 1], beta_e2_ptr[0], b_residual_ptr[kCount - 1]);

      intermediate[kCount - 1] = reinterpret_cast<half_t const &>(d_residual);
    }
  #endif
  }
};
/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace cutlass
