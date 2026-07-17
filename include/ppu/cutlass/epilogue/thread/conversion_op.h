/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved. 
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  \brief Functor performing conversion operations used by epilogues.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Converts the result without other operations
///
template <
  typename ElementOutput_,                             ///< Data type used to load and store tensors
  int Count,                                           ///< Number of elements computed per operation
  typename ElementAccumulator_ = ElementOutput_,       ///< Accumulator data type
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest
>
class AcConvert {
public:

  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementAccumulator_;
  using ElementScalar = ElementCompute;

  static int const kCount = Count;

  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using ComputeFragment = FragmentAccumulator;

  static FloatRoundStyle const kRound = Round;

  static bool const kIsHeavy = false;

  /// Host-constructable parameters structure
  struct Params {
//     // to be consistent with linear combination to use same kernel params for serial and parallel
//     ElementCompute alpha;                  ///< scales accumulators
//     ElementCompute beta;                   ///< scales source tensor
//     ElementCompute const *alpha_ptr;       ///< pointer to accumulator scalar - if not null, loads it from memory
//     ElementCompute const *beta_ptr;        ///< pointer to source scalar - if not null, loads it from memory

// #if SUPPORT_FP8_SCALING
//     ElementScalar scale_a = ElementScalar(1);
//     ElementScalar scale_b = ElementScalar(1);
//     ElementScalar scale_c = ElementScalar(1);
//     ElementScalar scale_d = ElementScalar(1);
//     ElementScalar const* scale_a_ptr = nullptr;
//     ElementScalar const* scale_b_ptr = nullptr;
//     ElementScalar const* scale_c_ptr = nullptr;
//     ElementScalar const* scale_d_ptr = nullptr;
// #endif

//     //
//     // Methods
//     //

//     CUTLASS_HOST_DEVICE
//     Params() {}
  };

public:

  /// Constructs the function object, possibly loading from pointers in host memory
  CUTLASS_HOST_DEVICE
  AcConvert() {

  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {

  }

  /// Returns true if source is needed based on state of runtime arguments
  CUTLASS_HOST_DEVICE
  constexpr bool is_source_needed() const {
    return false;
  }

  /// Constexpr function to enable the compiler to optimize away the source loading if it is
  /// never needed.
  CUTLASS_HOST_DEVICE
  constexpr bool is_source_ever_needed() const {
    return false;
  }

  /// Computes linear scaling: D = alpha * accumulator + beta * source
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const &accumulator,
    FragmentOutput const &source = FragmentOutput(),
    ElementCompute uniform = ElementCompute(0)) const {

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementAccumulator, kCount, Round> destination_converter;

    return destination_converter(accumulator);
  }

  CUTLASS_HOST_DEVICE
  ElementOutput operator()(ElementAccumulator const accumulator) const {
    NumericConverter<ElementOutput, ElementCompute, Round> destination_converter;
    return destination_converter(accumulator);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace cutlass
