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
  \brief Functor performing linear combination with a maximum operation used by epilogues.
*/

#pragma once

#include <cutlass/half.h>
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/scale_type.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies a linear combination operator to an array of elements.
///
/// D = alpha * accumulator + beta * source + uniform
///
template <
  typename ElementOutput_,                             ///< Data type used to load and store tensors
  int Count,                                           ///< Number of elements computed per operation
  typename ElementAccumulator_ = ElementOutput_,       ///< Accumulator data type
  typename ElementCompute_ = ElementOutput_,           ///< Data type used to compute linear combination
  ScaleType::Kind Scale = ScaleType::Default,          ///< Control Alpha and Beta scaling
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest
>
class LinearCombinationLeakyRelu {                     ///< Support one more input E for epligoue
public:

  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;

  static int const kCount = Count;

  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using ComputeFragment = Array<ElementCompute, kCount>;

  static FloatRoundStyle const kRound = Round;

  #if SAIL_FUSE_OP_EXT
  static int const kExtraEpilogueOpsNum = 0;
  static int const kExtraEpilogueInputs = 0;
  #endif

  /// Host-constructable parameters structure
  struct Params {

    ElementCompute alpha;                  ///< scales accumulators
    ElementCompute beta_c;                 ///< scales source tensor
    ElementCompute threshold;              ///< zero
    ElementCompute leaky_alpha;            ///< alpha value of leaky relu
    ElementCompute const *alpha_ptr;       ///< pointer to accumulator scalar - if not null, loads it from memory
    ElementCompute const *beta_c_ptr;      ///< pointer to source scalar - if not null, loads it from memory
    ElementCompute const *leaky_alpha_ptr; ///< pointer to leaky alpha value - if not null, loads it from memory
    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params():
      alpha(ElementCompute(1)),
      beta_c(ElementCompute(0)),
      threshold(ElementCompute(0)),
      leaky_alpha(ElementCompute(0)),
      alpha_ptr(nullptr),
      beta_c_ptr(nullptr),
      leaky_alpha_ptr(nullptr) { }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute alpha,
      ElementCompute leaky_alpha,
      ElementCompute beta_c = ElementCompute(0),
      ElementCompute threshold = ElementCompute(0)
    ): alpha(alpha), beta_c(beta_c),
    threshold(threshold), leaky_alpha(leaky_alpha), alpha_ptr(nullptr), beta_c_ptr(nullptr),
    leaky_alpha_ptr(nullptr) {

    }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute const *alpha_ptr,
      ElementCompute const *beta_c_ptr = nullptr,
      ElementCompute threshold = ElementCompute(0)
#if SAIL_TMP_WORKAROUND
    ): alpha(alpha_ptr ? *alpha_ptr : 0), beta_c(beta_c_ptr ? *beta_c_ptr : 0), threshold(threshold), alpha_ptr(alpha_ptr), beta_c_ptr(beta_c_ptr)
    {
#else
    ): alpha(0), beta_c(0), threshold(threshold), alpha_ptr(alpha_ptr), beta_c_ptr(beta_c_ptr)
    {
#endif
    }
  };

private:

  //
  // Data members
  //

  ElementCompute alpha_;
  ElementCompute beta_c_;
  ElementCompute threshold_;
  ElementCompute leaky_alpha_;
public:

  /// Constructs the function object, possibly loading from pointers in host memory
  CUTLASS_HOST_DEVICE
  LinearCombinationLeakyRelu(Params const &params) {
#if SAIL_TMP_WORKAROUND
    alpha_ = params.alpha;
    beta_c_ = params.beta_c;
#else
    alpha_ = (params.alpha_ptr ? *params.alpha_ptr : params.alpha);
    beta_c_ = (params.beta_c_ptr ? *params.beta_c_ptr : params.beta_c);
#endif
    threshold_ = params.threshold;
    leaky_alpha_ = params.leaky_alpha;
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    if (Scale == ScaleType::NoBetaScaling) return true;

    if (Scale == ScaleType::OnlyAlphaScaling) return false;

    return beta_c_ != ElementCompute(0);
  }

#if SAIL_EPILOGUE_OPT >= 1
  /// Returns true if output op is invariant
  CUTLASS_HOST_DEVICE
  bool is_invariant() const { return false; }
#endif

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    if (k_partition) {
      beta_c_ = ElementCompute(1);
    }

    if (k_partition != k_partition_count - 1) {
      // set to NaN to make ReLU no-op for all except last k partitions
      int64_t allones = -1;
      threshold_ = reinterpret_cast<ElementCompute const &>(allones);
    }
  }

  /// Computes linear scaling: D = alpha * accumulator + beta * source
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const &accumulator, 
    FragmentOutput const &source) const {

    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round> source_converter;
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    ComputeFragment converted_source = source_converter(source);
    ComputeFragment converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations
    ComputeFragment intermediate;

    multiplies<ComputeFragment> mul_add_source;
    multiply_add<ComputeFragment> mul_add_accumulator;
    LeakyReLu<ComputeFragment> leaky_relu;

    if (Scale == ScaleType::NoBetaScaling)
      intermediate = converted_source;
    else
      intermediate = mul_add_source(beta_c_, converted_source);                             // X =  beta * C + uniform

    intermediate = mul_add_accumulator(alpha_, converted_accumulator, intermediate);    // D = alpha * Accum + X

    // Compute threshold optionally
    intermediate = leaky_relu(threshold_, leaky_alpha_, intermediate);

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

    return destination_converter(intermediate);
  }

  /// Computes linear scaling: D = alpha * accumulator
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const &accumulator) const {

    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    ComputeFragment converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations
    ComputeFragment intermediate;

    multiplies<ComputeFragment> mul_accumulator;
    LeakyReLu<ComputeFragment> leaky_relu;

    intermediate = mul_accumulator(alpha_, converted_accumulator);    // D = alpha * Accum

    // Compute threshold optionally
    intermediate = leaky_relu(threshold_, leaky_alpha_, intermediate);

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

    return destination_converter(intermediate);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Conditional guards to enable partial specialization for packed integers
#if defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 100) && ((__HGGCCC_VER_MAJOR__ > 10) || ((__HGGCCC_VER_MAJOR__ >= 10) && (__HGGCCC_VER_MINOR__ >= 2)))

/// Applies a linear combination operator to an array of elements.
///
/// D = alpha * accumulator + beta * source + uniform
///
/// Special handling for int types

template <
  typename ElementOutput_,                             ///< Data type used to load and store tensors
  int Count,                                           ///< Number of elements computed per operation
  ScaleType::Kind Scale,                               ///< Control Alpha and Beta scaling
  FloatRoundStyle Round
>
class LinearCombinationLeakyRelu <ElementOutput_, Count, int, float, Scale, Round> {
public:

  using ElementOutput = ElementOutput_;
  using ElementAccumulator = int;
  using ElementCompute = float;

  static int const kCount = Count;

  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using ComputeFragment = Array<ElementCompute, kCount>;

  static FloatRoundStyle const kRound = Round;

  #if SAIL_FUSE_OP_EXT
  static int const kExtraEpilogueOpsNum = 0;
  static int const kExtraEpilogueInputs = 0;
  #endif
  /// Host-constructable parameters structure
  struct Params {

    ElementCompute alpha;                  ///< scales accumulators
    ElementCompute beta;                   ///< scales source tensor
    ElementCompute threshold;              ///< minimum value that is output 
    ElementCompute leaky_alpha;            ///< alpha of leaky relu
    ElementCompute const *alpha_ptr;       ///< pointer to accumulator scalar - if not null, loads it from memory
    ElementCompute const *beta_ptr;        ///< pointer to source scalar - if not null, loads it from memory
    ElementCompute const *leaky_alpha_ptr; ///< pointer to alpha value of leaky relu - if not null, loads it from memory
    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params(): 
      alpha(ElementCompute(1)), 
      beta(ElementCompute(0)),
      threshold(ElementCompute(0)),
      leaky_alpha(ElementCompute(1)),
      alpha_ptr(nullptr), 
      beta_ptr(nullptr) { }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute alpha,
      ElementCompute leaky_alpha,
      ElementCompute beta = ElementCompute(0),
      ElementCompute threshold = ElementCompute(0)
    ): alpha(alpha), beta(beta), threshold(threshold), leaky_alpha(leaky_alpha), alpha_ptr(nullptr), beta_ptr(nullptr) {

    }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute const *alpha_ptr,
      ElementCompute const *beta_ptr = nullptr,
      ElementCompute threshold = ElementCompute(0),
      ElementCompute leaky_alpha = ElementCompute(1)
#if SAIL_TMP_WORKAROUND
    ): alpha(alpha_ptr ? *alpha_ptr : 0), beta(beta_ptr ? *beta_ptr : 0), threshold(threshold), leaky_alpha(leaky_alpha), alpha_ptr(alpha_ptr), beta_ptr(beta_ptr) {
#else
    ): alpha(0), beta(0), threshold(threshold), leaky_alpha(leaky_alpha), alpha_ptr(alpha_ptr), beta_ptr(beta_ptr) {
#endif
    }
  };

private:

  //
  // Data members
  //

  ElementCompute alpha_;
  ElementCompute beta_;
  ElementCompute threshold_;
  ElementCompute leaky_alpha_;

public:

  /// Constructs the function object, possibly loading from pointers in host memory
  CUTLASS_HOST_DEVICE
  LinearCombinationLeakyRelu(Params const &params) {
#if SAIL_TMP_WORKAROUND
    alpha_ = params.alpha;
    beta_ = params.beta;
#else
    alpha_ = (params.alpha_ptr ? *params.alpha_ptr : params.alpha);
    beta_ = (params.beta_ptr ? *params.beta_ptr : params.beta);
#endif
    threshold_ = params.threshold;
    leaky_alpha_ = params.leaky_alpha_;
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    if (Scale == ScaleType::NoBetaScaling) return true;

    if (Scale == ScaleType::OnlyAlphaScaling) return false;

    return beta_ != ElementCompute(0);
  }

#if SAIL_EPILOGUE_OPT >= 1
  /// Returns true if output op is invariant
  CUTLASS_HOST_DEVICE
  bool is_invariant() const { return false; }
#endif

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    if (k_partition) {
      beta_ = ElementCompute(1);
    }

    if (k_partition != k_partition_count - 1) {
      // set to NaN to make ReLU no-op for all except last k partitions
      int64_t allones = -1;
      threshold_ = reinterpret_cast<ElementCompute const &>(allones);
    }
  }
  
  /// Computes linear scaling: D = alpha * accumulator + beta * source
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const &accumulator, 
    FragmentOutput const &source) const {

    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round> source_converter;
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    ComputeFragment converted_source = source_converter(source);
    ComputeFragment converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations
    ComputeFragment intermediate;

    multiplies<ComputeFragment> mul_add_source;
    multiply_add<ComputeFragment> mul_add_accumulator;
    LeakyReLu<ComputeFragment> leaky_relu;

    if (Scale == ScaleType::NoBetaScaling)
        intermediate = converted_source;
    else
        intermediate = mul_add_source(beta_, converted_source);                         // X =  beta * C + uniform

    intermediate = mul_add_accumulator(alpha_, converted_accumulator, intermediate);    // D = alpha * Accum + X

    // Compute threshold optionally
    intermediate = leaky_relu(threshold_, leaky_alpha_, intermediate);

    if (platform::is_same<ElementOutput, int32_t>::value ||
        platform::is_same<ElementOutput, uint32_t>::value ||
        platform::is_same<ElementOutput, int16_t>::value ||
        platform::is_same<ElementOutput, uint16_t>::value ||
        platform::is_same<ElementOutput, int8_t>::value ||
        platform::is_same<ElementOutput, uint8_t>::value ||
        platform::is_same<ElementOutput, cutlass::int4b_t>::value ||
        platform::is_same<ElementOutput, cutlass::uint4b_t>::value ||
        platform::is_same<ElementOutput, cutlass::uint1b_t>::value) {
      // Convert floats back to INT
      FragmentAccumulator scaled_accumulator;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) {
        scaled_accumulator[i] = __float2int_rn(intermediate[i]);
      }

      // Convert to destination numeric type
      NumericArrayConverter<ElementOutput, int, kCount, Round>
          destination_converter;

      return destination_converter(scaled_accumulator);
    } else {
      NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
          destination_converter;
      return destination_converter(intermediate);
    }
  }

  /// Computes linear scaling: D = alpha * accumulator
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const &accumulator) const {

    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    ComputeFragment converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations
    ComputeFragment intermediate;

    multiplies<ComputeFragment> mul_accumulator;
    LeakyReLu<ComputeFragment> leaky_relu;

    intermediate = mul_accumulator(alpha_, converted_accumulator);    // D = alpha * Accum

    // Compute threshold optionally
    intermediate = leaky_relu(threshold_, leaky_alpha_, intermediate);

    // Convert floats back to INT
    FragmentAccumulator scaled_accumulator;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      scaled_accumulator[i] = __float2int_rn(intermediate[i]);
    }

    if (platform::is_same<ElementOutput, int32_t>::value ||
        platform::is_same<ElementOutput, uint32_t>::value ||
        platform::is_same<ElementOutput, int16_t>::value ||
        platform::is_same<ElementOutput, uint16_t>::value ||
        platform::is_same<ElementOutput, int8_t>::value ||
        platform::is_same<ElementOutput, uint8_t>::value ||
        platform::is_same<ElementOutput, cutlass::int4b_t>::value ||
        platform::is_same<ElementOutput, cutlass::uint4b_t>::value ||
        platform::is_same<ElementOutput, cutlass::uint1b_t>::value) {
      // Convert floats back to INT
      FragmentAccumulator scaled_accumulator;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) {
        scaled_accumulator[i] = __float2int_rn(intermediate[i]);
      }

      // Convert to destination numeric type
      NumericArrayConverter<ElementOutput, int, kCount, Round>
          destination_converter;

      return destination_converter(scaled_accumulator);
    } else {
      NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
          destination_converter;
      return destination_converter(intermediate);
    }
  }
};

#endif // Conditional guards to enable partial specialization for packed integers

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
