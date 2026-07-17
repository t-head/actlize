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
class Convert {
public:

  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementAccumulator_;

  static int const kCount = Count;

  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using ComputeFragment = FragmentAccumulator;

  static FloatRoundStyle const kRound = Round;

  #if SAIL_FUSE_OP_EXT
  static int const kExtraEpilogueOpsNum = 0;
  static int const kExtraEpilogueInputs = 0;
  #endif

  /// Host-constructable parameters structure
  struct Params {

    //
    // Methods
    //

    ElementCompute alpha;                  ///< scales accumulators
    ElementCompute beta_c;                 ///< scales source tensor
    #if SAIL_FUSE_OP_EXT
    ElementCompute beta_e;                 ///< scales fuse op input E tensor
    #endif
    ElementCompute act_param0;              ///< zero
    ElementCompute act_param1;            ///< alpha value of leaky relu
    ElementCompute const *alpha_ptr;       ///< pointer to accumulator scalar - if not null, loads it from memory
    ElementCompute const *beta_c_ptr;      ///< pointer to source scalar - if not null, loads it from memory
    #if SAIL_FUSE_OP_EXT
    ElementCompute const *beta_e_ptr;      ///< pointer to input E scalar - if not null, loads it from memory
    #endif
    ElementCompute const *act_param0_ptr; ///< pointer to activation op's first parameter - if not null, loads it from memory
    ElementCompute const *act_param1_ptr; ///< pointer to leaky alpha value - if not null, loads it from memory
    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params():
      alpha(ElementCompute(1)),
      beta_c(ElementCompute(0)),
      #if SAIL_FUSE_OP_EXT
      beta_e(ElementCompute(1)),
      #endif
      act_param0(ElementCompute(0)),
      act_param1(ElementCompute(0)),
      alpha_ptr(nullptr),
      beta_c_ptr(nullptr),
      #if SAIL_FUSE_OP_EXT
      beta_e_ptr(nullptr),
      #endif
      act_param0_ptr(nullptr),
      act_param1_ptr(nullptr) { }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute alpha,
      ElementCompute beta_c = ElementCompute(0),
      ElementCompute beta_e = ElementCompute(1),
      ElementCompute act_param0 = ElementCompute(0),
      ElementCompute act_param1 = ElementCompute(1.0)
    ): alpha(alpha), beta_c(beta_c),
    #if SAIL_FUSE_OP_EXT
    beta_e(beta_e),
    #endif
    act_param0(act_param0), act_param1(act_param1), alpha_ptr(nullptr), beta_c_ptr(nullptr),
    #if SAIL_FUSE_OP_EXT
    beta_e_ptr(nullptr),
    #endif
    act_param0_ptr(nullptr),
    act_param1_ptr(nullptr) {
    }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute const *alpha_ptr,
      ElementCompute const *beta_c_ptr = nullptr,
      #if SAIL_FUSE_OP_EXT
      ElementCompute const *beta_e_ptr = nullptr,
      #endif
      ElementCompute const *act_param0_ptr = nullptr,
      ElementCompute const *act_param1_ptr = nullptr
    ): alpha(1), beta_c(0), alpha_ptr(alpha_ptr), beta_c_ptr(beta_c_ptr)
    #if SAIL_FUSE_OP_EXT
    , beta_e(1)
    , beta_e_ptr(beta_e_ptr)
    , act_param0(0.0)
    , act_param1(1.0)
    , act_param0_ptr(act_param0_ptr)
    , act_param1_ptr(act_param1_ptr)
    #endif
    {
    }
  };

public:

  /// Constructs the function object, possibly loading from pointers in host memory
  CUTLASS_HOST_DEVICE
  Convert(Params const &params = Params()) {

  }

  /// Returns true if source is needed based on state of runtime arguments
  CUTLASS_HOST_DEVICE
  constexpr bool is_source_needed() const {
    return false;
  }

#if SAIL_EPILOGUE_OPT >= 1
  /// Returns true if output op is invariant
  CUTLASS_HOST_DEVICE
  constexpr bool is_invariant() const { return false; }
#endif

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
  void set_k_partition(int k_partition, int k_partition_count) {}
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace cutlass
