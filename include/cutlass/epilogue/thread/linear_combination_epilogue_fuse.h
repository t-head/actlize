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

/*!file
 brief Functor performing linear combination with a maximum operation used by epilogues.
*/

#pragma once

#include <hggc_fp16.h>
#include <cfloat>

#include "cutlass/half.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"
#include "cutlass/utils.h"

#include "cutlass/numeric_conversion.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/epilogue/thread/synthesize_op.h"
/////////////////////////////////////////////////////////////////////////////////////////////////


namespace cutlass {
namespace epilogue {
namespace thread {


typedef enum {
  CUTLASS_POINTWISE_ADD        = 0,
  CUTLASS_POINTWISE_ADD_SQUARE = 5,
  CUTLASS_POINTWISE_DIV        = 6,
  CUTLASS_POINTWISE_MAX        = 3,
  CUTLASS_POINTWISE_MIN        = 2,
  CUTLASS_POINTWISE_MOD        = 7,
  CUTLASS_POINTWISE_MUL        = 1,
  CUTLASS_POINTWISE_POW        = 8,
  CUTLASS_POINTWISE_SUB        = 9,

  CUTLASS_POINTWISE_ABS   = 10,
  CUTLASS_POINTWISE_CEIL  = 11,
  CUTLASS_POINTWISE_COS   = 12,
  CUTLASS_POINTWISE_EXP   = 13,
  CUTLASS_POINTWISE_FLOOR = 14,
  CUTLASS_POINTWISE_LOG   = 15,
  CUTLASS_POINTWISE_NEG   = 16,
  CUTLASS_POINTWISE_RSQRT = 17,
  CUTLASS_POINTWISE_SIN   = 18,
  CUTLASS_POINTWISE_SQRT  = 4,
  CUTLASS_POINTWISE_TAN   = 19,

  CUTLASS_POINTWISE_RELU_FWD      = 100,
  CUTLASS_POINTWISE_TANH_FWD      = 101,
  CUTLASS_POINTWISE_SIGMOID_FWD   = 102,
  CUTLASS_POINTWISE_ELU_FWD       = 103,
  CUTLASS_POINTWISE_GELU_FWD      = 104,
  CUTLASS_POINTWISE_SOFTPLUS_FWD  = 105,
  CUTLASS_POINTWISE_SWISH_FWD     = 106,

  CUTLASS_POINTWISE_IDENTITY_FWD  = 2000,
  CUTLASS_POINTWISE_LEAKYRELU_FWD = 2001,
  CUTLASS_POINTWISE_CLIP_RELU_FWD = 2002,
  CUTLASS_POINTWISE_PRELU_FWD     = 2003
} epilogueOpMode_t;

template<typename EpilogueOp, typename T, typename EpilogueIn, typename EpilogueOut>
struct EpilogueOperandTraits
{
  /// Signature for unary epilogue ops
  template<typename U, EpilogueOut (U::*)(EpilogueIn const&) const> struct SFINAEUnary1 {};
  template<typename U> CUTLASS_HOST_DEVICE static char SignatureUnary1(SFINAEUnary1<U, &U::operator()>*);
  template<typename U> CUTLASS_HOST_DEVICE static int SignatureUnary1(...);
  template<typename U, EpilogueOut (U::*)(EpilogueIn) const> struct SFINAEUnary2 {};
  template<typename U> CUTLASS_HOST_DEVICE static char SignatureUnary2(SFINAEUnary2<U, &U::operator()>*);
  template<typename U> CUTLASS_HOST_DEVICE static int SignatureUnary2(...);
  /// Signature for binary epilogue ops
  template<typename U, EpilogueOut (U::*)(EpilogueIn const&, EpilogueIn const&) const> struct SFINAEBinary1 {};
  template<typename U> CUTLASS_HOST_DEVICE static char SignatureBinary1(SFINAEBinary1<U, &U::operator()>*);
  template<typename U> CUTLASS_HOST_DEVICE static int SignatureBinary1(...);
  template<typename U, EpilogueOut (U::*)(EpilogueIn, EpilogueIn const&) const> struct SFINAEBinary2 {};
  template<typename U> CUTLASS_HOST_DEVICE static char SignatureBinary2(SFINAEBinary2<U, &U::operator()>*);
  template<typename U> CUTLASS_HOST_DEVICE static int SignatureBinary2(...);
  template<typename U, EpilogueOut (U::*)(EpilogueIn const&, EpilogueIn) const> struct SFINAEBinary3 {};
  template<typename U> CUTLASS_HOST_DEVICE static char SignatureBinary3(SFINAEBinary3<U, &U::operator()>*);
  template<typename U> CUTLASS_HOST_DEVICE static int SignatureBinary3(...);
  template<typename U, EpilogueOut (U::*)(EpilogueIn, EpilogueIn) const> struct SFINAEBinary4 {};
  template<typename U> CUTLASS_HOST_DEVICE static char SignatureBinary4(SFINAEBinary4<U, &U::operator()>*);
  template<typename U> CUTLASS_HOST_DEVICE static int SignatureBinary4(...);
  static const bool unary_op = ((sizeof(SignatureUnary1<EpilogueOp>(0))
                                  + sizeof(SignatureUnary2<EpilogueOp>(0))
                                  ) < sizeof(int2));
  static const bool binary_op = ((sizeof(SignatureBinary1<EpilogueOp>(0))
                                                            + sizeof(SignatureBinary2<EpilogueOp>(0))
                                                            + sizeof(SignatureBinary3<EpilogueOp>(0))
                                                            + sizeof(SignatureBinary4<EpilogueOp>(0))
                                                          ) < sizeof(int4));
  static const int kFragOperandNum = unary_op? 1 :(binary_op ? 2 : 0);
  /// Signature for 0 scalar parameter activation ops
  template<typename U, EpilogueOut (U::*)(EpilogueIn const&) const> struct SFINAEParam0 {};
  template<typename U> CUTLASS_HOST_DEVICE static char SignatureParam0(SFINAEParam0<U, &U::operator()>*);
  template<typename U> CUTLASS_HOST_DEVICE static int SignatureParam0(...);
  /// Signature for 1 scalar parameters activation ops
  template<typename U, EpilogueOut (U::*)(T const&, EpilogueIn const&) const> struct SFINAEParam1 {};
  template<typename U> CUTLASS_HOST_DEVICE static char SignatureParam1(SFINAEParam1<U, &U::operator()>*);
  template<typename U> CUTLASS_HOST_DEVICE static int SignatureParam1(...);
  /// Signature for 2 scalar parameters activation ops
  template<typename U, EpilogueOut (U::*)(T const&, T const&, EpilogueIn const&) const> struct SFINAEParam2 {};
  template<typename U> CUTLASS_HOST_DEVICE static char SignatureParam2(SFINAEParam2<U, &U::operator()>*);
  template<typename U> CUTLASS_HOST_DEVICE static int SignatureParam2(...);
  static const bool param_zero = sizeof(SignatureParam0<EpilogueOp>(0)) < sizeof(int);
  static const bool param_one = sizeof(SignatureParam1<EpilogueOp>(0)) < sizeof(int);
  static const bool param_two = sizeof(SignatureParam2<EpilogueOp>(0)) < sizeof(int);
  static const int kScalarParamNum = unary_op ? 0: ( param_one ? 1 : (param_two ? 2 : -1) );
};

// Trait class to get the real input number of fused epilogue operators in order to save vreg usage, as some operator doesn't have vector input
template<typename EpiElementType, typename EpiInputFragType, typename EpiOutputFragType, typename EpilogueOp, typename... EpilogueOps>
struct EpilogueOpTraits {
  static const int kOperandInputNum = ((EpilogueOperandTraits<EpilogueOp, EpiElementType, EpiInputFragType, EpiOutputFragType>::kFragOperandNum == 2 ? 1 : 0)
                                                                            + EpilogueOpTraits<EpiElementType, EpiInputFragType, EpiOutputFragType, EpilogueOps...>::kOperandInputNum);
  static const bool kIfAnyHeavyEpilogueOp  = IsEpilogueFunctorHeavy<EpilogueOp>::value ? true : EpilogueOpTraits<EpiElementType, EpiInputFragType, EpiOutputFragType, EpilogueOps...>::kIfAnyHeavyEpilogueOp;
};
template<typename EpiElementType, typename EpiInputFragType, typename EpiOutputFragType, typename EpilogueOp>
struct EpilogueOpTraits<EpiElementType, EpiInputFragType, EpiOutputFragType, EpilogueOp> {
  static const int kOperandInputNum = (EpilogueOperandTraits<EpilogueOp, EpiElementType, EpiInputFragType, EpiOutputFragType>::kFragOperandNum == 2 ? 1 : 0);
  static const bool kIfAnyHeavyEpilogueOp  = IsEpilogueFunctorHeavy<EpilogueOp>::value;
};

template <
  typename EpilogueOp,
  typename T,
  typename EpilogueInput,
  typename EpilogueOutput,
  int kFragOperandNum_,
  int kScalarParamNum_
>
struct RunEpilogueOp {
  CUTLASS_DEVICE
  EpilogueOutput operator()( T const& act_param0,
                             T const& act_param1,
                             EpilogueInput const& intermediate,
                             EpilogueInput const& extra_input_val ) {
    static_assert(kFragOperandNum_ == 1
               || kFragOperandNum_ == 2
               || kScalarParamNum_ == 1
               || kScalarParamNum_ == 2,
                  "fused epilogue op's operand number should be 1 or 2, and for activation op's parameter should be 1 or 2");
  }
};

template <
  typename T,
  typename EpilogueInput,
  typename EpilogueOutput,
  int kScalarParamNum_>
struct RunEpilogueOp<void, T, EpilogueInput, EpilogueOutput, 1, kScalarParamNum_> {
  CUTLASS_DEVICE
  EpilogueOutput operator()( T const& act_param0,
                             T const& act_param1,
                             EpilogueInput const& intermediate,
                             EpilogueInput const& extra_input_val ) {
    return intermediate;
  }
};

template <
  typename EpilogueOp,
  typename T,
  typename EpilogueInput,
  typename EpilogueOutput,
  int kScalarParamNum_>
struct RunEpilogueOp<EpilogueOp, T, EpilogueInput, EpilogueOutput, 2, kScalarParamNum_> {
  CUTLASS_DEVICE
  EpilogueOutput operator()( T const& act_param0,
                             T const& act_param1,
                             EpilogueInput const& intermediate,
                             EpilogueInput const& extra_input_val ) {
    return EpilogueOp()(intermediate, extra_input_val);
  }
};

template <
  typename EpilogueOp,
  typename T,
  typename EpilogueInput,
  typename EpilogueOutput,
  int kScalarParamNum_>
struct RunEpilogueOp<EpilogueOp, T, EpilogueInput, EpilogueOutput, 1, kScalarParamNum_> {
  CUTLASS_DEVICE
  EpilogueOutput operator()( T const& act_param0,
                             T const& act_param1,
                             EpilogueInput const& intermediate,
                             EpilogueInput const& extra_input_val ) {
    return EpilogueOp()(intermediate);
  }
};


template <
  typename EpilogueOp,
  typename T,
  typename EpilogueInput,
  typename EpilogueOutput>
struct RunEpilogueOp<EpilogueOp, T, EpilogueInput, EpilogueOutput, 0, 1> {
  CUTLASS_DEVICE
  EpilogueOutput operator()( T const& act_param0,
                             T const& act_param1,
                             EpilogueInput const& intermediate,
                             EpilogueInput const& extra_input_val ) {
    return EpilogueOp()(act_param0, intermediate);
  }
};

template <
  typename EpilogueOp,
  typename T,
  typename EpilogueInput,
  typename EpilogueOutput>
struct RunEpilogueOp<EpilogueOp, T, EpilogueInput, EpilogueOutput, 0, 2> {
  CUTLASS_DEVICE
  EpilogueOutput operator()( T const& act_param0,
                             T const& act_param1,
                             EpilogueInput const& intermediate,
                             EpilogueInput const& extra_input_val ) {
    return EpilogueOp()(act_param0, act_param1, intermediate);
  }
};

template<epilogueOpMode_t op_type, typename ElementType, typename FragmentType>
struct EpilogueOpTypetoFunctor
{
  using CutlassOpType = void;
};

#define EpilogueOptoFunctorMacro(enum_type, cutlass_op_type) \
template<typename ElementType, typename FragmentType> \
struct EpilogueOpTypetoFunctor<enum_type, ElementType, FragmentType> \
{ \
  using CutlassOpType = cutlass_op_type<FragmentType>; \
  using EpilogueOperandTraitType = EpilogueOperandTraits<CutlassOpType, ElementType, FragmentType, FragmentType>; \
  using RunEpilogueOpType = RunEpilogueOp< \
                                          CutlassOpType, ElementType, FragmentType, FragmentType, \
                                          EpilogueOperandTraitType::kFragOperandNum, \
                                          EpilogueOperandTraitType::kScalarParamNum >; \
}

EpilogueOptoFunctorMacro(CUTLASS_POINTWISE_ADD, cutlass::plus);
EpilogueOptoFunctorMacro(CUTLASS_POINTWISE_MUL, cutlass::multiplies);
EpilogueOptoFunctorMacro(CUTLASS_POINTWISE_SUB, cutlass::minus);
EpilogueOptoFunctorMacro(CUTLASS_POINTWISE_DIV, cutlass::divides);
EpilogueOptoFunctorMacro(CUTLASS_POINTWISE_TAN, cutlass::epilogue::thread::Tanh);
EpilogueOptoFunctorMacro(CUTLASS_POINTWISE_TANH_FWD, cutlass::epilogue::thread::Tanh);
EpilogueOptoFunctorMacro(CUTLASS_POINTWISE_RELU_FWD, cutlass::epilogue::thread::ReLu);
EpilogueOptoFunctorMacro(CUTLASS_POINTWISE_PRELU_FWD, cutlass::epilogue::thread::PReLu);
EpilogueOptoFunctorMacro(CUTLASS_POINTWISE_LEAKYRELU_FWD, cutlass::epilogue::thread::LeakyReLu);
EpilogueOptoFunctorMacro(CUTLASS_POINTWISE_CLIP_RELU_FWD, cutlass::epilogue::thread::ClipReLu);
EpilogueOptoFunctorMacro(CUTLASS_POINTWISE_ELU_FWD, cutlass::epilogue::thread::ELu);
EpilogueOptoFunctorMacro(CUTLASS_POINTWISE_SIGMOID_FWD, cutlass::epilogue::thread::Sigmoid);
EpilogueOptoFunctorMacro(CUTLASS_POINTWISE_IDENTITY_FWD, cutlass::epilogue::thread::Identity);

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
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest,
  PrefetchStrategyType::Kind PrefetchStrategy_ = PrefetchStrategyType::kNoPrefetchNeeded,
  int EpilogueInputNumByParam = 0, ///< corresponds to EpilogueInputNumByParam > 0
  typename... FuseInputOps
>
class LinearCombinationEpilogueFuse {                  ///< Support one more input E for epligoue
public:

  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  static PrefetchStrategyType::Kind const PrefetchStrategy = PrefetchStrategy_;

  static int const kCount = Count;

  #if SAIL_FUSE_OP_EXT
  static int const kExtraEpilogueBinaryInputs = EpilogueInputNumByParam;
  static int const kExtraEpilogueInputs = EpilogueInputNumByParam;
  static int const kExtraEpilogueOpsNum = EpilogueInputNumByParam;
  #endif

  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using ComputeFragment = Array<ElementCompute, kCount>;

#if SAIL_FUSE_ALIGN_CUDNN
  using ElementEpiComputeType = ElementAccumulator;
  using FragmentEpiComputeType = FragmentAccumulator;
#else
// TODO: Now reuse ElementCompute for representing the compute type used in epilogue currently
//       Should use dedicated compute type for epilogue phase later on after fullly supporting backend API
  using ElementEpiComputeType = ElementCompute;
  using FragmentEpiComputeType = ComputeFragment;
#endif

  static FloatRoundStyle const kRound = Round;

  /// Host-constructable parameters structure
  struct Params {

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
    epilogueOpMode_t epi_ops_list[EpilogueInputNumByParam];
    #endif
    ElementCompute const *act_param0_ptr; ///< pointer to activation op's first parameter
    ElementCompute const *act_param1_ptr; ///< pointer to leaky alpha value - if not null, loads it from memory
    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params():
      alpha(ElementCompute(1)),
      beta_c(ElementCompute(0)),
      #if SAIL_FUSE_OP_EXT
      beta_e(ElementCompute(0)),
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
      const ElementCompute alpha,
      const ElementCompute beta_c = ElementCompute(0),
      const ElementCompute beta_e = ElementCompute(0),
      const ElementCompute act_param0 = ElementCompute(0),
      const ElementCompute act_param1 = ElementCompute(1.0),
      const epilogueOpMode_t *p_epi_ops_list = nullptr
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
      if (p_epi_ops_list) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < EpilogueInputNumByParam; i++) {
          epi_ops_list[i] = p_epi_ops_list[i];
        }
      }
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
    ): alpha(0), beta_c(0), alpha_ptr(alpha_ptr), beta_c_ptr(beta_c_ptr)
    #if SAIL_FUSE_OP_EXT
    , beta_e(0)
    , beta_e_ptr(beta_e_ptr)
    #endif
    , act_param0(0.0)
    , act_param1(1.0)
    , act_param0_ptr(act_param0_ptr)
    , act_param1_ptr(act_param1_ptr)
    {
    }
  };

  template <epilogueOpMode_t op_type>
  struct MapEpilogueOpTypeToFunctor {
    using EpilogueFunctor = typename EpilogueOpTypetoFunctor<op_type, ElementEpiComputeType, FragmentEpiComputeType>::RunEpilogueOpType;
  };

private:

  //
  // Data members
  //

  ElementCompute alpha_;
  ElementCompute beta_c_;
  ElementCompute beta_e_;
  epilogueOpMode_t epi_ops_list[EpilogueInputNumByParam];
  ElementCompute act_param0_;
  ElementCompute act_param1_;

public:

  /// Constructs the function object, possibly loading from pointers in host memory
  CUTLASS_HOST_DEVICE
  LinearCombinationEpilogueFuse(Params const &params) {
    alpha_ = (params.alpha_ptr ? *params.alpha_ptr : params.alpha);
    beta_c_ = (params.beta_c_ptr ? *params.beta_c_ptr : params.beta_c);
    #if SAIL_FUSE_OP_EXT
    beta_e_ = (params.beta_e_ptr ? *params.beta_e_ptr : params.beta_e);
    #endif
    act_param0_ = (params.act_param0_ptr ? *params.act_param0_ptr : params.act_param0);
    act_param1_ = (params.act_param1_ptr ? *params.act_param1_ptr : params.act_param1);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < EpilogueInputNumByParam; i++) {
      epi_ops_list[i] = params.epi_ops_list[i];
    }

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
      act_param0_ = reinterpret_cast<ElementCompute const &>(allones);
    }
  }

  /// Computes linear scaling: D = (alpha * accumulator + beta_c * source) + beta_e * input_e

  CUTLASS_DEVICE
  void dispatch_epilogue_op(epilogueOpMode_t epi_enum,
                            FragmentEpiComputeType& temp_e,
                            FragmentEpiComputeType& intermediate
                            ) const {
    switch(epi_enum)
    {
      case CUTLASS_POINTWISE_ADD:
          intermediate = typename MapEpilogueOpTypeToFunctor<CUTLASS_POINTWISE_ADD>::EpilogueFunctor()(ElementEpiComputeType(act_param0_), ElementEpiComputeType(act_param1_), intermediate, temp_e);
          break;
      case CUTLASS_POINTWISE_MUL:
          intermediate = typename MapEpilogueOpTypeToFunctor<CUTLASS_POINTWISE_MUL>::EpilogueFunctor()(ElementEpiComputeType(act_param0_), ElementEpiComputeType(act_param1_), intermediate, temp_e);
          break;
      case CUTLASS_POINTWISE_DIV:
          intermediate = typename MapEpilogueOpTypeToFunctor<CUTLASS_POINTWISE_DIV>::EpilogueFunctor()(ElementEpiComputeType(act_param0_), ElementEpiComputeType(act_param1_), intermediate, temp_e);
          break;
      case CUTLASS_POINTWISE_SUB:
          intermediate = typename MapEpilogueOpTypeToFunctor<CUTLASS_POINTWISE_SUB>::EpilogueFunctor()(ElementEpiComputeType(act_param0_), ElementEpiComputeType(act_param1_), intermediate, temp_e);
          break;
      case CUTLASS_POINTWISE_RELU_FWD:
          intermediate = typename MapEpilogueOpTypeToFunctor<CUTLASS_POINTWISE_RELU_FWD>::EpilogueFunctor()(ElementEpiComputeType(act_param0_), ElementEpiComputeType(act_param1_), intermediate, temp_e);
          break;
      case CUTLASS_POINTWISE_PRELU_FWD:
          intermediate = typename MapEpilogueOpTypeToFunctor<CUTLASS_POINTWISE_PRELU_FWD>::EpilogueFunctor()(ElementEpiComputeType(act_param0_), ElementEpiComputeType(act_param1_), intermediate, temp_e);
          break;
      case CUTLASS_POINTWISE_TAN:
      case CUTLASS_POINTWISE_TANH_FWD:
          intermediate = typename MapEpilogueOpTypeToFunctor<CUTLASS_POINTWISE_TAN>::EpilogueFunctor()(ElementEpiComputeType(act_param0_), ElementEpiComputeType(act_param1_), intermediate, temp_e);
          break;
      case CUTLASS_POINTWISE_LEAKYRELU_FWD:
          intermediate = typename MapEpilogueOpTypeToFunctor<CUTLASS_POINTWISE_LEAKYRELU_FWD>::EpilogueFunctor()(ElementEpiComputeType(act_param0_), ElementEpiComputeType(act_param1_), intermediate, temp_e);
          break;
      case CUTLASS_POINTWISE_SIGMOID_FWD:
          intermediate = typename MapEpilogueOpTypeToFunctor<CUTLASS_POINTWISE_SIGMOID_FWD>::EpilogueFunctor()(ElementEpiComputeType(act_param0_), ElementEpiComputeType(act_param1_), intermediate, temp_e);
          break;
      case CUTLASS_POINTWISE_CLIP_RELU_FWD:
          intermediate = typename MapEpilogueOpTypeToFunctor<CUTLASS_POINTWISE_CLIP_RELU_FWD>::EpilogueFunctor()(ElementEpiComputeType(act_param0_), ElementEpiComputeType(act_param1_), intermediate, temp_e);
          break;
      case CUTLASS_POINTWISE_ELU_FWD:
          intermediate = typename MapEpilogueOpTypeToFunctor<CUTLASS_POINTWISE_ELU_FWD>::EpilogueFunctor()(ElementEpiComputeType(act_param0_), ElementEpiComputeType(act_param1_), intermediate, temp_e);
          break;
      case CUTLASS_POINTWISE_IDENTITY_FWD:
          intermediate = typename MapEpilogueOpTypeToFunctor<CUTLASS_POINTWISE_IDENTITY_FWD>::EpilogueFunctor()(ElementEpiComputeType(act_param0_), ElementEpiComputeType(act_param1_), intermediate, temp_e);
          break;
      default :
          break;
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

    intermediate = mul_accumulator(alpha_, converted_accumulator);    // D = alpha * Accum

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

    return destination_converter(intermediate);
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

    intermediate = mul_add_source(beta_c_, converted_source);                             // X =  beta * C + uniform
    intermediate = mul_add_accumulator(alpha_, converted_accumulator, intermediate);    // D = alpha * Accum + X

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

    return destination_converter(intermediate);
  }

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const &accumulator,
    FragmentOutput const &source,
    FragmentOutput (&extra_input_vals)[kExtraEpilogueInputs] ) const {

    NumericArrayConverter<ElementEpiComputeType, ElementOutput, kCount, Round> source_converter;
    NumericArrayConverter<ElementEpiComputeType, ElementAccumulator, kCount, Round> accumulator_converter;

    FragmentEpiComputeType converted_source = source_converter(source);
    FragmentEpiComputeType converted_accumulator = accumulator_converter(accumulator);

    FragmentEpiComputeType intermediate;
    multiplies<FragmentEpiComputeType> mul_compute;
    multiply_add<FragmentEpiComputeType> mul_add_accumulator;
    if (Scale == ScaleType::NoBetaScaling) {
      intermediate = mul_add_accumulator(ElementEpiComputeType(alpha_), converted_accumulator, converted_source);
    } else {
      intermediate = mul_add_accumulator(ElementEpiComputeType(alpha_), converted_accumulator, mul_compute(ElementEpiComputeType(beta_c_), converted_source));
    }

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementEpiComputeType, kCount, Round> destination_converter;
    FragmentEpiComputeType temp_e;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < EpilogueInputNumByParam; i++) {
      temp_e = mul_compute(ElementEpiComputeType(beta_e_), source_converter(extra_input_vals[i]));
      dispatch_epilogue_op(epi_ops_list[i], temp_e, intermediate);
    } // for

    return destination_converter(intermediate);
  }

  /// Computes linear scaling: D = alpha * accumulator
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const &accumulator,
    FragmentOutput (&extra_input_vals)[kExtraEpilogueInputs] ) const {

    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementEpiComputeType, ElementOutput, kCount, Round> extra_input_converter;
    NumericArrayConverter<ElementEpiComputeType, ElementAccumulator, kCount, Round> accumulator_converter;

    FragmentEpiComputeType converted_accumulator = accumulator_converter(accumulator);

    // Perform binary or unary operations
    multiplies<FragmentEpiComputeType> mul;
    FragmentEpiComputeType intermediate = mul(ElementEpiComputeType(alpha_), converted_accumulator);

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementEpiComputeType, kCount, Round> destination_converter;
    FragmentEpiComputeType temp_e;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < EpilogueInputNumByParam; i++) {
      temp_e = mul(ElementEpiComputeType(beta_e_), extra_input_converter(extra_input_vals[i]));
      dispatch_epilogue_op(epi_ops_list[i], temp_e, intermediate);
    }

    return destination_converter(intermediate);
  }
};
template <
  typename ElementOutput_,                             ///< Data type used to load and store tensors
  int Count,                                           ///< Number of elements computed per operation
  typename ElementAccumulator_,       ///< Accumulator data type
  typename ElementCompute_,           ///< Data type used to compute linear combination
  ScaleType::Kind Scale,          ///< Control Alpha and Beta scaling
  FloatRoundStyle Round,
  PrefetchStrategyType::Kind PrefetchStrategy_,
  typename... FuseInputOps
>
class LinearCombinationEpilogueFuse< ElementOutput_,
                                     Count,
                                     ElementAccumulator_,
                                     ElementCompute_,
                                     Scale,
                                     Round,
                                     PrefetchStrategy_,
                                     0,
                                     FuseInputOps...
                                   > {                     ///< Support one more input E for epligoue
public:

  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;

  static PrefetchStrategyType::Kind const PrefetchStrategy = PrefetchStrategy_;
  static int const kCount = Count;


  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using ComputeFragment = Array<ElementCompute, kCount>;

#if SAIL_FUSE_ALIGN_CUDNN
  using ElementEpiComputeType = ElementAccumulator;
  using FragmentEpiComputeType = FragmentAccumulator;
#else
// TODO: Now reuse ElementCompute for representing the compute type used in epilogue currently
//       Should use dedicated compute type for epilogue phase later on after fullly supporting backend API
  using ElementEpiComputeType = ElementCompute;
  using FragmentEpiComputeType = ComputeFragment;
#endif

  static FloatRoundStyle const kRound = Round;

  #if SAIL_FUSE_OP_EXT
  static int const kExtraEpilogueBinaryInputs = EpilogueOpTraits<ElementEpiComputeType, FragmentEpiComputeType, FragmentEpiComputeType, FuseInputOps...>::kOperandInputNum;
  static int const kExtraEpilogueInputs  = kExtraEpilogueBinaryInputs > 1 ? kExtraEpilogueBinaryInputs : 1;
  static int const kExtraEpilogueOpsNum = sizeof...(FuseInputOps);
  static bool const kIsHeavy = EpilogueOpTraits<ElementEpiComputeType, FragmentEpiComputeType, FragmentEpiComputeType, FuseInputOps...>::kIfAnyHeavyEpilogueOp;
  #endif

  using SynthesizedOp = SynthesizeEpilogueOps<FragmentEpiComputeType, FragmentOutput, kExtraEpilogueInputs, kCount, FuseInputOps...>;


  /// Host-constructable parameters structure
  struct Params {

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

private:

  //
  // Data members
  //

  ElementCompute alpha_;
  ElementCompute beta_c_;
  ElementCompute beta_e_;
  ElementCompute act_param0_;
  ElementCompute act_param1_;

  bool is_fake_relu;
public:

  /// Constructs the function object, possibly loading from pointers in host memory
  CUTLASS_HOST_DEVICE
  LinearCombinationEpilogueFuse(Params const &params) {
    alpha_ = (params.alpha_ptr ? *params.alpha_ptr : params.alpha);
    beta_c_ = (params.beta_c_ptr ? *params.beta_c_ptr : params.beta_c);
    #if SAIL_FUSE_OP_EXT
    beta_e_ = (params.beta_e_ptr ? *params.beta_e_ptr : params.beta_e);
    act_param0_ = params.act_param0_ptr ? *params.act_param0_ptr : params.act_param0;
    act_param1_ = params.act_param1_ptr ? *params.act_param1_ptr : params.act_param1;

    // TODO: remove is_fake_relu if can support Conv+Add+Bias directly(now throught Conv+Add+Bias+FakeRelu)
    is_fake_relu = (platform::is_same<ElementCompute, float>::value && float(act_param0_) == -FLT_MAX)
                  || (platform::is_same<ElementCompute, half>::value && float(act_param0_) == -FLT_MAX);
    #endif
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
      act_param0_ = reinterpret_cast<ElementCompute const &>(allones);
    }
  }

  CUTLASS_HOST_DEVICE
  void load_epilogue_inputs(
    const void**    extra_ptrs,
    uint64_t*      extra_offsets,
    FragmentOutput (&extra_input_vals)[kExtraEpilogueInputs] ) const {

    if (kExtraEpilogueBinaryInputs != 0){
      int extra_input_seq = 0;

      (void)std::initializer_list<int>{
        (extra_input_vals[extra_input_seq] = *reinterpret_cast<const FragmentOutput*>
                                                                (reinterpret_cast<const ElementOutput*>(extra_ptrs[extra_input_seq]) + extra_offsets[extra_input_seq])
        , (EpilogueOperandTraits<FuseInputOps, ElementEpiComputeType, FragmentEpiComputeType, FragmentEpiComputeType>::kFragOperandNum == 2 && (extra_input_seq+1) < kExtraEpilogueInputs
            ? ++extra_input_seq : extra_input_seq)
        )...
      };
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

    intermediate = mul_accumulator(alpha_, converted_accumulator);    // D = alpha * Accum

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

    return destination_converter(intermediate);
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

    intermediate = mul_add_source(beta_c_, converted_source);                             // X =  beta * C + uniform
    intermediate = mul_add_accumulator(alpha_, converted_accumulator, intermediate);    // D = alpha * Accum + X

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

    return destination_converter(intermediate);
  }

  /// Computes linear scaling: D = (alpha * accumulator + beta_c * source) + beta_e * input_e
  #if SAIL_FUSE_OP_EXT
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const &accumulator,
    FragmentOutput const &source,
    FragmentOutput (&extra_input_vals)[kExtraEpilogueInputs] ) const {

    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementEpiComputeType, ElementOutput, kCount, Round> source_converter;
    NumericArrayConverter<ElementEpiComputeType, ElementAccumulator, kCount, Round> accumulator_converter;
    NumericConverter<ElementEpiComputeType, ElementCompute, Round> act_param_converter;

    FragmentEpiComputeType converted_source = source_converter(source);
    FragmentEpiComputeType converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations
    FragmentEpiComputeType intermediate;

    multiplies<FragmentEpiComputeType> mul;
    multiply_add<FragmentEpiComputeType> mul_add_accumulator;

    if (Scale == ScaleType::NoBetaScaling) {
      intermediate = mul_add_accumulator(ElementEpiComputeType(alpha_), converted_accumulator, converted_source);
    } else {
      intermediate = mul_add_accumulator(ElementEpiComputeType(alpha_), converted_accumulator, mul(ElementEpiComputeType(beta_c_), converted_source));
    }

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementEpiComputeType, kCount, Round> destination_converter;

    // TODO: remove is_fake_relu if can support Conv+Add+Bias directly(now throught Conv+Add+Bias+FakeRelu)
    if (platform::is_same<typename SynthesizedOp::type, void>::value == false && !is_fake_relu) {
      SynthesizedOp()(intermediate, beta_e_,  extra_input_vals);
    } else {
      if (kExtraEpilogueBinaryInputs == 0) {
        int extra_input_seq = 0;
        (void)std::initializer_list<int>{ ( intermediate =
                                      RunEpilogueOp<FuseInputOps, ElementEpiComputeType, FragmentEpiComputeType, FragmentEpiComputeType,
                                                    EpilogueOperandTraits<FuseInputOps, ElementEpiComputeType, FragmentEpiComputeType, FragmentEpiComputeType>::kFragOperandNum,
                                                    EpilogueOperandTraits<FuseInputOps, ElementEpiComputeType, FragmentEpiComputeType, FragmentEpiComputeType>::kScalarParamNum>
                                      ()(act_param_converter(act_param0_), act_param_converter(act_param1_), intermediate, intermediate), extra_input_seq
                                    )...
                                  };
      } else {
        int extra_input_seq = 0;
        (void)std::initializer_list<int>{ ( intermediate =
                                      RunEpilogueOp<FuseInputOps, ElementEpiComputeType, FragmentEpiComputeType, FragmentEpiComputeType,
                                                    EpilogueOperandTraits<FuseInputOps, ElementEpiComputeType, FragmentEpiComputeType, FragmentEpiComputeType>::kFragOperandNum,
                                                    EpilogueOperandTraits<FuseInputOps, ElementEpiComputeType, FragmentEpiComputeType, FragmentEpiComputeType>::kScalarParamNum>
                                      ()(act_param_converter(act_param0_), act_param_converter(act_param1_), intermediate, mul(act_param_converter(beta_e_), source_converter(extra_input_vals[extra_input_seq])))
                                      , (EpilogueOperandTraits<FuseInputOps, ElementEpiComputeType, FragmentEpiComputeType, FragmentEpiComputeType>::kFragOperandNum == 2 && (extra_input_seq+1) < kExtraEpilogueInputs
                                        ? ++extra_input_seq : extra_input_seq)
                                    )...
                                  };
      }
    }


    return destination_converter(intermediate);
  }
  #endif

  /// Computes linear scaling: D = alpha * accumulator
  #if SAIL_FUSE_OP_EXT
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const &accumulator,
    FragmentOutput (&extra_input_vals)[kExtraEpilogueInputs] ) const {
    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementEpiComputeType, ElementOutput, kCount, Round> extra_input_converter;
    NumericArrayConverter<ElementEpiComputeType, ElementAccumulator, kCount, Round> accumulator_converter;
    FragmentEpiComputeType converted_accumulator = accumulator_converter(accumulator);

    // Perform binary or unary operations
    multiplies<FragmentEpiComputeType> mul;
    FragmentAccumulator intermediate = mul(ElementEpiComputeType(alpha_), converted_accumulator);

    NumericArrayConverter<ElementOutput, ElementEpiComputeType, kCount, Round> destination_converter;

    // TODO: remove is_fake_relu if can support Conv+Add+Bias directly(now throught Conv+Add+Bias+FakeRelu)
    if (platform::is_same<typename SynthesizedOp::type, void>::value == false && !is_fake_relu) {
      SynthesizedOp()(intermediate, beta_e_,  extra_input_vals);
    } else {
      if (kExtraEpilogueBinaryInputs == 0) {
        int extra_input_seq = 0;
        (void)std::initializer_list<int>{ ( intermediate =
                                      RunEpilogueOp<FuseInputOps, ElementEpiComputeType, FragmentEpiComputeType, FragmentEpiComputeType,
                                                    EpilogueOperandTraits<FuseInputOps, ElementEpiComputeType, FragmentEpiComputeType, FragmentEpiComputeType>::kFragOperandNum,
                                                    EpilogueOperandTraits<FuseInputOps, ElementEpiComputeType, FragmentEpiComputeType, FragmentEpiComputeType>::kScalarParamNum>
                                      ()(ElementEpiComputeType(act_param0_), ElementEpiComputeType(act_param1_), intermediate, intermediate), extra_input_seq
                                    )...
                                  };
      } else {
        int extra_input_seq = 0;
        (void)std::initializer_list<int>{ ( intermediate =
                                      RunEpilogueOp<FuseInputOps, ElementEpiComputeType, FragmentEpiComputeType, FragmentEpiComputeType,
                                                    EpilogueOperandTraits<FuseInputOps, ElementEpiComputeType, FragmentEpiComputeType, FragmentEpiComputeType>::kFragOperandNum,
                                                    EpilogueOperandTraits<FuseInputOps, ElementEpiComputeType, FragmentEpiComputeType, FragmentEpiComputeType>::kScalarParamNum>
                                      ()(ElementEpiComputeType(act_param0_), ElementEpiComputeType(act_param1_), intermediate, mul(ElementEpiComputeType(beta_e_), extra_input_converter(extra_input_vals[extra_input_seq])))
                                      , (EpilogueOperandTraits<FuseInputOps, ElementEpiComputeType, FragmentEpiComputeType, FragmentEpiComputeType>::kFragOperandNum == 2 && (extra_input_seq+1) < kExtraEpilogueInputs
                                        ? ++extra_input_seq : extra_input_seq)
                                    )...
                                  };
      }
    }

    return destination_converter(intermediate);
  }
  #endif
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
class LinearCombinationEpilogueFuse <ElementOutput_, Count, int, float, Scale, Round> {
public:

  using ElementOutput = ElementOutput_;
  using ElementAccumulator = int;
  using ElementCompute = float;

  static int const kCount = Count;

  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using ComputeFragment = Array<ElementCompute, kCount>;

  static FloatRoundStyle const kRound = Round;

  /// Host-constructable parameters structure
  struct Params {

    ElementCompute alpha;                  ///< scales accumulators
    ElementCompute beta;                   ///< scales source tensor
    ElementCompute act_param0;              ///< minimum value that is output
    ElementCompute act_param1;            ///< alpha of leaky relu
    ElementCompute const *alpha_ptr;       ///< pointer to accumulator scalar - if not null, loads it from memory
    ElementCompute const *beta_ptr;        ///< pointer to source scalar - if not null, loads it from memory
    ElementCompute const *act_param1_ptr; ///< pointer to alpha value of leaky relu - if not null, loads it from memory
    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params():
      alpha(ElementCompute(1)),
      beta(ElementCompute(0)),
      act_param0(ElementCompute(0)),
      act_param1(ElementCompute(1)),
      alpha_ptr(nullptr),
      beta_ptr(nullptr) { }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute alpha,
      ElementCompute act_param1,
      ElementCompute beta = ElementCompute(0),
      ElementCompute act_param0 = ElementCompute(0)
    ): alpha(alpha), beta(beta), act_param0(act_param0), act_param1(act_param1), alpha_ptr(nullptr), beta_ptr(nullptr) {

    }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute const *alpha_ptr,
      ElementCompute const *beta_ptr = nullptr,
      ElementCompute act_param0 = ElementCompute(0),
      ElementCompute act_param1 = ElementCompute(1)
    ): alpha(0), beta(0), act_param0(act_param0), act_param1(act_param1), alpha_ptr(alpha_ptr), beta_ptr(beta_ptr) {
    }
  };

private:

  //
  // Data members
  //

  ElementCompute alpha_;
  ElementCompute beta_;
  ElementCompute act_param0_;
  ElementCompute act_param1_;

public:

  /// Constructs the function object, possibly loading from pointers in host memory
  CUTLASS_HOST_DEVICE
  LinearCombinationEpilogueFuse(Params const &params) {
    alpha_ = (params.alpha_ptr ? *params.alpha_ptr : params.alpha);
    beta_ = (params.beta_ptr ? *params.beta_ptr : params.beta);
    act_param0_ = params.act_param0;
    act_param1_ = params.act_param1_;

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
      act_param0_ = reinterpret_cast<ElementCompute const &>(allones);
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

    // Compute act_param0 optionally
    intermediate = leaky_relu(act_param0_, act_param1_, intermediate);

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

    // Compute act_param0 optionally
    intermediate = leaky_relu(act_param0_, act_param1_, intermediate);

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

