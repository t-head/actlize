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
    \brief This extends the contents of cutlass/functional.h with frequently used activation functions.

*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/constants.h"
#include "cutlass/complex.h"
#include "cutlass/array.h"
#include "cutlass/half.h"
#include "cutlass/functional.h"
#include "cutlass/fast_math.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct Identity {
  CUTLASS_HOST_DEVICE
  T operator()(T value) const {
    return value;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// ReLu operator - propagates NaNs
template <typename T>
struct ReLu {
  CUTLASS_HOST_DEVICE
  T operator()(T const & threshold, T value) const {
    if (value < threshold) {
      value = threshold;
    }
    return value;
  }
  CUTLASS_HOST_DEVICE
  T operator()(T value) const {
    if (value < T()) {
      value = T();
    }
    return value;
  }
};

template <typename T, int N>
struct ReLu<Array<T, N>> {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(T const & threshold, Array<T, N> const &frag) const {
    Array<T, N> result;
    ReLu<T> relu_op;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = relu_op(threshold, frag[i]);
    }
    return result;
  }
};

/// LeakyReLu operator - propagates NaNs
template <typename T>
struct LeakyReLu {
  CUTLASS_HOST_DEVICE
  T operator()(T const & threshold, T const & leaky_alpha, T value) const {
    if (value < threshold) {
      value *= leaky_alpha;
    }
    return value;
  }
  CUTLASS_HOST_DEVICE
  T operator()(T const & leaky_alpha, T value) const {
    if (value < T()) {
      value *= leaky_alpha;
    }
    return value;
  }
};

template <typename T, int N>
struct LeakyReLu<Array<T, N>> {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(T const & threshold, T const & leaky_alpha, Array<T, N> const &frag) const {
    Array<T, N> result;
    LeakyReLu<T> leaky_relu_op;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = leaky_relu_op(threshold, leaky_alpha, frag[i]);
    }
    return result;
  }
};

// Leaky Relu operator
template <typename T>
struct LeakyReLU {
  CUTLASS_HOST_DEVICE
  T operator()(T const &value, T const & alpha_recip) const {
    T res = value > T(0) ? value : value * alpha_recip;
    return res;
  }
};

template <typename T, int N>
struct LeakyReLU<Array<T, N> > {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &rhs, T const & alpha_recip) const {
    Array<T, N> y;
    LeakyReLU<T> leaky_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < int(rhs.size()); ++i) {
      y[i] = leaky_op(rhs[i], alpha_recip);
    }

    return y;
  }
};

/// ClippedReLu operator - propagates NaNs
template <typename T>
struct ClipReLu {
  CUTLASS_HOST_DEVICE
  T operator()(T const & threshold0, T const & threshold1, T value) const {
    if (value < threshold0) {
      value = threshold0;
    } else if(value > threshold1) {
      value = threshold1;
    }
    return value;
  }
};

template <typename T, int N>
struct ClipReLu<Array<T, N>> {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(T const & threshold0, T const & threshold1, Array<T, N> const &frag) const {
    Array<T, N> result;
    ClipReLu<T> clip_relu_op;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = clip_relu_op(threshold0, threshold1, frag[i]);
    }
    return result;
  }
};

/// ELu operator - propagates NaNs
template <typename T>
struct PReLu {
  CUTLASS_HOST_DEVICE
  T operator()(T value, T const & coeff) const {
    if (value < T(.0f)) {
      value = T(float(float(value) * float(coeff)));
    }
    return value;
  }
};

template <>
struct PReLu<float> {
  CUTLASS_HOST_DEVICE
  float operator()(float value, float const & coeff) const {
    if (value < .0f) {
      value = value * coeff;
    }
    return value;
  }
};

template <typename T, int N>
struct PReLu<Array<T, N>> {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &frag, Array<T, N> const & coeff) const {
    Array<T, N> result;
    PReLu<T> PReLu_op;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = PReLu_op(frag[i], coeff[i]);
    }
    return result;
  }
};

/// ELu operator - propagates NaNs
template <typename T>
struct ELu {
  CUTLASS_HOST_DEVICE
  T operator()(T const & threshold, T const & coeff, T value) const {
    if (value < threshold) {
      value = T(expf(float(value)) - float(1));
    }
    return value;
  }
};

template <>
struct ELu<float> {
  CUTLASS_HOST_DEVICE
  float operator()(float const & threshold, float const & coeff, float value) const {
    if (value < threshold) {
      value = expf(value) - 1.0f;
    }
    return value;
  }
};

template <typename T, int N>
struct ELu<Array<T, N>> {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(T const & param0, T const & param1, Array<T, N> const &frag) const {
    Array<T, N> result;
    ELu<T> elu_op;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = elu_op(param0, param1, frag[i]);
    }
    return result;
  }
};

// Sigmoid operator
template <typename T>
struct Sigmoid {
  CUTLASS_HOST_DEVICE
  T operator()(T const &scalar) const {
    #if defined(__HGGC_ARCH__)
      return T(__ppu_sgmdf(scalar));
    #else
      return T(float(1) / (float(1) + expf(float(-scalar))));
    #endif
  }
};

template <>
struct Sigmoid<float> {
  CUTLASS_HOST_DEVICE
  float operator()(float const &scalar) const {
    #if defined(__HGGC_ARCH__)
      return __ppu_sgmdf(scalar);
    #else
      return 1.0f / (1.0f + expf(-scalar));
    #endif
  }
};

template <typename T, int N>
struct Sigmoid<Array<T, N> > {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &rhs) const {
    Array<T, N> y;
    Sigmoid<T> sigmoid_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < int(rhs.size()); ++i) {
      y[i] = sigmoid_op(rhs[i]);
    }

    return y;
  }
};

// HSigmoid operator
template <typename T>
struct HSigmoid {
  CUTLASS_HOST_DEVICE
  T operator()(T const &scalar) const {
    if (scalar < T(-3.0)) {
      return T(0.0);
    } else if (scalar > T(3.0)) {
      return T(1.0);
    } else {
      return T(scalar * T(1.0 / 6.0) + T(0.5));
    }
  }
};

template <typename T, int N>
struct HSigmoid<Array<T, N> > {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &rhs) const {
    Array<T, N> y;
    HSigmoid<T> hsigmoid_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < int(rhs.size()); ++i) {
      y[i] = hsigmoid_op(rhs[i]);
    }

    return y;
  }
};

// SiLu (swish) operator introduced by Elfwing et al. in the following paper
// "Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning" (2017)
// https://arxiv.org/pdf/1702.03118.pdf
// It is used in EfficientNet and YOLOv5, for example.
// Reference: https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html
template <typename T>
struct SiLu {
  CUTLASS_HOST_DEVICE
  T operator()(T const &scalar) const {
    return T(scalar * Sigmoid<T>()(scalar));
  }
};

template <>
struct SiLu<float> {
  CUTLASS_HOST_DEVICE
  float operator()(float const &scalar) const {
    return scalar * Sigmoid<float>()(scalar);
  }
};

template <typename T, int N>
struct SiLu<Array<T, N> > {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &rhs) const {
    Array<T, N> y;
    Sigmoid<T> sigmoid_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < int(rhs.size()); ++i) {
      y[i] = rhs[i] * sigmoid_op(rhs[i]);
    }

    return y;
  }
};

template <typename T>
struct HSwish {
  CUTLASS_HOST_DEVICE
  T operator()(T const &scalar) const {
    return T(scalar * HSigmoid<T>()(scalar));
  }
};

template <typename T, int N>
struct HSwish<Array<T, N> > {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &rhs) const {
    Array<T, N> y;
    HSigmoid<T> hsigmoid_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < int(rhs.size()); ++i) {
      y[i] = rhs[i] * hsigmoid_op(rhs[i]);
    }

    return y;
  }
};

// Tanh operator
template <typename T>
struct Tanh {
  CUTLASS_HOST_DEVICE
  T operator()(T const &scalar) const {
    return fast_tanh(scalar);
  }
};

template <typename T, int N>
struct Tanh<Array<T, N> > {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &value) const {
    Array<T, N> y;
    Tanh<T> tanh_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      y[i] = tanh_op(value[i]);
    }

    return y;
  }
};

template <int N>
struct Tanh<Array<half_t, N>> {
  using T = half_t;

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const& z) const {
    fast_tanh_op<Array<T, N>, N> tanh;
    return tanh(z);

  }
};

template <int N>
struct Tanh<Array<float, N>> {
  using T = float;

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const& z) const {
    fast_tanh_op<Array<T, N>, N> tanh;
    return tanh(z);

  }
};
//
// GELU function definitions implemented as described by
//   Hendrycks, D., and Gimpel, K. in
//   "Gaussian Error Linear Units (GELUs)." (2020)
//   https://arxiv.org/pdf/1606.08415.pdf
//
// Floating-point constants are Taylor coefficients described in the paper.
//

// GELU operator
template <typename T>
struct GELU {
  CUTLASS_HOST_DEVICE
  T operator()(T const &scalar) const {
    return T(cutlass::constants::half<T>() * scalar *
      (cutlass::constants::one<T>() + T(erff( float(scalar / cutlass::constants::root_two<T>()) ))));
  }
};

template <>
struct GELU<float> {
  CUTLASS_HOST_DEVICE
  float operator()(float const &scalar) const {
    return cutlass::constants::half<float>() * scalar *
      (cutlass::constants::one<float>() + erff( scalar / cutlass::constants::root_two<float>() ));
  }
};

template <typename T, int N>
struct GELU<Array<T, N> > {
  #ifdef __HGGC_ARCH__
  static const bool kIsHeavy=false;
  #else
  static const bool kIsHeavy=true;
  #endif
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &rhs) const {
    Array<T, N> y;
    GELU<T> gelu_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      y[i] = gelu_op(rhs[i]);
    }

    return y;
  }
};

// GELU operator implemented using the Taylor series approximation
template <typename T>
struct GELU_taylor {
  CUTLASS_HOST_DEVICE
  T operator()(T const &z) const {

    T k0 = T(0.7978845608028654);
    T k1 = T(0.044715);

    return T(cutlass::constants::half<T>() * z *
      (cutlass::constants::one<T>() + fast_tanh(k0 * z * (cutlass::constants::one<T>() + k1 * z * z))));
  }
};

template <typename T, int N>
struct GELU_taylor<Array<T, N> > {
  #ifdef __HGGC_ARCH__
  static const bool kIsHeavy=false;
  #else
  static const bool kIsHeavy=true;
  #endif
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &rhs) const {
    Array<T, N> y;
    GELU_taylor<T> gelu_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      y[i] = gelu_op(rhs[i]);
    }

    return y;
  }
};

template <int N>
struct GELU_taylor<Array<half_t, N> > {
  #ifdef __HGGC_ARCH__
  static const bool kIsHeavy=false;
  #else
  static const bool kIsHeavy=true;
  #endif
  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const &z) const {
    using T = half_t;
    Array<half_t, N> y;

    half_t k0 = half_t(0.7978845608028654);
    half_t k1 = half_t(0.044715);

    multiply_add<Array<half_t, N>> fma;
    multiplies<Array<half_t, N>>   mul;
    plus<Array<half_t, N>>         add;

    fast_tanh_op<Array<half_t, N>, N> tanh;

    Array<half_t, N> u = mul(mul(k0, z), fma(mul(k1, z), z, cutlass::constants::one<T>()));

    y = mul(mul(z, cutlass::constants::half<T>()), add(cutlass::constants::one<T>(), tanh(u)));

    return y;
  }
};

/// Computes backwards pass for GELU operator assuming d_t is the layer gradient and
/// z is computed from the forward pass.
template <typename T>
struct dGELU {
  CUTLASS_HOST_DEVICE
  T operator()(T const &d_t, T const &z) const {

    T k0 = T(0.7978845608028654);
    T k1 = T(0.044715);
    T k2 = T(0.1070322243);

    T tanh_out = fast_tanh(k0 * z * (T(1.0) + k1 * z * z));

    T ff = constants::half<T>() * z * ((T(1.0) - tanh_out * tanh_out) * (k0 + k2 * z * z)) +
      constants::half<T>() * (T(1.0) + tanh_out);

    return ff * d_t;
  }
};

template <typename T, int N>
struct dGELU<Array<T, N> > {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &d_t, Array<T, N> const &z) const {
    Array<T, N> y;
    dGELU<T> gelu_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      y[i] = gelu_op(d_t[i], z[i]);
    }

    return y;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

