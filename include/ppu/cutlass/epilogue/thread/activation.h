/*
 * Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
 * All rights reserved.
 *
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
 */

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/constants.h"
#include "cutlass/complex.h"
#include "cutlass/array.h"
#include "cutlass/half.h"
#include "cutlass/functional.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

// additional threshold param than original LeakyRelu
template <typename T>
struct AcLeakyReLU {

  static const bool kIsHeavy = false;

  struct Arguments {
    T threshold = T(0);
    T leaky_alpha = T(0);
  };

  CUTLASS_HOST_DEVICE
  T operator()(T const& value, T const& threshold, T const& leaky_alpha) const {
    T res = value > threshold ? value : value * leaky_alpha;
    return res;
  }

  CUTLASS_HOST_DEVICE
  T operator()(T const& value, Arguments const& args = Arguments()) const {
    this->operator()(value, args.threshold, args.leaky_alpha);
  }
};

template <typename T, int N>
struct AcLeakyReLU<Array<T, N> > {

  static const bool kIsHeavy = false;

  using Arguments = typename AcLeakyReLU<T>::Arguments;

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const& values, T const& threshold, T const& leaky_alpha) const {
    Array<T, N> y;
    AcLeakyReLU<T> leaky_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < int(values.size()); ++i) {
      y[i] = leaky_op(values[i], threshold, leaky_alpha);
    }

    return y;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const& values, Arguments const& args = Arguments()) const {
    return this->operator()(values, args.threshold, args.leaky_alpha);
  }
};

// HSigmoid operator
template <typename T>
struct AcHSigmoid {
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
struct AcHSigmoid<Array<T, N> > {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &rhs) const {
    Array<T, N> y;
    AcHSigmoid<T> hsigmoid_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < int(rhs.size()); ++i) {
      y[i] = hsigmoid_op(rhs[i]);
    }

    return y;
  }
};


template <typename T>
struct AcHSwish {
  CUTLASS_HOST_DEVICE
  T operator()(T const &scalar) const {
    return T(scalar * AcHSigmoid<T>()(scalar));
  }
};

template <typename T, int N>
struct AcHSwish<Array<T, N> > {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &rhs) const {
    Array<T, N> y;
    AcHSigmoid<T> hsigmoid_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < int(rhs.size()); ++i) {
      y[i] = rhs[i] * hsigmoid_op(rhs[i]);
    }

    return y;
  }
};


/// ClippedReLu operator - propagates NaNs
template <typename T>
struct AcClipReLu {

  struct Arguments {
    T threshold0 = T(0);
    T threshold1 = T(0);
  };

  CUTLASS_HOST_DEVICE
  T operator()(T value, T const & threshold0, T const & threshold1) const {
    if (value < threshold0) {
      value = threshold0;
    } else if(value > threshold1) {
      value = threshold1;
    }
    return value;
  }

  CUTLASS_HOST_DEVICE
  T operator()(T const& value, Arguments const& args = Arguments()) const {
    this->operator()(value, args.threshold0, args.threshold1);
  }
};

template <typename T, int N>
struct AcClipReLu<Array<T, N>> {

  using Arguments = typename AcClipReLu<T>::Arguments;

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &frag, T const & threshold0, T const & threshold1) const {
    Array<T, N> result;
    AcClipReLu<T> clip_relu_op;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = clip_relu_op(frag[i], threshold0, threshold1);
    }
    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const& values, Arguments const& args = Arguments()) const {
    return this->operator()(values, args.threshold0, args.threshold1);
  }
};


/// ELu operator - propagates NaNs
template <typename T>
struct AcPReLu {

  CUTLASS_HOST_DEVICE
  T operator()(T value, T const & coeff) const {
    if (value < T(.0f)) {
      value = T(float(float(value) * float(coeff)));
    }
    return value;
  }
};

template <typename T, int N>
struct AcPReLu<Array<T, N>> {

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &frag, Array<T, N> const & coeff) const {
    Array<T, N> result;
    AcPReLu<T> PReLu_op;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = PReLu_op(frag[i], coeff[i]);
    }
    return result;
  }
};






} // namespace thread
} // namespace epilogue
} // namespace cutlass
