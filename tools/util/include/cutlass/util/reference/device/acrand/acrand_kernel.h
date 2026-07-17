// Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved.
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ACRAND_KERNEL_H_
#define ACRAND_KERNEL_H_

#include <hggc_runtime.h>
#include <math.h>

#define ACRAND_2POW32_INV (2.3283064e-10f)
#define ACRAND_2POW32_INV_DOUBLE (2.3283064365386963e-10)
#define ACRAND_2POW53_INV_DOUBLE (1.1102230246251565e-16)
#define ACRAND_2POW32_INV_2PI (2.3283064e-10f * 6.2831855f)
#define ACRAND_PI_DOUBLE  (3.1415926535897932)
#define ACRAND_SQRT2 (1.4142135f)
#define ACRAND_SQRT2_DOUBLE (1.4142135623730951)

#include "cutlass/util/reference/device/acrand/acrand_xorwow_precomputed.h"


namespace acrand_device {
namespace detail {

template<typename Engine>
struct engine_boxmuller_helper
{
    static __forceinline__ __device__ __host__ bool has_float(const Engine* engine)
    {
        return engine->m_state.boxmuller_float_state != 0;
    }

    static __forceinline__ __device__ __host__ float get_float(Engine* engine)
    {
        engine->m_state.boxmuller_float_state = 0;
        return engine->m_state.boxmuller_float;
    }

    static __forceinline__ __device__ __host__ void save_float(Engine* engine, float f)
    {
        engine->m_state.boxmuller_float_state = 1;
        engine->m_state.boxmuller_float = f;
    }

    static __forceinline__ __device__ __host__ bool has_double(const Engine* engine)
    {
        return engine->m_state.boxmuller_double_state != 0;
    }

    static __forceinline__ __device__ __host__ double get_double(Engine* engine)
    {
        engine->m_state.boxmuller_double_state = 0;
        return engine->m_state.boxmuller_double;
    }

    static __forceinline__ __device__ __host__ void save_double(Engine* engine, double d)
    {
        engine->m_state.boxmuller_double_state = 1;
        engine->m_state.boxmuller_double = d;
    }
};

__forceinline__ __device__ __host__ void copy_vec(unsigned int* dst, const unsigned int* src)
{
    for (int i = 0; i < XORWOW_N; i++)
    {
        dst[i] = src[i];
    }
}

__forceinline__ __device__ __host__ void mul_mat_vec_inplace(const unsigned int* m, unsigned int* v)
{
    unsigned int r[XORWOW_N] = { 0 };
    for (int ij = 0; ij < XORWOW_N * XORWOW_M; ij++)
    {
        const int i = ij / XORWOW_M;
        const int j = ij % XORWOW_M;
        const unsigned int b = (v[i] & (1U << j)) ? 0xffffffff : 0x0;
        for (int k = 0; k < XORWOW_N; k++)
        {
            r[k] ^= b & m[i * XORWOW_M * XORWOW_N + j * XORWOW_N + k];
        }
    }
    copy_vec(v, r);
}

struct two_uints
{
    unsigned int x;
    unsigned int y;
};

union two_uints_to_ulong
{
    two_uints uint2_value;
    unsigned long long int ulong_value;
};

__forceinline__ __device__ __host__ float uniform_distribution(unsigned int v)
{
    return ACRAND_2POW32_INV + (v * ACRAND_2POW32_INV);
}

__forceinline__ __device__ __host__ double uniform_distribution_double(unsigned int v1,
                                                                       unsigned int v2)
{
    two_uints_to_ulong v;
    v.uint2_value.x = v1;
    v.uint2_value.y = (v2 >> 11);
    return ACRAND_2POW53_INV_DOUBLE + (v.ulong_value * ACRAND_2POW53_INV_DOUBLE);
}

__forceinline__ __device__ __host__ float2 box_muller(unsigned int x, unsigned int y)
{
    float2 result;
    float u = ACRAND_2POW32_INV + (x * ACRAND_2POW32_INV);
    float v = ACRAND_2POW32_INV_2PI + (y * ACRAND_2POW32_INV_2PI);
    float s = sqrtf(-2.0f * logf(u));
    #ifdef __HGGC_ARCH__
        __sincosf(v, &result.x, &result.y);
        result.x *= s;
        result.y *= s;
    #else
        result.x = sinf(v) * s;
        result.y = cosf(v) * s;
    #endif
    return result;
}

__forceinline__ __device__ __host__ double2 box_muller_double(uint4 v)
{
    double2 result;
    unsigned long long int v1 = (unsigned long long int)v.x ^
        ((unsigned long long int)v.y << (53 - 32));
    double u = ACRAND_2POW53_INV_DOUBLE + (v1 * ACRAND_2POW53_INV_DOUBLE);
    unsigned long long int v2 = (unsigned long long int)v.z ^
        ((unsigned long long int)v.w << (53 - 32));
    double w = (ACRAND_2POW53_INV_DOUBLE * 2.0) +
        (v2 * (ACRAND_2POW53_INV_DOUBLE * 2.0));
    double s = sqrt(-2.0 * log(u));
    #ifdef __HGGC_ARCH__
        sincospi(w, &result.x, &result.y);
        result.x *= s;
        result.y *= s;
    #else
        result.x = sin(w * ACRAND_PI_DOUBLE) * s;
        result.y = cos(w * ACRAND_PI_DOUBLE) * s;
    #endif
    return result;
}

__forceinline__ __device__ __host__ float2 normal_distribution2(unsigned int v1, unsigned int v2)
{
    return box_muller(v1, v2);
}

__forceinline__ __device__ __host__ double2 normal_distribution_double2(uint4 v)
{
    return box_muller_double(v);
}

} // end namespace detail

#define ACRAND_XORWOW_DEFAULT_SEED 0ULL

class xorwow_engine
{
public:
    struct xorwow_state
    {
        unsigned int d;

        unsigned int boxmuller_float_state;
        unsigned int boxmuller_double_state;
        float boxmuller_float;
        double boxmuller_double;

        unsigned int x[5];
    };

    __forceinline__ __device__ __host__ xorwow_engine()
        : xorwow_engine(ACRAND_XORWOW_DEFAULT_SEED, 0, 0)
    {}

    __forceinline__ __device__ __host__ xorwow_engine(const unsigned long long seed,
                                                      const unsigned long long subsequence,
                                                      const unsigned long long offset)
    {
        m_state.x[0] = 123456789U;
        m_state.x[1] = 362436069U;
        m_state.x[2] = 521288629U;
        m_state.x[3] = 88675123U;
        m_state.x[4] = 5783321U;

        m_state.d = 6615241U;

        const unsigned int s0 = static_cast<unsigned int>(seed) ^ 0x2c7f967fU;
        const unsigned int s1 = static_cast<unsigned int>(seed >> 32) ^ 0xa03697cbU;
        const unsigned int t0 = 1228688033U * s0;
        const unsigned int t1 = 2073658381U * s1;
        m_state.x[0] += t0;
        m_state.x[1] ^= t0;
        m_state.x[2] += t1;
        m_state.x[3] ^= t1;
        m_state.x[4] += t0;
        m_state.d += t1 + t0;

        discard_subsequence(subsequence);
        discard(offset);

        m_state.boxmuller_float_state = 0;
        m_state.boxmuller_double_state = 0;
    }

    __forceinline__ __device__ __host__ void discard(unsigned long long offset)
    {
        #ifdef __HGGC_ARCH__
        jump(offset, d_xorwow_jump_matrices);
        #else
        jump(offset, h_xorwow_jump_matrices);
        #endif

        m_state.d += static_cast<unsigned int>(offset) * 362437;
    }

    __forceinline__ __device__ __host__ void discard_subsequence(unsigned long long subsequence)
    {
        #ifdef __HGGC_ARCH__
        jump(subsequence, d_xorwow_sequence_jump_matrices);
        #else
        jump(subsequence, h_xorwow_sequence_jump_matrices);
        #endif
    }

    __forceinline__ __device__ __host__ unsigned int operator()()
    {
        return next();
    }

    __forceinline__ __device__ __host__ unsigned int next()
    {
        const unsigned int t = m_state.x[0] ^ (m_state.x[0] >> 2);
        m_state.x[0] = m_state.x[1];
        m_state.x[1] = m_state.x[2];
        m_state.x[2] = m_state.x[3];
        m_state.x[3] = m_state.x[4];
        m_state.x[4] = (m_state.x[4] ^ (m_state.x[4] << 4)) ^ (t ^ (t << 1));

        m_state.d += 362437;

        return m_state.d + m_state.x[4];
    }

protected:
    __forceinline__ __device__ __host__ void
        jump(unsigned long long v,
             const unsigned int jump_matrices[XORWOW_JUMP_MATRICES][XORWOW_SIZE])
    {
        unsigned int mi = 0;
        while (v > 0)
        {
            const unsigned int is = static_cast<unsigned int>(v) & ((1 << XORWOW_JUMP_LOG2) - 1);
            for (unsigned int i = 0; i < is; i++)
            {
                detail::mul_mat_vec_inplace(jump_matrices[mi], m_state.x);
            }
            mi++;
            v >>= XORWOW_JUMP_LOG2;
        }
    }

protected:
    xorwow_state m_state;

    friend struct detail::engine_boxmuller_helper<xorwow_engine>;

}; // xorwow_engine class

} // end namespace acrand_device

typedef acrand_device::xorwow_engine acrand_state_xorwow;
typedef acrand_state_xorwow acrandState_t;
typedef acrand_state_xorwow acrandState;
typedef acrand_state_xorwow acrandStateXORWOW_t;

__forceinline__ __device__ __host__
void acrand_init(const unsigned long long seed,
                  const unsigned long long subsequence,
                  const unsigned long long offset,
                  acrand_state_xorwow*    state)
{
    *state = acrand_state_xorwow(seed, subsequence, offset);
}

__forceinline__ __device__ __host__
unsigned int acrand(acrand_state_xorwow* state)
{
    return state->next();
}

__forceinline__ __device__ __host__
float acrand_uniform(acrand_state_xorwow* state)
{
    return acrand_device::detail::uniform_distribution(acrand(state));
}

__forceinline__ __device__ __host__
double acrand_uniform_double(acrand_state_xorwow* state)
{
    auto state1 = acrand(state);
    auto state2 = acrand(state);
    return acrand_device::detail::uniform_distribution_double(state1, state2);
}

__forceinline__ __device__ __host__
float acrand_normal(acrand_state_xorwow* state)
{
    typedef acrand_device::detail::engine_boxmuller_helper<acrand_state_xorwow> bm_helper;

    if(bm_helper::has_float(state))
    {
        return bm_helper::get_float(state);
    }
    auto state1 = acrand(state);
    auto state2 = acrand(state);
    float2 r = acrand_device::detail::normal_distribution2(state1, state2);
    bm_helper::save_float(state, r.y);
    return r.x;
}

__forceinline__ __device__ __host__
double acrand_normal_double(acrand_state_xorwow* state)
{
    typedef acrand_device::detail::engine_boxmuller_helper<acrand_state_xorwow> bm_helper;

    if(bm_helper::has_double(state))
    {
        return bm_helper::get_double(state);
    }

    auto state1 = acrand(state);
    auto state2 = acrand(state);
    auto state3 = acrand(state);
    auto state4 = acrand(state);

    double2 r = acrand_device::detail::normal_distribution_double2(
        uint4 { state1, state2, state3, state4 }
    );
    bm_helper::save_double(state, r.y);
    return r.x;
}

#endif // ACRAND_KERNEL_H_
