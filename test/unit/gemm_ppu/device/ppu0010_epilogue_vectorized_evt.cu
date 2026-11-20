/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved.
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <hggc_runtime.h>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>
#include "cute/tensor.hpp"
#include "cute/atom/mma_traits_ppu0010.hpp"
#include "cute/algorithm/ppu_copy.hpp"
#include "cutlass/epilogue/collective/collective_builder_safe_store.hpp"
#ifndef ACTLIZE_EPILOGUE_STANDALONE
#include "../../common/cutlass_unit_test.h"
#endif

namespace {
using namespace cute;
using BlockTile = Shape<_64, _64, _32>;
using Warp = Shape<_32, _32, _32>;
using Schedule = cutlass::epilogue::EpilogueSimtVectorized;
using TiledMma = TiledMMA<MMA_Atom<PPU0010_16x16x16_F32BF16BF16F32_TN>,
    Layout<Shape<_2, _2, _1>>, Tile<_32, _32, _16>>;

template <class Output>
using Operation = cutlass::epilogue::fusion::PPUEVT<
    cutlass::epilogue::fusion::PPUCompute<cutlass::epilogue::thread::Identity,
        Output, float, cutlass::FloatRoundStyle::round_to_nearest>,
    cutlass::epilogue::fusion::PPUAccFetch>;

template <class Output, int Alignment>
using LegacyEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::PPU0010, cutlass::arch::OpClassTensorOp, BlockTile, Warp,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float,
    Output, cutlass::layout::RowMajor, Alignment,
    Output, cutlass::layout::RowMajor, Alignment,
    Schedule, Operation<Output>>::CollectiveOp;

template <class Output, int Alignment, class EpilogueSchedule = Schedule,
          class Arch = cutlass::arch::PPU0010>
using SafeBuilder = cutlass::epilogue::collective::CollectiveBuilderSafeStore<
    Arch, cutlass::arch::OpClassTensorOp, BlockTile, Warp,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float,
    Output, cutlass::layout::RowMajor, Alignment,
    Output, cutlass::layout::RowMajor, Alignment,
    EpilogueSchedule, Operation<Output>>;

template <class Output, int Alignment>
using SafeEpilogue = typename SafeBuilder<Output, Alignment>::CollectiveOp;

template <class Output, int Alignment>
constexpr bool check_abi() {
  using Old = LegacyEpilogue<Output, Alignment>;
  using New = SafeEpilogue<Output, Alignment>;
  using DirectOld = cutlass::epilogue::collective::EpilogueEvt<
      typename Old::StrideC, typename Old::StrideD, typename Old::FusionCallbacks,
      typename Old::SmemLayout, typename Old::CopyAtomR2S, typename Old::TiledCopyS2R,
      typename Old::CopyAtomR2G, Schedule>;
  using DirectNew = cutlass::epilogue::collective::EpilogueEvtSafeStore<
      typename Old::StrideC, typename Old::StrideD, typename Old::FusionCallbacks,
      typename Old::SmemLayout, typename Old::CopyAtomR2S, typename Old::TiledCopyS2R,
      typename Old::CopyAtomR2G, Schedule>;
  static_assert(is_same_v<Old, DirectOld>, "Legacy interface and type must remain unchanged.");
  static_assert(is_same_v<New, DirectNew> && !is_same_v<New, Old>, "Safe epilogue must be a separate type.");
  using OP = typename Old::Params; using NP = typename New::Params;
  using OA = typename Old::Arguments; using NA = typename New::Arguments;
  using OS = typename Old::SharedStorage; using NS = typename New::SharedStorage;
  static_assert(sizeof(OP) == sizeof(NP) && alignof(OP) == alignof(NP), "Params ABI must match.");
  static_assert(sizeof(OA) == sizeof(NA) && alignof(OA) == alignof(NA), "Arguments ABI must match.");
  static_assert(sizeof(OS) == sizeof(NS) && alignof(OS) == alignof(NS), "Shared storage ABI must match.");
  static_assert(offsetof(OP, thread) == offsetof(NP, thread) &&
                offsetof(OP, ptr_C) == offsetof(NP, ptr_C) && offsetof(OP, dC) == offsetof(NP, dC) &&
                offsetof(OP, ptr_D) == offsetof(NP, ptr_D) && offsetof(OP, dD) == offsetof(NP, dD),
                "Params field offsets must match.");
  static_assert(offsetof(OS, smem_epilogue) == offsetof(NS, smem_epilogue) &&
                offsetof(OS, thread) == offsetof(NS, thread), "Shared storage offsets must match.");
  return true;
}
static_assert(check_abi<float, 4>() && check_abi<cutlass::bfloat16_t, 8>());
static_assert(check_abi<float, 1>() && check_abi<float, 2>());


template <class Output, int BM, int BN, int WM, int WN, int Alignment, bool Exhaustive = true>
bool check_partition() {
  using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::PPU0010, cutlass::arch::OpClassTensorOp,
      Shape<Int<BM>, Int<BN>, _32>, Shape<Int<WM>, Int<WN>, _32>,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float,
      Output, cutlass::layout::RowMajor, Alignment,
      Output, cutlass::layout::RowMajor, Alignment, Schedule, Operation<Output>>::CollectiveOp;
  static_assert(Epi::is_row_vector_partition(), "The partition proof must scale to production tiles.");
  static_assert(Epi::is_store_shape_safe(BM - 1, BN), "Aligned tails must be constexpr-safe.");
  if (Epi::kAlignmentD != Alignment || !Epi::is_row_vector_partition()) return false;
  using Copy = typename Epi::TiledCopyS2R;
  constexpr int SM = size<0>(typename Epi::SmemLayout{});
  constexpr int SN = size<1>(typename Epi::SmemLayout{});
  auto identity = make_identity_tensor(make_shape(Int<SM>{}, Int<SN>{}));
  auto divided = flat_divide(identity, make_shape(Int<SM>{}, Int<SN>{}));
  auto scalar = coalesce(Epi::store_partition_layout());
  std::vector<std::vector<std::pair<int, int>>> fragments;
  for (int tid = 0; tid < Copy::TiledNumThr::value; ++tid) {
    auto coords = Copy{}.get_thread_slice(tid).partition_D(divided);
    auto tile = coords(_, _, _, 0, 0);
    for (int pn = 0; pn < size<2>(tile); ++pn)
      for (int pm = 0; pm < size<1>(tile); ++pm) {
        std::vector<std::pair<int, int>> fragment;
        for (int v = 0; v < size<0>(tile); ++v) {
          auto coordinate = tile(v, pm, pn);
          int row = get<0>(coordinate), col = get<1>(coordinate);
          int index = tid + Copy::TiledNumThr::value *
              (v + size<0>(tile) * (pm + size<1>(tile) * pn));
          if (scalar(index) != row * SN + col) return false;
          fragment.emplace_back(row, col);
        }
        fragments.push_back(fragment);
      }
  }
  std::vector<int> tails;
  if constexpr (Exhaustive) {
    for (int n = 1; n <= BN + 2 * Alignment; ++n) tails.push_back(n);
  } else {
    tails = {1, Alignment - 1, Alignment, Alignment + 1, BN - Alignment,
             BN - 1, BN, BN + 1, BN + Alignment};
  }
  for (int m : {1, BM - 1, BM, BM + 1})
    for (int n : tails) {
      int rm = m % SM == 0 ? SM : m % SM;
      int rn = n % SN == 0 ? SN : n % SN;
      bool raw_safe = true;
      for (auto const& fragment : fragments)
        if (fragment[0].first < rm && fragment[0].second < rn)
          for (auto const& value : fragment)
            if (value.first >= rm || value.second >= rn) raw_safe = false;
      if (raw_safe != (n % Alignment == 0) || Epi::is_store_shape_safe(m, n) != raw_safe)
        return false;
    }
  return true;
}

template <class Output>
bool partition_contracts() {
  constexpr int A = 16 / sizeof(Output);
  return check_partition<Output, 8, 16, 8, 16, A>() &&
      check_partition<Output, 64, 64, 32, 32, A>() &&
      check_partition<Output, 128, 128, 32, 64, A>() &&
      check_partition<Output, 64, 128, 64, 32, A>() &&
      check_partition<Output, 64, 64, 32, 32, 1>() &&
      check_partition<Output, 64, 64, 32, 32, 2>() &&
      check_partition<Output, 128, 256, 32, 64, A, false>() &&
      check_partition<Output, 256, 256, 64, 64, A, false>() &&
      check_partition<Output, 128, 512, 32, 64, A, false>() &&
      check_partition<Output, 256, 512, 64, 64, A, false>();
}

void check(hggcError_t result) {
  if (result != hggcSuccess) throw std::runtime_error(hggcGetErrorString(result));
}

template <class T>
struct DeviceBuffer {
  T* ptr = nullptr;
  explicit DeviceBuffer(size_t count) { check(hggcMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T))); }
  ~DeviceBuffer() { if (ptr) hggcFree(ptr); }
  DeviceBuffer(DeviceBuffer const&) = delete;
  DeviceBuffer& operator=(DeviceBuffer const&) = delete;
};

__host__ __device__ float fingerprint(int row, int col, int batch) {
  return float((row % 4 + 1) * (col % 4 + 1) + batch * 16) / 8;
}

template <class Epi, int StaticN = 0>
__global__ void store_kernel(typename Epi::Params params, int rows, int cols, int batches) {
  extern __shared__ char smem[];
  TiledMma mma;
  auto coords = make_identity_tensor(take<0,2>(BlockTile{}));
  auto thread_coords = mma.get_slice(threadIdx.x).partition_C(coords);
  auto accum = partition_fragment_C(mma, take<0,2>(BlockTile{}));
  for (int i = 0; i < size(accum); ++i) {
    int row = int(get<0>(thread_coords(i))) + int(blockIdx.x) * 64;
    int col = int(get<1>(thread_coords(i))) + int(blockIdx.y) * 64;
    accum(i) = fingerprint(row, col, blockIdx.z);
  }
  typename Epi::SharedStorage& storage = *reinterpret_cast<typename Epi::SharedStorage*>(smem);
  Epi epi(params, storage);
  auto n = [&]() { if constexpr (StaticN > 0) return Int<StaticN>{}; else return cols; }();
  epi(make_shape(rows, n, 32, batches), BlockTile{},
      make_coord(int(blockIdx.x), int(blockIdx.y), 0, int(blockIdx.z)),
      accum, mma, make_tuple(rows - int(blockIdx.x) * 64, cols - int(blockIdx.y) * 64, 0),
      threadIdx.x, smem);
}

template <class Epi, int StaticN = 0>
bool run_case(int rows, int cols, int offset, int padding, int batches) {
  using Output = typename Epi::ElementOutput;
  int pitch = cols + padding;
  int batch_stride = rows * pitch + 16;
  int begin = 32 + offset;
  size_t count = begin + batches * batch_stride + 32;
  Output sentinel(-123);
  std::vector<Output> initial(count, sentinel), expected = initial, actual(count);
  for (int b = 0; b < batches; ++b)
    for (int m = 0; m < rows; ++m)
      for (int n = 0; n < cols; ++n)
        expected[begin + b * batch_stride + m * pitch + n] = Output(fingerprint(m, n, b));
  DeviceBuffer<Output> device(count);
  typename Epi::Params params{};
  params.ptr_D = device.ptr + begin;
  params.dD = typename Epi::StrideD{pitch, _1{}, batch_stride};
  dim3 grid((rows + 63) / 64, (cols + 63) / 64, batches);
  for (int repeat = 0; repeat < 3; ++repeat) {
    check(hggcMemcpy(device.ptr, initial.data(), count * sizeof(Output), hggcMemcpyHostToDevice));
    store_kernel<Epi, StaticN><<<grid, 128, sizeof(typename Epi::SharedStorage)>>>(params, rows, cols, batches);
    check(hggcGetLastError());
    check(hggcDeviceSynchronize());
    check(hggcMemcpy(actual.data(), device.ptr, count * sizeof(Output), hggcMemcpyDeviceToHost));
    for (size_t i = 0; i < count; ++i) {
      if (float(actual[i]) != float(expected[i])) {
        std::cerr << "Mismatch: M=" << rows << " N=" << cols << " offset=" << offset
                  << " padding=" << padding << " batches=" << batches << " index=" << i
                  << " expected=" << float(expected[i]) << " actual=" << float(actual[i]) << '\n';
        return false;
      }
    }
  }
  return true;
}

template <class Output>
bool safe_boundaries() {
  using Epi = SafeEpilogue<Output, 16 / sizeof(Output)>;
  for (int m : {1, 3, 63, 64, 65})
    for (int n : {1, 2, 3, 6, 7, 8, 14, 16, 63, 64, 65, 66})
      for (int offset = 0; offset < int(16 / sizeof(Output)); ++offset)
        for (int padding : {0, 1})
          if (!run_case<Epi>(m, n, offset, padding, 3)) return false;
  return true;
}

template <class Output, int Alignment>
bool unchecked_boundaries() {
  using Epi = LegacyEpilogue<Output, Alignment>;
  for (int m : {1, 3, 63, 64, 65})
    for (int n : {1, 2, 3, 6, 7, 8, 14, 16, 63, 64, 65, 66}) {
      if (n % Alignment) continue;
      for (int offset : {0, Alignment})
        if (!run_case<Epi>(m, n, offset, Alignment, 3)) return false;
    }
  return true;
}

bool unchecked_all() {
  return unchecked_boundaries<float, 1>() && unchecked_boundaries<float, 2>() &&
      unchecked_boundaries<float, 4>() && unchecked_boundaries<cutlass::bfloat16_t, 1>() &&
      unchecked_boundaries<cutlass::bfloat16_t, 2>() && unchecked_boundaries<cutlass::bfloat16_t, 4>() &&
      unchecked_boundaries<cutlass::bfloat16_t, 8>() &&
      run_case<LegacyEpilogue<float, 1>, 14>(3, 14, 0, 0, 3) &&
      run_case<LegacyEpilogue<float, 2>, 14>(3, 14, 0, 0, 3) &&
      run_case<LegacyEpilogue<cutlass::bfloat16_t, 2>, 14>(3, 14, 0, 0, 3);
}

bool safe_narrow_and_static_tails() {
  for (int offset : {0, 1, 2, 3}) {
    if (!run_case<SafeEpilogue<float, 1>, 14>(3, 14, offset, 1, 3) ||
        !run_case<SafeEpilogue<float, 2>, 14>(3, 14, offset, 1, 3) ||
        !run_case<SafeEpilogue<float, 4>, 14>(3, 14, offset, 1, 3) ||
        !run_case<SafeEpilogue<cutlass::bfloat16_t, 1>, 14>(3, 14, offset, 1, 3) ||
        !run_case<SafeEpilogue<cutlass::bfloat16_t, 2>, 14>(3, 14, offset, 1, 3) ||
        !run_case<SafeEpilogue<cutlass::bfloat16_t, 8>, 14>(3, 14, offset, 1, 3)) return false;
  }
  return true;
}
} // namespace

#if defined(ACTLIZE_EPILOGUE_STANDALONE)
int main() {
#if defined(ACTLIZE_HOST_PARTITION)
  bool partition = partition_contracts<float>() && partition_contracts<cutlass::bfloat16_t>();
  std::cout << "host_cute_partition=" << partition << '\n';
  return partition ? 0 : 1;
#elif defined(ACTLIZE_TEST_NO_EVT)
  using Bad = SafeBuilder<float, 4, cutlass::epilogue::EpilogueSimtVectorizedWithoutEvt>;
  std::cout << sizeof(Bad);
#elif defined(ACTLIZE_TEST_WRONG_ARCH)
  using Bad = SafeBuilder<float, 4, Schedule, cutlass::arch::PPU0015>;
  std::cout << sizeof(Bad);
#elif defined(ACTLIZE_TEST_ZERO_ALIGNMENT)
  std::cout << sizeof(SafeBuilder<float, 0>);
#elif defined(ACTLIZE_TEST_LEGACY_FP32_TAIL)
  return run_case<LegacyEpilogue<float, 4>, 14>(3, 14, 0, 0, 1) ? 0 : 1;
#elif defined(ACTLIZE_TEST_LEGACY_BF16_TAIL)
  return run_case<LegacyEpilogue<cutlass::bfloat16_t, 8>, 14>(3, 14, 0, 0, 1) ? 0 : 1;
#elif defined(ACTLIZE_TEST_LEGACY_FRAGMENT_ALIGNMENT)
  using Old = LegacyEpilogue<float, 4>;
  using Bad = cutlass::epilogue::collective::EpilogueEvt<
      typename Old::StrideC, typename Old::StrideD, typename Old::FusionCallbacks,
      typename Old::SmemLayout, typename Old::CopyAtomR2S, typename Old::TiledCopyS2R,
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<256>, float>, Schedule>;
  return run_case<Bad>(3, 16, 0, 0, 1) ? 0 : 1;
#else
  try {
    bool fp32 = safe_boundaries<float>();
    bool bf16 = safe_boundaries<cutlass::bfloat16_t>();
    bool unchecked = unchecked_all();
    bool narrow = safe_narrow_and_static_tails();
    bool partition = partition_contracts<float>() && partition_contracts<cutlass::bfloat16_t>();
    std::cout << "safe_fp32=" << fp32 << " safe_bf16=" << bf16 << " legacy=" << unchecked
              << " safe_narrow_static=" << narrow << " host_cute_partition=" << partition << '\n';
    return fp32 && bf16 && unchecked && narrow && partition ? 0 : 1;
  } catch (std::exception const& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
#endif
}
#else
TEST(PPU0010_Epilogue_SafeStore, SafeFP32TailsOffsetsStridesBatches) {
  EXPECT_TRUE(safe_boundaries<float>());
}
TEST(PPU0010_Epilogue_SafeStore, SafeBF16TailsOffsetsStridesBatches) {
  EXPECT_TRUE(safe_boundaries<cutlass::bfloat16_t>());
}
TEST(PPU0010_Epilogue_SafeStore, LegacyScalarNarrowAndFullVectors) {
  EXPECT_TRUE(unchecked_all());
}
TEST(PPU0010_Epilogue_SafeStore, SafeNarrowAndStaticTails) {
  EXPECT_TRUE(safe_narrow_and_static_tails());
}
#endif
