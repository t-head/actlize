/*
 * Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <hggc_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "cute/tensor.hpp"
#include "cute/atom/mma_traits_ppu0010.hpp"
#include "cute/algorithm/ppu_copy.hpp"
#include "cutlass/epilogue/collective/builders/ppu_builder.inl"
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

template <class Output, int Alignment, bool Safe,
          class EpilogueSchedule = Schedule>
using Builder = cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::PPU0010, cutlass::arch::OpClassTensorOp, BlockTile, Warp,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float,
    Output, cutlass::layout::RowMajor, Alignment,
    Output, cutlass::layout::RowMajor, Alignment,
    EpilogueSchedule, Operation<Output>, void, Safe>;

template <class Output, int Alignment, bool Safe>
using Epilogue = typename Builder<Output, Alignment, Safe>::CollectiveOp;

using DefaultEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::PPU0010, cutlass::arch::OpClassTensorOp, BlockTile, Warp,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float,
    float, cutlass::layout::RowMajor, 4,
    float, cutlass::layout::RowMajor, 4, Schedule, Operation<float>>::CollectiveOp;
static_assert(!DefaultEpilogue::EnableSafeStore, "Default must preserve unchecked stores.");
static_assert(is_same_v<DefaultEpilogue, Epilogue<float, 4, false>>,
              "Omitted and explicitly disabled safe stores must have the same type.");
static_assert(Epilogue<float, 4, true>::EnableSafeStore, "Builder must forward safe stores.");
static_assert(Epilogue<float, 4, true>::kAlignmentD == 4, "Builder must forward alignment.");
static_assert(sizeof(DefaultEpilogue::Params) == sizeof(Epilogue<float, 4, true>::Params),
              "Store policy must not change launch parameter ABI.");
static_assert(sizeof(DefaultEpilogue::SharedStorage) == sizeof(Epilogue<float, 4, true>::SharedStorage),
              "Store policy must not change shared storage.");

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
  using Epi = Epilogue<Output, 16 / sizeof(Output), true>;
  for (int m : {1, 3, 63, 64, 65})
    for (int n : {1, 2, 3, 6, 7, 8, 14, 16, 63, 64, 65, 66})
      for (int offset = 0; offset < int(16 / sizeof(Output)); ++offset)
        for (int padding : {0, 1})
          if (!run_case<Epi>(m, n, offset, padding, 3)) return false;
  return true;
}

template <class Output, int Alignment>
bool unchecked_boundaries() {
  using Epi = Epilogue<Output, Alignment, false>;
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
      run_case<Epilogue<float, 1, false>, 14>(3, 14, 0, 0, 3) &&
      run_case<Epilogue<float, 2, false>, 14>(3, 14, 0, 0, 3) &&
      run_case<Epilogue<cutlass::bfloat16_t, 2, false>, 14>(3, 14, 0, 0, 3) &&
      run_case<Epilogue<float, 4, true>, 14>(3, 14, 1, 1, 3);
}

#if defined(ACTLIZE_TEST_FRAGMENT_MISMATCH)
using Wide = cutlass::gemm::config::DefaultGemm_Epilogue_Configuration<
    AutoVectorizingCopyWithAssumedAlignment<128>, float, 8, _64, _64, _2, 128>;
using Bad = cutlass::epilogue::collective::EpilogueEvt<
    DefaultEpilogue::StrideC, DefaultEpilogue::StrideD,
    DefaultEpilogue::FusionCallbacks, Wide::SmemLayoutO,
    DefaultEpilogue::CopyAtomR2S, Wide::GmemTiledCopyO,
    Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, float>, Schedule>;
#endif
} // namespace

#if defined(ACTLIZE_EPILOGUE_STANDALONE)
int main() {
#if defined(ACTLIZE_TEST_STATIC_TAIL)
  store_kernel<DefaultEpilogue, 14><<<1, 128>>>(DefaultEpilogue::Params{}, 1, 14, 1);
#elif defined(ACTLIZE_TEST_FRAGMENT_MISMATCH)
  store_kernel<Bad><<<1, 128>>>(Bad::Params{}, 1, 16, 1);
#elif defined(ACTLIZE_TEST_NO_EVT)
  using Bad = typename Builder<float, 4, true, cutlass::epilogue::EpilogueSimtVectorizedWithoutEvt>::CollectiveOp;
  std::cout << sizeof(Bad);
#elif defined(ACTLIZE_TEST_ZERO_ALIGNMENT)
  std::cout << sizeof(Epilogue<float, 0, false>);
#else
  try {
    bool fp32 = safe_boundaries<float>();
    bool bf16 = safe_boundaries<cutlass::bfloat16_t>();
    bool unchecked = unchecked_all();
    std::cout << "safe_fp32=" << fp32 << " safe_bf16=" << bf16 << " unchecked=" << unchecked << '\n';
    return fp32 && bf16 && unchecked ? 0 : 1;
  } catch (std::exception const& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
#endif
}
#else
TEST(PPU0010_Epilogue_VectorizedEvt, SafeFP32TailsOffsetsStridesBatches) {
  EXPECT_TRUE(safe_boundaries<float>());
}
TEST(PPU0010_Epilogue_VectorizedEvt, SafeBF16TailsOffsetsStridesBatches) {
  EXPECT_TRUE(safe_boundaries<cutlass::bfloat16_t>());
}
TEST(PPU0010_Epilogue_VectorizedEvt, UncheckedScalarNarrowAndFullVectors) {
  EXPECT_TRUE(unchecked_all());
}
#endif
