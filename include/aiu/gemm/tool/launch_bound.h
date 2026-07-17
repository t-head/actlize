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

namespace aiu {

#define CMAX(a,b) ((a > b) ? a : b)
#define CMIN(a,b) ((a < b) ? a : b)
#define SIZE_OF_VREG 4
#define DOUBLE_VREG 2

//constexpr int WARP_SIZE = 32;

constexpr int inline GetTileThreads(int TB_M, int TB_N, int WARP_M, int WARP_N) {
  return (WARP_SIZE * TB_M * TB_N) / (WARP_M * WARP_N);
}

constexpr int inline GetTileVregPerThread(int WARP_M, int WARP_N, int MMA_K, int SizeOfAB,
                                          int SizeOfACC) {
  return (DOUBLE_VREG * (WARP_M * MMA_K + WARP_N * MMA_K) * SizeOfAB +
          WARP_M * WARP_N * SizeOfACC) /
         (WARP_SIZE * SIZE_OF_VREG);
}

constexpr int inline GetWarpCount(int TB_M, int TB_N, int WARP_M, int WARP_N) {
  return (TB_M * TB_N) / (WARP_M * WARP_N);
}

constexpr int inline GetTileTsmInKB(int TB_M, int TB_N, int TB_K, int SizeOfAB, int Stages) {
  return (TB_M * TB_K + TB_N * TB_K) * SizeOfAB * Stages / 1024;
}

template <typename ElementAB, typename ElementAccumulator>
constexpr int inline GetVregOccupancy(int WARP_M, int WARP_N, int MMA_K, int WARP_COUNT) {
  return (512 * 8) /
         ( GetTileVregPerThread(WARP_M, WARP_N, MMA_K, sizeof(ElementAB), sizeof(ElementAccumulator)) * WARP_COUNT);
}

template <typename ElementAB>
constexpr int inline GetTsmOccupancy(int TB_M, int TB_N, int TB_K, int Stages) {
  return 256 / GetTileTsmInKB(TB_M, TB_N, TB_K, sizeof(ElementAB), Stages);
}

template <typename ElementAB, typename ElementAccumulator>
constexpr int inline GetTileThreadsWithOccupancy(int TB_M, int TB_N, int TB_K, int WARP_M,
                                                 int WARP_N, int MMA_K, int Stages) {
  return GetTileThreads(TB_M, TB_N, WARP_M, WARP_N) *
         CMIN((GetVregOccupancy<ElementAB, ElementAccumulator>(WARP_M, WARP_N, MMA_K, GetWarpCount(TB_M, TB_N, WARP_M, WARP_N))),
              (GetTsmOccupancy<ElementAB>(TB_M, TB_N, TB_K, Stages)));
}

template <typename ElementAB,
          typename ElementAccumulator,
          typename ThreadblockShape,
          typename WarpShape,
          typename MmaShape,
          int Stages>
class LaunchBound {
public:
  static constexpr int value = GetTileThreadsWithOccupancy<ElementAB, ElementAccumulator>(
      ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK, WarpShape::kM,
      WarpShape::kN, MmaShape::kK, Stages);
};

} /// namespace aiu
