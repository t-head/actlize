/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved. 
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <iostream>
#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/device/tensor_compare.h"

#include "helper.h"
#include "unfused_weight_dequantize.hpp"

#include "ppu_include.hpp"
#include "cutlass/gemm/collective/builders/ppu_mma_builder.inl"
#include "cutlass/gemm/collective/ppu_mma_aiu_multistage_with_scale.hpp"


using namespace cute;


/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////
using QuantType = cutlass::float_e4m3_t;
using ElementOutput = cutlass::half_t;

constexpr int Stages = 4;

// A matrix configuration
using         ElementA    = QuantType;                                      // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = QuantType;                                      // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand

constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)


using ElementScale = float;
using LayoutScale = cutlass::layout::RowMajor;

// C/D matrix configuration
using         ElementC    = ElementOutput;                                  // Element type for C and D matrix operands
using         LayoutC     = cutlass::layout::RowMajor;                      // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

// D matrix configuration
using         ElementD    = ElementOutput;
using         LayoutD     = LayoutC;
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;

// Core kernel configurations
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ElementCompute      = float;                                          // Element type for epilogue computation
using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag
using TileShape           = Shape<_192,_256,_128>;                          // Threadblock-level tile size
using WarpShape        = Shape<_48,_64,_128>;                            // Shape of the threadblocks in a cluster

using TileShapeRef           = Shape<_256,_256,_64>;                        // Threadblock-level tile size
using WarpShapeRef        = Shape<_64,_64,_64>;

using ScaleGranularityShape = Shape<_1,_128,_128>;

using ScaleConfig         = decltype(cutlass::detail::ppu_trivial_blockwise_scale_config<
                                     ScaleGranularityShape, false, true>(ScaleGranularityShape{}));
using LayoutSFA           = decltype(ScaleConfig::deduce_layoutSFA());                     // Layout type for SFA matrix operand
using LayoutSFB           = decltype(ScaleConfig::deduce_layoutSFB());                     // Layout type for SFB matrix operand

using KernelSchedule      = cutlass::gemm::KernelAiuMultistageWithBlockWiseScale;
using EpilogueSchedule    = cutlass::epilogue::EpilogueSimtVectorized;
using EpilogueTileType    = cutlass::epilogue::collective::EpilogueTileAuto;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::PPU0015, cutlass::arch::OpClassTensorOp,
    TileShape, WarpShape,
    EpilogueTileType,
    ElementCompute, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    EpilogueSchedule
  >::CollectiveOp;

// ============================================================ BlockWise SCALES ============================================================================
using CollectiveMainloopBlockWise = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::PPU0015, OperatorClass,
    ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
    ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB,
    ElementAccumulator,
    TileShape, WarpShape,
    Int<Stages>,
    KernelSchedule,
  >::CollectiveOp;

// Catch exhausted-iterator accesses that numerical checks can miss.
template <class Iterator>
struct CheckedKTileIterator {
  Iterator iterator;

  CUTLASS_DEVICE auto operator*() const {
    if (iterator == cute::ForwardCoordIteratorSentinel{}) {
      CUTE_INVALID_CONTROL_PATH("K-tile iterator accessed past the last tile\n");
    }
    return *iterator;
  }

  CUTLASS_DEVICE CheckedKTileIterator& operator++() {
    ++iterator;
    return *this;
  }
};

template <class Base>
struct CheckedCollective : Base {
  template <class... Ts, class FrgTensorC, class Iterator>
  CUTLASS_DEVICE void operator()(
      typename Base::Params const& params, cute::tuple<Ts...> const& inputs,
      FrgTensorC& accum, Iterator iterator, int tile_count, int thread_idx, char* smem_buf) {
    Base::operator()(params, inputs, accum,
        CheckedKTileIterator<Iterator>{iterator}, tile_count, thread_idx, smem_buf);
  }
};

using GemmKernelBlockWise = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, // Indicates ProblemShape
    CollectiveMainloopBlockWise,
    CollectiveEpilogue
>;

using GemmBlockWise = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelBlockWise>;

using CheckedGemmKernelBlockWise = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, CheckedCollective<CollectiveMainloopBlockWise>, CollectiveEpilogue>;
using CheckedGemmBlockWise = cutlass::gemm::device::GemmUniversalAdapter<CheckedGemmKernelBlockWise>;

using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;
using StrideC = cutlass::detail::TagToStrideC_t<LayoutC>;
using StrideD = cutlass::detail::TagToStrideC_t<LayoutD>;

//
// Data members
//

/// Initialization
StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideD stride_D;
LayoutSFA layout_SFA;
LayoutSFB layout_SFB;

uint64_t seed = 0;

cutlass::DeviceAllocation<ElementA> block_A;
cutlass::DeviceAllocation<ElementB> block_B;
cutlass::DeviceAllocation<ElementOutput> block_A_dq;
cutlass::DeviceAllocation<ElementOutput> block_B_dq;
cutlass::DeviceAllocation<ElementScale> block_scale_A;
cutlass::DeviceAllocation<ElementScale> block_scale_B;
cutlass::DeviceAllocation<ElementC> block_C;
cutlass::DeviceAllocation<ElementOutput> block_D;
cutlass::DeviceAllocation<ElementOutput> block_ref_D;


/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help = false;
  bool check_k_tiles = false;

  float alpha = 1.0f;
  float beta = 0.0f;
  int iterations = 10;
  int m = 5120, n = 4096, k = 4096;
  int l = 1;
  float eps = 0.001f;

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("l", l);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("eps", eps);
    check_k_tiles = cmd.check_cmd_line_flag("check-k-tiles");
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "104_ppu0015_fp8_gemm_with_blockwise_quant\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --l=<int>                   The number of independent gemm problems with mnk shape\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n"
      << "  --check-k-tiles              Check every mainloop K-tile iterator access\n"
      << "  --eps=<f32>          ";

    out
      << "\n\nExamples:\n\n"
      << "$ " << "104_ppu0015_fp8_gemm_with_blockwise_quant" << " --m=1024 --n=512 --k=1024 --g=128 --is_block_quant=1\n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * m * n * k * l;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

/// Result structure
struct Result
{
  double avg_runtime_ms = 0.0;
  double gflops = 0.0;
  cutlass::Status status = cutlass::Status::kSuccess;
  hggcError_t error = hggcSuccess;
  bool passed = false;

};


/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <class Element>
bool initialize_tensor(
  cutlass::DeviceAllocation<Element>& block,
  uint64_t seed=2023) {

  double scope_max, scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;
  int bits_output = cutlass::sizeof_bits<Element>::value;

  if (bits_input == 1) {
    scope_max = 2;
    scope_min = 0;
  }
  else if (bits_input <= 8) {
    scope_max = 2;
    scope_min = -2;
  }
  else if (bits_output == 16) {
    scope_max = 5;
    scope_min = -5;
  }
  else {
    scope_max = 8;
    scope_min = -8;
  }
  cutlass::reference::device::BlockFillRandomUniform(
      block.get(), block.size(), seed, Element(scope_max), Element(scope_min));

  return true;
}

template <typename Element>
bool initialize_quant_tensor(
  cutlass::DeviceAllocation<Element>& block,
  uint64_t seed=2023) {

  float scope_min = float(cutlass::platform::numeric_limits<Element>::lowest()) * 0.8;
  float scope_max = float(cutlass::platform::numeric_limits<Element>::max());

  cutlass::reference::device::BlockFillRandomUniform(
    block.get(), block.size(), seed, Element(scope_max), Element(scope_min));

  return true;
}

template <class Element>
bool initialize_scale(
  cutlass::DeviceAllocation<Element>& block,
  uint64_t seed=2023, bool no_scale=false) {

  float elt_max_f = float(cutlass::platform::numeric_limits<QuantType>::max());
  const float max_dequant_val = 4.0f;
  const float min_dequant_val = 0.5f;

  float scope_max(max_dequant_val / elt_max_f);
  float scope_min(min_dequant_val / elt_max_f);

  cutlass::reference::device::BlockFillRandomUniform(
    block.get(), block.size(), seed, Element(scope_max), Element(scope_min));
  if (no_scale) {
    // No scales, so just initialize with 1 so we can use the same kernel to dequantize the data.
    std::vector<Element> stage(block.size(), Element(0.01f));
    block.copy_from_host(stage.data());
  }
  return true;
}

template <class Element>
bool initialize_zero(
  cutlass::DeviceAllocation<Element>& block) {

  // No bias, so just initialize with 1 so we can use the same kernel to dequantize the data.
  std::vector<Element> stage(block.size(), Element(0.0f));
  block.copy_from_host(stage.data());
  return true;
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(Options const& options) {
  auto shape_a = cute::make_shape(options.m, options.k, options.l);
  auto shape_b = cute::make_shape(options.n, options.k, options.l);
  auto shape_c = cute::make_shape(options.m, options.n, options.l);

  auto ScaleGranularityM = size<0>(ScaleGranularityShape{});
  auto ScaleGranularityN = size<1>(ScaleGranularityShape{});
  auto ScaleGranularityK = size<2>(ScaleGranularityShape{});
  auto scale_k = (options.k + ScaleGranularityK - 1) / ScaleGranularityK;
  auto scale_m = (options.m + ScaleGranularityM - 1) / ScaleGranularityM;
  auto scale_n = (options.n + ScaleGranularityN - 1) / ScaleGranularityN;

  auto shape_scale_a = cute::make_shape(scale_m, scale_k, options.l);
  auto shape_scale_b = cute::make_shape(scale_n, scale_k, options.l);

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, shape_a);
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_b);
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, shape_c);
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, shape_c);

  auto layout_A = make_layout(shape_a, stride_A);
  auto layout_B = make_layout(shape_b, stride_B);

  layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(options.m, options.n, options.k, options.l));
  layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(options.m, options.n, options.k, options.l));

  auto a_coord = cutlass::make_Coord(options.m * options.l, options.k);
  auto b_coord = cutlass::make_Coord(options.k, options.n * options.l);
  auto c_coord = cutlass::make_Coord(options.m * options.l, options.n);

  block_A.reset(a_coord.product());
  block_B.reset(b_coord.product());
  block_A_dq.reset(a_coord.product());
  block_B_dq.reset(b_coord.product());
  block_C.reset(c_coord.product());
  block_D.reset(c_coord.product());
  block_ref_D.reset(c_coord.product());

  block_scale_A.reset(scale_k * options.l * scale_m);
  block_scale_B.reset(scale_k * options.l * scale_n);

  initialize_quant_tensor(block_A, seed + 2022);
  initialize_quant_tensor(block_B, seed + 2021);
  initialize_tensor(block_C, seed + 2020);
  initialize_scale(block_scale_A, seed+2023, false);
  initialize_scale(block_scale_B, seed+2024, false);

  dequantize_weight(block_A_dq.get(), block_A.get(), layout_A, block_scale_A.get(), layout_SFA);
  dequantize_weight(block_B_dq.get(), block_B.get(), layout_B, block_scale_B.get(), layout_SFB);
}

/// Populates a Gemm::Arguments structure from the given commandline options
template <typename Args>
Args args_from_options(Options const& options)
{
  return Args {
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k, options.l},
    {
      block_A.get(), stride_A, block_B.get(), stride_B, 4,
      block_scale_A.get(), layout_SFA, block_scale_B.get(), layout_SFB
    },
    {{options.alpha, options.beta}, block_C.get(), stride_C, block_D.get(), stride_D}
  };
}

template<typename T>
void check_eps(int len, const T* v1,  const T* v2, float eps, hggcStream_t stream) {
  T* host_v1 = new T[len];
  T* host_v2 = new T[len];
  hggcMemcpyAsync(host_v1, v1, sizeof(T) * len, hggcMemcpyDeviceToHost, stream);
  hggcMemcpyAsync(host_v2, v2, sizeof(T) * len, hggcMemcpyDeviceToHost, stream);
  hggcStreamSynchronize(stream);
  float max_eps = 0;
  for (int i = 0; i < len; ++i) {
    float cur_v1 = float(host_v1[i]);
    float cur_v2 = float(host_v2[i]);
    float cur_eps = abs(cur_v2 - cur_v1);
    if (cur_eps > eps) {
      printf("%.4f, %.4f, abs:%.4f\n", cur_v1, cur_v2, cur_eps);
    }
    max_eps = max(max_eps, cur_eps);
  }
  printf("max_eps:%.4f\n", max_eps);
  delete[] host_v1;
  delete[] host_v2;
}

bool verify(const Options &options) {
  //
  // Compute reference output
  //

  // In this example, we use the PPU default kernels as a reference (unfused scale)
  // This avoids numerical differences due to different accumulation order.

  using ScheduleRef = cutlass::gemm::collective::KernelScheduleAuto;

  using CollectiveMainloopRef = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::PPU0015, OperatorClass,
      ElementOutput, LayoutA, AlignmentA,
      ElementOutput, LayoutB, AlignmentB,
      float,
      TileShapeRef, WarpShapeRef,
      cute::Int<4>,
      ScheduleRef
    >::CollectiveOp;

  using CollectiveEpilogueRef = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::PPU0015, cutlass::arch::OpClassTensorOp,
      TileShapeRef, WarpShapeRef,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::NoSmemWarpSpecialized
    >::CollectiveOp;

  using GemmKernelRef = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>, // Indicates ProblemShape
      CollectiveMainloopRef,
      CollectiveEpilogueRef
  >;

  using GemmRef = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelRef>;

  typename GemmRef::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k, options.l},
    {block_A_dq.get(), stride_A, block_B_dq.get(), stride_B},
    {{options.alpha, options.beta}, block_C.get(), stride_C, block_ref_D.get(), stride_D}
  };

  // Run the gemm where the scaling is performed outside of the kernel.
  GemmRef gemm_ref;
  size_t workspace_size = GemmRef::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  CUTLASS_CHECK(gemm_ref.can_implement(arguments));
  CUTLASS_CHECK(gemm_ref.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(gemm_ref.run());

  // compare_reference
  ElementD const epsilon(5e-1f);
  ElementD const non_zero_floor(1e-4f);
  bool passed = cutlass::reference::device::BlockCompareRelativelyEqual(block_ref_D.get(), block_D.get(), block_D.size(), epsilon, non_zero_floor);
  // check_eps(block_D.size(), block_ref_D.get(), block_D.get(), options.eps, 0);
  return passed;
}

/// Execute a given example GEMM computation
template <class Gemm = GemmBlockWise>
int run(Options &options)
{
  initialize(options);

  // Instantiate kernel depending on templates
  Gemm gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options<typename Gemm::Arguments>(options);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check if the problem size is supported or not
  CUTLASS_CHECK(gemm.can_implement(arguments));

  // Initialize kernel with arguments and workspace pointer
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

  // Correctness / Warmup iteration
  CUTLASS_CHECK(gemm.run());


  // Check if output from kernel and reference kernel are equal or not
  Result result;
  result.passed = verify(options);

  std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;

  if (!result.passed) {
    exit(-1);
  }

  return 0;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char const **args) {
  hggcDeviceProp props;
  int current_device_id;
  CUTLASS_PPU_CHECK(hggcGetDevice(&current_device_id));
  CUTLASS_PPU_CHECK(hggcGetDeviceProperties(&props, current_device_id));
  hggcError_t error = hggcGetDeviceProperties(&props, 0);

  // should run on PPU 1.5
  if (props.major != 8 || props.minor != 9) {
    std::cerr << " This example should be run on PPU 1.5!!! " << std::endl;
    return 0;
  }
  //
  // Parse options
  //
  Options options;
  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  //
  // Evaluate kernels
  //
  if (options.check_k_tiles) {
    return run<CheckedGemmBlockWise>(options);
  }
  run(options);
  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
