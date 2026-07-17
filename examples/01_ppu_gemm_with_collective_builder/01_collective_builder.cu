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

/*! \file
    \brief PPU GEMM example leveraging collective operation builders.

    Example usage:
      $ ./examples/01_ppu_gemm_with_collective_builder/01_collective_builder \
            --m=2048 --n=2048 --k=2048 --l=2
*/

#include <iostream>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"

#ifdef __HGGCCC__
#define ACOMPUTE_VERSION 10000
#include "ppu/ppu_include.hpp"
#include "ppu/cutlass/gemm/collective/ppu_mma_builder.inl"
#endif

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Command line options parsing
struct Options {

  bool help;
  bool error;

  int m, n, k, l;
  float alpha, beta;

  Options():
    help(false),
    error(false),
    m(2048), n(2048), k(2048), l(1),
    alpha(1.f), beta(0.f)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m, 2048);
    cmd.get_cmd_line_argument("n", n, 2048);
    cmd.get_cmd_line_argument("k", k, 2048);
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "01_ppu_gemm_with_collective_builder\n\n"
      << "  This example showcases the use of collective operation builders to easily construct\n"
      << "  performant kernels targeting PPU architecture.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --l=<int>                   Sets the L extent (batch count) of the GEMM\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n";

    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <class Element>
bool initialize_block(
  cutlass::DeviceAllocation<Element>& block,
  uint64_t seed=2023) {

  Element scope_max, scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;

  if (bits_input == 1) {
    scope_max = 2;
    scope_min = 0;
  } else if (bits_input <= 8) {
    scope_max = 2;
    scope_min = -2;
  } else {
    scope_max = 8;
    scope_min = -8;
  }

  cutlass::reference::device::BlockFillRandomUniform(
    block.get(), block.size(), seed, scope_max, scope_min, 0);

  return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////



// Wrapper to construct, run, and verify a GEMM. This example showcases collective
// operation builders by specializing the GEMM only on the kernel schedule it will use and the
// number of pipeline stages.
//
// One can use a special `Auto` type that tells the CollectiveBuilder
// to select an appropriate value on its own. The CollectiveBuilder will attempt to select
// configurations that will result in the most-performant kernel, but this is not a guarantee.
//
// If relying on 'Auto' schedules, all builders must use the 'Auto' schedule to ensure compatiblity.
// For example, if `KernelScheduleAuto` is used for the mainloop builder, `EpilogueScheduleAuto` must
// be used for the epilogue builder.
//
// Furthermore, if an override schedule is selected, both epilogue and mainloop schedules must
// be specifically opt into a compatible selection.
//
// Behavior of the CollectiveBuilder with `Auto` types is subject to change in future releases
// -- do not rely on `Auto` if you require a specific scheduling policy.
template <
  // Type of kernel schedule to generate
  class MainloopScheduleType = cutlass::gemm::collective::KernelScheduleAuto,
  // Type of epilogue schedule to generate
  class EpilogueScheduleType = cutlass::epilogue::collective::EpilogueScheduleAuto,
  // Number of pipeline stages to use
  class StageCountType = cutlass::gemm::collective::StageCountAuto,
  // Type of tile scheduler to use
  class TileSchedulerType = cutlass::gemm::PersistentScheduler,
  // Do we use custom epilogue visitor tree (EVT) fusion
  bool UseCustomEVT = false
>
struct ExampleRunner {

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::ColumnMajor;

  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementScalar = float;

  // 16B alignment lets us use TMA
  static constexpr int AlignmentA = 16 / sizeof(ElementA);
  static constexpr int AlignmentB = 16 / sizeof(ElementB);
  static constexpr int AlignmentC = 16 / sizeof(ElementC);
  static constexpr int AlignmentD = 16 / sizeof(ElementD);

  static_assert(not UseCustomEVT ||
    (cute::is_same_v<EpilogueScheduleType, cutlass::epilogue::TmaWarpSpecialized> ||
     cute::is_same_v<EpilogueScheduleType, cutlass::epilogue::TmaWarpSpecializedCooperative>),
    "Epilogue visitor trees are currently only supported by the TMA warp-specialized epilogue");
  static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;

  // EVTs can be constructed by composing the fundamental load/store/compute visitor operations defined in include/cutlass/epilogue/fusion
  // For more complex examples of EVT construction please refer to include/cutlass/epilogue/fusion/ppu15_callbacks_tma_warpspecialized.hpp
  using CustomEVT =  // alpha * acc + beta * C
    cutlass::epilogue::fusion::PPU0015EVT<cutlass::epilogue::fusion::PPU0015Compute<cutlass::homogeneous_multiply_add, ElementD, ElementCompute, RoundStyle>, // beta * C + (alpha * acc)
      cutlass::epilogue::fusion::PPU0015ScalarBroadcast<ElementScalar>, // beta
      cutlass::epilogue::fusion::PPU0015SrcFetch<ElementC>, // C
      cutlass::epilogue::fusion::PPU0015EVT<cutlass::epilogue::fusion::PPU0015Compute<cutlass::multiplies, ElementCompute, ElementCompute, RoundStyle>, // alpha * acc
        cutlass::epilogue::fusion::PPU0015ScalarBroadcast<ElementScalar>, // alpha
        cutlass::epilogue::fusion::PPU0015AccFetch // acc
      >
    >;

  // A predefined set of fusion operations (implemented with EVT) are supported by the TMA warp-specialized epilogue.
  // Users can select one of these operations by passing one of the tags defined in include/cutlass/epilogue/fusion/operations.hpp
  // to the CollectiveBuilder. This frees the user from having to compute additional parameters such as stage counts and copy atoms/layouts.
  // These tags also provide additional metadata that can be queried at compile time.
  using DefaultOperation = cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute, ElementC, ElementScalar, RoundStyle>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::PPU0010, cutlass::arch::OpClassTensorOp,
      Shape<_128,_128,_64>, Shape<_1,_1,_1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      EpilogueScheduleType,
      cute::conditional_t<UseCustomEVT, CustomEVT, DefaultOperation>
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::PPU0010, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      Shape<_128,_128,_64>, Shape<_1,_1,_1>,
      cute::conditional_t<cute::is_same_v<StageCountType, cutlass::gemm::collective::StageCountAuto>,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
          StageCountType>,
      MainloopScheduleType
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      TileSchedulerType
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using LayoutTagA = cutlass::gemm::detail::StrideToLayoutTagA_t<StrideA>;
  using LayoutTagB = cutlass::gemm::detail::StrideToLayoutTagB_t<StrideB>;
  using LayoutTagC = cutlass::gemm::detail::StrideToLayoutTagC_t<StrideC>;
  using LayoutTagD = cutlass::gemm::detail::StrideToLayoutTagC_t<StrideD>;

  //
  // Data members
  //

  /// Initialization
  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;
  uint64_t seed = 0;

  cutlass::DeviceAllocation<typename Gemm::ElementA> block_A;
  cutlass::DeviceAllocation<typename Gemm::ElementB> block_B;
  cutlass::DeviceAllocation<typename Gemm::ElementC> block_C;
  cutlass::DeviceAllocation<typename Gemm::ElementD> block_D;
  cutlass::DeviceAllocation<typename Gemm::ElementD> block_ref_D;

  //
  // Methods
  //

  bool verify(const ProblemShapeType& problem_size, float alpha, float beta) {
    auto [M, N, K, L] = problem_size;

    cutlass::TensorRef ref_A(block_A.get(), Gemm::LayoutA::packed({M, K}));
    cutlass::TensorRef ref_B(block_B.get(), Gemm::LayoutB::packed({K, N}));
    cutlass::TensorRef ref_C(block_C.get(), Gemm::LayoutC::packed({M, N}));
    cutlass::TensorRef ref_D(block_ref_D.get(), Gemm::LayoutD::packed({M, N}));

    cutlass::reference::device::GemmComplex(
          {M, N, K},
          ElementScalar(alpha),
          ref_A,
          cutlass::ComplexTransform::kNone,
          ref_B,
          cutlass::ComplexTransform::kNone,
          ElementScalar(beta),
          ref_C,
          ref_D,
          ElementAccumulator(0),
          L,     // batch_count
          M * K, // batch_stride_A
          K * N, // batch_stride_B
          M * N, // batch_stride_C
          M * N  // batch_stride_D
        );

    hggcError_t result = hggcDeviceSynchronize();
    if (result != hggcSuccess) {
      std::cerr << "Reference kernel failed. Last device error: "
                << hggcGetErrorString(result) << std::endl;
      return false;
    }

    // Check if output from kernel and reference kernel are equal or not
    bool passed = cutlass::reference::device::BlockCompareEqual(block_ref_D.get(), block_D.get(), block_D.size());

    return passed;
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  void initialize(const ProblemShapeType& problem_size) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

    block_A.reset(M * K * L);
    block_B.reset(K * N * L);
    block_C.reset(M * N * L);
    block_D.reset(M * N * L);
    block_ref_D.reset(M * N * L);

    initialize_block(block_A, seed + 2023);
    initialize_block(block_B, seed + 2022);
    initialize_block(block_C, seed + 2021);
  }

  bool run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType problem_size = ProblemShapeType{options.m, options.n, options.k, options.l};

    initialize(problem_size);

    typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {block_A.get(), stride_A, block_B.get(), stride_B},
      {{}, // epilogue.thread
       block_C.get(), stride_C, block_D.get(), stride_D},
      hw_info
    };

    // Custom EVT fusions will have nested unnamed args, the structure of which
    // can be deduced from the type definition of the EVT.
    // Each node's arguments has the recursive structure of
    // {first_child_args, ..., last_child_args, op_args},
    // For more complex examples of EVT initialization please refer to
    // include/cutlass/epilogue/fusion/ppu15_callbacks_tma_warpspecialized.hpp
    if constexpr (UseCustomEVT) {
      arguments.epilogue.thread =
        {    // ternary op : beta * C + (alpha * acc)
          {{options.beta}}, // leaf op+args : beta
          {},               // leaf op+args : C
          {                 // binary op : alpha * acc
            {{options.alpha}}, // leaf op+args : alpha
            {},                // leaf op+args : acc
            {}              // binary args : multiplies
          },                // end binary op
          {} // ternary args : multiply_add
        };   // end ternary op
    }
    // Pre-defined fusions will have flat, named args for user-friendlyness
    else {
      arguments.epilogue.thread.alpha = options.alpha;
      arguments.epilogue.thread.beta = options.beta;
    }

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "This kernel is not supported. Last device error is: "
                << hggcGetErrorString(hggcGetLastError()) << std::endl;
      return false;
    }

    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to initialize the kernel. Last device error is: "
                << hggcGetErrorString(hggcGetLastError()) << std::endl;
      return false;
    }

    // Run the GEMM
    status = gemm_op.run();
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to launch the kernel. Last device error is: "
                << hggcGetErrorString(hggcGetLastError()) << std::endl;
      return false;
    }

    hggcError_t result = hggcDeviceSynchronize();
    if (result != hggcSuccess) {
      std::cerr << "Error running the kernel. Last device error is: "
                << hggcGetErrorString(result) << std::endl;
      return false;
    }

    // Verify that the result is correct
    bool passed = verify(problem_size, options.alpha, options.beta);
    if (!passed) {
      std::cerr << "Reference check failed" << std::endl;
    }

    return passed;
  }

};



///////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to print a description of the example run and its result
void print_result(const std::string& description, bool passed) {
  std::cout << description << ": " << (passed ? "Passed" : "Failed") << std::endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  hggcDeviceProp props;

  hggcError_t error = hggcGetDeviceProperties(&props, 0);
  if (error != hggcSuccess) {
    std::cerr << "hggcGetDeviceProperties() returned an error: " << hggcGetErrorString(error) << std::endl;
    return -1;
  }

  if (!(props.major == 8 && props.minor == 0)) {
    std::cerr << "unsupported PPU arch !!!" << std::endl;
    return -1;
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

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }



  //
  // Run examples
  //

  // The KernelHardwareInfo struct holds the number of CUs on the GPU with a given device ID. This
  // information is used by the underlying kernel.
  cutlass::KernelHardwareInfo hw_info;

  // Change device_id to another value if you are running on a machine with multiple GPUs and wish
  // to use a GPU other than that with device ID 0.
  hw_info.device_id = 0;
  hw_info.cu_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  bool passed;

  // This first example constructs a GEMM using the default schedule and stage count provided by
  // the CollectiveBuilder. The scheduling policy that is expected to be most performant will be
  // selected and the maximum number of stages that can fit in shared memory will be selected.
  //
  // This example is equivalent to declaring
  //    ExampleRunner<
  //        cutlass::gemm::collective::KernelScheduleAuto,
  //        cutlass::epilogue::collective::EpilogueScheduleAuto,
  //        cutlass::gemm::collective::StageCountAuto>
  // Each of the `Auto` types indicate that the CollectiveBuilder should determine the scheduling policy and
  // stage count. Note that the behavior of the CollectiveBuilder with `Auto` parameters is subject to change
  // -- do not rely on `Auto` if you require a specific scheduling policy.
  // If you opt in to a non-'Auto' schedule, make sure all collectives are built using specific, compatible schedules.
  ExampleRunner<> auto_schedule_auto_stage_runner;
  passed = auto_schedule_auto_stage_runner.run(options, hw_info);
  print_result("Automatically-selected schedule and stage count", passed);

  // One can override the stage count used in the GEMM by replacing cutlass::gemm::collective::StageCountAuto
  // with the number of stages to use (5 in this case).
  ExampleRunner<
    cutlass::gemm::collective::KernelScheduleAuto,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    _5> auto_schedule_5_stage_runner;

  passed = auto_schedule_5_stage_runner.run(options, hw_info);
  print_result("Automatically-selected schedule with 5 stages", passed);

  // One can also override the scheduling policy to use.
  ExampleRunner<cutlass::gemm::KernelTma, cutlass::epilogue::NoSmemWarpSpecialized> tma_schedule_auto_stage_runner;
  passed = tma_schedule_auto_stage_runner.run(options, hw_info);
  print_result("TMA schedule with automatically-selected stage count", passed);

  ExampleRunner<cutlass::gemm::KernelTmaWarpSpecialized, cutlass::epilogue::NoSmemWarpSpecialized> ws_schedule_auto_stage_runner;
  passed = ws_schedule_auto_stage_runner.run(options, hw_info);
  print_result("Warp-specialized TMA schedule with automatically-selected stage count", passed);

#if 0
  ExampleRunner<
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized> ws_pingpong_schedule_auto_stage_runner;
  passed = ws_pingpong_schedule_auto_stage_runner.run(options, hw_info);
  print_result("Ping-pong warp-specialized TMA schedule with automatically-selected stage count", passed);
#endif

  // Here, we override the scheduling policy to use stream-K problem decomposition atop the cooperative
  // warp-specialized scheduling policy. This kernel continues to leverage persistent thread blocks
  // as well aso TMA in both the mainloop and epilogue.
  ExampleRunner<
    cutlass::gemm::KernelAiuMultistageStreamK,
    cutlass::epilogue::NoSmemWarpSpecialized,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::StreamKScheduler> ws_cooperative_stream_k_schedule_auto_stage_runner;
  passed = ws_cooperative_stream_k_schedule_auto_stage_runner.run(options, hw_info);
  print_result("Cooperative warp-specialized TMA schedule using stream-K with automatically-selected stage count", passed);

#if 0
  // Here, we override the fusion operation to use a customized EVT fusion, in addition to the previous schedule overrides
  ExampleRunner<
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::PersistentScheduler,
    true> ws_cooperative_schedule_auto_stage_custom_evt_runner;
  passed = ws_cooperative_schedule_auto_stage_custom_evt_runner.run(options, hw_info);
  print_result("Cooperative warp-specialized TMA schedule using custom epilogue visitor tree with automatically-selected stage count", passed);
#endif

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
