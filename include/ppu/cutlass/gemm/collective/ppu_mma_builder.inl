#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cutlass/detail/collective.hpp"

#define ENABLE_AIU 1

namespace cutlass::gemm::collective {

namespace detail {

template <typename Element, typename Layout, int Alignment, int SizeK>
struct DefaultGemm_TensorOpPPU0010_OperandA;

template <typename Element, typename Layout, int Alignment, int SizeK>
struct DefaultGemm_TensorOpPPU0010_OperandB;

//
// F16: 128-by-128-by-64
//

/// Operand A - Row-major (K-Major)
template <>
struct DefaultGemm_TensorOpPPU0010_OperandA<half_t, layout::RowMajor, 8, 64>
{
  // Smem
  using SmemLayoutAtom = decltype(
    composition(Swizzle<3,3,3>{},
                Layout<Shape < _8,_64>,
                       Stride<_64, _1>>{}));
  using SmemCopyAtom = Copy_Atom<PPU_U32x4_LDSM_N, half_t>;

  // Gmem
  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, half_t>{},
                    Layout<Shape <_16,_8>,
                           Stride< _8,_1>>{},
                    Layout<Shape < _1,_8>>{}));
};

/// Operand A - Column-major (M-major)
template <int SizeK>
struct DefaultGemm_TensorOpPPU0010_OperandA<half_t, layout::ColumnMajor, 8, SizeK>
{
  // Smem
  using SmemLayoutAtom = decltype(
    composition(Swizzle<3,3,3>{},
                Layout<Shape <_64, _8>,
                       Stride< _1,_64>>{}));
  using SmemCopyAtom = Copy_Atom<PPU_U16x8_LDSM_T, half_t>;

  // Gmem
  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, half_t>{},
                    Layout<Shape <_16, _8>,
                           Stride< _1,_16>>{},
                    Layout<Shape < _8, _1>>{}));
};

// Because the F32F16 tiledMMA is A-B symmetric, we can reuse the DefaultOperands

// Operand B - Column-Major (K-major)
template <int Alignment, int SizeK>
struct DefaultGemm_TensorOpPPU0010_OperandB<half_t, layout::ColumnMajor, Alignment, SizeK>
     : DefaultGemm_TensorOpPPU0010_OperandA<half_t, layout::RowMajor,    Alignment, SizeK>
{};

// Operand B - Row-Major (N-major)
template <int Alignment, int SizeK>
struct DefaultGemm_TensorOpPPU0010_OperandB<half_t, layout::RowMajor,    Alignment, SizeK>
     : DefaultGemm_TensorOpPPU0010_OperandA<half_t, layout::ColumnMajor, Alignment, SizeK>
{};

//
// F16: 128-by-128-by-32 (small k-block)
//

/// Operand A - Row-major (K-Major)
template <>
struct DefaultGemm_TensorOpPPU0010_OperandA<half_t, layout::RowMajor, 8, 32>
{
  // Smem
  using SmemLayoutAtom = decltype(
    composition(Swizzle<2,3,3>{},
                Layout<Shape < _8,_32>,
                       Stride<_32, _1>>{}));
  using SmemCopyAtom = Copy_Atom<PPU_U32x4_LDSM_N, half_t>;

  // Gmem
  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<PPU_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, half_t>{},
                    Layout<Shape <_32,_4>,
                           Stride< _4,_1>>{},
                    Layout<Shape < _1,_8>>{}));
};

///////////////////////////////////////////////////////////////////////////////
#if ENABLE_AIU
// ========== aiu gemm ==========
template <
  typename Element,
  bool Trans,
  typename Block_MN,
  typename Block_K,
  bool Swap
> struct DefaultGemm_AIU_Operand;

template <
  typename Element,
  typename Block_MN,
  typename Block_K,
  bool Swap
> struct DefaultGemm_AIU_Operand<
  Element,
  false,
  Block_MN,
  Block_K,
  Swap
> {
  static constexpr int BlockContSize = Block_K{} * sizeof_bits<Element>::value / 8;
  static_assert(BlockContSize % 32 == 0, "aiu_trans: block contiguous size should be multiple of 32B");
  static_assert(BlockContSize > 128 ? (BlockContSize % 128 == 0) : (BlockContSize % 32 == 0), "aiu_trans: block contiguous size should be multiple of 128B or 32B");
  static constexpr int AiuContByteSize = BlockContSize > 128 ? 128 : BlockContSize;
  using AiuContElemSize = Int<AiuContByteSize / sizeof_bits<Element>::value * 8>;
  static constexpr int InstNum = Block_K{} / AiuContElemSize{};

  static constexpr int bits_per_aiu = Block_MN{} * AiuContElemSize{} * sizeof_bits<Element>::value;
  using CopyInst = PPU0010_AIU_LOAD<cute::C<bits_per_aiu>, Element, false>;

  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<CopyInst, Element>{},
                    Layout<Shape <_1,_1>,
                           Stride<_1,_1>>{},
                    Layout<Shape <Block_MN, AiuContElemSize>>{}));

  using SmemCopyOp = PPU0010_TSM_LD_SWZL<Element, Block_MN{}, AiuContElemSize{}, Swap, false, InstNum>;
  using SmemCopyAtom = Copy_Atom<SmemCopyOp, Element>;
  using SmemLayoutAtom = Layout<Shape<_8, AiuContElemSize>, Stride<AiuContElemSize, _1>>;
};

template <
  typename Element,
  typename Block_MN,
  typename Block_K,
  bool Swap
> struct DefaultGemm_AIU_Operand<
  Element,
  true,
  Block_MN,
  Block_K,
  Swap
> {
  static constexpr int BlockContSize = Block_MN{} * sizeof(Element);
  static_assert(BlockContSize % 64 == 0, "aiu_trans: block contiguous size should be multiple of 64B");
  static constexpr int AiuContByteSize = BlockContSize % 128 == 0 ? 128 : 64;
  using AiuContElemSize = Int<AiuContByteSize / sizeof(Element)>;
  static constexpr int InstNum = Block_MN{} / AiuContElemSize{};

  static constexpr int bits_per_aiu = AiuContByteSize * 8 * Block_K{};
  using CopyInst = PPU0010_AIU_LOAD<cute::C<bits_per_aiu>, Element, true>;

  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<CopyInst, Element>{},
                    Layout<Shape <_1,_1>,
                           Stride<_1,_1>>{},
                    Layout<Shape <AiuContElemSize,Block_K>>{}));

  using SmemCopyOp = PPU0010_TSM_LD_SWZL<Element, Block_K{}, AiuContElemSize{}, Swap, true, InstNum>;
  using SmemCopyAtom = Copy_Atom<SmemCopyOp, Element>;
  using SmemLayoutAtom = Layout<Shape<AiuContElemSize, Block_K>, Stride<_1, AiuContElemSize>>;
};

template <
  typename Block_MN,
  typename Block_K,
  bool Swap
> struct DefaultGemm_AIU_Operand<
  cutlass::int4b_t,
  false,
  Block_MN,
  Block_K,
  Swap,
> {
  static constexpr int BlockContSize = Block_K{} * sizeof_bits<cutlass::int4b_t>::value / 8;
  static_assert(BlockContSize % 32 == 0, "aiu_no_trans: block_k must be multiple of 32B");
  static_assert(BlockContSize > 128 ? (BlockContSize % 128 == 0) : (BlockContSize % 32 == 0), "aiu_trans: block contiguous size should be multiple of 128B or 32B");
  static constexpr int AiuContByteSize = BlockContSize > 128 ? 128 : BlockContSize;
  using AiuContElemSize = Int<AiuContByteSize / sizeof_bits<cutlass::int4b_t>::value * 8>;
  static constexpr int InstNum = Block_K{} / AiuContElemSize{};

  static constexpr int bits_per_aiu = Block_MN{} * AiuContByteSize * 8;
  using CopyInst = PPU0010_AIU_LOAD<cute::C<bits_per_aiu>, cutlass::int4b_t, false>;     // load as i8

  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<CopyInst, cutlass::int4b_t>{},
                    Layout<Shape <_1,_1>,
                           Stride<_1,_1>>{},
                    Layout<Shape <Block_MN, AiuContElemSize>>{}));

  using SmemCopyOp = PPU0010_TSM_LD_SWZL<int8_t, Block_MN{}, AiuContElemSize{} / 2, Swap, false, InstNum>;
  using SmemCopyAtom = Copy_Atom<SmemCopyOp, int8_t>;
  using SmemLayoutAtom = Layout<Shape<_8, AiuContElemSize>, Stride<AiuContElemSize, _1>>;
};

#endif

template < typename ElementA, typename ElementB = ElementA, typename ElementC = ElementA>
auto get_ppu_mma_atom () {}

template <>
auto get_ppu_mma_atom<half_t, half_t, float> () {
  return MMA_Atom<PPU_16x16x16_F32F16F16F32_TN>{};
}

template <>
auto get_ppu_mma_atom<tfloat32_t, tfloat32_t, float> () {
  return MMA_Atom<PPU_16x16x8_F32TF32TF32F32_TN>{};
}

template <>
auto get_ppu_mma_atom<float, float, float> () {
  return MMA_Atom<PPU_16x16x8_F32TF32TF32F32_TN>{};
}

template <>
auto get_ppu_mma_atom<int8_t, int8_t, int32_t> () {
  return MMA_Atom<PPU_16x16x32_S32S8S8S32_TN>{};
}

} // namespace detail


// AIU GEMM
template <
  class ElementA,
  class GmemLayoutA,
  int AlignmentA,
  class ElementB,
  class GmemLayoutB,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::PPU0010,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutA,
    AlignmentA,
    ElementB,
    GmemLayoutB,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
      (cute::is_same_v<KernelScheduleType, KernelScheduleAuto> ||
       cute::is_same_v<KernelScheduleType, KernelMultistage> ||
       cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecialized> ||
       cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecializedPingpong> ||
       cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecializedCooperative> ||
       cute::is_same_v<KernelScheduleType, KernelTma> ||
       cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecialized> ||
       cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedPingpong> ||
       cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedCooperative>)>
> {

  // For fp32 types, map to tf32 MMA value type
  using MmaElementA = ElementA; //cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using MmaElementB = ElementB; //cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

  using TiledMma = TiledMMA<
      decltype(detail::get_ppu_mma_atom<MmaElementA, MmaElementB, ElementAccumulator>()),
      Layout<Shape<_2,_2,_1>>,  // 2x2x1 thread group
      Tile<_64, _32, _32>>; // 2x1x1 value group for 16x16x16 MMA and LDSM

#if ENABLE_AIU
  static constexpr int blockM = cute::get<0>(TileShape_MNK{});
  static constexpr int blockN = cute::get<1>(TileShape_MNK{});
  static constexpr int blockK = cute::get<2>(TileShape_MNK{});
  using DispatchPolicy = MainloopPPUAiu<3>;
  static constexpr bool TransA = platform::is_same<GmemLayoutA, cutlass::layout::RowMajor>::value ? false : true;
  static constexpr bool TransB = platform::is_same<GmemLayoutB, cutlass::layout::ColumnMajor>::value ? false : true;
  using DefaultOperandA = detail::DefaultGemm_AIU_Operand<ElementA, TransA, Int<blockM>, Int<blockK>, true>;
  using DefaultOperandB = detail::DefaultGemm_AIU_Operand<ElementB, TransB, Int<blockN>, Int<blockK>, true>;
#else
  using DispatchPolicy = MainloopPPU0010CpAsync<3>;
  using DefaultOperandA = detail::DefaultGemm_TensorOpPPU0010_OperandA<
    ElementA, GmemLayoutA, AlignmentA, 32>;
  using DefaultOperandB = detail::DefaultGemm_TensorOpPPU0010_OperandB<
    ElementB, GmemLayoutB, AlignmentB, 32>;
#endif

  using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom; // M, K
  using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;

  // B
  using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom; // N, K
  using SmemCopyAtomB = typename DefaultOperandB::SmemCopyAtom;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

  // Mainloop
  using CollectiveOp = collective::CollectiveMma<
    DispatchPolicy, TileShape_MNK,
    MmaElementA, TagToStrideA_t<GmemLayoutA>,
    MmaElementA, TagToStrideB_t<GmemLayoutB>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

};


// AIU Mixed GEMM
template <
  class ElementPairA_,
  class GmemLayoutA,
  int AlignmentA,
  class ElementPairB_,
  class GmemLayoutB,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::PPU0010,
    arch::OpClassTensorOp,
    ElementPairA_,
    GmemLayoutA,
    AlignmentA,
    ElementPairB_,
    GmemLayoutB,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
      (cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedMixedInput> ||
       cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedPingpongMixedInput> ||
       cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedCooperativeMixedInput>)>
> {
private:
  using ScaleA = detail::deduce_mixed_width_dtype_t<1, ElementPairA_>;
  using ScaleB = detail::deduce_mixed_width_dtype_t<1, ElementPairB_>;
  using ZeroA = detail::deduce_mixed_width_dtype_t<2, ElementPairA_>;
  using ZeroB = detail::deduce_mixed_width_dtype_t<2, ElementPairB_>;
  static constexpr bool NeitherIsTuple = !cute::is_tuple<ElementPairA_>::value && !cute::is_tuple<ElementPairB_>::value;

public:
  using ElementA = detail::deduce_mixed_width_dtype_t<0, ElementPairA_>;
  using ElementB = detail::deduce_mixed_width_dtype_t<0, ElementPairB_>;
  static_assert(cute::is_tuple<ElementPairA_>::value ^ cute::is_tuple<ElementPairB_>::value ||
               (NeitherIsTuple && (sizeof_bits<ElementA>::value != sizeof_bits<ElementB>::value)),
    "Either A OR B must be a tuple or the widths of A and B must be different.");

  static constexpr bool IsANarrow = sizeof_bits<ElementA>::value < sizeof_bits<ElementB>::value;

  using ElementPairA = cute::conditional_t<IsANarrow && NeitherIsTuple, cute::tuple<ElementA>, ElementPairA_>;
  using ElementPairB = cute::conditional_t<!IsANarrow && NeitherIsTuple, cute::tuple<ElementB>, ElementPairB_>;

  static constexpr bool IsATransformed = cute::is_tuple<ElementPairA>::value;
  using ElementScale = cute::conditional_t<IsATransformed, ScaleA, ScaleB>;
  using ElementZero = cute::conditional_t<IsATransformed, ZeroA, ZeroB>;
  // For fp32 types, map to tf32 MMA value type
  // using MmaElementA = ElementA; //cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  // using MmaElementB = ElementB; //cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

  using ElementMma = cute::conditional_t<IsATransformed, ElementB, ElementA>;
  using TiledMma = TiledMMA<
      decltype(detail::get_ppu_mma_atom<ElementMma, ElementMma, ElementAccumulator>()),
      Layout<Shape<_2,_2,_1>>,  // 2x2x1 thread group
      Tile<_32, _32, Int<32 * 8 / sizeof_bits<ElementA>::value>>>; // 2x1x2 value group for 16x16x16 MMA and LDSM

#if ENABLE_AIU
  static constexpr int blockM = cute::get<0>(TileShape_MNK{});
  static constexpr int blockN = cute::get<1>(TileShape_MNK{});
  static constexpr int blockK = cute::get<2>(TileShape_MNK{});
  using DispatchPolicy = MainloopPPUAiuMixedInput<3>;
  static constexpr bool TransA = platform::is_same<GmemLayoutA, cutlass::layout::RowMajor>::value ? false : true;
  static constexpr bool TransB = platform::is_same<GmemLayoutB, cutlass::layout::ColumnMajor>::value ? false : true;
  using DefaultOperandA = detail::DefaultGemm_AIU_Operand<ElementA, TransA, Int<blockM>, Int<blockK>, true>;
  using DefaultOperandB = detail::DefaultGemm_AIU_Operand<ElementB, TransB, Int<blockN>, Int<blockK>, true>;
#else
  using DispatchPolicy = MainloopPPUAiuMixedInput<3>;
  using DefaultOperandA = detail::DefaultGemm_TensorOpPPU0010_OperandA<
    ElementA, GmemLayoutA, AlignmentA, 32>;
  using DefaultOperandB = detail::DefaultGemm_TensorOpPPU0010_OperandB<
    ElementB, GmemLayoutB, AlignmentB, 32>;
#endif
  using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom; // M, K
  using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;

  // B
  using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom; // N, K
  using SmemCopyAtomB = typename DefaultOperandB::SmemCopyAtom;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

  // Mainloop
  using CollectiveOp = collective::CollectiveMma<
    DispatchPolicy, TileShape_MNK,
    ElementPairA, TagToStrideA_t<GmemLayoutA>,
    ElementPairB, TagToStrideB_t<GmemLayoutB>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

};

// AIU GEMM for Batch Array
template <
  class ElementA,
  class GmemLayoutA,
  int AlignmentA,
  class ElementB,
  class GmemLayoutB,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::PPU0010,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutA,
    AlignmentA,
    ElementB,
    GmemLayoutB,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
      (cute::is_same_v<KernelScheduleType, KernelPtrArrayTmaWarpSpecializedCooperative>)>
> {

  // For fp32 types, map to tf32 MMA value type
  using MmaElementA = ElementA; //cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using MmaElementB = ElementB; //cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

  using TiledMma = TiledMMA<
      decltype(detail::get_ppu_mma_atom<MmaElementA, MmaElementB, ElementAccumulator>()),
      Layout<Shape<_2,_2,_1>>,  // 2x2x1 thread group
      Tile<_32, _32, _16>>; // 2x1x1 value group for 16x16x16 MMA and LDSM

#if ENABLE_AIU
  static constexpr int blockM = cute::get<0>(TileShape_MNK{});
  static constexpr int blockN = cute::get<1>(TileShape_MNK{});
  static constexpr int blockK = cute::get<2>(TileShape_MNK{});
  using DispatchPolicy = MainloopPPUAiuBatchArray<3>;
  static constexpr bool TransA = platform::is_same<typename TagToStrideA<GmemLayoutA>::tag, cutlass::layout::RowMajor>::value ? false : true;
  static constexpr bool TransB = platform::is_same<typename TagToStrideB<GmemLayoutB>::tag, cutlass::layout::ColumnMajor>::value ? false : true;
  using DefaultOperandA = detail::DefaultGemm_AIU_Operand<ElementA, TransA, Int<blockM>, Int<blockK>, true>;
  using DefaultOperandB = detail::DefaultGemm_AIU_Operand<ElementB, TransB, Int<blockN>, Int<blockK>, true>;
#else
  using DispatchPolicy = MainloopPPU0010CpAsync<3>;
  using DefaultOperandA = detail::DefaultGemm_TensorOpPPU0010_OperandA<
    ElementA, GmemLayoutA, AlignmentA, 32>;
  using DefaultOperandB = detail::DefaultGemm_TensorOpPPU0010_OperandB<
    ElementB, GmemLayoutB, AlignmentB, 32>;
#endif

  using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom; // M, K
  using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;

  // B
  using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom; // N, K
  using SmemCopyAtomB = typename DefaultOperandB::SmemCopyAtom;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

  // Mainloop
  using CollectiveOp = collective::CollectiveMma<
    DispatchPolicy, TileShape_MNK,
    MmaElementA, TagToStrideA_t<GmemLayoutA>,
    MmaElementA, TagToStrideB_t<GmemLayoutB>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

};

// AIU GEMM for StreamK
template <
  class ElementA,
  class GmemLayoutA,
  int AlignmentA,
  class ElementB,
  class GmemLayoutB,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::PPU0010,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutA,
    AlignmentA,
    ElementB,
    GmemLayoutB,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
      (cute::is_same_v<KernelScheduleType, KernelAiuMultistageStreamK>)>
> {

  // For fp32 types, map to tf32 MMA value type
  using MmaElementA = ElementA; //cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using MmaElementB = ElementB; //cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

  using TiledMma = TiledMMA<
      decltype(detail::get_ppu_mma_atom<MmaElementA, MmaElementB, ElementAccumulator>()),
      Layout<Shape<_2,_2,_1>>,  // 2x2x1 thread group
      Tile<_32, _32, _16>>; // 2x1x1 value group for 16x16x16 MMA and LDSM

#if ENABLE_AIU
  static constexpr int blockM = cute::get<0>(TileShape_MNK{});
  static constexpr int blockN = cute::get<1>(TileShape_MNK{});
  static constexpr int blockK = cute::get<2>(TileShape_MNK{});
  using DispatchPolicy = MainloopPPUAiu<3, KernelAiuMultistageStreamK>;
  static constexpr bool TransA = platform::is_same<GmemLayoutA, cutlass::layout::RowMajor>::value ? false : true;
  static constexpr bool TransB = platform::is_same<GmemLayoutB, cutlass::layout::ColumnMajor>::value ? false : true;
  using DefaultOperandA = detail::DefaultGemm_AIU_Operand<ElementA, TransA, Int<blockM>, Int<blockK>, true>;
  using DefaultOperandB = detail::DefaultGemm_AIU_Operand<ElementB, TransB, Int<blockN>, Int<blockK>, true>;
#else
  using DispatchPolicy = MainloopPPU0010CpAsync<3>;
  using DefaultOperandA = detail::DefaultGemm_TensorOpPPU0010_OperandA<
    ElementA, GmemLayoutA, AlignmentA, 32>;
  using DefaultOperandB = detail::DefaultGemm_TensorOpPPU0010_OperandB<
    ElementB, GmemLayoutB, AlignmentB, 32>;
#endif

  using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom; // M, K
  using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;

  // B
  using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom; // N, K
  using SmemCopyAtomB = typename DefaultOperandB::SmemCopyAtom;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

  // Mainloop
  using CollectiveOp = collective::CollectiveMma<
    DispatchPolicy, TileShape_MNK,
    MmaElementA, TagToStrideA_t<GmemLayoutA>,
    MmaElementA, TagToStrideB_t<GmemLayoutB>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

};



} // namespace cutlass::gemm::collective