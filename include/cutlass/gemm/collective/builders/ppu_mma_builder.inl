#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder_decl.hpp"

#include "cutlass/detail/collective.hpp"

#include "cutlass/gemm/collective/builders/tile_shape_infer.inl"
#include "cutlass/gemm/config/gemm_operands.hpp"

#define ENABLE_AIU 1

namespace cutlass::gemm::collective {

namespace ppu_detail {

constexpr int ppu10000_smem_capacity_bytes = 262144;
// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count.
template<int CapacityBytes, class ElementA, class ElementB, class TileShapeMNK, int stages>
constexpr int
compute_stage_count_or_override(StageCount<stages> stage_count) {
  return stages;
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count.
template<int CapacityBytes, class ElementA, class ElementB, class TileShapeMNK, int stages>
constexpr int
compute_stage_count_or_override(cute::Int<stages> stage_count) {
  return stages;
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count.
template<int CapacityBytes, class ElementA, class ElementB, class TileShapeMNK, int carveout_bytes>
constexpr int
compute_stage_count_or_override(StageCountAutoCarveout<carveout_bytes> stage_count) {
  static_assert(carveout_bytes < ppu10000_smem_capacity_bytes, "epilogue carved out shm size should be smaller than total shm size");
  constexpr auto a_bits = cute::sizeof_bits_v<ElementA>;
  constexpr auto b_bits = cute::sizeof_bits_v<ElementB>;
  constexpr int stage_bytes =
    cutlass::bits_to_bytes(a_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    cutlass::bits_to_bytes(b_bits * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{}));
  constexpr int compute_stages = (CapacityBytes - carveout_bytes) / stage_bytes;
  constexpr int out_stages = min(compute_stages, Int<5>{});
  return out_stages;
}
} // namespace ppu_detail
namespace detail {

///////////////////////////////////////////////////////////////////////////////
#if ENABLE_AIU
template <
  typename Element,
  bool Trans,
  typename Block_MN,
  typename Block_K,
  bool Swap
> struct MixGemm_AIU_Operand;

template <
  typename Element,
  typename Block_MN,
  typename Block_K,
  bool Swap
> struct MixGemm_AIU_Operand<
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
  typename Block_MN,
  typename Block_K,
  bool Swap
> struct MixGemm_AIU_Operand<
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

template <typename Arch,
          typename ElementA,
          typename ElementB,
          typename ElementAccumulator,
          typename TileShape_MNK,
          typename ClusterShape_MNK,
          typename PermutionK_ = void
          >
struct get_tiled_mma {
  using MmaInst = typename config::GetMmaInst<Arch, ElementA, ElementB, ElementAccumulator>::type;

  static constexpr int blockM = cute::get<0>(TileShape_MNK{});
  static constexpr int blockN = cute::get<1>(TileShape_MNK{});
  static constexpr int blockK = cute::get<2>(TileShape_MNK{});

  // User can configure custom warp tile shape through ClusterShape_MNK
  static constexpr bool CustomWarpShape = cute::get<0>(ClusterShape_MNK{}) != 1 ||
                                          cute::get<1>(ClusterShape_MNK{}) != 1 ||
                                          cute::get<2>(ClusterShape_MNK{}) != 1;
  static constexpr auto WarpShapeStage = MapBlockShapeToWarpShapeStage<Arch, ElementA, blockM, blockN, blockK>();

  static constexpr int InstM = cute::get<0>(typename MMA_Traits<MmaInst>::Shape_MNK{});
  static constexpr int InstN = cute::get<1>(typename MMA_Traits<MmaInst>::Shape_MNK{});
  static constexpr int InstK = cute::get<2>(typename MMA_Traits<MmaInst>::Shape_MNK{});

  static constexpr int warpM = max(Int<InstM>{}, CustomWarpShape ? cute::get<0>(ClusterShape_MNK{}) : cute::get<0>(WarpShapeStage));
  static constexpr int warpN = max(Int<InstN>{}, CustomWarpShape ? cute::get<1>(ClusterShape_MNK{}) : cute::get<1>(WarpShapeStage));

  using WarpOnM = Int<blockM / warpM>;
  using WarpOnN = Int<blockN / warpN>;

  using PermutionK = cute::conditional_t<cute::is_void_v<PermutionK_>, Int<InstK>, PermutionK_>;
  static_assert(PermutionK{} % InstK == 0, "PermutionK must be multiple of InstK.");

  using TiledMma = cute::conditional_t<cute::is_void_v<PermutionK_>,
                      TiledMMA<MMA_Atom<MmaInst>,
                              cute::Layout<Shape<WarpOnM, WarpOnN, _1>>>,
                      TiledMMA<MMA_Atom<MmaInst>,
                              cute::Layout<Shape<WarpOnM, WarpOnN, _1>>,
                              Tile<Int<blockM / warpM * InstM>, Int<blockN / warpN * InstN>, PermutionK>>>;
};

} // namespace detail


// AIU GEMM
template <
  typename Arch,
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
    Arch,
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
       cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedCooperative> ||
       cute::is_same_v<KernelScheduleType, KernelAiuMultistagePersistent>)>
> {
  // For fp32 types, map to tf32 MMA value type
  using MmaElementA = cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using MmaElementB = cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

  using TiledMma = typename detail::get_tiled_mma<Arch, MmaElementA, MmaElementB, ElementAccumulator, TileShape_MNK, ClusterShape_MNK>::TiledMma;
  using PPUKernelScheduleType = cute::conditional_t<cute::is_same_v<KernelScheduleType, KernelAiuMultistagePersistent>,
                                                    KernelAiuMultistagePersistent,
                                                    KernelAiuMultistage>;
  static constexpr int PipelineStages = ppu_detail::compute_stage_count_or_override<ppu_detail::ppu10000_smem_capacity_bytes,
      MmaElementA, MmaElementB, TileShape_MNK>(StageCountType{});
#if ENABLE_AIU
  static constexpr int blockM = cute::get<0>(TileShape_MNK{});
  static constexpr int blockN = cute::get<1>(TileShape_MNK{});
  static constexpr int blockK = cute::get<2>(TileShape_MNK{});
  using DispatchPolicy = MainloopPPUAiu<PipelineStages, PPUKernelScheduleType>;
  static constexpr bool TransA = platform::is_same<GmemLayoutA, cutlass::layout::RowMajor>::value ? false : true;
  static constexpr bool TransB = platform::is_same<GmemLayoutB, cutlass::layout::ColumnMajor>::value ? false : true;


  using DefaultOperandA = config::DefaultGemm_AIU_Operand<Arch, ElementA, TransA, Int<blockM>, Int<blockK>, false>;
  using DefaultOperandB = config::DefaultGemm_AIU_Operand<Arch, ElementB, TransB, Int<blockN>, Int<blockK>, true>;
#else
  using DispatchPolicy = MainloopPPUCpAsync<3>;
  using DefaultOperandA = detail::DefaultGemm_TensorOpPPU_OperandA<
    ElementA, GmemLayoutA, AlignmentA, 32>;
  using DefaultOperandB = detail::DefaultGemm_TensorOpPPU_OperandB<
    ElementB, GmemLayoutB, AlignmentB, 32>;
#endif

  using TransformA = typename platform::conditional<
    platform::is_same<ElementA, float>::value && platform::is_same<MmaElementA, cutlass::tfloat32_t>::value,
    cute::convert<cutlass::tfloat32_t>,
    cute::identity
  >::type;

  using TransformB = typename platform::conditional<
    platform::is_same<ElementB, float>::value && platform::is_same<MmaElementB, cutlass::tfloat32_t>::value,
    cute::convert<cutlass::tfloat32_t>,
    cute::identity
  >::type;

  using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom; // M, K
  using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;

  // B
  using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom; // N, K
  using SmemCopyAtomB = typename DefaultOperandB::SmemCopyAtom;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

  // Mainloop
  using CollectiveOp = collective::CollectiveMma<
    Arch, DispatchPolicy, TileShape_MNK,
    ElementA, TagToStrideA_t<GmemLayoutA>,
    ElementB, TagToStrideB_t<GmemLayoutB>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, TransformA,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, TransformB   // B
  >;

};

// AIU GEMM with scale (for a8w8 block-wise quant)
template <
  typename Arch,
  class ElementA,
  class GmemLayoutPairA,
  int AlignmentA,
  class ElementB,
  class GmemLayoutPairB,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    Arch,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutPairA,
    AlignmentA,
    ElementB,
    GmemLayoutPairB,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<(cute::is_same_v<KernelScheduleType, KernelAiuMultistageWithScale>
      || cute::is_same_v<KernelScheduleType, KernelAiuMultistageWithBlockWiseScale>)
  >
> {
  using GmemLayoutATag   = cute::remove_cvref_t<decltype(get<0>(GmemLayoutPairA{}))>;
  using GmemLayoutSFATag = cute::remove_cvref_t<decltype(get<1>(GmemLayoutPairA{}))>;
  using GmemLayoutBTag   = cute::remove_cvref_t<decltype(get<0>(GmemLayoutPairB{}))>;
  using GmemLayoutSFBTag = cute::remove_cvref_t<decltype(get<1>(GmemLayoutPairB{}))>;
  static_assert(cute::depth(cute::remove_pointer_t<GmemLayoutSFATag>{}) == 2 and
                cute::depth(cute::remove_pointer_t<GmemLayoutSFBTag>{}) == 2,
      "Expect SFA and SFB layout to be depth of two with shape ((SFVecMN, restMN),(SFVecK, restK), L)");
  static_assert(size<1,0>(cute::remove_pointer_t<GmemLayoutSFATag>{}) ==
                size<1,0>(cute::remove_pointer_t<GmemLayoutSFBTag>{}),
      "SFA and SFB must have equivalent SF vector sizes along K");
  static_assert(sizeof_bits<ElementA>::value == 8, "Only int8 or fp8 is supported for ElementA at KernelAiuMultistageWithScale.");
  static_assert(sizeof_bits<ElementB>::value == 8, "Only int8 or fp8 is supported for ElementB at KernelAiuMultistageWithScale.");

  static constexpr auto ScaleGranularityM = size<0,0>(cute::remove_pointer_t<GmemLayoutSFATag>{});
  static constexpr auto ScaleGranularityN = size<0,0>(cute::remove_pointer_t<GmemLayoutSFBTag>{});
  static constexpr auto ScaleGranularityK = size<1,0>(cute::remove_pointer_t<GmemLayoutSFATag>{});

  using ElementScale = float;

  using TiledMma = typename detail::get_tiled_mma<Arch, ElementA, ElementB, ElementAccumulator, TileShape_MNK, ClusterShape_MNK>::TiledMma;
  using PPUKernelScheduleType = cute::conditional_t<cute::is_same_v<KernelScheduleType, KernelAiuMultistagePersistent>,
                                                    KernelAiuMultistagePersistent,
                                                    KernelAiuMultistage>;
  static constexpr int PipelineStages = ppu_detail::compute_stage_count_or_override<ppu_detail::ppu10000_smem_capacity_bytes,
      ElementA, ElementB, TileShape_MNK>(StageCountType{});


#if ENABLE_AIU
  using DispatchPolicy = MainloopWithScalePPUAiu<PipelineStages, KernelScheduleType>;
  static constexpr int blockM = cute::get<0>(TileShape_MNK{});
  static constexpr int blockN = cute::get<1>(TileShape_MNK{});
  static constexpr int blockK = cute::get<2>(TileShape_MNK{});

  static constexpr int ScaleMsPerTile = cute::ceil_div(Int<blockM>{}, Int<ScaleGranularityM>{});
  static constexpr int ScaleNsPerTile = cute::ceil_div(Int<blockN>{}, Int<ScaleGranularityN>{});
  static constexpr int ScaleKsPerTile = cute::ceil_div(Int<blockK>{}, Int<ScaleGranularityK>{});

  static_assert(blockK > ScaleGranularityK ? (blockK % ScaleGranularityK) == 0 : (ScaleGranularityK % blockK) == 0,
              "block scaling granularity must evenly divide tile shape along K.");

  static constexpr bool TransA = platform::is_same<GmemLayoutATag, cutlass::layout::RowMajor>::value ? false : true;
  static constexpr bool TransB = platform::is_same<GmemLayoutBTag, cutlass::layout::ColumnMajor>::value ? false : true;

  static constexpr bool TransSFA = is_static<decltype(stride<1>(GmemLayoutSFATag{}))>::value ? false : true;
  static constexpr bool TransSFB = is_static<decltype(stride<1>(GmemLayoutSFBTag{}))>::value ? false : true;

  using DefaultOperandA = config::DefaultGemm_AIU_Operand<Arch, ElementA, TransA, Int<blockM>, Int<blockK>, false>;
  using DefaultOperandB = config::DefaultGemm_AIU_Operand<Arch, ElementB, TransB, Int<blockN>, Int<blockK>, true>;

  // Linear scale loads use the compact logical tile on both PPU generations.
  using DefaultOperandSFA = config::DefaultGemm_AIU_Operand<
      Arch, ElementScale, TransSFA, Int<ScaleMsPerTile>, Int<ScaleKsPerTile>, false, 1, false>;
  using DefaultOperandSFB = config::DefaultGemm_AIU_Operand<
      Arch, ElementScale, TransSFB, Int<ScaleNsPerTile>, Int<ScaleKsPerTile>, true, 1, false>;

#else
  using DispatchPolicy = MainloopPPUCpAsync<3>;
  using DefaultOperandA = detail::DefaultGemm_TensorOpPPU_OperandA<
    ElementA, GmemLayoutA, AlignmentA, 32>;
  using DefaultOperandB = detail::DefaultGemm_TensorOpPPU_OperandB<
    ElementB, GmemLayoutB, AlignmentB, 32>;
#endif

  using TransformA = cute::identity;
  using TransformB = cute::identity;

  // A
  using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom; // M, K
  using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;

  // B
  using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom; // N, K
  using SmemCopyAtomB = typename DefaultOperandB::SmemCopyAtom;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

  // scaleA
  using SmemLayoutAtomSFA = typename DefaultOperandSFA::SmemLayoutAtom; // M, K
  using GmemTiledCopySFA = typename DefaultOperandSFA::GmemTiledCopy;

  // scaleB
  using SmemLayoutAtomSFB = typename DefaultOperandSFB::SmemLayoutAtom; // N, K
  using GmemTiledCopySFB = typename DefaultOperandSFB::GmemTiledCopy;

  // Mainloop
  using CollectiveOp = collective::CollectiveMma<
    Arch, DispatchPolicy, TileShape_MNK,
    ElementA, cute::tuple<TagToStrideA_t<GmemLayoutATag>, TagToStrideA_t<GmemLayoutSFATag>>,
    ElementB, cute::tuple<TagToStrideB_t<GmemLayoutBTag>, TagToStrideB_t<GmemLayoutSFBTag>>,
    TiledMma,
    cute::tuple<GmemTiledCopyA, GmemTiledCopySFA>,
    cute::tuple<SmemLayoutAtomA, SmemLayoutAtomSFA>,
    SmemCopyAtomA, TransformA,     // A
    cute::tuple<GmemTiledCopyB, GmemTiledCopySFB>,
    cute::tuple<SmemLayoutAtomB, SmemLayoutAtomSFB>,
    SmemCopyAtomB, TransformB      // B
  >;
};

// AIU Mixed GEMM
template <
  typename Arch,
  class ElementPairA_,
  class GmemLayoutA_,
  int AlignmentA,
  class ElementPairB_,
  class GmemLayoutB_,
  int AlignmentB,
  class ElementAccumulator,
  class TileShapePair_,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    Arch,
    arch::OpClassTensorOp,
    ElementPairA_,
    GmemLayoutA_,
    AlignmentA,
    ElementPairB_,
    GmemLayoutB_,
    AlignmentB,
    ElementAccumulator,
    TileShapePair_,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
      (cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedMixedInput> ||
       cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedPingpongMixedInput> ||
       cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedCooperativeMixedInput> ||
       cute::is_same_v<KernelScheduleType, KernelAiuMultistageMixedInputPerCol> ||
       cute::is_same_v<KernelScheduleType, KernelAiuMultistageMixedInputFinegrainedGs128> ||
       cute::is_same_v<KernelScheduleType, KernelAiuMultistageMixedInputFinegrainedGs64>)>
> {
private:
  using ScaleA = detail::deduce_mixed_width_dtype_t<1, ElementPairA_>;
  using ScaleB = detail::deduce_mixed_width_dtype_t<1, ElementPairB_>;
  using ZeroA = detail::deduce_mixed_width_dtype_t<2, ElementPairA_>;
  using ZeroB = detail::deduce_mixed_width_dtype_t<2, ElementPairB_>;
  static constexpr bool NeitherIsTuple = !cute::is_tuple<ElementPairA_>::value && !cute::is_tuple<ElementPairB_>::value;

public:
  using TileShape_MNK = detail::deduce_mixed_width_dtype_t<0, TileShapePair_>;
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

  using ElementMma = cute::conditional_t<IsATransformed, ElementB, ElementA>;
  using RealInternalElementA = cute::conditional_t<IsATransformed, ElementB, ElementA>;
  using RealInternalElementB = cute::conditional_t<IsATransformed, ElementA, ElementB>;

  // currently only support a16w8 / a16w4 mix gemm
  // static_assert(IsATransformed, "currently only A is supported for quantization.");
  static_assert(sizeof_bits<RealInternalElementA>::value == 16 && (sizeof_bits<RealInternalElementB>::value == 8 || sizeof_bits<RealInternalElementB>::value == 4),
    "currently only support a16w8 / a16w4 mix gemm");
  // For fp32 types, map to tf32 MMA value type
  // using MmaElementA = ElementA; //cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  // using MmaElementB = ElementB; //cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

  using TiledMma = typename detail::get_tiled_mma<
        Arch, ElementMma, ElementMma, ElementAccumulator, TileShape_MNK, ClusterShape_MNK,
        Int<32 * 8 / sizeof_bits<RealInternalElementB>::value>>::TiledMma;

  static constexpr int PipelineStages = ppu_detail::compute_stage_count_or_override<ppu_detail::ppu10000_smem_capacity_bytes,
      ElementMma, ElementMma, TileShape_MNK>(StageCountType{});

  // currently only support k-major
  static_assert(cute::is_same_v<GmemLayoutA_, cutlass::layout::RowMajor> || cute::is_same_v<GmemLayoutA_, cutlass::layout::RowMajorInterleaved<256>>,
      "invalid GmemLayoutA, currently only support k-major or k256-major");
  static_assert(cute::is_same_v<GmemLayoutB_, cutlass::layout::ColumnMajor> || cute::is_same_v<GmemLayoutB_, cutlass::layout::ColumnMajorInterleaved<256>>,
      "invalid GmemLayoutB, currently only support k-major or k256-major");

  using kContinousA = cute::conditional_t<cute::is_same_v<GmemLayoutA_, cutlass::layout::RowMajorInterleaved<256>>, Int<256>, Int<1>>;
  using kContinousB = cute::conditional_t<cute::is_same_v<GmemLayoutB_, cutlass::layout::ColumnMajorInterleaved<256>>, Int<256>, Int<1>>;
  using kContinous = cute::conditional_t<IsATransformed, kContinousA, kContinousB>;
  using DispatchPolicy = MainloopPPUAiuMixedInput<PipelineStages, kContinous, KernelScheduleType>;

  using GmemLayoutA = cutlass::layout::RowMajor;
  using GmemLayoutB = cutlass::layout::ColumnMajor;

#if ENABLE_AIU
  static constexpr int blockM = cute::get<0>(TileShape_MNK{});
  static constexpr int blockN = cute::get<1>(TileShape_MNK{});
  static constexpr int blockK = cute::get<2>(TileShape_MNK{});

  using DefaultOperandA = detail::MixGemm_AIU_Operand<RealInternalElementA, false, Int<blockM>, Int<blockK>, true>;
  using DefaultOperandB = detail::MixGemm_AIU_Operand<RealInternalElementB, false, Int<blockN>, Int<blockK>, true>;
#elif 0 // async_cp not work now
  static_assert(false, "async_cp not work now");
  using DispatchPolicy = MainloopPPUAiuMixedInput<PipelineStages, kContinous, KernelScheduleType>;
  using DefaultOperandA = detail::DefaultGemm_TensorOpPPU_OperandA<
    RealInternalElementA, GmemLayoutA, cute::conditional_t<IsATransformed, AlignmentA, AlignmentB>, 32>;
  using DefaultOperandB = detail::DefaultGemm_TensorOpPPU_OperandB<
    RealInternalElementB, GmemLayoutB, cute::conditional_t<IsATransformed, AlignmentB, AlignmentA>, 32>;
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
    Arch, DispatchPolicy, TileShapePair_,
    ElementPairA, TagToStrideA_t<GmemLayoutA>,
    ElementPairB, TagToStrideB_t<GmemLayoutB>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

};

// AIU GEMM for Batch Array
template <
  typename Arch,
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
    Arch,
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
  using MmaElementA = cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using MmaElementB = cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;
  using TiledMma = typename detail::get_tiled_mma<Arch, MmaElementA, MmaElementB, ElementAccumulator, TileShape_MNK, ClusterShape_MNK>::TiledMma;
  using PPUKernelScheduleType = cute::conditional_t<cute::is_same_v<KernelScheduleType, KernelAiuMultistagePersistent>,
                                                    KernelAiuMultistagePersistent,
                                                    KernelAiuMultistage>;
  static constexpr int PipelineStages = ppu_detail::compute_stage_count_or_override<ppu_detail::ppu10000_smem_capacity_bytes,
      MmaElementA, MmaElementB, TileShape_MNK>(StageCountType{});

#if ENABLE_AIU
  static constexpr int blockM = cute::get<0>(TileShape_MNK{});
  static constexpr int blockN = cute::get<1>(TileShape_MNK{});
  static constexpr int blockK = cute::get<2>(TileShape_MNK{});
  using DispatchPolicy = MainloopPPUAiuBatchArray<PipelineStages>;
  static constexpr bool TransA = platform::is_same<typename TagToStrideA<GmemLayoutA>::tag, cutlass::layout::RowMajor>::value ? false : true;
  static constexpr bool TransB = platform::is_same<typename TagToStrideB<GmemLayoutB>::tag, cutlass::layout::ColumnMajor>::value ? false : true;
  using DefaultOperandA = config::DefaultGemm_AIU_Operand<Arch, ElementA, TransA, Int<blockM>, Int<blockK>, false>;
  using DefaultOperandB = config::DefaultGemm_AIU_Operand<Arch, ElementB, TransB, Int<blockN>, Int<blockK>, true>;
#else
  using DispatchPolicy = MainloopPPUCpAsync<PipelineStages>;
  using DefaultOperandA = detail::DefaultGemm_TensorOpPPU_OperandA<
    ElementA, GmemLayoutA, AlignmentA, 32>;
  using DefaultOperandB = detail::DefaultGemm_TensorOpPPU_OperandB<
    ElementB, GmemLayoutB, AlignmentB, 32>;
#endif

  using TransformA = typename platform::conditional<
    platform::is_same<ElementA, float>::value && platform::is_same<MmaElementA, cutlass::tfloat32_t>::value,
    cute::convert<cutlass::tfloat32_t>,
    cute::identity
  >::type;

  using TransformB = typename platform::conditional<
    platform::is_same<ElementB, float>::value && platform::is_same<MmaElementB, cutlass::tfloat32_t>::value,
    cute::convert<cutlass::tfloat32_t>,
    cute::identity
  >::type;

  using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom; // M, K
  using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;

  // B
  using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom; // N, K
  using SmemCopyAtomB = typename DefaultOperandB::SmemCopyAtom;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

  // Mainloop
  using CollectiveOp = collective::CollectiveMma<
    Arch, DispatchPolicy, TileShape_MNK,
    ElementA, TagToStrideA_t<GmemLayoutA>,
    ElementB, TagToStrideB_t<GmemLayoutB>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, TransformA,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, TransformB   // B
  >;

};

// AIU GEMM for StreamK
template <
  typename Arch,
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
    Arch,
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
  using ElementMma = cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;

  // TODO: 32x64x64 warp tile would make the barrier wait in ppu_tile_scheduler_stream_k.hpp hang
  //       but 64x64x64 warp tile is OK. This kind of bug already existed when Xiaohui first enabled Stream-K, would investigate later.
  #if 0
  using TiledMma = typename detail::get_tiled_mma<Arch, ElementMma, ElementMma, ElementAccumulator, TileShape_MNK, ClusterShape_MNK>::TiledMma;
  #else
  using MmaInst = typename config::GetMmaInst<Arch, ElementMma, ElementMma, ElementAccumulator>::type;
  using TiledMma = TiledMMA<
      MMA_Atom<MmaInst>,
      Layout<Shape<_2,_2,_1>>,  // 2x2x1 thread group
      Tile<_32, _32, _16>>; // 2x1x1 value group for 16x16x16 MMA and LDSM
  #endif

  using PPUKernelScheduleType = cute::conditional_t<cute::is_same_v<KernelScheduleType, KernelAiuMultistagePersistent>,
                                                    KernelAiuMultistagePersistent,
                                                    KernelAiuMultistage>;
  static constexpr int PipelineStages = ppu_detail::compute_stage_count_or_override<ppu_detail::ppu10000_smem_capacity_bytes,
      ElementMma, ElementMma, TileShape_MNK>(StageCountType{});

#if ENABLE_AIU
  static constexpr int blockM = cute::get<0>(TileShape_MNK{});
  static constexpr int blockN = cute::get<1>(TileShape_MNK{});
  static constexpr int blockK = cute::get<2>(TileShape_MNK{});
  using DispatchPolicy = MainloopPPUAiu<PipelineStages, KernelAiuMultistageStreamK>;
  static constexpr bool TransA = platform::is_same<GmemLayoutA, cutlass::layout::RowMajor>::value ? false : true;
  static constexpr bool TransB = platform::is_same<GmemLayoutB, cutlass::layout::ColumnMajor>::value ? false : true;
  using DefaultOperandA = config::DefaultGemm_AIU_Operand<Arch, ElementA, TransA, Int<blockM>, Int<blockK>, false>;
  using DefaultOperandB = config::DefaultGemm_AIU_Operand<Arch, ElementB, TransB, Int<blockN>, Int<blockK>, true>;
#else
  using DispatchPolicy = MainloopPPUCpAsync<PipelineStages>;
  using DefaultOperandA = detail::DefaultGemm_TensorOpPPU_OperandA<
    ElementA, GmemLayoutA, AlignmentA, 32>;
  using DefaultOperandB = detail::DefaultGemm_TensorOpPPU_OperandB<
    ElementB, GmemLayoutB, AlignmentB, 32>;
#endif

  using TransformA = typename platform::conditional<
    platform::is_same<ElementA, float>::value && platform::is_same<ElementMma, cutlass::tfloat32_t>::value,
    cute::convert<cutlass::tfloat32_t>,
    cute::identity
  >::type;

  using TransformB = typename platform::conditional<
    platform::is_same<ElementB, float>::value && platform::is_same<ElementMma, cutlass::tfloat32_t>::value,
    cute::convert<cutlass::tfloat32_t>,
    cute::identity
  >::type;

  using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom; // M, K
  using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;

  // B
  using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom; // N, K
  using SmemCopyAtomB = typename DefaultOperandB::SmemCopyAtom;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

  // Mainloop
  using CollectiveOp = collective::CollectiveMma<
    Arch, DispatchPolicy, TileShape_MNK,
    MmaElementA, TagToStrideA_t<GmemLayoutA>,
    MmaElementB, TagToStrideB_t<GmemLayoutB>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, TransformA,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, TransformB   // B
  >;

};

} // namespace cutlass::gemm::collective
