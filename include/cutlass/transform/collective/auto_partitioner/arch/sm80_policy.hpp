#pragma once

#include <cstdint>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits_sm75.hpp>
#include <cute/atom/copy_traits_sm80.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/gemm/collective/builders/sm90_common.inl>
#include <cutlass/gemm/gemm.h>
#include <cutlass/numeric_types.h>
#include <type_traits>

#include "../auto_partitioner.hpp"

namespace autopartition {
namespace detail {

// ------------------------------ 类型路由 ------------------------------
// SIMT 路径使用线程级 FMA，因此覆盖常见标量类型；TensorOp 路径只开放
// SM80 mma.sync / ldmatrix 能稳定覆盖的类型。新增类型时只需要扩展这些
// trait 和下面的 TensorOpTraits，不需要改 AutoPartitioner 主模板。
template <class Element>
struct IsSm80SimtElement
    : std::integral_constant<bool,
                             std::is_same<Element, float>::value || std::is_same<Element, double>::value
                                 || std::is_same<Element, cutlass::half_t>::value
                                 || std::is_same<Element, cutlass::bfloat16_t>::value>
{
};

template <class Element>
struct IsSm80TensorOpElement
    : std::integral_constant<bool,
                             std::is_same<Element, float>::value || std::is_same<Element, cutlass::tfloat32_t>::value
                                 || std::is_same<Element, cutlass::half_t>::value
                                 || std::is_same<Element, cutlass::bfloat16_t>::value
                                 || std::is_same<Element, int8_t>::value || std::is_same<Element, uint8_t>::value>
{
};

template <int TileM, int TileN, int ThreadCount> struct OptimalSimtThreadLayout
{
    // SIMT 的线程布局只决定每个 CTA 内线程如何铺到 M/N 方向。这里不推导
    // pipeline stage，也不假设 A/B 两个 operand 的形状关系。
    static_assert(ThreadCount == 64 || ThreadCount == 128 || ThreadCount == 256,
                  "SM80 SIMT supports 64, 128, or 256 CTA threads.");
    static constexpr int TM = (ThreadCount == 256) ? ((TileM > TileN) ? 32 : 16) : 16;
    static constexpr int TN = ThreadCount / TM;
    static_assert(ThreadCount % TM == 0, "Invalid SM80 SIMT thread layout.");
    using Layout = cute::Layout<cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::_1>>;
};

template <int TileM, int TileN, int ThreadCount> struct OptimalTensorOpThreadLayout
{
    // TensorOp 按 warp 数铺开。WarpM/WarpN 是“局部最优”的静态选择，
    // 让较长维度拿到更多 warp，同时保持只依赖 TileShape 和 ThreadCount。
    static_assert(ThreadCount % 32 == 0, "SM80 TensorOp requires whole warps.");
    static constexpr int WarpCount = ThreadCount / 32;
    static_assert(WarpCount == 1 || WarpCount == 2 || WarpCount == 4 || WarpCount == 8,
                  "SM80 TensorOp supports 1, 2, 4, or 8 warps.");

    static constexpr int WarpM = (WarpCount == 8) ? ((TileM >= TileN) ? 4 : 2)
                               : (WarpCount == 4) ? ((TileM >= TileN) ? 2 : 1)
                               : (WarpCount == 2) ? ((TileM >= TileN) ? 2 : 1)
                                                  : 1;
    static constexpr int WarpN = WarpCount / WarpM;

    using Layout = cute::Layout<cute::Shape<cute::Int<WarpM>, cute::Int<WarpN>, cute::_1>>;
};

template <class Element, int ContiguousElements> struct GmemVectorAlignment
{
    // cp.async 最喜欢 16B 对齐；当 tile 连续维度不满足 16B 向量化时，
    // 逐级退到 8B/4B。这个值只描述单个张量的全局内存访问粒度。
    static constexpr int ElementBytes = int(sizeof(Element));
    static constexpr int Align16      = (ElementBytes <= 16 && (16 % ElementBytes) == 0) ? (16 / ElementBytes) : 0;
    static constexpr int Align8       = (ElementBytes <= 8 && (8 % ElementBytes) == 0) ? (8 / ElementBytes) : 0;
    static constexpr int Align4       = (ElementBytes <= 4 && (4 % ElementBytes) == 0) ? (4 / ElementBytes) : 0;

    static constexpr int value = (Align16 != 0 && (ContiguousElements % Align16) == 0) ? Align16
                               : (Align8 != 0 && (ContiguousElements % Align8) == 0)   ? Align8
                               : (Align4 != 0 && (ContiguousElements % Align4) == 0)   ? Align4
                                                                                       : 0;

    static_assert(value != 0, "No legal cp.async vector width for this element type and contiguous tile extent.");
};

template <class Element> struct SmemPaddingElements
{
    // SIMT shared load 的主要风险是固定 stride 打到同一组 bank。
    // 对 SIMT 来说，简单 padding 比引入 swizzle 更稳：地址仍保持单调，
    // gmem->smem 的 tiled copy 也更容易保持向量化。TensorOp/UMMA 需要
    // ldmatrix/descriptor 的 canonical swizzle，则在专门路径里处理。
    static constexpr int value = (sizeof(Element) < 16) ? (16 / int(sizeof(Element))) : 1;
};

template <class Element, int AlignmentElements>
using VectorizedCopyAtom =
    cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<AlignmentElements *int(sizeof(Element)) * 8>,
                    Element>;

template <class Element, class GmemStride, int TileMN, int TileK, int ThreadCount> struct Sm80SimtMainloopRole
{
    // RoleA/RoleB 的共同 SIMT 主循环图纸。TileMN 对 A 表示 M，对 B 表示 N。
    // GmemStride 决定 MN-major 或 K-major，从而决定 shared layout 的 padding 方向。
    static constexpr bool IsMnMajor = cutlass::gemm::detail::is_mn_major<GmemStride>();
    static constexpr int  Padding   = SmemPaddingElements<Element>::value;

    using SmemLayoutAtom = cute::conditional_t<IsMnMajor,
                                               cute::Layout<cute::Shape<cute::Int<TileMN>, cute::Int<TileK>>,
                                                            cute::Stride<cute::_1, cute::Int<TileMN + Padding>>>,
                                               cute::Layout<cute::Shape<cute::Int<TileMN>, cute::Int<TileK>>,
                                                            cute::Stride<cute::Int<TileK + Padding>, cute::_1>>>;
    using SmemLayout     = SmemLayoutAtom;

    static constexpr int ContiguousDimLength = IsMnMajor ? TileMN : TileK;
    static constexpr int AlignmentElements   = GmemVectorAlignment<Element, ContiguousDimLength>::value;
    using AlignmentType                      = cute::uint_byte_t<AlignmentElements *int(sizeof(Element))>;

    using GmemCopyAtom   = cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<AlignmentType>, Element>;
    using GmemToSmemCopy = decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<GmemCopyAtom,
                                                                                                 ThreadCount,
                                                                                                 AlignmentElements,
                                                                                                 GmemStride,
                                                                                                 cute::Int<TileMN>,
                                                                                                 cute::Int<TileK>>());

    // SIMT 的寄存器搬运保持 DefaultCopy，让 CUTE 根据 partition 后的 tensor
    // 选择 LDS/STS 形态；全局写回则显式使用向量化 copy，便于生成 STG.E.128。
    using SmemToRegCopy  = cute::Copy_Atom<cute::DefaultCopy, Element>;
    using RegToSmemCopy  = cute::Copy_Atom<cute::DefaultCopy, Element>;
    using SmemToGmemCopy = decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
                                    VectorizedCopyAtom<Element, AlignmentElements>,
                                    ThreadCount,
                                    AlignmentElements,
                                    GmemStride,
                                    cute::Int<TileMN>,
                                    cute::Int<TileK>>());

    using GlobalToSharedCopy   = GmemToSmemCopy;
    using SharedToRegisterCopy = SmemToRegCopy;
    using RegisterToSharedCopy = RegToSmemCopy;
    using SharedToGlobalCopy   = SmemToGmemCopy;
};

template <class Element, class GmemStride, class TileShape_MNK, int ThreadCount>
struct Sm80SimtRoleA
    : Sm80SimtMainloopRole<Element,
                           GmemStride,
                           cute::size<0>(TileShape_MNK{}),
                           cute::size<2>(TileShape_MNK{}),
                           ThreadCount>
{
};

template <class Element, class GmemStride, class TileShape_MNK, int ThreadCount>
struct Sm80SimtRoleB
    : Sm80SimtMainloopRole<Element,
                           GmemStride,
                           cute::size<1>(TileShape_MNK{}),
                           cute::size<2>(TileShape_MNK{}),
                           ThreadCount>
{
};

template <class Element, class GmemStride, class TileShape_MNK, int ThreadCount> struct Sm80SimtRoleC
{
    static constexpr int BlkM = cute::size<0>(TileShape_MNK{});
    static constexpr int BlkN = cute::size<1>(TileShape_MNK{});

    using ThreadLayout = typename OptimalSimtThreadLayout<BlkM, BlkN, ThreadCount>::Layout;
    using MmaAtom      = cute::MMA_Atom<cute::UniversalFMA<Element, Element, Element>>;
    using TiledMma     = decltype(cute::make_tiled_mma(MmaAtom{}, ThreadLayout{}));

    static constexpr bool IsMnMajor = cutlass::gemm::detail::is_mn_major<GmemStride>();
    static constexpr int  Padding   = SmemPaddingElements<Element>::value;

    using SmemLayoutAtom = cute::conditional_t<
        IsMnMajor,
        cute::Layout<cute::Shape<cute::Int<BlkM>, cute::Int<BlkN>>, cute::Stride<cute::_1, cute::Int<BlkM + Padding>>>,
        cute::Layout<cute::Shape<cute::Int<BlkM>, cute::Int<BlkN>>, cute::Stride<cute::Int<BlkN + Padding>, cute::_1>>>;
    using SmemLayout = SmemLayoutAtom;

    static constexpr int ContiguousDimLength = IsMnMajor ? BlkM : BlkN;
    static constexpr int AlignmentElements   = GmemVectorAlignment<Element, ContiguousDimLength>::value;
    using AlignmentType                      = cute::uint_byte_t<AlignmentElements *int(sizeof(Element))>;

    using GmemToSmemCopy = decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
                                    cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<AlignmentType>, Element>,
                                    ThreadCount,
                                    AlignmentElements,
                                    GmemStride,
                                    cute::Int<BlkM>,
                                    cute::Int<BlkN>>());
    using RegToSmemCopy  = cute::Copy_Atom<cute::DefaultCopy, Element>;
    using SmemToRegCopy  = cute::Copy_Atom<cute::DefaultCopy, Element>;
    using SmemToGmemCopy = decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
                                    VectorizedCopyAtom<Element, AlignmentElements>,
                                    ThreadCount,
                                    AlignmentElements,
                                    GmemStride,
                                    cute::Int<BlkM>,
                                    cute::Int<BlkN>>());

    using GlobalToSharedCopy   = GmemToSmemCopy;
    using SharedToRegisterCopy = SmemToRegCopy;
    using RegisterToSharedCopy = RegToSmemCopy;
    using SharedToGlobalCopy   = SmemToGmemCopy;
};

template <class Element> struct Sm80TensorOpTraits;

// ------------------------------ SM80 TensorOp atom 表 ------------------------------
// 每个数据类型绑定一个 SM80 mma.sync atom、累加类型以及推荐 shared->register
// copy atom。RoleA/RoleB/RoleC 只引用 trait，不在偏特化里写类型分支。
template <> struct Sm80TensorOpTraits<cutlass::half_t>
{
    using MmaOperation = cute::SM80_16x8x16_F32F16F16F32_TN;
    using Accumulator  = float;
    using SmemCopyAtom = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, cutlass::half_t>;
};

template <> struct Sm80TensorOpTraits<cutlass::bfloat16_t>
{
    using MmaOperation = cute::SM80_16x8x16_F32BF16BF16F32_TN;
    using Accumulator  = float;
    using SmemCopyAtom = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, cutlass::bfloat16_t>;
};

template <> struct Sm80TensorOpTraits<cutlass::tfloat32_t>
{
    using MmaOperation = cute::SM80_16x8x8_F32TF32TF32F32_TN;
    using Accumulator  = float;
    using SmemCopyAtom = cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, cutlass::tfloat32_t>;
};

template <> struct Sm80TensorOpTraits<float> : Sm80TensorOpTraits<cutlass::tfloat32_t>
{
    using SmemCopyAtom = cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, float>;
};

template <> struct Sm80TensorOpTraits<int8_t>
{
    using MmaOperation = cute::SM80_16x8x32_S32S8S8S32_TN;
    using Accumulator  = int32_t;
    using SmemCopyAtom = cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, int8_t>;
};

template <> struct Sm80TensorOpTraits<uint8_t>
{
    using MmaOperation = cute::SM80_16x8x32_S32U8U8S32_TN;
    using Accumulator  = int32_t;
    using SmemCopyAtom = cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, uint8_t>;
};

template <int TileMN, int TileK, bool UseLdMatrix> struct Sm80TensorOpSmemLayoutSelector;

template <int TileMN, int TileK> struct Sm80TensorOpSmemLayoutSelector<TileMN, TileK, true>
{
    // half/bfloat16 TensorOp 使用 SM80 经典 128B swizzle 布局，目标是服务
    // ldmatrix。这个 layout 是“图纸”；真正的 shared memory 分配由下游 kernel 做。
    using SwizzleAtom =
        decltype(cute::composition(cute::Swizzle<3, 3, 3>{},
                                   cute::Layout<cute::Shape<cute::_8, cute::Shape<cute::_8, cute::_8>>,
                                                cute::Stride<cute::_8, cute::Stride<cute::_1, cute::_64>>>{}));
    using type = decltype(cute::tile_to_shape(SwizzleAtom{}, cute::Shape<cute::Int<TileMN>, cute::Int<TileK>>{}));
};

template <int TileMN, int TileK> struct Sm80TensorOpSmemLayoutSelector<TileMN, TileK, false>
{
    // TF32 / int8 等当前未走 ldmatrix swizzle 的类型使用 padding layout。
    // 这样仍然可以通过 cp.async/vectorized copy 跑通全局到 shared 的搬运。
    using type =
        cute::Layout<cute::Shape<cute::Int<TileMN>, cute::Int<TileK>>, cute::Stride<cute::_1, cute::Int<TileMN + 4>>>;
};

template <class Element, class GmemStride, int TileMN, int TileK, int ThreadCount> struct Sm80TensorOpMainloopRole
{
    // 是否启用 ldmatrix swizzle 完全由当前单个 operand 的 Element 和 tile 决定，
    // 不读取另一个 operand 的参数，避免 A/B 推导耦合。
    static constexpr bool UseLdMatrix =
        (std::is_same<Element, cutlass::half_t>::value || std::is_same<Element, cutlass::bfloat16_t>::value)
        && (TileMN % 8 == 0) && (TileK % 64 == 0);

    using SmemLayoutAtom = typename Sm80TensorOpSmemLayoutSelector<TileMN, TileK, UseLdMatrix>::type;
    using SmemLayout     = SmemLayoutAtom;

    static constexpr bool IsMnMajor           = cutlass::gemm::detail::is_mn_major<GmemStride>();
    static constexpr int  ContiguousDimLength = IsMnMajor ? TileMN : TileK;
    static constexpr int  AlignmentElements   = GmemVectorAlignment<Element, ContiguousDimLength>::value;
    using AlignmentType                       = cute::uint_byte_t<AlignmentElements *int(sizeof(Element))>;

    using GmemCopyAtom   = cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<AlignmentType>, Element>;
    using GmemToSmemCopy = decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<GmemCopyAtom,
                                                                                                 ThreadCount,
                                                                                                 AlignmentElements,
                                                                                                 GmemStride,
                                                                                                 cute::Int<TileMN>,
                                                                                                 cute::Int<TileK>>());

    // SmemToRegCopy 暴露 ldmatrix/向量化 copy atom。对于 swizzled layout，
    // 某些完整 kernel 会选择 cute::cooperative_copy 或自定义 TiledCopy 先填充 shared；
    // AutoPartitioner 只提供 atom 与 layout，不负责组织 pipeline。
    using SmemToRegCopy  = typename Sm80TensorOpTraits<Element>::SmemCopyAtom;
    using RegToSmemCopy  = cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, Element>;
    using SmemToGmemCopy = decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
                                    VectorizedCopyAtom<Element, AlignmentElements>,
                                    ThreadCount,
                                    AlignmentElements,
                                    GmemStride,
                                    cute::Int<TileMN>,
                                    cute::Int<TileK>>());

    using GlobalToSharedCopy   = GmemToSmemCopy;
    using SharedToRegisterCopy = SmemToRegCopy;
    using RegisterToSharedCopy = RegToSmemCopy;
    using SharedToGlobalCopy   = SmemToGmemCopy;
};

template <class Element, class GmemStride, class TileShape_MNK, int ThreadCount>
struct Sm80TensorOpRoleA
    : Sm80TensorOpMainloopRole<Element,
                               GmemStride,
                               cute::size<0>(TileShape_MNK{}),
                               cute::size<2>(TileShape_MNK{}),
                               ThreadCount>
{
};

template <class Element, class GmemStride, class TileShape_MNK, int ThreadCount>
struct Sm80TensorOpRoleB
    : Sm80TensorOpMainloopRole<Element,
                               GmemStride,
                               cute::size<1>(TileShape_MNK{}),
                               cute::size<2>(TileShape_MNK{}),
                               ThreadCount>
{
};

template <class Element, class GmemStride, class TileShape_MNK, int ThreadCount> struct Sm80TensorOpRoleC
{
    // RoleC 负责给 epilogue/accumulator 提供 MMA atom、C shared layout 与写回 copy。
    // 对 half/bfloat16，Accumulator 是 float；输出是否 cast 回 Element 由下游 epilogue 决定。
    static constexpr int BlkM = cute::size<0>(TileShape_MNK{});
    static constexpr int BlkN = cute::size<1>(TileShape_MNK{});

    using ThreadLayout = typename OptimalTensorOpThreadLayout<BlkM, BlkN, ThreadCount>::Layout;
    using MmaAtom      = cute::MMA_Atom<typename Sm80TensorOpTraits<Element>::MmaOperation>;
    using TiledMma     = decltype(cute::make_tiled_mma(MmaAtom{}, ThreadLayout{}));
    using Accumulator  = typename Sm80TensorOpTraits<Element>::Accumulator;

    static constexpr bool IsMnMajor = cutlass::gemm::detail::is_mn_major<GmemStride>();
    static constexpr int  Padding   = SmemPaddingElements<Element>::value;
    using SmemLayoutAtom            = cute::conditional_t<
                   IsMnMajor,
                   cute::Layout<cute::Shape<cute::Int<BlkM>, cute::Int<BlkN>>, cute::Stride<cute::_1, cute::Int<BlkM + Padding>>>,
                   cute::Layout<cute::Shape<cute::Int<BlkM>, cute::Int<BlkN>>, cute::Stride<cute::Int<BlkN + Padding>, cute::_1>>>;
    using SmemLayout = SmemLayoutAtom;

    static constexpr int ContiguousDimLength = IsMnMajor ? BlkM : BlkN;
    static constexpr int AlignmentElements   = GmemVectorAlignment<Element, ContiguousDimLength>::value;
    using AlignmentType                      = cute::uint_byte_t<AlignmentElements *int(sizeof(Element))>;

    using GmemToSmemCopy = decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
                                    cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<AlignmentType>, Element>,
                                    ThreadCount,
                                    AlignmentElements,
                                    GmemStride,
                                    cute::Int<BlkM>,
                                    cute::Int<BlkN>>());
    using SmemToRegCopy  = cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, Element>;
    using RegToSmemCopy  = cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, Element>;
    using SmemToGmemCopy = decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
                                    VectorizedCopyAtom<Element, AlignmentElements>,
                                    ThreadCount,
                                    AlignmentElements,
                                    GmemStride,
                                    cute::Int<BlkM>,
                                    cute::Int<BlkN>>());

    using GlobalToSharedCopy   = GmemToSmemCopy;
    using SharedToRegisterCopy = SmemToRegCopy;
    using RegisterToSharedCopy = RegToSmemCopy;
    using SharedToGlobalCopy   = SmemToGmemCopy;
};

} // namespace detail

// ------------------------------ SFINAE 外部偏特化 ------------------------------
// 注意这里没有嵌套 dispatcher，也没有在一个大类里写 if constexpr。
// ArchTag + OpClass + Element 共同决定是否匹配该偏特化；不匹配时回到空壳主模板。
template <typename Element, typename GmemStride, typename TileShape_MNK, int ThreadCount>
struct AutoPartitioner<cutlass::arch::Sm80,
                       cutlass::arch::OpClassSimt,
                       Element,
                       GmemStride,
                       TileShape_MNK,
                       ThreadCount,
                       std::enable_if_t<detail::IsSm80SimtElement<Element>::value>>
{
    using RoleA = detail::Sm80SimtRoleA<Element, GmemStride, TileShape_MNK, ThreadCount>;
    using RoleB = detail::Sm80SimtRoleB<Element, GmemStride, TileShape_MNK, ThreadCount>;
    using RoleC = detail::Sm80SimtRoleC<Element, GmemStride, TileShape_MNK, ThreadCount>;
};

template <typename Element, typename GmemStride, typename TileShape_MNK, int ThreadCount>
struct AutoPartitioner<cutlass::arch::Sm80,
                       cutlass::arch::OpClassTensorOp,
                       Element,
                       GmemStride,
                       TileShape_MNK,
                       ThreadCount,
                       std::enable_if_t<detail::IsSm80TensorOpElement<Element>::value>>
{
    using RoleA = detail::Sm80TensorOpRoleA<Element, GmemStride, TileShape_MNK, ThreadCount>;
    using RoleB = detail::Sm80TensorOpRoleB<Element, GmemStride, TileShape_MNK, ThreadCount>;
    using RoleC = detail::Sm80TensorOpRoleC<Element, GmemStride, TileShape_MNK, ThreadCount>;
};

} // namespace autopartition
