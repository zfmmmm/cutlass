#pragma once

#include <cstdint>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
#include <cute/atom/copy_traits_sm100_tma.hpp>
#include <cute/atom/copy_traits_sm80.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/gemm/collective/collective_builder_decl.hpp>
#include <cutlass/gemm/collective/collective_mma_decl.hpp>
#include <cutlass/gemm/collective/builders/sm100_common.inl>
#include <cutlass/gemm/collective/builders/sm100_simt_builder.inl>
#include <cutlass/gemm/collective/builders/sm120_common.inl>
#include <cutlass/gemm/collective/builders/sm90_common.inl>
#include <cutlass/gemm/gemm.h>
#include <cutlass/numeric_types.h>
#include <type_traits>

#include "../auto_partitioner.hpp"
#include "sm80_policy.hpp"

namespace autopartition {
namespace detail {

// ------------------------------ 类型路由 ------------------------------
// SM100 SIMT builder 在 CUTLASS 中专门服务 SGEMM，因此 SIMT 只开放 float。
// TensorOp 走 Blackwell UMMA/TMEM 路径，开放当前 sm100_make_trivial_tiled_mma
// 能稳定覆盖的输入类型。SM120 复用这组 Blackwell policy。
template <class Element> struct IsSm100SimtElement : std::is_same<Element, float>
{
};

template <class Element>
struct IsSm100TensorOpElement
    : std::integral_constant<bool,
                             std::is_same<Element, float>::value || std::is_same<Element, cutlass::half_t>::value
                                 || std::is_same<Element, cutlass::bfloat16_t>::value
                                 || std::is_same<Element, int8_t>::value || std::is_same<Element, uint8_t>::value>
{
};

template <class Element>
struct IsSm120TensorOpElement
    : std::integral_constant<bool,
                             std::is_same<Element, cutlass::float_e4m3_t>::value
                                 || std::is_same<Element, cutlass::float_e5m2_t>::value>
{
};

template <class Element> struct Sm100TensorOpAccumulator
{
    using type = float;
};

template <> struct Sm100TensorOpAccumulator<int8_t>
{
    using type = int32_t;
};

template <> struct Sm100TensorOpAccumulator<uint8_t>
{
    using type = int32_t;
};

template <class Element, int ContiguousElements>
struct Sm100GmemVectorAlignment : GmemVectorAlignment<Element, ContiguousElements>
{
};

template <class Element, class GmemStride, int TileMN, int TileK, int ThreadCount> struct Sm100SimtMainloopRole
{
    // SM100 SIMT SGEMM 的官方 builder 要求 TileK=16。这里复用官方布局策略：
    // MN-major shared load 可走 128-bit，K-major 用 8B padding/transpose 避免 bank conflict。
    static_assert(std::is_same<Element, float>::value, "SM100 SIMT policy currently targets SGEMM.");
    static_assert(TileK == 16, "SM100 SIMT SGEMM kernels require TileShape_K = 16.");

    using TileShape = cute::Shape<cute::Int<TileMN>, cute::_1, cute::Int<TileK>>;

    static constexpr bool IsMnMajor           = cutlass::gemm::detail::is_mn_major<GmemStride>();
    static constexpr int  SmemAlignmentOffset = IsMnMajor ? 0 : 2;
    static constexpr int  ContiguousDimLength = IsMnMajor ? TileMN : TileK;
    // K-major SIMT operand 会被写入带 8B padding 的 MN-major shared layout。
    // 目标 shared copy 是转置写入，不能安全承载连续 16B/8B cp.async 向量；
    // 因此 Gmem->Smem 搬运退到 32-bit。后续 Smem->Reg 仍按官方策略走
    // 64-bit，保证 bank-conflict-free 的读取模式。
    static constexpr int AlignmentElements =
        IsMnMajor ? Sm100GmemVectorAlignment<Element, ContiguousDimLength>::value : 1;
    using AlignmentType = cute::uint_byte_t<AlignmentElements *int(sizeof(Element))>;

    using SmemLayoutAtom = cute::Layout<cute::Shape<cute::Int<TileMN>, cute::Int<TileK>>,
                                        cute::Stride<cute::_1, cute::Int<TileMN + SmemAlignmentOffset>>>;
    using SmemLayout     = SmemLayoutAtom;

    using SmemToRegCopy =
        cute::conditional_t<IsMnMajor,
                            cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, Element>,
                            cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<64>, Element>>;
    using RegToSmemCopy = SmemToRegCopy;

    using GmemCopyAtom   = cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<AlignmentType>, Element>;
    using GmemToSmemCopy = decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<GmemCopyAtom,
                                                                                                 ThreadCount,
                                                                                                 AlignmentElements,
                                                                                                 GmemStride,
                                                                                                 cute::Int<TileMN>,
                                                                                                 cute::Int<TileK>>());
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
struct Sm100SimtRoleA
    : Sm100SimtMainloopRole<Element,
                            GmemStride,
                            cute::size<0>(TileShape_MNK{}),
                            cute::size<2>(TileShape_MNK{}),
                            ThreadCount>
{
};

template <class Element, class GmemStride, class TileShape_MNK, int ThreadCount>
struct Sm100SimtRoleB
    : Sm100SimtMainloopRole<Element,
                            GmemStride,
                            cute::size<1>(TileShape_MNK{}),
                            cute::size<2>(TileShape_MNK{}),
                            ThreadCount>
{
};

template <class Element, class GmemStride, class TileShape_MNK, int ThreadCount> struct Sm100SimtRoleC
{
    // RoleC 用官方 sm100_make_simt_f32_tiled_mma 生成 SIMT tiled_mma。
    // ThreadCount 必须等于官方 warp shape selector 推导出来的线程数。
    static_assert(std::is_same<Element, float>::value, "SM100 SIMT policy currently targets SGEMM.");
    static_assert(cute::size<2>(TileShape_MNK{}) == 16, "SM100 SIMT SGEMM kernels require TileShape_K = 16.");

    using WarpShape_MNK =
        decltype(cutlass::gemm::collective::detail::sm100_simt_f32_warp_shape_mnk_selector<TileShape_MNK>());
    static constexpr int OfficialThreadCount = cute::size(WarpShape_MNK{}) * cutlass::NumThreadsPerWarp;
    static_assert(ThreadCount == OfficialThreadCount,
                  "ThreadCount must match the SM100 SIMT warp-shape selector for this TileShape.");

    using TiledMma =
        decltype(cutlass::gemm::collective::detail::
                     sm100_make_simt_f32_tiled_mma<GmemStride, 1, GmemStride, 1, TileShape_MNK, WarpShape_MNK>());
    using MmaAtom = typename TiledMma::Atom;

    static constexpr int  BlkM      = cute::size<0>(TileShape_MNK{});
    static constexpr int  BlkN      = cute::size<1>(TileShape_MNK{});
    static constexpr bool IsMnMajor = cutlass::gemm::detail::is_mn_major<GmemStride>();
    static constexpr int  Padding   = SmemPaddingElements<Element>::value;

    using SmemLayoutAtom = cute::conditional_t<
        IsMnMajor,
        cute::Layout<cute::Shape<cute::Int<BlkM>, cute::Int<BlkN>>, cute::Stride<cute::_1, cute::Int<BlkM + Padding>>>,
        cute::Layout<cute::Shape<cute::Int<BlkM>, cute::Int<BlkN>>, cute::Stride<cute::Int<BlkN + Padding>, cute::_1>>>;
    using SmemLayout = SmemLayoutAtom;

    static constexpr int ContiguousDimLength = IsMnMajor ? BlkM : BlkN;
    static constexpr int AlignmentElements   = Sm100GmemVectorAlignment<Element, ContiguousDimLength>::value;
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

template <class Element, class GmemStride, class TileShape_MNK, bool IsRoleA> struct Sm100TensorOpMainloopRole;

template <class Element, class GmemStride, class TileShape_MNK>
struct Sm100TensorOpMainloopRole<Element, GmemStride, TileShape_MNK, true>
{
    // SM100/SM120 TensorOp 的 A/B operand 由 UMMA descriptor 从 shared memory 读取。
    // 这里生成 canonical UMMA shared layout 和 TMA copy opcode；真实 TMA descriptor、
    // mbarrier、TMEM allocator 由下游 mainloop 管理。
    using ElementMma =
        decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<Element>());
    using SmemAllocElement = cute::conditional_t<(cute::sizeof_bits_v<ElementMma> < 8), uint8_t, ElementMma>;

    static constexpr cute::UMMA::Major Major = cutlass::gemm::collective::detail::tag_to_umma_major_A<GmemStride>();
    static constexpr int               BlkM  = cute::size<0>(TileShape_MNK{});
    static constexpr int               BlkK  = cute::size<2>(TileShape_MNK{});

    using SmemLayoutAtom =
        decltype(cutlass::gemm::collective::detail::
                     sm100_smem_selector<Major, SmemAllocElement, cute::Int<BlkM>, cute::Int<BlkK>>());
    using SmemLayout = decltype(cute::tile_to_shape(SmemLayoutAtom{}, cute::Shape<cute::Int<BlkM>, cute::Int<BlkK>>{}));

    // TMA load/store 是 SM100 UMMA 的推荐全局搬运抽象。这里故意只暴露 atom/op，
    // 不创建 descriptor，也不绑定 pipeline stage。
    using GmemToSmemCopy     = cute::SM90_TMA_LOAD;
    using GlobalToSharedCopy = GmemToSmemCopy;

    using GmemToSmemCpAsyncCopy =
        decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
                 cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<cute::uint_byte_t<16>>, Element>,
                 128,
                 16 / int(sizeof(Element)),
                 GmemStride,
                 cute::Int<BlkM>,
                 cute::Int<BlkK>>());

    using SmemToRegCopy        = void;
    using SharedToRegisterCopy = SmemToRegCopy;
    using RegToSmemCopy        = void;
    using RegisterToSharedCopy = RegToSmemCopy;
    using SmemToGmemCopy       = cute::SM90_TMA_STORE;
    using SharedToGlobalCopy   = SmemToGmemCopy;
};

template <class Element, class GmemStride, class TileShape_MNK>
struct Sm100TensorOpMainloopRole<Element, GmemStride, TileShape_MNK, false>
{
    // B operand 与 A operand 对称，但 Major 的解释使用 tag_to_umma_major_B。
    // 这仍然只依赖当前输入张量的 stride，不读取 RoleA 的任何参数。
    using ElementMma =
        decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<Element>());
    using SmemAllocElement = cute::conditional_t<(cute::sizeof_bits_v<ElementMma> < 8), uint8_t, ElementMma>;

    static constexpr cute::UMMA::Major Major = cutlass::gemm::collective::detail::tag_to_umma_major_B<GmemStride>();
    static constexpr int               BlkN  = cute::size<1>(TileShape_MNK{});
    static constexpr int               BlkK  = cute::size<2>(TileShape_MNK{});

    using SmemLayoutAtom =
        decltype(cutlass::gemm::collective::detail::
                     sm100_smem_selector<Major, SmemAllocElement, cute::Int<BlkN>, cute::Int<BlkK>>());
    using SmemLayout = decltype(cute::tile_to_shape(SmemLayoutAtom{}, cute::Shape<cute::Int<BlkN>, cute::Int<BlkK>>{}));

    using GmemToSmemCopy     = cute::SM90_TMA_LOAD;
    using GlobalToSharedCopy = GmemToSmemCopy;

    using GmemToSmemCpAsyncCopy =
        decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
                 cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<cute::uint_byte_t<16>>, Element>,
                 128,
                 16 / int(sizeof(Element)),
                 GmemStride,
                 cute::Int<BlkN>,
                 cute::Int<BlkK>>());

    using SmemToRegCopy        = void;
    using SharedToRegisterCopy = SmemToRegCopy;
    using RegToSmemCopy        = void;
    using RegisterToSharedCopy = RegToSmemCopy;
    using SmemToGmemCopy       = cute::SM90_TMA_STORE;
    using SharedToGlobalCopy   = SmemToGmemCopy;
};

template <class Element, class GmemStride, class TileShape_MNK, int ThreadCount>
struct Sm100TensorOpRoleA : Sm100TensorOpMainloopRole<Element, GmemStride, TileShape_MNK, true>
{
};

template <class Element, class GmemStride, class TileShape_MNK, int ThreadCount>
struct Sm100TensorOpRoleB : Sm100TensorOpMainloopRole<Element, GmemStride, TileShape_MNK, false>
{
};

template <class Element, class GmemStride, class TileShape_MNK, int ThreadCount> struct Sm100TensorOpRoleC
{
    // RoleC 输出 UMMA tiled_mma。UMMA 的 accumulator 存在 Tensor Memory(TMEM)，
    // 因此 TiledMma 的 fragment type 会带有 cute::UMMA::tmem_frg_base 标记。
    // TiledMmaFor<MajorA, MajorB> 允许下游把 RoleA/RoleB 的 Major 显式传入，
    // 避免 AutoPartitioner 在内部把 A/B 参数耦合起来。
    using ElementMma =
        decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<Element>());
    using Accumulator      = typename Sm100TensorOpAccumulator<Element>::type;
    using ClusterShape_MNK = cute::Shape<cute::_1, cute::_1, cute::_1>;

    template <cute::UMMA::Major MajorA, cute::UMMA::Major MajorB>
    using TiledMmaFor = decltype(cutlass::gemm::collective::detail::sm100_make_trivial_tiled_mma<
                                 ElementMma,
                                 ElementMma,
                                 Accumulator,
                                 TileShape_MNK,
                                 ClusterShape_MNK,
                                 MajorA,
                                 MajorB,
                                 cutlass::gemm::collective::KernelScheduleAuto>());

    static constexpr cute::UMMA::Major DefaultMajorA =
        cutlass::gemm::collective::detail::tag_to_umma_major_A<GmemStride>();
    static constexpr cute::UMMA::Major DefaultMajorB =
        cutlass::gemm::collective::detail::tag_to_umma_major_B<GmemStride>();
    using TiledMma = TiledMmaFor<DefaultMajorA, DefaultMajorB>;
    using MmaAtom  = typename TiledMma::Atom;

    static constexpr int  BlkM      = cute::size<0>(TileShape_MNK{});
    static constexpr int  BlkN      = cute::size<1>(TileShape_MNK{});
    static constexpr bool IsMnMajor = cutlass::gemm::detail::is_mn_major<GmemStride>();
    static constexpr int  Padding   = SmemPaddingElements<Element>::value;

    using SmemLayoutAtom = cute::conditional_t<
        IsMnMajor,
        cute::Layout<cute::Shape<cute::Int<BlkM>, cute::Int<BlkN>>, cute::Stride<cute::_1, cute::Int<BlkM + Padding>>>,
        cute::Layout<cute::Shape<cute::Int<BlkM>, cute::Int<BlkN>>, cute::Stride<cute::Int<BlkN + Padding>, cute::_1>>>;
    using SmemLayout = SmemLayoutAtom;

    static constexpr int ContiguousDimLength = IsMnMajor ? BlkM : BlkN;
    static constexpr int AlignmentElements   = Sm100GmemVectorAlignment<Element, ContiguousDimLength>::value;
    using AlignmentType                      = cute::uint_byte_t<AlignmentElements *int(sizeof(Element))>;

    using GmemToSmemCopy = cute::SM90_TMA_LOAD;
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

template <class Element, class GmemStride, class TileShape_MNK, int ThreadCount, bool IsRoleA>
struct Sm120TensorOpMainloopRole
{
    // SM120 consumer Blackwell 当前公开的是 F8/F6/F4 MMA，而不是 SM100 的
    // TCGEN05/TMEM 半精度 UMMA。FP8 的 MMA 指令使用 uint8_t 原始寄存器载荷；
    // 因此 shared layout 用 uint8_t 分配类型，但模板路由仍以用户输入
    // Element(float_e4m3_t/float_e5m2_t) 为准。
    static_assert(IsSm120TensorOpElement<Element>::value, "SM120 TensorOp example path currently targets FP8 inputs.");

    using ElementMma       = decltype(cutlass::gemm::collective::detail::
                                          sm1xx_kernel_input_element_to_mma_input_element<Element>());
    using SmemAllocElement = uint8_t;

    static constexpr int BlkMN = IsRoleA ? cute::size<0>(TileShape_MNK{}) : cute::size<1>(TileShape_MNK{});
    static constexpr int BlkK  = cute::size<2>(TileShape_MNK{});

    using SmemLayoutAtom =
        decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<SmemAllocElement, cute::Int<BlkK>>());
    using SmemLayout = decltype(cute::tile_to_shape(SmemLayoutAtom{}, cute::Shape<cute::Int<BlkMN>, cute::Int<BlkK>>{}));

    using GmemToSmemCopy     = cute::AutoCopyAsync;
    using GlobalToSharedCopy = GmemToSmemCopy;

    using SmemToRegCopyOperation = cute::conditional_t<
        IsRoleA,
        decltype(cutlass::gemm::collective::detail::sm120_rr_smem_copy_selector_A<Element, Element, true>()),
        decltype(cutlass::gemm::collective::detail::sm120_rr_smem_copy_selector_B<Element, Element, true>())>;
    using SmemToRegCopy        = cute::Copy_Atom<SmemToRegCopyOperation, SmemAllocElement>;
    using SharedToRegisterCopy = SmemToRegCopy;

    using RegToSmemCopy        = cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, SmemAllocElement>;
    using RegisterToSharedCopy = RegToSmemCopy;
    using SmemToGmemCopy       = cute::AutoCopyAsync;
    using SharedToGlobalCopy   = SmemToGmemCopy;

    using SmemElement = SmemAllocElement;
};

template <class Element, class GmemStride, class TileShape_MNK, int ThreadCount>
struct Sm120TensorOpRoleA : Sm120TensorOpMainloopRole<Element, GmemStride, TileShape_MNK, ThreadCount, true>
{
};

template <class Element, class GmemStride, class TileShape_MNK, int ThreadCount>
struct Sm120TensorOpRoleB : Sm120TensorOpMainloopRole<Element, GmemStride, TileShape_MNK, ThreadCount, false>
{
};

template <class Element, class GmemStride, class TileShape_MNK, int ThreadCount> struct Sm120TensorOpRoleC
{
    // RoleC 生成 SM120 FP8 Tensor Core 的 TiledMma 和 FP32 epilogue shared
    // 布局。这里仍然只生成图纸：没有 stage 计算，没有 shared 分配，也没有
    // tile 坐标切分。
    static_assert(IsSm120TensorOpElement<Element>::value, "SM120 TensorOp path currently targets FP8 inputs.");

    using Accumulator     = float;
    using EpilogueElement = Accumulator;

    using PermTileM = decltype(cute::min(cute::size<0>(TileShape_MNK{}), cute::_128{}));
    using PermTileN = decltype(cute::min(cute::size<1>(TileShape_MNK{}), cute::_32{}));
    using MmaAtom    = cute::MMA_Atom<decltype(cute::rr_op_selector_sm120<Element, Element, Accumulator>())>;
    using AtomLayout = cute::Layout<cute::Shape<cute::_4, cute::_2, cute::_1>>;
    using TiledMma   = decltype(cute::make_tiled_mma(MmaAtom{}, AtomLayout{}, cute::Tile<PermTileM, PermTileN, cute::_32>{}));

    static constexpr int  BlkM      = cute::size<0>(TileShape_MNK{});
    static constexpr int  BlkN      = cute::size<1>(TileShape_MNK{});
    static constexpr bool IsMnMajor = cutlass::gemm::detail::is_mn_major<GmemStride>();
    static constexpr int  Padding   = SmemPaddingElements<EpilogueElement>::value;

    using SmemLayoutAtom = cute::conditional_t<
        IsMnMajor,
        cute::Layout<cute::Shape<cute::Int<BlkM>, cute::Int<BlkN>>, cute::Stride<cute::_1, cute::Int<BlkM + Padding>>>,
        cute::Layout<cute::Shape<cute::Int<BlkM>, cute::Int<BlkN>>, cute::Stride<cute::Int<BlkN + Padding>, cute::_1>>>;
    using SmemLayout = SmemLayoutAtom;

    static constexpr int ContiguousDimLength = IsMnMajor ? BlkM : BlkN;
    static constexpr int AlignmentElements   = Sm100GmemVectorAlignment<EpilogueElement, ContiguousDimLength>::value;

    using GmemToSmemCopy          = cute::AutoCopyAsync;
    using SmemToRegCopyOperation  = cute::AutoVectorizingCopyWithAssumedAlignment<128>;
    using RegToSmemCopyOperation  = cute::AutoVectorizingCopyWithAssumedAlignment<128>;
    using SmemToRegCopy           = cute::Copy_Atom<SmemToRegCopyOperation, EpilogueElement>;
    using RegToSmemCopy           = cute::Copy_Atom<RegToSmemCopyOperation, EpilogueElement>;
    using SmemToGmemCopy       = decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
                                    VectorizedCopyAtom<EpilogueElement, AlignmentElements>,
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

// ------------------------------ SFINAE 外部偏特化：SM100 ------------------------------
template <typename Element, typename GmemStride, typename TileShape_MNK, int ThreadCount>
struct AutoPartitioner<cutlass::arch::Sm100,
                       cutlass::arch::OpClassSimt,
                       Element,
                       GmemStride,
                       TileShape_MNK,
                       ThreadCount,
                       std::enable_if_t<detail::IsSm100SimtElement<Element>::value>>
{
    using RoleA = detail::Sm100SimtRoleA<Element, GmemStride, TileShape_MNK, ThreadCount>;
    using RoleB = detail::Sm100SimtRoleB<Element, GmemStride, TileShape_MNK, ThreadCount>;
    using RoleC = detail::Sm100SimtRoleC<Element, GmemStride, TileShape_MNK, ThreadCount>;
};

template <typename Element, typename GmemStride, typename TileShape_MNK, int ThreadCount>
struct AutoPartitioner<cutlass::arch::Sm100,
                       cutlass::arch::OpClassTensorOp,
                       Element,
                       GmemStride,
                       TileShape_MNK,
                       ThreadCount,
                       std::enable_if_t<detail::IsSm100TensorOpElement<Element>::value>>
{
    using RoleA = detail::Sm100TensorOpRoleA<Element, GmemStride, TileShape_MNK, ThreadCount>;
    using RoleB = detail::Sm100TensorOpRoleB<Element, GmemStride, TileShape_MNK, ThreadCount>;
    using RoleC = detail::Sm100TensorOpRoleC<Element, GmemStride, TileShape_MNK, ThreadCount>;
};

// ------------------------------ SFINAE 外部偏特化：SM120 ------------------------------
// RTX 50 系列 / Blackwell consumer 设备通常以 sm_120 编译。当前 CUTLASS 的
// SM120 仍然保留 SM100 UMMA/TMEM 语义，同时增加 FP4/FP8 等高阶路径。
// AutoPartitioner 这里先复用 SM100 的稳定蓝图，保证 ArchTag=Sm120 能参与路由；
// 后续如要支持 SM120 blockscaled/sparse，可继续加独立 sm120_policy.hpp。
template <typename Element, typename GmemStride, typename TileShape_MNK, int ThreadCount>
struct AutoPartitioner<cutlass::arch::Sm120,
                       cutlass::arch::OpClassSimt,
                       Element,
                       GmemStride,
                       TileShape_MNK,
                       ThreadCount,
                       std::enable_if_t<detail::IsSm100SimtElement<Element>::value>>
{
    // sm_120 当前未启用 SM100 f32x2 SIMT PTX 宏，因此实际可执行示例走
    // UniversalFMA SIMT 图纸；SM100 原生 policy 仍保留在 ArchTag=Sm100。
    using RoleA = detail::Sm80SimtRoleA<Element, GmemStride, TileShape_MNK, ThreadCount>;
    using RoleB = detail::Sm80SimtRoleB<Element, GmemStride, TileShape_MNK, ThreadCount>;
    using RoleC = detail::Sm80SimtRoleC<Element, GmemStride, TileShape_MNK, ThreadCount>;
};

template <typename Element, typename GmemStride, typename TileShape_MNK, int ThreadCount>
struct AutoPartitioner<cutlass::arch::Sm120,
                       cutlass::arch::OpClassTensorOp,
                       Element,
                       GmemStride,
                       TileShape_MNK,
                       ThreadCount,
                       std::enable_if_t<detail::IsSm100TensorOpElement<Element>::value>>
{
    using RoleA = detail::Sm100TensorOpRoleA<Element, GmemStride, TileShape_MNK, ThreadCount>;
    using RoleB = detail::Sm100TensorOpRoleB<Element, GmemStride, TileShape_MNK, ThreadCount>;
    using RoleC = detail::Sm100TensorOpRoleC<Element, GmemStride, TileShape_MNK, ThreadCount>;
};

template <typename Element, typename GmemStride, typename TileShape_MNK, int ThreadCount>
struct AutoPartitioner<cutlass::arch::Sm120,
                       cutlass::arch::OpClassTensorOp,
                       Element,
                       GmemStride,
                       TileShape_MNK,
                       ThreadCount,
                       std::enable_if_t<detail::IsSm120TensorOpElement<Element>::value>>
{
    using RoleA = detail::Sm120TensorOpRoleA<Element, GmemStride, TileShape_MNK, ThreadCount>;
    using RoleB = detail::Sm120TensorOpRoleB<Element, GmemStride, TileShape_MNK, ThreadCount>;
    using RoleC = detail::Sm120TensorOpRoleC<Element, GmemStride, TileShape_MNK, ThreadCount>;
};

} // namespace autopartition
