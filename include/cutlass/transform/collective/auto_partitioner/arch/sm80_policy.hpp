#pragma once

// =================================================================================================
// [AutoPartitioner SM80 Policy: 核心硬件抽象层与泛型路由]
//
// 本文件基于 C++17 模板元编程 (Template Metaprogramming) 与 SFINAE 机制，
// 为 SM80 (Ampere) 架构下的 SIMT (CUDA Cores) 与 TensorOp (Tensor Cores) 提供了
// 高度泛化且物理内存安全的 Layout、TiledCopy 与 TiledMMA 生成器。
//
// 核心演进特性：
// 1. 物理对齐解耦：彻底摒弃基于逻辑 Tile 推导显存访问位宽的危险假设，引入显式 `GmemAlignmentBytes` 兜底。
// 2. 角色感知 LDSM (Role-Aware)：基于 MMA 指令特性的 XOR 异或逻辑，精准决策 LDSM_N 与 LDSM_T。
// 3. 动态 Swizzle 行宽：根据 TileK 的动态字节数，自适应分配 128B/64B/32B Swizzle 宏，支持极细粒度流水线。
// 4. 计算与存储类型隔离：在 RoleC Epilogue 阶段隔离 ElementCompute 与 ElementOutput，防止类型坍缩。
// =================================================================================================

#include <cstdint>
#include <cute/arch/copy.hpp>
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

// =================================================================================================
// 第一部分：编译期类型与硬件能力侦测 (Type & Architecture Routing)
// =================================================================================================

/// @brief 侦测当前数据类型是否受 SM80 传统标量/向量计算单元 (SIMT FMA) 支持。
template <class Element>
struct IsSm80SimtElement
    : std::integral_constant<bool,
                             std::is_same<Element, float>::value || std::is_same<Element, double>::value
                                 || std::is_same<Element, cutlass::half_t>::value
                                 || std::is_same<Element, cutlass::bfloat16_t>::value>
{
};

/// @brief 侦测当前数据类型是否受 SM80 硬件矩阵乘加引擎 (Tensor Cores mma.sync) 支持。
/// @note SM80 架构引入了 DMMA (FP64)，此处将其纳入泛型路由体系，以支持高性能科学计算。
template <class Element>
struct IsSm80TensorOpElement
    : std::integral_constant<bool,
                             std::is_same<Element, double>::value || std::is_same<Element, float>::value
                                 || std::is_same<Element, cutlass::tfloat32_t>::value
                                 || std::is_same<Element, cutlass::half_t>::value
                                 || std::is_same<Element, cutlass::bfloat16_t>::value
                                 || std::is_same<Element, int8_t>::value || std::is_same<Element, uint8_t>::value>
{
};

// =================================================================================================
// 第二部分：线程束拓扑排布策略 (Thread Topology & Layout Policies)
// =================================================================================================

/// @brief 针对 SIMT 计算的二维线程网格 (Grid) 启发式分配策略。
/// @details 通过解构 M 和 N 维度的长短关系，将一维的 ThreadCount (如 256) 映射为二维形状。
/// 目标是使线程块读取 Global Memory 时能够最大化实现内存合并 (Memory Coalescing)。
template <int TileM, int TileN, int ThreadCount> struct OptimalSimtThreadLayout
{
    static_assert(ThreadCount == 64 || ThreadCount == 128 || ThreadCount == 256,
                  "SM80 SIMT supports 64, 128, or 256 CTA threads.");

    // 当处理的长边为 M 时，赋予 M 维度更多的线程组
    static constexpr int TM = (ThreadCount == 256) ? ((TileM > TileN) ? 32 : 16) : 16;
    static constexpr int TN = ThreadCount / TM;

    static_assert(ThreadCount % TM == 0, "Invalid SM80 SIMT thread layout.");

    using Layout = cute::Layout<cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::_1>>;
};

/// @brief 针对 Tensor Core MMA 计算的 Warp 级别二维排布策略。
/// @details Tensor Core 的基本调度单位是 Warp (32 线程)。此处将 ThreadCount 折算为 WarpCount (1,2,4,8)，
/// 并倾向于将更多的 Warp 铺设在 Tile 较长的维度上。这种“偏置”策略能够最大化 Shared Memory 的数据复用率，
/// 显著降低 Global 到 Shared 的访存带宽压力。
template <int TileM, int TileN, int ThreadCount> struct OptimalTensorOpThreadLayout
{
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

// =================================================================================================
// 第三部分：物理对齐与向量化访存引擎 (Physical Alignment & Vectorization Engine)
// =================================================================================================

/// @brief 纯线性连续内存的向量化推导引擎。
/// @details 结合逻辑连续长度 (`ContiguousElements`) 与外部承诺的物理绝对对齐边界 (`MaxAlignmentBytes`)，
/// 推导出安全且最大化的 PTX 访存位宽 (128-bit/64-bit/32-bit)。这从根本上杜绝了 Misaligned Address 硬件异常。
template <class Element, int ContiguousElements, int MaxAlignmentBytes = 16> struct GmemVectorAlignment
{
    static_assert(MaxAlignmentBytes == 4 || MaxAlignmentBytes == 8 || MaxAlignmentBytes == 16,
                  "Gmem alignment must be one of 4, 8, or 16 bytes.");

    static constexpr int ElementBytes = int(sizeof(Element));

    // 计算在当前元素大小下，要填满 16B/8B/4B 分别需要多少个元素
    static constexpr int Align16 =
        (MaxAlignmentBytes >= 16 && ElementBytes <= 16 && (16 % ElementBytes) == 0) ? (16 / ElementBytes) : 0;
    static constexpr int Align8 =
        (MaxAlignmentBytes >= 8 && ElementBytes <= 8 && (8 % ElementBytes) == 0) ? (8 / ElementBytes) : 0;
    static constexpr int Align4 =
        (MaxAlignmentBytes >= 4 && ElementBytes <= 4 && (4 % ElementBytes) == 0) ? (4 / ElementBytes) : 0;

    // 贪心降级策略：取能够同时被物理边界和逻辑连续长度整除的最大值
    static constexpr int value = [] {
        if constexpr (Align16 != 0) {
            if constexpr ((ContiguousElements % Align16) == 0) {
                return Align16;
            }
        }
        if constexpr (Align8 != 0) {
            if constexpr ((ContiguousElements % Align8) == 0) {
                return Align8;
            }
        }
        if constexpr (Align4 != 0) {
            if constexpr ((ContiguousElements % Align4) == 0) {
                return Align4;
            }
        }
        return 0;
    }();
    static constexpr int bytes = value * ElementBytes;

    static_assert(value != 0,
                  "No legal cp.async vector width for this element type, tile extent, and physical alignment.");
};

/// @brief TiledCopy 线程映射合法性检验器。
/// @details 防止在生成 `make_simt_gmem_tiled_copy` 时，分配给单线程的向量化分块跨越了逻辑 Tile 的物理边界。
template <int AlignmentElements, int ThreadCount, int TileMN, int TileK, bool IsMnMajor>
struct IsLegalSimtGmemTiledCopyAlignment
{
    static constexpr bool value = [] {
        if constexpr (AlignmentElements == 0) {
            return false;
        }
        else {
            constexpr int MajorExtent = IsMnMajor ? TileMN : TileK;
            constexpr int MinorExtent = IsMnMajor ? TileK : TileMN;
            constexpr int MajorThreads =
                (ThreadCount >= MajorExtent / AlignmentElements) ? (MajorExtent / AlignmentElements) : ThreadCount;
            if constexpr (MajorThreads <= 0) {
                return false;
            }
            else if constexpr ((ThreadCount % MajorThreads) != 0) {
                return false; // 无法均匀切分线程
            }
            else {
                constexpr int MinorThreads = ThreadCount / MajorThreads;
                return (MinorThreads == 0) || ((MinorExtent % MinorThreads) == 0);
            }
        }
    }();
};

/// @brief 基于 TiledCopy 拓扑的二维张量安全向量化引擎。
/// @details 这是 Gmem 到 Smem 搬运的最终权威推断器。它不仅考虑单行的对齐约束，还全面融合了线程块维度的切分合法性。
template <class Element, int TileMN, int TileK, int ThreadCount, bool IsMnMajor, int MaxAlignmentBytes = 16>
struct GmemTiledCopyAlignment
{
    static_assert(MaxAlignmentBytes == 4 || MaxAlignmentBytes == 8 || MaxAlignmentBytes == 16,
                  "Gmem alignment must be one of 4, 8, or 16 bytes.");

    static constexpr int ElementBytes       = int(sizeof(Element));
    static constexpr int ContiguousElements = IsMnMajor ? TileMN : TileK;
    static constexpr int Align16 =
        (MaxAlignmentBytes >= 16 && ElementBytes <= 16 && (16 % ElementBytes) == 0) ? (16 / ElementBytes) : 0;
    static constexpr int Align8 =
        (MaxAlignmentBytes >= 8 && ElementBytes <= 8 && (8 % ElementBytes) == 0) ? (8 / ElementBytes) : 0;
    static constexpr int Align4 =
        (MaxAlignmentBytes >= 4 && ElementBytes <= 4 && (4 % ElementBytes) == 0) ? (4 / ElementBytes) : 0;

    static constexpr int value = [] {
        if constexpr (Align16 != 0) {
            if constexpr ((ContiguousElements % Align16) == 0
                          && IsLegalSimtGmemTiledCopyAlignment<Align16, ThreadCount, TileMN, TileK, IsMnMajor>::value) {
                return Align16;
            }
        }
        if constexpr (Align8 != 0) {
            if constexpr ((ContiguousElements % Align8) == 0
                          && IsLegalSimtGmemTiledCopyAlignment<Align8, ThreadCount, TileMN, TileK, IsMnMajor>::value) {
                return Align8;
            }
        }
        if constexpr (Align4 != 0) {
            if constexpr ((ContiguousElements % Align4) == 0
                          && IsLegalSimtGmemTiledCopyAlignment<Align4, ThreadCount, TileMN, TileK, IsMnMajor>::value) {
                return Align4;
            }
        }
        return 0;
    }();
    static constexpr int bytes = value * ElementBytes;

    static_assert(value != 0,
                  "No legal gmem tiled-copy vector width for this element type, tile extent, thread layout, and "
                  "physical alignment.");
};

/// @brief Shared Memory 银行冲突 (Bank Conflict) 防御引擎。
/// @details 共享内存拥有 32 个 Banks，单 Bank 位宽为 4 Bytes (总跨度 128 Bytes)。
/// 若连续维度未开启 Swizzle 且跨度恰为 128 Bytes 倍数，列向访问将产生极其严重的硬件串行化排队。
/// 此处通过强制插入 padding，人为打破对齐周期，使得逻辑上同一列的元素物理上错落在不同的 Bank 中。
template <class Element> struct SmemPaddingElements
{
    static constexpr int value = (sizeof(Element) < 16) ? (16 / int(sizeof(Element))) : 1;
};

template <class Element, int AlignmentElements>
using VectorizedCopyAtom =
    cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<AlignmentElements *int(sizeof(Element)) * 8>,
                    Element>;

// =================================================================================================
// 第四部分：SIMT (CUDA Core) 流水线角色构建协议
// =================================================================================================

/// @brief 抽象 SIMT 计算场景下矩阵 A 与 B 的通用 Mainloop 行为。
template <class Element, class GmemStride, int TileMN, int TileK, int ThreadCount, int GmemAlignmentBytes>
struct Sm80SimtMainloopRole
{
    static constexpr bool IsMnMajor = cutlass::gemm::detail::is_mn_major<GmemStride>();
    static constexpr int  Padding   = SmemPaddingElements<Element>::value;

    // 静态构建 Shared Memory Layout。通过在非连续维度的 Stride 上施加 Padding 实现 Bank 错位。
    using SmemLayoutAtom = cute::conditional_t<IsMnMajor,
                                               cute::Layout<cute::Shape<cute::Int<TileMN>, cute::Int<TileK>>,
                                                            cute::Stride<cute::_1, cute::Int<TileMN + Padding>>>,
                                               cute::Layout<cute::Shape<cute::Int<TileMN>, cute::Int<TileK>>,
                                                            cute::Stride<cute::Int<TileK + Padding>, cute::_1>>>;
    using SmemLayout     = SmemLayoutAtom;

    static constexpr int ContiguousDimLength = IsMnMajor ? TileMN : TileK;

    // 获取受物理对齐和 Tile 约束保护的最终向量化位宽
    static constexpr int VectorAlignmentElements =
        GmemTiledCopyAlignment<Element, TileMN, TileK, ThreadCount, IsMnMajor, GmemAlignmentBytes>::value;
    static constexpr int VectorAlignmentBytes =
        GmemTiledCopyAlignment<Element, TileMN, TileK, ThreadCount, IsMnMajor, GmemAlignmentBytes>::bytes;

    static constexpr int GmemToSmemAlignmentElements = VectorAlignmentElements;
    static constexpr int GmemToSmemAlignmentBytes    = VectorAlignmentBytes;
    using AlignmentType = cute::uint_byte_t<GmemToSmemAlignmentElements *int(sizeof(Element))>;

    // 绑定 SM80 专属的异步拷贝硬件指令 (cp.async)
    using GmemCopyAtom = cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<AlignmentType>, Element>;

    // 构建线程协同的异步拷贝分发器
    using TiledGmemToSmemCopy =
        decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<GmemCopyAtom,
                                                                              ThreadCount,
                                                                              GmemToSmemAlignmentElements,
                                                                              GmemStride,
                                                                              cute::Int<TileMN>,
                                                                              cute::Int<TileK>>());
    using GmemToSmemCopy = TiledGmemToSmemCopy;

    using SmemToRegCopy  = cute::Copy_Atom<cute::DefaultCopy, Element>;
    using RegToSmemCopy  = cute::Copy_Atom<cute::DefaultCopy, Element>;
    using SmemToGmemCopy = decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
                                    VectorizedCopyAtom<Element, VectorAlignmentElements>,
                                    ThreadCount,
                                    VectorAlignmentElements,
                                    GmemStride,
                                    cute::Int<TileMN>,
                                    cute::Int<TileK>>());

    using GlobalToSharedCopy   = GmemToSmemCopy;
    using SharedToRegisterCopy = SmemToRegCopy;
    using RegisterToSharedCopy = RegToSmemCopy;
    using SharedToGlobalCopy   = SmemToGmemCopy;
};

template <class Element, class GmemStride, class TileShape_MNK, int ThreadCount, int GmemAlignmentBytes>
struct Sm80SimtRoleA
    : Sm80SimtMainloopRole<Element,
                           GmemStride,
                           cute::size<0>(TileShape_MNK{}),
                           cute::size<2>(TileShape_MNK{}),
                           ThreadCount,
                           GmemAlignmentBytes>
{
};

template <class Element, class GmemStride, class TileShape_MNK, int ThreadCount, int GmemAlignmentBytes>
struct Sm80SimtRoleB
    : Sm80SimtMainloopRole<Element,
                           GmemStride,
                           cute::size<1>(TileShape_MNK{}),
                           cute::size<2>(TileShape_MNK{}),
                           ThreadCount,
                           GmemAlignmentBytes>
{
};

/// @brief 抽象 SIMT 计算场景下输出矩阵 C 的 Epilogue 行为。
template <class Element, class ElementC, class GmemStride, class TileShape_MNK, int ThreadCount, int GmemAlignmentBytes>
struct Sm80SimtRoleC
{
    static constexpr int BlkM = cute::size<0>(TileShape_MNK{});
    static constexpr int BlkN = cute::size<1>(TileShape_MNK{});

    using ThreadLayout = typename OptimalSimtThreadLayout<BlkM, BlkN, ThreadCount>::Layout;
    using MmaAtom      = cute::MMA_Atom<cute::UniversalFMA<Element, Element, Element>>;
    using TiledMma     = decltype(cute::make_tiled_mma(MmaAtom{}, ThreadLayout{}));

    static constexpr bool IsMnMajor = cutlass::gemm::detail::is_mn_major<GmemStride>();
    static constexpr int  Padding   = SmemPaddingElements<Element>::value;
    using SmemLayoutAtom            = cute::conditional_t<
                   IsMnMajor,
                   cute::Layout<cute::Shape<cute::Int<BlkM>, cute::Int<BlkN>>, cute::Stride<cute::_1, cute::Int<BlkM + Padding>>>,
                   cute::Layout<cute::Shape<cute::Int<BlkM>, cute::Int<BlkN>>, cute::Stride<cute::Int<BlkN + Padding>, cute::_1>>>;
    using SmemLayout = SmemLayoutAtom;

    static constexpr int ContiguousDimLength = IsMnMajor ? BlkM : BlkN;
    static constexpr int AlignmentElements =
        GmemTiledCopyAlignment<Element, BlkM, BlkN, ThreadCount, IsMnMajor, GmemAlignmentBytes>::value;
    static constexpr int AlignmentBytes =
        GmemTiledCopyAlignment<Element, BlkM, BlkN, ThreadCount, IsMnMajor, GmemAlignmentBytes>::bytes;
    static constexpr int GmemToSmemAlignmentBytes = AlignmentBytes;
    using AlignmentType                           = cute::uint_byte_t<AlignmentElements *int(sizeof(Element))>;

    using GmemToSmemCopy = decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
                                    cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<AlignmentType>, Element>,
                                    ThreadCount,
                                    AlignmentElements,
                                    GmemStride,
                                    cute::Int<BlkM>,
                                    cute::Int<BlkN>>());

    static constexpr int AlignmentBits = AlignmentElements * int(sizeof(Element)) * 8;
    using RegToSmemCopy  = cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<AlignmentBits>, Element>;
    using SmemToRegCopy  = cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<AlignmentBits>, Element>;
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

// =================================================================================================
// 第五部分：TensorOp (Tensor Core) 架构特征抽象与微架构操作原语
// =================================================================================================

template <class Element> struct Sm80TensorOpTraits;
template <class MmaOperation, bool IsRoleA> struct MmaOperandContiguity;
template <bool NeedTranspose> struct Sm80TensorOpLdsmCopyOperation;
template <class Element, class MmaOperation, bool IsRoleA, bool SmemIsMnMajor, int AlignmentElements, bool UseLdMatrix>
struct Sm80TensorOpSmemCopyOperation;

// ---------------- 1. 数据类型到 MMA 原语的映射 ----------------

/// @brief FP16 Tensor Core 绑定。
/// @note `_TN` 表示在计算 $D = A \times B + C$ 时，硬件要求 A 在寄存器中维持 T(Transpose, 即 K 连续)，
/// B 维持 N(Normal, 亦即 K 连续)。因此对于 _TN 指令族，无论 A 还是 B 都渴望物理访存 K 维度连续。
template <> struct Sm80TensorOpTraits<cutlass::half_t>
{
    using MmaOperation = cute::SM80_16x8x16_F32F16F16F32_TN;
    using Accumulator  = float;
};

template <> struct Sm80TensorOpTraits<cutlass::bfloat16_t>
{
    using MmaOperation = cute::SM80_16x8x16_F32BF16BF16F32_TN;
    using Accumulator  = float;
};

template <> struct Sm80TensorOpTraits<cutlass::tfloat32_t>
{
    using MmaOperation = cute::SM80_16x8x8_F32TF32TF32F32_TN; // TF32 宽度加倍，因此 K 维缩为 8
    using Accumulator  = float;
};
template <> struct Sm80TensorOpTraits<float> : Sm80TensorOpTraits<cutlass::tfloat32_t>
{
};

/// @brief FP64 DMMA (Double-Precision Tensor Core) 绑定。
template <> struct Sm80TensorOpTraits<double>
{
    using MmaOperation = cute::SM80_8x8x4_F64F64F64F64_TN;
    using Accumulator  = double;
};

template <> struct Sm80TensorOpTraits<int8_t>
{
    using MmaOperation = cute::SM80_16x8x32_S32S8S8S32_TN;
    using Accumulator  = int32_t;
};

template <> struct Sm80TensorOpTraits<uint8_t>
{
    using MmaOperation = cute::SM80_16x8x32_S32U8U8S32_TN;
    using Accumulator  = int32_t;
};

// ---------------- 2. MMA 操作数连续性萃取器 (Contiguity Extractor) ----------------
// 用于编译期静态反射当前 MMA 硬件指令对其操作数 (RoleA 或 RoleB) 在寄存器层面的物理连续性期待。

template <bool IsRoleA> struct MmaOperandContiguity<cute::SM80_16x8x16_F32F16F16F32_TN, IsRoleA>
{
    static constexpr bool RequiresMnMajor = false;
};

template <bool IsRoleA> struct MmaOperandContiguity<cute::SM80_16x8x16_F32BF16BF16F32_TN, IsRoleA>
{
    static constexpr bool RequiresMnMajor = false;
};

template <bool IsRoleA> struct MmaOperandContiguity<cute::SM80_16x8x8_F32TF32TF32F32_TN, IsRoleA>
{
    static constexpr bool RequiresMnMajor = false;
};

template <bool IsRoleA> struct MmaOperandContiguity<cute::SM80_8x8x4_F64F64F64F64_TN, IsRoleA>
{
    static constexpr bool RequiresMnMajor = false;
};

template <bool IsRoleA> struct MmaOperandContiguity<cute::SM80_16x8x32_S32S8S8S32_TN, IsRoleA>
{
    static constexpr bool RequiresMnMajor = false;
};

template <bool IsRoleA> struct MmaOperandContiguity<cute::SM80_16x8x32_S32U8U8S32_TN, IsRoleA>
{
    static constexpr bool RequiresMnMajor = false;
};


// ---------------- 3. 基于异或逻辑的 LDSM (ldmatrix) 动态路由引擎 ----------------

/// @brief LDSM 硬件指令分发器
/// @details ldmatrix.sync.aligned.m8n8.x4.shared.b16 是一条专为 Tensor Core 供水的极速总线指令。
/// `LDSM_N` 原封不动搬运连续维，`LDSM_T` 则在搬运的瞬间利用硬件电路对 16x16 矩阵微块执行零开销物理转置。
template <> struct Sm80TensorOpLdsmCopyOperation<false>
{
    using type = cute::SM75_U32x4_LDSM_N; // 不需要转置
};
template <> struct Sm80TensorOpLdsmCopyOperation<true>
{
    using type = cute::SM75_U16x8_LDSM_T; // 需要硬件隐式转置
};

/// @brief 泛型 Shared-to-Register (LdMatrix 侧) 操作推断器。
/// @details 核心 XOR 门阵列：比较 "Smem 目前连续的方向" 与 "MMA 针对当前 Role 期望的方向"。
/// 如果两者方向不一致 (SmemIsMnMajor != MmaRequiresMnMajor)，则触发 NeedTranspose，选用 LDSM_T。
/// 这一设计彻底解决了硬编码 IsMnMajor 导致的非对称架构 (如 _NN 或未来 SM90 _TMA) 下寄存器加载错乱的 Bug。
template <class Element, class MmaOperation, bool IsRoleA, bool SmemIsMnMajor, int AlignmentElements>
struct Sm80TensorOpSmemCopyOperation<Element, MmaOperation, IsRoleA, SmemIsMnMajor, AlignmentElements, true>
{
    static constexpr bool MmaRequiresMnMajor = MmaOperandContiguity<MmaOperation, IsRoleA>::RequiresMnMajor;
    static constexpr bool NeedTranspose      = (SmemIsMnMajor != MmaRequiresMnMajor);
    using type                               = typename Sm80TensorOpLdsmCopyOperation<NeedTranspose>::type;
};

// 退化分支：非 FP16/BF16 类型不走 LDSM，按照能承诺的物理对齐宽度回退至常规向量化读取指令。
template <class Element, class MmaOperation, bool IsRoleA, bool SmemIsMnMajor, int AlignmentElements>
struct Sm80TensorOpSmemCopyOperation<Element, MmaOperation, IsRoleA, SmemIsMnMajor, AlignmentElements, false>
{
    static constexpr int  AlignmentBits = AlignmentElements * int(sizeof(Element)) * 8;
    static constexpr bool NeedTranspose = false;
    using type                          = cute::AutoVectorizingCopyWithAssumedAlignment<AlignmentBits>;
};


// ---------------- 4. 动态 Swizzle (地址异或) 布局发生器 ----------------

/// @brief 动态侦测 TileK 的单行物理宽度，并自适应匹配最优异或掩码 (Swizzle Mask)。
/// @details 为了突破固定 128 Bytes 宽度的限制，此处将 `TileK * sizeof(Element)` 化简，
/// 支持 128B (Base=3, 跨度8组)、64B (Base=2) 和 32B (Base=1) 的多档位 Swizzle。
/// 这种设计允许算子在处理细粒度流水线 (例如 `TileK=32` 即单行 64B) 时，依然能享受 LdMatrix + Swizzle 的极速带宽。
template <class Element, int TileK> struct Sm80TensorOpSwizzleRow
{
    static constexpr int  RowBytes  = TileK * int(sizeof(Element));
    static constexpr int  Bytes     = (RowBytes >= 128 && (RowBytes % 128) == 0) ? 128
                                    : (RowBytes >= 64 && (RowBytes % 64) == 0)   ? 64
                                    : (RowBytes >= 32 && (RowBytes % 32) == 0)   ? 32
                                                                                 : 0;
    static constexpr int  Base      = (Bytes == 128) ? 3 : (Bytes == 64) ? 2 : (Bytes == 32) ? 1 : 0;
    static constexpr int  Elements  = Bytes / int(sizeof(Element));
    static constexpr bool Supported = (Bytes != 0);
};

template <class Element, int TileMN, int TileK, bool UseLdMatrix, bool IsMnMajor> struct Sm80TensorOpSmemLayoutSelector;

/// @brief K 连续的共享内存 Swizzle 布局。
/// @details Base 控制 XOR 的起始 bit 位移。例如 `Swizzle<3,3,3>` 会在地址计算的底层劫持连续维和跨步维的 index，
/// 混合形成伪随机排布，完美抹平 128 Bytes 颗粒度带来的 32-Bank 冲突。
template <class Element, int TileMN, int TileK>
struct Sm80TensorOpSmemLayoutSelector<Element, TileMN, TileK, true, false>
{
    static constexpr int SwizzleBase = Sm80TensorOpSwizzleRow<Element, TileK>::Base;
    static constexpr int RowElements = Sm80TensorOpSwizzleRow<Element, TileK>::Elements;
    static_assert(Sm80TensorOpSwizzleRow<Element, TileK>::Supported,
                  "LdMatrix shared layout requires a 32, 64, or 128 byte row.");

    using SwizzleAtom = decltype(cute::composition(
        cute::Swizzle<SwizzleBase, 3, 3>{},
        cute::Layout<cute::Shape<cute::_8, cute::Int<RowElements>>, cute::Stride<cute::Int<RowElements>, cute::_1>>{}));
    using type = decltype(cute::tile_to_shape(SwizzleAtom{}, cute::Shape<cute::Int<TileMN>, cute::Int<TileK>>{}));
};

/// @brief M/N 连续的共享内存 Swizzle 布局。
template <class Element, int TileMN, int TileK>
struct Sm80TensorOpSmemLayoutSelector<Element, TileMN, TileK, true, true>
{
    static constexpr int SwizzleBase = Sm80TensorOpSwizzleRow<Element, TileK>::Base;
    static constexpr int RowElements = Sm80TensorOpSwizzleRow<Element, TileK>::Elements;
    static_assert(Sm80TensorOpSwizzleRow<Element, TileK>::Supported,
                  "LdMatrix shared layout requires a 32, 64, or 128 byte row.");

    using SwizzleAtom = decltype(cute::composition(
        cute::Swizzle<SwizzleBase, 3, 3>{},
        cute::Layout<cute::Shape<cute::Int<RowElements>, cute::_8>, cute::Stride<cute::_1, cute::Int<RowElements>>>{}));
    using type = decltype(cute::tile_to_shape(SwizzleAtom{}, cute::Shape<cute::Int<TileMN>, cute::Int<TileK>>{}));
};

/// @brief 退化保护分支 (Fallback)：不满足 LdMatrix 条件时，使用传统的 Padding 方法防止冲突。
template <class Element, int TileMN, int TileK, bool IsMnMajor>
struct Sm80TensorOpSmemLayoutSelector<Element, TileMN, TileK, false, IsMnMajor>
{
    static constexpr int Padding = SmemPaddingElements<Element>::value;
    using type                   = cute::conditional_t<IsMnMajor,
                                                       cute::Layout<cute::Shape<cute::Int<TileMN>, cute::Int<TileK>>,
                                                                    cute::Stride<cute::_1, cute::Int<TileMN + Padding>>>,
                                                       cute::Layout<cute::Shape<cute::Int<TileMN>, cute::Int<TileK>>,
                                                                    cute::Stride<cute::Int<TileK + Padding>, cute::_1>>>;
};


// ---------------- 5. 块级别 TiledMMA 协同乘加器装配 ----------------
template <class Element, class MmaAtom, class ThreadLayout> struct Sm80TensorOpTiledMmaSelector
{
    // 通用默认实现：依赖 CuTe 编译器后端的启发式排布（如 DMMA 的特殊 8x8x4 footprint 将路由至此）
    using type = decltype(cute::make_tiled_mma(MmaAtom{}, ThreadLayout{}));
};

template <class MmaAtom, class ThreadLayout> struct Sm80TensorOpTiledMmaSelector<cutlass::half_t, MmaAtom, ThreadLayout>
{
    // 强制 FP16 在块级别绑定为 32x32x16 规模。这一微观尺寸是为了使整个 CTA 内部的
    // 数据分发刚好能够被基于 128B Swizzle 的 LDSM.x4 指令完全填满而不产生气泡。
    using type = cute::TiledMMA<MmaAtom, ThreadLayout, cute::Tile<cute::_32, cute::_32, cute::_16>>;
};

template <class MmaAtom, class ThreadLayout>
struct Sm80TensorOpTiledMmaSelector<cutlass::bfloat16_t, MmaAtom, ThreadLayout>
{
    using type = cute::TiledMMA<MmaAtom, ThreadLayout, cute::Tile<cute::_32, cute::_32, cute::_16>>;
};


// ---------------- 6. Tensor Core 流水线角色的全局集成 ----------------

/// @brief 抽象 Tensor Core 计算场景下矩阵 A 与 B 在 Mainloop 中的宏观行为和指令路由。
template <class Element, class GmemStride, int TileMN, int TileK, int ThreadCount, bool IsRoleA, int GmemAlignmentBytes>
struct Sm80TensorOpMainloopRole
{
    static constexpr bool IsMnMajor           = cutlass::gemm::detail::is_mn_major<GmemStride>();
    static constexpr int  ContiguousDimLength = IsMnMajor ? TileMN : TileK;

    // 安全摄取物理界限下的最大合法位宽
    static constexpr int AlignmentElements =
        GmemTiledCopyAlignment<Element, TileMN, TileK, ThreadCount, IsMnMajor, GmemAlignmentBytes>::value;
    static constexpr int AlignmentBytes =
        GmemTiledCopyAlignment<Element, TileMN, TileK, ThreadCount, IsMnMajor, GmemAlignmentBytes>::bytes;
    static constexpr int AlignmentBits               = AlignmentBytes * 8;
    static constexpr int GmemToSmemAlignmentElements = AlignmentElements;
    static constexpr int GmemToSmemAlignmentBytes    = AlignmentBytes;

    // LdMatrix 准入的联合门禁逻辑：16bit精度 + M/N维切分合法 + 探测出合法的动态 Swizzle 方案
    static constexpr bool UseLdMatrix =
        (std::is_same<Element, cutlass::half_t>::value || std::is_same<Element, cutlass::bfloat16_t>::value)
        && (TileMN % 8 == 0) && Sm80TensorOpSwizzleRow<Element, TileK>::Supported;

    static constexpr int SwizzleBase = UseLdMatrix ? Sm80TensorOpSwizzleRow<Element, TileK>::Base : 0;
    using MmaOperation               = typename Sm80TensorOpTraits<Element>::MmaOperation;

    // 路由物理内存 Layout (带或不带 Swizzle)
    using SmemLayoutAtom =
        typename Sm80TensorOpSmemLayoutSelector<Element, TileMN, TileK, UseLdMatrix, IsMnMajor>::type;
    using SmemLayout = SmemLayoutAtom;

    using AlignmentType = cute::uint_byte_t<AlignmentElements *int(sizeof(Element))>;
    using GmemCopyAtom  = cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<AlignmentType>, Element>;
    using TiledGmemToSmemCopy =
        decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<GmemCopyAtom,
                                                                              ThreadCount,
                                                                              AlignmentElements,
                                                                              GmemStride,
                                                                              cute::Int<TileMN>,
                                                                              cute::Int<TileK>>());

    // 关键修正：TensorOp shared layout 无论是 Swizzle 还是 Padding，都会改变线性二维地址空间。
    // 公开安全入口使用 AutoCopyAsync，由 cooperative_copy 在实际 src/dst layout 上重新推导分块；
    // 显式 tiled cp.async 仍保留给能证明布局完全匹配的低层用户。
    using GmemToSmemCpAsyncCopy = TiledGmemToSmemCopy;
    using GmemToSmemCopy        = cute::AutoCopyAsync;

    // 将 MMA 原子指令特征和 RoleA 身份注入，生成 Role-Aware LDSM 原子
    using SmemCopySelector =
        Sm80TensorOpSmemCopyOperation<Element, MmaOperation, IsRoleA, IsMnMajor, AlignmentElements, UseLdMatrix>;
    static constexpr bool SmemToRegNeedTranspose = SmemCopySelector::NeedTranspose;
    using SmemToRegCopyOperation                 = typename SmemCopySelector::type;
    using SmemToRegCopy                          = cute::Copy_Atom<SmemToRegCopyOperation, Element>;

    using RegToSmemCopyOperation = cute::AutoVectorizingCopyWithAssumedAlignment<AlignmentBits>;
    using RegToSmemCopy          = cute::Copy_Atom<RegToSmemCopyOperation, Element>;
    using SmemToGmemCopy         = decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
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

// 偏特化装载 Mainloop 协议 (A 属于 RoleA，B 属于非 RoleA)
template <class Element, class GmemStride, class TileShape_MNK, int ThreadCount, int GmemAlignmentBytes>
struct Sm80TensorOpRoleA
    : Sm80TensorOpMainloopRole<Element,
                               GmemStride,
                               cute::size<0>(TileShape_MNK{}),
                               cute::size<2>(TileShape_MNK{}),
                               ThreadCount,
                               true,
                               GmemAlignmentBytes>
{
};

template <class Element, class GmemStride, class TileShape_MNK, int ThreadCount, int GmemAlignmentBytes>
struct Sm80TensorOpRoleB
    : Sm80TensorOpMainloopRole<Element,
                               GmemStride,
                               cute::size<1>(TileShape_MNK{}),
                               cute::size<2>(TileShape_MNK{}),
                               ThreadCount,
                               false,
                               GmemAlignmentBytes>
{
};

/// @brief Tensor Core 场景下 C 矩阵（累加结果）后处理（Epilogue）的严格隔离区。
/// @details 极其重要的架构分离：计算往往在宽精度类型（ElementCompute，如 FP32）的寄存器中累加，
/// 而最终结果需要截断写回低精度（ElementOutput，如 FP16）的物理显存。
/// 此处的类型定义严格把控了 "寄存器->Shared" 和 "Shared->Global" 的数据类型和地址对齐计算准则。
template <class Element, class ElementC, class GmemStride, class TileShape_MNK, int ThreadCount, int GmemAlignmentBytes>
struct Sm80TensorOpRoleC
{
    static constexpr int BlkM = cute::size<0>(TileShape_MNK{});
    static constexpr int BlkN = cute::size<1>(TileShape_MNK{});

    using ThreadLayout = typename OptimalTensorOpThreadLayout<BlkM, BlkN, ThreadCount>::Layout;
    using MmaAtom      = cute::MMA_Atom<typename Sm80TensorOpTraits<Element>::MmaOperation>;

    // 明确定义各类层级的物理存储表征
    using ElementInput = Element;                                             // A/B 原始类型
    using ElementCompute = typename Sm80TensorOpTraits<Element>::Accumulator; // MMA 计算时的累加器类型 (如 FP32)
    using ElementOutput   = ElementC; // 用户指定，需要最终落盘入显存的类型 (如 FP16)
    using Accumulator     = ElementCompute;
    using EpilogueElement = ElementCompute; // 暂存 Epilogue Shared 阶段的宽泛类型
    using OutputElement   = ElementOutput;

    using TiledMma = typename Sm80TensorOpTiledMmaSelector<Element, MmaAtom, ThreadLayout>::type;

    static constexpr bool IsMnMajor = cutlass::gemm::detail::is_mn_major<GmemStride>();

    // C 的 Shared Memory 依然采用 Padding，不引入复杂的 Swizzle。
    static constexpr int Padding = SmemPaddingElements<EpilogueElement>::value;
    using SmemLayoutAtom         = cute::conditional_t<
                IsMnMajor,
                cute::Layout<cute::Shape<cute::Int<BlkM>, cute::Int<BlkN>>, cute::Stride<cute::_1, cute::Int<BlkM + Padding>>>,
                cute::Layout<cute::Shape<cute::Int<BlkM>, cute::Int<BlkN>>, cute::Stride<cute::Int<BlkN + Padding>, cute::_1>>>;
    using SmemLayout = SmemLayoutAtom;

    static constexpr int OutputPadding = SmemPaddingElements<ElementOutput>::value;
    using OutputSmemLayoutAtom         = cute::conditional_t<
                IsMnMajor,
                cute::Layout<cute::Shape<cute::Int<BlkM>, cute::Int<BlkN>>,
                             cute::Stride<cute::_1, cute::Int<BlkM + OutputPadding>>>,
                cute::Layout<cute::Shape<cute::Int<BlkM>, cute::Int<BlkN>>,
                             cute::Stride<cute::Int<BlkN + OutputPadding>, cute::_1>>>;
    using OutputSmemLayout = OutputSmemLayoutAtom;

    // 内部数据的换向搬运基于计算高精度 (EpilogueElement)
    static constexpr int ContiguousDimLength = IsMnMajor ? BlkM : BlkN;
    static constexpr int EpilogueAlignmentElements =
        GmemVectorAlignment<EpilogueElement, ContiguousDimLength, 16>::value;
    static constexpr int EpilogueAlignmentBits = EpilogueAlignmentElements * int(sizeof(EpilogueElement)) * 8;
    static constexpr int AlignmentElements     = EpilogueAlignmentElements;
    static constexpr int AlignmentBits         = EpilogueAlignmentBits;

    using SmemToRegCopyOperation = cute::AutoVectorizingCopyWithAssumedAlignment<EpilogueAlignmentBits>;
    using RegToSmemCopyOperation = cute::AutoVectorizingCopyWithAssumedAlignment<EpilogueAlignmentBits>;
    using SmemToRegCopy          = cute::Copy_Atom<SmemToRegCopyOperation, EpilogueElement>;
    using RegToSmemCopy          = cute::Copy_Atom<RegToSmemCopyOperation, EpilogueElement>;

    // 核心屏障：所有写出到 Global Memory 的操作，强制基于 ElementOutput (如 FP16) 类型去推导位宽和字节数！
    // 这切断了把 FP32 当做 FP16 数据量写回导致的致命内存越界 (Out-of-Bounds memory access)。
    static constexpr int OutputAlignmentElements =
        GmemTiledCopyAlignment<ElementOutput, BlkM, BlkN, ThreadCount, IsMnMajor, GmemAlignmentBytes>::value;
    static constexpr int OutputAlignmentBytes =
        GmemTiledCopyAlignment<ElementOutput, BlkM, BlkN, ThreadCount, IsMnMajor, GmemAlignmentBytes>::bytes;
    static constexpr int OutputAlignmentBits      = OutputAlignmentBytes * 8;
    static constexpr int GmemToSmemAlignmentBytes = OutputAlignmentBytes;
    using OutputAlignmentType                     = cute::uint_byte_t<OutputAlignmentBytes>;

    using GmemToSmemCopy =
        decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
                 cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<OutputAlignmentType>, ElementOutput>,
                 ThreadCount,
                 OutputAlignmentElements,
                 GmemStride,
                 cute::Int<BlkM>,
                 cute::Int<BlkN>>());

    using OutputSmemToGmemCopy = decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
                                          VectorizedCopyAtom<ElementOutput, OutputAlignmentElements>,
                                          ThreadCount,
                                          OutputAlignmentElements,
                                          GmemStride,
                                          cute::Int<BlkM>,
                                          cute::Int<BlkN>>());
    using SmemToGmemCopy       = OutputSmemToGmemCopy;

    using GlobalToSharedCopy   = GmemToSmemCopy;
    using SharedToRegisterCopy = SmemToRegCopy;
    using RegisterToSharedCopy = RegToSmemCopy;
    using SharedToGlobalCopy   = SmemToGmemCopy;
    using SharedToGlobalLayout = OutputSmemLayout;
};

} // namespace detail

// =================================================================================================
// 第六部分：AutoPartitioner (前端拦截与静态派发器)
// =================================================================================================

/// @brief SIMT 架构特征静态捕获层。
/// @details 当前端 `AutoPartitioner` 探测到 `OpClassSimt` 标签，并且元素类型受 SIMT 引擎支持时被 SFINAE 激活。
template <typename Element,
          typename GmemStride,
          typename TileShape_MNK,
          int ThreadCount,
          typename ElementC,
          int GmemAlignmentA,
          int GmemAlignmentB,
          int GmemAlignmentC,
          typename ClusterShape_MNK>
struct AutoPartitioner<cutlass::arch::Sm80,
                       cutlass::arch::OpClassSimt,
                       Element,
                       GmemStride,
                       TileShape_MNK,
                       ThreadCount,
                       ElementC,
                       GmemAlignmentA,
                       GmemAlignmentB,
                       GmemAlignmentC,
                       ClusterShape_MNK,
                       std::enable_if_t<detail::IsSm80SimtElement<Element>::value>>
{
    using RoleA = detail::Sm80SimtRoleA<Element, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentA>;
    using RoleB = detail::Sm80SimtRoleB<Element, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentB>;
    using RoleC = detail::Sm80SimtRoleC<Element, ElementC, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentC>;
};

/// @brief TensorOp 架构特征静态捕获层。
/// @details 当前端探测到 `OpClassTensorOp` 标签，并识别出可利用硬件矩阵乘加速引擎（如 FP16, DMMA 等）时激活。
template <typename Element,
          typename GmemStride,
          typename TileShape_MNK,
          int ThreadCount,
          typename ElementC,
          int GmemAlignmentA,
          int GmemAlignmentB,
          int GmemAlignmentC,
          typename ClusterShape_MNK>
struct AutoPartitioner<cutlass::arch::Sm80,
                       cutlass::arch::OpClassTensorOp,
                       Element,
                       GmemStride,
                       TileShape_MNK,
                       ThreadCount,
                       ElementC,
                       GmemAlignmentA,
                       GmemAlignmentB,
                       GmemAlignmentC,
                       ClusterShape_MNK,
                       std::enable_if_t<detail::IsSm80TensorOpElement<Element>::value>>
{
    using RoleA = detail::Sm80TensorOpRoleA<Element, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentA>;
    using RoleB = detail::Sm80TensorOpRoleB<Element, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentB>;
    using RoleC = detail::Sm80TensorOpRoleC<Element, ElementC, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentC>;
};

} // namespace autopartition
