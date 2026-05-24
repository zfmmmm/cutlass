#pragma once // 确保头文件在单次编译中只被包含一次，防止重复定义

// =====================================================================================
// 引入必要的标准库和核心依赖
// =====================================================================================
#include <cstdint>     // 提供精确宽度的整数类型 (int8_t, uint8_t, int32_t 等)
#include <type_traits> // 编译期类型推导与变换核心库 (std::is_same, std::enable_if_t)

// clang-format off
// -------------------------------------------------------------------------------------
// CUTE 原子指令与特征层 (Atom & Traits)
// CUTE 负责将底层 PTX 汇编指令封装为代数拓扑概念的 Layout 和 Tensor 映射
// -------------------------------------------------------------------------------------
#include <cute/atom/copy_atom.hpp>             // 提供 Copy_Atom 抽象，绑定访存指令与数据类型
#include <cute/atom/copy_traits_sm100.hpp>     // SM100 专属访存指令（如 TMEM_LOAD/STORE，LSMEM 等）
#include <cute/atom/copy_traits_sm100_tma.hpp> // SM100 TMA (Tensor Memory Accelerator) 硬件异步搬运描述符与指令
#include <cute/atom/copy_traits_sm80.hpp> // 保留 SM80 特征，因为 SIMT fallback 和某些非齐整内存依旧需要 cp.async
#include <cute/atom/mma_atom.hpp>         // 提供 MMA_Atom 抽象，封装矩阵乘加指令
#include <cute/atom/mma_traits_sm100.hpp> // SM100 核心：TCGEN05 (UMMA) 描述符分发与计算原语
#include <cute/atom/mma_traits_sm120.hpp> // SM120 核心：为 Blackwell 消费级 (RTX 50系) 引入 FP8/FP4 特化指令
#include <cute/layout.hpp>                // Shape o Stride，逻辑与物理坐标映射的核心代数引擎
#include <cute/tensor.hpp> // 结合 Layout 和物理引擎(显存/共享内存/寄存器/TMEM)的张量抽象

// -------------------------------------------------------------------------------------
// CUTLASS 3.x 架构与 Builder 层 (Architecture & Builders)
// 提供现代化的标签 (Tags) 和官方验证过的标准构建器
// -------------------------------------------------------------------------------------
#include <cutlass/arch/arch.h>                               // 架构标签定义 (Sm100, Sm120 等)
#include <cutlass/arch/mma.h>                                // 操作类型标签 (OpClassSimt, OpClassTensorOp 等)
#include <cutlass/gemm/collective/collective_builder_decl.hpp>     // CUTLASS 3.x CollectiveBuilder 声明
#include <cutlass/gemm/collective/collective_mma_decl.hpp>         // Collective MMA 声明
#include <cutlass/gemm/collective/builders/sm100_common.inl> // SM100 官方通用的辅助推导函数 (如 smem_selector)
#include <cutlass/gemm/collective/builders/sm100_simt_builder.inl> // SM100 SIMT (CUDA Core) 官方布局构建器
#include <cutlass/gemm/collective/builders/sm120_common.inl>       // SM120 官方通用辅助函数
#include <cutlass/gemm/collective/builders/sm90_common.inl>        // 借用 SM90 的 TMA 和一些底层通用机制
#include <cutlass/gemm/gemm.h>                                     // GEMM 基础定义
#include <cutlass/numeric_types.h> // 硬件级数值类型 (half_t, bfloat16_t, float_e4m3_t 等)

#include "../auto_partitioner.hpp" // 引入本项目的泛型前端
#include "sm80_policy.hpp"         // 引入 SM80 策略作为部分架构退化(Fallback)的备选
// clang-format on

namespace autopartition {
namespace detail {

// =====================================================================================
// 1. 硬件类型路由特征 (Type Routing Traits)
// =====================================================================================

// [SM100 SIMT 检查]
// 在 Blackwell 架构上，CUTLASS 官方的 SIMT builder 被严格收束，目前主要服务于 SGEMM (单精度)。
template <class Element> struct IsSm100SimtElement : std::is_same<Element, float>
{
};

// [SM100 TensorOp 检查]
// 检查输入类型是否支持放入 UMMA (User-level MMA) 并使用 TMEM (Tensor Memory) 累加。
// 支持 FP32(TF32), FP16, BF16, 以及 INT8/UINT8 量化。
template <class Element>
struct IsSm100TensorOpElement
    : std::integral_constant<bool,
                             std::is_same<Element, float>::value || std::is_same<Element, cutlass::half_t>::value
                                 || std::is_same<Element, cutlass::bfloat16_t>::value
                                 || std::is_same<Element, int8_t>::value || std::is_same<Element, uint8_t>::value>
{
};

// [SM120 TensorOp 检查 (前瞻)]
// 区分 SM120 的关键：引入了原生的 FP8 (e4m3 和 e5m2) 硬件级支持。
template <class Element>
struct IsSm120TensorOpElement
    : std::integral_constant<bool,
                             std::is_same<Element, cutlass::float_e4m3_t>::value
                                 || std::is_same<Element, cutlass::float_e5m2_t>::value>
{
};

// [UMMA 累加器类型映射]
// 保证计算精度，防止溢出。浮点一律用 FP32 累加，8位整数一律用 INT32 累加。
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

// [通用显存向量化对齐推导]
// 复用基础策略中的对齐逻辑，但将最大对齐字节限制为 16B (即 128-bit，这是单线程 cp.async 的极限)。
template <class Element, int ContiguousElements, int MaxAlignmentBytes = 16>
struct Sm100GmemVectorAlignment : GmemVectorAlignment<Element, ContiguousElements, MaxAlignmentBytes>
{
};

constexpr int sm100_constexpr_gcd(int lhs, int rhs)
{
    lhs = lhs < 0 ? -lhs : lhs;
    rhs = rhs < 0 ? -rhs : rhs;
    while (rhs != 0) {
        int tmp = lhs % rhs;
        lhs     = rhs;
        rhs     = tmp;
    }
    return lhs;
}

// Shared-memory row pitch selector for vectorized SIMT/epilogue traffic.
// It chooses the smallest element padding that keeps each row aligned to the vector width
// and rotates 128B bank groups with a coprime stride.
template <class Element, int MajorExtent, int VectorBits = 128, int BankCount = 32, int BankWidthBytes = 4>
struct Sm100SmemBankPaddingElements
{
    static_assert(VectorBits % 8 == 0, "SM100 shared-memory vector width must be byte-addressable.");
    static_assert(MajorExtent > 0, "SM100 shared-memory major extent must be positive.");

    static constexpr int ElementBytes   = int(sizeof(Element));
    static constexpr int VectorBytes    = VectorBits / 8;
    static constexpr int BankSpanBytes  = BankCount * BankWidthBytes;
    static constexpr int VectorBankSets = BankSpanBytes / VectorBytes;

    static_assert((BankSpanBytes % VectorBytes) == 0,
                  "SM100 shared-memory vector width must divide the 32-bank span.");

    static constexpr bool is_candidate(int padding_elements)
    {
        int pitch_bytes = (MajorExtent + padding_elements) * ElementBytes;
        if ((pitch_bytes % VectorBytes) != 0) {
            return false;
        }
        int bank_set_stride = (pitch_bytes / VectorBytes) % VectorBankSets;
        return bank_set_stride != 0 && sm100_constexpr_gcd(bank_set_stride, VectorBankSets) == 1;
    }

    static constexpr int select_padding_elements()
    {
        for (int padding_elements = 0; padding_elements <= BankSpanBytes / ElementBytes; ++padding_elements) {
            if (is_candidate(padding_elements)) {
                return padding_elements;
            }
        }
        return -1;
    }

    static constexpr int value = select_padding_elements();
    static constexpr int bytes = value * ElementBytes;

    static_assert(value >= 0, "No legal SM100 shared-memory bank padding found for this element and vector width.");
};

template <int GmemAlignmentBytes> struct Sm100UseTma : std::integral_constant<bool, (GmemAlignmentBytes >= 16)>
{
};

template <class TileShape_MNK, int ThreadCount> struct Sm100SimtNativeEligible
{
    using WarpShape_MNK =
        decltype(cutlass::gemm::collective::detail::sm100_simt_f32_warp_shape_mnk_selector<TileShape_MNK>());

    static constexpr bool value = (cute::size<2>(TileShape_MNK{}) == 16)
                               && (ThreadCount == int(cute::size(WarpShape_MNK{})) * int(cutlass::NumThreadsPerWarp));
};

// =====================================================================================
// 2. SM100 SIMT (CUDA Core) 策略定义
// =====================================================================================
// 虽然 Blackwell 的重心是 Tensor Core，但在处理极其不规则的形状或特殊后处理时，
// 依然需要高度优化的 SIMT 路径作为兜底。

template <class Element, class GmemStride, int TileMN, int TileK, int ThreadCount> struct Sm100SimtMainloopRole
{
    // [严苛的官方对齐要求]
    static_assert(std::is_same<Element, float>::value, "SM100 SIMT policy currently targets SGEMM.");

    using TileShape = cute::Shape<cute::Int<TileMN>, cute::_1, cute::Int<TileK>>;

    // 探测显存布局是行主序还是列主序
    static constexpr bool IsMnMajor = cutlass::gemm::detail::is_mn_major<GmemStride>();

    // [SIMT Shared Memory Bank Conflict 消除策略]
    // K 连续时，shared 行距由元素大小、TileMN 和 128-bit shared 读写宽度共同决定。
    static constexpr int SmemVectorAlignmentBits = 128;
    static constexpr int SmemAlignmentOffset =
        IsMnMajor ? 0 : Sm100SmemBankPaddingElements<Element, TileMN, SmemVectorAlignmentBits>::value;
    static constexpr int ContiguousDimLength = IsMnMajor ? TileMN : TileK;

    // [降级保护] K-major 时，写入 Shared Memory 需要转置，不能用 128-bit 连续异步拷贝，强制退化为 32-bit (1个元素)。
    static constexpr int AlignmentElements =
        IsMnMajor ? Sm100GmemVectorAlignment<Element, ContiguousDimLength>::value : 1;
    using AlignmentType = cute::uint_byte_t<AlignmentElements *int(sizeof(Element))>;

    // 静态构建带 Padding 保护的 Shared Memory Layout
    using SmemLayoutAtom = cute::Layout<cute::Shape<cute::Int<TileMN>, cute::Int<TileK>>,
                                        cute::Stride<cute::_1, cute::Int<TileMN + SmemAlignmentOffset>>>;
    using SmemLayout     = SmemLayoutAtom;

    // [寄存器读取策略]
    // Padding 已保证 shared 行距按 128-bit bank-group 旋转，因此两种方向都保持 x4 读写宽度。
    using SmemToRegCopy =
        cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<SmemVectorAlignmentBits>, Element>;
    using RegToSmemCopy = SmemToRegCopy;

    // Global 到 Shared 依然依赖 SM80 引入的硬件异步指令 (cp.async)
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

// 偏特化：绑定 A 矩阵角色
template <class Element, class GmemStride, class TileShape_MNK, int ThreadCount>
struct Sm100SimtRoleA
    : Sm100SimtMainloopRole<Element,
                            GmemStride,
                            cute::size<0>(TileShape_MNK{}),
                            cute::size<2>(TileShape_MNK{}),
                            ThreadCount>
{
};

// 偏特化：绑定 B 矩阵角色
template <class Element, class GmemStride, class TileShape_MNK, int ThreadCount>
struct Sm100SimtRoleB
    : Sm100SimtMainloopRole<Element,
                            GmemStride,
                            cute::size<1>(TileShape_MNK{}),
                            cute::size<2>(TileShape_MNK{}),
                            ThreadCount>
{
};

// C 矩阵 / 累加器的 SIMT 角色定义
template <class Element, class GmemStride, class TileShape_MNK, int ThreadCount> struct Sm100SimtRoleC
{
    static_assert(std::is_same<Element, float>::value, "SM100 SIMT policy currently targets SGEMM.");

    // 调用官方构建器，自动推导 SIMT 模式下的 Warp 拓扑 (如 2x4 还是 4x2)
    using WarpShape_MNK =
        decltype(cutlass::gemm::collective::detail::sm100_simt_f32_warp_shape_mnk_selector<TileShape_MNK>());

    // 强制校验用户传入的线程数必须和官方拓扑严丝合缝匹配
    static constexpr int OfficialThreadCount = cute::size(WarpShape_MNK{}) * cutlass::NumThreadsPerWarp;

    // 生成 SIMT 核心计算原语 (Universal FMA 的平铺版本)
    using TiledMma =
        decltype(cutlass::gemm::collective::detail::
                     sm100_make_simt_f32_tiled_mma<GmemStride, 1, GmemStride, 1, TileShape_MNK, WarpShape_MNK>());
    using MmaAtom = typename TiledMma::Atom;

    static constexpr int  BlkM      = cute::size<0>(TileShape_MNK{});
    static constexpr int  BlkN      = cute::size<1>(TileShape_MNK{});
    static constexpr bool IsMnMajor = cutlass::gemm::detail::is_mn_major<GmemStride>();
    static constexpr int  Padding   = SmemPaddingElements<Element>::value;

    // C 矩阵的 Shared Memory Layout (防 Bank Conflict Padding)
    using SmemLayoutAtom = cute::conditional_t<
        IsMnMajor,
        cute::Layout<cute::Shape<cute::Int<BlkM>, cute::Int<BlkN>>, cute::Stride<cute::_1, cute::Int<BlkM + Padding>>>,
        cute::Layout<cute::Shape<cute::Int<BlkM>, cute::Int<BlkN>>, cute::Stride<cute::Int<BlkN + Padding>, cute::_1>>>;
    using SmemLayout = SmemLayoutAtom;

    static constexpr int ContiguousDimLength = IsMnMajor ? BlkM : BlkN;
    static constexpr int AlignmentElements   = Sm100GmemVectorAlignment<Element, ContiguousDimLength>::value;
    using AlignmentType                      = cute::uint_byte_t<AlignmentElements *int(sizeof(Element))>;

    // C 矩阵搬运策略 (全开 128-bit 位宽，因为通常不需要在主循环内转置)
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

// =====================================================================================
// 3. SM100 TensorOp (UMMA + TMEM + TMA) 策略核心定义
// =====================================================================================
// 这是 Blackwell 架构的灵魂所在：全自动的数据搬运(TMA)和无寄存器介入的矩阵计算(UMMA)。

template <class Element,
          class GmemStride,
          class TileShape_MNK,
          int ThreadCount,
          int GmemAlignmentBytes,
          class ClusterShape,
          bool IsRoleA>
struct Sm100TensorOpMainloopRole;

// [特化：RoleA (A矩阵主循环)]
template <class Element,
          class GmemStride,
          class TileShape_MNK,
          int ThreadCount,
          int GmemAlignmentBytes,
          class ClusterShape>
struct Sm100TensorOpMainloopRole<Element,
                                 GmemStride,
                                 TileShape_MNK,
                                 ThreadCount,
                                 GmemAlignmentBytes,
                                 ClusterShape,
                                 true>
{
    // 将上层类型转化为底层 PTX 对应的 MMA 载荷类型 (如 float -> tfloat32)
    using ElementMma =
        decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<Element>());
    // 决定 Shared Memory 中分配颗粒度 (如果小于 8bit，按照 8bit 对齐分配)
    using SmemAllocElement = cute::conditional_t<(cute::sizeof_bits_v<ElementMma> < 8), uint8_t, ElementMma>;
    using Accumulator      = typename Sm100TensorOpAccumulator<Element>::type;

    // [核心亮点：基于名字和 Tag 的路由]
    // 不再手动编写繁杂的 Swizzle<3,3,3>，而是通过 tag_to_umma_major_A 从 Stride 中提取物理主序。
    // 这完美契合了 CUTLASS 3.x CollectiveBuilder "无内部结构体，全靠标准标签" 的现代设计哲学。
    static constexpr cute::UMMA::Major Major     = cutlass::gemm::collective::detail::tag_to_umma_major_A<GmemStride>();
    static constexpr int               BlkM      = cute::size<0>(TileShape_MNK{});
    static constexpr int               BlkK      = cute::size<2>(TileShape_MNK{});
    static constexpr bool              IsMnMajor = cutlass::gemm::detail::is_mn_major<GmemStride>();
    using TiledMma = decltype(cutlass::gemm::collective::detail::sm100_make_trivial_tiled_mma<
                              ElementMma,
                              ElementMma,
                              Accumulator,
                              TileShape_MNK,
                              ClusterShape,
                              Major,
                              cutlass::gemm::collective::detail::tag_to_umma_major_B<GmemStride>(),
                              cutlass::gemm::collective::KernelScheduleAuto>());

    // 让硬件和官方 selector 根据 Major Tag 自动推导配合 UMMA 描述符所需的 128B Swizzle Layout
    using SmemLayoutAtom =
        decltype(cutlass::gemm::collective::detail::
                     sm100_smem_selector<Major, SmemAllocElement, cute::Int<BlkM>, cute::Int<BlkK>>());
    using MmaShapeA =
        decltype(cute::partition_shape_A(TiledMma{}, cute::make_shape(cute::Int<BlkM>{}, cute::Int<BlkK>{})));
    using SmemLayout = decltype(cute::UMMA::tile_to_mma_shape(SmemLayoutAtom{}, MmaShapeA{}));

    static constexpr bool UsesTmaLoad = Sm100UseTma<GmemAlignmentBytes>::value;
    using ClusterLayoutVMNK           = decltype(cute::tiled_divide(cute::make_layout(ClusterShape{}),
                                                          cute::make_tile(typename TiledMma::AtomThrID{})));
    using GmemTiledCopyTmaOperation =
        decltype(cutlass::gemm::collective::detail::sm90_cluster_shape_to_tma_atom(cute::size<1>(ClusterShape{})));
    using GmemToSmemTmaCopy =
        decltype(cute::make_tma_atom_A_sm100(GmemTiledCopyTmaOperation{},
                                             cute::make_tensor(cute::make_gmem_ptr(static_cast<Element *>(nullptr)),
                                                               cute::make_shape(cute::Int<BlkM>{}, cute::Int<BlkK>{}),
                                                               GmemStride{}),
                                             SmemLayout{},
                                             TileShape_MNK{},
                                             TiledMma{},
                                             ClusterLayoutVMNK{}));

    // 保留 cp.async 作为后备降级路径 (Fallback)，用于无法满足 TMA 16B 对齐要求时的灾难恢复
    static constexpr int GmemToSmemAlignmentElements =
        GmemTiledCopyAlignment<Element, BlkM, BlkK, ThreadCount, IsMnMajor, GmemAlignmentBytes>::value;
    static constexpr int GmemToSmemAlignmentBytes =
        GmemTiledCopyAlignment<Element, BlkM, BlkK, ThreadCount, IsMnMajor, GmemAlignmentBytes>::bytes;
    using CpAsyncAlignmentType = cute::uint_byte_t<GmemToSmemAlignmentBytes>;
    using GmemToSmemCpAsyncCopy =
        decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
                 cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<CpAsyncAlignmentType>, Element>,
                 ThreadCount,
                 GmemToSmemAlignmentElements,
                 GmemStride,
                 cute::Int<BlkM>,
                 cute::Int<BlkK>>());
    using GmemToSmemCopy     = cute::conditional_t<UsesTmaLoad, GmemToSmemTmaCopy, cute::AutoCopyAsync>;
    using GlobalToSharedCopy = GmemToSmemCopy;

    // [破局点：寄存器的彻底消失]
    // Blackwell 的 UMMA 通过描述符直接吸取 Shared Memory 中的数据。
    // 抽象层中 Smem -> Reg 的搬运逻辑直接设为 void，极大降低了寄存器压力 (Register Pressure)。
    using SmemToRegCopy        = void;
    using SharedToRegisterCopy = SmemToRegCopy;
    using RegToSmemCopy        = void;
    using RegisterToSharedCopy = RegToSmemCopy;

    using SmemToGmemCopy     = void;
    using SharedToGlobalCopy = SmemToGmemCopy;
};

// [特化：RoleB (B矩阵主循环)]
template <class Element,
          class GmemStride,
          class TileShape_MNK,
          int ThreadCount,
          int GmemAlignmentBytes,
          class ClusterShape>
struct Sm100TensorOpMainloopRole<Element,
                                 GmemStride,
                                 TileShape_MNK,
                                 ThreadCount,
                                 GmemAlignmentBytes,
                                 ClusterShape,
                                 false>
{
    // 逻辑与 RoleA 镜像对称。区别在于 Major 标签必须使用 tag_to_umma_major_B 提取，
    // 因为 B 矩阵在 K 维度上的布局要求与 A 矩阵有物理交叉矩阵(Crossbar)上的互补性。
    using ElementMma =
        decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<Element>());
    using SmemAllocElement = cute::conditional_t<(cute::sizeof_bits_v<ElementMma> < 8), uint8_t, ElementMma>;
    using Accumulator      = typename Sm100TensorOpAccumulator<Element>::type;

    static constexpr cute::UMMA::Major Major     = cutlass::gemm::collective::detail::tag_to_umma_major_B<GmemStride>();
    static constexpr int               BlkN      = cute::size<1>(TileShape_MNK{});
    static constexpr int               BlkK      = cute::size<2>(TileShape_MNK{});
    static constexpr bool              IsMnMajor = cutlass::gemm::detail::is_mn_major<GmemStride>();
    using TiledMma = decltype(cutlass::gemm::collective::detail::sm100_make_trivial_tiled_mma<
                              ElementMma,
                              ElementMma,
                              Accumulator,
                              TileShape_MNK,
                              ClusterShape,
                              cutlass::gemm::collective::detail::tag_to_umma_major_A<GmemStride>(),
                              Major,
                              cutlass::gemm::collective::KernelScheduleAuto>());

    using SmemLayoutAtom =
        decltype(cutlass::gemm::collective::detail::
                     sm100_smem_selector<Major, SmemAllocElement, cute::Int<BlkN>, cute::Int<BlkK>>());
    using MmaShapeB =
        decltype(cute::partition_shape_B(TiledMma{}, cute::make_shape(cute::Int<BlkN>{}, cute::Int<BlkK>{})));
    using SmemLayout = decltype(cute::UMMA::tile_to_mma_shape(SmemLayoutAtom{}, MmaShapeB{}));

    static constexpr bool UsesTmaLoad = Sm100UseTma<GmemAlignmentBytes>::value;
    using ClusterLayoutVMNK           = decltype(cute::tiled_divide(cute::make_layout(ClusterShape{}),
                                                          cute::make_tile(typename TiledMma::AtomThrID{})));
    using GmemTiledCopyTmaOperation =
        decltype(cutlass::gemm::collective::detail::sm90_cluster_shape_to_tma_atom(cute::size<0>(ClusterShape{})));
    using GmemToSmemTmaCopy =
        decltype(cute::make_tma_atom_B_sm100(GmemTiledCopyTmaOperation{},
                                             cute::make_tensor(cute::make_gmem_ptr(static_cast<Element *>(nullptr)),
                                                               cute::make_shape(cute::Int<BlkN>{}, cute::Int<BlkK>{}),
                                                               GmemStride{}),
                                             SmemLayout{},
                                             TileShape_MNK{},
                                             TiledMma{},
                                             ClusterLayoutVMNK{}));

    static constexpr int GmemToSmemAlignmentElements =
        GmemTiledCopyAlignment<Element, BlkN, BlkK, ThreadCount, IsMnMajor, GmemAlignmentBytes>::value;
    static constexpr int GmemToSmemAlignmentBytes =
        GmemTiledCopyAlignment<Element, BlkN, BlkK, ThreadCount, IsMnMajor, GmemAlignmentBytes>::bytes;
    using CpAsyncAlignmentType = cute::uint_byte_t<GmemToSmemAlignmentBytes>;
    using GmemToSmemCpAsyncCopy =
        decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
                 cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<CpAsyncAlignmentType>, Element>,
                 ThreadCount,
                 GmemToSmemAlignmentElements,
                 GmemStride,
                 cute::Int<BlkN>,
                 cute::Int<BlkK>>());
    using GmemToSmemCopy     = cute::conditional_t<UsesTmaLoad, GmemToSmemTmaCopy, cute::AutoCopyAsync>;
    using GlobalToSharedCopy = GmemToSmemCopy;

    using SmemToRegCopy        = void;
    using SharedToRegisterCopy = SmemToRegCopy;
    using RegToSmemCopy        = void;
    using RegisterToSharedCopy = RegToSmemCopy;
    using SmemToGmemCopy       = void;
    using SharedToGlobalCopy   = SmemToGmemCopy;
};

// 偏特化装载外壳
template <class Element,
          class GmemStride,
          class TileShape_MNK,
          int ThreadCount,
          int GmemAlignmentBytes,
          class ClusterShape>
struct Sm100TensorOpRoleA
    : Sm100TensorOpMainloopRole<Element, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentBytes, ClusterShape, true>
{
};

template <class Element,
          class GmemStride,
          class TileShape_MNK,
          int ThreadCount,
          int GmemAlignmentBytes,
          class ClusterShape>
struct Sm100TensorOpRoleB
    : Sm100TensorOpMainloopRole<Element,
                                GmemStride,
                                TileShape_MNK,
                                ThreadCount,
                                GmemAlignmentBytes,
                                ClusterShape,
                                false>
{
};

// [C 矩阵 / 累加器角色 (连接 TMEM)]
template <class Element,
          class ElementC,
          class GmemStride,
          class TileShape_MNK,
          int ThreadCount,
          int GmemAlignmentBytes,
          class ClusterShape>
struct Sm100TensorOpRoleC
{
    using ElementInput  = Element;
    using ElementOutput = ElementC;
    using ElementMma =
        decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<Element>());
    using Accumulator     = typename Sm100TensorOpAccumulator<Element>::type;
    using ElementCompute  = Accumulator;
    using EpilogueElement = Accumulator;

    using ClusterShape_MNK = ClusterShape;

    // [生成带 TMEM 标记的 MMA 拓扑]
    // TiledMmaFor 允许通过模板显式注入 A/B 的 Major 属性，防止 Partitioner 内部耦合。
    // sm100_make_trivial_tiled_mma 生成的 fragment 不再是寄存器阵列，而是 cute::UMMA::tmem_frg_base (TMEM 寻址指针)。
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
    static constexpr int  SmemMajorExtent = IsMnMajor ? BlkM : BlkN;
    static constexpr int  Padding         = Sm100SmemBankPaddingElements<EpilogueElement, SmemMajorExtent, 128>::value;

    using SmemLayoutAtom = cute::conditional_t<
        IsMnMajor,
        cute::Layout<cute::Shape<cute::Int<BlkM>, cute::Int<BlkN>>, cute::Stride<cute::_1, cute::Int<BlkM + Padding>>>,
        cute::Layout<cute::Shape<cute::Int<BlkM>, cute::Int<BlkN>>, cute::Stride<cute::Int<BlkN + Padding>, cute::_1>>>;
    using SmemLayout = SmemLayoutAtom;

    static constexpr int OutputPadding = Sm100SmemBankPaddingElements<ElementOutput, SmemMajorExtent, 128>::value;
    using OutputSmemLayoutAtom = cute::conditional_t<
        IsMnMajor,
        cute::Layout<cute::Shape<cute::Int<BlkM>, cute::Int<BlkN>>,
                     cute::Stride<cute::_1, cute::Int<BlkM + OutputPadding>>>,
        cute::Layout<cute::Shape<cute::Int<BlkM>, cute::Int<BlkN>>,
                     cute::Stride<cute::Int<BlkN + OutputPadding>, cute::_1>>>;
    using OutputSmemLayout = OutputSmemLayoutAtom;

    static constexpr bool UsesTmaLoad  = Sm100UseTma<GmemAlignmentBytes>::value;
    static constexpr bool UsesTmaStore = Sm100UseTma<GmemAlignmentBytes>::value;
    static constexpr int  GmemToSmemAlignmentElements =
        GmemTiledCopyAlignment<ElementOutput, BlkM, BlkN, ThreadCount, IsMnMajor, GmemAlignmentBytes>::value;
    static constexpr int GmemToSmemAlignmentBytes =
        GmemTiledCopyAlignment<ElementOutput, BlkM, BlkN, ThreadCount, IsMnMajor, GmemAlignmentBytes>::bytes;
    static constexpr int SmemToGmemAlignmentElements = GmemToSmemAlignmentElements;
    static constexpr int SmemToGmemAlignmentBytes    = GmemToSmemAlignmentBytes;
    using GmemAlignmentType                          = cute::uint_byte_t<GmemToSmemAlignmentBytes>;

    using GmemToSmemTmaCopy =
        decltype(cute::make_tma_copy(cute::SM90_TMA_LOAD{},
                                     cute::make_tensor(cute::make_gmem_ptr(static_cast<ElementOutput *>(nullptr)),
                                                       cute::make_shape(cute::Int<BlkM>{}, cute::Int<BlkN>{}),
                                                       GmemStride{}),
                                     OutputSmemLayout{},
                                     cute::make_shape(cute::Int<BlkM>{}, cute::Int<BlkN>{}),
                                     cute::Int<1>{}));
    using GmemToSmemCpAsyncCopy =
        decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
                 cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<GmemAlignmentType>, ElementOutput>,
                 ThreadCount,
                 GmemToSmemAlignmentElements,
                 GmemStride,
                 cute::Int<BlkM>,
                 cute::Int<BlkN>>());
    using GmemToSmemCopy = cute::conditional_t<UsesTmaLoad, GmemToSmemTmaCopy, GmemToSmemCpAsyncCopy>;

    using TmemToSmemCopyOperation =
        cute::conditional_t<(BlkM == 64), cute::SM100_TMEM_STORE_16dp256b1x, cute::SM100_TMEM_STORE_32dp32b32x>;
    using TmemToSmemCopy = cute::Copy_Atom<TmemToSmemCopyOperation, Accumulator>;

    using SmemToRegCopyOperation = cute::AutoVectorizingCopyWithAssumedAlignment<128>;
    using RegToSmemCopyOperation = cute::AutoVectorizingCopyWithAssumedAlignment<128>;
    using SmemToRegCopy          = cute::Copy_Atom<SmemToRegCopyOperation, EpilogueElement>;
    using RegToSmemCopy          = cute::Copy_Atom<RegToSmemCopyOperation, ElementOutput>;
    using SmemToGmemTmaCopy =
        decltype(cute::make_tma_copy(cute::SM90_TMA_STORE{},
                                     cute::make_tensor(cute::make_gmem_ptr(static_cast<ElementOutput *>(nullptr)),
                                                       cute::make_shape(cute::Int<BlkM>{}, cute::Int<BlkN>{}),
                                                       GmemStride{}),
                                     OutputSmemLayout{},
                                     cute::make_shape(cute::Int<BlkM>{}, cute::Int<BlkN>{}),
                                     cute::Int<1>{}));
    using SmemToGmemVectorCopy = decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
                                          VectorizedCopyAtom<ElementOutput, SmemToGmemAlignmentElements>,
                                          ThreadCount,
                                          SmemToGmemAlignmentElements,
                                          GmemStride,
                                          cute::Int<BlkM>,
                                          cute::Int<BlkN>>());
    using SmemToGmemCopy       = cute::conditional_t<UsesTmaStore, SmemToGmemTmaCopy, SmemToGmemVectorCopy>;

    using GlobalToSharedCopy       = GmemToSmemCopy;
    using TensorMemoryToSharedCopy = TmemToSmemCopy;
    using SharedToRegisterCopy     = SmemToRegCopy;
    using RegisterToSharedCopy     = RegToSmemCopy;
    using SharedToGlobalCopy       = SmemToGmemCopy;
    using SharedToRegisterLayout   = SmemLayout;
    using RegisterToSharedLayout   = OutputSmemLayout;
    using SharedToGlobalLayout     = OutputSmemLayout;
};

// =====================================================================================
// 4. SM120 特化与前瞻 (FP8/FP4 与消费级 Blackwell)
// =====================================================================================

template <class Element,
          class GmemStride,
          class TileShape_MNK,
          int ThreadCount,
          int GmemAlignmentBytes,
          class ClusterShape,
          bool IsRoleA>
struct Sm120TensorOpMainloopRole
{
    // [极低精度量化屏障]
    static_assert(IsSm120TensorOpElement<Element>::value, "SM120 TensorOp example path currently targets FP8 inputs.");

    using ElementInput                            = Element;
    using ClusterShape_MNK                        = ClusterShape;
    static constexpr int GmemToSmemAlignmentBytes = GmemAlignmentBytes;

    using ElementMma =
        decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<Element>());

    // FP8 运算在底层寄存器载荷上采用 uint8_t 无符号字节作为数据流转实体
    using SmemAllocElement = uint8_t;

    static constexpr int BlkMN = IsRoleA ? cute::size<0>(TileShape_MNK{}) : cute::size<1>(TileShape_MNK{});
    static constexpr int BlkK  = cute::size<2>(TileShape_MNK{});

    // 针对超低精度的 Register-to-Register (rr) 共享内存选取策略
    using SmemLayoutAtom =
        decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<SmemAllocElement, cute::Int<BlkK>>());
    using SmemLayout =
        decltype(cute::tile_to_shape(SmemLayoutAtom{}, cute::Shape<cute::Int<BlkMN>, cute::Int<BlkK>>{}));

    using ScaleElement                   = float;
    static constexpr int ScaleVectorSize = 32;
    static constexpr int ScaleKBlocks    = (BlkK + ScaleVectorSize - 1) / ScaleVectorSize;
    using ScaleSmemLayout                = cute::Layout<cute::Shape<cute::Int<BlkMN>, cute::Int<ScaleKBlocks>>>;

    // 对于极细粒度的 FP8 混合扩展，往往不绑定静态的 TMA 载荷，转而使用 AutoCopyAsync 让编译器根据 Scale Factor
    // 联编分配。
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

    using SmemElement = SmemAllocElement; // 对外暴漏分配类型，用于上层 Scale/Bias 的共享内存分配计算
};

template <class Element,
          class GmemStride,
          class TileShape_MNK,
          int ThreadCount,
          int GmemAlignmentBytes,
          class ClusterShape>
struct Sm120TensorOpRoleA
    : Sm120TensorOpMainloopRole<Element, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentBytes, ClusterShape, true>
{
};

template <class Element,
          class GmemStride,
          class TileShape_MNK,
          int ThreadCount,
          int GmemAlignmentBytes,
          class ClusterShape>
struct Sm120TensorOpRoleB
    : Sm120TensorOpMainloopRole<Element,
                                GmemStride,
                                TileShape_MNK,
                                ThreadCount,
                                GmemAlignmentBytes,
                                ClusterShape,
                                false>
{
};

template <class Element,
          class ElementC,
          class GmemStride,
          class TileShape_MNK,
          int ThreadCount,
          int GmemAlignmentBytes,
          class ClusterShape>
struct Sm120TensorOpRoleC
{
    static_assert(IsSm120TensorOpElement<Element>::value, "SM120 TensorOp path currently targets FP8 inputs.");

    using ElementInput     = Element;
    using ElementOutput    = ElementC;
    using ClusterShape_MNK = ClusterShape;
    using Accumulator     = float; // 哪怕是 FP8，累加依然需要使用全精度的 FP32 以保持数值稳定性
    using ElementCompute  = Accumulator;
    using EpilogueElement = Accumulator;

    // [微操：排列 Tile 约束 (Permuted Tile)]
    // 强制截断 M 和 N 的最大尺寸，防止 FP8 导致的超大规模寄存器或共享内存访问越界
    using PermTileM = decltype(cute::min(cute::size<0>(TileShape_MNK{}), cute::_128{}));
    using PermTileN = decltype(cute::min(cute::size<1>(TileShape_MNK{}), cute::_32{}));

    // 使用 Register-to-Register (rr) 专有选择器进行 FP8 混合乘加
    using MmaAtom    = cute::MMA_Atom<decltype(cute::rr_op_selector_sm120<Element, Element, Accumulator>())>;
    using AtomLayout = cute::Layout<cute::Shape<cute::_4, cute::_2, cute::_1>>;
    using TiledMma =
        decltype(cute::make_tiled_mma(MmaAtom{}, AtomLayout{}, cute::Tile<PermTileM, PermTileN, cute::_32>{}));

    static constexpr int  BlkM      = cute::size<0>(TileShape_MNK{});
    static constexpr int  BlkN      = cute::size<1>(TileShape_MNK{});
    static constexpr bool IsMnMajor = cutlass::gemm::detail::is_mn_major<GmemStride>();
    static constexpr int  Padding   = SmemPaddingElements<EpilogueElement>::value;

    using SmemLayoutAtom = cute::conditional_t<
        IsMnMajor,
        cute::Layout<cute::Shape<cute::Int<BlkM>, cute::Int<BlkN>>, cute::Stride<cute::_1, cute::Int<BlkM + Padding>>>,
        cute::Layout<cute::Shape<cute::Int<BlkM>, cute::Int<BlkN>>, cute::Stride<cute::Int<BlkN + Padding>, cute::_1>>>;
    using SmemLayout = SmemLayoutAtom;

    static constexpr int OutputAlignmentElements =
        GmemTiledCopyAlignment<ElementOutput, BlkM, BlkN, ThreadCount, IsMnMajor, GmemAlignmentBytes>::value;
    static constexpr int GmemToSmemAlignmentBytes =
        GmemTiledCopyAlignment<ElementOutput, BlkM, BlkN, ThreadCount, IsMnMajor, GmemAlignmentBytes>::bytes;

    using GmemToSmemCopy         = cute::AutoCopyAsync;
    using SmemToRegCopyOperation = cute::AutoVectorizingCopyWithAssumedAlignment<128>;
    using RegToSmemCopyOperation = cute::AutoVectorizingCopyWithAssumedAlignment<128>;
    using SmemToRegCopy          = cute::Copy_Atom<SmemToRegCopyOperation, EpilogueElement>;
    using RegToSmemCopy          = cute::Copy_Atom<RegToSmemCopyOperation, EpilogueElement>;
    using SmemToGmemCopy         = decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
                                            VectorizedCopyAtom<ElementOutput, OutputAlignmentElements>,
                                            ThreadCount,
                                            OutputAlignmentElements,
                                            GmemStride,
                                            cute::Int<BlkM>,
                                            cute::Int<BlkN>>());

    using GlobalToSharedCopy   = GmemToSmemCopy;
    using SharedToRegisterCopy = SmemToRegCopy;
    using RegisterToSharedCopy = RegToSmemCopy;
    using SharedToGlobalCopy   = SmemToGmemCopy;
};

} // namespace detail

// =====================================================================================
// 5. 编译期装载器与分发面板 (The SFINAE Dispatchers)
// =====================================================================================
// 利用 std::enable_if_t 在编译阶段根据操作级别和架构自动捕获最匹配的特化实现。

// [SM100 - SIMT 兜底分发]
template <typename Element,
          typename GmemStride,
          typename TileShape_MNK,
          int ThreadCount,
          typename ElementC,
          int GmemAlignmentA,
          int GmemAlignmentB,
          int GmemAlignmentC,
          typename ClusterShape_MNK>
struct AutoPartitioner<cutlass::arch::Sm100,
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
                       std::enable_if_t<detail::IsSm100SimtElement<Element>::value
                                        && detail::Sm100SimtNativeEligible<TileShape_MNK, ThreadCount>::value>>
{
    using RoleA = detail::Sm100SimtRoleA<Element, GmemStride, TileShape_MNK, ThreadCount>;
    using RoleB = detail::Sm100SimtRoleB<Element, GmemStride, TileShape_MNK, ThreadCount>;
    using RoleC = detail::Sm100SimtRoleC<Element, GmemStride, TileShape_MNK, ThreadCount>;
};

// [SM100 - SIMT 安全兜底]
// 不满足 SM100 官方 SIMT TileK/Thread 拓扑时，静默退到更宽容的 SM80 SIMT policy。
template <typename Element,
          typename GmemStride,
          typename TileShape_MNK,
          int ThreadCount,
          typename ElementC,
          int GmemAlignmentA,
          int GmemAlignmentB,
          int GmemAlignmentC,
          typename ClusterShape_MNK>
struct AutoPartitioner<cutlass::arch::Sm100,
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
                       std::enable_if_t<detail::IsSm100SimtElement<Element>::value
                                        && !detail::Sm100SimtNativeEligible<TileShape_MNK, ThreadCount>::value>>
{
    using RoleA = detail::Sm80SimtRoleA<Element, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentA>;
    using RoleB = detail::Sm80SimtRoleB<Element, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentB>;
    using RoleC = detail::Sm80SimtRoleC<Element, ElementC, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentC>;
};

// [SM100 - TensorOp (主战线) 分发]
template <typename Element,
          typename GmemStride,
          typename TileShape_MNK,
          int ThreadCount,
          typename ElementC,
          int GmemAlignmentA,
          int GmemAlignmentB,
          int GmemAlignmentC,
          typename ClusterShape_MNK>
struct AutoPartitioner<cutlass::arch::Sm100,
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
                       std::enable_if_t<detail::IsSm100TensorOpElement<Element>::value>>
{
    using RoleA =
        detail::Sm100TensorOpRoleA<Element, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentA, ClusterShape_MNK>;
    using RoleB =
        detail::Sm100TensorOpRoleB<Element, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentB, ClusterShape_MNK>;
    using RoleC = detail::
        Sm100TensorOpRoleC<Element, ElementC, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentC, ClusterShape_MNK>;
};

// [SM120 - 消费级架构的 SIMT 降级]
// 因为 SM120 并未完全开启 SM100 数据中心级别的 f32x2 指令集，
// SIMT 分支强行退化挂载到 Ampere (SM80) 的 policy 上，以保障泛用性。
template <typename Element,
          typename GmemStride,
          typename TileShape_MNK,
          int ThreadCount,
          typename ElementC,
          int GmemAlignmentA,
          int GmemAlignmentB,
          int GmemAlignmentC,
          typename ClusterShape_MNK>
struct AutoPartitioner<cutlass::arch::Sm120,
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
                       std::enable_if_t<detail::IsSm100SimtElement<Element>::value>>
{
    using RoleA = detail::Sm80SimtRoleA<Element, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentA>;
    using RoleB = detail::Sm80SimtRoleB<Element, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentB>;
    using RoleC = detail::Sm80SimtRoleC<Element, ElementC, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentC>;
};

// [SM120 - 常规精度 TensorOp (FP16/BF16)]
// 如果元素是标准的半精度浮点或整数，复用 SM100 的坚实底座。
template <typename Element,
          typename GmemStride,
          typename TileShape_MNK,
          int ThreadCount,
          typename ElementC,
          int GmemAlignmentA,
          int GmemAlignmentB,
          int GmemAlignmentC,
          typename ClusterShape_MNK>
struct AutoPartitioner<cutlass::arch::Sm120,
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
                       std::enable_if_t<detail::IsSm100TensorOpElement<Element>::value>>
{
    using RoleA =
        detail::Sm100TensorOpRoleA<Element, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentA, ClusterShape_MNK>;
    using RoleB =
        detail::Sm100TensorOpRoleB<Element, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentB, ClusterShape_MNK>;
    using RoleC = detail::
        Sm100TensorOpRoleC<Element, ElementC, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentC, ClusterShape_MNK>;
};

// [SM120 - 极限前沿 TensorOp (FP8/FP4)]
// 检测到极低精度类型，激活专属的 SM120 Register-to-Register 路径配置。
template <typename Element,
          typename GmemStride,
          typename TileShape_MNK,
          int ThreadCount,
          typename ElementC,
          int GmemAlignmentA,
          int GmemAlignmentB,
          int GmemAlignmentC,
          typename ClusterShape_MNK>
struct AutoPartitioner<cutlass::arch::Sm120,
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
                       std::enable_if_t<detail::IsSm120TensorOpElement<Element>::value>>
{
    using RoleA =
        detail::Sm120TensorOpRoleA<Element, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentA, ClusterShape_MNK>;
    using RoleB =
        detail::Sm120TensorOpRoleB<Element, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentB, ClusterShape_MNK>;
    using RoleC = detail::
        Sm120TensorOpRoleC<Element, ElementC, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentC, ClusterShape_MNK>;
};

} // namespace autopartition
