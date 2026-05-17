#pragma once // 确保头文件在一次编译中只被包含一次，防止重定义错误

// ------------------------------------------------------------------
// 包含必要的标准库和 CUTE/CUTLASS 核心头文件
// ------------------------------------------------------------------
#include <cstdint>     // 提供固定宽度的整数类型，如 int8_t, uint8_t, int32_t 等
#include <type_traits> // 提供编译期类型信息和类型变换（如 std::is_same, std::enable_if_t）

// CUTE 头文件：CUTE 是一个基于代数拓扑概念的 C++ 模板库，专门用于描述和操作高维 Layout 和 Tensor
#include <cute/arch/copy.hpp>      // 提供对底层 PTX 内存拷贝指令（如 ld.global, st.shared）的封装
#include <cute/atom/copy_atom.hpp> // 定义 Copy_Atom，将基础的拷贝指令与特定数据类型绑定
#include <cute/atom/copy_traits_sm75.hpp> // 包含 SM75 引入的指令特征（如 ldmatrix 用于 Tensor Core）
#include <cute/atom/copy_traits_sm80.hpp> // 包含 SM80 引入的指令特征（如 cp.async 异步拷贝）
#include <cute/atom/mma_atom.hpp>         // 定义 MMA_Atom，将矩阵乘累加指令（MMA）封装为 CUTE 原语
#include <cute/atom/mma_traits_sm80.hpp>  // 包含 SM80 Tensor Core mma.sync 指令集的特征定义
#include <cute/layout.hpp> // CUTE 的核心：Layout = Shape o Stride，描述逻辑坐标到物理索引的映射
#include <cute/tensor.hpp> // 提供对 CUTE Tensor 的支持，Tensor 是 Layout 和物理指针/引擎的结合

// CUTLASS 头文件：提供架构标签和枚举类型
#include <cutlass/arch/arch.h> // 提供 SM 架构的 Tag 标签（如 cutlass::arch::Sm80）
#include <cutlass/arch/mma.h>  // 提供操作类别的 Tag 标签（如 OpClassSimt, OpClassTensorOp）
#include <cutlass/gemm/collective/builders/sm90_common.inl> // 借用 SM90 builder 中的一些通用辅助函数（如 is_mn_major）
#include <cutlass/gemm/gemm.h>                              // CUTLASS GEMM 的核心定义
#include <cutlass/numeric_types.h> // 提供特定精度的数值类型（如 half_t, bfloat16_t, tfloat32_t）

#include "../auto_partitioner.hpp" // 引入项目上游定义的基础泛型模板 AutoPartitioner

namespace autopartition {
namespace detail {

// ==================================================================
// 1. 类型路由特征 (Type Routing Traits)
// ==================================================================
// 目的：在编译期通过 SFINAE (Substitution Failure Is Not An Error) 机制，
// 决定当前数据类型（Element）是否可以使用 SIMT（CUDA 核心）或 TensorOp（Tensor 核心）。

// 判断输入类型是否受 SM80 SIMT (FMA指令) 支持。SIMT 使用传统的标量/向量计算单元。
template <class Element>
struct IsSm80SimtElement
    : std::integral_constant<bool,                                       // 继承自标准库的编译期布尔常量
                             std::is_same<Element, float>::value ||      // 单精度浮点 (FP32)
                                 std::is_same<Element, double>::value || // 双精度浮点 (FP64)
                                 std::is_same<Element, cutlass::half_t>::value ||   // 半精度浮点 (FP16)
                                 std::is_same<Element, cutlass::bfloat16_t>::value> // 脑浮点 (BF16)
{
};

// 判断输入类型是否受 SM80 TensorOp 支持。TensorOp 依赖特定的硬件矩阵乘引擎（MMA指令）。
template <class Element>
struct IsSm80TensorOpElement
    : std::integral_constant<bool,
                             std::is_same<Element, float>::value || // FP32（底层可能会转为 TF32 执行）
                                 std::is_same<Element, cutlass::tfloat32_t>::value || // TensorFloat-32 (TF32)
                                 std::is_same<Element, cutlass::half_t>::value ||     // FP16
                                 std::is_same<Element, cutlass::bfloat16_t>::value || // BF16
                                 std::is_same<Element, int8_t>::value || // 8位有符号整数（常用于量化推理）
                                 std::is_same<Element, uint8_t>::value> // 8位无符号整数
{
};

// ==================================================================
// 2. 线程排布映射 (Thread Layout Policies)
// ==================================================================
// 目的：决定一个 CTA (Cooperative Thread Array, 即 Thread Block) 内部的线程
// 如何在逻辑 M 和 N 维度上进行二维映射。这直接影响访存合并和数据复用。

// SIMT 模式的线程布局策略
template <int TileM, int TileN, int ThreadCount> struct OptimalSimtThreadLayout
{
    // 强制限制线程数只能是 64, 128, 256。这是为了对齐常用的 Block 设定（2~8个 Warp）。
    static_assert(ThreadCount == 64 || ThreadCount == 128 || ThreadCount == 256,
                  "SM80 SIMT supports 64, 128, or 256 CTA threads.");

    // 启发式分配 M 维度的线程数 (TM)。
    // 如果总线程是 256（8个 Warp），且当前处理的 M 边长大于 N 边长，则给 M 分配 32 个线程，否则给 M 分配 16 个线程。
    // 如果总线程不是 256，则 M 维度固定分配 16 个线程。
    static constexpr int TM = (ThreadCount == 256) ? ((TileM > TileN) ? 32 : 16) : 16;

    // N 维度的线程数由总线程数除以 M 维度线程数得出。
    static constexpr int TN = ThreadCount / TM;

    // 确保线程数可以整除，保证生成的矩阵块是完整的（没有“半个”线程被分配）。
    static_assert(ThreadCount % TM == 0, "Invalid SM80 SIMT thread layout.");

    // cute::Layout 接受 Shape 作为参数。这里的 Shape 是 <TM, TN, 1>。
    // cute::Int<> 将运行期常量转为编译期类型常量，实现零运行时开销。
    using Layout = cute::Layout<cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::_1>>;
};

// TensorOp 模式的线程布局策略。TensorOp 是以 Warp (32 线程) 为基本单位调度 MMA 指令的。
template <int TileM, int TileN, int ThreadCount> struct OptimalTensorOpThreadLayout
{
    // TensorOp 操作要求线程数必须是 Warp 大小的整数倍。
    static_assert(ThreadCount % 32 == 0, "SM80 TensorOp requires whole warps.");
    static constexpr int WarpCount = ThreadCount / 32; // 计算总 Warp 数量

    // 限制 Warp 数量只能是 1, 2, 4, 8，这是为了适配后续 TiledMMA 分布式的物理限制。
    static_assert(WarpCount == 1 || WarpCount == 2 || WarpCount == 4 || WarpCount == 8,
                  "SM80 TensorOp supports 1, 2, 4, or 8 warps.");

    // 启发式 Warp 分配逻辑：将更多的 Warp 铺垫在较长的维度上。
    // 这样做的核心目的是“最大化 Shared Memory 的数据重用率”。
    static constexpr int WarpM = (WarpCount == 8) ? ((TileM >= TileN) ? 4 : 2)
                               : (WarpCount == 4) ? ((TileM >= TileN) ? 2 : 1)
                               : (WarpCount == 2) ? ((TileM >= TileN) ? 2 : 1)
                                                  : 1;
    static constexpr int WarpN = WarpCount / WarpM; // 剩余的 Warp 铺到 N 维

    // 生成二维 Warp 排布 Layout。
    using Layout = cute::Layout<cute::Shape<cute::Int<WarpM>, cute::Int<WarpN>, cute::_1>>;
};

// ==================================================================
// 3. 内存对齐与访存向量化 (Memory Alignment & Vectorization)
// ==================================================================

// 动态推导 Global Memory（显存）访问的最优向量化粒度（16B, 8B, 4B）。
template <class Element,
          int ContiguousElements,
          int MaxAlignmentBytes = 16> // ContiguousElements 指在内存中最内层连续的元素数量
struct GmemVectorAlignment
{
    static_assert(MaxAlignmentBytes == 4 || MaxAlignmentBytes == 8 || MaxAlignmentBytes == 16,
                  "Gmem alignment must be one of 4, 8, or 16 bytes.");

    static constexpr int ElementBytes = int(sizeof(Element)); // 单个元素占据的字节数

    // 计算满足 16B/8B/4B 对齐分别需要多少个元素。
    // SM80 架构下单线程单次内存事务最高效的宽度是 128-bit (16 Bytes)，常用于 cp.async 指令。
    static constexpr int Align16 = (MaxAlignmentBytes >= 16 && ElementBytes <= 16 && (16 % ElementBytes) == 0)
                                     ? (16 / ElementBytes)
                                     : 0;
    static constexpr int Align8 = (MaxAlignmentBytes >= 8 && ElementBytes <= 8 && (8 % ElementBytes) == 0)
                                    ? (8 / ElementBytes)
                                    : 0;
    static constexpr int Align4 = (MaxAlignmentBytes >= 4 && ElementBytes <= 4 && (4 % ElementBytes) == 0)
                                    ? (4 / ElementBytes)
                                    : 0;

    // 贪心策略：优先选择能被连续元素个数整除的最大对齐宽度。
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

    // 防御性断言：如果找不到合法的向量化宽度（通常说明参数配置极其恶劣或不对齐），在编译期熔断。
    static_assert(value != 0,
                  "No legal cp.async vector width for this element type, tile extent, and physical alignment.");
};

// 计算 Shared Memory（共享内存）为了防止 Bank Conflict 需要的 Padding 数量。
template <class Element> struct SmemPaddingElements
{
    // Shared memory 由 32 个 Bank 组成，每个 Bank 位宽 4 Bytes (共128 Bytes)。
    // 如果元素的行长刚好是 32 的倍数，不同行同列的元素会落在同一个 Bank，导致严重的访问串行化（Bank Conflict）。
    // 此处策略：强行在连续维度的末尾补足 (16 Bytes / 元素大小) 个占位元素，使得下一行的起始地址错开。
    static constexpr int value = (sizeof(Element) < 16) ? (16 / int(sizeof(Element))) : 1;
};

// 工具类型：使用 AutoVectorizingCopyWithAssumedAlignment 生成通用的全局向量化 Copy Atom。
template <class Element, int AlignmentElements>
using VectorizedCopyAtom =
    cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<AlignmentElements *int(sizeof(Element)) * 8>,
                    Element>;

// ==================================================================
// 4. SIMT (CUDA Core) 模式的架构定义
// ==================================================================

// 为 SIMT 提取 A 和 B 矩阵在主循环（Mainloop）中的公共访存和布局特征。
template <class Element, class GmemStride, int TileMN, int TileK, int ThreadCount, int GmemAlignmentBytes>
struct Sm80SimtMainloopRole
{
    // 判断 Global Memory 传进来的 Stride 是按 M/N 连续（Col/Row-Major）还是 K 连续。
    static constexpr bool IsMnMajor = cutlass::gemm::detail::is_mn_major<GmemStride>();
    static constexpr int  Padding   = SmemPaddingElements<Element>::value;

    // 静态构建 Shared Memory 内部的 Layout。
    // 如果是 MnMajor，对列维（Stride 的第二维）施加 Padding；反之对行维施加 Padding。
    using SmemLayoutAtom = cute::conditional_t<IsMnMajor,
                                               cute::Layout<cute::Shape<cute::Int<TileMN>, cute::Int<TileK>>,
                                                            cute::Stride<cute::_1, cute::Int<TileMN + Padding>>>,
                                               cute::Layout<cute::Shape<cute::Int<TileMN>, cute::Int<TileK>>,
                                                            cute::Stride<cute::Int<TileK + Padding>, cute::_1>>>;
    using SmemLayout     = SmemLayoutAtom;

    // 确定物理内存中连续存放的那一维的长度。
    static constexpr int ContiguousDimLength = IsMnMajor ? TileMN : TileK;

    // 全局写回 (STG) 使用的对齐宽度。
    static constexpr int VectorAlignmentElements =
        GmemVectorAlignment<Element, ContiguousDimLength, GmemAlignmentBytes>::value;
    static constexpr int VectorAlignmentBytes =
        GmemVectorAlignment<Element, ContiguousDimLength, GmemAlignmentBytes>::bytes;

    // Gmem 到 Smem 的对齐宽度与连续维度匹配，优先使用 16B cp.async，不能整除时退到 8B/4B。
    static constexpr int GmemToSmemAlignmentElements = VectorAlignmentElements;
    static constexpr int GmemToSmemAlignmentBytes    = VectorAlignmentBytes;
    using AlignmentType = cute::uint_byte_t<GmemToSmemAlignmentElements *int(sizeof(Element))>;

    // 构建 Gmem 到 Smem 的 Copy_Atom。此处采用 SM80 独有的 CP.ASYNC 硬件指令。
    // CACHEALWAYS 表示数据在全局内存经过 L2 时缓存；ZFILL 用于越界访问时自动填零。
    using GmemCopyAtom = cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<AlignmentType>, Element>;

    // 实例化一个 TiledCopy，负责调度所有线程协作完成 TileMN x TileK 数据的异步搬运。
    using TiledGmemToSmemCopy =
        decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<GmemCopyAtom,
                                                                              ThreadCount,
                                                                              GmemToSmemAlignmentElements,
                                                                              GmemStride,
                                                                              cute::Int<TileMN>,
                                                                              cute::Int<TileK>>());
    // SIMT shared layout 没有 swizzle 重解释，直接公开包含线程拓扑的 TiledCopy。
    using GmemToSmemCopy = TiledGmemToSmemCopy;

    // Shared -> Register 的数据加载策略。DefaultCopy 使得 CUTE 编译器自己选择 LDS 指令。
    using SmemToRegCopy = cute::Copy_Atom<cute::DefaultCopy, Element>;
    // Register -> Shared 的回写策略（用于 Epilogue 等场景）。
    using RegToSmemCopy = cute::Copy_Atom<cute::DefaultCopy, Element>;
    // 最终结果 Shared -> Global 写回策略，显式声明了前面推导出的最大向量化配置 VectorAlignmentElements。
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

// A 矩阵特化（M 维度参与推导）
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

// B 矩阵特化（N 维度参与推导）
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

// C 矩阵 / Accumulator 在 SIMT 模式下的定义。
template <class Element, class ElementC, class GmemStride, class TileShape_MNK, int ThreadCount, int GmemAlignmentBytes>
struct Sm80SimtRoleC
{
    static constexpr int BlkM = cute::size<0>(TileShape_MNK{});
    static constexpr int BlkN = cute::size<1>(TileShape_MNK{});

    // 采用此前定义的 SIMT 最优线程排布
    using ThreadLayout = typename OptimalSimtThreadLayout<BlkM, BlkN, ThreadCount>::Layout;

    // 定义核心计算 Atom：UniversalFMA 代表通用的 Fused Multiply-Add (a * b + c) 指令。
    using MmaAtom = cute::MMA_Atom<cute::UniversalFMA<Element, Element, Element>>;
    // 根据 ThreadLayout 将 MmaAtom 扩展为块级别的 TiledMma。
    using TiledMma = decltype(cute::make_tiled_mma(MmaAtom{}, ThreadLayout{}));

    // C 矩阵 Layout 推导过程同 A/B
    static constexpr bool IsMnMajor = cutlass::gemm::detail::is_mn_major<GmemStride>();
    static constexpr int  Padding   = SmemPaddingElements<Element>::value;
    using SmemLayoutAtom            = cute::conditional_t<
                   IsMnMajor,
                   cute::Layout<cute::Shape<cute::Int<BlkM>, cute::Int<BlkN>>, cute::Stride<cute::_1, cute::Int<BlkM + Padding>>>,
                   cute::Layout<cute::Shape<cute::Int<BlkM>, cute::Int<BlkN>>, cute::Stride<cute::Int<BlkN + Padding>, cute::_1>>>;
    using SmemLayout = SmemLayoutAtom;

    static constexpr int ContiguousDimLength = IsMnMajor ? BlkM : BlkN;
    static constexpr int AlignmentElements =
        GmemVectorAlignment<Element, ContiguousDimLength, GmemAlignmentBytes>::value;
    static constexpr int AlignmentBytes = GmemVectorAlignment<Element, ContiguousDimLength, GmemAlignmentBytes>::bytes;
    static constexpr int GmemToSmemAlignmentBytes = AlignmentBytes;
    using AlignmentType                      = cute::uint_byte_t<AlignmentElements *int(sizeof(Element))>;

    // 与 Mainloop 不同，C 矩阵的 GmemToSmem 这里强制使用了最大 AlignmentElements，
    // 因为通常 C 的搬运不在极其关键的 inner-loop 中，可以直接开足马力。
    using GmemToSmemCopy = decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
                                    cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<AlignmentType>, Element>,
                                    ThreadCount,
                                    AlignmentElements,
                                    GmemStride,
                                    cute::Int<BlkM>,
                                    cute::Int<BlkN>>());
    // Register/Shared 之间只承诺当前连续维度真正支持的向量宽度。
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

// ==================================================================
// 5. TensorOp (Tensor Core) 模式的架构定义
// ==================================================================

// 声明硬件特征映射模板
template <class Element> struct Sm80TensorOpTraits;
template <class MmaOperation, bool IsRoleA> struct MmaOperandContiguity;
template <bool NeedTranspose> struct Sm80TensorOpLdsmCopyOperation;
template <class Element,
          class MmaOperation,
          bool IsRoleA,
          bool SmemIsMnMajor,
          int AlignmentElements,
          bool UseLdMatrix>
struct Sm80TensorOpSmemCopyOperation;

// 针对 FP16 数据类型的 TensorOp 硬件原语绑定。
template <> struct Sm80TensorOpTraits<cutlass::half_t>
{
    // CUTE 提供的预封装 MMA 指令：16x8x16 规模。
    // 计算公式 D = A * B + C。
    // F32F16F16F32 含义: C(FP32) = A(FP16) * B(FP16) + C(FP32)。混合精度计算，保真度高。
    // _TN 含义：硬件指令要求输入数据 A 留在寄存器中表现为 Transpose(T，行主序)，B 表现为 Normal(N，列主序)。
    // 半精度默认使用 CUTLASS/Ampere 常用的 k16 atom；k8 atom 需要同步重配 TiledMMA 与 ldmatrix
    // 分块，不能在这里只替换 MmaOperation。
    using MmaOperation = cute::SM80_16x8x16_F32F16F16F32_TN;
    using Accumulator  = float; // 累加器强制采用 FP32
};

// 针对 BF16 数据类型（指令规模与半精度一致）
template <> struct Sm80TensorOpTraits<cutlass::bfloat16_t>
{
    using MmaOperation = cute::SM80_16x8x16_F32BF16BF16F32_TN;
    using Accumulator  = float;
};

// 针对 TF32 (TensorFloat-32) 数据类型。
template <> struct Sm80TensorOpTraits<cutlass::tfloat32_t>
{
    // 注意 K 维变成了 8（16x8x8），因为 TF32 数据更宽（32 bit）。
    using MmaOperation = cute::SM80_16x8x8_F32TF32TF32F32_TN;
    using Accumulator  = float;
};
template <> struct Sm80TensorOpTraits<float> : Sm80TensorOpTraits<cutlass::tfloat32_t>
{
};

// 针对 INT8 的量化计算。
template <> struct Sm80TensorOpTraits<int8_t>
{
    // INT8 的吞吐量极大，单条指令能算 16x8x32，采用 S32(有符号32位) 进行无损累加。
    using MmaOperation = cute::SM80_16x8x32_S32S8S8S32_TN;
    using Accumulator  = int32_t;
};

template <> struct Sm80TensorOpTraits<uint8_t>
{
    using MmaOperation = cute::SM80_16x8x32_S32U8U8S32_TN;
    using Accumulator  = int32_t;
};

template <bool IsRoleA>
struct MmaOperandContiguity<cute::SM80_16x8x16_F32F16F16F32_TN, IsRoleA>
{
    static constexpr bool RequiresMnMajor = false;
};

template <bool IsRoleA>
struct MmaOperandContiguity<cute::SM80_16x8x16_F32BF16BF16F32_TN, IsRoleA>
{
    static constexpr bool RequiresMnMajor = false;
};

template <bool IsRoleA>
struct MmaOperandContiguity<cute::SM80_16x8x8_F32TF32TF32F32_TN, IsRoleA>
{
    static constexpr bool RequiresMnMajor = false;
};

template <bool IsRoleA>
struct MmaOperandContiguity<cute::SM80_16x8x32_S32S8S8S32_TN, IsRoleA>
{
    static constexpr bool RequiresMnMajor = false;
};

template <bool IsRoleA>
struct MmaOperandContiguity<cute::SM80_16x8x32_S32U8U8S32_TN, IsRoleA>
{
    static constexpr bool RequiresMnMajor = false;
};

// ---------------- LDMATRIX 指令派发器 ----------------
// ldmatrix 只负责把 shared 中已经顺势存好的连续方向搬进寄存器。
// 当前 32x32x16 TiledMMA 需要固定的 x4 载入；gmem cp.async alignment 只影响 global->shared。
// K-major shared 与 _TN 寄存器要求一致，用 LDSM_N；MN-major shared 在载入瞬间转置，用 LDSM_T。
template <> struct Sm80TensorOpLdsmCopyOperation<false>
{
    using type = cute::SM75_U32x4_LDSM_N;
};
template <> struct Sm80TensorOpLdsmCopyOperation<true>
{
    using type = cute::SM75_U16x8_LDSM_T;
};

template <class Element, class MmaOperation, bool IsRoleA, bool SmemIsMnMajor, int AlignmentElements>
struct Sm80TensorOpSmemCopyOperation<Element, MmaOperation, IsRoleA, SmemIsMnMajor, AlignmentElements, true>
{
    static constexpr bool MmaRequiresMnMajor = MmaOperandContiguity<MmaOperation, IsRoleA>::RequiresMnMajor;
    static constexpr bool NeedTranspose      = (SmemIsMnMajor != MmaRequiresMnMajor);
    using type                               = typename Sm80TensorOpLdsmCopyOperation<NeedTranspose>::type;
};

// 非 ldmatrix 路径仍按连续维度可承诺的对齐位宽生成普通 vector copy。
template <class Element, class MmaOperation, bool IsRoleA, bool SmemIsMnMajor, int AlignmentElements>
struct Sm80TensorOpSmemCopyOperation<Element, MmaOperation, IsRoleA, SmemIsMnMajor, AlignmentElements, false>
{
    static constexpr int AlignmentBits = AlignmentElements * int(sizeof(Element)) * 8;
    static constexpr bool NeedTranspose = false;
    using type                         = cute::AutoVectorizingCopyWithAssumedAlignment<AlignmentBits>;
};


// ---------------- Swizzle (地址异或置换) Layout 生成器 ----------------
template <class Element, int TileK>
struct Sm80TensorOpSwizzleRow
{
    static constexpr int RowBytes = TileK * int(sizeof(Element));
    static constexpr int Bytes = (RowBytes >= 128 && (RowBytes % 128) == 0) ? 128
                               : (RowBytes >= 64 && (RowBytes % 64) == 0)   ? 64
                               : (RowBytes >= 32 && (RowBytes % 32) == 0)   ? 32
                                                                            : 0;
    static constexpr int Base = (Bytes == 128) ? 3 : (Bytes == 64) ? 2 : (Bytes == 32) ? 1 : 0;
    static constexpr int Elements = Bytes / int(sizeof(Element));
    static constexpr bool Supported = (Bytes != 0);
};

template <class Element, int TileMN, int TileK, bool UseLdMatrix, bool IsMnMajor> struct Sm80TensorOpSmemLayoutSelector;

// K-major shared：逻辑第二维 K 连续，cp.async 顺着 K 写，LDSM_N 原样读。
template <class Element, int TileMN, int TileK>
struct Sm80TensorOpSmemLayoutSelector<Element, TileMN, TileK, true, false>
{
    static constexpr int SwizzleBase = Sm80TensorOpSwizzleRow<Element, TileK>::Base;
    static constexpr int RowElements = Sm80TensorOpSwizzleRow<Element, TileK>::Elements;
    static_assert(Sm80TensorOpSwizzleRow<Element, TileK>::Supported,
                  "LdMatrix shared layout requires a 32, 64, or 128 byte row.");

    // Swizzle 的核心目的是解决极高带宽访存时的 Shared Memory Bank Conflict。
    // Swizzle<3, 3, 3> 表示对地址坐标的特定比特位执行异或（XOR）操作。
    // 第一个 3 (Base): XOR 开始的比特偏移。
    // 第二个 3 (Shift): XOR 对应坐标的偏移。
    // 第三个 3 (Mask): 异或涉及的比特位数为 3 位（即 8 种可能组合）。
    // 在这里配合后面的 Shape<_8, _64>，它实现的是经典的 128 Bytes 颗粒度 Swizzle (16 个 FP16 =
    // 32B，乘以跨度构成了物理上错位存储）。
    using SwizzleAtom = decltype(cute::composition(
        cute::Swizzle<SwizzleBase, 3, 3>{},
        cute::Layout<cute::Shape<cute::_8, cute::Int<RowElements>>,
                     cute::Stride<cute::Int<RowElements>, cute::_1>>{})); // 行主序核心块
    // 将 Tile 形状映射到带 Swizzle 特性的物理地址布局中
    using type = decltype(cute::tile_to_shape(SwizzleAtom{}, cute::Shape<cute::Int<TileMN>, cute::Int<TileK>>{}));
};

// MN-major shared：逻辑第一维 M/N 连续，cp.async 顺着 M/N 写，LDSM_T 在读入寄存器时完成转置。
template <class Element, int TileMN, int TileK>
struct Sm80TensorOpSmemLayoutSelector<Element, TileMN, TileK, true, true>
{
    static constexpr int SwizzleBase = Sm80TensorOpSwizzleRow<Element, TileK>::Base;
    static constexpr int RowElements = Sm80TensorOpSwizzleRow<Element, TileK>::Elements;
    static_assert(Sm80TensorOpSwizzleRow<Element, TileK>::Supported,
                  "LdMatrix shared layout requires a 32, 64, or 128 byte row.");

    // 这里保持 Global 的 M/N 连续方向不变，Shared 也按 M/N 连续落地；
    // 后续 LDSM_T 在 shared -> register 的瞬间完成 Tensor Core 所需的转置。
    using SwizzleAtom = decltype(cute::composition(
        cute::Swizzle<SwizzleBase, 3, 3>{},
        cute::Layout<cute::Shape<cute::Int<RowElements>, cute::_8>,
                     cute::Stride<cute::_1, cute::Int<RowElements>>>{}));
    using type = decltype(cute::tile_to_shape(SwizzleAtom{}, cute::Shape<cute::Int<TileMN>, cute::Int<TileK>>{}));
};

// 如果不使用 LdMatrix（例如类型是 TF32 / Int8），退回到传统的 Padding 方法规避冲突
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


// ---------------- MMA 指令的 Tile 组合器 ----------------
template <class Element, class MmaAtom, class ThreadLayout> struct Sm80TensorOpTiledMmaSelector
{
    // 默认回退：按 CUTE 内置的基础行为排布。
    using type = decltype(cute::make_tiled_mma(MmaAtom{}, ThreadLayout{}));
};

template <class MmaAtom, class ThreadLayout> struct Sm80TensorOpTiledMmaSelector<cutlass::half_t, MmaAtom, ThreadLayout>
{
    // 针对 FP16 特化：设定 Value Tile 为 32x32x16。
    // 这代表着整个 ThreadBlock 的 MMA 指令是在逻辑概念上的 32x32x16 矩阵微块层面展开协同计算。
    // 这个尺寸与之前定义的 128B Swizzle Layout 和 LDSM 四倍载入完美契合。
    using type = cute::TiledMMA<MmaAtom, ThreadLayout, cute::Tile<cute::_32, cute::_32, cute::_16>>;
};

template <class MmaAtom, class ThreadLayout>
struct Sm80TensorOpTiledMmaSelector<cutlass::bfloat16_t, MmaAtom, ThreadLayout>
{
    using type = cute::TiledMMA<MmaAtom, ThreadLayout, cute::Tile<cute::_32, cute::_32, cute::_16>>;
};


// ---------------- TensorOp A/B 矩阵通用构建协议 ----------------
template <class Element, class GmemStride, int TileMN, int TileK, int ThreadCount, bool IsRoleA, int GmemAlignmentBytes>
struct Sm80TensorOpMainloopRole
{
    static constexpr bool IsMnMajor           = cutlass::gemm::detail::is_mn_major<GmemStride>();
    static constexpr int  ContiguousDimLength = IsMnMajor ? TileMN : TileK;
    static constexpr int AlignmentElements =
        GmemVectorAlignment<Element, ContiguousDimLength, GmemAlignmentBytes>::value;
    static constexpr int AlignmentBytes = GmemVectorAlignment<Element, ContiguousDimLength, GmemAlignmentBytes>::bytes;
    static constexpr int AlignmentBits  = AlignmentBytes * 8;
    static constexpr int GmemToSmemAlignmentElements = AlignmentElements;
    static constexpr int GmemToSmemAlignmentBytes    = AlignmentBytes;

    // 判断是否具备开启极致性能 LdMatrix 指令的条件：
    // 1. 类型必须是 16bit 浮点。
    // 2. MN 维度的块长必须是 8 的倍数（LDSM_T 最小单元限制）。
    // 3. K 维度的块长必须是 64 的倍数（为了与 128 Bytes/行 的最佳 Swizzle 适配，64 个 FP16 = 128B）。
    static constexpr bool UseLdMatrix =
        (std::is_same<Element, cutlass::half_t>::value || std::is_same<Element, cutlass::bfloat16_t>::value)
        && (TileMN % 8 == 0) && Sm80TensorOpSwizzleRow<Element, TileK>::Supported;
    static constexpr int SwizzleBase = UseLdMatrix ? Sm80TensorOpSwizzleRow<Element, TileK>::Base : 0;
    using MmaOperation = typename Sm80TensorOpTraits<Element>::MmaOperation;

    // 基于上述判断抽取当前应当使用的 SmemLayout。
    using SmemLayoutAtom =
        typename Sm80TensorOpSmemLayoutSelector<Element, TileMN, TileK, UseLdMatrix, IsMnMajor>::type;
    using SmemLayout = SmemLayoutAtom;

    using AlignmentType = cute::uint_byte_t<AlignmentElements *int(sizeof(Element))>;

    using GmemCopyAtom = cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<AlignmentType>, Element>;
    using TiledGmemToSmemCopy =
        decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<GmemCopyAtom,
                                                                              ThreadCount,
                                                                              AlignmentElements,
                                                                              GmemStride,
                                                                              cute::Int<TileMN>,
                                                                              cute::Int<TileK>>());

    // 如果开启了 Swizzle (UseLdMatrix为真)，传统的 TiledCopy 的线性映射会与物理 Swizzle 冲突！
    // 必须退化回声明 AutoCopyAsync，强迫后续的 Kernel 侧使用 cute::cooperative_copy 进行编排重整。
    using GmemToSmemCopy = cute::conditional_t<UseLdMatrix, cute::AutoCopyAsync, TiledGmemToSmemCopy>;

    // 向寄存器的加载应用专门的 LDSM (ldmatrix) atom
    using SmemCopySelector =
        Sm80TensorOpSmemCopyOperation<Element, MmaOperation, IsRoleA, IsMnMajor, AlignmentElements, UseLdMatrix>;
    static constexpr bool SmemToRegNeedTranspose = SmemCopySelector::NeedTranspose;
    using SmemToRegCopyOperation                 = typename SmemCopySelector::type;
    using SmemToRegCopy = cute::Copy_Atom<SmemToRegCopyOperation, Element>;
    // 寄存器写回同样只承诺当前连续维度推导出的对齐位宽。
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

// 特化 RoleA/RoleB
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

// C矩阵 (输出累加) 特化
template <class Element, class ElementC, class GmemStride, class TileShape_MNK, int ThreadCount, int GmemAlignmentBytes>
struct Sm80TensorOpRoleC
{
    static constexpr int BlkM = cute::size<0>(TileShape_MNK{});
    static constexpr int BlkN = cute::size<1>(TileShape_MNK{});

    // C 矩阵基于 Warp 的二维分布排布 (WarpM x WarpN)
    using ThreadLayout = typename OptimalTensorOpThreadLayout<BlkM, BlkN, ThreadCount>::Layout;

    // 引入前面特化好的 TensorOp 硬件原子操作
    using MmaAtom = cute::MMA_Atom<typename Sm80TensorOpTraits<Element>::MmaOperation>;
    // 确定计算过程中的累加类型（如 FP16 计算，累加类型可能是 FP32 避免溢出）
    using ElementInput   = Element;
    using ElementCompute = typename Sm80TensorOpTraits<Element>::Accumulator;
    using ElementOutput  = ElementC;
    using Accumulator    = ElementCompute;
    using EpilogueElement = ElementCompute;
    using OutputElement   = ElementOutput;

    // 将 MmaAtom 和 ThreadLayout 混合构成块级的 TiledMMA
    using TiledMma = typename Sm80TensorOpTiledMmaSelector<Element, MmaAtom, ThreadLayout>::type;

    // C 矩阵的共享内存推导：不引入复杂的 Swizzle（因为 C 的写回往往不再参与核心的极限迭代）。
    // 依然靠加 Padding 防护冲突。
    static constexpr bool IsMnMajor = cutlass::gemm::detail::is_mn_major<GmemStride>();
    static constexpr int  Padding   = SmemPaddingElements<EpilogueElement>::value;
    using SmemLayoutAtom            = cute::conditional_t<
                   IsMnMajor,
                   cute::Layout<cute::Shape<cute::Int<BlkM>, cute::Int<BlkN>>, cute::Stride<cute::_1, cute::Int<BlkM + Padding>>>,
                   cute::Layout<cute::Shape<cute::Int<BlkM>, cute::Int<BlkN>>, cute::Stride<cute::Int<BlkN + Padding>, cute::_1>>>;
    using SmemLayout = SmemLayoutAtom;

    static constexpr int ContiguousDimLength = IsMnMajor ? BlkM : BlkN;
    static constexpr int EpilogueAlignmentElements =
        GmemVectorAlignment<EpilogueElement, ContiguousDimLength, 16>::value;
    static constexpr int EpilogueAlignmentBits =
        EpilogueAlignmentElements * int(sizeof(EpilogueElement)) * 8;
    static constexpr int AlignmentElements = EpilogueAlignmentElements;
    static constexpr int AlignmentBits     = EpilogueAlignmentBits;

    // 后处理的各类 Copy 设置，按 C 的连续维度推导实际可承诺的向量宽度。
    using SmemToRegCopyOperation = cute::AutoVectorizingCopyWithAssumedAlignment<EpilogueAlignmentBits>;
    using RegToSmemCopyOperation = cute::AutoVectorizingCopyWithAssumedAlignment<EpilogueAlignmentBits>;
    using SmemToRegCopy          = cute::Copy_Atom<SmemToRegCopyOperation, EpilogueElement>;
    using RegToSmemCopy          = cute::Copy_Atom<RegToSmemCopyOperation, EpilogueElement>;

    static constexpr int OutputAlignmentElements =
        GmemVectorAlignment<ElementOutput, ContiguousDimLength, GmemAlignmentBytes>::value;
    static constexpr int OutputAlignmentBytes =
        GmemVectorAlignment<ElementOutput, ContiguousDimLength, GmemAlignmentBytes>::bytes;
    static constexpr int OutputAlignmentBits = OutputAlignmentBytes * 8;
    static constexpr int GmemToSmemAlignmentBytes = OutputAlignmentBytes;
    using OutputAlignmentType = cute::uint_byte_t<OutputAlignmentBytes>;

    using GmemToSmemCopy = decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
                                    cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<OutputAlignmentType>,
                                                    ElementOutput>,
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
    using SmemToGmemCopy = OutputSmemToGmemCopy;

    using GlobalToSharedCopy   = GmemToSmemCopy;
    using SharedToRegisterCopy = SmemToRegCopy;
    using RegisterToSharedCopy = RegToSmemCopy;
    using SharedToGlobalCopy   = SmemToGmemCopy;
};

} // namespace detail

// ==================================================================
// 6. SFINAE 外部偏特化装载器 (The Dispatcher)
// ==================================================================
// 将上述详细定义好的内部部件，根据编译期的特征（SM版本、Op类别、数据类型有效性）
// 映射到全局通用接口 AutoPartitioner 上。

// SIMT 分支装载。仅当 OpClass 为 OpClassSimt 且该元素支持 SIMT 时匹配该模板特化。
template <typename Element,
          typename GmemStride,
          typename TileShape_MNK,
          int ThreadCount,
          typename ElementC,
          int GmemAlignmentA,
          int GmemAlignmentB,
          int GmemAlignmentC>
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
                       std::enable_if_t<detail::IsSm80SimtElement<Element>::value>>
{
    using RoleA = detail::Sm80SimtRoleA<Element, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentA>;
    using RoleB = detail::Sm80SimtRoleB<Element, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentB>;
    using RoleC = detail::Sm80SimtRoleC<Element, ElementC, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentC>;
};

// TensorOp 分支装载。仅当 OpClass 为 OpClassTensorOp 且该元素支持 TensorOp 时匹配。
template <typename Element,
          typename GmemStride,
          typename TileShape_MNK,
          int ThreadCount,
          typename ElementC,
          int GmemAlignmentA,
          int GmemAlignmentB,
          int GmemAlignmentC>
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
                       std::enable_if_t<detail::IsSm80TensorOpElement<Element>::value>>
{
    using RoleA = detail::Sm80TensorOpRoleA<Element, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentA>;
    using RoleB = detail::Sm80TensorOpRoleB<Element, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentB>;
    using RoleC = detail::Sm80TensorOpRoleC<Element, ElementC, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentC>;
};

} // namespace autopartition
