#pragma once

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

namespace autopartition
{
namespace detail
{
/**
 * @brief Type trait to check if a given element type is supported for SM80 SIMT operations.
 *
 * @tparam Element The data type to check (e.g., float, double, cutlass::half_t, or cutlass::bfloat16_t).
 */
template <class Element>
struct IsSm80SimtElement
    : std::integral_constant<bool, std::is_same<Element, float>::value || std::is_same<Element, double>::value ||
                                       std::is_same<Element, cutlass::half_t>::value ||
                                       std::is_same<Element, cutlass::bfloat16_t>::value>
{
};
/**
 * @brief Type trait to check if a given element type is supported for SM80 TensorOp (MMA) operations.
 *
 * @tparam Element The data type to check (e.g., cutlass::tfloat32_t, int8_t, cutlass::half_t, etc.).
 */
template <class Element>
struct IsSm80TensorOpElement
    : std::integral_constant<bool, std::is_same<Element, double>::value || std::is_same<Element, float>::value ||
                                       std::is_same<Element, cutlass::tfloat32_t>::value ||
                                       std::is_same<Element, cutlass::half_t>::value ||
                                       std::is_same<Element, cutlass::bfloat16_t>::value ||
                                       std::is_same<Element, int8_t>::value || std::is_same<Element, uint8_t>::value>
{
};
/**
 * @brief Computes the absolute value of an integer at compile time.
 *
 * @param value The input integer.
 * @return constexpr int The absolute value of the input.
 */
constexpr int sm80_constexpr_abs(int value)
{
    return value < 0 ? -value : value;
}
/**
 * @brief Determines the optimal SIMT thread layout (Thread M x Thread N) for a given CTA tile size.
 * It evaluates candidate layouts based on squareness and tile shape alignment to minimize cost.
 *
 * @tparam TileM The M extent of the threadblock tile.
 * @tparam TileN The N extent of the threadblock tile.
 * @tparam ThreadCount The total number of threads in the threadblock (e.g., 64, 128, 256).
 * @tparam CIsMnMajor Boolean flag indicating if the C matrix layout is M-major or N-major.
 */
template <int TileM, int TileN, int ThreadCount, bool CIsMnMajor = (TileM >= TileN)> struct OptimalSimtThreadLayout
{
    static_assert(ThreadCount == 64 || ThreadCount == 128 || ThreadCount == 256,
                  "SM80 SIMT supports 64, 128, or 256 CTA threads.");
    // 连续维度至少需要的线程数
    static constexpr int ContinuousLaneTarget = CIsMnMajor ? ((TileM < 32) ? TileM : 32) : ((TileN < 32) ? TileN : 32);
    // 判断线程布局合法性：能整除，达到ContinuousLaneTarget目标
    static constexpr bool is_candidate_legal(int tm)
    {
        if (tm <= 0 || ThreadCount % tm != 0)
        {
            return false;
        }
        int tn = ThreadCount / tm;
        if (tm > TileM || tn > TileN)
        {
            return false;
        }
        if ((TileM % tm) != 0 || (TileN % tn) != 0)
        {
            return false;
        }
        return CIsMnMajor ? (tm >= ContinuousLaneTarget) : (tn >= ContinuousLaneTarget);
    }
    // 给候选布局打分
    static constexpr int candidate_cost(int tm)
    {
        int tn = ThreadCount / tm;
        // 线程布局尽量接近正方形
        int square_cost = sm80_constexpr_abs(tm - tn) * 64; // 64为代价权重
        // 每个线程负责的 M/N 子块尽量接近正方形
        int tile_cost = sm80_constexpr_abs((TileM / tm) - (TileN / tn));
        return square_cost + tile_cost;
    }
    // 遍历所有布局，找出最优布局
    static constexpr int select_tm()
    {
        int best_tm = 0;
        int best_score = 1 << 30;
        for (int tm = 1; tm <= ThreadCount; tm *= 2)
        {
            if (is_candidate_legal(tm))
            {
                int score = candidate_cost(tm);
                if (score < best_score)
                {
                    best_score = score;
                    best_tm = tm;
                }
            }
        }
        return best_tm;
    }

    static constexpr int TM = select_tm();
    static constexpr int TN = (TM == 0) ? 0 : ThreadCount / TM;

    static_assert(TM != 0, "No legal SM80 SIMT thread layout for this tile, thread count, and C layout.");
    static_assert(ThreadCount % TM == 0, "Invalid SM80 SIMT thread layout.");
    // 线程布局
    using Layout = cute::Layout<cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::_1>>;
};

/**
 * @brief Determines the optimal warp layout (Warp M x Warp N) for SM80 TensorOp MMA operations.
 * Calculates the most efficient distribution of warps across the M and N dimensions based on atom size.
 *
 * @tparam TileM The M extent of the threadblock tile.
 * @tparam TileN The N extent of the threadblock tile.
 * @tparam ThreadCount The total number of threads in the threadblock (must be a multiple of 32 for whole warps).
 * @tparam MmaOperation The specific CuTe MMA atom operation mapped to hardware (e.g.,
 * cute::SM80_16x8x16_F32F16F16F32_TN).
 */
template <int TileM, int TileN, int ThreadCount, class MmaOperation = cute::SM80_16x8x16_F32F16F16F32_TN>
struct OptimalTensorOpThreadLayout
{
    static_assert(ThreadCount % 32 == 0, "SM80 TensorOp requires whole warps.");
    static constexpr int WarpCount = ThreadCount / 32;

    static_assert(WarpCount == 1 || WarpCount == 2 || WarpCount == 4 || WarpCount == 8,
                  "SM80 TensorOp supports 1, 2, 4, or 8 warps.");
    // 获取 MMA atom 的shape
    using AtomShape = typename cute::MMA_Traits<MmaOperation>::Shape_MNK;
    static constexpr int AtomM = cute::size<0>(AtomShape{});
    static constexpr int AtomN = cute::size<1>(AtomShape{});
    // 判断warp候选是否合法，warp能否整除，warp内线程能否被atom整除
    static constexpr bool is_candidate_legal(int warp_m)
    {
        if (warp_m <= 0 || WarpCount % warp_m != 0)
        {
            return false;
        }
        int warp_n = WarpCount / warp_m;
        if (warp_m > TileM || warp_n > TileN)
        {
            return false;
        }
        if ((TileM % warp_m) != 0 || (TileN % warp_n) != 0)
        {
            return false;
        }
        int warp_tile_m = TileM / warp_m;
        int warp_tile_n = TileN / warp_n;
        return (warp_tile_m % AtomM) == 0 && (warp_tile_n % AtomN) == 0;
    }
    // 给候选 warp layout 打分
    static constexpr int candidate_cost(int warp_m)
    {
        int warp_n = WarpCount / warp_m;
        int warp_tile_m = TileM / warp_m;
        int warp_tile_n = TileN / warp_n;
        int repeat_m = warp_tile_m / AtomM;
        int repeat_n = warp_tile_n / AtomN;
        int max_repeat = (repeat_m > repeat_n) ? repeat_m : repeat_n;

        int repeat_balance_cost = sm80_constexpr_abs(repeat_m - repeat_n) *
                                  1024; // 每个 warp 内 MMA atom 的两个维度的重复次数应该尽可能相近，权重最高
        int warp_square_cost = sm80_constexpr_abs(warp_tile_m - warp_tile_n) * 8; // 每个 warp 负责的 tile 形状
        int pressure_cost = max_repeat * 32 + repeat_m * repeat_n; // 惩罚某个方向 repeat 太长+惩罚总的 atom 数量
        int layout_cost = sm80_constexpr_abs(warp_m - warp_n);     // warp layout正
        return repeat_balance_cost + warp_square_cost + pressure_cost + layout_cost;
    }
    // 选择最优warp_m
    static constexpr int select_warp_m()
    {
        int best_warp_m = 0;
        int best_score = 1 << 30;
        for (int warp_m = 1; warp_m <= WarpCount; warp_m *= 2)
        {
            if (is_candidate_legal(warp_m))
            {
                int score = candidate_cost(warp_m);
                if (score < best_score)
                {
                    best_score = score;
                    best_warp_m = warp_m;
                }
            }
        }
        return best_warp_m;
    }

    static constexpr int WarpM = select_warp_m();
    static constexpr int WarpN = (WarpM == 0) ? 0 : WarpCount / WarpM;
    static constexpr int WarpTileM = (WarpM == 0) ? 0 : TileM / WarpM;
    static constexpr int WarpTileN = (WarpN == 0) ? 0 : TileN / WarpN;
    static constexpr int RepeatM = (WarpTileM == 0) ? 0 : WarpTileM / AtomM;
    static constexpr int RepeatN = (WarpTileN == 0) ? 0 : WarpTileN / AtomN;

    static_assert(WarpM != 0, "No legal SM80 TensorOp warp layout for this tile, thread count, and MMA atom.");
    using Layout = cute::Layout<cute::Shape<cute::Int<WarpM>, cute::Int<WarpN>, cute::_1>>;
};

/**
 * @brief Calculates the optimal global memory vector alignment in elements based on physical alignment constraints.
 *
 * @tparam Element The data type of the elements being accessed.
 * @tparam ContiguousElements The number of contiguous elements in the memory layout.
 * @tparam MaxAlignmentBytes The maximum allowed physical memory alignment in bytes (usually 16, 8, or 4).
 */
template <class Element, int ContiguousElements, int MaxAlignmentBytes = 16> struct GmemVectorAlignment
{
    static_assert(MaxAlignmentBytes == 4 || MaxAlignmentBytes == 8 || MaxAlignmentBytes == 16,
                  "Gmem alignment must be one of 4, 8, or 16 bytes.");

    static constexpr int ElementBytes = int(sizeof(Element));

    static constexpr int Align16 =
        (MaxAlignmentBytes >= 16 && ElementBytes <= 16 && (16 % ElementBytes) == 0) ? (16 / ElementBytes) : 0;
    static constexpr int Align8 =
        (MaxAlignmentBytes >= 8 && ElementBytes <= 8 && (8 % ElementBytes) == 0) ? (8 / ElementBytes) : 0;
    static constexpr int Align4 =
        (MaxAlignmentBytes >= 4 && ElementBytes <= 4 && (4 % ElementBytes) == 0) ? (4 / ElementBytes) : 0;

    static constexpr int value = [] {
        if constexpr (Align16 != 0)
        {
            if constexpr ((ContiguousElements % Align16) == 0)
            {
                return Align16;
            }
        }
        if constexpr (Align8 != 0)
        {
            if constexpr ((ContiguousElements % Align8) == 0)
            {
                return Align8;
            }
        }
        if constexpr (Align4 != 0)
        {
            if constexpr ((ContiguousElements % Align4) == 0)
            {
                return Align4;
            }
        }
        return 0;
    }();
    static constexpr int bytes = value * ElementBytes;

    static_assert(
        value != 0,
        "No legal cp.async vector width for this element type, tile extent, and physical alignment."); // 只看数据本身允许的最大vector
                                                                                                       // load
};

/**
 * @brief Checks if a specified vector alignment is mathematically valid for a SIMT global memory tiled copy operation.
 * Ensures that the thread layout can evenly divide the memory extents given the proposed alignment.
 *
 * @tparam AlignmentElements The proposed vector alignment in terms of element count.
 * @tparam ThreadCount The total number of threads participating in the copy.
 * @tparam TileMN The extent of the major (contiguous) dimension.
 * @tparam TileK The extent of the minor dimension.
 * @tparam IsMnMajor Boolean flag indicating if the layout is MN-major (true) or K-major (false).
 */
template <int AlignmentElements, int ThreadCount, int TileMN, int TileK, bool IsMnMajor>
struct IsLegalSimtGmemTiledCopyAlignment
{
    static constexpr bool value = [] {
        if constexpr (AlignmentElements == 0)
        {
            return false;
        }
        else
        {
            constexpr int MajorExtent = IsMnMajor ? TileMN : TileK;
            constexpr int MinorExtent = IsMnMajor ? TileK : TileMN;
            constexpr int MajorThreads =
                (ThreadCount >= MajorExtent / AlignmentElements) ? (MajorExtent / AlignmentElements) : ThreadCount;
            if constexpr (MajorThreads <= 0)
            {
                return false;
            }
            else if constexpr ((ThreadCount % MajorThreads) != 0)
            {
                return false;
            }
            else
            {
                constexpr int MinorThreads = ThreadCount / MajorThreads;
                return (MinorThreads == 0) || ((MinorExtent % MinorThreads) == 0);
            }
        }
    }();
};

/**
 * @brief Determines the optimal and legally compliant alignment for global memory tiled copies.
 * Combines physical byte alignment limits with SIMT thread layout legality checks.
 *
 * @tparam Element The data type of the elements being copied.
 * @tparam TileMN The spatial extent of the dimension (M or N).
 * @tparam TileK The extent of the K dimension.
 * @tparam ThreadCount The total number of threads executing the copy.
 * @tparam IsMnMajor Boolean flag indicating if the contiguous dimension is MN (true) or K (false).
 * @tparam MaxAlignmentBytes The maximum physical memory alignment allowed in bytes (default 16).
 */
template <class Element, int TileMN, int TileK, int ThreadCount, bool IsMnMajor, int MaxAlignmentBytes = 16>
struct GmemTiledCopyAlignment
{
    static_assert(MaxAlignmentBytes == 4 || MaxAlignmentBytes == 8 || MaxAlignmentBytes == 16,
                  "Gmem alignment must be one of 4, 8, or 16 bytes.");

    static constexpr int ElementBytes = int(sizeof(Element));
    static constexpr int ContiguousElements = IsMnMajor ? TileMN : TileK;
    static constexpr int Align16 =
        (MaxAlignmentBytes >= 16 && ElementBytes <= 16 && (16 % ElementBytes) == 0) ? (16 / ElementBytes) : 0;
    static constexpr int Align8 =
        (MaxAlignmentBytes >= 8 && ElementBytes <= 8 && (8 % ElementBytes) == 0) ? (8 / ElementBytes) : 0;
    static constexpr int Align4 =
        (MaxAlignmentBytes >= 4 && ElementBytes <= 4 && (4 % ElementBytes) == 0) ? (4 / ElementBytes) : 0;

    static constexpr int value = [] {
        if constexpr (Align16 != 0)
        {
            if constexpr ((ContiguousElements % Align16) == 0 &&
                          IsLegalSimtGmemTiledCopyAlignment<Align16, ThreadCount, TileMN, TileK, IsMnMajor>::value)
            {
                return Align16;
            }
        }
        if constexpr (Align8 != 0)
        {
            if constexpr ((ContiguousElements % Align8) == 0 &&
                          IsLegalSimtGmemTiledCopyAlignment<Align8, ThreadCount, TileMN, TileK, IsMnMajor>::value)
            {
                return Align8;
            }
        }
        if constexpr (Align4 != 0)
        {
            if constexpr ((ContiguousElements % Align4) == 0 &&
                          IsLegalSimtGmemTiledCopyAlignment<Align4, ThreadCount, TileMN, TileK, IsMnMajor>::value)
            {
                return Align4;
            }
        }
        return 0;
    }();
    static constexpr int bytes = value * ElementBytes;

    static_assert(value != 0,
                  "No legal gmem tiled-copy vector width for this element type, tile extent, thread layout, and "
                  "physical alignment."); // 数据本身允许，并且 ThreadCount + TileShape 下，make_simt_gmem_tiled_copy
                                          // 也能合法分配线程
};

/**
 * @brief Computes the number of padding elements needed in shared memory to avoid bank conflicts.
 * Typically pads up to a 16-byte boundary.
 *
 * @tparam Element The data type of the elements stored in shared memory.
 */
template <class Element> struct SmemPaddingElements
{
    static constexpr int value = (sizeof(Element) < 16) ? (16 / int(sizeof(Element))) : 1;
};

/**
 * @brief Alias for an automatically vectorizing copy atom configured with an assumed memory alignment.
 *
 * @tparam Element The data type being copied.
 * @tparam AlignmentElements The memory alignment guaranteed, expressed in the number of elements.
 */
template <class Element, int AlignmentElements>
using VectorizedCopyAtom =
    cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<AlignmentElements *int(sizeof(Element)) * 8>,
                    Element>; // 假定 N bit 对齐,自动选择合适的向量化 load/store 指令
/**
 * @brief Defines the shared and global memory layout, alignment, and copy operations for an SM80 SIMT mainloop.
 * Acts as a base configuration for specific matrix roles (A or B).
 *
 * @tparam Element The data type of the matrix tile.
 * @tparam GmemStride The layout stride of the matrix in global memory.
 * @tparam TileMN The M or N extent of the threadblock tile.
 * @tparam TileK The K extent of the threadblock tile.
 * @tparam ThreadCount The number of threads in the threadblock.
 * @tparam GmemAlignmentBytes The guaranteed global memory alignment in bytes.
 */
template <class Element, class GmemStride, int TileMN, int TileK, int ThreadCount, int GmemAlignmentBytes>
struct Sm80SimtMainloopRole
{
    static constexpr bool IsMnMajor = cutlass::gemm::detail::is_mn_major<GmemStride>();
    static constexpr int Padding = SmemPaddingElements<Element>::value;

    using SmemLayoutAtom = cute::conditional_t<IsMnMajor,
                                               cute::Layout<cute::Shape<cute::Int<TileMN>, cute::Int<TileK>>,
                                                            cute::Stride<cute::_1, cute::Int<TileMN + Padding>>>,
                                               cute::Layout<cute::Shape<cute::Int<TileMN>, cute::Int<TileK>>,
                                                            cute::Stride<cute::Int<TileK + Padding>, cute::_1>>>;
    using SmemLayout = SmemLayoutAtom;

    static constexpr int ContiguousDimLength = IsMnMajor ? TileMN : TileK;
    // 最小搬运颗粒度元素数
    static constexpr int VectorAlignmentElements =
        GmemTiledCopyAlignment<Element, TileMN, TileK, ThreadCount, IsMnMajor, GmemAlignmentBytes>::value;
    // 最小搬运颗粒度字节数
    static constexpr int VectorAlignmentBytes =
        GmemTiledCopyAlignment<Element, TileMN, TileK, ThreadCount, IsMnMajor, GmemAlignmentBytes>::bytes;

    static constexpr int GmemToSmemAlignmentElements = VectorAlignmentElements;
    static constexpr int GmemToSmemAlignmentBytes = VectorAlignmentBytes;
    using AlignmentType = cute::uint_byte_t<GmemToSmemAlignmentBytes>;
    // 采用zero fill的cp.acync原子搬运
    using GmemCopyAtom = cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<AlignmentType>, Element>;
    // 根据 copy atom、线程数、向量化宽度、global layout stride、tile shape，自动构造一个 CTA 级 global → shared 的
    // tiled copy 对象
    using TiledGmemToSmemCopy =
        decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<GmemCopyAtom, ThreadCount,
                                                                              GmemToSmemAlignmentElements, GmemStride,
                                                                              cute::Int<TileMN>, cute::Int<TileK>>());
    using GmemToSmemCopy = TiledGmemToSmemCopy;

    using SmemToRegCopy = cute::Copy_Atom<cute::DefaultCopy, Element>;
    using RegToSmemCopy = cute::Copy_Atom<cute::DefaultCopy, Element>;
    // S2G不能使用cp.async
    using SmemToGmemCopy = decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
                                    VectorizedCopyAtom<Element, VectorAlignmentElements>, ThreadCount,
                                    VectorAlignmentElements, GmemStride, cute::Int<TileMN>, cute::Int<TileK>>());

    using GlobalToSharedCopy = GmemToSmemCopy;
    using SharedToRegisterCopy = SmemToRegCopy;
    using RegisterToSharedCopy = RegToSmemCopy;
    using SharedToGlobalCopy = SmemToGmemCopy;
};

/**
 * @brief Specializes the SIMT mainloop data movement configuration role for operand A.
 *
 * @tparam Element The data type of operand A.
 * @tparam GmemStride The layout stride of operand A in global memory.
 * @tparam TileShape_MNK The overall threadblock tile shape (M, N, K).
 * @tparam ThreadCount The number of threads in the threadblock.
 * @tparam GmemAlignmentBytes The guaranteed global memory alignment in bytes.
 */
template <class Element, class GmemStride, class TileShape_MNK, int ThreadCount, int GmemAlignmentBytes>
struct Sm80SimtRoleA : Sm80SimtMainloopRole<Element, GmemStride, cute::size<0>(TileShape_MNK{}),
                                            cute::size<2>(TileShape_MNK{}), ThreadCount, GmemAlignmentBytes>
{
};

/**
 * @brief Specializes the SIMT mainloop data movement configuration role for operand B.
 *
 * @tparam Element The data type of operand B.
 * @tparam GmemStride The layout stride of operand B in global memory.
 * @tparam TileShape_MNK The overall threadblock tile shape (M, N, K).
 * @tparam ThreadCount The number of threads in the threadblock.
 * @tparam GmemAlignmentBytes The guaranteed global memory alignment in bytes.
 */
template <class Element, class GmemStride, class TileShape_MNK, int ThreadCount, int GmemAlignmentBytes>
struct Sm80SimtRoleB : Sm80SimtMainloopRole<Element, GmemStride, cute::size<1>(TileShape_MNK{}),
                                            cute::size<2>(TileShape_MNK{}), ThreadCount, GmemAlignmentBytes>
{
};

/**
 * @brief Defines the layout, threading, and copy operations for the output matrix (Role C) in an SM80 SIMT kernel.
 *
 * @tparam Element The computational data type (e.g., accumulator type).
 * @tparam ElementC The final output data type to be written to global memory.
 * @tparam GmemStride The layout stride of operand C in global memory.
 * @tparam TileShape_MNK The overall threadblock tile shape (M, N, K).
 * @tparam ThreadCount The number of threads in the threadblock.
 * @tparam GmemAlignmentBytes The guaranteed global memory alignment in bytes.
 */
template <class Element, class ElementC, class GmemStride, class TileShape_MNK, int ThreadCount, int GmemAlignmentBytes>
struct Sm80SimtRoleC
{
    static constexpr int BlkM = cute::size<0>(TileShape_MNK{});
    static constexpr int BlkN = cute::size<1>(TileShape_MNK{});

    static constexpr bool IsMnMajor = cutlass::gemm::detail::is_mn_major<GmemStride>();
    static constexpr int Padding = SmemPaddingElements<Element>::value;

    using ThreadLayout = typename OptimalSimtThreadLayout<BlkM, BlkN, ThreadCount, IsMnMajor>::Layout;
    using MmaAtom = cute::MMA_Atom<cute::UniversalFMA<Element, Element, Element>>;
    using TiledMma = decltype(cute::make_tiled_mma(MmaAtom{}, ThreadLayout{}));
    using SmemLayoutAtom = cute::conditional_t<
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
    using AlignmentType = cute::uint_byte_t<AlignmentBytes>;

    using GmemToSmemCopy = decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
                                    cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<AlignmentType>, Element>,
                                    ThreadCount, AlignmentElements, GmemStride, cute::Int<BlkM>, cute::Int<BlkN>>());

    static constexpr int AlignmentBits = AlignmentBytes * 8;
    using RegToSmemCopy = cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<AlignmentBits>, Element>;
    using SmemToRegCopy = cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<AlignmentBits>, Element>;
    using SmemToGmemCopy = decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
                                    VectorizedCopyAtom<Element, AlignmentElements>, ThreadCount, AlignmentElements,
                                    GmemStride, cute::Int<BlkM>, cute::Int<BlkN>>());

    using GlobalToSharedCopy = GmemToSmemCopy;
    using SharedToRegisterCopy = SmemToRegCopy;
    using RegisterToSharedCopy = RegToSmemCopy;
    using SharedToGlobalCopy = SmemToGmemCopy;
};

template <class Element> struct Sm80TensorOpTraits;
template <class MmaOperation, bool IsRoleA> struct MmaOperandContiguity;
template <bool NeedTranspose> struct Sm80TensorOpLdsmCopyOperation;
template <class Element, class MmaOperation, bool IsRoleA, bool SmemIsMnMajor, int AlignmentElements, bool UseLdMatrix>
struct Sm80TensorOpSmemCopyOperation;
/**
 * @brief SM80 TensorOp traits specialization for FP16 (half_t) elements.
 * Maps to the 16x8x16 FP16 MMA atom accumulating in FP32.
 *
 * @tparam (Specialization for cutlass::half_t)
 */
template <> struct Sm80TensorOpTraits<cutlass::half_t>
{
    using MmaOperation = cute::SM80_16x8x16_F32F16F16F32_TN;
    using Accumulator = float;
};
/**
 * @brief SM80 TensorOp traits specialization for BF16 (bfloat16_t) elements.
 * Maps to the 16x8x16 BF16 MMA atom accumulating in FP32.
 *
 * @tparam (Specialization for cutlass::bfloat16_t)
 */
template <> struct Sm80TensorOpTraits<cutlass::bfloat16_t>
{
    using MmaOperation = cute::SM80_16x8x16_F32BF16BF16F32_TN;
    using Accumulator = float;
};
/**
 * @brief SM80 TensorOp traits specialization for TF32 (tfloat32_t) elements.
 * Maps to the 16x8x8 TF32 MMA atom accumulating in FP32.
 *
 * @tparam (Specialization for cutlass::tfloat32_t)
 */
template <> struct Sm80TensorOpTraits<cutlass::tfloat32_t>
{
    using MmaOperation = cute::SM80_16x8x8_F32TF32TF32F32_TN;
    using Accumulator = float;
};
/**
 * @brief SM80 TensorOp traits specialization for standard FP32 (float) elements.
 * Implicitly falls back to using hardware TF32 TensorOp instructions.
 *
 * @tparam (Specialization for float)
 */
template <> struct Sm80TensorOpTraits<float> : Sm80TensorOpTraits<cutlass::tfloat32_t>
{
};
/**
 * @brief SM80 TensorOp traits specialization for FP64 (double) elements.
 * Maps to the 8x8x4 FP64 MMA atom accumulating in FP64.
 *
 * @tparam (Specialization for double)
 */
template <> struct Sm80TensorOpTraits<double>
{
    using MmaOperation = cute::SM80_8x8x4_F64F64F64F64_TN;
    using Accumulator = double;
};
/**
 * @brief SM80 TensorOp traits specialization for signed 8-bit integers.
 * Maps to the 16x8x32 S8 MMA atom accumulating in S32.
 *
 * @tparam (Specialization for int8_t)
 */
template <> struct Sm80TensorOpTraits<int8_t>
{
    using MmaOperation = cute::SM80_16x8x32_S32S8S8S32_TN;
    using Accumulator = int32_t;
};
/**
 * @brief SM80 TensorOp traits specialization for unsigned 8-bit integers.
 * Maps to the 16x8x32 U8 MMA atom accumulating in S32.
 *
 * @tparam (Specialization for uint8_t)
 */
template <> struct Sm80TensorOpTraits<uint8_t>
{
    using MmaOperation = cute::SM80_16x8x32_S32U8U8S32_TN;
    using Accumulator = int32_t;
};
/**
 * @brief Determines whether a specific SM80 TensorOp MMA operation requires an MN-major shared memory layout.
 * Defaults to false unless specialized to strictly require it.
 *
 * @tparam IsRoleA Boolean flag indicating if we are querying operand A (true) or operand B (false).
 */
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
/**
 * @brief Selects the optimal ldmatrix (Load Matrix) hardware instruction based on transpose requirements.
 *
 * @tparam (Specialization for no transpose)
 */
template <> struct Sm80TensorOpLdsmCopyOperation<false>
{
    using type = cute::SM75_U32x4_LDSM_N;
};
template <> struct Sm80TensorOpLdsmCopyOperation<true>
{
    using type = cute::SM75_U16x8_LDSM_T;
};
/**
 * @brief Determines the copy operation (ldmatrix mapping) from shared memory to registers for TensorOp execution.
 * Computes if a hardware transpose is required based on MMA contiguity requirements and SMEM layout.
 *
 * @tparam Element The data type being loaded.
 * @tparam MmaOperation The target CuTe MMA instruction atom.
 * @tparam IsRoleA Flag indicating if this copy applies to operand A.
 * @tparam SmemIsMnMajor Flag indicating if the shared memory is stored in an MN-major format.
 * @tparam AlignmentElements Vectorization alignment constraint in elements.
 */
template <class Element, class MmaOperation, bool IsRoleA, bool SmemIsMnMajor, int AlignmentElements>
struct Sm80TensorOpSmemCopyOperation<Element, MmaOperation, IsRoleA, SmemIsMnMajor, AlignmentElements, true>
{
    static constexpr bool MmaRequiresMnMajor = MmaOperandContiguity<MmaOperation, IsRoleA>::RequiresMnMajor;
    // 判断shared memory 当前布局的连续方向，和 MMA atom 期望的 operand 连续方向是否一致
    static constexpr bool NeedTranspose = (SmemIsMnMajor != MmaRequiresMnMajor);
    // 选择ldmatrix版本
    using type = typename Sm80TensorOpLdsmCopyOperation<NeedTranspose>::type;
};
/**
 * @brief Determines the fallback copy operation (standard auto-vectorized copy) from shared memory to registers.
 * Used when hardware ldmatrix instructions are inapplicable.
 *
 * @tparam Element The data type being loaded.
 * @tparam MmaOperation The target CuTe MMA instruction atom.
 * @tparam IsRoleA Flag indicating if this copy applies to operand A.
 * @tparam SmemIsMnMajor Flag indicating if the shared memory is stored in an MN-major format.
 * @tparam AlignmentElements Vectorization alignment constraint in elements.
 */
template <class Element, class MmaOperation, bool IsRoleA, bool SmemIsMnMajor, int AlignmentElements>
struct Sm80TensorOpSmemCopyOperation<Element, MmaOperation, IsRoleA, SmemIsMnMajor, AlignmentElements, false>
{
    static constexpr int AlignmentBits = AlignmentElements * int(sizeof(Element)) * 8;
    static constexpr bool NeedTranspose = false;
    using type = cute::AutoVectorizingCopyWithAssumedAlignment<AlignmentBits>;
};
/**
 * @brief Computes shared memory swizzle configuration parameters to prevent bank conflicts
 * during ldmatrix instructions, categorizing based on 128, 64, or 32-byte row sizes.
 *
 * @tparam Element The data type of the stored elements.
 * @tparam TileK The K dimension extent of the tile.
 */
template <class Element, int TileK> struct Sm80TensorOpSwizzleRow
{
    static constexpr int RowBytes = TileK * int(sizeof(Element)); // 一行字节数
    static constexpr int Bytes = (RowBytes >= 128 && (RowBytes % 128) == 0) ? 128
                                 : (RowBytes >= 64 && (RowBytes % 64) == 0) ? 64
                                 : (RowBytes >= 32 && (RowBytes % 32) == 0) ? 32
                                                                            : 0; // 选择 swizzle 粒度
    static constexpr int Base = (Bytes == 128) ? 3 : (Bytes == 64) ? 2 : (Bytes == 32) ? 1 : 0;
    static constexpr int Elements = Bytes / int(sizeof(Element));
    static constexpr bool Supported = (Bytes != 0);
};
/**
 * @brief Selects the optimal shared memory layout configuration (swizzled or padded) depending on whether
 * ldmatrix instructions are utilized and the contiguous layout dimension (M/N major vs K major).
 *
 * @tparam Element The data type stored in shared memory.
 * @tparam TileMN The spatial extent (M or N).
 * @tparam TileK The K contiguous extent.
 * @tparam UseLdMatrix Flag indicating if ldmatrix swizzle layouts should be actively generated.
 * @tparam IsMnMajor Flag indicating if the layout should be primarily MN-major.
 */
template <class Element, int TileMN, int TileK, bool UseLdMatrix, bool IsMnMajor> struct Sm80TensorOpSmemLayoutSelector;

template <class Element, int TileMN, int TileK>
struct Sm80TensorOpSmemLayoutSelector<Element, TileMN, TileK, true, false>
{
    static constexpr int SwizzleBase = Sm80TensorOpSwizzleRow<Element, TileK>::Base;
    static constexpr int RowElements = Sm80TensorOpSwizzleRow<Element, TileK>::Elements;
    static_assert(Sm80TensorOpSwizzleRow<Element, TileK>::Supported,
                  "LdMatrix shared layout requires a 32, 64, or 128 byte row.");

    using SwizzleAtom = decltype(cute::composition(
        cute::Swizzle<SwizzleBase, 3, 3>{}, // 对于128bytes,16B,MS固定为3,3
        cute::Layout<cute::Shape<cute::_8, cute::Int<RowElements>>, cute::Stride<cute::Int<RowElements>, cute::_1>>{}));
    using type = decltype(cute::tile_to_shape(
        SwizzleAtom{}, cute::Shape<cute::Int<TileMN>, cute::Int<TileK>>{})); // 铺满整个 shared memory tile 的 layout
};

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

template <class Element, int TileMN, int TileK, bool IsMnMajor>
struct Sm80TensorOpSmemLayoutSelector<Element, TileMN, TileK, false, IsMnMajor>
{
    static constexpr int Padding = SmemPaddingElements<Element>::value;
    using type = cute::conditional_t<IsMnMajor,
                                     cute::Layout<cute::Shape<cute::Int<TileMN>, cute::Int<TileK>>,
                                                  cute::Stride<cute::_1, cute::Int<TileMN + Padding>>>,
                                     cute::Layout<cute::Shape<cute::Int<TileMN>, cute::Int<TileK>>,
                                                  cute::Stride<cute::Int<TileK + Padding>, cute::_1>>>;
};
/**
 * @brief Creates a generic CuTe TiledMMA operation descriptor mapping from a base MMA atom and thread layout.
 * Wraps standard dynamic layout resolution for TensorOp computation.
 *
 * @tparam Element The operational data type.
 * @tparam MmaAtom The base CuTe MMA atom pointing to the hardware instruction.
 * @tparam ThreadLayout The thread/warp configuration overlay.
 * @tparam TileM Total M dimension of the threadblock tile.
 * @tparam TileN Total N dimension of the threadblock tile.
 * @tparam TileK Total K dimension of the threadblock tile.
 */
template <class Element, class MmaAtom, class ThreadLayout, int TileM = 0, int TileN = 0, int TileK = 0>
struct Sm80TensorOpTiledMmaSelector
{
    using type = decltype(cute::make_tiled_mma(MmaAtom{}, ThreadLayout{}));
};
/**
 * @brief Specialized TiledMMA selector specifically designed for SM80 ldmatrix operations,
 * explicitly constructing the TiledMMA descriptor based on warp tile mappings and atom shape repeat counts.
 *
 * @tparam Element The operational data type.
 * @tparam MmaAtom The underlying CuTe TensorOp MMA atom.
 * @tparam ThreadLayout The multi-warp thread layout configuration.
 * @tparam TileM Total M dimension of the threadblock tile.
 * @tparam TileN Total N dimension of the threadblock tile.
 * @tparam TileK Total K dimension of the threadblock tile.
 */
template <class Element, class MmaAtom, class ThreadLayout, int TileM, int TileN, int TileK>
struct Sm80LdMatrixTiledMmaSelector
{
    using AtomShape = typename MmaAtom::Shape_MNK;
    static constexpr int AtomM = cute::size<0>(AtomShape{});
    static constexpr int AtomN = cute::size<1>(AtomShape{});
    static constexpr int AtomK = cute::size<2>(AtomShape{});
    static constexpr int WarpM = cute::size<0>(ThreadLayout{});
    static constexpr int WarpN = cute::size<1>(ThreadLayout{});
    static constexpr int WarpTileM = TileM / WarpM;
    static constexpr int WarpTileN = TileN / WarpN;
    static constexpr int RepeatM = WarpTileM / AtomM;
    static constexpr int RepeatN = WarpTileN / AtomN;

    static_assert((TileM % WarpM) == 0 && (TileN % WarpN) == 0,
                  "Selected SM80 TensorOp thread layout must evenly divide the CTA tile.");
    static_assert((WarpTileM % AtomM) == 0 && (WarpTileN % AtomN) == 0,
                  "Selected SM80 TensorOp warp tile must be exactly covered by the MMA atom.");
    // 根据warp layout和warp tile，在 CTA 内组成一个完整的 tiled MMA
    using type =
        cute::TiledMMA<MmaAtom, ThreadLayout, cute::Tile<cute::Int<WarpTileM>, cute::Int<WarpTileN>, cute::Int<AtomK>>>;
};
/**
 * @brief Base specialization of the TiledMMA selector for FP16 elements lacking explicit tile sizing context.
 *
 * @tparam MmaAtom The associated FP16 MMA atom.
 * @tparam ThreadLayout The planned multi-warp thread layout configuration.
 */
template <class MmaAtom, class ThreadLayout>
struct Sm80TensorOpTiledMmaSelector<cutlass::half_t, MmaAtom, ThreadLayout, 0, 0, 0>
{
    using type = decltype(cute::make_tiled_mma(MmaAtom{}, ThreadLayout{}));
}; //  fallback 特化
/**
 * @brief Specialization of the TiledMMA selector for FP16 elements incorporating explicit threadblock tile sizes.
 * Delegates structural computation to the custom ldmatrix TiledMMA builder.
 *
 * @tparam MmaAtom The associated FP16 MMA atom.
 * @tparam ThreadLayout The planned multi-warp thread layout configuration.
 * @tparam TileM Total M dimension of the threadblock tile.
 * @tparam TileN Total N dimension of the threadblock tile.
 * @tparam TileK Total K dimension of the threadblock tile.
 */
template <class MmaAtom, class ThreadLayout, int TileM, int TileN, int TileK>
struct Sm80TensorOpTiledMmaSelector<cutlass::half_t, MmaAtom, ThreadLayout, TileM, TileN, TileK>
    : Sm80LdMatrixTiledMmaSelector<cutlass::half_t, MmaAtom, ThreadLayout, TileM, TileN, TileK>
{
};

template <class MmaAtom, class ThreadLayout, int TileM, int TileN, int TileK>
struct Sm80TensorOpTiledMmaSelector<cutlass::bfloat16_t, MmaAtom, ThreadLayout, TileM, TileN, TileK>
    : Sm80LdMatrixTiledMmaSelector<cutlass::bfloat16_t, MmaAtom, ThreadLayout, TileM, TileN, TileK>
{
};
/**
 * @brief Defines the shared/global memory layout, alignment limits, swizzle requirements, and
 * asynchronous copy strategies (including cp.async and ldmatrix) for an SM80 TensorOp mainloop.
 *
 * @tparam Element The data type of the matrix block.
 * @tparam GmemStride The layout stride configuration of the matrix in global memory.
 * @tparam TileMN The M or N spatial extent of the threadblock tile.
 * @tparam TileK The K contiguous extent of the threadblock tile.
 * @tparam ThreadCount The total number of threads inside the executing threadblock.
 * @tparam IsRoleA Boolean flag determining if this struct represents operand A (true) or operand B (false).
 * @tparam GmemAlignmentBytes The guaranteed global memory pointer alignment in bytes.
 */
template <class Element, class GmemStride, int TileMN, int TileK, int ThreadCount, bool IsRoleA, int GmemAlignmentBytes>
struct Sm80TensorOpMainloopRole
{
    static constexpr bool IsMnMajor = cutlass::gemm::detail::is_mn_major<GmemStride>();
    static constexpr int ContiguousDimLength = IsMnMajor ? TileMN : TileK;

    static constexpr int AlignmentElements =
        GmemTiledCopyAlignment<Element, TileMN, TileK, ThreadCount, IsMnMajor, GmemAlignmentBytes>::value;
    static constexpr int AlignmentBytes =
        GmemTiledCopyAlignment<Element, TileMN, TileK, ThreadCount, IsMnMajor, GmemAlignmentBytes>::bytes;
    static constexpr int AlignmentBits = AlignmentBytes * 8;
    static constexpr int GmemToSmemAlignmentElements = AlignmentElements;
    static constexpr int GmemToSmemAlignmentBytes = AlignmentBytes;

    static constexpr bool UseLdMatrix =
        (std::is_same<Element, cutlass::half_t>::value || std::is_same<Element, cutlass::bfloat16_t>::value) &&
        (TileMN % 8 == 0) && Sm80TensorOpSwizzleRow<Element, TileK>::Supported;

    static constexpr int SwizzleBase = UseLdMatrix ? Sm80TensorOpSwizzleRow<Element, TileK>::Base : 0;
    static constexpr int SwizzleBytes = UseLdMatrix ? Sm80TensorOpSwizzleRow<Element, TileK>::Bytes : 0;
    static constexpr int SwizzleElements = UseLdMatrix ? Sm80TensorOpSwizzleRow<Element, TileK>::Elements : 0;
    using MmaOperation = typename Sm80TensorOpTraits<Element>::MmaOperation;

    using SmemLayoutAtom =
        typename Sm80TensorOpSmemLayoutSelector<Element, TileMN, TileK, UseLdMatrix, IsMnMajor>::type;
    using SmemLayout = SmemLayoutAtom;

    using AlignmentType = cute::uint_byte_t<AlignmentBytes>;
    using GmemCopyAtom = cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<AlignmentType>, Element>;
    using TiledGmemToSmemCopy =
        decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
                 GmemCopyAtom, ThreadCount, AlignmentElements, GmemStride, cute::Int<TileMN>, cute::Int<TileK>>());

    using GmemToSmemCpAsyncCopy = TiledGmemToSmemCopy;
    using GmemToSmemCopy = cute::AutoCopyAsync;

    using SmemCopySelector =
        Sm80TensorOpSmemCopyOperation<Element, MmaOperation, IsRoleA, IsMnMajor, AlignmentElements, UseLdMatrix>;
    static constexpr bool SmemToRegNeedTranspose = SmemCopySelector::NeedTranspose;
    using SmemToRegCopyOperation = typename SmemCopySelector::type; // 获取操作atom
    using SmemToRegCopy = cute::Copy_Atom<SmemToRegCopyOperation, Element>;

    using RegToSmemCopyOperation = cute::AutoVectorizingCopyWithAssumedAlignment<AlignmentBits>;
    using RegToSmemCopy = cute::Copy_Atom<RegToSmemCopyOperation, Element>;
    using SmemToGmemCopy = decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
                                    VectorizedCopyAtom<Element, AlignmentElements>, ThreadCount, AlignmentElements,
                                    GmemStride, cute::Int<TileMN>, cute::Int<TileK>>());

    // using GlobalToSharedCopy = GmemToSmemCopy;
    using GlobalToSharedCopy = GmemToSmemCpAsyncCopy; // 使用算好的最优copy布局
    using SharedToRegisterCopy = SmemToRegCopy;
    using RegisterToSharedCopy = RegToSmemCopy;
    using SharedToGlobalCopy = SmemToGmemCopy;
};
/**
 * @brief Specializes the TensorOp mainloop data movement and instruction mapping strategy for operand A.
 *
 * @tparam Element The data type of operand A.
 * @tparam GmemStride The memory layout strides for operand A.
 * @tparam TileShape_MNK The overall GEMM threadblock tile shape (M, N, K).
 * @tparam ThreadCount The total number of execution threads in the threadblock.
 * @tparam GmemAlignmentBytes The alignment of operand A's global memory pointer in bytes.
 */
template <class Element, class GmemStride, class TileShape_MNK, int ThreadCount, int GmemAlignmentBytes>
struct Sm80TensorOpRoleA
    : Sm80TensorOpMainloopRole<Element, GmemStride, cute::size<0>(TileShape_MNK{}), cute::size<2>(TileShape_MNK{}),
                               ThreadCount, true, GmemAlignmentBytes>
{
};
/**
 * @brief Specializes the TensorOp mainloop data movement and instruction mapping strategy for operand B.
 *
 * @tparam Element The data type of operand B.
 * @tparam GmemStride The memory layout strides for operand B.
 * @tparam TileShape_MNK The overall GEMM threadblock tile shape (M, N, K).
 * @tparam ThreadCount The total number of execution threads in the threadblock.
 * @tparam GmemAlignmentBytes The alignment of operand B's global memory pointer in bytes.
 */
template <class Element, class GmemStride, class TileShape_MNK, int ThreadCount, int GmemAlignmentBytes>
struct Sm80TensorOpRoleB
    : Sm80TensorOpMainloopRole<Element, GmemStride, cute::size<1>(TileShape_MNK{}), cute::size<2>(TileShape_MNK{}),
                               ThreadCount, false, GmemAlignmentBytes>
{
};

template <int Bytes> struct Sm80EpilogueSwizzleBase;
template <> struct Sm80EpilogueSwizzleBase<32>
{
    static constexpr int value = 1;
};
template <> struct Sm80EpilogueSwizzleBase<64>
{
    static constexpr int value = 2;
};
template <> struct Sm80EpilogueSwizzleBase<128>
{
    static constexpr int value = 3;
};

template <int InstructionBytes> struct Sm80EpilogueSwizzleBytes
{
    static constexpr int value = (InstructionBytes >= 128) ? 128 : (InstructionBytes >= 64) ? 64
                                                                 : (InstructionBytes >= 32) ? 32
                                                                                            : 0;
    static_assert(value != 0, "SM80 epilogue swizzle requires at least a 32-byte instruction span.");
};

template <class Element, int LogicalMajorExtent, int VectorBytes, bool IsMnMajor>
struct Sm80TensorOpEpilogueSmemLayoutSelector
{
    static constexpr int InstructionThreads = 16;
    static constexpr int InstructionBytes = InstructionThreads * VectorBytes;
    static constexpr int SwizzleBytes = Sm80EpilogueSwizzleBytes<InstructionBytes>::value;
    static constexpr int SwizzleBase = Sm80EpilogueSwizzleBase<SwizzleBytes>::value;
    static constexpr int RowElements = SwizzleBytes / int(sizeof(Element));

    static_assert((SwizzleBytes % int(sizeof(Element))) == 0,
                  "SM80 epilogue swizzle span must be element-addressable.");
    static_assert((LogicalMajorExtent % RowElements) == 0,
                  "SM80 epilogue logical row must be covered by whole swizzle atoms.");

    using SwizzleAtom = cute::conditional_t<
        IsMnMajor,
        decltype(cute::composition(
            cute::Swizzle<SwizzleBase, 3, 3>{},
            cute::Layout<cute::Shape<cute::Int<RowElements>, cute::_16>, cute::Stride<cute::_1, cute::Int<RowElements>>>{})),
        decltype(cute::composition(
            cute::Swizzle<SwizzleBase, 3, 3>{},
            cute::Layout<cute::Shape<cute::_16, cute::Int<RowElements>>, cute::Stride<cute::Int<RowElements>, cute::_1>>{}))>;
};
/**
 * @brief Defines the output layout, threading structure, accumulator tracking, and epilogue data copies
 * for the C/D matrix role in an SM80 TensorOp kernel.
 *
 * @tparam Element The incoming computational data type matching the core MMA accumulator.
 * @tparam ElementC The final outbound data type to be stored physically in global memory.
 * @tparam GmemStride The layout stride configuration of matrix C in global memory.
 * @tparam TileShape_MNK The overall GEMM threadblock tile shape (M, N, K).
 * @tparam ThreadCount The total number of execution threads in the threadblock.
 * @tparam GmemAlignmentBytes The alignment guarantee for matrix C's global memory pointer in bytes.
 */
template <class Element, class ElementC, class GmemStride, class TileShape_MNK, int ThreadCount, int GmemAlignmentBytes>
struct Sm80TensorOpRoleC
{
    static constexpr int BlkM = cute::size<0>(TileShape_MNK{});
    static constexpr int BlkN = cute::size<1>(TileShape_MNK{});
    static constexpr int BlkK = cute::size<2>(TileShape_MNK{});

    using MmaOperation = typename Sm80TensorOpTraits<Element>::MmaOperation;
    using ThreadLayoutPlan = OptimalTensorOpThreadLayout<BlkM, BlkN, ThreadCount, MmaOperation>;
    using ThreadLayout = typename ThreadLayoutPlan::Layout;
    using MmaAtom = cute::MMA_Atom<MmaOperation>;

    static constexpr int WarpM = ThreadLayoutPlan::WarpM;
    static constexpr int WarpN = ThreadLayoutPlan::WarpN;
    static constexpr int WarpTileM = ThreadLayoutPlan::WarpTileM;
    static constexpr int WarpTileN = ThreadLayoutPlan::WarpTileN;
    static constexpr int RepeatM = ThreadLayoutPlan::RepeatM;
    static constexpr int RepeatN = ThreadLayoutPlan::RepeatN;

    using ElementInput = Element;
    using ElementCompute = typename Sm80TensorOpTraits<Element>::Accumulator;
    using ElementOutput = ElementC;
    using Accumulator = ElementCompute;
    using EpilogueElement = ElementCompute;
    using OutputElement = ElementOutput;

    using TiledMmaSelector = Sm80TensorOpTiledMmaSelector<Element, MmaAtom, ThreadLayout, BlkM, BlkN, BlkK>;
    using TiledMma = typename TiledMmaSelector::type;

    static constexpr bool IsMnMajor = cutlass::gemm::detail::is_mn_major<GmemStride>();

    static constexpr int OutputAlignmentElements =
        GmemTiledCopyAlignment<ElementOutput, BlkM, BlkN, ThreadCount, IsMnMajor, GmemAlignmentBytes>::value;
    static constexpr int OutputAlignmentBytes =
        GmemTiledCopyAlignment<ElementOutput, BlkM, BlkN, ThreadCount, IsMnMajor, GmemAlignmentBytes>::bytes;
    static constexpr int OutputAlignmentBits = OutputAlignmentBytes * 8;

    static constexpr int EpilogueInstructionThreads = 16;
    static constexpr int EpilogueVectorElements = OutputAlignmentElements;
    static constexpr int EpilogueVectorBytes = OutputAlignmentBytes;
    static constexpr int EpilogueVectorBits = OutputAlignmentBits;
    static constexpr int EpilogueLogicalMajorExtent = IsMnMajor ? BlkM : BlkN;
    static constexpr int AlignmentElements = OutputAlignmentElements;
    static constexpr int AlignmentBits = OutputAlignmentBits;
    static constexpr int EpilogueAlignmentElements = OutputAlignmentElements;
    static constexpr int EpilogueAlignmentBits = OutputAlignmentBits;
    static constexpr int GmemToSmemAlignmentBytes = OutputAlignmentBytes;
    using OutputAlignmentType = cute::uint_byte_t<OutputAlignmentBytes>;

    using OutputSwizzleSelector = Sm80TensorOpEpilogueSmemLayoutSelector<ElementOutput,
                                                                         EpilogueLogicalMajorExtent,
                                                                         EpilogueVectorBytes,
                                                                         IsMnMajor>;
    static constexpr int EpilogueSwizzleBytes = OutputSwizzleSelector::SwizzleBytes;
    static constexpr int EpilogueSwizzleBase = OutputSwizzleSelector::SwizzleBase;
    using OutputSmemLayoutAtom = typename OutputSwizzleSelector::SwizzleAtom;
    using OutputSmemLayout = decltype(cute::tile_to_shape(
        OutputSmemLayoutAtom{}, cute::Shape<cute::Int<BlkM>, cute::Int<BlkN>>{}));
    using SmemLayoutAtom = OutputSmemLayoutAtom;
    using SmemLayout = OutputSmemLayout;

    using SmemToRegCopyOperation = cute::AutoVectorizingCopyWithAssumedAlignment<OutputAlignmentBits>;
    using RegToSmemCopyOperation = cute::AutoVectorizingCopyWithAssumedAlignment<OutputAlignmentBits>;
    using SmemToRegCopy = cute::Copy_Atom<SmemToRegCopyOperation, ElementOutput>;
    using RegToSmemCopy = cute::Copy_Atom<RegToSmemCopyOperation, ElementOutput>;

    using GmemToSmemCopy =
        decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
                 cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<OutputAlignmentType>, ElementOutput>,
                 ThreadCount, OutputAlignmentElements, GmemStride, cute::Int<BlkM>, cute::Int<BlkN>>());

    using OutputSmemToGmemCopy = decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
                                          VectorizedCopyAtom<ElementOutput, OutputAlignmentElements>, ThreadCount,
                                          OutputAlignmentElements, GmemStride, cute::Int<BlkM>, cute::Int<BlkN>>());
    using SmemToGmemCopy = OutputSmemToGmemCopy;

    using GlobalToSharedCopy = GmemToSmemCopy;
    using SharedToRegisterCopy = SmemToRegCopy;
    using RegisterToSharedCopy = RegToSmemCopy;
    using SharedToGlobalCopy = SmemToGmemCopy;
    using SharedToGlobalLayout = OutputSmemLayout;
};

} // namespace detail
/**
 * @brief AutoPartitioner specialization for SM80 architectures leveraging SIMT (CUDA Core) execution instructions.
 * It constructs and associates the respective RoleA, RoleB, and RoleC data movement/mapping components for a GEMM
 * kernel.
 *
 * @tparam Element The basic operational data type (must validate true for SM80 SIMT compatability).
 * @tparam GmemStride Struct characterizing layout strides mapped across global memory.
 * @tparam TileShape_MNK The overall threadblock spatial configuration shape (M, N, K).
 * @tparam ThreadCount The total thread count participating concurrently within a threadblock.
 * @tparam ElementC The resulting destination format/type written to memory.
 * @tparam GmemAlignmentA Guaranteed alignment constraints for Operand A array block.
 * @tparam GmemAlignmentB Guaranteed alignment constraints for Operand B array block.
 * @tparam GmemAlignmentC Guaranteed alignment constraints for Operand C output array block.
 * @tparam ClusterShape_MNK Threadblock cluster dimension size (traditionally 1x1x1 pre-Hopper).
 */
template <typename Element, typename GmemStride, typename TileShape_MNK, int ThreadCount, typename ElementC,
          int GmemAlignmentA, int GmemAlignmentB, int GmemAlignmentC, typename ClusterShape_MNK>
struct AutoPartitioner<cutlass::arch::Sm80, cutlass::arch::OpClassSimt, Element, GmemStride, TileShape_MNK, ThreadCount,
                       ElementC, GmemAlignmentA, GmemAlignmentB, GmemAlignmentC, ClusterShape_MNK,
                       std::enable_if_t<detail::IsSm80SimtElement<Element>::value>>
{
    using RoleA = detail::Sm80SimtRoleA<Element, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentA>;
    using RoleB = detail::Sm80SimtRoleB<Element, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentB>;
    using RoleC = detail::Sm80SimtRoleC<Element, ElementC, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentC>;
};
/**
 * @brief AutoPartitioner specialization for SM80 architectures leveraging TensorOp (Tensor Core MMA) instructions.
 * Constructs Roles A, B, and C by integrating Ampere-native features such as cp.async and ldmatrix capabilities.
 *
 * @tparam Element The basic operational data type (must validate true for SM80 TensorOp compatibility).
 * @tparam GmemStride Struct characterizing layout strides mapped across global memory.
 * @tparam TileShape_MNK The overall threadblock spatial configuration shape (M, N, K).
 * @tparam ThreadCount The total thread count executing per block (needs integer multiplicity matching warp bounds).
 * @tparam ElementC The resulting destination format/type written to memory.
 * @tparam GmemAlignmentA Guaranteed alignment constraints for Operand A memory pointers.
 * @tparam GmemAlignmentB Guaranteed alignment constraints for Operand B memory pointers.
 * @tparam GmemAlignmentC Guaranteed alignment constraints for Operand C memory pointers.
 * @tparam ClusterShape_MNK Threadblock cluster dimension size (traditionally 1x1x1 pre-Hopper).
 */
template <typename Element, typename GmemStride, typename TileShape_MNK, int ThreadCount, typename ElementC,
          int GmemAlignmentA, int GmemAlignmentB, int GmemAlignmentC, typename ClusterShape_MNK>
struct AutoPartitioner<cutlass::arch::Sm80, cutlass::arch::OpClassTensorOp, Element, GmemStride, TileShape_MNK,
                       ThreadCount, ElementC, GmemAlignmentA, GmemAlignmentB, GmemAlignmentC, ClusterShape_MNK,
                       std::enable_if_t<detail::IsSm80TensorOpElement<Element>::value>>
{
    using RoleA = detail::Sm80TensorOpRoleA<Element, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentA>;
    using RoleB = detail::Sm80TensorOpRoleB<Element, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentB>;
    using RoleC = detail::Sm80TensorOpRoleC<Element, ElementC, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentC>;
};

} // namespace autopartition
