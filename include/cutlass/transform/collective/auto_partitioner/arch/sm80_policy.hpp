#pragma once

#include "../auto_partitioner.hpp"
#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/gemm/collective/builders/sm90_common.inl>
namespace autopartition
{
// ==============================================================================
// 辅助元函数：推导最优线程拓扑
// ==============================================================================
template <int TileM, int TileN, int ThreadCount>
struct OptimalSimtThreadLayout
{
  static constexpr int TM = (ThreadCount == 256) ? ((TileM > TileN) ? 32 : 16) : 16;
  static constexpr int TN = ThreadCount / TM;
  using Layout            = cute::Layout<cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::_1>>;
};

// ==============================================================================
// SM80 + OpClassSimt 偏特化实现 (已修复 16-Byte 对齐的 Padding)
// ==============================================================================
template <typename Element, typename GmemStride, typename TileShape_MNK, int ThreadCount>
struct AutoPartitioner<cutlass::arch::Sm80,
                       cutlass::arch::OpClassSimt,
                       Element,
                       GmemStride,
                       TileShape_MNK,
                       ThreadCount,
                       cute::enable_if_t<true>>
{
  static constexpr int BlkM_val = cute::size<0>(TileShape_MNK{});
  static constexpr int BlkN_val = cute::size<1>(TileShape_MNK{});
  static constexpr int BlkK_val = cute::size<2>(TileShape_MNK{});

  // 提取推导最优线程拓扑
  using ThreadLayout = typename OptimalSimtThreadLayout<BlkM_val, BlkN_val, ThreadCount>::Layout;

  // 【核心修复】：计算 16 字节对齐所需要的最小元素个数 (对于 float 是 4)
  static constexpr int PaddingElements = 16 / sizeof(Element);

  // ==========================================================================
  // [Role A]
  // ==========================================================================
  struct RoleA
  {
    static constexpr bool is_M_contiguous = cute::is_same_v<decltype(cute::get<0>(GmemStride{})), cute::Int<1>>;

    // 使用 PaddingElements (4) 进行补齐，绝不使用 1！
    using SmemLayoutAtom =
      cute::conditional_t<is_M_contiguous,
                          cute::Layout<cute::Shape<cute::Int<BlkM_val>, cute::Int<BlkK_val>>,
                                       cute::Stride<cute::_1, cute::Int<BlkM_val + PaddingElements>>>,
                          cute::Layout<cute::Shape<cute::Int<BlkM_val>, cute::Int<BlkK_val>>,
                                       cute::Stride<cute::Int<BlkK_val + PaddingElements>, cute::_1>>>;
    using SmemLayout = SmemLayoutAtom;

    static constexpr int ContiguousDimLength = is_M_contiguous ? BlkM_val : BlkK_val;
    static constexpr int AlignmentElements   = (ContiguousDimLength % PaddingElements == 0) ? PaddingElements : 1;

    using CpAsyncAtom =
      cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS<cute::uint_byte_t<AlignmentElements * sizeof(Element)>>, Element>;

    using GmemToSmemCopy =
      decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
               CpAsyncAtom,
               ThreadCount,
               AlignmentElements,
               GmemStride,
               cute::Int<BlkM_val>,
               cute::Int<BlkK_val>>());

    using SmemToRegCopy = cute::Copy_Atom<cute::DefaultCopy, Element>;
  };

  // ==========================================================================
  // [Role B]
  // ==========================================================================
  struct RoleB
  {
    static constexpr bool is_N_contiguous = cute::is_same_v<decltype(cute::get<0>(GmemStride{})), cute::Int<1>>;

    using SmemLayoutAtom =
      cute::conditional_t<is_N_contiguous,
                          cute::Layout<cute::Shape<cute::Int<BlkN_val>, cute::Int<BlkK_val>>,
                                       cute::Stride<cute::_1, cute::Int<BlkN_val + PaddingElements>>>,
                          cute::Layout<cute::Shape<cute::Int<BlkN_val>, cute::Int<BlkK_val>>,
                                       cute::Stride<cute::Int<BlkK_val + PaddingElements>, cute::_1>>>;
    using SmemLayout = SmemLayoutAtom;

    static constexpr int ContiguousDimLength = is_N_contiguous ? BlkN_val : BlkK_val;
    static constexpr int AlignmentElements   = (ContiguousDimLength % PaddingElements == 0) ? PaddingElements : 1;

    using CpAsyncAtom =
      cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS<cute::uint_byte_t<AlignmentElements * sizeof(Element)>>, Element>;

    using GmemToSmemCopy =
      decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
               CpAsyncAtom,
               ThreadCount,
               AlignmentElements,
               GmemStride,
               cute::Int<BlkN_val>,
               cute::Int<BlkK_val>>());

    using SmemToRegCopy = cute::Copy_Atom<cute::DefaultCopy, Element>;
  };

  // ==========================================================================
  // [Role C / Epilogue]
  // ==========================================================================
  struct RoleC
  {
    using MmaAtom  = cute::MMA_Atom<cute::UniversalFMA<Element, Element, Element>>;
    using TiledMma = decltype(cute::make_tiled_mma(MmaAtom{}, ThreadLayout{}));

    static constexpr bool is_C_M_contiguous = cute::is_same_v<decltype(cute::get<0>(GmemStride{})), cute::Int<1>>;

    using SmemLayoutAtom =
      cute::conditional_t<is_C_M_contiguous,
                          cute::Layout<cute::Shape<cute::Int<BlkM_val>, cute::Int<BlkN_val>>,
                                       cute::Stride<cute::_1, cute::Int<BlkM_val + PaddingElements>>>,
                          cute::Layout<cute::Shape<cute::Int<BlkM_val>, cute::Int<BlkN_val>>,
                                       cute::Stride<cute::Int<BlkN_val + PaddingElements>, cute::_1>>>;
    using SmemLayout = SmemLayoutAtom;

    using RegToSmemCopy = cute::Copy_Atom<cute::DefaultCopy, Element>;

    static constexpr int ContiguousDimLength = is_C_M_contiguous ? BlkM_val : BlkN_val;
    static constexpr int AlignmentElements   = (ContiguousDimLength % PaddingElements == 0) ? PaddingElements : 1;

    using SmemToGmemCopy =
      decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
               cute::Copy_Atom<cute::DefaultCopy, Element>,
               ThreadCount,
               AlignmentElements,
               GmemStride,
               cute::Int<BlkM_val>,
               cute::Int<BlkN_val>>());
  };
};
} // namespace autopartition
