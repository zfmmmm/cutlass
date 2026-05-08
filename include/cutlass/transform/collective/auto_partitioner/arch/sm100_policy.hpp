#pragma once

#include <cstdint>
#include <type_traits>

#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits_sm80.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
#include <cute/atom/copy_traits_sm100_tma.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/collective/collective_builder_decl.hpp>
#include <cutlass/gemm/collective/collective_mma_decl.hpp>
#include <cutlass/gemm/collective/builders/sm90_common.inl>
#include <cutlass/gemm/collective/builders/sm100_common.inl>
#include <cutlass/gemm/collective/builders/sm100_simt_builder.inl>
#include <cutlass/numeric_types.h>

#include "../auto_partitioner.hpp"
#include "sm80_policy.hpp"

namespace autopartition {
namespace detail {

template <class Element>
struct IsSm100SimtElement : std::is_same<Element, float> {};

template <class Element>
struct IsSm100TensorOpElement : std::integral_constant<bool,
  std::is_same<Element, float>::value ||
  std::is_same<Element, cutlass::half_t>::value ||
  std::is_same<Element, cutlass::bfloat16_t>::value ||
  std::is_same<Element, int8_t>::value ||
  std::is_same<Element, uint8_t>::value> {};

template <class Element>
struct Sm100TensorOpAccumulator
{
  using type = float;
};

template <>
struct Sm100TensorOpAccumulator<int8_t>
{
  using type = int32_t;
};

template <>
struct Sm100TensorOpAccumulator<uint8_t>
{
  using type = int32_t;
};

template <class Element, int ContiguousElements>
struct Sm100GmemVectorAlignment : GmemVectorAlignment<Element, ContiguousElements> {};

template <class Element, class GmemStride, int TileMN, int TileK, int ThreadCount>
struct Sm100SimtMainloopRole
{
  static_assert(std::is_same<Element, float>::value, "SM100 SIMT policy currently targets SGEMM.");
  static_assert(TileK == 16, "SM100 SIMT SGEMM kernels require TileShape_K = 16.");

  using TileShape = cute::Shape<cute::Int<TileMN>, cute::_1, cute::Int<TileK>>;

  static constexpr bool IsMnMajor = cutlass::gemm::detail::is_mn_major<GmemStride>();
  static constexpr int SmemAlignmentOffset = IsMnMajor ? 0 : 2;
  static constexpr int ContiguousDimLength = IsMnMajor ? TileMN : TileK;
  static constexpr int AlignmentElements = Sm100GmemVectorAlignment<Element, ContiguousDimLength>::value;
  using AlignmentType = cute::uint_byte_t<AlignmentElements * int(sizeof(Element))>;

  using SmemLayoutAtom =
    cute::Layout<cute::Shape<cute::Int<TileMN>, cute::Int<TileK>>,
                 cute::Stride<cute::_1, cute::Int<TileMN + SmemAlignmentOffset>>>;
  using SmemLayout = SmemLayoutAtom;

  using SmemToRegCopy = cute::conditional_t<
    IsMnMajor,
    cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, Element>,
    cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<64>, Element>>;
  using RegToSmemCopy = SmemToRegCopy;

  using GmemCopyAtom = cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<AlignmentType>, Element>;
  using GmemToSmemCopy = decltype(
    cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
      GmemCopyAtom,
      ThreadCount,
      AlignmentElements,
      GmemStride,
      cute::Int<TileMN>,
      cute::Int<TileK>>());
  using SmemToGmemCopy = decltype(
    cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
      VectorizedCopyAtom<Element, AlignmentElements>,
      ThreadCount,
      AlignmentElements,
      GmemStride,
      cute::Int<TileMN>,
      cute::Int<TileK>>());

  using GlobalToSharedCopy = GmemToSmemCopy;
  using SharedToRegisterCopy = SmemToRegCopy;
  using RegisterToSharedCopy = RegToSmemCopy;
  using SharedToGlobalCopy = SmemToGmemCopy;
};

template <class Element, class GmemStride, class TileShape_MNK, int ThreadCount>
struct Sm100SimtRoleA
  : Sm100SimtMainloopRole<Element,
                          GmemStride,
                          cute::size<0>(TileShape_MNK{}),
                          cute::size<2>(TileShape_MNK{}),
                          ThreadCount> {};

template <class Element, class GmemStride, class TileShape_MNK, int ThreadCount>
struct Sm100SimtRoleB
  : Sm100SimtMainloopRole<Element,
                          GmemStride,
                          cute::size<1>(TileShape_MNK{}),
                          cute::size<2>(TileShape_MNK{}),
                          ThreadCount> {};

template <class Element, class GmemStride, class TileShape_MNK, int ThreadCount>
struct Sm100SimtRoleC
{
  static_assert(std::is_same<Element, float>::value, "SM100 SIMT policy currently targets SGEMM.");
  static_assert(cute::size<2>(TileShape_MNK{}) == 16, "SM100 SIMT SGEMM kernels require TileShape_K = 16.");

  using WarpShape_MNK = decltype(cutlass::gemm::collective::detail::sm100_simt_f32_warp_shape_mnk_selector<TileShape_MNK>());
  static constexpr int OfficialThreadCount = cute::size(WarpShape_MNK{}) * cutlass::NumThreadsPerWarp;
  static_assert(ThreadCount == OfficialThreadCount,
                "ThreadCount must match the SM100 SIMT warp-shape selector for this TileShape.");

  using TiledMma = decltype(
    cutlass::gemm::collective::detail::sm100_make_simt_f32_tiled_mma<
      GmemStride,
      1,
      GmemStride,
      1,
      TileShape_MNK,
      WarpShape_MNK>());
  using MmaAtom = typename TiledMma::Atom;

  static constexpr int BlkM = cute::size<0>(TileShape_MNK{});
  static constexpr int BlkN = cute::size<1>(TileShape_MNK{});
  static constexpr bool IsMnMajor = cutlass::gemm::detail::is_mn_major<GmemStride>();
  static constexpr int Padding = SmemPaddingElements<Element>::value;

  using SmemLayoutAtom = cute::conditional_t<
    IsMnMajor,
    cute::Layout<cute::Shape<cute::Int<BlkM>, cute::Int<BlkN>>,
                 cute::Stride<cute::_1, cute::Int<BlkM + Padding>>>,
    cute::Layout<cute::Shape<cute::Int<BlkM>, cute::Int<BlkN>>,
                 cute::Stride<cute::Int<BlkN + Padding>, cute::_1>>>;
  using SmemLayout = SmemLayoutAtom;

  static constexpr int ContiguousDimLength = IsMnMajor ? BlkM : BlkN;
  static constexpr int AlignmentElements = Sm100GmemVectorAlignment<Element, ContiguousDimLength>::value;
  using AlignmentType = cute::uint_byte_t<AlignmentElements * int(sizeof(Element))>;

  using GmemToSmemCopy = decltype(
    cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
      cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<AlignmentType>, Element>,
      ThreadCount,
      AlignmentElements,
      GmemStride,
      cute::Int<BlkM>,
      cute::Int<BlkN>>());
  using SmemToRegCopy = cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, Element>;
  using RegToSmemCopy = cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, Element>;
  using SmemToGmemCopy = decltype(
    cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
      VectorizedCopyAtom<Element, AlignmentElements>,
      ThreadCount,
      AlignmentElements,
      GmemStride,
      cute::Int<BlkM>,
      cute::Int<BlkN>>());

  using GlobalToSharedCopy = GmemToSmemCopy;
  using SharedToRegisterCopy = SmemToRegCopy;
  using RegisterToSharedCopy = RegToSmemCopy;
  using SharedToGlobalCopy = SmemToGmemCopy;
};

template <class Element, class GmemStride, class TileShape_MNK, bool IsRoleA>
struct Sm100TensorOpMainloopRole;

template <class Element, class GmemStride, class TileShape_MNK>
struct Sm100TensorOpMainloopRole<Element, GmemStride, TileShape_MNK, true>
{
  using ElementMma = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<Element>());
  using SmemAllocElement = cute::conditional_t<(cute::sizeof_bits_v<ElementMma> < 8), uint8_t, ElementMma>;

  static constexpr cute::UMMA::Major Major = cutlass::gemm::collective::detail::tag_to_umma_major_A<GmemStride>();
  static constexpr int BlkM = cute::size<0>(TileShape_MNK{});
  static constexpr int BlkK = cute::size<2>(TileShape_MNK{});

  using SmemLayoutAtom = decltype(
    cutlass::gemm::collective::detail::sm100_smem_selector<
      Major,
      SmemAllocElement,
      cute::Int<BlkM>,
      cute::Int<BlkK>>());
  using SmemLayout = decltype(cute::tile_to_shape(SmemLayoutAtom{}, cute::Shape<cute::Int<BlkM>, cute::Int<BlkK>>{}));

  using GmemToSmemCopy = cute::SM90_TMA_LOAD;
  using GlobalToSharedCopy = GmemToSmemCopy;

  using GmemToSmemCpAsyncCopy = decltype(
    cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
      cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<cute::uint_byte_t<16>>, Element>,
      128,
      16 / int(sizeof(Element)),
      GmemStride,
      cute::Int<BlkM>,
      cute::Int<BlkK>>());

  using SmemToRegCopy = void;
  using SharedToRegisterCopy = SmemToRegCopy;
  using RegToSmemCopy = void;
  using RegisterToSharedCopy = RegToSmemCopy;
  using SmemToGmemCopy = cute::SM90_TMA_STORE;
  using SharedToGlobalCopy = SmemToGmemCopy;
};

template <class Element, class GmemStride, class TileShape_MNK>
struct Sm100TensorOpMainloopRole<Element, GmemStride, TileShape_MNK, false>
{
  using ElementMma = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<Element>());
  using SmemAllocElement = cute::conditional_t<(cute::sizeof_bits_v<ElementMma> < 8), uint8_t, ElementMma>;

  static constexpr cute::UMMA::Major Major = cutlass::gemm::collective::detail::tag_to_umma_major_B<GmemStride>();
  static constexpr int BlkN = cute::size<1>(TileShape_MNK{});
  static constexpr int BlkK = cute::size<2>(TileShape_MNK{});

  using SmemLayoutAtom = decltype(
    cutlass::gemm::collective::detail::sm100_smem_selector<
      Major,
      SmemAllocElement,
      cute::Int<BlkN>,
      cute::Int<BlkK>>());
  using SmemLayout = decltype(cute::tile_to_shape(SmemLayoutAtom{}, cute::Shape<cute::Int<BlkN>, cute::Int<BlkK>>{}));

  using GmemToSmemCopy = cute::SM90_TMA_LOAD;
  using GlobalToSharedCopy = GmemToSmemCopy;

  using GmemToSmemCpAsyncCopy = decltype(
    cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
      cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<cute::uint_byte_t<16>>, Element>,
      128,
      16 / int(sizeof(Element)),
      GmemStride,
      cute::Int<BlkN>,
      cute::Int<BlkK>>());

  using SmemToRegCopy = void;
  using SharedToRegisterCopy = SmemToRegCopy;
  using RegToSmemCopy = void;
  using RegisterToSharedCopy = RegToSmemCopy;
  using SmemToGmemCopy = cute::SM90_TMA_STORE;
  using SharedToGlobalCopy = SmemToGmemCopy;
};

template <class Element, class GmemStride, class TileShape_MNK, int ThreadCount>
struct Sm100TensorOpRoleA : Sm100TensorOpMainloopRole<Element, GmemStride, TileShape_MNK, true> {};

template <class Element, class GmemStride, class TileShape_MNK, int ThreadCount>
struct Sm100TensorOpRoleB : Sm100TensorOpMainloopRole<Element, GmemStride, TileShape_MNK, false> {};

template <class Element, class GmemStride, class TileShape_MNK, int ThreadCount>
struct Sm100TensorOpRoleC
{
  using ElementMma = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<Element>());
  using Accumulator = typename Sm100TensorOpAccumulator<Element>::type;
  using ClusterShape_MNK = cute::Shape<cute::_1, cute::_1, cute::_1>;

  template <cute::UMMA::Major MajorA, cute::UMMA::Major MajorB>
  using TiledMmaFor = decltype(
    cutlass::gemm::collective::detail::sm100_make_trivial_tiled_mma<
      ElementMma,
      ElementMma,
      Accumulator,
      TileShape_MNK,
      ClusterShape_MNK,
      MajorA,
      MajorB,
      cutlass::gemm::collective::KernelScheduleAuto>());

  static constexpr cute::UMMA::Major DefaultMajorA = cutlass::gemm::collective::detail::tag_to_umma_major_A<GmemStride>();
  static constexpr cute::UMMA::Major DefaultMajorB = cutlass::gemm::collective::detail::tag_to_umma_major_B<GmemStride>();
  using TiledMma = TiledMmaFor<DefaultMajorA, DefaultMajorB>;
  using MmaAtom = typename TiledMma::Atom;

  static constexpr int BlkM = cute::size<0>(TileShape_MNK{});
  static constexpr int BlkN = cute::size<1>(TileShape_MNK{});
  static constexpr bool IsMnMajor = cutlass::gemm::detail::is_mn_major<GmemStride>();
  static constexpr int Padding = SmemPaddingElements<Element>::value;

  using SmemLayoutAtom = cute::conditional_t<
    IsMnMajor,
    cute::Layout<cute::Shape<cute::Int<BlkM>, cute::Int<BlkN>>,
                 cute::Stride<cute::_1, cute::Int<BlkM + Padding>>>,
    cute::Layout<cute::Shape<cute::Int<BlkM>, cute::Int<BlkN>>,
                 cute::Stride<cute::Int<BlkN + Padding>, cute::_1>>>;
  using SmemLayout = SmemLayoutAtom;

  static constexpr int ContiguousDimLength = IsMnMajor ? BlkM : BlkN;
  static constexpr int AlignmentElements = Sm100GmemVectorAlignment<Element, ContiguousDimLength>::value;
  using AlignmentType = cute::uint_byte_t<AlignmentElements * int(sizeof(Element))>;

  using GmemToSmemCopy = cute::SM90_TMA_LOAD;
  using SmemToRegCopy = cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, Element>;
  using RegToSmemCopy = cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, Element>;
  using SmemToGmemCopy = decltype(
    cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
      VectorizedCopyAtom<Element, AlignmentElements>,
      ThreadCount,
      AlignmentElements,
      GmemStride,
      cute::Int<BlkM>,
      cute::Int<BlkN>>());

  using GlobalToSharedCopy = GmemToSmemCopy;
  using SharedToRegisterCopy = SmemToRegCopy;
  using RegisterToSharedCopy = RegToSmemCopy;
  using SharedToGlobalCopy = SmemToGmemCopy;
};

} // namespace detail

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

} // namespace autopartition
