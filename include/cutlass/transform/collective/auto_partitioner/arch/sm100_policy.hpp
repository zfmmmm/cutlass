#pragma once

#include <cstdint>
#include <type_traits>

// clang-format off
// #include <cute/atom/copy_atom.hpp>
// #include <cute/atom/copy_traits_sm100.hpp>
// #include <cute/atom/copy_traits_sm100_tma.hpp>
// #include <cute/atom/copy_traits_sm80.hpp>
// #include <cute/atom/mma_atom.hpp>
// #include <cute/atom/mma_traits_sm100.hpp>
// #include <cute/layout.hpp>
// #include <cute/tensor.hpp>

// #include <cutlass/arch/arch.h>
// #include <cutlass/arch/mma.h>
// #include <cutlass/epilogue/collective/collective_builder.hpp>
// #include <cutlass/gemm/collective/collective_builder_decl.hpp>
// #include <cutlass/gemm/collective/collective_mma_decl.hpp>
// #include <cutlass/gemm/collective/builders/sm100_common.inl>
// #include <cutlass/gemm/collective/builders/sm100_simt_builder.inl>
// #include <cutlass/gemm/collective/builders/sm90_common.inl>
// #include <cutlass/gemm/gemm.h>
// #include <cutlass/layout/matrix.h>
// #include <cutlass/numeric_types.h>

// #include "../auto_partitioner.hpp"
// #include "sm80_policy.hpp"
// clang-format on
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
#include <cute/atom/copy_traits_sm100_tma.hpp>
#include <cute/atom/copy_traits_sm80.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
//
#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder_decl.hpp>
//
#include <cutlass/gemm/collective/collective_mma_decl.hpp>
//
#include <cutlass/gemm/collective/builders/sm100_common.inl>
#include <cutlass/gemm/collective/builders/sm100_simt_builder.inl>
//
#include <cutlass/gemm/collective/builders/sm90_common.inl>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

#include "../auto_partitioner.hpp"
#include "sm80_policy.hpp"

namespace autopartition {
namespace detail {

template <class Element> struct IsSm100SimtElement : std::is_same<Element, float>
{
};
/**
 * @brief Checks if the given data type is supported by SM100 Tensor Cores (MMA instructions).
 * SM100 supports a variety of precision types including floating point and integers.
 * * @tparam Element The data type to evaluate. Supports float, half_t, bfloat16_t, int8_t, and uint8_t.
 */
template <class Element>
struct IsSm100TensorOpElement
    : std::integral_constant<bool,
                             std::is_same<Element, float>::value || std::is_same<Element, cutlass::half_t>::value
                                 || std::is_same<Element, cutlass::bfloat16_t>::value
                                 || std::is_same<Element, int8_t>::value || std::is_same<Element, uint8_t>::value>
{
};
/**
 * @brief Type trait to determine the appropriate math accumulator data type
 * based on the input element type for SM100 Tensor Core operations.
 * * @tparam Element The input data type (e.g., float, int8_t, uint8_t).
 */
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
/**
 * @brief Determines the optimal global memory (Gmem) vectorization alignment
 * for a specific element and contiguous dimension length on SM100.
 * This ensures efficient vectorized memory loads/stores (e.g., 128-bit LDS/STS).
 * * @tparam Element The data type being accessed in global memory.
 * @tparam ContiguousElements The number of elements in the contiguous dimension.
 * @tparam MaxAlignmentBytes The maximum allowed alignment in bytes (default is 16 bytes, i.e., 128 bits).
 */
template <class Element, int ContiguousElements, int MaxAlignmentBytes = 16>
struct Sm100GmemVectorAlignment : GmemVectorAlignment<Element, ContiguousElements, MaxAlignmentBytes>
{
};
/**
 * @brief A compile-time helper function to calculate the Greatest Common Divisor (GCD) of two integers.
 * Useful for calculating shared memory bank swizzling and padding to avoid bank conflicts.
 * * @param lhs The left-hand side integer.
 * @param rhs The right-hand side integer.
 * @return constexpr int The calculated greatest common divisor.
 */
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
/**
 * @brief Computes the optimal dynamic padding for Shared Memory (Smem) allocations on SM100.
 * The padding prevents shared memory bank conflicts when threads access data through
 * vectorized loads (e.g., 128-bit memory instructions).
 * * @tparam Element The data type of the elements stored in shared memory.
 * @tparam MajorExtent The size (in elements) of the contiguous/major dimension of the memory block.
 * @tparam VectorBits The width of the vector memory access in bits (default: 128).
 * @tparam BankCount The number of shared memory banks in the architecture (default: 32).
 * @tparam BankWidthBytes The width of each individual shared memory bank in bytes (default: 4).
 */
template <class Element, int MajorExtent, int VectorBits = 128, int BankCount = 32, int BankWidthBytes = 4>
struct Sm100SmemBankPaddingElements
{
    static_assert(VectorBits % 8 == 0, "SM100 shared-memory vector width must be byte-addressable.");
    static_assert(MajorExtent > 0, "SM100 shared-memory major extent must be positive.");

    static constexpr int ElementBytes   = int(sizeof(Element));
    static constexpr int VectorBytes    = VectorBits / 8;
    static constexpr int BankSpanBytes  = BankCount * BankWidthBytes;
    static constexpr int VectorBankSets = BankSpanBytes / VectorBytes;

    static_assert((BankSpanBytes % VectorBytes) == 0, "SM100 shared-memory vector width must divide the 32-bank span.");
    // 断当前尝试的 Padding 元素数量是否满足消除冲突的条件
    static constexpr bool is_candidate(int padding_elements)
    {
        int pitch_bytes = (MajorExtent + padding_elements) * ElementBytes;
        if ((pitch_bytes % VectorBytes) != 0) {
            return false;
        }
        int bank_set_stride = (pitch_bytes / VectorBytes) % VectorBankSets;
        return bank_set_stride != 0 && sm100_constexpr_gcd(bank_set_stride, VectorBankSets) == 1;
    }
    // 断当前尝试的 Padding 元素数量是否满足消除冲突的条件
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
/**
 * @brief Determines whether the Tensor Memory Accelerator (TMA) should be enabled for SM100.
 * TMA relies on strict memory alignment constraints; it is only activated if the
 * global memory alignment meets or exceeds the 16-byte (128-bit) requirement.
 * * @tparam GmemAlignmentBytes The guaranteed alignment of the global memory pointer in bytes.
 */
template <int GmemAlignmentBytes> struct Sm100UseTma : std::integral_constant<bool, (GmemAlignmentBytes >= 16)>
{
};

template <class GmemStride> struct Sm100EpilogueLayoutTag
{
    using type = cute::conditional_t<cutlass::gemm::detail::is_mn_major<GmemStride>(),
                                     cutlass::layout::ColumnMajor,
                                     cutlass::layout::RowMajor>;
};
/**
 * @brief Validates if the chosen tile shape and thread count configuration is natively eligible
 * for an SM100 SIMT (Single Instruction, Multiple Threads) based mainloop.
 * Ensures the Block dimensions align with SM100 warp scheduling characteristics.
 * * @tparam TileShape_MNK A CuTe shape representing the CTA-level tile size <M, N, K>.
 * @tparam ThreadCount The total number of threads assigned to the CTA.
 */
template <class TileShape_MNK, int ThreadCount> struct Sm100SimtNativeEligible
{
    // 根据 CTA tile shape 自动选择 warpMN组织方式
    using WarpShape_MNK =
        decltype(cutlass::gemm::collective::detail::sm100_simt_f32_warp_shape_mnk_selector<TileShape_MNK>());

    static constexpr bool value = (cute::size<2>(TileShape_MNK{}) == 16)
                               && (ThreadCount == int(cute::size(WarpShape_MNK{})) * int(cutlass::NumThreadsPerWarp));
};
/**
 * @brief Base configuration struct for defining the Mainloop role (either Operand A or Operand B)
 * for SIMT-based matrix multiplication operations on the SM100 architecture.
 * It dictates layout creation, copy operations (Gmem->Smem, Smem->Reg), and alignments.
 * * @tparam Element The data type of the input matrix.
 * @tparam GmemStride The structural stride of the matrix in global memory.
 * @tparam TileMN The CTA-level tile size in the spatial dimension (M for RoleA, N for RoleB).
 * @tparam TileK The CTA-level tile size in the reduction/inner dimension K.
 * @tparam ThreadCount Total number of threads available in the threadblock (CTA).
 */
template <class Element, class GmemStride, int TileMN, int TileK, int ThreadCount> struct Sm100SimtMainloopRole
{
    static_assert(std::is_same<Element, float>::value, "SM100 SIMT policy currently targets SGEMM.");

    using TileShape = cute::Shape<cute::Int<TileMN>, cute::_1, cute::Int<TileK>>;

    static constexpr bool IsMnMajor = cutlass::gemm::detail::is_mn_major<GmemStride>();
    // 一个是需要根据 IsMnMajor 来判断主序然后加 padding，一个是要根据支持向量化粒度来选择 SmemVectorAlignmentBits
    static constexpr int SmemVectorAlignmentBits = 128;
    static constexpr int SmemAlignmentOffset =
        IsMnMajor ? 0 : Sm100SmemBankPaddingElements<Element, TileMN, SmemVectorAlignmentBits>::value;
    static constexpr int ContiguousDimLength = IsMnMajor ? TileMN : TileK;

    static constexpr int AlignmentElements =
        IsMnMajor ? Sm100GmemVectorAlignment<Element, ContiguousDimLength>::value : 1;
    using AlignmentType = cute::uint_byte_t<AlignmentElements *int(sizeof(Element))>;

    using SmemLayoutAtom = cute::Layout<cute::Shape<cute::Int<TileMN>, cute::Int<TileK>>,
                                        cute::Stride<cute::_1, cute::Int<TileMN + SmemAlignmentOffset>>>;
    using SmemLayout     = SmemLayoutAtom;

    using SmemToRegCopy =
        cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<SmemVectorAlignmentBits>, Element>;
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
/**
 * @brief Specializes the SIMT Mainloop Role for Operand A (LHS matrix).
 * Extracts the M and K dimensions from the MNK TileShape.
 * * @tparam Element Data type of Operand A.
 * @tparam GmemStride Memory stride of Operand A.
 * @tparam TileShape_MNK The CTA-level tile shape <M, N, K>.
 * @tparam ThreadCount Thread count per CTA.
 */
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
/**
 * @brief Configures the Epilogue Role (Operand C/Accumulator) for SM100 SIMT matrix multiplication.
 * Handles the memory layouts and copy strategies needed to move computed results from
 * registers (or thread-local memory) back to Shared Memory, and ultimately to Global Memory.
 * * @tparam Element Data type of the result matrix (typically floating point).
 * @tparam GmemStride The memory stride pattern of the output matrix in global memory.
 * @tparam TileShape_MNK The CTA-level tile shape <M, N, K> used during the computation.
 * @tparam ThreadCount Number of threads handling the epilogue write-back.
 */
template <class Element, class GmemStride, class TileShape_MNK, int ThreadCount> struct Sm100SimtRoleC
{
    static_assert(std::is_same<Element, float>::value, "SM100 SIMT policy currently targets SGEMM.");

    using WarpShape_MNK =
        decltype(cutlass::gemm::collective::detail::sm100_simt_f32_warp_shape_mnk_selector<TileShape_MNK>());

    static constexpr int OfficialThreadCount = cute::size(WarpShape_MNK{}) * cutlass::NumThreadsPerWarp;

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
/**
 * @brief Forward declaration for the Tensor Core (TensorOp) Mainloop Role on SM100.
 * This structure manages advanced memory fetching strategies such as TMA (Tensor Memory Accelerator),
 * CP_ASYNC, and UMMA (Unmma) layout mappings.
 * * @tparam Element Data type of the input matrix.
 * @tparam GmemStride Stride configuration in Global Memory.
 * @tparam TileShape_MNK The CTA-level tile shape <M, N, K>.
 * @tparam ThreadCount Number of threads in the CTA.
 * @tparam GmemAlignmentBytes Memory alignment of the input matrix in bytes.
 * @tparam ClusterShape The shape of the Threadblock Cluster (Threadblock clustering for SM90+ architectures).
 * @tparam IsRoleA Boolean flag: true if configuring Operand A, false if configuring Operand B.
 */
template <class Element,
          class GmemStride,
          class TileShape_MNK,
          int ThreadCount,
          int GmemAlignmentBytes,
          class ClusterShape,
          bool IsRoleA>
struct Sm100TensorOpMainloopRole;
/**
 * @brief Specialization of Sm100TensorOpMainloopRole for Operand A (IsRoleA = true).
 * Sets up TMA load configurations, Shared Memory mappings, and WGMMA layouts
 * tailored to matrix A's dimensions (M and K).
 * * @tparam Element Data type of Operand A.
 * @tparam GmemStride Stride configuration in Global Memory for Operand A.
 * @tparam TileShape_MNK The CTA-level tile shape <M, N, K>.
 * @tparam ThreadCount Number of threads in the CTA.
 * @tparam GmemAlignmentBytes Memory alignment of Operand A in bytes.
 * @tparam ClusterShape The shape of the Threadblock Cluster.
 */
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
    using ElementMma =
        decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<Element>());
    using SmemAllocElement =
        cute::conditional_t<(cute::sizeof_bits_v<ElementMma> < 8), uint8_t, ElementMma>; // 小于 8 bit用uint8_t
    using Accumulator = typename Sm100TensorOpAccumulator<Element>::type;

    static constexpr cute::UMMA::Major Major =
        cutlass::gemm::collective::detail::tag_to_umma_major_A<GmemStride>(); // 判断 A 的 UMMA major
    static constexpr int  BlkM      = cute::size<0>(TileShape_MNK{});
    static constexpr int  BlkK      = cute::size<2>(TileShape_MNK{});
    static constexpr bool IsMnMajor = cutlass::gemm::detail::is_mn_major<GmemStride>();
    using TiledMma =
        decltype(cutlass::gemm::collective::detail::sm100_make_trivial_tiled_mma<
                 ElementMma,
                 ElementMma,
                 Accumulator,
                 TileShape_MNK,
                 ClusterShape,
                 Major,
                 cutlass::gemm::collective::detail::tag_to_umma_major_B<GmemStride>(),
                 cutlass::gemm::collective::KernelScheduleAuto>()); // 构造一个 SM100 TensorOp/UMMA 的 TiledMma 类型
    using SmemLayoutAtom =
        decltype(cutlass::gemm::collective::detail::sm100_smem_selector<Major,
                                                                        SmemAllocElement,
                                                                        cute::Int<BlkM>,
                                                                        cute::Int<BlkK>>()); // 选择 A 的 shared memory
                                                                                             // layout atom
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
    using GmemToSmemCopy     = cute::conditional_t<UsesTmaLoad, GmemToSmemTmaCopy, GmemToSmemCpAsyncCopy>;
    using GlobalToSharedCopy = GmemToSmemCopy;

    using SmemToRegCopy        = void;
    using SharedToRegisterCopy = SmemToRegCopy;
    using RegToSmemCopy        = void;
    using RegisterToSharedCopy = RegToSmemCopy;

    using SmemToGmemCopy     = void;
    using SharedToGlobalCopy = SmemToGmemCopy;
};
/**
 * @brief Specialization of Sm100TensorOpMainloopRole for Operand B (IsRoleA = false).
 * Sets up TMA load configurations, Shared Memory mappings, and WGMMA layouts
 * tailored to matrix B's dimensions (N and K).
 * * @tparam Element Data type of Operand B.
 * @tparam GmemStride Stride configuration in Global Memory for Operand B.
 * @tparam TileShape_MNK The CTA-level tile shape <M, N, K>.
 * @tparam ThreadCount Number of threads in the CTA.
 * @tparam GmemAlignmentBytes Memory alignment of Operand B in bytes.
 * @tparam ClusterShape The shape of the Threadblock Cluster.
 */
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
/**
 * @brief Helper alias struct for Operand A configuring the SM100 TensorOp Mainloop.
 * Inherits from the specialized Sm100TensorOpMainloopRole where IsRoleA is true.
 * * @tparam Element Data type of Operand A.
 * @tparam GmemStride Stride configuration in Global Memory.
 * @tparam TileShape_MNK The CTA-level tile shape <M, N, K>.
 * @tparam ThreadCount Number of threads in the CTA.
 * @tparam GmemAlignmentBytes Memory alignment of Operand A in bytes.
 * @tparam ClusterShape The shape of the Threadblock Cluster.
 */
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
/**
 * @brief Helper alias struct for Operand B configuring the SM100 TensorOp Mainloop.
 * Inherits from the specialized Sm100TensorOpMainloopRole where IsRoleA is false.
 * * @tparam Element Data type of Operand B.
 * @tparam GmemStride Stride configuration in Global Memory.
 * @tparam TileShape_MNK The CTA-level tile shape <M, N, K>.
 * @tparam ThreadCount Number of threads in the CTA.
 * @tparam GmemAlignmentBytes Memory alignment of Operand B in bytes.
 * @tparam ClusterShape The shape of the Threadblock Cluster.
 */
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
/**
 * @brief Configures the SM100 TensorOp Epilogue Role (Operand C).
 * Handles accumulator conversions, TMEM (Tensor Memory) to Shared Memory transfers,
 * and final writes (via TMA Store or Vectorized Copy) from Shared Memory to Global Memory.
 * * @tparam Element The compute/input data type of the matrix operands.
 * @tparam ElementC The expected output data type for matrix C.
 * @tparam GmemStride The global memory layout stride for matrix C.
 * @tparam TileShape_MNK The CTA-level tile shape <M, N, K>.
 * @tparam ThreadCount Total number of threads designated to the Epilogue.
 * @tparam GmemAlignmentBytes Global memory alignment in bytes for matrix C.
 * @tparam ClusterShape The Threadblock cluster shape layout.
 */
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

    using GmemLayoutTagC = typename Sm100EpilogueLayoutTag<GmemStride>::type;
    static_assert(GmemAlignmentBytes >= int(sizeof(ElementOutput)),
                  "SM100 RoleC output alignment must cover at least one element.");
    static_assert((GmemAlignmentBytes % int(sizeof(ElementOutput))) == 0,
                  "SM100 RoleC output alignment must be expressed in whole output elements.");
    static constexpr int AlignmentElements = GmemAlignmentBytes / int(sizeof(ElementOutput));
    static constexpr int AlignmentBytes    = AlignmentElements * int(sizeof(ElementOutput));
    static constexpr int AlignmentBits     = AlignmentBytes * 8;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm100,
        cutlass::arch::OpClassTensorOp,
        TileShape_MNK,
        ClusterShape_MNK,
        cutlass::epilogue::collective::EpilogueTileAuto,
        Accumulator,
        Accumulator,
        ElementOutput,
        GmemLayoutTagC,
        AlignmentElements,
        ElementOutput,
        GmemLayoutTagC,
        AlignmentElements,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

    using EpilogueTile      = typename CollectiveEpilogue::EpilogueTile;
    using EpilogueTileShape = decltype(cute::product_each(cute::shape(EpilogueTile{})));
    using EpilogueModeOrder =
        cute::conditional_t<IsMnMajor, cute::Step<cute::_2, cute::_1>, cute::Step<cute::_1, cute::_2>>;

    using SmemLayoutAtom = typename CollectiveEpilogue::SmemLayoutAtomC;
    using SmemLayout     = decltype(cute::tile_to_shape(SmemLayoutAtom{}, EpilogueTileShape{}, EpilogueModeOrder{}));
    using AccumulatorSmemLayout = SmemLayout;
    using FusionSmemLayout      = SmemLayout;
    using FusionElement         = EpilogueElement;

    using OutputSmemLayoutAtom = typename CollectiveEpilogue::SmemLayoutAtomD;
    using OutputSmemLayout =
        decltype(cute::tile_to_shape(OutputSmemLayoutAtom{}, EpilogueTileShape{}, EpilogueModeOrder{}));

    static constexpr bool UsesTmaLoad                 = Sm100UseTma<GmemAlignmentBytes>::value;
    static constexpr bool UsesTmaStore                = Sm100UseTma<GmemAlignmentBytes>::value;
    static constexpr int  GmemToSmemAlignmentElements = AlignmentElements;
    static constexpr int  GmemToSmemAlignmentBytes    = AlignmentBytes;
    static constexpr int  SmemToGmemAlignmentElements = AlignmentElements;
    static constexpr int  SmemToGmemAlignmentBytes    = AlignmentBytes;
    using GmemAlignmentType                           = cute::uint_byte_t<GmemToSmemAlignmentBytes>;

    using GmemToSmemCopyOperation = typename CollectiveEpilogue::CopyOpG2S;
    using GmemToSmemTmaCopy =
        decltype(cute::make_tma_copy(GmemToSmemCopyOperation{},
                                     cute::make_tensor(cute::make_gmem_ptr(static_cast<ElementOutput *>(nullptr)),
                                                       cute::make_shape(cute::Int<BlkM>{}, cute::Int<BlkN>{}),
                                                       GmemStride{}),
                                     SmemLayout{},
                                     EpilogueTile{},
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

    using TmemToRegisterCopyOperation   = typename CollectiveEpilogue::CopyOpT2R;
    using TmemToRegisterCopy            = TmemToRegisterCopyOperation;
    using TmemToSmemCopyOperation       = TmemToRegisterCopyOperation;
    using TmemToSmemCopy                = TmemToRegisterCopy;
    using SmemToRegCopyOperation        = typename CollectiveEpilogue::CopyOpS2R;
    using RegToSmemCopyOperation        = typename CollectiveEpilogue::CopyOpR2S;
    using SharedToGlobalCopyOperation   = typename CollectiveEpilogue::CopyOpS2G;
    using RegisterToSharedCopyOperation = RegToSmemCopyOperation;
    using SharedToRegisterCopyOperation = SmemToRegCopyOperation;
    using SmemToRegCopy                 = cute::Copy_Atom<SmemToRegCopyOperation, ElementOutput>;
    using RegToSmemCopy                 = cute::Copy_Atom<RegToSmemCopyOperation, ElementOutput>;
    using SmemToGmemTmaCopy =
        decltype(cute::make_tma_copy(SharedToGlobalCopyOperation{},
                                     cute::make_tensor(cute::make_gmem_ptr(static_cast<ElementOutput *>(nullptr)),
                                                       cute::make_shape(cute::Int<BlkM>{}, cute::Int<BlkN>{}),
                                                       GmemStride{}),
                                     OutputSmemLayout{},
                                     EpilogueTile{},
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

    using FusionRegisterToSharedCopyOperation = RegToSmemCopyOperation;
    using FusionSharedToRegisterCopyOperation = SmemToRegCopyOperation;
    using FusionRegisterToSharedCopy          = RegToSmemCopy;
    using FusionSharedToRegisterCopy          = SmemToRegCopy;
    using FusionSharedLayout                  = FusionSmemLayout;

    using AccumulatorRegisterLayout = decltype(TiledMma{}.get_layoutC_TV());
    using RegisterReuseAsALayout    = decltype(TiledMma{}.get_layoutA_TV());
    using RegisterReuseAsBLayout    = decltype(TiledMma{}.get_layoutB_TV());

    using EpilogueRegisterToRegisterCopyOperation = typename CollectiveEpilogue::CopyOpR2R;
    using RegisterToRegisterCopyOperation         = cute::DefaultCopy;
    using RegisterToOperandACopyOperation         = RegisterToRegisterCopyOperation;
    using RegisterToOperandBCopyOperation         = RegisterToRegisterCopyOperation;

    static constexpr bool HasZeroGlueEpilogueMapping = true;
    static constexpr bool HasFusionSharedMapping     = true;
    static constexpr bool HasRegisterReuseMapping    = true;
    static constexpr bool HasRegisterShuffleMapping  = false;
    static constexpr bool CanReuseAccumulatorAsOperand = std::is_same<EpilogueElement, ElementInput>::value;

    template <class NextTiledMma, class CopyOperation = RegisterToOperandACopyOperation>
    CUTE_HOST_DEVICE static auto make_register_to_operand_A_copy(NextTiledMma const &next_mma)
    {
        return cute::make_tiled_copy_A(cute::Copy_Atom<CopyOperation, EpilogueElement>{}, next_mma);
    }

    template <class NextTiledMma, class CopyOperation = RegisterToOperandBCopyOperation>
    CUTE_HOST_DEVICE static auto make_register_to_operand_B_copy(NextTiledMma const &next_mma)
    {
        return cute::make_tiled_copy_B(cute::Copy_Atom<CopyOperation, EpilogueElement>{}, next_mma);
    }

    template <class ThreadCopy, class RegisterTensor, class OperandRegisterTensor>
    CUTE_HOST_DEVICE static auto retile_register_to_operand(ThreadCopy const           &thr_copy,
                                                            RegisterTensor const       &register_tensor,
                                                            OperandRegisterTensor const &operand_tensor)
    {
        return cute::make_tuple(thr_copy.retile_S(register_tensor), thr_copy.retile_D(operand_tensor));
    }

    template <class ThreadCopy, class RegisterTensor, class SmemTensor>
    CUTE_HOST_DEVICE static auto retile_register_to_fusion_smem(ThreadCopy const     &thr_copy,
                                                                RegisterTensor const &register_tensor,
                                                                SmemTensor const     &smem_tensor)
    {
        return cute::make_tuple(thr_copy.retile_S(register_tensor), thr_copy.partition_D(smem_tensor));
    }

    template <class ThreadCopy, class SmemTensor, class RegisterTensor>
    CUTE_HOST_DEVICE static auto retile_fusion_smem_to_register(ThreadCopy const     &thr_copy,
                                                                SmemTensor const     &smem_tensor,
                                                                RegisterTensor const &register_tensor)
    {
        return cute::make_tuple(thr_copy.partition_S(smem_tensor), thr_copy.retile_D(register_tensor));
    }

    template <class ThreadCopy, class SmemTensor, class OutputTensor>
    CUTE_HOST_DEVICE static auto
    retile_smem_to_output(ThreadCopy const &thr_copy, SmemTensor const &smem_tensor, OutputTensor const &output_tensor)
    {
        return cute::make_tuple(thr_copy.partition_S(smem_tensor), thr_copy.partition_D(output_tensor));
    }

    template <class ThreadCopy, class RegisterTensor, class OutputTensor>
    CUTE_HOST_DEVICE static auto retile_register_to_output(ThreadCopy const     &thr_copy,
                                                           RegisterTensor const &register_tensor,
                                                           OutputTensor const   &output_tensor)
    {
        return cute::make_tuple(thr_copy.retile_S(register_tensor), thr_copy.partition_D(output_tensor));
    }
};
} // namespace detail
/**
 * @brief Specialization of the AutoPartitioner policy for the SM100 architecture executing SIMT
 * (Single Instruction, Multiple Threads) kernels. This specialization is chosen when the
 * element type and tile configuration are strictly natively eligible for SM100 SIMT execution.
 * * @tparam Element Input data type (Operand A and B).
 * @tparam GmemStride Global memory stride definition.
 * @tparam TileShape_MNK The structural tile shape <M, N, K>.
 * @tparam ThreadCount Total number of threads.
 * @tparam ElementC Output data type (Operand C).
 * @tparam GmemAlignmentA Operand A byte alignment.
 * @tparam GmemAlignmentB Operand B byte alignment.
 * @tparam GmemAlignmentC Operand C byte alignment.
 * @tparam ClusterShape_MNK Threadblock cluster dimension shape.
 */
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
/**
 * @brief Fallback specialization of the AutoPartitioner policy for the SM100 architecture executing SIMT.
 * If the given tile shape and thread count are NOT natively eligible on SM100,
 * this policy seamlessly falls back to legacy SM80 SIMT Roles.
 * * @tparam Element Input data type (Operand A and B).
 * @tparam GmemStride Global memory stride definition.
 * @tparam TileShape_MNK The structural tile shape <M, N, K>.
 * @tparam ThreadCount Total number of threads.
 * @tparam ElementC Output data type (Operand C).
 * @tparam GmemAlignmentA Operand A byte alignment.
 * @tparam GmemAlignmentB Operand B byte alignment.
 * @tparam GmemAlignmentC Operand C byte alignment.
 * @tparam ClusterShape_MNK Threadblock cluster dimension shape.
 */
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
/**
 * @brief Specialization of the AutoPartitioner policy for the SM100 architecture using Tensor Cores (TensorOp).
 * Activated when the data type is verified as natively supported by SM100 TensorOp architectures.
 * Dispatches the core computations to `Sm100TensorOpRole` definitions.
 * * @tparam Element Input data type (Operand A and B).
 * @tparam GmemStride Global memory stride definition.
 * @tparam TileShape_MNK The structural tile shape <M, N, K>.
 * @tparam ThreadCount Total number of threads.
 * @tparam ElementC Output data type (Operand C).
 * @tparam GmemAlignmentA Operand A byte alignment.
 * @tparam GmemAlignmentB Operand B byte alignment.
 * @tparam GmemAlignmentC Operand C byte alignment.
 * @tparam ClusterShape_MNK Threadblock cluster dimension shape.
 */
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
} // namespace autopartition
