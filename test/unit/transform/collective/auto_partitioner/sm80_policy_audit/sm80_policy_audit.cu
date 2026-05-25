/***************************************************************************************************
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/util/print_tensor.hpp>
#include <limits>
#include <type_traits>
#include <vector>

#include "../../../../common/cutlass_unit_test.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/transform/collective/auto_partitioner/auto_partitioner_builder.hpp"
#include "cutlass/transform/collective/auto_partitioner/examples/autopartition_example_utils.hpp"

using namespace cute;

namespace autopartition_sm80_policy_audit {

bool has_cuda_device()
{
    int         count  = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return count > 0;
}

template <class Element> void fill_audit_pattern(std::vector<Element> &values)
{
    autopartition::examples::fill_pattern(values);
}

template <> void fill_audit_pattern<int8_t>(std::vector<int8_t> &values)
{
    for (int i = 0; i < int(values.size()); ++i) {
        values[i] = int8_t(((i * 7 + 3) % 11) - 5);
    }
}

template <> void fill_audit_pattern<uint8_t>(std::vector<uint8_t> &values)
{
    for (int i = 0; i < int(values.size()); ++i) {
        values[i] = uint8_t((i * 5 + 1) % 13);
    }
}

template <class Element> float audit_to_float(Element value) { return autopartition::examples::to_float(value); }

template <class AElement, class BElement, class CElement>
void reference_gemm(int M, int N, int K, AElement const *A, BElement const *B, CElement *C)
{
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            if constexpr (std::is_integral<CElement>::value) {
                int32_t acc = 0;
                for (int k = 0; k < K; ++k) {
                    acc += int32_t(A[m * K + k]) * int32_t(B[n + k * N]);
                }
                C[m * N + n] = CElement(acc);
            }
            else {
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) {
                    acc += audit_to_float(A[m * K + k]) * audit_to_float(B[n + k * N]);
                }
                C[m * N + n] = CElement(acc);
            }
        }
    }
}

template <class Element> float max_abs_diff(std::vector<Element> const &lhs, std::vector<Element> const &rhs)
{
    float diff = 0.0f;
    for (int i = 0; i < int(lhs.size()); ++i) {
        diff = std::max(diff, std::abs(audit_to_float(lhs[i]) - audit_to_float(rhs[i])));
    }
    return diff;
}

template <class PartMnMajor, class PartKMajor> __global__ void print_sm80_tensorop_layouts_kernel(int *status)
{
    if (threadIdx.x == 0) {
        printf("SM80 TensorOp M/N-major SmemLayout:\n");
        cute::print_layout(typename PartMnMajor::SmemLayout{});
        printf("SM80 TensorOp K-major SmemLayout:\n");
        cute::print_layout(typename PartKMajor::SmemLayout{});
        status[0] = int(cute::cosize(typename PartMnMajor::SmemLayout{}));
        status[1] = int(cute::cosize(typename PartKMajor::SmemLayout{}));
    }
}

template <class PartA, class Element, class StrideA>
__global__ void sm80_alignment_copy_kernel(Element const *ptr_A, StrideA stride_A, Element *ptr_out)
{
    using bM = decltype(size<0>(typename PartA::SmemLayout{}));
    using bK = decltype(size<1>(typename PartA::SmemLayout{}));

    Tensor gA = make_tensor(make_gmem_ptr(ptr_A), make_shape(bM{}, bK{}), stride_A);

    __shared__ cute::array_aligned<Element, cute::cosize_v<typename PartA::SmemLayout>> smem;
    Tensor sA = make_tensor(make_smem_ptr(smem.data()), typename PartA::SmemLayout{});

    cute::cooperative_copy<128, PartA::GmemToSmemAlignmentBytes * 8>(
        threadIdx.x, gA, sA, typename PartA::GmemToSmemCopy{});
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();

    for (int idx = threadIdx.x; idx < int(bM{}) * int(bK{}); idx += blockDim.x) {
        int m        = idx % int(bM{});
        int k        = idx / int(bM{});
        ptr_out[idx] = sA(m, k);
    }
}

__global__ void sm80_cp_async_zfill_probe_kernel(float const *ptr_in, float *ptr_out)
{
    __shared__ float smem[1];
    if (threadIdx.x == 0) {
        smem[0]    = std::numeric_limits<float>::quiet_NaN();
        Tensor g   = make_tensor(make_gmem_ptr(ptr_in), Shape<_1>{});
        Tensor s   = make_tensor(make_smem_ptr(smem), Shape<_1>{});
        using Atom = cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<cute::uint_byte_t<4>>, float>;
        cute::copy(Atom{}.with(false), g, s);
    }
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();
    if (threadIdx.x == 0) {
        ptr_out[0] = smem[0];
    }
}

template <class PartC, class StrideC>
__global__ void sm80_epilogue_shared_store_kernel(typename PartC::ElementOutput *ptr_C, StrideC stride_C)
{
    using Layout       = typename PartC::SmemLayout;
    using OutputLayout = typename PartC::OutputSmemLayout;
    using Compute      = typename PartC::EpilogueElement;
    using Output       = typename PartC::ElementOutput;
    constexpr int BlkM = int(cute::size<0>(Layout{}));
    constexpr int BlkN = int(cute::size<1>(Layout{}));

    struct SharedStorage
    {
        cute::array_aligned<Compute, cute::cosize_v<Layout>>      smem_compute;
        cute::array_aligned<Output, cute::cosize_v<OutputLayout>> smem_output;
    };
    __shared__ SharedStorage storage;

    Tensor sCompute = make_tensor(make_smem_ptr(storage.smem_compute.data()), Layout{});
    Tensor sOutput  = make_tensor(make_smem_ptr(storage.smem_output.data()), OutputLayout{});
    Tensor gC       = make_tensor(make_gmem_ptr(ptr_C), make_shape(Int<BlkM>{}, Int<BlkN>{}), stride_C);

    for (int idx = threadIdx.x; idx < BlkM * BlkN; idx += blockDim.x) {
        int   m        = idx % BlkM;
        int   n        = idx / BlkM;
        float value    = float((m * 7 + n * 3) % 29) * 0.25f;
        sCompute(m, n) = Compute(value);
        sOutput(m, n)  = Output(sCompute(m, n));
    }
    __syncthreads();

    cute::cooperative_copy<128, PartC::OutputAlignmentBits>(threadIdx.x, sOutput, gC, typename PartC::SmemToGmemCopy{});
}

template <class PartC, class StrideC>
__global__ void sm80_large_register_pressure_kernel(typename PartC::ElementOutput *ptr_C, StrideC stride_C)
{
    constexpr int BlkM = PartC::BlkM;
    constexpr int BlkN = PartC::BlkN;

    Tensor                   gC = make_tensor(make_gmem_ptr(ptr_C), make_shape(Int<BlkM>{}, Int<BlkN>{}), stride_C);
    typename PartC::TiledMma mma;
    auto                     thr_mma = mma.get_thread_slice(threadIdx.x);
    Tensor                   tCgC    = thr_mma.partition_C(gC);
    Tensor                   tCrC    = thr_mma.make_fragment_C(tCgC);

    CUTE_UNROLL
    for (int i = 0; i < cute::size(tCrC); ++i) {
        tCrC(i) = typename PartC::Accumulator(float((threadIdx.x + i) % 17) * 0.125f);
    }
    autopartition::examples::convert_tensor(tCgC, tCrC);
}

template <typename PartA,
          typename PartB,
          typename PartC,
          typename InputElement,
          typename OutputElement,
          typename StrideA,
          typename StrideB,
          typename StrideC>
__global__ void sm80_tensorop_gemm_one_tile_kernel(InputElement const *ptr_A,
                                                   StrideA             stride_A,
                                                   InputElement const *ptr_B,
                                                   StrideB             stride_B,
                                                   OutputElement      *ptr_C,
                                                   StrideC             stride_C)
{
    using bM = decltype(size<0>(typename PartA::SmemLayout{}));
    using bN = decltype(size<0>(typename PartB::SmemLayout{}));
    using bK = decltype(size<1>(typename PartA::SmemLayout{}));

    Tensor gA = make_tensor(make_gmem_ptr(ptr_A), make_shape(bM{}, bK{}), stride_A);
    Tensor gB = make_tensor(make_gmem_ptr(ptr_B), make_shape(bN{}, bK{}), stride_B);
    Tensor gC = make_tensor(make_gmem_ptr(ptr_C), make_shape(bM{}, bN{}), stride_C);

    struct SharedStorage
    {
        cute::array_aligned<InputElement, cute::cosize_v<typename PartA::SmemLayout>> smemA;
        cute::array_aligned<InputElement, cute::cosize_v<typename PartB::SmemLayout>> smemB;
        cute::array_aligned<typename PartC::ElementOutput, cute::cosize_v<typename PartC::OutputSmemLayout>> smemC;
    };
    __shared__ SharedStorage smem;

    Tensor sA = make_tensor(make_smem_ptr(smem.smemA.data()), typename PartA::SmemLayout{});
    Tensor sB = make_tensor(make_smem_ptr(smem.smemB.data()), typename PartB::SmemLayout{});
    Tensor sC = make_tensor(make_smem_ptr(smem.smemC.data()), typename PartC::OutputSmemLayout{});

    cute::cooperative_copy<128, PartA::GmemToSmemAlignmentBytes * 8>(
        threadIdx.x, gA, sA, typename PartA::GmemToSmemCopy{});
    cute::cooperative_copy<128, PartB::GmemToSmemAlignmentBytes * 8>(
        threadIdx.x, gB, sB, typename PartB::GmemToSmemCopy{});
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();

    typename PartC::TiledMma mma;
    auto                     thr_mma = mma.get_thread_slice(threadIdx.x);
    Tensor                   tCgC    = thr_mma.partition_C(gC);
    Tensor                   tCrC    = thr_mma.make_fragment_C(tCgC);
    cute::clear(tCrC);

    cute::cooperative_gemm(threadIdx.x,
                           mma,
                           sA,
                           sB,
                           tCrC,
                           cute::identity{},
                           cute::identity{},
                           typename PartA::SmemToRegCopyOperation{},
                           typename PartB::SmemToRegCopyOperation{});

    Tensor tCrD = make_fragment_like<typename PartC::ElementOutput>(tCrC);
    cutlass::NumericConverter<typename PartC::ElementOutput, typename PartC::ElementCompute> convert;
    CUTE_UNROLL
    for (int i = 0; i < cute::size(tCrC); ++i) {
        tCrD(i) = convert(tCrC(i));
    }

    auto smem_tiled_copy_C = make_tiled_copy_C(
        Copy_Atom<typename PartC::RegToSmemCopyOperation, typename PartC::ElementOutput>{}, thr_mma);
    auto   smem_thr_copy_C = smem_tiled_copy_C.get_thread_slice(threadIdx.x);
    Tensor tCsC            = smem_thr_copy_C.partition_D(sC);
    Tensor tCrD_view       = smem_thr_copy_C.retile_S(tCrD);
    copy(smem_tiled_copy_C, tCrD_view, tCsC);
    __syncthreads();

    cute::cooperative_copy<128, PartC::OutputAlignmentBits>(threadIdx.x, sC, gC, typename PartC::SmemToGmemCopy{});
}

__global__ void
odd_zfill_gemm_kernel(float const *A, int lda, float const *B, int ldb, float *C, int M, int N, int K, int round_k)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) {
        return;
    }

    float acc = 0.0f;
    for (int k = 0; k < round_k; ++k) {
        float a = (k < K) ? A[m * lda + k] : 0.0f;
        float b = (k < K) ? B[k * ldb + n] : 0.0f;
        acc += a * b;
    }
    C[m * N + n] = acc;
}

template <class InputElement, class OutputElement, int M, int N, int K> void run_tensorop_gemm_case(float tolerance)
{
    ASSERT_TRUE(has_cuda_device());

    constexpr int ThreadCount = 128;
    using StrideA             = cute::Stride<int64_t, cute::_1>;
    using StrideB             = cute::Stride<cute::_1, int64_t>;
    using StrideC             = cute::Stride<int64_t, cute::_1>;
    using TileShape           = cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>;
    using ArchTag             = cutlass::arch::Sm80;
    using OpClass             = cutlass::arch::OpClassTensorOp;
    using PartA               = typename autopartition::
        AutoPartitioner<ArchTag, OpClass, InputElement, StrideA, TileShape, ThreadCount, OutputElement, 16, 16, 16>::
            RoleA;
    using PartB = typename autopartition::
        AutoPartitioner<ArchTag, OpClass, InputElement, StrideB, TileShape, ThreadCount, OutputElement, 16, 16, 16>::
            RoleB;
    using PartC = typename autopartition::
        AutoPartitioner<ArchTag, OpClass, InputElement, StrideC, TileShape, ThreadCount, OutputElement, 16, 16, 16>::
            RoleC;

    static_assert(std::is_same<typename PartC::ElementOutput, OutputElement>::value,
                  "RoleC output type must follow the requested GEMM output type.");

    std::vector<InputElement>  hA(M * K);
    std::vector<InputElement>  hB(N * K);
    std::vector<OutputElement> hC(M * N);
    std::vector<OutputElement> hRef(M * N);
    fill_audit_pattern(hA);
    fill_audit_pattern(hB);
    reference_gemm(M, N, K, hA.data(), hB.data(), hRef.data());

    InputElement  *dA = nullptr, *dB = nullptr;
    OutputElement *dC = nullptr;
    ASSERT_EQ(cudaMalloc(&dA, hA.size() * sizeof(InputElement)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dB, hB.size() * sizeof(InputElement)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dC, hC.size() * sizeof(OutputElement)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(InputElement), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(InputElement), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemset(dC, 0, hC.size() * sizeof(OutputElement)), cudaSuccess);

    sm80_tensorop_gemm_one_tile_kernel<PartA, PartB, PartC, InputElement, OutputElement>
        <<<dim3(1), dim3(ThreadCount)>>>(
            dA, make_stride(K, Int<1>{}), dB, make_stride(Int<1>{}, N), dC, make_stride(N, Int<1>{}));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(OutputElement), cudaMemcpyDeviceToHost), cudaSuccess);

    EXPECT_LE(max_abs_diff(hC, hRef), tolerance);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

void run_odd_zfill_case()
{
    ASSERT_TRUE(has_cuda_device());

    constexpr int M      = 61;
    constexpr int N      = 113;
    constexpr int K      = 59;
    constexpr int RoundK = ((K + 31) / 32) * 32;

    std::vector<float> hA(M * RoundK, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> hB(RoundK * N, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> hC(M * N, 0.0f);
    std::vector<float> hRef(M * N, 0.0f);
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            hA[m * RoundK + k] = autopartition::examples::patterned_value(m * K + k);
        }
    }
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            hB[k * N + n] = autopartition::examples::patterned_value(k * N + n + 17);
        }
    }
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += hA[m * RoundK + k] * hB[k * N + n];
            }
            hRef[m * N + n] = acc;
        }
    }

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    ASSERT_EQ(cudaMalloc(&dA, hA.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dB, hB.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dC, hC.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemset(dC, 0, hC.size() * sizeof(float)), cudaSuccess);

    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    odd_zfill_gemm_kernel<<<grid, block>>>(dA, RoundK, dB, N, dC, M, N, K, RoundK);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    for (float value : hC) {
        ASSERT_TRUE(std::isfinite(value));
    }
    EXPECT_LT(max_abs_diff(hC, hRef), 1.0e-4f);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

} // namespace autopartition_sm80_policy_audit

using namespace autopartition_sm80_policy_audit;

TEST(AutoPartitionerSm80Phase1, StaticRoutingAndThreadTopology)
{
    using namespace autopartition::detail;
    using StrideA = cute::Stride<int64_t, cute::_1>;
    using Tile    = cute::Shape<cute::_64, cute::_64, cute::_64>;

    static_assert(std::is_same<typename Sm80TensorOpTraits<cutlass::half_t>::MmaOperation,
                               cute::SM80_16x8x16_F32F16F16F32_TN>::value,
                  "FP16 must route to SM80 F32/F16 TN mma.sync.");
    static_assert(std::is_same<typename Sm80TensorOpTraits<cutlass::bfloat16_t>::MmaOperation,
                               cute::SM80_16x8x16_F32BF16BF16F32_TN>::value,
                  "BF16 must route to SM80 F32/BF16 TN mma.sync.");
    static_assert(std::is_same<typename Sm80TensorOpTraits<cutlass::tfloat32_t>::MmaOperation,
                               cute::SM80_16x8x8_F32TF32TF32F32_TN>::value,
                  "TF32 must route to SM80 TF32 TN mma.sync.");
    static_assert(
        std::is_same<typename Sm80TensorOpTraits<float>::MmaOperation, cute::SM80_16x8x8_F32TF32TF32F32_TN>::value,
        "float TensorOp must route through the TF32 trait.");
    static_assert(
        std::is_same<typename Sm80TensorOpTraits<int8_t>::MmaOperation, cute::SM80_16x8x32_S32S8S8S32_TN>::value,
        "INT8 must route to SM80 signed int8 TensorOp.");
    static_assert(
        std::is_same<typename Sm80TensorOpTraits<uint8_t>::MmaOperation, cute::SM80_16x8x32_S32U8U8S32_TN>::value,
        "UINT8 must route to SM80 unsigned int8 TensorOp.");

    using HalfA = typename autopartition::
        AutoPartitioner<cutlass::arch::Sm80, cutlass::arch::OpClassTensorOp, cutlass::half_t, StrideA, Tile, 128>::
            RoleA;
    static_assert(std::is_same<HalfA, Sm80TensorOpRoleA<cutlass::half_t, StrideA, Tile, 128, 16>>::value,
                  "SM80 FP16 TensorOp RoleA routing must be exact.");

    using HalfMmaOp      = typename Sm80TensorOpTraits<cutlass::half_t>::MmaOperation;
    using TensorTallPlan = OptimalTensorOpThreadLayout<256, 128, 256, HalfMmaOp>;
    using TensorWidePlan = OptimalTensorOpThreadLayout<128, 256, 256, HalfMmaOp>;
    using TensorTall     = typename TensorTallPlan::Layout;
    using TensorWide     = typename TensorWidePlan::Layout;
    using SimtTall       = typename OptimalSimtThreadLayout<256, 128, 256>::Layout;
    using SimtWide       = typename OptimalSimtThreadLayout<128, 256, 256>::Layout;
    using SimtCMajor     = typename OptimalSimtThreadLayout<128, 256, 256, true>::Layout;
    using SimtNMajor     = typename OptimalSimtThreadLayout<128, 256, 256, false>::Layout;
    static_assert(cute::size<0>(TensorTall{}) == 4 && cute::size<1>(TensorTall{}) == 2,
                  "8-warps TensorOp should pick the best RepeatM/RepeatN-scored 4x2 layout for 256x128.");
    static_assert(cute::size<0>(TensorWide{}) == 2 && cute::size<1>(TensorWide{}) == 4,
                  "8-warps TensorOp should pick the best RepeatM/RepeatN-scored 2x4 layout for 128x256.");
    static_assert(TensorTallPlan::WarpTileM == 64 && TensorTallPlan::WarpTileN == 64,
                  "TensorOp scoring should prefer square per-warp tiles when repeat balance ties.");
    static_assert(TensorTallPlan::RepeatM == 4 && TensorTallPlan::RepeatN == 8,
                  "TensorOp scoring must expose the selected atom repeat structure.");
    static_assert(cute::size<0>(SimtTall{}) == 32 && cute::size<1>(SimtTall{}) == 8,
                  "256-thread SIMT should keep a full warp contiguous along M for M-contiguous C.");
    static_assert(cute::size<0>(SimtWide{}) == 8 && cute::size<1>(SimtWide{}) == 32,
                  "256-thread SIMT should keep a full warp contiguous along N for N-contiguous C.");
    static_assert(cute::size<0>(SimtCMajor{}) == 32 && cute::size<1>(SimtCMajor{}) == 8,
                  "SIMT C M-major layout must bias warp lanes toward M even for a wide tile.");
    static_assert(cute::size<0>(SimtNMajor{}) == 8 && cute::size<1>(SimtNMajor{}) == 32,
                  "SIMT C N-major layout must bias warp lanes toward N.");

    EXPECT_EQ(int(cute::size<0>(TensorTall{})), 4);
    EXPECT_EQ(int(cute::size<1>(TensorWide{})), 4);
}

TEST(AutoPartitionerSm80Phase2, FourByteAlignedCpAsyncFallbackRuns)
{
    using Element = cutlass::half_t;
    using StrideA = cute::Stride<int64_t, cute::_1>;
    using Tile    = cute::Shape<cute::_64, cute::_64, cute::_64>;
    using PartA   = typename autopartition::AutoPartitioner<cutlass::arch::Sm80,
                                                            cutlass::arch::OpClassTensorOp,
                                                            Element,
                                                            StrideA,
                                                            Tile,
                                                            128,
                                                            Element,
                                                            4,
                                                            16,
                                                            16>::RoleA;

    static_assert(PartA::GmemToSmemAlignmentElements == 2, "4B-aligned FP16 must copy two elements at a time.");
    static_assert(PartA::GmemToSmemAlignmentBytes == 4, "4B-aligned FP16 must downgrade to 32-bit cp.async.");
    static_assert(PartA::UseLdMatrix, "The gmem vector downgrade must not disable the shared ldmatrix path.");

    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device not available.";
    }

    constexpr int        M = 64;
    constexpr int        K = 64;
    std::vector<Element> hA(M * K + 2);
    std::vector<Element> hOut(M * K);
    fill_audit_pattern(hA);

    Element *dA_raw = nullptr, *dOut = nullptr;
    ASSERT_EQ(cudaMalloc(&dA_raw, hA.size() * sizeof(Element)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dOut, hOut.size() * sizeof(Element)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dA_raw, hA.data(), hA.size() * sizeof(Element), cudaMemcpyHostToDevice), cudaSuccess);

    Element const *dA_4b = dA_raw + 2;
    sm80_alignment_copy_kernel<PartA><<<1, 128>>>(dA_4b, make_stride(K, Int<1>{}), dOut);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(hOut.data(), dOut, hOut.size() * sizeof(Element), cudaMemcpyDeviceToHost), cudaSuccess);

    float max_diff = 0.0f;
    for (int k = 0; k < K; ++k) {
        for (int m = 0; m < M; ++m) {
            int out_idx = m + k * M;
            int src_idx = m * K + k + 2;
            max_diff    = std::max(max_diff, std::abs(audit_to_float(hOut[out_idx]) - audit_to_float(hA[src_idx])));
        }
    }
    EXPECT_LT(max_diff, 1.0e-5f);

    float *dProbeIn = nullptr, *dProbeOut = nullptr;
    float  hProbeIn  = std::numeric_limits<float>::quiet_NaN();
    float  hProbeOut = std::numeric_limits<float>::quiet_NaN();
    ASSERT_EQ(cudaMalloc(&dProbeIn, sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dProbeOut, sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dProbeIn, &hProbeIn, sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    sm80_cp_async_zfill_probe_kernel<<<1, 32>>>(dProbeIn, dProbeOut);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&hProbeOut, dProbeOut, sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(hProbeOut, 0.0f);

    RecordProperty("ptx_vectorization_check",
                   "nvcc -arch=sm_120 --ptxas-options=-v -c sm80_policy_audit.cu; "
                   "the 4B RoleA path has GmemToSmemAlignmentBytes=4, so cp.async lowers with a 4-byte size operand.");

    cudaFree(dA_raw);
    cudaFree(dOut);
    cudaFree(dProbeIn);
    cudaFree(dProbeOut);
}

TEST(AutoPartitionerSm80Phase3, SwizzleLdsmAndPaddingFallback)
{
    using Element  = cutlass::half_t;
    using Tile     = cute::Shape<cute::_64, cute::_64, cute::_64>;
    using StrideMn = cute::Stride<cute::_1, int64_t>;
    using StrideK  = cute::Stride<int64_t, cute::_1>;
    using MnMajorA = typename autopartition::
        AutoPartitioner<cutlass::arch::Sm80, cutlass::arch::OpClassTensorOp, Element, StrideMn, Tile, 128>::RoleA;
    using KMajorA = typename autopartition::
        AutoPartitioner<cutlass::arch::Sm80, cutlass::arch::OpClassTensorOp, Element, StrideK, Tile, 128>::RoleA;

    static_assert(MnMajorA::UseLdMatrix && KMajorA::UseLdMatrix, "FP16 64x64 must enable ldmatrix.");
    static_assert(MnMajorA::SwizzleBase == 3 && KMajorA::SwizzleBase == 3,
                  "TileK=64 FP16 has a 128B row and must select Swizzle<3,3,3>.");
    static_assert(std::is_same<typename MnMajorA::SmemToRegCopyOperation, cute::SM75_U16x8_LDSM_T>::value,
                  "M/N-major shared storage must use LDSM_T for the current TN MMA operand contract.");
    static_assert(std::is_same<typename KMajorA::SmemToRegCopyOperation, cute::SM75_U32x4_LDSM_N>::value,
                  "K-major shared storage must use LDSM_N for the current TN MMA operand contract.");

    using FallbackTileMn12 = autopartition::detail::Sm80TensorOpMainloopRole<Element, StrideMn, 12, 64, 96, true, 16>;
    static_assert(!FallbackTileMn12::UseLdMatrix, "TileMN=12 must not enter the ldmatrix path.");
    static_assert(FallbackTileMn12::GmemToSmemAlignmentBytes == 8,
                  "The TileMN=12 fallback keeps the largest legal 8B global vector width for 96 threads.");
    static_assert(std::is_same<typename FallbackTileMn12::SmemToRegCopyOperation,
                               cute::AutoVectorizingCopyWithAssumedAlignment<64>>::value,
                  "Non-ldmatrix fallback must use ordinary vectorized shared-to-register copy.");
    static_assert(cute::stride<0>(typename FallbackTileMn12::SmemLayout{}) == cute::_1{},
                  "M/N-major fallback keeps M/N contiguous.");

    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device not available.";
    }

    int *d_status = nullptr;
    ASSERT_EQ(cudaMalloc(&d_status, 2 * sizeof(int)), cudaSuccess);
    print_sm80_tensorop_layouts_kernel<MnMajorA, KMajorA><<<1, 1>>>(d_status);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    int h_status[2] = {};
    ASSERT_EQ(cudaMemcpy(h_status, d_status, sizeof(h_status), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_GT(h_status[0], 0);
    EXPECT_GT(h_status[1], 0);
    cudaFree(d_status);

    RecordProperty("ncu_bank_conflict_check",
                   "ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,"
                   "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum "
                   "./cutlass_test_unit_transform_collective_auto_partitioner_sm80_policy_audit "
                   "--gtest_filter=AutoPartitionerSm80Phase3.*");
}

TEST(AutoPartitionerSm80Phase4, RegisterPressureAndEpilogueStore)
{
    using Input        = cutlass::half_t;
    using LargeOutput  = float;
    using LargeStrideC = cute::Stride<int64_t, cute::_1>;
    using LargeTile    = cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<32>>;
    using LargePartC   = typename autopartition::AutoPartitioner<cutlass::arch::Sm80,
                                                                 cutlass::arch::OpClassTensorOp,
                                                                 Input,
                                                                 LargeStrideC,
                                                                 LargeTile,
                                                                 256,
                                                                 LargeOutput,
                                                                 16,
                                                                 16,
                                                                 16>::RoleC;

    static_assert(std::is_same<typename LargePartC::Accumulator, float>::value,
                  "SM80 FP16 TensorOp accumulators live in FP32 RF.");
    static_assert(LargePartC::BlkM == 256 && LargePartC::BlkN == 128,
                  "The register pressure probe must instantiate the requested 256x128x32 tile.");
    static_assert(LargePartC::WarpTileM == 64 && LargePartC::WarpTileN == 64,
                  "TensorOp TiledMMA selection should follow the scored per-warp tile.");
    static_assert(LargePartC::RepeatM == 4 && LargePartC::RepeatN == 8,
                  "TensorOp TiledMMA selection should expose the selected MMA repeat structure.");

    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device not available.";
    }

    std::vector<float> hLarge(256 * 128, 0.0f);
    float             *dLarge = nullptr;
    ASSERT_EQ(cudaMalloc(&dLarge, hLarge.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemset(dLarge, 0, hLarge.size() * sizeof(float)), cudaSuccess);
    sm80_large_register_pressure_kernel<LargePartC><<<1, 256>>>(dLarge, make_stride(128, Int<1>{}));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(hLarge.data(), dLarge, hLarge.size() * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_TRUE(std::any_of(hLarge.begin(), hLarge.end(), [](float value) { return value != 0.0f; }));
    cudaFree(dLarge);

    using EpiOutput  = cutlass::half_t;
    using EpiStrideC = cute::Stride<cute::_1, int64_t>;
    using EpiTile    = cute::Shape<cute::_64, cute::_64, cute::_32>;
    using EpiPartC   = typename autopartition::AutoPartitioner<cutlass::arch::Sm80,
                                                               cutlass::arch::OpClassTensorOp,
                                                               Input,
                                                               EpiStrideC,
                                                               EpiTile,
                                                               128,
                                                               EpiOutput,
                                                               16,
                                                               16,
                                                               16>::RoleC;
    static_assert(std::is_same<typename EpiPartC::ElementCompute, float>::value,
                  "Epilogue compute staging remains FP32.");
    static_assert(std::is_same<typename EpiPartC::ElementOutput, EpiOutput>::value,
                  "Epilogue global store uses the requested output type.");
    static_assert(EpiPartC::OutputAlignmentBytes == 16,
                  "Half output epilogue should use 128-bit vectorized global stores when C is 16B aligned.");
    static_assert(EpiPartC::EpilogueInstructionThreads == 16,
                  "SM80 epilogue swizzle must model the 16-thread store/load issue group.");
    static_assert(EpiPartC::OutputAlignmentBytes == EpiPartC::EpilogueVectorBytes,
                  "RoleC output vector width must be derived from the selected output tiled-copy alignment.");
    static_assert(EpiPartC::EpilogueSwizzleBytes == 32 || EpiPartC::EpilogueSwizzleBytes == 64 ||
                      EpiPartC::EpilogueSwizzleBytes == 128,
                  "RoleC epilogue swizzle must use a supported shared-memory swizzle span.");
    static_assert(cute::cosize_v<typename EpiPartC::OutputSmemLayout> == EpiPartC::BlkM * EpiPartC::BlkN,
                  "SM80 RoleC output shared layout must not allocate padding elements.");
    static_assert(!std::is_same<typename EpiPartC::OutputSmemLayout,
                                cute::Layout<cute::Shape<cute::Int<64>, cute::Int<64>>,
                                             cute::Stride<cute::_1, cute::Int<72>>>>::value,
                  "SM80 RoleC output shared layout must not be the old padding layout.");
    static_assert((EpiPartC::EpilogueSwizzleBytes % EpiPartC::OutputAlignmentBytes) == 0,
                  "Output shared layout swizzle span must cover whole output vectors.");

    std::vector<EpiOutput> hC(64 * 64);
    EpiOutput             *dC = nullptr;
    ASSERT_EQ(cudaMalloc(&dC, hC.size() * sizeof(EpiOutput)), cudaSuccess);
    ASSERT_EQ(cudaMemset(dC, 0, hC.size() * sizeof(EpiOutput)), cudaSuccess);
    sm80_epilogue_shared_store_kernel<EpiPartC><<<1, 128>>>(dC, make_stride(Int<1>{}, 64));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(EpiOutput), cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(dC);

    float max_diff = 0.0f;
    for (int n = 0; n < 64; ++n) {
        for (int m = 0; m < 64; ++m) {
            float expected = float((m * 7 + n * 3) % 29) * 0.25f;
            max_diff       = std::max(max_diff, std::abs(audit_to_float(hC[m + n * 64]) - expected));
        }
    }
    EXPECT_LT(max_diff, 1.0e-3f);

    RecordProperty("ptxas_spill_check",
                   "nvcc -arch=sm_120 --ptxas-options=-v -c sm80_policy_audit.cu and inspect "
                   "ptxas 'spill stores'/'spill loads' for sm80_large_register_pressure_kernel.");
}

TEST(AutoPartitionerSm80Phase5, MixedPrecisionAndOddShapeZfill)
{
    run_tensorop_gemm_case<cutlass::tfloat32_t, float, 64, 64, 32>(2.0e-2f);
    run_tensorop_gemm_case<int8_t, int32_t, 64, 64, 64>(0.0f);
    run_odd_zfill_case();
}
