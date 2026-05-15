#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <iostream>
#include <vector>

#include "autopartition_example_utils.hpp"
//
#include "auto_partitioner_builder.hpp"

using namespace cute;

// SM100/SM120 + OpClassSimt 的单 CTA 示例。SM100 SIMT builder 只支持 SGEMM，
// 并要求 TileK=16、ThreadCount 等于官方 warp-shape selector 的结果。
template <typename PartA,
          typename PartB,
          typename PartC,
          typename Element,
          typename StrideA,
          typename StrideB,
          typename StrideC>
__global__ void sm100_simt_autopartition_kernel(Element const *ptr_A,
                                                StrideA        stride_A,
                                                Element const *ptr_B,
                                                StrideB        stride_B,
                                                Element       *ptr_C,
                                                StrideC        stride_C)
{
    using bM = decltype(size<0>(typename PartA::SmemLayout{}));
    using bN = decltype(size<0>(typename PartB::SmemLayout{}));
    using bK = decltype(size<1>(typename PartA::SmemLayout{}));

    Tensor gA = make_tensor(make_gmem_ptr(ptr_A), make_shape(bM{}, bK{}), stride_A);
    Tensor gB = make_tensor(make_gmem_ptr(ptr_B), make_shape(bN{}, bK{}), stride_B);
    Tensor gC = make_tensor(make_gmem_ptr(ptr_C), make_shape(bM{}, bN{}), stride_C);

    union SharedStorage
    {
        struct
        {
            cute::array_aligned<Element, cute::cosize_v<typename PartA::SmemLayout>> smemA;
            cute::array_aligned<Element, cute::cosize_v<typename PartB::SmemLayout>> smemB;
        } ab;
        cute::array_aligned<Element, cute::cosize_v<typename PartC::SmemLayout>> smemC;
    };
    __shared__ SharedStorage smem;

    Tensor sA = make_tensor(make_smem_ptr(smem.ab.smemA.data()), typename PartA::SmemLayout{});
    Tensor sB = make_tensor(make_smem_ptr(smem.ab.smemB.data()), typename PartB::SmemLayout{});
    Tensor sC = make_tensor(make_smem_ptr(smem.smemC.data()), typename PartC::SmemLayout{});

    constexpr int Pad = 4;
    for (int p = threadIdx.x; p < (int(bM{}) + Pad) * int(bK{}); p += 128) {
        int m = p % (int(bM{}) + Pad);
        int k = p / (int(bM{}) + Pad);
        if (m < int(bM{})) {
            sA(m, k) = gA(m, k);
        }
    }
    for (int p = threadIdx.x; p < int(bN{}) * (int(bK{}) + Pad); p += 128) {
        int k = p % (int(bK{}) + Pad);
        int n = p / (int(bK{}) + Pad);
        if (k < int(bK{})) {
            sB(n, k) = gB(n, k);
        }
    }
    __syncthreads();

    typename PartC::TiledMma mma;
    auto                     thr_mma = mma.get_thread_slice(threadIdx.x);

    Tensor tCrA = thr_mma.partition_fragment_A(sA);
    Tensor tCrB = thr_mma.partition_fragment_B(sB);
    Tensor tCrC = thr_mma.partition_fragment_C(sC);
    clear(tCrC);

    cute::copy(typename PartA::SmemToRegCopy{}, thr_mma.partition_A(sA), tCrA);
    cute::copy(typename PartB::SmemToRegCopy{}, thr_mma.partition_B(sB), tCrB);
    cute::gemm(mma, tCrA, tCrB, tCrC);
    __syncthreads();

    cute::copy(tCrC, thr_mma.partition_C(gC));
}

int main()
{
    constexpr int M           = 64;
    constexpr int N           = 64;
    constexpr int K           = 16;
    constexpr int ThreadCount = 128;

    using Element   = float;
    using StrideA   = cute::Stride<cute::_1, int64_t>;
    using StrideB   = cute::Stride<int64_t, cute::_1>;
    using StrideC   = cute::Stride<cute::_1, int64_t>;
    using TileShape = cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>;

    // sm_120 构建时使用 ArchTag=Sm120。该路径在 policy 中选择可实际执行的
    // UniversalFMA SIMT 图纸，避免 sm_120 上误用 SM100 f32x2 保护宏。
    using ArchTag = cutlass::arch::Sm120;
    using OpClass = cutlass::arch::OpClassSimt;

    using PartA =
        typename autopartition::AutoPartitioner<ArchTag, OpClass, Element, StrideA, TileShape, ThreadCount>::RoleA;
    using PartB =
        typename autopartition::AutoPartitioner<ArchTag, OpClass, Element, StrideB, TileShape, ThreadCount>::RoleB;
    using PartC =
        typename autopartition::AutoPartitioner<ArchTag, OpClass, Element, StrideC, TileShape, ThreadCount>::RoleC;

    std::vector<Element> hA(M * K);
    std::vector<Element> hB(N * K);
    std::vector<float>   hRef(M * N);
    std::vector<float>   hAuto(M * N);
    std::vector<float>   hBaseline(M * N);
    autopartition::examples::fill_pattern(hA);
    autopartition::examples::fill_pattern(hB);
    autopartition::examples::reference_gemm(M, N, K, hA.data(), 1, M, hB.data(), K, 1, hRef.data(), 1, M);

    Element *dA = nullptr, *dB = nullptr, *dC = nullptr, *dBaseline = nullptr;
    cudaMalloc(&dA, M * K * sizeof(Element));
    cudaMalloc(&dB, N * K * sizeof(Element));
    cudaMalloc(&dC, M * N * sizeof(Element));
    cudaMalloc(&dBaseline, M * N * sizeof(Element));
    cudaMemcpy(dA, hA.data(), M * K * sizeof(Element), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), N * K * sizeof(Element), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, M * N * sizeof(Element));
    cudaMemset(dBaseline, 0, M * N * sizeof(Element));

    sm100_simt_autopartition_kernel<PartA, PartB, PartC, Element><<<dim3(1), dim3(ThreadCount)>>>(
        dA, make_stride(Int<1>{}, M), dB, make_stride(K, Int<1>{}), dC, make_stride(Int<1>{}, M));
    cudaError_t err = cudaDeviceSynchronize();
    if (!autopartition::examples::check_cuda(err, "sm100_simt_autopartition_kernel")) {
        return 1;
    }

    autopartition::examples::conventional_gemm_kernel<Element, Element, Element>
        <<<dim3((M + 15) / 16, (N + 15) / 16), dim3(16, 16)>>>(dA, 1, M, dB, K, 1, dBaseline, 1, M, M, N, K);
    err = cudaDeviceSynchronize();
    if (!autopartition::examples::check_cuda(err, "sm100_simt_conventional_kernel")) {
        return 1;
    }

    cudaMemcpy(hAuto.data(), dC, M * N * sizeof(Element), cudaMemcpyDeviceToHost);
    cudaMemcpy(hBaseline.data(), dBaseline, M * N * sizeof(Element), cudaMemcpyDeviceToHost);

    float max_error = std::max(autopartition::examples::max_abs_diff(hAuto, hRef),
                               autopartition::examples::max_abs_diff(hBaseline, hRef));
    float auto_ms   = autopartition::examples::time_launch_ms(
        [&]() {
            sm100_simt_autopartition_kernel<PartA, PartB, PartC, Element><<<dim3(1), dim3(ThreadCount)>>>(
                dA, make_stride(Int<1>{}, M), dB, make_stride(K, Int<1>{}), dC, make_stride(Int<1>{}, M));
        },
        50);
    float baseline_ms = autopartition::examples::time_launch_ms(
        [&]() {
            autopartition::examples::conventional_gemm_kernel<Element, Element, Element>
                <<<dim3((M + 15) / 16, (N + 15) / 16), dim3(16, 16)>>>(dA, 1, M, dB, K, 1, dBaseline, 1, M, M, N, K);
        },
        50);

    autopartition::examples::print_result(
        "sm100_simt_example(sm_120 build): AutoPartitioner GEMM vs conventional GEMM", max_error, auto_ms, baseline_ms);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dBaseline);
    return max_error < 1.0e-4f ? 0 : 1;
}
