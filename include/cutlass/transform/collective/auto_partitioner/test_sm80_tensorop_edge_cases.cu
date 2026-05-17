#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <iostream>
#include <type_traits>
#include <vector>

#include "autopartition_example_utils.hpp"
#include "auto_partitioner_builder.hpp"

using namespace cute;

template <typename PartA,
          typename PartB,
          typename PartC,
          typename InputElement,
          typename OutputElement,
          typename StrideA,
          typename StrideB,
          typename StrideC>
__global__ void sm80_edge_tensorop_kernel(InputElement const *ptr_A,
                                          StrideA stride_A,
                                          InputElement const *ptr_B,
                                          StrideB stride_B,
                                          OutputElement *ptr_C,
                                          StrideC stride_C)
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
    };
    __shared__ SharedStorage smem;

    Tensor sA = make_tensor(make_smem_ptr(smem.smemA.data()), typename PartA::SmemLayout{});
    Tensor sB = make_tensor(make_smem_ptr(smem.smemB.data()), typename PartB::SmemLayout{});

    cooperative_copy<128, PartA::GmemToSmemAlignmentBytes * 8>(threadIdx.x, gA, sA, typename PartA::GmemToSmemCopy{});
    cooperative_copy<128, PartB::GmemToSmemAlignmentBytes * 8>(threadIdx.x, gB, sB, typename PartB::GmemToSmemCopy{});
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    typename PartC::TiledMma mma;
    auto thr_mma = mma.get_thread_slice(threadIdx.x);
    Tensor tCgC = thr_mma.partition_C(gC);
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);
    clear(tCrC);

    cooperative_gemm(threadIdx.x,
                     mma,
                     sA,
                     sB,
                     tCrC,
                     identity{},
                     identity{},
                     typename PartA::SmemToRegCopyOperation{},
                     typename PartB::SmemToRegCopyOperation{});

    autopartition::examples::convert_tensor(tCgC, tCrC);
}

bool negative_comparison_sanity()
{
    std::vector<cutlass::half_t> values(16);
    std::vector<cutlass::half_t> reference(16);
    autopartition::examples::fill_pattern(values);
    reference = values;
    autopartition::examples::corrupt_first(values);
    float diff = autopartition::examples::max_abs_diff(int(values.size()), values.data(), reference.data());
    std::cout << "negative_comparison_sanity diff = " << diff << "\n";
    return diff > 0.5f;
}

template <int K, int AlignA, int AlignB, class OutputElement>
bool run_half_case(char const *name)
{
    constexpr int M = 64;
    constexpr int N = 64;
    constexpr int ThreadCount = 128;

    using InputElement = cutlass::half_t;
    using StrideA = cute::Stride<int64_t, cute::_1>;
    using StrideB = cute::Stride<cute::_1, int64_t>;
    using StrideC = cute::Stride<int64_t, cute::_1>;
    using TileShape = cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>;
    using ArchTag = cutlass::arch::Sm80;
    using OpClass = cutlass::arch::OpClassTensorOp;
    using PartA = typename autopartition::AutoPartitioner<ArchTag, OpClass, InputElement, StrideA, TileShape, ThreadCount,
                                                          OutputElement, AlignA, AlignB, 4>::RoleA;
    using PartB = typename autopartition::AutoPartitioner<ArchTag, OpClass, InputElement, StrideB, TileShape, ThreadCount,
                                                          OutputElement, AlignA, AlignB, 4>::RoleB;
    using PartC = typename autopartition::AutoPartitioner<ArchTag, OpClass, InputElement, StrideC, TileShape, ThreadCount,
                                                          OutputElement, AlignA, AlignB, 4>::RoleC;

    static_assert(std::is_same<typename PartC::ElementOutput, OutputElement>::value, "Output type must flow to RoleC.");

    std::vector<InputElement> hA(M * K + 2);
    std::vector<InputElement> hB(N * K + 2);
    std::vector<float> hRef(M * N);
    std::vector<OutputElement> hAuto(M * N);
    autopartition::examples::fill_pattern(hA);
    autopartition::examples::fill_pattern(hB);

    InputElement const *hostA = hA.data() + ((AlignA == 4) ? 2 : 0);
    InputElement const *hostB = hB.data() + ((AlignB == 4) ? 2 : 0);
    autopartition::examples::reference_gemm(M, N, K, hostA, K, 1, hostB, 1, N, hRef.data(), N, 1);

    InputElement *dA_raw = nullptr;
    InputElement *dB_raw = nullptr;
    OutputElement *dC = nullptr;
    cudaMalloc(&dA_raw, (M * K + 2) * sizeof(InputElement));
    cudaMalloc(&dB_raw, (N * K + 2) * sizeof(InputElement));
    cudaMalloc(&dC, M * N * sizeof(OutputElement));
    cudaMemcpy(dA_raw, hA.data(), (M * K + 2) * sizeof(InputElement), cudaMemcpyHostToDevice);
    cudaMemcpy(dB_raw, hB.data(), (N * K + 2) * sizeof(InputElement), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, M * N * sizeof(OutputElement));

    InputElement const *dA = dA_raw + ((AlignA == 4) ? 2 : 0);
    InputElement const *dB = dB_raw + ((AlignB == 4) ? 2 : 0);
    sm80_edge_tensorop_kernel<PartA, PartB, PartC, InputElement, OutputElement>
        <<<dim3(1), dim3(ThreadCount)>>>(dA, make_stride(K, Int<1>{}),
                                         dB, make_stride(Int<1>{}, N),
                                         dC, make_stride(N, Int<1>{}));
    cudaError_t err = cudaDeviceSynchronize();
    if (!autopartition::examples::check_cuda(err, name)) {
        cudaFree(dA_raw);
        cudaFree(dB_raw);
        cudaFree(dC);
        return false;
    }

    cudaMemcpy(hAuto.data(), dC, M * N * sizeof(OutputElement), cudaMemcpyDeviceToHost);
    float diff = autopartition::examples::max_abs_diff(M * N, hAuto.data(), hRef.data());
    std::cout << name << " diff = " << diff << "\n";

    cudaFree(dA_raw);
    cudaFree(dB_raw);
    cudaFree(dC);
    return diff < (std::is_same<OutputElement, cutlass::half_t>::value ? 2.0e-2f : 2.0e-2f);
}

int main()
{
    bool ok = true;
    ok = negative_comparison_sanity() && ok;
    ok = run_half_case<64, 4, 16, float>("half_k64_a4_float_c") && ok;
    ok = run_half_case<64, 16, 16, cutlass::half_t>("half_k64_half_c") && ok;
    ok = run_half_case<32, 16, 16, float>("half_k32_float_c") && ok;
    return ok ? 0 : 1;
}
