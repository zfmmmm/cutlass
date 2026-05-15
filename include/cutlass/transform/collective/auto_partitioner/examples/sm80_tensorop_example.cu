#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <iostream>
#include <type_traits>
#include <vector>

#include "autopartition_example_utils.hpp"
//
#include "auto_partitioner_builder.hpp"

using namespace cute;

// SM80 + OpClassTensorOp 的真实 GEMM 示例。
// AutoPartitioner 只生成 A/B/C 的 layout、copy op 和 TiledMma；shared memory
// 分配、同步、C tile 清零和写回顺序都在 kernel 里显式表达，保持生成器纯净。
template <typename PartA,
          typename PartB,
          typename PartC,
          typename InputElement,
          typename OutputElement,
          typename StrideA,
          typename StrideB,
          typename StrideC>
__global__ void sm80_tensorop_autopartition_kernel(InputElement const *ptr_A,
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
    };
    __shared__ SharedStorage smem;

    Tensor sA = make_tensor(make_smem_ptr(smem.smemA.data()), typename PartA::SmemLayout{});
    Tensor sB = make_tensor(make_smem_ptr(smem.smemB.data()), typename PartB::SmemLayout{});

    // swizzled shared layout 使用 cooperative_copy + AutoCopyAsync 填充，避免把
    // gmem->smem 的连续向量化假设硬塞进 swizzle 地址空间。
    cooperative_copy<128, 128>(threadIdx.x, gA, sA, typename PartA::GmemToSmemCopy{});
    cooperative_copy<128, 128>(threadIdx.x, gB, sB, typename PartB::GmemToSmemCopy{});
    cp_async_fence();
    cp_async_wait<0>();

    __syncthreads();

    typename PartC::TiledMma mma;
    auto                     thr_mma = mma.get_thread_slice(threadIdx.x);
    Tensor                   tCgC    = thr_mma.partition_C(gC);
    Tensor                   tCrC    = thr_mma.make_fragment_C(tCgC);
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
    cute::copy(tCrC, tCgC);
}

int main()
{
    constexpr int M           = 64;
    constexpr int N           = 64;
    constexpr int K           = 64;
    constexpr int ThreadCount = 128;

    using InputElement  = cutlass::half_t;
    using OutputElement = float;
    using StrideA       = cute::Stride<int64_t, cute::_1>;
    using StrideB       = cute::Stride<cute::_1, int64_t>;
    using StrideC       = cute::Stride<int64_t, cute::_1>;
    using TileShape     = cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>;
    using ArchTag       = cutlass::arch::Sm80;
    using OpClass       = cutlass::arch::OpClassTensorOp;

    using PartA =
        typename autopartition::AutoPartitioner<ArchTag, OpClass, InputElement, StrideA, TileShape, ThreadCount>::RoleA;
    using PartB =
        typename autopartition::AutoPartitioner<ArchTag, OpClass, InputElement, StrideB, TileShape, ThreadCount>::RoleB;
    using PartC =
        typename autopartition::AutoPartitioner<ArchTag, OpClass, InputElement, StrideC, TileShape, ThreadCount>::RoleC;

    static_assert(std::is_same<typename PartC::Accumulator, OutputElement>::value,
                  "SM80 half TensorOp example stores FP32 accumulators.");
    static_assert(cute::cosize_v<typename PartA::SmemLayout> > 0, "PartA swizzled smem layout must be valid.");
    static_assert(cute::cosize_v<typename PartB::SmemLayout> > 0, "PartB swizzled smem layout must be valid.");

    std::vector<InputElement>  hA(M * K);
    std::vector<InputElement>  hB(N * K);
    std::vector<OutputElement> hRef(M * N);
    std::vector<OutputElement> hAuto(M * N);
    std::vector<OutputElement> hBaseline(M * N);
    autopartition::examples::fill_pattern(hA);
    autopartition::examples::fill_pattern(hB);
    autopartition::examples::reference_gemm(M, N, K, hA.data(), K, 1, hB.data(), 1, N, hRef.data(), N, 1);

    InputElement  *dA        = nullptr;
    InputElement  *dB        = nullptr;
    OutputElement *dC        = nullptr;
    OutputElement *dBaseline = nullptr;
    cudaMalloc(&dA, M * K * sizeof(InputElement));
    cudaMalloc(&dB, N * K * sizeof(InputElement));
    cudaMalloc(&dC, M * N * sizeof(OutputElement));
    cudaMalloc(&dBaseline, M * N * sizeof(OutputElement));
    cudaMemcpy(dA, hA.data(), M * K * sizeof(InputElement), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), N * K * sizeof(InputElement), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, M * N * sizeof(OutputElement));
    cudaMemset(dBaseline, 0, M * N * sizeof(OutputElement));

    sm80_tensorop_autopartition_kernel<PartA, PartB, PartC, InputElement, OutputElement>
        <<<dim3(1), dim3(ThreadCount)>>>(
            dA, make_stride(K, Int<1>{}), dB, make_stride(Int<1>{}, N), dC, make_stride(N, Int<1>{}));
    cudaError_t err = cudaDeviceSynchronize();
    if (!autopartition::examples::check_cuda(err, "sm80_tensorop_autopartition_kernel")) {
        return 1;
    }

    autopartition::examples::conventional_gemm_kernel<InputElement, InputElement, OutputElement>
        <<<dim3((M + 15) / 16, (N + 15) / 16), dim3(16, 16)>>>(dA, K, 1, dB, 1, N, dBaseline, N, 1, M, N, K);
    err = cudaDeviceSynchronize();
    if (!autopartition::examples::check_cuda(err, "sm80_tensorop_conventional_kernel")) {
        return 1;
    }

    cudaMemcpy(hAuto.data(), dC, M * N * sizeof(OutputElement), cudaMemcpyDeviceToHost);
    cudaMemcpy(hBaseline.data(), dBaseline, M * N * sizeof(OutputElement), cudaMemcpyDeviceToHost);

    float max_error = std::max(autopartition::examples::max_abs_diff(hAuto, hRef),
                               autopartition::examples::max_abs_diff(hBaseline, hRef));

    float auto_ms = autopartition::examples::time_launch_ms(
        [&]() {
            sm80_tensorop_autopartition_kernel<PartA, PartB, PartC, InputElement, OutputElement>
                <<<dim3(1), dim3(ThreadCount)>>>(
                    dA, make_stride(K, Int<1>{}), dB, make_stride(Int<1>{}, N), dC, make_stride(N, Int<1>{}));
        },
        50);
    float baseline_ms = autopartition::examples::time_launch_ms(
        [&]() {
            autopartition::examples::conventional_gemm_kernel<InputElement, InputElement, OutputElement>
                <<<dim3((M + 15) / 16, (N + 15) / 16), dim3(16, 16)>>>(dA, K, 1, dB, 1, N, dBaseline, N, 1, M, N, K);
        },
        50);

    autopartition::examples::print_result(
        "sm80_tensorop_example: AutoPartitioner GEMM vs conventional GEMM", max_error, auto_ms, baseline_ms);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dBaseline);
    return max_error < 2.0e-2f ? 0 : 1;
}
