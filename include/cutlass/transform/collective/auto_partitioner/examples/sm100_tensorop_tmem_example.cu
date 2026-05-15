#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <iostream>
#include <type_traits>
#include <vector>

#include "autopartition_example_utils.hpp"
//
#include "auto_partitioner_builder.hpp"

using namespace cute;

// 这个文件同时体现两个事实：
// 1. ArchTag=Sm100 + OpClassTensorOp 会生成 UMMA/TMEM 蓝图，证明 SM100 路由
//    支持 Tensor Memory accumulator。
// 2. RTX 50 / sm_120 实际可执行的高阶 TensorOp 是 SM120 FP8/F6/F4 MMA，
//    因此下面的 kernel 使用 ArchTag=Sm120 + FP8 输入做真实 GEMM。
template <typename PartA,
          typename PartB,
          typename PartC,
          typename InputElement,
          typename OutputElement,
          typename StrideA,
          typename StrideB,
          typename StrideC>
__global__ void sm120_tensorop_autopartition_kernel(InputElement const *ptr_A,
                                                    StrideA             stride_A,
                                                    InputElement const *ptr_B,
                                                    StrideB             stride_B,
                                                    OutputElement      *ptr_C,
                                                    StrideC             stride_C)
{
    using bM          = decltype(size<0>(typename PartA::SmemLayout{}));
    using bN          = decltype(size<0>(typename PartB::SmemLayout{}));
    using bK          = decltype(size<1>(typename PartA::SmemLayout{}));
    using SmemElement = typename PartA::SmemElement;

    auto const *raw_A = reinterpret_cast<SmemElement const *>(ptr_A);
    auto const *raw_B = reinterpret_cast<SmemElement const *>(ptr_B);

    Tensor gA = make_tensor(make_gmem_ptr(raw_A), make_shape(bM{}, bK{}), stride_A);
    Tensor gB = make_tensor(make_gmem_ptr(raw_B), make_shape(bN{}, bK{}), stride_B);
    Tensor gC = make_tensor(make_gmem_ptr(ptr_C), make_shape(bM{}, bN{}), stride_C);

    struct SharedStorage
    {
        cute::array_aligned<SmemElement, cute::cosize_v<typename PartA::SmemLayout>> smemA;
        cute::array_aligned<SmemElement, cute::cosize_v<typename PartB::SmemLayout>> smemB;
    };
    __shared__ SharedStorage smem;

    Tensor sA = make_tensor(make_smem_ptr(smem.smemA.data()), typename PartA::SmemLayout{});
    Tensor sB = make_tensor(make_smem_ptr(smem.smemB.data()), typename PartB::SmemLayout{});

    cooperative_copy<256, 128>(threadIdx.x, gA, sA, typename PartA::GmemToSmemCopy{});
    cooperative_copy<256, 128>(threadIdx.x, gB, sB, typename PartB::GmemToSmemCopy{});
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
    // 先做 SM100 UMMA/TMEM 蓝图确认。这个确认不把 pipeline/TMEM allocator 写进
    // AutoPartitioner，只验证 RoleC 的 fragment 类型确实是 TMEM accumulator。
    {
        using Element100 = cutlass::half_t;
        using StrideA100 = cute::Stride<cute::_1, int64_t>;
        using StrideB100 = cute::Stride<int64_t, cute::_1>;
        using StrideC100 = cute::Stride<cute::_1, int64_t>;
        using Tile100    = cute::Shape<cute::_64, cute::_128, cute::_64>;
        using PartA100   = typename autopartition::AutoPartitioner<cutlass::arch::Sm100,
                                                                   cutlass::arch::OpClassTensorOp,
                                                                   Element100,
                                                                   StrideA100,
                                                                   Tile100,
                                                                   128>::RoleA;
        using PartB100   = typename autopartition::AutoPartitioner<cutlass::arch::Sm100,
                                                                   cutlass::arch::OpClassTensorOp,
                                                                   Element100,
                                                                   StrideB100,
                                                                   Tile100,
                                                                   128>::RoleB;
        using PartC100   = typename autopartition::AutoPartitioner<cutlass::arch::Sm100,
                                                                   cutlass::arch::OpClassTensorOp,
                                                                   Element100,
                                                                   StrideC100,
                                                                   Tile100,
                                                                   128>::RoleC;
        using Mma100     = typename PartC100::template TiledMmaFor<PartA100::Major, PartB100::Major>;
        static_assert(cute::is_base_of<cute::UMMA::tmem_frg_base, typename Mma100::FrgTypeC>::value,
                      "SM100 TensorOp accumulator must be backed by Tensor Memory.");
    }

    constexpr int M           = 64;
    constexpr int N           = 32;
    constexpr int K           = 64;
    constexpr int ThreadCount = 256;

    using InputElement  = cutlass::float_e4m3_t;
    using OutputElement = float;
    using StrideA       = cute::Stride<int64_t, cute::_1>;
    using StrideB       = cute::Stride<int64_t, cute::_1>;
    using StrideC       = cute::Stride<int64_t, cute::_1>;
    using TileShape     = cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>;
    using ArchTag       = cutlass::arch::Sm120;
    using OpClass       = cutlass::arch::OpClassTensorOp;

    using PartA =
        typename autopartition::AutoPartitioner<ArchTag, OpClass, InputElement, StrideA, TileShape, ThreadCount>::RoleA;
    using PartB =
        typename autopartition::AutoPartitioner<ArchTag, OpClass, InputElement, StrideB, TileShape, ThreadCount>::RoleB;
    using PartC =
        typename autopartition::AutoPartitioner<ArchTag, OpClass, InputElement, StrideC, TileShape, ThreadCount>::RoleC;

    static_assert(std::is_same<typename PartC::Accumulator, OutputElement>::value,
                  "SM120 FP8 TensorOp example stores FP32 accumulators.");
    static_assert(cute::cosize_v<typename PartA::SmemLayout> > 0, "PartA SM120 smem layout must be valid.");
    static_assert(cute::cosize_v<typename PartB::SmemLayout> > 0, "PartB SM120 smem layout must be valid.");

    std::vector<InputElement>  hA(M * K);
    std::vector<InputElement>  hB(N * K);
    std::vector<OutputElement> hRef(M * N);
    std::vector<OutputElement> hAuto(M * N);
    std::vector<OutputElement> hBaseline(M * N);
    autopartition::examples::fill_pattern(hA);
    autopartition::examples::fill_pattern(hB);
    autopartition::examples::reference_gemm(M, N, K, hA.data(), K, 1, hB.data(), K, 1, hRef.data(), N, 1);

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

    sm120_tensorop_autopartition_kernel<PartA, PartB, PartC, InputElement, OutputElement>
        <<<dim3(1), dim3(ThreadCount)>>>(
            dA, make_stride(K, Int<1>{}), dB, make_stride(K, Int<1>{}), dC, make_stride(N, Int<1>{}));
    cudaError_t err = cudaDeviceSynchronize();
    if (!autopartition::examples::check_cuda(err, "sm120_tensorop_autopartition_kernel")) {
        return 1;
    }

    autopartition::examples::conventional_gemm_kernel<InputElement, InputElement, OutputElement>
        <<<dim3((M + 15) / 16, (N + 15) / 16), dim3(16, 16)>>>(dA, K, 1, dB, K, 1, dBaseline, N, 1, M, N, K);
    err = cudaDeviceSynchronize();
    if (!autopartition::examples::check_cuda(err, "sm120_tensorop_conventional_kernel")) {
        return 1;
    }

    cudaMemcpy(hAuto.data(), dC, M * N * sizeof(OutputElement), cudaMemcpyDeviceToHost);
    cudaMemcpy(hBaseline.data(), dBaseline, M * N * sizeof(OutputElement), cudaMemcpyDeviceToHost);

    float max_error = std::max(autopartition::examples::max_abs_diff(hAuto, hRef),
                               autopartition::examples::max_abs_diff(hBaseline, hRef));

    float auto_ms = autopartition::examples::time_launch_ms(
        [&]() {
            sm120_tensorop_autopartition_kernel<PartA, PartB, PartC, InputElement, OutputElement>
                <<<dim3(1), dim3(ThreadCount)>>>(
                    dA, make_stride(K, Int<1>{}), dB, make_stride(K, Int<1>{}), dC, make_stride(N, Int<1>{}));
        },
        50);
    float baseline_ms = autopartition::examples::time_launch_ms(
        [&]() {
            autopartition::examples::conventional_gemm_kernel<InputElement, InputElement, OutputElement>
                <<<dim3((M + 15) / 16, (N + 15) / 16), dim3(16, 16)>>>(dA, K, 1, dB, K, 1, dBaseline, N, 1, M, N, K);
        },
        50);

    autopartition::examples::print_result(
        "sm100_tensorop_tmem_example: SM100 TMEM blueprint + SM120 FP8 AutoPartitioner GEMM",
        max_error,
        auto_ms,
        baseline_ms);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dBaseline);
    return max_error < 1.0e-1f ? 0 : 1;
}
