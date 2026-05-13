#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <iostream>

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

    typename PartA::GmemToSmemCopy g2s_A;
    typename PartB::GmemToSmemCopy g2s_B;
    typename PartC::TiledMma       mma;
    typename PartC::RegToSmemCopy  r2s_C;
    typename PartC::SmemToGmemCopy s2g_C;

    auto thr_g2s_A = g2s_A.get_slice(threadIdx.x);
    auto thr_g2s_B = g2s_B.get_slice(threadIdx.x);
    auto thr_mma   = mma.get_thread_slice(threadIdx.x);
    auto thr_s2g_C = s2g_C.get_slice(threadIdx.x);

    cute::copy(g2s_A, thr_g2s_A.partition_S(gA), thr_g2s_A.partition_D(sA));
    cute::copy(g2s_B, thr_g2s_B.partition_S(gB), thr_g2s_B.partition_D(sB));
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();

    Tensor tCrA = thr_mma.partition_fragment_A(sA);
    Tensor tCrB = thr_mma.partition_fragment_B(sB);
    Tensor tCrC = thr_mma.partition_fragment_C(sC);
    clear(tCrC);

#if defined(CUTE_ARCH_FFMA2_SM100_ENABLED)
    cute::copy(typename PartA::SmemToRegCopy{}, thr_mma.partition_A(sA), tCrA);
    cute::copy(typename PartB::SmemToRegCopy{}, thr_mma.partition_B(sB), tCrB);
    cute::gemm(mma, tCrA, tCrB, tCrC);
#else
    // sm_120 当前不会定义 CUTE_ARCH_FFMA2_SM100_ENABLED。此时示例仍然验证
    // AutoPartitioner 生成的 G2S/S2G copy 和 shared layout，但不执行 SM100
    // f32x2 SIMT MMA，避免触发 CuTe 的架构保护断言。
    (void)tCrA;
    (void)tCrB;
#endif
    __syncthreads();

    cute::copy(r2s_C, tCrC, thr_mma.partition_C(sC));
    __syncthreads();
    cute::copy(s2g_C, thr_s2g_C.partition_S(sC), thr_s2g_C.partition_D(gC));
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

    // sm_120 构建时也可以使用 ArchTag=Sm120；该偏特化复用 SM100 Blackwell
    // SIMT policy，便于 RTX 50 系列直接编译验证。
    using ArchTag = cutlass::arch::Sm120;
    using OpClass = cutlass::arch::OpClassSimt;

    using PartA =
        typename autopartition::AutoPartitioner<ArchTag, OpClass, Element, StrideA, TileShape, ThreadCount>::RoleA;
    using PartB =
        typename autopartition::AutoPartitioner<ArchTag, OpClass, Element, StrideB, TileShape, ThreadCount>::RoleB;
    using PartC =
        typename autopartition::AutoPartitioner<ArchTag, OpClass, Element, StrideC, TileShape, ThreadCount>::RoleC;

    Element *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaMalloc(&dA, M * K * sizeof(Element));
    cudaMalloc(&dB, N * K * sizeof(Element));
    cudaMalloc(&dC, M * N * sizeof(Element));
    cudaMemset(dA, 0, M * K * sizeof(Element));
    cudaMemset(dB, 0, N * K * sizeof(Element));
    cudaMemset(dC, 0, M * N * sizeof(Element));

    sm100_simt_autopartition_kernel<PartA, PartB, PartC, Element><<<dim3(1), dim3(ThreadCount)>>>(
        dA, make_stride(Int<1>{}, M), dB, make_stride(K, Int<1>{}), dC, make_stride(Int<1>{}, M));

    cudaError_t err = cudaDeviceSynchronize();
    std::cout << "sm100_simt_example(sm_120 build): " << cudaGetErrorString(err) << "\n";

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return err == cudaSuccess ? 0 : 1;
}
