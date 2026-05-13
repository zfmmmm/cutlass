#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <iostream>

#include "auto_partitioner_builder.hpp"

using namespace cute;

// SM80 + OpClassSimt 的完整单 CTA 示例。
// 这个 kernel 故意不做 pipeline stage 推导，也不隐藏 shared memory 分配：
// AutoPartitioner 只提供 PartA/PartB/PartC 的 layout/copy/MMA 图纸，
// 资源管理和 tile 坐标由下游 kernel 显式完成。
template <typename PartA,
          typename PartB,
          typename PartC,
          typename Element,
          typename StrideA,
          typename StrideB,
          typename StrideC>
__global__ void sm80_simt_autopartition_kernel(Element const *ptr_A,
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

    // Gmem -> Smem: SM80 policy 会选择 cp.async zfill 和合适的向量宽度。
    cute::copy(g2s_A, thr_g2s_A.partition_S(gA), thr_g2s_A.partition_D(sA));
    cute::copy(g2s_B, thr_g2s_B.partition_S(gB), thr_g2s_B.partition_D(sB));
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();

    // Smem -> Register -> SIMT FMA。这里直接使用 TiledMma 的 partition，
    // layout/padding 由 PartA/PartB 提供。
    Tensor tCrA = thr_mma.partition_fragment_A(sA);
    Tensor tCrB = thr_mma.partition_fragment_B(sB);
    Tensor tCrC = thr_mma.partition_fragment_C(sC);
    clear(tCrC);

    cute::copy(thr_mma.partition_A(sA), tCrA);
    cute::copy(thr_mma.partition_B(sB), tCrB);
    cute::gemm(mma, tCrA, tCrB, tCrC);
    __syncthreads();

    // Register -> Smem -> Gmem。显式使用 RoleC 暴露的 copy 类型，便于 Nsight
    // 分别观察 shared store 和 global store。
    cute::copy(r2s_C, tCrC, thr_mma.partition_C(sC));
    __syncthreads();
    cute::copy(s2g_C, thr_s2g_C.partition_S(sC), thr_s2g_C.partition_D(gC));
}

int main()
{
    constexpr int M           = 64;
    constexpr int N           = 64;
    constexpr int K           = 16;
    constexpr int ThreadCount = 256;

    using Element   = float;
    using StrideA   = cute::Stride<cute::_1, int64_t>;
    using StrideB   = cute::Stride<int64_t, cute::_1>;
    using StrideC   = cute::Stride<cute::_1, int64_t>;
    using TileShape = cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>;
    using ArchTag   = cutlass::arch::Sm80;
    using OpClass   = cutlass::arch::OpClassSimt;

    using PartA =
        typename autopartition::AutoPartitioner<ArchTag, OpClass, Element, StrideA, TileShape, ThreadCount>::RoleA;
    using PartB =
        typename autopartition::AutoPartitioner<ArchTag, OpClass, Element, StrideB, TileShape, ThreadCount>::RoleB;
    using PartC =
        typename autopartition::AutoPartitioner<ArchTag, OpClass, Element, StrideC, TileShape, ThreadCount>::RoleC;

    static_assert(cute::cosize_v<typename PartA::SmemLayout> > 0, "PartA shared layout must be valid.");
    static_assert(cute::cosize_v<typename PartB::SmemLayout> > 0, "PartB shared layout must be valid.");
    static_assert(cute::cosize_v<typename PartC::SmemLayout> > 0, "PartC shared layout must be valid.");

    Element *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaMalloc(&dA, M * K * sizeof(Element));
    cudaMalloc(&dB, N * K * sizeof(Element));
    cudaMalloc(&dC, M * N * sizeof(Element));
    cudaMemset(dA, 0, M * K * sizeof(Element));
    cudaMemset(dB, 0, N * K * sizeof(Element));
    cudaMemset(dC, 0, M * N * sizeof(Element));

    sm80_simt_autopartition_kernel<PartA, PartB, PartC, Element><<<dim3(1), dim3(ThreadCount)>>>(
        dA, make_stride(Int<1>{}, M), dB, make_stride(K, Int<1>{}), dC, make_stride(Int<1>{}, M));

    cudaError_t err = cudaDeviceSynchronize();
    std::cout << "sm80_simt_example: " << cudaGetErrorString(err) << "\n";

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return err == cudaSuccess ? 0 : 1;
}
