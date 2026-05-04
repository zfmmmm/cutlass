#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "auto_partitioner_builder.hpp"
#include <cublas_v2.h>
#include <cute/tensor.hpp>
using namespace cute;

template <typename PartA, typename PartB, typename PartC, typename Element, typename StrideA, typename StrideB, typename StrideC>
__global__ void simt_gemm_kernel(
  Element const* ptr_A,
  StrideA stride_A,
  Element const* ptr_B,
  StrideB stride_B,
  Element* ptr_C,
  StrideC stride_C,
  int M,
  int N,
  int K)
{
  Tensor gA = make_tensor(make_gmem_ptr(ptr_A), make_shape(M, K), stride_A);
  Tensor gB = make_tensor(make_gmem_ptr(ptr_B), make_shape(N, K), stride_B);
  Tensor gC = make_tensor(make_gmem_ptr(ptr_C), make_shape(M, N), stride_C);

  int cta_m = blockIdx.x, cta_n = blockIdx.y;
  using bM = decltype(size<0>(typename PartA::SmemLayout{}));
  using bN = decltype(size<0>(typename PartB::SmemLayout{}));
  using bK = decltype(size<1>(typename PartA::SmemLayout{}));

  Tensor gA_blk = local_tile(gA, make_shape(bM{}, bK{}), make_coord(cta_m, _));
  Tensor gB_blk = local_tile(gB, make_shape(bN{}, bK{}), make_coord(cta_n, _));
  Tensor gC_blk = local_tile(gC, make_shape(bM{}, bN{}), make_coord(cta_m, cta_n));

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

  typename PartA::GmemToSmemCopy copy_A;
  typename PartB::GmemToSmemCopy copy_B;
  typename PartC::TiledMma mma;
  auto thr_copy_A = copy_A.get_slice(threadIdx.x);
  auto thr_copy_B = copy_B.get_slice(threadIdx.x);
  auto thr_mma    = mma.get_thread_slice(threadIdx.x);

  Tensor tCrC = thr_mma.partition_fragment_C(sC);
  clear(tCrC);

  int num_k_blocks = size<2>(gA_blk);
  for (int k = 0; k < num_k_blocks; ++k)
  {
    cute::copy(copy_A, thr_copy_A.partition_S(gA_blk(_, _, k)), thr_copy_A.partition_D(sA));
    cute::copy(copy_B, thr_copy_B.partition_S(gB_blk(_, _, k)), thr_copy_B.partition_D(sB));
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();

    Tensor tCrA = thr_mma.partition_fragment_A(sA);
    cute::copy(thr_mma.partition_A(sA), tCrA);
    Tensor tCrB = thr_mma.partition_fragment_B(sB);
    cute::copy(thr_mma.partition_B(sB), tCrB);
    cute::gemm(mma, tCrA, tCrB, tCrC);
    __syncthreads();
  }

  cute::copy(tCrC, thr_mma.partition_C(sC));
  __syncthreads();
  typename PartC::SmemToGmemCopy copy_C;
  auto thr_copy_C = copy_C.get_slice(threadIdx.x);
  cute::copy(copy_C, thr_copy_C.partition_S(sC), thr_copy_C.partition_D(gC_blk));
}

// ==============================================================================
// 3. 可视化与 Benchmark 引擎
// ==============================================================================
void draw_ascii_bar(std::string name, double tflops, double max_tflops)
{
  int bar_width = 40;
  int filled    = static_cast<int>((tflops / max_tflops) * bar_width);
  std::cout << std::setw(15) << name << " | ";
  for (int i = 0; i < bar_width; ++i)
  {
    if (i < filled)
    {
      std::cout << "█";
    }
    else
    {
      std::cout << " ";
    }
  }
  std::cout << " | " << std::fixed << std::setprecision(2) << tflops << " TFLOPS\n";
}

int main()
{
  std::cout << "========================================================\n";
  std::cout << "  AutoPartitioner vs Official cuBLAS Performance Test\n";
  std::cout << "========================================================\n\n";

  cublasHandle_t handle;
  cublasCreate(&handle);

  // 测试矩阵的多种尺寸
  std::vector<int> sizes = {256, 512, 1024, 2048};

  for (int SIZE : sizes)
  {
    int M = SIZE, N = SIZE, K = SIZE;
    using Element = float;

    // A: Col-Major, B: Row-Major, C: Col-Major
    using StrideA = cute::Stride<cute::_1, int64_t>;
    using StrideB = cute::Stride<int64_t, cute::_1>;
    using StrideC = cute::Stride<cute::_1, int64_t>;

    using TileShape           = cute::Shape<cute::_64, cute::_64, cute::_16>;
    constexpr int ThreadCount = 256;

    using ArchTag = cutlass::arch::Sm80;
    using OpClass = cutlass::arch::OpClassSimt;

    using PartA = autopartition::AutoPartitioner<ArchTag, OpClass, Element, StrideA, TileShape, ThreadCount>::RoleA;
    using PartB = autopartition::AutoPartitioner<ArchTag, OpClass, Element, StrideB, TileShape, ThreadCount>::RoleB;
    using PartC = autopartition::AutoPartitioner<ArchTag, OpClass, Element, StrideC, TileShape, ThreadCount>::RoleC;

    Element *dA, *dB, *dC_auto, *dC_cublas;
    cudaMalloc(&dA, M * K * sizeof(Element));
    cudaMalloc(&dB, N * K * sizeof(Element));
    cudaMalloc(&dC_auto, M * N * sizeof(Element));
    cudaMalloc(&dC_cublas, M * N * sizeof(Element));

    float alpha = 1.0f, beta = 0.0f;
    dim3 block(ThreadCount);
    dim3 grid(M / cute::size<0>(TileShape{}), N / cute::size<1>(TileShape{}));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    const int ITERS = 50; // 迭代次数取平均

    // ---------------------------------------------------------
    // 测速 1: AutoPartitioner
    // ---------------------------------------------------------
    for (int i = 0; i < 5; ++i)
    { // Warmup
      simt_gemm_kernel<PartA, PartB, PartC, Element><<<grid, block>>>(
        dA, make_stride(Int<1>{}, M), dB, make_stride(N, Int<1>{}), dC_auto, make_stride(Int<1>{}, M), M, N, K);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < ITERS; ++i)
    {
      simt_gemm_kernel<PartA, PartB, PartC, Element><<<grid, block>>>(
        dA, make_stride(Int<1>{}, M), dB, make_stride(N, Int<1>{}), dC_auto, make_stride(Int<1>{}, M), M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_auto = 0;
    cudaEventElapsedTime(&ms_auto, start, stop);
    double tflops_auto = (2.0 * M * N * K * ITERS) / (ms_auto * 1e9);

    // ---------------------------------------------------------
    // 测速 2: cuBLAS 官方基准
    // ---------------------------------------------------------
    for (int i = 0; i < 5; ++i)
    { // Warmup
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, M, N, K, &alpha, dA, M, dB, N, &beta, dC_cublas, M);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < ITERS; ++i)
    {
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, M, N, K, &alpha, dA, M, dB, N, &beta, dC_cublas, M);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_cublas = 0;
    cudaEventElapsedTime(&ms_cublas, start, stop);
    double tflops_cublas = (2.0 * M * N * K * ITERS) / (ms_cublas * 1e9);

    // ---------------------------------------------------------
    // 绘制终端图表
    // ---------------------------------------------------------
    std::cout << "Matrix Size: M=N=K=" << SIZE << "\n";
    double max_t = std::max(tflops_auto, tflops_cublas) * 1.1; // 留 10% 裕量
    draw_ascii_bar("AutoPartition", tflops_auto, max_t);
    draw_ascii_bar("cuBLAS (Ref)", tflops_cublas, max_t);
    std::cout
      << "Speedup vs Official: " << std::fixed << std::setprecision(2) << (tflops_auto / tflops_cublas) << "x\n";
    std::cout << "--------------------------------------------------------\n";

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC_auto);
    cudaFree(dC_cublas);
  }

  cublasDestroy(handle);
  return 0;
}
