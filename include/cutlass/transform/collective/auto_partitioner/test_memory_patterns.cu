#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include <cute/tensor.hpp>

// 引入你已经修复好 16-Bytes Padding 的 auto_partitioner.hpp
#include "auto_partitioner_builder.hpp"

using namespace cute;

// ==============================================================================
// 专为 Nsight Compute 打造的极简、无流水线 Kernel
// ==============================================================================
template <typename PartA, typename PartB, typename PartC, typename Element, typename StrideA, typename StrideB, typename StrideC>
__global__ void memory_profiling_kernel(
  Element const* ptr_A, StrideA stride_A, Element const* ptr_B, StrideB stride_B, Element* ptr_C, StrideC stride_C)
{
  // 提取静态 Tile 维度 (假设矩阵大小正好等于 Tile 大小，仅运行 1 个 Block)
  using bM = decltype(size<0>(typename PartA::SmemLayout{}));
  using bN = decltype(size<0>(typename PartB::SmemLayout{}));
  using bK = decltype(size<1>(typename PartA::SmemLayout{}));

  Tensor gA = make_tensor(make_gmem_ptr(ptr_A), make_shape(bM{}, bK{}), stride_A);
  Tensor gB = make_tensor(make_gmem_ptr(ptr_B), make_shape(bN{}, bK{}), stride_B);
  Tensor gC = make_tensor(make_gmem_ptr(ptr_C), make_shape(bM{}, bN{}), stride_C);

  // 运行时 Smem 分配 (无流水线，最朴素的分配)
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

  // 实例化算子
  typename PartA::GmemToSmemCopy copy_A;
  typename PartB::GmemToSmemCopy copy_B;
  typename PartC::TiledMma mma;
  typename PartC::SmemToGmemCopy copy_C;

  auto thr_copy_A = copy_A.get_slice(threadIdx.x);
  auto thr_copy_B = copy_B.get_slice(threadIdx.x);
  auto thr_mma    = mma.get_thread_slice(threadIdx.x);
  auto thr_copy_C = copy_C.get_slice(threadIdx.x);

  Tensor tCrC = thr_mma.partition_fragment_C(sC);
  clear(tCrC);

  // ==========================================================================
  // Nsight Compute 观测点 1：Gmem 到 Smem 的 cp.async 向量化加载
  // 预期汇编：cp.async.cg.shared.global.16 (如果 16Bytes 对齐成功)
  // ==========================================================================
  Tensor tAgA = thr_copy_A.partition_S(gA);
  Tensor tAsA = thr_copy_A.partition_D(sA);
  cute::copy(copy_A, tAgA, tAsA);

  Tensor tBgB = thr_copy_B.partition_S(gB);
  Tensor tBsB = thr_copy_B.partition_D(sB);
  cute::copy(copy_B, tBgB, tBsB);

  cute::cp_async_fence();
  cute::cp_async_wait<0>();
  __syncthreads();

  // ==========================================================================
  // Nsight Compute 观测点 2：Smem 到 Register 的无冲突读取
  // 预期汇编：LDS.128 (ld.shared.v4，因为 SIMT 的 TiledCopy 被优化为了 DefaultCopy)
  // 预期 Bank Conflict：0
  // ==========================================================================
  Tensor tCrA = thr_mma.partition_fragment_A(sA);
  cute::copy(thr_mma.partition_A(sA), tCrA);

  Tensor tCrB = thr_mma.partition_fragment_B(sB);
  cute::copy(thr_mma.partition_B(sB), tCrB);

  // ==========================================================================
  // Nsight Compute 观测点 3：计算单元 FFMA 狂飙
  // 预期汇编：大量 FFMA 指令密布
  // ==========================================================================
  cute::gemm(mma, tCrA, tCrB, tCrC);
  __syncthreads();

  // ==========================================================================
  // Nsight Compute 观测点 4：Epilogue 写回 Smem 然后写出到 Gmem
  // 预期汇编 Smem->Gmem: STG.E.128 (st.global.v4)
  // 预期 Bank Conflict：0
  // ==========================================================================
  cute::copy(tCrC, thr_mma.partition_C(sC));
  __syncthreads();

  Tensor tCsC = thr_copy_C.partition_S(sC);
  Tensor tCgC = thr_copy_C.partition_D(gC);
  cute::copy(copy_C, tCsC, tCgC);
}

// ==============================================================================
// Host Launch
// ==============================================================================
int main()
{
  // 极简设置：正好等于一个 Tile，只产生一个 Block！
  constexpr int M = 64;
  constexpr int N = 64;
  constexpr int K = 16;

  using Element = float;

  using StrideA = cute::Stride<cute::_1, int64_t>; // Col-Major
  using StrideB = cute::Stride<int64_t, cute::_1>; // Row-Major
  using StrideC = cute::Stride<cute::_1, int64_t>; // Col-Major

  using TileShape           = cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>;
  constexpr int ThreadCount = 256;

  using ArchTag = cutlass::arch::Sm80;
  using OpClass = cutlass::arch::OpClassSimt;

  using PartA = autopartition::AutoPartitioner<ArchTag, OpClass, Element, StrideA, TileShape, ThreadCount>::RoleA;
  using PartB = autopartition::AutoPartitioner<ArchTag, OpClass, Element, StrideB, TileShape, ThreadCount>::RoleB;
  using PartC = autopartition::AutoPartitioner<ArchTag, OpClass, Element, StrideC, TileShape, ThreadCount>::RoleC;

  Element *dA, *dB, *dC;
  cudaMalloc(&dA, M * K * sizeof(Element));
  cudaMalloc(&dB, N * K * sizeof(Element));
  cudaMalloc(&dC, M * N * sizeof(Element));

  // 为纯净测试，忽略 Host 验证逻辑，直接塞入伪数据
  cudaMemset(dA, 0, M * K * sizeof(Element));
  cudaMemset(dB, 0, N * K * sizeof(Element));
  cudaMemset(dC, 0, M * N * sizeof(Element));

  dim3 block(ThreadCount);
  dim3 grid(1, 1); // 绝对控制：只有 1 个 Block！

  std::cout << "Launching Single-Block Profiling Kernel..." << std::endl;

  memory_profiling_kernel<PartA, PartB, PartC, Element>
    <<<grid, block>>>(dA, make_stride(Int<1>{}, M), dB, make_stride(K, Int<1>{}), dC, make_stride(Int<1>{}, M));

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess)
  {
    std::cerr << "Kernel Error: " << cudaGetErrorString(err) << std::endl;
  }
  else
  {
    std::cout << "Kernel execution completed cleanly. Ready for Nsight Compute!" << std::endl;
  }

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  return 0;
}
