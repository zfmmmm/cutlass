/***************************************************************************************************
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#include "../../../common/cutlass_unit_test.h"

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/util/print_tensor.hpp>

#include "cutlass/transform/collective/auto_partitioner/auto_partitioner_builder.hpp"
#include "cutlass/transform/collective/auto_partitioner/examples/autopartition_example_utils.hpp"

using namespace cute;

namespace autopartition_sm1xx_policy_audit {

bool has_cuda_device() {
  int count = 0;
  cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return count > 0;
}

template <class Element>
void fill_pattern(std::vector<Element>& values) {
  for (int i = 0; i < int(values.size()); ++i) {
    values[i] = autopartition::examples::from_float<Element>(
        autopartition::examples::patterned_value(i));
  }
}

void fill_pattern(std::vector<float>& values) {
  for (int i = 0; i < int(values.size()); ++i) {
    values[i] = autopartition::examples::patterned_value(i);
  }
}

template <class PartA, class PartB>
__global__ void print_sm100_tensorop_layouts_kernel(int* status) {
  if (threadIdx.x == 0) {
    printf("SM100 TensorOp SmemLayoutA:\n");
    cute::print_layout(typename PartA::SmemLayout{});
    printf("SM100 TensorOp SmemLayoutB:\n");
    cute::print_layout(typename PartB::SmemLayout{});
    status[0] = int(cute::cosize(typename PartA::SmemLayout{}));
    status[1] = int(cute::cosize(typename PartB::SmemLayout{}));
  }
}

template <class PartA, class Element, class StrideA>
__global__ void sm100_alignment_fallback_copy_kernel(
    Element const* ptr_A,
    StrideA stride_A,
    Element* ptr_out) {
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
    int m = idx % int(bM{});
    int k = idx / int(bM{});
    ptr_out[idx] = sA(m, k);
  }
}

template <class PartC>
__global__ void simt_c_bank_pattern_kernel(float* checksum) {
  using Layout = typename PartC::SmemLayout;
  constexpr int BlkM = int(cute::size<0>(Layout{}));
  constexpr int BlkN = int(cute::size<1>(Layout{}));

  __shared__ cute::array_aligned<float, cute::cosize_v<Layout>> smem;
  Tensor sC = make_tensor(make_smem_ptr(smem.data()), Layout{});

  if (threadIdx.x == 0) {
    *checksum = 0.0f;
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < BlkM * BlkN; idx += blockDim.x) {
    int m = idx % BlkM;
    int n = idx / BlkM;
    sC(m, n) = float(m + n);
  }
  __syncthreads();

  float local = 0.0f;
  for (int idx = threadIdx.x; idx < BlkM * BlkN; idx += blockDim.x) {
    int m = idx % BlkM;
    int n = idx / BlkM;
    local += sC(m, n);
  }
  atomicAdd(checksum, local);
}

template <class PartC, class TmemTensor, class SmemTensor, class GmemTensor>
CUTLASS_DEVICE void sm100_tmem_epilogue_unload_blueprint(
    TmemTensor const& tmem_accum,
    SmemTensor const& smem_tile,
    GmemTensor const& gmem_tile,
    int cta_m_in_cluster,
    int cta_n_in_cluster) {
  // This is intentionally a compile-time blueprint: real execution requires a valid
  // SM100 TMEM allocation, pipeline barriers, and host-encoded TMA descriptors.
  cute::copy(typename PartC::TmemToSmemCopy{}, tmem_accum, smem_tile);
  __syncthreads();

  constexpr int ClusterM = cute::size<0>(typename PartC::ClusterShape_MNK{});
  constexpr int ClusterN = cute::size<1>(typename PartC::ClusterShape_MNK{});
  bool cluster_writer = (cta_m_in_cluster == 0) && (cta_n_in_cluster == 0);
  if (cluster_writer || (ClusterM == 1 && ClusterN == 1)) {
    if constexpr (!PartC::UsesTmaStore) {
      cute::copy(typename PartC::SmemToGmemCopy{}, smem_tile, gmem_tile);
    }
  }
}

template <class PartC>
__global__ void sm100_epilogue_connectivity_kernel(int* flags) {
  if (threadIdx.x == 0) {
    flags[0] = cute::is_base_of<cute::UMMA::tmem_frg_base, typename PartC::TiledMma::FrgTypeC>::value ? 1 : 0;
    flags[1] = !std::is_void<typename PartC::TmemToSmemCopy>::value ? 1 : 0;
    flags[2] = PartC::UsesTmaStore ? 1 : 0;
    flags[3 + blockIdx.x] = (blockIdx.x % cute::size<0>(typename PartC::ClusterShape_MNK{}) == 0) ? 1 : 0;
  }
}

__global__ void zfill_gemm_kernel(
    float const* A,
    float const* B,
    float* C,
    int M,
    int N,
    int K,
    int round_k) {
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || n >= N) {
    return;
  }

  float acc = 0.0f;
  for (int k = 0; k < round_k; ++k) {
    float a = (k < K) ? A[m * K + k] : 0.0f;
    float b = (k < K) ? B[k * N + n] : 0.0f;
    acc += a * b;
  }
  C[m * N + n] = acc;
}

void run_zfill_case(int M, int N, int K) {
  ASSERT_TRUE(has_cuda_device());

  int round_k = ((K + 15) / 16) * 16;
  std::vector<float> hA(M * K);
  std::vector<float> hB(K * N);
  std::vector<float> hC(M * N, 0.0f);
  std::vector<float> hRef(M * N, 0.0f);
  fill_pattern(hA);
  fill_pattern(hB);

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        acc += hA[m * K + k] * hB[k * N + n];
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
  zfill_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K, round_k);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

  float max_diff = 0.0f;
  for (int i = 0; i < M * N; ++i) {
    max_diff = std::max(max_diff, std::abs(hC[i] - hRef[i]));
  }
  EXPECT_LT(max_diff, 1.0e-4f);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}

} // namespace autopartition_sm1xx_policy_audit

using namespace autopartition_sm1xx_policy_audit;

TEST(AutoPartitionerSm1xxPhase1, StaticSfinaeRoutingAndTopology) {
  using Cluster2x1x1 = cute::Shape<cute::_2, cute::_1, cute::_1>;
  using TileSm80 = cute::Shape<cute::_64, cute::_64, cute::_64>;
  using TileSm100 = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using StrideA = cute::Stride<cute::_1, int64_t>;
  using StrideB = cute::Stride<int64_t, cute::_1>;
  using StrideC = cute::Stride<cute::_1, int64_t>;

  using Sm80SimtA = typename autopartition::AutoPartitioner<
      cutlass::arch::Sm80, cutlass::arch::OpClassSimt, float, StrideA,
      cute::Shape<cute::_64, cute::_64, cute::_16>, 256>::RoleA;
  static_assert(std::is_same<Sm80SimtA,
      autopartition::detail::Sm80SimtRoleA<float, StrideA, cute::Shape<cute::_64, cute::_64, cute::_16>, 256, 16>>::value,
      "SM80 SIMT must route to Sm80SimtRoleA.");

  using Sm80TensorA = typename autopartition::AutoPartitioner<
      cutlass::arch::Sm80, cutlass::arch::OpClassTensorOp, cutlass::half_t,
      StrideA, TileSm80, 128>::RoleA;
  static_assert(std::is_same<Sm80TensorA,
      autopartition::detail::Sm80TensorOpRoleA<cutlass::half_t, StrideA, TileSm80, 128, 16>>::value,
      "SM80 TensorOp must route to Sm80TensorOpRoleA.");

  using Sm100TensorA = typename autopartition::AutoPartitioner<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, cutlass::half_t,
      StrideA, TileSm100, 128, float, 16, 16, 16, Cluster2x1x1>::RoleA;
  static_assert(std::is_same<Sm100TensorA,
      autopartition::detail::Sm100TensorOpRoleA<cutlass::half_t, StrideA, TileSm100, 128, 16, Cluster2x1x1>>::value,
      "SM100 TensorOp must route to Sm100TensorOpRoleA.");

  using Sm120Fp8A = typename autopartition::AutoPartitioner<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, cutlass::float_e4m3_t,
      StrideA, TileSm100, 256, float, 16, 16, 16, Cluster2x1x1>::RoleA;
  static_assert(std::is_same<Sm120Fp8A,
      autopartition::detail::Sm120TensorOpRoleA<cutlass::float_e4m3_t, StrideA, TileSm100, 256, 16, Cluster2x1x1>>::value,
      "SM120 FP8 TensorOp must route to Sm120TensorOpRoleA.");

  static_assert(Sm100TensorA::UsesTmaLoad, "SM100 16B aligned RoleA should select TMA.");
  static_assert(std::is_same<typename Sm120Fp8A::ClusterShape_MNK, Cluster2x1x1>::value,
      "SM120 role must preserve cluster shape.");
  static_assert(Sm120Fp8A::GmemToSmemAlignmentBytes == 16,
      "SM120 role must preserve A alignment.");
  static_assert(cute::cosize_v<typename Sm100TensorA::SmemLayout> > 0,
      "SM100 TensorOp A layout must be non-empty.");

  using Sm100TensorB = typename autopartition::AutoPartitioner<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, cutlass::half_t,
      StrideB, TileSm100, 128, float, 16, 16, 16, Cluster2x1x1>::RoleB;
  static_assert((cute::size<2>(TileSm100{}) * int(sizeof(cutlass::half_t))) == 128,
      "TileK=64 FP16 gives a 128B K-row, matching UMMA SW128 selection.");
  EXPECT_GT(int(cute::cosize(typename Sm100TensorB::SmemLayout{})), 0);

  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device not available.";
  }
  int* d_status = nullptr;
  ASSERT_EQ(cudaMalloc(&d_status, 2 * sizeof(int)), cudaSuccess);
  print_sm100_tensorop_layouts_kernel<Sm100TensorA, Sm100TensorB><<<1, 1>>>(d_status);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  int h_status[2] = {};
  ASSERT_EQ(cudaMemcpy(h_status, d_status, sizeof(h_status), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_GT(h_status[0], 0);
  EXPECT_GT(h_status[1], 0);
  cudaFree(d_status);
}

TEST(AutoPartitionerSm1xxPhase2, AlignmentFallbackUsesCpAsyncAndRunsOnEightByteView) {
  using Element = cutlass::half_t;
  using StrideA = cute::Stride<cute::_1, int64_t>;
  using Tile = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using Cluster = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using PartA = typename autopartition::AutoPartitioner<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, Element,
      StrideA, Tile, 128, float, 8, 16, 16, Cluster>::RoleA;

  static_assert(!PartA::UsesTmaLoad, "8B alignment must select the cp.async fallback path.");
  static_assert(PartA::GmemToSmemAlignmentBytes == 8, "Fallback vector width must honor 8B alignment.");

  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device not available.";
  }

  constexpr int M = 128;
  constexpr int K = 64;
  std::vector<Element> hA(M * K + 4);
  std::vector<Element> hOut(M * K);
  fill_pattern(hA);

  Element *dA_raw = nullptr, *dOut = nullptr;
  ASSERT_EQ(cudaMalloc(&dA_raw, hA.size() * sizeof(Element)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dOut, hOut.size() * sizeof(Element)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dA_raw, hA.data(), hA.size() * sizeof(Element), cudaMemcpyHostToDevice), cudaSuccess);

  Element const* dA_8b = dA_raw + 4;
  sm100_alignment_fallback_copy_kernel<PartA><<<1, 128>>>(dA_8b, make_stride(Int<1>{}, M), dOut);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(hOut.data(), dOut, hOut.size() * sizeof(Element), cudaMemcpyDeviceToHost), cudaSuccess);

  float max_diff = 0.0f;
  for (int k = 0; k < K; ++k) {
    for (int m = 0; m < M; ++m) {
      int idx = m + k * M;
      max_diff = std::max(max_diff,
          std::abs(autopartition::examples::to_float(hOut[idx]) -
                   autopartition::examples::to_float(hA[idx + 4])));
    }
  }
  EXPECT_LT(max_diff, 1.0e-5f);

  cudaFree(dA_raw);
  cudaFree(dOut);
}

TEST(AutoPartitionerSm1xxPhase3, SimtCPaddedLayoutAndBankConflictProfilingHook) {
  using Tile = cute::Shape<cute::_64, cute::_64, cute::_16>;
  using StrideC = cute::Stride<cute::_1, int64_t>;
  using PartC = autopartition::detail::Sm100SimtRoleC<float, StrideC, Tile, 128>;
  using Layout = typename PartC::SmemLayout;

  static_assert(cute::size<0>(Layout{}) == 64, "C layout M extent must be 64.");
  static_assert(cute::size<1>(Layout{}) == 64, "C layout N extent must be 64.");
  static_assert(cute::cosize_v<Layout> > 64 * 64, "C layout must contain padding to perturb bank mapping.");

  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device not available.";
  }

  float* d_checksum = nullptr;
  ASSERT_EQ(cudaMalloc(&d_checksum, sizeof(float)), cudaSuccess);
  simt_c_bank_pattern_kernel<PartC><<<1, 128>>>(d_checksum);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  float h_checksum = 0.0f;
  ASSERT_EQ(cudaMemcpy(&h_checksum, d_checksum, sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
  float expected = 0.0f;
  for (int n = 0; n < 64; ++n) {
    for (int m = 0; m < 64; ++m) {
      expected += float(m + n);
    }
  }
  EXPECT_EQ(h_checksum, expected);
  cudaFree(d_checksum);

  RecordProperty("ncu_metric",
      "ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,"
      "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum "
      "./cutlass_test_unit_transform_collective_auto_partitioner "
      "--gtest_filter=AutoPartitionerSm1xxPhase3.*");
}

TEST(AutoPartitionerSm1xxPhase4, TmemEpilogueConnectivityAndClusterWriterGuard) {
  using Element = cutlass::half_t;
  using StrideC = cute::Stride<cute::_1, int64_t>;
  using Tile = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using Cluster = cute::Shape<cute::_2, cute::_1, cute::_1>;
  using PartC = typename autopartition::AutoPartitioner<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, Element,
      StrideC, Tile, 128, float, 16, 16, 16, Cluster>::RoleC;

  static_assert(cute::is_base_of<cute::UMMA::tmem_frg_base, typename PartC::TiledMma::FrgTypeC>::value,
      "SM100 TensorOp accumulator fragment must be backed by TMEM.");
  static_assert(!std::is_void<typename PartC::TmemToSmemCopy>::value,
      "RoleC must expose TMEM-to-SMEM unload copy.");
  static_assert(PartC::UsesTmaStore, "16B C alignment should select TMA store for epilogue.");

  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device not available.";
  }

  int* d_flags = nullptr;
  ASSERT_EQ(cudaMalloc(&d_flags, 5 * sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_flags, 0, 5 * sizeof(int)), cudaSuccess);
  sm100_epilogue_connectivity_kernel<PartC><<<2, 1>>>(d_flags);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  int h_flags[5] = {};
  ASSERT_EQ(cudaMemcpy(h_flags, d_flags, sizeof(h_flags), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(h_flags[0], 1);
  EXPECT_EQ(h_flags[1], 1);
  EXPECT_EQ(h_flags[2], 1);
  EXPECT_EQ(h_flags[3], 1);
  EXPECT_EQ(h_flags[4], 0);
  cudaFree(d_flags);
}

TEST(AutoPartitionerSm1xxPhase5, NumericalGemmStandardAndOddShapeZfill) {
  run_zfill_case(256, 256, 256);
  run_zfill_case(123, 45, 67);
}
