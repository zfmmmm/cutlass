// PyTorch CUDA extension backend using the repository AutoPartitioner roles.

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/algorithm/cooperative_gemm.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cutlass/transform/collective/auto_partitioner/auto_partitioner_builder.hpp>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cmath>
#include <stdexcept>
#include <type_traits>

using namespace cute;

namespace zfm_autopartition_runtime {

template <class PartA,
          class PartB,
          class PartC,
          class InputElement,
          class OutputElement,
          class StrideA,
          class StrideB,
          class StrideC,
          int ThreadCount>
__global__ __launch_bounds__(ThreadCount, 4) void fused_gemm_bias_scale_gelu_kernel(InputElement const *ptr_A,
                                                                                     StrideA             stride_A,
                                                                                     InputElement const *ptr_B,
                                                                                     StrideB             stride_B,
                                                                                     OutputElement      *ptr_C,
                                                                                     StrideC             stride_C,
                                                                                     int                 m,
                                                                                     int                 n,
                                                                                     int                 k)
{
    using bM = decltype(size<0>(typename PartA::SmemLayout{}));
    using bN = decltype(size<0>(typename PartB::SmemLayout{}));
    using bK = decltype(size<1>(typename PartA::SmemLayout{}));

    Tensor gA_full = make_tensor(make_gmem_ptr(ptr_A), make_shape(m, k), stride_A);
    Tensor gB_full = make_tensor(make_gmem_ptr(ptr_B), make_shape(n, k), stride_B);
    Tensor gC_full = make_tensor(make_gmem_ptr(ptr_C), make_shape(m, n), stride_C);

    Tensor gA = local_tile(gA_full, make_tile(bM{}, bK{}), make_coord(blockIdx.x, _));
    Tensor gB = local_tile(gB_full, make_tile(bN{}, bK{}), make_coord(blockIdx.y, _));
    Tensor gC = local_tile(gC_full, make_tile(bM{}, bN{}), make_coord(blockIdx.x, blockIdx.y));

    struct MainloopStorage
    {
        cute::array_aligned<InputElement, cute::cosize_v<typename PartA::SmemLayout>> smemA;
        cute::array_aligned<InputElement, cute::cosize_v<typename PartB::SmemLayout>> smemB;
    };
    struct EpilogueStorage
    {
        cute::array_aligned<typename PartC::EpilogueElement, cute::cosize_v<typename PartC::SmemLayout>> smemC;
    };
    union SharedStorage
    {
        MainloopStorage mainloop;
        EpilogueStorage epilogue;
    };
    __shared__ SharedStorage shared;

    Tensor sA = make_tensor(make_smem_ptr(shared.mainloop.smemA.data()), typename PartA::SmemLayout{});
    Tensor sB = make_tensor(make_smem_ptr(shared.mainloop.smemB.data()), typename PartB::SmemLayout{});
    Tensor sC = make_tensor(make_smem_ptr(shared.epilogue.smemC.data()), typename PartC::SmemLayout{});

    typename PartC::TiledMma mma;
    auto                     thr_mma = mma.get_thread_slice(threadIdx.x);
    Tensor                   tCgC    = thr_mma.partition_C(gC);
    Tensor                   tCrC    = thr_mma.make_fragment_C(tCgC);
    clear(tCrC);

    typename PartA::GlobalToSharedCopy tiled_copy_A;
    typename PartB::GlobalToSharedCopy tiled_copy_B;
    auto                               thr_copy_A = tiled_copy_A.get_thread_slice(threadIdx.x);
    auto                               thr_copy_B = tiled_copy_B.get_thread_slice(threadIdx.x);

    int k_tiles = size<2>(gA);
#pragma unroll 1
    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        Tensor gA_k = gA(_, _, k_tile);
        Tensor gB_k = gB(_, _, k_tile);

        Tensor tAgA = thr_copy_A.partition_S(gA_k);
        Tensor tAsA = thr_copy_A.partition_D(sA);
        Tensor tBgB = thr_copy_B.partition_S(gB_k);
        Tensor tBsB = thr_copy_B.partition_D(sB);
        copy(tiled_copy_A, tAgA, tAsA);
        copy(tiled_copy_B, tBgB, tBsB);
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        cooperative_gemm(threadIdx.x,
                         mma,
                         sA,
                         sB,
                         tCrC,
                         identity{},
                         identity{},
                         typename PartA::SmemToRegCopyOperation{},
                         typename PartB::SmemToRegCopyOperation{});
        __syncthreads();
    }

    auto r2s_tiled_copy_C =
        make_tiled_copy_C(Copy_Atom<typename PartC::RegToSmemCopyOperation, typename PartC::EpilogueElement>{},
                          thr_mma);
    auto   r2s_thr_copy_C = r2s_tiled_copy_C.get_thread_slice(threadIdx.x);
    Tensor tRS_rAcc       = r2s_thr_copy_C.retile_S(tCrC);
    Tensor tRS_sAcc       = r2s_thr_copy_C.partition_D(sC);
    copy(r2s_tiled_copy_C, tRS_rAcc, tRS_sAcc);
    __syncthreads();

    typename PartC::SharedToOutputRegisterCopy s2r_tiled_copy_C;
    auto   s2r_thr_copy_C = s2r_tiled_copy_C.get_thread_slice(threadIdx.x);
    auto   s2r_contract   = PartC::retile_smem_to_output(s2r_thr_copy_C, sC, gC);
    Tensor tSR_sAcc       = cute::get<0>(s2r_contract);
    Tensor tSR_rAcc       = make_tensor<typename PartC::EpilogueElement>(shape(tSR_sAcc));
    Tensor tSR_rD         = make_tensor<OutputElement>(shape(tSR_sAcc));

    copy(s2r_tiled_copy_C, tSR_sAcc, tSR_rAcc);

    cutlass::NumericConverter<OutputElement, typename PartC::EpilogueElement> convert;
    CUTE_UNROLL
    for (int i = 0; i < size(tSR_rAcc); ++i) {
        tSR_rD(i) = convert(tSR_rAcc(i));
    }

    typename PartC::OutputRegisterToGlobalCopy r2g_tiled_copy_C;
    auto r2g_thr_copy_C = r2g_tiled_copy_C.get_thread_slice(threadIdx.x);
    auto r2g_contract   = PartC::retile_register_to_output(r2g_thr_copy_C, tSR_rD, gC);
    copy(r2g_tiled_copy_C, cute::get<0>(r2g_contract), cute::get<1>(r2g_contract));
}

template <class BiasElement, class OutputElement>
__global__ void bias_scale_gelu_epilogue_kernel(float const *acc,
                                                BiasElement const *bias,
                                                float scale,
                                                OutputElement *out,
                                                int total,
                                                int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int   col = idx % n;
    float x   = (acc[idx] + static_cast<float>(bias[col])) * scale;
    float y   = 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
    cutlass::NumericConverter<OutputElement, float> convert;
    out[idx] = convert(y);
}

torch::Tensor fused_gemm_bias_scale_gelu(torch::Tensor A, torch::Tensor B, torch::Tensor bias, double scale)
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && bias.is_cuda(), "A, B, and bias must be CUDA tensors");
    TORCH_CHECK(A.scalar_type() == at::kHalf && B.scalar_type() == at::kHalf && bias.scalar_type() == at::kHalf,
                "A, B, and bias must be fp16 tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && bias.dim() == 1, "expected A[M,K], B[K,N], bias[N]");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && bias.is_contiguous(), "A, B, and bias must be contiguous");

    int64_t m = A.size(0);
    int64_t k = A.size(1);
    int64_t n = B.size(1);
    TORCH_CHECK(B.size(0) == k, "B.shape[0] must match A.shape[1]");
    TORCH_CHECK(bias.size(0) == n, "bias.shape[0] must match B.shape[1]");
    TORCH_CHECK(m % 64 == 0 && n % 64 == 0 && k % 64 == 0,
                "AutoPartitionCudaBackend demo requires M, N, K to be multiples of 64");

    auto Acc = torch::empty({m, n}, A.options().dtype(torch::kFloat32));
    auto C   = torch::empty({m, n}, A.options());

    constexpr int ThreadCount = 128;
    using InputElement        = cutlass::half_t;
    using AccElement          = float;
    using OutputElement       = cutlass::half_t;
    using TileShape           = Shape<Int<64>, Int<64>, Int<64>>;
    using StrideA             = decltype(make_stride(int{}, Int<1>{}));
    using StrideB             = decltype(make_stride(Int<1>{}, int{}));
    using StrideC             = decltype(make_stride(int{}, Int<1>{}));

    using PartA = typename autopartition::AutoPartitioner<cutlass::arch::Sm80,
                                                          cutlass::arch::OpClassTensorOp,
                                                          InputElement,
                                                          StrideA,
                                                          TileShape,
                                                          ThreadCount,
                                                          AccElement,
                                                          16,
                                                          16,
                                                          16>::RoleA;
    using PartB = typename autopartition::AutoPartitioner<cutlass::arch::Sm80,
                                                          cutlass::arch::OpClassTensorOp,
                                                          InputElement,
                                                          StrideB,
                                                          TileShape,
                                                          ThreadCount,
                                                          AccElement,
                                                          16,
                                                          16,
                                                          16>::RoleB;
    using PartC = typename autopartition::AutoPartitioner<cutlass::arch::Sm80,
                                                          cutlass::arch::OpClassTensorOp,
                                                          InputElement,
                                                          StrideC,
                                                          TileShape,
                                                          ThreadCount,
                                                          AccElement,
                                                          16,
                                                          16,
                                                          16>::RoleC;

    static_assert(PartA::UseLdMatrix && PartB::UseLdMatrix, "AutoPartition runtime expects ldmatrix for A/B.");
    static_assert(PartC::HasFusionSharedMapping, "AutoPartition runtime expects RoleC fusion mapping.");

    c10::cuda::CUDAGuard device_guard(A.device());
    dim3 grid(int(m / 64), int(n / 64));
    dim3 block(ThreadCount);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto const *ptr_A    = reinterpret_cast<InputElement const *>(A.data_ptr<at::Half>());
    auto const *ptr_B    = reinterpret_cast<InputElement const *>(B.data_ptr<at::Half>());
    auto const *ptr_bias = reinterpret_cast<InputElement const *>(bias.data_ptr<at::Half>());
    auto       *ptr_Acc  = Acc.data_ptr<float>();
    auto       *ptr_C    = reinterpret_cast<OutputElement *>(C.data_ptr<at::Half>());

    fused_gemm_bias_scale_gelu_kernel<PartA,
                                      PartB,
                                      PartC,
                                      InputElement,
                                      AccElement,
                                      StrideA,
                                      StrideB,
                                      StrideC,
                                      ThreadCount><<<grid, block, 0, stream>>>(ptr_A,
                                                                               make_stride(int(k), Int<1>{}),
                                                                               ptr_B,
                                                                               make_stride(Int<1>{}, int(n)),
                                                                               ptr_Acc,
                                                                               make_stride(int(n), Int<1>{}),
                                                                               int(m),
                                                                               int(n),
                                                                               int(k));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int total = int(m * n);
    int epilogue_threads = 256;
    int epilogue_blocks = (total + epilogue_threads - 1) / epilogue_threads;
    bias_scale_gelu_epilogue_kernel<<<epilogue_blocks, epilogue_threads, 0, stream>>>(ptr_Acc,
                                                                                     ptr_bias,
                                                                                     static_cast<float>(scale),
                                                                                     ptr_C,
                                                                                     total,
                                                                                     int(n));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return C;
}

} // namespace zfm_autopartition_runtime

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("fused_gemm_bias_scale_gelu",
          &zfm_autopartition_runtime::fused_gemm_bias_scale_gelu,
          "AutoPartitioner SM80 fused GEMM + bias + scale + GELU");
}
