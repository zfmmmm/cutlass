// PyTorch CUDA extension for the torch.compile AutoPartition backend.

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

#include <type_traits>

using namespace cute;

namespace autopartition_torch_runtime {

template <class PartA,
          class PartB,
          class PartC,
          class InputElement,
          class OutputElement,
          class StrideA,
          class StrideB,
          class StrideC,
          int ThreadCount>
__global__ __launch_bounds__(ThreadCount, 4) void autopartition_gemm_accum_kernel(InputElement const *ptr_A,
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

template <bool ApplyGelu, class BiasElement, class OutputElement>
__global__ void bias_epilogue_kernel(float const *acc,
                                     BiasElement const *bias,
                                     OutputElement *out,
                                     int total,
                                     int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int   col = idx % n;
    float x   = acc[idx] + static_cast<float>(bias[col]);
    if constexpr (ApplyGelu) {
        x = 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
    }
    cutlass::NumericConverter<OutputElement, float> convert;
    out[idx] = convert(x);
}

template <class OutputElement>
__global__ void cast_epilogue_kernel(float const *acc, OutputElement *out, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    cutlass::NumericConverter<OutputElement, float> convert;
    out[idx] = convert(acc[idx]);
}

void check_gemm_inputs(torch::Tensor const &A, torch::Tensor const &B)
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
    TORCH_CHECK(A.scalar_type() == at::kHalf && B.scalar_type() == at::kHalf, "A and B must be fp16 tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "expected A[M,K] and B[K,N]");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "A and B must be contiguous");
    TORCH_CHECK(B.size(0) == A.size(1), "B.shape[0] must match A.shape[1]");
    TORCH_CHECK(A.size(0) % 64 == 0 && A.size(1) % 64 == 0 && B.size(1) % 64 == 0,
                "AutoPartition runtime requires M, N, K to be multiples of 64");
}

void check_bias(torch::Tensor const &B, torch::Tensor const &bias)
{
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(bias.scalar_type() == at::kHalf, "bias must be fp16");
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == B.size(1), "bias must have shape [N]");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
}

torch::Tensor autopartition_gemm_accum(torch::Tensor A, torch::Tensor B)
{
    check_gemm_inputs(A, B);
    int64_t m = A.size(0);
    int64_t k = A.size(1);
    int64_t n = B.size(1);
    auto Acc = torch::empty({m, n}, A.options().dtype(torch::kFloat32));

    constexpr int ThreadCount = 128;
    using InputElement        = cutlass::half_t;
    using AccElement          = float;
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
    static_assert(!std::is_void<typename PartA::GlobalToSharedCopy>::value, "RoleA must expose GlobalToSharedCopy.");
    static_assert(!std::is_void<typename PartB::GlobalToSharedCopy>::value, "RoleB must expose GlobalToSharedCopy.");
    static_assert(!std::is_void<typename PartA::SmemToRegCopyOperation>::value,
                  "RoleA must expose SmemToRegCopyOperation.");
    static_assert(!std::is_void<typename PartB::SmemToRegCopyOperation>::value,
                  "RoleB must expose SmemToRegCopyOperation.");
    static_assert(!std::is_void<typename PartC::TiledMma>::value, "RoleC must expose TiledMma.");
    static_assert(!std::is_void<typename PartC::OutputRegisterToGlobalCopy>::value,
                  "RoleC must expose OutputRegisterToGlobalCopy.");
    static_assert(PartC::HasFusionSharedMapping, "RoleC must expose fusion shared-memory mapping.");

    c10::cuda::CUDAGuard device_guard(A.device());
    dim3 grid(int(m / 64), int(n / 64));
    dim3 block(ThreadCount);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto const *ptr_A   = reinterpret_cast<InputElement const *>(A.data_ptr<at::Half>());
    auto const *ptr_B   = reinterpret_cast<InputElement const *>(B.data_ptr<at::Half>());
    auto       *ptr_Acc = Acc.data_ptr<float>();

    autopartition_gemm_accum_kernel<PartA,
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
    return Acc;
}

torch::Tensor fused_gemm_bias_gelu(torch::Tensor A, torch::Tensor B, torch::Tensor bias)
{
    check_gemm_inputs(A, B);
    check_bias(B, bias);
    auto Acc = autopartition_gemm_accum(A, B);
    auto C   = torch::empty({A.size(0), B.size(1)}, A.options());

    c10::cuda::CUDAGuard device_guard(A.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int total = int(A.size(0) * B.size(1));
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    auto const *ptr_bias = reinterpret_cast<cutlass::half_t const *>(bias.data_ptr<at::Half>());
    auto       *ptr_C    = reinterpret_cast<cutlass::half_t *>(C.data_ptr<at::Half>());
    bias_epilogue_kernel<true><<<blocks, threads, 0, stream>>>(Acc.data_ptr<float>(), ptr_bias, ptr_C, total, int(B.size(1)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return C;
}

torch::Tensor fused_gemm_bias(torch::Tensor A, torch::Tensor B, torch::Tensor bias)
{
    check_gemm_inputs(A, B);
    check_bias(B, bias);
    auto Acc = autopartition_gemm_accum(A, B);
    auto C   = torch::empty({A.size(0), B.size(1)}, A.options());

    c10::cuda::CUDAGuard device_guard(A.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int total = int(A.size(0) * B.size(1));
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    auto const *ptr_bias = reinterpret_cast<cutlass::half_t const *>(bias.data_ptr<at::Half>());
    auto       *ptr_C    = reinterpret_cast<cutlass::half_t *>(C.data_ptr<at::Half>());
    bias_epilogue_kernel<false><<<blocks, threads, 0, stream>>>(Acc.data_ptr<float>(), ptr_bias, ptr_C, total, int(B.size(1)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return C;
}

torch::Tensor gemm(torch::Tensor A, torch::Tensor B)
{
    auto Acc = autopartition_gemm_accum(A, B);
    auto C   = torch::empty({A.size(0), B.size(1)}, A.options());

    c10::cuda::CUDAGuard device_guard(A.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int total = int(A.size(0) * B.size(1));
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    auto *ptr_C = reinterpret_cast<cutlass::half_t *>(C.data_ptr<at::Half>());
    cast_epilogue_kernel<<<blocks, threads, 0, stream>>>(Acc.data_ptr<float>(), ptr_C, total);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return C;
}

} // namespace autopartition_torch_runtime

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("fused_gemm_bias_gelu",
          &autopartition_torch_runtime::fused_gemm_bias_gelu,
          "AutoPartitioner SM80 GEMM + bias + GELU");
    m.def("fused_gemm_bias",
          &autopartition_torch_runtime::fused_gemm_bias,
          "AutoPartitioner SM80 GEMM + bias");
    m.def("gemm", &autopartition_torch_runtime::gemm, "AutoPartitioner SM80 GEMM");
}
