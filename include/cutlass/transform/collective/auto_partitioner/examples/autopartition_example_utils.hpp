#pragma once

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/numeric_conversion.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace autopartition::examples {

// 示例里的错误检查保持轻量：失败时打印 CUDA runtime 的真实错误，main 根据
// bool 返回值决定退出码。这样 Nsight 采集时也能看到 kernel 是否真正执行成功。
inline bool check_cuda(cudaError_t status, char const *what)
{
    if (status != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(status) << "\n";
        return false;
    }
    return true;
}

template <class T> __host__ __device__ inline float to_float(T value)
{
    return static_cast<float>(value);
}

template <class T> inline T from_float(float value)
{
    return T(value);
}

// 生成小幅度、可重复、不会让 FP8/FP16 轻易溢出的输入值。
inline float patterned_value(int index)
{
    int lane = (index * 13 + 7) % 23;
    return (float(lane) - 11.0f) * 0.03125f;
}

template <class Element> void fill_pattern(std::vector<Element> &data)
{
    for (int i = 0; i < int(data.size()); ++i) {
        data[i] = from_float<Element>(patterned_value(i));
    }
}

template <class DstTensor, class SrcTensor>
__device__ void convert_tensor(DstTensor &&dst, SrcTensor const &src)
{
    using Dst = typename cute::remove_cvref_t<DstTensor>::value_type;
    using Src = typename cute::remove_cvref_t<SrcTensor>::value_type;
    cutlass::NumericConverter<Dst, Src> convert;
    CUTE_UNROLL
    for (int i = 0; i < cute::size(dst); ++i) {
        dst(i) = convert(src(i));
    }
}

// 约定 B 的逻辑形状为 (N,K)，GEMM 计算 C(m,n)=sum_k A(m,k)*B(n,k)。
// stride 参数以“元素”为单位，和 CuTe make_stride 传入 kernel 的值保持一致。
template <class ElementA, class ElementB>
void reference_gemm(int              M,
                    int              N,
                    int              K,
                    ElementA const  *A,
                    int              stride_am,
                    int              stride_ak,
                    ElementB const  *B,
                    int              stride_bn,
                    int              stride_bk,
                    float           *C,
                    int              stride_cm,
                    int              stride_cn)
{
    for (int n = 0; n < N; ++n) {
        for (int m = 0; m < M; ++m) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += to_float(A[m * stride_am + k * stride_ak]) * to_float(B[n * stride_bn + k * stride_bk]);
            }
            C[m * stride_cm + n * stride_cn] = acc;
        }
    }
}

inline float max_abs_diff(std::vector<float> const &lhs, std::vector<float> const &rhs)
{
    float diff = 0.0f;
    for (int i = 0; i < int(lhs.size()); ++i) {
        diff = std::max(diff, std::abs(lhs[i] - rhs[i]));
    }
    return diff;
}

template <class LhsElement, class RhsElement>
float max_abs_diff(int count, LhsElement const *lhs, RhsElement const *rhs)
{
    float diff = 0.0f;
    for (int i = 0; i < count; ++i) {
        diff = std::max(diff, std::abs(to_float(lhs[i]) - to_float(rhs[i])));
    }
    return diff;
}

template <class Element>
void corrupt_first(std::vector<Element> &data)
{
    if (!data.empty()) {
        data[0] = from_float<Element>(to_float(data[0]) + 1.0f);
    }
}

template <class ElementA, class ElementB, class ElementC>
__global__ void conventional_gemm_kernel(ElementA const *A,
                                         int             stride_am,
                                         int             stride_ak,
                                         ElementB const *B,
                                         int             stride_bn,
                                         int             stride_bk,
                                         ElementC       *C,
                                         int             stride_cm,
                                         int             stride_cn,
                                         int             M,
                                         int             N,
                                         int             K)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) {
        return;
    }

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += to_float(A[m * stride_am + k * stride_ak]) * to_float(B[n * stride_bn + k * stride_bk]);
    }
    C[m * stride_cm + n * stride_cn] = ElementC(acc);
}

template <class Kernel, class... Args>
float time_kernel_ms(Kernel kernel, dim3 grid, dim3 block, size_t smem_bytes, int iterations, Args... args)
{
    for (int i = 0; i < 3; ++i) {
        kernel<<<grid, block, smem_bytes>>>(args...);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start{};
    cudaEvent_t stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        kernel<<<grid, block, smem_bytes>>>(args...);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsed_ms / float(iterations);
}

template <class Launch>
float time_launch_ms(Launch launch, int iterations)
{
    for (int i = 0; i < 3; ++i) {
        launch();
    }
    cudaDeviceSynchronize();

    cudaEvent_t start{};
    cudaEvent_t stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        launch();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsed_ms / float(iterations);
}

inline void print_result(std::string const &name, float max_error, float auto_ms, float baseline_ms)
{
    std::cout << name << "\n";
    std::cout << "  max_abs_error = " << max_error << "\n";
    std::cout << "  autopartition_ms = " << auto_ms << "\n";
    std::cout << "  conventional_ms = " << baseline_ms << "\n";
}

} // namespace autopartition::examples
