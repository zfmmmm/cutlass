#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/half.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

namespace {

struct Options
{
    int  m          = 512;
    int  n          = 1024;
    int  k          = 256;
    int  iterations = 5;
    int  warmup     = 2;
    bool verify     = true;
};

int round_up(int value, int multiple) { return ((value + multiple - 1) / multiple) * multiple; }

bool check_cuda(cudaError_t status, char const *what)
{
    if (status != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(status) << "\n";
        return false;
    }
    return true;
}

bool parse_int_arg(char const *arg, char const *prefix, int &value)
{
    auto n = std::strlen(prefix);
    if (std::strncmp(arg, prefix, n) != 0) {
        return false;
    }
    value = std::atoi(arg + n);
    return true;
}

Options parse_options(int argc, char **argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        if (parse_int_arg(argv[i], "--m=", options.m) || parse_int_arg(argv[i], "--n=", options.n)
            || parse_int_arg(argv[i], "--k=", options.k) || parse_int_arg(argv[i], "--iterations=", options.iterations)
            || parse_int_arg(argv[i], "--warmup=", options.warmup)) {
            continue;
        }
        if (std::strcmp(argv[i], "--skip-reference") == 0) {
            options.verify = false;
        }
    }
    return options;
}

template <class Element>
Element from_float(float value)
{
    return cutlass::NumericConverter<Element, float>{}(value);
}

template <class Element>
float to_float(Element value)
{
    return static_cast<float>(value);
}

template <class Element>
void fill_a_b_padded(std::vector<Element> &a,
                     std::vector<Element> &b,
                     int                   m,
                     int                   n,
                     int                   k,
                     int                   padded_m,
                     int                   padded_n,
                     int                   padded_k)
{
    std::mt19937                          rng(20260517);
    std::uniform_real_distribution<float> dist(-0.25f, 0.25f);

    std::fill(a.begin(), a.end(), Element{});
    std::fill(b.begin(), b.end(), Element{});
    for (int row = 0; row < m; ++row) {
        for (int kk = 0; kk < k; ++kk) {
            a[row * padded_k + kk] = from_float<Element>(dist(rng));
        }
    }
    for (int kk = 0; kk < k; ++kk) {
        for (int col = 0; col < n; ++col) {
            b[kk * padded_n + col] = from_float<Element>(dist(rng));
        }
    }
}

template <class ElementA, class ElementB>
void reference_gemm(std::vector<ElementA> const &a,
                    std::vector<ElementB> const &b,
                    std::vector<float>          &c,
                    int                          m,
                    int                          n,
                    int                          k,
                    int                          padded_n,
                    int                          padded_k)
{
    std::fill(c.begin(), c.end(), 0.0f);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float acc = 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                acc += to_float(a[row * padded_k + kk]) * to_float(b[kk * padded_n + col]);
            }
            c[row * n + col] = acc;
        }
    }
}

float max_abs_diff_active(std::vector<float> const &actual,
                          std::vector<float> const &reference,
                          int                       m,
                          int                       n,
                          int                       padded_n)
{
    float max_diff = 0.0f;
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            max_diff = std::max(max_diff, std::abs(actual[row * padded_n + col] - reference[row * n + col]));
        }
    }
    return max_diff;
}

template <class Launch>
float time_launch_ms(Launch launch, int warmup, int iterations)
{
    for (int i = 0; i < warmup; ++i) {
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

double tflops(int m, int n, int k, float ms)
{
    return (2.0 * double(m) * double(n) * double(k)) / (double(ms) * 1.0e-3) / 1.0e12;
}

bool is_sm100a_device()
{
    cudaDeviceProp props{};
    cudaError_t    error = cudaGetDeviceProperties(&props, 0);
    if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << "\n";
        return false;
    }
    if (props.major != 10 || props.minor != 0) {
        std::cerr << "This baseline requires an SM100/100a Blackwell datacenter GPU. Found " << props.major << "."
                  << props.minor << "\n";
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char **argv)
{
    using namespace cute;

    Options options = parse_options(argc, argv);

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    using ElementA           = cutlass::half_t;
    using ElementB           = cutlass::half_t;
    using ElementC           = float;
    using ElementD           = float;
    using ElementAccumulator = float;
    using ElementCompute     = float;
    using LayoutA            = cutlass::layout::RowMajor;
    using LayoutB            = cutlass::layout::RowMajor;
    using LayoutC            = cutlass::layout::RowMajor;
    using LayoutD            = cutlass::layout::RowMajor;
    using TileShape          = Shape<Int<128>, Int<256>, Int<64>>;
    using ClusterShape       = Shape<_1, _1, _1>;

    constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm100,
        cutlass::arch::OpClassTensorOp,
        TileShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator,
        ElementCompute,
        ElementC,
        LayoutC,
        AlignmentC,
        ElementD,
        LayoutD,
        AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100,
        cutlass::arch::OpClassTensorOp,
        ElementA,
        LayoutA,
        AlignmentA,
        ElementB,
        LayoutB,
        AlignmentB,
        ElementAccumulator,
        TileShape,
        ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                                           CollectiveMainloop,
                                                           CollectiveEpilogue>;
    using Gemm       = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideA    = typename Gemm::GemmKernel::StrideA;
    using StrideB    = typename Gemm::GemmKernel::StrideB;
    using StrideC    = typename Gemm::GemmKernel::StrideC;
    using StrideD    = typename Gemm::GemmKernel::StrideD;

    constexpr int bM = cute::size<0>(TileShape{});
    constexpr int bN = cute::size<1>(TileShape{});
    constexpr int bK = cute::size<2>(TileShape{});
    int           padded_m = round_up(options.m, bM);
    int           padded_n = round_up(options.n, bN);
    int           padded_k = round_up(options.k, bK);

    std::cout << "SM100 CUTLASS official CollectiveBuilder TMA/UMMA baseline\n"
              << "  logical_mnk = " << options.m << "x" << options.n << "x" << options.k << "\n"
              << "  padded_mnk  = " << padded_m << "x" << padded_n << "x" << padded_k << "\n"
              << "  warmup/iters = " << options.warmup << "/" << options.iterations << "\n";

    if (!is_sm100a_device()) {
        return 255;
    }

    std::vector<ElementA> hA(padded_m * padded_k);
    std::vector<ElementB> hB(padded_k * padded_n);
    std::vector<ElementD> hD(padded_m * padded_n, ElementD{});
    std::vector<float>    hRef(options.m * options.n);
    fill_a_b_padded(hA, hB, options.m, options.n, options.k, padded_m, padded_n, padded_k);
    if (options.verify) {
        reference_gemm(hA, hB, hRef, options.m, options.n, options.k, padded_n, padded_k);
    }

    ElementA *dA = nullptr;
    ElementB *dB = nullptr;
    ElementD *dD = nullptr;
    if (!check_cuda(cudaMalloc(&dA, hA.size() * sizeof(ElementA)), "cudaMalloc(A)")
        || !check_cuda(cudaMalloc(&dB, hB.size() * sizeof(ElementB)), "cudaMalloc(B)")
        || !check_cuda(cudaMalloc(&dD, hD.size() * sizeof(ElementD)), "cudaMalloc(D)")
        || !check_cuda(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(ElementA), cudaMemcpyHostToDevice), "copy A")
        || !check_cuda(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(ElementB), cudaMemcpyHostToDevice), "copy B")
        || !check_cuda(cudaMemset(dD, 0, hD.size() * sizeof(ElementD)), "zero D")) {
        return 1;
    }

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(padded_m, padded_k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(padded_n, padded_k, 1));
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(padded_m, padded_n, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(padded_m, padded_n, 1));

    typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                                       {padded_m, padded_n, padded_k, 1},
                                       {dA, stride_A, dB, stride_B},
                                       {{1.0f, 0.0f}, dD, stride_C, dD, stride_D}};

    Gemm gemm;
    auto status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS can_implement failed\n";
        return 1;
    }

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    status = gemm.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS initialize failed\n";
        return 1;
    }

    auto launch = [&]() {
        cutlass::Status launch_status = gemm.run();
        if (launch_status != cutlass::Status::kSuccess) {
            std::cerr << "CUTLASS launch failed\n";
        }
    };

    launch();
    if (!check_cuda(cudaDeviceSynchronize(), "CUTLASS SM100 GEMM")) {
        return 1;
    }
    if (!check_cuda(cudaMemcpy(hD.data(), dD, hD.size() * sizeof(ElementD), cudaMemcpyDeviceToHost), "copy D")) {
        return 1;
    }

    float max_diff = options.verify ? max_abs_diff_active(hD, hRef, options.m, options.n, padded_n) : 0.0f;
    float ms       = time_launch_ms(launch, options.warmup, options.iterations);

    std::cout << "  max_abs_diff = " << max_diff << "\n"
              << "  runtime_ms   = " << ms << "\n"
              << "  tflops       = " << tflops(options.m, options.n, options.k, ms) << "\n";

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dD);
    return (!options.verify || max_diff < 2.5e-2f) ? 0 : 2;
#else
    std::cout << "CUTLASS_ARCH_MMA_SM100_SUPPORTED must be enabled, but it is not. Test is waived.\n";
    return 0;
#endif
}
