#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/half.h"
#include "cutlass/numeric_conversion.h"

namespace {

struct Options
{
    int  m          = 512;
    int  n          = 512;
    int  k          = 512;
    int  iterations = 20;
    int  warmup     = 5;
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

} // namespace

int main(int argc, char **argv)
{
    using InputElement  = cutlass::half_t;
    using OutputElement = float;
    using Layout        = cutlass::layout::RowMajor;

    using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 64>;
    using WarpShape        = cutlass::gemm::GemmShape<32, 32, 64>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
    using EpilogueOp       = cutlass::epilogue::thread::LinearCombination<OutputElement, 4, float, float>;
    using CutlassGemm      = cutlass::gemm::device::Gemm<InputElement,
                                                    Layout,
                                                    InputElement,
                                                    Layout,
                                                    OutputElement,
                                                    Layout,
                                                    float,
                                                    cutlass::arch::OpClassTensorOp,
                                                    cutlass::arch::Sm80,
                                                    ThreadblockShape,
                                                    WarpShape,
                                                    InstructionShape,
                                                    EpilogueOp,
                                                    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
                                                    3,
                                                    8,
                                                    8>;

    Options options  = parse_options(argc, argv);
    int     padded_m = round_up(options.m, 64);
    int     padded_n = round_up(options.n, 64);
    int     padded_k = round_up(options.k, 64);

    std::cout << "SM80 CUTLASS official device::Gemm baseline\n"
              << "  logical_mnk = " << options.m << "x" << options.n << "x" << options.k << "\n"
              << "  padded_mnk  = " << padded_m << "x" << padded_n << "x" << padded_k << "\n"
              << "  warmup/iters = " << options.warmup << "/" << options.iterations << "\n";

    std::vector<InputElement>  hA(padded_m * padded_k);
    std::vector<InputElement>  hB(padded_k * padded_n);
    std::vector<OutputElement> hC(padded_m * padded_n, OutputElement{});
    std::vector<float>         hRef(options.m * options.n);
    fill_a_b_padded(hA, hB, options.m, options.n, options.k, padded_m, padded_n, padded_k);
    if (options.verify) {
        reference_gemm(hA, hB, hRef, options.m, options.n, options.k, padded_n, padded_k);
    }

    InputElement  *dA = nullptr;
    InputElement  *dB = nullptr;
    OutputElement *dC = nullptr;
    if (!check_cuda(cudaMalloc(&dA, hA.size() * sizeof(InputElement)), "cudaMalloc(A)")
        || !check_cuda(cudaMalloc(&dB, hB.size() * sizeof(InputElement)), "cudaMalloc(B)")
        || !check_cuda(cudaMalloc(&dC, hC.size() * sizeof(OutputElement)), "cudaMalloc(C)")
        || !check_cuda(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(InputElement), cudaMemcpyHostToDevice), "copy A")
        || !check_cuda(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(InputElement), cudaMemcpyHostToDevice), "copy B")
        || !check_cuda(cudaMemset(dC, 0, hC.size() * sizeof(OutputElement)), "zero C")) {
        return 1;
    }

    CutlassGemm gemm;
    typename CutlassGemm::Arguments arguments({padded_m, padded_n, padded_k},
                                              {dA, padded_k},
                                              {dB, padded_n},
                                              {dC, padded_n},
                                              {dC, padded_n},
                                              {1.0f, 0.0f});

    cutlass::Status status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS can_implement failed\n";
        return 1;
    }

    auto launch = [&]() {
        cutlass::Status launch_status = gemm(arguments);
        if (launch_status != cutlass::Status::kSuccess) {
            std::cerr << "CUTLASS launch failed\n";
        }
    };

    launch();
    if (!check_cuda(cudaDeviceSynchronize(), "CUTLASS SM80 GEMM")) {
        return 1;
    }
    if (!check_cuda(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(OutputElement), cudaMemcpyDeviceToHost), "copy C")) {
        return 1;
    }

    float max_diff = options.verify ? max_abs_diff_active(hC, hRef, options.m, options.n, padded_n) : 0.0f;
    float ms       = time_launch_ms(launch, options.warmup, options.iterations);

    std::cout << "  max_abs_diff = " << max_diff << "\n"
              << "  runtime_ms   = " << ms << "\n"
              << "  tflops       = " << tflops(options.m, options.n, options.k, ms) << "\n";

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return (!options.verify || max_diff < 2.5e-2f) ? 0 : 2;
}
