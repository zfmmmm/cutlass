#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <nvtx3/nvToolsExt.h>
#include <random>
#include <vector>

#include "cutlass/numeric_conversion.h"

namespace autopartition_bench {

class NvtxRange
{
public:
    explicit NvtxRange(char const *name) { nvtxRangePushA(name); }
    ~NvtxRange() { nvtxRangePop(); }

    NvtxRange(NvtxRange const &) = delete;
    NvtxRange &operator=(NvtxRange const &) = delete;
};

struct GemmOptions
{
    int  m             = 512;
    int  n             = 512;
    int  k             = 512;
    int  iterations    = 20;
    int  warmup        = 5;
    bool verify        = true;
    bool print_layouts = false;
};

struct OutputStats
{
    double        sum      = 0.0;
    double        abs_sum  = 0.0;
    double        sq_sum   = 0.0;
    std::uint64_t hash     = 0;
};

inline int round_up(int value, int multiple) { return ((value + multiple - 1) / multiple) * multiple; }

inline bool check_cuda(cudaError_t status, char const *what)
{
    if (status != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(status) << "\n";
        return false;
    }
    return true;
}

inline bool parse_int_arg(char const *arg, char const *prefix, int &value)
{
    auto n = std::strlen(prefix);
    if (std::strncmp(arg, prefix, n) != 0) {
        return false;
    }
    value = std::atoi(arg + n);
    return true;
}

inline GemmOptions parse_options(int argc, char **argv)
{
    GemmOptions options;
    for (int i = 1; i < argc; ++i) {
        if (parse_int_arg(argv[i], "--m=", options.m) || parse_int_arg(argv[i], "--n=", options.n)
            || parse_int_arg(argv[i], "--k=", options.k) || parse_int_arg(argv[i], "--iterations=", options.iterations)
            || parse_int_arg(argv[i], "--warmup=", options.warmup)) {
            continue;
        }
        if (std::strcmp(argv[i], "--skip-reference") == 0) {
            options.verify = false;
            continue;
        }
        if (std::strcmp(argv[i], "--print-layouts") == 0) {
            options.print_layouts = true;
            continue;
        }
    }
    return options;
}

inline void print_options(char const *name, GemmOptions const &options, int padded_m, int padded_n, int padded_k)
{
    std::cout << name << "\n"
              << "  logical_mnk = " << options.m << "x" << options.n << "x" << options.k << "\n"
              << "  padded_mnk  = " << padded_m << "x" << padded_n << "x" << padded_k << "\n"
              << "  warmup/iters = " << options.warmup << "/" << options.iterations << "\n"
              << "  input_seed   = 20260517\n"
              << "  layouts      = A:row_major(M,K) B:row_major(K,N) C:row_major(M,N)\n";
}

template <class Element>
inline Element from_float(float value)
{
    return cutlass::NumericConverter<Element, float>{}(value);
}

template <class Element>
inline float to_float(Element value)
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
void reference_gemm_kn(std::vector<ElementA> const &a,
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

template <class ElementC>
float max_abs_diff_active(std::vector<ElementC> const &actual,
                          std::vector<float> const    &reference,
                          int                          m,
                          int                          n,
                          int                          padded_n)
{
    float max_diff = 0.0f;
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            max_diff = std::max(max_diff, std::abs(to_float(actual[row * padded_n + col]) - reference[row * n + col]));
        }
    }
    return max_diff;
}

inline std::uint64_t fnv1a_bytes(void const *data, std::size_t bytes)
{
    auto const   *ptr  = static_cast<unsigned char const *>(data);
    std::uint64_t hash = 1469598103934665603ull;
    for (std::size_t i = 0; i < bytes; ++i) {
        hash ^= std::uint64_t(ptr[i]);
        hash *= 1099511628211ull;
    }
    return hash;
}

template <class Element>
std::uint64_t vector_hash(std::vector<Element> const &data)
{
    return fnv1a_bytes(data.data(), data.size() * sizeof(Element));
}

template <class ElementC>
OutputStats output_stats_active(std::vector<ElementC> const &actual, int m, int n, int padded_n)
{
    OutputStats stats;
    stats.hash = 1469598103934665603ull;
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            ElementC const value = actual[row * padded_n + col];
            float const    f     = to_float(value);
            stats.sum += double(f);
            stats.abs_sum += double(std::abs(f));
            stats.sq_sum += double(f) * double(f);
            stats.hash ^= fnv1a_bytes(&value, sizeof(ElementC));
            stats.hash *= 1099511628211ull;
        }
    }
    return stats;
}

template <class Launch>
bool time_launch_ms(Launch      launch,
                    int         warmup,
                    int         iterations,
                    float      &elapsed_ms_out,
                    char const *warmup_range,
                    char const *timed_range)
{
    {
        NvtxRange range(warmup_range);
        for (int i = 0; i < warmup; ++i) {
            if (!launch()) {
                return false;
            }
        }
        if (!check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warmup)")) {
            return false;
        }
    }

    cudaEvent_t start{};
    cudaEvent_t stop{};
    if (!check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)")
        || !check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)")) {
        return false;
    }
    {
        NvtxRange range(timed_range);
        if (!check_cuda(cudaEventRecord(start), "cudaEventRecord(start)")) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return false;
        }
        for (int i = 0; i < iterations; ++i) {
            if (!launch()) {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                return false;
            }
        }
        if (!check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)")
            || !check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)")
            || !check_cuda(cudaGetLastError(), "cudaGetLastError(timed launch)")) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return false;
        }
    }

    float elapsed_ms = 0.0f;
    if (!check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime")) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return false;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    elapsed_ms_out = elapsed_ms / float(iterations);
    return true;
}

inline double tflops(int m, int n, int k, float ms)
{
    return (2.0 * double(m) * double(n) * double(k)) / (double(ms) * 1.0e-3) / 1.0e12;
}

inline void print_hashes(std::uint64_t input_a_hash, std::uint64_t input_b_hash)
{
    std::cout << "  input_a_hash = " << input_a_hash << "\n"
              << "  input_b_hash = " << input_b_hash << "\n";
}

inline void print_output_stats(OutputStats const &stats)
{
    std::cout << "  output_sum   = " << stats.sum << "\n"
              << "  output_abs_sum = " << stats.abs_sum << "\n"
              << "  output_sq_sum = " << stats.sq_sum << "\n"
              << "  output_hash  = " << stats.hash << "\n";
}

} // namespace autopartition_bench
