#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "cutlass/numeric_conversion.h"

namespace autopartition::examples::production {

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
              << "  warmup/iters = " << options.warmup << "/" << options.iterations << "\n";
}

template <class Element> inline Element from_float(float value)
{
    return cutlass::NumericConverter<Element, float>{}(value);
}

template <class Element> inline float to_float(Element value) { return static_cast<float>(value); }

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
        for (int col = 0; col < k; ++col) {
            a[row * padded_k + col] = from_float<Element>(dist(rng));
        }
    }
    for (int col_n = 0; col_n < n; ++col_n) {
        for (int col_k = 0; col_k < k; ++col_k) {
            b[col_n + col_k * padded_n] = from_float<Element>(dist(rng));
        }
    }
}

template <class ElementA, class ElementB>
void reference_gemm_abt(std::vector<ElementA> const &a,
                        std::vector<ElementB> const &b,
                        std::vector<float>          &c,
                        int                          m,
                        int                          n,
                        int                          k,
                        int                          padded_k,
                        int                          padded_n)
{
    std::fill(c.begin(), c.end(), 0.0f);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float acc = 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                acc += to_float(a[row * padded_k + kk]) * to_float(b[col + kk * padded_n]);
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

template <class Launch> float time_launch_ms(Launch launch, int warmup, int iterations)
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

inline double tflops(int m, int n, int k, float ms)
{
    return (2.0 * double(m) * double(n) * double(k)) / (double(ms) * 1.0e-3) / 1.0e12;
}

inline void print_ncu_hint(char const *binary, char const *filter = "")
{
    std::cout << "NCU bank-conflict probe:\n"
              << "  ncu --metrics " << "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,"
              << "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum " << binary << " " << filter << "\n";
}

} // namespace autopartition::examples::production
