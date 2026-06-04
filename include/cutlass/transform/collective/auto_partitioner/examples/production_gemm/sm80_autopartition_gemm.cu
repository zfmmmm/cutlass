#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/algorithm/cooperative_gemm.hpp>
#include <cute/util/print_tensor.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#include "cutlass/numeric_conversion.h"
#include "cutlass/transform/collective/auto_partitioner/auto_partitioner_builder.hpp"
#include "tools/auto_partitioner_bench/sm80_benchmark_common.hpp"
using namespace cute;

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
        for (int col = 0; col < k; ++col) {
            a[row * padded_k + col] = from_float<Element>(dist(rng));
        }
    }
    for (int row = 0; row < n; ++row) {
        for (int kk = 0; kk < k; ++kk) {
            b[row + kk * padded_n] = from_float<Element>(dist(rng));
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

inline double tflops(int m, int n, int k, float ms)
{
    return (2.0 * double(m) * double(n) * double(k)) / (double(ms) * 1.0e-3) / 1.0e12;
}

inline void print_ncu_hint(char const *binary, char const *filter = "")
{
    std::cout << "NCU bank-conflict probe:\n"
              << "  ncu --metrics "
              << "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,"
              << "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum " << binary << " " << filter << "\n";
}

} // namespace autopartition::examples::production

namespace autopartition_sm80_production {

template <class PartA,
          class PartB,
          class PartC,
          class InputElement,
          class OutputElement,
          class StrideA,
          class StrideB,
          class StrideC,
          int ThreadCount>
__global__ __launch_bounds__(ThreadCount, 4) void sm80_autopartition_gemm_kernel(InputElement const *ptr_A,
                                                                                 StrideA             stride_A,
                                                                                 InputElement const *ptr_B,
                                                                                 StrideB             stride_B,
                                                                                 OutputElement      *ptr_C,
                                                                                 StrideC             stride_C,
                                                                                 int                 padded_m,
                                                                                 int                 padded_n,
                                                                                 int                 padded_k)
{
    using bM = decltype(size<0>(typename PartA::SmemLayout{}));
    using bN = decltype(size<0>(typename PartB::SmemLayout{}));
    using bK = decltype(size<1>(typename PartA::SmemLayout{}));

    Tensor gA_full = make_tensor(make_gmem_ptr(ptr_A), make_shape(padded_m, padded_k), stride_A);
    Tensor gB_full = make_tensor(make_gmem_ptr(ptr_B), make_shape(padded_n, padded_k), stride_B);
    Tensor gC_full = make_tensor(make_gmem_ptr(ptr_C), make_shape(padded_m, padded_n), stride_C);

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

        Tensor                             tAgA       = thr_copy_A.partition_S(gA_k);
        Tensor                             tAsA       = thr_copy_A.partition_D(sA);
        Tensor                             tBgB       = thr_copy_B.partition_S(gB_k);
        Tensor                             tBsB       = thr_copy_B.partition_D(sB);
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
    auto r2g_thr_copy_C   = r2g_tiled_copy_C.get_thread_slice(threadIdx.x);
    auto r2g_contract     = PartC::retile_register_to_output(r2g_thr_copy_C, tSR_rD, gC);
    copy(r2g_tiled_copy_C, cute::get<0>(r2g_contract), cute::get<1>(r2g_contract));
}

__global__ void sm80_cp_async_zfill_probe_kernel(float const *in, float *out)
{
    __shared__ float smem[1];
    Tensor           g = make_tensor(make_gmem_ptr(in), make_shape(Int<1>{}));
    Tensor           s = make_tensor(make_smem_ptr(smem), make_shape(Int<1>{}));
    if (threadIdx.x == 0) {
        using Atom = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint_byte_t<4>>, float>;
        copy(Atom{}.with(false), g, s);
    }
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();
    if (threadIdx.x == 0) {
        *out = smem[0];
    }
}

} // namespace autopartition_sm80_production

int main(int argc, char **argv)
{
    using namespace autopartition::examples::production;
    using namespace autopartition_sm80_production;

    autopartition_bench::GemmOptions options = autopartition_bench::parse_options(argc, argv);
    constexpr int ThreadCount = 128;
    using InputElement        = cutlass::half_t;
    using OutputElement       = float;
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
                                                          OutputElement,
                                                          16,
                                                          16,
                                                          16>::RoleA;
    using PartB = typename autopartition::AutoPartitioner<cutlass::arch::Sm80,
                                                          cutlass::arch::OpClassTensorOp,
                                                          InputElement,
                                                          StrideB,
                                                          TileShape,
                                                          ThreadCount,
                                                          OutputElement,
                                                          16,
                                                          16,
                                                          16>::RoleB;
    using PartC = typename autopartition::AutoPartitioner<cutlass::arch::Sm80,
                                                          cutlass::arch::OpClassTensorOp,
                                                          InputElement,
                                                          StrideC,
                                                          TileShape,
                                                          ThreadCount,
                                                          OutputElement,
                                                          16,
                                                          16,
                                                          16>::RoleC;

    static_assert(PartA::UseLdMatrix && PartB::UseLdMatrix, "SM80 production example must use LDSM.");
    static_assert(PartA::SwizzleBase == 3 && PartB::SwizzleBase == 3,
                  "64x64x64 half tiles must route to Swizzle<3,3,3> shared layouts.");
    static_assert(!std::is_void<typename PartC::RegisterToGlobalCopy>::value,
                  "SM80 epilogue must expose a register-to-global vectorized store.");
    static_assert(PartC::EpilogueInstructionThreads == 16,
                  "SM80 epilogue writeback must use the 16-thread shared-memory issue group model.");
    static_assert(PartC::EpilogueLayoutCandidateCount >= 32,
                  "SM80 epilogue must choose from many generic shared-memory layout candidates.");
    static_assert(PartC::HasZeroGlueEpilogueMapping,
                  "SM80 epilogue must expose a complete shared/register/global mapping contract.");
    static_assert(std::is_same<typename PartC::EpilogueElement, typename PartC::ElementCompute>::value,
                  "SM80 production epilogue shared memory must store accumulator elements.");
    static_assert(cute::cosize_v<typename PartC::SmemLayout>
                      == int(size<0>(TileShape{})) * int(size<1>(TileShape{})),
                  "SM80 production epilogue shared layout must be padding-free.");

    int padded_m = autopartition_bench::round_up(options.m, int(size<0>(TileShape{})));
    int padded_n = autopartition_bench::round_up(options.n, int(size<1>(TileShape{})));
    int padded_k = autopartition_bench::round_up(options.k, int(size<2>(TileShape{})));
    autopartition_bench::print_options("SM80 AutoPartitioner cp.async/LDSM/mma.sync GEMM", options, padded_m, padded_n, padded_k);
    std::cout << "  threadblock = 64x64x64\n"
              << "  warp        = autopartitioner\n"
              << "  instruction = 16x8x16\n"
              << "  stages      = 1\n"
              << "  alignments  = A8 / B8\n";

    if (options.print_layouts) {
        std::cout << "PartA::SmemLayout\n";
        print_layout(typename PartA::SmemLayout{});
        std::cout << "PartB::SmemLayout\n";
        print_layout(typename PartB::SmemLayout{});
        std::cout << "PartC::SmemLayout\n";
        print_layout(typename PartC::SmemLayout{});
        std::cout << "PartC::EpilogueLayoutCandidateCount = " << PartC::EpilogueLayoutCandidateCount << "\n";
        std::cout << "PartC::EpilogueSwizzle<Base,M,S> = Swizzle<" << PartC::EpilogueSwizzleBase << ","
                  << PartC::EpilogueSwizzleMBase << "," << PartC::EpilogueSwizzleShift << ">\n";
        std::cout << "PartC::EpilogueSwizzleRowElements = " << PartC::EpilogueSwizzleRowElements << "\n";
        std::cout << "PartC::EpilogueSwizzleMinorElements = " << PartC::EpilogueSwizzleMinorElements << "\n";
        std::cout << "PartC::EpilogueBankConflictScore = " << PartC::EpilogueBankConflictScore << "\n";
        std::cout << "PartC::EpilogueNaiveBankConflictScore = " << PartC::EpilogueNaiveBankConflictScore << "\n";
        std::cout << "PartC::EpilogueVectorAlignmentPenalty = " << PartC::EpilogueVectorAlignmentPenalty << "\n";
    }

    std::vector<InputElement>  hA(padded_m * padded_k);
    std::vector<InputElement>  hB(padded_n * padded_k);
    std::vector<OutputElement> hC(padded_m * padded_n, OutputElement{});
    std::vector<float>         hRef(options.m * options.n);
    autopartition_bench::fill_a_b_padded(hA, hB, options.m, options.n, options.k, padded_m, padded_n, padded_k);
    autopartition_bench::print_hashes(autopartition_bench::vector_hash(hA), autopartition_bench::vector_hash(hB));
    if (options.verify) {
        autopartition_bench::reference_gemm_kn(hA, hB, hRef, options.m, options.n, options.k, padded_n, padded_k);
    }

    InputElement  *dA = nullptr;
    InputElement  *dB = nullptr;
    OutputElement *dC = nullptr;
    if (!autopartition_bench::check_cuda(cudaMalloc(&dA, hA.size() * sizeof(InputElement)), "cudaMalloc(A)")
        || !autopartition_bench::check_cuda(cudaMalloc(&dB, hB.size() * sizeof(InputElement)), "cudaMalloc(B)")
        || !autopartition_bench::check_cuda(cudaMalloc(&dC, hC.size() * sizeof(OutputElement)), "cudaMalloc(C)")
        || !autopartition_bench::check_cuda(
            cudaMemcpy(dA, hA.data(), hA.size() * sizeof(InputElement), cudaMemcpyHostToDevice), "copy A")
        || !autopartition_bench::check_cuda(
            cudaMemcpy(dB, hB.data(), hB.size() * sizeof(InputElement), cudaMemcpyHostToDevice), "copy B")
        || !autopartition_bench::check_cuda(cudaMemset(dC, 0, hC.size() * sizeof(OutputElement)), "zero C")) {
        return 1;
    }

    dim3 grid(padded_m / int(size<0>(TileShape{})), padded_n / int(size<1>(TileShape{})));
    dim3 block(ThreadCount);
    auto launch = [&]() -> bool {
        sm80_autopartition_gemm_kernel<PartA,
                                       PartB,
                                       PartC,
                                       InputElement,
                                       OutputElement,
                                       StrideA,
                                       StrideB,
                                       StrideC,
                                       ThreadCount><<<grid, block>>>(dA,
                                                                     make_stride(padded_k, Int<1>{}),
                                                                     dB,
                                                                     make_stride(Int<1>{}, padded_n),
                                                                     dC,
                                                                     make_stride(padded_n, Int<1>{}),
                                                                     padded_m,
                                                                     padded_n,
                                                                     padded_k);
        return autopartition_bench::check_cuda(cudaGetLastError(), "launch SM80 AutoPartitioner GEMM");
    };

    if (!launch()) {
        return 1;
    }
    if (!autopartition_bench::check_cuda(cudaDeviceSynchronize(), "SM80 GEMM kernel")) {
        return 1;
    }
    if (!autopartition_bench::check_cuda(
            cudaMemcpy(hC.data(), dC, hC.size() * sizeof(OutputElement), cudaMemcpyDeviceToHost), "copy C")) {
        return 1;
    }

    autopartition_bench::OutputStats stats = autopartition_bench::output_stats_active(hC, options.m, options.n, padded_n);
    float max_diff =
        options.verify ? autopartition_bench::max_abs_diff_active(hC, hRef, options.m, options.n, padded_n) : 0.0f;
    float ms = 0.0f;
    if (!autopartition_bench::time_launch_ms(launch, options.warmup, options.iterations, ms)) {
        return 1;
    }
    autopartition_bench::print_output_stats(stats);
    std::cout << "  max_abs_diff = " << max_diff << "\n"
              << "  runtime_ms   = " << ms << "\n"
              << "  tflops       = " << autopartition_bench::tflops(options.m, options.n, options.k, ms) << "\n";

    float *dProbeIn = nullptr, *dProbeOut = nullptr;
    float  hProbeIn  = std::numeric_limits<float>::quiet_NaN();
    float  hProbeOut = std::numeric_limits<float>::quiet_NaN();
    check_cuda(cudaMalloc(&dProbeIn, sizeof(float)), "cudaMalloc(zfill in)");
    check_cuda(cudaMalloc(&dProbeOut, sizeof(float)), "cudaMalloc(zfill out)");
    check_cuda(cudaMemcpy(dProbeIn, &hProbeIn, sizeof(float), cudaMemcpyHostToDevice), "copy zfill in");
    sm80_cp_async_zfill_probe_kernel<<<1, 32>>>(dProbeIn, dProbeOut);
    check_cuda(cudaDeviceSynchronize(), "SM80 ZFILL probe");
    check_cuda(cudaMemcpy(&hProbeOut, dProbeOut, sizeof(float), cudaMemcpyDeviceToHost), "copy zfill out");
    std::cout << "  cp.async_zfill_false_predicate = " << hProbeOut << "\n";
    print_ncu_hint("./sm80_autopartition_gemm", "--print-layouts");

    cudaFree(dProbeIn);
    cudaFree(dProbeOut);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return (!options.verify || max_diff < 2.5e-2f) && hProbeOut == 0.0f ? 0 : 2;
}
