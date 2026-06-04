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
#include "tools/auto_partitioner_bench/sm80_benchmark_common.hpp"

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

    autopartition_bench::GemmOptions options = autopartition_bench::parse_options(argc, argv);
    int padded_m = autopartition_bench::round_up(options.m, 64);
    int padded_n = autopartition_bench::round_up(options.n, 64);
    int padded_k = autopartition_bench::round_up(options.k, 64);

    autopartition_bench::print_options("SM80 CUTLASS official device::Gemm baseline", options, padded_m, padded_n, padded_k);
    std::cout << "  threadblock = 64x64x64\n"
              << "  warp        = 32x32x64\n"
              << "  instruction = 16x8x16\n"
              << "  stages      = 3\n"
              << "  alignments  = A8 / B8\n";

    std::vector<InputElement>  hA(padded_m * padded_k);
    std::vector<InputElement>  hB(padded_k * padded_n);
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

    CutlassGemm                     gemm;
    typename CutlassGemm::Arguments arguments(
        {padded_m, padded_n, padded_k}, {dA, padded_k}, {dB, padded_n}, {dC, padded_n}, {dC, padded_n}, {1.0f, 0.0f});

    cutlass::Status status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS can_implement failed: " << cutlassGetStatusString(status) << "\n";
        return 1;
    }

    auto launch = [&]() -> bool {
        cutlass::Status launch_status = gemm(arguments);
        if (launch_status != cutlass::Status::kSuccess) {
            std::cerr << "CUTLASS launch failed: " << cutlassGetStatusString(launch_status) << "\n";
            return false;
        }
        return true;
    };

    if (!launch()) {
        return 1;
    }
    if (!autopartition_bench::check_cuda(cudaDeviceSynchronize(), "CUTLASS SM80 GEMM")) {
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

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return (!options.verify || max_diff < 2.5e-2f) ? 0 : 2;
}
