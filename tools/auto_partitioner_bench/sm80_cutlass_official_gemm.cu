#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass/gemm/kernel/gemm.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/threadblock/mma_singlestage.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/half.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "tools/auto_partitioner_bench/sm80_benchmark_common.hpp"

namespace {

template <class GemmKernel, class ThreadblockShape, class ThreadblockSwizzle>
class KernelLevelGemmRunner
{
public:
    using Params = typename GemmKernel::Params;

    cutlass::Status initialize(cutlass::gemm::GemmCoord const                  &problem_size,
                               typename GemmKernel::Mma::IteratorA::TensorRef  ref_A,
                               typename GemmKernel::Mma::IteratorB::TensorRef  ref_B,
                               typename GemmKernel::Epilogue::OutputTileIterator::TensorRef ref_C,
                               typename GemmKernel::Epilogue::OutputTileIterator::TensorRef ref_D,
                               typename GemmKernel::OutputOp::Params const     &epilogue)
    {
        auto status = GemmKernel::can_implement(problem_size, ref_A, ref_B, ref_C, ref_D);
        if (status != cutlass::Status::kSuccess) {
            return status;
        }

        ThreadblockSwizzle swizzle;
        cutlass::gemm::GemmCoord grid_shape =
            swizzle.get_tiled_shape(problem_size,
                                    {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
                                    1);

        params_ = Params{problem_size, grid_shape, ref_A, ref_B, ref_C, ref_D, epilogue};
        return cutlass::Status::kSuccess;
    }

    cutlass::Status run(cudaStream_t stream = nullptr)
    {
        ThreadblockSwizzle swizzle;
        dim3               grid  = swizzle.get_grid_shape(params_.grid_tiled_shape);
        dim3               block = dim3(GemmKernel::kThreadCount, 1, 1);
        int                smem_size = int(sizeof(typename GemmKernel::SharedStorage));

        if (smem_size >= (48 << 10)) {
            cudaError_t attr_status = cudaFuncSetAttribute(cutlass::Kernel<GemmKernel>,
                                                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                           smem_size);
            if (attr_status != cudaSuccess) {
                return cutlass::Status::kErrorInternal;
            }
        }

        cutlass::arch::synclog_setup();
        cutlass::Kernel<GemmKernel><<<grid, block, smem_size, stream>>>(params_);
        return cudaGetLastError() == cudaSuccess ? cutlass::Status::kSuccess : cutlass::Status::kErrorInternal;
    }

    cutlass::Status operator()(cudaStream_t stream = nullptr) { return run(stream); }

private:
    Params params_{};
};

} // namespace

int main(int argc, char **argv)
{
    using InputElement  = cutlass::half_t;
    using OutputElement = float;
    using Layout        = cutlass::layout::RowMajor;

    using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 64>;
    using WarpShape        = cutlass::gemm::GemmShape<32, 32, 64>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
    using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
    using EpilogueOp       = cutlass::epilogue::thread::LinearCombination<OutputElement, 4, float, float>;
    using Operator         = cutlass::arch::OpMultiplyAdd;
    using MmaCore          = cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape,
                                                                        WarpShape,
                                                                        InstructionShape,
                                                                        InputElement,
                                                                        Layout,
                                                                        InputElement,
                                                                        Layout,
                                                                        float,
                                                                        Layout,
                                                                        cutlass::arch::OpClassTensorOp,
                                                                        1,
                                                                        Operator,
                                                                        false,
                                                                        cutlass::arch::CacheOperation::Always,
                                                                        cutlass::arch::CacheOperation::Always>;
    using IteratorA        = cutlass::transform::threadblock::PredicatedTileIterator<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        InputElement,
        Layout,
        1,
        typename MmaCore::IteratorThreadMapA,
        8>;
    using IteratorB        = cutlass::transform::threadblock::PredicatedTileIterator<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        InputElement,
        Layout,
        0,
        typename MmaCore::IteratorThreadMapB,
        8>;
    using SmemIteratorA    = cutlass::transform::threadblock::RegularTileIterator<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        InputElement,
        typename MmaCore::SmemLayoutA,
        0,
        typename MmaCore::IteratorThreadMapA>;
    using SmemIteratorB    = cutlass::transform::threadblock::RegularTileIterator<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        InputElement,
        typename MmaCore::SmemLayoutB,
        0,
        typename MmaCore::IteratorThreadMapB>;
    using Mma              = cutlass::gemm::threadblock::MmaSingleStage<typename MmaCore::Shape,
                                                                        IteratorA,
                                                                        SmemIteratorA,
                                                                        IteratorB,
                                                                        SmemIteratorB,
                                                                        float,
                                                                        Layout,
                                                                        typename MmaCore::MmaPolicy>;
    static int const kPartitionsK = ThreadblockShape::kK / WarpShape::kK;
    using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<ThreadblockShape,
                                                                                      typename Mma::Operator,
                                                                                      kPartitionsK,
                                                                                      EpilogueOp,
                                                                                      EpilogueOp::kCount>::Epilogue;
    using GemmKernel = cutlass::gemm::kernel::Gemm<Mma, Epilogue, ThreadblockSwizzle, false>;
    using CutlassGemm = KernelLevelGemmRunner<GemmKernel, ThreadblockShape, ThreadblockSwizzle>;

    autopartition_bench::GemmOptions options = autopartition_bench::parse_options(argc, argv);
    int padded_m = autopartition_bench::round_up(options.m, 64);
    int padded_n = autopartition_bench::round_up(options.n, 64);
    int padded_k = autopartition_bench::round_up(options.k, 64);

    autopartition_bench::print_options("SM80 CUTLASS official device::Gemm baseline", options, padded_m, padded_n, padded_k);
    std::cout << "  threadblock = 64x64x64\n"
              << "  warp        = 32x32x64\n"
              << "  instruction = 16x8x16\n"
              << "  stages      = 1\n"
              << "  alignments  = A8 / B8\n";

    std::vector<InputElement>  hA(padded_m * padded_k);
    std::vector<InputElement>  hB(padded_k * padded_n);
    std::vector<OutputElement> hC(padded_m * padded_n, OutputElement{});
    std::vector<float>         hRef(options.m * options.n);
    {
        autopartition_bench::NvtxRange range("input-initialize");
        autopartition_bench::fill_a_b_padded(hA, hB, options.m, options.n, options.k, padded_m, padded_n, padded_k);
        autopartition_bench::print_hashes(autopartition_bench::vector_hash(hA), autopartition_bench::vector_hash(hB));
        if (options.verify) {
            autopartition_bench::reference_gemm_kn(hA, hB, hRef, options.m, options.n, options.k, padded_n, padded_k);
        }
    }

    InputElement  *dA = nullptr;
    InputElement  *dB = nullptr;
    OutputElement *dC = nullptr;
    {
        autopartition_bench::NvtxRange range("device-setup");
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
    }

    CutlassGemm gemm;
    cutlass::Status status = gemm.initialize({padded_m, padded_n, padded_k},
                                             {dA, padded_k},
                                             {dB, padded_n},
                                             {dC, padded_n},
                                             {dC, padded_n},
                                             {1.0f, 0.0f});
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS can_implement failed: " << cutlassGetStatusString(status) << "\n";
        return 1;
    }

    auto launch = [&]() -> bool {
        cutlass::Status launch_status = gemm();
        if (launch_status != cutlass::Status::kSuccess) {
            std::cerr << "CUTLASS launch failed: " << cutlassGetStatusString(launch_status) << "\n";
            return false;
        }
        return true;
    };

    {
        autopartition_bench::NvtxRange range("correctness-launch");
        if (!launch()) {
            return 1;
        }
        if (!autopartition_bench::check_cuda(cudaDeviceSynchronize(), "CUTLASS SM80 GEMM")) {
            return 1;
        }
    }
    {
        autopartition_bench::NvtxRange range("result-copy");
        if (!autopartition_bench::check_cuda(
                cudaMemcpy(hC.data(), dC, hC.size() * sizeof(OutputElement), cudaMemcpyDeviceToHost), "copy C")) {
            return 1;
        }
    }

    autopartition_bench::OutputStats stats{};
    float                            max_diff = 0.0f;
    {
        autopartition_bench::NvtxRange range("verification");
        stats = autopartition_bench::output_stats_active(hC, options.m, options.n, padded_n);
        max_diff =
            options.verify ? autopartition_bench::max_abs_diff_active(hC, hRef, options.m, options.n, padded_n) : 0.0f;
    }
    float ms = 0.0f;
    if (!autopartition_bench::time_launch_ms(
            launch, options.warmup, options.iterations, ms, "warmup", "timed")) {
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
