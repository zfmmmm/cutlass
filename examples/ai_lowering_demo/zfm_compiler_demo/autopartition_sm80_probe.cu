// Instantiates the repository AutoPartitioner and prints a machine-readable SM80 plan.

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <type_traits>

#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/numeric_types.h>
#include <cutlass/transform/collective/auto_partitioner/auto_partitioner_builder.hpp>

using namespace cute;

namespace {

struct Options
{
    int         m         = 1024;
    int         n         = 1024;
    int         k         = 1024;
    int         threads   = 128;
    int         align_a   = 16;
    int         align_b   = 16;
    int         align_out = 16;
    char const *epilogue  = "bias_scale_gelu";
    char const *target_sm = "sm80";
    char const *tile      = "64x64x64";
};

bool parse_int_arg(char const *arg, char const *prefix, int &value)
{
    auto n = std::strlen(prefix);
    if (std::strncmp(arg, prefix, n) != 0) {
        return false;
    }
    value = std::atoi(arg + n);
    return true;
}

bool parse_string_arg(char const *arg, char const *prefix, char const *&value)
{
    auto n = std::strlen(prefix);
    if (std::strncmp(arg, prefix, n) != 0) {
        return false;
    }
    value = arg + n;
    return true;
}

Options parse_options(int argc, char **argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        if (parse_int_arg(argv[i], "--m=", options.m) || parse_int_arg(argv[i], "--n=", options.n)
            || parse_int_arg(argv[i], "--k=", options.k) || parse_int_arg(argv[i], "--threads=", options.threads)
            || parse_int_arg(argv[i], "--align-a=", options.align_a)
            || parse_int_arg(argv[i], "--align-b=", options.align_b)
            || parse_int_arg(argv[i], "--align-out=", options.align_out)
            || parse_string_arg(argv[i], "--epilogue=", options.epilogue)
            || parse_string_arg(argv[i], "--target-sm=", options.target_sm)
            || parse_string_arg(argv[i], "--tile=", options.tile)) {
            continue;
        }
        std::cerr << "unknown argument: " << argv[i] << "\n";
        std::exit(2);
    }
    return options;
}

bool supported_options(Options const &options)
{
    return std::strcmp(options.target_sm, "sm80") == 0 && std::strcmp(options.tile, "64x64x64") == 0
        && options.threads == 128 && options.align_a == 16 && options.align_b == 16 && options.align_out == 16
        && std::strcmp(options.epilogue, "bias_scale_gelu") == 0;
}

int round_up(int value, int multiple)
{
    return ((value + multiple - 1) / multiple) * multiple;
}

} // namespace

int main(int argc, char **argv)
{
    Options options = parse_options(argc, argv);
    if (!supported_options(options)) {
        std::cerr << "autopartition_sm80_probe supports only sm80 tensorop fp16, tile=64x64x64, threads=128, "
                     "16-byte alignments, epilogue=bias_scale_gelu\n";
        return 2;
    }

    constexpr int ThreadCount = 128;
    using InputElement        = cutlass::half_t;
    using OutputElement       = cutlass::half_t;
    using TileShape           = Shape<Int<64>, Int<64>, Int<64>>;

    // A is viewed as [M, K] row-major. B follows the production GEMM convention:
    // the original row-major [K, N] tensor is planned through an [N, K] CuTe view.
    using StrideA = decltype(make_stride(int{}, Int<1>{}));
    using StrideB = decltype(make_stride(Int<1>{}, int{}));
    using StrideC = decltype(make_stride(int{}, Int<1>{}));

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

    static_assert(PartA::UseLdMatrix && PartB::UseLdMatrix, "SM80 tensorop fp16 path should use ldmatrix.");
    static_assert(!std::is_void<typename PartA::GlobalToSharedCopy>::value, "RoleA must expose gmem-to-smem copy.");
    static_assert(!std::is_void<typename PartB::GlobalToSharedCopy>::value, "RoleB must expose gmem-to-smem copy.");
    static_assert(!std::is_void<typename PartC::TiledMma>::value, "RoleC must expose tiled MMA.");
    static_assert(!std::is_void<typename PartC::OutputRegisterToGlobalCopy>::value,
                  "RoleC must expose output register-to-global copy.");
    static_assert(PartC::HasZeroGlueEpilogueMapping, "RoleC must expose zero-glue epilogue mapping.");
    static_assert(PartC::HasFusionSharedMapping, "RoleC must expose fusion shared-memory mapping.");
    static_assert(std::is_same<typename PartC::EpilogueElement, typename PartC::ElementCompute>::value,
                  "Epilogue shared path must store accumulator elements.");

    int padded_m = round_up(options.m, int(size<0>(TileShape{})));
    int padded_n = round_up(options.n, int(size<1>(TileShape{})));
    int padded_k = round_up(options.k, int(size<2>(TileShape{})));

    std::cout << std::boolalpha;
    std::cout << "source=autopartitioner_probe\n";
    std::cout << "target_sm=sm80\n";
    std::cout << "op_class=tensorop\n";
    std::cout << "input_dtype=fp16\n";
    std::cout << "output_dtype=fp16\n";
    std::cout << "acc_dtype=fp32\n";
    std::cout << "logical_m=" << options.m << "\n";
    std::cout << "logical_n=" << options.n << "\n";
    std::cout << "logical_k=" << options.k << "\n";
    std::cout << "padded_m=" << padded_m << "\n";
    std::cout << "padded_n=" << padded_n << "\n";
    std::cout << "padded_k=" << padded_k << "\n";
    std::cout << "tile_m=64\n";
    std::cout << "tile_n=64\n";
    std::cout << "tile_k=64\n";
    std::cout << "thread_count=" << ThreadCount << "\n";

    std::cout << "role_a_use_ldmatrix=" << PartA::UseLdMatrix << "\n";
    std::cout << "role_b_use_ldmatrix=" << PartB::UseLdMatrix << "\n";
    std::cout << "role_a_swizzle_base=" << PartA::SwizzleBase << "\n";
    std::cout << "role_b_swizzle_base=" << PartB::SwizzleBase << "\n";
    std::cout << "role_a_smem_cosize=" << cute::cosize_v<typename PartA::SmemLayout> << "\n";
    std::cout << "role_b_smem_cosize=" << cute::cosize_v<typename PartB::SmemLayout> << "\n";

    std::cout << "role_c_warp_m=" << PartC::WarpM << "\n";
    std::cout << "role_c_warp_n=" << PartC::WarpN << "\n";
    std::cout << "role_c_repeat_m=" << PartC::RepeatM << "\n";
    std::cout << "role_c_repeat_n=" << PartC::RepeatN << "\n";
    std::cout << "role_c_epilogue_instruction_threads=" << PartC::EpilogueInstructionThreads << "\n";
    std::cout << "role_c_epilogue_vector_bytes=" << PartC::EpilogueVectorBytes << "\n";
    std::cout << "role_c_output_alignment_bytes=" << PartC::OutputAlignmentBytes << "\n";
    std::cout << "role_c_epilogue_layout_candidate_count=" << PartC::EpilogueLayoutCandidateCount << "\n";
    std::cout << "role_c_epilogue_swizzle_base=" << PartC::EpilogueSwizzleBase << "\n";
    std::cout << "role_c_epilogue_swizzle_mbase=" << PartC::EpilogueSwizzleMBase << "\n";
    std::cout << "role_c_epilogue_swizzle_shift=" << PartC::EpilogueSwizzleShift << "\n";
    std::cout << "role_c_epilogue_swizzle_row_elements=" << PartC::EpilogueSwizzleRowElements << "\n";
    std::cout << "role_c_epilogue_swizzle_minor_elements=" << PartC::EpilogueSwizzleMinorElements << "\n";
    std::cout << "role_c_epilogue_bank_conflict_score=" << PartC::EpilogueBankConflictScore << "\n";
    std::cout << "role_c_epilogue_naive_bank_conflict_score=" << PartC::EpilogueNaiveBankConflictScore << "\n";
    std::cout << "role_c_epilogue_vector_alignment_penalty=" << PartC::EpilogueVectorAlignmentPenalty << "\n";
    std::cout << "role_c_has_zero_glue_epilogue_mapping=" << PartC::HasZeroGlueEpilogueMapping << "\n";
    std::cout << "role_c_has_fusion_shared_mapping=" << PartC::HasFusionSharedMapping << "\n";
    std::cout << "role_c_has_register_reuse_mapping=" << PartC::HasRegisterReuseMapping << "\n";
    std::cout << "role_c_has_register_shuffle_mapping=" << PartC::HasRegisterShuffleMapping << "\n";
    return 0;
}
