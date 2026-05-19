#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <iostream>
#include <type_traits>

#include "autopartition_production_common.hpp"
#include "cutlass/transform/collective/auto_partitioner/auto_partitioner_builder.hpp"

namespace autopartition_sm100_production {

using Element               = cutlass::half_t;
using OutputElement         = float;
using StrideA               = cute::Stride<int, cute::_1>;
using StrideB               = cute::Stride<int, cute::_1>;
using StrideC               = cute::Stride<int, cute::_1>;
using TileShape             = cute::Shape<cute::Int<128>, cute::Int<256>, cute::Int<64>>;
using ClusterShape          = cute::Shape<cute::_1, cute::_1, cute::_1>;
using ClusterShapeMulticast = cute::Shape<cute::_2, cute::_1, cute::_1>;

using PartA        = typename autopartition::AutoPartitioner<cutlass::arch::Sm100,
                                                             cutlass::arch::OpClassTensorOp,
                                                             Element,
                                                             StrideA,
                                                             TileShape,
                                                             128,
                                                             OutputElement,
                                                             16,
                                                             16,
                                                             16,
                                                             ClusterShape>::RoleA;
using PartB        = typename autopartition::AutoPartitioner<cutlass::arch::Sm100,
                                                             cutlass::arch::OpClassTensorOp,
                                                             Element,
                                                             StrideB,
                                                             TileShape,
                                                             128,
                                                             OutputElement,
                                                             16,
                                                             16,
                                                             16,
                                                             ClusterShape>::RoleB;
using PartC        = typename autopartition::AutoPartitioner<cutlass::arch::Sm100,
                                                             cutlass::arch::OpClassTensorOp,
                                                             Element,
                                                             StrideC,
                                                             TileShape,
                                                             128,
                                                             OutputElement,
                                                             16,
                                                             16,
                                                             16,
                                                             ClusterShape>::RoleC;
using PartACluster = typename autopartition::AutoPartitioner<cutlass::arch::Sm100,
                                                             cutlass::arch::OpClassTensorOp,
                                                             Element,
                                                             StrideA,
                                                             TileShape,
                                                             128,
                                                             OutputElement,
                                                             16,
                                                             16,
                                                             16,
                                                             ClusterShapeMulticast>::RoleA;

static_assert(PartA::UsesTmaLoad && PartB::UsesTmaLoad, "SM100 16B-aligned mainloop must route to TMA load.");
static_assert(std::is_void<typename PartA::SmemToRegCopy>::value && std::is_void<typename PartB::SmemToRegCopy>::value,
              "UMMA consumes SMEM descriptors directly; SmemToRegCopy must be retired.");
static_assert(cute::is_base_of<cute::UMMA::tmem_frg_base, typename PartC::TiledMma::FrgTypeC>::value,
              "SM100 TensorOp accumulator must be backed by TMEM.");
static_assert(!std::is_void<typename PartC::TmemToSmemCopy>::value, "RoleC must expose a TMEM-to-SMEM unload channel.");
static_assert(PartC::UsesTmaStore, "SM100 16B-aligned epilogue must route to TMA store.");
static_assert(cute::size<0>(ClusterShapeMulticast{}) == 2, "ClusterShape is preserved for multicast-capable routing.");

inline bool encode_tma_descriptor_probe(void *ptr, int rows, int cols, int leading_dim_elements)
{
    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        std::cerr << "cuInit failed while probing cuTensorMapEncodeTiled: " << int(result) << "\n";
        return false;
    }

    CUtensorMap tensor_map{};
    cuuint64_t  global_dim[2]     = {cuuint64_t(cols), cuuint64_t(rows)};
    cuuint64_t  global_stride[1]  = {cuuint64_t(leading_dim_elements * int(sizeof(Element)))};
    cuuint32_t  box_dim[2]        = {cuuint32_t(cute::size<2>(TileShape{})), cuuint32_t(cute::size<0>(TileShape{}))};
    cuuint32_t  element_stride[2] = {1, 1};

    result = cuTensorMapEncodeTiled(&tensor_map,
                                    CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
                                    2,
                                    ptr,
                                    global_dim,
                                    global_stride,
                                    box_dim,
                                    element_stride,
                                    CU_TENSOR_MAP_INTERLEAVE_NONE,
                                    CU_TENSOR_MAP_SWIZZLE_128B,
                                    CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (result != CUDA_SUCCESS) {
        std::cerr << "cuTensorMapEncodeTiled probe failed: " << int(result) << "\n";
        return false;
    }
    std::cout << "  cuTensorMapEncodeTiled probe = success, SWIZZLE_128B, box=(" << box_dim[1] << "x" << box_dim[0]
              << ")\n";
    return true;
}

inline void print_autopartitioner_contract()
{
    std::cout << "SM100 AutoPartitioner contract\n"
              << "  PartA UsesTmaLoad = " << PartA::UsesTmaLoad << "\n"
              << "  PartB UsesTmaLoad = " << PartB::UsesTmaLoad << "\n"
              << "  PartC UsesTmaStore = " << PartC::UsesTmaStore << "\n"
              << "  PartA smem cosize = " << cute::cosize(typename PartA::SmemLayout{}) << "\n"
              << "  PartB smem cosize = " << cute::cosize(typename PartB::SmemLayout{}) << "\n"
              << "  Cluster multicast example = <" << cute::size<0>(ClusterShapeMulticast{}) << ","
              << cute::size<1>(ClusterShapeMulticast{}) << "," << cute::size<2>(ClusterShapeMulticast{}) << ">\n";
}

} // namespace autopartition_sm100_production

// Reuse the official CUTLASS/CuTe Blackwell tutorial kernel as the executable body.
// It contains the full device pipeline: SM90_TMA_LOAD, tcgen05 UMMA, TMEM accumulator,
// TMEM -> register epilogue staging, register -> SMEM staging, and SM90_TMA_STORE.
#define print(...) ((void)0)
#define main       cutlass_cute_tutorial_05_sm100_main
#include "examples/cute/tutorial/blackwell/05_mma_tma_epi_sm100.cu"
#undef main
#undef print

namespace autopartition_sm100_production {

inline autopartition::examples::production::GemmOptions parse_sm100_options(int argc, char **argv)
{
    using namespace autopartition::examples::production;
    GemmOptions options;
    options.m          = 512;
    options.n          = 1024;
    options.k          = 256;
    options.iterations = 5;
    options.warmup     = 2;

    int positional = 0;
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
        if (argv[i][0] != '-') {
            int value = std::atoi(argv[i]);
            if (positional == 0) {
                options.m = value;
            }
            else if (positional == 1) {
                options.n = value;
            }
            else if (positional == 2) {
                options.k = value;
            }
            ++positional;
        }
    }
    return options;
}

inline bool is_sm100a_device()
{
    cudaDeviceProp props{};
    cudaError_t    error = cudaGetDeviceProperties(&props, 0);
    if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << "\n";
        return false;
    }
    if ((props.major != 10) || (props.major == 10 && props.minor > 1)) {
        std::cerr << "This example requires NVIDIA's Blackwell Architecture GPU with compute capability 100a.\n";
        std::cerr << "  Found " << props.major << "." << props.minor << "\n";
        return false;
    }
    return true;
}

} // namespace autopartition_sm100_production

int main(int argc, char **argv)
{
    using namespace autopartition_sm100_production;
    using namespace autopartition::examples::production;

    GemmOptions options = parse_sm100_options(argc, argv);
    print_autopartitioner_contract();

    Element *descriptor_probe_ptr = nullptr;
    if (check_cuda(cudaMalloc(&descriptor_probe_ptr, 128 * 64 * sizeof(Element)), "cudaMalloc descriptor probe")) {
        encode_tma_descriptor_probe(descriptor_probe_ptr, 128, 64, 64);
        cudaFree(descriptor_probe_ptr);
    }

    std::cout << "NCU bank-conflict probe:\n"
              << "  ncu --metrics " << "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,"
              << "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum "
              << "./sm100_autopartition_tma_umma_gemm 512 1024 256\n";
    std::cout << std::flush;

    if (!is_sm100a_device()) {
        return 255;
    }

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    using TypeA           = cutlass::half_t;
    using TypeB           = cutlass::half_t;
    using TypeC           = float;
    using TypeD           = float;
    using TypeAccumulator = float;

    Layout layout_A = make_layout(make_shape(options.m, options.k), make_stride(options.k, Int<1>{}));
    Layout layout_B = make_layout(make_shape(options.n, options.k), make_stride(options.k, Int<1>{}));
    Layout layout_C = make_layout(make_shape(options.m, options.n), make_stride(options.n, Int<1>{}));
    Layout layout_D = make_layout(make_shape(options.m, options.n), make_stride(options.n, Int<1>{}));

    print_options("SM100 AutoPartitioner TMA/UMMA/TMEM GEMM", options, options.m, options.n, options.k);

    thrust::host_vector<TypeA> host_A(options.m * options.k);
    thrust::host_vector<TypeB> host_B(options.n * options.k);
    thrust::host_vector<TypeC> host_C(options.m * options.n);
    Tensor                     host_tensor_A = make_tensor(host_A.data(), layout_A);
    Tensor                     host_tensor_B = make_tensor(host_B.data(), layout_B);
    Tensor                     host_tensor_C = make_tensor(host_C.data(), layout_C);
    initialize_tensor(host_tensor_A);
    initialize_tensor(host_tensor_B);
    initialize_tensor(host_tensor_C);

    thrust::device_vector<TypeA> device_A = host_A;
    thrust::device_vector<TypeB> device_B = host_B;
    thrust::device_vector<TypeC> device_C = host_C;
    thrust::device_vector<TypeD> device_D(options.m * options.n);

    float alpha  = 1.0f;
    float beta   = 0.0f;
    auto  launch = [&]() {
        gemm_host_f16xf16_f32_f32_tnt(device_A.data().get(),
                                      layout_A,
                                      device_B.data().get(),
                                      layout_B,
                                      device_C.data().get(),
                                      layout_C,
                                      device_D.data().get(),
                                      layout_D,
                                      alpha,
                                      beta);
    };

    for (int i = 0; i < options.warmup; ++i) {
        launch();
    }
    if (!check_cuda(cudaDeviceSynchronize(), "SM100 warmup")) {
        return 1;
    }

    cudaEvent_t start{};
    cudaEvent_t stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < options.iterations; ++i) {
        launch();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    float runtime_ms = elapsed_ms / float(options.iterations);

    thrust::host_vector<TypeD> host_D        = device_D;
    Tensor                     host_tensor_D = make_tensor(host_D.data(), layout_D);

    double relative_error = 0.0;
    if (options.verify) {
        thrust::host_vector<TypeD> host_reference_D(options.m * options.n);
        Tensor                     host_reference_tensor_D = make_tensor(host_reference_D.data(), layout_D);
        reference_gemm<TypeAccumulator>(
            host_tensor_A, host_tensor_B, host_tensor_C, host_reference_tensor_D, alpha, beta);
        relative_error = print_matrix_multiply_mollified_relative_error(
            "half_t", host_tensor_A, "half_t", host_tensor_B, "float", host_tensor_D, host_reference_tensor_D);
    }

    std::cout << "  relative_error = " << relative_error << "\n"
              << "  runtime_ms     = " << runtime_ms << "\n"
              << "  tflops         = " << tflops(options.m, options.n, options.k, runtime_ms) << "\n";
    return (!options.verify || relative_error <= 1.0e-3) ? 0 : 2;
#else
    std::cout << "CUTLASS_ARCH_MMA_SM100_SUPPORTED must be enabled, but it is not. Test is waived.\n";
    return 0;
#endif
}
