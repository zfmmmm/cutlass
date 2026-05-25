// clang-format off
#include <cstdio>
#include <iostream>
#include <type_traits>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/half.h>
#include <cutlass/numeric_conversion.h>

#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/numeric/integral_constant.hpp>

#include "autopartition_production_common.hpp"
#include "cutlass/transform/collective/auto_partitioner/auto_partitioner_builder.hpp"
// clang-format on

namespace autopartition_sm100_production {

using Element             = cutlass::half_t;
using OutputElement       = float;
using StrideA             = cute::Stride<int, cute::_1>;
using StrideB             = cute::Stride<int, cute::_1>;
using StrideC             = cute::Stride<int, cute::_1>;
using TileShape           = cute::Shape<cute::Int<128>, cute::Int<256>, cute::Int<64>>;
using ClusterShape        = cute::Shape<cute::_1, cute::_1, cute::_1>;
constexpr int ThreadCount = 128;

using PartA    = typename autopartition::AutoPartitioner<cutlass::arch::Sm100,
                                                         cutlass::arch::OpClassTensorOp,
                                                         Element,
                                                         StrideA,
                                                         TileShape,
                                                         ThreadCount,
                                                         OutputElement,
                                                         16,
                                                         16,
                                                         16,
                                                         ClusterShape>::RoleA;
using PartB    = typename autopartition::AutoPartitioner<cutlass::arch::Sm100,
                                                         cutlass::arch::OpClassTensorOp,
                                                         Element,
                                                         StrideB,
                                                         TileShape,
                                                         ThreadCount,
                                                         OutputElement,
                                                         16,
                                                         16,
                                                         16,
                                                         ClusterShape>::RoleB;
using PartC    = typename autopartition::AutoPartitioner<cutlass::arch::Sm100,
                                                         cutlass::arch::OpClassTensorOp,
                                                         Element,
                                                         StrideC,
                                                         TileShape,
                                                         ThreadCount,
                                                         OutputElement,
                                                         16,
                                                         16,
                                                         16,
                                                         ClusterShape>::RoleC;
using TiledMma = typename PartC::template TiledMmaFor<PartA::Major, PartB::Major>;

static_assert(PartA::UsesTmaLoad && PartB::UsesTmaLoad, "A/B must use AutoPartitioner-selected TMA loads.");
static_assert(std::is_void<typename PartA::SmemToRegCopy>::value && std::is_void<typename PartB::SmemToRegCopy>::value,
              "SM100 UMMA consumes SMEM descriptors directly.");
static_assert(cute::is_base_of<cute::UMMA::tmem_frg_base, typename TiledMma::FrgTypeC>::value,
              "SM100 accumulator must be backed by TMEM.");
static_assert(!std::is_void<typename PartC::TmemToRegisterCopyOperation>::value,
              "RoleC must expose the builder-selected TMEM-to-register copy.");
static_assert(!std::is_void<typename PartC::RegisterToSharedCopyOperation>::value,
              "RoleC must expose the builder-selected register-to-shared copy.");
static_assert(!std::is_void<typename PartC::SharedToGlobalCopyOperation>::value,
              "RoleC must expose the builder-selected shared-to-global copy.");
static_assert(cute::cosize_v<typename PartC::SharedToGlobalLayout> > 0,
              "SM100 RoleC must expose a concrete output shared-memory layout.");

template <class TypeA, class TypeB, class TypeD, class ASmemLayout, class BSmemLayout, class DSmemLayout>
struct SharedStorage
{
    alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> A;
    alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout>> B;
    alignas(128) cute::ArrayEngine<TypeD, cute::cosize_v<DSmemLayout>> D;

    alignas(16) cute::uint64_t mma_barrier;
    alignas(16) cute::uint64_t tma_barrier;
    alignas(16) cute::uint32_t tmem_base_ptr;

    CUTE_DEVICE constexpr auto tensor_sA() { return cute::make_tensor(cute::make_smem_ptr(A.begin()), ASmemLayout{}); }
    CUTE_DEVICE constexpr auto tensor_sB() { return cute::make_tensor(cute::make_smem_ptr(B.begin()), BSmemLayout{}); }
    CUTE_DEVICE constexpr auto tensor_sD() { return cute::make_tensor(cute::make_smem_ptr(D.begin()), DSmemLayout{}); }
};

template <class SharedStorage,
          class ATensor,
          class BTensor,
          class DTensor,
          class MmaTiler,
          class TiledMMA,
          class TmaAtomA,
          class TmaAtomB,
          class TmaAtomD,
          class Alpha>
__global__ void autopartition_sm100_tma_umma_kernel(ATensor                           mA,
                                                    BTensor                           mB,
                                                    DTensor                           mD,
                                                    MmaTiler                          mma_tiler,
                                                    TiledMMA                          tiled_mma,
                                                    CUTE_GRID_CONSTANT TmaAtomA const tma_atom_A,
                                                    CUTE_GRID_CONSTANT TmaAtomB const tma_atom_B,
                                                    CUTE_GRID_CONSTANT TmaAtomD const tma_store_D,
                                                    Alpha                             alpha)
{
    using namespace cute;

    auto   mma_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA        = local_tile(mA, mma_tiler, mma_coord, Step<_1, X, _1>{});
    Tensor gB        = local_tile(mB, mma_tiler, mma_coord, Step<X, _1, _1>{});
    Tensor gD        = local_tile(mD, mma_tiler, mma_coord, Step<_1, _1, X>{});

    extern __shared__ char shared_memory[];
    SharedStorage         &shared_storage = *reinterpret_cast<SharedStorage *>(shared_memory);
    Tensor                 tCsA           = shared_storage.tensor_sA();
    Tensor                 tCsB           = shared_storage.tensor_sB();
    Tensor                 sD             = as_position_independent_swizzle_tensor(shared_storage.tensor_sD());

    ThrMMA cta_mma = tiled_mma.get_slice(Int<0>{});
    Tensor tCgA    = cta_mma.partition_A(gA);
    Tensor tCgB    = cta_mma.partition_B(gB);
    Tensor tCgD    = cta_mma.partition_C(gD);
    Tensor tCrA    = cta_mma.make_fragment_A(tCsA);
    Tensor tCrB    = cta_mma.make_fragment_B(tCsB);
    Tensor tCtAcc  = cta_mma.make_fragment_C(tCgD);

    uint32_t elect_one_thr  = cute::elect_one_sync();
    uint32_t elect_one_warp = (threadIdx.x / 32 == 0);

    using TmemAllocator = cute::conditional_t<cute::size(cute::shape<0>(typename TiledMMA::ThrLayoutVMNK{})) == 1,
                                              cute::TMEM::Allocator1Sm,
                                              cute::TMEM::Allocator2Sm>;
    TmemAllocator tmem_allocator{};
    if (elect_one_warp) {
        tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);
    }
    __syncthreads();
    tCtAcc.data() = shared_storage.tmem_base_ptr;

    auto [tAgA, tAsA] =
        tma_partition(tma_atom_A, Int<0>{}, Layout<_1>{}, group_modes<0, 3>(tCsA), group_modes<0, 3>(tCgA));
    auto [tBgB, tBsB] =
        tma_partition(tma_atom_B, Int<0>{}, Layout<_1>{}, group_modes<0, 3>(tCsB), group_modes<0, 3>(tCgB));

    int tma_transaction_bytes = sizeof(make_tensor_like(tAsA)) + sizeof(make_tensor_like(tBsB));
    if (elect_one_warp && elect_one_thr) {
        cute::initialize_barrier(shared_storage.mma_barrier, 1);
        cute::initialize_barrier(shared_storage.tma_barrier, 1);
    }
    int mma_barrier_phase_bit = 0;
    int tma_barrier_phase_bit = 0;
    __syncthreads();

    tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
    for (int k_tile = 0; k_tile < size<3>(tCgA); ++k_tile) {
        if (elect_one_warp && elect_one_thr) {
            cute::set_barrier_transaction_bytes(shared_storage.tma_barrier, tma_transaction_bytes);
            copy(tma_atom_A.with(shared_storage.tma_barrier), tAgA(_, k_tile), tAsA);
            copy(tma_atom_B.with(shared_storage.tma_barrier), tBgB(_, k_tile), tBsB);
        }

        cute::wait_barrier(shared_storage.tma_barrier, tma_barrier_phase_bit);
        tma_barrier_phase_bit ^= 1;

        if (elect_one_warp) {
            for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
                gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCtAcc);
                tiled_mma.accumulate_ = UMMA::ScaleOut::One;
            }
            cutlass::arch::umma_arrive(&shared_storage.mma_barrier);
        }
        cute::wait_barrier(shared_storage.mma_barrier, mma_barrier_phase_bit);
        mma_barrier_phase_bit ^= 1;
    }

    Tensor tAcc     = tCtAcc(make_coord(_, _), _0{}, _0{});
    Tensor tAcc_epi = flat_divide(tAcc, typename PartC::EpilogueTile{});
    Tensor gD_epi   = flat_divide(gD, typename PartC::EpilogueTile{});

    TiledCopy tiled_t2r_copy =
        make_tmem_copy(typename PartC::TmemToRegisterCopyOperation{}, tAcc_epi(_, _, _0{}, _0{}));
    ThrCopy thr_t2r_copy = tiled_t2r_copy.get_slice(threadIdx.x);
    Tensor  tTR_tAcc     = thr_t2r_copy.partition_S(tAcc_epi);
    Tensor  tTR_sD       = thr_t2r_copy.partition_D(sD);

    using ComputeElement = typename PartC::ElementCompute;
    Tensor tTR_rAcc      = make_tensor<ComputeElement>(shape(tTR_sD));
    Tensor tTR_rD        = make_tensor<OutputElement>(shape(tTR_sD));

    TiledCopy tiled_r2s_copy =
        make_tiled_copy_D(Copy_Atom<typename PartC::RegisterToSharedCopyOperation, OutputElement>{}, tiled_t2r_copy);
    ThrCopy thr_r2s_copy = tiled_r2s_copy.get_slice(threadIdx.x);
    Tensor  tRS_rD       = thr_r2s_copy.retile_S(tTR_rD);
    Tensor  tRS_sD       = thr_r2s_copy.partition_D(sD);

    ThrCopy thrblk_s2g_copy = tma_store_D.get_slice(Int<0>{});
    Tensor  bSG_sD          = thrblk_s2g_copy.partition_S(sD);
    Tensor  bSG_gD          = thrblk_s2g_copy.partition_D(gD_epi);

    cutlass::NumericConverter<OutputElement, ComputeElement> convert;
    Layout tmem_warp_layout =
        typename decltype(make_tmem_warp_partitioner(tAcc_epi(_, _, _0{}, _0{})))::TiledLayout_TV{};
    constexpr bool predicate_tmem_load = size(tmem_warp_layout) != cosize(tmem_warp_layout);
    int            warp_idx            = threadIdx.x / cutlass::NumThreadsPerWarp;

    constexpr int NumEpiSubtilesN = CUTE_STATIC_V(size<3>(gD_epi));
    constexpr int NumEpiSubtilesM = CUTE_STATIC_V(size<2>(gD_epi));
#pragma unroll
    for (int epi_n = 0; epi_n < NumEpiSubtilesN; ++epi_n) {
#pragma unroll
        for (int epi_m = 0; epi_m < NumEpiSubtilesM; ++epi_m) {
            Tensor tTR_tAcc_mn  = tTR_tAcc(_, _, _, epi_m, epi_n);
            bool   issue_t2r    = true;
            if constexpr (predicate_tmem_load) {
                int subpart_idx = (tTR_tAcc_mn.data().dp_ / 32) % 4;
                issue_t2r       = warp_idx == subpart_idx;
            }

            if (issue_t2r) {
                copy(tiled_t2r_copy, tTR_tAcc_mn, tTR_rAcc);
                CUTE_UNROLL
                for (int i = 0; i < size(tTR_rAcc); ++i) {
                    tTR_rD(i) = convert(alpha * tTR_rAcc(i));
                }
                copy(tiled_r2s_copy, tRS_rD, tRS_sD);
            }

            tma_store_fence();
            __syncthreads();
            if (elect_one_thr) {
                copy(tma_store_D, bSG_sD, bSG_gD(_, _, _, epi_m, epi_n));
                tma_store_arrive();
                tma_store_wait<0>();
            }
            __syncthreads();
        }
    }

    __syncthreads();
    if (elect_one_warp) {
        tmem_allocator.release_allocation_lock();
        tmem_allocator.free(shared_storage.tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
    }
}

inline bool is_sm100a_device()
{
    cudaDeviceProp props{};
    cudaError_t    error = cudaGetDeviceProperties(&props, 0);
    if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << "\n";
        return false;
    }
    if (props.major != 10 || props.minor != 0) {
        std::cerr << "This example requires an SM100/100a Blackwell datacenter GPU. Found " << props.major << "."
                  << props.minor << "\n";
        return false;
    }
    return true;
}

inline autopartition::examples::production::GemmOptions parse_sm100_options(int argc, char **argv)
{
    using namespace autopartition::examples::production;
    GemmOptions options;
    options.m          = 512;
    options.n          = 1024;
    options.k          = 256;
    options.iterations = 5;
    options.warmup     = 2;
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

inline void
fill_inputs(thrust::host_vector<Element> &a, thrust::host_vector<Element> &b, int m, int n, int k, int padded_k)
{
    for (int row = 0; row < m; ++row) {
        for (int kk = 0; kk < k; ++kk) {
            a[row * padded_k + kk] =
                autopartition::examples::production::from_float<Element>(((row * 13 + kk * 7) % 17 - 8) * 0.03125f);
        }
    }
    for (int row = 0; row < n; ++row) {
        for (int kk = 0; kk < k; ++kk) {
            b[row * padded_k + kk] =
                autopartition::examples::production::from_float<Element>(((row * 5 + kk * 11) % 19 - 9) * 0.03125f);
        }
    }
}

inline void reference_gemm(thrust::host_vector<Element> const &a,
                           thrust::host_vector<Element> const &b,
                           thrust::host_vector<OutputElement> &d,
                           int                                 m,
                           int                                 n,
                           int                                 k,
                           int                                 padded_k)
{
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float acc = 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                acc += autopartition::examples::production::to_float(a[row * padded_k + kk])
                     * autopartition::examples::production::to_float(b[col * padded_k + kk]);
            }
            d[row * n + col] = acc;
        }
    }
}

inline float max_abs_diff(thrust::host_vector<OutputElement> const &actual,
                          thrust::host_vector<OutputElement> const &reference,
                          int                                       m,
                          int                                       n,
                          int                                       padded_n)
{
    float diff = 0.0f;
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            diff = std::max(diff, std::abs(actual[row * padded_n + col] - reference[row * n + col]));
        }
    }
    return diff;
}

} // namespace autopartition_sm100_production

int main(int argc, char **argv)
{
    using namespace cute;
    using namespace autopartition_sm100_production;
    using namespace autopartition::examples::production;

    GemmOptions   options  = parse_sm100_options(argc, argv);
    constexpr int bM       = cute::size<0>(TileShape{});
    constexpr int bN       = cute::size<1>(TileShape{});
    constexpr int bK       = cute::size<2>(TileShape{});
    int           padded_m = round_up(options.m, bM);
    int           padded_n = round_up(options.n, bN);
    int           padded_k = round_up(options.k, bK);
    print_options("SM100 AutoPartitioner-owned TMA/UMMA GEMM", options, padded_m, padded_n, padded_k);
    print_ncu_hint("./sm100_autopartition_tma_umma_gemm");

    if (!is_sm100a_device()) {
        return 255;
    }

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    auto layout_A = make_layout(make_shape(padded_m, padded_k), make_stride(padded_k, Int<1>{}));
    auto layout_B = make_layout(make_shape(padded_n, padded_k), make_stride(padded_k, Int<1>{}));
    auto layout_D = make_layout(make_shape(padded_m, padded_n), make_stride(padded_n, Int<1>{}));

    thrust::host_vector<Element>       host_A(padded_m * padded_k);
    thrust::host_vector<Element>       host_B(padded_n * padded_k);
    thrust::host_vector<OutputElement> host_D(padded_m * padded_n);
    thrust::host_vector<OutputElement> host_ref(options.m * options.n);
    fill_inputs(host_A, host_B, options.m, options.n, options.k, padded_k);

    thrust::device_vector<Element>       device_A = host_A;
    thrust::device_vector<Element>       device_B = host_B;
    thrust::device_vector<OutputElement> device_D(padded_m * padded_n);

    Tensor mA = make_tensor(make_gmem_ptr(device_A.data().get()), layout_A);
    Tensor mB = make_tensor(make_gmem_ptr(device_B.data().get()), layout_B);
    Tensor mD = make_tensor(make_gmem_ptr(device_D.data().get()), layout_D);

    TiledMma tiled_mma;
    auto     mma_tiler           = TileShape{};
    auto     cluster_layout_vmnk = tiled_divide(make_layout(ClusterShape{}), make_tile(typename TiledMma::AtomThrID{}));
    auto     tma_atom_A          = make_tma_atom_A_sm100(typename PartA::GmemTiledCopyTmaOperation{},
                                            mA,
                                            typename PartA::SmemLayout{},
                                            mma_tiler,
                                            tiled_mma,
                                            cluster_layout_vmnk);
    auto     tma_atom_B          = make_tma_atom_B_sm100(typename PartB::GmemTiledCopyTmaOperation{},
                                            mB,
                                            typename PartB::SmemLayout{},
                                            mma_tiler,
                                            tiled_mma,
                                            cluster_layout_vmnk);
    auto     tma_store_D         = make_tma_copy(typename PartC::SharedToGlobalCopyOperation{},
                                        mD,
                                        typename PartC::SharedToGlobalLayout{},
                                        typename PartC::EpilogueTile{},
                                        _1{});
    Tensor   mA_tma              = tma_atom_A.get_tma_tensor(shape(mA));
    Tensor   mB_tma              = tma_atom_B.get_tma_tensor(shape(mB));
    Tensor   mD_tma              = tma_store_D.get_tma_tensor(shape(mD));

    using Storage = SharedStorage<Element,
                                  Element,
                                  OutputElement,
                                  typename PartA::SmemLayout,
                                  typename PartB::SmemLayout,
                                  typename PartC::SharedToGlobalLayout>;
    auto *kernel     = &autopartition_sm100_tma_umma_kernel<Storage,
                                                            decltype(mA_tma),
                                                            decltype(mB_tma),
                                                            decltype(mD_tma),
                                                            decltype(mma_tiler),
                                                            TiledMma,
                                                            decltype(tma_atom_A),
                                                            decltype(tma_atom_B),
                                                            decltype(tma_store_D),
                                                            float>;
    int   smem_bytes = sizeof(Storage);
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    dim3 block(ThreadCount);
    dim3 grid(padded_m / bM, padded_n / bN);
    auto launch = [&]() {
        kernel<<<grid, block, smem_bytes>>>(
            mA_tma, mB_tma, mD_tma, mma_tiler, tiled_mma, tma_atom_A, tma_atom_B, tma_store_D, 1.0f);
    };

    float runtime_ms = time_launch_ms(launch, options.warmup, options.iterations);
    if (!check_cuda(cudaDeviceSynchronize(), "autopartition_sm100_tma_umma_kernel")) {
        return 1;
    }

    host_D     = device_D;
    float diff = 0.0f;
    if (options.verify) {
        reference_gemm(host_A, host_B, host_ref, options.m, options.n, options.k, padded_k);
        diff = max_abs_diff(host_D, host_ref, options.m, options.n, padded_n);
    }

    std::cout << "  max_abs_error = " << diff << "\n"
              << "  runtime_ms    = " << runtime_ms << "\n"
              << "  tflops        = " << tflops(options.m, options.n, options.k, runtime_ms) << "\n";
    return (!options.verify || diff <= 1.0e-2f) ? 0 : 2;
#else
    std::cout << "CUTLASS_ARCH_MMA_SM100_SUPPORTED must be enabled, but it is not. Test is waived.\n";
    return 0;
#endif
}
