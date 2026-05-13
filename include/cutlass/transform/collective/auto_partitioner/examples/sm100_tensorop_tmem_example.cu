#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <iostream>
#include <type_traits>

#include "auto_partitioner_builder.hpp"

using namespace cute;

// SM100/SM120 + OpClassTensorOp 的 TMEM/UMMA 蓝图示例。
// Blackwell UMMA 的 accumulator 位于 Tensor Memory(TMEM)，A/B operand
// 通过 UMMA shared descriptor 消费 shared memory。AutoPartitioner 只生成
// descriptor-compatible 的 smem layout、TMA copy opcode 和 TiledMma；
// TMEM allocator、mbarrier、producer/consumer pipeline 必须由下游 mainloop 实现。
int main()
{
    constexpr int M           = 64;
    constexpr int N           = 128;
    constexpr int K           = 64;
    constexpr int ThreadCount = 128;

    using Element   = cutlass::half_t;
    using StrideA   = cute::Stride<cute::_1, int64_t>;
    using StrideB   = cute::Stride<int64_t, cute::_1>;
    using StrideC   = cute::Stride<cute::_1, int64_t>;
    using TileShape = cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>;

    using PartA100 = typename autopartition::AutoPartitioner<cutlass::arch::Sm100,
                                                             cutlass::arch::OpClassTensorOp,
                                                             Element,
                                                             StrideA,
                                                             TileShape,
                                                             ThreadCount>::RoleA;
    using PartB100 = typename autopartition::AutoPartitioner<cutlass::arch::Sm100,
                                                             cutlass::arch::OpClassTensorOp,
                                                             Element,
                                                             StrideB,
                                                             TileShape,
                                                             ThreadCount>::RoleB;
    using PartC100 = typename autopartition::AutoPartitioner<cutlass::arch::Sm100,
                                                             cutlass::arch::OpClassTensorOp,
                                                             Element,
                                                             StrideC,
                                                             TileShape,
                                                             ThreadCount>::RoleC;

    using PartA120 = typename autopartition::AutoPartitioner<cutlass::arch::Sm120,
                                                             cutlass::arch::OpClassTensorOp,
                                                             Element,
                                                             StrideA,
                                                             TileShape,
                                                             ThreadCount>::RoleA;
    using PartB120 = typename autopartition::AutoPartitioner<cutlass::arch::Sm120,
                                                             cutlass::arch::OpClassTensorOp,
                                                             Element,
                                                             StrideB,
                                                             TileShape,
                                                             ThreadCount>::RoleB;
    using PartC120 = typename autopartition::AutoPartitioner<cutlass::arch::Sm120,
                                                             cutlass::arch::OpClassTensorOp,
                                                             Element,
                                                             StrideC,
                                                             TileShape,
                                                             ThreadCount>::RoleC;

    using Mma100 = typename PartC100::template TiledMmaFor<PartA100::Major, PartB100::Major>;
    using Mma120 = typename PartC120::template TiledMmaFor<PartA120::Major, PartB120::Major>;

    static_assert(std::is_same<typename PartA100::GmemToSmemCopy, cute::SM90_TMA_LOAD>::value,
                  "SM100 TensorOp uses TMA load as the global-to-shared blueprint.");
    static_assert(std::is_same<typename PartB100::SmemToGmemCopy, cute::SM90_TMA_STORE>::value,
                  "SM100 TensorOp exposes TMA store as the shared-to-global blueprint.");
    static_assert(std::is_same<typename PartA100::SmemToRegCopy, void>::value,
                  "UMMA consumes smem descriptors/TMEM, not normal per-thread smem-to-register operands.");
    static_assert(cute::is_base_of<cute::UMMA::tmem_frg_base, typename Mma100::FrgTypeC>::value,
                  "SM100 UMMA accumulator fragment must live in Tensor Memory.");
    static_assert(cute::is_base_of<cute::UMMA::tmem_frg_base, typename Mma120::FrgTypeC>::value,
                  "SM120 build path must preserve the UMMA/TMEM accumulator fragment.");
    static_assert(cute::cosize_v<typename PartA120::SmemLayout> > 0, "SM120 PartA smem layout must be valid.");
    static_assert(cute::cosize_v<typename PartB120::SmemLayout> > 0, "SM120 PartB smem layout must be valid.");

    std::cout << "sm100_tensorop_tmem_example: SM100 and SM120 UMMA/TMEM blueprints generated.\n";
    std::cout << "PartA100 smem elements = " << cute::cosize_v<typename PartA100::SmemLayout> << "\n";
    std::cout << "PartB100 smem elements = " << cute::cosize_v<typename PartB100::SmemLayout> << "\n";
    std::cout << "PartA120 smem elements = " << cute::cosize_v<typename PartA120::SmemLayout> << "\n";
    std::cout << "PartB120 smem elements = " << cute::cosize_v<typename PartB120::SmemLayout> << "\n";
    return 0;
}
