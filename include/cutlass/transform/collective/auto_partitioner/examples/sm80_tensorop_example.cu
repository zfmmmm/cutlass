#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <iostream>
#include <type_traits>

#include "auto_partitioner_builder.hpp"

using namespace cute;

// SM80 + OpClassTensorOp 的蓝图示例。
// TensorOp 的完整 mainloop 需要下游 kernel 决定：如何组织 pipeline stages、
// 如何给 swizzled shared layout 填数据、以及 accumulator 如何 cast/store。
// 这里刻意只验证 AutoPartitioner 能为单个输入张量生成 RoleA/RoleB/RoleC
// 所需的 layout/copy/MMA 类型，不把资源管理逻辑塞回生成器。
int main()
{
    constexpr int M           = 64;
    constexpr int N           = 64;
    constexpr int K           = 64;
    constexpr int ThreadCount = 128;

    using Element   = cutlass::half_t;
    using StrideA   = cute::Stride<int64_t, cute::_1>;
    using StrideB   = cute::Stride<cute::_1, int64_t>;
    using StrideC   = cute::Stride<cute::_1, int64_t>;
    using TileShape = cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>;
    using ArchTag   = cutlass::arch::Sm80;
    using OpClass   = cutlass::arch::OpClassTensorOp;

    using PartA =
        typename autopartition::AutoPartitioner<ArchTag, OpClass, Element, StrideA, TileShape, ThreadCount>::RoleA;
    using PartB =
        typename autopartition::AutoPartitioner<ArchTag, OpClass, Element, StrideB, TileShape, ThreadCount>::RoleB;
    using PartC =
        typename autopartition::AutoPartitioner<ArchTag, OpClass, Element, StrideC, TileShape, ThreadCount>::RoleC;

    using Mma         = typename PartC::TiledMma;
    using Accumulator = typename PartC::Accumulator;

    static_assert(cute::cosize_v<typename PartA::SmemLayout> > 0, "PartA swizzled/padded smem layout must be valid.");
    static_assert(cute::cosize_v<typename PartB::SmemLayout> > 0, "PartB swizzled/padded smem layout must be valid.");
    static_assert(cute::cosize_v<typename PartC::SmemLayout> > 0, "PartC epilogue smem layout must be valid.");
    static_assert(std::is_same<Accumulator, float>::value, "SM80 half TensorOp accumulates in float.");
    static_assert(sizeof(typename PartA::GmemToSmemCopy) > 0, "PartA exposes a global-to-shared copy blueprint.");
    static_assert(sizeof(typename PartB::SmemToRegCopy) > 0, "PartB exposes a shared-to-register copy atom.");
    static_assert(sizeof(Mma) > 0, "RoleC exposes a tiled MMA blueprint.");

    std::cout << "sm80_tensorop_example: blueprint generated for half TensorOp.\n";
    std::cout << "PartA smem elements = " << cute::cosize_v<typename PartA::SmemLayout> << "\n";
    std::cout << "PartB smem elements = " << cute::cosize_v<typename PartB::SmemLayout> << "\n";
    std::cout << "PartC smem elements = " << cute::cosize_v<typename PartC::SmemLayout> << "\n";
    return 0;
}
