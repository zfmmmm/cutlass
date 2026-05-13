#pragma once
#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <type_traits>

namespace autopartition {
// AutoPartitioner 是一个纯模板“图纸生成器”：
// 1. 输入单个张量视角的 ArchTag / OpClass / Element / GmemStride / TileShape。
// 2. 通过外部偏特化输出 RoleA、RoleB、RoleC 三组布局与 copy/MMA atom 类型。
// 3. 不在这里计算 pipeline stages，不分配 shared memory，也不做 local_tile 坐标变换。
//
// 主模板保持空壳，并保留 Enable = void。所有真实逻辑必须写在架构 policy 文件中的
// std::enable_if_t 偏特化里，这样 NVCC 在模板匹配阶段就能精确路由到 SM80/SM100、
// SIMT/TensorOp 以及具体数据类型。
template <typename ArchTag,
          typename OpClass,
          typename Element,
          typename GmemStride,
          typename TileShape_MNK,
          int ThreadCount,
          typename Enable = void>
struct AutoPartitioner
{
    static_assert(sizeof(Element) == 0,
                  "[AutoPartitioner] Unsupported parameters! Check ArchTag, OpClass, or Element.");
};
} // namespace autopartition
