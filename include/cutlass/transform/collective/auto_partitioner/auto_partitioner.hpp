#pragma once
#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/gemm/collective/builders/sm90_common.inl>

namespace autopartition
{
template <typename ArchTag,
          typename OpClass,
          typename Element,
          typename GmemStride,
          typename TileShape_MNK,
          int ThreadCount,
          typename Enable = void>
struct AutoPartitioner
{
  static_assert(sizeof(Element) == 0, "[AutoPartitioner] Unsupported parameters! Check ArchTag, OpClass, or Element.");
};
} // namespace autopartition
