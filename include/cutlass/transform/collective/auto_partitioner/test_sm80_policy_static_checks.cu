#include <type_traits>

#include <cute/tensor.hpp>

#include "auto_partitioner_builder.hpp"

using namespace cute;

namespace {

constexpr int ThreadCount = 128;
using ArchTag = cutlass::arch::Sm80;
using OpClass = cutlass::arch::OpClassTensorOp;
using Element = cutlass::half_t;
using ElementC = cutlass::half_t;
using StrideA = cute::Stride<int64_t, cute::_1>;
using StrideB = cute::Stride<cute::_1, int64_t>;
using StrideC = cute::Stride<int64_t, cute::_1>;
using TileShape64 = cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<64>>;

using LegacyPartA =
    typename autopartition::AutoPartitioner<ArchTag, OpClass, Element, StrideA, TileShape64, ThreadCount>::RoleA;
using LegacyPartB =
    typename autopartition::AutoPartitioner<ArchTag, OpClass, Element, StrideB, TileShape64, ThreadCount>::RoleB;
using LegacyPartC =
    typename autopartition::AutoPartitioner<ArchTag, OpClass, Element, StrideC, TileShape64, ThreadCount>::RoleC;

using ExtendedPartA = typename autopartition::AutoPartitioner<ArchTag,
                                                              OpClass,
                                                              Element,
                                                              StrideA,
                                                              TileShape64,
                                                              ThreadCount,
                                                              ElementC,
                                                              16,
                                                              16,
                                                              4>::RoleA;
using ExtendedPartB = typename autopartition::AutoPartitioner<ArchTag,
                                                              OpClass,
                                                              Element,
                                                              StrideB,
                                                              TileShape64,
                                                              ThreadCount,
                                                              ElementC,
                                                              16,
                                                              16,
                                                              4>::RoleB;
using ExtendedPartC = typename autopartition::AutoPartitioner<ArchTag,
                                                              OpClass,
                                                              Element,
                                                              StrideC,
                                                              TileShape64,
                                                              ThreadCount,
                                                              ElementC,
                                                              16,
                                                              16,
                                                              4>::RoleC;

static_assert(cute::cosize_v<typename LegacyPartA::SmemLayout> > 0, "Legacy RoleA must still instantiate.");
static_assert(cute::cosize_v<typename LegacyPartB::SmemLayout> > 0, "Legacy RoleB must still instantiate.");
static_assert(cute::cosize_v<typename LegacyPartC::SmemLayout> > 0, "Legacy RoleC must still instantiate.");
static_assert(cute::cosize_v<typename ExtendedPartA::SmemLayout> > 0, "Extended RoleA must instantiate.");
static_assert(cute::cosize_v<typename ExtendedPartB::SmemLayout> > 0, "Extended RoleB must instantiate.");
static_assert(cute::cosize_v<typename ExtendedPartC::SmemLayout> > 0, "Extended RoleC must instantiate.");

} // namespace

int main() { return 0; }
