#include <cute/tensor.hpp>

#include "cutlass/half.h"
#include "cutlass/transform/collective/auto_partitioner/auto_partitioner_builder.hpp"

using namespace cute;

int main()
{
    constexpr int ThreadCount = 128;

    using InputElement  = cutlass::half_t;
    using OutputElement = float;
    using TileShape     = Shape<Int<64>, Int<64>, Int<64>>;
    using StrideC       = decltype(make_stride(int{}, Int<1>{}));

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

    static_assert(PartC::EpilogueLayoutCandidateCount >= 32,
                  "RoleC must score many generic epilogue shared-memory layout candidates.");
    static_assert(PartC::EpilogueBankConflictScore <= PartC::EpilogueNaiveBankConflictScore,
                  "The selected epilogue layout must not score worse than the unswizzled baseline.");
    static_assert(PartC::HasZeroGlueEpilogueMapping,
                  "RoleC must expose a complete smem-to-output mapping contract.");
    static_assert(!std::is_void<typename PartC::RegisterToGlobalCopyAtom>::value,
                  "RoleC must expose a register-to-global copy atom compatible with the epilogue mapping.");

    return 0;
}
