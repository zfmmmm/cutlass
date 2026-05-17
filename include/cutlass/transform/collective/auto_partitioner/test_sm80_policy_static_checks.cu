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

using SimtArch = cutlass::arch::Sm80;
using SimtOpClass = cutlass::arch::OpClassSimt;
using FloatElement = float;
using SimtStrideA = cute::Stride<cute::_1, int64_t>;
using SimtTileShape = cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<16>>;
constexpr int SimtThreadCount = 256;

using SimtPartA16 = typename autopartition::AutoPartitioner<SimtArch,
                                                            SimtOpClass,
                                                            FloatElement,
                                                            SimtStrideA,
                                                            SimtTileShape,
                                                            SimtThreadCount,
                                                            FloatElement,
                                                            16,
                                                            16,
                                                            16>::RoleA;
using SimtPartA4 = typename autopartition::AutoPartitioner<SimtArch,
                                                           SimtOpClass,
                                                           FloatElement,
                                                           SimtStrideA,
                                                           SimtTileShape,
                                                           SimtThreadCount,
                                                           FloatElement,
                                                           4,
                                                           16,
                                                           16>::RoleA;

static_assert(SimtPartA16::GmemToSmemAlignmentBytes == 16, "16-byte physical alignment should allow 16-byte cp.async.");
static_assert(SimtPartA4::GmemToSmemAlignmentBytes == 4, "4-byte physical alignment should force 4-byte cp.async.");
static_assert(SimtPartA4::GmemToSmemAlignmentElements == 1, "float 4-byte copy uses one element.");

using PartA_KMajor = LegacyPartA;
using PartA_MnMajor = typename autopartition::AutoPartitioner<ArchTag,
                                                              OpClass,
                                                              Element,
                                                              cute::Stride<cute::_1, int64_t>,
                                                              TileShape64,
                                                              ThreadCount>::RoleA;
using PartB_MnMajor = LegacyPartB;
using PartB_KMajor = typename autopartition::AutoPartitioner<ArchTag,
                                                             OpClass,
                                                             Element,
                                                             cute::Stride<int64_t, cute::_1>,
                                                             TileShape64,
                                                             ThreadCount>::RoleB;

static_assert(!PartA_KMajor::SmemToRegNeedTranspose, "K-major A should not transpose for current TN MMA.");
static_assert(PartA_MnMajor::SmemToRegNeedTranspose, "MN-major A should transpose for current TN MMA.");
static_assert(PartB_MnMajor::SmemToRegNeedTranspose, "MN-major B should transpose for current TN MMA.");
static_assert(!PartB_KMajor::SmemToRegNeedTranspose, "K-major B should not transpose for current TN MMA.");
static_assert(std::is_same<typename PartA_KMajor::SmemToRegCopyOperation, cute::SM75_U32x4_LDSM_N>::value,
              "No-transpose ldmatrix should use LDSM_N x4.");
static_assert(std::is_same<typename PartB_MnMajor::SmemToRegCopyOperation, cute::SM75_U16x8_LDSM_T>::value,
              "Transpose ldmatrix should use LDSM_T x4.");

using TileShapeK32 = cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<32>>;
using PartA_K32 = typename autopartition::AutoPartitioner<ArchTag,
                                                          OpClass,
                                                          Element,
                                                          StrideA,
                                                          TileShapeK32,
                                                          ThreadCount>::RoleA;
using PartB_K32 = typename autopartition::AutoPartitioner<ArchTag,
                                                          OpClass,
                                                          Element,
                                                          StrideB,
                                                          TileShapeK32,
                                                          ThreadCount>::RoleB;

static_assert(PartA_K32::UseLdMatrix, "FP16 TileK=32 should keep ldmatrix enabled.");
static_assert(PartB_K32::UseLdMatrix, "FP16 TileK=32 should keep ldmatrix enabled.");
static_assert(PartA_K32::SwizzleBase == 2, "FP16 TileK=32 uses a 64-byte swizzle row.");
static_assert(PartB_K32::SwizzleBase == 2, "FP16 TileK=32 uses a 64-byte swizzle row.");
static_assert(cute::cosize_v<typename PartA_K32::SmemLayout> > 0, "TileK=32 A shared layout must be valid.");
static_assert(cute::cosize_v<typename PartB_K32::SmemLayout> > 0, "TileK=32 B shared layout must be valid.");

} // namespace

int main() { return 0; }
