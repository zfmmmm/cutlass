#include <cute/tensor.hpp>
#include <type_traits>

#include "auto_partitioner_builder.hpp"

using namespace cute;

namespace {

constexpr int ThreadCount = 128;
using Element             = cutlass::half_t;
using ElementC            = float;
using StrideA             = cute::Stride<cute::_1, int64_t>;
using StrideB             = cute::Stride<int64_t, cute::_1>;
using StrideC             = cute::Stride<cute::_1, int64_t>;
using TileShape           = cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<64>>;
using Cluster2x1x1        = cute::Shape<cute::_2, cute::_1, cute::_1>;

using PartA_Tma     = typename autopartition::AutoPartitioner<cutlass::arch::Sm100,
                                                              cutlass::arch::OpClassTensorOp,
                                                              Element,
                                                              StrideA,
                                                              TileShape,
                                                              ThreadCount,
                                                              ElementC,
                                                              16,
                                                              16,
                                                              16,
                                                              Cluster2x1x1>::RoleA;
using PartA_CpAsync = typename autopartition::AutoPartitioner<cutlass::arch::Sm100,
                                                              cutlass::arch::OpClassTensorOp,
                                                              Element,
                                                              StrideA,
                                                              TileShape,
                                                              ThreadCount,
                                                              ElementC,
                                                              8,
                                                              16,
                                                              16,
                                                              Cluster2x1x1>::RoleA;
using PartB_Tma     = typename autopartition::AutoPartitioner<cutlass::arch::Sm100,
                                                              cutlass::arch::OpClassTensorOp,
                                                              Element,
                                                              StrideB,
                                                              TileShape,
                                                              ThreadCount,
                                                              ElementC,
                                                              16,
                                                              16,
                                                              16,
                                                              Cluster2x1x1>::RoleB;
using PartC_Tma     = typename autopartition::AutoPartitioner<cutlass::arch::Sm100,
                                                              cutlass::arch::OpClassTensorOp,
                                                              Element,
                                                              StrideC,
                                                              TileShape,
                                                              ThreadCount,
                                                              ElementC,
                                                              16,
                                                              16,
                                                              16,
                                                              Cluster2x1x1>::RoleC;

static_assert(std::is_same<typename PartC_Tma::ClusterShape_MNK, Cluster2x1x1>::value,
              "ClusterShape must flow through SM100 TensorOp RoleC.");
static_assert(cute::size<0>(typename PartC_Tma::ClusterShape_MNK{}) == 2,
              "SM100 TensorOp RoleC must preserve the M cluster dimension.");
static_assert(PartA_Tma::UsesTmaLoad, "16-byte aligned A should use TMA.");
static_assert(PartB_Tma::UsesTmaLoad, "16-byte aligned B should use TMA.");
static_assert(!std::is_same<typename PartA_Tma::GmemToSmemCopy, cute::SM90_TMA_LOAD>::value,
              "TMA load must expose a host-encoded TiledCopy, not a raw opcode tag.");
static_assert(!PartA_CpAsync::UsesTmaLoad, "Sub-16-byte aligned A must fall back to cp.async.");
static_assert(PartA_CpAsync::GmemToSmemAlignmentBytes == 8, "cp.async fallback should honor A alignment.");
static_assert(cute::cosize_v<typename PartA_Tma::SmemLayout> > 0, "A smem layout must instantiate.");
static_assert(cute::cosize_v<typename PartB_Tma::SmemLayout> > 0, "B smem layout must instantiate.");

static_assert(std::is_same<typename PartC_Tma::ElementOutput, ElementC>::value,
              "SM100 RoleC must preserve the global output element type.");
static_assert(cute::cosize_v<typename PartC_Tma::SmemLayout> > 0, "C epilogue smem layout must instantiate.");
static_assert(!std::is_same<typename PartC_Tma::SmemToGmemCopy, cute::SM90_TMA_STORE>::value,
              "Epilogue store must expose a host-encoded TMA TiledCopy, not a raw opcode tag.");
static_assert(!std::is_void<typename PartC_Tma::TmemToSmemCopy>::value,
              "SM100 RoleC must expose the TMEM-to-SMEM unload copy.");

using Sm120PartA = typename autopartition::AutoPartitioner<cutlass::arch::Sm120,
                                                           cutlass::arch::OpClassTensorOp,
                                                           cutlass::float_e4m3_t,
                                                           StrideA,
                                                           TileShape,
                                                           256,
                                                           ElementC,
                                                           16,
                                                           16,
                                                           16,
                                                           Cluster2x1x1>::RoleA;
using Sm120PartB = typename autopartition::AutoPartitioner<cutlass::arch::Sm120,
                                                           cutlass::arch::OpClassTensorOp,
                                                           cutlass::float_e4m3_t,
                                                           StrideB,
                                                           TileShape,
                                                           256,
                                                           ElementC,
                                                           16,
                                                           16,
                                                           16,
                                                           Cluster2x1x1>::RoleB;

static_assert(std::is_same<typename Sm120PartA::ScaleElement, float>::value,
              "SM120 FP8 scale factors should use FP32 storage by default.");
static_assert(Sm120PartA::GmemToSmemAlignmentBytes == 16,
              "SM120 RoleA must preserve the public A alignment contract.");
static_assert(std::is_same<typename Sm120PartA::ClusterShape_MNK, Cluster2x1x1>::value,
              "SM120 RoleA must preserve the public cluster shape contract.");
static_assert(Sm120PartB::GmemToSmemAlignmentBytes == 16,
              "SM120 RoleB must preserve the public B alignment contract.");
static_assert(std::is_same<typename Sm120PartB::ClusterShape_MNK, Cluster2x1x1>::value,
              "SM120 RoleB must preserve the public cluster shape contract.");
static_assert(cute::cosize_v<typename Sm120PartA::ScaleSmemLayout> > 0,
              "SM120 RoleA must expose a scale-factor smem layout.");
static_assert(cute::cosize_v<typename Sm120PartB::ScaleSmemLayout> > 0,
              "SM120 RoleB must expose a scale-factor smem layout.");
using Sm120PartC = typename autopartition::AutoPartitioner<cutlass::arch::Sm120,
                                                           cutlass::arch::OpClassTensorOp,
                                                           cutlass::float_e4m3_t,
                                                           StrideC,
                                                           TileShape,
                                                           256,
                                                           ElementC,
                                                           16,
                                                           16,
                                                           8,
                                                           Cluster2x1x1>::RoleC;
static_assert(std::is_same<typename Sm120PartC::ElementOutput, ElementC>::value,
              "SM120 RoleC must preserve the public output element type.");
static_assert(Sm120PartC::GmemToSmemAlignmentBytes == 8,
              "SM120 RoleC must preserve the public C alignment contract.");
static_assert(std::is_same<typename Sm120PartC::ClusterShape_MNK, Cluster2x1x1>::value,
              "SM120 RoleC must preserve the public cluster shape contract.");

using SimtK32Tile          = cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<32>>;
using Sm100SimtK32Fallback = typename autopartition::AutoPartitioner<cutlass::arch::Sm100,
                                                                     cutlass::arch::OpClassSimt,
                                                                     float,
                                                                     StrideA,
                                                                     SimtK32Tile,
                                                                     ThreadCount,
                                                                     float,
                                                                     16,
                                                                     16,
                                                                     16,
                                                                     Cluster2x1x1>::RoleA;
static_assert(cute::cosize_v<typename Sm100SimtK32Fallback::SmemLayout> > 0,
              "SM100 SIMT TileK != 16 should route to a safe fallback instead of a hard assert.");

} // namespace

int main() { return 0; }
