# AutoPartition Epilogue Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor SM80 and SM100 AutoPartitioner TensorOp RoleC epilogue layout generation to use swizzled, padding-free write-back layouts and complete AutoPartitioner-driven GEMM examples.

**Architecture:** AutoPartitioner remains a compile-time layout/type generator. SM80 RoleC gets an epilogue-specific swizzle selector derived from output element, logical M/N row width, vector-copy legality, and the 16-thread epilogue instruction group. SM100 RoleC wraps official CUTLASS SM100 epilogue builder-derived types behind the existing RoleC interface.

**Tech Stack:** C++17/CUDA, CuTe layouts/copy atoms/MMA atoms, CUTLASS AutoPartitioner policy headers, CUTLASS unit tests, NVCC/CMake.

---

## File Structure

- Modify `include/cutlass/transform/collective/auto_partitioner/arch/sm80_policy.hpp`
  - Add SM80 epilogue-specific swizzle helpers.
  - Replace TensorOp RoleC padding layouts with generated swizzled layouts.
  - Keep RoleC as pure type/constant generation.
- Modify `include/cutlass/transform/collective/auto_partitioner/arch/sm100_policy.hpp`
  - Include official epilogue collective builder.
  - Remove AutoPartitioner SM120 test implementation and routing.
  - Refactor TensorOp RoleC to derive epilogue layout/copy operation types from official SM100 epilogue builder machinery.
- Modify `test/unit/transform/collective/auto_partitioner/sm80_policy_audit/sm80_policy_audit.cu`
  - Add static and runtime-connectivity checks for SM80 RoleC epilogue swizzle, selected vector width, no padding, ldmatrix, and complete one-tile GEMM write-back.
- Modify `test/unit/transform/collective/auto_partitioner/sm1xx_policy_audit.cu`
  - Remove SM120 AutoPartitioner assertions.
  - Add SM100 RoleC official-epilogue-derived type checks.
- Modify `include/cutlass/transform/collective/auto_partitioner/test_sm80_policy_static_checks.cu`
  - Update static checks for SM80 RoleC no-padding epilogue swizzle.
- Modify `include/cutlass/transform/collective/auto_partitioner/test_sm100_policy_static_checks.cu`
  - Remove SM120 static checks.
  - Add SM100 epilogue builder type checks.
- Modify `include/cutlass/transform/collective/auto_partitioner/examples/production_gemm/sm80_autopartition_gemm.cu`
  - Use `PartC` output layout/copy types for register-to-shared and shared-to-global write-back.
- Modify `include/cutlass/transform/collective/auto_partitioner/examples/production_gemm/sm100_autopartition_tma_umma_gemm.cu`
  - Use `PartC` official-derived epilogue layout/copy operation types for TMEM unload, register-to-shared staging, and final store.
- Optional compile-only command output files remain untracked and are not part of the implementation.

## Task 1: Add Failing SM80 RoleC Epilogue Tests

**Files:**
- Modify: `test/unit/transform/collective/auto_partitioner/sm80_policy_audit/sm80_policy_audit.cu`
- Modify: `include/cutlass/transform/collective/auto_partitioner/test_sm80_policy_static_checks.cu`

- [ ] **Step 1: Add static checks that require the new SM80 RoleC epilogue contract**

Add these checks near the existing `AutoPartitionerSm80Phase4` RoleC assertions:

```cpp
using EpiOutput = float;
using EpiStrideC = cute::Stride<int64_t, cute::_1>;
using EpiTile = cute::Shape<cute::_64, cute::_64, cute::_64>;
using EpiPartC = typename autopartition::AutoPartitioner<
    cutlass::arch::Sm80, cutlass::arch::OpClassTensorOp, cutlass::half_t,
    EpiStrideC, EpiTile, 128, EpiOutput, 16, 16, 16>::RoleC;

static_assert(EpiPartC::EpilogueInstructionThreads == 16,
              "SM80 epilogue swizzle must model the 16-thread store/load issue group.");
static_assert(EpiPartC::OutputAlignmentBytes == EpiPartC::EpilogueVectorBytes,
              "RoleC output vector width must be derived from the selected output tiled-copy alignment.");
static_assert(EpiPartC::EpilogueSwizzleBytes == 32 ||
              EpiPartC::EpilogueSwizzleBytes == 64 ||
              EpiPartC::EpilogueSwizzleBytes == 128,
              "RoleC epilogue swizzle must use a supported shared-memory swizzle span.");
static_assert(cute::cosize_v<typename EpiPartC::OutputSmemLayout> == EpiPartC::BlkM * EpiPartC::BlkN,
              "SM80 RoleC output shared layout must not allocate padding elements.");
static_assert(!std::is_same<typename EpiPartC::OutputSmemLayout,
                            cute::Layout<cute::Shape<cute::Int<64>, cute::Int<64>>,
                                         cute::Stride<cute::Int<68>, cute::_1>>>::value,
              "SM80 RoleC output shared layout must not be the old padding layout.");
```

- [ ] **Step 2: Add a complete one-tile write-back check using `PartC` layout**

Replace the direct `autopartition::examples::convert_tensor(tCgC, tCrC);` tail in the SM80 audit one-tile GEMM helper with an audit-only path that converts registers to `PartC::ElementOutput`, stores through `PartC::OutputSmemLayout`, then uses `PartC::SmemToGmemCopy`:

```cpp
Tensor tCrD = make_fragment_like<typename PartC::ElementOutput>(tCrC);
cutlass::NumericConverter<typename PartC::ElementOutput, typename PartC::ElementCompute> convert;
CUTE_UNROLL
for (int i = 0; i < cute::size(tCrC); ++i) {
    tCrD(i) = convert(tCrC(i));
}

__shared__ cute::array_aligned<typename PartC::ElementOutput,
                               cute::cosize_v<typename PartC::OutputSmemLayout>> smemC;
Tensor sC = make_tensor(make_smem_ptr(smemC.data()), typename PartC::OutputSmemLayout{});
auto smem_tiled_copy_C =
    make_tiled_copy_C(Copy_Atom<typename PartC::RegToSmemCopyOperation,
                                typename PartC::ElementOutput>{}, thr_mma);
auto smem_thr_copy_C = smem_tiled_copy_C.get_thread_slice(threadIdx.x);
Tensor tCsC = smem_thr_copy_C.partition_D(sC);
Tensor tCrD_view = smem_thr_copy_C.retile_S(tCrD);
copy(smem_tiled_copy_C, tCrD_view, tCsC);
__syncthreads();

cute::cooperative_copy<128, PartC::OutputAlignmentBits>(
    threadIdx.x, sC, gC, typename PartC::SmemToGmemCopy{});
```

- [ ] **Step 3: Run the SM80 audit target and confirm it fails on missing symbols**

Run:

```bash
cmake --build build --target cutlass_test_unit_transform_collective_auto_partitioner_sm80_policy_audit -j$(nproc)
```

Expected: compile failure mentioning missing `EpilogueInstructionThreads`, `EpilogueVectorBytes`, or `EpilogueSwizzleBytes` in `Sm80TensorOpRoleC`.

- [ ] **Step 4: Commit the failing tests**

```bash
git add test/unit/transform/collective/auto_partitioner/sm80_policy_audit/sm80_policy_audit.cu \
        include/cutlass/transform/collective/auto_partitioner/test_sm80_policy_static_checks.cu
git commit -m "test: require sm80 tensorop epilogue swizzle layout"
```

## Task 2: Implement SM80 Padding-Free Epilogue Swizzle RoleC

**Files:**
- Modify: `include/cutlass/transform/collective/auto_partitioner/arch/sm80_policy.hpp`

- [ ] **Step 1: Add SM80 epilogue swizzle helpers above `Sm80TensorOpRoleC`**

Add these helper declarations:

```cpp
template <int Bytes> struct Sm80EpilogueSwizzleBase;
template <> struct Sm80EpilogueSwizzleBase<32>  { static constexpr int value = 1; };
template <> struct Sm80EpilogueSwizzleBase<64>  { static constexpr int value = 2; };
template <> struct Sm80EpilogueSwizzleBase<128> { static constexpr int value = 3; };

template <int InstructionBytes>
struct Sm80EpilogueSwizzleBytes {
    static constexpr int value = (InstructionBytes >= 128) ? 128 :
                                 (InstructionBytes >= 64)  ? 64  :
                                 (InstructionBytes >= 32)  ? 32  : 0;
    static_assert(value != 0, "SM80 epilogue swizzle requires at least a 32-byte instruction span.");
};

template <class Element, int LogicalMajorExtent, int VectorBytes, bool IsMnMajor>
struct Sm80TensorOpEpilogueSmemLayoutSelector {
    static constexpr int InstructionThreads = 16;
    static constexpr int InstructionBytes = InstructionThreads * VectorBytes;
    static constexpr int SwizzleBytes = Sm80EpilogueSwizzleBytes<InstructionBytes>::value;
    static constexpr int SwizzleBase = Sm80EpilogueSwizzleBase<SwizzleBytes>::value;
    static constexpr int RowElements = SwizzleBytes / int(sizeof(Element));
    static_assert((SwizzleBytes % int(sizeof(Element))) == 0,
                  "SM80 epilogue swizzle span must be element-addressable.");
    static_assert((LogicalMajorExtent % RowElements) == 0,
                  "SM80 epilogue logical row must be covered by whole swizzle atoms.");

    using SwizzleAtom = cute::conditional_t<
        IsMnMajor,
        decltype(cute::composition(
            cute::Swizzle<SwizzleBase, 3, 3>{},
            cute::Layout<cute::Shape<cute::Int<RowElements>, cute::_16>,
                         cute::Stride<cute::_1, cute::Int<RowElements>>>{})),
        decltype(cute::composition(
            cute::Swizzle<SwizzleBase, 3, 3>{},
            cute::Layout<cute::Shape<cute::_16, cute::Int<RowElements>>,
                         cute::Stride<cute::Int<RowElements>, cute::_1>>>{}))>;
};
```

- [ ] **Step 2: Replace `Sm80TensorOpRoleC` padding layouts**

Inside `Sm80TensorOpRoleC`, remove `Padding`, `OutputPadding`, and padding-based `SmemLayoutAtom` definitions. Replace them with:

```cpp
static constexpr int OutputAlignmentElements =
    GmemTiledCopyAlignment<ElementOutput, BlkM, BlkN, ThreadCount, IsMnMajor, GmemAlignmentBytes>::value;
static constexpr int OutputAlignmentBytes =
    GmemTiledCopyAlignment<ElementOutput, BlkM, BlkN, ThreadCount, IsMnMajor, GmemAlignmentBytes>::bytes;
static constexpr int OutputAlignmentBits = OutputAlignmentBytes * 8;

static constexpr int EpilogueInstructionThreads = 16;
static constexpr int EpilogueVectorElements = OutputAlignmentElements;
static constexpr int EpilogueVectorBytes = OutputAlignmentBytes;
static constexpr int EpilogueLogicalMajorExtent = IsMnMajor ? BlkM : BlkN;

using OutputSwizzleSelector =
    Sm80TensorOpEpilogueSmemLayoutSelector<ElementOutput,
                                           EpilogueLogicalMajorExtent,
                                           EpilogueVectorBytes,
                                           IsMnMajor>;
static constexpr int EpilogueSwizzleBytes = OutputSwizzleSelector::SwizzleBytes;
static constexpr int EpilogueSwizzleBase = OutputSwizzleSelector::SwizzleBase;
using OutputSmemLayoutAtom = typename OutputSwizzleSelector::SwizzleAtom;
using OutputSmemLayout = decltype(cute::tile_to_shape(
    OutputSmemLayoutAtom{}, cute::Shape<cute::Int<BlkM>, cute::Int<BlkN>>{}));
using SmemLayout = OutputSmemLayout;
using SmemLayoutAtom = OutputSmemLayoutAtom;
```

Keep `RegToSmemCopyOperation`, `SmemToGmemCopy`, and public aliases, but make them use `ElementOutput` for the output staging path.

- [ ] **Step 3: Run the SM80 audit target and confirm it passes compilation**

Run:

```bash
cmake --build build --target cutlass_test_unit_transform_collective_auto_partitioner_sm80_policy_audit -j$(nproc)
```

Expected: target builds successfully. Runtime may skip or fail on non-SM80 hardware; compile success is required.

- [ ] **Step 4: Commit SM80 RoleC implementation**

```bash
git add include/cutlass/transform/collective/auto_partitioner/arch/sm80_policy.hpp
git commit -m "feat: add sm80 tensorop epilogue swizzle layout"
```

## Task 3: Remove AutoPartitioner SM120 Test Surface

**Files:**
- Modify: `include/cutlass/transform/collective/auto_partitioner/arch/sm100_policy.hpp`
- Modify: `include/cutlass/transform/collective/auto_partitioner/auto_partitioner_builder.hpp`
- Modify: `include/cutlass/transform/collective/auto_partitioner/test_sm100_policy_static_checks.cu`
- Modify: `test/unit/transform/collective/auto_partitioner/sm1xx_policy_audit.cu`

- [ ] **Step 1: Remove SM120 includes and traits from AutoPartitioner policy**

Delete from `sm100_policy.hpp`:

```cpp
#include <cute/atom/mma_traits_sm120.hpp>
#include <cutlass/gemm/collective/builders/sm120_common.inl>
template <class Element> struct IsSm120TensorOpElement ...
// the commented Sm120TensorOpRoleA/B/C implementation block
// the commented AutoPartitioner<cutlass::arch::Sm120, ...> specialization block
```

Update the builder comment to say SM80/SM100:

```cpp
// SM80/SM100 policy specializations.
```

- [ ] **Step 2: Remove SM120 static assertions from tests**

Delete the `Sm120PartA`, `Sm120PartB`, `Sm120PartC`, and `Sm120Fp8A` blocks from the two SM100 audit/static check files.

- [ ] **Step 3: Run the SM1xx audit target and confirm SM120 references are gone**

Run:

```bash
rg -n "Sm120|SM120|sm120" include/cutlass/transform/collective/auto_partitioner test/unit/transform/collective/auto_partitioner
cmake --build build --target cutlass_test_unit_transform_collective_auto_partitioner -j$(nproc)
```

Expected: `rg` reports no AutoPartitioner SM120 policy/test references except unrelated prose outside this scope; build reaches the next known failure only if SM100 RoleC tests have already been added and not implemented.

- [ ] **Step 4: Commit SM120 cleanup**

```bash
git add include/cutlass/transform/collective/auto_partitioner/arch/sm100_policy.hpp \
        include/cutlass/transform/collective/auto_partitioner/auto_partitioner_builder.hpp \
        include/cutlass/transform/collective/auto_partitioner/test_sm100_policy_static_checks.cu \
        test/unit/transform/collective/auto_partitioner/sm1xx_policy_audit.cu
git commit -m "chore: remove autopartitioner sm120 test path"
```

## Task 4: Add Failing SM100 Official Epilogue RoleC Tests

**Files:**
- Modify: `include/cutlass/transform/collective/auto_partitioner/test_sm100_policy_static_checks.cu`
- Modify: `test/unit/transform/collective/auto_partitioner/sm1xx_policy_audit.cu`

- [ ] **Step 1: Add SM100 RoleC official epilogue type checks**

Add checks for `PartC_Tma`:

```cpp
static_assert(!std::is_void<typename PartC_Tma::CollectiveEpilogue>::value,
              "SM100 RoleC must expose the official CollectiveBuilder epilogue type.");
static_assert(!std::is_void<typename PartC_Tma::EpilogueTile>::value,
              "SM100 RoleC must expose the official epilogue tile.");
static_assert(cute::cosize_v<typename PartC_Tma::SharedToGlobalLayout> == PartC_Tma::BlkM * PartC_Tma::BlkN ||
              cute::cosize_v<typename PartC_Tma::SharedToGlobalLayout> > 0,
              "SM100 RoleC must expose a usable official-derived shared-to-global layout.");
static_assert(!std::is_void<typename PartC_Tma::TmemToRegisterCopyOperation>::value,
              "SM100 RoleC must expose official TMEM-to-register copy op.");
static_assert(!std::is_void<typename PartC_Tma::RegisterToSharedCopyOperation>::value,
              "SM100 RoleC must expose official register-to-shared copy op.");
static_assert(!std::is_void<typename PartC_Tma::SharedToGlobalCopyOperation>::value,
              "SM100 RoleC must expose official shared-to-global copy op.");
```

- [ ] **Step 2: Run the SM1xx audit target and confirm it fails on missing RoleC aliases**

Run:

```bash
cmake --build build --target cutlass_test_unit_transform_collective_auto_partitioner -j$(nproc)
```

Expected: compile failure mentioning one of `CollectiveEpilogue`, `EpilogueTile`, `TmemToRegisterCopyOperation`, `RegisterToSharedCopyOperation`, or `SharedToGlobalCopyOperation`.

- [ ] **Step 3: Commit the failing SM100 tests**

```bash
git add include/cutlass/transform/collective/auto_partitioner/test_sm100_policy_static_checks.cu \
        test/unit/transform/collective/auto_partitioner/sm1xx_policy_audit.cu
git commit -m "test: require sm100 official epilogue rolec types"
```

## Task 5: Implement SM100 RoleC With Official Epilogue Builder Types

**Files:**
- Modify: `include/cutlass/transform/collective/auto_partitioner/arch/sm100_policy.hpp`

- [ ] **Step 1: Include official epilogue builder**

Add:

```cpp
#include <cutlass/epilogue/collective/collective_builder.hpp>
```

- [ ] **Step 2: Add stride-to-layout-tag helper**

Add in `autopartition::detail`:

```cpp
template <class GmemStride>
struct Sm100EpilogueLayoutTag {
    using type = cute::conditional_t<cutlass::gemm::detail::is_mn_major<GmemStride>(),
                                     cutlass::layout::ColumnMajor,
                                     cutlass::layout::RowMajor>;
};
```

- [ ] **Step 3: Replace SM100 TensorOp RoleC padding layout**

Inside `Sm100TensorOpRoleC`, derive official types:

```cpp
using GmemLayoutTagC = typename Sm100EpilogueLayoutTag<GmemStride>::type;
static constexpr int AlignmentElements =
    (GmemAlignmentBytes >= int(sizeof(ElementOutput))) ? GmemAlignmentBytes / int(sizeof(ElementOutput)) : 1;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100,
    cutlass::arch::OpClassTensorOp,
    TileShape_MNK,
    ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    Accumulator,
    Accumulator,
    ElementOutput,
    GmemLayoutTagC,
    AlignmentElements,
    ElementOutput,
    GmemLayoutTagC,
    AlignmentElements,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using EpilogueTile = typename CollectiveEpilogue::EpilogueTile;
using SmemLayoutAtom = typename CollectiveEpilogue::SmemLayoutAtomD;
using SharedToGlobalLayout = decltype(cute::tile_to_shape(
    SmemLayoutAtom{},
    cute::product_each(cute::shape(EpilogueTile{})),
    cute::conditional_t<IsMnMajor, cute::Step<cute::_2, cute::_1>, cute::Step<cute::_1, cute::_2>>{}));
using OutputSmemLayout = SharedToGlobalLayout;
using SmemLayout = OutputSmemLayout;

using TmemToRegisterCopyOperation = typename CollectiveEpilogue::CopyOpT2R;
using RegisterToSharedCopyOperation = typename CollectiveEpilogue::CopyOpR2S;
using SharedToGlobalCopyOperation = typename CollectiveEpilogue::CopyOpS2G;
using TmemToRegisterCopy = cute::Copy_Atom<TmemToRegisterCopyOperation, Accumulator>;
using RegisterToSharedCopy = cute::Copy_Atom<RegisterToSharedCopyOperation, ElementOutput>;
using SharedToRegisterCopy = cute::Copy_Atom<typename CollectiveEpilogue::CopyOpS2R, ElementOutput>;
```

Keep compatibility aliases:

```cpp
using TmemToSmemCopy = TmemToRegisterCopy;
using SmemToRegCopy = SharedToRegisterCopy;
using RegToSmemCopy = RegisterToSharedCopy;
using SmemToGmemCopy = decltype(cute::make_tma_copy(
    SharedToGlobalCopyOperation{},
    cute::make_tensor(cute::make_gmem_ptr(static_cast<ElementOutput *>(nullptr)),
                      cute::make_shape(cute::Int<BlkM>{}, cute::Int<BlkN>{}), GmemStride{}),
    SharedToGlobalLayout{},
    cute::make_shape(cute::Int<BlkM>{}, cute::Int<BlkN>{}),
    cute::Int<1>{}));
```

- [ ] **Step 4: Run the SM1xx audit target**

Run:

```bash
cmake --build build --target cutlass_test_unit_transform_collective_auto_partitioner -j$(nproc)
```

Expected: target builds successfully. Runtime execution is handled in final verification because the local SM120 device may not execute SM80/SM100 kernels.

- [ ] **Step 5: Commit SM100 RoleC implementation**

```bash
git add include/cutlass/transform/collective/auto_partitioner/arch/sm100_policy.hpp
git commit -m "feat: derive sm100 tensorop rolec epilogue from collective builder"
```

## Task 6: Rewrite SM80 Production GEMM Epilogue To Use PartC

**Files:**
- Modify: `include/cutlass/transform/collective/auto_partitioner/examples/production_gemm/sm80_autopartition_gemm.cu`

- [ ] **Step 1: Replace shared C storage with `PartC::OutputSmemLayout`**

Use:

```cpp
cute::array_aligned<OutputElement, cute::cosize_v<typename PartC::OutputSmemLayout>> smemC;
Tensor sC = make_tensor(make_smem_ptr(shared.smemC.data()), typename PartC::OutputSmemLayout{});
```

- [ ] **Step 2: Keep register conversion in the example kernel**

Use:

```cpp
Tensor tCrD = make_fragment_like<OutputElement>(tCrC);
cutlass::NumericConverter<OutputElement, typename PartC::ElementCompute> convert;
CUTE_UNROLL
for (int i = 0; i < size(tCrC); ++i) {
    tCrD(i) = convert(tCrC(i));
}
```

- [ ] **Step 3: Write through `PartC` register-to-shared and shared-to-global interfaces**

Use:

```cpp
auto smem_tiled_copy_C =
    make_tiled_copy_C(Copy_Atom<typename PartC::RegToSmemCopyOperation, OutputElement>{}, thr_mma);
auto smem_thr_copy_C = smem_tiled_copy_C.get_thread_slice(threadIdx.x);
Tensor tCsC = smem_thr_copy_C.partition_D(sC);
Tensor tCrD_view = smem_thr_copy_C.retile_S(tCrD);
copy(smem_tiled_copy_C, tCrD_view, tCsC);
__syncthreads();
cooperative_copy<ThreadCount, PartC::OutputAlignmentBits>(
    threadIdx.x, sC, gC, typename PartC::SmemToGmemCopy{});
```

- [ ] **Step 4: Compile the SM80 production example**

Run:

```bash
nvcc -std=c++17 -Iinclude -Itools/util/include \
  -arch=sm_120 \
  include/cutlass/transform/collective/auto_partitioner/examples/production_gemm/sm80_autopartition_gemm.cu \
  -o /tmp/sm80_autopartition_gemm_compile
```

Expected: compile succeeds on the SM120 development machine. Running may fail or be unsupported because the kernel targets SM80 instructions.

- [ ] **Step 5: Commit SM80 production example**

```bash
git add include/cutlass/transform/collective/auto_partitioner/examples/production_gemm/sm80_autopartition_gemm.cu
git commit -m "feat: route sm80 production gemm epilogue through rolec layout"
```

## Task 7: Rewrite SM100 Production GEMM Epilogue To Use PartC

**Files:**
- Modify: `include/cutlass/transform/collective/auto_partitioner/examples/production_gemm/sm100_autopartition_tma_umma_gemm.cu`

- [ ] **Step 1: Add D shared storage using `PartC::SharedToGlobalLayout`**

Extend shared storage with:

```cpp
alignas(128) cute::ArrayEngine<TypeD, cute::cosize_v<typename PartC::SharedToGlobalLayout>> D;
CUTE_DEVICE constexpr auto tensor_sD() {
    return cute::make_tensor(cute::make_smem_ptr(D.begin()), typename PartC::SharedToGlobalLayout{});
}
```

- [ ] **Step 2: Replace direct TMEM-to-global copy with official-derived PartC stages**

Use:

```cpp
TiledCopy tiled_t2r_copy = make_tmem_copy(typename PartC::TmemToRegisterCopyOperation{}, tCtAcc);
ThrCopy thr_t2r_copy = tiled_t2r_copy.get_slice(threadIdx.x);
Tensor tDtAcc = thr_t2r_copy.partition_S(tCtAcc);
Tensor tDsD = thr_t2r_copy.partition_D(tCsD);
Tensor tDrD = make_tensor<typename decltype(tCtAcc)::value_type>(shape(tDsD));
copy(tiled_t2r_copy, tDtAcc, tDrD);

auto smem_tiled_copy_D =
    make_tiled_copy(Copy_Atom<typename PartC::RegisterToSharedCopyOperation, OutputElement>{},
                    typename PartC::SharedToGlobalLayout{});
copy(smem_tiled_copy_D, tDrD, tDsD);
__syncthreads();
```

The implementation uses the `PartC::TmemToRegisterCopyOperation` partition shape as the source of truth: `tDsD` is partitioned from `PartC::SharedToGlobalLayout`, and the register-to-shared tiled copy is built from `PartC::RegisterToSharedCopyOperation` over that exact destination partition.

- [ ] **Step 3: Use `PartC::SmemToGmemCopy` or `PartC::SharedToGlobalCopyOperation` for final store**

Use the host-created TMA store descriptor or vector fallback from `PartC`:

```cpp
auto tma_store_D = make_tma_copy(typename PartC::SharedToGlobalCopyOperation{},
                                 mD,
                                 typename PartC::SharedToGlobalLayout{},
                                 typename PartC::EpilogueTile{},
                                 Int<1>{});
```

The kernel receives this descriptor and copies `tCsD` to the global D tile through it.

- [ ] **Step 4: Compile the SM100 production example**

Run:

```bash
nvcc -std=c++17 -Iinclude -Itools/util/include \
  -arch=sm_120 \
  include/cutlass/transform/collective/auto_partitioner/examples/production_gemm/sm100_autopartition_tma_umma_gemm.cu \
  -o /tmp/sm100_autopartition_tma_umma_gemm_compile
```

Expected: compile succeeds. A compiler diagnostic that says the local CUDA toolkit or selected architecture does not enable SM100/UMMA is recorded as an environment blocker in the final response.

- [ ] **Step 5: Commit SM100 production example**

```bash
git add include/cutlass/transform/collective/auto_partitioner/examples/production_gemm/sm100_autopartition_tma_umma_gemm.cu
git commit -m "feat: route sm100 production gemm epilogue through rolec"
```

## Task 8: Final Verification

**Files:**
- Verify all modified files from previous tasks.

- [ ] **Step 1: Build AutoPartitioner unit targets**

Run:

```bash
cmake --build build --target cutlass_test_unit_transform_collective_auto_partitioner -j$(nproc)
cmake --build build --target cutlass_test_unit_transform_collective_auto_partitioner_sm80_policy_audit -j$(nproc)
```

Expected: both targets compile.

- [ ] **Step 2: Compile complete production GEMM examples**

Run:

```bash
nvcc -std=c++17 -Iinclude -Itools/util/include \
  -arch=sm_120 \
  include/cutlass/transform/collective/auto_partitioner/examples/production_gemm/sm80_autopartition_gemm.cu \
  -o /tmp/sm80_autopartition_gemm_compile

nvcc -std=c++17 -Iinclude -Itools/util/include \
  -arch=sm_120 \
  include/cutlass/transform/collective/auto_partitioner/examples/production_gemm/sm100_autopartition_tma_umma_gemm.cu \
  -o /tmp/sm100_autopartition_tma_umma_gemm_compile
```

Expected: examples compile, or the SM100 example reports an architecture guard/toolkit support limitation that is documented in the final response.

- [ ] **Step 3: Run unit binaries if compilation and hardware allow**

Run:

```bash
./build/test/unit/transform/collective/auto_partitioner/cutlass_test_unit_transform_collective_auto_partitioner
./build/test/unit/transform/collective/auto_partitioner/sm80_policy_audit/cutlass_test_unit_transform_collective_auto_partitioner_sm80_policy_audit
```

Expected: tests pass or skip runtime sections when SM80/SM100 execution is unavailable on the SM120 device.

- [ ] **Step 4: Inspect git diff**

Run:

```bash
git diff --stat
git diff --check
```

Expected: no whitespace errors and only scoped files changed.

- [ ] **Step 5: Commit final verification changes when the verification diff is non-empty**

```bash
git add include/cutlass/transform/collective/auto_partitioner/arch/sm80_policy.hpp \
        include/cutlass/transform/collective/auto_partitioner/arch/sm100_policy.hpp \
        include/cutlass/transform/collective/auto_partitioner/auto_partitioner_builder.hpp \
        include/cutlass/transform/collective/auto_partitioner/test_sm80_policy_static_checks.cu \
        include/cutlass/transform/collective/auto_partitioner/test_sm100_policy_static_checks.cu \
        include/cutlass/transform/collective/auto_partitioner/examples/production_gemm/sm80_autopartition_gemm.cu \
        include/cutlass/transform/collective/auto_partitioner/examples/production_gemm/sm100_autopartition_tma_umma_gemm.cu \
        test/unit/transform/collective/auto_partitioner/sm80_policy_audit/sm80_policy_audit.cu \
        test/unit/transform/collective/auto_partitioner/sm1xx_policy_audit.cu
git commit -m "test: verify autopartition epilogue layout refactor"
```
