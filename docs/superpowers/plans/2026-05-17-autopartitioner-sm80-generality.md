# AutoPartitioner SM80 Generality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make SM80 AutoPartitioner alignment-aware, role-aware for ldmatrix, output-type-safe for TensorOp epilogues, more flexible for TileK swizzles, and complete for DMMA while preserving the existing six-argument API.

**Architecture:** Extend the primary `AutoPartitioner` signature with defaulted `ElementC` and alignment parameters, then thread those contracts through SM80/SM100/SM120 specializations. Keep policy files as blueprint generators; examples and tests remain responsible for shared storage, numeric conversion, and kernel launch.

**Tech Stack:** CUDA C++17, CuTe/CUTLASS templates, `nvcc`, standalone `.cu` example/test programs under `include/cutlass/transform/collective/auto_partitioner/`.

---

## File Structure

- Modify `include/cutlass/transform/collective/auto_partitioner/auto_partitioner.hpp`: extend the public template signature with defaulted `ElementC` and byte-alignment parameters.
- Modify `include/cutlass/transform/collective/auto_partitioner/arch/sm80_policy.hpp`: implement alignment-aware copy selection, role-aware ldmatrix routing, dynamic swizzle rows, ElementC-aware RoleC, and DMMA traits.
- Modify `include/cutlass/transform/collective/auto_partitioner/arch/sm100_policy.hpp`: accept the extended template signature so old and new public calls route cleanly.
- Modify `include/cutlass/transform/collective/auto_partitioner/examples/autopartition_example_utils.hpp`: add generic diff helpers and optional output corruption utilities used by edge-case tests.
- Modify `include/cutlass/transform/collective/auto_partitioner/examples/sm80_tensorop_example.cu`: use the new ElementC-aware RoleC aliases and explicit accumulator-to-output conversion.
- Create `include/cutlass/transform/collective/auto_partitioner/test_sm80_policy_static_checks.cu`: compile-time checks for API compatibility, alignment, ldmatrix routing, dynamic swizzle, ElementC, and DMMA traits.
- Create `include/cutlass/transform/collective/auto_partitioner/test_sm80_tensorop_edge_cases.cu`: runtime checks for negative comparison sanity, 4-byte aligned subtensor copies, FP16 output epilogue, TileK=32 TensorOp, and double TensorOp when supported.

## Task 1: Extend AutoPartitioner Public Interface

**Files:**
- Modify: `include/cutlass/transform/collective/auto_partitioner/auto_partitioner.hpp`
- Modify: `include/cutlass/transform/collective/auto_partitioner/arch/sm80_policy.hpp`
- Modify: `include/cutlass/transform/collective/auto_partitioner/arch/sm100_policy.hpp`
- Create: `include/cutlass/transform/collective/auto_partitioner/test_sm80_policy_static_checks.cu`

- [ ] **Step 1: Write the failing API compatibility test**

Create `include/cutlass/transform/collective/auto_partitioner/test_sm80_policy_static_checks.cu` with this content:

```cpp
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
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
/usr/local/cuda/bin/nvcc -std=c++17 \
  -Iinclude \
  -Iinclude/cutlass/transform/collective/auto_partitioner \
  -Iinclude/cutlass/transform/collective/auto_partitioner/examples \
  -arch=sm_120 \
  include/cutlass/transform/collective/auto_partitioner/test_sm80_policy_static_checks.cu \
  -o /tmp/test_sm80_policy_static_checks
```

Expected: compilation fails with an error equivalent to "too many template arguments" for the extended `AutoPartitioner` form.

- [ ] **Step 3: Extend the primary template**

In `auto_partitioner.hpp`, replace the primary template declaration with:

```cpp
template <typename ArchTag,
          typename OpClass,
          typename Element,
          typename GmemStride,
          typename TileShape_MNK,
          int ThreadCount,
          typename ElementC = Element,
          int GmemAlignmentA = 16,
          int GmemAlignmentB = 16,
          int GmemAlignmentC = 16,
          typename Enable = void>
struct AutoPartitioner
{
    static_assert(sizeof(Element) == 0,
                  "[AutoPartitioner] Unsupported parameters! Check ArchTag, OpClass, Element, or alignment.");
};
```

- [ ] **Step 4: Update SM80 partial specialization signatures**

In `sm80_policy.hpp`, change both SM80 partial specialization headers to this shape:

```cpp
template <typename Element,
          typename GmemStride,
          typename TileShape_MNK,
          int ThreadCount,
          typename ElementC,
          int GmemAlignmentA,
          int GmemAlignmentB,
          int GmemAlignmentC>
struct AutoPartitioner<cutlass::arch::Sm80,
                       cutlass::arch::OpClassSimt,
                       Element,
                       GmemStride,
                       TileShape_MNK,
                       ThreadCount,
                       ElementC,
                       GmemAlignmentA,
                       GmemAlignmentB,
                       GmemAlignmentC,
                       std::enable_if_t<detail::IsSm80SimtElement<Element>::value>>
{
    using RoleA = detail::Sm80SimtRoleA<Element, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentA>;
    using RoleB = detail::Sm80SimtRoleB<Element, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentB>;
    using RoleC = detail::Sm80SimtRoleC<Element, ElementC, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentC>;
};
```

Use the same parameter list for `OpClassTensorOp`, mapping to:

```cpp
using RoleA = detail::Sm80TensorOpRoleA<Element, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentA>;
using RoleB = detail::Sm80TensorOpRoleB<Element, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentB>;
using RoleC = detail::Sm80TensorOpRoleC<Element, ElementC, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentC>;
```

- [ ] **Step 5: Add temporary role template parameters**

Update the SM80 role declarations so the specializations compile before later tasks fill in the alignment behavior:

```cpp
template <class Element, class GmemStride, int TileMN, int TileK, int ThreadCount, int GmemAlignmentBytes>
struct Sm80SimtMainloopRole;

template <class Element, class GmemStride, class TileShape_MNK, int ThreadCount, int GmemAlignmentBytes>
struct Sm80SimtRoleA;

template <class Element, class GmemStride, class TileShape_MNK, int ThreadCount, int GmemAlignmentBytes>
struct Sm80SimtRoleB;

template <class Element, class ElementC, class GmemStride, class TileShape_MNK, int ThreadCount, int GmemAlignmentBytes>
struct Sm80SimtRoleC;

template <class Element, class GmemStride, int TileMN, int TileK, int ThreadCount, bool IsRoleA, int GmemAlignmentBytes>
struct Sm80TensorOpMainloopRole;

template <class Element, class GmemStride, class TileShape_MNK, int ThreadCount, int GmemAlignmentBytes>
struct Sm80TensorOpRoleA;

template <class Element, class GmemStride, class TileShape_MNK, int ThreadCount, int GmemAlignmentBytes>
struct Sm80TensorOpRoleB;

template <class Element, class ElementC, class GmemStride, class TileShape_MNK, int ThreadCount, int GmemAlignmentBytes>
struct Sm80TensorOpRoleC;
```

Apply these parameters to the existing definitions and keep current behavior inside the bodies for this task.

- [ ] **Step 6: Update SM100/SM120 partial specialization signatures**

In `sm100_policy.hpp`, update every SM100 and SM120 `AutoPartitioner` partial specialization to accept:

```cpp
typename ElementC,
int GmemAlignmentA,
int GmemAlignmentB,
int GmemAlignmentC
```

between `ThreadCount` and `Enable`. Keep the role aliases using current role templates for this task. For SM120 SIMT aliases to SM80 roles, pass the new alignment parameters:

```cpp
using RoleA = detail::Sm80SimtRoleA<Element, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentA>;
using RoleB = detail::Sm80SimtRoleB<Element, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentB>;
using RoleC = detail::Sm80SimtRoleC<Element, ElementC, GmemStride, TileShape_MNK, ThreadCount, GmemAlignmentC>;
```

- [ ] **Step 7: Run the API compatibility test**

Run the command from Step 2 again.

Expected: compilation succeeds and creates `/tmp/test_sm80_policy_static_checks`.

- [ ] **Step 8: Run legacy examples to verify compatibility**

Run:

```bash
/usr/local/cuda/bin/nvcc -std=c++17 \
  -Iinclude \
  -Iinclude/cutlass/transform/collective/auto_partitioner \
  -Iinclude/cutlass/transform/collective/auto_partitioner/examples \
  -arch=sm_120 \
  include/cutlass/transform/collective/auto_partitioner/examples/sm80_tensorop_example.cu \
  -o /tmp/sm80_tensorop_api_compat

/usr/local/cuda/bin/nvcc -std=c++17 \
  -Iinclude \
  -Iinclude/cutlass/transform/collective/auto_partitioner \
  -Iinclude/cutlass/transform/collective/auto_partitioner/examples \
  -arch=sm_120 \
  include/cutlass/transform/collective/auto_partitioner/examples/sm80_simt_example.cu \
  -o /tmp/sm80_simt_api_compat
```

Expected: both compile.

- [ ] **Step 9: Commit Task 1**

Run:

```bash
git add include/cutlass/transform/collective/auto_partitioner/auto_partitioner.hpp \
        include/cutlass/transform/collective/auto_partitioner/arch/sm80_policy.hpp \
        include/cutlass/transform/collective/auto_partitioner/arch/sm100_policy.hpp \
        include/cutlass/transform/collective/auto_partitioner/test_sm80_policy_static_checks.cu
git commit -m "feat: extend autopartitioner interface compatibly"
```

## Task 2: Make Gmem Vector Alignment Physical-Alignment-Aware

**Files:**
- Modify: `include/cutlass/transform/collective/auto_partitioner/arch/sm80_policy.hpp`
- Modify: `include/cutlass/transform/collective/auto_partitioner/arch/sm100_policy.hpp`
- Modify: `include/cutlass/transform/collective/auto_partitioner/test_sm80_policy_static_checks.cu`

- [ ] **Step 1: Add failing static alignment checks**

Add these declarations and assertions before `int main()` in `test_sm80_policy_static_checks.cu`:

```cpp
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
```

- [ ] **Step 2: Run the static checks and verify failure**

Run:

```bash
/usr/local/cuda/bin/nvcc -std=c++17 \
  -Iinclude \
  -Iinclude/cutlass/transform/collective/auto_partitioner \
  -Iinclude/cutlass/transform/collective/auto_partitioner/examples \
  -arch=sm_120 \
  include/cutlass/transform/collective/auto_partitioner/test_sm80_policy_static_checks.cu \
  -o /tmp/test_sm80_policy_static_checks
```

Expected: compilation fails because `GmemToSmemAlignmentBytes` is missing or because 4-byte physical alignment still selects 16-byte copies.

- [ ] **Step 3: Replace GmemVectorAlignment**

In `sm80_policy.hpp`, replace `GmemVectorAlignment` with:

```cpp
template <class Element, int ContiguousElements, int MaxAlignmentBytes = 16>
struct GmemVectorAlignment
{
    static_assert(MaxAlignmentBytes == 4 || MaxAlignmentBytes == 8 || MaxAlignmentBytes == 16,
                  "Gmem alignment must be one of 4, 8, or 16 bytes.");

    static constexpr int ElementBytes = int(sizeof(Element));
    static constexpr int Align16 = (MaxAlignmentBytes >= 16 && ElementBytes <= 16 && (16 % ElementBytes) == 0)
                                     ? (16 / ElementBytes)
                                     : 0;
    static constexpr int Align8 = (MaxAlignmentBytes >= 8 && ElementBytes <= 8 && (8 % ElementBytes) == 0)
                                    ? (8 / ElementBytes)
                                    : 0;
    static constexpr int Align4 = (MaxAlignmentBytes >= 4 && ElementBytes <= 4 && (4 % ElementBytes) == 0)
                                    ? (4 / ElementBytes)
                                    : 0;

    static constexpr int value = (Align16 != 0 && (ContiguousElements % Align16) == 0) ? Align16
                               : (Align8 != 0 && (ContiguousElements % Align8) == 0)   ? Align8
                               : (Align4 != 0 && (ContiguousElements % Align4) == 0)   ? Align4
                                                                                       : 0;
    static constexpr int bytes = value * ElementBytes;

    static_assert(value != 0,
                  "No legal cp.async vector width for this element type, tile extent, and physical alignment.");
};
```

- [ ] **Step 4: Thread alignment through SM80 SIMT roles**

Inside `Sm80SimtMainloopRole`, change:

```cpp
static constexpr int VectorAlignmentElements = GmemVectorAlignment<Element, ContiguousDimLength>::value;
static constexpr int GmemToSmemAlignmentElements = VectorAlignmentElements;
```

to:

```cpp
static constexpr int VectorAlignmentElements =
    GmemVectorAlignment<Element, ContiguousDimLength, GmemAlignmentBytes>::value;
static constexpr int VectorAlignmentBytes =
    GmemVectorAlignment<Element, ContiguousDimLength, GmemAlignmentBytes>::bytes;
static constexpr int GmemToSmemAlignmentElements = VectorAlignmentElements;
static constexpr int GmemToSmemAlignmentBytes = VectorAlignmentBytes;
```

Use `GmemToSmemAlignmentElements` to build `AlignmentType` as before.

- [ ] **Step 5: Thread alignment through SM80 TensorOp A/B and RoleC**

In `Sm80TensorOpMainloopRole`, replace the alignment computation with:

```cpp
static constexpr int AlignmentElements =
    GmemVectorAlignment<Element, ContiguousDimLength, GmemAlignmentBytes>::value;
static constexpr int AlignmentBytes =
    GmemVectorAlignment<Element, ContiguousDimLength, GmemAlignmentBytes>::bytes;
static constexpr int AlignmentBits = AlignmentBytes * 8;
static constexpr int GmemToSmemAlignmentElements = AlignmentElements;
static constexpr int GmemToSmemAlignmentBytes = AlignmentBytes;
```

In `Sm80TensorOpRoleC`, keep compute staging as-is for this task, but calculate global C alignment with:

```cpp
static constexpr int AlignmentElements =
    GmemVectorAlignment<EpilogueElement, ContiguousDimLength, GmemAlignmentBytes>::value;
static constexpr int AlignmentBytes =
    GmemVectorAlignment<EpilogueElement, ContiguousDimLength, GmemAlignmentBytes>::bytes;
static constexpr int AlignmentBits = AlignmentBytes * 8;
static constexpr int GmemToSmemAlignmentBytes = AlignmentBytes;
```

- [ ] **Step 6: Update SM100 alignment wrappers**

In `sm100_policy.hpp`, replace:

```cpp
template <class Element, int ContiguousElements>
struct Sm100GmemVectorAlignment : GmemVectorAlignment<Element, ContiguousElements>
{
};
```

with:

```cpp
template <class Element, int ContiguousElements, int MaxAlignmentBytes = 16>
struct Sm100GmemVectorAlignment : GmemVectorAlignment<Element, ContiguousElements, MaxAlignmentBytes>
{
};
```

Update SM100/SM120 call sites to pass the relevant `GmemAlignmentBytes` where those role templates have it. For roles that do not yet accept alignment, keep the default third argument omitted.

- [ ] **Step 7: Run static checks and legacy examples**

Run:

```bash
/usr/local/cuda/bin/nvcc -std=c++17 \
  -Iinclude \
  -Iinclude/cutlass/transform/collective/auto_partitioner \
  -Iinclude/cutlass/transform/collective/auto_partitioner/examples \
  -arch=sm_120 \
  include/cutlass/transform/collective/auto_partitioner/test_sm80_policy_static_checks.cu \
  -o /tmp/test_sm80_policy_static_checks

/usr/local/cuda/bin/nvcc -std=c++17 \
  -Iinclude \
  -Iinclude/cutlass/transform/collective/auto_partitioner \
  -Iinclude/cutlass/transform/collective/auto_partitioner/examples \
  -arch=sm_120 \
  include/cutlass/transform/collective/auto_partitioner/examples/sm80_simt_example.cu \
  -o /tmp/sm80_simt_alignment
```

Expected: both compile.

- [ ] **Step 8: Commit Task 2**

Run:

```bash
git add include/cutlass/transform/collective/auto_partitioner/arch/sm80_policy.hpp \
        include/cutlass/transform/collective/auto_partitioner/arch/sm100_policy.hpp \
        include/cutlass/transform/collective/auto_partitioner/test_sm80_policy_static_checks.cu
git commit -m "fix: make autopartitioner gmem alignment explicit"
```

## Task 3: Implement Role-Aware LDSM Routing

**Files:**
- Modify: `include/cutlass/transform/collective/auto_partitioner/arch/sm80_policy.hpp`
- Modify: `include/cutlass/transform/collective/auto_partitioner/test_sm80_policy_static_checks.cu`

- [ ] **Step 1: Add failing role-aware ldmatrix checks**

Add these assertions before `int main()` in `test_sm80_policy_static_checks.cu`:

```cpp
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
```

- [ ] **Step 2: Run static checks and verify failure**

Run the static-check `nvcc` command from Task 2 Step 7.

Expected: compilation fails because `SmemToRegNeedTranspose` is not exposed or because the selector still only takes `IsMnMajor`.

- [ ] **Step 3: Add MmaOperandContiguity traits**

In `sm80_policy.hpp`, add after `Sm80TensorOpTraits` declarations:

```cpp
template <class MmaOperation, bool IsRoleA>
struct MmaOperandContiguity;

template <bool IsRoleA>
struct MmaOperandContiguity<cute::SM80_16x8x16_F32F16F16F32_TN, IsRoleA>
{
    static constexpr bool RequiresMnMajor = false;
};

template <bool IsRoleA>
struct MmaOperandContiguity<cute::SM80_16x8x16_F32BF16BF16F32_TN, IsRoleA>
{
    static constexpr bool RequiresMnMajor = false;
};

template <bool IsRoleA>
struct MmaOperandContiguity<cute::SM80_16x8x8_F32TF32TF32F32_TN, IsRoleA>
{
    static constexpr bool RequiresMnMajor = false;
};

template <bool IsRoleA>
struct MmaOperandContiguity<cute::SM80_16x8x32_S32S8S8S32_TN, IsRoleA>
{
    static constexpr bool RequiresMnMajor = false;
};

template <bool IsRoleA>
struct MmaOperandContiguity<cute::SM80_16x8x32_S32U8U8S32_TN, IsRoleA>
{
    static constexpr bool RequiresMnMajor = false;
};
```

- [ ] **Step 4: Replace ldmatrix copy selector**

Replace the current `Sm80TensorOpLdsmCopyOperation` and ldmatrix `Sm80TensorOpSmemCopyOperation` declarations with:

```cpp
template <bool NeedTranspose>
struct Sm80TensorOpLdsmCopyOperation;

template <>
struct Sm80TensorOpLdsmCopyOperation<false>
{
    using type = cute::SM75_U32x4_LDSM_N;
};

template <>
struct Sm80TensorOpLdsmCopyOperation<true>
{
    using type = cute::SM75_U16x8_LDSM_T;
};

template <class Element,
          class MmaOperation,
          bool IsRoleA,
          bool SmemIsMnMajor,
          int AlignmentElements,
          bool UseLdMatrix>
struct Sm80TensorOpSmemCopyOperation;

template <class Element, class MmaOperation, bool IsRoleA, bool SmemIsMnMajor, int AlignmentElements>
struct Sm80TensorOpSmemCopyOperation<Element, MmaOperation, IsRoleA, SmemIsMnMajor, AlignmentElements, true>
{
    static constexpr bool MmaRequiresMnMajor =
        MmaOperandContiguity<MmaOperation, IsRoleA>::RequiresMnMajor;
    static constexpr bool NeedTranspose = (SmemIsMnMajor != MmaRequiresMnMajor);
    using type = typename Sm80TensorOpLdsmCopyOperation<NeedTranspose>::type;
};

template <class Element, class MmaOperation, bool IsRoleA, bool SmemIsMnMajor, int AlignmentElements>
struct Sm80TensorOpSmemCopyOperation<Element, MmaOperation, IsRoleA, SmemIsMnMajor, AlignmentElements, false>
{
    static constexpr int AlignmentBits = AlignmentElements * int(sizeof(Element)) * 8;
    static constexpr bool NeedTranspose = false;
    using type = cute::AutoVectorizingCopyWithAssumedAlignment<AlignmentBits>;
};
```

- [ ] **Step 5: Wire the selector in Sm80TensorOpMainloopRole**

Inside `Sm80TensorOpMainloopRole`, add:

```cpp
using MmaOperation = typename Sm80TensorOpTraits<Element>::MmaOperation;
using SmemCopySelector =
    Sm80TensorOpSmemCopyOperation<Element, MmaOperation, IsRoleA, IsMnMajor, AlignmentElements, UseLdMatrix>;
static constexpr bool SmemToRegNeedTranspose = SmemCopySelector::NeedTranspose;
```

Then replace `SmemToRegCopyOperation` with:

```cpp
using SmemToRegCopyOperation = typename SmemCopySelector::type;
```

- [ ] **Step 6: Run static checks and tensorop example**

Run:

```bash
/usr/local/cuda/bin/nvcc -std=c++17 \
  -Iinclude \
  -Iinclude/cutlass/transform/collective/auto_partitioner \
  -Iinclude/cutlass/transform/collective/auto_partitioner/examples \
  -arch=sm_120 \
  include/cutlass/transform/collective/auto_partitioner/test_sm80_policy_static_checks.cu \
  -o /tmp/test_sm80_policy_static_checks

/usr/local/cuda/bin/nvcc -std=c++17 \
  -Iinclude \
  -Iinclude/cutlass/transform/collective/auto_partitioner \
  -Iinclude/cutlass/transform/collective/auto_partitioner/examples \
  -arch=sm_120 \
  include/cutlass/transform/collective/auto_partitioner/examples/sm80_tensorop_example.cu \
  -o /tmp/sm80_tensorop_role_aware

/tmp/sm80_tensorop_role_aware
```

Expected: compilation succeeds; runtime prints `max_abs_error = 0`.

- [ ] **Step 7: Commit Task 3**

Run:

```bash
git add include/cutlass/transform/collective/auto_partitioner/arch/sm80_policy.hpp \
        include/cutlass/transform/collective/auto_partitioner/test_sm80_policy_static_checks.cu
git commit -m "fix: make sm80 ldmatrix routing role aware"
```

## Task 4: Add Dynamic LDSM Swizzle Rows

**Files:**
- Modify: `include/cutlass/transform/collective/auto_partitioner/arch/sm80_policy.hpp`
- Modify: `include/cutlass/transform/collective/auto_partitioner/test_sm80_policy_static_checks.cu`

- [ ] **Step 1: Add failing TileK=32 ldmatrix checks**

Add these declarations before `int main()` in `test_sm80_policy_static_checks.cu`:

```cpp
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
```

- [ ] **Step 2: Run static checks and verify failure**

Run the static-check `nvcc` command from Task 2 Step 7.

Expected: compilation fails because `UseLdMatrix` is false for `TileK=32` or `SwizzleBase` is missing.

- [ ] **Step 3: Add swizzle row helper**

In `sm80_policy.hpp`, add before `Sm80TensorOpSmemLayoutSelector`:

```cpp
template <class Element, int TileK>
struct Sm80TensorOpSwizzleRow
{
    static constexpr int RowBytes = TileK * int(sizeof(Element));
    static constexpr int Bytes = (RowBytes >= 128 && (RowBytes % 128) == 0) ? 128
                               : (RowBytes >= 64 && (RowBytes % 64) == 0)   ? 64
                               : (RowBytes >= 32 && (RowBytes % 32) == 0)   ? 32
                                                                            : 0;
    static constexpr int Base = (Bytes == 128) ? 3 : (Bytes == 64) ? 2 : (Bytes == 32) ? 1 : 0;
    static constexpr int Elements = Bytes / int(sizeof(Element));
    static constexpr bool Supported = (Bytes != 0);
};
```

- [ ] **Step 4: Replace swizzled layout selector with dynamic rows**

Replace the K-major `Sm80TensorOpSmemLayoutSelector<Element, TileMN, TileK, true, false>` specialization with:

```cpp
template <class Element, int TileMN, int TileK>
struct Sm80TensorOpSmemLayoutSelector<Element, TileMN, TileK, true, false>
{
    static constexpr int SwizzleBase = Sm80TensorOpSwizzleRow<Element, TileK>::Base;
    static constexpr int RowElements = Sm80TensorOpSwizzleRow<Element, TileK>::Elements;
    static_assert(Sm80TensorOpSwizzleRow<Element, TileK>::Supported,
                  "LdMatrix shared layout requires a 32, 64, or 128 byte row.");

    using SwizzleAtom = decltype(cute::composition(
        cute::Swizzle<SwizzleBase, 3, 3>{},
        cute::Layout<cute::Shape<cute::_8, cute::Int<RowElements>>,
                     cute::Stride<cute::Int<RowElements>, cute::_1>>{}));
    using type = decltype(cute::tile_to_shape(SwizzleAtom{}, cute::Shape<cute::Int<TileMN>, cute::Int<TileK>>{}));
};
```

Replace the MN-major specialization with:

```cpp
template <class Element, int TileMN, int TileK>
struct Sm80TensorOpSmemLayoutSelector<Element, TileMN, TileK, true, true>
{
    static constexpr int SwizzleBase = Sm80TensorOpSwizzleRow<Element, TileK>::Base;
    static constexpr int RowElements = Sm80TensorOpSwizzleRow<Element, TileK>::Elements;
    static_assert(Sm80TensorOpSwizzleRow<Element, TileK>::Supported,
                  "LdMatrix shared layout requires a 32, 64, or 128 byte row.");

    using SwizzleAtom = decltype(cute::composition(
        cute::Swizzle<SwizzleBase, 3, 3>{},
        cute::Layout<cute::Shape<cute::Int<RowElements>, cute::_8>,
                     cute::Stride<cute::_1, cute::Int<RowElements>>>{}));
    using type = decltype(cute::tile_to_shape(SwizzleAtom{}, cute::Shape<cute::Int<TileMN>, cute::Int<TileK>>{}));
};
```

- [ ] **Step 5: Relax UseLdMatrix**

Inside `Sm80TensorOpMainloopRole`, replace the `UseLdMatrix` expression with:

```cpp
static constexpr bool UseLdMatrix =
    (std::is_same<Element, cutlass::half_t>::value || std::is_same<Element, cutlass::bfloat16_t>::value)
    && (TileMN % 8 == 0) && Sm80TensorOpSwizzleRow<Element, TileK>::Supported;
static constexpr int SwizzleBase = UseLdMatrix ? Sm80TensorOpSwizzleRow<Element, TileK>::Base : 0;
```

- [ ] **Step 6: Run static checks**

Run the static-check `nvcc` command from Task 2 Step 7.

Expected: compilation succeeds and TileK=32 asserts pass.

- [ ] **Step 7: Commit Task 4**

Run:

```bash
git add include/cutlass/transform/collective/auto_partitioner/arch/sm80_policy.hpp \
        include/cutlass/transform/collective/auto_partitioner/test_sm80_policy_static_checks.cu
git commit -m "feat: support dynamic sm80 ldmatrix swizzle rows"
```

## Task 5: Split TensorOp RoleC Compute and Output Types

**Files:**
- Modify: `include/cutlass/transform/collective/auto_partitioner/arch/sm80_policy.hpp`
- Modify: `include/cutlass/transform/collective/auto_partitioner/examples/autopartition_example_utils.hpp`
- Modify: `include/cutlass/transform/collective/auto_partitioner/examples/sm80_tensorop_example.cu`
- Modify: `include/cutlass/transform/collective/auto_partitioner/test_sm80_policy_static_checks.cu`

- [ ] **Step 1: Add failing ElementC static checks**

Add these assertions before `int main()` in `test_sm80_policy_static_checks.cu`:

```cpp
static_assert(std::is_same<typename ExtendedPartC::ElementInput, cutlass::half_t>::value,
              "RoleC should expose input element type.");
static_assert(std::is_same<typename ExtendedPartC::ElementCompute, float>::value,
              "RoleC should expose FP32 compute for FP16 TensorOp.");
static_assert(std::is_same<typename ExtendedPartC::ElementOutput, cutlass::half_t>::value,
              "RoleC should expose FP16 global output type.");
static_assert(std::is_same<typename ExtendedPartC::Accumulator, float>::value,
              "Accumulator remains compute type.");
static_assert(ExtendedPartC::OutputAlignmentBytes == 4, "RoleC should use explicit C alignment.");
```

- [ ] **Step 2: Run static checks and verify failure**

Run the static-check `nvcc` command from Task 2 Step 7.

Expected: compilation fails because RoleC aliases are missing or RoleC still aliases epilogue output to accumulator.

- [ ] **Step 3: Refactor Sm80TensorOpRoleC aliases**

In `Sm80TensorOpRoleC`, add these aliases near the top:

```cpp
using ElementInput = Element;
using ElementCompute = typename Sm80TensorOpTraits<Element>::Accumulator;
using ElementOutput = ElementC;
using Accumulator = ElementCompute;
using EpilogueElement = ElementCompute;
using OutputElement = ElementOutput;
```

Remove the old accumulator and epilogue aliases that bind epilogue storage directly to the accumulator type.

- [ ] **Step 4: Keep accumulator staging based on ElementCompute**

Ensure the RoleC shared layout, `SmemToRegCopy`, and `RegToSmemCopy` use `EpilogueElement`. The body should contain:

```cpp
static constexpr int ContiguousDimLength = IsMnMajor ? BlkM : BlkN;
static constexpr int EpilogueAlignmentElements =
    GmemVectorAlignment<EpilogueElement, ContiguousDimLength, 16>::value;
static constexpr int EpilogueAlignmentBits =
    EpilogueAlignmentElements * int(sizeof(EpilogueElement)) * 8;

using SmemToRegCopyOperation = cute::AutoVectorizingCopyWithAssumedAlignment<EpilogueAlignmentBits>;
using RegToSmemCopyOperation = cute::AutoVectorizingCopyWithAssumedAlignment<EpilogueAlignmentBits>;
using SmemToRegCopy = cute::Copy_Atom<SmemToRegCopyOperation, EpilogueElement>;
using RegToSmemCopy = cute::Copy_Atom<RegToSmemCopyOperation, EpilogueElement>;
```

- [ ] **Step 5: Add output global copy traits based on ElementOutput**

Add:

```cpp
static constexpr int OutputAlignmentElements =
    GmemVectorAlignment<ElementOutput, ContiguousDimLength, GmemAlignmentBytes>::value;
static constexpr int OutputAlignmentBytes =
    GmemVectorAlignment<ElementOutput, ContiguousDimLength, GmemAlignmentBytes>::bytes;
static constexpr int OutputAlignmentBits = OutputAlignmentBytes * 8;

using OutputSmemToGmemCopy = decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
                                      VectorizedCopyAtom<ElementOutput, OutputAlignmentElements>,
                                      ThreadCount,
                                      OutputAlignmentElements,
                                      GmemStride,
                                      cute::Int<BlkM>,
                                      cute::Int<BlkN>>());
using SmemToGmemCopy = OutputSmemToGmemCopy;
using SharedToGlobalCopy = SmemToGmemCopy;
```

Remove old global C copy aliases that use `EpilogueElement` as the stored type.

- [ ] **Step 6: Add conversion helper to example utils**

Add this helper to `autopartition_example_utils.hpp`:

```cpp
template <class DstTensor, class SrcTensor>
__device__ void convert_tensor(DstTensor &&dst, SrcTensor const &src)
{
    using Dst = typename cute::remove_cvref_t<DstTensor>::value_type;
    using Src = typename cute::remove_cvref_t<SrcTensor>::value_type;
    cutlass::NumericConverter<Dst, Src> convert;
    CUTE_UNROLL
    for (int i = 0; i < cute::size(dst); ++i) {
        dst(i) = convert(src(i));
    }
}
```

Add this include near the top:

```cpp
#include <cutlass/numeric_conversion.h>
```

- [ ] **Step 7: Update sm80_tensorop_example for explicit output conversion**

In `sm80_tensorop_example.cu`, keep `OutputElement = float` for the existing path, but replace:

```cpp
cute::copy(tCrC, tCgC);
```

with:

```cpp
autopartition::examples::convert_tensor(tCgC, tCrC);
```

Replace the accumulator static assert with:

```cpp
static_assert(std::is_same<typename PartC::ElementCompute, OutputElement>::value,
              "Default SM80 half TensorOp example stores FP32 output.");
static_assert(std::is_same<typename PartC::ElementOutput, OutputElement>::value,
              "Default RoleC output element remains compatible.");
```

- [ ] **Step 8: Run static checks and tensorop example**

Run:

```bash
/usr/local/cuda/bin/nvcc -std=c++17 \
  -Iinclude \
  -Iinclude/cutlass/transform/collective/auto_partitioner \
  -Iinclude/cutlass/transform/collective/auto_partitioner/examples \
  -arch=sm_120 \
  include/cutlass/transform/collective/auto_partitioner/test_sm80_policy_static_checks.cu \
  -o /tmp/test_sm80_policy_static_checks

/usr/local/cuda/bin/nvcc -std=c++17 \
  -Iinclude \
  -Iinclude/cutlass/transform/collective/auto_partitioner \
  -Iinclude/cutlass/transform/collective/auto_partitioner/examples \
  -arch=sm_120 \
  include/cutlass/transform/collective/auto_partitioner/examples/sm80_tensorop_example.cu \
  -o /tmp/sm80_tensorop_elementc

/tmp/sm80_tensorop_elementc
```

Expected: static checks compile; tensorop example prints `max_abs_error = 0`.

- [ ] **Step 9: Commit Task 5**

Run:

```bash
git add include/cutlass/transform/collective/auto_partitioner/arch/sm80_policy.hpp \
        include/cutlass/transform/collective/auto_partitioner/examples/autopartition_example_utils.hpp \
        include/cutlass/transform/collective/auto_partitioner/examples/sm80_tensorop_example.cu \
        include/cutlass/transform/collective/auto_partitioner/test_sm80_policy_static_checks.cu
git commit -m "fix: split sm80 tensorop compute and output types"
```

## Task 6: Add SM80 DMMA Coverage

**Files:**
- Modify: `include/cutlass/transform/collective/auto_partitioner/arch/sm80_policy.hpp`
- Modify: `include/cutlass/transform/collective/auto_partitioner/test_sm80_policy_static_checks.cu`

- [ ] **Step 1: Add failing DMMA static checks**

Add these declarations before `int main()` in `test_sm80_policy_static_checks.cu`:

```cpp
using DoubleTileShape = cute::Shape<cute::Int<16>, cute::Int<16>, cute::Int<8>>;
using DoublePartA = typename autopartition::AutoPartitioner<ArchTag,
                                                            OpClass,
                                                            double,
                                                            StrideA,
                                                            DoubleTileShape,
                                                            ThreadCount>::RoleA;
using DoublePartC = typename autopartition::AutoPartitioner<ArchTag,
                                                            OpClass,
                                                            double,
                                                            StrideC,
                                                            DoubleTileShape,
                                                            ThreadCount>::RoleC;

static_assert(autopartition::detail::IsSm80TensorOpElement<double>::value,
              "SM80 TensorOp should support double through DMMA.");
static_assert(std::is_same<typename autopartition::detail::Sm80TensorOpTraits<double>::MmaOperation,
                           cute::SM80_8x8x4_F64F64F64F64_TN>::value,
              "double TensorOp should bind SM80 DMMA.");
static_assert(std::is_same<typename DoublePartC::Accumulator, double>::value,
              "double TensorOp accumulates in double.");
static_assert(cute::cosize_v<typename DoublePartA::SmemLayout> > 0, "Double RoleA shared layout must instantiate.");
```

- [ ] **Step 2: Run static checks and verify failure**

Run the static-check `nvcc` command from Task 2 Step 7.

Expected: compilation fails because double is not in SM80 TensorOp routing.

- [ ] **Step 3: Add double routing and traits**

In `IsSm80TensorOpElement`, add:

```cpp
std::is_same<Element, double>::value ||
```

before the existing `float` case.

Add this trait specialization after the float/TF32 trait:

```cpp
template <>
struct Sm80TensorOpTraits<double>
{
    using MmaOperation = cute::SM80_8x8x4_F64F64F64F64_TN;
    using Accumulator = double;
};
```

- [ ] **Step 4: Add DMMA contiguity trait**

Add:

```cpp
template <bool IsRoleA>
struct MmaOperandContiguity<cute::SM80_8x8x4_F64F64F64F64_TN, IsRoleA>
{
    static constexpr bool RequiresMnMajor = false;
};
```

- [ ] **Step 5: Ensure double avoids half/bfloat16 tiled-mma selector**

Confirm no double specialization is added to `Sm80TensorOpTiledMmaSelector`. The generic selector must remain:

```cpp
template <class Element, class MmaAtom, class ThreadLayout>
struct Sm80TensorOpTiledMmaSelector
{
    using type = decltype(cute::make_tiled_mma(MmaAtom{}, ThreadLayout{}));
};
```

- [ ] **Step 6: Run static checks**

Run the static-check `nvcc` command from Task 2 Step 7.

Expected: compilation succeeds.

- [ ] **Step 7: Commit Task 6**

Run:

```bash
git add include/cutlass/transform/collective/auto_partitioner/arch/sm80_policy.hpp \
        include/cutlass/transform/collective/auto_partitioner/test_sm80_policy_static_checks.cu
git commit -m "feat: add sm80 dmma autopartitioner traits"
```

## Task 7: Add Runtime Edge-Case Tests

**Files:**
- Modify: `include/cutlass/transform/collective/auto_partitioner/examples/autopartition_example_utils.hpp`
- Create: `include/cutlass/transform/collective/auto_partitioner/test_sm80_tensorop_edge_cases.cu`

- [ ] **Step 1: Add generic diff helpers**

In `autopartition_example_utils.hpp`, add:

```cpp
template <class LhsElement, class RhsElement>
float max_abs_diff(int count, LhsElement const *lhs, RhsElement const *rhs)
{
    float diff = 0.0f;
    for (int i = 0; i < count; ++i) {
        diff = std::max(diff, std::abs(to_float(lhs[i]) - to_float(rhs[i])));
    }
    return diff;
}

template <class Element>
void corrupt_first(std::vector<Element> &data)
{
    if (!data.empty()) {
        data[0] = from_float<Element>(to_float(data[0]) + 1.0f);
    }
}
```

- [ ] **Step 2: Write the failing runtime edge-case test**

Create `include/cutlass/transform/collective/auto_partitioner/test_sm80_tensorop_edge_cases.cu` with this content:

```cpp
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <iostream>
#include <type_traits>
#include <vector>

#include "autopartition_example_utils.hpp"
#include "auto_partitioner_builder.hpp"

using namespace cute;

template <typename PartA,
          typename PartB,
          typename PartC,
          typename InputElement,
          typename OutputElement,
          typename StrideA,
          typename StrideB,
          typename StrideC>
__global__ void sm80_edge_tensorop_kernel(InputElement const *ptr_A,
                                          StrideA stride_A,
                                          InputElement const *ptr_B,
                                          StrideB stride_B,
                                          OutputElement *ptr_C,
                                          StrideC stride_C)
{
    using bM = decltype(size<0>(typename PartA::SmemLayout{}));
    using bN = decltype(size<0>(typename PartB::SmemLayout{}));
    using bK = decltype(size<1>(typename PartA::SmemLayout{}));

    Tensor gA = make_tensor(make_gmem_ptr(ptr_A), make_shape(bM{}, bK{}), stride_A);
    Tensor gB = make_tensor(make_gmem_ptr(ptr_B), make_shape(bN{}, bK{}), stride_B);
    Tensor gC = make_tensor(make_gmem_ptr(ptr_C), make_shape(bM{}, bN{}), stride_C);

    struct SharedStorage
    {
        cute::array_aligned<InputElement, cute::cosize_v<typename PartA::SmemLayout>> smemA;
        cute::array_aligned<InputElement, cute::cosize_v<typename PartB::SmemLayout>> smemB;
    };
    __shared__ SharedStorage smem;

    Tensor sA = make_tensor(make_smem_ptr(smem.smemA.data()), typename PartA::SmemLayout{});
    Tensor sB = make_tensor(make_smem_ptr(smem.smemB.data()), typename PartB::SmemLayout{});

    cooperative_copy<128, 128>(threadIdx.x, gA, sA, typename PartA::GmemToSmemCopy{});
    cooperative_copy<128, 128>(threadIdx.x, gB, sB, typename PartB::GmemToSmemCopy{});
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    typename PartC::TiledMma mma;
    auto thr_mma = mma.get_thread_slice(threadIdx.x);
    Tensor tCgC = thr_mma.partition_C(gC);
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);
    clear(tCrC);

    cooperative_gemm(threadIdx.x,
                     mma,
                     sA,
                     sB,
                     tCrC,
                     identity{},
                     identity{},
                     typename PartA::SmemToRegCopyOperation{},
                     typename PartB::SmemToRegCopyOperation{});

    autopartition::examples::convert_tensor(tCgC, tCrC);
}

bool negative_comparison_sanity()
{
    std::vector<cutlass::half_t> values(16);
    std::vector<cutlass::half_t> reference(16);
    autopartition::examples::fill_pattern(values);
    reference = values;
    autopartition::examples::corrupt_first(values);
    float diff = autopartition::examples::max_abs_diff(int(values.size()), values.data(), reference.data());
    std::cout << "negative_comparison_sanity diff = " << diff << "\n";
    return diff > 0.5f;
}

template <int K, int AlignA, int AlignB, class OutputElement>
bool run_half_case(char const *name)
{
    constexpr int M = 64;
    constexpr int N = 64;
    constexpr int ThreadCount = 128;

    using InputElement = cutlass::half_t;
    using StrideA = cute::Stride<int64_t, cute::_1>;
    using StrideB = cute::Stride<cute::_1, int64_t>;
    using StrideC = cute::Stride<int64_t, cute::_1>;
    using TileShape = cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>;
    using ArchTag = cutlass::arch::Sm80;
    using OpClass = cutlass::arch::OpClassTensorOp;
    using PartA = typename autopartition::AutoPartitioner<ArchTag, OpClass, InputElement, StrideA, TileShape, ThreadCount,
                                                          OutputElement, AlignA, AlignB, 4>::RoleA;
    using PartB = typename autopartition::AutoPartitioner<ArchTag, OpClass, InputElement, StrideB, TileShape, ThreadCount,
                                                          OutputElement, AlignA, AlignB, 4>::RoleB;
    using PartC = typename autopartition::AutoPartitioner<ArchTag, OpClass, InputElement, StrideC, TileShape, ThreadCount,
                                                          OutputElement, AlignA, AlignB, 4>::RoleC;

    static_assert(std::is_same<typename PartC::ElementOutput, OutputElement>::value, "Output type must flow to RoleC.");

    std::vector<InputElement> hA(M * K + 2);
    std::vector<InputElement> hB(N * K + 2);
    std::vector<float> hRef(M * N);
    std::vector<OutputElement> hAuto(M * N);
    autopartition::examples::fill_pattern(hA);
    autopartition::examples::fill_pattern(hB);

    InputElement const *hostA = hA.data() + ((AlignA == 4) ? 2 : 0);
    InputElement const *hostB = hB.data() + ((AlignB == 4) ? 2 : 0);
    autopartition::examples::reference_gemm(M, N, K, hostA, K, 1, hostB, 1, N, hRef.data(), N, 1);

    InputElement *dA_raw = nullptr;
    InputElement *dB_raw = nullptr;
    OutputElement *dC = nullptr;
    cudaMalloc(&dA_raw, (M * K + 2) * sizeof(InputElement));
    cudaMalloc(&dB_raw, (N * K + 2) * sizeof(InputElement));
    cudaMalloc(&dC, M * N * sizeof(OutputElement));
    cudaMemcpy(dA_raw, hA.data(), (M * K + 2) * sizeof(InputElement), cudaMemcpyHostToDevice);
    cudaMemcpy(dB_raw, hB.data(), (N * K + 2) * sizeof(InputElement), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, M * N * sizeof(OutputElement));

    InputElement const *dA = dA_raw + ((AlignA == 4) ? 2 : 0);
    InputElement const *dB = dB_raw + ((AlignB == 4) ? 2 : 0);
    sm80_edge_tensorop_kernel<PartA, PartB, PartC, InputElement, OutputElement>
        <<<dim3(1), dim3(ThreadCount)>>>(dA, make_stride(K, Int<1>{}),
                                         dB, make_stride(Int<1>{}, N),
                                         dC, make_stride(N, Int<1>{}));
    cudaError_t err = cudaDeviceSynchronize();
    if (!autopartition::examples::check_cuda(err, name)) {
        cudaFree(dA_raw);
        cudaFree(dB_raw);
        cudaFree(dC);
        return false;
    }

    cudaMemcpy(hAuto.data(), dC, M * N * sizeof(OutputElement), cudaMemcpyDeviceToHost);
    float diff = autopartition::examples::max_abs_diff(M * N, hAuto.data(), hRef.data());
    std::cout << name << " diff = " << diff << "\n";

    cudaFree(dA_raw);
    cudaFree(dB_raw);
    cudaFree(dC);
    return diff < (std::is_same<OutputElement, cutlass::half_t>::value ? 2.0e-2f : 2.0e-2f);
}

int main()
{
    bool ok = true;
    ok = negative_comparison_sanity() && ok;
    ok = run_half_case<64, 4, 16, float>("half_k64_a4_float_c") && ok;
    ok = run_half_case<64, 16, 16, cutlass::half_t>("half_k64_half_c") && ok;
    ok = run_half_case<32, 16, 16, float>("half_k32_float_c") && ok;
    return ok ? 0 : 1;
}
```

- [ ] **Step 3: Compile and verify failure before implementation is complete**

Run:

```bash
/usr/local/cuda/bin/nvcc -std=c++17 \
  -Iinclude \
  -Iinclude/cutlass/transform/collective/auto_partitioner \
  -Iinclude/cutlass/transform/collective/auto_partitioner/examples \
  -arch=sm_120 \
  include/cutlass/transform/collective/auto_partitioner/test_sm80_tensorop_edge_cases.cu \
  -o /tmp/test_sm80_tensorop_edge_cases
```

Expected: if prior tasks are not complete, compilation fails. After prior tasks are complete, compilation succeeds.

- [ ] **Step 4: Run the edge-case test**

Run:

```bash
/tmp/test_sm80_tensorop_edge_cases
```

Expected output includes:

```text
negative_comparison_sanity diff =
half_k64_a4_float_c diff = 0
half_k64_half_c diff =
half_k32_float_c diff = 0
```

Expected exit code: `0`. The half-output diff may be nonzero but must be below the FP16 tolerance.

- [ ] **Step 5: Commit Task 7**

Run:

```bash
git add include/cutlass/transform/collective/auto_partitioner/examples/autopartition_example_utils.hpp \
        include/cutlass/transform/collective/auto_partitioner/test_sm80_tensorop_edge_cases.cu
git commit -m "test: add sm80 tensorop edge cases"
```

## Task 8: Final Verification and Hardware Evidence

**Files:**
- No required source changes.

- [ ] **Step 1: Run all compile checks**

Run:

```bash
/usr/local/cuda/bin/nvcc -std=c++17 \
  -Iinclude \
  -Iinclude/cutlass/transform/collective/auto_partitioner \
  -Iinclude/cutlass/transform/collective/auto_partitioner/examples \
  -arch=sm_120 \
  include/cutlass/transform/collective/auto_partitioner/test_sm80_policy_static_checks.cu \
  -o /tmp/test_sm80_policy_static_checks

/usr/local/cuda/bin/nvcc -std=c++17 \
  -Iinclude \
  -Iinclude/cutlass/transform/collective/auto_partitioner \
  -Iinclude/cutlass/transform/collective/auto_partitioner/examples \
  -arch=sm_120 \
  include/cutlass/transform/collective/auto_partitioner/test_sm80_tensorop_edge_cases.cu \
  -o /tmp/test_sm80_tensorop_edge_cases

/usr/local/cuda/bin/nvcc -std=c++17 \
  -Iinclude \
  -Iinclude/cutlass/transform/collective/auto_partitioner \
  -Iinclude/cutlass/transform/collective/auto_partitioner/examples \
  -arch=sm_120 \
  include/cutlass/transform/collective/auto_partitioner/examples/sm80_tensorop_example.cu \
  -o /tmp/sm80_tensorop_final

/usr/local/cuda/bin/nvcc -std=c++17 \
  -Iinclude \
  -Iinclude/cutlass/transform/collective/auto_partitioner \
  -Iinclude/cutlass/transform/collective/auto_partitioner/examples \
  -arch=sm_120 \
  include/cutlass/transform/collective/auto_partitioner/examples/sm80_simt_example.cu \
  -o /tmp/sm80_simt_final
```

Expected: all commands exit `0`.

- [ ] **Step 2: Run runtime correctness tests**

Run:

```bash
/tmp/test_sm80_tensorop_edge_cases
/tmp/sm80_tensorop_final
/tmp/sm80_simt_final
```

Expected:

- `test_sm80_tensorop_edge_cases` exits `0`;
- `sm80_tensorop_final` prints `max_abs_error = 0`;
- `sm80_simt_final` prints `max_abs_error = 0`.

- [ ] **Step 3: Check SASS for TensorOp evidence**

Run:

```bash
/usr/local/cuda/bin/cuobjdump --dump-sass /tmp/sm80_tensorop_final | \
  rg -n "LDSM\\.16\\.M88\\.4|LDSM\\.16\\.MT88\\.4|HMMA\\.16816\\.F32|LDGSTS\\.E"
```

Expected: output contains `LDSM.16.M88.4`, `LDSM.16.MT88.4`, `HMMA.16816.F32`, and `LDGSTS.E` lines.

- [ ] **Step 4: Run Nsight Compute bank-conflict sample when available**

Run:

```bash
/usr/local/cuda/bin/ncu \
  --target-processes all \
  --kernel-name regex:sm80_tensorop_autopartition_kernel \
  --launch-count 1 \
  --metrics sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm,sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldgsts,sm__inst_executed_pipe_tensor_subpipe_hmma_op_hmma \
  /tmp/sm80_tensorop_final
```

Expected when permissions allow profiling:

- `sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm.sum` is `0`;
- `sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldgsts.sum` is `0`;
- `sm__inst_executed_pipe_tensor_subpipe_hmma_op_hmma.sum` is nonzero.

If profiling is blocked by permissions, record the profiler error in the final summary and rely on compile/runtime/SASS verification.

- [ ] **Step 5: Compile DMMA for SM80 target**

Run:

```bash
/usr/local/cuda/bin/nvcc -std=c++17 \
  -Iinclude \
  -Iinclude/cutlass/transform/collective/auto_partitioner \
  -Iinclude/cutlass/transform/collective/auto_partitioner/examples \
  -arch=sm_80 \
  include/cutlass/transform/collective/auto_partitioner/test_sm80_policy_static_checks.cu \
  -o /tmp/test_sm80_policy_static_checks_sm80
```

Expected: compilation succeeds. Runtime execution of this binary is only required on SM80-compatible hardware.

- [ ] **Step 6: Commit verification adjustments if any**

If verification required source changes, run:

```bash
git add include/cutlass/transform/collective/auto_partitioner
git commit -m "test: finalize sm80 autopartitioner verification"
```

If no source changes were needed, do not create an empty commit.

## Self-Review

Spec coverage:

- Public API compatibility: Task 1.
- Physical alignment: Task 2 and Task 7.
- Role-aware LDSM: Task 3.
- ElementCompute/ElementOutput split: Task 5 and Task 7.
- Dynamic TileK swizzle: Task 4 and Task 7.
- DMMA: Task 6 and Task 8.
- Negative comparison sanity: Task 7.

Placeholder scan:

- The plan contains concrete file paths, code snippets, commands, and expected outcomes.
- No task depends on an unnamed future helper; helper names and code are specified before use.

Type consistency:

- `ElementC` is the public API parameter.
- RoleC aliases are consistently named `ElementInput`, `ElementCompute`, `ElementOutput`, `Accumulator`, `EpilogueElement`, and `OutputElement`.
- Alignment byte aliases are consistently named `GmemToSmemAlignmentBytes`, `AlignmentBytes`, and `OutputAlignmentBytes`.
