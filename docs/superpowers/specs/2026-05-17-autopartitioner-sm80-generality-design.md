# AutoPartitioner SM80 Generality Design

## Context

`AutoPartitioner` currently acts as a template-only blueprint generator for A, B, and C roles. The SM80 policy now covers useful SIMT and TensorOp examples, but several assumptions are still too strong for production use:

- global-memory vector width is inferred from tile shape rather than from the physical alignment contract;
- SM80 TensorOp ldmatrix routing is tied to shared-memory major mode only;
- TensorOp RoleC assumes global C has the accumulator type;
- ldmatrix swizzle selection is locked to 128-byte rows;
- SM80 TensorOp routing omits double-precision tensor cores.

The refactor must keep the existing six-argument `AutoPartitioner` call form compiling while adding explicit hooks for output type and alignment-sensitive code paths.

## Goals

1. Keep existing call sites source-compatible.
2. Let users explicitly express the physical global-memory alignment guarantee for A, B, and C.
3. Split input element, accumulator element, and global C output element in SM80 TensorOp RoleC.
4. Make ldmatrix copy selection role-aware and MMA-operation-aware.
5. Support half/bfloat16 ldmatrix paths for `TileK` row widths of 128, 64, and 32 bytes.
6. Add SM80 double TensorOp support through DMMA.
7. Add tests that first prove the comparison path can fail, then validate the corrected paths.

## Non-Goals

- Do not replace the current blueprint style with a full GEMM kernel builder.
- Do not introduce runtime pointer inspection inside policy templates; physical alignment remains a compile-time contract supplied by the caller.
- Do not redesign SM100, SM120, or TMA/UMMA policies beyond keeping them compatible with the extended `AutoPartitioner` signature.
- Do not promise perfect performance for every `TileK`; the immediate target is correctness and avoiding accidental scalar fallback for supported 32/64/128-byte ldmatrix rows.

## Public Interface

Extend `AutoPartitioner` with defaulted template parameters after `ThreadCount`:

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
struct AutoPartitioner;
```

`Element` remains the A/B input element type. `ElementC` is the global C storage type. The alignment parameters are byte counts and mean "all pointers and relevant dynamic strides supplied to the generated copy path satisfy at least this alignment." Defaults preserve the current behavior.

SM80 role aliases will map those alignment parameters as follows:

- `RoleA`: `Element`, `GmemAlignmentA`
- `RoleB`: `Element`, `GmemAlignmentB`
- `RoleC`: `Element` for compute traits, `ElementC` for global C storage, `GmemAlignmentC` for C global copy

SM100 and SM120 partial specializations should accept the same extended template parameter list and may initially pass `ElementC` through only where C copy traits are present. This keeps the public interface uniform.

## Alignment Model

Replace shape-only `GmemVectorAlignment<Element, ContiguousElements>` with an alignment-aware form:

```cpp
template <class Element, int ContiguousElements, int MaxAlignmentBytes>
struct GmemVectorAlignment;
```

The selected vector width is the largest legal element count that satisfies all three constraints:

- the vector byte width is one of 16, 8, or 4 bytes;
- `ContiguousElements` is divisible by the vector element count;
- `MaxAlignmentBytes` is at least the vector byte width.

If no legal width exists, the static assertion should name the element size, contiguous extent, and alignment contract. This prevents 128-bit `cp.async` generation when the caller only promises 4-byte alignment.

The existing `VectorizedCopyAtom` helper should be driven by the selected alignment elements. `GmemCopyAtom` must use `cute::uint_byte_t<AlignmentElements * sizeof(Element)>`, not a hard-coded 16-byte type.

## Role-Aware LDSM Routing

Introduce an MMA operand contiguity trait:

```cpp
template <class MmaOperation, bool IsRoleA>
struct MmaOperandContiguity;
```

The trait exposes:

```cpp
static constexpr bool RequiresMnMajor;
```

For the existing SM80 `_TN` operations, both A and B should initially require K-major register-facing data, so `RequiresMnMajor = false`. This covers:

- `cute::SM80_16x8x16_F32F16F16F32_TN`
- `cute::SM80_16x8x16_F32BF16BF16F32_TN`
- `cute::SM80_16x8x8_F32TF32TF32F32_TN`
- `cute::SM80_16x8x32_S32S8S8S32_TN`
- `cute::SM80_16x8x32_S32U8U8S32_TN`
- `cute::SM80_8x8x4_F64F64F64F64_TN`

Then make ldmatrix selection a joint decision:

```cpp
NeedTranspose = (SmemIsMnMajor != MmaOperandContiguity<MmaOperation, IsRoleA>::RequiresMnMajor)
```

For half and bfloat16 ldmatrix paths:

- `NeedTranspose == false`: use `cute::SM75_U32x4_LDSM_N`
- `NeedTranspose == true`: use `cute::SM75_U16x8_LDSM_T`

Non-ldmatrix paths continue using `cute::AutoVectorizingCopyWithAssumedAlignment<AlignmentBits>`.

`Sm80TensorOpMainloopRole` must pass `IsRoleA` and `typename Sm80TensorOpTraits<Element>::MmaOperation` into the shared-to-register copy selector. This avoids baking `_TN` assumptions into shared layout alone.

## TensorOp RoleC Type Split

Refactor `Sm80TensorOpRoleC` to expose separate aliases:

```cpp
using ElementInput = Element;
using ElementCompute = typename Sm80TensorOpTraits<Element>::Accumulator;
using ElementOutput = ElementC;
using Accumulator = ElementCompute;
```

`TiledMma` remains based on `ElementInput` and `ElementCompute`. RoleC shared epilogue layout and register/shared copy operations should store `ElementCompute` when they represent accumulator staging. Global C copy traits must use `ElementOutput` and `GmemAlignmentC`.

The policy should expose enough aliases for kernels to make the conversion explicit:

```cpp
using EpilogueElement = ElementCompute;
using OutputElement = ElementOutput;
using OutputSmemToGmemCopy = ... // built with ElementOutput
```

Example kernels should not copy an accumulator fragment directly into a half global tensor. They should create an output fragment or tensor view with `ElementOutput`, convert each value with `cutlass::NumericConverter<ElementOutput, ElementCompute>`, then write global C. For direct register-to-global examples, the conversion can happen in a small local fragment before `cute::copy`.

This fixes FP16 output storage for FP16 input with FP32 accumulation and prevents out-of-bounds FP32 writes into FP16 C buffers.

## Dynamic LDSM Swizzle Rows

Remove the `TileK % 64 == 0` ldmatrix gate. Replace it with row-byte classification for 16-bit ldmatrix-capable types:

- `TileK * sizeof(Element) >= 128` and divisible by 128: 128-byte swizzle row, `cute::Swizzle<3, 3, 3>`
- `TileK * sizeof(Element) >= 64` and divisible by 64: 64-byte swizzle row, `cute::Swizzle<2, 3, 3>`
- `TileK * sizeof(Element) >= 32` and divisible by 32: 32-byte swizzle row, `cute::Swizzle<1, 3, 3>`

For K-major shared layouts, the atom layout should keep K contiguous:

```cpp
Layout<Shape<_8, Int<RowElements>>, Stride<Int<RowElements>, _1>>
```

For MN-major shared layouts, the atom layout should keep MN contiguous:

```cpp
Layout<Shape<Int<RowElements>, _8>, Stride<_1, Int<RowElements>>>
```

`RowElements = RowBytes / sizeof(Element)`. If the row width is not one of the supported classes, fall back to the padded non-ldmatrix layout with a clear static path. This enables `TileK = 32` for FP16/BF16 to remain on ldmatrix with a 64-byte row.

## SM80 DMMA Coverage

Extend SM80 TensorOp routing:

- add `double` to `IsSm80TensorOpElement`;
- add `Sm80TensorOpTraits<double>` with `cute::SM80_8x8x4_F64F64F64F64_TN` and `Accumulator = double`;
- let double use the generic `Sm80TensorOpTiledMmaSelector` path based on `cute::make_tiled_mma`.

Do not reuse half/bfloat16 `Tile<_32, _32, _16>` for double. DMMA has a different instruction shape and register footprint.

## Tests

Add focused CUDA examples or unit-test-style programs under `include/cutlass/transform/collective/auto_partitioner/` or its `examples/` directory, matching the current repository style.

Required validation cases:

1. **Negative comparison sanity**
   - Run a deliberately wrong kernel or deliberately corrupt one output element.
   - Confirm `max_abs_error` is nonzero and the test returns failure.
   - This proves the reference comparison is not always returning zero.

2. **Alignment downgrade**
   - Instantiate SM80 SIMT or TensorOp roles with `GmemAlignmentA = 4` and/or `GmemAlignmentB = 4`.
   - Add static assertions that the selected gmem-to-smem copy alignment is 4 bytes.
   - Use a pointer offset compatible with 4-byte alignment but not 16-byte alignment.
   - Confirm the kernel runs without misaligned-address failure and matches reference.

3. **Role-aware ldmatrix**
   - Static assert both `NeedTranspose` branches for A and B by varying gmem/shared major mode.
   - Run at least one A/B stride combination through GEMM and compare to reference.

4. **FP16 output epilogue**
   - Use FP16 A/B, FP32 accumulation, and FP16 global C.
   - Confirm C allocation is `M * N * sizeof(cutlass::half_t)`.
   - Convert reference values to FP16 before comparison or use a tolerance consistent with FP16 output.

5. **TileK 32 ldmatrix**
   - Instantiate FP16 or BF16 TensorOp with `TileK = 32`.
   - Static assert `UseLdMatrix == true`.
   - Static assert the selected swizzle base is 2 for the 64-byte row case.
   - Run GEMM and compare to reference.

6. **DMMA**
   - Instantiate double SM80 TensorOp with a small legal tile.
   - Compile and run on hardware that supports the generated target, or compile-only when the local GPU cannot execute the target.
   - Compare to double reference when runtime support is available.

Optional hardware evidence:

- use `cuobjdump` or `nvdisasm` to check for `LDSM`/`HMMA`/`DMMA` SASS patterns;
- use Nsight Compute shared-bank-conflict metrics when permissions and hardware support allow it.

## Migration Notes

Existing code using the six-argument form keeps compiling because `ElementC` and alignments have defaults. New code should pass `ElementC` when C storage differs from accumulator storage, and should pass alignment byte counts when using subtensors or pointer offsets that weaken the default 16-byte assumption.

Example:

```cpp
using PartC = typename autopartition::AutoPartitioner<
    cutlass::arch::Sm80,
    cutlass::arch::OpClassTensorOp,
    cutlass::half_t,
    StrideC,
    TileShape,
    128,
    cutlass::half_t,
    16,
    16,
    4>::RoleC;
```

This means FP16 input, FP16 C storage, 16-byte A/B alignment, and 4-byte C alignment.

## Risks

- Extending the primary template requires updating every architecture partial specialization to avoid ambiguous or missing matches.
- Direct register-to-global epilogues may need example-level conversion helpers because CuTe copy atoms do not perform numeric conversion by themselves.
- Some DMMA code paths may compile only for SM80-class targets and may not execute on all local development GPUs.
- Swizzle row selection must be validated carefully; a layout that compiles can still feed ldmatrix with the wrong logical orientation if the atom shape is wrong.
