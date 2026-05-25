# AutoPartition Epilogue Layout Refactor Design

## Goal

Refactor the AutoPartitioner SM80 and SM100 TensorOp epilogue layout path so it generates correct, bank-conflict-aware write-back layouts and copy atoms without hand-written padding. The production GEMM examples must use AutoPartitioner-provided layouts, copy atoms, and MMA types for complete GEMM execution and output write-back.

## Scope

This work focuses on SM80 and SM100 AutoPartitioner policy and production examples:

- `include/cutlass/transform/collective/auto_partitioner/arch/sm80_policy.hpp`
- `include/cutlass/transform/collective/auto_partitioner/arch/sm100_policy.hpp`
- `include/cutlass/transform/collective/auto_partitioner/examples/production_gemm/sm80_autopartition_gemm.cu`
- `include/cutlass/transform/collective/auto_partitioner/examples/production_gemm/sm100_autopartition_tma_umma_gemm.cu`
- AutoPartitioner static and unit tests that validate SM80 and SM100 routing, layout, copy atom, ldmatrix, MMA, and epilogue connectivity.

SM120 AutoPartitioner test implementation and tests are removed from the AutoPartitioner policy surface for this refactor. Existing official CUTLASS SM120 examples and unrelated tests remain out of scope.

## Architecture

AutoPartitioner remains a pure compile-time layout generator. Policies expose types and constants only: shared-memory layouts, swizzle layout atoms, tiled MMA types, copy atoms, tiled copies, alignment values, and architecture routing decisions. Policies do not perform runtime element conversion, register processing, epilogue math, memory allocation, or kernel control flow.

The production examples are responsible for runtime GEMM behavior. They allocate memory, tile global tensors, execute mainloop copies and MMA, convert accumulator fragments to the output element type when needed, and use AutoPartitioner-provided epilogue layouts and copy atoms for shared-memory staging and global write-back.

## SM80 TensorOp RoleC

SM80 TensorOp RoleC no longer uses padding-based C/D shared-memory layouts. It introduces an epilogue-specific shared-memory swizzle selector. This selector is independent from A/B shared-memory swizzle because epilogue write-back has different logical row width, vector granularity, participating instruction lanes, and bank-conflict constraints.

The selector derives its layout from:

- `ElementCompute`, the accumulator element type.
- `ElementOutput`, the final global output element type.
- `GmemStride`, which determines whether the output is M-major or N-major.
- `TileShape_MNK`, specifically the block tile `M/N` extents used as the logical row basis.
- `ThreadCount`.
- `GmemAlignmentBytes`.
- The selected copy instruction and legal vector granularity.

The vector granularity is not a default target. It is selected from legal values implied by element size, alignment, logical contiguous extent, and tiled-copy legality. The epilogue swizzle is selected to make the relevant 16-thread shared-memory store/load group conflict-free for the chosen vector width. The layout must preserve the logical `BlkM x BlkN` tile shape while mapping addresses through a swizzle appropriate for the C/D epilogue row width.

If `ElementCompute` and `ElementOutput` differ, RoleC exposes both accumulator staging and output staging types/layouts as needed. The actual numeric conversion is performed by the example kernel in registers before writing the `ElementOutput` fragment to shared memory. RoleC only provides the fixed compile-time parameters needed for that write-back path.

## SM100 TensorOp RoleC

SM100 TensorOp RoleC uses CUTLASS official epilogue builder logic for epilogue tile and shared-memory swizzle selection. It removes RoleC hand-written padding and avoids `Sm100SmemBankPaddingElements` for the tensorop epilogue path.

RoleC uses `cutlass::epilogue::collective::CollectiveBuilder` or the same official SM100 epilogue helper machinery to derive:

- Epilogue tile shape.
- C/D shared-memory swizzle layout atoms.
- TMEM load/copy type where applicable.
- Shared-to-global store path, including TMA store when alignment and schedule support it.
- Register/shared copy atoms required by the exposed epilogue path.

AutoPartitioner wraps these derived official types behind its existing RoleC-style interface so examples continue to consume AutoPartitioner first. Direct calls to official examples are not used.

## Examples

`sm80_autopartition_gemm.cu` remains a complete SM80 TensorOp GEMM example:

- A/B global-to-shared copy uses `PartA` and `PartB`.
- Shared-to-register and MMA use AutoPartitioner-selected ldmatrix and tiled MMA types.
- Accumulator fragments are converted to `ElementOutput` in registers when required.
- Register-to-shared and shared-to-global write-back use `PartC` layouts and copy atoms.
- Optional layout printing includes A/B shared layouts and C/D epilogue write-back layout.
- Numerical verification compares active output against a host reference.

`sm100_autopartition_tma_umma_gemm.cu` remains a complete SM100 TMA/UMMA GEMM example:

- A/B TMA descriptors and shared layouts come from `PartA` and `PartB`.
- UMMA tiled MMA and TMEM accumulator handling use AutoPartitioner-exposed types.
- TMEM unload and final global write-back use `PartC` epilogue interfaces derived from official SM100 epilogue builder logic.
- Numerical verification compares active output against a host reference.

## Testing And Verification

The hard verification target is compile-time correctness and static routing on the current SM120 development machine. Runtime execution of SM80 or SM100 device kernels may be unavailable on SM120 hardware; when unavailable, tests must still compile and validate the generated types.

Verification must include:

- Static tests for SM80 TensorOp RoleC epilogue swizzle selection, no padding-based C/D layouts, selected vector granularity, copy atom type, ldmatrix use for A/B, and tiled MMA instantiation.
- Static tests for SM100 TensorOp RoleC official epilogue-derived layout and copy types, including TMA store exposure when alignment permits.
- Tests confirming SM120 AutoPartitioner test roles and assertions are removed from the AutoPartitioner-specific test surface.
- Compilation of both complete production GEMM examples.
- Runtime numerical verification where the architecture and build target support execution.

Performance acceptance is qualitative on hardware where kernels can run: the refactor must not introduce an obvious severe slowdown relative to the existing AutoPartitioner example path. Since the current available device is SM120, this refactor treats successful compilation plus detailed static validation as the required local acceptance path, with runtime checks left intact for compatible SM80 and SM100 systems.

