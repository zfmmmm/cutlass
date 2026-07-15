# SM80 Register Shuffle Reuse Design

## Goal

Replace the current metadata-only accumulator-to-operand retile contract with a real warp-register shuffle path for chaining an SM80 FP32 accumulator into the A or B operand of a subsequent SM80 TF32 MMA. The data path must stay in registers, preserve logical matrix coordinates, and be proven by both numerical execution and generated SASS.

## Current Problem

`Sm80TensorOpRoleC` currently advertises register shuffle support, but its operand copy aliases use `cute::DefaultCopy`. `retile_register_to_operand()` only returns `retile_S()` and `retile_D()` tensor views. Those operations change the view expected by a copy but do not themselves exchange values between warp lanes. The declared `SM50_Shuffle_U32_2x2Trans_XOR1` and `SM50_Shuffle_U32_2x2Trans_XOR4` operations are not selected or executed by any production or audit kernel.

Consequently, `HasRegisterShuffleMapping` is currently stronger than the implementation warrants.

## Scope

The first supported chain is:

- source: FP32 accumulator fragment produced by `SM80_16x8x8_F32TF32TF32F32_TN`;
- destination: A or B register operand fragment consumed by a subsequent SM80 TF32 MMA;
- transport: warp-register shuffle through CuTe copy atoms;
- conversion: FP32 to `cutlass::tfloat32_t` in registers before the second MMA;
- no shared-memory staging between the first accumulator and second MMA.

FP32-to-FP16 packing, arbitrary MMA shapes, cross-warp exchange, and SM100 TMEM reuse are outside this change.

## Architecture

### Compile-Time Mapping Contract

The policy compares the source accumulator TV layout with the destination operand TV layout. For every logical coordinate used by the next MMA, the contract identifies the source `(lane, value)` and destination `(lane, value)` pair. A supported mapping selects a composition of CuTe's existing 32-bit shuffle copy operations rather than `cute::DefaultCopy`.

The policy exposes separate A and B contracts because their operand layouts differ. Availability predicates include architecture, MMA operation, source and destination value widths, fragment shape, and whether the selected mapping performs a real lane exchange.

`HasRegisterShuffleMapping` is true only when at least one validated shuffle contract is available. Unsupported combinations fail through a focused compile-time diagnostic instead of silently falling back to a local register copy.

### Type Conversion

The first MMA accumulates into `float`, while TF32 MMA operands are `cutlass::tfloat32_t`. This conversion is numerically meaningful and cannot be replaced by a bit reinterpretation. AutoPartitioner remains a compile-time generator: it exposes the destination element type and conversion contract, while the chained kernel owns the register values and invokes CUTLASS numeric conversion before the generated shuffle copy.

No global or shared-memory object is created by AutoPartitioner.

### Execution Flow

The validation kernel performs:

1. load TF32-compatible A and B fragments;
2. execute the first TF32 MMA into an FP32 accumulator fragment;
3. convert that fragment to `cutlass::tfloat32_t` in registers;
4. invoke the AutoPartitioner-generated C-to-A or C-to-B shuffle copy;
5. feed the resulting operand fragment directly to a second TF32 MMA;
6. store the second accumulator for comparison with a host reference.

A shared-memory implementation may be used only as a test oracle or diagnostic comparison. It is not part of the target data path.

## Interfaces

`Sm80TensorOpRoleC` will expose explicit names for:

- destination operand element type;
- C-to-A shuffle copy plan;
- C-to-B shuffle copy plan;
- availability predicates for A and B;
- constructors for the selected tiled copies;
- helpers that return source and destination tensor views accepted by `cute::copy`.

Existing register-to-output and fusion shared-memory contracts remain unchanged. Existing ambiguous aliases may be retained temporarily for source compatibility only if they resolve to the real shuffle implementation; they must not continue to resolve to `cute::DefaultCopy` while claiming shuffle support.

## Verification

### Compile-Time Audit

The policy audit must prove:

- the FP32/TF32 chain is supported;
- selected C-to-A/B operations are not `cute::DefaultCopy`;
- source and destination layouts cover the same logical coordinates;
- unsupported FP32-to-FP16 reuse remains rejected;
- the selected copy atom has 32-bit register operands and a warp-wide thread layout.

### Runtime Correctness

A CUDA unit test runs a complete two-MMA chain and compares its output with a CPU reference using TF32-rounded intermediate values. Tests cover both C-to-A and C-to-B reuse where the logical shapes are valid.

The test must fail before the production contract is added: either through the missing API/compile-time predicate or through incorrect output from the existing `DefaultCopy` path.

### Generated-Code Proof

Build the validation kernel with line information, disassemble it, and verify:

- `SHFL` instructions occur in the accumulator-to-operand region;
- the second MMA occurs after those shuffle instructions;
- no `STS`, `LDS`, or barrier is used to transport the intermediate matrix between the two MMAs.

Compiler spills are checked separately. Local-memory spill traffic is a failure, but unrelated input/output memory instructions are expected.

### Regression

Run the existing SM80 policy audit, production GEMM correctness test, and benchmark build. The existing one-stage GEMM path must continue to compile and produce the same result because its epilogue contracts are not replaced.

## Evidence Boundary

Runtime validation is performed on an RTX 5060 Ti (SM120) by compiling the SM80-style `mma.sync`/shuffle path for SM120. This proves the contract and generated instruction path on the available machine; it is not an A100 performance claim. Native SM80 compilation remains part of compile verification when supported by the installed CUDA toolkit.
