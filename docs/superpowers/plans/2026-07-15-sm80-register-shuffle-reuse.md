# SM80 Register Shuffle Reuse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and prove a real register-only warp shuffle that converts an SM80 FP32 accumulator fragment into the A or B operand layout of a subsequent SM80 TF32 MMA.

**Architecture:** Extend `Sm80TensorOpRoleC` with explicit C-to-A and C-to-B contracts backed by CuTe warp-shuffle copy atoms, not `DefaultCopy`. Keep numeric values in the kernel: convert FP32 accumulator values to `cutlass::tfloat32_t` registers, execute the policy-generated shuffle, and immediately issue a second MMA. Verify logical-coordinate equivalence, end-to-end numerics, and generated `SHFL` instructions without shared-memory staging.

**Tech Stack:** C++17, CUDA 13, CUTLASS, CuTe, GoogleTest, NVCC, cuobjdump/nvdisasm

## Global Constraints

- AutoPartitioner remains a compile-time layout/copy-parameter generator and owns no runtime tensor storage.
- Initial support is limited to SM80 `16x8x8` FP32 accumulator to TF32 A/B operand reuse.
- FP32-to-TF32 conversion occurs in registers in the caller and must not be implemented as a bit reinterpretation.
- The intermediate path between the two MMAs must contain a real warp shuffle and no shared-memory staging.
- Existing epilogue, production GEMM, and unrelated dirty working-tree changes must remain intact.
- Runtime evidence on RTX 5060 Ti SM120 is reported as an SM80-style path compiled for SM120, not native A100 performance.

---

### Task 1: Failing Contract Audit

**Files:**
- Modify: `test/unit/transform/collective/auto_partitioner/sm80_policy_audit/sm80_policy_audit.cu`
- Modify: `test/unit/transform/collective/auto_partitioner/CMakeLists.txt` only if a new test target is required

**Interfaces:**
- Consumes: current `Sm80TensorOpRoleC` register reuse aliases and helper functions
- Produces: compile-time and runtime tests that distinguish a true lane shuffle from `retile` plus `DefaultCopy`

- [ ] **Step 1: Add compile-time expectations for explicit TF32 C-to-A/B contracts**

Add assertions requiring dedicated availability predicates, `cutlass::tfloat32_t` destination elements, and non-`DefaultCopy` copy operations. Keep the existing FP16 rejection assertion.

- [ ] **Step 2: Add a coordinate-permutation CUDA test kernel**

Initialize each source accumulator register with an encoding of its logical matrix coordinate. Invoke the future policy API to convert and shuffle into A and B fragments, then write `(destination coordinate, observed source coordinate)` pairs to global memory. The host test requires every destination coordinate to receive the value from the identical logical source coordinate.

- [ ] **Step 3: Build the focused audit and verify RED**

Run the existing CMake/Ninja target for `sm80_policy_audit`. Expected result: compilation fails because the explicit C-to-A/B shuffle API is missing or because the operation remains `cute::DefaultCopy`. Confirm the failure is caused by the new requirement.

- [ ] **Step 4: Commit the failing test**

Commit only the audit/CMake changes with message `test: require real SM80 accumulator shuffle`.

### Task 2: CuTe Shuffle Mapping Contract

**Files:**
- Modify: `include/cutlass/transform/collective/auto_partitioner/arch/sm80_policy.hpp`
- Test: `test/unit/transform/collective/auto_partitioner/sm80_policy_audit/sm80_policy_audit.cu`

**Interfaces:**
- Consumes: `TiledMma::get_layoutC_TV()`, next MMA A/B TV layouts, CuTe `SM50_Shuffle_U32_2x2Trans_XOR1/XOR4`, and `cutlass::tfloat32_t`
- Produces: `CanRegisterShuffleToOperandA`, `CanRegisterShuffleToOperandB`, `RegisterShuffleOperandElement`, explicit C-to-A/B copy operations, tiled-copy constructors, and source/destination retile helpers

- [ ] **Step 1: Derive the C-to-A/B logical mapping**

Represent source and destination TV layouts through compile-time CuTe layouts. Add static legality checks for 32 lanes, 32-bit values, matching logical coordinate domains, and the supported `16x8x8` TF32 MMA operation.

- [ ] **Step 2: Select real CuTe shuffle atoms**

Compose or sequence the official XOR1/XOR4 2x2 transpose copy atoms needed by the derived mapping. Do not retain a `DefaultCopy` fallback under a true availability predicate. Unsupported mappings produce a focused static assertion when their constructor is instantiated.

- [ ] **Step 3: Expose unambiguous policy APIs**

Add separate A and B constructors and retile helpers. Set `HasRegisterShuffleMapping` from the validated availability predicates. Correct `CanReuseAccumulatorAsOperand` so it also reflects the TF32 destination type and required conversion rather than claiming raw type identity is sufficient.

- [ ] **Step 4: Run the focused audit and verify GREEN**

Build and run the audit on the available GPU. Confirm coordinate mapping passes for every tested lane/value and FP16 remains rejected.

- [ ] **Step 5: Run existing policy audit regressions**

Build all AutoPartitioner policy audit targets and run their CTest entries. Fix only regressions caused by the new contract.

- [ ] **Step 6: Commit the policy implementation**

Commit policy and audit changes with message `feat: add SM80 accumulator operand shuffle`.

### Task 3: Two-MMA Numerical and SASS Proof

**Files:**
- Modify: `test/unit/transform/collective/auto_partitioner/sm80_policy_audit/sm80_policy_audit.cu` or create a focused `.cu` beside it if compile cost warrants separation
- Modify: matching test `CMakeLists.txt` when creating a target
- Create: `docs/auto_partitioner_sm80_register_shuffle_validation.md`

**Interfaces:**
- Consumes: Task 2 C-to-A/B contracts
- Produces: executable chained MMA validation and reproducible disassembly evidence

- [ ] **Step 1: Add a failing chained-MMA correctness test**

Implement a one-warp/tile kernel that computes a first TF32 MMA, converts its FP32 accumulator fragment using CUTLASS numeric conversion, invokes the generated C-to-A shuffle, and uses the result in a second TF32 MMA. Add the analogous C-to-B case when its logical tile domain is valid. Compare against a host reference that applies TF32 rounding to the intermediate matrix. Before wiring the new contract, verify the test fails numerically or cannot compile through the old path.

- [ ] **Step 2: Wire the generated shuffle into the second MMA and verify GREEN**

Run the focused test. Require finite outputs and a tolerance appropriate to two TF32 MMA operations. Ensure changing the intermediate values changes the second result so the test cannot pass while bypassing reuse.

- [ ] **Step 3: Inspect generated SASS**

Build with line information, disassemble the exact kernel, and record the command and relevant raw excerpt. Verify `SHFL` appears before the second `HMMA`/MMA sequence. Inspect the intermediate source region for `STS`, `LDS`, `BAR`, and local-memory spill traffic; fail the validation if it stages the intermediate through memory.

- [ ] **Step 4: Run production regressions**

Run `tools/auto_partitioner_bench/build_benchmarks.sh`, execute the SM80 production example at a correctness size, and run the official baseline with identical arguments. This task does not claim a chained-MMA performance result; it verifies the existing GEMM path remains correct.

- [ ] **Step 5: Document usage and evidence boundary**

Document the new policy API, a minimal call sequence, test commands, numerical output, SASS excerpt, and RTX 5060 Ti SM120 boundary. Clearly distinguish register conversion from lane shuffle.

- [ ] **Step 6: Commit validation and documentation**

Commit with message `test: validate chained SM80 register MMA reuse`.

### Task 4: Final Verification

**Files:**
- Verify only

**Interfaces:**
- Consumes: all prior tasks
- Produces: final evidence summary

- [ ] **Step 1: Rebuild from a clean feature build directory**

Configure the required audit and benchmark targets without relying on stale objects.

- [ ] **Step 2: Run all focused and regression tests**

Capture exit status, numerical error, and device information.

- [ ] **Step 3: Re-run disassembly checks**

Confirm the final binary still contains the expected shuffle-to-MMA instruction ordering and no intermediate shared-memory path.

- [ ] **Step 4: Review the diff**

Run `git diff --check`, inspect the complete branch diff, and verify unrelated files are absent.
