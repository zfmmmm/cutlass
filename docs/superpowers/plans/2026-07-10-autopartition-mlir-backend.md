# AutoPartition-MLIR-Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and verify a real PyTorch -> torch.export/torch-mlir -> StableHLO -> Linalg-on-Tensors -> C++ MLIR fusion/legalization/runtime lowering -> CUTLASS/CuTe AutoPartition fused CUDA backend pipeline.

**Architecture:** `export_model.py` captures one `GemmBiasGelu` ExportedProgram and uses torch-mlir's version-adapted API to emit the HLO module. The saved HLO module is then passed through the registered StableHLO-to-Linalg pipeline and saved as `exported_linalg.mlir`; no hand-written Linalg file is used as the primary input. `tools/autopartition-opt.cpp` owns three real MLIR passes: use-def fusion into an unregistered-but-real `autopartition.gemm_bias_gelu` operation, attribute-based backend legalization, and runtime-call lowering. `runtime.py` reads the lowered MLIR before dispatching to a PyTorch CUDA extension whose single GEMM kernel performs accumulation, bias load, GELU, conversion, and final store in the same kernel.

**Tech Stack:** Python 3.12, PyTorch/torch.export, torch-mlir, StableHLO, MLIR C++ PassManager, LLVM CMake/Ninja, CUDA 13.0, CUTLASS/CuTe AutoPartitioner, PyTorch C++ extension API, pytest.

## Global Constraints

- Project directory is exactly `autopartition_mlir_backend/` with the user-specified shallow file layout.
- The primary path must start from the PyTorch model and a real `torch.export.ExportedProgram`.
- `exported_stablehlo.mlir` must be emitted by torch-mlir and contain `stablehlo` or `mhlo` operations.
- `exported_linalg.mlir` must be produced by lowering the saved HLO module and contain `linalg.matmul` plus a verifiable bias/GELU epilogue.
- Fusion/legalization/lowering must be real MLIR C++ passes and must not be Python text replacement or dataclass lowering.
- Fallback is explicit, carries a concrete `fallback_reason`, and is selected from `lowered.mlir` by `runtime.py`.
- The legal path requires f16, rank-2 row-major contiguous A/B/output, rank-1 bias `[N]`, static M/N/K multiples of 64, 16-byte alignment, and an SM80-compatible target.
- The CUDA primary path must use `AutoPartitioner::RoleA`, `RoleB`, and `RoleC` and must fuse bias/GELU into the GEMM kernel epilogue; a second epilogue kernel is not acceptable.
- Builds and caches go under `/media/zfm/System/AutoPartition-MLIR-Backend-cache` when possible; repository outputs stay under `autopartition_mlir_backend/`.
- Existing unrelated dirty files are preserved.

## File Map

- `autopartition_mlir_backend/README.md`: Chinese architecture, setup, troubleshooting, limitations, and interview explanation.
- `autopartition_mlir_backend/env_check.py`: environment discovery, version/API probes, mounted-disk cache selection, and actionable dependency setup.
- `autopartition_mlir_backend/export_model.py`: model/input construction, ExportedProgram capture, torch-mlir HLO export, HLO-to-Linalg conversion, and graph printing.
- `autopartition_mlir_backend/mlir_pipeline.py`: build/discover `autopartition-opt`, run the three pass stages, and validate all four artifacts.
- `autopartition_mlir_backend/runtime.py`: extension loading, lowered-MLIR parsing, explicit backend dispatch, correctness comparison, and CUDA/PyTorch benchmarking.
- `autopartition_mlir_backend/run.py`: one-command orchestration and required Chinese status lines.
- `autopartition_mlir_backend/tools/CMakeLists.txt`: MLIR/LLVM-compatible standalone executable build.
- `autopartition_mlir_backend/tools/autopartition-opt.cpp`: the three MLIR passes and tool registration.
- `autopartition_mlir_backend/csrc/autopartition_runtime.cu`: one-kernel AutoPartitioner/CuTe GEMM plus in-register bias/GELU epilogue.
- `autopartition_mlir_backend/mlir/*.mlir`: generated HLO, Linalg, fused, and lowered artifacts.
- `autopartition_mlir_backend/tests/test_pipeline.py`: end-to-end artifact, pass, fallback, and CUDA correctness tests.

### Task 1: Environment and toolchain bootstrap

Detect the active interpreter and existing torch/triton/CUDA installation, then probe torch-mlir's `OutputType`, `export_and_import`, and StableHLO-to-Linalg pass registration without assuming an old API. Try the current cp312 torch-mlir wheel with `--no-deps` in the external cache first. If it cannot import or cannot lower the model with the installed PyTorch, build a matching torch-mlir/LLVM/StableHLO toolchain in the external mounted-disk cache, using `/dev/shm` as temporary build space if required by the root disk limit. Detect the actual RTX 5060 Ti `sm_120`, retain the fixed backend target `sm80`, and compile the extension with an explicit compatible `TORCH_CUDA_ARCH_LIST`.

Verification: environment probes print every required item; a small exported model probe succeeds; `autopartition-opt --help` lists all three custom passes.

### Task 2: Real HLO and Linalg export

Implement `GemmBiasGelu`, `[1024,1024]` f16 input construction, and ExportedProgram capture. Export HLO with torch-mlir's StableHLO output path and save it. Feed that saved module through the StableHLO-to-Linalg pass pipeline (`stablehlo-aggressive-simplification`, `stablehlo-legalize-to-linalg`, `stablehlo-convert-to-signless`, `canonicalize`, with version fallbacks) and save the resulting module. Preserve source locations/semantic evidence needed to identify the expanded default GELU through use-def analysis.

Verification: HLO contains StableHLO/MHLO; Linalg contains `linalg.matmul`, rank-1 bias use, and the GELU scalar math or equivalent source-located expansion; no source file is copied into either artifact.

### Task 3: C++ fusion pass

Register `--autopartition-fuse-gemm-epilogue`. Walk actual MLIR operations and SSA uses, locate `linalg.matmul`, resolve the following broadcast/add and GELU chain, validate single-use and side-effect constraints, infer static M/N/K and element type, and create `autopartition.gemm_bias_gelu` with all required attributes. Keep operands/results typed and retain enough operand metadata for later runtime lowering. The matcher accepts named `linalg.generic`/`linalg.map` pointwise forms and the current torch-mlir GELU expansion, using source locations only as an additional semantic guard rather than a text search.

Verification: a real input artifact transforms to a readable custom operation with M/N/K, dtype, target, tile, thread, alignment, backend, and fallback attributes; malformed or extra-use patterns remain unfused.

### Task 4: Backend legalization and runtime lowering passes

Implement `--autopartition-legalize-backend` to set either `backend = "AutoPartitionBackend"` or `backend = "TorchFallbackBackend"` with a nonempty reason for every fallback. Implement `--autopartition-lower-to-runtime` to replace the custom op with an explicit private function declaration and `func.call` named `@autopartition_fused_gemm_bias_gelu` or `@torch_fallback_gemm_bias_gelu`, while preserving backend/fallback attributes in the module. Build the tool through `find_package(MLIR CONFIG)` and the local/system MLIR CMake package, with an external-cache fallback.

Verification: legal 1024 shapes produce the AutoPartition call; a generated illegal-shape fixture produces the fallback call and reason; `mlir-opt`/the project tool can parse all output.

### Task 5: Single-kernel fused CUDA runtime

Refactor the existing AutoPartitioner kernel pattern into the new project runtime. Keep `RoleA`/`RoleB`/`RoleC` tiled copies and cooperative GEMM, but after accumulator shared-to-register transfer compute the output column from the same epilogue mapping, load `bias[col]`, apply exact GELU in registers, convert to f16, and use `RoleC::OutputRegisterToGlobalCopy` for the only output store. Remove the intermediate global accumulator tensor and the separate epilogue launch. Expose `fused_gemm_bias_gelu(A,B,bias)` and validate CUDA, dtype, shape, contiguity, alignment, and multiples of 64.

Verification: compile/load the extension on the available GPU, inspect the source/build log to ensure one kernel launch, compare against PyTorch with max absolute error below `1e-1`, and benchmark warmup/repeat timings.

### Task 6: MLIR-driven runtime dispatch and orchestration

Make `runtime.py` parse `lowered.mlir` for the call symbol, backend, and fallback reason before selecting the implementation. Make `run.py` execute environment check, tool build, model export, both C++ pass stages, artifact checks, extension load, dispatch, correctness, and benchmark in order, printing the exact required Chinese status messages. Keep fallback visible and never report AutoPartition when the lowered artifact requests fallback.

Verification: `python run.py` generates all four MLIR files and prints the required success lines; an illegal fixture demonstrates explicit PyTorch fallback.

### Task 7: Tests and documentation

Add pytest coverage for capture, HLO, Linalg, fusion attributes, legal and illegal legalization, runtime call names, and conditional CUDA correctness. Document the real route and every file, environment fallback, mounted-disk cache, pass contract, fused kernel contract, current SM80/static-shape limits, and dynamic-shape extension path. Do not add an archive/report framework.

Verification: run `python -m pytest tests/test_pipeline.py -q`, then run the default full demo and re-run the artifact checks from a clean generated-output state.

## Completion Gate

Do not claim completion until the default CUDA run has `actual_backend = AutoPartitionBackend`, `correctness = pass`, a benchmark line, all four required artifacts, and the focused pytest suite passing. If the hardware/compiler combination forces an explicit fallback, report the exact reason and continue fixing the primary path until the target machine's AutoPartition path passes.
