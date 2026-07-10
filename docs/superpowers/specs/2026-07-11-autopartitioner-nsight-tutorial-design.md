# AutoPartitioner Nsight Bottleneck Tutorial Design

## Goal

Create a reproducible, evidence-driven tutorial that teaches how to use NVIDIA Nsight Systems and NVIDIA Nsight Compute directly to diagnose CUDA GEMM bottlenecks. The tutorial uses the AutoPartitioner SM80-style GEMM optimization history as its running case study and demonstrates a complete reasoning chain from an initial performance gap to performance comparable with the controlled CUTLASS baseline.

The tutorial must make the reader capable of diagnosing a new kernel independently. It must not be only a command catalog or a retrospective list of optimizations.

## Hardware And Claims

All new measurements are collected on:

- GPU: NVIDIA GeForce RTX 5060 Ti
- Compute capability: SM120
- Nsight Systems: 2025.3.2
- Nsight Compute: 2025.3.0
- Driver: 580.142

The benchmark named `sm80_autopartition_gemm` uses an SM80-style `mma.sync`, `ldmatrix`, and `cp.async` implementation compiled for and executed on SM120. The tutorial must state this at the beginning and beside result summaries. Measurements from this machine must not be presented as native A100 or RTX 30-series SM80 performance.

The tutorial will also provide the compile and profile command changes required on native SM80 hardware, but those commands are not described as locally verified.

## Teaching Structure

The tutorial combines a reference manual with a chronological case study.

### Diagnostic Model

The reader first learns a hierarchical workflow:

1. Establish correctness and a fair benchmark contract.
2. Use Nsight Systems to determine whether time is lost outside the target kernel or inside it.
3. Use Nsight Compute Speed of Light and workload data to classify the kernel broadly.
4. Test specific hypotheses with focused metric groups.
5. Correlate the confirmed metric with source or SASS.
6. Change one relevant behavior while preserving controlled variables.
7. Rebuild, verify correctness, repeat timing, and re-profile.
8. Keep only improvements that are repeatable and whose metric movement agrees with the proposed mechanism.

This workflow is represented as a decision tree. Every branch names the next native Nsight command and the evidence required to continue down that branch.

### Nsight Systems Curriculum

The Nsight Systems section covers:

- What Nsight Systems measures and what it cannot explain.
- How to add NVTX ranges around initialization, warmup, timed launches, verification, and individual optimization phases.
- Native `nsys profile` commands and the meaning of trace, capture-range, sample, output, overwrite, and CUDA-backtrace options.
- Native `nsys stats` reports for CUDA API summary, GPU kernel summary, GPU memory operations, NVTX ranges, and synchronization.
- How to inspect CPU launch gaps, accidental synchronizations, memory-transfer pollution, warmup contamination, serialization, kernel duration variance, and lack of overlap.
- How to read the GUI timeline and map a timeline event back to source behavior.
- Which observations justify moving to Nsight Compute and which require fixing the host-side benchmark first.

Each lesson includes the literal command, representative unedited terminal output captured on the RTX 5060 Ti, an annotated interpretation, and common false conclusions.

### Nsight Compute Curriculum

The Nsight Compute section proceeds from low-cost broad collection to focused expensive collection:

- Kernel discovery and filtering.
- `basic` and Speed of Light sections.
- Launch configuration and achieved/theoretical occupancy.
- Scheduler issue efficiency and eligible/active warps.
- Warp stall reasons, including barrier, long scoreboard, short scoreboard, MIO throttle, math-pipe throttle, not selected, wait, and dispatch stall.
- Tensor-pipe activity and tensor instruction count.
- DRAM, L2, L1/TEX, shared-memory throughput, sectors, and bytes.
- Shared-memory load/store bank conflicts and conflict ratios.
- Register count, shared-memory use, occupancy limits, waves per SM, and tail effects.
- Source counters, line information, SASS correlation, and instruction sampling where supported.
- Roofline as a consistency check rather than a one-command diagnosis.

For every important metric the tutorial records:

- Exact metric or section name.
- Unit and denominator.
- What hardware event it approximates.
- What high and low values do and do not imply.
- Metrics needed for cross-checking.
- Typical code causes in this GEMM.
- Reasonable next experiment.

The tutorial explicitly distinguishes elapsed-peak percentages from active-cycle percentages and raw counts from normalized rates. Comparisons only use matching launches and identical workloads.

## AutoPartitioner Case Study

The historical `v00` through `v08` AutoPartitioner snapshots are reconstructed from Git commit `15394589` into an isolated worktree or independent source/build area. The current dirty worktree is not reset or overwritten.

The case study includes these optimization stages:

- `v00`: single-stage baseline.
- `v01`: policy-generated tiled global-to-shared copies.
- `v02`: tiled-copy objects and thread slices hoisted out of the K loop.
- `v03` to `v05`: launch-bounds and residency experiments.
- `v06`: shared-memory lifetime reuse with a union.
- `v07`: compile-time selected padding-free epilogue swizzle and complete output mapping contract.
- `v08`: controlled benchmark protocol and focused Nsight diagnosis.
- Failed double-buffer and launch-bound experiments, retained as examples of disproven hypotheses.

For each retained stage the tutorial shows:

1. The performance symptom inherited from the previous stage.
2. The Nsight Systems observation and whether it rules out host overhead.
3. The first Nsight Compute collection.
4. The exact metric values that create a hypothesis.
5. A second independent metric or source view used to confirm it.
6. The relevant source diff.
7. Why that code change should affect those metrics.
8. Correctness and fairness checks.
9. Repeated runtime distribution and throughput change.
10. The post-change profile and whether the predicted counters changed.
11. The next remaining bottleneck.

Historical claims that cannot be reproduced with the current checkout, compiler, or hardware are labeled historical. They are not silently mixed with new measurements.

## Fair Comparison Contract

AutoPartitioner and official CUTLASS measurements must use identical:

- Logical and padded M/N/K.
- Element types and accumulation type.
- A, B, and C memory layouts.
- Tile, warp, instruction shape, stage count, alignment, alpha, and beta where the comparison intends to isolate layout behavior.
- Input seed and input buffers.
- Warmup count, timed iteration count, timing mechanism, stream, and synchronization boundaries.
- Correctness checks and output hashes.
- Clock/power environment and process conditions as far as the local machine permits.

The tutorial distinguishes a layout-controlled official baseline from a fastest-available official implementation. Only the controlled baseline is used to claim that a layout is comparable. A different pipeline depth or kernel family is reported separately.

## Native Commands Only

The learning path does not hide Nsight behind a new wrapper script. It uses direct commands such as:

- `nvcc` or the existing benchmark build command.
- `nsys profile`.
- `nsys stats`.
- `ncu --set ...`.
- `ncu --section ...`.
- `ncu --metrics ...`.
- `ncu --import ...`.

Existing benchmark programs may be used to provide deterministic inputs and timing. No new profiling command wrapper is introduced.

## Artifacts

The implementation produces:

- Main tutorial: `docs/auto_partitioner_nsight_bottleneck_tutorial.md`.
- Raw text exports grouped by tool, version, and problem size.
- `.nsys-rep` and `.ncu-rep` report files for representative stages.
- Images exported or captured from Nsight views where a timeline, source correlation, or roofline is materially clearer than text.
- CSV data and comparison curves for runtime, TFLOP/s, selected stalls, tensor utilization, occupancy, and shared-memory conflicts.
- Source diffs or links for every optimization stage.

Large generated reports may remain under the ignored build/results tree when unsuitable for source control. The tutorial records exact regeneration commands and identifies which compact text/image artifacts are committed.

## Validation

The final validation requires:

- Current AutoPartitioner and official benchmark binaries build successfully.
- Representative correctness runs pass and matching-input/output hashes are recorded.
- Every documented profiling command is executed locally when supported.
- Every quoted new metric can be traced to a saved raw output or report.
- Repeated timing is used for performance claims; a single profiled launch is not used as benchmark timing.
- At least one Nsight Systems report and focused Nsight Compute reports are collected for both AutoPartitioner and controlled official implementations.
- The tutorial ends with an independent unknown-kernel checklist and exercises it once without relying on the historical optimization explanation.

## Success Criteria

The reader should be able to answer all of the following without guessing:

- Which profiler to run first and why.
- Which native command to run next after a given observation.
- What each reported percentage is divided by.
- Whether a high stall percentage is a root cause, a symptom, or harmless backpressure.
- How to distinguish host overhead, memory bandwidth, shared-memory conflicts, dependency latency, synchronization, occupancy limits, and compute saturation.
- How to connect a metric to a source region and formulate a falsifiable optimization hypothesis.
- How to prove that an optimization improved the intended mechanism rather than merely benefiting from noise or changed inputs.
- How the AutoPartitioner evolved through evidence-backed steps to reach performance comparable with the fair official baseline on the stated RTX 5060 Ti test system.
