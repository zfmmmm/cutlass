# AutoPartitioner RTX 3090 Historical Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Benchmark v00-v08 and latest SM80 AutoPartitioner implementations against the controlled official CUTLASS single-stage GEMM on an RTX 3090 and return auditable results locally.

**Architecture:** Extend the existing benchmark tooling with a manifest-driven historical builder and runner. Each preserved source snapshot becomes a separately named binary using the same benchmark contract; a Python orchestrator records build/run state and derives all summaries and the report from raw samples.

**Tech Stack:** CUDA C++17, CUTLASS/CuTe, CMake/NVCC, Python 3 standard library, matplotlib when available, SSH/rsync.

## Global Constraints

- Test only SM80 on the RTX 3090.
- Include v00-v08, latest, and the official CUTLASS single-stage comparator.
- Use sizes 256,512,1024,2048,4096,8192; warmup 10; iterations 50; repeats 5.
- Preserve failures and raw logs; never fabricate or omit a version.
- Validate 256 and 512 against CPU reference and compare deterministic hashes for every implementation.
- Copy all results back to the local worktree with a matching content hash.

---

### Task 1: Version manifest and discovery

**Files:**
- Create: `tools/auto_partitioner_bench/history_manifest.py`
- Create: `test/python/auto_partitioner/test_history_manifest.py`

**Interfaces:**
- Produces: `discover_versions(repo: Path) -> list[VersionSpec]` with v00-v08 and latest in stable order.

- [ ] Write tests asserting ten unique versions, required paths, introducing commits, and SHA256 values.
- [ ] Run the tests and confirm failure because the module does not exist.
- [ ] Implement immutable `VersionSpec` records and snapshot discovery.
- [ ] Run the tests and confirm they pass.

### Task 2: Historical source build driver

**Files:**
- Create: `tools/auto_partitioner_bench/build_history.py`
- Create: `test/python/auto_partitioner/test_build_history.py`

**Interfaces:**
- Consumes: `discover_versions()`.
- Produces: compile commands, per-version logs, binary paths, and classified build results in JSON.

- [ ] Write tests for command generation and failure classification using a temporary fake compiler.
- [ ] Run the tests and confirm the expected failure.
- [ ] Implement compilation of official, v00-v08, and latest sources with `-arch=sm_80` and common include/link flags from the existing tutorial build script.
- [ ] Run the tests and confirm they pass.

### Task 3: Fair historical sweep and summaries

**Files:**
- Create: `tools/auto_partitioner_bench/run_history_benchmark.py`
- Create: `test/python/auto_partitioner/test_history_benchmark.py`

**Interfaces:**
- Consumes: build-result JSON and existing binary output contract.
- Produces: `raw_samples.csv`, `summary.csv`, `summary.json`, `official_comparison.csv`, logs, and failure records.

- [ ] Write tests for parser behavior, alternating order, median/std/CV, official ratios, correctness rejection, and partial-version failure retention.
- [ ] Run tests and verify the module is missing.
- [ ] Implement the runner using subprocess timeouts and atomic output writes.
- [ ] Run tests and confirm they pass.

### Task 4: Derived report and plots

**Files:**
- Create: `tools/auto_partitioner_bench/report_history.py`
- Create: `test/python/auto_partitioner/test_history_report.py`

**Interfaces:**
- Consumes: raw and summary artifacts.
- Produces: `AutoPartitioner_RTX3090_Report.md`, PNG plots, and an acceptance JSON.

- [ ] Write tests proving report values come from CSV and failed versions remain visible.
- [ ] Run tests and confirm failure.
- [ ] Implement tables for per-size TFLOP/s, AP/official percentage, best version, stability, and failure analysis.
- [ ] Run tests and confirm they pass.

### Task 5: Remote upload and execution wrapper

**Files:**
- Create: `tools/auto_partitioner_bench/run_3090_history.sh`
- Create: `test/python/auto_partitioner/test_remote_wrapper.py`

**Interfaces:**
- Produces: a single remote command that captures environment, builds, runs, reports, and validates artifacts.

- [ ] Write a static test for strict shell mode, exact tutorial parameters, environment capture, and non-destructive output directories.
- [ ] Run it and confirm failure.
- [ ] Implement the shell wrapper.
- [ ] Run shell syntax and Python tests.

### Task 6: Execute on RTX 3090 and return results

**Files:**
- Create remotely: `/sxs/cutlass-autopartitioner-3090`
- Create locally: `validation/autopartitioner_3090/results_<UTC timestamp>/`

**Interfaces:**
- Consumes: the committed benchmark worktree.
- Produces: the final remote and local artifact sets with matching hashes.

- [ ] Record clean GPU state, CUDA toolchain, disk, and source commits.
- [ ] Upload the Git bundle/worktree without local build artifacts.
- [ ] Run the single remote wrapper to completion, monitoring failures without dropping later versions.
- [ ] Validate expected version/result cardinality and report traceability.
- [ ] Rsync the result directory back and compare content hashes.
- [ ] Run the local artifact validator and commit tooling changes.
