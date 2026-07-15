# AutoPartitioner SM80 Historical Benchmark Design

## Goal

Measure every preserved SM80 AutoPartitioner implementation attempt from v00 through v08 plus the current implementation on one RTX 3090, compare each against the controlled official CUTLASS single-stage GEMM, and retain enough evidence to reproduce or audit every result.

## Version scope

The authoritative historical implementations are the source snapshots already preserved under `tools/auto_partitioner_bench/snapshots/`:

- v00 baseline single-stage
- v01 policy tiled G2S
- v02 hoisted G2S tiled copy
- v03 launch bounds
- v04 launch bounds, three CTA target
- v05 launch bounds, four CTA target
- v06 shared-storage union
- v07 epilogue autoselect contract
- v08 fair-benchmark baseline
- latest, from the production GEMM at branch HEAD

This list represents real implementation attempts. Documentation-only commits, test-only commits, SM100-only changes, and unrelated upstream merges are excluded. The manifest records the snapshot path, introducing commit, SHA256, and label for each version.

## Fair comparison

All versions and the official baseline use the tutorial's controlled SM80 FP16 Tensor Core GEMM setup: identical M/N/K, input generation, strides, alpha/beta, output type, CTA/warp/instruction shape where the snapshot encodes it, one-stage pipeline, CUDA stream, warmup, iteration count, and synchronization boundary. The official comparator is `tools/auto_partitioner_bench/sm80_cutlass_official_gemm.cu`, not a three-stage or heuristic-selected profiler kernel.

Correctness is mandatory before performance is accepted. Sizes 256 and 512 run CPU reference validation. v08 and latest expose deterministic input/output hashes for comparison with the official implementation; v00-v07 predate that instrumentation, so their larger skip-reference sizes explicitly record hash evidence as unavailable. The preserved kernels are not modified merely to manufacture newer telemetry. A build or correctness failure is recorded as a result, never silently omitted.

## Workloads and statistics

The tutorial full sweep is used on the RTX 3090:

```text
M=N=K: 256, 512, 1024, 2048, 4096, 8192
warmup: 10
iterations per CUDA Event interval: 50
process-level repeat runs: 5
```

Execution order alternates by repeat to reduce temperature and DVFS ordering bias. The primary metric is median TFLOP/s across five process runs. Raw runtimes, TFLOP/s samples, min, max, mean, standard deviation, coefficient of variation, AP/official ratio, hashes, and correctness status are retained.

## Remote execution

The local Git history and benchmark worktree are packaged and uploaded to `/sxs/cutlass-autopartitioner-3090`. The remote runner records Git commits, GPU UUID, driver, CUDA compiler/runtime, clocks, temperature, power, CMake, compiler versions, build commands, per-version logs, and return codes. It builds for SM80 only and never overwrites the local source checkout.

## Outputs

Remote and copied-back outputs live under `results/autopartitioner_3090_<UTC timestamp>/` and contain:

- `version_manifest.json`
- `environment.json` and `environment.txt`
- one build log per implementation
- one raw run log per version/size/repeat
- `raw_samples.csv`
- `summary.csv` and `summary.json`
- `official_comparison.csv`
- throughput and relative-performance plots
- `AutoPartitioner_RTX3090_Report.md`
- `commands.sh`

The final report separates compile failures, correctness failures, runtime failures, and valid performance results. It does not infer performance for versions that cannot be built on the current branch/toolchain.

## Acceptance

- All v00-v08 snapshots and latest appear in the manifest.
- The official single-stage baseline is built and measured with the same harness.
- Every version passes its preserved correctness path at 256 and 512; v08/latest additionally match the official deterministic hash contract where emitted.
- Every expected size has five raw samples unless its version failed, in which case a classified failure and log exist.
- Summary numbers are derived from raw CSV, not typed manually.
- Remote results and local copied-back results have matching content hashes.
