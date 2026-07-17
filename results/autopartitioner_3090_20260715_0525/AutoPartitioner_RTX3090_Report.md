# AutoPartitioner v00→latest RTX 3090 Benchmark Report

## Scope and fairness

This report measures every preserved SM80 implementation attempt (v00-v08 and latest) against the controlled official CUTLASS single-stage GEMM. It follows the tutorial harness rather than CUTLASS Profiler heuristic selection.

- Sizes: `256,512,1024,2048,4096,8192`
- Warmup launches: `10`
- Timed iterations: `50`
- Process repeats: `5`
- Primary statistic: median TFLOP/s; execution order reverses on alternating repeats.
- CPU reference: 256 and 512. Later historical sources do not all expose hashes; missing hashes are reported as unknown rather than invented.

## Version manifest

|Version|Attempt|Introducing commit|Source SHA256|
|---|---|---|---|
|v00|baseline_single_stage|`07ee9235d795`|`39a6af9379d24c21…`|
|v01|policy_tiled_g2s|`07ee9235d795`|`0ee8c5b421a709f7…`|
|v02|hoist_g2s_tiled_copy|`07ee9235d795`|`c6f5eb10e0b9788f…`|
|v03|launch_bounds|`07ee9235d795`|`e48207e44417caed…`|
|v04|launch_bounds_3cta|`07ee9235d795`|`c41c589a131a7392…`|
|v05|launch_bounds_4cta|`07ee9235d795`|`593258018e24c630…`|
|v06|shared_union|`07ee9235d795`|`6a1c9afbb1589a2c…`|
|v07|epilogue_autoselect_contract|`27d44ffc970e`|`dd6c37e79bc0bad4…`|
|v08|fair_benchmark_baseline|`15394589212c`|`b7c9c5f7367b7cae…`|
|latest|current_production_head|`a9573ccf65a5`|`57a230709c641299…`|

## Throughput and official comparison

|Version|M=N=K|Median TFLOP/s|Official TFLOP/s|AP / official|CV|Status|
|---|---:|---:|---:|---:|---:|---|
|latest|256|5.559|5.554|100.08%|0.0410|success|
|v00|256|5.340|5.554|96.15%|0.0563|success|
|v01|256|5.522|5.554|99.42%|0.0126|success|
|v02|256|5.412|5.554|97.45%|0.0538|success|
|v03|256|5.573|5.554|100.34%|0.0566|success|
|v04|256|5.448|5.554|98.09%|0.0738|success|
|v05|256|5.425|5.554|97.68%|0.0690|success|
|v06|256|5.445|5.554|98.05%|0.0553|success|
|v07|256|5.372|5.554|96.72%|0.0622|success|
|v08|256|5.511|5.554|99.23%|0.0315|success|
|latest|512|24.545|24.731|99.25%|0.0708|success|
|v00|512|20.971|24.731|84.80%|0.0486|success|
|v01|512|21.527|24.731|87.05%|0.0513|success|
|v02|512|21.593|24.731|87.31%|0.0439|success|
|v03|512|21.809|24.731|88.19%|0.0438|success|
|v04|512|21.701|24.731|87.75%|0.0136|success|
|v05|512|22.140|24.731|89.53%|0.0516|success|
|v06|512|21.737|24.731|87.89%|0.0600|success|
|v07|512|22.482|24.731|90.91%|0.0457|success|
|v08|512|22.367|24.731|90.44%|0.0540|success|
|latest|1024|50.705|50.054|101.30%|0.0076|success|
|v00|1024|45.452|50.054|90.81%|0.0558|success|
|v01|1024|44.147|50.054|88.20%|0.0545|success|
|v02|1024|42.886|50.054|85.68%|0.0599|success|
|v03|1024|45.576|50.054|91.05%|0.0530|success|
|v04|1024|45.354|50.054|90.61%|0.0403|success|
|v05|1024|45.158|50.054|90.22%|0.0535|success|
|v06|1024|50.364|50.054|100.62%|0.0152|success|
|v07|1024|50.607|50.054|101.11%|0.0164|success|
|v08|1024|50.754|50.054|101.40%|0.0138|success|
|latest|2048|62.452|63.918|97.71%|0.0008|success|
|v00|2048|57.394|63.918|89.79%|0.0463|success|
|v01|2048|55.273|63.918|86.47%|0.0458|success|
|v02|2048|55.214|63.918|86.38%|0.0006|success|
|v03|2048|61.169|63.918|95.70%|0.0007|success|
|v04|2048|60.950|63.918|95.36%|0.0006|success|
|v05|2048|60.924|63.918|95.32%|0.0003|success|
|v06|2048|62.184|63.918|97.29%|0.0005|success|
|v07|2048|62.443|63.918|97.69%|0.0007|success|
|v08|2048|62.420|63.918|97.66%|0.0008|success|
|latest|4096|53.394|54.185|98.54%|0.0010|success|
|v00|4096|52.568|54.185|97.02%|0.0039|success|
|v01|4096|51.767|54.185|95.54%|0.0078|success|
|v02|4096|51.718|54.185|95.45%|0.0088|success|
|v03|4096|53.002|54.185|97.82%|0.0041|success|
|v04|4096|53.371|54.185|98.50%|0.0060|success|
|v05|4096|53.308|54.185|98.38%|0.0037|success|
|v06|4096|53.414|54.185|98.58%|0.0011|success|
|v07|4096|53.402|54.185|98.55%|0.0009|success|
|v08|4096|53.417|54.185|98.58%|0.0006|success|
|latest|8192|49.503|48.911|101.21%|0.0012|success|
|v00|8192|48.804|48.911|99.78%|0.0013|success|
|v01|8192|48.062|48.911|98.26%|0.0017|success|
|v02|8192|48.137|48.911|98.42%|0.0009|success|
|v03|8192|49.107|48.911|100.40%|0.0011|success|
|v04|8192|49.124|48.911|100.44%|0.0010|success|
|v05|8192|49.091|48.911|100.37%|0.0008|success|
|v06|8192|49.484|48.911|101.17%|0.0012|success|
|v07|8192|49.476|48.911|101.16%|0.0020|success|
|v08|8192|49.613|48.911|101.43%|0.0020|success|

## Best attempt by size

|M=N=K|Best version|TFLOP/s|Official ratio|
|---:|---|---:|---:|
|256|v03|5.573|100.34%|
|512|latest|24.545|99.25%|
|1024|v08|50.754|101.40%|
|2048|latest|62.452|97.71%|
|4096|v08|53.417|98.58%|
|8192|v08|49.613|101.43%|

## Cross-size aggregate

|Version|Valid sizes|Mean AP/official|
|---|---:|---:|
|latest|6|99.68%|
|v00|6|93.06%|
|v01|6|92.49%|
|v02|6|91.78%|
|v03|6|95.58%|
|v04|6|95.12%|
|v05|6|95.25%|
|v06|6|97.27%|
|v07|6|97.69%|
|v08|6|98.12%|

## Failures and unavailable evidence

No build, correctness, runtime, parse, or timeout failures were recorded.

## Interpretation rules

- A ratio above 100% means the AutoPartitioner attempt exceeded the controlled official single-stage comparator for that shape; it does not claim superiority over every CUTLASS kernel.
- Ratios within one percent are treated as performance parity unless repeat dispersion clearly separates them.
- 256 is dominated by launch and fixed overhead; 4096/8192 better represent sustained GEMM throughput.
- Results apply to this RTX 3090, driver, CUDA toolchain, clock/thermal state, and the preserved source revisions.

## Artifact traceability

Raw process samples are in `raw_samples.csv`; all build and run stdout/stderr are retained under `build_logs/` and `run_logs/`. `environment.json`, `commands.sh`, `build_results.json`, and `version_manifest.json` provide the reproduction record.
