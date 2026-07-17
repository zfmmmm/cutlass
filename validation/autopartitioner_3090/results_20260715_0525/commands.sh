#!/usr/bin/env bash
set -euo pipefail
cd '/sxs/cutlass-autopartitioner-3090'
python3 tools/auto_partitioner_bench/build_history.py --repo . --output '/sxs/cutlass-autopartitioner-3090/results/autopartitioner_3090_20260715_0525'
python3 tools/auto_partitioner_bench/run_history_benchmark.py --build-results '/sxs/cutlass-autopartitioner-3090/results/autopartitioner_3090_20260715_0525'/build_results.json --output '/sxs/cutlass-autopartitioner-3090/results/autopartitioner_3090_20260715_0525' --sizes 256,512,1024,2048,4096,8192 --warmup 10 --iterations 50 --repeat-runs 5 --verify-sizes 256,512
python3 tools/auto_partitioner_bench/report_history.py --results '/sxs/cutlass-autopartitioner-3090/results/autopartitioner_3090_20260715_0525'
