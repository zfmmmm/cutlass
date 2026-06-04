```bash
tools/auto_partitioner_bench/build_benchmarks.sh
python3 tools/auto_partitioner_bench/run_gemm_sweep.py --arch sm80 --sizes 256,512,1024,2048,4096,8192 --warmup 3 --iterations 10 --repeat-runs 3 --skip-reference --verify-sizes 256,512 --plot --output build/auto_partitioner_bench/results/sm80_sweep_official_singlestage.csv
```