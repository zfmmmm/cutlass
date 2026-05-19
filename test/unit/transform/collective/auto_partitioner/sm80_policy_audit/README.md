# SM80 AutoPartitioner Policy Audit

This folder contains the executable GTest harness for the SM80 AutoPartitioner policy.

| Test / example | Phase | Purpose |
| --- | --- | --- |
| `AutoPartitionerSm80Phase1.StaticRoutingAndThreadTopology` | 1 | Checks FP16, BF16, TF32, FP32-as-TF32, INT8, and UINT8 TensorOp trait routing, plus SIMT/TensorOp 2D thread topology heuristics. |
| `AutoPartitionerSm80Phase2.FourByteAlignedCpAsyncFallbackRuns` | 2 | Launches a 4-byte-aligned half tensor copy through the policy-selected global-to-shared path and verifies the copy runs without misaligned access. |
| `sm80_cp_async_zfill_probe_kernel` | 2 / 5 | Uses `SM80_CP_ASYNC_CACHEALWAYS_ZFILL` with a false predicate to prove the hardware path writes zero instead of propagating dirty data. |
| `AutoPartitionerSm80Phase3.SwizzleLdsmAndPaddingFallback` | 3 | Prints K-major and M/N-major shared layouts, verifies LDSM_N/LDSM_T routing, and checks the TileMN=12 padding fallback selector. |
| `AutoPartitionerSm80Phase4.RegisterPressureAndEpilogueStore` | 4 | Instantiates a 256x128x32 RF accumulator pressure kernel, records the ptxas spill-audit command, and runs a shared-to-global epilogue store. |
| `AutoPartitionerSm80Phase5.MixedPrecisionAndOddShapeZfill` | 5 | Runs TF32 and INT8 TensorOp GEMM tiles, then validates a 61x113x59 predicated ZFILL-style corner case with NaN-padded tails. |

Useful manual probes:

```bash
cmake --build build --target cutlass_test_unit_transform_collective_auto_partitioner_sm80_policy_audit -j 8
./build/test/unit/transform/collective/auto_partitioner/sm80_policy_audit/cutlass_test_unit_transform_collective_auto_partitioner_sm80_policy_audit

/usr/local/cuda/bin/nvcc --expt-relaxed-constexpr -std=c++17 -Iinclude -Itest/unit/common \
  -Itools/util/include -Ibuild/_deps/googletest-src/googletest/include -arch=sm_120 \
  --ptxas-options=-v -c test/unit/transform/collective/auto_partitioner/sm80_policy_audit/sm80_policy_audit.cu \
  -o /tmp/sm80_policy_audit.o
```

For bank-conflict profiling, run the Phase 3 filter under Nsight Compute with:

```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
  ./build/test/unit/transform/collective/auto_partitioner/sm80_policy_audit/cutlass_test_unit_transform_collective_auto_partitioner_sm80_policy_audit \
  --gtest_filter=AutoPartitionerSm80Phase3.*
```
