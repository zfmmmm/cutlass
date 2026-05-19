# AutoPartitioner Production GEMM Examples

这个目录把 AutoPartitioner 的 SM80 与 SM100 路由落成两个可编译、可运行的端到端 GEMM 示例。示例只使用 CUTLASS/CuTe 的标准架构 Tag 和布局名字，例如 `cutlass::arch::Sm80`、`cutlass::arch::Sm100`、`cutlass::arch::OpClassTensorOp`、`cute::UMMA::Major`，不再引入私有架构包装标签。

## 文件清单

| 文件 | 测试项 | 目的 |
| --- | --- | --- |
| `autopartition_production_common.hpp` | Host 初始化、CPU reference、CUDA Event 计时、NCU 指令提示 | 让 SM80 与 SM100 示例共享同一套精度验证和性能输出格式 |
| `sm80_autopartition_gemm.cu` | SM80 `cp.async`、`ldmatrix`、`mma.sync`、RF 累加、RF->SMEM->GMEM epilogue、奇异尺寸 padding、`cp.async` false-predicate ZFILL probe | 验证 AutoPartitioner 在 Ampere TensorOp 路径下能生成 `Swizzle<3,3,3>` 共享内存布局和安全的非齐整矩阵执行路径 |
| `sm100_autopartition_tma_umma_gemm.cu` | SM100 TMA descriptor host probe、TMA load/store、UMMA、TMEM accumulator、TMA multicast cluster 约束检查、CUDA Event timing | 验证 AutoPartitioner 对 SM100 TMA/UMMA/TMEM 的静态契约，并复用 CUTLASS 官方 Blackwell Tutorial 05 kernel 作为完整设备流水线 |
| `build_examples.sh` | 单命令构建 | 生成两个示例二进制，SM100 需要 CUDA Driver API 链接 `-lcuda` |

## 构建

```bash
cd /home/zfm/Desktop/cutlass
include/cutlass/transform/collective/auto_partitioner/examples/production_gemm/build_examples.sh
```

脚本默认输出到 `build/auto_partitioner_production_gemm`。也可以手动构建：

```bash
/usr/local/cuda/bin/nvcc --expt-relaxed-constexpr -std=c++17 \
  -Iinclude \
  -Iinclude/cutlass/transform/collective/auto_partitioner/examples/production_gemm \
  -arch=sm_80 \
  include/cutlass/transform/collective/auto_partitioner/examples/production_gemm/sm80_autopartition_gemm.cu \
  -o build/auto_partitioner_production_gemm/sm80_autopartition_gemm

/usr/local/cuda/bin/nvcc --expt-relaxed-constexpr -std=c++17 \
  -I. -Iinclude -Itools/util/include -Iexamples/cute/tutorial \
  -Iinclude/cutlass/transform/collective/auto_partitioner/examples/production_gemm \
  -arch=sm_100a \
  include/cutlass/transform/collective/auto_partitioner/examples/production_gemm/sm100_autopartition_tma_umma_gemm.cu \
  -o build/auto_partitioner_production_gemm/sm100_autopartition_tma_umma_gemm \
  -lcuda
```

## 运行

SM80 示例支持 `--m/--n/--k`、`--warmup`、`--iterations`、`--skip-reference`、`--print-layouts`：

```bash
build/auto_partitioner_production_gemm/sm80_autopartition_gemm \
  --m=123 --n=113 --k=59 --warmup=5 --iterations=20 --print-layouts
```

SM100 示例先打印 AutoPartitioner contract 和 `cuTensorMapEncodeTiled` 探针结果，然后进入 CUTLASS 官方 Blackwell Tutorial 05 kernel。wrapper 支持 `--warmup`、`--iterations`、`--skip-reference`，也兼容 Tutorial 的位置参数 `M N K`。Tutorial 05 当前要求问题规模被其 MMA tiler 整除，推荐先用 512x1024x256 或 4096x4096x4096：

```bash
build/auto_partitioner_production_gemm/sm100_autopartition_tma_umma_gemm 512 1024 256 --warmup=2 --iterations=5
build/auto_partitioner_production_gemm/sm100_autopartition_tma_umma_gemm 4096 4096 4096 --skip-reference --warmup=5 --iterations=20
```

## Nsight Compute

SM80 重点看 shared load/store bank conflict 是否被 `Swizzle<3,3,3>` 和 C 矩阵 padding 压住：

```bash
ncu --metrics \
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
build/auto_partitioner_production_gemm/sm80_autopartition_gemm \
--m=1024 --n=1024 --k=1024 --skip-reference
```

SM100 重点看 TMA/UMMA/TMEM epilogue trace 和 shared store/load 冲突。TMA 本身绕过普通 LSU path，冲突指标主要用于确认 epilogue staging 没有意外退化：

```bash
ncu --metrics \
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
build/auto_partitioner_production_gemm/sm100_autopartition_tma_umma_gemm \
512 1024 256
```

## 说明

SM80 示例是直接由 AutoPartitioner 类型驱动的完整 kernel。为保证奇异尺寸安全，Host 端把 `M/N/K` padding 到 tile 边界并把 inactive 区域填零，同时 kernel 里包含独立的 `cp.async` false-predicate ZFILL probe。

SM100 示例的设备端完整 TMA/UMMA/TMEM pipeline 复用 CUTLASS 官方 Tutorial 05 kernel，wrapper 负责把 AutoPartitioner 的 SM100 contract、16B TMA 对齐、TMEM accumulator 类型、TMA store、ClusterShape multicast、`cuTensorMapEncodeTiled` host-side descriptor encoding、CUDA Event timing 和 TFLOPS 输出显式串起来。若要把 SM100 示例扩展到任意非齐整形状，需要在 Tutorial 05 的 host tiler 外层增加与 SM80 示例一致的 padding 或补充 predicated TMA OOB store 边界处理。
