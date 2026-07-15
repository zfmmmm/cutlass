# GEMM 融合 Lowering Demo

这个目录是一个面向面试讲解的轻量 AI 编译器 lowering demo。它不是完整 AI 编译器，不实现 MLIR、Torch FX、动态图、复杂调度器、cuBLASLt 后端或 autotune 系统。它只演示一条固定链路：

```text
高层表达式
    -> GraphIR
    -> 图优化 pass
    -> fused_gemm_epilogue
    -> LoweredIR
    -> 后端合法性判断和路由
    -> Torch reference / Triton fused kernel / AutoPartition plan
    -> correctness + benchmark
```

固定表达式是：

```python
Y = gelu((A @ B + bias) * scale)
```

其中 `A: [M, K] fp16`，`B: [K, N] fp16`，`bias: [N] fp16`，`scale: fp32 scalar`，输出 `Y: [M, N] fp16`，GEMM accumulator 按 `fp32` 描述。

## 文件说明

为了学习时不用在很多文件之间跳转，现在 Python 主流程只保留三个核心文件：

- `demo.py`：固定 `M=N=K=1024` 的可运行入口，创建输入 tensor，调用 tiny frontend 构图，再进入编译流程。
- `compiler.py`：从上到下放在一个文件里：`Tensor/Node/Graph`、构图、pass、fusion、`LoweredGemmEpilogueOp`、lowering、backend legality、运行、correctness、benchmark。
- `backends.py`：Torch reference、Triton fused kernel、AutoPartition plan backend、AutoPartition CUDA backend。
- `autopartition_sm80_probe.cu`：最小 C++ adapter，实例化仓库里的 `autopartition::AutoPartitioner` 并打印 RoleA/RoleB/RoleC 计划。
- `autopartition_sm80_runtime.cu`：PyTorch CUDA extension backend，使用 AutoPartitioner RoleA/RoleB/RoleC 执行真实 GEMM，并把结果返回 Python。
- `test_compiler_flow.py`：轻量行为测试，覆盖 IR、fusion、lowering 和 AutoPartition plan。

建议先忽略 `__init__.py` 和测试文件。真正学习 lowering 主线时，按下面顺序看。

## 学习顺序

1. `README.md`：先看目标和边界，记住这个 demo 只讲一条固定表达式的 lowering 闭环。
2. `demo.py`：看真实入口。重点是它没有直接把 PyTorch 表达式当主流程，而是先用 `build_demo_graph(M, N, K)` 构造自己的 GraphIR。
3. `compiler.py` 前半段：看 `Tensor`、`Node`、`Graph` 和 `build_demo_graph`。这里对应“高层表达式 -> GraphIR”。
4. `compiler.py` 中段：看 `run_passes`、`infer_shapes`、`infer_dtypes`、`eliminate_dead_code`、`fuse_gemm_epilogue`。重点是 pattern：

```text
gelu(mul(add(matmul(A, B), bias), scale))
    -> fused_gemm_epilogue(A, B, bias, scale)
```

5. `compiler.py` 后半段：看 `LoweredGemmEpilogueOp` 和 `lower_graph`。这里把图节点变成后端合同，明确 `M/N/K`、dtype、layout、acc dtype、tile shape、target SM。
6. `compiler.py` 的 `compile` 函数：按打印顺序看完整流程：Original GraphIR、Optimized GraphIR、LoweredIR、Backend Legality、AutoPartition Plan、执行、正确性、benchmark。
7. `backends.py`：先看 `TorchReferenceBackend`，再看 `TritonFusedGemmBackend`，最后看 `AutoPartitionBackend` 和 `AutoPartitionCudaBackend`。重点是后端只消费 `LoweredGemmEpilogueOp`，不关心高层图怎么来的。
8. `autopartition_sm80_probe.cu`：看 plan 路径如何实例化你的 `AutoPartitioner`，并把 RoleA/RoleB/RoleC 的选择打印回 Python。
9. `autopartition_sm80_runtime.cu`：看真实 CUDA 计算路径。它用 AutoPartitioner 的 RoleA/RoleB/RoleC 做 GEMM 主计算，再用一个小 epilogue kernel 做 bias、scale、GELU、fp16 写回。

## GraphIR 是什么

GraphIR 是这个 demo 自己定义的小计算图。它只表达算子依赖和 tensor metadata，例如：

```text
%0 = matmul(%A, %B)
%1 = add(%0, %bias)
%2 = mul(%1, %scale)
%Y = gelu(%2)
```

它的目标不是覆盖所有 PyTorch 语义，而是让你能清楚说明：高层表达式先进入一个框架无关的中间表达，再由 pass 和 lowering 逐步变成后端友好的形式。

## Pass 做了什么

`compiler.py` 中的核心 pass 是 pattern fusion。它只识别这一种模式：

```text
gelu(mul(add(matmul(A, B), bias), scale))
```

识别后替换成：

```text
%Y = fused_gemm_epilogue(%A, %B, %bias, %scale) {
    epilogue = bias_scale_gelu
    original_pattern = gelu((A @ B + bias) * scale)
}
```

这个 pass 的意义是把多个高层 op 合成一个后端更容易实现的 fused GEMM epilogue 合同。

## Lowering 做了什么

`compiler.py` 里的 `lower_graph` 把 fused node 变成 `LoweredGemmEpilogueOp`，记录：

- `M/N/K`
- A/B/bias/scale/out dtype
- `acc_dtype = fp32`
- A/B/out layout
- epilogue 类型
- target
- `allow_cublaslt_rewrite`

第一版只支持 `row_major`，避免把 demo 写成复杂 layout 框架。

## 后端分工

`TorchReferenceBackend` 用 PyTorch 计算 reference。实现里把 A/B/bias 转成 `fp32` 参与计算，再 cast 到 `fp16` 输出，便于对应 “fp32 accumulator” 这个 lowering 合同。

`TritonFusedGemmBackend` 是实际可运行的 fused backend。kernel 内完成：

```text
acc = A @ B
tmp = (acc + bias[n]) * scale
out = gelu(tmp)
```

它使用 mask 支持非整除 shape，但不追求极致性能。

`AutoPartitionCudaBackend` 是实际可运行的 AutoPartition CUDA backend。它从同一个 `LoweredGemmEpilogueOp` 读取 `M/N/K`、dtype、layout、tile shape、thread count 和 alignment，然后通过 PyTorch CUDA extension 调用 `autopartition_sm80_runtime.cu`。

当前实现为了先形成可靠闭环，采用两段 CUDA kernel：

```text
GraphIR
  -> fused_gemm_epilogue
  -> LoweredGemmEpilogueOp
  -> AutoPartitionCudaBackend
  -> AutoPartitioner RoleA/RoleB/RoleC GEMM kernel
  -> fp32 accumulator temporary
  -> CUDA epilogue kernel: bias + scale + gelu + fp16 output
  -> Python torch.Tensor
```

第一段 GEMM 主计算真实使用你的 AutoPartitioner 选择：

```text
RoleA::GlobalToSharedCopy
RoleB::GlobalToSharedCopy
RoleA/RoleB::SmemToRegCopyOperation
RoleC::TiledMma
RoleC::OutputRegisterToGlobalCopy
```

第二段 epilogue 是一个小 CUDA kernel，用来把 `fp32 accumulator` 变成最终 `fp16 Y`。这样避免在 demo 第一版里猜测尚未稳定公开的 fusion epilogue 坐标接口。未来如果 AutoPartitioner 暴露了完整 fusion epilogue atom/coordinate contract，可以把第二段 epilogue 合回第一段 kernel，变成真正单 kernel fused GEMM epilogue。

`AutoPartitionBackend` 是 plan backend，不是 launch backend。当前仓库中的 AutoPartitioner 是 C++ 模板 layout/template planner，通过 `RoleA/RoleB/RoleC` 选择 global-to-shared、shared-to-register、MMA、epilogue writeback 等 layout 和 atom。这个 demo 不把它假设成完整 GEMM runtime，而是生成：

```text
LoweredGemmEpilogueOp -> AutoPartitionConfig -> AutoPartitionPlan
```

现在这个 backend 会优先调用 `autopartition_sm80_probe.cu`。这个 probe 仿照 production GEMM example，实例化：

```cpp
autopartition::AutoPartitioner<
    cutlass::arch::Sm80,
    cutlass::arch::OpClassTensorOp,
    cutlass::half_t,
    StrideA/StrideB/StrideC,
    Shape<Int<64>, Int<64>, Int<64>>,
    128,
    cutlass::half_t,
    16, 16, 16>
```

然后把 `RoleA/RoleB/RoleC` 暴露的计划打印回 Python，例如：

```text
RoleA::UseLdMatrix
RoleB::UseLdMatrix
RoleA::SwizzleBase
RoleB::SwizzleBase
RoleC::EpilogueLayoutCandidateCount
RoleC::HasFusionSharedMapping
RoleC::OutputAlignmentBytes
```

Python 侧会把这些字段纳入 `AutoPartitionPlan`，所以当前链路已经真实走了你的 AutoPartitioner 模板接口。它仍然打印：

```text
gmem_to_smem = RoleA/RoleB::GlobalToSharedCopy selected by AutoPartitioner
smem_to_reg = RoleA/RoleB::SmemToRegCopyOperation selected by AutoPartitioner
mma_atom = RoleC::TiledMma selected by AutoPartitioner
epilogue_layout = RoleC::FusionSmemLayout / OutputRegisterToGlobalCopy selected by AutoPartitioner
runnable = false
```

第一次运行会用 `nvcc` 把 probe 编译到 `~/.cache/zfm_compiler_demo/autopartition/`。如果 `autopartition_sm80_probe.cu`、`auto_partitioner.hpp`、`auto_partitioner_builder.hpp` 或 `sm80_policy.hpp` 更新，Python adapter 会自动重编译 probe。

未来要接成真正 Cutlass/CuTe kernel launch，可以让 `AutoPartitionPlan` 生成 C++ kernel template 参数，复用 probe 中的 `PartA/PartB/PartC` 类型，再把 epilogue math 接进 production GEMM kernel。

## 运行

只做 IR dump 和 AutoPartition plan：

```bash
python3 examples/ai_lowering_demo/zfm_compiler_demo/demo.py
```

有 CUDA、PyTorch、Triton 时，同一个命令会额外运行 Torch reference、Triton fused kernel、AutoPartition CUDA backend、correctness 和 benchmark。

当前机器上已发现可用环境：

```bash
/home/zfm/Desktop/qwen_quant/qwen35_quant_vllm/.venv/bin/python \
  examples/ai_lowering_demo/zfm_compiler_demo/demo.py
```

该环境包含 `torch 2.11.0+cu130` 和 `triton 3.6.0`，可以完整跑通 demo。

成功输出中应该能看到：

```text
TorchReferenceBackend: legal
TritonFusedGemmBackend: legal
AutoPartitionCudaBackend: legal
AutoPartitionBackend: plan_only

AutoPartition CUDA Result
shape = (1024, 1024), dtype = torch.float16, device = cuda:0

autopartition_cuda.passed = true
autopartition_cuda: ... ms
```

运行测试：

```bash
python3 -m unittest discover -s examples/ai_lowering_demo -p 'test_*.py'
```

## 面试讲法

可以按这条主线讲：

1. 前端没有直接依赖 PyTorch FX，而是手写 tiny frontend，目的是把表达式转成可控 GraphIR。
2. GraphIR 保留 op 依赖和 tensor metadata，方便 pass 分析。
3. pass 识别 GEMM 后接 bias、scale、gelu 的 epilogue pattern，并融合成一个 `fused_gemm_epilogue`。
4. lowering 把图节点变成后端合同，明确 M/N/K、dtype、layout、accumulator 和 epilogue。
5. backend legality 决定哪个 backend 能执行，哪个只能生成 plan。
6. Torch 是 correctness reference，Triton 是实际 fused kernel，AutoPartition 是 layout/template planning backend。
7. 这个 demo 故意不做大而全，价值在于把 AI 编译器的核心链路讲清楚。

cuBLASLt 可以作为后续 backend。当前表达式在某些场景下可重写为 `gelu(scale * A@B + scale * bias)` 来适配部分 epilogue 能力，但第一版不实现，避免把 demo 复杂化。
