# AutoPartition Torch Backend Demo

这是一个面向 AI 编译器开发面试的 PyTorch 编译后端项目。它把 PyTorch 上层函数接入 `torch.compile` 自定义 backend，通过 TorchDynamo 捕获真实 FX Graph，识别 GEMM / Bias / GELU / MLP pattern，生成多层 lowering 记录，最后按 legalize 结果调用仓库内 AutoPartitioner 二次封装 CUDA extension。

## 项目目标

固定主线：

```text
PyTorch Module / Function
  -> torch.compile custom backend
  -> FX Graph
  -> Pattern Match / FusionPlan
  -> TensorContract
  -> AutoPartitionPlan
  -> AutoPartition CUDA extension
  -> RuntimeCall / Correctness / Benchmark / Report
```

AutoPartition backend 的主 GEMM 不使用 `torch.matmul`、cuBLAS、Triton 或官方 CUTLASS 默认 GEMM 偷换实现。CUDA extension 中实例化并使用：

```text
autopartition::AutoPartitioner<...>::RoleA
autopartition::AutoPartitioner<...>::RoleB
autopartition::AutoPartitioner<...>::RoleC
RoleA/RoleB::GlobalToSharedCopy
RoleA/RoleB::SmemToRegCopyOperation
RoleC::TiledMma
RoleC::OutputRegisterToGlobalCopy
```

当前 runtime 策略是 two-stage：

```text
AutoPartitioner GEMM mainloop -> fp32 accumulator temporary
CUDA epilogue kernel -> bias / GELU / fp16 output
```

报告中会明确记录这不是 single-kernel fused epilogue。后续可以把 epilogue 合入 AutoPartitioner 的 fusion epilogue contract。

## 一键运行

推荐使用你已配好的虚拟环境：

```bash
cd /home/zfm/Desktop/cutlass/autopartition_torch_backend
/home/zfm/Desktop/qwen_quant/qwen35_quant_vllm/.venv/bin/python run_demo.py
```

快速只跑默认 shape：

```bash
/home/zfm/Desktop/qwen_quant/qwen35_quant_vllm/.venv/bin/python run_demo.py --quick
```

默认会生成：

```text
reports/latest/report.md
reports/run_YYYYMMDD_HHMMSS/report.md
archive/v01_env_check/
archive/v02_fx_backend/
archive/v03_pattern_fusion/
archive/v04_lowering_report/
archive/v05_autopartition_extension/
archive/v06_runtime_call/
archive/v07_final_delivery/
archive/v08_interview_material/
```

## 环境要求

- Python + PyTorch，当前已发现可用环境：`torch 2.11.0+cu130`
- Triton 可选，只作为 baseline
- CUDA/nvcc，用于编译 PyTorch CUDA extension
- 当前仓库 `include/cutlass/transform/collective/auto_partitioner/...`

如果 CUDA、nvcc 或 GPU 架构不满足要求，demo 不会假装成功，会 fallback 到 PyTorch reference，并在 `reports/debug_log.md` 和报告中记录原因。

当前机器 GPU 是 RTX 5060 Ti，compute capability 为 `sm120`。本项目 runtime 选择 `sm80` AutoPartition policy，并通过 `TORCH_CUDA_ARCH_LIST=8.0+PTX` 尝试 PTX 兼容运行；如果驱动或代码路径不兼容，报告会明确记录 fallback。

## 文件说明

- `run_demo.py`：默认入口，负责环境检测、八阶段流水线、correctness、benchmark、报告和归档。
- `backend.py`：torch.compile backend、FX matcher、FusionPlan、TensorContract、AutoPartitionPlan、legalize、runtime 路由。
- `csrc/autopartition_runtime.cu`：PyTorch CUDA extension，主 GEMM 使用 AutoPartitioner RoleA/RoleB/RoleC。
- `tests/test_basic.py`：轻量行为测试，覆盖 FX pattern、lowering contract 和 CUDA 源码 Role 接入。
- `reports/`：每次运行的中文报告。
- `archive/`：每阶段版本存档。

## 支持的 Pattern

第一阶段：

```python
Y = gelu(X @ W + bias)
```

lowering 为 `GEMM_BIAS_GELU`。

第二阶段：

```python
Y = gelu(X @ W1 + b1) @ W2 + b2
```

lowering 为 `MLP_TWO_GEMM`，当前执行策略是两个 AutoPartition GEMM kernel 串联：第一段 `fused_gemm_bias_gelu`，第二段 `fused_gemm_bias`。

第三阶段 SwiGLU / SiLU gate 目前列为后续扩展，不影响主流程交付。

## 支持的 dtype / shape / backend

- dtype：输入 fp16，weight fp16，bias fp16，输出 fp16，accumulator fp32
- layout：row-major contiguous；非 contiguous 会在 runtime 前调用 `.contiguous()`，报告记录 layout normalization
- target policy：sm80 AutoPartitioner
- tile：`64x64x64`
- thread count：`128`
- shape：默认测试 `1024x1024x1024`，并测试 `M=512,K=1024,N=2048`、`M=2048,K=1024,N=4096`
- shape 限制：M/N/K 必须为 64 的倍数，否则 fallback 并记录原因

## 如何判断是否真的走了 AutoPartition backend

看 `reports/latest/report.md`：

1. `AutoPartitionPlan` 中应出现 RoleA / RoleB / RoleC、tile shape、thread count、alignment、probe 或 static runtime template 信息。
2. `RuntimeCall` 中 backend 如果是 `AutoPartitionCudaBackend`，说明 extension 调用成功。
3. 如果 backend 是 `PyTorchFallbackBackend`，报告会写明 fallback reason，例如非 CUDA、dtype 不合法、shape 不满足 64 倍数、extension build/runtime 报错。
4. `csrc/autopartition_runtime.cu` 中主计算路径直接使用 AutoPartitioner Role 类型，不是调用 torch/cuBLAS/Triton。

## 测试

```bash
cd /home/zfm/Desktop/cutlass
/home/zfm/Desktop/qwen_quant/qwen35_quant_vllm/.venv/bin/python -m pytest autopartition_torch_backend/tests/test_basic.py -q
```

## 面试讲解版本

一句话介绍：

这个项目是一个 PyTorch `torch.compile` 自定义后端，从 FX Graph 识别 GEMM epilogue 和两层 MLP pattern，lowering 成 AutoPartitioner 可消费的 backend contract，并通过 CUDA extension 调用我自己的 CUTLASS/CuTe AutoPartitioner GEMM runtime。

整体 pipeline：

```text
torch.compile 捕获 FX Graph
  -> 记录原始 FX nodes
  -> matcher 识别 mm/add/gelu 或 MLP_TWO_GEMM
  -> FusionPlan 表示图优化结果
  -> TensorContract 固化 M/N/K、dtype、layout、stride
  -> AutoPartitionPlan 固化 sm80 policy、tile、thread、alignment、Role 信息
  -> legalize 判断能否走 AutoPartition backend
  -> RuntimeCall 执行 AutoPartition extension 或显式 fallback
```

图优化做了什么：

`mm/add/gelu` 被融合为 `GEMM_BIAS_GELU`，因为 matmul 结果只被 bias add 消费，add 结果只被 GELU 消费，融合不会改变语义。MLP 中 `linear1 -> gelu -> linear2` 被识别为 `MLP_TWO_GEMM`，当前不做跨两个 GEMM 的单 kernel fusion，而是记录为两次 AutoPartition GEMM 调用。

lowering 分几层：

报告固定展示 FX Graph、Pattern Match、FusionPlan、TensorContract、AutoPartitionPlan、RuntimeCall 六层。每层记录的信息不同，便于面试时说明“图优化”和“后端合法化”不是一个函数直接调 CUDA。

AutoPartitioner 后端做了什么：

CUDA extension 参考 `include/cutlass/transform/collective/auto_partitioner/examples/production_gemm/sm80_autopartition_gemm.cu`，用 AutoPartitioner 选择出的 RoleA/RoleB/RoleC 完成 global-to-shared、shared-to-register、MMA 和 output writeback。Triton 只作为 baseline，不是核心 backend。

和 TorchInductor / Triton / CUTLASS 的关系：

TorchDynamo/FX 是前端捕获与图表示；TorchInductor 是对比 baseline，不是本项目目标后端；Triton 是可选性能 baseline；CUTLASS/CuTe 是 AutoPartitioner runtime 的底层模板与 layout 表达。本项目价值在于把 PyTorch 图、lowering 合同、backend legalize 和自研 AutoPartitioner runtime 串成闭环。

当前限制和后续优化：

- 当前只做 whole-graph pattern match，后续可扩展为子图替换。
- 当前 epilogue 是 two-stage，后续可接入 AutoPartitioner fusion epilogue 做 single-kernel。
- 当前 policy 固定 sm80，非 sm80 GPU 走 PTX 兼容尝试或 fallback，后续可补 sm90/sm100/sm120 policy。
- 当前 M/N/K 要求 64 倍数，后续可加入 padding、mask 或 residue kernel。
