# AutoPartition-MLIR-Backend

> 第一次学习或希望从空目录逐步复现，请先阅读 [TUTORIAL.md](TUTORIAL.md)。它按“写代码 -> 执行工具 -> 检查 IR -> 接入 CUDA -> 验收”的顺序讲完整开发过程，并单独列出已实现和未实现能力。

这是一个面向 AI 编译器面试和工程验证的真实全链路项目：

```text
PyTorch GemmBiasGelu
  -> torch.export / ExportedProgram
  -> torch-mlir StableHLO/MHLO
  -> StableHLO legalize-to-Linalg-on-Tensors
  -> MLIR C++ use-def fusion
  -> autopartition.gemm_bias_gelu
  -> backend legalize
  -> runtime call
  -> PyTorch CUDA extension
  -> CUTLASS/CuTe AutoPartitioner fused GEMM
  -> correctness + benchmark
```

## 设计定位

项目参考 ByteIR 和 BladeDISC 的企业级路线：复用 StableHLO/MHLO、Linalg-on-Tensors 等成熟 MLIR 方言，不重新发明一套深度学习 IR。StableHLO/MHLO 负责模型语义和跨框架接口；Linalg-on-Tensors 负责可组合的张量级优化；AutoPartition backend 只负责最后一公里的目标硬件选择、合法性和 runtime 接入。

ByteIR 的思想是让通用图、循环和张量优化在 MHLO/Linalg 层复用，后端聚焦 codegen/runtime；BladeDISC 也采用 PyTorch/TorchScript 到 MHLO，再逐步 lowering 到运行时抽象层的路线。本项目用 torch.export/torch-mlir 取代旧 TorchScript 入口，但保留相同的分层边界。

固定模型是 `Y = GELU(A @ B + bias)`，默认 A 为 `[1024,1024]`、B 为 `[1024,1024]`、bias 为 `[1024]`、fp16 CUDA。合法后端要求静态 M/N/K 都是 64 的倍数、16 字节对齐、row-major contiguous、目标为 `sm80`。当前机器 RTX 5060 Ti 是 `sm120`，运行时使用 SM80 AutoPartitioner policy 的 PTX 兼容路径。

## 文件说明

- `run.py`：一键执行环境检查、导出、C++ pass、runtime、正确性和 benchmark。
- `env_check.py`：探测 Python、PyTorch、CUDA、nvcc、CMake、Ninja、LLVM/MLIR、torch-mlir、StableHLO 和 CUTLASS；构建缓存优先放到挂载盘。
- `export_model.py`：实现 `build_model`、`build_inputs`、`export_to_stablehlo`、`export_to_linalg`、`print_fx_or_exported_graph`。FX 只用于观察捕获结果，不能替代 MLIR。
- `mlir_pipeline.py`：构建 `tools/autopartition-opt`，依次运行 fusion/legalize 和 lower-to-runtime，并检查产物。
- `runtime.py`：实现 `load_autopartition_extension`、`run_autopartition_fused`、`run_torch_reference`、`dispatch_from_lowered_mlir`、`benchmark`、`compare_outputs`。dispatch 首先读取 `lowered.mlir` 的 call 和 backend 属性。
- `tools/autopartition-opt.cpp`：三个真实 MLIR C++ pass。第一版使用动态注册的 generic MLIR op，不用 TableGen，但仍由 MLIR Operation/PassManager 创建和转换。
- `tools/CMakeLists.txt`：复用系统或本地 LLVM/MLIR CMake package。
- `csrc/autopartition_runtime.cu`：RoleA/RoleB/RoleC cooperative GEMM；bias 读取、GELU、转换和最终写回位于同一个 kernel epilogue。
- `mlir/exported_stablehlo.mlir`：HLO 层真实导出结果。
- `mlir/exported_linalg.mlir`：从上一个 HLO 文件经 StableHLO-to-Linalg pass 得到的 Tensor IR。
- `mlir/fused.mlir`：fusion 和 backend legalize 结果。
- `mlir/lowered.mlir`：最终 runtime call。
- `tests/test_pipeline.py`：端到端文件、属性、fallback 和 CUDA correctness 测试。

## 三个 C++ Pass

`--autopartition-fuse-gemm-epilogue` 沿 SSA use-def 链查找 `linalg.matmul`、一维 bias 广播/add 和 GELU 后缀，检查唯一主链、rank、shape、静态维度和无副作用条件后创建 `autopartition.gemm_bias_gelu`。它附带 `M/N/K/dtype/target_sm/tile_m/tile_n/tile_k/thread_count/alignment_bytes/backend/fallback_reason` 属性。

`--autopartition-legalize-backend` 检查 fp16、rank-2 A/B、rank-1 bias `[N]`、64 倍数、16 字节 alignment、row-major contiguous 假设和 sm80 target。合法时设置 `backend = "AutoPartitionBackend"`；非法时设置 `backend = "TorchFallbackBackend"`，并写入具体 `fallback_reason`。

`--autopartition-lower-to-runtime` 把 custom op 转为 `func.call @autopartition_fused_gemm_bias_gelu` 或 `func.call @torch_fallback_gemm_bias_gelu`，并在 call 和 private declaration 上保留 backend/reason 属性。

## 运行

在已有 PyTorch/CUDA venv 中执行：

```bash
cd /home/zfm/Desktop/cutlass/autopartition_mlir_backend
python run.py
python -m pytest tests/test_pipeline.py -q
```

`run.py` 会优先使用当前解释器；如果当前 Python 没有 PyTorch，会自动扫描 `/media/zfm/System`、`/media/zfm/Software` 和 home 下的 venv。torch-mlir 优先安装官方 cp312 dev wheel；如果 wheel/API 不兼容，应该使用与当前 PyTorch/LLVM 匹配的源码构建。MLIR CMake 找不到时，设置 `MLIR_DIR=/path/to/lib/cmake/mlir`；项目不会用手写 Linalg 文件绕过环境问题。

查看中间 IR：

```bash
less mlir/exported_stablehlo.mlir
less mlir/exported_linalg.mlir
less mlir/fused.mlir
less mlir/lowered.mlir
tools/build/autopartition-opt --help
```

## Runtime 和 fused backend

合法 lowered IR 只能由 `runtime.py` 识别后进入 `AutoPartitionBackend`，再调用 `extension.fused_gemm_bias_gelu(A, B, bias)`。CUDA kernel 使用 CUTLASS/CuTe AutoPartitioner 的 RoleA/RoleB/RoleC 主循环；累加结果经 AutoPartitioner epilogue mapping 进入寄存器后读取 bias，执行 `0.5*x*(1+erf(x/sqrt(2)))`，转换到 fp16 并通过 RoleC 输出 copy 写回。不存在中间 global accumulator，也不存在第二个 bias/GELU kernel。

非法 IR 不会静默报告成功：runtime 只能根据 `lowered.mlir` 的 fallback call 进入 `torch.nn.functional.gelu(A @ B + bias)`，并输出 `actual_backend` 和 `fallback_reason`。

## 面试讲法

本项目参考 ByteIR / BladeDISC 的企业级 AI 编译器路线，不从零发明深度学习 IR，而是复用 StableHLO/MHLO 和 Linalg-on-Tensors 等成熟 MLIR 方言。前端通过 torch.export / torch-mlir 将 PyTorch 模型导出到 HLO 层，再 lowering 到 Linalg-on-Tensors。在 Linalg 层，我实现 MLIR C++ pass 沿 use-def 链识别 linalg.matmul + bias + GELU pattern，判断融合合法性后生成 autopartition.gemm_bias_gelu。随后通过 backend legalize pass 检查 dtype、shape、layout、alignment 和 target 是否满足 AutoPartition 后端要求，并通过 runtime lowering pass 转成 call @autopartition_fused_gemm_bias_gelu。最后 runtime.py 根据 lowered.mlir dispatch 到 PyTorch CUDA extension，调用基于 CUTLASS/CuTe AutoPartitioner 的 fused GEMM 后端，并与 PyTorch reference 做 correctness 和 benchmark。

## 当前限制和扩展路线

当前实现针对静态二维 fp16 GEMM、row-major contiguous、SM80 policy 和默认 GELU。动态 shape 下一步需要在 Linalg 层保留 `tensor.dim`/shape constraints，在 legalize 中生成 runtime guard 和多版本 dispatch；布局扩展需要把 stride/layout 作为显式 backend contract，而不是只接受 contiguous；更高架构可以增加 SM90/SM100/SM120 policy，并让 AutoPartitioner 根据运行时 capability 选择实例。正式生产版本还应把 generic operation 升级为 ODS/TableGen dialect、把 runtime call 接入统一 ABI，并增加 lit/FileCheck、跨 GPU 和多 stream 测试。
