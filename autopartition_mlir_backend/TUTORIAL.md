# AutoPartition-MLIR-Backend 从零复现教程

这份教程不是只告诉你运行 `python run.py`。它按真实开发顺序解释：先建立什么接口、写什么代码、运行什么工具、每一步生成什么 IR、怎样判断这一步没有被“假实现”绕过，以及最后怎样接入 fused CUDA backend。

教程对应当前仓库中的已验证实现。建议先完整读一遍，再从“阶段 0”开始操作。

## 1. 最终要解决什么问题

输入模型只有一行计算：

```python
Y = torch.nn.functional.gelu(A @ B + bias)
```

固定输入为：

```text
A:    [1024, 1024], fp16, CUDA
B:    [1024, 1024], fp16, CUDA
bias: [1024],       fp16, CUDA
```

我们不是直接读取 FX Graph 后调用 CUDA，而是让模型依次经过四层可观察的编译表示：

```text
PyTorch Module
  -> torch.export ExportedProgram
  -> torch-mlir StableHLO
  -> StableHLO legalize-to-Linalg-on-Tensors
  -> autopartition.gemm_bias_gelu
  -> func.call @autopartition_fused_gemm_bias_gelu
  -> runtime.py 读取 call 并 dispatch
  -> PyTorch CUDA extension
  -> AutoPartitioner/CUTLASS/CuTe 单 kernel
```

其中四份落盘证据分别是：

| 文件 | 应看到的核心内容 | 证明什么 |
|---|---|---|
| `mlir/exported_stablehlo.mlir` | `stablehlo.dot_general`、`stablehlo.add` | PyTorch 已进入真实 HLO 层 |
| `mlir/exported_linalg.mlir` | `linalg.matmul` 和 epilogue 运算 | HLO 已通过 StableHLO pass lowering |
| `mlir/fused.mlir` | `autopartition.gemm_bias_gelu` | C++ pass 已识别并融合模式 |
| `mlir/lowered.mlir` | `call @autopartition_fused_gemm_bias_gelu` | backend op 已变成 runtime ABI |

## 2. 先理解为什么这样分层

ByteIR 和 BladeDISC 的核心工程思想不是“所有 IR 都自己造”，而是把不同职责放在不同层：

1. `torch.export` 负责从动态图世界捕获可编译程序。
2. StableHLO/MHLO 保存框架无关的高层张量语义。
3. Linalg-on-Tensors 提供适合分析 use-def、融合和后续 bufferization 的结构。
4. 自定义 `autopartition` op 只表达后端已经理解的候选融合。
5. legalize pass 决定该候选能否交给当前 CUDA 实现。
6. runtime call 建立编译 IR 与实际 PyTorch Tensor/CUDA kernel 的边界。

这样做的价值是：前端变化不要求重写 CUDA kernel，后端 tile policy 变化也不要求重新设计 PyTorch IR。

## 3. 阶段 0：进入环境并确认基础工具

当前机器上已经验证通过的解释器是：

```bash
export PROJECT=/home/zfm/Desktop/cutlass/autopartition_mlir_backend
export PYTHON=/media/zfm/System/store/PLAF_runtime/venvs/mast3r-slam/bin/python
cd "$PROJECT"
```

先执行最小检查：

```bash
$PYTHON --version
$PYTHON -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvcc --version
cmake --version
ninja --version
```

再运行项目的完整探测：

```bash
$PYTHON env_check.py
```

本机已验证组合为 Python 3.12.3、PyTorch 2.9.1+cu130、CUDA Toolkit 13.0、torch-mlir dev wheel 20260531.828、LLVM/MLIR 20。GPU 是 RTX 5060 Ti，compute capability 为 12.0。

`env_check.py` 的开发顺序是：

1. 用 `shutil.which` 找 `nvcc/cmake/ninja/mlir-opt`。
2. 用 `importlib.util.find_spec` 检查 `torch`、`torch_mlir` 和 StableHLO 扩展。
3. 用 PyTorch 查询 GPU、CUDA 和 compute capability。
4. 检查 CUTLASS include 和 AutoPartitioner header。
5. 把大体积 wheel、LLVM 包和 extension build 放到挂载盘缓存。

关键缓存位置：

```text
/media/zfm/System/AutoPartition-MLIR-Backend-cache/
```

若 `torch_mlir` 缺失，当前脚本会尝试：

```bash
$PYTHON -m pip install --no-deps --pre torch-mlir \
  -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels
```

若 `MLIRConfig.cmake` 缺失，不能用 Python 文本处理代替 C++ pass。需要安装匹配版本的 MLIR 开发包或从 LLVM 源码构建，然后设置：

```bash
export MLIR_DIR=/path/to/llvm/lib/cmake/mlir
```

当前机器可直接使用：

```bash
export MLIR_DIR=/media/zfm/System/AutoPartition-MLIR-Backend-cache/mlir-prefix/usr/lib/llvm-20/lib/cmake/mlir
```

## 4. 阶段 1：先写 PyTorch 模型和输入

第一个应写的业务文件是 `export_model.py`。此时先不考虑 MLIR pass 和 CUDA，只定义源程序：

```python
class GemmBiasGelu(torch.nn.Module):
    def forward(self, x, w, bias):
        return torch.nn.functional.gelu(x @ w + bias)


def build_model():
    return GemmBiasGelu().eval()


def build_inputs(device="cuda"):
    factory = {"device": device, "dtype": torch.float16}
    return (
        torch.randn((1024, 1024), **factory),
        torch.randn((1024, 1024), **factory),
        torch.randn((1024,), **factory),
    )
```

先直接检查语义：

```bash
$PYTHON - <<'PY'
from export_model import build_model, build_inputs
A, B, bias = build_inputs()
Y = build_model()(A, B, bias)
print(Y.shape, Y.dtype, Y.device)
PY
```

预期输出包含：

```text
torch.Size([1024, 1024]) torch.float16 cuda:0
```

这一阶段只确认模型本身正确，不能把它当作编译成功。

## 5. 阶段 2：用 torch.export 捕获程序

第二步实现 `_exported_program` 和 `print_fx_or_exported_graph`：

```python
program = torch.export.export(model, tuple(inputs), {}, strict=False)
print(program.graph_module.print_readable())
```

运行：

```bash
$PYTHON - <<'PY'
from export_model import build_model, build_inputs, print_fx_or_exported_graph
print_fx_or_exported_graph(build_model(), build_inputs())
PY
```

预期 graph 中能看到：

```text
torch.ops.aten.matmul.default
torch.ops.aten.add.Tensor
torch.ops.aten.gelu.default
```

这里的 `ExportedProgram` 是 torch-mlir 的输入，不是最终编译 IR。当前代码通过 `inspect.signature` 探测 `torch.export.export` 是否支持 `strict`，避免写死某个旧版本 API。

## 6. 阶段 3：调用 torch-mlir 生成 StableHLO

接下来实现 `export_to_stablehlo`。关键调用不是手写 MLIR，而是：

```python
from torch_mlir.compiler_utils import OutputType
from torch_mlir.fx import export_and_import

module = export_and_import(
    program,
    output_type=OutputType.get("STABLEHLO"),
    decomposition_table={},
    strict=False,
)
```

把 `str(module)` 保存为 `mlir/exported_stablehlo.mlir`，并强制检查文本中至少出现 `stablehlo` 或 `mhlo`。执行：

```bash
$PYTHON - <<'PY'
from export_model import build_model, build_inputs, export_to_stablehlo
export_to_stablehlo(build_model(), build_inputs(), "mlir/exported_stablehlo.mlir")
PY
rg -n "stablehlo.dot_general|stablehlo.add" mlir/exported_stablehlo.mlir
```

你应看到 `stablehlo.dot_general` 表示矩阵乘，`stablehlo.broadcast_in_dim` 表示 bias 广播，后续 `stablehlo.add/multiply/rsqrt/...` 表示 GELU 展开。

如果这里只有 ATen/FX 文本，说明没有真正进入 HLO；如果文件是人工模板，也不满足主链路要求。

## 7. 阶段 4：把已保存的 StableHLO lowering 到 Linalg

这一步必须读取刚才落盘的 HLO 文件，再由 StableHLO 注册的 MLIR pass 转换：

```python
from torch_mlir import ir
import torch_mlir._mlir_libs._stablehlo as stablehlo
from torch_mlir.passmanager import PassManager

context = ir.Context()
stablehlo.register_dialect(context)
stablehlo.register_stablehlo_passes()
module = ir.Module.parse(stable_text, context)
pm = PassManager.parse(
    "builtin.module(stablehlo-legalize-to-linalg, "
    "stablehlo-convert-to-signless, canonicalize)",
    context=context,
)
pm.run(module.operation)
```

当前实现准备了三个候选 pipeline，以适配 wheel 中是否需要 aggressive simplification 或 CHLO legalize。只有输出确实包含 `linalg.matmul` 才接受。

执行：

```bash
$PYTHON - <<'PY'
from export_model import build_model, build_inputs, export_to_linalg
export_to_linalg(
    build_model(), build_inputs(),
    "mlir/exported_linalg.mlir",
    "mlir/exported_stablehlo.mlir",
)
PY
rg -n "linalg.matmul|linalg.generic|arith.addf|math" mlir/exported_linalg.mlir
```

至此主路径已证明是 `PyTorch -> HLO -> Linalg`，不是手写 `input_linalg.mlir`。

## 8. 阶段 5：先定义 MLIR backend contract

在写 C++ pass 前，先明确 custom op 的输入、输出和属性：

```mlir
%y = "autopartition.gemm_bias_gelu"(%a, %b, %bias) {
  M = 1024 : i64,
  N = 1024 : i64,
  K = 1024 : i64,
  dtype = "f16",
  target_sm = "sm80",
  tile_m = 64 : i64,
  tile_n = 64 : i64,
  tile_k = 64 : i64,
  thread_count = 128 : i64,
  alignment_bytes = 16 : i64,
  backend = "unknown",
  fallback_reason = ""
} : (...) -> tensor<1024x1024xf16>
```

第一版在 `tools/autopartition-opt.cpp` 中注册 `AutoPartitionDialect` 并允许 generic operations。它仍是 MLIR `Operation`，由 `OpBuilder` 创建和 use-def 重写，只是尚未使用 ODS/TableGen 生成强类型 op 类。

## 9. 阶段 6：实现三个真实 MLIR C++ pass

### 9.1 Fusion pass

先注册 pass 名称：

```cpp
StringRef getArgument() const final {
  return "autopartition-fuse-gemm-epilogue";
}
```

然后按以下顺序实现：

1. `module.walk` 找 `linalg.matmul`，不做字符串替换。
2. 从 `func.return` 的返回值反向遍历 SSA producer DAG。
3. 只允许 `linalg/tensor/arith/math/chlo` epilogue op。
4. 找到 rank-1 bias 来源，并确认 bias 长度等于 N。
5. 通过 `erf/tanh/rsqrt` 或 GELU location 识别 GELU 证据。
6. 确认 matmul 没有 DAG 外额外用户。
7. 从 RankedTensorType 推导静态 M/N/K。
8. 用 `OperationState` 创建 `autopartition.gemm_bias_gelu`。
9. `replaceAllUsesWith` 替换输出，并删除无副作用死计算。

### 9.2 Backend legalize pass

这个 pass 不负责融合，只检查 backend contract：

```text
dtype == f16
A/B/output rank == 2
bias rank == 1 且 bias.shape[0] == N
M/N/K 静态且都是 64 的倍数
alignment_bytes >= 16
target_sm == sm80
```

合法时：

```mlir
backend = "AutoPartitionBackend", fallback_reason = ""
```

非法时：

```mlir
backend = "TorchFallbackBackend",
fallback_reason = "M must be a multiple of 64"
```

### 9.3 Lower-to-runtime pass

最后一个 pass 根据 backend 创建私有函数声明和 `func.call`：

```mlir
%0 = call @autopartition_fused_gemm_bias_gelu(%a, %b, %bias)
```

或者显式 fallback：

```mlir
%0 = call @torch_fallback_gemm_bias_gelu(%a, %b, %bias)
```

call 和声明都保留 `autopartition.backend`、`autopartition.fallback_reason`，供 Python runtime 读取。

## 10. 阶段 7：用 CMake 构建 autopartition-opt

`tools/CMakeLists.txt` 先执行 `find_package(MLIR REQUIRED CONFIG)`，然后链接 parser、IR、pass、opt driver 和使用到的 dialect 库。手工构建命令如下：

```bash
cmake -S tools -B tools/build -G Ninja \
  -DMLIR_DIR="$MLIR_DIR" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build tools/build --target autopartition-opt
```

检查三个 pass 已注册：

```bash
tools/build/autopartition-opt --help | rg "autopartition-(fuse|legalize|lower)"
```

预期看到：

```text
--autopartition-fuse-gemm-epilogue
--autopartition-legalize-backend
--autopartition-lower-to-runtime
```

## 11. 阶段 8：手工运行 MLIR pipeline

先运行 fusion：

```bash
tools/build/autopartition-opt \
  mlir/exported_linalg.mlir \
  --autopartition-fuse-gemm-epilogue \
  -o mlir/fused.prelegalize.mlir
```

再运行 legalize：

```bash
tools/build/autopartition-opt \
  mlir/fused.prelegalize.mlir \
  --autopartition-legalize-backend \
  -o mlir/fused.mlir
```

检查：

```bash
rg -n "autopartition.gemm_bias_gelu|backend|fallback_reason" mlir/fused.mlir
```

最后 lowering 到 runtime call：

```bash
tools/build/autopartition-opt \
  mlir/fused.mlir \
  --autopartition-lower-to-runtime \
  -o mlir/lowered.mlir
rg -n "call @|autopartition.backend|fallback_reason" mlir/lowered.mlir
```

日常代码中 `mlir_pipeline.py` 的 `run_fusion_and_legalize` 和 `run_lower_to_runtime` 封装了这些命令，并在失败时把 stdout/stderr 放进异常。

## 12. 阶段 9：实现真正的 fused CUDA backend

`csrc/autopartition_runtime.cu` 的开发可以拆成四层。

### 12.1 PyTorch extension ABI

先暴露固定 API：

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_gemm_bias_gelu", &fused_gemm_bias_gelu);
}
```

Python 最终只调用：

```python
extension.fused_gemm_bias_gelu(A, B, bias)
```

### 12.2 Runtime 输入检查

CUDA wrapper 检查 CUDA device、fp16、rank、contiguous、`B.shape[0] == A.shape[1]`、bias `[N]`，以及 M/N/K 都是 64 的倍数。静态 legalize 和这里的动态检查是双保险。

### 12.3 AutoPartitioner policy

当前 kernel 使用：

```cpp
TileShape   = Shape<Int<64>, Int<64>, Int<64>>
ThreadCount = 128
Arch        = cutlass::arch::Sm80
```

分别实例化 `AutoPartitioner<...>::RoleA`、`RoleB`、`RoleC`。RoleA/B 提供 global-to-shared 和 shared-to-register copy，RoleC 提供 MMA 与 fused epilogue 的 shared/register/global mapping。

### 12.4 单 kernel 数据流

kernel 内部顺序是：

```text
A/B global memory
  -> AutoPartition cooperative copy
  -> shared memory
  -> cooperative_gemm / Tensor Core accumulator
  -> RoleC register-to-shared-to-output-register mapping
  -> 每个输出元素读取 bias[col]
  -> 0.5*x*(1+erf(x/sqrt(2)))
  -> fp16 conversion
  -> RoleC output-register-to-global copy
```

源文件中只有一次 `<<<grid, block>>>` launch。没有中间 global accumulator，没有第二个 epilogue kernel，也没有 `torch.matmul`、cuBLAS 或 Triton 冒充 backend。

## 13. 阶段 10：编译和加载 CUDA extension

`runtime.py` 使用 PyTorch 的 JIT extension loader：

```python
from torch.utils.cpp_extension import load

extension = load(
    name="autopartition_mlir_runtime",
    sources=["csrc/autopartition_runtime.cu"],
    extra_include_paths=["/home/zfm/Desktop/cutlass/include"],
    extra_cuda_cflags=["-std=c++17", "--expt-relaxed-constexpr"],
)
```

当前实现默认设置：

```bash
export TORCH_CUDA_ARCH_LIST="8.0+PTX"
export MAX_JOBS=1
export AUTOPARTITION_MLIR_EXT_DIR=/media/zfm/System/AutoPartition-MLIR-Backend-cache/torch_extension
```

第一次单独触发编译：

```bash
$PYTHON - <<'PY'
from runtime import load_autopartition_extension
ext = load_autopartition_extension()
print(ext, hasattr(ext, "fused_gemm_bias_gelu"))
PY
```

后续运行会复用缓存。`8.0+PTX` 是当前 SM80 policy 的 PTX 兼容构建策略，并不等于已经实现原生 SM120 专用 kernel。

## 14. 阶段 11：runtime 必须服从 lowered.mlir

`runtime.py` 先读取 `mlir/lowered.mlir`，用 `_lowered_metadata` 得到：

```text
callee
autopartition.backend
autopartition.fallback_reason
```

dispatch 规则是：

| lowered call | backend 属性 | 实际执行 |
|---|---|---|
| `autopartition_fused_gemm_bias_gelu` | `AutoPartitionBackend` | CUDA extension |
| `torch_fallback_gemm_bias_gelu` | `TorchFallbackBackend` | PyTorch reference |
| 任意不一致组合 | 任意 | 抛出错误 |

fallback call 如果没有原因也会抛错，因此不会静默 fallback。

正确性比较：

```python
ref = torch.nn.functional.gelu(A @ B + bias)
max_abs_error = (out.float() - ref.float()).abs().max()
passed = max_abs_error < 1e-1
```

benchmark 使用 CUDA Event，先 warmup 10 次，再测 50 次并同步。

## 15. 阶段 12：组合成一键入口

所有独立部件验证完后，最后才写 `run.py`。它的调用顺序是：

```text
print_environment
ensure_torch_mlir
build_model / build_inputs
print_fx_or_exported_graph
export_to_stablehlo
export_to_linalg
run_fusion_and_legalize
run_lower_to_runtime
check_mlir_outputs
dispatch_from_lowered_mlir
compare_outputs
benchmark
```

执行完整验收：

```bash
cd /home/zfm/Desktop/cutlass/autopartition_mlir_backend
$PYTHON run.py
```

当前机器最近一次完整运行的关键输出是：

```text
PyTorch 前端捕获成功
StableHLO/MHLO 导出成功：mlir/exported_stablehlo.mlir
Linalg-on-Tensors 导出成功：mlir/exported_linalg.mlir
MLIR fusion pass 成功：mlir/fused.mlir
backend legalize 成功：AutoPartitionBackend
runtime lowering 成功：mlir/lowered.mlir
actual_backend = AutoPartitionBackend
max_abs_error = 0
max_rel_error = 0
correctness = pass
benchmark 已生成: AutoPartitionBackend avg_ms=0.0618, PyTorch avg_ms=0.0698
```

性能数字会受频率、温度和后台任务影响，不应把某一次数字当作稳定结论。

## 16. 如何逐项测试而不是只相信 run.py

执行：

```bash
$PYTHON -m pytest tests/test_pipeline.py -q
```

当前 6 项测试覆盖：

1. PyTorch 捕获以及 StableHLO/MHLO 存在。
2. Linalg 中存在 matmul 和 bias/GELU epilogue 证据。
3. fusion op 和全部 backend 属性存在，legalize 选择 AutoPartitionBackend。
4. lowered IR 中存在规范 runtime call。
5. 65x64x64 非法 shape 被明确 legalize 为 fallback，并带原因。
6. CUDA 可用时实际 backend 是 AutoPartitionBackend，最大绝对误差小于 `1e-1`。

也可以做禁止项审计：

```bash
rg -n "<<<" csrc/autopartition_runtime.cu
rg -n "torch::matmul|at::matmul|cublas|triton" csrc/autopartition_runtime.cu
```

第一条应只有一个 kernel launch；第二条应没有匹配。

## 17. 常见故障和定位顺序

### torch-mlir import 失败

先确认安装 wheel 的 Python 与运行脚本的 Python 是同一个：

```bash
which python
$PYTHON -c "import sys, torch_mlir; print(sys.executable, torch_mlir.__file__)"
```

cp311 wheel 不能装进 cp312。若没有兼容 wheel，必须按 torch-mlir 当前源码要求选择匹配的 PyTorch/LLVM revision 构建，不能改成手写 Linalg。

### StableHLO pass 未注册

确认在创建 `PassManager` 前执行：

```python
stablehlo.register_dialect(context)
stablehlo.register_stablehlo_passes()
```

### CMake 找不到 MLIR

定位配置：

```bash
find /usr /media/zfm -name MLIRConfig.cmake 2>/dev/null | head
```

然后设置 `MLIR_DIR` 到包含该文件的目录，而不是文件本身。

### fusion 后没有 custom op

按顺序检查：

```bash
rg -n "linalg.matmul" mlir/exported_linalg.mlir
rg -n "tensor<1024xf16>" mlir/exported_linalg.mlir
rg -n "erf|tanh|rsqrt|gelu" mlir/exported_linalg.mlir
```

若 torch-mlir 新版本改变 GELU 展开，应修改 C++ use-def matcher 支持新的等价 op DAG，并增加测试；不能靠文本替换输出文件。

### CUDA extension 编译失败

依次检查 `nvcc --version`、PyTorch 的 CUDA 版本、C++ ABI、CUTLASS include 路径和 `TORCH_CUDA_ARCH_LIST`。清理 extension 缓存会触发完整重编译，通常成本较高，应先读 Ninja 编译错误。

### correctness 失败

先用 64x64x64 缩小问题，再分别核对 B 的逻辑 stride、tile 坐标、bias column、累加类型和 GELU 近似。当前实现为了对齐 PyTorch fp16 eager 语义，在寄存器中模拟 GEMM 输出和 bias add 后的 fp16 rounding。

## 18. 现在已经实现了什么

以下能力已经有代码、生成物和测试，不是规划项：

- 真实 `torch.export` 前端捕获。
- 真实 torch-mlir StableHLO 导出。
- 从保存的 StableHLO 经注册 pass lowering 到 Linalg-on-Tensors。
- 三个可由 opt-like CLI 调用的 MLIR C++ pass。
- 沿 SSA DAG 识别 matmul、rank-1 bias 和 GELU 证据。
- 创建 generic MLIR backend op，并附完整静态配置属性。
- dtype、静态 shape、64 倍数、rank、alignment、target 的 backend legalize。
- 非法 shape 的明确 fallback 和具体 `fallback_reason`。
- backend op 到 `func.call` 的 runtime lowering。
- runtime 读取 lowered IR 后 dispatch，不绕过 lowered IR。
- PyTorch CUDA extension 的 Tensor 输入输出包装。
- AutoPartitioner RoleA/RoleB/RoleC 驱动的 fp16 Tensor Core GEMM。
- bias 和 exact-erf GELU 位于同一个 kernel 的寄存器 epilogue。
- 正确性、CUDA Event benchmark 和 6 项 pytest 验收。
- 本机 RTX 5060 Ti 上主路径实际运行，最近一次 `max_abs_error = 0`。

## 19. 现在还没有实现什么

下面是当前边界，不能把它们描述成已经完成：

- **没有通用模型支持。** 当前 matcher 和 runtime 只针对 GEMM + 一维 bias + GELU。
- **没有 dynamic shape。** M/N/K 必须静态，且都是 64 的倍数。
- **没有多布局。** 只支持 row-major contiguous A/B/output 和 contiguous bias。
- **没有 fp32、bf16、fp8。** backend contract 当前只接受 fp16。
- **没有原生 SM90/SM100/SM120 policy dispatch。** 当前使用 SM80 policy 和 `8.0+PTX` 兼容路径。
- **没有完整 ODS/TableGen dialect。** custom op 是已注册 dialect 下的 generic operation，没有自动 verifier、parser/printer 和类型化 C++ API。
- **没有标准 ExecutionEngine/LLVM ABI。** lowered MLIR 是 runtime 调度契约，实际由 Python 读取文本元数据后调用 extension；尚未把 MLIR module 编译成可直接执行的 LLVM/CUDA binary。
- **没有完整 bufferization/codegen pipeline。** 本项目在 Linalg 层做融合后进入自定义 backend，不演示 Linalg-to-loops、GPU dialect、NVVM lowering。
- **没有跨 op cost model。** tile、线程数和 backend target 是固定策略，不是运行时 autotune。
- **没有多 stream、CUDA Graph、分布式和显存规划。** 只使用当前 PyTorch CUDA stream。
- **没有完善生产测试矩阵。** 尚缺 lit/FileCheck、随机 shape 大规模测试、跨 GPU、性能回归和 sanitizer。
- **环境自修复不是完全无人值守。** wheel 缺失时会尝试 pip；但需要 sudo 的系统包安装、torch-mlir/LLVM 大规模源码构建仍需根据机器权限执行，脚本会报出准确缺项而不会伪造降级结果。
- **runtime metadata 解析仍是演示级。** 当前 Python 用正则读取固定 call/属性；生产系统应使用 MLIR bytecode/API 或稳定的序列化 executable metadata。

## 20. 下一步如何扩展成更接近生产系统

建议按这个顺序扩展：

1. 为 `autopartition.gemm_bias_gelu` 增加 TableGen 定义、verifier 和 canonicalization。
2. 用 `tensor.dim`、shape constraints 和 runtime guards 支持动态 M/N/K。
3. legalize 生成多个 kernel variant，runtime 根据 shape/alignment/capability 选择。
4. 增加 bf16 和 SM90/SM100/SM120 AutoPartitioner policy。
5. 把 layout/stride 写进 op contract，支持 transpose 和非连续输入。
6. 把 Python 正则 dispatch 替换成编译产物 metadata 或 MLIR ExecutionEngine ABI。
7. 增加更多 epilogue、多个 GEMM、attention 子图和收益模型。
8. 建立 lit/FileCheck、pytest、跨架构 correctness 和稳定性能基线。

## 21. 一段完整的面试讲法

本项目参考 ByteIR 和 BladeDISC 的企业级 AI 编译器分层路线，没有从零发明深度学习 IR。前端先用 `torch.export` 捕获 PyTorch 程序，再用 torch-mlir 导出 StableHLO，并通过 StableHLO 官方注册的 pass lowering 到 Linalg-on-Tensors。优化阶段由自建的 `autopartition-opt` 执行三个 MLIR C++ pass：第一个从函数输出反向遍历 SSA use-def DAG，识别 `linalg.matmul + bias + GELU` 并创建 backend op；第二个检查 dtype、静态 shape、layout 假设、alignment、tile 和 target，决定 AutoPartition 或带原因的 fallback；第三个生成明确的 runtime call。运行时读取 lowered MLIR 的 call 和 backend metadata，只有合法路径才加载 PyTorch CUDA extension。CUDA 侧使用 CUTLASS/CuTe AutoPartitioner 的 RoleA、RoleB、RoleC 完成 cooperative GEMM，并在同一个 kernel 的寄存器 epilogue 中读取 bias 和计算 GELU。最后用 PyTorch reference 做 correctness，并使用 CUDA Event benchmark。当前原型完整证明了前端、MLIR 优化、backend legalize、runtime ABI 和 fused kernel 的纵向链路，但 dynamic shape、完整 TableGen dialect、原生多架构 codegen 和生产级 executable runtime 仍是后续工作。
