# AutoPartition Torch Backend 运行报告

## 项目运行环境
- python_executable: `/home/zfm/Desktop/qwen_quant/qwen35_quant_vllm/.venv/bin/python`
- python_version: `3.12.3 (main, Mar 23 2026, 19:04:32) [GCC 13.3.0]`
- torch_import_error: ``
- triton_import_error: ``
- nvcc_path: `/usr/local/cuda/bin/nvcc`
- nvcc_version: `nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Jul_16_07:30:01_PM_PDT_2025
Cuda compilation tools, release 13.0, V13.0.48
Build cuda_13.0.r13.0/compiler.36260728_0`
- nvidia_smi: `Tue Jul  7 22:02:46 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.142                Driver Version: 580.142        CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5060 Ti     Off |   00000000:02:00.0  On |                  N/A |
| 31%   50C    P3             26W /  180W |     595MiB /  16311MiB |     14%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            3029      G   /usr/lib/xorg/Xorg                      211MiB |
|    0   N/A  N/A            3232      G   /usr/bin/gnome-shell                     60MiB |
|    0   N/A  N/A            4418      G   ...ns-seed-version --log-level=2         11MiB |
|    0   N/A  N/A            4492      G   ...rack-uuid=3190708988185955192        104MiB |
|    0   N/A  N/A            5731    C+G   /usr/bin/wezterm-gui                     35MiB |
|    0   N/A  N/A            5839      G   ...exec/xdg-desktop-portal-gnome          2MiB |
|    0   N/A  N/A           13179      G   /usr/share/code/code                     83MiB |
+-----------------------------------------------------------------------------------------+`
- cutlass_include_exists: `True`
- autopartitioner_reference_exists: `True`
- torch_cuda_arch_list: `unset; demo will use 8.0+PTX for extension build`
- torch_version: `2.11.0+cu130`
- torch_cuda_available: `True`
- torch_cuda_version: `13.0`
- gpu_name: `NVIDIA GeForce RTX 5060 Ti`
- compute_capability: `(12, 0)`
- triton_version: `3.6.0`

## 第一层：FX Graph 原始图

## 第二层：规范化图 / Pattern Match 结果

## 第三层：FusionPlan

## 第四层：TensorContract

## 第五层：AutoPartitionPlan

## Backend Legalize

## 第六层：RuntimeCall / Correctness / Benchmark

## Debug Log 摘要
本次运行没有捕获到需要写入 debug_log 的错误。

## 面试讲法
这个项目可以这样讲：我用 torch.compile 自定义 backend 接住 PyTorch 前端，让 TorchDynamo 负责动态图捕获，FX Graph 作为第一层 IR。随后我做 whole-graph pattern match，把 mm/add/gelu 识别成 GEMM_BIAS_GELU，把 linear1/gelu/linear2 识别成 MLP_TWO_GEMM，并记录融合覆盖的原始 FX nodes。

lowering 分成 FusionPlan、TensorContract、AutoPartitionPlan 和 RuntimeCall。TensorContract 说明 M/N/K、dtype、layout、stride、contiguous 状态和 bias shape；AutoPartitionPlan 说明 sm80 policy、tile shape、thread count、alignment、candidate 和 RoleA/RoleB/RoleC 选择；RuntimeCall 说明最终走 AutoPartitionCudaBackend 还是 PyTorchFallbackBackend。

AutoPartitioner 后端不是把计算偷换成 cuBLAS/Triton/torch.matmul，而是在 CUDA extension 中实例化仓库里的 AutoPartitioner RoleA/RoleB/RoleC，主 GEMM 使用其 GlobalToSharedCopy、SmemToRegCopyOperation、TiledMma 和 OutputRegisterToGlobalCopy。当前 epilogue 是 two-stage CUDA kernel，报告中明确说明，后续可以把 bias/GELU 合入 AutoPartitioner 的单 kernel epilogue contract。
