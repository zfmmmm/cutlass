# AutoPartitioner Nsight 瓶颈定位教程实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 RTX 5060 Ti 上实际复现 AutoPartitioner 优化时间线，产出一份能够教会读者独立使用 Nsight Systems 和 Nsight Compute 建立瓶颈证据链的中文教程。

**Architecture:** 当前工作树只承载 NVTX 探针、中文教程和小型文本/图片产物；历史 `v00-v08` 从提交 `15394589` 恢复到隔离 worktree，所有大型 `.nsys-rep`、`.ncu-rep` 与临时二进制写入 `build/auto_partitioner_nsight/`。采样遵循“先 Systems、后 Compute；先宽后窄；每个结论至少两项证据”的顺序，并始终将 AutoPartitioner 与一级流水官方基线置于同一控制变量下。

**Tech Stack:** CUDA 13.x/NVCC、CUTLASS/CuTe、NVTX3、Nsight Systems 2025.3.2、Nsight Compute 2025.3.0、Markdown、CSV、现有 benchmark 绘图功能。

## Global Constraints

- 所有新测量明确标注测试 GPU 为 NVIDIA GeForce RTX 5060 Ti（SM120）。
- `sm80_autopartition_gemm` 是编译到 SM120 的 SM80 风格实现，结果不得表述为原生 A100/RTX 30 的 SM80 性能。
- 教程正文、图注和分析使用中文；Nsight 指标名、命令和原生输出保持原样。
- 不新增封装 `nsys` 或 `ncu` 的脚本，所有采样使用原生命令。
- 不重置、覆盖或回退当前脏工作树中的用户修改。
- 性能结论使用重复的非 profile 计时；profile 中的 kernel 时间只用于结构和计数器分析。
- AutoPartitioner 与官方基线必须保持相同输入、输出、问题规模、tile、stage、warmup、iteration 和同步边界。

---

### Task 1: 建立隔离实验环境和环境清单

**Files:**
- Create: `build/auto_partitioner_nsight/environment.txt`
- Create: `build/auto_partitioner_nsight/commands/environment_commands.txt`
- Reference: `tools/auto_partitioner_bench/build_benchmarks.sh`
- Reference: `tools/auto_partitioner_bench/sm80_benchmark_common.hpp`

**Interfaces:**
- Consumes: 当前提交、历史提交 `15394589`、本机 CUDA/Nsight/GPU 环境。
- Produces: 隔离 worktree 路径、统一输出目录、可引用的环境原始输出。

- [ ] **Step 1: 使用 `superpowers:using-git-worktrees` 创建隔离历史 worktree**

使用技能选择安全目录，并将 `15394589` checkout 为 detached worktree。不得修改当前工作树。

- [ ] **Step 2: 记录 GPU、驱动、CUDA、编译器与 Nsight 版本**

Run:

```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,power.limit,clocks.max.sm,clocks.max.memory --format=csv
nvcc --version
nsys --version
ncu --version
nsys status --environment
```

Expected: 输出 RTX 5060 Ti、compute capability 12.0、Nsight Systems 2025.3.2 和 Nsight Compute 2025.3.0；如 system-wide CPU sampling 不可用，只使用 process-tree/CUDA trace。

- [ ] **Step 3: 查询当前版本真实支持的 Systems 报告和 Compute section**

Run:

```bash
nsys stats --help-reports
ncu --list-sets
ncu --list-sections
ncu --query-metrics-mode suffix --metrics sm__throughput,smsp__warp_issue_stalled_barrier
```

Expected: 保存原生名称，后续命令只使用本机版本实际存在的 report/section/metric。

- [ ] **Step 4: 记录当前工作树与历史快照来源**

Run:

```bash
git status --short
git show --stat --oneline 15394589
git ls-tree -r --name-only 15394589 tools/auto_partitioner_bench/snapshots
```

Expected: 当前用户修改保持不变，历史树能看到 `v00-v08` 和 `sm80_optimization_log.md`。

### Task 2: 为两个受控 benchmark 加入相同 NVTX 探针

**Files:**
- Modify: `tools/auto_partitioner_bench/sm80_benchmark_common.hpp`
- Modify: `include/cutlass/transform/collective/auto_partitioner/examples/production_gemm/sm80_autopartition_gemm.cu`
- Modify: `tools/auto_partitioner_bench/sm80_cutlass_official_gemm.cu`
- Modify: `tools/auto_partitioner_bench/build_benchmarks.sh`
- Test: both benchmark binaries

**Interfaces:**
- Consumes: 现有 `time_launch_ms(Launch, int, int, float&)` 与两个 `main()` 的初始化、验证、计时阶段。
- Produces: `autopartition_bench::NvtxRange`、统一的 `setup`、`correctness`、`warmup`、`timed`、`result-copy` NVTX ranges。

- [ ] **Step 1: 先写编译探针测试并确认当前代码没有 NVTX range**

Run:

```bash
rg -n 'nvtx|NvtxRange|NVTX' tools/auto_partitioner_bench/sm80_benchmark_common.hpp include/cutlass/transform/collective/auto_partitioner/examples/production_gemm/sm80_autopartition_gemm.cu tools/auto_partitioner_bench/sm80_cutlass_official_gemm.cu
```

Expected: 没有用于 benchmark 阶段划分的 NVTX range。

- [ ] **Step 2: 实现最小 RAII NVTX range**

在公共头文件中加入 `<nvtx3/nvToolsExt.h>`，实现只负责 `nvtxRangePushA(name)`/`nvtxRangePop()` 的不可复制 `NvtxRange`。range 名称固定为阶段名，不在 kernel 热路径中动态格式化字符串。

- [ ] **Step 3: 给公共计时函数标记 warmup 和 timed 区间**

将签名扩展为：

```cpp
template <class Launch>
bool time_launch_ms(
    Launch launch,
    int warmup,
    int iterations,
    float &elapsed_ms_out,
    char const *warmup_range,
    char const *timed_range);
```

两个 range 分别包住 warmup 循环和 CUDA event 计时循环。range 只帮助 Systems/Compute 选择 launch，不改变 CUDA event 的起止点。

- [ ] **Step 4: 在两个 benchmark 中加入对称阶段标记**

两个程序都使用相同名称：`setup`、`correctness-launch`、`result-copy`、`warmup`、`timed`。AutoPartitioner 额外的 `cp.async-zfill-probe` 单独标记，避免被误认为 GEMM。

- [ ] **Step 5: 更新链接参数并编译**

Run:

```bash
tools/auto_partitioner_bench/build_benchmarks.sh
```

Expected: 两个 SM120 二进制构建成功；如果 NVTX3 头采用动态注入无需链接库则不增加无效链接参数，否则只增加本机实际要求的 `-ldl` 或 `-lnvToolsExt`。

- [ ] **Step 6: 验证结果一致性**

Run:

```bash
build/auto_partitioner_bench/bin/sm80_autopartition_gemm --m=256 --n=256 --k=256 --warmup=2 --iterations=5
build/auto_partitioner_bench/bin/sm80_cutlass_official_gemm --m=256 --n=256 --k=256 --warmup=2 --iterations=5
```

Expected: 两者 `input_a_hash`、`input_b_hash`、`output_hash` 相同，`max_abs_diff` 低于已有阈值。

### Task 3: 建立非 profiler 公平性能基线

**Files:**
- Create: `build/auto_partitioner_nsight/timing/current_fair_sweep.csv`
- Create: `build/auto_partitioner_nsight/timing/current_fair_sweep.png`
- Create: `tools/auto_partitioner_bench/nsight_artifacts/current_fair_sweep_summary.csv`

**Interfaces:**
- Consumes: Task 2 的两个二进制。
- Produces: profile 前的重复 runtime/TFLOP/s 分布、输入输出 hash 公平性证据。

- [ ] **Step 1: 进行短正确性 sweep**

Run:

```bash
python3 tools/auto_partitioner_bench/run_gemm_sweep.py --arch sm80 --sizes 256,512 --warmup 2 --iterations 5 --repeat-runs 2 --verify-sizes 256,512 --output build/auto_partitioner_nsight/timing/correctness.csv
```

Expected: 所有实现通过；每个规模的输入和输出 hash 匹配。

- [ ] **Step 2: 进行完整公平 sweep**

Run:

```bash
python3 tools/auto_partitioner_bench/run_gemm_sweep.py --arch sm80 --sizes 256,512,1024,2048,4096,8192 --warmup 10 --iterations 50 --repeat-runs 5 --skip-reference --verify-sizes 256,512 --plot --output build/auto_partitioner_nsight/timing/current_fair_sweep.csv
```

Expected: 保存每次 raw sample、median runtime、TFLOP/s、hash；将 profile 前性能作为后续所有结论的基线。

- [ ] **Step 3: 记录时钟波动和测试边界**

Run before and after:

```bash
nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.current.sm,clocks.current.memory,utilization.gpu --format=csv
```

Expected: 教程解释消费级 GPU 的 DVFS/温度会带来波动，不能用单次结果声称提升。

### Task 4: 用 Nsight Systems 完成系统级定位

**Files:**
- Create: `build/auto_partitioner_nsight/nsys/current_ap.nsys-rep`
- Create: `build/auto_partitioner_nsight/nsys/current_official.nsys-rep`
- Create: `tools/auto_partitioner_bench/nsight_artifacts/nsys_current_ap_stats.txt`
- Create: `tools/auto_partitioner_bench/nsight_artifacts/nsys_current_official_stats.txt`

**Interfaces:**
- Consumes: NVTX 标记后的当前二进制。
- Produces: host/API/GPU 时间线证据，回答“慢在 kernel 内还是 kernel 外”。

- [ ] **Step 1: 直接采集 AutoPartitioner Systems report**

Run:

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --force-overwrite=true --output=build/auto_partitioner_nsight/nsys/current_ap build/auto_partitioner_bench/bin/sm80_autopartition_gemm --m=2048 --n=2048 --k=2048 --warmup=5 --iterations=20 --skip-reference
```

Expected: 生成 `.nsys-rep`，时间线中能区分 correctness、warmup、timed 和 zfill probe。

- [ ] **Step 2: 使用完全相同参数采集官方基线**

Run:

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --force-overwrite=true --output=build/auto_partitioner_nsight/nsys/current_official build/auto_partitioner_bench/bin/sm80_cutlass_official_gemm --m=2048 --n=2048 --k=2048 --warmup=5 --iterations=20 --skip-reference
```

Expected: 两份 report 的 timed range 有相同 launch 数量。

- [ ] **Step 3: 用原生 `nsys stats` 导出摘要**

先以 `nsys stats --help-reports` 的实际名称为准，然后分别运行：

```bash
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum,cuda_gpu_mem_time_sum build/auto_partitioner_nsight/nsys/current_ap.nsys-rep
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum,cuda_gpu_mem_time_sum build/auto_partitioner_nsight/nsys/current_official.nsys-rep
```

Expected: 原始表格能看出 kernel 总时间、平均/最小/最大时长、CUDA API 同步时间、memcpy 时间和 NVTX 阶段占比。

- [ ] **Step 4: 建立 Systems 判断链**

在教程中逐项回答：timed 区间是否被 memcpy 污染、kernel 之间是否有 CPU launch gap、是否出现意外 `cudaDeviceSynchronize`、kernel 时长方差是否过大、两实现 launch 数是否一致。只有确认主要差异位于 GEMM kernel 内，才进入 NCU。

### Task 5: 用 Nsight Compute 从宽到窄定位当前瓶颈

**Files:**
- Create: `build/auto_partitioner_nsight/ncu/current_ap_basic.ncu-rep`
- Create: `build/auto_partitioner_nsight/ncu/current_official_basic.ncu-rep`
- Create: `build/auto_partitioner_nsight/ncu/current_ap_focused.ncu-rep`
- Create: `build/auto_partitioner_nsight/ncu/current_official_focused.ncu-rep`
- Create: `tools/auto_partitioner_bench/nsight_artifacts/ncu_current_basic.txt`
- Create: `tools/auto_partitioner_bench/nsight_artifacts/ncu_current_focused.txt`

**Interfaces:**
- Consumes: Task 4 证明是 kernel 内瓶颈的目标 launch。
- Produces: SOL、occupancy、scheduler、warp stall、tensor pipe、memory 和 bank-conflict 的交叉证据。

- [ ] **Step 1: 发现真实 demangled kernel 名称**

Run:

```bash
ncu --set basic --kernel-name-base demangled --launch-skip 1 --launch-count 1 build/auto_partitioner_bench/bin/sm80_autopartition_gemm --m=2048 --n=2048 --k=2048 --warmup=0 --iterations=1 --skip-reference
ncu --set basic --kernel-name-base demangled --launch-skip 1 --launch-count 1 build/auto_partitioner_bench/bin/sm80_cutlass_official_gemm --m=2048 --n=2048 --k=2048 --warmup=0 --iterations=1 --skip-reference
```

Expected: 从原始输出复制 kernel 名称模式，不凭源码猜 filter。

- [ ] **Step 2: 采集低成本 basic/SOL/launch/occupancy 报告**

Run with the discovered filters:

```bash
ncu --set basic --kernel-name-base demangled --kernel-name 'regex:.*sm80_autopartition_gemm_kernel.*' --launch-skip 1 --launch-count 1 --export build/auto_partitioner_nsight/ncu/current_ap_basic --force-overwrite build/auto_partitioner_bench/bin/sm80_autopartition_gemm --m=2048 --n=2048 --k=2048 --warmup=0 --iterations=1 --skip-reference
```

对官方 kernel 使用相同参数和对应 filter。Expected: 首先判断 compute/memory throughput、grid waves、寄存器、shared memory 和 occupancy limit，不立即收集 `full`。

- [ ] **Step 3: 采集 focused 原始指标**

Run:

```bash
ncu --kernel-name-base demangled --kernel-name 'regex:.*sm80_autopartition_gemm_kernel.*' --launch-skip 1 --launch-count 1 --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,smsp__inst_executed_pipe_tensor.sum,smsp__average_warps_active_per_issue_active,smsp__warp_issue_stalled_barrier_per_warp_active,smsp__warp_issue_stalled_long_scoreboard_per_warp_active,smsp__warp_issue_stalled_short_scoreboard_per_warp_active,smsp__warp_issue_stalled_mio_throttle_per_warp_active,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active,smsp__warp_issue_stalled_not_selected_per_warp_active,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum --export build/auto_partitioner_nsight/ncu/current_ap_focused --force-overwrite build/auto_partitioner_bench/bin/sm80_autopartition_gemm --m=2048 --n=2048 --k=2048 --warmup=0 --iterations=1 --skip-reference
```

对官方 kernel 重复完全相同 metric 列表。若 SM120 的 metric 名称发生变化，先用 `ncu --query-metrics-mode suffix` 查询并在教程中记录替代关系。

- [ ] **Step 4: 用 section 交叉验证，不依赖孤立百分比**

Run:

```bash
ncu --section SchedulerStats --section WarpStateStats --section Occupancy --section MemoryWorkloadAnalysis --kernel-name-base demangled --kernel-name 'regex:.*sm80_autopartition_gemm_kernel.*' --launch-skip 1 --launch-count 1 build/auto_partitioner_bench/bin/sm80_autopartition_gemm --m=2048 --n=2048 --k=2048 --warmup=0 --iterations=1 --skip-reference
```

Expected: 对每个候选瓶颈至少检查“利用率/吞吐 + stall + 资源或事务计数”三类中的两类。

- [ ] **Step 5: 做源码/SASS 关联**

Run:

```bash
ncu --section SourceCounters --section InstructionStats --kernel-name-base demangled --kernel-name 'regex:.*sm80_autopartition_gemm_kernel.*' --launch-skip 1 --launch-count 1 --export build/auto_partitioner_nsight/ncu/current_ap_source --force-overwrite build/auto_partitioner_bench/bin/sm80_autopartition_gemm --m=2048 --n=2048 --k=2048 --warmup=0 --iterations=1 --skip-reference
```

Expected: `-lineinfo` 能将热点/采样关联到主循环、barrier、shared load/store 或 epilogue；若某个源级 sampling 功能在 SM120/当前权限下不可用，保留原始报错并改用 SASS view，不伪造截图。

### Task 6: 重建并实测 v00-v08 优化时间线

**Files:**
- Reference: isolated worktree `tools/auto_partitioner_bench/snapshots/sm80_autopartition_gemm_v00_baseline_single_stage.cu`
- Reference: snapshots `v01` through `v08`
- Create: `build/auto_partitioner_nsight/history/bin/`
- Create: `build/auto_partitioner_nsight/history/timing.csv`
- Create: `tools/auto_partitioner_bench/nsight_artifacts/history_metrics.csv`
- Create: `tools/auto_partitioner_bench/nsight_artifacts/history_diffs.txt`

**Interfaces:**
- Consumes: 历史快照与当前 CUDA 13/SM120 编译环境。
- Produces: 每阶段可执行文件、重复 timing、代表性 NCU 指标、源码 diff、可复现/不可复现标签。

- [ ] **Step 1: 使用原生 NVCC 命令逐个编译快照**

使用 `build_benchmarks.sh` 中已展开的 include、define、`-O3`、`-lineinfo` 和 SM120 gencode 参数，直接将每个 `v00-v08` `.cu` 编译为独立二进制；命令逐条写入教程，不创建编译包装脚本。

- [ ] **Step 2: 每个版本先做 256 正确性，再做 1024/2048 重复计时**

Run pattern:

```bash
build/auto_partitioner_nsight/history/bin/v00 --m=256 --n=256 --k=256 --warmup=2 --iterations=5
build/auto_partitioner_nsight/history/bin/v00 --m=2048 --n=2048 --k=2048 --warmup=10 --iterations=50 --skip-reference
```

Expected: 每个结果标注当前重新测得或仅历史可用；失败版本保留编译/运行错误原文。

- [ ] **Step 3: 为关键转折点采集相同 focused metrics**

至少采集 `v00`、`v02`、`v05`、`v06`、`v07/v08` 和官方基线。对纯主机对象 hoist 等变化，先用 Systems/kernel timing 判断是否产生可测差异；对 launch bounds、shared union、epilogue swizzle 分别重点观察 registers/occupancy、static shared memory/residency、bank conflicts/output mapping。

- [ ] **Step 4: 生成相邻版本源码 diff**

Run pattern:

```bash
git diff --no-index tools/auto_partitioner_bench/snapshots/sm80_autopartition_gemm_v00_baseline_single_stage.cu tools/auto_partitioner_bench/snapshots/sm80_autopartition_gemm_v01_policy_tiled_g2s.cu
```

Expected: 教程只引用与该次假设相关的 diff hunk，并解释它应改变哪个硬件行为。

- [ ] **Step 5: 复核失败实验**

从历史日志恢复 naive double-buffer 和 launch-bounds=2 的命令/结果；能够安全重建则重测，不能重建则明确标注历史结果。说明为何“barrier stall 高”不足以推出“简单双缓冲必然更快”。

### Task 7: 编写完整中文学习笔记和图表

**Files:**
- Create: `docs/auto_partitioner_nsight_bottleneck_tutorial.md`
- Create: `tools/auto_partitioner_bench/nsight_artifacts/diagnostic_tree.png` or an equivalent Markdown-native diagram
- Create: `tools/auto_partitioner_bench/nsight_artifacts/history_tflops.png`
- Create: `tools/auto_partitioner_bench/nsight_artifacts/history_bottlenecks.png`

**Interfaces:**
- Consumes: Tasks 1-6 的原始输出、reports、CSV 和 diff。
- Produces: 中文主教程、诊断树、性能和关键指标变化图。

- [ ] **Step 1: 写 Nsight Systems 中文课程**

必须包含命令逐项解释、原生输出块、表格列释义、GUI 时间线阅读顺序、NVTX 加探针方式、何时不能进入 NCU，以及本项目 Systems 实测判断。

- [ ] **Step 2: 写 Nsight Compute 中文课程**

必须解释 metric 命名层次（unit/subunit/event/reduction/normalization）、raw count 与 rate、active 与 elapsed、peak sustained 与 theoretical、section 与 metric、replay 开销、kernel filter 和 launch filter。

- [ ] **Step 3: 建立全面指标词典**

覆盖 SOL、occupancy、waves、scheduler、eligible/active warps、各主要 stall、tensor pipe、DRAM/L2/L1/shared、bank conflicts、register/shared resource limits、source/SASS correlation。每项给出定义、误区、交叉指标和下一条命令。

- [ ] **Step 4: 按 v00-v08 写完整判断叙事**

每阶段按“症状 → 命令 → 原始输出 → 解释 → 假设 → 交叉验证 → diff → 重测 → 新瓶颈”结构书写，明确体现是根据证据逐步优化，而不是事后把修改包装成结论。

- [ ] **Step 5: 绘制图表**

使用已有 sweep 绘图能力或直接从 CSV 绘图；不得创建 Nsight 命令包装器。图中同时显示 AutoPartitioner、受控官方基线、版本节点和误差/重复样本范围。指标图至少展示 occupancy/shared footprint、barrier/scoreboard stalls、tensor utilization 和 bank conflicts 的演变。

- [ ] **Step 6: 写独立诊断清单**

教程最后给出面对未知 CUDA kernel 时从正确性、公平计时、Systems、Compute、源码关联到复测的逐项 checklist；每个分支给出下一条原生命令。

### Task 8: 最终复现和教程审计

**Files:**
- Modify: `docs/auto_partitioner_nsight_bottleneck_tutorial.md`
- Verify: all committed text/CSV/PNG artifacts

**Interfaces:**
- Consumes: 完整教程和当前 benchmark。
- Produces: 可复现、无虚构输出、边界清晰的最终结果。

- [ ] **Step 1: 使用 `superpowers:verification-before-completion` 执行最终验证**

重新运行 benchmark 构建、256 正确性、当前公平 sweep 的代表规模、Systems 采集和两个 focused Compute 采集。

- [ ] **Step 2: 审计所有数字来源**

每个新测量数字必须能定位到 raw `.txt`、CSV 或 `.ncu-rep/.nsys-rep`；历史数字必须带“历史记录”标签。

- [ ] **Step 3: 审计公平性和表述**

确认官方比较是相同一级流水、相同 tile/类型/布局/输入/计时协议；确认所有结果写作“RTX 5060 Ti（SM120）上的 SM80 风格实现”。

- [ ] **Step 4: 审计命令可执行性**

逐个检查教程命令的路径、引号、kernel regex、report 名称和输出文件名。不能在当前机器运行的原生 SM80 命令单独标注“未在本机验证”。

- [ ] **Step 5: 检查文档完整性**

Run:

```bash
rg -n 'TBD|TODO|待补充|稍后|PLACEHOLDER' docs/auto_partitioner_nsight_bottleneck_tutorial.md tools/auto_partitioner_bench/nsight_artifacts
git diff --check
```

Expected: 没有占位符或空白错误；所有图片链接存在；教程从零开始可顺序执行。
