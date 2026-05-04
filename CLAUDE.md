# CLAUDE.md - Context & Guardrails

## 1. 项目概览 (Project Context)
- **WHAT:** 基于 C++ 与 CUTLASS/CuTe 实现的 `AutoPartitioner` 布局自动生成引擎，利用模板推导为不同架构（当前侧重 SM80/SM100）生成全局、共享内存、寄存器及 MMA 之间的最优数据 Layout 及 Swizzle 方案。
- **WHY:** 简化 Kernel 开发，实现单张量多面体菜单模式，彻底解耦布局图纸生成与运行时资源管理。
- **核心技术栈:** C++17, CUDA, CUTLASS, CuTe。

## 2. 架构与目录铁律 (Architecture Rules)
- **核心生成器 (HOW):** `auto_partitioner.hpp`, `auto_partitioner_builder.hpp`
- **架构特化层:** `arch/sm80_policy.hpp` (SM100 规范待补充)
- **测试与验证:** `test_memory_patterns.cu`, `test_simt.cu`
- **只读参考域:** `include/cutlass/gemm/collective/` (特别是 sm80/sm90 的官方实现)。**绝对禁止**修改官方 CUTLASS 依赖库源码。
- [详细文件结构约束请参考 @.claude/skills/architecture-rules.md]

## 3. 编码规范 (Coding Standards)
- **单张量多面体模式 (Monolithic Traits Bundle):** 绝不允许在推导器中耦合 A 和 B 的参数。`AutoPartitioner` 必须针对单一输入张量，在同一偏特化结构体中同时输出 RoleA (MMA A), RoleB (MMA B), RoleC (Epilogue) 的全套最优 Layout。
- **零资源管理 (Zero Resource Management):** Partitioner 仅仅是“图纸生成器”。**严禁**在推导引擎内部包含 Pipeline Stages 计算、`__shared__` 内存块分配或 `local_tile` 坐标系转换逻辑。资源管理全权交由下游 Kernel Mainloop 负责。
- **SFINAE 静态路由:** 彻底弃用大类嵌套 `struct`。必须使用带 `Enable = void` 的空壳主模板，通过 `std::enable_if_t` 依据 `ArchTag` (Sm80/Sm100)、`OpClass` (TensorOp/Simt) 和数据类型，将所有推导逻辑写入外部偏特化版本进行精准路由。

## 4. 关键工作流指令 (Workflows)
- **构建与测试编译 (CUDA SIMT):**
  `nvcc test.cu -o test_simt -lineinfo -std=c++17 -O3 -arch=sm_80 -I/home/zfm/Desktop/cutlass/include -I/home/zfm/Desktop/cutlass/tools/util/include -I.`
- **性能分析 (Profiling):** 强制使用 Nsight Systems (`nsys`) 与 Nsight Compute (`ncu`) 进行 Kernel 级别的内存访问和指令级优化分析。

## 5. AI 执行边界与熔断机制 (Agent Guardrails)
- **执行边界:** 你的核心职责是编写和推导基于 SFINAE 的 C++ 模板元编程代码。禁止主动重构脱离当前目标架构(SM80/SM100)的基础设计。
- **错误熔断机制 (CIRCUIT BREAKER - 铁律) :**
  - 在尝试修复编译报错（特别是复杂的 C++ 模板实例化错误）、处理依赖冲突或修正运行结果时，**连续重试/修改代码的次数绝对不得超过 3 次！**
  - **一旦达到 3 次失败**，必须立刻停止执行！
  - 停止后，必须保留最后一次报错现场的核心 Log，输出简明的失败原因总结，并向人类用户请求接管。
  - **严禁**擅自猜测、无限生成无用代码测试或陷入死循环。