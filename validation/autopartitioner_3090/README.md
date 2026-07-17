# AutoPartitioner RTX 3090 验证

这里保存 AutoPartitioner 历史版本与官方 CUTLASS 对照的可审计验证结果。

- 实验结果：`results_20260715_0525/`
- 报告：`results_20260715_0525/AutoPartitioner_RTX3090_Report.md`
- 原始样本：`results_20260715_0525/raw_samples.csv`
- 官方对比：`results_20260715_0525/official_comparison.csv`
- 复现入口：`../../tools/auto_partitioner_bench/run_3090_history.sh`

工具代码仍位于项目标准的 `tools/auto_partitioner_bench/`，测试位于
`test/python/auto_partitioner/`；`.worktrees/` 只是开发期间的临时 Git 隔离目录，
不属于项目交付物。

从项目根目录运行：

```bash
tools/auto_partitioner_bench/run_3090_history.sh
```

默认输出会写入新的：
`validation/autopartitioner_3090/results_<UTC timestamp>/`。
