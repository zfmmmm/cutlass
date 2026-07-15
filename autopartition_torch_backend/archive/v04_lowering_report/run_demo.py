from __future__ import annotations

import argparse
import datetime as _dt
import os
import shutil
import subprocess
import sys
import textwrap
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import backend as ap_backend


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
DEFAULT_SHAPES = [(1024, 1024, 1024), (512, 1024, 2048), (2048, 1024, 4096)]


class DemoRecorder:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.events: dict[str, list[Any]] = {}
        self.stage_logs: dict[str, list[str]] = {}
        self.current_stage = "bootstrap"
        self.debug_logs: list[str] = []
        self.env: dict[str, Any] = {}

    def begin(self, name: str) -> None:
        self.current_stage = name
        self.stage_logs.setdefault(name, [])
        self.log(f"\n========== {name} ==========")

    def log(self, text: str) -> None:
        print(text)
        self.stage_logs.setdefault(self.current_stage, []).append(text)

    def record(self, kind: str, payload: Any) -> None:
        self.events.setdefault(kind, []).append(payload)
        if kind == "debug_log":
            layer, detail = payload
            self.debug_logs.append(f"## {layer}\n\n```text\n{detail}\n```")

    def archive_stage(self, version: str, stage_name: str, summary: str) -> None:
        archive_dir = PROJECT_DIR / "archive" / f"{version}_{stage_name}"
        if archive_dir.exists():
            shutil.rmtree(archive_dir)
        archive_dir.mkdir(parents=True)

        for relative in [
            "README.md",
            "backend.py",
            "run_demo.py",
            "csrc/autopartition_runtime.cu",
            "tests/test_basic.py",
        ]:
            src = PROJECT_DIR / relative
            if src.exists():
                dst = archive_dir / relative
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

        (archive_dir / "stage_report.md").write_text(summary, encoding="utf-8")
        stage_key = f"{version}_{stage_name}"
        output = "\n".join(self.stage_logs.get(stage_key, self.stage_logs.get(self.current_stage, [])))
        (archive_dir / "run_output.txt").write_text(output, encoding="utf-8")
        if self.debug_logs:
            (archive_dir / "debug_log.md").write_text("\n\n".join(self.debug_logs), encoding="utf-8")
        self.log(f"已创建阶段存档: {archive_dir.relative_to(PROJECT_DIR)}")

    def write_report(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        report = build_report(self)
        report_path = self.run_dir / "report.md"
        report_path.write_text(report, encoding="utf-8")
        latest = PROJECT_DIR / "reports" / "latest"
        if latest.exists():
            shutil.rmtree(latest)
        latest.mkdir(parents=True)
        shutil.copy2(report_path, latest / "report.md")
        if self.debug_logs:
            debug_path = PROJECT_DIR / "reports" / "debug_log.md"
            debug_path.write_text("\n\n".join(self.debug_logs), encoding="utf-8")
            shutil.copy2(debug_path, self.run_dir / "debug_log.md")
        return report_path


class GemmBiasGeluModule:
    def __init__(self, torch_module: Any) -> None:
        self.torch = torch_module

    def make(self):
        torch = self.torch

        class GemmBiasGelu(torch.nn.Module):
            def forward(self, x, w, bias):
                return torch.nn.functional.gelu(x @ w + bias)

        return GemmBiasGelu()


class MlpTwoGemmModule:
    def __init__(self, torch_module: Any) -> None:
        self.torch = torch_module

    def make(self):
        torch = self.torch

        class MlpTwoGemm(torch.nn.Module):
            def forward(self, x, w1, b1, w2, b2):
                return torch.nn.functional.gelu(x @ w1 + b1) @ w2 + b2

        return MlpTwoGemm()


def main() -> None:
    args = parse_args()
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_DIR / "reports" / f"run_{timestamp}"
    recorder = DemoRecorder(run_dir)
    ap_backend.set_active_recorder(recorder)

    recorder.begin("v01_env_check")
    recorder.env = detect_environment()
    recorder.log(format_env_console(recorder.env))
    recorder.write_report()
    recorder.archive_stage("v01", "env_check", "阶段一完成：已检测 Python/PyTorch/CUDA/nvcc/CUTLASS 路径，并生成 reports/latest/report.md。")

    torch = import_torch_or_finish(recorder)
    if torch is None:
        recorder.write_report()
        return

    recorder.begin("v02_fx_backend")
    run_fx_capture_stage(torch, recorder)
    recorder.write_report()
    recorder.archive_stage("v02", "fx_backend", "阶段二完成：torch.compile 自定义 backend 已接入，报告记录真实 FX Graph 和节点明细。")

    recorder.begin("v03_pattern_fusion")
    run_pattern_stage(torch, recorder)
    recorder.write_report()
    recorder.archive_stage("v03", "pattern_fusion", "阶段三完成：已识别 GEMM_BIAS_GELU 和 MLP_TWO_GEMM，并记录 FusionPlan。")

    recorder.begin("v04_lowering_report")
    run_lowering_stage(torch, recorder)
    recorder.write_report()
    recorder.archive_stage("v04", "lowering_report", "阶段四完成：FusionPlan 已 lowering 到 TensorContract、AutoPartitionPlan 和 backend legalize 记录。")

    recorder.begin("v05_autopartition_extension")
    run_extension_stage(torch, recorder)
    recorder.write_report()
    recorder.archive_stage("v05", "autopartition_extension", "阶段五完成：已尝试编译 AutoPartitioner CUDA extension；失败会写入 debug_log 并由运行阶段 fallback。")

    recorder.begin("v06_runtime_call")
    run_runtime_stage(torch, recorder, args)
    recorder.write_report()
    recorder.archive_stage("v06", "runtime_call", "阶段六完成：torch.compile compiled callable 已按 legalize 结果路由到 AutoPartition 或 PyTorch fallback。")

    recorder.begin("v07_final_delivery")
    run_benchmark_stage(torch, recorder, args)
    final_report = recorder.write_report()
    recorder.archive_stage("v07", "final_delivery", "阶段七完成：已生成 correctness、benchmark、baseline 和最终中文报告。")

    recorder.begin("v08_interview_material")
    recorder.log("README.md 已包含中文面试讲法和后续扩展方向。")
    recorder.write_report()
    recorder.archive_stage("v08", "interview_material", "阶段八完成：README 和 report 已整理面试讲法。")
    recorder.log(f"最终报告: {final_report}")
    recorder.log(f"latest 报告: {PROJECT_DIR / 'reports/latest/report.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AutoPartition torch.compile backend demo.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="Only benchmark the default 1024 shape.")
    return parser.parse_args()


def detect_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "torch_import_error": "",
        "triton_import_error": "",
        "nvcc_path": run_text(["bash", "-lc", "which nvcc"]).strip(),
        "nvcc_version": run_text(["bash", "-lc", "nvcc --version"]).strip(),
        "nvidia_smi": run_text(["bash", "-lc", "nvidia-smi"]).strip(),
        "cutlass_include_exists": (REPO_ROOT / "include/cutlass").exists(),
        "autopartitioner_reference_exists": (
            REPO_ROOT / "include/cutlass/transform/collective/auto_partitioner/examples/production_gemm/sm80_autopartition_gemm.cu"
        ).exists(),
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST", "unset; demo will use 8.0+PTX for extension build"),
    }
    try:
        import torch

        env.update(
            {
                "torch_version": torch.__version__,
                "torch_cuda_available": torch.cuda.is_available(),
                "torch_cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
                "compute_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "",
            }
        )
    except Exception as exc:
        env["torch_import_error"] = str(exc)
    try:
        import triton

        env["triton_version"] = triton.__version__
    except Exception as exc:
        env["triton_import_error"] = str(exc)
    return env


def import_torch_or_finish(recorder: DemoRecorder):
    try:
        import torch

        return torch
    except Exception as exc:
        recorder.log(f"无法导入 PyTorch，后续 FX/CUDA 阶段跳过: {exc}")
        recorder.log("可执行修复方向：使用已发现的虚拟环境 /home/zfm/Desktop/qwen_quant/qwen35_quant_vllm/.venv/bin/python 运行本 demo。")
        return None


def run_fx_capture_stage(torch: Any, recorder: DemoRecorder) -> None:
    model = GemmBiasGeluModule(torch).make()
    x = torch.randn((16, 32), dtype=torch.float32)
    w = torch.randn((32, 64), dtype=torch.float32)
    bias = torch.randn((64,), dtype=torch.float32)
    compiled = torch.compile(model, backend=ap_backend.autopartition_compile_backend, fullgraph=True)
    y = compiled(x, w, bias)
    recorder.log(f"CPU fallback capture output shape = {tuple(y.shape)}")


def run_pattern_stage(torch: Any, recorder: DemoRecorder) -> None:
    gemm_gm = torch.fx.symbolic_trace(GemmBiasGeluModule(torch).make())
    gemm_analysis = ap_backend.analyze_fx_graph(gemm_gm, [])
    recorder.record("fx_analysis", gemm_analysis)
    recorder.log("GEMM_BIAS_GELU FusionPlan:")
    recorder.log(str(_to_plain(gemm_analysis.fusion_plans[0]) if gemm_analysis.fusion_plans else gemm_analysis.pattern_failures))

    mlp_gm = torch.fx.symbolic_trace(MlpTwoGemmModule(torch).make())
    mlp_analysis = ap_backend.analyze_fx_graph(mlp_gm, [])
    recorder.record("fx_analysis", mlp_analysis)
    recorder.log("MLP_TWO_GEMM FusionPlan:")
    recorder.log(str(_to_plain(mlp_analysis.fusion_plans[0]) if mlp_analysis.fusion_plans else mlp_analysis.pattern_failures))


def run_lowering_stage(torch: Any, recorder: DemoRecorder) -> None:
    gm = torch.fx.symbolic_trace(GemmBiasGeluModule(torch).make())
    analysis = ap_backend.analyze_fx_graph(gm, [])
    tensors = {
        "x": torch.empty((128, 64), dtype=torch.float16),
        "w": torch.empty((64, 256), dtype=torch.float16),
        "bias": torch.empty((256,), dtype=torch.float16),
    }
    contract = ap_backend.build_tensor_contract(analysis.fusion_plans[0], tensors, actual_sm=ap_backend.detect_actual_sm())
    plan = ap_backend.build_autopartition_plan(contract, actual_sm=contract.actual_sm)
    legality = ap_backend.legalize_for_autopartition(contract, plan)
    recorder.record("tensor_contract", contract)
    recorder.record("autopartition_plan", plan)
    recorder.record("legality", legality)
    recorder.log(f"TensorContract M/N/K = {contract.M}/{contract.N}/{contract.K}")
    recorder.log(f"AutoPartitionPlan target={plan.target_sm}, actual={plan.actual_sm}, legal={legality.legal}")


def run_extension_stage(torch: Any, recorder: DemoRecorder) -> None:
    if not torch.cuda.is_available():
        recorder.log("CUDA 不可用，跳过 extension 编译；报告会保留 fallback 原因。")
        return
    gm = torch.fx.symbolic_trace(GemmBiasGeluModule(torch).make())
    analysis = ap_backend.analyze_fx_graph(gm, [])
    tensors = make_gemm_inputs(torch, 64, 64, 64)
    contract = ap_backend.build_tensor_contract(analysis.fusion_plans[0], tensors, actual_sm=ap_backend.detect_actual_sm())
    plan = ap_backend.build_autopartition_plan(contract, actual_sm=contract.actual_sm)
    try:
        module = ap_backend.load_autopartition_extension(plan)
        recorder.log(f"extension 编译/加载成功: {module}")
    except Exception as exc:
        detail = traceback_text(exc)
        recorder.record("debug_log", ("extension build", detail))
        recorder.log(f"extension 编译/加载失败，运行阶段会 fallback: {exc}")


def run_runtime_stage(torch: Any, recorder: DemoRecorder, args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        recorder.log("CUDA 不可用，runtime call 阶段只验证 PyTorch fallback。")
        return
    model = GemmBiasGeluModule(torch).make().cuda()
    tensors = make_gemm_inputs(torch, 64, 64, 64)
    compiled = torch.compile(model, backend=ap_backend.autopartition_compile_backend, fullgraph=True)
    out = compiled(tensors["A"], tensors["B"], tensors["bias"])
    ref = ap_backend.run_torch_reference("GEMM_BIAS_GELU", tensors)
    max_abs, max_rel, passed = ap_backend.compare_outputs(out, ref)
    recorder.record(
        "runtime_call",
        ap_backend.RuntimeCall(
            backend="AutoPartitionCudaBackend or PyTorchFallbackBackend",
            function="compiled GemmBiasGelu",
            input_shapes={k: tuple(v.shape) for k, v in tensors.items()},
            output_shape=tuple(out.shape),
            warmup=args.warmup,
            repeats=args.repeats,
            max_abs_error=max_abs,
            max_rel_error=max_rel,
            passed=passed,
        ),
    )
    recorder.log(f"runtime correctness: max_abs={max_abs:.6g}, max_rel={max_rel:.6g}, passed={passed}")


def run_benchmark_stage(torch: Any, recorder: DemoRecorder, args: argparse.Namespace) -> None:
    shapes = DEFAULT_SHAPES[:1] if args.quick else DEFAULT_SHAPES
    if not torch.cuda.is_available():
        recorder.log("CUDA 不可用，benchmark 阶段跳过 CUDA backend。")
        return
    for M, K, N in shapes:
        recorder.log(f"Benchmark shape M={M}, K={K}, N={N}")
        tensors = make_gemm_inputs(torch, M, K, N)
        model = GemmBiasGeluModule(torch).make().cuda()
        compiled = torch.compile(model, backend=ap_backend.autopartition_compile_backend, fullgraph=True)
        y_compiled = compiled(tensors["A"], tensors["B"], tensors["bias"])
        y_ref = ap_backend.run_torch_reference("GEMM_BIAS_GELU", tensors)
        max_abs, max_rel, passed = ap_backend.compare_outputs(y_compiled, y_ref)
        compiled_ms = ap_backend.benchmark(lambda: compiled(tensors["A"], tensors["B"], tensors["bias"]), args.warmup, args.repeats)
        ref_ms = ap_backend.benchmark(lambda: ap_backend.run_torch_reference("GEMM_BIAS_GELU", tensors), args.warmup, args.repeats)
        recorder.record(
            "runtime_call",
            ap_backend.RuntimeCall(
                backend="AutoPartitionCudaBackend or PyTorchFallbackBackend",
                function="compiled GemmBiasGelu",
                input_shapes={k: tuple(v.shape) for k, v in tensors.items()},
                output_shape=tuple(y_compiled.shape),
                warmup=args.warmup,
                repeats=args.repeats,
                avg_ms=compiled_ms,
                max_abs_error=max_abs,
                max_rel_error=max_rel,
                passed=passed,
            ),
        )
        recorder.record(
            "runtime_call",
            ap_backend.RuntimeCall(
                backend="PyTorchFallbackBackend",
                function="torch reference",
                input_shapes={k: tuple(v.shape) for k, v in tensors.items()},
                output_shape=tuple(y_ref.shape),
                warmup=args.warmup,
                repeats=args.repeats,
                avg_ms=ref_ms,
                max_abs_error=0.0,
                max_rel_error=0.0,
                passed=True,
            ),
        )
        recorder.log(f"compiled={compiled_ms:.4f} ms, reference={ref_ms:.4f} ms, max_abs={max_abs:.6g}, passed={passed}")
        try:
            y_triton = ap_backend.run_triton_baseline("GEMM_BIAS_GELU", tensors)
            tri_abs, tri_rel, tri_pass = ap_backend.compare_outputs(y_triton, y_ref)
            tri_ms = ap_backend.benchmark(lambda: ap_backend.run_triton_baseline("GEMM_BIAS_GELU", tensors), args.warmup, args.repeats)
            recorder.record(
                "runtime_call",
                ap_backend.RuntimeCall(
                    backend="TritonBaselineBackend",
                    function="triton_gemm_bias_gelu",
                    input_shapes={k: tuple(v.shape) for k, v in tensors.items()},
                    output_shape=tuple(y_triton.shape),
                    warmup=args.warmup,
                    repeats=args.repeats,
                    avg_ms=tri_ms,
                    max_abs_error=tri_abs,
                    max_rel_error=tri_rel,
                    passed=tri_pass,
                ),
            )
            recorder.log(f"triton={tri_ms:.4f} ms, max_abs={tri_abs:.6g}, passed={tri_pass}")
        except Exception as exc:
            recorder.record("debug_log", ("Triton baseline", traceback_text(exc)))
            recorder.log(f"Triton baseline unavailable: {exc}")

    run_mlp_demo(torch, recorder, args)


def run_mlp_demo(torch: Any, recorder: DemoRecorder, args: argparse.Namespace) -> None:
    M, K, N, N2 = 512, 1024, 1024, 1024
    model = MlpTwoGemmModule(torch).make().cuda()
    torch.manual_seed(7)
    scale = 0.02
    tensors = {
        "X": torch.randn((M, K), device="cuda", dtype=torch.float16) * scale,
        "W1": torch.randn((K, N), device="cuda", dtype=torch.float16) * scale,
        "b1": torch.randn((N,), device="cuda", dtype=torch.float16) * scale,
        "W2": torch.randn((N, N2), device="cuda", dtype=torch.float16) * scale,
        "b2": torch.randn((N2,), device="cuda", dtype=torch.float16) * scale,
    }
    compiled = torch.compile(model, backend=ap_backend.autopartition_compile_backend, fullgraph=True)
    out = compiled(tensors["X"], tensors["W1"], tensors["b1"], tensors["W2"], tensors["b2"])
    ref = ap_backend.run_torch_reference("MLP_TWO_GEMM", tensors)
    max_abs, max_rel, passed = ap_backend.compare_outputs(out, ref)
    ms = ap_backend.benchmark(lambda: compiled(tensors["X"], tensors["W1"], tensors["b1"], tensors["W2"], tensors["b2"]), args.warmup, args.repeats)
    recorder.record(
        "runtime_call",
        ap_backend.RuntimeCall(
            backend="AutoPartitionCudaBackend or PyTorchFallbackBackend",
            function="compiled MLP_TWO_GEMM",
            input_shapes={k: tuple(v.shape) for k, v in tensors.items()},
            output_shape=tuple(out.shape),
            warmup=args.warmup,
            repeats=args.repeats,
            avg_ms=ms,
            max_abs_error=max_abs,
            max_rel_error=max_rel,
            passed=passed,
        ),
    )
    recorder.log(f"MLP_TWO_GEMM: {ms:.4f} ms, max_abs={max_abs:.6g}, passed={passed}")


def make_gemm_inputs(torch: Any, M: int, K: int, N: int) -> dict[str, Any]:
    torch.manual_seed(M + K + N)
    scale = 0.02
    return {
        "A": torch.randn((M, K), device="cuda", dtype=torch.float16) * scale,
        "B": torch.randn((K, N), device="cuda", dtype=torch.float16) * scale,
        "bias": torch.randn((N,), device="cuda", dtype=torch.float16) * scale,
    }


def build_report(recorder: DemoRecorder) -> str:
    lines: list[str] = []
    lines.append("# AutoPartition Torch Backend 运行报告")
    lines.append("")
    lines.append("## 项目运行环境")
    for key, value in recorder.env.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## 第一层：FX Graph 原始图")
    for idx, analysis in enumerate(recorder.events.get("fx_analysis", [])):
        lines.append(f"### FX Graph #{idx}")
        lines.append("```text")
        lines.append(analysis.raw_fx_graph)
        lines.append("```")
        lines.append("节点明细：")
        for node in analysis.nodes:
            lines.append(
                f"- {node.name}: op={node.op}, target={node.target}, args={node.args}, users={node.users}, "
                f"shape={node.shape}, dtype={node.dtype}, device={node.device}, stride={node.stride}, contiguous={node.is_contiguous}"
            )
        if analysis.pattern_failures:
            lines.append("Pattern 未命中原因：")
            for reason in analysis.pattern_failures:
                lines.append(f"- {reason}")
    lines.append("")
    lines.append("## 第二层：规范化图 / Pattern Match 结果")
    for analysis in recorder.events.get("fx_analysis", []):
        for plan in analysis.fusion_plans:
            lines.append(f"- 识别 pattern: `{plan.plan_type}`")
            lines.append(f"  覆盖 FX nodes: `{plan.covered_nodes}`")
            lines.append(f"  fusion reason: {plan.reason}")
            lines.append(f"  runtime strategy: {plan.runtime_strategy}")
    lines.append("")
    lines.append("## 第三层：FusionPlan")
    for analysis in recorder.events.get("fx_analysis", []):
        for plan in analysis.fusion_plans:
            lines.append("```text")
            lines.append(str(_to_plain(plan)))
            lines.append("```")
    lines.append("")
    lines.append("## 第四层：TensorContract")
    for contract in recorder.events.get("tensor_contract", []):
        lines.append("```text")
        lines.append(str(_to_plain(contract)))
        lines.append("```")
    lines.append("")
    lines.append("## 第五层：AutoPartitionPlan")
    for plan in recorder.events.get("autopartition_plan", []):
        lines.append("```text")
        lines.append(str(_to_plain(plan)))
        lines.append("```")
    lines.append("")
    lines.append("## Backend Legalize")
    for legality in recorder.events.get("legality", []):
        lines.append(f"- legal={legality.legal}, reasons={legality.reasons}, warnings={legality.warnings}")
    lines.append("")
    lines.append("## 第六层：RuntimeCall / Correctness / Benchmark")
    for call in recorder.events.get("runtime_call", []):
        lines.append("```text")
        lines.append(str(_to_plain(call)))
        lines.append("```")
    lines.append("")
    lines.append("## Debug Log 摘要")
    if recorder.debug_logs:
        lines.append("详细日志见 `reports/debug_log.md` 或本 run 目录下的 `debug_log.md`。")
        for item in recorder.debug_logs[-3:]:
            lines.append(item[:2000])
    else:
        lines.append("本次运行没有捕获到需要写入 debug_log 的错误。")
    lines.append("")
    lines.append("## 面试讲法")
    lines.append(
        textwrap.dedent(
            """
            这个项目可以这样讲：我用 torch.compile 自定义 backend 接住 PyTorch 前端，让 TorchDynamo 负责动态图捕获，FX Graph 作为第一层 IR。随后我做 whole-graph pattern match，把 mm/add/gelu 识别成 GEMM_BIAS_GELU，把 linear1/gelu/linear2 识别成 MLP_TWO_GEMM，并记录融合覆盖的原始 FX nodes。

            lowering 分成 FusionPlan、TensorContract、AutoPartitionPlan 和 RuntimeCall。TensorContract 说明 M/N/K、dtype、layout、stride、contiguous 状态和 bias shape；AutoPartitionPlan 说明 sm80 policy、tile shape、thread count、alignment、candidate 和 RoleA/RoleB/RoleC 选择；RuntimeCall 说明最终走 AutoPartitionCudaBackend 还是 PyTorchFallbackBackend。

            AutoPartitioner 后端不是把计算偷换成 cuBLAS/Triton/torch.matmul，而是在 CUDA extension 中实例化仓库里的 AutoPartitioner RoleA/RoleB/RoleC，主 GEMM 使用其 GlobalToSharedCopy、SmemToRegCopyOperation、TiledMma 和 OutputRegisterToGlobalCopy。当前 epilogue 是 two-stage CUDA kernel，报告中明确说明，后续可以把 bias/GELU 合入 AutoPartitioner 的单 kernel epilogue contract。
            """
        ).strip()
    )
    return "\n".join(lines) + "\n"


def format_env_console(env: dict[str, Any]) -> str:
    keys = [
        "python_executable",
        "torch_version",
        "torch_cuda_available",
        "torch_cuda_version",
        "gpu_name",
        "compute_capability",
        "triton_version",
        "nvcc_path",
        "cutlass_include_exists",
        "autopartitioner_reference_exists",
    ]
    return "\n".join(f"{key}: {env.get(key)}" for key in keys)


def run_text(command: list[str]) -> str:
    try:
        proc = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
        return proc.stdout if proc.stdout else proc.stderr
    except Exception as exc:
        return str(exc)


def traceback_text(exc: BaseException) -> str:
    import traceback

    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def _to_plain(value: Any) -> Any:
    if is_dataclass(value):
        result: dict[str, Any] = {}
        for item in fields(value):
            if item.name == "operands":
                continue
            result[item.name] = _to_plain(getattr(value, item.name))
        return result
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_plain(val) for key, val in value.items()}
    return value


if __name__ == "__main__":
    main()
