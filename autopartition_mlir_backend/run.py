"""一键运行 AutoPartition-MLIR-Backend 全链路。"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def _reexec_with_torch_python() -> None:
    if importlib.util.find_spec("torch") is not None:
        return
    from env_check import find_torch_python

    candidate = find_torch_python()
    if candidate.resolve() != Path(sys.executable).resolve():
        os.execv(str(candidate), [str(candidate), *sys.argv])


_reexec_with_torch_python()

import torch

from env_check import ensure_torch_mlir, print_environment
from export_model import build_inputs, build_model, export_to_linalg, export_to_stablehlo, print_fx_or_exported_graph
from mlir_pipeline import (
    check_mlir_outputs,
    run_fusion_and_legalize,
    run_lower_to_runtime,
)
from runtime import benchmark, compare_outputs, dispatch_from_lowered_mlir, run_torch_reference


def main() -> int:
    info = print_environment()
    if not info.get("torch_importable"):
        raise RuntimeError("未找到 PyTorch 环境")
    ensure_torch_mlir()

    root = Path(__file__).resolve().parent
    mlir = root / "mlir"
    model = build_model()
    inputs = build_inputs()
    print_fx_or_exported_graph(model, inputs)
    print("StableHLO/MHLO 导出中...")
    export_to_stablehlo(model, inputs, mlir / "exported_stablehlo.mlir")
    print("StableHLO/MHLO 导出成功：mlir/exported_stablehlo.mlir")
    export_to_linalg(model, inputs, mlir / "exported_linalg.mlir", mlir / "exported_stablehlo.mlir")
    print("Linalg-on-Tensors 导出成功：mlir/exported_linalg.mlir")

    run_fusion_and_legalize(mlir / "exported_linalg.mlir", mlir / "fused.mlir")
    fused_text = (mlir / "fused.mlir").read_text(encoding="utf-8")
    print("MLIR fusion pass 成功：mlir/fused.mlir")
    legal_backend = "backend = \"AutoPartitionBackend\"" in fused_text
    print(f"backend legalize 成功：{'AutoPartitionBackend' if legal_backend else 'TorchFallbackBackend'}")

    run_lower_to_runtime(mlir / "fused.mlir", mlir / "lowered.mlir")
    print("runtime lowering 成功：mlir/lowered.mlir")
    check_mlir_outputs(root)

    A, B, bias = inputs
    if legal_backend and torch.cuda.is_available():
        out, actual_backend, reason = dispatch_from_lowered_mlir(mlir / "lowered.mlir", A, B, bias)
        ref = run_torch_reference(A, B, bias)
        metrics = compare_outputs(out, ref)
        print(f"max_abs_error = {metrics['max_abs_error']:.6g}")
        print(f"max_rel_error = {metrics['max_rel_error']:.6g}")
        print(f"correctness = {'pass' if metrics['passed'] else 'fail'}")
        if not metrics["passed"]:
            raise RuntimeError(f"AutoPartition correctness failed: {metrics}")
        avg_backend = benchmark(lambda: dispatch_from_lowered_mlir(mlir / "lowered.mlir", A, B, bias, announce=False)[0])
        avg_reference = benchmark(lambda: run_torch_reference(A, B, bias))
        print(f"benchmark 已生成: AutoPartitionBackend avg_ms={avg_backend:.4f}, PyTorch avg_ms={avg_reference:.4f}")
    else:
        if legal_backend:
            raise RuntimeError("合法 MLIR backend 但 CUDA 不可用，不能静默 fallback")
        reason = ""
        print("actual_backend = TorchFallbackBackend")
        print("correctness = skip (CUDA unavailable or MLIR explicitly legalized to fallback)")
        print("fallback_reason = " + reason)
        print("benchmark 已生成: fallback path")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
