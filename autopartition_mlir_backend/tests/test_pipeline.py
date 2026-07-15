from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def pipeline_run():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, str(ROOT / "run.py")], cwd=ROOT, env=env, text=True, capture_output=True)
    if result.returncode != 0:
        pytest.fail(f"run.py failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return result.stdout


def _text(name: str) -> str:
    return (ROOT / "mlir" / name).read_text(encoding="utf-8")


def test_pytorch_capture_and_stablehlo(pipeline_run):
    assert "PyTorch 前端捕获成功" in pipeline_run
    assert "stablehlo" in _text("exported_stablehlo.mlir").lower() or "mhlo" in _text("exported_stablehlo.mlir").lower()


def test_linalg_contains_matmul_bias_gelu(pipeline_run):
    text = _text("exported_linalg.mlir")
    assert "linalg.matmul" in text
    assert re.search(r"bias|erf|tanh|gelu|rsqrt|arith\.addf", text, re.IGNORECASE)


def test_fusion_and_legalization(pipeline_run):
    text = _text("fused.mlir")
    assert "autopartition.gemm_bias_gelu" in text
    assert 'backend = "AutoPartitionBackend"' in text
    for attr in ("M", "N", "K", "dtype", "target_sm", "tile_m", "tile_n", "tile_k", "thread_count", "alignment_bytes"):
        assert attr in text


def test_lowered_runtime_call(pipeline_run):
    text = _text("lowered.mlir")
    assert "call @autopartition_fused_gemm_bias_gelu" in text or "call @torch_fallback_gemm_bias_gelu" in text


def test_illegal_shape_has_explicit_fallback(pipeline_run, tmp_path):
    tool = ROOT / "tools/build/autopartition-opt"
    source = tmp_path / "illegal.mlir"
    output = tmp_path / "illegal_out.mlir"
    source.write_text(
        """module {
  func.func @main(%a: tensor<65x64xf16>, %b: tensor<64x64xf16>, %bias: tensor<64xf16>) -> tensor<65x64xf16> {
    %0 = \"autopartition.gemm_bias_gelu\"(%a, %b, %bias) {M = 65 : i64, N = 64 : i64, K = 64 : i64, dtype = \"f16\", target_sm = \"sm80\", tile_m = 64 : i64, tile_n = 64 : i64, tile_k = 64 : i64, thread_count = 128 : i64, alignment_bytes = 16 : i64, backend = \"unknown\", fallback_reason = \"\"} : (tensor<65x64xf16>, tensor<64x64xf16>, tensor<64xf16>) -> tensor<65x64xf16>
    return %0 : tensor<65x64xf16>
  }
}
""",
        encoding="utf-8",
    )
    subprocess.run([str(tool), str(source), "--autopartition-legalize-backend", "-o", str(output)], check=True)
    text = output.read_text(encoding="utf-8")
    assert 'backend = "TorchFallbackBackend"' in text
    assert "fallback_reason" in text


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_actual_backend_and_correctness(pipeline_run):
    from runtime import compare_outputs, dispatch_from_lowered_mlir, run_torch_reference
    from export_model import build_inputs

    A, B, bias = build_inputs("cuda", (64, 64, 64))
    out, actual, reason = dispatch_from_lowered_mlir(ROOT / "mlir/lowered.mlir", A, B, bias)
    metrics = compare_outputs(out, run_torch_reference(A, B, bias))
    assert actual == "AutoPartitionBackend", reason
    assert metrics["max_abs_error"] < 1e-1
