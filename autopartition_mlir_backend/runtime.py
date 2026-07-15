"""MLIR-driven runtime dispatch and CUDA correctness/benchmark helpers."""

from __future__ import annotations

import re
import os
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
_EXTENSION = None


def load_autopartition_extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    from torch.utils.cpp_extension import load

    source = PROJECT_ROOT / "csrc/autopartition_runtime.cu"
    build_dir = Path(
        os.environ.get(
            "AUTOPARTITION_MLIR_EXT_DIR",
            "/media/zfm/System/AutoPartition-MLIR-Backend-cache/torch_extension",
        )
    )
    build_dir.mkdir(parents=True, exist_ok=True)
    old_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    old_jobs = os.environ.get("MAX_JOBS")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0+PTX")
    os.environ.setdefault("MAX_JOBS", "1")
    try:
        _EXTENSION = load(
            name="autopartition_mlir_runtime",
            sources=[str(source)],
            extra_include_paths=[str(REPO_ROOT / "include")],
            extra_cflags=["-std=c++17"],
            extra_cuda_cflags=["-std=c++17", "--expt-relaxed-constexpr"],
            build_directory=str(build_dir),
            verbose=False,
        )
    finally:
        if old_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = old_arch
        if old_jobs is None:
            os.environ.pop("MAX_JOBS", None)
        else:
            os.environ["MAX_JOBS"] = old_jobs
    return _EXTENSION


def run_autopartition_fused(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor):
    if not torch.cuda.is_available():
        raise RuntimeError("AutoPartitionBackend requires CUDA")
    return load_autopartition_extension().fused_gemm_bias_gelu(A, B, bias)


def run_torch_reference(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor):
    return F.gelu(A @ B + bias)


def _lowered_metadata(path: str | Path) -> tuple[str, str, str]:
    text = Path(path).read_text(encoding="utf-8")
    call = re.search(r"call\s+@([A-Za-z0-9_]+)", text)
    if not call:
        raise RuntimeError(f"lowered MLIR contains no runtime call: {path}")
    callee = call.group(1)
    backend_match = re.search(r'autopartition\.backend\s*=\s*"([^"]*)"', text)
    reason_match = re.search(r'autopartition\.fallback_reason\s*=\s*"([^"]*)"', text)
    backend = backend_match.group(1) if backend_match else (
        "AutoPartitionBackend" if callee.startswith("autopartition_") else "TorchFallbackBackend"
    )
    reason = reason_match.group(1) if reason_match else ""
    return callee, backend, reason


def dispatch_from_lowered_mlir(lowered_mlir_path, A, B, bias, announce: bool = True):
    callee, backend, reason = _lowered_metadata(lowered_mlir_path)
    if callee == "autopartition_fused_gemm_bias_gelu" and backend == "AutoPartitionBackend":
        actual_backend = "AutoPartitionBackend"
        output = run_autopartition_fused(A, B, bias)
    elif callee == "torch_fallback_gemm_bias_gelu":
        actual_backend = "TorchFallbackBackend"
        if not reason:
            raise RuntimeError("Torch fallback call has no fallback_reason")
        output = run_torch_reference(A, B, bias)
    else:
        raise RuntimeError(f"unsupported lowered call/backend: {callee}/{backend}")
    if announce:
        print(f"actual_backend = {actual_backend}")
        print(f"fallback_reason = {reason}")
    return output, actual_backend, reason


def benchmark(fn: Callable[[], torch.Tensor], warmup: int = 10, repeat: int = 50) -> float:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / repeat
    import time

    begin = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - begin) * 1000.0 / repeat


def compare_outputs(out: torch.Tensor, ref: torch.Tensor) -> dict[str, object]:
    diff = (out.float() - ref.float()).abs()
    max_abs = float(diff.max().item())
    max_rel = float((diff / ref.float().abs().clamp_min(1e-6)).max().item())
    return {"max_abs_error": max_abs, "max_rel_error": max_rel, "passed": max_abs < 1e-1}
