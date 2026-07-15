"""Environment discovery and dependency bootstrap for the demo."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
SYSTEM_CACHE = Path("/media/zfm/System/AutoPartition-MLIR-Backend-cache")
SOFTWARE_CACHE = Path("/media/zfm/Software/AutoPartition-MLIR-Backend-cache")
CACHE_ROOT = SYSTEM_CACHE if SYSTEM_CACHE.parent.exists() else SOFTWARE_CACHE
TORCH_MLIR_WHEEL_INDEX = (
    "https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels"
)


def choose_cache() -> Path:
    for candidate in (SYSTEM_CACHE, SOFTWARE_CACHE, Path.home() / ".cache" / "autopartition_mlir_backend"):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return candidate
        except OSError:
            continue
    return Path("/tmp/autopartition_mlir_backend")


def python_candidates() -> list[Path]:
    candidates = [Path(sys.executable)]
    for root in (Path("/media/zfm/System"), Path("/media/zfm/Software"), Path.home()):
        if not root.exists():
            continue
        try:
            candidates.extend(root.glob("**/.venv/bin/python"))
            candidates.extend(root.glob("**/venvs/*/bin/python"))
        except OSError:
            pass
    unique = []
    for candidate in candidates:
        if candidate.exists() and candidate not in unique:
            unique.append(candidate)
    return unique


def find_torch_python() -> Path:
    for candidate in python_candidates():
        probe = subprocess.run(
            [str(candidate), "-c", "import torch; print(torch.__version__)"],
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            return candidate
    return Path(sys.executable)


def command_path(name: str) -> str:
    return shutil.which(name) or ""


def collect_environment() -> dict[str, object]:
    local_mlir_opt = (
        choose_cache() / "mlir-prefix/usr/lib/llvm-20/bin/mlir-opt"
    )
    torch_mlir_spec = importlib.util.find_spec("torch_mlir")
    torch_mlir_opt = ""
    stablehlo_tool = ""
    if torch_mlir_spec and torch_mlir_spec.submodule_search_locations:
        package_root = Path(next(iter(torch_mlir_spec.submodule_search_locations)))
        wheel_opt = package_root / "_mlir_libs/torch-mlir-opt"
        if wheel_opt.exists():
            torch_mlir_opt = str(wheel_opt)
        stablehlo_spec = importlib.util.find_spec("torch_mlir._mlir_libs._stablehlo")
        if stablehlo_spec and stablehlo_spec.origin:
            stablehlo_tool = stablehlo_spec.origin
    result: dict[str, object] = {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "torch_mlir_importable": importlib.util.find_spec("torch_mlir") is not None,
        "torch_importable": importlib.util.find_spec("torch") is not None,
        "triton_importable": importlib.util.find_spec("triton") is not None,
        "nvcc": command_path("nvcc"),
        "cmake": command_path("cmake"),
        "ninja": command_path("ninja"),
        "mlir_opt": command_path("mlir-opt") or (str(local_mlir_opt) if local_mlir_opt.exists() else ""),
        "stablehlo_opt": command_path("stablehlo-opt") or stablehlo_tool,
        "torch_mlir_opt": command_path("torch-mlir-opt") or torch_mlir_opt,
        "cutlass_include": str(REPO_ROOT / "include"),
        "autopartitioner_header": str(
            REPO_ROOT / "include/cutlass/transform/collective/auto_partitioner/auto_partitioner_builder.hpp"
        ),
        "cache_root": str(choose_cache()),
    }
    try:
        import torch

        result.update(
            {
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
                "compute_capability": torch.cuda.get_device_capability(0)
                if torch.cuda.is_available()
                else (),
            }
        )
    except Exception as exc:
        result["torch_error"] = str(exc)
    return result


def ensure_torch_mlir() -> dict[str, object]:
    info = collect_environment()
    if info.get("torch_mlir_importable"):
        return info
    python = Path(sys.executable)
    command = [
        str(python),
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--pre",
        "torch-mlir",
        "-f",
        TORCH_MLIR_WHEEL_INDEX,
    ]
    completed = subprocess.run(command, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "torch-mlir wheel installation failed; source build is required. "
            f"Command: {' '.join(command)}"
        )
    importlib.invalidate_caches()
    return collect_environment()


def print_environment(info: dict[str, object] | None = None) -> dict[str, object]:
    info = info or collect_environment()
    print("环境检查：")
    for key in (
        "python",
        "python_version",
        "torch_version",
        "cuda_available",
        "cuda_version",
        "gpu_name",
        "compute_capability",
        "nvcc",
        "cmake",
        "ninja",
        "mlir_opt",
        "stablehlo_opt",
        "torch_mlir_opt",
        "torch_mlir_importable",
        "cache_root",
    ):
        print(f"  {key}: {info.get(key, '')}")
    return info


if __name__ == "__main__":
    print_environment()
