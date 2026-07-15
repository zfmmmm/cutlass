"""Build and invoke the real C++ AutoPartition MLIR passes."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
TOOLS_ROOT = PROJECT_ROOT / "tools"
DEFAULT_BUILD = TOOLS_ROOT / "build"


def _mlir_dir() -> Path | None:
    candidates = []
    if os.environ.get("MLIR_DIR"):
        candidates.append(Path(os.environ["MLIR_DIR"]))
    for version in ("20", "19", "18", "17"):
        candidates.extend(
            [
                Path(f"/usr/lib/llvm-{version}/lib/cmake/mlir"),
                Path(f"/usr/lib/llvm-{version}/lib/cmake/mlir"),
            ]
        )
    candidates.append(
        Path("/media/zfm/System/AutoPartition-MLIR-Backend-cache/mlir-prefix/usr/lib/llvm-20/lib/cmake/mlir")
    )
    candidates.extend(Path(p) for p in os.environ.get("CMAKE_PREFIX_PATH", "").split(":") if p)
    for candidate in candidates:
        if (candidate / "MLIRConfig.cmake").exists():
            return candidate
    return None


def build_autopartition_opt() -> Path:
    existing = DEFAULT_BUILD / "autopartition-opt"
    if existing.exists() and os.access(existing, os.X_OK):
        return existing
    mlir_dir = _mlir_dir()
    if mlir_dir is None:
        raise RuntimeError(
            "找不到 MLIRConfig.cmake。请先安装 MLIR 开发包，或设置 MLIR_DIR；"
            "当前 env_check 会优先尝试本机 wheel/工具链并保留准确路径。"
        )
    DEFAULT_BUILD.mkdir(parents=True, exist_ok=True)
    command = [
        "cmake",
        "-S",
        str(TOOLS_ROOT),
        "-B",
        str(DEFAULT_BUILD),
        "-G",
        "Ninja",
        f"-DMLIR_DIR={mlir_dir}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    subprocess.run(command, check=True)
    subprocess.run(["cmake", "--build", str(DEFAULT_BUILD), "--target", "autopartition-opt"], check=True)
    if not existing.exists():
        raise RuntimeError(f"autopartition-opt 构建后不存在: {existing}")
    return existing


def _run(opt: Path, input_path: Path, output_path: Path, flag: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(opt), str(input_path), flag, "-o", str(output_path)]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"MLIR pass failed: {' '.join(command)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def run_fusion_and_legalize(input_linalg: str | Path, fused_output: str | Path) -> Path:
    opt = build_autopartition_opt()
    input_path = Path(input_linalg)
    fused_path = Path(fused_output)
    intermediate = fused_path.with_name(f"{fused_path.stem}.prelegalize.mlir")
    _run(opt, input_path, intermediate, "--autopartition-fuse-gemm-epilogue")
    _run(opt, intermediate, fused_path, "--autopartition-legalize-backend")
    intermediate.unlink(missing_ok=True)
    return fused_path


def run_lower_to_runtime(fused_input: str | Path, lowered_output: str | Path) -> Path:
    opt = build_autopartition_opt()
    _run(opt, Path(fused_input), Path(lowered_output), "--autopartition-lower-to-runtime")
    return Path(lowered_output)


def check_mlir_outputs(root: str | Path = PROJECT_ROOT) -> dict[str, bool]:
    root = Path(root)
    outputs = {
        "stablehlo": root / "mlir/exported_stablehlo.mlir",
        "linalg": root / "mlir/exported_linalg.mlir",
        "fused": root / "mlir/fused.mlir",
        "lowered": root / "mlir/lowered.mlir",
    }
    texts = {key: path.read_text(encoding="utf-8") for key, path in outputs.items() if path.exists()}
    checks = {
        "stablehlo": "stablehlo" in texts.get("stablehlo", "").lower() or "mhlo" in texts.get("stablehlo", "").lower(),
        "linalg": "linalg.matmul" in texts.get("linalg", ""),
        "fused": "autopartition.gemm_bias_gelu" in texts.get("fused", ""),
        "lowered": "call @autopartition_fused_gemm_bias_gelu" in texts.get("lowered", "")
        or "call @torch_fallback_gemm_bias_gelu" in texts.get("lowered", ""),
    }
    missing = [key for key, ok in checks.items() if not ok]
    if missing:
        raise RuntimeError(f"MLIR artifact checks failed: {missing}")
    return checks
