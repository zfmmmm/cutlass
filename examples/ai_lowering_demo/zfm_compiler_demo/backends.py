"""Reference, Triton, and AutoPartition plan backends for the demo."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .lowering import LoweredGemmEpilogueOp


@dataclass
class BackendStatus:
    name: str
    status: str
    reason: str = ""

    def pretty(self) -> str:
        return f"{self.name}: {self.status}" + (f" ({self.reason})" if self.reason else "")


class TorchReferenceBackend:
    name = "TorchReferenceBackend"

    def legality(self, op: LoweredGemmEpilogueOp) -> BackendStatus:
        if op.op != "fused_gemm_epilogue":
            return BackendStatus(self.name, "illegal", "unsupported op")
        try:
            import torch  # noqa: F401
        except Exception as exc:
            return BackendStatus(self.name, "unavailable", f"torch import failed: {exc}")
        return BackendStatus(self.name, "legal")

    def run(self, op: LoweredGemmEpilogueOp, inputs: dict[str, Any]):
        import torch
        import torch.nn.functional as F

        a = inputs["A"].to(torch.float32)
        b = inputs["B"].to(torch.float32)
        bias = inputs["bias"].to(torch.float32)
        scale = float(inputs["scale"])
        out = F.gelu((a @ b + bias) * scale)
        return out.to(_torch_dtype(torch, op.dtype_out))

    def benchmark(self, op: LoweredGemmEpilogueOp, inputs: dict[str, Any], warmup: int, repeats: int) -> float:
        import torch

        return _time_cuda_or_cpu(lambda: self.run(op, inputs), torch, warmup, repeats)


class TritonFusedGemmBackend:
    name = "TritonFusedGemmBackend"

    def legality(self, op: LoweredGemmEpilogueOp) -> BackendStatus:
        if op.op != "fused_gemm_epilogue":
            return BackendStatus(self.name, "illegal", "unsupported op")
        if (op.dtype_a, op.dtype_b, op.dtype_out, op.acc_dtype) != ("fp16", "fp16", "fp16", "fp32"):
            return BackendStatus(self.name, "illegal", "demo kernel only supports fp16/fp32acc/fp16")
        if (op.layout_a, op.layout_b, op.layout_out) != ("row_major", "row_major", "row_major"):
            return BackendStatus(self.name, "illegal", "demo kernel only supports row_major")
        try:
            import triton  # noqa: F401
            import triton.language as tl
        except Exception as exc:
            return BackendStatus(self.name, "unavailable", f"triton import failed: {exc}")
        if not hasattr(tl, "erf"):
            return BackendStatus(self.name, "unavailable", "triton.language.erf is required for exact GELU")
        return BackendStatus(self.name, "legal")

    def run(self, op: LoweredGemmEpilogueOp, inputs: dict[str, Any]):
        import torch
        import triton

        kernel = _triton_kernel()
        a = inputs["A"].contiguous()
        b = inputs["B"].contiguous()
        bias = inputs["bias"].contiguous()
        out = torch.empty((op.M, op.N), device=a.device, dtype=_torch_dtype(torch, op.dtype_out))
        block_m, block_n, block_k = 16, 32, 32
        grid = (triton.cdiv(op.M, block_m), triton.cdiv(op.N, block_n))
        kernel[grid](
            a,
            b,
            bias,
            float(inputs["scale"]),
            out,
            op.M,
            op.N,
            op.K,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=4,
            num_stages=3,
        )
        return out

    def benchmark(self, op: LoweredGemmEpilogueOp, inputs: dict[str, Any], warmup: int, repeats: int) -> float:
        import torch

        return _time_cuda_or_cpu(lambda: self.run(op, inputs), torch, warmup, repeats)


class AutoPartitionCudaBackend:
    name = "AutoPartitionCudaBackend"

    def legality(self, op: LoweredGemmEpilogueOp) -> BackendStatus:
        if op.op != "fused_gemm_epilogue":
            return BackendStatus(self.name, "illegal", "unsupported op")
        if (op.dtype_a, op.dtype_b, op.dtype_bias, op.dtype_out, op.acc_dtype) != (
            "fp16",
            "fp16",
            "fp16",
            "fp16",
            "fp32",
        ):
            return BackendStatus(self.name, "illegal", "demo extension only supports fp16/fp32acc/fp16")
        if (op.layout_a, op.layout_b, op.layout_out) != ("row_major", "row_major", "row_major"):
            return BackendStatus(self.name, "illegal", "demo extension only supports row_major")
        if op.target_sm != "sm80" or op.tile_shape != (64, 64, 64) or op.thread_count != 128:
            return BackendStatus(self.name, "illegal", "demo extension uses the SM80 64x64x64/128-thread AutoPartition path")
        if op.M % 64 != 0 or op.N % 64 != 0 or op.K % 64 != 0:
            return BackendStatus(self.name, "illegal", "demo extension requires M, N, K multiples of 64")
        try:
            import torch
            import torch.utils.cpp_extension  # noqa: F401
        except Exception as exc:
            return BackendStatus(self.name, "unavailable", f"torch extension import failed: {exc}")
        if not torch.cuda.is_available():
            return BackendStatus(self.name, "unavailable", "CUDA is not available")
        if not (_repo_root() / "examples/ai_lowering_demo/zfm_compiler_demo/autopartition_sm80_runtime.cu").exists():
            return BackendStatus(self.name, "unavailable", "AutoPartition runtime extension source not found")
        return BackendStatus(self.name, "legal")

    def run(self, _op: LoweredGemmEpilogueOp, inputs: dict[str, Any]):
        module = _load_autopartition_runtime_extension()
        return module.fused_gemm_bias_scale_gelu(
            inputs["A"].contiguous(),
            inputs["B"].contiguous(),
            inputs["bias"].contiguous(),
            float(inputs["scale"]),
        )

    def benchmark(self, op: LoweredGemmEpilogueOp, inputs: dict[str, Any], warmup: int, repeats: int) -> float:
        import torch

        return _time_cuda_or_cpu(lambda: self.run(op, inputs), torch, warmup, repeats)


@dataclass
class AutoPartitionConfig:
    M: int
    N: int
    K: int
    dtype: str
    acc_dtype: str
    layout_a: str
    layout_b: str
    layout_out: str
    epilogue: str
    tile_shape: tuple[int, int, int]
    thread_count: int
    gmem_alignment_a: int
    gmem_alignment_b: int
    gmem_alignment_out: int
    candidate_tile_shapes: list[tuple[int, int, int]]
    candidate_warp_shapes: list[tuple[int, int]]
    candidate_stage_counts: list[int]
    target_sm: str


@dataclass
class AutoPartitionPlan:
    config: AutoPartitionConfig
    runnable: bool
    reason: str
    detected_interfaces: list[str]
    source: str
    selected: dict[str, str]
    probe_command: list[str] | None = None
    probe_error: str = ""

    def pretty(self) -> str:
        detected = ", ".join(self.detected_interfaces) if self.detected_interfaces else "none"
        lines = [
            "AutoPartitionPlan:",
            "    op = fused_gemm_epilogue",
            f"    M/N/K = {self.config.M}/{self.config.N}/{self.config.K}",
            f"    dtype = {self.config.dtype}",
            f"    A/B/out layout = {self.config.layout_a}/{self.config.layout_b}/{self.config.layout_out}",
            f"    acc_dtype = {self.config.acc_dtype}",
            f"    epilogue = {self.config.epilogue}",
            f"    selected_tile_shape = {self.config.tile_shape}",
            f"    thread_count = {self.config.thread_count}",
            f"    gmem_alignment_bytes = A{self.config.gmem_alignment_a} / B{self.config.gmem_alignment_b} / OUT{self.config.gmem_alignment_out}",
            f"    candidate_tile_shapes = {self.config.candidate_tile_shapes}",
            f"    candidate_warp_shapes = {self.config.candidate_warp_shapes}",
            f"    candidate_stage_counts = {self.config.candidate_stage_counts}",
            f"    target_sm = {self.config.target_sm}",
            f"    source = {self.source}",
        ]
        if self.selected:
            lines.extend(
                [
                    "    gmem_to_smem = RoleA/RoleB::GlobalToSharedCopy selected by AutoPartitioner",
                    "    smem_to_reg = RoleA/RoleB::SmemToRegCopyOperation selected by AutoPartitioner",
                    "    mma_atom = RoleC::TiledMma selected by AutoPartitioner",
                    "    epilogue_layout = RoleC::FusionSmemLayout / OutputRegisterToGlobalCopy selected by AutoPartitioner",
                    f"    RoleA::UseLdMatrix = {self.selected.get('role_a_use_ldmatrix', 'unknown')}",
                    f"    RoleB::UseLdMatrix = {self.selected.get('role_b_use_ldmatrix', 'unknown')}",
                    f"    RoleA::SwizzleBase = {self.selected.get('role_a_swizzle_base', 'unknown')}",
                    f"    RoleB::SwizzleBase = {self.selected.get('role_b_swizzle_base', 'unknown')}",
                    f"    RoleC::EpilogueLayoutCandidateCount = {self.selected.get('role_c_epilogue_layout_candidate_count', 'unknown')}",
                    f"    RoleC::HasZeroGlueEpilogueMapping = {self.selected.get('role_c_has_zero_glue_epilogue_mapping', 'unknown')}",
                    f"    RoleC::HasFusionSharedMapping = {self.selected.get('role_c_has_fusion_shared_mapping', 'unknown')}",
                    f"    RoleC::EpilogueSwizzleBase = {self.selected.get('role_c_epilogue_swizzle_base', 'unknown')}",
                    f"    RoleC::EpilogueBankConflictScore = {self.selected.get('role_c_epilogue_bank_conflict_score', 'unknown')}",
                    f"    RoleC::OutputAlignmentBytes = {self.selected.get('role_c_output_alignment_bytes', 'unknown')}",
                ]
            )
        else:
            lines.extend(
                [
                    "    gmem_to_smem = to be selected by user autopartition tool",
                    "    smem_to_reg = to be selected by user autopartition tool",
                    "    mma_atom = to be selected by user autopartition tool",
                    "    epilogue_layout = to be selected by user autopartition tool",
                ]
            )
        lines.append(f"    detected_interfaces = {detected}")
        if self.probe_command:
            lines.append(f"    probe_command = {' '.join(self.probe_command)}")
        if self.probe_error:
            lines.append(f"    probe_error = {self.probe_error}")
        lines.append(f"    runnable = {str(self.runnable).lower()}")
        lines.append(f"    reason = {self.reason}")
        return "\n".join(lines)


class AutoPartitionBackend:
    name = "AutoPartitionBackend"

    def __init__(self, probe_runner: Callable[[AutoPartitionConfig], dict[str, str]] | None = None) -> None:
        self._probe_runner = probe_runner

    def legality(self, op: LoweredGemmEpilogueOp) -> BackendStatus:
        if op.op != "fused_gemm_epilogue":
            return BackendStatus(self.name, "illegal", "unsupported op")
        return BackendStatus(self.name, "plan_only")

    def plan(self, op: LoweredGemmEpilogueOp) -> AutoPartitionPlan:
        config = AutoPartitionConfig(
            M=op.M,
            N=op.N,
            K=op.K,
            dtype=op.dtype_a,
            acc_dtype=op.acc_dtype,
            layout_a=op.layout_a,
            layout_b=op.layout_b,
            layout_out=op.layout_out,
            epilogue=op.epilogue,
            tile_shape=op.tile_shape,
            thread_count=op.thread_count,
            gmem_alignment_a=op.gmem_alignment_a,
            gmem_alignment_b=op.gmem_alignment_b,
            gmem_alignment_out=op.gmem_alignment_out,
            candidate_tile_shapes=[(64, 64, 64), (128, 64, 64), (64, 128, 64)],
            candidate_warp_shapes=[(2, 2), (1, 4), (4, 1)],
            candidate_stage_counts=[1, 2, 3],
            target_sm=op.target_sm,
        )
        probe_error = ""
        probe_command = None
        try:
            selected = self._call_probe(config)
            source = selected.get("source", "autopartitioner_probe")
            probe_command_value = selected.pop("probe_command", "")
            probe_command = probe_command_value.split("\t") if probe_command_value else None
        except Exception as exc:
            selected = {}
            source = "python_fallback"
            probe_error = str(exc)
        return AutoPartitionPlan(
            config=config,
            runnable=False,
            reason="AutoPartition tool is a layout/template planner, not a direct launch backend yet.",
            detected_interfaces=_detect_autopartition_interfaces(),
            source=source,
            selected=selected,
            probe_command=probe_command,
            probe_error=probe_error,
        )

    def _call_probe(self, config: AutoPartitionConfig) -> dict[str, str]:
        if self._probe_runner is not None:
            return self._probe_runner(config)
        return _run_autopartition_probe(config)


_TRITON_KERNEL = None


def _triton_kernel():
    global _TRITON_KERNEL
    if _TRITON_KERNEL is not None:
        return _TRITON_KERNEL

    import triton
    import triton.language as tl

    @triton.jit
    def _kernel(A, B, BIAS, SCALE, OUT, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_idxs = k0 + offs_k
            a = tl.load(A + offs_m[:, None] * K + k_idxs[None, :],
                        mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K), other=0.0)
            b = tl.load(B + k_idxs[:, None] * N + offs_n[None, :],
                        mask=(k_idxs[:, None] < K) & (offs_n[None, :] < N), other=0.0)
            acc += tl.dot(a, b, out_dtype=tl.float32)

        bias = tl.load(BIAS + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        x = (acc + bias[None, :]) * SCALE
        y = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))
        tl.store(OUT + offs_m[:, None] * N + offs_n[None, :],
                 y, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    _TRITON_KERNEL = _kernel
    return _TRITON_KERNEL


def _time_cuda_or_cpu(fn, torch_module, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()
        start = torch_module.cuda.Event(enable_timing=True)
        end = torch_module.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        torch_module.cuda.synchronize()
        return float(start.elapsed_time(end) / repeats)

    start_s = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start_s) * 1000.0 / repeats


def _torch_dtype(torch_module, dtype: str):
    mapping = {
        "fp16": torch_module.float16,
        "fp32": torch_module.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"unsupported torch dtype: {dtype}")
    return mapping[dtype]


def _detect_autopartition_interfaces() -> list[str]:
    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        "include/cutlass/transform/collective/auto_partitioner/auto_partitioner.hpp",
        "include/cutlass/transform/collective/auto_partitioner/auto_partitioner_builder.hpp",
        "include/cutlass/transform/collective/auto_partitioner/examples/production_gemm/sm80_autopartition_gemm.cu",
        "include/cutlass/transform/collective/auto_partitioner/examples/production_gemm/sm100_autopartition_tma_umma_gemm.cu",
    ]
    return [path for path in candidates if (repo_root / path).exists()]


def _run_autopartition_probe(config: AutoPartitionConfig) -> dict[str, str]:
    if config.target_sm != "sm80":
        raise RuntimeError(f"autopartition probe only supports sm80, got {config.target_sm}")
    if config.dtype != "fp16" or config.acc_dtype != "fp32":
        raise RuntimeError(f"autopartition probe only supports fp16 input with fp32 accumulator, got {config.dtype}/{config.acc_dtype}")
    if config.tile_shape != (64, 64, 64) or config.thread_count != 128:
        raise RuntimeError(f"autopartition probe only supports tile 64x64x64 and 128 threads, got {config.tile_shape}/{config.thread_count}")

    exe = _ensure_autopartition_probe_built()
    command = [
        str(exe),
        f"--m={config.M}",
        f"--n={config.N}",
        f"--k={config.K}",
        f"--epilogue={config.epilogue}",
        f"--target-sm={config.target_sm}",
        f"--tile={config.tile_shape[0]}x{config.tile_shape[1]}x{config.tile_shape[2]}",
        f"--threads={config.thread_count}",
        f"--align-a={config.gmem_alignment_a}",
        f"--align-b={config.gmem_alignment_b}",
        f"--align-out={config.gmem_alignment_out}",
    ]
    proc = subprocess.run(command, cwd=_repo_root(), text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        stdout = proc.stdout.strip()
        raise RuntimeError(f"autopartition probe failed with code {proc.returncode}: {stderr or stdout}")

    result = _parse_key_value_output(proc.stdout)
    result["probe_command"] = "\t".join(command)
    return result


_AUTOPARTITION_RUNTIME_MODULE = None


def _load_autopartition_runtime_extension():
    global _AUTOPARTITION_RUNTIME_MODULE
    if _AUTOPARTITION_RUNTIME_MODULE is not None:
        return _AUTOPARTITION_RUNTIME_MODULE

    import torch
    from torch.utils.cpp_extension import load

    repo_root = _repo_root()
    source = repo_root / "examples/ai_lowering_demo/zfm_compiler_demo/autopartition_sm80_runtime.cu"
    build_dir = Path(os.environ.get("ZFM_COMPILER_DEMO_TORCH_EXT_DIR", Path.home() / ".cache/zfm_compiler_demo/torch_extensions"))
    build_dir.mkdir(parents=True, exist_ok=True)

    old_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    old_max_jobs = os.environ.get("MAX_JOBS")
    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("TORCH_CUDA_ARCH_LIST", "8.0+PTX")
    os.environ["MAX_JOBS"] = os.environ.get("MAX_JOBS", "1")
    try:
        _AUTOPARTITION_RUNTIME_MODULE = load(
            name="zfm_autopartition_sm80_runtime",
            sources=[str(source)],
            extra_include_paths=[str(repo_root / "include")],
            extra_cflags=["-std=c++17"],
            extra_cuda_cflags=["-std=c++17", "--expt-relaxed-constexpr"],
            build_directory=str(build_dir),
            verbose=False,
        )
    finally:
        if old_arch_list is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = old_arch_list
        if old_max_jobs is None:
            os.environ.pop("MAX_JOBS", None)
        else:
            os.environ["MAX_JOBS"] = old_max_jobs

    # Touch CUDA once so extension load errors surface close to the backend call.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return _AUTOPARTITION_RUNTIME_MODULE


def _ensure_autopartition_probe_built() -> Path:
    repo_root = _repo_root()
    source = repo_root / "examples/ai_lowering_demo/zfm_compiler_demo/autopartition_sm80_probe.cu"
    if not source.exists():
        raise RuntimeError(f"autopartition probe source not found: {source}")

    build_dir = Path(os.environ.get("ZFM_COMPILER_DEMO_BUILD_DIR", Path.home() / ".cache/zfm_compiler_demo/autopartition"))
    build_dir.mkdir(parents=True, exist_ok=True)
    exe = build_dir / "autopartition_sm80_probe"
    deps = [
        source,
        repo_root / "include/cutlass/transform/collective/auto_partitioner/auto_partitioner.hpp",
        repo_root / "include/cutlass/transform/collective/auto_partitioner/auto_partitioner_builder.hpp",
        repo_root / "include/cutlass/transform/collective/auto_partitioner/arch/sm80_policy.hpp",
    ]
    newest_dep_mtime = max(path.stat().st_mtime for path in deps if path.exists())
    if exe.exists() and exe.stat().st_mtime >= newest_dep_mtime:
        return exe

    nvcc = os.environ.get("NVCC", "nvcc")
    command = [
        nvcc,
        "-std=c++17",
        "-O2",
        "-arch=sm_80",
        f"-I{repo_root / 'include'}",
        str(source),
        "-o",
        str(exe),
    ]
    proc = subprocess.run(command, cwd=repo_root, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError("failed to build autopartition probe with nvcc: " + (proc.stderr.strip() or proc.stdout.strip()))
    return exe


def _parse_key_value_output(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if sep:
            result[key.strip()] = value.strip()
    return result


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]
