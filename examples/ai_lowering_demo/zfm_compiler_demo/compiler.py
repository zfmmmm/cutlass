"""Straight-line compiler driver for the tiny GEMM lowering demo."""

from __future__ import annotations

from typing import Any

from .backends import AutoPartitionBackend, AutoPartitionCudaBackend, TritonFusedGemmBackend, TorchReferenceBackend
from .ir import Graph
from .lowering import LoweredGemmEpilogueOp, lower_graph
from .passes import run_passes


def build_demo_graph(M: int, N: int, K: int) -> Graph:
    graph = Graph()
    a = graph.placeholder("A", (M, K), "fp16")
    b = graph.placeholder("B", (K, N), "fp16")
    bias = graph.placeholder("bias", (N,), "fp16")
    scale = graph.scalar("scale", 0.5, "fp32")
    y = graph.gelu(graph.mul(graph.add(graph.matmul(a, b), bias), scale), name="Y")
    graph.set_output(y)
    return graph


def compile(graph: Graph, inputs: dict[str, Any] | None = None, warmup: int = 5, repeats: int = 20) -> dict[str, Any]:
    _section("Original GraphIR")
    print(graph.pretty())

    optimized = run_passes(graph)
    _section("Optimized GraphIR")
    print(optimized.pretty())

    lowered = lower_graph(optimized)
    op = lowered[0]
    _section("LoweredIR")
    print(op.pretty())

    torch_backend = TorchReferenceBackend()
    triton_backend = TritonFusedGemmBackend()
    autopart_cuda_backend = AutoPartitionCudaBackend()
    autopart_backend = AutoPartitionBackend()
    statuses = [
        torch_backend.legality(op),
        triton_backend.legality(op),
        autopart_cuda_backend.legality(op),
        autopart_backend.legality(op),
    ]

    _section("Backend Legality")
    for status in statuses:
        print(status.pretty())

    plan = autopart_backend.plan(op)
    _section("AutoPartition Plan")
    print(plan.pretty())

    results: dict[str, Any] = {"lowered": lowered, "autopartition_plan": plan}
    if inputs is None:
        _section("Execution")
        print("runtime inputs unavailable; skipped Torch reference, Triton fused run, correctness, and benchmark")
        return results

    torch_status = statuses[0].status
    triton_status = statuses[1].status
    autopart_cuda_status = statuses[2].status
    y_ref = None
    y_triton = None
    y_autopart = None

    _section("Torch Reference Result")
    if torch_status == "legal":
        y_ref = torch_backend.run(op, inputs)
        results["torch_reference"] = y_ref
        print(_tensor_summary(y_ref))
    else:
        print(f"skipped: {statuses[0].reason}")

    _section("Triton Fused Result")
    if triton_status == "legal":
        y_triton = triton_backend.run(op, inputs)
        results["triton_fused"] = y_triton
        print(_tensor_summary(y_triton))
    else:
        print(f"skipped: {statuses[1].reason}")

    _section("AutoPartition CUDA Result")
    if autopart_cuda_status == "legal":
        try:
            y_autopart = autopart_cuda_backend.run(op, inputs)
            results["autopartition_cuda"] = y_autopart
            print(_tensor_summary(y_autopart))
        except Exception as exc:
            print(f"failed: {exc}")
            results["autopartition_cuda_error"] = str(exc)
    else:
        print(f"skipped: {statuses[2].reason}")

    _section("Correctness")
    if y_ref is not None:
        correctness: dict[str, dict[str, Any]] = {}
        if y_triton is not None:
            max_abs, max_rel, passed = _compare_outputs(y_triton, y_ref)
            correctness["triton_fused"] = {
                "max_abs_error": max_abs,
                "max_rel_error": max_rel,
                "passed": passed,
            }
            print(f"triton_fused.max_abs_error = {max_abs:.6g}")
            print(f"triton_fused.max_rel_error = {max_rel:.6g}")
            print(f"triton_fused.passed = {str(passed).lower()}")
        if y_autopart is not None:
            max_abs, max_rel, passed = _compare_outputs(y_autopart, y_ref)
            correctness["autopartition_cuda"] = {
                "max_abs_error": max_abs,
                "max_rel_error": max_rel,
                "passed": passed,
            }
            print(f"autopartition_cuda.max_abs_error = {max_abs:.6g}")
            print(f"autopartition_cuda.max_rel_error = {max_rel:.6g}")
            print(f"autopartition_cuda.passed = {str(passed).lower()}")
        if correctness:
            results["correctness"] = correctness
        else:
            print("skipped: no executable backend output is available")
    else:
        print("skipped: Torch reference output is required")

    _section("Benchmark")
    if torch_status == "legal":
        torch_ms = torch_backend.benchmark(op, inputs, warmup, repeats)
        results["torch_reference_ms"] = torch_ms
        print(f"torch_reference: {torch_ms:.4f} ms")
    else:
        print(f"torch_reference: skipped ({statuses[0].reason})")
    if triton_status == "legal":
        triton_ms = triton_backend.benchmark(op, inputs, warmup, repeats)
        results["triton_fused_ms"] = triton_ms
        print(f"triton_fused: {triton_ms:.4f} ms")
    else:
        print(f"triton_fused: skipped ({statuses[1].reason})")
    if autopart_cuda_status == "legal" and y_autopart is not None:
        autopart_ms = autopart_cuda_backend.benchmark(op, inputs, warmup, repeats)
        results["autopartition_cuda_ms"] = autopart_ms
        print(f"autopartition_cuda: {autopart_ms:.4f} ms")
    elif autopart_cuda_status == "legal":
        print("autopartition_cuda: skipped (runtime failed before benchmark)")
    else:
        print(f"autopartition_cuda: skipped ({statuses[2].reason})")

    return results


def _section(title: str) -> None:
    print(f"\n========== {title} ==========")


def _tensor_summary(tensor: Any) -> str:
    shape = tuple(tensor.shape)
    return f"shape = {shape}, dtype = {tensor.dtype}, device = {tensor.device}"


def _compare_outputs(actual: Any, expected: Any) -> tuple[float, float, bool]:
    diff = (actual.to("cpu").float() - expected.to("cpu").float()).abs()
    expected_abs = expected.to("cpu").float().abs()
    max_abs = float(diff.max().item())
    max_rel = float((diff / (expected_abs + 1.0e-6)).max().item())
    return max_abs, max_rel, max_abs < 1.0e-1
