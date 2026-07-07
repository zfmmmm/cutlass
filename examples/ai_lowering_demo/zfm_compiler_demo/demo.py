"""Executable entry point for the tiny GEMM lowering demo."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zfm_compiler_demo.compiler import build_demo_graph, compile


def main() -> None:
    M = 1024
    N = 1024
    K = 1024
    graph = build_demo_graph(M, N, K)

    inputs = _make_inputs(M, N, K)
    compile(graph, inputs=inputs)


def _make_inputs(M: int, N: int, K: int):
    try:
        import torch
    except Exception as exc:
        print(f"torch unavailable: {exc}")
        return None

    if not torch.cuda.is_available():
        print("CUDA unavailable: IR dump and AutoPartition plan will still run")
        return None

    torch.manual_seed(0)
    return {
        "A": torch.randn((M, K), device="cuda", dtype=torch.float16),
        "B": torch.randn((K, N), device="cuda", dtype=torch.float16),
        "bias": torch.randn((N,), device="cuda", dtype=torch.float16),
        "scale": 0.5,
    }


if __name__ == "__main__":
    main()
