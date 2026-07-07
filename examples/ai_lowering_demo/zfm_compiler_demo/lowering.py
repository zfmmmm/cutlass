"""Lower fused GraphIR nodes into backend-friendly GEMM contracts."""

from __future__ import annotations

from dataclasses import dataclass

from .ir import Graph


@dataclass
class LoweredGemmEpilogueOp:
    M: int
    N: int
    K: int
    dtype_a: str
    dtype_b: str
    dtype_bias: str
    dtype_scale: str
    dtype_out: str
    acc_dtype: str
    layout_a: str
    layout_b: str
    layout_out: str
    epilogue: str
    target: str
    target_sm: str
    tile_shape: tuple[int, int, int]
    thread_count: int
    gmem_alignment_a: int
    gmem_alignment_b: int
    gmem_alignment_out: int
    allow_cublaslt_rewrite: bool

    @property
    def op(self) -> str:
        return "fused_gemm_epilogue"

    def pretty(self) -> str:
        return "\n".join(
            [
                "op = fused_gemm_epilogue",
                f"M = {self.M}, N = {self.N}, K = {self.K}",
                f"A = {self.dtype_a} {self.layout_a}",
                f"B = {self.dtype_b} {self.layout_b}",
                f"OUT = {self.dtype_out} {self.layout_out}",
                f"ACC = {self.acc_dtype}",
                "epilogue = acc + bias, then scale, then gelu",
                f"target = {self.target}",
                f"target_sm = {self.target_sm}",
                f"tile_shape = {self.tile_shape[0]}x{self.tile_shape[1]}x{self.tile_shape[2]}",
                f"thread_count = {self.thread_count}",
                f"gmem_alignment = A{self.gmem_alignment_a} / B{self.gmem_alignment_b} / OUT{self.gmem_alignment_out} bytes",
                f"allow_cublaslt_rewrite = {str(self.allow_cublaslt_rewrite).lower()}",
            ]
        )


def lower_graph(graph: Graph, target: str = "cuda") -> list[LoweredGemmEpilogueOp]:
    if graph.output is None or graph.output.producer is None:
        raise ValueError("graph has no output node to lower")

    node = graph.output.producer
    if node.op != "fused_gemm_epilogue":
        raise ValueError(f"only fused_gemm_epilogue can be lowered, got {node.op}")

    a, b, bias, scale = node.inputs
    if a.shape is None or b.shape is None or bias.shape is None:
        raise ValueError("lowering expects inferred input shapes")
    m, k = a.shape
    k_b, n = b.shape
    if k != k_b:
        raise ValueError(f"lowering K mismatch: {a.shape} vs {b.shape}")
    if bias.shape != (n,):
        raise ValueError(f"bias shape mismatch: expected ({n},), got {bias.shape}")

    return [
        LoweredGemmEpilogueOp(
            M=m,
            N=n,
            K=k,
            dtype_a=_require_dtype(a.dtype, "A"),
            dtype_b=_require_dtype(b.dtype, "B"),
            dtype_bias=_require_dtype(bias.dtype, "bias"),
            dtype_scale=_require_dtype(scale.dtype, "scale"),
            dtype_out=_require_dtype(graph.output.dtype, "Y"),
            acc_dtype="fp32",
            layout_a=a.layout,
            layout_b=b.layout,
            layout_out=graph.output.layout,
            epilogue=node.attrs["epilogue"],
            target=target,
            target_sm="sm80",
            tile_shape=(64, 64, 64),
            thread_count=128,
            gmem_alignment_a=16,
            gmem_alignment_b=16,
            gmem_alignment_out=16,
            allow_cublaslt_rewrite=False,
        )
    ]


def _require_dtype(dtype: str | None, name: str) -> str:
    if dtype is None:
        raise ValueError(f"{name} has no inferred dtype")
    return dtype
