"""Straight-line GraphIR -> passes -> LoweredIR -> backend driver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .backends import AutoPartitionBackend, AutoPartitionCudaBackend, TritonFusedGemmBackend, TorchReferenceBackend


Shape = tuple[int, ...]


@dataclass
class Tensor:
    name: str
    shape: Shape | None = None
    dtype: str | None = None
    producer: "Node | None" = None
    value: Any = None
    layout: str = "row_major"

    def ref(self) -> str:
        return f"%{self.name}"

    @property
    def op(self) -> str | None:
        return self.producer.op if self.producer is not None else None

    @property
    def attrs(self) -> dict[str, Any]:
        return self.producer.attrs if self.producer is not None else {}


@dataclass
class Node:
    op: str
    inputs: list[Tensor]
    output: Tensor
    attrs: dict[str, Any] = field(default_factory=dict)


class Graph:
    def __init__(self) -> None:
        self.nodes: list[Node] = []
        self.tensors: dict[str, Tensor] = {}
        self.output: Tensor | None = None
        self._next_tmp = 0

    def placeholder(self, name: str, shape: Shape, dtype: str, layout: str = "row_major") -> Tensor:
        tensor = Tensor(name=name, shape=shape, dtype=dtype, layout=layout)
        self.tensors[name] = tensor
        return tensor

    def scalar(self, name: str, value: Any, dtype: str) -> Tensor:
        tensor = Tensor(name=name, shape=(), dtype=dtype, value=value)
        self.tensors[name] = tensor
        return tensor

    def matmul(self, a: Tensor, b: Tensor, name: str | None = None) -> Tensor:
        return self._emit("matmul", [a, b], name)

    def add(self, a: Tensor, b: Tensor, name: str | None = None) -> Tensor:
        return self._emit("add", [a, b], name)

    def mul(self, a: Tensor, b: Tensor, name: str | None = None) -> Tensor:
        return self._emit("mul", [a, b], name)

    def gelu(self, x: Tensor, name: str | None = None) -> Tensor:
        return self._emit("gelu", [x], name)

    def set_output(self, tensor: Tensor) -> None:
        self.output = tensor

    def pretty(self) -> str:
        lines: list[str] = []
        for node in self.nodes:
            args = ", ".join(t.ref() for t in node.inputs)
            line = f"{node.output.ref()} = {node.op}({args})"
            if node.attrs:
                lines.append(line + " {")
                for key, value in node.attrs.items():
                    lines.append(f"    {key} = {value}")
                lines.append("}")
            else:
                lines.append(line)
        return "\n".join(lines)

    def _emit(
        self,
        op: str,
        inputs: list[Tensor],
        name: str | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> Tensor:
        if name is None:
            name = str(self._next_tmp)
            self._next_tmp += 1
        tensor = Tensor(name=name)
        node = Node(op=op, inputs=inputs, output=tensor, attrs=attrs or {})
        tensor.producer = node
        self.nodes.append(node)
        self.tensors[name] = tensor
        return tensor


def build_demo_graph(M: int, N: int, K: int) -> Graph:
    graph = Graph()
    a = graph.placeholder("A", (M, K), "fp16")
    b = graph.placeholder("B", (K, N), "fp16")
    bias = graph.placeholder("bias", (N,), "fp16")
    scale = graph.scalar("scale", 0.5, "fp32")
    y = graph.gelu(graph.mul(graph.add(graph.matmul(a, b), bias), scale), name="Y")
    graph.set_output(y)
    return graph


def run_passes(graph: Graph) -> Graph:
    infer_shapes(graph)
    infer_dtypes(graph)
    eliminate_dead_code(graph)
    fuse_gemm_epilogue(graph)
    infer_shapes(graph)
    infer_dtypes(graph)
    eliminate_dead_code(graph)
    return graph


def infer_shapes(graph: Graph) -> None:
    for node in graph.nodes:
        if node.op == "matmul":
            a, b = node.inputs
            if a.shape is None or b.shape is None:
                raise ValueError("matmul expects shaped inputs")
            m, k = a.shape
            k_b, n = b.shape
            if k != k_b:
                raise ValueError(f"matmul K mismatch: {a.shape} vs {b.shape}")
            node.output.shape = (m, n)
        elif node.op == "add":
            node.output.shape = _binary_shape(node.inputs[0], node.inputs[1], allow_bias=True)
        elif node.op == "mul":
            node.output.shape = _binary_shape(node.inputs[0], node.inputs[1], allow_scalar=True)
        elif node.op == "gelu":
            node.output.shape = node.inputs[0].shape
        elif node.op == "fused_gemm_epilogue":
            a, b, bias, _scale = node.inputs
            if a.shape is None or b.shape is None:
                raise ValueError("fused_gemm_epilogue expects shaped A and B")
            m, k = a.shape
            k_b, n = b.shape
            if k != k_b or bias.shape != (n,):
                raise ValueError(f"bad fused_gemm_epilogue shapes: A={a.shape}, B={b.shape}, bias={bias.shape}")
            node.output.shape = (m, n)
        else:
            raise ValueError(f"unsupported op for shape inference: {node.op}")


def infer_dtypes(graph: Graph) -> None:
    for node in graph.nodes:
        if node.op in {"matmul", "add", "mul", "gelu", "fused_gemm_epilogue"}:
            node.output.dtype = _first_tensor_dtype(node.inputs)
        else:
            raise ValueError(f"unsupported op for dtype inference: {node.op}")


def eliminate_dead_code(graph: Graph) -> None:
    if graph.output is None:
        graph.nodes = []
        return

    live: set[int] = set()

    def visit(tensor: Tensor) -> None:
        if tensor.producer is None or id(tensor.producer) in live:
            return
        live.add(id(tensor.producer))
        for inp in tensor.producer.inputs:
            visit(inp)

    visit(graph.output)
    graph.nodes = [node for node in graph.nodes if id(node) in live]


def fuse_gemm_epilogue(graph: Graph) -> None:
    if graph.output is None:
        return

    gelu = graph.output.producer
    if gelu is None or gelu.op != "gelu":
        return
    mul = gelu.inputs[0].producer
    if mul is None or mul.op != "mul":
        return
    add_tensor, scale = _split_by_producer(mul.inputs, "add")
    if add_tensor is None or scale is None:
        return
    add = add_tensor.producer
    matmul_tensor, bias = _split_by_producer(add.inputs, "matmul")
    if matmul_tensor is None or bias is None:
        return
    matmul = matmul_tensor.producer
    a, b = matmul.inputs

    fused = Node(
        op="fused_gemm_epilogue",
        inputs=[a, b, bias, scale],
        output=graph.output,
        attrs={
            "epilogue": "bias_scale_gelu",
            "original_pattern": "gelu((A @ B + bias) * scale)",
        },
    )
    graph.output.producer = fused
    graph.nodes = [fused]


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
    if k != k_b or bias.shape != (n,):
        raise ValueError(f"bad lowering shapes: A={a.shape}, B={b.shape}, bias={bias.shape}")

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
        print("runtime inputs unavailable; skipped Torch reference, executable backends, correctness, and benchmark")
        return results

    _section("Torch Reference Result")
    y_ref = None
    if statuses[0].status != "legal":
        print(f"skipped: {statuses[0].reason}")
    else:
        try:
            y_ref = torch_backend.run(op, inputs)
            results["torch_reference"] = y_ref
            print(f"shape = {tuple(y_ref.shape)}, dtype = {y_ref.dtype}, device = {y_ref.device}")
        except Exception as exc:
            results["torch_reference_error"] = str(exc)
            print(f"failed: {exc}")

    _section("Triton Fused Result")
    y_triton = None
    if statuses[1].status != "legal":
        print(f"skipped: {statuses[1].reason}")
    else:
        try:
            y_triton = triton_backend.run(op, inputs)
            results["triton_fused"] = y_triton
            print(f"shape = {tuple(y_triton.shape)}, dtype = {y_triton.dtype}, device = {y_triton.device}")
        except Exception as exc:
            results["triton_fused_error"] = str(exc)
            print(f"failed: {exc}")

    _section("AutoPartition CUDA Result")
    y_autopart = None
    if statuses[2].status != "legal":
        print(f"skipped: {statuses[2].reason}")
    else:
        try:
            y_autopart = autopart_cuda_backend.run(op, inputs)
            results["autopartition_cuda"] = y_autopart
            print(f"shape = {tuple(y_autopart.shape)}, dtype = {y_autopart.dtype}, device = {y_autopart.device}")
        except Exception as exc:
            results["autopartition_cuda_error"] = str(exc)
            print(f"failed: {exc}")

    _section("Correctness")
    if y_ref is None:
        print("skipped: Torch reference output is required")
    else:
        if y_triton is None:
            print("triton_fused: skipped")
        else:
            max_abs, max_rel, passed = _compare_outputs(y_triton, y_ref)
            results.setdefault("correctness", {})["triton_fused"] = {
                "max_abs_error": max_abs,
                "max_rel_error": max_rel,
                "passed": passed,
            }
            print(f"triton_fused.max_abs_error = {max_abs:.6g}")
            print(f"triton_fused.max_rel_error = {max_rel:.6g}")
            print(f"triton_fused.passed = {str(passed).lower()}")

        if y_autopart is None:
            print("autopartition_cuda: skipped")
        else:
            max_abs, max_rel, passed = _compare_outputs(y_autopart, y_ref)
            results.setdefault("correctness", {})["autopartition_cuda"] = {
                "max_abs_error": max_abs,
                "max_rel_error": max_rel,
                "passed": passed,
            }
            print(f"autopartition_cuda.max_abs_error = {max_abs:.6g}")
            print(f"autopartition_cuda.max_rel_error = {max_rel:.6g}")
            print(f"autopartition_cuda.passed = {str(passed).lower()}")

    _section("Benchmark")
    if statuses[0].status != "legal":
        print(f"torch_reference: skipped ({statuses[0].reason})")
    elif y_ref is None:
        print("torch_reference: skipped (runtime failed before benchmark)")
    else:
        ms = torch_backend.benchmark(op, inputs, warmup, repeats)
        results["torch_reference_ms"] = ms
        print(f"torch_reference: {ms:.4f} ms")

    if statuses[1].status != "legal":
        print(f"triton_fused: skipped ({statuses[1].reason})")
    elif y_triton is None:
        print("triton_fused: skipped (runtime failed before benchmark)")
    else:
        ms = triton_backend.benchmark(op, inputs, warmup, repeats)
        results["triton_fused_ms"] = ms
        print(f"triton_fused: {ms:.4f} ms")

    if statuses[2].status != "legal":
        print(f"autopartition_cuda: skipped ({statuses[2].reason})")
    elif y_autopart is None:
        print("autopartition_cuda: skipped (runtime failed before benchmark)")
    else:
        ms = autopart_cuda_backend.benchmark(op, inputs, warmup, repeats)
        results["autopartition_cuda_ms"] = ms
        print(f"autopartition_cuda: {ms:.4f} ms")
    return results


def _binary_shape(lhs: Tensor, rhs: Tensor, allow_bias: bool = False, allow_scalar: bool = False) -> tuple[int, ...]:
    if lhs.shape is None or rhs.shape is None:
        raise ValueError("binary op expects shaped inputs")
    if lhs.shape == rhs.shape:
        return lhs.shape
    if allow_bias and len(lhs.shape) == 2 and rhs.shape == (lhs.shape[1],):
        return lhs.shape
    if allow_scalar and rhs.shape == ():
        return lhs.shape
    raise ValueError(f"unsupported broadcast: {lhs.shape} and {rhs.shape}")


def _first_tensor_dtype(inputs: list[Tensor]) -> str:
    for tensor in inputs:
        if tensor.shape != ():
            return _require_dtype(tensor.dtype, tensor.name)
    return _require_dtype(inputs[0].dtype, inputs[0].name)


def _split_by_producer(inputs: list[Tensor], op: str) -> tuple[Tensor | None, Tensor | None]:
    first, second = inputs
    if first.producer is not None and first.producer.op == op:
        return first, second
    if second.producer is not None and second.producer.op == op:
        return second, first
    return None, None


def _require_dtype(dtype: str | None, name: str) -> str:
    if dtype is None:
        raise ValueError(f"{name} has no inferred dtype")
    return dtype


def _section(title: str) -> None:
    print(f"\n========== {title} ==========")


def _compare_outputs(actual: Any, expected: Any) -> tuple[float, float, bool]:
    diff = (actual.to("cpu").float() - expected.to("cpu").float()).abs()
    expected_abs = expected.to("cpu").float().abs()
    max_abs = float(diff.max().item())
    max_rel = float((diff / (expected_abs + 1.0e-6)).max().item())
    return max_abs, max_rel, max_abs < 1.0e-1
