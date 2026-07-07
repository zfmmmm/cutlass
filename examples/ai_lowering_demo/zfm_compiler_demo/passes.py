"""Small GraphIR passes for shape/dtype inference and GEMM epilogue fusion."""

from __future__ import annotations

from .ir import Graph, Node, Tensor


def run_passes(graph: Graph) -> Graph:
    infer_shapes(graph)
    infer_dtypes(graph)
    canonicalize(graph)
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
            if a.shape is None or b.shape is None or len(a.shape) != 2 or len(b.shape) != 2:
                raise ValueError("matmul expects rank-2 inputs")
            m, k = a.shape
            k_b, n = b.shape
            if k != k_b:
                raise ValueError(f"matmul K mismatch: {a.shape} vs {b.shape}")
            node.output.shape = (m, n)
        elif node.op == "add":
            lhs, rhs = node.inputs
            node.output.shape = _binary_shape(lhs, rhs, allow_bias=True)
        elif node.op == "mul":
            lhs, rhs = node.inputs
            node.output.shape = _binary_shape(lhs, rhs, allow_scalar=True)
        elif node.op == "gelu":
            node.output.shape = node.inputs[0].shape
        elif node.op == "fused_gemm_epilogue":
            a, b, bias, _scale = node.inputs
            if a.shape is None or b.shape is None:
                raise ValueError("fused_gemm_epilogue expects shaped A and B")
            m, k = a.shape
            k_b, n = b.shape
            if k != k_b:
                raise ValueError(f"fused GEMM K mismatch: {a.shape} vs {b.shape}")
            if bias.shape != (n,):
                raise ValueError(f"bias must have shape ({n},), got {bias.shape}")
            node.output.shape = (m, n)
        else:
            raise ValueError(f"unsupported op for shape inference: {node.op}")


def infer_dtypes(graph: Graph) -> None:
    for node in graph.nodes:
        if node.op == "matmul":
            node.output.dtype = node.inputs[0].dtype
        elif node.op in {"add", "mul", "gelu"}:
            node.output.dtype = _first_tensor_dtype(node.inputs)
        elif node.op == "fused_gemm_epilogue":
            node.output.dtype = node.inputs[0].dtype
        else:
            raise ValueError(f"unsupported op for dtype inference: {node.op}")


def canonicalize(_graph: Graph) -> None:
    # The demo expression is already canonical. This named pass keeps the flow explicit.
    return


def eliminate_dead_code(graph: Graph) -> None:
    if graph.output is None:
        graph.nodes = []
        return

    live: set[int] = set()

    def visit(tensor: Tensor) -> None:
        if tensor.producer is None:
            return
        node = tensor.producer
        node_id = id(node)
        if node_id in live:
            return
        live.add(node_id)
        for inp in node.inputs:
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


def _binary_shape(lhs: Tensor, rhs: Tensor, allow_bias: bool = False, allow_scalar: bool = False) -> tuple[int, ...]:
    if lhs.shape is None or rhs.shape is None:
        raise ValueError("binary op expects shaped inputs")
    if lhs.shape == rhs.shape:
        return lhs.shape
    if allow_bias and len(lhs.shape) == 2 and rhs.shape == (lhs.shape[1],):
        return lhs.shape
    if allow_bias and len(rhs.shape) == 2 and lhs.shape == (rhs.shape[1],):
        return rhs.shape
    if allow_scalar and lhs.shape == ():
        return rhs.shape
    if allow_scalar and rhs.shape == ():
        return lhs.shape
    raise ValueError(f"unsupported broadcast: {lhs.shape} and {rhs.shape}")


def _first_tensor_dtype(inputs: list[Tensor]) -> str:
    for tensor in inputs:
        if tensor.shape != ():
            if tensor.dtype is None:
                raise ValueError(f"tensor %{tensor.name} has no dtype")
            return tensor.dtype
    if inputs[0].dtype is None:
        raise ValueError("input has no dtype")
    return inputs[0].dtype


def _split_by_producer(inputs: list[Tensor], op: str) -> tuple[Tensor | None, Tensor | None]:
    first, second = inputs
    if first.producer is not None and first.producer.op == op:
        return first, second
    if second.producer is not None and second.producer.op == op:
        return second, first
    return None, None

