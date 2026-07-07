"""Minimal GraphIR objects and tiny frontend helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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

    def fused_gemm_epilogue(self, a: Tensor, b: Tensor, bias: Tensor, scale: Tensor, name: str = "Y") -> Tensor:
        return self._emit(
            "fused_gemm_epilogue",
            [a, b, bias, scale],
            name,
            attrs={
                "epilogue": "bias_scale_gelu",
                "original_pattern": "gelu((A @ B + bias) * scale)",
            },
        )

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
