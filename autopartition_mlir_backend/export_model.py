"""PyTorch -> torch.export -> torch-mlir HLO/Linalg front end."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F


class GemmBiasGelu(torch.nn.Module):
    def forward(self, x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return F.gelu(x @ w + bias)


def build_model() -> torch.nn.Module:
    return GemmBiasGelu().eval()


def build_inputs(
    device: str | torch.device | None = None,
    shape: tuple[int, int, int] = (1024, 1024, 1024),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, k, n = shape
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    factory = dict(device=device, dtype=torch.float16)
    return (
        torch.randn((m, k), **factory),
        torch.randn((k, n), **factory),
        torch.randn((n,), **factory),
    )


def _exported_program(model: torch.nn.Module, inputs: Iterable[torch.Tensor]):
    args = tuple(inputs)
    export = torch.export.export
    export_kwargs = {}
    signature = inspect.signature(export)
    if "strict" in signature.parameters:
        export_kwargs["strict"] = False
    return export(model, args, {}, **export_kwargs)


def print_fx_or_exported_graph(model: torch.nn.Module, inputs: Iterable[torch.Tensor]):
    program = _exported_program(model, inputs)
    print("PyTorch 前端捕获成功")
    print(program.graph_module.print_readable())
    return program


def _torch_mlir_module(program, output_type: str):
    from torch_mlir.compiler_utils import OutputType
    from torch_mlir.fx import export_and_import

    output = OutputType.get(output_type)
    # Empty decomposition tables avoid coupling this project to a private
    # decomposition registry that changes more often than the public API.
    return export_and_import(
        program,
        output_type=output,
        decomposition_table={},
        strict=False,
        enable_graph_printing=False,
        enable_ir_printing=False,
    )


def export_to_stablehlo(model, inputs, output_path: str | Path):
    program = _exported_program(model, inputs)
    module = _torch_mlir_module(program, "STABLEHLO")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = str(module)
    lowered = text.lower()
    if "stablehlo" not in lowered and "mhlo" not in lowered:
        raise RuntimeError("torch-mlir did not produce StableHLO/MHLO operations")
    output.write_text(text + "\n", encoding="utf-8")
    return program, text


def _run_hlo_to_linalg(module):
    from torch_mlir.passmanager import PassManager

    candidates = (
        "builtin.module(func.func(stablehlo-aggressive-simplification), stablehlo-legalize-to-linalg, stablehlo-convert-to-signless, canonicalize)",
        "builtin.module(stablehlo-legalize-to-linalg, stablehlo-convert-to-signless, canonicalize)",
        "builtin.module(chlo-legalize-to-stablehlo, stablehlo-legalize-to-linalg, stablehlo-convert-to-signless, canonicalize)",
    )
    errors = []
    for pipeline in candidates:
        try:
            manager = PassManager.parse(pipeline, context=module.context)
            manager.run(module.operation)
            text = str(module)
            if "linalg.matmul" in text:
                return text, pipeline
        except Exception as exc:  # API/pass registration varies by wheel.
            errors.append(f"{pipeline}: {exc}")
    raise RuntimeError("StableHLO -> Linalg lowering failed:\n" + "\n".join(errors))


def export_to_linalg(model, inputs, output_path: str | Path, stablehlo_path: str | Path | None = None):
    from torch_mlir import ir
    # The wheel keeps StableHLO registration in a separate extension module.
    # Importing it before parsing the saved HLO is required on current wheels.
    import torch_mlir._mlir_libs._stablehlo as stablehlo

    if stablehlo_path is None:
        stablehlo_path = Path(output_path).with_name("exported_stablehlo.mlir")
        export_to_stablehlo(model, inputs, stablehlo_path)
    stable_text = Path(stablehlo_path).read_text(encoding="utf-8")
    context = ir.Context()
    stablehlo.register_dialect(context)
    stablehlo.register_stablehlo_passes()
    module = ir.Module.parse(stable_text, context)
    text, pipeline = _run_hlo_to_linalg(module)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    if "linalg.matmul" not in text:
        raise RuntimeError("exported_linalg.mlir lacks linalg.matmul")
    return text, pipeline


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    model = build_model()
    inputs = build_inputs()
    print_fx_or_exported_graph(model, inputs)
    export_to_stablehlo(model, inputs, root / "mlir/exported_stablehlo.mlir")
    export_to_linalg(model, inputs, root / "mlir/exported_linalg.mlir")
