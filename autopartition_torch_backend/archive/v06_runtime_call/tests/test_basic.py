from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _load_backend():
    spec = importlib.util.spec_from_file_location("ap_backend", PROJECT_DIR / "backend.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class GemmBiasGelu(torch.nn.Module):
    def forward(self, x, w, bias):
        return F.gelu(x @ w + bias)


class MlpTwoGemm(torch.nn.Module):
    def forward(self, x, w1, b1, w2, b2):
        return F.gelu(x @ w1 + b1) @ w2 + b2


def test_detects_gemm_bias_gelu_from_fx():
    backend = _load_backend()
    gm = torch.fx.symbolic_trace(GemmBiasGelu())

    analysis = backend.analyze_fx_graph(gm, [])

    assert analysis.fusion_plans
    plan = analysis.fusion_plans[0]
    assert plan.plan_type == "GEMM_BIAS_GELU"
    assert plan.activation == "gelu"
    assert plan.source_nodes
    assert "fusion legal" in plan.reason
    assert analysis.raw_fx_graph.strip()
    assert all(record.name and record.op for record in analysis.nodes)


def test_detects_mlp_two_gemm_from_fx():
    backend = _load_backend()
    gm = torch.fx.symbolic_trace(MlpTwoGemm())

    analysis = backend.analyze_fx_graph(gm, [])

    assert analysis.fusion_plans
    plan = analysis.fusion_plans[0]
    assert plan.plan_type == "MLP_TWO_GEMM"
    assert plan.activation == "gelu"
    assert len(plan.gemm_chains) == 2
    assert "two AutoPartition GEMM" in plan.runtime_strategy


def test_tensor_contract_and_autopartition_plan_record_legality():
    backend = _load_backend()
    gm = torch.fx.symbolic_trace(GemmBiasGelu())
    analysis = backend.analyze_fx_graph(gm, [])
    tensors = {
        "x": torch.empty((128, 64), dtype=torch.float16),
        "w": torch.empty((64, 256), dtype=torch.float16),
        "bias": torch.empty((256,), dtype=torch.float16),
    }

    contract = backend.build_tensor_contract(analysis.fusion_plans[0], tensors, actual_sm="sm80")
    plan = backend.build_autopartition_plan(contract, actual_sm="sm80")
    legality = backend.legalize_for_autopartition(contract, plan)

    assert contract.M == 128
    assert contract.N == 256
    assert contract.K == 64
    assert contract.input_dtype == "fp16"
    assert contract.bias_shape == (256,)
    assert plan.target_sm == "sm80"
    assert plan.tile_shape == (64, 64, 64)
    assert plan.thread_count == 128
    assert any("device must be CUDA" in reason for reason in legality.reasons)


def test_cuda_source_uses_autopartitioner_roles():
    source = (PROJECT_DIR / "csrc" / "autopartition_runtime.cu").read_text(encoding="utf-8")

    assert "autopartition::AutoPartitioner" in source
    assert "::RoleA" in source
    assert "::RoleB" in source
    assert "::RoleC" in source
    assert "GlobalToSharedCopy" in source
    assert "SmemToRegCopyOperation" in source
    assert "TiledMma" in source
    assert "OutputRegisterToGlobalCopy" in source


def test_report_serialization_skips_fx_operands_without_deepcopy():
    backend = _load_backend()
    run_demo_spec = importlib.util.spec_from_file_location("ap_run_demo", PROJECT_DIR / "run_demo.py")
    assert run_demo_spec and run_demo_spec.loader
    run_demo = importlib.util.module_from_spec(run_demo_spec)
    sys.modules[run_demo_spec.name] = run_demo
    sys.modules["backend"] = backend
    run_demo_spec.loader.exec_module(run_demo)

    class BadDeepcopy:
        def __deepcopy__(self, _memo):
            raise RuntimeError("operands must not be deep-copied")

    plan = backend.FusionPlan(
        plan_type="GEMM_BIAS_GELU",
        source_nodes=["matmul", "add", "gelu"],
        covered_nodes=["matmul", "add", "gelu"],
        A="x",
        B="w",
        bias="bias",
        activation="gelu",
        reason="fusion legal",
        runtime_strategy="two-stage",
        output_node="gelu",
        operands={"A": BadDeepcopy()},
    )

    plain = run_demo._to_plain(plan)

    assert "operands" not in plain
    assert plain["plan_type"] == "GEMM_BIAS_GELU"


def test_compiled_autopartition_backend_preserves_user_output_shape_on_cuda():
    if not torch.cuda.is_available():
        return
    backend = _load_backend()

    class GemmBiasGeluCuda(torch.nn.Module):
        def forward(self, x, w, bias):
            return F.gelu(x @ w + bias)

    x = torch.randn((64, 64), device="cuda", dtype=torch.float16) * 0.02
    w = torch.randn((64, 64), device="cuda", dtype=torch.float16) * 0.02
    bias = torch.randn((64,), device="cuda", dtype=torch.float16) * 0.02
    compiled = torch.compile(GemmBiasGeluCuda().cuda(), backend=backend.autopartition_compile_backend, fullgraph=True)

    out = compiled(x, w, bias)

    assert tuple(out.shape) == (64, 64)
