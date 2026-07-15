"""torch.compile backend and lowering records for the AutoPartition demo."""

from __future__ import annotations

import operator
import os
import subprocess
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class FxNodeRecord:
    name: str
    op: str
    target: str
    args: str
    users: list[str]
    shape: tuple[int, ...] | None = None
    dtype: str = ""
    device: str = ""
    stride: tuple[int, ...] | None = None
    is_contiguous: bool | None = None


@dataclass
class FusionPlan:
    plan_type: str
    source_nodes: list[str]
    covered_nodes: list[str]
    A: str
    B: str
    bias: str
    activation: str
    reason: str
    runtime_strategy: str
    output_node: str
    gemm_chains: list[dict[str, str]] = field(default_factory=list)
    operands: dict[str, Any] = field(default_factory=dict, repr=False)
    weight_transposed: bool = False


@dataclass
class FxAnalysis:
    raw_fx_graph: str
    nodes: list[FxNodeRecord]
    fusion_plans: list[FusionPlan]
    pattern_failures: list[str]


@dataclass
class GemmContract:
    name: str
    M: int
    N: int
    K: int
    A_name: str
    B_name: str
    bias_name: str
    A_shape: tuple[int, ...]
    B_shape: tuple[int, ...]
    bias_shape: tuple[int, ...]
    output_shape: tuple[int, ...]


@dataclass
class TensorContract:
    plan_type: str
    input_dtype: str
    weight_dtype: str
    bias_dtype: str
    output_dtype: str
    acc_dtype: str
    device: str
    A_layout: str
    B_layout: str
    bias_layout: str
    output_layout: str
    A_stride: tuple[int, ...]
    B_stride: tuple[int, ...]
    bias_stride: tuple[int, ...]
    A_contiguous: bool
    B_contiguous: bool
    bias_contiguous: bool
    bias_shape: tuple[int, ...]
    layout_normalization: list[str]
    gemms: list[GemmContract]
    actual_sm: str

    @property
    def M(self) -> int:
        return self.gemms[0].M

    @property
    def N(self) -> int:
        return self.gemms[0].N

    @property
    def K(self) -> int:
        return self.gemms[0].K


@dataclass
class AutoPartitionPlan:
    target_sm: str
    actual_sm: str
    policy: str
    tile_shape: tuple[int, int, int]
    thread_count: int
    alignment_bytes: int
    candidate_tile_shapes: list[tuple[int, int, int]]
    candidate_warp_shapes: list[tuple[int, int]]
    candidate_stage_counts: list[int]
    selected: dict[str, str]
    detected_interfaces: list[str]
    extension_functions: list[str]
    compile_arch_list: str
    probe_command: list[str] | None = None
    probe_error: str = ""


@dataclass
class LegalityResult:
    legal: bool
    reasons: list[str]
    warnings: list[str]
    fallback_backend: str = "PyTorchFallbackBackend"


@dataclass
class RuntimeCall:
    backend: str
    function: str
    input_shapes: dict[str, tuple[int, ...]]
    output_shape: tuple[int, ...] | None
    warmup: int = 0
    repeats: int = 0
    avg_ms: float | None = None
    max_abs_error: float | None = None
    max_rel_error: float | None = None
    passed: bool | None = None
    fallback_reason: str = ""


_ACTIVE_RECORDER: Any = None
_AUTOPARTITION_EXTENSION: Any = None


def set_active_recorder(recorder: Any | None) -> None:
    global _ACTIVE_RECORDER
    _ACTIVE_RECORDER = recorder


def record_event(kind: str, payload: Any) -> None:
    if _ACTIVE_RECORDER is not None and hasattr(_ACTIVE_RECORDER, "record"):
        _ACTIVE_RECORDER.record(kind, payload)


def autopartition_compile_backend(gm: Any, example_inputs: list[Any]):
    analysis = analyze_fx_graph(gm, example_inputs)
    record_event("fx_analysis", analysis)

    if not analysis.fusion_plans:
        reason = "; ".join(analysis.pattern_failures) or "unsupported whole graph pattern"
        record_event("fallback", reason)

        def fallback_callable(*args: Any):
            record_event(
                "runtime_call",
                RuntimeCall(
                    backend="PyTorchFallbackBackend",
                    function="gm.forward",
                    input_shapes=_input_shape_dict(args),
                    output_shape=None,
                    fallback_reason=reason,
                ),
            )
            return gm.forward(*args)

        return fallback_callable

    plan = analysis.fusion_plans[0]
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    placeholder_index = {node.name: idx for idx, node in enumerate(placeholders)}

    def compiled_callable(*args: Any):
        tensor_map = _runtime_tensor_map(plan, placeholder_index, args, gm)
        actual_sm = detect_actual_sm()
        contract = build_tensor_contract(plan, tensor_map, actual_sm=actual_sm)
        ap_plan = build_autopartition_plan(contract, actual_sm=actual_sm)
        legality = legalize_for_autopartition(contract, ap_plan)
        record_event("tensor_contract", contract)
        record_event("autopartition_plan", ap_plan)
        record_event("legality", legality)

        if not legality.legal:
            reason = "; ".join(legality.reasons)
            record_event("fallback", reason)
            out = gm.forward(*args)
            record_event(
                "runtime_call",
                RuntimeCall(
                    backend="PyTorchFallbackBackend",
                    function="gm.forward",
                    input_shapes=_input_shape_dict(args),
                    output_shape=_shape_tuple(out),
                    fallback_reason=reason,
                ),
            )
            return out

        try:
            module = load_autopartition_extension(ap_plan)
            if plan.plan_type == "GEMM_BIAS_GELU":
                A = _normalized_tensor(tensor_map["A"], "A", contract)
                B = _normalized_tensor(tensor_map["B"], "B", contract)
                bias = _normalized_tensor(tensor_map["bias"], "bias", contract)
                out = module.fused_gemm_bias_gelu(A, B, bias)
                function = "fused_gemm_bias_gelu"
            elif plan.plan_type == "MLP_TWO_GEMM":
                X = _normalized_tensor(tensor_map["X"], "X", contract)
                W1 = _normalized_tensor(tensor_map["W1"], "W1", contract)
                b1 = _normalized_tensor(tensor_map["b1"], "b1", contract)
                W2 = _normalized_tensor(tensor_map["W2"], "W2", contract)
                b2 = _normalized_tensor(tensor_map["b2"], "b2", contract)
                hidden = module.fused_gemm_bias_gelu(X, W1, b1)
                out = module.fused_gemm_bias(hidden, W2, b2)
                function = "fused_gemm_bias_gelu + fused_gemm_bias"
            else:
                raise RuntimeError(f"unsupported fusion plan at runtime: {plan.plan_type}")

            record_event(
                "runtime_call",
                RuntimeCall(
                    backend="AutoPartitionCudaBackend",
                    function=function,
                    input_shapes={key: tuple(value.shape) for key, value in tensor_map.items() if hasattr(value, "shape")},
                    output_shape=_shape_tuple(out),
                ),
            )
            return _pack_graph_output(gm, out)
        except Exception as exc:
            debug_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            record_event("debug_log", ("CUDA runtime", debug_text))
            out = gm.forward(*args)
            record_event(
                "runtime_call",
                RuntimeCall(
                    backend="PyTorchFallbackBackend",
                    function="gm.forward",
                    input_shapes=_input_shape_dict(args),
                    output_shape=_shape_tuple(out),
                    fallback_reason=f"AutoPartition extension failed: {exc}",
                ),
            )
            return out

    return compiled_callable


def analyze_fx_graph(gm: Any, example_inputs: list[Any]) -> FxAnalysis:
    del example_inputs
    nodes = [_record_node(node) for node in gm.graph.nodes]
    failures: list[str] = []
    output = _graph_output_node(gm)
    plans: list[FusionPlan] = []

    if output is None:
        failures.append("FX graph output is not a Tensor-producing node")
    else:
        mlp = _match_mlp_two_gemm(output)
        if mlp is not None:
            plans.append(mlp)
        else:
            failures.append("MLP_TWO_GEMM pattern did not match whole graph output")

        gemm_bias_gelu = _match_gemm_bias_gelu(output)
        if gemm_bias_gelu is not None:
            plans.append(gemm_bias_gelu)
        elif not plans:
            failures.append("GEMM_BIAS_GELU pattern did not match whole graph output")

    return FxAnalysis(raw_fx_graph=str(gm.graph), nodes=nodes, fusion_plans=plans[:1], pattern_failures=failures)


def build_tensor_contract(plan: FusionPlan, tensors: dict[str, Any], actual_sm: str | None = None) -> TensorContract:
    actual = actual_sm or detect_actual_sm()
    if plan.plan_type == "GEMM_BIAS_GELU":
        A = _lookup_tensor(tensors, "A", plan)
        B = _lookup_tensor(tensors, "B", plan)
        bias = _lookup_tensor(tensors, "bias", plan)
        gemm = _build_gemm_contract("gemm0", "A", A, "B", B, "bias", bias, output_name="Y")
        return _make_contract(plan.plan_type, A, B, bias, [gemm], actual)

    if plan.plan_type == "MLP_TWO_GEMM":
        X = _lookup_tensor(tensors, "X", plan)
        W1 = _lookup_tensor(tensors, "W1", plan)
        b1 = _lookup_tensor(tensors, "b1", plan)
        W2 = _lookup_tensor(tensors, "W2", plan)
        b2 = _lookup_tensor(tensors, "b2", plan)
        first = _build_gemm_contract("gemm0", "X", X, "W1", W1, "b1", b1, output_name="hidden")
        hidden_shape = first.output_shape
        second = _build_gemm_contract_from_shapes(
            "gemm1",
            "hidden",
            hidden_shape,
            "W2",
            tuple(W2.shape),
            "b2",
            tuple(b2.shape),
            output_name="Y",
        )
        return _make_contract(plan.plan_type, X, W1, b1, [first, second], actual)

    raise ValueError(f"unsupported plan type: {plan.plan_type}")


def build_autopartition_plan(contract: TensorContract, actual_sm: str | None = None) -> AutoPartitionPlan:
    actual = actual_sm or contract.actual_sm or detect_actual_sm()
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "8.0+PTX")
    selected, probe_command, probe_error = _probe_autopartitioner(contract)
    if not selected:
        selected = {
            "role_a": "AutoPartitioner<Sm80, TensorOp, fp16, StrideA, Shape<64,64,64>, 128, fp32>::RoleA",
            "role_b": "AutoPartitioner<Sm80, TensorOp, fp16, StrideB, Shape<64,64,64>, 128, fp32>::RoleB",
            "role_c": "AutoPartitioner<Sm80, TensorOp, fp16, StrideC, Shape<64,64,64>, 128, fp32>::RoleC",
            "role_a_use_ldmatrix": "selected at compile time",
            "role_b_use_ldmatrix": "selected at compile time",
            "role_c_output_alignment_bytes": "16",
        }
    return AutoPartitionPlan(
        target_sm="sm80",
        actual_sm=actual,
        policy="Sm80 AutoPartitioner RoleA/RoleB/RoleC policy; non-sm80 GPUs use PTX compatibility attempt",
        tile_shape=(64, 64, 64),
        thread_count=128,
        alignment_bytes=16,
        candidate_tile_shapes=[(64, 64, 64), (128, 64, 64), (64, 128, 64)],
        candidate_warp_shapes=[(2, 2), (1, 4), (4, 1)],
        candidate_stage_counts=[1, 2, 3],
        selected=selected,
        detected_interfaces=_detect_autopartition_interfaces(),
        extension_functions=["fused_gemm_bias_gelu", "fused_gemm_bias", "gemm"],
        compile_arch_list=arch_list,
        probe_command=probe_command,
        probe_error=probe_error,
    )


def legalize_for_autopartition(contract: TensorContract, plan: AutoPartitionPlan) -> LegalityResult:
    reasons: list[str] = []
    warnings: list[str] = []
    if contract.input_dtype != "fp16" or contract.weight_dtype != "fp16":
        reasons.append(f"input/weight dtype must be fp16, got {contract.input_dtype}/{contract.weight_dtype}")
    if contract.bias_dtype != "fp16" or contract.output_dtype != "fp16" or contract.acc_dtype != "fp32":
        reasons.append(
            f"bias/output/acc dtype must be fp16/fp16/fp32, got {contract.bias_dtype}/{contract.output_dtype}/{contract.acc_dtype}"
        )
    if not contract.device.startswith("cuda"):
        reasons.append(f"device must be CUDA for AutoPartition backend, got {contract.device}")
    for gemm in contract.gemms:
        if len(gemm.A_shape) != 2 or len(gemm.B_shape) != 2:
            reasons.append(f"{gemm.name} expects 2D GEMM tensors, got A={gemm.A_shape}, B={gemm.B_shape}")
        if gemm.bias_shape != (gemm.N,):
            reasons.append(f"{gemm.name} bias must be [{gemm.N}], got {gemm.bias_shape}")
        if gemm.M % 64 != 0 or gemm.N % 64 != 0 or gemm.K % 64 != 0:
            reasons.append(f"{gemm.name} M/N/K must be multiples of 64, got {gemm.M}/{gemm.N}/{gemm.K}")
    if plan.target_sm != "sm80":
        reasons.append(f"runtime currently instantiates sm80 policy only, got target {plan.target_sm}")
    actual_num = _sm_number(plan.actual_sm)
    if actual_num is not None and actual_num < 80:
        reasons.append(f"actual GPU {plan.actual_sm} is older than sm80")
    if actual_num is not None and actual_num != 80:
        warnings.append(f"actual GPU is {plan.actual_sm}; using sm80 AutoPartition policy through PTX compatibility attempt")
    if plan.alignment_bytes != 16:
        reasons.append(f"AutoPartition runtime requires 16-byte alignment, got {plan.alignment_bytes}")
    for item in contract.layout_normalization:
        warnings.append(item)
    return LegalityResult(legal=not reasons, reasons=reasons, warnings=warnings)


def load_autopartition_extension(plan: AutoPartitionPlan | None = None):
    global _AUTOPARTITION_EXTENSION
    if _AUTOPARTITION_EXTENSION is not None:
        return _AUTOPARTITION_EXTENSION

    import torch
    from torch.utils.cpp_extension import load

    repo = repo_root()
    project = project_root()
    source = project / "csrc" / "autopartition_runtime.cu"
    build_dir = Path(os.environ.get("AUTOPARTITION_TORCH_EXT_DIR", Path.home() / ".cache/autopartition_torch_backend/torch_extensions"))
    build_dir.mkdir(parents=True, exist_ok=True)

    old_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    old_jobs = os.environ.get("MAX_JOBS")
    os.environ["TORCH_CUDA_ARCH_LIST"] = old_arch or (plan.compile_arch_list if plan else "8.0+PTX")
    os.environ["MAX_JOBS"] = old_jobs or "1"
    try:
        _AUTOPARTITION_EXTENSION = load(
            name="autopartition_torch_runtime",
            sources=[str(source)],
            extra_include_paths=[str(repo / "include")],
            extra_cflags=["-std=c++17"],
            extra_cuda_cflags=["-std=c++17", "--expt-relaxed-constexpr"],
            build_directory=str(build_dir),
            verbose=False,
        )
    finally:
        if old_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = old_arch
        if old_jobs is None:
            os.environ.pop("MAX_JOBS", None)
        else:
            os.environ["MAX_JOBS"] = old_jobs
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return _AUTOPARTITION_EXTENSION


def run_torch_reference(plan_type: str, tensors: dict[str, Any]):
    import torch.nn.functional as F

    if plan_type == "GEMM_BIAS_GELU":
        return F.gelu(tensors["A"].float() @ tensors["B"].float() + tensors["bias"].float()).to(tensors["A"].dtype)
    if plan_type == "MLP_TWO_GEMM":
        hidden = F.gelu(tensors["X"].float() @ tensors["W1"].float() + tensors["b1"].float())
        return (hidden @ tensors["W2"].float() + tensors["b2"].float()).to(tensors["X"].dtype)
    raise ValueError(f"unsupported reference plan: {plan_type}")


def run_triton_baseline(plan_type: str, tensors: dict[str, Any]):
    if plan_type != "GEMM_BIAS_GELU":
        raise RuntimeError("Triton baseline only covers GEMM_BIAS_GELU in this demo")
    import torch
    import triton
    import triton.language as tl

    if not hasattr(tl, "erf"):
        raise RuntimeError("triton.language.erf is unavailable")
    kernel = _triton_gemm_bias_gelu_kernel()
    A = tensors["A"].contiguous()
    B = tensors["B"].contiguous()
    bias = tensors["bias"].contiguous()
    M, K = A.shape
    _, N = B.shape
    out = torch.empty((M, N), device=A.device, dtype=A.dtype)
    block_m, block_n, block_k = 16, 32, 32
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    kernel[grid](A, B, bias, out, M, N, K, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, num_warps=4, num_stages=3)
    return out


_TRITON_KERNEL: Any = None


def _triton_gemm_bias_gelu_kernel():
    global _TRITON_KERNEL
    if _TRITON_KERNEL is not None:
        return _TRITON_KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def _kernel(A, B, BIAS, OUT, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
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
        x = acc + bias[None, :]
        y = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))
        tl.store(OUT + offs_m[:, None] * N + offs_n[None, :],
                 y, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    _TRITON_KERNEL = _kernel
    return _TRITON_KERNEL


def benchmark(fn: Callable[[], Any], warmup: int = 5, repeats: int = 20) -> float:
    import torch

    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end) / repeats)
    start_s = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start_s) * 1000.0 / repeats


def compare_outputs(actual: Any, expected: Any, threshold: float = 1.0e-1) -> tuple[float, float, bool]:
    diff = (actual.detach().cpu().float() - expected.detach().cpu().float()).abs()
    expected_abs = expected.detach().cpu().float().abs()
    max_abs = float(diff.max().item())
    max_rel = float((diff / (expected_abs + 1.0e-6)).max().item())
    return max_abs, max_rel, max_abs < threshold


def detect_actual_sm() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return f"sm{major}{minor}"
    except Exception:
        pass
    return "unknown"


def project_root() -> Path:
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    return project_root().parent


def _record_node(node: Any) -> FxNodeRecord:
    value = node.meta.get("example_value") if hasattr(node, "meta") else None
    return FxNodeRecord(
        name=str(node.name),
        op=str(node.op),
        target=_target_text(node.target),
        args=_args_text(node.args),
        users=[str(user.name) for user in node.users],
        shape=_value_shape(value),
        dtype=_dtype_text(getattr(value, "dtype", "")),
        device=str(getattr(value, "device", "")) if value is not None else "",
        stride=_safe_stride(value),
        is_contiguous=_safe_contiguous(value),
    )


def _graph_output_node(gm: Any):
    for node in gm.graph.nodes:
        if node.op == "output":
            return _unwrap_output_arg(node.args[0])
    return None


def _pack_graph_output(gm: Any, value: Any):
    for node in gm.graph.nodes:
        if node.op != "output":
            continue
        spec = node.args[0]
        if isinstance(spec, tuple) and len(spec) == 1:
            return (value,)
        if isinstance(spec, list) and len(spec) == 1:
            return [value]
        return value
    return value


def _unwrap_output_arg(arg: Any):
    if isinstance(arg, (tuple, list)) and len(arg) == 1:
        return _unwrap_output_arg(arg[0])
    return arg if hasattr(arg, "op") else None


def _match_gemm_bias_gelu(output: Any) -> FusionPlan | None:
    if not _is_gelu(output):
        return None
    add = _single_arg_node(output)
    if add is None or not _is_add(add):
        linear = _single_arg_node(output)
        if linear is not None and _is_linear(linear):
            x, weight, bias = _linear_args(linear)
            return _gemm_bias_gelu_plan(output, linear, x, weight, bias, weight_transposed=True)
        return None
    matmul, bias = _split_binary_by_predicate(add, _is_matmul)
    if matmul is None or bias is None:
        return None
    A, B = _binary_nodes(matmul)
    return _gemm_bias_gelu_plan(output, add, A, B, bias, matmul_node=matmul)


def _gemm_bias_gelu_plan(output: Any, add_or_linear: Any, A: Any, B: Any, bias: Any, matmul_node: Any | None = None,
                         weight_transposed: bool = False) -> FusionPlan:
    source_nodes = [node.name for node in [matmul_node, add_or_linear, output] if node is not None]
    return FusionPlan(
        plan_type="GEMM_BIAS_GELU",
        source_nodes=source_nodes,
        covered_nodes=source_nodes,
        A=str(A.name),
        B=str(B.name),
        bias=str(bias.name),
        activation="gelu",
        reason="matmul result only consumed by bias add, add result only consumed by gelu; fusion legal",
        runtime_strategy="AutoPartition GEMM mainloop plus CUDA bias+GELU epilogue",
        output_node=str(output.name),
        gemm_chains=[{"A": str(A.name), "B": str(B.name), "bias": str(bias.name), "activation": "gelu"}],
        operands={"A": A, "B": B, "bias": bias},
        weight_transposed=weight_transposed,
    )


def _match_mlp_two_gemm(output: Any) -> FusionPlan | None:
    add2 = output if _is_add(output) else None
    if add2 is None:
        return None
    matmul2, b2 = _split_binary_by_predicate(add2, _is_matmul)
    if matmul2 is None or b2 is None:
        return None
    hidden, W2 = _binary_nodes(matmul2)
    if hidden is None or not _is_gelu(hidden):
        return None
    add1 = _single_arg_node(hidden)
    if add1 is None or not _is_add(add1):
        return None
    matmul1, b1 = _split_binary_by_predicate(add1, _is_matmul)
    if matmul1 is None or b1 is None:
        return None
    X, W1 = _binary_nodes(matmul1)
    source_nodes = [str(node.name) for node in [matmul1, add1, hidden, matmul2, add2]]
    return FusionPlan(
        plan_type="MLP_TWO_GEMM",
        source_nodes=source_nodes,
        covered_nodes=source_nodes,
        A=str(X.name),
        B=str(W1.name),
        bias=str(b1.name),
        activation="gelu",
        reason="linear1 feeds gelu, gelu feeds linear2, each intermediate has a single consumer; fusion legal as MLP_TWO_GEMM plan",
        runtime_strategy="two AutoPartition GEMM calls: fused_gemm_bias_gelu then fused_gemm_bias",
        output_node=str(output.name),
        gemm_chains=[
            {"A": str(X.name), "B": str(W1.name), "bias": str(b1.name), "activation": "gelu"},
            {"A": "hidden", "B": str(W2.name), "bias": str(b2.name), "activation": "none"},
        ],
        operands={"X": X, "W1": W1, "b1": b1, "W2": W2, "b2": b2},
    )


def _is_matmul(node: Any) -> bool:
    return _node_name(node) in {"matmul", "mm"} or "matmul" in _target_text(node.target).lower() or "aten.mm" in _target_text(node.target).lower()


def _is_add(node: Any) -> bool:
    name = _node_name(node)
    target = _target_text(node.target).lower()
    return name == "add" or "operator.add" in target or "built-in function add" in target or "aten.add" in target


def _is_gelu(node: Any) -> bool:
    return "gelu" in _node_name(node) or "gelu" in _target_text(node.target).lower()


def _is_linear(node: Any) -> bool:
    return "linear" in _node_name(node) or "linear" in _target_text(node.target).lower() or "addmm" in _target_text(node.target).lower()


def _node_name(node: Any) -> str:
    return str(getattr(getattr(node, "target", None), "__name__", getattr(node, "target", ""))).lower()


def _single_arg_node(node: Any):
    args = list(getattr(node, "args", ()))
    return args[0] if args and hasattr(args[0], "op") else None


def _binary_nodes(node: Any):
    args = list(getattr(node, "args", ()))
    if len(args) < 2:
        return None, None
    return args[0], args[1]


def _split_binary_by_predicate(node: Any, predicate: Callable[[Any], bool]):
    lhs, rhs = _binary_nodes(node)
    if lhs is not None and predicate(lhs):
        return lhs, rhs
    if rhs is not None and predicate(rhs):
        return rhs, lhs
    return None, None


def _linear_args(node: Any):
    args = list(getattr(node, "args", ()))
    if len(args) == 3:
        return args[0], args[1], args[2]
    raise ValueError(f"linear node expected 3 args, got {args}")


def _runtime_tensor_map(plan: FusionPlan, placeholder_index: dict[str, int], args: tuple[Any, ...], gm: Any) -> dict[str, Any]:
    del gm
    result: dict[str, Any] = {}
    for role, node in plan.operands.items():
        if node.op != "placeholder":
            raise RuntimeError(f"runtime operand {role} is not a placeholder in this first-version backend: {node}")
        result[role] = args[placeholder_index[node.name]]
    if plan.plan_type == "GEMM_BIAS_GELU":
        result.setdefault("A", result.get("x"))
    return result


def _lookup_tensor(tensors: dict[str, Any], key: str, plan: FusionPlan):
    if key in tensors:
        return tensors[key]
    node = plan.operands.get(key)
    names = []
    if node is not None:
        names.extend([str(node.name), str(node.target), _canonical_placeholder_name(str(node.name)), _canonical_placeholder_name(str(node.target))])
    for name in names:
        if name in tensors:
            return tensors[name]
    raise KeyError(f"tensor for role {key} not found; tried {names}")


def _build_gemm_contract(name: str, A_name: str, A: Any, B_name: str, B: Any, bias_name: str, bias: Any,
                         output_name: str) -> GemmContract:
    return _build_gemm_contract_from_shapes(name, A_name, tuple(A.shape), B_name, tuple(B.shape), bias_name, tuple(bias.shape), output_name)


def _build_gemm_contract_from_shapes(name: str, A_name: str, A_shape: tuple[int, ...], B_name: str, B_shape: tuple[int, ...],
                                     bias_name: str, bias_shape: tuple[int, ...], output_name: str) -> GemmContract:
    if len(A_shape) != 2 or len(B_shape) != 2:
        M = N = K = -1
        output_shape = ()
    else:
        M, K = A_shape
        K_b, N = B_shape
        if K_b != K:
            K = -1
        output_shape = (M, N)
    del output_name
    return GemmContract(name, M, N, K, A_name, B_name, bias_name, A_shape, B_shape, bias_shape, output_shape)


def _make_contract(plan_type: str, A: Any, B: Any, bias: Any, gemms: list[GemmContract], actual_sm: str) -> TensorContract:
    layout_norm = []
    for label, tensor in [("A", A), ("B", B), ("bias", bias)]:
        if hasattr(tensor, "is_contiguous") and not tensor.is_contiguous():
            layout_norm.append(f"{label} is non-contiguous; runtime will call contiguous() before extension")
    return TensorContract(
        plan_type=plan_type,
        input_dtype=_dtype_name(getattr(A, "dtype", None)),
        weight_dtype=_dtype_name(getattr(B, "dtype", None)),
        bias_dtype=_dtype_name(getattr(bias, "dtype", None)),
        output_dtype="fp16",
        acc_dtype="fp32",
        device=str(getattr(A, "device", "unknown")),
        A_layout=_layout_name(A),
        B_layout=_layout_name(B),
        bias_layout=_layout_name(bias),
        output_layout="row_major",
        A_stride=tuple(A.stride()) if hasattr(A, "stride") else (),
        B_stride=tuple(B.stride()) if hasattr(B, "stride") else (),
        bias_stride=tuple(bias.stride()) if hasattr(bias, "stride") else (),
        A_contiguous=bool(A.is_contiguous()) if hasattr(A, "is_contiguous") else False,
        B_contiguous=bool(B.is_contiguous()) if hasattr(B, "is_contiguous") else False,
        bias_contiguous=bool(bias.is_contiguous()) if hasattr(bias, "is_contiguous") else False,
        bias_shape=tuple(bias.shape),
        layout_normalization=layout_norm,
        gemms=gemms,
        actual_sm=actual_sm,
    )


def _normalized_tensor(tensor: Any, name: str, contract: TensorContract):
    del contract
    if hasattr(tensor, "is_contiguous") and not tensor.is_contiguous():
        record_event("layout_normalization", f"{name}.contiguous()")
        return tensor.contiguous()
    return tensor


def _probe_autopartitioner(contract: TensorContract) -> tuple[dict[str, str], list[str] | None, str]:
    source = repo_root() / "examples/ai_lowering_demo/zfm_compiler_demo/autopartition_sm80_probe.cu"
    if not source.exists():
        return {}, None, f"probe source not found: {source}"
    try:
        exe = _ensure_probe_built(source)
        command = [
            str(exe),
            f"--m={contract.M}",
            f"--n={contract.N}",
            f"--k={contract.K}",
            "--epilogue=bias_scale_gelu",
            "--target-sm=sm80",
            "--tile=64x64x64",
            "--threads=128",
            "--align-a=16",
            "--align-b=16",
            "--align-out=16",
        ]
        proc = subprocess.run(command, cwd=repo_root(), text=True, capture_output=True, check=False)
        if proc.returncode != 0:
            return {}, command, (proc.stderr.strip() or proc.stdout.strip())
        return _parse_key_value_output(proc.stdout), command, ""
    except Exception as exc:
        return {}, None, str(exc)


def _ensure_probe_built(source: Path) -> Path:
    build_dir = Path(os.environ.get("AUTOPARTITION_TORCH_PROBE_DIR", Path.home() / ".cache/autopartition_torch_backend/probe"))
    build_dir.mkdir(parents=True, exist_ok=True)
    exe = build_dir / "autopartition_sm80_probe"
    deps = [
        source,
        repo_root() / "include/cutlass/transform/collective/auto_partitioner/auto_partitioner.hpp",
        repo_root() / "include/cutlass/transform/collective/auto_partitioner/auto_partitioner_builder.hpp",
        repo_root() / "include/cutlass/transform/collective/auto_partitioner/arch/sm80_policy.hpp",
    ]
    newest = max(path.stat().st_mtime for path in deps if path.exists())
    if exe.exists() and exe.stat().st_mtime >= newest:
        return exe
    nvcc = os.environ.get("NVCC", "nvcc")
    command = [nvcc, "-std=c++17", "-O2", "-arch=sm_80", f"-I{repo_root() / 'include'}", str(source), "-o", str(exe)]
    proc = subprocess.run(command, cwd=repo_root(), text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return exe


def _detect_autopartition_interfaces() -> list[str]:
    candidates = [
        "include/cutlass/transform/collective/auto_partitioner/auto_partitioner.hpp",
        "include/cutlass/transform/collective/auto_partitioner/auto_partitioner_builder.hpp",
        "include/cutlass/transform/collective/auto_partitioner/arch/sm80_policy.hpp",
        "include/cutlass/transform/collective/auto_partitioner/examples/production_gemm/sm80_autopartition_gemm.cu",
    ]
    return [path for path in candidates if (repo_root() / path).exists()]


def _parse_key_value_output(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in text.splitlines():
        key, sep, value = raw.partition("=")
        if sep:
            result[key.strip()] = value.strip()
    return result


def _target_text(target: Any) -> str:
    return str(target)


def _args_text(args: Any) -> str:
    return str(args)


def _value_shape(value: Any) -> tuple[int, ...] | None:
    if value is None or not hasattr(value, "shape"):
        return None
    try:
        return tuple(int(dim) for dim in value.shape)
    except Exception:
        return None


def _dtype_text(dtype: Any) -> str:
    return str(dtype) if dtype else ""


def _safe_stride(value: Any) -> tuple[int, ...] | None:
    try:
        return tuple(value.stride())
    except Exception:
        return None


def _safe_contiguous(value: Any) -> bool | None:
    try:
        return bool(value.is_contiguous())
    except Exception:
        return None


def _dtype_name(dtype: Any) -> str:
    text = str(dtype)
    if "float16" in text or "Half" in text:
        return "fp16"
    if "float32" in text or text == "torch.float":
        return "fp32"
    if "bfloat16" in text:
        return "bf16"
    return text


def _layout_name(tensor: Any) -> str:
    if not hasattr(tensor, "shape") or not hasattr(tensor, "stride"):
        return "unknown"
    shape = tuple(tensor.shape)
    stride = tuple(tensor.stride())
    if len(shape) == 2 and stride == (shape[1], 1):
        return "row_major"
    if len(shape) == 1 and stride == (1,):
        return "contiguous_vector"
    if hasattr(tensor, "is_contiguous") and tensor.is_contiguous():
        return "contiguous"
    return f"strided{stride}"


def _canonical_placeholder_name(name: str) -> str:
    value = name
    if value.startswith("L_"):
        value = value[2:]
    if value.startswith("l_"):
        value = value[2:]
    return value.strip("_").lower()


def _input_shape_dict(args: tuple[Any, ...]) -> dict[str, tuple[int, ...]]:
    return {f"arg{idx}": tuple(arg.shape) for idx, arg in enumerate(args) if hasattr(arg, "shape")}


def _shape_tuple(value: Any) -> tuple[int, ...] | None:
    if isinstance(value, (tuple, list)) and len(value) == 1:
        return _shape_tuple(value[0])
    return tuple(value.shape) if hasattr(value, "shape") else None


def _sm_number(sm: str) -> int | None:
    if not sm.startswith("sm"):
        return None
    try:
        return int(sm[2:])
    except ValueError:
        return None
