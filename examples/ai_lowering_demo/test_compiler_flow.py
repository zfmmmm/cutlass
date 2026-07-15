"""Behavior tests for the tiny GEMM lowering demo."""

import unittest

from zfm_compiler_demo.backends import AutoPartitionBackend, AutoPartitionCudaBackend
from zfm_compiler_demo.compiler import Graph, lower_graph, run_passes


class TinyCompilerFlowTest(unittest.TestCase):
    def build_graph(self):
        graph = Graph()
        a = graph.placeholder("A", (128, 64), "fp16")
        b = graph.placeholder("B", (64, 256), "fp16")
        bias = graph.placeholder("bias", (256,), "fp16")
        scale = graph.scalar("scale", 0.5, "fp32")
        y = graph.gelu(graph.mul(graph.add(graph.matmul(a, b), bias), scale), name="Y")
        graph.set_output(y)
        return graph

    def test_original_graph_pretty_prints_expression(self):
        graph = self.build_graph()

        text = graph.pretty()

        self.assertIn("%0 = matmul(%A, %B)", text)
        self.assertIn("%1 = add(%0, %bias)", text)
        self.assertIn("%2 = mul(%1, %scale)", text)
        self.assertIn("%Y = gelu(%2)", text)

    def test_passes_fuse_bias_scale_gelu_gemm_epilogue(self):
        graph = self.build_graph()

        optimized = run_passes(graph)

        self.assertEqual(optimized.output.op, "fused_gemm_epilogue")
        self.assertEqual(optimized.output.shape, (128, 256))
        self.assertEqual(optimized.output.dtype, "fp16")
        self.assertEqual(optimized.output.attrs["epilogue"], "bias_scale_gelu")
        self.assertEqual(optimized.output.attrs["original_pattern"], "gelu((A @ B + bias) * scale)")
        self.assertIn("%Y = fused_gemm_epilogue(%A, %B, %bias, %scale)", optimized.pretty())

    def test_lowering_records_backend_friendly_gemm_contract(self):
        optimized = run_passes(self.build_graph())

        lowered = lower_graph(optimized)

        self.assertEqual(len(lowered), 1)
        op = lowered[0]
        self.assertEqual(op.M, 128)
        self.assertEqual(op.N, 256)
        self.assertEqual(op.K, 64)
        self.assertEqual(op.dtype_a, "fp16")
        self.assertEqual(op.dtype_b, "fp16")
        self.assertEqual(op.dtype_bias, "fp16")
        self.assertEqual(op.dtype_scale, "fp32")
        self.assertEqual(op.dtype_out, "fp16")
        self.assertEqual(op.acc_dtype, "fp32")
        self.assertEqual(op.layout_a, "row_major")
        self.assertEqual(op.layout_b, "row_major")
        self.assertEqual(op.layout_out, "row_major")
        self.assertEqual(op.epilogue, "bias_scale_gelu")
        self.assertEqual(op.target_sm, "sm80")
        self.assertEqual(op.tile_shape, (64, 64, 64))
        self.assertEqual(op.thread_count, 128)
        self.assertEqual(op.gmem_alignment_a, 16)
        self.assertEqual(op.gmem_alignment_b, 16)
        self.assertEqual(op.gmem_alignment_out, 16)
        self.assertFalse(op.allow_cublaslt_rewrite)

    def test_autopartition_backend_generates_plan_only_contract(self):
        lowered = lower_graph(run_passes(self.build_graph()))[0]

        plan = AutoPartitionBackend().plan(lowered)

        self.assertFalse(plan.runnable)
        self.assertEqual(plan.config.M, 128)
        self.assertEqual(plan.config.N, 256)
        self.assertEqual(plan.config.K, 64)
        self.assertEqual(plan.config.epilogue, "bias_scale_gelu")
        self.assertIn((64, 64, 64), plan.config.candidate_tile_shapes)
        self.assertIn("layout/template planner", plan.reason)
        self.assertIn("AutoPartitionPlan:", plan.pretty())
        self.assertIn("gmem_to_smem", plan.pretty())

    def test_autopartition_backend_consumes_autopartitioner_probe_result(self):
        lowered = lower_graph(run_passes(self.build_graph()))[0]

        def fake_probe(_config):
            return {
                "source": "autopartitioner_probe",
                "role_a_use_ldmatrix": "true",
                "role_b_use_ldmatrix": "true",
                "role_a_swizzle_base": "3",
                "role_b_swizzle_base": "3",
                "role_c_epilogue_layout_candidate_count": "48",
                "role_c_has_zero_glue_epilogue_mapping": "true",
                "role_c_has_fusion_shared_mapping": "true",
                "role_c_epilogue_swizzle_base": "3",
                "role_c_epilogue_bank_conflict_score": "0",
                "role_c_output_alignment_bytes": "16",
            }

        plan = AutoPartitionBackend(probe_runner=fake_probe).plan(lowered)

        self.assertEqual(plan.source, "autopartitioner_probe")
        self.assertEqual(plan.selected["role_a_use_ldmatrix"], "true")
        self.assertEqual(plan.selected["role_c_has_fusion_shared_mapping"], "true")
        self.assertIn("RoleA::UseLdMatrix = true", plan.pretty())
        self.assertIn("RoleC::EpilogueLayoutCandidateCount = 48", plan.pretty())

    def test_autopartition_cuda_backend_is_real_execution_backend(self):
        lowered = lower_graph(run_passes(self.build_graph()))[0]

        backend = AutoPartitionCudaBackend()
        status = backend.legality(lowered)

        self.assertEqual(backend.name, "AutoPartitionCudaBackend")
        self.assertIn(status.status, {"legal", "unavailable"})
        self.assertNotEqual(status.status, "plan_only")


if __name__ == "__main__":
    unittest.main()
