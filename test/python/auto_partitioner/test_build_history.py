import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/auto_partitioner_bench"))

from build_history import compile_command, run_compile


class BuildHistoryTest(unittest.TestCase):
    def test_compile_command_targets_real_ampere(self):
        command = compile_command(
            nvcc=Path("/cuda/bin/nvcc"),
            source=Path("kernel.cu"),
            output=Path("kernel"),
            repo=Path("/repo"),
            cuda_home=Path("/cuda"),
        )
        joined = " ".join(map(str, command))
        self.assertIn("arch=compute_80,code=sm_80", joined)
        self.assertNotIn("sm_120", joined)
        self.assertIn("-std=c++17", command)

    def test_run_compile_classifies_failure_and_preserves_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            compiler = tmp / "nvcc"
            compiler.write_text("#!/bin/sh\necho intentional-compile-error\nexit 7\n")
            compiler.chmod(0o755)
            result = run_compile(
                name="v00",
                command=[str(compiler)],
                log_path=tmp / "v00.log",
                timeout=10,
            )
            self.assertEqual(result["status"], "build_failed")
            self.assertEqual(result["returncode"], 7)
            self.assertIn("intentional-compile-error", (tmp / "v00.log").read_text())


if __name__ == "__main__":
    unittest.main()
