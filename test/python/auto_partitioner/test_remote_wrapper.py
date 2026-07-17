import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


class RemoteWrapperTest(unittest.TestCase):
    def test_wrapper_uses_tutorial_full_parameters_and_environment_capture(self):
        text = (ROOT / "tools/auto_partitioner_bench/run_3090_history.sh").read_text()
        self.assertIn("set -euo pipefail", text)
        self.assertIn("256,512,1024,2048,4096,8192", text)
        self.assertIn("--warmup 10", text)
        self.assertIn("--iterations 50", text)
        self.assertIn("--repeat-runs 5", text)
        self.assertIn("nvidia-smi", text)
        self.assertIn('bin/nvcc" --version', text)
        self.assertIn("validation/autopartitioner_3090/results_", text)
        self.assertNotIn("rm -rf", text)


if __name__ == "__main__":
    unittest.main()
