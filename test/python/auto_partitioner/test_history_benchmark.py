import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/auto_partitioner_bench"))

from run_history_benchmark import alternating_order, parse_output, summarize_values


class HistoryBenchmarkTest(unittest.TestCase):
    def test_parse_output_accepts_legacy_and_instrumented_formats(self):
        legacy = parse_output("max_abs_diff = 0.01\nruntime_ms = 2.5\ntflops = 7.25\n")
        self.assertEqual(legacy["tflops"], 7.25)
        self.assertIsNone(legacy["output_hash"])
        current = parse_output(
            "runtime_ms=1\ntflops=8\noutput_hash=42\n"
            "input_a_hash=11\ninput_b_hash=12\noutput_sum=3\n"
        )
        self.assertEqual(current["output_hash"], 42)
        self.assertEqual(current["input_a_hash"], 11)

    def test_order_reverses_between_repeats(self):
        names = ["official", "v00", "v01"]
        self.assertEqual(alternating_order(names, 0), names)
        self.assertEqual(alternating_order(names, 1), list(reversed(names)))

    def test_summary_reports_median_std_and_cv(self):
        summary = summarize_values([10.0, 11.0, 9.0, 10.0, 10.0])
        self.assertEqual(summary["median"], 10.0)
        self.assertEqual(summary["mean"], 10.0)
        self.assertGreater(summary["std"], 0.0)
        self.assertAlmostEqual(summary["cv"], summary["std"] / 10.0)


if __name__ == "__main__":
    unittest.main()
