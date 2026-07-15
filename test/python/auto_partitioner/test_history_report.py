import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/auto_partitioner_bench"))

from report_history import build_report


class HistoryReportTest(unittest.TestCase):
    def test_report_uses_summary_and_keeps_failed_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "summary.json").write_text(
                json.dumps(
                    {
                        "configuration": {"sizes": "256", "warmup": 10, "iterations": 50, "repeat_runs": 5},
                        "summaries": [
                            {"implementation": "official_cutlass", "m": 256, "status": "success", "tflops_median": 8.0, "tflops_cv": 0.01},
                            {"implementation": "v00", "m": 256, "status": "success", "tflops_median": 7.0, "tflops_cv": 0.02},
                        ],
                        "comparisons": [
                            {"implementation": "v00", "m": 256, "status": "success", "tflops_median": 7.0, "ap_over_official_percent": 87.5}
                        ],
                        "failures": [{"name": "v01", "status": "build_failed", "log": "build_logs/v01.log"}],
                    }
                )
            )
            (tmp / "version_manifest.json").write_text(
                json.dumps([
                    {"version": "v00", "label": "baseline", "introducing_commit": "a" * 40, "sha256": "b" * 64},
                    {"version": "v01", "label": "broken", "introducing_commit": "c" * 40, "sha256": "d" * 64},
                ])
            )
            report = build_report(tmp)
            self.assertIn("87.50%", report)
            self.assertIn("v01", report)
            self.assertIn("build_failed", report)


if __name__ == "__main__":
    unittest.main()
