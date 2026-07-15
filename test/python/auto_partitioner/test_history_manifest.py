import hashlib
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/auto_partitioner_bench"))

from history_manifest import discover_versions


class HistoryManifestTest(unittest.TestCase):
    def test_discovers_v00_through_v08_and_latest(self):
        versions = discover_versions(ROOT)
        self.assertEqual(
            [version.version for version in versions],
            [*(f"v{i:02d}" for i in range(9)), "latest"],
        )
        self.assertEqual(len({version.binary_name for version in versions}), 10)

    def test_every_source_exists_and_sha_matches(self):
        for version in discover_versions(ROOT):
            source = ROOT / version.source
            self.assertTrue(source.is_file(), source)
            self.assertEqual(
                version.sha256,
                hashlib.sha256(source.read_bytes()).hexdigest(),
            )
            self.assertRegex(version.introducing_commit, r"^[0-9a-f]{40}$")


if __name__ == "__main__":
    unittest.main()
