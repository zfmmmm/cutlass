#!/usr/bin/env python3
"""Immutable manifest for the preserved SM80 AutoPartitioner attempts."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class VersionSpec:
    version: str
    label: str
    source: str
    introducing_commit: str
    sha256: str

    @property
    def binary_name(self) -> str:
        return f"sm80_autopartition_gemm_{self.version}"

    def to_dict(self) -> dict:
        return {**asdict(self), "binary_name": self.binary_name}


SNAPSHOTS = (
    ("v00", "baseline_single_stage", "sm80_autopartition_gemm_v00_baseline_single_stage.cu", "07ee9235"),
    ("v01", "policy_tiled_g2s", "sm80_autopartition_gemm_v01_policy_tiled_g2s.cu", "07ee9235"),
    ("v02", "hoist_g2s_tiled_copy", "sm80_autopartition_gemm_v02_hoist_g2s_tiled_copy.cu", "07ee9235"),
    ("v03", "launch_bounds", "sm80_autopartition_gemm_v03_launch_bounds.cu", "07ee9235"),
    ("v04", "launch_bounds_3cta", "sm80_autopartition_gemm_v04_launch_bounds_3cta.cu", "07ee9235"),
    ("v05", "launch_bounds_4cta", "sm80_autopartition_gemm_v05_launch_bounds_4cta.cu", "07ee9235"),
    ("v06", "shared_union", "sm80_autopartition_gemm_v06_shared_union.cu", "07ee9235"),
    ("v07", "epilogue_autoselect_contract", "sm80_autopartition_gemm_v07_epilogue_autoselect_contract.cu", "27d44ffc"),
    ("v08", "fair_benchmark_baseline", "sm80_autopartition_gemm_v08_fair_benchmark_baseline.cu", "15394589"),
)


def _full_commit(repo: Path, revision: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", revision], text=True
    ).strip()


def discover_versions(repo: Path) -> list[VersionSpec]:
    repo = repo.resolve()
    snapshot_root = Path("tools/auto_partitioner_bench/snapshots")
    versions = []
    for version, label, filename, commit in SNAPSHOTS:
        relative = snapshot_root / filename
        source = repo / relative
        versions.append(
            VersionSpec(
                version=version,
                label=label,
                source=str(relative),
                introducing_commit=_full_commit(repo, commit),
                sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            )
        )
    latest = Path(
        "include/cutlass/transform/collective/auto_partitioner/examples/production_gemm/"
        "sm80_autopartition_gemm.cu"
    )
    versions.append(
        VersionSpec(
            version="latest",
            label="current_production_head",
            source=str(latest),
            introducing_commit=_full_commit(repo, "HEAD"),
            sha256=hashlib.sha256((repo / latest).read_bytes()).hexdigest(),
        )
    )
    return versions
