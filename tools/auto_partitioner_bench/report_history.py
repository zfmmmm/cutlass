#!/usr/bin/env python3
"""Create the auditable RTX 3090 historical benchmark report."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def fmt(value, digits=3, suffix=""):
    return "N/A" if value is None else f"{float(value):.{digits}f}{suffix}"


def build_report(result_dir: Path) -> str:
    data = json.loads((result_dir / "summary.json").read_text())
    manifest = json.loads((result_dir / "version_manifest.json").read_text())
    configuration = data["configuration"]
    summaries = data["summaries"]
    comparisons = data["comparisons"]
    failures = data["failures"]
    official = {
        row["m"]: row
        for row in summaries
        if row["implementation"] == "official_cutlass"
    }

    lines = [
        "# AutoPartitioner v00→latest RTX 3090 Benchmark Report",
        "",
        "## Scope and fairness",
        "",
        "This report measures every preserved SM80 implementation attempt (v00-v08 and latest) against the controlled official CUTLASS single-stage GEMM. It follows the tutorial harness rather than CUTLASS Profiler heuristic selection.",
        "",
        f"- Sizes: `{configuration.get('sizes')}`",
        f"- Warmup launches: `{configuration.get('warmup')}`",
        f"- Timed iterations: `{configuration.get('iterations')}`",
        f"- Process repeats: `{configuration.get('repeat_runs')}`",
        "- Primary statistic: median TFLOP/s; execution order reverses on alternating repeats.",
        "- CPU reference: 256 and 512. Later historical sources do not all expose hashes; missing hashes are reported as unknown rather than invented.",
        "",
        "## Version manifest",
        "",
        "|Version|Attempt|Introducing commit|Source SHA256|",
        "|---|---|---|---|",
    ]
    for row in manifest:
        lines.append(
            f"|{row['version']}|{row['label']}|`{row['introducing_commit'][:12]}`|`{row['sha256'][:16]}…`|"
        )

    lines += [
        "",
        "## Throughput and official comparison",
        "",
        "|Version|M=N=K|Median TFLOP/s|Official TFLOP/s|AP / official|CV|Status|",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in comparisons:
        baseline = official.get(row["m"], {})
        lines.append(
            f"|{row['implementation']}|{row['m']}|{fmt(row.get('tflops_median'))}|"
            f"{fmt(baseline.get('tflops_median'))}|{fmt(row.get('ap_over_official_percent'), 2, '%')}|"
            f"{fmt(row.get('tflops_cv'), 4)}|{row['status']}|"
        )

    valid = [row for row in comparisons if row.get("ap_over_official_percent") is not None]
    by_size = defaultdict(list)
    by_version = defaultdict(list)
    for row in valid:
        by_size[row["m"]].append(row)
        by_version[row["implementation"]].append(row["ap_over_official_percent"])
    lines += ["", "## Best attempt by size", "", "|M=N=K|Best version|TFLOP/s|Official ratio|", "|---:|---|---:|---:|"]
    for size, rows in sorted(by_size.items()):
        best = max(rows, key=lambda row: row["tflops_median"])
        lines.append(
            f"|{size}|{best['implementation']}|{best['tflops_median']:.3f}|{best['ap_over_official_percent']:.2f}%|"
        )
    lines += ["", "## Cross-size aggregate", "", "|Version|Valid sizes|Mean AP/official|", "|---|---:|---:|"]
    for name, ratios in sorted(by_version.items()):
        lines.append(f"|{name}|{len(ratios)}|{sum(ratios) / len(ratios):.2f}%|")

    lines += ["", "## Failures and unavailable evidence", ""]
    if failures:
        lines += ["|Stage/name|Status|Log|", "|---|---|---|"]
        for row in failures:
            name = row.get("name") or f"{row.get('implementation')} m{row.get('m')} r{row.get('repeat')}"
            lines.append(f"|{name}|{row.get('status')}|`{row.get('log', 'N/A')}`|")
    else:
        lines.append("No build, correctness, runtime, parse, or timeout failures were recorded.")
    lines += [
        "",
        "## Interpretation rules",
        "",
        "- A ratio above 100% means the AutoPartitioner attempt exceeded the controlled official single-stage comparator for that shape; it does not claim superiority over every CUTLASS kernel.",
        "- Ratios within one percent are treated as performance parity unless repeat dispersion clearly separates them.",
        "- 256 is dominated by launch and fixed overhead; 4096/8192 better represent sustained GEMM throughput.",
        "- Results apply to this RTX 3090, driver, CUDA toolchain, clock/thermal state, and the preserved source revisions.",
        "",
        "## Artifact traceability",
        "",
        "Raw process samples are in `raw_samples.csv`; all build and run stdout/stderr are retained under `build_logs/` and `run_logs/`. `environment.json`, `commands.sh`, `build_results.json`, and `version_manifest.json` provide the reproduction record.",
    ]
    return "\n".join(lines) + "\n"


def make_plots(result_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    data = json.loads((result_dir / "summary.json").read_text())
    comparisons = [row for row in data["comparisons"] if row.get("tflops_median") is not None]
    official = [row for row in data["summaries"] if row["implementation"] == "official_cutlass" and row.get("tflops_median") is not None]
    grouped = defaultdict(list)
    for row in comparisons:
        grouped[row["implementation"]].append(row)
    figure, axis = plt.subplots(figsize=(11, 6.5))
    for name, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["m"])
        axis.plot([row["m"] for row in rows], [row["tflops_median"] for row in rows], marker="o", linewidth=1.2, label=name)
    official.sort(key=lambda row: row["m"])
    axis.plot([row["m"] for row in official], [row["tflops_median"] for row in official], marker="s", color="black", linewidth=2.4, label="official_cutlass")
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Square GEMM size M=N=K")
    axis.set_ylabel("Median TFLOP/s")
    axis.set_title("RTX 3090 AutoPartitioner Historical Throughput")
    axis.grid(True, alpha=0.3)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    throughput = result_dir / "history_throughput.png"
    figure.savefig(throughput, dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 6.5))
    for name, rows in sorted(grouped.items()):
        rows = [row for row in rows if row.get("ap_over_official_percent") is not None]
        rows.sort(key=lambda row: row["m"])
        axis.plot([row["m"] for row in rows], [row["ap_over_official_percent"] for row in rows], marker="o", linewidth=1.2, label=name)
    axis.axhline(100.0, color="black", linestyle="--", linewidth=1.5)
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Square GEMM size M=N=K")
    axis.set_ylabel("AutoPartitioner / official (%)")
    axis.set_title("RTX 3090 Relative Performance vs Official Single-Stage CUTLASS")
    axis.grid(True, alpha=0.3)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    relative = result_dir / "history_vs_official.png"
    figure.savefig(relative, dpi=180)
    plt.close(figure)
    return [str(throughput), str(relative)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    result_dir = args.results.resolve()
    report = build_report(result_dir)
    (result_dir / "AutoPartitioner_RTX3090_Report.md").write_text(report)
    plots = make_plots(result_dir)
    data = json.loads((result_dir / "summary.json").read_text())
    manifest = json.loads((result_dir / "version_manifest.json").read_text())
    acceptance = {
        "manifest_versions": len(manifest),
        "expected_versions_present": [row["version"] for row in manifest] == [*(f"v{i:02d}" for i in range(9)), "latest"],
        "official_present": any(row["implementation"] == "official_cutlass" for row in data["summaries"]),
        "failures": len(data["failures"]),
        "plots": plots,
        "report": str(result_dir / "AutoPartitioner_RTX3090_Report.md"),
    }
    (result_dir / "acceptance.json").write_text(json.dumps(acceptance, indent=2) + "\n")
    print(json.dumps(acceptance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
