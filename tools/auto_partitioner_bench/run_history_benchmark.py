#!/usr/bin/env python3
"""Run the tutorial's fair SM80 sweep across all historical implementations."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import time
from pathlib import Path


PATTERNS = {
    "runtime_ms": re.compile(r"runtime_ms\s*=\s*([0-9.eE+\-]+)"),
    "tflops": re.compile(r"tflops\s*=\s*([0-9.eE+\-]+)"),
    "max_abs_diff": re.compile(r"max_abs_(?:diff|error)\s*=\s*([0-9.eE+\-]+)"),
    "input_a_hash": re.compile(r"input_a_hash\s*=\s*([0-9]+)"),
    "input_b_hash": re.compile(r"input_b_hash\s*=\s*([0-9]+)"),
    "output_hash": re.compile(r"output_hash\s*=\s*([0-9]+)"),
    "output_sum": re.compile(r"output_sum\s*=\s*([0-9.eE+\-]+)"),
    "output_abs_sum": re.compile(r"output_abs_sum\s*=\s*([0-9.eE+\-]+)"),
    "output_sq_sum": re.compile(r"output_sq_sum\s*=\s*([0-9.eE+\-]+)"),
}


def parse_output(text: str) -> dict:
    parsed = {}
    for key, pattern in PATTERNS.items():
        match = pattern.search(text)
        if key.endswith("hash"):
            parsed[key] = int(match.group(1)) if match else None
        else:
            parsed[key] = float(match.group(1)) if match else None
    if parsed["runtime_ms"] is None or parsed["tflops"] is None:
        raise ValueError("benchmark output does not contain runtime_ms and tflops")
    return parsed


def alternating_order(names: list[str], repeat_index: int) -> list[str]:
    return list(names if repeat_index % 2 == 0 else reversed(names))


def summarize_values(values: list[float]) -> dict:
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "median": statistics.median(values),
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "cv": std / mean if mean else None,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def parse_sizes(text: str) -> list[int]:
    values = [int(value) for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("size list is empty")
    return values


def run_sample(
    name: str,
    binary: Path,
    size: int,
    repeat: int,
    warmup: int,
    iterations: int,
    verify: bool,
    timeout: int,
    log_path: Path,
) -> dict:
    command = [
        str(binary),
        f"--m={size}",
        f"--n={size}",
        f"--k={size}",
        f"--warmup={warmup}",
        f"--iterations={iterations}",
    ]
    if not verify:
        command.append("--skip-reference")
    started = time.time()
    status = "success"
    parsed = {}
    returncode = None
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        returncode = completed.returncode
        text = completed.stdout
        if completed.returncode != 0:
            status = "correctness_failed" if completed.returncode == 2 else "runtime_failed"
        else:
            try:
                parsed = parse_output(text)
            except ValueError:
                status = "output_parse_failed"
    except subprocess.TimeoutExpired as error:
        text = (error.stdout or "") + "\nRUN TIMEOUT\n"
        status = "timeout"
        returncode = 124
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("COMMAND: " + " ".join(command) + "\n\n" + text)
    return {
        "implementation": name,
        "m": size,
        "n": size,
        "k": size,
        "repeat": repeat,
        "warmup": warmup,
        "iterations": iterations,
        "reference_checked": verify,
        "status": status,
        "returncode": returncode,
        "wall_time_sec": round(time.time() - started, 3),
        "log": str(log_path),
        **parsed,
    }


def build_summaries(samples: list[dict], expected_repeats: int) -> list[dict]:
    groups = {}
    for row in samples:
        groups.setdefault((row["implementation"], row["m"]), []).append(row)
    summaries = []
    for (name, size), rows in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        successes = [row for row in rows if row["status"] == "success"]
        record = {
            "implementation": name,
            "m": size,
            "n": size,
            "k": size,
            "expected_samples": expected_repeats,
            "successful_samples": len(successes),
            "status": "success" if len(successes) == expected_repeats else "incomplete",
        }
        if successes:
            for metric in ("runtime_ms", "tflops"):
                metric_summary = summarize_values([row[metric] for row in successes])
                record.update({f"{metric}_{key}": value for key, value in metric_summary.items()})
            record["max_abs_diff"] = max(
                (row.get("max_abs_diff") or 0.0) for row in successes
            )
            hashes = {row.get("output_hash") for row in successes if row.get("output_hash") is not None}
            record["output_hash"] = hashes.pop() if len(hashes) == 1 else None
        summaries.append(record)
    return summaries


def comparisons(summaries: list[dict]) -> list[dict]:
    official = {
        row["m"]: row
        for row in summaries
        if row["implementation"] == "official_cutlass" and row["status"] == "success"
    }
    rows = []
    for row in summaries:
        if row["implementation"] == "official_cutlass":
            continue
        baseline = official.get(row["m"])
        ratio = None
        if baseline and row["status"] == "success":
            ratio = row["tflops_median"] / baseline["tflops_median"]
        official_hash = None if baseline is None else baseline.get("output_hash")
        output_hash = row.get("output_hash")
        hash_matches = (
            None
            if official_hash is None or output_hash is None
            else output_hash == official_hash
        )
        rows.append(
            {
                **row,
                "ap_over_official": ratio,
                "ap_over_official_percent": None if ratio is None else ratio * 100.0,
                "hash_matches_official": hash_matches,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", default="256,512,1024,2048,4096,8192")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeat-runs", type=int, default=5)
    parser.add_argument("--verify-sizes", default="256,512")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    build = json.loads(args.build_results.read_text())
    successful_builds = {
        row["name"]: Path(row["binary"])
        for row in build["builds"]
        if row["status"] == "success"
    }
    expected_names = ["official_cutlass", *[row["version"] for row in build["versions"]]]
    runnable = [name for name in expected_names if name in successful_builds]
    sizes = parse_sizes(args.sizes)
    verify_sizes = set(parse_sizes(args.verify_sizes))
    samples = []
    for size in sizes:
        for repeat in range(args.repeat_runs):
            for name in alternating_order(runnable, repeat):
                print(f"size={size} repeat={repeat + 1}/{args.repeat_runs} implementation={name}", flush=True)
                row = run_sample(
                    name,
                    successful_builds[name],
                    size,
                    repeat,
                    args.warmup,
                    args.iterations,
                    size in verify_sizes,
                    args.timeout,
                    output / "run_logs" / f"{name}_m{size}_r{repeat + 1}.log",
                )
                samples.append(row)
                print(f"  status={row['status']} tflops={row.get('tflops')}", flush=True)
    summaries = build_summaries(samples, args.repeat_runs)
    comparison_rows = comparisons(summaries)
    failures = [
        row for row in build["builds"] if row["status"] != "success"
    ] + [row for row in samples if row["status"] != "success"]
    write_csv(output / "raw_samples.csv", samples)
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "official_comparison.csv", comparison_rows)
    configuration = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    configuration["output"] = str(output)
    atomic_json(output / "summary.json", {"configuration": configuration, "summaries": summaries, "comparisons": comparison_rows, "failures": failures})
    atomic_json(output / "failures.json", failures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
