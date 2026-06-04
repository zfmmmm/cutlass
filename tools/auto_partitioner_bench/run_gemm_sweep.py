#!/usr/bin/env python3
import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path


TFLOPS_RE = re.compile(r"tflops\s*=\s*([0-9.+\-eE]+)")
RUNTIME_RE = re.compile(r"runtime_ms\s*=\s*([0-9.+\-eE]+)")
DIFF_RE = re.compile(r"max_abs_(?:diff|error)\s*=\s*([0-9.+\-eE]+)")
INPUT_A_HASH_RE = re.compile(r"input_a_hash\s*=\s*([0-9]+)")
INPUT_B_HASH_RE = re.compile(r"input_b_hash\s*=\s*([0-9]+)")
OUTPUT_SUM_RE = re.compile(r"output_sum\s*=\s*([0-9.+\-eE]+)")
OUTPUT_ABS_SUM_RE = re.compile(r"output_abs_sum\s*=\s*([0-9.+\-eE]+)")
OUTPUT_SQ_SUM_RE = re.compile(r"output_sq_sum\s*=\s*([0-9.+\-eE]+)")
OUTPUT_HASH_RE = re.compile(r"output_hash\s*=\s*([0-9]+)")


def parse_sizes(text):
    sizes = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        sizes.append(int(item))
    if not sizes:
        raise ValueError("size list is empty")
    return sizes


def run_one(binary, size, args):
    cmd = [
        str(binary),
        f"--m={size}",
        f"--n={size}",
        f"--k={size}",
        f"--warmup={args.warmup}",
        f"--iterations={args.iterations}",
    ]
    verify_sizes = set(parse_sizes(args.verify_sizes)) if args.verify_sizes else set()
    if args.skip_reference and size not in verify_sizes:
        cmd.append("--skip-reference")

    completed = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = completed.stdout
    if completed.returncode != 0:
        sys.stderr.write(output)
        raise RuntimeError(f"{binary.name} failed for {size}x{size}x{size} with exit code {completed.returncode}")

    tflops_match = TFLOPS_RE.search(output)
    runtime_match = RUNTIME_RE.search(output)
    diff_match = DIFF_RE.search(output)
    input_a_hash_match = INPUT_A_HASH_RE.search(output)
    input_b_hash_match = INPUT_B_HASH_RE.search(output)
    output_sum_match = OUTPUT_SUM_RE.search(output)
    output_abs_sum_match = OUTPUT_ABS_SUM_RE.search(output)
    output_sq_sum_match = OUTPUT_SQ_SUM_RE.search(output)
    output_hash_match = OUTPUT_HASH_RE.search(output)
    if not tflops_match or not runtime_match:
        sys.stderr.write(output)
        raise RuntimeError(f"could not parse runtime/tflops from {binary.name}")
    if not input_a_hash_match or not input_b_hash_match:
        sys.stderr.write(output)
        raise RuntimeError(f"could not parse input hashes from {binary.name}")
    if not output_sum_match or not output_abs_sum_match or not output_sq_sum_match or not output_hash_match:
        sys.stderr.write(output)
        raise RuntimeError(f"could not parse output stats from {binary.name}")

    return {
        "runtime_ms": float(runtime_match.group(1)),
        "tflops": float(tflops_match.group(1)),
        "max_abs_diff": float(diff_match.group(1)) if diff_match else 0.0,
        "input_a_hash": int(input_a_hash_match.group(1)),
        "input_b_hash": int(input_b_hash_match.group(1)),
        "output_sum": float(output_sum_match.group(1)),
        "output_abs_sum": float(output_abs_sum_match.group(1)),
        "output_sq_sum": float(output_sq_sum_match.group(1)),
        "output_hash": int(output_hash_match.group(1)),
        "stdout": output,
    }


def median_summary(results):
    runtimes = [item["runtime_ms"] for item in results]
    diffs = [item["max_abs_diff"] for item in results]
    median_runtime = statistics.median(runtimes)
    median_tflops = statistics.median([item["tflops"] for item in results])
    representative = min(results, key=lambda item: abs(item["runtime_ms"] - median_runtime))
    return {
        "runtime_ms": median_runtime,
        "tflops": median_tflops,
        "max_abs_diff": max(diffs) if diffs else 0.0,
        "input_a_hash": representative["input_a_hash"],
        "input_b_hash": representative["input_b_hash"],
        "output_sum": representative["output_sum"],
        "output_abs_sum": representative["output_abs_sum"],
        "output_sq_sum": representative["output_sq_sum"],
        "output_hash": representative["output_hash"],
        "samples_runtime_ms": runtimes,
        "samples_tflops": [item["tflops"] for item in results],
        "representative_stdout": representative["stdout"],
    }


def rel_diff(lhs, rhs):
    return abs(lhs - rhs) / max(1.0, abs(lhs), abs(rhs))


def check_control_variables(size, summaries, output_rel_tol):
    if len(summaries) < 2:
        return

    reference_name, reference = summaries[0]
    for name, summary in summaries[1:]:
        if summary["input_a_hash"] != reference["input_a_hash"]:
            raise RuntimeError(
                f"input A hash mismatch at {size}: {reference_name}={reference['input_a_hash']} "
                f"{name}={summary['input_a_hash']}"
            )
        if summary["input_b_hash"] != reference["input_b_hash"]:
            raise RuntimeError(
                f"input B hash mismatch at {size}: {reference_name}={reference['input_b_hash']} "
                f"{name}={summary['input_b_hash']}"
            )
        if summary["output_hash"] != reference["output_hash"]:
            raise RuntimeError(
                f"output hash mismatch at {size}: {reference_name}={reference['output_hash']} "
                f"{name}={summary['output_hash']}"
            )

        abs_sum_diff = rel_diff(summary["output_abs_sum"], reference["output_abs_sum"])
        sq_sum_diff = rel_diff(summary["output_sq_sum"], reference["output_sq_sum"])
        if abs_sum_diff > output_rel_tol or sq_sum_diff > output_rel_tol:
            raise RuntimeError(
                f"output stats mismatch at {size}: {reference_name} vs {name}; "
                f"abs_sum_rel_diff={abs_sum_diff:.6g}, sq_sum_rel_diff={sq_sum_diff:.6g}, "
                f"tolerance={output_rel_tol:.6g}"
            )


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "arch",
                "implementation",
                "m",
                "n",
                "k",
                "runtime_ms",
                "tflops",
                "max_abs_diff",
                "input_a_hash",
                "input_b_hash",
                "output_sum",
                "output_abs_sum",
                "output_sq_sum",
                "output_hash",
                "runtime_samples_ms",
                "tflops_samples",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_csv(csv_path, png_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; CSV was written, plot skipped.")
        return

    rows = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            row["m"] = int(row["m"])
            row["tflops"] = float(row["tflops"])
            rows.append(row)

    grouped = {}
    for row in rows:
        grouped.setdefault(row["implementation"], []).append(row)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for name, data in sorted(grouped.items()):
        data = sorted(data, key=lambda item: item["m"])
        ax.plot([item["m"] for item in data], [item["tflops"] for item in data], marker="o", label=name)

    arch = rows[0]["arch"] if rows else "gemm"
    ax.set_title(f"{arch.upper()} GEMM Throughput")
    ax.set_xlabel("Square GEMM size M=N=K")
    ax.set_ylabel("TFLOP/s")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=180)
    print(f"plot_png={png_path}")


def main():
    parser = argparse.ArgumentParser(description="Sweep AutoPartitioner and official CUTLASS GEMM binaries.")
    parser.add_argument("--arch", choices=["sm80", "sm100"], required=True)
    parser.add_argument("--bin-dir", default="build/auto_partitioner_bench/bin")
    parser.add_argument("--sizes", default="256,512,1024,2048,4096,8192")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--verify-sizes", default="", help="Comma-separated sizes that still run CPU reference when --skip-reference is set.")
    parser.add_argument("--output-rel-tol", type=float, default=1.0e-4)
    parser.add_argument("--output", default="")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    bin_dir = Path(args.bin_dir)
    if args.arch == "sm80":
        binaries = [
            ("autopartitioner", bin_dir / "sm80_autopartition_gemm"),
            ("official_cutlass", bin_dir / "sm80_cutlass_official_gemm"),
        ]
    else:
        binaries = [
            ("autopartitioner", bin_dir / "sm100_autopartition_tma_umma_gemm"),
            ("official_cutlass", bin_dir / "sm100_cutlass_official_tma_umma_gemm"),
        ]

    sizes = parse_sizes(args.sizes)
    output = Path(args.output) if args.output else Path("build/auto_partitioner_bench/results") / f"{args.arch}_sweep.csv"

    if args.repeat_runs <= 0:
        raise ValueError("--repeat-runs must be positive")

    rows = []
    for size in sizes:
        per_impl_results = {implementation: [] for implementation, _ in binaries}
        for repeat_idx in range(args.repeat_runs):
            order = binaries if (repeat_idx % 2) == 0 else list(reversed(binaries))
            for implementation, binary in order:
                if not binary.exists():
                    raise FileNotFoundError(f"missing binary: {binary}")
                print(
                    f"running {implementation} {size}x{size}x{size} "
                    f"(sample {repeat_idx + 1}/{args.repeat_runs})"
                )
                result = run_one(binary, size, args)
                per_impl_results[implementation].append(result)
                print(f"  runtime_ms={result['runtime_ms']:.6g} tflops={result['tflops']:.6g}")

        summaries = []
        for implementation, _binary in binaries:
            summary = median_summary(per_impl_results[implementation])
            summaries.append((implementation, summary))
            runtime_samples = ",".join(f"{value:.6g}" for value in summary["samples_runtime_ms"])
            tflops_samples = ",".join(f"{value:.6g}" for value in summary["samples_tflops"])
            rows.append(
                {
                    "arch": args.arch,
                    "implementation": implementation,
                    "m": size,
                    "n": size,
                    "k": size,
                    "runtime_ms": summary["runtime_ms"],
                    "tflops": summary["tflops"],
                    "max_abs_diff": summary["max_abs_diff"],
                    "input_a_hash": summary["input_a_hash"],
                    "input_b_hash": summary["input_b_hash"],
                    "output_sum": summary["output_sum"],
                    "output_abs_sum": summary["output_abs_sum"],
                    "output_sq_sum": summary["output_sq_sum"],
                    "output_hash": summary["output_hash"],
                    "runtime_samples_ms": runtime_samples,
                    "tflops_samples": tflops_samples,
                }
            )
            print(
                f"summary {implementation} {size}x{size}x{size}: "
                f"median_runtime_ms={summary['runtime_ms']:.6g} "
                f"median_tflops={summary['tflops']:.6g}"
            )
            print(f"  runtime_samples_ms=[{runtime_samples}]")
            print(f"  tflops_samples=[{tflops_samples}]")
            print(f"  input_hashes=A{summary['input_a_hash']} B{summary['input_b_hash']}")
            print(
                f"  output_stats=sum{summary['output_sum']:.6g} "
                f"abs{summary['output_abs_sum']:.6g} sq{summary['output_sq_sum']:.6g} "
                f"hash{summary['output_hash']}"
            )
        check_control_variables(size, summaries, args.output_rel_tol)

    write_csv(output, rows)
    print(f"csv={output}")
    if args.plot:
        plot_csv(output, output.with_suffix(".png"))


if __name__ == "__main__":
    main()
