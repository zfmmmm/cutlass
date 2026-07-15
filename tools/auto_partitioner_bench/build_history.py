#!/usr/bin/env python3
"""Build every preserved SM80 AutoPartitioner attempt and official CUTLASS."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path

from history_manifest import discover_versions


LEGACY_REVISION = "33a51510^"


def compile_command(
    nvcc: Path, source: Path, output: Path, repo: Path, cuda_home: Path
) -> list[str]:
    return [
        str(nvcc),
        "-forward-unknown-to-host-compiler",
        "-std=c++17",
        "-O3",
        "-lineinfo",
        f"-I{repo}",
        f"-I{repo / 'include'}",
        f"-I{repo / 'tools/util/include'}",
        "-isystem",
        str(cuda_home / "include"),
        "-isystem",
        str(cuda_home / "include/cccl"),
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        "--expt-relaxed-constexpr",
        "-ftemplate-backtrace-limit=0",
        "--generate-code=arch=compute_80,code=sm_80",
        str(source),
        "-o",
        str(output),
    ]


def run_compile(name: str, command: list[str], log_path: Path, timeout: int) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        output = completed.stdout
        status = "success" if completed.returncode == 0 else "build_failed"
        returncode = completed.returncode
    except subprocess.TimeoutExpired as error:
        output = (error.stdout or "") + "\nBUILD TIMEOUT\n"
        status = "build_timeout"
        returncode = 124
    log_path.write_text("COMMAND: " + " ".join(command) + "\n\n" + output)
    return {
        "name": name,
        "status": status,
        "returncode": returncode,
        "elapsed_sec": round(time.time() - started, 3),
        "command": command,
        "log": str(log_path),
    }


def extract_legacy_tree(repo: Path, destination: Path) -> str:
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", LEGACY_REVISION], text=True
    ).strip()
    marker = destination / ".source_commit"
    if marker.exists() and marker.read_text().strip() == revision:
        return revision
    if destination.exists():
        import shutil

        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    archive = destination.parent / "legacy_source.tar"
    subprocess.run(
        ["git", "-C", str(repo), "archive", "-o", str(archive), revision],
        check=True,
    )
    with tarfile.open(archive) as stream:
        stream.extractall(destination, filter="data")
    archive.unlink()
    marker.write_text(revision + "\n")
    return revision


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda-home", type=Path, default=Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")))
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    repo = args.repo.resolve()
    output = args.output.resolve()
    bin_dir = output / "bin"
    log_dir = output / "build_logs"
    bin_dir.mkdir(parents=True, exist_ok=True)
    legacy = output / "legacy_source"
    legacy_commit = extract_legacy_tree(repo, legacy)
    versions = discover_versions(repo)
    nvcc = args.cuda_home / "bin/nvcc"

    jobs = [
        (
            "official_cutlass",
            repo / "tools/auto_partitioner_bench/sm80_cutlass_official_gemm.cu",
            repo,
            bin_dir / "sm80_cutlass_official_gemm",
        )
    ]
    for version in versions:
        source_root = repo if version.version == "latest" else legacy
        jobs.append(
            (
                version.version,
                source_root / version.source,
                source_root,
                bin_dir / version.binary_name,
            )
        )

    results = []
    for name, source, source_root, binary in jobs:
        command = compile_command(nvcc, source, binary, source_root, args.cuda_home)
        result = run_compile(name, command, log_dir / f"{name}.log", args.timeout)
        result.update({"source": str(source), "binary": str(binary)})
        if result["status"] == "success" and not binary.exists():
            result["status"] = "artifact_missing"
        results.append(result)
        print(f"{name}: {result['status']} ({result['elapsed_sec']} sec)", flush=True)

    payload = {
        "repo": str(repo),
        "head_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "legacy_commit": legacy_commit,
        "cuda_home": str(args.cuda_home),
        "versions": [version.to_dict() for version in versions],
        "builds": results,
    }
    atomic_json(output / "build_results.json", payload)
    atomic_json(output / "version_manifest.json", payload["versions"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
