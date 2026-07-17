#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
timestamp="$(date -u +%Y%m%d_%H%M%S)"
result_dir="${1:-${repo_root}/validation/autopartitioner_3090/results_${timestamp}}"
mkdir -p "${result_dir}"

exec > >(tee -a "${result_dir}/pipeline.log") 2>&1

echo "repo_root=${repo_root}"
echo "result_dir=${result_dir}"
echo "started_utc=$(date -u --iso-8601=seconds)"

{
  echo "date_utc=$(date -u --iso-8601=seconds)"
  echo "hostname=$(hostname)"
  echo "kernel=$(uname -a)"
  echo "git_head=$(git -C "${repo_root}" rev-parse HEAD)"
  echo "git_status_begin"
  git -C "${repo_root}" status --short
  echo "git_status_end"
  echo "nvidia_smi_begin"
  nvidia-smi
  echo "nvidia_smi_end"
  echo "nvcc_begin"
  "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" --version
  echo "nvcc_end"
  echo "cmake=$(cmake --version | head -1)"
  echo "cxx=$(${CXX:-g++} --version | head -1)"
  echo "python=$(python3 --version)"
  echo "disk_begin"
  df -h "${repo_root}"
  echo "disk_end"
} | tee "${result_dir}/environment.txt"

python3 - "${result_dir}" "${repo_root}" <<'PY'
import json, platform, subprocess, sys
from pathlib import Path

out, repo = map(Path, sys.argv[1:])
def command(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
payload = {
    "hostname": platform.node(),
    "platform": platform.platform(),
    "python": platform.python_version(),
    "git_head": command("git", "-C", str(repo), "rev-parse", "HEAD"),
    "gpu_query": command("nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version,temperature.gpu,power.limit", "--format=csv,noheader"),
    "nvcc": command(str(Path("/usr/local/cuda/bin/nvcc")), "--version"),
}
(out / "environment.json").write_text(json.dumps(payload, indent=2) + "\n")
PY

cat > "${result_dir}/commands.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd ${repo_root@Q}
python3 tools/auto_partitioner_bench/build_history.py --repo . --output ${result_dir@Q}
python3 tools/auto_partitioner_bench/run_history_benchmark.py --build-results ${result_dir@Q}/build_results.json --output ${result_dir@Q} --sizes 256,512,1024,2048,4096,8192 --warmup 10 --iterations 50 --repeat-runs 5 --verify-sizes 256,512
python3 tools/auto_partitioner_bench/report_history.py --results ${result_dir@Q}
EOF
chmod +x "${result_dir}/commands.sh"

python3 "${repo_root}/tools/auto_partitioner_bench/build_history.py" \
  --repo "${repo_root}" \
  --output "${result_dir}"

python3 "${repo_root}/tools/auto_partitioner_bench/run_history_benchmark.py" \
  --build-results "${result_dir}/build_results.json" \
  --output "${result_dir}" \
  --sizes 256,512,1024,2048,4096,8192 \
  --warmup 10 \
  --iterations 50 \
  --repeat-runs 5 \
  --verify-sizes 256,512

python3 "${repo_root}/tools/auto_partitioner_bench/report_history.py" \
  --results "${result_dir}"

echo "finished_utc=$(date -u --iso-8601=seconds)"
echo "report=${result_dir}/AutoPartitioner_RTX3090_Report.md"
