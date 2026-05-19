#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../../" && pwd)"
cuda_bin="${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"
out_dir="${1:-${repo_root}/build/auto_partitioner_production_gemm}"

mkdir -p "${out_dir}"

common_flags=(
  --expt-relaxed-constexpr
  -std=c++17
  -I"${repo_root}/include"
  -I"${repo_root}/include/cutlass/transform/collective/auto_partitioner/examples/production_gemm"
)

"${cuda_bin}" "${common_flags[@]}" \
  -arch=sm_80 \
  "${repo_root}/include/cutlass/transform/collective/auto_partitioner/examples/production_gemm/sm80_autopartition_gemm.cu" \
  -o "${out_dir}/sm80_autopartition_gemm"

"${cuda_bin}" "${common_flags[@]}" \
  -I"${repo_root}" \
  -I"${repo_root}/tools/util/include" \
  -I"${repo_root}/examples/cute/tutorial" \
  -arch=sm_100a \
  "${repo_root}/include/cutlass/transform/collective/auto_partitioner/examples/production_gemm/sm100_autopartition_tma_umma_gemm.cu" \
  -o "${out_dir}/sm100_autopartition_tma_umma_gemm" \
  -lcuda

echo "Built:"
echo "  ${out_dir}/sm80_autopartition_gemm"
echo "  ${out_dir}/sm100_autopartition_tma_umma_gemm"
