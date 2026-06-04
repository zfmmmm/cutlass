#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cuda_home="${CUDA_HOME:-/usr/local/cuda}"
nvcc="${cuda_home}/bin/nvcc"
out_dir="${1:-${repo_root}/build/auto_partitioner_bench/bin}"

mkdir -p "${out_dir}"

common_flags=(
  -forward-unknown-to-host-compiler
  -std=c++17
  -O3
  -lineinfo
  -I"${repo_root}"
  -I"${repo_root}/include"
  -I"${repo_root}/build/include"
  -I"${repo_root}/tools/util/include"
  -isystem "${cuda_home}/include"
  -isystem "${cuda_home}/include/cccl"
  -DCUTLASS_VERSIONS_GENERATED
  -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1
  -DCUTLASS_ENABLE_GDC_FOR_SM100=1
  --expt-relaxed-constexpr
  -ftemplate-backtrace-limit=0
)

sm80_arch="${SM80_ARCH:-120}"
sm100_arch="${SM100_ARCH:-100a}"

sm80_gencode=(
  "--generate-code=arch=compute_${sm80_arch},code=sm_${sm80_arch}"
  "--generate-code=arch=compute_${sm80_arch},code=compute_${sm80_arch}"
)

"${nvcc}" "${common_flags[@]}" "${sm80_gencode[@]}" \
  "${repo_root}/include/cutlass/transform/collective/auto_partitioner/examples/production_gemm/sm80_autopartition_gemm.cu" \
  -o "${out_dir}/sm80_autopartition_gemm"

"${nvcc}" "${common_flags[@]}" "${sm80_gencode[@]}" \
  "${repo_root}/tools/auto_partitioner_bench/sm80_cutlass_official_gemm.cu" \
  -o "${out_dir}/sm80_cutlass_official_gemm"

if [[ "${BUILD_SM100:-0}" == "1" ]]; then
  "${nvcc}" "${common_flags[@]}" "-arch=sm_${sm100_arch}" \
    "${repo_root}/include/cutlass/transform/collective/auto_partitioner/examples/production_gemm/sm100_autopartition_tma_umma_gemm.cu" \
    -o "${out_dir}/sm100_autopartition_tma_umma_gemm" \
    -lcuda

  "${nvcc}" "${common_flags[@]}" "-arch=sm_${sm100_arch}" \
    "${repo_root}/tools/auto_partitioner_bench/sm100_cutlass_official_tma_umma_gemm.cu" \
    -o "${out_dir}/sm100_cutlass_official_tma_umma_gemm" \
    -lcuda
fi

echo "Built benchmark binaries in ${out_dir}"
echo "  ${out_dir}/sm80_autopartition_gemm"
echo "  ${out_dir}/sm80_cutlass_official_gemm"
if [[ "${BUILD_SM100:-0}" == "1" ]]; then
  echo "  ${out_dir}/sm100_autopartition_tma_umma_gemm"
  echo "  ${out_dir}/sm100_cutlass_official_tma_umma_gemm"
fi
