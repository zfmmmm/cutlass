# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

cutlass_test_unit_add_executable(
  cutlass_test_unit_transform_collective_auto_partitioner
  sm1xx_policy_audit.cu
)

add_subdirectory(sm80_policy_audit)
