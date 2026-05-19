# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

add_subdirectory(auto_partitioner)

add_custom_target(
  cutlass_test_unit_transform_collective
  DEPENDS
  cutlass_test_unit_transform_collective_auto_partitioner
  cutlass_test_unit_transform_collective_auto_partitioner_sm80_policy_audit
)

add_custom_target(
  test_unit_transform_collective
  DEPENDS
  test_unit_transform_collective_auto_partitioner
  test_unit_transform_collective_auto_partitioner_sm80_policy_audit
)
