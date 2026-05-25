#pragma once
// 统一入口头文件：用户只需要 include 这个 builder，就能拿到当前支持的
// SM80/SM100 policy 偏特化。具体推导逻辑仍然分散在各架构文件里，
// 保持 SFINAE 路由清晰，也避免一个大类里继续嵌套大量 struct。
#include "cutlass/transform/collective/auto_partitioner/arch/sm100_policy.hpp"
#include "cutlass/transform/collective/auto_partitioner/arch/sm80_policy.hpp"
