#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <iostream>
#include <type_traits>
#include <vector>

#include "auto_partitioner_builder.hpp"
#include "autopartition_example_utils.hpp"

using namespace cute;

// =================================================================================================
// 生产级 Global GEMM Kernel (支持任意矩阵规模)
// 目标架构：SM120 (Blackwell 消费级 / RTX 50 系)
// 核心载荷：FP8 (e4m3) 输入 -> FP32 累加 -> FP32 输出
// =================================================================================================
template <typename PartA,
          typename PartB,
          typename PartC,
          typename InputElement,
          typename OutputElement,
          typename StrideA,
          typename StrideB,
          typename StrideC,
          int ThreadCount>
__global__ void sm120_tensorop_global_autopartition_kernel(InputElement const *ptr_A,
                                                           StrideA             stride_A,
                                                           InputElement const *ptr_B,
                                                           StrideB             stride_B,
                                                           OutputElement      *ptr_C,
                                                           StrideC             stride_C,
                                                           int                 M,
                                                           int                 N,
                                                           int                 K) // 引入全局矩阵维度
{
    // 【1. 静态 Tile 尺寸推断】
    // 依靠 AutoPartitioner 生成的 SmemLayout，在编译期静态提取当前 Thread Block 负责的子矩阵尺寸。
    // 这解耦了 Kernel 逻辑与具体的 Tile 调优参数。
    using bM = decltype(size<0>(typename PartA::SmemLayout{}));
    using bN = decltype(size<0>(typename PartB::SmemLayout{}));
    using bK = decltype(size<1>(typename PartA::SmemLayout{}));

    // 【2. FP8 物理载荷适配 (极其关键)】
    // Blackwell SM120 的低精度 rr (Register-to-Register) MMA 要求极高。
    // Policy 中已将 FP8 映射为 uint8_t 连续内存块进行高效搬运。
    // 此处的 reinterpret_cast 是安全的，因为 float_e4m3_t 和 uint8_t 在物理层面上都是 1 Byte。
    using SmemElement = typename PartA::SmemElement;
    auto const *raw_A = reinterpret_cast<SmemElement const *>(ptr_A);
    auto const *raw_B = reinterpret_cast<SmemElement const *>(ptr_B);

    // 【3. 全局张量视图构建】
    // 结合传入的 M, N, K 和步长 (Stride)，建立整个大矩阵的寻址代数空间。
    Tensor gA_full = make_tensor(make_gmem_ptr(raw_A), make_shape(M, K), stride_A);
    Tensor gB_full = make_tensor(make_gmem_ptr(raw_B), make_shape(N, K), stride_B);
    Tensor gC_full = make_tensor(make_gmem_ptr(ptr_C), make_shape(M, N), stride_C);

    // 【4. 基于 blockIdx 的空间域切分 (Global Tiling)】
    // 利用 local_tile 将全局张量分割为当前 CTA 应负责的网格。
    // make_coord(blockIdx.x, _) 意味着：在 M 维锁定当前 blockIdx.x 对应的子块，而在 K 维保留所有子块，等待流水线消费。
    // 返回的 gA 形状为 (bM, bK, num_k_tiles)。
    Tensor gA = local_tile(gA_full, make_tile(bM{}, bK{}), make_coord(blockIdx.x, _));
    Tensor gB = local_tile(gB_full, make_tile(bN{}, bK{}), make_coord(blockIdx.y, _));
    // C 矩阵不需要在 K 维迭代，直接锁定二维坐标 (blockIdx.x, blockIdx.y)。
    Tensor gC = local_tile(gC_full, make_tile(bM{}, bN{}), make_coord(blockIdx.x, blockIdx.y));

    // 【5. 物理绝对安全的 Shared Memory 分配】
    struct SharedStorage
    {
        // cosize_v 能够穿透 AutoPartitioner 生成的任意复杂的 Swizzle/Padding 掩码，
        // 精确计算出物理上需要的 uint8_t 字节总数。
        cute::array_aligned<SmemElement, cute::cosize_v<typename PartA::SmemLayout>> smemA;
        cute::array_aligned<SmemElement, cute::cosize_v<typename PartB::SmemLayout>> smemB;
    };
    __shared__ SharedStorage smem;

    // 将物理共享内存与 Policy 规划的最优 SmemLayout 绑定。
    Tensor sA = make_tensor(make_smem_ptr(smem.smemA.data()), typename PartA::SmemLayout{});
    Tensor sB = make_tensor(make_smem_ptr(smem.smemB.data()), typename PartB::SmemLayout{});

    // 【6. MMA 引擎组装与累加器初始化】
    // PartC::TiledMma 在 SM120 FP8 路径下，绑定的是 rr_op_selector_sm120 (Register-to-Register MMA)。
    typename PartC::TiledMma mma;
    auto                     thr_mma = mma.get_thread_slice(threadIdx.x);

    // partition_C 为当前线程在 C 矩阵上划分逻辑坐标。
    Tensor tCgC = thr_mma.partition_C(gC);
    // make_fragment_C 在当前线程的 RF (寄存器堆) 中分配物理累加器。
    // 依据 Policy，这里是 FP32 (float) 类型的数组，以防止 FP8 累加带来的严重精度溢出。
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);
    clear(tCrC); // 初始化累加器为 0.0f

    // 【7. K 维度时间域流水线 (The Mainloop)】
    int num_k_tiles = size<2>(gA);

#pragma unroll 1 // 禁止编译器过度展开主循环，控制寄存器溢出 (Register Spilling)
    for (int k_step = 0; k_step < num_k_tiles; ++k_step) {

        // 7.1 取出当前流水线周期的 K 维切片 (bM x bK 和 bN x bK)
        Tensor gA_k = gA(_, _, k_step);
        Tensor gB_k = gB(_, _, k_step);

        // 7.2 全局显存 -> 共享内存的异步搬运
        // 在 SM120 FP8 模式下，GmemToSmemCopy 降级为 AutoCopyAsync。
        // 因为 FP8 粒度极细，且常需要配合 Scale 缩放因子，编译器会依据类型和对齐边界自动联编最优的 cp.async 或
        // ld.global。 <ThreadCount, 128> 表示 128-bit 向量化假设。
        cooperative_copy<ThreadCount, 128>(threadIdx.x, gA_k, sA, typename PartA::GmemToSmemCopy{});
        cooperative_copy<ThreadCount, 128>(threadIdx.x, gB_k, sB, typename PartB::GmemToSmemCopy{});

        // 7.3 提交异步批次并阻塞
        cp_async_fence();
        cp_async_wait<0>();

        // 7.4 读后写 (RAW) 防御：确保 Shared Memory 数据就绪
        __syncthreads();

        // 7.5 发射 Tensor Core 计算指令
        // cooperative_gemm 会利用 PartA/B::SmemToRegCopyOperation (例如 sm120_rr_smem_copy_selector)
        // 将 FP8 (uint8_t) 数据从 Smem 提拉至寄存器，随后触发 Blackwell FP8 MMA。
        // 计算结果原位累加至 tCrC。
        cooperative_gemm(threadIdx.x,
                         mma,
                         sA,
                         sB,
                         tCrC,
                         identity{},
                         identity{},
                         typename PartA::SmemToRegCopyOperation{},
                         typename PartB::SmemToRegCopyOperation{});

        // 7.6 写后写 (WAW) 防御：防止下一次循环的显存加载覆盖当前正在被 MMA 消费的 Shared Memory 数据
        __syncthreads();
    }

    // 【8. Epilogue：寄存器直写回显存】
    // 经过循环累加，tCrC (FP32) 包含了最终结果。
    // 在本 Policy 中，tCgC 的值类型推导为 float (OutputElement)，与 tCrC 类型一致。
    // cute::copy 能够自动识别同类型张量，并底层展开为高度并行的 st.global (Store Global) 汇编指令流。
    cute::copy(tCrC, tCgC);
}

// =================================================================================================
// Host 端调度引擎与工程化验证
// =================================================================================================
int main()
{
    // ==========================================================================
    // [模块 1] 纯编译期静态断言：SM100 TMEM / UMMA 蓝图验证
    // 证明 AutoPartitioner 能够零开销、正确识别并路由至 Tensor Memory 架构。
    // ==========================================================================
    {
        using Element100 = cutlass::half_t;
        using StrideA100 = cute::Stride<cute::_1, int64_t>;
        using StrideB100 = cute::Stride<int64_t, cute::_1>;
        using StrideC100 = cute::Stride<cute::_1, int64_t>;
        using Tile100    = cute::Shape<cute::_64, cute::_128, cute::_64>;

        using PartA100 = typename autopartition::AutoPartitioner<cutlass::arch::Sm100,
                                                                 cutlass::arch::OpClassTensorOp,
                                                                 Element100,
                                                                 StrideA100,
                                                                 Tile100,
                                                                 128>::RoleA;
        using PartB100 = typename autopartition::AutoPartitioner<cutlass::arch::Sm100,
                                                                 cutlass::arch::OpClassTensorOp,
                                                                 Element100,
                                                                 StrideB100,
                                                                 Tile100,
                                                                 128>::RoleB;
        using PartC100 = typename autopartition::AutoPartitioner<cutlass::arch::Sm100,
                                                                 cutlass::arch::OpClassTensorOp,
                                                                 Element100,
                                                                 StrideC100,
                                                                 Tile100,
                                                                 128>::RoleC;

        // 提取带 TMEM 定向标的 MMA
        using Mma100 = typename PartC100::template TiledMmaFor<PartA100::Major, PartB100::Major>;

        // 静态断言：证明 C 矩阵的 Fragment 类型已经不再是传统的 Array/Register，而是 tmem_frg_base (TMEM 指针句柄)。
        static_assert(cute::is_base_of<cute::UMMA::tmem_frg_base, typename Mma100::FrgTypeC>::value,
                      "SM100 TensorOp accumulator must be backed by Tensor Memory.");
    }

    // ==========================================================================
    // [模块 2] 动态业务层：SM120 FP8 Global GEMM 部署与验证
    // ==========================================================================

    // 设定实际业务张量规模 (M, N, K)
    constexpr int M           = 1024;
    constexpr int N           = 1024;
    constexpr int K           = 1024;
    constexpr int ThreadCount = 256; // 8 Warps (适用于较大 Tile 的吞吐型配置)

    // 类型与排布配置 (FP8 输入，FP32 输出)
    using InputElement  = cutlass::float_e4m3_t;
    using OutputElement = float;
    // 使用行主序 (Row-Major) 配置
    using StrideA = decltype(make_stride(int{}, Int<1>{}));
    using StrideB = decltype(make_stride(int{}, Int<1>{}));
    using StrideC = decltype(make_stride(int{}, Int<1>{}));

    // CTA Tile 规模设定
    using TileShape = cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<64>>;
    using ArchTag   = cutlass::arch::Sm120;
    using OpClass   = cutlass::arch::OpClassTensorOp;

    // 前端路由：获取当前参数下最优的 Policy 布局对象
    using PartA =
        typename autopartition::AutoPartitioner<ArchTag, OpClass, InputElement, StrideA, TileShape, ThreadCount>::RoleA;
    using PartB =
        typename autopartition::AutoPartitioner<ArchTag, OpClass, InputElement, StrideB, TileShape, ThreadCount>::RoleB;
    using PartC =
        typename autopartition::AutoPartitioner<ArchTag, OpClass, InputElement, StrideC, TileShape, ThreadCount>::RoleC;

    // 前端防呆校验
    static_assert(std::is_same<typename PartC::Accumulator, OutputElement>::value,
                  "SM120 FP8 TensorOp example must compute and store FP32 accumulators to prevent overflow.");
    static_assert(cute::cosize_v<typename PartA::SmemLayout> > 0, "PartA SM120 smem layout collapsed.");

    // Host 数据初始化
    std::vector<InputElement>  hA(M * K);
    std::vector<InputElement>  hB(N * K);
    std::vector<OutputElement> hRef(M * N);
    std::vector<OutputElement> hAuto(M * N);

    autopartition::examples::fill_pattern(hA);
    autopartition::examples::fill_pattern(hB);
    // 运行 CPU 端参考运算
    autopartition::examples::reference_gemm(M, N, K, hA.data(), K, 1, hB.data(), K, 1, hRef.data(), N, 1);

    // Device 内存分配与拷贝
    InputElement  *dA = nullptr;
    InputElement  *dB = nullptr;
    OutputElement *dC = nullptr;
    cudaMalloc(&dA, M * K * sizeof(InputElement));
    cudaMalloc(&dB, N * K * sizeof(InputElement));
    cudaMalloc(&dC, M * N * sizeof(OutputElement));
    cudaMemcpy(dA, hA.data(), M * K * sizeof(InputElement), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), N * K * sizeof(InputElement), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, M * N * sizeof(OutputElement));

    // 计算 Grid 维度：向上取整计算需要的 CTA 数量
    constexpr int bM = cute::size<0>(TileShape{});
    constexpr int bN = cute::size<1>(TileShape{});
    dim3          grid_dim((M + bM - 1) / bM, (N + bN - 1) / bN);
    dim3          block_dim(ThreadCount);

    std::cout << "Launching SM120 FP8 Kernel: Grid(" << grid_dim.x << ", " << grid_dim.y << "), Block(" << block_dim.x
              << ")\n";

    // 启动全局 AutoPartition Kernel
    sm120_tensorop_global_autopartition_kernel<PartA,
                                               PartB,
                                               PartC,
                                               InputElement,
                                               OutputElement,
                                               StrideA,
                                               StrideB,
                                               StrideC,
                                               ThreadCount><<<grid_dim, block_dim>>>(
        dA, make_stride(K, Int<1>{}), dB, make_stride(K, Int<1>{}), dC, make_stride(N, Int<1>{}), M, N, K);

    cudaError_t err = cudaDeviceSynchronize();
    if (!autopartition::examples::check_cuda(err, "sm120_tensorop_global_autopartition_kernel")) {
        return 1;
    }

    // 结果拷贝与验证
    cudaMemcpy(hAuto.data(), dC, M * N * sizeof(OutputElement), cudaMemcpyDeviceToHost);

    // 针对 FP8 运算的精度容差
    // 由于 e4m3 只有 3 bit 尾数，FP8 乘法会引入较大的截断误差，容差设置为 1.0e-1f 较为合理
    float max_error = autopartition::examples::max_abs_diff(hAuto, hRef);

    // 性能跑分 (平均 50 次)
    float auto_ms = autopartition::examples::time_launch_ms(
        [&]() {
            sm120_tensorop_global_autopartition_kernel<PartA,
                                                       PartB,
                                                       PartC,
                                                       InputElement,
                                                       OutputElement,
                                                       StrideA,
                                                       StrideB,
                                                       StrideC,
                                                       ThreadCount><<<grid_dim, block_dim>>>(
                dA, make_stride(K, Int<1>{}), dB, make_stride(K, Int<1>{}), dC, make_stride(N, Int<1>{}), M, N, K);
        },
        50);

    std::cout << "Max Absolute Error (FP8 e4m3): " << max_error << "\n";
    std::cout << "Kernel execution time: " << auto_ms << " ms\n";

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return max_error < 1.0e-1f ? 0 : 1;
}
