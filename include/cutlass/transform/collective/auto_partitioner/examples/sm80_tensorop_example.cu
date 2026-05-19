#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <iostream>
#include <type_traits>
#include <vector>

#include "auto_partitioner_builder.hpp"
#include "autopartition_example_utils.hpp"

using namespace cute;

// =================================================================================================
// 工程级 Global GEMM Kernel
// =================================================================================================
// 核心改动说明：
// 1. 引入了全局维度 M, N, K，使 Kernel 能够处理动态大小的张量。
// 2. 引入了基于 blockIdx 的 CuTe local_tile 坐标映射机制。
// 3. 构建了沿 K 维度的 Mainloop，实现了最基础的单级流水线 (1-Stage Pipeline)。
template <typename PartA,
          typename PartB,
          typename PartC,
          typename InputElement,
          typename OutputElement,
          typename StrideA,
          typename StrideB,
          typename StrideC,
          int ThreadCount> // 将 ThreadCount 提升为模板参数，提高内部 API 调用的鲁棒性
__global__ void sm80_tensorop_global_autopartition_kernel(InputElement const *ptr_A,
                                                          StrideA             stride_A,
                                                          InputElement const *ptr_B,
                                                          StrideB             stride_B,
                                                          OutputElement      *ptr_C,
                                                          StrideC             stride_C,
                                                          int                 M,
                                                          int                 N,
                                                          int                 K) // 动态传入全局矩阵规模
{
    // 【1. 提取编译期确定的 Tile 尺寸】
    // bM, bN, bK 代表单个 Thread Block 在一次计算周期内处理的子块大小 (例如 64x64x64)。
    using bM = decltype(size<0>(typename PartA::SmemLayout{}));
    using bN = decltype(size<0>(typename PartB::SmemLayout{}));
    using bK = decltype(size<1>(typename PartA::SmemLayout{}));

    // 【2. 构建全局张量视图 (Global Tensor View)】
    // 这里不再使用局部 bM/bK 构造，而是将传入的裸指针直接映射为完整大小的 M*K / N*K / M*N 逻辑张量。
    // 配合传入的步长 (Stride)，CuTe 会在底层代数空间建立完整的寻址公式。
    Tensor gA_full = make_tensor(make_gmem_ptr(ptr_A), make_shape(M, K), stride_A);
    Tensor gB_full = make_tensor(make_gmem_ptr(ptr_B), make_shape(N, K), stride_B);
    Tensor gC_full = make_tensor(make_gmem_ptr(ptr_C), make_shape(M, N), stride_C);

    // 【3. 基于 CTA 坐标的全局切片 (Global Tiling)】
    // local_tile 是 CuTe 中极其强大的算子。它根据你提供的 Tile 形状 (bM, bK) 对全局张量进行网格化。
    // make_coord(blockIdx.x, _) 的意义是：
    // - 在 M 维度，我只要当前 blockIdx.x 对应的那一个分块。
    // - 在 K 维度，我保留所有的分块（用占位符 `_` 表示），因为 K 维度需要在这个 CTA 内部被循环消费。
    // 返回的 gA 形状变为 (bM, bK, num_k_tiles)，其中第3维代表 K 方向需要迭代的次数。
    Tensor gA = local_tile(gA_full, make_tile(bM{}, bK{}), make_coord(blockIdx.x, _));
    Tensor gB = local_tile(gB_full, make_tile(bN{}, bK{}), make_coord(blockIdx.y, _));
    // 输出矩阵 C 不需要沿 K 迭代，所以它的坐标被完全固定为一个 2D 张量 (bM, bN)。
    Tensor gC = local_tile(gC_full, make_tile(bM{}, bN{}), make_coord(blockIdx.x, blockIdx.y));

    // 【4. 物理安全的 Shared Memory 显式分配】
    struct SharedStorage
    {
        // 依托 AutoPartitioner 的 cosize_v 静态推导，无论内部做了多复杂的 Swizzle (地址异或) 或 Padding (填空)，
        // 这里申请的物理内存字节数绝对安全，不会发生 OOM 或越界。
        cute::array_aligned<InputElement, cute::cosize_v<typename PartA::SmemLayout>> smemA;
        cute::array_aligned<InputElement, cute::cosize_v<typename PartB::SmemLayout>> smemB;
    };
    __shared__ SharedStorage smem;

    // 将分配的裸内存和 AutoPartitioner 决策出的 Layout (如 LdMatrix-Friendly Swizzle Layout) 绑定。
    Tensor sA = make_tensor(make_smem_ptr(smem.smemA.data()), typename PartA::SmemLayout{});
    Tensor sB = make_tensor(make_smem_ptr(smem.smemB.data()), typename PartB::SmemLayout{});

    // 【5. Tensor Core MMA 引擎与寄存器分配】
    typename PartC::TiledMma mma;
    auto                     thr_mma = mma.get_thread_slice(threadIdx.x);
    // 根据 TiledMma 的拓扑，在当前线程分配它应负责的局部 C 矩阵碎片 (位于寄存器堆)。
    Tensor tCgC = thr_mma.partition_C(gC);
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);
    clear(tCrC); // 累加器初始化为 0

    // 【6. K 维度流水线 (Mainloop)】
    // size<2>(gA) 获取的正是沿着 K 维度被切分出的 Tile 数量 (即 K / bK)。
    int num_k_tiles = size<2>(gA);

#pragma unroll 1 // 防止编译器过度展开外层主循环导致寄存器溢出
    for (int k_step = 0; k_step < num_k_tiles; ++k_step) {

        // 6.1 提取当前周期的 Global Tile
        // gA(_, _, k_step) 取出当前迭代步对应的二维显存切片 (bM, bK)。
        Tensor gA_k = gA(_, _, k_step);
        Tensor gB_k = gB(_, _, k_step);

        // 6.2 异步数据搬运 (Gmem -> Smem)
        // 使用 cp.async 将数据从显存推入共享内存。
        cooperative_copy<ThreadCount, PartA::GmemToSmemAlignmentBytes * 8>(
            threadIdx.x, gA_k, sA, typename PartA::GmemToSmemCopy{});
        cooperative_copy<ThreadCount, PartB::GmemToSmemAlignmentBytes * 8>(
            threadIdx.x, gB_k, sB, typename PartB::GmemToSmemCopy{});

        // 6.3 异步流水线栅栏与同步
        // 提交当前周期的 DMA 搬运请求。
        cp_async_fence();
        // 阻塞当前 CTA，直到当前提交的这一批次数据完全落入 Shared Memory。
        cp_async_wait<0>();

        // 必须插入全局同步！这是为了防止后续的计算线程跑得太快，
        // 在前一个周期的 DMA 还未完成时，就读取了未初始化的 Shared Memory，
        // 或者防止 DMA 写入覆盖了上一个周期还没被算完的数据。
        __syncthreads();

        // 6.4 协同矩阵乘加 (WGMMA)
        // Tensor Core 在这里轰鸣。由于 C 的累加器 tCrC 被定义在了循环外部，
        // 每次 cooperative_gemm 的结果都会原位累加到 tCrC 的物理寄存器中。
        // SmemToRegCopyOperation 负责触发 LDSM_N 或 LDSM_T 将 sA/sB 拉入寄存器。
        cooperative_gemm(threadIdx.x,
                         mma,
                         sA,
                         sB,
                         tCrC,
                         identity{},
                         identity{},
                         typename PartA::SmemToRegCopyOperation{},
                         typename PartB::SmemToRegCopyOperation{});

        // 6.5 极其关键的写后写 (WAW) 防御屏障
        // 计算完成后，准备进入下一次 k_step。下一次循环的 DMA 拷贝会向同一块 sA/sB 物理地址写入数据。
        // 如果这里不同步，跑得快的线程可能已经开始下个周期的写入，从而毁掉了跑得慢的线程还在使用的当前周期数据。
        __syncthreads();
    }

    // 【7. Epilogue 写回 (Register -> Gmem)】
    // 当整个 K 维度被消费完毕，tCrC 中就包含了最终的矩阵乘结果。
    // 此处的 convert_tensor 将 FP32 的累加结果安全截断并写回 FP16 的全局显存。
    // （工程注：在极高性能要求的场景中，C 的写回通常也需要经过 Shared Memory 来实现 128-bit 向量化访存聚合，
    // AutoPartitioner 的 RoleC 已经为您生成了 Epilogue SmemLayout，本例为保持简洁沿用了标量/寄存器直写。）
    autopartition::examples::convert_tensor(tCgC, tCrC);
}

// =================================================================================================
// Host 端调度与测试
// =================================================================================================
int main()
{
    // 定义生产级的测试矩阵规模（这里假设 M, N, K 都能被 Tile 大小整除，即 1024 都是 64 的倍数）。
    // 在真实生产环境算子中，你需要在此处或 kernel 内部添加 Predication (边界谓词判定) 来处理零头。
    constexpr int M           = 1024;
    constexpr int N           = 1024;
    constexpr int K           = 1024;
    constexpr int ThreadCount = 128; // 4 Warps

    // 类型与排布配置（经典 NN 布局：A 行主序，B 列主序，方便 K 维度的连续读取）
    using InputElement  = cutlass::half_t;
    using OutputElement = float;
    using StrideA       = decltype(make_stride(int{}, Int<1>{}));
    using StrideB       = decltype(make_stride(Int<1>{}, int{}));
    using StrideC       = decltype(make_stride(int{}, Int<1>{}));

    // 设定 CTA 处理的 Tile 大小
    using TileShape = cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<64>>;
    using ArchTag   = cutlass::arch::Sm80;
    using OpClass   = cutlass::arch::OpClassTensorOp;

    // 利用 AutoPartitioner 生成图纸（与上一版相同，完全依赖编译期推断）
    using PartA =
        typename autopartition::AutoPartitioner<ArchTag, OpClass, InputElement, StrideA, TileShape, ThreadCount>::RoleA;
    using PartB =
        typename autopartition::AutoPartitioner<ArchTag, OpClass, InputElement, StrideB, TileShape, ThreadCount>::RoleB;
    using PartC = typename autopartition::
        AutoPartitioner<ArchTag, OpClass, InputElement, StrideC, TileShape, ThreadCount, OutputElement>::RoleC;

    // 分配并初始化 Host 内存
    std::vector<InputElement>  hA(M * K);
    std::vector<InputElement>  hB(N * K);
    std::vector<OutputElement> hRef(M * N);
    std::vector<OutputElement> hAuto(M * N);

    autopartition::examples::fill_pattern(hA);
    autopartition::examples::fill_pattern(hB);
    // 运行 CPU 参考 GEMM 用于后续校验精度
    autopartition::examples::reference_gemm(M, N, K, hA.data(), K, 1, hB.data(), 1, N, hRef.data(), N, 1);

    // 分配 Device 内存
    InputElement  *dA = nullptr;
    InputElement  *dB = nullptr;
    OutputElement *dC = nullptr;
    cudaMalloc(&dA, M * K * sizeof(InputElement));
    cudaMalloc(&dB, N * K * sizeof(InputElement));
    cudaMalloc(&dC, M * N * sizeof(OutputElement));
    cudaMemcpy(dA, hA.data(), M * K * sizeof(InputElement), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), N * K * sizeof(InputElement), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, M * N * sizeof(OutputElement));

    // 【核心改动：计算全局 Grid 大小】
    // 提取 bM 和 bN 来计算需要的 CTA 总数。
    constexpr int bM = cute::size<0>(TileShape{});
    constexpr int bN = cute::size<1>(TileShape{});

    // (M + bM - 1) / bM 是一种防御性写法，确保能向上取整覆盖整个矩阵。
    dim3 grid_dim((M + bM - 1) / bM, (N + bN - 1) / bN);
    dim3 block_dim(ThreadCount);

    std::cout << "Launching Kernel with Grid: (" << grid_dim.x << ", " << grid_dim.y << ") and Block: (" << block_dim.x
              << ")\n";

    // 启动全局 Kernel
    sm80_tensorop_global_autopartition_kernel<PartA,
                                              PartB,
                                              PartC,
                                              InputElement,
                                              OutputElement,
                                              StrideA,
                                              StrideB,
                                              StrideC,
                                              ThreadCount><<<grid_dim, block_dim>>>(
        dA, make_stride(K, Int<1>{}), dB, make_stride(Int<1>{}, N), dC, make_stride(N, Int<1>{}), M, N, K);

    cudaError_t err = cudaDeviceSynchronize();
    if (!autopartition::examples::check_cuda(err, "sm80_tensorop_global_autopartition_kernel")) {
        return 1;
    }

    // 精度校验与性能测试
    cudaMemcpy(hAuto.data(), dC, M * N * sizeof(OutputElement), cudaMemcpyDeviceToHost);
    float max_error = autopartition::examples::max_abs_diff(hAuto, hRef);

    float auto_ms = autopartition::examples::time_launch_ms(
        [&]() {
            sm80_tensorop_global_autopartition_kernel<PartA,
                                                      PartB,
                                                      PartC,
                                                      InputElement,
                                                      OutputElement,
                                                      StrideA,
                                                      StrideB,
                                                      StrideC,
                                                      ThreadCount><<<grid_dim, block_dim>>>(
                dA, make_stride(K, Int<1>{}), dB, make_stride(Int<1>{}, N), dC, make_stride(N, Int<1>{}), M, N, K);
        },
        50);

    std::cout << "Max Absolute Error: " << max_error << "\n";
    std::cout << "Kernel execution time: " << auto_ms << " ms\n";
    std::cout << "Estimated TFLOPS: " << (2.0 * M * N * K) / (auto_ms * 1e-3) / 1e12 << " TFLOPS\n";

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return max_error < 2.0e-2f ? 0 : 1;
}
