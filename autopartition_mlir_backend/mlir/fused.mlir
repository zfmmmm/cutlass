module {
  func.func @main(%arg0: tensor<1024x1024xf16>, %arg1: tensor<1024x1024xf16>, %arg2: tensor<1024xf16>) -> tensor<1024x1024xf16> {
    %0 = "autopartition.gemm_bias_gelu"(%arg0, %arg1, %arg2) {K = 1024 : i64, M = 1024 : i64, N = 1024 : i64, alignment_bytes = 16 : i64, backend = "AutoPartitionBackend", dtype = "f16", fallback_reason = "", target_sm = "sm80", thread_count = 128 : i64, tile_k = 64 : i64, tile_m = 64 : i64, tile_n = 64 : i64} : (tensor<1024x1024xf16>, tensor<1024x1024xf16>, tensor<1024xf16>) -> tensor<1024x1024xf16>
    return %0 : tensor<1024x1024xf16>
  }
}

