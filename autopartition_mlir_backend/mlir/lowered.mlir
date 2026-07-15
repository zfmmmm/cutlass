module {
  func.func @main(%arg0: tensor<1024x1024xf16>, %arg1: tensor<1024x1024xf16>, %arg2: tensor<1024xf16>) -> tensor<1024x1024xf16> {
    %0 = call @autopartition_fused_gemm_bias_gelu(%arg0, %arg1, %arg2) {autopartition.backend = "AutoPartitionBackend", autopartition.fallback_reason = ""} : (tensor<1024x1024xf16>, tensor<1024x1024xf16>, tensor<1024xf16>) -> tensor<1024x1024xf16>
    return %0 : tensor<1024x1024xf16>
  }
  func.func private @autopartition_fused_gemm_bias_gelu(tensor<1024x1024xf16>, tensor<1024x1024xf16>, tensor<1024xf16>) -> tensor<1024x1024xf16> attributes {autopartition.backend = "AutoPartitionBackend", autopartition.fallback_reason = ""}
}

