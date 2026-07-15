#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%arg0: tensor<1024x1024xf16>, %arg1: tensor<1024x1024xf16>, %arg2: tensor<1024xf16>) -> tensor<1024x1024xf16> {
    %cst = arith.constant 1.000000e+00 : f16
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant -1.000000e+00 : f32
    %cst_2 = arith.constant -0.0142647391 : f32
    %cst_3 = arith.constant -0.00737332925 : f32
    %cst_4 = arith.constant -0.00168282702 : f32
    %cst_5 = arith.constant -2.13374049E-4 : f32
    %cst_6 = arith.constant -1.45660715E-5 : f32
    %cst_7 = arith.constant -0.0160960332 : f32
    %cst_8 = arith.constant -2.954600e-03 : f32
    %cst_9 = arith.constant -7.34990637E-4 : f32
    %cst_10 = arith.constant -5.69250624E-5 : f32
    %cst_11 = arith.constant -2.10102394E-6 : f32
    %cst_12 = arith.constant 2.77068146E-8 : f32
    %cst_13 = arith.constant -2.72614237E-10 : f32
    %cst_14 = arith.constant 4.000000e+00 : f32
    %cst_15 = arith.constant -4.000000e+00 : f32
    %cst_16 = arith.constant 2.000000e+00 : f16
    %cst_17 = arith.constant 5.000000e-01 : f16
    %cst_18 = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1024x1024xf16>
    %1 = linalg.fill ins(%cst_18 : f16) outs(%0 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<1024x1024xf16>, tensor<1024x1024xf16>) outs(%1 : tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
    %3 = tensor.empty() : tensor<1024x1024xf16>
    %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<1024xf16>) outs(%3 : tensor<1024x1024xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1024x1024xf16>
    %5 = tensor.empty() : tensor<1024x1024xf16>
    %6 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%2, %4 : tensor<1024x1024xf16>, tensor<1024x1024xf16>) outs(%5 : tensor<1024x1024xf16>) {
    ^bb0(%in: f16, %in_19: f16, %out: f16):
      %71 = arith.addf %in, %in_19 : f16
      linalg.yield %71 : f16
    } -> tensor<1024x1024xf16>
    %7 = tensor.empty() : tensor<1024x1024xf16>
    %8 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<1024x1024xf16>) outs(%7 : tensor<1024x1024xf16>) {
    ^bb0(%in: f16, %out: f16):
      %71 = arith.mulf %in, %cst_17 : f16
      linalg.yield %71 : f16
    } -> tensor<1024x1024xf16>
    %9 = tensor.empty() : tensor<1024x1024xf16>
    %10 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%9 : tensor<1024x1024xf16>) {
    ^bb0(%out: f16):
      %71 = math.rsqrt %cst_16 : f16
      linalg.yield %71 : f16
    } -> tensor<1024x1024xf16>
    %11 = tensor.empty() : tensor<1024x1024xf16>
    %12 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%6, %10 : tensor<1024x1024xf16>, tensor<1024x1024xf16>) outs(%11 : tensor<1024x1024xf16>) {
    ^bb0(%in: f16, %in_19: f16, %out: f16):
      %71 = arith.mulf %in, %in_19 : f16
      linalg.yield %71 : f16
    } -> tensor<1024x1024xf16>
    %13 = tensor.empty() : tensor<1024x1024xf32>
    %14 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%12 : tensor<1024x1024xf16>) outs(%13 : tensor<1024x1024xf32>) {
    ^bb0(%in: f16, %out: f32):
      %71 = arith.extf %in : f16 to f32
      linalg.yield %71 : f32
    } -> tensor<1024x1024xf32>
    %15 = tensor.empty() : tensor<1024x1024xf32>
    %16 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%14 : tensor<1024x1024xf32>) outs(%15 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %71 = arith.maximumf %in, %cst_15 : f32
      %72 = arith.minimumf %71, %cst_14 : f32
      linalg.yield %72 : f32
    } -> tensor<1024x1024xf32>
    %17 = tensor.empty() : tensor<1024x1024xf32>
    %18 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%16, %16 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%17 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %in_19: f32, %out: f32):
      %71 = arith.mulf %in, %in_19 : f32
      linalg.yield %71 : f32
    } -> tensor<1024x1024xf32>
    %19 = tensor.empty() : tensor<1024x1024xf32>
    %20 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%18 : tensor<1024x1024xf32>) outs(%19 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %71 = arith.mulf %in, %cst_13 : f32
      linalg.yield %71 : f32
    } -> tensor<1024x1024xf32>
    %21 = tensor.empty() : tensor<1024x1024xf32>
    %22 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%20 : tensor<1024x1024xf32>) outs(%21 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %71 = arith.addf %in, %cst_12 : f32
      linalg.yield %71 : f32
    } -> tensor<1024x1024xf32>
    %23 = tensor.empty() : tensor<1024x1024xf32>
    %24 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%22, %18 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%23 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %in_19: f32, %out: f32):
      %71 = arith.mulf %in, %in_19 : f32
      linalg.yield %71 : f32
    } -> tensor<1024x1024xf32>
    %25 = tensor.empty() : tensor<1024x1024xf32>
    %26 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%24 : tensor<1024x1024xf32>) outs(%25 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %71 = arith.addf %in, %cst_11 : f32
      linalg.yield %71 : f32
    } -> tensor<1024x1024xf32>
    %27 = tensor.empty() : tensor<1024x1024xf32>
    %28 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%26, %18 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%27 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %in_19: f32, %out: f32):
      %71 = arith.mulf %in, %in_19 : f32
      linalg.yield %71 : f32
    } -> tensor<1024x1024xf32>
    %29 = tensor.empty() : tensor<1024x1024xf32>
    %30 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%28 : tensor<1024x1024xf32>) outs(%29 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %71 = arith.addf %in, %cst_10 : f32
      linalg.yield %71 : f32
    } -> tensor<1024x1024xf32>
    %31 = tensor.empty() : tensor<1024x1024xf32>
    %32 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%30, %18 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%31 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %in_19: f32, %out: f32):
      %71 = arith.mulf %in, %in_19 : f32
      linalg.yield %71 : f32
    } -> tensor<1024x1024xf32>
    %33 = tensor.empty() : tensor<1024x1024xf32>
    %34 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%32 : tensor<1024x1024xf32>) outs(%33 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %71 = arith.addf %in, %cst_9 : f32
      linalg.yield %71 : f32
    } -> tensor<1024x1024xf32>
    %35 = tensor.empty() : tensor<1024x1024xf32>
    %36 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%34, %18 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%35 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %in_19: f32, %out: f32):
      %71 = arith.mulf %in, %in_19 : f32
      linalg.yield %71 : f32
    } -> tensor<1024x1024xf32>
    %37 = tensor.empty() : tensor<1024x1024xf32>
    %38 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%36 : tensor<1024x1024xf32>) outs(%37 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %71 = arith.addf %in, %cst_8 : f32
      linalg.yield %71 : f32
    } -> tensor<1024x1024xf32>
    %39 = tensor.empty() : tensor<1024x1024xf32>
    %40 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%38, %18 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%39 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %in_19: f32, %out: f32):
      %71 = arith.mulf %in, %in_19 : f32
      linalg.yield %71 : f32
    } -> tensor<1024x1024xf32>
    %41 = tensor.empty() : tensor<1024x1024xf32>
    %42 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%40 : tensor<1024x1024xf32>) outs(%41 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %71 = arith.addf %in, %cst_7 : f32
      linalg.yield %71 : f32
    } -> tensor<1024x1024xf32>
    %43 = tensor.empty() : tensor<1024x1024xf32>
    %44 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%18 : tensor<1024x1024xf32>) outs(%43 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %71 = arith.mulf %in, %cst_6 : f32
      linalg.yield %71 : f32
    } -> tensor<1024x1024xf32>
    %45 = tensor.empty() : tensor<1024x1024xf32>
    %46 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%44 : tensor<1024x1024xf32>) outs(%45 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %71 = arith.addf %in, %cst_5 : f32
      linalg.yield %71 : f32
    } -> tensor<1024x1024xf32>
    %47 = tensor.empty() : tensor<1024x1024xf32>
    %48 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%46, %18 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%47 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %in_19: f32, %out: f32):
      %71 = arith.mulf %in, %in_19 : f32
      linalg.yield %71 : f32
    } -> tensor<1024x1024xf32>
    %49 = tensor.empty() : tensor<1024x1024xf32>
    %50 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%48 : tensor<1024x1024xf32>) outs(%49 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %71 = arith.addf %in, %cst_4 : f32
      linalg.yield %71 : f32
    } -> tensor<1024x1024xf32>
    %51 = tensor.empty() : tensor<1024x1024xf32>
    %52 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%50, %18 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%51 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %in_19: f32, %out: f32):
      %71 = arith.mulf %in, %in_19 : f32
      linalg.yield %71 : f32
    } -> tensor<1024x1024xf32>
    %53 = tensor.empty() : tensor<1024x1024xf32>
    %54 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%52 : tensor<1024x1024xf32>) outs(%53 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %71 = arith.addf %in, %cst_3 : f32
      linalg.yield %71 : f32
    } -> tensor<1024x1024xf32>
    %55 = tensor.empty() : tensor<1024x1024xf32>
    %56 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%54, %18 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%55 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %in_19: f32, %out: f32):
      %71 = arith.mulf %in, %in_19 : f32
      linalg.yield %71 : f32
    } -> tensor<1024x1024xf32>
    %57 = tensor.empty() : tensor<1024x1024xf32>
    %58 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%56 : tensor<1024x1024xf32>) outs(%57 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %71 = arith.addf %in, %cst_2 : f32
      linalg.yield %71 : f32
    } -> tensor<1024x1024xf32>
    %59 = tensor.empty() : tensor<1024x1024xf32>
    %60 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%16, %42 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%59 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %in_19: f32, %out: f32):
      %71 = arith.mulf %in, %in_19 : f32
      linalg.yield %71 : f32
    } -> tensor<1024x1024xf32>
    %61 = tensor.empty() : tensor<1024x1024xf32>
    %62 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%60, %58 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%61 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %in_19: f32, %out: f32):
      %71 = arith.divf %in, %in_19 : f32
      linalg.yield %71 : f32
    } -> tensor<1024x1024xf32>
    %63 = tensor.empty() : tensor<1024x1024xf32>
    %64 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%62 : tensor<1024x1024xf32>) outs(%63 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %71 = arith.maximumf %in, %cst_1 : f32
      %72 = arith.minimumf %71, %cst_0 : f32
      linalg.yield %72 : f32
    } -> tensor<1024x1024xf32>
    %65 = tensor.empty() : tensor<1024x1024xf16>
    %66 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%64 : tensor<1024x1024xf32>) outs(%65 : tensor<1024x1024xf16>) {
    ^bb0(%in: f32, %out: f16):
      %71 = arith.truncf %in : f32 to f16
      linalg.yield %71 : f16
    } -> tensor<1024x1024xf16>
    %67 = tensor.empty() : tensor<1024x1024xf16>
    %68 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%66 : tensor<1024x1024xf16>) outs(%67 : tensor<1024x1024xf16>) {
    ^bb0(%in: f16, %out: f16):
      %71 = arith.addf %in, %cst : f16
      linalg.yield %71 : f16
    } -> tensor<1024x1024xf16>
    %69 = tensor.empty() : tensor<1024x1024xf16>
    %70 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%68, %8 : tensor<1024x1024xf16>, tensor<1024x1024xf16>) outs(%69 : tensor<1024x1024xf16>) {
    ^bb0(%in: f16, %in_19: f16, %out: f16):
      %71 = arith.mulf %in, %in_19 : f16
      linalg.yield %71 : f16
    } -> tensor<1024x1024xf16>
    return %70 : tensor<1024x1024xf16>
  }
}

