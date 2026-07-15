module {
  func.func @main(%arg0: tensor<1024x1024xf16>, %arg1: tensor<1024x1024xf16>, %arg2: tensor<1024xf16>) -> tensor<1024x1024xf16> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<1024x1024xf16>
    %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<1024x1024xf16>
    %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<1024x1024xf16>
    %cst_2 = stablehlo.constant dense<-4.000000e+00> : tensor<1024x1024xf32>
    %cst_3 = stablehlo.constant dense<4.000000e+00> : tensor<1024x1024xf32>
    %cst_4 = stablehlo.constant dense<-2.72614237E-10> : tensor<1024x1024xf32>
    %cst_5 = stablehlo.constant dense<2.77068146E-8> : tensor<1024x1024xf32>
    %cst_6 = stablehlo.constant dense<-2.10102394E-6> : tensor<1024x1024xf32>
    %cst_7 = stablehlo.constant dense<-5.69250624E-5> : tensor<1024x1024xf32>
    %cst_8 = stablehlo.constant dense<-7.34990637E-4> : tensor<1024x1024xf32>
    %cst_9 = stablehlo.constant dense<-2.954600e-03> : tensor<1024x1024xf32>
    %cst_10 = stablehlo.constant dense<-0.0160960332> : tensor<1024x1024xf32>
    %cst_11 = stablehlo.constant dense<-1.45660715E-5> : tensor<1024x1024xf32>
    %cst_12 = stablehlo.constant dense<-2.13374049E-4> : tensor<1024x1024xf32>
    %cst_13 = stablehlo.constant dense<-0.00168282702> : tensor<1024x1024xf32>
    %cst_14 = stablehlo.constant dense<-0.00737332925> : tensor<1024x1024xf32>
    %cst_15 = stablehlo.constant dense<-0.0142647391> : tensor<1024x1024xf32>
    %cst_16 = stablehlo.constant dense<-1.000000e+00> : tensor<1024x1024xf32>
    %cst_17 = stablehlo.constant dense<1.000000e+00> : tensor<1024x1024xf32>
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1024x1024xf16>, tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<1024xf16>) -> tensor<1024x1024xf16>
    %2 = stablehlo.add %0, %1 : tensor<1024x1024xf16>
    %3 = stablehlo.multiply %2, %cst_1 : tensor<1024x1024xf16>
    %4 = stablehlo.rsqrt %cst_0 : tensor<1024x1024xf16>
    %5 = stablehlo.multiply %2, %4 : tensor<1024x1024xf16>
    %6 = stablehlo.convert %5 : (tensor<1024x1024xf16>) -> tensor<1024x1024xf32>
    %7 = stablehlo.clamp %cst_2, %6, %cst_3 : tensor<1024x1024xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<1024x1024xf32>
    %9 = stablehlo.multiply %cst_4, %8 : tensor<1024x1024xf32>
    %10 = stablehlo.add %9, %cst_5 : tensor<1024x1024xf32>
    %11 = stablehlo.multiply %10, %8 : tensor<1024x1024xf32>
    %12 = stablehlo.add %11, %cst_6 : tensor<1024x1024xf32>
    %13 = stablehlo.multiply %12, %8 : tensor<1024x1024xf32>
    %14 = stablehlo.add %13, %cst_7 : tensor<1024x1024xf32>
    %15 = stablehlo.multiply %14, %8 : tensor<1024x1024xf32>
    %16 = stablehlo.add %15, %cst_8 : tensor<1024x1024xf32>
    %17 = stablehlo.multiply %16, %8 : tensor<1024x1024xf32>
    %18 = stablehlo.add %17, %cst_9 : tensor<1024x1024xf32>
    %19 = stablehlo.multiply %18, %8 : tensor<1024x1024xf32>
    %20 = stablehlo.add %19, %cst_10 : tensor<1024x1024xf32>
    %21 = stablehlo.multiply %cst_11, %8 : tensor<1024x1024xf32>
    %22 = stablehlo.add %21, %cst_12 : tensor<1024x1024xf32>
    %23 = stablehlo.multiply %22, %8 : tensor<1024x1024xf32>
    %24 = stablehlo.add %23, %cst_13 : tensor<1024x1024xf32>
    %25 = stablehlo.multiply %24, %8 : tensor<1024x1024xf32>
    %26 = stablehlo.add %25, %cst_14 : tensor<1024x1024xf32>
    %27 = stablehlo.multiply %26, %8 : tensor<1024x1024xf32>
    %28 = stablehlo.add %27, %cst_15 : tensor<1024x1024xf32>
    %29 = stablehlo.multiply %7, %20 : tensor<1024x1024xf32>
    %30 = stablehlo.divide %29, %28 : tensor<1024x1024xf32>
    %31 = stablehlo.clamp %cst_16, %30, %cst_17 : tensor<1024x1024xf32>
    %32 = stablehlo.convert %31 : (tensor<1024x1024xf32>) -> tensor<1024x1024xf16>
    %33 = stablehlo.add %32, %cst : tensor<1024x1024xf16>
    %34 = stablehlo.multiply %33, %3 : tensor<1024x1024xf16>
    return %34 : tensor<1024x1024xf16>
  }
}

