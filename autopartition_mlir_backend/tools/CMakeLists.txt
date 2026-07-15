cmake_minimum_required(VERSION 3.20)
project(autopartition_opt LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard")
find_package(MLIR REQUIRED CONFIG)

message(STATUS "MLIR_DIR=${MLIR_DIR}")
message(STATUS "LLVM_DIR=${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

set(AUTOPARTITION_OPT_LIBS
    MLIROptLib
    MLIRParser
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRArithDialect
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRMathDialect
    MLIRTensorDialect)

add_llvm_executable(autopartition-opt autopartition-opt.cpp)
llvm_update_compile_flags(autopartition-opt)
target_link_libraries(autopartition-opt PRIVATE ${AUTOPARTITION_OPT_LIBS})
