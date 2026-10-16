cmake_minimum_required(VERSION 3.20.0)

# options
set(LLVM_TARGETS_TO_BUILD "host")
set(LLVM_ENABLE_ASSERTIONS ON)
set(CMAKE_BUILD_TYPE DEBUG)
# set(CMAKE_C_COMPILER clang)
# set(CMAKE_CXX_COMPILER clang++)
set(LLVM_ENABLE_LLD ON)
option(LLVM_INCLUDE_TOOLS "Generate build targets for the LLVM tools." ON)
option(LLVM_BUILD_TOOLS "Build the LLVM tools. If OFF, just generate build targets." ON)

set(STREAMHLS_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(STREAMHLS_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

project(streamhls LANGUAGES CXX C)

# ================= MLIR/LLVM Configurations =================
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

# sets binary and lib dirs
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

# lit configuration for testing
set(LIT_ARGS_DEFAULT "-sv")
if (MSVC_IDE OR XCODE)
  set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
endif()
set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

message(STATUS "Using Python binding")
include(MLIRDetectPythonEnv)
mlir_configure_python_dev_packages()

# add_subdirectory(tmp)
#add_subdirectory(include)
add_subdirectory(include/streamhls)
add_subdirectory(lib)
# add_subdirectory(tools)
# add_subdirectory(test)
