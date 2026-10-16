cmake_minimum_required(VERSION 3.20.0)

# options
set(LLVM_TARGETS_TO_BUILD "host")
set(LLVM_ENABLE_ASSERTIONS ON)
# Release unless the caller asks otherwise (pip: -C cmake.build-type=Debug). This used
# to be hardcoded to DEBUG, which made every install a ~4x larger, slower debug build.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# Link with lld when it is installed (faster), otherwise with the system linker. The
# prebuilt LLVM does not ship ld.lld, and forcing it fails at configure time.
find_program(STFLOW_LD_LLD ld.lld)
if(STFLOW_LD_LLD)
  set(LLVM_ENABLE_LLD ON)
else()
  set(LLVM_ENABLE_LLD OFF)
endif()

# ============== Locate the prebuilt LLVM/MLIR and its toolchain ==============
# This runs BEFORE project() on purpose: CMAKE_<LANG>_COMPILER has no effect once
# the languages are enabled. Search order, first hit wins:
#
#   1. -DMLIR_DIR / -DLLVM_DIR, or the same variables in the environment
#   2. $LLVM_BUILD_DIR or $STFLOW_LLVM_DIR (an LLVM install/build prefix)
#   3. ~/opt/llvm-stflow (where the README extracts the prebuilt release)
#   4. the download cache, ${STFLOW_LLVM_CACHE_DIR}/llvm-stflow
#   5. download the prebuilt release into that cache (STFLOW_FETCH_LLVM=ON)
#
# `source env.sh` sets (1); a bare `pip install .` relies on (3)-(5).
option(STFLOW_FETCH_LLVM "Download the prebuilt LLVM/MLIR when none is found" ON)
set(STFLOW_LLVM_URL
    "https://zenodo.org/records/21881993/files/llvm-stflow-ubuntu-latest.tar.gz"
    CACHE STRING "Prebuilt LLVM/MLIR tarball (root directory: llvm-stflow/)")
set(STFLOW_LLVM_MD5 "4ba7221c1a2a1b96803a31cdff5329b1"
    CACHE STRING "MD5 of STFLOW_LLVM_URL, as published by Zenodo")
if(DEFINED ENV{XDG_CACHE_HOME})
  set(_stflow_cache_default "$ENV{XDG_CACHE_HOME}/stflow")
else()
  set(_stflow_cache_default "$ENV{HOME}/.cache/stflow")
endif()
set(STFLOW_LLVM_CACHE_DIR "${_stflow_cache_default}"
    CACHE PATH "Where a downloaded LLVM/MLIR is kept between builds")

foreach(_var MLIR_DIR LLVM_DIR)
  if(NOT ${_var} AND DEFINED ENV{${_var}})
    set(${_var} "$ENV{${_var}}" CACHE PATH "")
  endif()
endforeach()

if(LLVM_DIR)
  get_filename_component(STREAMHLS_LLVM_ROOT "${LLVM_DIR}/../../.." ABSOLUTE)
else()
  set(_stflow_candidates "$ENV{LLVM_BUILD_DIR}" "$ENV{STFLOW_LLVM_DIR}"
      "$ENV{HOME}/opt/llvm-stflow" "${STFLOW_LLVM_CACHE_DIR}/llvm-stflow")
  foreach(_root IN LISTS _stflow_candidates)
    if(_root AND EXISTS "${_root}/lib/cmake/mlir/MLIRConfig.cmake")
      set(STREAMHLS_LLVM_ROOT "${_root}")
      break()
    endif()
  endforeach()

  if(NOT STREAMHLS_LLVM_ROOT AND STFLOW_FETCH_LLVM)
    set(_tarball "${STFLOW_LLVM_CACHE_DIR}/llvm-stflow.tar.gz")
    message(STATUS "No LLVM/MLIR found; downloading the prebuilt release (1.3 GB) "
                   "into ${STFLOW_LLVM_CACHE_DIR}")
    file(DOWNLOAD "${STFLOW_LLVM_URL}" "${_tarball}"
         EXPECTED_HASH MD5=${STFLOW_LLVM_MD5} SHOW_PROGRESS STATUS _dl_status)
    list(GET _dl_status 0 _dl_code)
    if(NOT _dl_code EQUAL 0)
      message(FATAL_ERROR "Downloading ${STFLOW_LLVM_URL} failed: ${_dl_status}")
    endif()
    file(ARCHIVE_EXTRACT INPUT "${_tarball}" DESTINATION "${STFLOW_LLVM_CACHE_DIR}")
    file(REMOVE "${_tarball}")
    set(STREAMHLS_LLVM_ROOT "${STFLOW_LLVM_CACHE_DIR}/llvm-stflow")
  endif()

  if(STREAMHLS_LLVM_ROOT)
    set(LLVM_DIR "${STREAMHLS_LLVM_ROOT}/lib/cmake/llvm" CACHE PATH "" FORCE)
  endif()
endif()
if(STREAMHLS_LLVM_ROOT AND NOT MLIR_DIR)
  set(MLIR_DIR "${STREAMHLS_LLVM_ROOT}/lib/cmake/mlir" CACHE PATH "" FORCE)
endif()
if(NOT MLIR_DIR OR NOT LLVM_DIR)
  message(FATAL_ERROR
    "No LLVM/MLIR 20 found. Set STFLOW_LLVM_DIR to a prebuilt LLVM/MLIR, enable "
    "STFLOW_FETCH_LLVM, or build one from extern/llvm-project with "
    "scripts/build_llvm.sh (see the installation guide).")
endif()

# The prebuilt release is linked against glibc 2.34. On an older system its tools
# cannot run, and every later step would fail with an obscure linker error, so check
# once, here, and point at the source build instead.
if(STREAMHLS_LLVM_ROOT AND EXISTS "${STREAMHLS_LLVM_ROOT}/bin/llvm-config")
  execute_process(COMMAND "${STREAMHLS_LLVM_ROOT}/bin/llvm-config" --version
                  RESULT_VARIABLE _llvm_rc OUTPUT_QUIET ERROR_VARIABLE _llvm_err)
  if(NOT _llvm_rc EQUAL 0)
    message(FATAL_ERROR
      "The LLVM/MLIR at ${STREAMHLS_LLVM_ROOT} does not run on this system:\n"
      "${_llvm_err}\nThe prebuilt release needs glibc >= 2.34. Build LLVM/MLIR from "
      "source with scripts/build_llvm.sh and set STFLOW_LLVM_DIR to the result.")
  endif()
endif()

# Compile with the clang that ships alongside that LLVM, so the compiler and the MLIR
# libraries come from one toolchain instead of whatever PATH resolves. Pass
# -DCMAKE_CXX_COMPILER=... to override.
#
# That clang uses the system's GCC installation for the C++ standard library, and on
# some systems it picks a GCC whose libstdc++ development files are not installed
# ("cannot find -lstdc++"). So only use it if it can link a test program; otherwise
# fall back to the default compiler.
if(NOT CMAKE_CXX_COMPILER AND STREAMHLS_LLVM_ROOT
   AND EXISTS "${STREAMHLS_LLVM_ROOT}/bin/clang++")
  set(_probe_dir "${CMAKE_BINARY_DIR}/stflow-clang-probe")
  file(WRITE "${_probe_dir}/probe.cpp" "#include <string>\nint main() { return std::string(\"ok\").size() != 2; }\n")
  execute_process(
    COMMAND "${STREAMHLS_LLVM_ROOT}/bin/clang++" -std=c++17 probe.cpp -o probe
    WORKING_DIRECTORY "${_probe_dir}"
    RESULT_VARIABLE _probe_rc OUTPUT_QUIET ERROR_VARIABLE _probe_err)
  if(_probe_rc EQUAL 0)
    set(CMAKE_CXX_COMPILER "${STREAMHLS_LLVM_ROOT}/bin/clang++"
        CACHE FILEPATH "C++ compiler from the prebuilt LLVM")
    set(CMAKE_C_COMPILER "${STREAMHLS_LLVM_ROOT}/bin/clang"
        CACHE FILEPATH "C compiler from the prebuilt LLVM")
    message(STATUS "Toolchain: clang from ${STREAMHLS_LLVM_ROOT}")
  else()
    message(STATUS "Toolchain: the clang in ${STREAMHLS_LLVM_ROOT} cannot link here "
                   "(${_probe_err}); using the default compiler instead")
  endif()
endif()
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
