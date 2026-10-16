# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

include(AddMLIRPython)

# The directory at which the Python import tree begins.
# See documentation for `declare_mlir_python_sources`'s ROOT_DIR
# argument.
# set(ALLO_MLIR_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Allo/Bindings/allo")
set(STREAM_HLS_MLIR_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Dataflow/Bindings/streamhls")

# for both streamhls and allo
set(OUT_MLIR_PYTHON_PACKAGES_DIR "${PROJECT_BINARY_DIR}/tools/streamhls")

# set(ALLO_MLIR_PYTHON_SOURCE_DIR "${MLIR_MAIN_SRC_DIR}/lib/Dialect/Allo/Bindings")
# set(ALLO_PYTHON_SOURCE_DIR "${PROJECT_SOURCE_DIR}/lib/Dialect/Allo/Bindings")

set(STREAM_HLS_MLIR_PYTHON_SOURCE_DIR "${MLIR_MAIN_SRC_DIR}/lib/Dialect/Dataflow/Bindings")
set(STREAM_HLS_PYTHON_SOURCE_DIR "${PROJECT_SOURCE_DIR}/lib/Dialect/Dataflow/Bindings")

# include_directories(${ALLO_MLIR_PYTHON_SOURCE_DIR})
include_directories(${STREAM_HLS_MLIR_PYTHON_SOURCE_DIR})

# We vendor our own MLIR instance in the `stflow._mlir` namespace.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=stflow._mlir.")

################################################################################
# Sources
################################################################################

# declare_mlir_python_sources(AlloMLIRPythonSources)
# declare_mlir_python_sources(AlloMLIRPythonExtensions)
declare_mlir_python_sources(StreamHLSMLIRPythonSources)
declare_mlir_python_sources(StreamHLSMLIRPythonExtensions)

# declare_mlir_python_sources(StreamHLSMLIRPythonSources.AlloDialect
#   ROOT_DIR "${ALLO_MLIR_PYTHON_ROOT_DIR}"
#   ADD_TO_PARENT StreamHLSMLIRPythonSources
# )

# declare_mlir_dialect_python_bindings(
#   ADD_TO_PARENT StreamHLSMLIRPythonSources.AlloDialect
#   ROOT_DIR "${ALLO_MLIR_PYTHON_ROOT_DIR}"
#   TD_FILE dialects/AlloBinding.td
#   SOURCES
#     dialects/allo.py
#     exceptions.py
#     __init__.py
#   DIALECT_NAME allo
# )

declare_mlir_python_sources(StreamHLSMLIRPythonSources.HLSDialect
  ROOT_DIR "${STREAM_HLS_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT StreamHLSMLIRPythonSources
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT StreamHLSMLIRPythonSources.HLSDialect
  ROOT_DIR "${STREAM_HLS_MLIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/StreamHLSBinding.td
  SOURCES
    dialects/streamhls.py
    # exceptions.py
    # __init__.py
  DIALECT_NAME streamhls
)

################################################################################
# Extensions
################################################################################

# declare_mlir_python_extension(StreamHLSMLIRPythonExtensions.AlloMain.Pybind
#   MODULE_NAME _allo
#   ADD_TO_PARENT StreamHLSMLIRPythonExtensions
#   ROOT_DIR "/"
#   SOURCES
#     ${ALLO_PYTHON_SOURCE_DIR}/AlloModule.cpp
#     ${ALLO_PYTHON_SOURCE_DIR}/AlloTypes.cpp
#     ${ALLO_PYTHON_SOURCE_DIR}/AlloAttributes.cpp
#   EMBED_CAPI_LINK_LIBS
#     MLIRCAPIIR
#     MLIRCAPIDebug
#     MLIRAlloCAPI
#     MLIRAlloCAPIEmitHLSCpp
#   PRIVATE_LINK_LIBS
#     MLIRPass
#     MLIRAlloPasses
#     LLVMSupport
# )

declare_mlir_python_extension(StreamHLSMLIRPythonExtensions.StreamHLSMain.Pybind
  MODULE_NAME _streamhls
  ADD_TO_PARENT StreamHLSMLIRPythonExtensions
  ROOT_DIR "/"
  SOURCES
    ${STREAM_HLS_PYTHON_SOURCE_DIR}/StreamHLSModule.cpp
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPIDebug
    MLIRStreamHLSCAPI
  PRIVATE_LINK_LIBS
    MLIRPass
    MLIR_StreamHLS_Transforms
    MLIR_StreamHLS_Passes
    # MLIR_StreamHLS_EmitHLSCpp
    MLIRStreamHLSCAPIEmitHLSCpp
    LLVMSupport
)

################################################################################
# Generate packages and shared library
# Downstreams typically will not use these, but they are useful for local
# testing.
################################################################################

set(_source_components
  StreamHLSMLIRPythonSources
  StreamHLSMLIRPythonExtensions
  MLIRPythonSources
  MLIRPythonExtension.RegisterEverything
)

add_mlir_python_common_capi_library(StreamHLSMLIRAggregateCAPI
  INSTALL_COMPONENT StreamHLSMLIRPythonModules
  INSTALL_DESTINATION _mlir
  OUTPUT_DIRECTORY "${OUT_MLIR_PYTHON_PACKAGES_DIR}/_mlir"
  RELATIVE_INSTALL_ROOT "../.."
  DECLARED_HEADERS
    MLIRPythonCAPI.HeaderSources
  DECLARED_SOURCES
    ${_source_components}
)

add_mlir_python_modules(StreamHLSMLIRPythonModules
  ROOT_PREFIX "${OUT_MLIR_PYTHON_PACKAGES_DIR}/_mlir"
  INSTALL_PREFIX "_mlir"
  DECLARED_SOURCES ${_source_components}
  COMMON_CAPI_LINK_LIBS
    StreamHLSMLIRAggregateCAPI
)

add_subdirectory(Dataflow)

