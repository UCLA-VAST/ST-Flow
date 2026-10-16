add_mlir_library(streamhls_runtime_utils
    SHARED
    StreamHLSRuntimeUtils.cpp
)
target_compile_definitions(streamhls_runtime_utils PRIVATE streamhls_runtime_utils_EXPORTS)