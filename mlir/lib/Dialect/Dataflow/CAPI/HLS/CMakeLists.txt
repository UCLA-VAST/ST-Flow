add_mlir_public_c_api_library(MLIRStreamHLSCAPI
  Dataflow.cpp
  Registeration.cpp
  ${PROJECT_SOURCE_DIR}/lib/Dialect/Dataflow/Passes.cpp
  # ${PROJECT_SOURCE_DIR}/lib/Transforms/LoopTransformations.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir-c

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRCAPIIR
  MLIRSupport
  StreamHLS_Dataflow
  MLIR_StreamHLS_Transforms
  MLIR_StreamHLS_Analysis
  MLIR_StreamHLS_Passes
)
