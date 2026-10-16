add_mlir_library(MLIR_StreamHLS_EmitHLSCpp
  Utils.cpp
  EmitHLS.cpp
  EmitTAPA.cpp
  EmitVivadoHLS.cpp

  LINK_LIBS PUBLIC
  StreamHLS_Dataflow
  MLIR_StreamHLS_Support
  MLIRMemRefDialect
  MLIRAnalysis
)