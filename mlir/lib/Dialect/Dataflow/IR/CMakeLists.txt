add_mlir_dialect_library(StreamHLS_Dataflow
  Dataflow.cpp

  DEPENDS
  MLIRStreamHLSDataflowIncGen
  MLIRDataflowAttributesIncGen
  MLIRDataflowInterfacesIncGen
  MLIRDataflowEnumsIncGen
  )
