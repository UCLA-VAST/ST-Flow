add_mlir_library(MLIR_StreamHLS_Passes
  Passes.cpp
  
  DEPENDS
  MLIR_StreamHLS_TransformsIncGen
  MLIRStreamHLSDataflowIncGen
  MLIRDataflowEnumsIncGen
  MLIRDataflowInterfacesIncGen
  MLIRDataflowAttributesIncGen
  
  LINK_LIBS PUBLIC
  StreamHLS_Dataflow
  MLIR_StreamHLS_Analysis
  MLIR_StreamHLS_Transforms
  MLIRTensorToLinalg
  MLIRPass
)
add_subdirectory(CAPI)
add_subdirectory(IR)
add_subdirectory(Transforms)
add_subdirectory(Translation)
add_subdirectory(Support)
add_subdirectory(Analysis)