file(GLOB globbed *.cpp)

add_mlir_library(MLIR_StreamHLS_Analysis
  ${globbed}
  
  DEPENDS
  MLIR_StreamHLS_TransformsIncGen
  MLIRStreamHLSDataflowIncGen
  MLIRDataflowEnumsIncGen
  MLIRDataflowInterfacesIncGen
  MLIRDataflowAttributesIncGen

  LINK_LIBS PUBLIC
  StreamHLS_Dataflow
  MLIR_StreamHLS_Support
  MLIRPass
  )
