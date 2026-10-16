file(GLOB globbed *.cpp)

add_mlir_library(MLIR_StreamHLS_Transforms
  ${globbed}
  ConverterInsertion.cpp
  MergeKernels/Utils.cpp
  MergeKernels/Candidates.cpp
  MergeKernels/Equations.cpp
  MergeKernels/Merging.cpp
  MergeKernels/Solution.cpp
  MergeKernels/Transform.cpp
  MergeKernels/Pass.cpp

  DEPENDS
  MLIR_StreamHLS_TransformsIncGen
  MLIRStreamHLSDataflowIncGen
  MLIRDataflowEnumsIncGen
  MLIRDataflowInterfacesIncGen
  MLIRDataflowAttributesIncGen
  
  LINK_LIBS PUBLIC
  StreamHLS_Dataflow
  MLIRPass
  )
