# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

add_mlir_library(MLIR_StreamHLS_Support
  AffineAnalysis.cpp
  AffineMemAccess.cpp
  DFG.cpp
  LoopCombinedModel.cpp
  LoopPermutationModel.cpp
  LoopUnrollingModel.cpp
  PerformanceModel.cpp
  Equations.cpp
  Utils.cpp
  DEPENDS
  MLIRStreamHLSDataflowIncGen
  MLIRDataflowEnumsIncGen
  MLIRDataflowInterfacesIncGen
  MLIRDataflowAttributesIncGen

  LINK_LIBS PUBLIC
  StreamHLS_Dataflow
  MLIRAffineUtils
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSPIRVDialect
  MLIRTranslateLib
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRAnalysis
  MLIRAffineAnalysis
)

# target_include_directories(MLIR_StreamHLS_Support_Gurobi PRIVATE ${GUROBI_INCLUDE_DIRS})
# target_link_libraries(MLIR_StreamHLS_Support_Gurobi optimized ${GUROBI_CXX_LIBRARY}
#                         debug ${GUROBI_CXX_DEBUG_LIBRARY})
# target_link_libraries(MLIR_StreamHLS_Support_Gurobi PRIVATE ${GUROBI_LIBRARY})
