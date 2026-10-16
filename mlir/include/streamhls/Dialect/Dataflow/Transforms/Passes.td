//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Parts of this file are adapted from ScaleHLS (https://github.com/UIUC-ChenLab/scalehls) and modified
// for Stream-HLS and ST-Flow. See NOTICE.
//
//===----------------------------------------------------------------------===//

#ifndef STREAMHLS_TRANSFORMS_PASSES_TD
#define STREAMHLS_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Transform Passes
//===----------------------------------------------------------------------===//

def RemoveRedundantMemRefs : Pass<"streamhls-remove-redundant-memrefs", "func::FuncOp"> {
  let summary = "Remove redundant memrefs";
  let constructor = "mlir::streamhls::createRemoveRedundantMemRefsPass()";
  let dependentDialects = ["mlir::dataflow::DataflowDialect"];
}

def RemoveRedundantOps : Pass<"streamhls-remove-redundant-ops", "ModuleOp"> {
  let summary = "Remove ml global seed";
  let constructor = "mlir::streamhls::createRemoveRedundantOpsPass()";
  let dependentDialects = ["mlir::dataflow::DataflowDialect"];
}

def RemoveLoopsOfUnitIter : Pass<"streamhls-remove-loops-of-unit-iter", "func::FuncOp"> {
  let summary = "Remove loops of unit iteration";
  let constructor = "mlir::streamhls::createRemoveLoopsOfUnitIterPass()";
  let dependentDialects = ["mlir::dataflow::DataflowDialect"];
}

def LowerSubviewToAffine : Pass<"streamhls-lower-subview-to-affine", "func::FuncOp"> {
  let summary = "Lower subview to affine";
  let constructor = "mlir::streamhls::createLowerSubviewToAffinePass()";
  let dependentDialects = ["mlir::dataflow::DataflowDialect"];
}

def LowerTensorConcatToSlices : Pass<"streamhls-lower-tensor-concat-to-slices", "func::FuncOp"> {
  let summary = "Lower tensor concat to slices";
  let constructor = "mlir::streamhls::createLowerTensorConcatToSlicesPass()";
  let dependentDialects = ["mlir::dataflow::DataflowDialect"];
}


def BufferizeFuncArgs : Pass<"streamhls-bufferize-func-args", "func::FuncOp"> {
  let summary = "Bufferize function arguments";
  let constructor = "mlir::streamhls::createBufferizeFuncArgsPass()";
  let dependentDialects = ["mlir::dataflow::DataflowDialect"];
}

// ScaleHLS
def LowerCopyToAffine : Pass<"scalehls-lower-copy-to-affine", "func::FuncOp"> {
  let summary = "Convert copy and assign to affine loops";
  let constructor = "mlir::streamhls::createLowerCopyToAffinePass()";

  let options = [
    Option<"internalCopyOnly", "internal-copy-only", "bool",
           /*default=*/"false", "only convert copy between internal buffers">
  ];
  let dependentDialects = ["mlir::dataflow::DataflowDialect"];

}

def StencilDataReuse :
      Pass<"streamhls-stencil-data-reuse", "func::FuncOp"> {
  let summary = "Stencil data reuse";
  let constructor = "mlir::streamhls::createStencilDataReusePass()";
  let dependentDialects = ["mlir::dataflow::DataflowDialect"];
}

def CollapseUnitDims : Pass<"streamhls-collapse-unit-dims", "ModuleOp"> {
  let summary = "Collapse unit dimensions in memrefs";
  let constructor = "mlir::streamhls::createCollapseUnitDimsPass()";
  let dependentDialects = ["mlir::dataflow::DataflowDialect"];
}

def ArrayStreamsToVectorStreamPass :
        Pass<"streamhls-array-streams-to-vector-stream", "mlir::ModuleOp"> {
  let summary = "Flatten 1D array-of-streams into a single vector stream";
  let description = [{
    Converts tensor<N x !dataflow.stream<T>> into
    !dataflow.stream<vector<NxT>>.
    Rewrites aofs_read/aofs_write into stream_read/stream_write using
    vector.extract/vector.insert.
  }];
  let constructor = "mlir::streamhls::createArrayStreamsToVectorStreamPass()";
  let dependentDialects = ["mlir::dataflow::DataflowDialect"];
}

def EliminateSingleWriteMultipleReads : Pass<"streamhls-eliminate-single-write-multiple-reads", "func::FuncOp"> {
  let summary = "Eliminate single-write-multiple-reads pattern by creating buffer copies";
  let description = [{
    For each memref that has one write and multiple reads, create buffer copies
    so that each read has its own copy. This helps to avoid read-after-write
    hazards in HLS synthesis.
  }];
   let options = [
    Option<"eliminationType", "elimination-type", "std::string",
           /*default=*/"\"skip\"", "Elimination type: skip, original, or duplicate">
  ];
  let constructor = "mlir::streamhls::createEliminateSingleWriteMultipleReadsPass()";
  let dependentDialects = ["mlir::dataflow::DataflowDialect"];
}

#endif // STREAMHLS_TRANSFORMS_PASSES_TD