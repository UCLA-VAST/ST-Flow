//===- ConverterInsertion.h -------------------------------------*- C++ -*-===//
// Header for DFG-level converter insertion logic.
//===----------------------------------------------------------------------===//

#ifndef STREAMHLS_TRANSFORMS_CONVERTERINSERTION_H
#define STREAMHLS_TRANSFORMS_CONVERTERINSERTION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "streamhls/Dialect/Dataflow/Support/DFG.h"

namespace mlir {
namespace streamhls {

/// Resolves DFG edges and inserts converter nodes if necessary.
void resolveDFGEdges(DFG &dfg);

/// Returns the argument index of a function with the given buffer name.
int getArgIdxByBufferName(ModuleOp module, func::FuncOp func, StringAttr bufferName, StringRef io = "");

} // namespace streamhls
} // namespace mlir

#endif // STREAMHLS_TRANSFORMS_CONVERTERINSERTION_H
