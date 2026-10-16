#pragma once
#include "streamhls/Dialect/Dataflow/Transforms/MergeKernels/DataTypes.h"

namespace mlir {
namespace streamhls {
/// ----------- Gather functions and their outermost loops from modules -----------
std::pair<SmallVector<func::FuncOp>, SmallVector<affine::AffineForOp>> gatherFuncsAndForOps(
  SmallVectorImpl<ModuleOp> &modules
);

/// ---------- Merge multiple kernel modules into a single module/function ----------
std::tuple<ModuleOp, AffineForOp, IRMapping> mergeKernels(
  SmallVectorImpl<ModuleOp> &modules, 
  OpBuilder &b,
  std::string moduleName
);
}
}