#pragma once
#include "streamhls/Dialect/Dataflow/Transforms/MergeKernels/DataTypes.h"

namespace mlir {
namespace streamhls {
// ---------- simplify affine ops in the module ----------
void simplifyAffineOps(ModuleOp &module);
// ---------- tile loops to 2 levels ----------
AffineForOp apply2LevelTiling(
  AffineForOp &forOp,
  OpBuilder &builder,
  const TilingSolution &tilingSolution
);
// ---------- apply buffering strategy ----------
void applyBufferingStrategy(
  ModuleOp &module,
  AffineForOp &newForOp,
  OpBuilder &builder,
  MemrefSolutions &memrefSolutions
);
// ---------- update loop bounds based on max loop info ----------
void updateLoopBounds(
  AffineForOp &forOp,
  const LoopBandInfo &maxLoopInfo
);
}
}