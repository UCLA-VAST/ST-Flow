#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "streamhls/Dialect/Dataflow/Support/Utils.h"
#include "streamhls/Dialect/Dataflow/Transforms/Passes.h"

using namespace mlir;
using namespace streamhls;
using namespace dataflow;
using namespace affine;

#define DEBUG_TYPE "streamhls-loop-permutations"


/// Interchange the two innermost loops of every (maximal) perfectly-nested
/// affine loop band of depth >= 2.
///  - depth 2:  [0,1]   -> [1,0]
///  - depth 3:  [0,1,2] -> [0,2,1]
///  - depth 4:  [0,1,2,3] -> [0,1,3,2]
ModuleOp streamhls::raiseInnerLoopByOneLevel(ModuleOp &module) {
  // Only process outermost affine.for loops to avoid permuting the same nest
  // multiple times.
  SmallVector<AffineForOp, 16> outerLoops;
  module.walk([&](AffineForOp forOp) {
    if (!forOp->getParentOfType<AffineForOp>())
      outerLoops.push_back(forOp);
  });

  for (AffineForOp outer : outerLoops) {
    AffineLoopBand band;

    // If you prefer your own utilities, replace this line with:
    //   band = getLoopNestFromOutermost(outer);
    // or similar.
    getPerfectlyNestedLoops(band, outer);

    // getPerfectlyNestedLoops() returns outermost->innermost.
    if (band.size() < 2)
      continue;

    SmallVector<unsigned, 8> permMap;
    permMap.reserve(band.size());
    for (unsigned i = 0, e = band.size(); i < e; ++i)
      permMap.push_back(i);

    // Swap the two innermost loops.
    std::swap(permMap[band.size() - 2], permMap[band.size() - 1]);

    if (!isValidLoopInterchangePermutation(band, permMap)) {
      llvm::errs() << "[raiseInnerLoopByOneLevel] WARNING: invalid interchange "
                      "for a loop band of depth "
                   << band.size() << "\n";
      continue;
    }

    // This mutates the IR in-place (it will move bodies and update IV uses).
    (void)permuteLoops(band, permMap);
  }

  return module;
}
