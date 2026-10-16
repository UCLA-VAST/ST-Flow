//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Adapted from ScaleHLS (https://github.com/UIUC-ChenLab/scalehls) and modified
// for Stream-HLS and ST-Flow. See NOTICE.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "streamhls/Dialect/Dataflow/Transforms/Passes.h"
// #include "streamhls/Dialect/Dataflow/Transforms/Utils.h"

// #include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
// #include "mlir/Analysis/Presburger/PresburgerSpace.h"

#include "streamhls/Dialect/Dataflow/Support/Utils.h"
#include "streamhls/Dialect/Dataflow/Support/AffineMemAccess.h"

#include "mlir/IR/IntegerSet.h"


using namespace mlir;
using namespace streamhls;
using namespace dataflow;

#define DEBUG_TYPE "streamhls-remove-loops-of-unit-iter"

namespace {
  struct RemoveLoopsOfUnitIterPattern : public OpRewritePattern<AffineForOp> {
    using OpRewritePattern<AffineForOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(
      AffineForOp forOp,
      PatternRewriter &rewriter
    ) const override {
      // LLVM_DEBUG(
      //   llvm::dbgs() << "RemoveLoopsOfUnitIter\n"
      // );
      // if does not have constant bounds
      if(!forOp.hasConstantBounds()){
        return failure();
      }
      // if has non-unit stride
      if(forOp.getStep() != 1){
        return failure();
      }
      auto lb = forOp.getConstantLowerBound();
      auto ub = forOp.getConstantUpperBound();
      // LLVM_DEBUG(
        // llvm::dbgs() << "lb: " << lb << "\n";
        // llvm::dbgs() << "ub: " << ub << "\n";
      // );
      if(lb!=0 || ub!=1){
        return failure();
      }
      // LLVM_DEBUG(
      //   llvm::dbgs() << "Starting to remove loop\n"
      // );
      auto loc = rewriter.getUnknownLoc();
      auto iv = forOp.getInductionVar();
      auto constantZero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      rewriter.replaceAllUsesWith(iv, constantZero.getResult());
      return success();
    }
  };
}

namespace {
  struct RemoveLoopsOfSingleIterPattern : public OpRewritePattern<AffineForOp> {
    using OpRewritePattern<AffineForOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(
      AffineForOp forOp,
      PatternRewriter &rewriter
    ) const override {
      // for nested loops where all loops have single iteration, we want to keep at least one loop
      auto parentOp = forOp->getParentOp();
      if(isa<func::FuncOp>(parentOp)){
        AffineLoopBand band;
        getLoopBandFromOutermost(forOp, band);
        if(band.size() == 1){
          return failure();
        }
      }
      return promoteIfSingleIteration(forOp);
    }
  };
}

// SCALEHLS code
static LogicalResult collapseMemref(Value memref) {
  auto type = memref.getType().dyn_cast<MemRefType>();
  if (!type)
    return failure();

  // TODO: Support non-identity buffers.
  if (!type.getLayout().getAffineMap().isIdentity() ||
      !llvm::any_of(type.getShape(),
                    [](int64_t dimSize) { return dimSize == 1; }))
    return failure();

  // TODO: Support non-affine read/write interfaces.
  if (llvm::any_of(memref.getUsers(), [](Operation *op) {
        return !isa<AffineReadOpInterface, AffineWriteOpInterface>(op);
      }))
    return failure();

  // Construct new shape.
  SmallVector<int64_t> newShape;
  SmallVector<unsigned> remainDims;
  for (auto dimSize : llvm::enumerate(type.getShape()))
    if (dimSize.value() != 1) {
      newShape.push_back(dimSize.value());
      remainDims.push_back(dimSize.index());
    }

  // Set the buffer to a new type.
  auto newType = MemRefType::get(newShape, type.getElementType(), AffineMap(),
                                 type.getMemorySpace());
  memref.setType(newType);

  // Update buffer users.
  for (auto user : memref.getUsers()) {
    AffineMap map;
    if (auto read = dyn_cast<affine::AffineReadOpInterface>(user))
      map = read.getAffineMap();
    else if (auto write = dyn_cast<affine::AffineWriteOpInterface>(user))
      map = write.getAffineMap();

    SmallVector<AffineExpr> newResults;
    for (auto dim : remainDims)
      newResults.push_back(map.getResult(dim));
    auto newMap = AffineMap::get(map.getNumDims(), map.getNumSymbols(),
                                 newResults, map.getContext());
    user->setAttr("map", AffineMapAttr::get(newMap));
  }
  return success();
}

namespace {
struct CollapseFuncMemref : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    Value memref = allocOp.getResult();
    return collapseMemref(memref);
  }
};
} // namespace


namespace {
struct RemoveLoopsOfUnitIter
    : public RemoveLoopsOfUnitIterBase<RemoveLoopsOfUnitIter> {
  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();
    mlir::RewritePatternSet patterns(context);

    // patterns.add<RemoveLoopsOfUnitIterPattern>(context);
    // (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
    // patterns.clear();
    patterns.add<RemoveLoopsOfSingleIterPattern>(context);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
    // patterns.clear();
    // patterns.add<CollapseFuncMemref>(context);
    // (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<Pass> streamhls::createRemoveLoopsOfUnitIterPass() {
  return std::make_unique<RemoveLoopsOfUnitIter>();
}