//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
//
//===----------------------------------------------------------------------===//


#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "streamhls/Dialect/Dataflow/Transforms/Passes.h"

#include "streamhls/Dialect/Dataflow/Support/Utils.h"
#include "streamhls/Dialect/Dataflow/Support/AffineMemAccess.h"

#include "mlir/IR/IntegerSet.h"


using namespace mlir;
using namespace streamhls;
using namespace dataflow;

#define DEBUG_TYPE "scalehls-lower-copy-to-affine"

namespace {
struct LowerCopy : public OpRewritePattern<memref::CopyOp> {
  LowerCopy(MLIRContext *context, bool internalCopyOnly = true)
      : OpRewritePattern(context), internalCopyOnly(internalCopyOnly) {}

  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copy,
                                PatternRewriter &rewriter) const override {
    // Check whether the copy op communicates with inputs or outputs.
    auto isExternalCopy = false;//isExternalBuffer(copy.getSource()) ||
                          //isExternalBuffer(copy.getTarget());

    // Return failure if we don't need to lower external copies.
    if (internalCopyOnly && isExternalCopy)
      return failure();

    rewriter.setInsertionPoint(copy);
    auto loc = copy.getLoc();
    auto memrefType = copy.getSource().getType().cast<MemRefType>();

    // Create explicit memory copy using an affine loop nest.
    SmallVector<Value, 4> ivs;
    auto constantZero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    for (auto dimSize : memrefType.getShape()) {
      // if (dimSize == 1) {
      //   ivs.push_back(constantZero);
      //   continue;
      // }
      auto loop = rewriter.create<AffineForOp>(loc, 0, dimSize);
      // setParallelAttr(loop);
      // If the copy op is not external, we consider the loop as point loop
      // that needs to be optimized later.
      // if (!isExternalCopy)
      //   setPointAttr(loop);
      rewriter.setInsertionPointToStart(loop.getBody());
      ivs.push_back(loop.getInductionVar());
    }

    // Create affine load/store operations.
    auto value =
        rewriter.create<AffineLoadOp>(loc, copy.getSource(), ivs);
    rewriter.create<AffineStoreOp>(loc, value, copy.getTarget(), ivs);

    rewriter.eraseOp(copy);
    return success();
  }

private:
  bool internalCopyOnly = true;
};
} // namespace

namespace {
struct LowerCopyToAffine : public LowerCopyToAffineBase<LowerCopyToAffine> {
  LowerCopyToAffine() = default;
  LowerCopyToAffine(bool argInternalCopyOnly) {
    internalCopyOnly = argInternalCopyOnly;
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto context = module.getContext();
    auto DT = DominanceInfo(module);

    // Lower copy operation.
    mlir::RewritePatternSet patterns(context);
    patterns.add<LowerCopy>(context, internalCopyOnly);
    (void)applyPatternsAndFoldGreedily(module, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<Pass> streamhls::createLowerCopyToAffinePass(bool internalCopyOnly) {
  return std::make_unique<LowerCopyToAffine>(internalCopyOnly);
}