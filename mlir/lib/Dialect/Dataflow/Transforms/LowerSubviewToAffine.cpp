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

#define DEBUG_TYPE "streamhls-lower-subview-to-affine"

namespace {
struct LowerSubview : public OpRewritePattern<memref::SubViewOp> {
  using OpRewritePattern<memref::SubViewOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::SubViewOp subviewOp,
                                PatternRewriter &rewriter) const override {
    auto inShape = subviewOp.getSourceType().getShape();
    auto outShape = subviewOp.getType().getShape();
    auto inRank = inShape.size();
    auto outRank = outShape.size();
    if(inRank != outRank)
      return failure();
    if(!subviewOp.hasUnitStride() || !subviewOp.hasZeroOffset())
      return failure();
    // TODO: FIXME: This is just a hack for TinyBert
    rewriter.replaceAllUsesWith(subviewOp.getResult(), subviewOp.getOperand(0));
    // auto lbs = subviewOp.getStaticOffsets();
    // auto ubs = subviewOp.getStaticSizes();
    // auto steps = subviewOp.getStaticStrides();
    // auto memrefType = subviewOp.getType().cast<MemRefType>();
    // auto loc = rewriter.getUnknownLoc();
    // rewriter.setInsertionPoint(subviewOp);
    // auto allocResult = rewriter.create<memref::AllocOp>(loc, memrefType);
    // affine::buildAffineLoopNest(rewriter, loc, lbs, ubs, steps,
    //   [&](OpBuilder &b, Location loc, ValueRange ivs) {
    //     // affine load op
    //     auto loadOpResult = b.create<affine::AffineLoadOp>(loc, subviewOp.getOperand(), ivs);
    //     // affine store op
    //     b.create<affine::AffineStoreOp>(loc, loadOpResult, allocResult, ivs);
    //   });
    // rewriter.replaceAllUsesWith(allocResult);
    return success();
  }
};
} // namespace

namespace {
struct LowerSubviewToAffine : public LowerSubviewToAffineBase<LowerSubviewToAffine> {
  LowerSubviewToAffine() = default;

  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();

    // Lower copy operation.
    mlir::RewritePatternSet patterns(context);
    patterns.add<LowerSubview>(context);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<Pass> streamhls::createLowerSubviewToAffinePass() {
  return std::make_unique<LowerSubviewToAffine>();
}