#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "streamhls/Dialect/Dataflow/Transforms/Passes.h"

using namespace mlir;
using namespace streamhls;
using namespace dataflow;

#define DEBUG_TYPE "streamhls-lower-tensor-concat-to-slices"
namespace {
struct LowerTensorConcatToSlicesPattern : OpRewritePattern<tensor::ConcatOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::ConcatOp op,
                                PatternRewriter &rewriter) const override {
    // Only support 2-operand concat for now. Extend if needed.
    if (op.getInputs().size() != 2) return failure();

    Location loc = op.getLoc();
    auto outTy = cast<RankedTensorType>(op.getType());
    int64_t dim = op.getDim();

    // Allocate result
    Value init = rewriter.create<tensor::EmptyOp>(loc, outTy.getShape(),
                                                  outTy.getElementType());

    // Helper to build slice sizes/offsets.
    auto makeSlice = [&](Value src, int64_t offsetAlongDim) -> Value {
      auto srcTy = cast<RankedTensorType>(src.getType());
      SmallVector<OpFoldResult> offsets(outTy.getRank(),
                                        rewriter.getIndexAttr(0));
      SmallVector<OpFoldResult> sizes;
      SmallVector<OpFoldResult> strides(outTy.getRank(),
                                        rewriter.getIndexAttr(1));
      for (int64_t d = 0; d < outTy.getRank(); ++d) {
        int64_t sz = (d == dim) ? srcTy.getDimSize(d) : outTy.getDimSize(d);
        sizes.push_back(rewriter.getIndexAttr(sz));
      }
      offsets[dim] = rewriter.getIndexAttr(offsetAlongDim);
      return rewriter.create<tensor::InsertSliceOp>(loc, src, init,
                                                    offsets, sizes, strides);
    };

    // Compute sizes along concat dim.
    auto a = op.getInputs()[0];
    auto b = op.getInputs()[1];
    int64_t aLen = cast<RankedTensorType>(a.getType()).getDimSize(dim);
    // First insert A at offset 0.
    Value withA = makeSlice(a, /*offset=*/0);
    // Rebase 'init' to chain the second insert.
    init = withA;
    // Then insert B at offset aLen.
    Value withAB = makeSlice(b, /*offset=*/aLen);

    rewriter.replaceOp(op, withAB);
    return success();
  }
};
} // end anonymous namespace

// std::unique_ptr<Pass> createLowerTensorConcatPass() {
//   struct P : PassWrapper<P, OperationPass<ModuleOp>> {
//     void runOnOperation() override {
//       MLIRContext *ctx = &getContext();
//       RewritePatternSet ps(ctx);
//       ps.add<LowerTensorConcatToSlices>(ctx);
//       if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(ps))))
//         signalPassFailure();
//     }
//   };
//   return std::make_unique<P>();
// }

namespace {
struct LowerTensorConcatToSlices : public LowerTensorConcatToSlicesBase<LowerTensorConcatToSlices> {
  LowerTensorConcatToSlices() = default;
  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();

    // Lower copy operation.
    mlir::RewritePatternSet patterns(context);
    patterns.add<LowerTensorConcatToSlicesPattern>(context);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<Pass> streamhls::createLowerTensorConcatToSlicesPass() {
  return std::make_unique<LowerTensorConcatToSlices>();
}