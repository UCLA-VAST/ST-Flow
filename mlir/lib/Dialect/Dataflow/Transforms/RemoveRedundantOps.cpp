#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "streamhls/Dialect/Dataflow/Transforms/Passes.h"

using namespace mlir;
using namespace streamhls;
using namespace dataflow;

#define DEBUG_TYPE "streamhls-remove-redundant-ops"

namespace {
  struct RemoveGlobalSeed : public OpRewritePattern<ml_program::GlobalOp> {
    using OpRewritePattern<ml_program::GlobalOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(
      ml_program::GlobalOp op,
      PatternRewriter &rewriter
    ) const override {
      // TODO: This is a hack to remove the global seed
      // make sure it is not used anywhere
      if(op.use_empty()) {
        rewriter.eraseOp(op);
        return success();
      }else{
        return failure();
      }
    }
  };
} // end anonymous namespace

namespace {
  struct AnnotateFuncArgs : public OpRewritePattern<func::FuncOp> {
    using OpRewritePattern<func::FuncOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(
      func::FuncOp funcOp,
      PatternRewriter &rewriter
    ) const override {
      SmallVector<Attribute> argAttrs;
      for(int i = 0; i < funcOp.getNumArguments(); i++){
        auto inputStrAttr = rewriter.getStringAttr("input_" + std::to_string(i));
        funcOp.setArgAttr(i, "builtin.argKind", inputStrAttr);
        // argAttrs.push_back(inputStrAttr);
      }
      for(int i = 0; i < funcOp.getNumResults(); i++){
        auto outputStrAttr = rewriter.getStringAttr("output_" + std::to_string(i));
        funcOp.setResultAttr(i, "builtin.argKind", outputStrAttr);
        // argAttrs.push_back(outputStrAttr);
      }
      // funcOp.setAllArgAttrs(argAttrs);
      // funcOp.dump();
      return success();
    }
  };
} // end anonymous namespace

// This pass takes creates the weight bins for the host function
namespace {
struct RemoveRedundantOps : public RemoveRedundantOpsBase<RemoveRedundantOps> {
  void runOnOperation() override {
    auto module = getOperation();
    auto context = module.getContext();
    // auto func = module.lookupSymbol<func::FuncOp>("forward");
    RewritePatternSet patterns(context);
    patterns.add<RemoveGlobalSeed>(context);
    // patterns.add<AnnotateFuncArgs>(context);
    (void)applyPatternsAndFoldGreedily(module, std::move(patterns));
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> streamhls::createRemoveRedundantOpsPass() {
  return std::make_unique<RemoveRedundantOps>();
}