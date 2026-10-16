#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/IntegerSet.h"
#include "streamhls/Dialect/Dataflow/Support/Utils.h"
#include "streamhls/Dialect/Dataflow/Transforms/Passes.h"

using namespace mlir;
using namespace streamhls;
using namespace dataflow;
using namespace affine;

#define DEBUG_TYPE "streamhls-sencil-data-reuse"

namespace {

  struct StencilDataReusePattern : public OpRewritePattern<AffineLoadOp> {
    using OpRewritePattern<AffineLoadOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(
      AffineLoadOp loadOp,
      PatternRewriter &rewriter
    ) const override {
      auto accessMap = loadOp.getAffineMap();
      if(accessMap.isIdentity())
        return failure();
      // target load should have access map with dims > results
      if(accessMap.getNumDims() <= accessMap.getNumResults())
        return failure();
      
      // accessMap.dump();

      auto accessIVs = loadOp.getMapOperands();
      // we need to check that an affine map is of the form:
      // (d0, d1, d2, d3) -> (d0 * c1 + d1, d2 * c2 + d3)
      // where c1 and c2 are constants
      // H and W dims have binaryExprs
      // other dims have dimExprs
      SmallVector<Value, 4> tensorNonHWIVs;
      SmallVector<Value, 4> tensor_H_W_IVs;
      SmallVector<uint64_t, 4> strides;
      SmallVector<Value, 4> kernelIVs;
      // get expressions from the access map
      auto exprs = accessMap.getResults();
      SmallVector<AffineExpr, 4> dimExprs;
      SmallVector<AffineBinaryOpExpr, 4> binaryExprs;
      for(auto expr : exprs) {
        if(isa<AffineDimExpr>(expr)) {
          dimExprs.push_back(expr);
        } else if(isa<AffineBinaryOpExpr>(expr)) {
          binaryExprs.push_back(cast<AffineBinaryOpExpr>(expr));
        } else {
          return failure();
        }
      }

      for(auto dimExpr : dimExprs) {
        tensorNonHWIVs.push_back(accessIVs[cast<AffineDimExpr>(dimExpr).getPosition()]);
      }
      // TODO: check for in channels and out channels
      for(auto binaryExpr : binaryExprs) {
        // auto binaryExpr = dyn_cast<AffineBinaryOpExpr>(expr);
        // lhs has two cases: dimExpr or binaryExpr
        auto lhs = binaryExpr.getLHS();
        if(auto dimExpr = dyn_cast<AffineDimExpr>(lhs)) {
          // get input idx from dim expr
          auto tensorIdx = dimExpr.getPosition();
          auto stride = 1;
          tensor_H_W_IVs.push_back(accessIVs[tensorIdx]);
          strides.push_back(stride);
        } else if(auto binaryExpr = dyn_cast<AffineBinaryOpExpr>(lhs)) {
          auto lhs = binaryExpr.getLHS();
          auto rhs = binaryExpr.getRHS();
          if(!isa<AffineDimExpr>(lhs) || !isa<AffineConstantExpr>(rhs))
            return failure();
          // check if lhs is a dim expr
          auto inputIdx = cast<AffineDimExpr>(lhs).getPosition();
          auto stride = cast<AffineConstantExpr>(rhs).getValue();
          tensor_H_W_IVs.push_back(accessIVs[inputIdx]);
          strides.push_back(stride);
        } else {
          return failure();
        }
        auto rhs = binaryExpr.getRHS();
        if(!isa<AffineDimExpr>(rhs))
          return failure();
        auto kernelIdx = cast<AffineDimExpr>(rhs).getPosition();
        kernelIVs.push_back(accessIVs[kernelIdx]);
      }
      // if(tensor_I_H_W_IVs.size() != 3 || kernelIVs.size() != 2)
      //   return failure();
      SmallVector<AffineForOp, 4> tensorNonHWLoops;
      SmallVector<AffineForOp, 4> tensor_H_W_Loops;
      SmallVector<AffineForOp, 4> kernelLoops;
      for(auto iv : tensorNonHWIVs){
        if(auto forOp = dyn_cast<AffineForOp>(iv.getParentBlock()->getParentOp())) {
          tensorNonHWLoops.push_back(forOp);
        }else {
          return failure();
        }
      }
      for(auto iv : tensor_H_W_IVs) {
        if(auto forOp = dyn_cast<AffineForOp>(iv.getParentBlock()->getParentOp())) {
          tensor_H_W_Loops.push_back(forOp);
        }else {
          return failure();
        }
      }
      for(auto iv : kernelIVs) {
        if(auto forOp = dyn_cast<AffineForOp>(iv.getParentBlock()->getParentOp())) {
          kernelLoops.push_back(forOp);
        }else {
          return failure();
        }
      }
      // Step 1: set i, h and w loop bounds to match the input tensor i, h and w
      auto memref = loadOp.getMemRef();
      auto memrefType = memref.getType().cast<MemRefType>();
      SmallVector<int64_t, 4> memrefShape;
      for(auto shapeIdx : llvm::enumerate(memrefType.getShape())) {
        if(shapeIdx.index() < tensorNonHWIVs.size())
          continue;
        memrefShape.push_back(shapeIdx.value());
      }
      // skip nonHW dims in memref shape
      for(auto loopShapePair : llvm::zip(tensor_H_W_Loops, memrefShape)) {
        auto loop = std::get<0>(loopShapePair);
        auto shape = std::get<1>(loopShapePair);
        if(loop.getConstantUpperBound() == shape)
          return failure();
        loop.setConstantUpperBound(shape);
      }
      // for(auto loopShapePair : llvm::zip(tensor_I_H_W_Loops, memrefShape)) {
      //   auto loop = std::get<0>(loopShapePair);
      //   auto shape = std::get<1>(loopShapePair);
      //   loop.setConstantUpperBound(shape);
      // }
      // if (h - K + 1) >= 0 and (w - K + 1) >= 0 and (((h - K + 1)) % S) == 0 and (((w - K + 1)) % S) == 0:
      //   # print(h, ifset)
      //   nh = (h - K + 1) // S
      //   nw = (w - K + 1) // S
      //   ih = S*nh + p
      //   iw = S*nw + q
      //   oh = nh
      //   ow = nw
      // // Step 2: create if guard expressions
      // // get loop bounds
      // int64_t I = tensor_I_H_W_Loops[0].getConstantUpperBound();
      // int64_t H = tensor_H_W_Loops[0].getConstantUpperBound();
      // int64_t W = tensor_H_W_Loops[1].getConstantUpperBound();
      int64_t P = kernelLoops[0].getConstantUpperBound();
      int64_t Q = kernelLoops[1].getConstantUpperBound();
      int64_t HStride = strides[0];
      int64_t WStride = strides[1];
      AffineExpr hExpr = rewriter.getAffineDimExpr(0);
      AffineExpr wExpr = rewriter.getAffineDimExpr(1);
      AffineExpr pExpr = rewriter.getAffineConstantExpr(P);
      AffineExpr qExpr = rewriter.getAffineConstantExpr(Q);
      AffineExpr hStrideExpr = rewriter.getAffineConstantExpr(HStride);
      AffineExpr wStrideExpr = rewriter.getAffineConstantExpr(WStride);
      SmallVector<AffineExpr, 4> guardExprs;
      SmallVector<bool, 4> guardEqFlags;
      // create if guard expressions
      // h - P + 1 >= 0
      guardExprs.push_back(hExpr - pExpr + 1);
      guardEqFlags.push_back(false);

      // w - Q + 1 >= 0
      guardExprs.push_back(wExpr - qExpr + 1);
      guardEqFlags.push_back(false);

      // (h - P + 1) % hStride == 0
      if((hExpr - pExpr + 1) % hStrideExpr != 0){
        guardExprs.push_back((hExpr - pExpr + 1) % hStrideExpr);
        guardEqFlags.push_back(true);
      }

      // (w - Q + 1) % wStride == 0
      if((wExpr - qExpr + 1) % wStrideExpr != 0){
        guardExprs.push_back((wExpr - qExpr + 1) % wStrideExpr);
        guardEqFlags.push_back(true);
      }

      // create if condition

      auto ifCondition = IntegerSet::get(tensor_H_W_Loops.size(), 0, guardExprs, guardEqFlags);
      // ifCondition.dump();
      // accessMap.dump();

      // get inner most loop using the loadOp
      auto innerMostLoop = loadOp.getOperation()->getParentOfType<AffineForOp>();
      AffineLoopBand band;
      getLoopBandFromInnermost(innerMostLoop, band);
      SmallVector<Value, 4> irrelevantIVs;

      for(auto forOp : band){
        // forOp.dump();
        auto iv = forOp.getInductionVar();
        // if not in tensorNonHWIVs, add to irrelevantIVs
        if(!llvm::is_contained(tensorNonHWIVs, iv) && !llvm::is_contained(tensor_H_W_IVs, iv))
          irrelevantIVs.push_back(iv);
      }
      // // get the ops of the inner most loop
      Block* innerMostLoopBody = innerMostLoop.getBody();
      SmallVector<Operation*, 4> loopOps;
      for(auto &op : innerMostLoopBody->getOperations()) {
        loopOps.push_back(&op);
      }
      auto loc = rewriter.getUnknownLoc();
      rewriter.setInsertionPointToStart(innerMostLoopBody);
      // create if operation
      auto ifOp = rewriter.create<AffineIfOp>(loc, ifCondition, tensor_H_W_IVs, false);
      // move all ops in the inner most loop to the then block of the if operation
      for(auto op : loopOps) {
        if(isa<AffineYieldOp>(op))
          continue;
        // llvm::dbgs() << "moving op: " << *op << "\n";
        op->moveBefore(ifOp.getThenBlock()->getTerminator());
      }

      // Step 2: update access maps
      rewriter.setInsertionPointToStart(ifOp.getThenBlock());
      // new h = (h - K + 1) // S
      AffineExpr newHExpr = (rewriter.getAffineDimExpr(0) - pExpr + 1).floorDiv(rewriter.getAffineConstantExpr(HStride));
      auto hExprList = {newHExpr};
      ValueRange hIVs = {tensor_H_W_IVs[0]};
      // create affineApplyOp
      auto hAffineApplyOp = rewriter.create<AffineApplyOp>(loc, hExprList, hIVs);

      // new w = (w - K + 1) // S
      AffineExpr newWExpr = (rewriter.getAffineDimExpr(0) - qExpr + 1).floorDiv(rewriter.getAffineConstantExpr(WStride));
      auto wExprList = {newWExpr};
      ValueRange wIVs = {tensor_H_W_IVs[1]};
      // create affineApplyOp
      auto wAffineApplyOp = rewriter.create<AffineApplyOp>(loc, wExprList, wIVs);

      SmallPtrSet<Operation*, 4> hExceptions = {hAffineApplyOp, ifOp};
      SmallPtrSet<Operation*, 4> wExceptions = {wAffineApplyOp, ifOp};
      tensor_H_W_IVs[0].replaceAllUsesExcept(hAffineApplyOp.getResult(), hExceptions);
      tensor_H_W_IVs[1].replaceAllUsesExcept(wAffineApplyOp.getResult(), wExceptions);
      
      // Step 3: create new reuse buffer
      // original buffer
      auto originalBuffer = memref.getDefiningOp();
      rewriter.setInsertionPoint(originalBuffer);
      auto newBuffer = rewriter.create<memref::AllocOp>(loc, memrefType);
      // create new if statement
      guardExprs.clear();
      guardEqFlags.clear();
      // create if guard expressions using irrelevant IVs
      for(auto idx = 0; idx < irrelevantIVs.size(); idx++) {
        guardExprs.push_back(rewriter.getAffineDimExpr(idx));
        guardEqFlags.push_back(true);
      }

      // create if condition
      ifCondition = IntegerSet::get(irrelevantIVs.size(), 0, guardExprs, guardEqFlags);

      // create if operation
      rewriter.setInsertionPointToStart(innerMostLoopBody);

      auto ifOp2 = rewriter.create<AffineIfOp>(loc, ifCondition, irrelevantIVs, false);
      // set insertion point to then block
      rewriter.setInsertionPointToStart(ifOp2.getThenBlock());
      // create new load op
      // append tensor_H_W_IVs to tensorNonHWIVs
      SmallVector<Value, 4> allIVs;
      allIVs.append(tensorNonHWIVs.begin(), tensorNonHWIVs.end());
      allIVs.append(tensor_H_W_IVs.begin(), tensor_H_W_IVs.end());

      // llvm::dbgs() << "allIVs: " << allIVs.size() << "\n";
      // if allIVs.size != newBuffer rank, fail
      if(allIVs.size() != newBuffer.getType().cast<MemRefType>().getRank())
        return failure();

      auto newLoadOp = rewriter.create<AffineLoadOp>(
        loc, 
        memref,
        allIVs
      );
      // new store op
      rewriter.create<AffineStoreOp>(
        loc,
        newLoadOp.getResult(),
        newBuffer,
        allIVs
      );

      // // replace old buffer with new buffer
      loadOp.setMemRef(newBuffer);
        
      return success();
    }
  };
} // end anonymous namespace


namespace {
  static AffineMap fixAccessMap(AffineMap map){
    SmallVector<AffineExpr, 2> replacements;
    for(unsigned i = 0; i < map.getNumSymbols(); i++){
      replacements.push_back(getAffineDimExpr(map.getNumDims() + i, map.getContext()));
    }
    return map.replaceDimsAndSymbols(
      {}, 
      replacements,
      map.getNumDims() + map.getNumSymbols(),
      0
    );
}
  struct PaddingReusePattern : public OpRewritePattern<AffineStoreOp> {
    using OpRewritePattern<AffineStoreOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(
      AffineStoreOp storeOp,
      PatternRewriter &rewriter
    ) const override {
      auto accessMap = fixAccessMap(storeOp.getAffineMap());
      if(accessMap.isIdentity())
        return failure();
      // target load should have access map with dims > results
      if(accessMap.getNumDims() != accessMap.getNumResults())
        return failure();

      storeOp.setMap(accessMap); // Check if this is correct

      auto accessIVs = storeOp.getMapOperands();
      // we need to check that an affine map is of the form: 
      // (d0, d1, d2) -> (d0, d1 + c1, d2 + c2)
      // results: (dimExpr, binaryExpr, binaryExpr)
      SmallVector<AffineExpr, 4> dimExprs;
      SmallVector<AffineBinaryOpExpr, 4> binaryExprs;
      for(auto expr : accessMap.getResults()) {
        if(auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
          dimExprs.push_back(dimExpr);
        } else if(auto binaryExpr = dyn_cast<AffineBinaryOpExpr>(expr)) {
          binaryExprs.push_back(binaryExpr);
        } else {
          return failure();
        }
      }
      if(dimExprs.size() != 1 || binaryExprs.size() != 2)
        return failure();
      SmallVector<Value, 4> tensorNonHWIVs;
      SmallVector<Value, 4> tensor_H_W_IVs;
      SmallVector<uint64_t, 4> paddings;
      for(auto dimExpr : dimExprs) {
        tensorNonHWIVs.push_back(accessIVs[cast<AffineDimExpr>(dimExpr).getPosition()]);
      }
      for(auto binaryExpr : binaryExprs) {
        auto lhs = binaryExpr.getLHS();
        auto rhs = binaryExpr.getRHS();
        if(!isa<AffineDimExpr>(lhs) || !isa<AffineConstantExpr>(rhs))
          return failure();
        auto tensorIdx = cast<AffineDimExpr>(lhs).getPosition();
        auto padding = cast<AffineConstantExpr>(rhs).getValue();
        paddings.push_back(padding);
        tensor_H_W_IVs.push_back(accessIVs[tensorIdx]);
      }

      SmallVector<AffineForOp, 4> tensorNonHWLoops;
      SmallVector<AffineForOp, 4> tensor_H_W_Loops;
      SmallVector<Value, 4> allIVs;
      for(auto iv : tensorNonHWIVs){
        if(auto forOp = dyn_cast<AffineForOp>(iv.getParentBlock()->getParentOp())) {
          tensorNonHWLoops.push_back(forOp);
          allIVs.push_back(forOp.getInductionVar());
        }else {
          return failure();
        }
      }
      for(auto iv : tensor_H_W_IVs) {
        if(auto forOp = dyn_cast<AffineForOp>(iv.getParentBlock()->getParentOp())) {
          tensor_H_W_Loops.push_back(forOp);
          allIVs.push_back(forOp.getInductionVar());
        }else {
          return failure();
        }
      }

      // Step 1: set i, h and w loop bounds to match the input tensor i, h and w
      auto memref = storeOp.getMemRef();
      auto memrefType = memref.getType().cast<MemRefType>();
      SmallVector<int64_t, 4> memrefShape;
      for(auto shapeIdx : llvm::enumerate(memrefType.getShape())) {
        if(shapeIdx.index() < tensorNonHWIVs.size())
          continue;
        memrefShape.push_back(shapeIdx.value());
      }
      // skip nonHW dims in memref shape
      for(auto loopShapePair : llvm::zip(tensor_H_W_Loops, memrefShape)) {
        auto loop = std::get<0>(loopShapePair);
        auto shape = std::get<1>(loopShapePair);
        if(loop.getConstantUpperBound() == shape)
          return failure();
        loop.setConstantUpperBound(shape);
      }

      // Step 2: create if guard expressions
      SmallVector<AffineExpr, 4> guardExprs;
      SmallVector<bool, 4> guardEqFlags;
      // guardExprs.push_back(paddings[0] - rewriter.getAffineDimExpr(0));
      // guardEqFlags.push_back(false);
      // guardExprs.push_back(paddings[1] - rewriter.getAffineDimExpr(1));
      // guardEqFlags.push_back(false);
      // guardExprs.push_back(rewriter.getAffineDimExpr(0) - memrefShape[0] + 2*paddings[0]);
      // guardEqFlags.push_back(false);
      // guardExprs.push_back(rewriter.getAffineDimExpr(1) - memrefShape[1] + 2*paddings[1]);
      // guardEqFlags.push_back(false);

      guardExprs.push_back(rewriter.getAffineDimExpr(0) - paddings[0]);
      guardEqFlags.push_back(false);
      guardExprs.push_back(rewriter.getAffineDimExpr(1) - paddings[1]);
      guardEqFlags.push_back(false);
      guardExprs.push_back((memrefShape[0] - 2*paddings[0]) - rewriter.getAffineDimExpr(0));
      guardEqFlags.push_back(false);
      guardExprs.push_back((memrefShape[1] - 2*paddings[1]) - rewriter.getAffineDimExpr(1));
      guardEqFlags.push_back(false);

      auto ifCondition = IntegerSet::get(tensor_H_W_Loops.size(), 0, guardExprs, guardEqFlags);

      auto innerMostLoop = storeOp.getOperation()->getParentOfType<AffineForOp>();
      AffineLoopBand band;
      getLoopBandFromInnermost(innerMostLoop, band);
      // // get the ops of the inner most loop
      Block* innerMostLoopBody = innerMostLoop.getBody();
      SmallVector<Operation*, 4> loopOps;
      for(auto &op : innerMostLoopBody->getOperations()) {
        loopOps.push_back(&op);
      }
      auto loc = rewriter.getUnknownLoc();
      rewriter.setInsertionPointToStart(innerMostLoopBody);
      // create if operation
      auto ifOp = rewriter.create<AffineIfOp>(loc, ifCondition, tensor_H_W_IVs, false);
      // move all ops in the inner most loop to the then block of the if operation
      for(auto op : loopOps) {
        if(isa<AffineYieldOp>(op))
          continue;
        // llvm::dbgs() << "moving op: " << *op << "\n";
        op->moveBefore(ifOp.getThenBlock()->getTerminator());
      }

      // Step 2: update access maps
      rewriter.setInsertionPointToStart(ifOp.getThenBlock());

      AffineExpr newHExpr = (rewriter.getAffineDimExpr(0) - paddings[0]);
      auto hExprList = {newHExpr};
      ValueRange hIVs = {tensor_H_W_IVs[0]};
      auto hAffineApplyOp = rewriter.create<AffineApplyOp>(loc, hExprList, hIVs);

      AffineExpr newWExpr = (rewriter.getAffineDimExpr(0) - paddings[1]);
      auto wExprList = {newWExpr};
      ValueRange wIVs = {tensor_H_W_IVs[1]};
      auto wAffineApplyOp = rewriter.create<AffineApplyOp>(loc, wExprList, wIVs);

      SmallPtrSet<Operation*, 4> hExceptions = {hAffineApplyOp, ifOp};
      SmallPtrSet<Operation*, 4> wExceptions = {wAffineApplyOp, ifOp};

      tensor_H_W_IVs[0].replaceAllUsesExcept(hAffineApplyOp.getResult(), hExceptions);
      tensor_H_W_IVs[1].replaceAllUsesExcept(wAffineApplyOp.getResult(), wExceptions);

      // create new reuse buffer
      auto originalBuffer = memref.getDefiningOp();
      rewriter.setInsertionPoint(originalBuffer);
      auto newBuffer = rewriter.create<memref::AllocOp>(loc, memrefType);
      // create new load op after if statement

      rewriter.setInsertionPointAfter(ifOp);
      auto loadOp = rewriter.create<AffineLoadOp>(
        loc,
        memref,
        allIVs
      );
      // create new store op
      rewriter.create<AffineStoreOp>(
        loc,
        loadOp.getResult(),
        newBuffer.getMemref(),
        allIVs
      );

      // replace old buffer with new buffer
      // SmallPtrSet<Operation*, 4> exceptions = {storeOp, loadOp};
      // memref.replaceAllUsesExcept(newBuffer.getResult(), exceptions);
      for(auto user : memref.getUsers()) {
        if(auto currLoadOp = dyn_cast<AffineLoadOp>(user)) {
          if(currLoadOp != loadOp){
            currLoadOp.setMemRef(newBuffer);
          }
        }
      }

      return success();
    }
  };
} // end anonymous namespace


namespace {
struct StencilDataReuse : public StencilDataReuseBase<StencilDataReuse> {
  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.add<StencilDataReusePattern>(context);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));

    patterns.clear();
    patterns.add<PaddingReusePattern>(context);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> streamhls::createStencilDataReusePass() {
  return std::make_unique<StencilDataReuse>();
}
