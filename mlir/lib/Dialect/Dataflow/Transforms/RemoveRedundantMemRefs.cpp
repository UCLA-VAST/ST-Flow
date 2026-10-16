#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "streamhls/Dialect/Dataflow/Transforms/Passes.h"

#include "streamhls/Dialect/Dataflow/Support/Utils.h"
#include "streamhls/Dialect/Dataflow/Support/AffineMemAccess.h"

#include "mlir/IR/IntegerSet.h"


using namespace mlir;
using namespace streamhls;
using namespace dataflow;

#define DEBUG_TYPE "streamhls-remove-redundant-memrefs"

// namespace {
//   struct RemoveRedundantMemRefsRewriter : public OpRewritePattern<memref::AllocOp> {
//     using OpRewritePattern<memref::AllocOp>::OpRewritePattern;
//     LogicalResult matchAndRewrite(
//       memref::AllocOp allocOp,
//       PatternRewriter &rewriter
//     ) const override {
//       auto memRef = allocOp.getResult();
//       SmallVector<AffineLoadOp, 4> loadOps;
//       SmallVector<AffineStoreOp, 4> storeOps;
//       for(auto user : memRef.getUsers()){
//         if(auto loadOp = dyn_cast<AffineLoadOp>(user)){
//           loadOps.push_back(loadOp);
//         }
//         if(auto storeOp = dyn_cast<AffineStoreOp>(user)){
//           storeOps.push_back(storeOp);
//         }
//       }

//       // // if there is only one load and one store
//       // if(loadOps.size() == 1 && storeOps.size() == 1){
//       //   LLVM_DEBUG(
//       //     llvm::dbgs() << "loadOps.size() == 1 && storeOps.size() == 1\n"
//       //   );
//       //   auto loadOp = loadOps[0];
//       //   auto storeOp = storeOps[0];
//       for(auto loadOp : loadOps){
//         for(auto storeOp : storeOps){
//           streamhls::MemRefAccess loadAccess(loadOp);
//           streamhls::MemRefAccess storeAccess(storeOp);
//           FlatAffineRelation loadRelation;
//           FlatAffineRelation storeRelation;
//           if(failed(loadAccess.getAccessRelation(loadRelation))){
//             LLVM_DEBUG(
//               llvm::dbgs() << "loadAccess.getAccessRelation failed\n"
//             );
//             // return failure();
//             continue;
//           }
//           if(failed(storeAccess.getAccessRelation(storeRelation))){
//             LLVM_DEBUG(
//               llvm::dbgs() << "storeAccess.getAccessRelation failed\n"
//             );
//             // return failure();
//             continue;
//           }

//           // if load and store are in the same block
//           if(loadOp->getBlock() != storeOp->getBlock()){
//             LLVM_DEBUG(
//               llvm::dbgs() << "\tloadOp->getBlock() != storeOp->getBlock()\n"
//             );
//             // return failure();
//             continue;
//           }
//           // if the load is after the store
//           if(loadOp->isBeforeInBlock(storeOp)){
//             LLVM_DEBUG(
//               llvm::dbgs() << "\tloadOp->isBeforeInBlock(storeOp)\n"
//             );
//             // return failure();
//             continue;
//           }
//           auto loadSpace = loadRelation.getSpace();
//           auto storeSpace = storeRelation.getSpace();
//           if(!loadSpace.isCompatible(storeSpace)){
//             LLVM_DEBUG(
//               llvm::dbgs() << "\t!loadSpace.isCompatible(storeSpace)\n"
//             );
//             // return failure();
//             continue;
//           }
//           if(!loadRelation.isEqual(storeRelation)){
//             LLVM_DEBUG(
//               llvm::dbgs() << "\t!loadRelation.isEqual(storeRelation)\n"
//             );
//             // return failure();
//             continue;
//           }
//           LLVM_DEBUG(
//             llvm::dbgs() << "Removing redundant memrefs\n"
//           );
//           rewriter.replaceOp(loadOp, storeOp.getOperand(0));
//           rewriter.eraseOp(storeOp);
//           if(allocOp.use_empty()){
//             rewriter.eraseOp(allocOp);
//           }
//           return success();
//           // if(loadOp->isBeforeInBlock(storeOp)){
//           //   // LLVM_DEBUG(
//           //   //   llvm::dbgs() << "\t\tloadOp->isBeforeInBlock(storeOp)\n";
//           //   //   loadOp.dump();
//           //   //   storeOp.dump();
//           //   //   loadOp->getBlock()->dump();
//           //   // );
//           // }else{
//           //   // LLVM_DEBUG(
//           //   //   storeOp->getBlock()->dump();
//           //   // );
            
//           // }
//         }
//       }
//       return success();
//     }
//   };
// }
namespace {
  struct RemoveRedundantMemRefsRewriter : public OpRewritePattern<AffineStoreOp> {
    using OpRewritePattern<AffineStoreOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(
      AffineStoreOp storeOp,
      PatternRewriter &rewriter
    ) const override {
      auto valueToStore = storeOp.getValueToStore();
      auto owner = valueToStore.getDefiningOp();
      if(auto constOp = dyn_cast<arith::ConstantOp>(owner)){
        auto memref = storeOp.getMemRef();
        auto numStores = 0;
        auto numLoads = 0;
        for(auto user : memref.getUsers()){
          if(auto loadOp = dyn_cast<AffineLoadOp>(user)){
            numLoads++;
          }else if(auto storeOp = dyn_cast<AffineStoreOp>(user)){
            numStores++;
          }
        }
        // llvm::dbgs() << "numLoads: " << numLoads << "\n";
        // llvm::dbgs() << "numStores: " << numStores << "\n";
        // auto allocOp = memref.getDefiningOp<memref::AllocOp>();
        for(auto user : memref.getUsers()){
          if(auto loadOp = dyn_cast<AffineLoadOp>(user)){
            if(numStores == 1 && numLoads == 1){
              rewriter.replaceOp(loadOp, constOp.getResult());
              rewriter.eraseOp(storeOp);
            }
            // rewriter.replaceOp(loadOp, constOp.getResult());
          }else if(auto storeOp = dyn_cast<AffineStoreOp>(user)){
            // do nothing
          }else{
            assert(false && "memref is used by an op other than AffineStore or AffineLoadOp");
          }
        }
      }
      return success();
    }
  };
}

namespace{
  struct AnnotateForLoops : public OpRewritePattern<AffineForOp> {
    using OpRewritePattern<AffineForOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(
      AffineForOp forOp,
      PatternRewriter &rewriter
    ) const override {
      AffineLoopBand band;
      getLoopBandFromOutermost(forOp, band);
      for(auto loop : band){
        if(hasLoopTypeAttr(loop)){
          // already annotated
          continue;
        }
        auto iv = loop.getInductionVar();
        auto numStores = 0;
        for(auto user : iv.getUsers()){
          if(isa<AffineStoreOp>(user)){
            numStores++;
          }
        }
        if(numStores >= 1){
          setLoopTypeAttr(loop, "parallel");
        }else{
          setLoopTypeAttr(loop, "reduction");
        }
      }
      return success();
    }
  };
}

namespace {
struct RemoveRedundantMemRefs
    : public RemoveRedundantMemRefsBase<RemoveRedundantMemRefs> {
  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();
    RewritePatternSet patterns(context);
    patterns.add<RemoveRedundantMemRefsRewriter>(context);
    patterns.add<AnnotateForLoops>(context);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<Pass> streamhls::createRemoveRedundantMemRefsPass() {
  return std::make_unique<RemoveRedundantMemRefs>();
}

ModuleOp streamhls::annotateLoops(ModuleOp &module) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::dataflow::DataflowDialect, mlir::func::FuncDialect,
                  mlir::arith::ArithDialect, mlir::tensor::TensorDialect,
                  mlir::affine::AffineDialect, mlir::math::MathDialect,
                  mlir::memref::MemRefDialect>();
  PassManager pm = PassManager::on<ModuleOp>(module.getContext());

  pm.getDependentDialects(registry); // check if this is necessary
  pm.nest<func::FuncOp>().addPass(streamhls::createRemoveRedundantMemRefsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  if (failed(pm.run(module)))
    assert(false && "Stream-HLS annotateLoops pass failed.");

  return module;
}