#include "streamhls/Dialect/Dataflow/Transforms/MergeKernels/Transform.h"

using namespace mlir;
using namespace streamhls;

#define DEBUG_TYPE "streamhls-merge-kernels-transform"

namespace mlir {
namespace streamhls {

// ---------- simplify affine ops in the module ----------
void simplifyAffineOps(ModuleOp &module) {
  PassManager pm = PassManager::on<ModuleOp>(module.getContext());

  // Apply the pass only to functions inside the module
  // pm.nest<func::FuncOp>().addPass(createSimplifyAffineStructuresPass());
  pm.nest<func::FuncOp>().addPass(createCanonicalizerPass());

  // Run the pass manager
  (void)pm.run(module);
}

// ---------- tile loops to 2 levels ----------
AffineForOp apply2LevelTiling(
  AffineForOp &forOp,
  OpBuilder &builder,
  const TilingSolution &tilingSolution
){
  auto l1TilingFactors = tilingSolution.l1TilingFactors;
  auto l2TilingFactors = tilingSolution.l2TilingFactors;
  AffineLoopBand band;
  getLoopBandFromOutermost(forOp, band);
  SmallVector<uint, 3> currTilingFactors;
  for(auto tutm : llvm::zip(l1TilingFactors, l2TilingFactors)){
    auto l1TilingFactor = std::get<0>(tutm);
    auto l2TilingFactor = std::get<1>(tutm);
    currTilingFactors.push_back(l1TilingFactor*l2TilingFactor);
  }
  // tile the for loop
  AffineLoopBand tiledBand1L;
  streamhls::tilePerfectlyNested(band, ArrayRef(currTilingFactors), &tiledBand1L, /*allDivisors*/ true);

  // innerloopband = tiledBand1L[size()/2 : size()] using SmallVector, do not use slice
  AffineLoopBand innerLoopsBand;
  for(auto loop : llvm::enumerate(tiledBand1L)){
    auto idx = loop.index();
    auto loopOp = loop.value();
    if(idx >= tiledBand1L.size() / 2){
      innerLoopsBand.push_back(loopOp);
    }
  }
  AffineLoopBand tiledBand2L;
  tilePerfectlyNested(innerLoopsBand, ArrayRef(l2TilingFactors), &tiledBand2L, true);
  tiledBand1L[0].walk([&](Operation *op) {
    if (auto affineParallel = dyn_cast<AffineParallelOp>(op))
      normalizeAffineParallel(affineParallel);
    else if (auto affineFor = dyn_cast<AffineForOp>(op))
      (void)streamhls::normalizeAffineForDivisorsOnly(affineFor);
  });
  // set pragma pipeline to unroll the innermost loops
  setLoopDirective(tiledBand1L[band.size() - 1], true, 1, false, false);
  return tiledBand1L[0];
}

// ---------- apply buffering strategy ----------
void applyBufferingStrategy(
  ModuleOp &module,
  AffineForOp &newForOp,
  OpBuilder &builder,
  MemrefSolutions &memrefSolutions
){
  auto loc = builder.getUnknownLoc();
  auto func = newForOp->getParentOfType<func::FuncOp>();
  auto *entryBlock = &func.getBlocks().front();

  // 1) Build memref -> {load/store ops} mapping once.
  DenseMap<Value, SmallVector<Operation*>> memrefToOps;
  newForOp.walk([&](Operation *op) {
    if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
      memrefToOps[loadOp.getMemRef()].push_back(op);
    } else if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
      memrefToOps[storeOp.getMemRef()].push_back(op);
    }
  });

  // 2) For each solution (which knows its memref), create buffer and rewrite uses.
  for (MemRefSolution &S : memrefSolutions) {
    Value memref = S.memref;

    auto it = memrefToOps.find(memref);
    if (it == memrefToOps.end())
      continue; // no uses in this loop band

    auto &memrefOps = it->second;

    // Allocate the local buffer once per memref solution.
    builder.setInsertionPointToStart(entryBlock);
    MemRefType bufferType = MemRefType::get(S.sizes, builder.getF32Type());
    Value localBuffer =
        builder.create<memref::AllocOp>(loc, bufferType).getResult();
    SmallVector<dataflow::PartitionKind> kinds;
    for (size_t i = 0; i < S.sizes.size(); ++i)
      kinds.push_back(dataflow::PartitionKind::CYCLIC);
    applyArrayPartition(localBuffer, S.fifoSizes, kinds);
    // Rewrite loads/stores for this memref.
    for (Operation *op : memrefOps) {
      if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
        builder.setInsertionPoint(loadOp);

        // Guarded preload into buffer.
        auto ifOp = builder.create<AffineIfOp>(
            loc, S.loadSet, S.loadSetOperands, /*withElseRegion=*/false);

        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
        if(S.isStore && S.isLoad){
          auto zeroConst = builder.create<arith::ConstantOp>(
            loc, 
            loadOp.getMemRef().getType().cast<MemRefType>().getElementType(),
            builder.getF32FloatAttr(0.0f)
          );
          builder.create<AffineStoreOp>(
            loc, 
            zeroConst, 
            localBuffer, 
            S.map,
            S.mapOperands
          );
        }else{ 
          auto tmpVal = builder.create<AffineLoadOp>(
            loc, loadOp.getMemRef(), loadOp.getMap(), loadOp.getMapOperands());

          setFifoAttr(
            builder.getContext(),
            tmpVal,
            S.fifoSizes,
            S.fifoMapOperands
          );
          builder.create<AffineStoreOp>(
            loc, 
            tmpVal, 
            localBuffer, 
            S.map,
            S.mapOperands
          );
        }
        // Replace original load with buffered load.
        builder.setInsertionPointAfter(ifOp);
        auto bufferedVal = builder.create<AffineLoadOp>(
            loc, localBuffer, S.map, S.mapOperands);
        loadOp.replaceAllUsesWith(bufferedVal.getResult());
        loadOp.erase();

      } else if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
        builder.setInsertionPoint(storeOp);

        // First, write to local buffer.
        builder.create<AffineStoreOp>(
            loc, storeOp.getValueToStore(), localBuffer, S.map, S.mapOperands);

        // Then guarded write-back to original memref.
        auto ifOp = builder.create<AffineIfOp>(
            loc, S.storeSet, S.storeSetOperands, /*withElseRegion=*/false);

        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
        auto bufferedVal = builder.create<AffineLoadOp>(
            loc, localBuffer, S.map, S.mapOperands);
        auto tmpVal = builder.create<AffineStoreOp>(
            loc, bufferedVal, storeOp.getMemRef(),
            storeOp.getMap(), storeOp.getMapOperands());
        setFifoAttr(
          builder.getContext(),
          tmpVal,
          S.fifoSizes,
          S.fifoMapOperands
        );
        storeOp.erase();
      }
    }
  }
}

// ---------- update loop bounds based on max loop info ----------
void updateLoopBounds(
  AffineForOp &forOp,
  const LoopBandInfo &maxLoopInfo
){
  AffineLoopBand allLoops;
  getLoopBandFromOutermost(forOp, allLoops);
  // make the upper bound of level 0 loops max(1, original_ub)
  for(auto loopPair : llvm::zip(allLoops, maxLoopInfo)){
    auto affineForOp = std::get<0>(loopPair);
    auto loopInfo = std::get<1>(loopPair);
    if(loopInfo.level != 0)
      continue;
    mlir::AffineMap ubMap = affineForOp.getUpperBoundMap();
    if (!ubMap) continue;

    mlir::MLIRContext *ctx = affineForOp.getContext();
    unsigned numDims  = ubMap.getNumDims();
    unsigned numSyms  = ubMap.getNumSymbols();

    // Transform each result expr: floordiv -> ceildiv.
    llvm::SmallVector<mlir::AffineExpr, 4> newResults;
    newResults.reserve(ubMap.getNumResults());
    for (mlir::AffineExpr r : ubMap.getResults())
      newResults.push_back(mlir::simplifyAffineExpr(floorToCeilDiv(r), numDims, numSyms));

    mlir::AffineMap newUBMap = mlir::AffineMap::get(numDims, numSyms, newResults, ctx);

    // Reuse original operands.
    llvm::SmallVector<mlir::Value, 4> ubOperands(affineForOp.getUpperBoundOperands());

    affineForOp.setUpperBoundMap(newUBMap);

    LLVM_DEBUG({
      llvm::dbgs() << "[merge-kernels] UB map floordiv→ceildiv: ";
      newUBMap.print(llvm::dbgs());
      llvm::dbgs() << "\n";
    });
  }
}
} // end namespace streamhls
} // end namespace mlir