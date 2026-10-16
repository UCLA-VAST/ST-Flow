#include "streamhls/Dialect/Dataflow/Transforms/MergeKernels/Merging.h"

using namespace mlir;
using namespace streamhls;

#define DEBUG_TYPE "streamhls-merge-kernels-merging"

// === general helpers for merging kernels into a single module/function
namespace {
func::FuncOp getOnlyFunction(ModuleOp m) {
  func::FuncOp f;
  for (auto fo : m.getOps<func::FuncOp>()) {
    if (!f) f = fo;
  }
  assert(f && "expected at least one func.func in each module");
  return f;
}

MemRefType maxRankedMemRefShape(MemRefType a, MemRefType b) {
  assert(a.getRank() == b.getRank() && "rank mismatch");
  SmallVector<int64_t, 4> shape(a.getRank());
  for (int i = 0, e = a.getRank(); i < e; ++i) {
    int64_t da = a.getDimSize(i);
    int64_t db = b.getDimSize(i);
    if (ShapedType::isDynamic(da) && ShapedType::isDynamic(db)) {
      shape[i] = ShapedType::kDynamic;
    } else if (ShapedType::isDynamic(da)) {
      shape[i] = db;
    } else if (ShapedType::isDynamic(db)) {
      shape[i] = da;
    } else {
      shape[i] = std::max(da, db);
    }
  }
  return MemRefType::get(shape, a.getElementType(), a.getLayout(),
                         a.getMemorySpace());
}

/// --- merge kernels helpers ----------------------------------------------------------------

/// Gather the outer-to-inner single loop band (assumes a single perfect nest).
void collectOutermostBand(func::FuncOp f,
                                 SmallVector<affine::AffineForOp, 4> &band) {
  affine::AffineForOp outer;
  // Find outermost affine.for (one whose parent is not an affine.for).
  f.walk([&](affine::AffineForOp fo) {
    if (!fo->getParentOfType<affine::AffineForOp>())
      if (!outer)
        outer = fo;
  });
  assert(outer && "no affine.for found in function");
  // Walk down the direct nested chain.
  auto cur = outer;
  while (cur) {
    band.push_back(cur);
    affine::AffineForOp next;
    cur.getBody()->walk([&](affine::AffineForOp inner) {
      if (inner->getParentOp() == cur) next = inner;
    });
    cur = next;
  }
}


/// Unify function type across N funcs by taking elementwise max on memrefs.
/// Also returns number of UB params to add (loop depth from the first func).
FunctionType buildUnifiedTypeWithUBs(OpBuilder &b,
                                            ArrayRef<func::FuncOp> funcs,
                                            unsigned &loopDepth) {
  assert(!funcs.empty());
  auto t0 = const_cast<func::FuncOp &>(funcs.front()).getFunctionType();

  SmallVector<Type> inTypes(t0.getInputs().begin(), t0.getInputs().end());
  SmallVector<Type> outTypes(t0.getResults().begin(), t0.getResults().end());

  for (auto f : funcs) {
    auto ft = f.getFunctionType();
    assert(ft.getNumInputs() == inTypes.size() &&
           ft.getNumResults() == outTypes.size() &&
           "arity mismatch among identical kernels");

    for (unsigned i = 0, e = inTypes.size(); i < e; ++i) {
      Type a = inTypes[i], bty = ft.getInput(i);
      if (auto ma = a.dyn_cast<MemRefType>()) {
        auto mb = bty.cast<MemRefType>();
        inTypes[i] = maxRankedMemRefShape(ma, mb);
      } else {
        assert(a == bty && "non-memref arg type mismatch");
      }
    }
    for (unsigned i = 0, e = outTypes.size(); i < e; ++i) {
      Type a = outTypes[i], bty = ft.getResult(i);
      if (auto ma = a.dyn_cast<MemRefType>()) {
        auto mb = bty.cast<MemRefType>();
        outTypes[i] = maxRankedMemRefShape(ma, mb);
      } else {
        assert(a == bty && "non-memref result type mismatch");
      }
    }
  }

  // Determine loop depth from the first func.
  SmallVector<affine::AffineForOp, 4> band;
  collectOutermostBand(funcs.front(), band);
  loopDepth = band.size();

  // Append index-typed UB params at the end of inputs.
  SmallVector<Type> mergedInputs = inTypes;
  mergedInputs.resize(inTypes.size() + loopDepth, b.getIndexType());

  return b.getFunctionType(mergedInputs, outTypes);
}

/// Clone all ops in entry block up to (but excluding) the first affine.for/return.
void clonePrefixBeforeFirstFor(OpBuilder &b, Block *srcEntry,
                                      Block *dstEntry, IRMapping &map) {
  OpBuilder scoped = OpBuilder::atBlockEnd(dstEntry);
  for (Operation &op : *srcEntry) {
    if (isa<affine::AffineForOp>(op) || isa<func::ReturnOp>(op)) break;
    scoped.clone(op, map);
  }
}

/// Clone loop nest of `srcOuter`, wiring UBs from `ubParams` (outer->inner),
/// by building a fresh nest via affine::buildAffineLoopNest.
void cloneNestWithParamUBs(OpBuilder &builder,
                                                 affine::AffineForOp srcOuter,
                                                 IRMapping &map,
                                                 ArrayRef<Value> ubParams) {
  // 1) Collect the source band starting at srcOuter (outer -> inner).
  SmallVector<affine::AffineForOp, 4> srcBand;
  {
    auto cur = srcOuter;
    while (cur) {
      srcBand.push_back(cur);
      affine::AffineForOp next;
      cur.getBody()->walk([&](affine::AffineForOp inner) {
        if (inner->getParentOp() == cur) next = inner;
      });
      cur = next;
    }
  }
  assert(srcBand.size() == ubParams.size() && "ub param depth mismatch");

  Location loc = srcOuter.getLoc();
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToEnd(builder.getInsertionBlock());

  // 2) Prepare per-level LBs/UBs/steps for the helper (Value lbs, Value ubs).
  SmallVector<Value> lbs;
  SmallVector<Value> ubs;
  SmallVector<int64_t> steps;
  lbs.reserve(srcBand.size());
  ubs.reserve(srcBand.size());
  steps.reserve(srcBand.size());

  // Lower bounds: use 0 as Value (constant index 0), but if the original
  // loop had non-zero/mapped LB operands+map, you could compute it here instead.
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  for (unsigned i = 0; i < srcBand.size(); ++i) {
    lbs.push_back(c0);
    ubs.push_back(ubParams[i]);                 // UB is the i-th param Value.
    steps.push_back(srcBand[i].getStepAsInt());      // preserve original step.
  }

  // 3) Build the new loop nest. We capture srcBand by value to reach its
  //    innermost body for cloning once we have IVs.
  affine::AffineForOp newOuter = nullptr;
  affine::buildAffineLoopNest(
    builder, loc, lbs, ubs, steps,
    [&](OpBuilder &b, Location bodyLoc, ValueRange ivs) {
      // First invocation gives us the full set of IVs at innermost level.
      // Map each source IV -> new IV (outer..inner).
      assert(ivs.size() == srcBand.size() && "IV count mismatch");
      for (auto it : llvm::enumerate(srcBand))
        map.map(it.value().getInductionVar(), ivs[it.index()]);

      // Clone the innermost body (skip terminator).
      affine::AffineForOp innermostSrc = srcBand.back();
      for (Operation &op : *innermostSrc.getBody()) {
        if (op.hasTrait<OpTrait::IsTerminator>()) continue;
        b.clone(op, map);
      }
    });
}
} // end anonymous namespace

namespace mlir {
namespace streamhls {

/// ----------- Gather functions and their outermost loops from modules -----------
std::pair<SmallVector<func::FuncOp>, SmallVector<affine::AffineForOp>> gatherFuncsAndForOps(
  SmallVectorImpl<ModuleOp> &modules
) {
  SmallVector<func::FuncOp> funcOps;
  SmallVector<affine::AffineForOp> forOps;
  for(auto m : modules){
    auto func = getOnlyFunction(m);
    funcOps.push_back(func);
    // find the outermost affine.for in func
    affine::AffineForOp outermostFor;
    func.walk([&](affine::AffineForOp fo) {
      if (!fo->getParentOfType<affine::AffineForOp>())
        if (!outermostFor) outermostFor = fo;
    });
    assert(outermostFor && "no affine.for in source function");
    forOps.push_back(outermostFor);
  }
  return {funcOps, forOps};
}

/// ---------- Merge multiple kernel modules into a single module/function ----------
std::tuple<ModuleOp, AffineForOp, IRMapping> mergeKernels(
  SmallVectorImpl<ModuleOp> &modules, 
  OpBuilder &b,
  std::string moduleName
) {
  assert(!modules.empty() && "need at least one module to merge");
  MLIRContext *ctx = b.getContext();
  auto loc = b.getUnknownLoc();
  auto [funcOps, forOps] = gatherFuncsAndForOps(modules);
  // 1) Gather the per-module function handles (assumes one kernel per module).
  SmallVector<func::FuncOp, 8> funcs;
  funcs.reserve(modules.size());
  for (auto m : modules) funcs.push_back(getOnlyFunction(m));

  // 2) Create a fresh module to host the merged kernel.
  ModuleOp mergedModule = ModuleOp::create(loc);


  // 3) Build unified signature + UB params.
  unsigned loopDepth = 0;
  // for(auto func : funcs){
  //   llvm::dbgs() << "Function: " << func.getName() << "\n";
  //   llvm::dbgs() << "Function Type: " << func.getFunctionType() << "\n";
  // }
  auto unifiedTy = buildUnifiedTypeWithUBs(b, funcs, loopDepth);
  // llvm::dbgs() << "Unified function type: " << unifiedTy << "\n";
  // 4) Create merged func.
  b.setInsertionPointToStart(mergedModule.getBody());
  auto mergedFunc =
      b.create<func::FuncOp>(loc, StringAttr::get(ctx, moduleName), unifiedTy);
  Block *entry = mergedFunc.addEntryBlock();
  b.setInsertionPointToStart(entry);


  // 5) Map original block args -> merged block args for the first function.
  //    (We use f0 as the structural template.)
  func::FuncOp f0 = funcs.front();
  IRMapping map;
  auto oldNumInputs = f0.getFunctionType().getNumInputs();

  // Propagate argument attributes from the first function.
  for (unsigned i = 0; i < oldNumInputs; ++i) {
    if (auto dict = f0.getArgAttrDict(i)) {
      mergedFunc.setArgAttrs(i, dict.getValue());
    }
  }

  for (unsigned i = 0; i < oldNumInputs; ++i)
    map.map(f0.getBody().front().getArgument(i), entry->getArgument(i));

  // 6) Clone the prefix ops (e.g., arith.constant) before the first loop.
  clonePrefixBeforeFirstFor(b, &f0.getBody().front(), entry, map);

  // 7) Gather UB params from the tail of merged inputs.
  SmallVector<Value> ubParams;
  ubParams.reserve(loopDepth);
  auto totalInputs = mergedFunc.getFunctionType().getNumInputs();
  for (unsigned i = oldNumInputs; i < totalInputs; ++i)
    ubParams.push_back(entry->getArgument(i));

  // (Alternative: derive UBs from memref dims — replace above with `memref.dim`
  // on an input and pass those SSA values to cloneNestWithParamUBs.)

  // 8) Clone the loop nest, wiring UBs to the new params.
  // Find f0’s outermost loop.
  affine::AffineForOp outermostF0;
  f0.walk([&](affine::AffineForOp fo) {
    if (!fo->getParentOfType<affine::AffineForOp>())
      if (!outermostF0) outermostF0 = fo;
  });
  assert(outermostF0 && "no affine.for in source function");
  b.setInsertionPointToEnd(entry);

  // IRMapping blockArgsMapping;
  // auto blockArgs = funcOps[0].getBlocks().front().getArguments();
  // auto newBlockArgs = mergedFunc.getBlocks().front().getArguments();
  // for(auto argPair : llvm::zip(blockArgs, newBlockArgs)){
  //   auto arg = std::get<0>(argPair);
  //   auto newArg = std::get<1>(argPair);
  //   blockArgsMapping.map(arg, newArg);
  // }
  // llvm::dbgs() << "Num block args: " << blockArgs.size() << "\n";
  // llvm::dbgs() << "Num new block args: " << newBlockArgs.size() << "\n";
  // SmallVector<Value> newLoopUBs;
  // for(auto idx = blockArgs.size(); idx < newBlockArgs.size(); idx++){
  //   newLoopUBs.push_back(newBlockArgs[idx]);
  // }
  // llvm::dbgs() << "Num new loop UBs: " << newLoopUBs.size() << "\n";

  // auto newForOp = forOps[0].clone();

  // auto originalForOpBlock = forOps[0].getBody();
  // auto newForOpBlock = newForOp.getBody();

  // // insert the for loop into the function
  // entry->getOperations().insert(entry->begin(), newForOp);

  // newForOp->walk([&](Operation *op) {
  //   if(auto loadOp = dyn_cast<AffineLoadOp>(op)){
  //     auto memref = loadOp.getMemRef();
  //     auto newMemRef = map.lookupOrDefault(memref);
  //     // newMemRef.dump();
  //     loadOp.setMemRef(newMemRef);
  //     // loadOp.dump();
  //   }
  //   if(auto storeOp = dyn_cast<AffineStoreOp>(op)){
  //     auto memref = storeOp.getMemRef();
  //     auto newMemRef = map.lookupOrDefault(memref);
  //     // newMemRef.dump();
  //     storeOp.setMemRef(newMemRef);
  //     // storeOp.dump();
  //   }
  // });

  // AffineLoopBand band;
  // getLoopBandFromOutermost(newForOp, band);
  // int64_t idx = 0;
  // for(auto loop : band){
  //   // auto affineMap = AffineMap::get(0, 1, builder.getAffineSymbolExpr(0));
  //   auto affineMap = AffineMap::get(1, 0, b.getAffineDimExpr(0));
  //   loop.setUpperBound({newLoopUBs[idx]}, affineMap);
  //   idx++;
  // }

  cloneNestWithParamUBs(b, outermostF0, map, ubParams);

  AffineForOp newForOp;
  entry->walk([&](AffineForOp af) {
    if (!af->getParentOfType<AffineForOp>())
      if (!newForOp) newForOp = af;
  });

  // 9) Return.
  b.setInsertionPointToEnd(entry);
  b.create<func::ReturnOp>(loc);
  // mergedModule.dump();
  return {mergedModule, newForOp, map};
}

} // namespace streamhls
} // namespace mlir