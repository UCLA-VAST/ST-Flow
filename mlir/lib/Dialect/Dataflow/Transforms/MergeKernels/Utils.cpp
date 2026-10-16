#include "streamhls/Dialect/Dataflow/Transforms/MergeKernels/Utils.h"
#include <optional>

using namespace mlir;
using namespace mlir::dataflow;

#define DEBUG_TYPE "streamhls-merge-kernels-utils"

namespace {
  
/// Gather the outer-to-inner single loop band (assumes a single perfect nest).
void collectOutermostBand(func::FuncOp f, SmallVectorImpl<affine::AffineForOp> &band) {
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
// ====================== merge variants utils ======================

// Optionally collect loop info (depth + steps of the outer perfect nest).
// If you don’t want this, you can omit this completely.
DictionaryAttr collectLoopInfo(OpBuilder &b, func::FuncOp f) {
  SmallVector<affine::AffineForOp, 4> band;
  collectOutermostBand(f, band); // you already have this helper
  SmallVector<int64_t> ubs;
  ubs.reserve(band.size());
  for (auto fo : band) {
    ubs.push_back(fo.getConstantUpperBound());
  }

  return b.getDictionaryAttr({
    b.getNamedAttr("depth", b.getI64IntegerAttr((int64_t)band.size())),
    b.getNamedAttr("ubs", b.getDenseI64ArrayAttr(ubs)),
  });
}

std::pair<int64_t, SmallVector<int64_t, 4>> getLoopDepthAndUBs(DictionaryAttr loopInfo) {
  int64_t depth = -1;
  SmallVector<int64_t, 4> ubs;

  if (!loopInfo) return {depth, ubs};

  // depth
  if (auto depthAttr = loopInfo.getAs<IntegerAttr>("depth"))
    depth = depthAttr.getInt();

  // ubs: prefer DenseI64ArrayAttr, but accept ArrayAttr<int>
  if (auto dense = loopInfo.getAs<DenseI64ArrayAttr>("ubs")) {
    auto arr = dense.asArrayRef();              // ArrayRef<int64_t>
    ubs.assign(arr.begin(), arr.end());
  } 

  // If depth is absent but ubs is present, infer it.
  if (depth < 0 && !ubs.empty())
    depth = static_cast<int64_t>(ubs.size());

  return {depth, ubs};
}

/// Helpers that match MLIR affine semantics (floordiv/mod).
inline int64_t floorDiv64(int64_t a, int64_t b) {
  assert(b != 0 && "division by zero");
  if (b < 0) { a = -a; b = -b; }
  // floor division
  int64_t q = a / b;
  int64_t r = a % b;
  if (r != 0 && ((r > 0) != (b > 0))) --q;
  return q;
}
inline int64_t ceilDiv64(int64_t a, int64_t b) {
  assert(b != 0 && "division by zero");
  if (b < 0) { a = -a; b = -b; }
  // ceil(a/b) = -floor((-a)/b)
  return -floorDiv64(-a, b);
}
inline int64_t floorMod64(int64_t a, int64_t b) {
  assert(b != 0 && "mod by zero");
  int64_t q = floorDiv64(a, b);
  return a - q * b;
}

/// Evaluate an AffineExpr with 0 dims and N symbols given symbol values.
int64_t evalExpr(AffineExpr e, ArrayRef<int64_t> symVals) {
  switch (e.getKind()) {
  case AffineExprKind::Constant:
    return e.cast<AffineConstantExpr>().getValue();

  case AffineExprKind::DimId:
    llvm_unreachable("unexpected dim in a ()[...] map");

  case AffineExprKind::SymbolId: {
    unsigned idx = e.cast<AffineSymbolExpr>().getPosition();
    assert(idx < symVals.size() && "symbol index out of range");
    return symVals[idx];
  }

  case AffineExprKind::Add: {
    auto be = e.cast<AffineBinaryOpExpr>();
    return evalExpr(be.getLHS(), symVals) + evalExpr(be.getRHS(), symVals);
  }
  case AffineExprKind::Mul: {
    auto be = e.cast<AffineBinaryOpExpr>();
    return evalExpr(be.getLHS(), symVals) * evalExpr(be.getRHS(), symVals);
  }
  case AffineExprKind::FloorDiv: {
    auto be = e.cast<AffineBinaryOpExpr>();
    int64_t lhs = evalExpr(be.getLHS(), symVals);
    int64_t rhs = evalExpr(be.getRHS(), symVals);
    return floorDiv64(lhs, rhs);
  }
  case AffineExprKind::CeilDiv: {
    auto be = e.cast<AffineBinaryOpExpr>();
    int64_t lhs = evalExpr(be.getLHS(), symVals);
    int64_t rhs = evalExpr(be.getRHS(), symVals);
    return ceilDiv64(lhs, rhs);
  }
  case AffineExprKind::Mod: {
    auto be = e.cast<AffineBinaryOpExpr>();
    int64_t lhs = evalExpr(be.getLHS(), symVals);
    int64_t rhs = evalExpr(be.getRHS(), symVals);
    return floorMod64(lhs, rhs);
  }
  }
  llvm_unreachable("unknown AffineExprKind");
}

/// Evaluate a map of shape ()[s0] -> (EXPR) for a single symbol value.
/// Returns failure if the map shape is not (0 dims, 1 symbol, 1 result).
FailureOr<int64_t> evalUnarySymbolMap(AffineMap map, int64_t symVal) {
  if (map.getNumDims() != 0 || map.getNumSymbols() != 1 || map.getNumResults() != 1)
    return failure();
  AffineExpr e = map.getResult(0);
  int64_t v = evalExpr(e, /*symVals=*/ArrayRef<int64_t>{symVal});
  return v;
}

static void collectDimIds(mlir::AffineExpr expr,
                          llvm::SmallVectorImpl<unsigned> &out) {
  using namespace mlir;
  switch (expr.getKind()) {
  case AffineExprKind::DimId: {
    auto d = expr.cast<AffineDimExpr>();
    out.push_back(d.getPosition());
    break;
  }
  case AffineExprKind::SymbolId:
  case AffineExprKind::Constant:
    // no dims
    break;
  case AffineExprKind::Add:
  case AffineExprKind::Mul:
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv:
  case AffineExprKind::Mod: {
    auto bin = expr.cast<AffineBinaryOpExpr>();
    collectDimIds(bin.getLHS(), out);
    collectDimIds(bin.getRHS(), out);
    break;
  }
  }
}

using Bounds = std::pair<int64_t, int64_t>;

static std::optional<int64_t> inferMaxRangeFromExprRecursive(AffineExpr expr, ArrayRef<int64_t> extentPerDim) {
    if (auto dimExpr = expr.dyn_cast<AffineDimExpr>()) {
        unsigned pos = dimExpr.getPosition();
        if (pos >= extentPerDim.size()) return std::nullopt;
        // Max value is extent - 1
        return extentPerDim[pos] - 1;
    }
    if (auto constExpr = expr.dyn_cast<AffineConstantExpr>()) {
        return constExpr.getValue();
    }
    if (auto binOp = expr.dyn_cast<AffineBinaryOpExpr>()) {
        auto lhs = inferMaxRangeFromExprRecursive(binOp.getLHS(), extentPerDim);
        auto rhs = inferMaxRangeFromExprRecursive(binOp.getRHS(), extentPerDim);
        if (!lhs || !rhs) return std::nullopt;

        int64_t l = *lhs;
        int64_t r = *rhs;

        switch (expr.getKind()) {
        case AffineExprKind::Add:
            return l + r;
        case AffineExprKind::Mul:
            return l * r;
        case AffineExprKind::FloorDiv:
            if (r == 0) return std::nullopt;
            return l / r;
        case AffineExprKind::CeilDiv:
             if (r == 0) return std::nullopt;
             return (l + r - 1) / r;            
        case AffineExprKind::Mod:
            // Max value of (a mod b) is b - 1
            return r - 1;
        default:
            return std::nullopt;
        }
    }
    return std::nullopt;
}

FailureOr<std::pair<AffineMap, SmallVector<Bounds>>>
buildAddressLinearMapFromAccess(Operation *accessOp,
                                ArrayRef<Value> keptIVs,
                                ArrayRef<Bounds> mapExtents) {
  MLIRContext *ctx = accessOp->getContext();

  // 1) Get original index map + operands.
  AffineMap origMap;
  SmallVector<Value, 8> origOperands;

  if (auto ld = dyn_cast<affine::AffineLoadOp>(accessOp)) {
    origMap = ld.getAffineMap();
    origOperands = llvm::to_vector(ld.getMapOperands());
  } else if (auto st = dyn_cast<affine::AffineStoreOp>(accessOp)) {
    origMap = st.getAffineMap();
    origOperands = llvm::to_vector(st.getMapOperands());
  } else {
    return failure();
  }
  // extents
  LLVM_DEBUG(
    llvm::dbgs() << "Extents: ";
    for (auto [lo, hi] : mapExtents)
      llvm::dbgs() << "[" << lo << "," << hi << ") ";
    llvm::dbgs() << "\n";
    // original map
    llvm::dbgs() << "Original Map: " << origMap << "\n";
  );
  const unsigned oldDims = origMap.getNumDims();
  const unsigned oldSyms = origMap.getNumSymbols();
  if (origOperands.size() != oldDims + oldSyms)
    return failure();

  const unsigned rank = origMap.getNumResults();
  if (keptIVs.size() != mapExtents.size()) {
    llvm::errs() << "buildAddressLinearMapFromAccess: keptIVs/mapExtents mismatch\n";
    return failure();
  }

  const unsigned numDims = keptIVs.size();

  // 2) Map each kept IV to its band index (dim id in the *new* map space).
  //    dim i <=> keptIVs[i] (band order) – this preserves your d0,d1,... pattern.
  llvm::DenseMap<Value, unsigned> bandIndexOf;
  for (unsigned i = 0; i < numDims; ++i)
    bandIndexOf[keptIVs[i]] = i;

  // 3) Build dim/sym replacements:
  //    - If original dim operand is one of keptIVs, map it to the dim at its
  //      band index i.
  //    - Otherwise, map to constant 0.
  SmallVector<AffineExpr, 8> dimRepls(oldDims), symRepls(oldSyms);
  auto c0 = getAffineConstantExpr(0, ctx);

  for (unsigned i = 0; i < oldDims; ++i) {
    Value v = origOperands[i];
    auto it = bandIndexOf.find(v);
    if (it != bandIndexOf.end()) {
      unsigned dimPos = it->second; // band order (0..numDims-1)
      dimRepls[i] = getAffineDimExpr(dimPos, ctx);
    } else {
      dimRepls[i] = c0;
    }
  }

  for (unsigned s = 0; s < oldSyms; ++s)
    symRepls[s] = c0;

  // 4) Rewrite each index expression in terms of dims 0..numDims-1 (band order).
  SmallVector<AffineExpr, 4> newIdxExprs;
  newIdxExprs.reserve(rank);
  for (AffineExpr e : origMap.getResults())
    newIdxExprs.push_back(e.replaceDimsAndSymbols(dimRepls, symRepls));
  // At this point, for your example:
  //   newIdxExprs[0] = d4 + d0 + d2
  //   newIdxExprs[1] = d5 + d1 + d3

  // 5) Build reorderedMapExtents in "map dim order" (as you had before).
  SmallVector<Bounds> reorderedMapExtents;
  reorderedMapExtents.reserve(numDims);
  for (unsigned d = 0; d < oldDims; ++d) {
    Value dimVal = origOperands[d]; // original map dim operand in map order
    auto it = bandIndexOf.find(dimVal);
    if (it == bandIndexOf.end())
      continue; // not one of kept IVs
    unsigned bandIdx = it->second;           // index in keptIVs/mapExtents
    reorderedMapExtents.push_back(mapExtents[bandIdx]);
  }
  // For your example, this gives: [0,1) [0,128) [0,1) [0,1) [0,64) [0,1)

  // 6) Compute loop extent per dim in *band order* (used for logical shape).
  SmallVector<int64_t, 8> extentPerDim(numDims, 1);
  for (unsigned d = 0; d < numDims; ++d) {
    auto [lo, hi] = mapExtents[d]; // mapExtents is in band order
    int64_t e = hi - lo;
    if (e <= 0) e = 1;
    extentPerDim[d] = e;
  }

  // 7) Infer logical shape per memref dimension from newIdxExprs.
  SmallVector<int64_t, 4> logicalShape(rank, 1);
  for (unsigned r = 0; r < rank; ++r) {
    LLVM_DEBUG({
      llvm::dbgs() << "\n[LogicalShape] Analyzing result " << r << "\n";
      llvm::dbgs() << "  Expr: " << newIdxExprs[r] << "\n";
    });

    // Use the recursive inference to handle floorDiv and mod
    // We infer MAX value, so extent is max + 1
    if (auto inferred = inferMaxRangeFromExprRecursive(newIdxExprs[r], extentPerDim)) {
      logicalShape[r] = *inferred + 1;
      LLVM_DEBUG(llvm::dbgs() << "  -> Inferred extent: " << logicalShape[r] << "\n");
    } else {
      assert(false && "unable to infer logical shape from index expression");
      // // Emergency Fallback: only if the expression is too complex for the recursion
      // llvm::SmallVector<unsigned, 8> usedDims;
      // collectDimIds(newIdxExprs[r], usedDims);
      // int64_t prod = 1;
      // for (unsigned d : usedDims) {
      //   if (d < numDims) prod *= extentPerDim[d];
      // }
      // logicalShape[r] = std::max<int64_t>(prod, 1);
      // LLVM_DEBUG(llvm::dbgs() << "  -> Fallback (product) extent: " << logicalShape[r] << "\n");
    }
  }

  LLVM_DEBUG(
    llvm::dbgs() << "Logical Shape: ";
    for (int64_t s : logicalShape)
      llvm::dbgs() << s << "x";
    llvm::dbgs() << "\n";
  );
  // // For your example:
  // //   result 0 uses dims {4,0,2} with extents {1,128,1} → logicalShape[0] = 128
  // //   result 1 uses dims {5,1,3} with extents {1,64,1}  → logicalShape[1] = 64

  // 8) Row-major strides from logicalShape (NOT from memref type).
  SmallVector<int64_t, 4> strides(rank, 1);
  int64_t s = 1;
  for (int i = (int)rank - 1; i >= 0; --i) {
    strides[i] = s;
    int64_t dim = logicalShape[i];
    if (dim <= 0) dim = 1;
    s *= dim;
  }
  // For {128, 64}: strides = {64, 1}.

  // 9) Compose linear address: Σ_p newIdxExprs[p] * strides[p]
  AffineExpr addr = c0;
  for (unsigned p = 0; p < rank; ++p) {
    addr = addr + newIdxExprs[p] *
                     getAffineConstantExpr(strides[p], ctx);
  }
  // In your example:
  //   addr = (d4 + d0 + d2)*64 + (d5 + d1 + d3)

  // 10) Final map: dims = keptIVs.size() (band order), no symbols.
  AffineMap linearMap =
      AffineMap::get(/*dims=*/numDims, /*syms=*/0, addr, ctx);
  LLVM_DEBUG(
    llvm::dbgs() << "Linearized Map: " << linearMap << "\n";
    llvm::dbgs() << "\n";
  );

  return std::make_pair(linearMap, std::move(reorderedMapExtents));
}

} // namespace


namespace mlir {
namespace streamhls {
AffineExpr getConstOne(MLIRContext *ctx) { return getAffineConstantExpr(1, ctx); }
AffineExpr getConstZero(MLIRContext *ctx) { return getAffineConstantExpr(0, ctx); }
// ---------- get min and max loop stats across all loop bands ----------
void getMinAndMaxLoopStatsMerge(SmallVectorImpl<LoopBandInfo> &allLoopBandsInfo, LoopBandInfo &minBandInfo, LoopBandInfo &maxBandInfo) {
  for(auto bandInfo : allLoopBandsInfo){      
    LLVM_DEBUG(
      llvm::dbgs() << "Curr Loop ubs: ";
      for(auto loopInfo : bandInfo){
        llvm::dbgs() << loopInfo.ub << " ";
      }
      llvm::dbgs() << "\n";
    );
    for(auto loopInfoZip : llvm::zip(minBandInfo, bandInfo)){
      auto &minLoopInfo = std::get<0>(loopInfoZip);
      auto loopInfo = std::get<1>(loopInfoZip);
      if(minLoopInfo.lb > loopInfo.lb){
        minLoopInfo.lb = loopInfo.lb;
      }
      if(minLoopInfo.ub > loopInfo.ub){
        minLoopInfo.ub = loopInfo.ub;
      }
    }
    for(auto loopInfoZip : llvm::zip(maxBandInfo, bandInfo)){
      auto &maxLoopInfo = std::get<0>(loopInfoZip);
      auto loopInfo = std::get<1>(loopInfoZip);
      if(maxLoopInfo.lb < loopInfo.lb){
        maxLoopInfo.lb = loopInfo.lb;
      }
      if(maxLoopInfo.ub < loopInfo.ub){
        maxLoopInfo.ub = loopInfo.ub;
      }
    }
    LLVM_DEBUG(
      llvm::dbgs() << "Min Loop info: ";
      for(auto loopInfo : minBandInfo){
        llvm::dbgs() << loopInfo.ub << " ";
      }
      llvm::dbgs() << "\n\n";
    );
  }
}

// // Example merge.variant attribute
// // Wrapped merged module:
// // #map = affine_map<(d0, d1) -> (d0 * 2 + d1)>
// // #map1 = affine_map<(d0, d1) -> (d0 * 64 + d1)>
// // #map2 = affine_map<(d0, d1) -> (d0 * 128 + d1)>
// // #map3 = affine_map<(d0, d1) -> (d0 * 256 + d1)>
// // #map4 = affine_map<(d0) -> (d0)>
// // #set = affine_set<(d0) : (d0 == 0)>
// // #set1 = affine_set<(d0) : (d0 - 127 == 0)>
// // module attributes {
// //   merge.variants = [
// //     {
// //       access_info_list = [
// //         {access.info = #dataflow.access_info<load, #map, [#dataflow.access_bound<0, 128>, #dataflow.access_bound<0, 2>], 256>}, 
// //         {access.info = #dataflow.access_info<load, #map1, [#dataflow.access_bound<0, 128>, #dataflow.access_bound<0, 64>], 8192>}, 
// //         {access.info = #dataflow.access_info<store, #map, [#dataflow.access_bound<0, 64>, #dataflow.access_bound<0, 2>], 128>}
// //       ], 
// //       buffer_info_list = [
// //         {buffer.info = {name = "buf33", type = "shared-buffer"}}, 
// //         {buffer.info = {name = "buf17", type = "shared-buffer"}}, 
// //         {buffer.info = {name = "buf35", type = "shared-buffer"}}
// //       ], 
// //       loop = {depth = 3 : i64, ubs = array<i64: 2, 64, 128>}
// //     }, 
// //     {
// //       access_info_list = [
// //         {access.info = #dataflow.access_info<load, #map, [#dataflow.access_bound<0, 64>, #dataflow.access_bound<0, 2>], 128>}, 
// //         {access.info = #dataflow.access_info<load, #map2, [#dataflow.access_bound<0, 64>, #dataflow.access_bound<0, 128>], 8192>}, 
// //         {access.info = #dataflow.access_info<store, #map, [#dataflow.access_bound<0, 128>, #dataflow.access_bound<0, 2>], 256>}
// //       ], 
// //       buffer_info_list = [
// //         {buffer.info = {name = "buf37", type = "shared-buffer"}}, 
// //         {buffer.info = {name = "buf19", type = "shared-buffer"}}, 
// //         {buffer.info = {name = "buf39", type = "shared-buffer"}}
// //       ], 
// //       loop = {depth = 3 : i64, ubs = array<i64: 2, 128, 64>}
// //     }, 
// //     {
// //       access_info_list = [
// //         {access.info = #dataflow.access_info<load, #map, [#dataflow.access_bound<0, 128>, #dataflow.access_bound<0, 2>], 256>}, 
// //         {access.info = #dataflow.access_info<load, #map3, [#dataflow.access_bound<0, 128>, #dataflow.access_bound<0, 256>], 32768>}, 
// //         {access.info = #dataflow.access_info<store, #map, [#dataflow.access_bound<0, 256>, #dataflow.access_bound<0, 2>], 512>}
// //       ], 
// //       buffer_info_list = [
// //         {buffer.info = {name = "buf41", type = "shared-buffer"}}, 
// //         {buffer.info = {name = "buf21", type = "shared-buffer"}}, 
// //         {buffer.info = {name = "buf43", type = "shared-buffer"}}
// //       ], 
// //       loop = {depth = 3 : i64, ubs = array<i64: 2, 256, 128>}
// //     }
// //   ]
// // } {
struct AccessOrder {
  SmallVector<Value, 8> ivs;        // outer -> inner varying IVs used by the access
  SmallVector<int64_t, 8> extents;  // per-iv extent (trip count)
  int64_t numAccess = 0;            // product of extents (0 if none)
  AffineMap linear;                 // row-major linearization over these dims
};

/// `mergedFunc` is the function whose body contains the merged accesses, and
/// `band` is its outermost loop band.
DictionaryAttr buildVariantMetadataForFunc(OpBuilder &b,
                                                  func::FuncOp srcFunc,
                                                  func::FuncOp mergedFunc,
                                                  const AffineLoopBand &band) {
  LLVM_DEBUG(
    llvm::dbgs() << "================================================START===================================================\n";
  );
  MLIRContext *ctx = b.getContext();
  AffineLoopBand srcBand;
  collectOutermostBand(srcFunc, srcBand); 
  LLVM_DEBUG(
    llvm::dbgs() << "Building variant metadata for merged function: "
                 << mergedFunc.getName() << "\n";
    auto topLoop = band.front();
    // topLoop.dump();
    // srcBand[0].dump();
    // srcFunc.dump();
    // mergedFunc.dump();
  );

  DictionaryAttr originalLoopInfo;
  SmallVector<affine::AffineForOp, 4> tmp;
  srcFunc.walk([&](affine::AffineForOp fo) {
    if (!fo->getParentOfType<affine::AffineForOp>())
      if (tmp.empty())
        tmp.push_back(fo);
  });
  originalLoopInfo = tmp.empty() ? b.getDictionaryAttr({})
                          : collectLoopInfo(b, srcFunc);                                 
  // --------------------------------------------------------------------------
  // 1) Build "loop" field: loopInfo derived from the source function.
  // --------------------------------------------------------------------------
  DictionaryAttr loopInfo;
  if(srcFunc == mergedFunc) {
    SmallVector<affine::AffineForOp, 4> tmp;
    srcFunc.walk([&](affine::AffineForOp fo) {
      if (!fo->getParentOfType<affine::AffineForOp>())
        if (tmp.empty())
          tmp.push_back(fo);
    });
    loopInfo = tmp.empty() ? b.getDictionaryAttr({})
                           : collectLoopInfo(b, srcFunc);
  }else{
    SmallVector<int64_t> ubs;
    ubs.reserve(band.size());
    for (auto foPair : llvm::enumerate(band)) {
      auto fo = foPair.value();
      auto idx = foPair.index();
      if(fo.hasConstantUpperBound()){
        ubs.push_back(fo.getConstantUpperBound());
      }else{
        auto srcForOp = srcBand[idx];
        if(srcForOp.hasConstantUpperBound()){
          auto constUB = srcForOp.getConstantUpperBound();
          auto res = evalUnarySymbolMap(fo.getUpperBoundMap(), constUB);
          if (failed(res)) {
            llvm::errs() << "Failed to eval UB map\n";
            assert(false && "Failed to eval UB map");
          } else {
            ubs.push_back(*res);
          }
        }else{
          assert(false && "Non-constant upper bound in source function");
        }
      }
    }

    loopInfo = b.getDictionaryAttr({
      b.getNamedAttr("depth", b.getI64IntegerAttr((int64_t)band.size())),
      b.getNamedAttr("ubs", b.getDenseI64ArrayAttr(ubs)),
    });
  }
  LLVM_DEBUG(
    llvm::dbgs() << "  Loop info: " << loopInfo << "\n";
  );

  // --------------------------------------------------------------------------
  // 2) Build "buffer_info_list" from srcFunc arguments (if any buffer.info attrs)
  // --------------------------------------------------------------------------
  SmallVector<Attribute> bufInfoWrapped;
  bufInfoWrapped.reserve(srcFunc.getNumArguments());
  for (unsigned a = 0, e = srcFunc.getNumArguments(); a < e; ++a) {
    if (auto dict =
            srcFunc.getArgAttrOfType<DictionaryAttr>(a, "buffer.info")) {
      bufInfoWrapped.push_back(
          b.getDictionaryAttr({b.getNamedAttr("buffer.info", dict)}));
    }
  }

  // --------------------------------------------------------------------------
  // 3) Build "access_info_list" from the merged function and its loop band.
  //    Loop UBs come from the per-variant loopInfo we just computed.
  // --------------------------------------------------------------------------
  auto [depth, ubs] = getLoopDepthAndUBs(loopInfo);
  LLVM_DEBUG(
    llvm::dbgs() << "Building access_info_list for merged function: "
                 << mergedFunc.getName() << "\n";
    llvm::dbgs() << "  Loop depth: " << depth << "\n";
    llvm::dbgs() << "  Loop UBs: ";
    for (auto u : ubs)
      llvm::dbgs() << u << " ";
    llvm::dbgs() << "\n";
  );

  SmallVector<Attribute> accessInfoWrapped;

  for (auto funcArg : mergedFunc.getArguments()) {
    LLVM_DEBUG(
      llvm::dbgs() << "Processing arg: " << funcArg << "\n";
    );
    if (!isa<MemRefType>(funcArg.getType()))
      continue;
    if (!funcArg.hasOneUse()) {
      llvm::errs()
          << "Argument has multiple uses, cannot annotate access info\n";
      continue;
    }

    Operation *loadOrStoreOp = *funcArg.getUsers().begin();
    LLVM_DEBUG(
      llvm::dbgs() << "  Found load/store op: " << *loadOrStoreOp << "\n";
    );
    // Reuse IVs from parent if (dims only).
    SmallVector<Value, 3> reuseIVs;
    if (auto parentIfOp =
            loadOrStoreOp->getParentOfType<affine::AffineIfOp>()) {
      IntegerSet set = parentIfOp.getIntegerSet();
      LLVM_DEBUG(
        llvm::dbgs() << "  Found parent if op with set: ";
        set.dump();
      );
      auto ops = parentIfOp.getOperands();
      reuseIVs.assign(ops.begin(), ops.begin() + set.getNumDims());
    }

    // Kind + original map/operands.
    AccessKind kind;
    SmallVector<Value> origMapOperands;
    AffineMap origMap;
    if (auto loadOp = dyn_cast<AffineLoadOp>(loadOrStoreOp)) {
      origMapOperands = loadOp.getMapOperands();
      origMap = loadOp.getAffineMap();
      kind = AccessKind::load;
    } else if (auto storeOp = dyn_cast<AffineStoreOp>(loadOrStoreOp)) {
      origMapOperands = storeOp.getMapOperands();
      origMap = storeOp.getAffineMap();
      kind = AccessKind::store;
    } else {
      llvm::errs() << "Unsupported op using the memref argument\n";
      continue;
    }

    // 1) Kept IVs (band order) + per-iv map extents.
    SmallVector<Value, 8> keptIVs;
    SmallVector<std::pair<int64_t, int64_t>> mapExtents;
    for (auto forOpPair : llvm::enumerate(band)) {
      unsigned bidx = forOpPair.index();
      auto forOp = forOpPair.value();
      Value iv = forOp.getInductionVar();

      if (llvm::is_contained(reuseIVs, iv))
        continue; // pinned
      // if (llvm::is_contained(origMapOperands, iv)){
        keptIVs.push_back(iv);
        mapExtents.emplace_back(0, ubs[bidx]);
      // }
      // if (forOp.hasConstantBounds()) {
      //   mapExtents.emplace_back(forOp.getConstantLowerBound(),
      //                           forOp.getConstantUpperBound());
      // } else {
      //   auto res = evalUnarySymbolMap(forOp.getUpperBoundMap(), ubs[bidx]);
      //   if (failed(res)) {
      //     llvm::errs() << "Failed to eval UB map\n";
      //     mapExtents.emplace_back(0, 1);
      //   } else {
      //     mapExtents.emplace_back(0, *res);
      //   }
      // }
    }
    // for(auto iv : origMapOperands){
    //   LLVM_DEBUG(
    //     llvm::dbgs() << "  Orig map operand: " << iv << "\n";
    //   );
    // }
    LLVM_DEBUG(
      origMap.dump();
      llvm::dbgs() << "  Num Reuse IVs: " << reuseIVs.size() << "\n";
      llvm::dbgs() << "  Num Kept IVs: " << keptIVs.size() << "\n";
    );
    for(auto [lo, hi] : mapExtents){
      LLVM_DEBUG(
        llvm::dbgs() << "    Map extent: [" << lo << "," << hi << ")\n";
      );
    }


    // 2) Address linear map + reordered extents (map-dim order).
    
    auto resultOr =
        buildAddressLinearMapFromAccess(loadOrStoreOp, keptIVs, mapExtents);
    if (failed(resultOr)) {
      llvm::errs() << "Failed to build address-linear map for access\n";
      continue;
    }
    auto [linearMap, reorderedMapExtents] = *resultOr;

    // 3) Count (product over all band IVs; if you want only kept, change source).
    SmallVector<int64_t, 8> extents;
    for (auto forOpPair : llvm::enumerate(band)) {
      unsigned bidx = forOpPair.index();
      auto forOp = forOpPair.value();
      int64_t extent = 1;
      if (!llvm::is_contained(reuseIVs, forOp.getInductionVar())) {
        if (forOp.hasConstantBounds())
          extent = forOp.getConstantUpperBound();
        else if (auto res =
                     evalUnarySymbolMap(forOp.getUpperBoundMap(), ubs[bidx]);
                 succeeded(res))
          extent = *res;
      }
      extents.push_back(extent);
    }
    int64_t numAccess = 1;
    for (auto [lo, hi] : mapExtents)
      numAccess *= std::max<int64_t>(hi - lo, 1);
    assert(band.size() % 3 == 0 && "Tiled loop nest must be multiple of 3");
    auto unrollStartIdx = band.size() - ((band.size() / 3) * 2);
    auto accessInfoAttr = AccessInfoAttr::get(
        ctx, kind, linearMap, mapExtents, numAccess, unrollStartIdx);
    // print memref op, attribute, and function it came from
    // llvm::errs() << "MemRef op: " << *loadOrStoreOp << "\n";
    // llvm::errs() << "Access Info Attr: " << accessInfoAttr << "\n";
    // llvm::errs() << "From function: " << mergedFunc << "\n\n";
    accessInfoWrapped.push_back(
        b.getDictionaryAttr({b.getNamedAttr("access.info", accessInfoAttr)}));
  }

  // --------------------------------------------------------------------------
  // 4) Assemble final dictionary: loop + buffer_info_list + access_info_list.
  // --------------------------------------------------------------------------
  NamedAttrList fields;
  fields.append("loop", originalLoopInfo);
  if (!bufInfoWrapped.empty())
    fields.append("buffer_info_list", b.getArrayAttr(bufInfoWrapped));
  if (!accessInfoWrapped.empty())
    fields.append("access_info_list", b.getArrayAttr(accessInfoWrapped));
  LLVM_DEBUG(
    llvm::dbgs() << "================================================END===================================================\n";
  );
  return b.getDictionaryAttr(fields);
}

/// Top-level function: attach "merge.variants" to the merged module.
void attachVariantMetadata(ModuleOp mergedModule,
                           ArrayRef<func::FuncOp> funcs) {
  OpBuilder b(mergedModule.getContext());

  // Access pattern info (same for the merged function’s body, UBs vary per variant).
  func::FuncOp mergedFunc = *mergedModule.getOps<func::FuncOp>().begin();
  AffineLoopBand band;
  collectOutermostBand(mergedFunc, band);

  // Build all variant dictionaries.
  SmallVector<Attribute> variants;
  variants.reserve(funcs.size());
  for (func::FuncOp f : funcs)
    variants.push_back(buildVariantMetadataForFunc(b, f, mergedFunc, band));

  // Attach the final array.
  mergedModule->setAttr("merge.variants", b.getArrayAttr(variants));
}

} // end namespace streamhls
} // end namespace mlir