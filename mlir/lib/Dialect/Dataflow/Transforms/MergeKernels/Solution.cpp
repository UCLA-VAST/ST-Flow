#include "streamhls/Dialect/Dataflow/Transforms/MergeKernels/Utils.h"
#include "streamhls/Dialect/Dataflow/Transforms/MergeKernels/Solution.h"

using namespace mlir;
using namespace streamhls;

#define DEBUG_TYPE "streamhls-merge-kernels-solution"

namespace {
//===----------------------------------------------------------------------===//
// Small helpers to keep getMemRefInfo lean
//===----------------------------------------------------------------------===//


/// Merge two affine maps into a single map by concatenating inputs and results.
AffineMap mergeAffineMaps(const AffineMap &mapA, const AffineMap &mapB) {
  MLIRContext *ctx = mapA.getContext();

  unsigned aDims = mapA.getNumDims();
  unsigned aSyms = mapA.getNumSymbols();
  unsigned bDims = mapB.getNumDims();
  unsigned bSyms = mapB.getNumSymbols();

  unsigned totalDims = aDims + bDims;
  unsigned totalSyms = aSyms + bSyms;

  // Remap A's dims/syms to [0, ..., aDims-1] and [totalDims, ..., totalDims+aSyms-1]
  SmallVector<AffineExpr> aDimReplacements, aSymReplacements;
  for (unsigned i = 0; i < aDims; ++i)
    aDimReplacements.push_back(getAffineDimExpr(i, ctx));
  for (unsigned i = 0; i < aSyms; ++i)
    aSymReplacements.push_back(getAffineSymbolExpr(totalDims + i, ctx));

  // Remap B's dims/syms to [aDims, ..., aDims+bDims-1] and [totalDims+aSyms, ..., totalSyms-1]
  SmallVector<AffineExpr> bDimReplacements, bSymReplacements;
  for (unsigned i = 0; i < bDims; ++i)
    bDimReplacements.push_back(getAffineDimExpr(aDims + i, ctx));
  for (unsigned i = 0; i < bSyms; ++i)
    bSymReplacements.push_back(getAffineSymbolExpr(totalDims + aSyms + i, ctx));

  // Apply replacements to both maps
  auto mapAReplaced = mapA.replaceDimsAndSymbols(aDimReplacements, aSymReplacements, totalDims, totalSyms);
  auto mapBReplaced = mapB.replaceDimsAndSymbols(bDimReplacements, bSymReplacements, totalDims, totalSyms);

  // Combine results
  SmallVector<AffineExpr> combinedResults;
  combinedResults.append(mapAReplaced.getResults().begin(), mapAReplaced.getResults().end());
  combinedResults.append(mapBReplaced.getResults().begin(), mapBReplaced.getResults().end());

  return AffineMap::get(totalDims, totalSyms, combinedResults, ctx);
}

/// Compute the effective UB for a loop `ubIdx` (1-based index into the *tiled*
/// band). If the source loop has a constant UB, return it. Otherwise, derive it
/// from the max band UB by dividing by Tm*Tu (first and second tiling levels).
int64_t resolveEffectiveUB(unsigned ubIdx1Based,
                                  AffineLoopBand &allLoops,
                                  const LoopBandInfo &maxBandInfo) {
  unsigned i = ubIdx1Based - 1; // to 0-based
  if (allLoops[i].hasConstantUpperBound())
    return allLoops[i].getConstantUpperBound();

  // Fallback: derive from the max band & tiles (same logic you had).
  auto numLoops = static_cast<unsigned>(maxBandInfo.size());
  int64_t maxUB   = maxBandInfo[i].ub;
  int64_t tmUBVal = allLoops[i + numLoops].getConstantUpperBound();
  int64_t tuUBVal = allLoops[i + 2 * numLoops].getConstantUpperBound();
  return maxUB / (tmUBVal * tuUBVal);
}

// AffineExpr resolveEffectiveUBExpr(unsigned ubIdx1Based,
//                                          AffineLoopBand &allLoops,
//                                          const LoopBandInfo &maxBandInfo,
//                                          OpBuilder &builder) {
//   unsigned i = ubIdx1Based - 1; // to 0-based
//   if (allLoops[i].hasConstantUpperBound())
//     return builder.getAffineConstantExpr(allLoops[i].getConstantUpperBound());

//   // Fallback: derive from the max band & tiles (same logic you had).
//   auto numLoops = static_cast<unsigned>(maxBandInfo.size());
//   AffineExpr maxUBExpr = builder.getAffineConstantExpr(maxBandInfo[i].ub);
//   AffineExpr tmUBExpr = builder.getAffineConstantExpr(
//       allLoops[i + numLoops].getConstantUpperBound());
//   AffineExpr tuUBExpr = builder.getAffineConstantExpr(
//       allLoops[i + 2 * numLoops].getConstantUpperBound());
//   return maxUBExpr.floorDiv(tmUBExpr * tuUBExpr);
// }

/// Replace *all* symbols in `expr` with constants given by UB indices.
/// Assumes the AffineExpr uses symbols in the same order as `ubOperands`.
AffineExpr replaceSymbolsWithConstantUBs(AffineExpr expr,
                                                ArrayRef<unsigned> ubOperands,
                                                OpBuilder &builder,
                                                AffineLoopBand &allLoops,
                                                const LoopBandInfo &maxBandInfo) {
  SmallVector<AffineExpr, 8> repl;
  repl.reserve(ubOperands.size());
  for (unsigned ubIdx1Based : ubOperands) {
    int64_t ub = resolveEffectiveUB(ubIdx1Based, allLoops, maxBandInfo);
    repl.push_back(builder.getAffineConstantExpr(ub));
  }
  return expr.replaceSymbols(repl);
}

/// Replace symbols in `expr` with *upper-bound expressions* (not constants).
/// It also collects the *map operands* needed by those UB expressions.
/// This is used for RHS where you want dynamic UB composition.
AffineExpr replaceSymbolsWithUBMaps(AffineExpr expr,
                                           ArrayRef<unsigned> ubOperands,
                                           OpBuilder &builder,
                                           AffineLoopBand &allLoops,
                                           SmallVectorImpl<Value> &outOperands) {
  SmallVector<AffineExpr, 8> repl;
  repl.reserve(ubOperands.size());
  for (unsigned ubIdx1Based : ubOperands) {
    unsigned i = ubIdx1Based - 1;
    AffineMap ubMap = allLoops[i].getUpperBoundMap();
    // We only support single-result UB maps here (same as your code).
    assert(ubMap.getNumResults() == 1 && "Expected single-result UB map");
    repl.push_back(ubMap.getResult(0));
    // Append its operands to the final set operands.
    for (Value v : allLoops[i].getUpperBoundOperands())
      outOperands.push_back(v);
  }
  return expr.replaceSymbols(repl);
}

/// Build an affine map + operands from one access spec.
///  - Replaces UB symbols with constants (LHS-like behavior).
///  - Map dims = number of IV operands referenced by the access.
///  - Map results = the (now symbol-free) access expr.
std::pair<AffineMap, SmallVector<Value>>
buildAccessMapAndOperands(const AffineExprInfo &access,
                          OpBuilder &builder,
                          AffineLoopBand &allLoops,
                          const LoopBandInfo &maxBandInfo) {
  AffineExpr expr = replaceSymbolsWithConstantUBs(access.expr, access.ubOperands,
                                                  builder, allLoops, maxBandInfo);
  auto map = AffineMap::get(/*dims=*/access.ivOperands.size(),
                            /*symbols=*/0, expr, builder.getContext());
  SmallVector<Value> ops;
  ops.reserve(access.ivOperands.size());
  for (unsigned iv1Based : access.ivOperands)
    ops.push_back(allLoops[iv1Based - 1].getInductionVar());
  return {map, std::move(ops)};
}

/// Merge a list of maps (same dim/sym counts) into a single product map and
/// concatenate operands. Your `mergeAffineMaps` composes results; we keep it.
std::pair<AffineMap, SmallVector<Value>>
mergeMapsAndOperands(ArrayRef<AffineMap> maps,
                     ArrayRef<SmallVector<Value>> opsPerMap) {
  assert(!maps.empty() && "Need at least one map");
  AffineMap merged = maps.front();
  SmallVector<Value> mergedOps = opsPerMap.front();
  for (size_t i = 1; i < maps.size(); ++i) {
    merged = mergeAffineMaps(merged, maps[i]);
    mergedOps.append(opsPerMap[i].begin(), opsPerMap[i].end());
  }
  return {merged, std::move(mergedOps)};
}

/// Build (loadSet, loadSetOperands) from the LHS guards:
///  - Every LHS constraint is assumed "== 0" (eqFlag=true) per your code.
std::pair<IntegerSet, SmallVector<Value>>
buildLoadGuardSet(ArrayRef<AffineExprInfo> lhsList,
                  OpBuilder &builder,
                  AffineLoopBand &allLoops,
                  const LoopBandInfo &maxBandInfo) {
  MLIRContext *ctx = builder.getContext();

  // 1) Build a shared dim list: union of all IVs used across constraints.
  SmallVector<unsigned, 16> sharedIvIdx1Based; // stable order
  auto getOrAddSharedPos = [&](unsigned iv1Based) -> unsigned {
    auto it = llvm::find(sharedIvIdx1Based, iv1Based);
    if (it != sharedIvIdx1Based.end())
      return (unsigned)std::distance(sharedIvIdx1Based.begin(), it);
    sharedIvIdx1Based.push_back(iv1Based);
    return (unsigned)(sharedIvIdx1Based.size() - 1);
  };

  for (const auto &lhs : lhsList)
    for (unsigned iv1Based : lhs.ivOperands)
      (void)getOrAddSharedPos(iv1Based);

  // 2) Build operands once, matching shared dim list.
  SmallVector<Value> operands;
  operands.reserve(sharedIvIdx1Based.size());
  for (unsigned iv1Based : sharedIvIdx1Based)
    operands.push_back(allLoops[iv1Based - 1].getInductionVar());

  // 3) Build constraints with proper dim remapping into the shared dim space.
  SmallVector<AffineExpr> exprs;
  SmallVector<bool> eqFlags;
  exprs.reserve(lhsList.size());
  eqFlags.reserve(lhsList.size());

  for (const auto &lhs : lhsList) {
    // Replace UB symbols with constants.
    AffineExpr e = replaceSymbolsWithConstantUBs(lhs.expr, lhs.ubOperands,
                                                 builder, allLoops, maxBandInfo);
    e = floorToCeilDiv(e);

    // Remap dims used by this expr to the shared dim positions.
    //
    // IMPORTANT: This assumes lhs.expr dims are 0..(lhs.ivOperands.size()-1),
    // aligned with lhs.ivOperands order.
    SmallVector<AffineExpr, 8> dimRepl;
    dimRepl.reserve(lhs.ivOperands.size());
    for (unsigned local = 0; local < lhs.ivOperands.size(); ++local) {
      unsigned iv1Based = lhs.ivOperands[local];
      unsigned sharedPos = getOrAddSharedPos(iv1Based); // already present
      dimRepl.push_back(getAffineDimExpr(sharedPos, ctx));
    }
    e = e.replaceDims(dimRepl);

    exprs.push_back(e);
    eqFlags.push_back(true);
  }

  unsigned numDims = sharedIvIdx1Based.size();
  return {IntegerSet::get(numDims, /*numSymbols=*/0, exprs, eqFlags),
          std::move(operands)};
}

// std::pair<IntegerSet, SmallVector<Value>>
// buildLoadGuardSet(ArrayRef<AffineExprInfo> lhsList,
//                   OpBuilder &builder,
//                   AffineLoopBand &allLoops,
//                   const LoopBandInfo &maxBandInfo) {
//   SmallVector<AffineExpr> exprs;
//   SmallVector<bool> eqFlags;
//   SmallVector<Value> operands;
//   for (const auto &lhs : lhsList) {
//     // Replace UB symbols with constants.
//     AffineExpr e = replaceSymbolsWithConstantUBs(lhs.expr, lhs.ubOperands,
//                                                  builder, allLoops, maxBandInfo);
//     e = floorToCeilDiv(e);
//     exprs.push_back(e);
//     eqFlags.push_back(/*isEq=*/true);
//     // Append IV operands (dims)
//     for (unsigned iv1Based : lhs.ivOperands)
//       operands.push_back(allLoops[iv1Based - 1].getInductionVar());
//   }
//   // All constraints share the same dims list, so we simply report dim count as
//   // the number of IV operands appended. (Matches your original construction.)
//   unsigned numDims = operands.size();
//   return {IntegerSet::get(numDims, /*numSymbols=*/0, exprs, eqFlags),
//           std::move(operands)};
// }

/// Build (storeSet, storeSetOperands) from paired LHS/RHS guards:
///  - For each (lhs, rhs): constraint is (lhs - rhs == 0),
///    where lhs uses constant-UB replacement, rhs uses UB maps with operands.
///  - We return one IntegerSet with all constraints AND-ed.
std::pair<IntegerSet, SmallVector<Value>>
buildStoreGuardSet(ArrayRef<AffineExprInfo> lhsList,
                   ArrayRef<AffineExprInfo> rhsList,
                   OpBuilder &builder,
                   AffineLoopBand &allLoops) {
  assert(lhsList.size() == rhsList.size() && "lhs/rhs size mismatch");

  SmallVector<AffineExpr> exprs;
  SmallVector<bool> eqFlags;
  SmallVector<Value> operands;     // dims first, then symbols

  // We will append *all* IVs (dims) for every pair exactly as you did, then
  // append RHS UB operands (symbols). That preserves your original layout.
  unsigned totalSymbolCount = 0;

  for (auto [lhs, rhs] : llvm::zip(lhsList, rhsList)) {
    // 1) LHS: replace UB symbols with constant UBs.
    AffineExpr lhsE = lhs.expr;
    {
      SmallVector<AffineExpr, 8> repl;
      repl.reserve(lhs.ubOperands.size());
      for (unsigned ub1 : lhs.ubOperands) {
        unsigned i = ub1 - 1;
        int64_t ub = allLoops[i].getConstantUpperBound();
        repl.push_back(builder.getAffineConstantExpr(ub));
      }
      lhsE = lhsE.replaceSymbols(repl);
    }

    // 2) RHS: replace UB symbols with UB maps (and collect UB operands).
    SmallVector<Value> rhsUBOps;
    AffineExpr rhsE = replaceSymbolsWithUBMaps(rhs.expr, rhs.ubOperands,
                                               builder, allLoops, rhsUBOps);

    lhsE = floorToCeilDiv(lhsE);
    rhsE = floorToCeilDiv(rhsE);

    // 3) Form constraint: lhs - rhs == 0
    exprs.push_back(lhsE - rhsE);
    eqFlags.push_back(true);

    // 4) Dims (IVs) for this constraint — same order as your code.
    for (unsigned iv1Based : lhs.ivOperands)
      operands.push_back(allLoops[iv1Based - 1].getInductionVar());

    // 5) Symbols (UB map operands) for this constraint.
    operands.append(rhsUBOps.begin(), rhsUBOps.end());
    totalSymbolCount += rhsUBOps.size();
  }

  // numDims = total IVs appended; numSymbols = appended UB-operand count.
  unsigned numDims    = operands.size() - totalSymbolCount;
  unsigned numSymbols = totalSymbolCount;

  return {IntegerSet::get(numDims, numSymbols, exprs, eqFlags),
          std::move(operands)};
}
} // end anonymous namespace

namespace mlir {
namespace streamhls {
// ----------- Extract detailed memref info from AMPL result -----------
void getMemRefInfo(
  OpBuilder &builder,
  const std::string &result,
  AffineLoopBand &allLoops,
  const AllCandidates &loopCandidates,
  const LoopBandInfo &maxBandInfo,
  MemrefSolutions &memrefSolutions
){
  std::smatch match;

  // We only use the *first* loop band of candidates (keep your early break).
  const auto &memrefList = loopCandidates.front();

  for (auto memIt : llvm::enumerate(memrefList)) {
    const std::string &memrefName               = memIt.value().first;
    const SingleMemrefCandidates &memrefCands   = memIt.value().second;

    // Iterate all candidates for this memref; pick the one selected by the solver.
    for (auto candIt : llvm::enumerate(memrefCands)) {
      unsigned candidateIdx = candIt.index();
      const MemrefCandidateInfo &cand = candIt.value();

      // If indicator var "<mem>_rl_<idx> = 1" is present in `result`, we select it.
      const std::string indVar = memrefName + "_rl_" + std::to_string(candidateIdx);
      // const std::regex  indRe(indVar + " = 1");
      // if (!std::regex_search(result, match, indRe))
      //   continue;

      // llvm::dbgs() << indVar << " = 1\n";

      MemRefSolution S;
      S.memref = cand.memref;
      S.isLoad = cand.isLoad;
      S.isStore = cand.isStore;

      // ------------------------------------------------------------------
      // 1) Buffer sizes: replace UB symbols → constants; expect constants.
      // ------------------------------------------------------------------
      for (const AffineExprInfo &size : cand.bufferSizes) {
        AffineExpr e = replaceSymbolsWithConstantUBs(size.expr, size.ubOperands,
                                                      builder, allLoops, maxBandInfo);
        if (auto cst = llvm::dyn_cast<AffineConstantExpr>(e)) {
          S.sizes.push_back(cst.getValue());
        } else {
          llvm::errs() << "Error: buffer size is not constant after replacement: ";
          size.expr.dump();
          assert(false && "non-constant buffer size");
        }
      }

      // ------------------------------------------------------------------
      // 2) Access map: build maps for each index expr, then merge.
      // ------------------------------------------------------------------
      SmallVector<AffineMap, 4> maps;
      SmallVector<SmallVector<Value>, 4> opsPerMap;

      for (const AffineExprInfo &acc : cand.accessIndices) {
        auto [m, ops] = buildAccessMapAndOperands(acc, builder, allLoops, maxBandInfo);
        maps.push_back(m);
        opsPerMap.push_back(std::move(ops));
      }
      std::tie(S.map, S.mapOperands) = mergeMapsAndOperands(maps, opsPerMap);

      // ------------------------------------------------------------------
      // 2.5) FIFO sizes and access map: build maps for each index expr
      // ------------------------------------------------------------------
      // FIFO sizes
      for (const AffineExprInfo &size : cand.fifoSizes) {
        AffineExpr e = replaceSymbolsWithConstantUBs(size.expr, size.ubOperands,
                                                      builder, allLoops, maxBandInfo);
        if (auto cst = llvm::dyn_cast<AffineConstantExpr>(e)) {
          S.fifoSizes.push_back(cst.getValue());
        } else {
          llvm::errs() << "Error: fifo size is not constant after replacement: ";
          size.expr.dump();
          assert(false && "non-constant fifo size");
        }
      }
      // FIFO access map
      maps.clear();
      opsPerMap.clear();
      for (const AffineExprInfo &acc : cand.fifoAccessIndices) {
        auto [m, ops] = buildAccessMapAndOperands(acc, builder, allLoops, maxBandInfo);
        maps.push_back(m);
        opsPerMap.push_back(std::move(ops));
      }
      std::tie(S.fifoMap, S.fifoMapOperands) = mergeMapsAndOperands(maps, opsPerMap);
      // ------------------------------------------------------------------
      // 3) Guarding sets: (a) loadSet from LHS, (b) storeSet from LHS/RHS.
      // ------------------------------------------------------------------
      if(!cand.guardLHS.empty()){
        LLVM_DEBUG(llvm::dbgs() << "Building load guard set for " << memrefName << "\n";);
        LLVM_DEBUG(
          for(const auto &lhs : cand.guardLHS){
            llvm::dbgs() << " LHS expr: ";
            lhs.expr.dump();
          }
        );
        LLVM_DEBUG(llvm::dbgs() << "\n";);
        auto [set, ops] = buildLoadGuardSet(cand.guardLHS, builder, allLoops, maxBandInfo);
        S.loadSet = set;
        S.loadSetOperands = std::move(ops);
      }else{
        S.loadSet =  IntegerSet::get(0, 0, getConstZero(builder.getContext()), true);
        S.loadSetOperands = {};
      }
      if(!cand.guardRHS.empty()){
        // LLVM_DEBUG(llvm::dbgs() << "Building store guard set for " << memrefName << "\n";);
        // LLVM_DEBUG(
        //   for(const auto &rhs : cand.guardRHS){
        //     llvm::dbgs() << " RHS expr: ";
        //     rhs.expr.dump();
        //   }
        // );
        // LLVM_DEBUG(llvm::dbgs() << "\n";);
        // LLVM_DEBUG(
        //   for(const auto &lhs : cand.guardLHS){
        //     llvm::dbgs() << " LHS expr: ";
        //     lhs.expr.dump();
        //   }
        // );
        // LLVM_DEBUG(llvm::dbgs() << "\n";);
        auto [set, ops] = buildStoreGuardSet(cand.guardLHS, cand.guardRHS, builder, allLoops);
        S.storeSet = set;
        S.storeSetOperands = std::move(ops);
      }else{
        S.storeSet = IntegerSet::get(0, 0, getConstZero(builder.getContext()), true);
        S.storeSetOperands = {};
      }

      memrefSolutions.push_back(std::move(S));
      break;
    }
  }
}

// ----------- Extract tiling factors from AMPL result -----------
void getTilingFactors(
  const std::string &result,
  TilingSolution &tilingSolution
){
  // llvm::dbgs() << "AMPL result: " << result << "\n";
  // DenseMap<const char*, bool> solution;
  // go through result checking for pattern Tu(digit) = (int)
  std::regex tm("Tm([0-9]+) = ([0-9]+)"); // l1 tiling factors
  std::regex tu("Tu([0-9]+) = ([0-9]+)"); // l2 tiling factors
  
  std::smatch match;
  std::string::const_iterator searchStart(result.cbegin());
  while (std::regex_search(searchStart, result.cend(), match, tm)) {
    // llvm::dbgs() << "Tm: " << match[1] << " = " << match[2] << "\n";
    auto idx = std::stoi(match[1]);
    auto value = std::stoi(match[2]);
    tilingSolution.l1TilingFactors.push_back(value);
    searchStart = match.suffix().first;
  }
  searchStart = result.cbegin();
  while (std::regex_search(searchStart, result.cend(), match, tu)) {
    // llvm::dbgs() << "Tu: " << match[1] << " = " << match[2] << "\n";
    auto idx = std::stoi(match[1]);
    auto value = std::stoi(match[2]);
    tilingSolution.l2TilingFactors.push_back(value);
    searchStart = match.suffix().first;
  }
}

// ----------- Extract tiling factors from AMPL result -----------
void setFixedTilingFactors(
  const std::string &result,
  const unsigned fixedTm,
  const unsigned fixedTu,
  TilingSolution &tilingSolution
){
  // llvm::dbgs() << "AMPL result: " << result << "\n";
  // DenseMap<const char*, bool> solution;
  // go through result checking for pattern Tu(digit) = (int)
  std::regex tm("Tm([0-9]+) = ([0-9]+)"); // l1 tiling factors
  std::regex tu("Tu([0-9]+) = ([0-9]+)"); // l2 tiling factors
  
  std::smatch match;
  std::string::const_iterator searchStart(result.cbegin());
  while (std::regex_search(searchStart, result.cend(), match, tm)) {
    // llvm::dbgs() << "Tm: " << match[1] << " = " << match[2] << "\n";
    auto idx = std::stoi(match[1]);
    // auto value = std::stoi(match[2]);
    tilingSolution.l1TilingFactors.push_back(fixedTm);
    searchStart = match.suffix().first;
  }
  searchStart = result.cbegin();
  while (std::regex_search(searchStart, result.cend(), match, tu)) {
    // llvm::dbgs() << "Tu: " << match[1] << " = " << match[2] << "\n";
    auto idx = std::stoi(match[1]);
    // auto value = std::stoi(match[2]);
    tilingSolution.l2TilingFactors.push_back(fixedTu);
    searchStart = match.suffix().first;
  }
}

void printMemRefSolutions(const MemrefSolutions &S) {
  // struct MemRefSolution{
  //   Value memref;
  //   SmallVector<int64_t, 4> sizes;
  //   SmallVector<int64_t, 4> fifoSizes;
  //   AffineMap map;
  //   SmallVector<Value> mapOperands;
  //   AffineMap fifoMap;
  //   SmallVector<Value> fifoMapOperands;
  //   IntegerSet loadSet;
  //   SmallVector<Value> loadSetOperands;
  //   IntegerSet storeSet;
  //   SmallVector<Value> storeSetOperands;

  //   // local sets
  //   IntegerSet localLoadSet;
  //   SmallVector<Value> localLoadSetOperands;
  //   IntegerSet localStoreSet;
  //   SmallVector<Value> localStoreSetOperands;
  //   Operation *loadOp;
  //   Operation *storeOp;
  //   bool isStore, isLoad;
  // };
  for (const auto &memS : S) {
    LLVM_DEBUG(
      llvm::dbgs() << "====================== MemRef: " << memS.memref << " ======================\n";
      llvm::dbgs() << "  isLoad: " << memS.isLoad << ", isStore: " << memS.isStore << "\n";
      llvm::dbgs() << "  Sizes: ";
    );
    // for (auto s : memS.sizes)
    //   LLVM_DEBUG(llvm::dbgs() << s << " ");
    // LLVM_DEBUG(
    //   llvm::dbgs() << "\n";
    //   llvm::dbgs() << "  FIFO Sizes: ";
    // );
    // for (auto s : memS.fifoSizes)
    //   LLVM_DEBUG(llvm::dbgs() << s << " ");
    LLVM_DEBUG(
      // llvm::dbgs() << "\n";
      // llvm::dbgs() << "  Access Map: ";
      // memS.map.print(llvm::dbgs());
      // llvm::dbgs() << "\n";
      // llvm::dbgs() << "  FIFO Access Map: ";
      // memS.fifoMap.print(llvm::dbgs());
      llvm::dbgs() << "\n";
      llvm::dbgs() << "  Load Set: ";
      memS.loadSet.print(llvm::dbgs());
      llvm::dbgs() << "\n";
      // llvm::dbgs() << "  Store Set: ";
      // memS.storeSet.print(llvm::dbgs());
      // llvm::dbgs() << "\n";
    );
    // llvm::dbgs() << "  Local Load Set: \n";
    // memS.localLoadSet.print(llvm::dbgs());
    // llvm::dbgs() << "\n";
    // llvm::dbgs() << "  Local Store Set: \n";
    // memS.localStoreSet.print(llvm::dbgs());
    // llvm::dbgs() << "\n";
  }
}

} // end namespace streamhls
} // end namespace mlir