
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Debug.h"

#include "streamhls/Dialect/Dataflow/Support/AffineAnalysis.h"
#include "streamhls/Dialect/Dataflow/Support/Utils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/AffineExpr.h"

#include <sstream>
#define DEBUG_TYPE "streamhls-affine-analysis"
using namespace mlir;
using namespace streamhls;
namespace{
std::string joinProd(const SmallVectorImpl<std::string> &items) {
  std::string result;
  bool first = true;
  for (const auto &item : items) {
    if (item.empty()) continue;
    if (!first) result += " * ";
    result += item;
    first = false;
  }
  return result;
}

AffineExprInfo packExpr(AffineExpr e,
                                llvm::ArrayRef<unsigned> ivOperands,
                                llvm::ArrayRef<unsigned> ubOperands,
                                llvm::SmallVectorImpl<std::string> &ubStrs) {
  AffineExprInfo info;
  info.expr = e;
  info.ivOperands.assign(ivOperands.begin(), ivOperands.end());
  info.ubOperands.assign(ubOperands.begin(), ubOperands.end());
  info.exprStr = joinProd(ubStrs);
  return info;
}

AffineExpr productOfSymbols(MLIRContext *ctx, llvm::ArrayRef<unsigned> symIdxs) {
  AffineExpr prod = getAffineConstantExpr(1, ctx);
  for (unsigned s : symIdxs)
    prod = prod * getAffineSymbolExpr(s, ctx);
  return prod;
}

AffineExprInfo getMixedRadixExprInfo(SmallVectorImpl<std::pair<size_t, LoopInfo>> &loops) {
  AffineExprInfo exprInfo;
  if (loops.empty()) return exprInfo;
  AffineExpr mainExpr = getAffineConstantExpr(0, loops[0].second.iv.getContext());
  for(auto pair : llvm::enumerate(loops)){
    auto i = pair.index();
    auto loopInfoPair = pair.value();
    auto originalIdx = loopInfoPair.first;
    auto loopInfo = loopInfoPair.second;
    auto ctx = loopInfo.iv.getContext();
    exprInfo.ivOperands.push_back(originalIdx);
    auto ubExpr = getAffineConstantExpr(1, ctx); 
    for(auto j = i+1; j < loops.size(); j++){
      auto nextLoopInfoPair = loops[j];
      auto nextOriginalIdx = nextLoopInfoPair.first;
      ubExpr = ubExpr * getAffineSymbolExpr(j - 1, ctx);
      exprInfo.ubOperands.push_back(nextOriginalIdx);
    }
    auto dimExpr = getAffineDimExpr(i, ctx);
    mainExpr = mainExpr + (dimExpr * ubExpr);
  }
  exprInfo.expr = mainExpr;
  return exprInfo;
}
}

namespace mlir { namespace streamhls {

int64_t evaluateAffineExpr(AffineExpr expr, ArrayRef<int64_t> values, MLIRContext *ctx) {
  if (!expr) return -1;
  unsigned maxSymIdx = 0;
  expr.walk([&](AffineExpr e) {
    if (auto sym = llvm::dyn_cast<AffineSymbolExpr>(e))
      maxSymIdx = std::max(maxSymIdx, sym.getPosition() + 1);
  });
  if (values.size() < maxSymIdx) return -1;

  if (maxSymIdx > 0) {
    SmallVector<AffineExpr, 4> replacements;
    for (auto v : values) replacements.push_back(getAffineConstantExpr(v, ctx));
    AffineExpr ev = expr.replaceSymbols(replacements);
    if (auto cst = llvm::dyn_cast<AffineConstantExpr>(ev)) return cst.getValue();
  } else if (auto cst = llvm::dyn_cast<AffineConstantExpr>(expr)) {
    return cst.getValue();
  }
  return -1;
}

SmallVector<unsigned> inversePerm(const SmallVectorImpl<unsigned> &permutation) {
  SmallVector<unsigned> inversePermutation(permutation.size());
  for (int i = 0; i < permutation.size(); ++i) {
    inversePermutation[permutation[i]] = i;
  }
  return inversePermutation;
}

AffineMap fixAccessMap(AffineMap map){
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

uint64_t getTimeFunction(Operation *op, bool isFirst){
  if(!isa<AffineLoadOp, AffineStoreOp>(op)){
    assert(false && "op is not a load or store operation");
  }
  AffineLoopBand band;
  auto innerMostForOp = op->getParentOfType<AffineForOp>();
  getLoopBandFromInnermost(innerMostForOp, band);
  
  SmallVector<IV, 4> loopsInfo;
  for(auto forOp : band){
    if(!forOp.hasConstantBounds())
      assert(false && "bounds are not constant");
    auto lb = forOp.getConstantLowerBound();
    auto ub = forOp.getConstantUpperBound();
    auto tripCount = ub - lb;
    IV info = {forOp.getInductionVar(), lb, ub, tripCount};
    loopsInfo.push_back(info);
  }
  std::reverse(loopsInfo.begin(), loopsInfo.end());

  SmallVector<Value, 4> operands;
  AffineMap accessMap;
  if(auto loadOp = dyn_cast<AffineLoadOp>(op)){
    operands = loadOp.getMapOperands();
    accessMap = loadOp.getAffineMap();
  }else if(auto storeOp = dyn_cast<AffineStoreOp>(op)){
    operands = storeOp.getMapOperands();
    accessMap = storeOp.getAffineMap();
  }

  int64_t currFactor = 1;
  AffineExpr timeExpr = getAffineConstantExpr(0, op->getContext());
  for(size_t idx = 0; idx < loopsInfo.size(); idx++){
    auto currIV = loopsInfo[idx].value;
    auto currUB = loopsInfo[idx].ub;
    auto prevTripCount = (idx > 0) ? loopsInfo[idx-1].tripCount : 1;
    currFactor *= prevTripCount;

    if(std::find(operands.begin(), operands.end(), currIV) != operands.end()){
      AffineExpr ivExpr = isFirst ? getAffineConstantExpr(0, op->getContext()) 
                                  : getAffineConstantExpr(currUB - 1, op->getContext());
      timeExpr = timeExpr + ivExpr * currFactor;
    }else{
      if(isa<AffineLoadOp>(op)){
        timeExpr = timeExpr + getAffineConstantExpr(0, op->getContext()) * currFactor;
      }
      else if(isa<AffineStoreOp>(op)){
        timeExpr = timeExpr + getAffineConstantExpr(currUB - 1, op->getContext()) * currFactor;
      }
    }
  }

  uint64_t timeVal = -1;
  if(timeExpr.getKind() == AffineExprKind::Constant){
    timeVal = timeExpr.cast<AffineConstantExpr>().getValue();
  }
  assert(timeVal != (uint64_t)-1 && "timeExpr.getKind() != AffineExprKind::Constant");
  return timeVal;
}

std::string getVariableTimeFunction(unsigned nodeId, const SmallVectorImpl<unsigned> &permutation, Operation *op, bool isFirst){
  if(!isa<AffineLoadOp, AffineStoreOp>(op)){
    assert(false && "op is not a load or store operation");
  }
  AffineLoopBand band;
  auto innerMostForOp = op->getParentOfType<AffineForOp>();
  getLoopBandFromInnermost(innerMostForOp, band);

  size_t numLoops = permutation.size();
  if (band.size() > numLoops) {
    AffineLoopBand subBand;
    subBand.append(band.end() - numLoops, band.end());
    band = subBand;
  } else if (band.size() < numLoops) {
    numLoops = band.size();
  }

  SmallVector<IV, 4> loopsInfo;
  for(auto forOp : band){
    if(!forOp.hasConstantBounds())
      assert(false && "bounds are not constant");
    auto lb = forOp.getConstantLowerBound();
    auto ub = forOp.getConstantUpperBound();
    auto tripCount = ub - lb;
    IV info = {forOp.getInductionVar(), lb, ub, tripCount};
    loopsInfo.push_back(info);
  }
  std::reverse(loopsInfo.begin(), loopsInfo.end());

  SmallVector<Value, 4> operands;
  if(auto loadOp = dyn_cast<AffineLoadOp>(op)){
    operands = loadOp.getMapOperands();
  }else if(auto storeOp = dyn_cast<AffineStoreOp>(op)){
    operands = storeOp.getMapOperands();
  }

  std::stringstream currFactor;
  currFactor << "1";
  std::stringstream timeExpr;
  timeExpr << "0";
  for(size_t idx = 0; idx < loopsInfo.size(); idx++){
    auto currIV = loopsInfo[idx].value;
    std::stringstream currUB;
    if (idx < permutation.size())
      currUB << "x" << nodeId << "_" << permutation[idx];
    else
      currUB << "x" << nodeId << "_" << idx;
    
    std::stringstream prevTripCount;
    if(idx > 0){
      prevTripCount << "x" << nodeId << "_" << permutation[loopsInfo.size() - idx];
    }else{
      prevTripCount << 1;
    }
    currFactor << " * " << prevTripCount.str();

    if(std::find(operands.begin(), operands.end(), currIV) != operands.end()){
      std::stringstream ivExpr;
      if(isFirst)
        ivExpr << "0";
      else
        ivExpr << "(" + currUB.str() + " - 1)";
      timeExpr << " + " << ivExpr.str() << " * " << currFactor.str();
    }else{
      if(isa<AffineLoadOp>(op))
        timeExpr << " + 0 * " << currFactor.str();
      else if(isa<AffineStoreOp>(op))
        timeExpr << " + (" << currUB.str() << " - 1) * " << currFactor.str();
    }
  }
  return "(" + timeExpr.str() + ")";
}

std::string getVariableTimeFunction(unsigned nodeId, Operation *op, bool isFirst){
  if(!isa<AffineLoadOp, AffineStoreOp>(op)){
    assert(false && "op is not a load or store operation");
  }
  AffineLoopBand band;
  auto innerMostForOp = op->getParentOfType<AffineForOp>();
  getLoopBandFromInnermost(innerMostForOp, band);

  SmallVector<IV, 4> loopsInfo;
  for(auto forOp : band){
    if(!forOp.hasConstantBounds())
      assert(false && "bounds are not constant");
    auto lb = forOp.getConstantLowerBound();
    auto ub = forOp.getConstantUpperBound();
    auto tripCount = ub - lb;
    IV info = {forOp.getInductionVar(), lb, ub, tripCount};
    loopsInfo.push_back(info);
  }
  std::reverse(loopsInfo.begin(), loopsInfo.end());

  SmallVector<Value, 4> operands;
  if(auto loadOp = dyn_cast<AffineLoadOp>(op)){
    operands = loadOp.getMapOperands();
  }else if(auto storeOp = dyn_cast<AffineStoreOp>(op)){
    operands = storeOp.getMapOperands();
  }

  std::string currFactorStr = "";
  std::string timeVarExpr = "0";
  for(size_t idx = 0; idx < loopsInfo.size(); idx++){
    auto currIV = loopsInfo[idx].value;
    std::string currUBStr = "x" + std::to_string(nodeId) + "_" + std::to_string(idx);
    if(idx > 0){
      currFactorStr += " * x" + std::to_string(nodeId) + "_" + std::to_string(idx-1);
    }
    std::string newTerm = "";
    if(std::find(operands.begin(), operands.end(), currIV) != operands.end()){
      if(isFirst) continue;
      newTerm = "(" + currUBStr + " - 1)" + currFactorStr;
    }else{
      if(isa<AffineLoadOp>(op)) continue;
      else if(isa<AffineStoreOp>(op)) {
        newTerm = "(" + currUBStr + " - 1)" + currFactorStr;
      }
    }
    timeVarExpr += " + " + newTerm;
  }
  return "(" + timeVarExpr + ")";
}

AffineExprInfo getTimeFunc(
  Operation *op,
  bool isFirst
){
    if(!isa<AffineLoadOp, AffineStoreOp>(op)){
    assert(false && "op is not a load or store operation");
  }
  AffineLoopBand band;
  auto innerMostForOp = op->getParentOfType<AffineForOp>();
  getLoopBandFromInnermost(innerMostForOp, band);

  SmallVector<IV, 4> loopsInfo;
  for(auto forOp : band){
    if(!forOp.hasConstantBounds())
      assert(false && "bounds are not constant");
    auto lb = forOp.getConstantLowerBound();
    auto ub = forOp.getConstantUpperBound();
    auto tripCount = ub - lb;
    IV info = {forOp.getInductionVar(), lb, ub, tripCount};
    loopsInfo.push_back(info);
  }
  std::reverse(loopsInfo.begin(), loopsInfo.end());
  // llvm::dbgs() << "loopsInfo: " << loopsInfo.size() << " ubs: ";
  // for(auto &info : loopsInfo){
  //   llvm::dbgs() << info.ub << " ";
  // }
  // llvm::dbgs() << "\n";
  
  SmallVector<Value, 4> operands;
  if(auto loadOp = dyn_cast<AffineLoadOp>(op)){
    operands = loadOp.getMapOperands();
  }else if(auto storeOp = dyn_cast<AffineStoreOp>(op)){
    operands = storeOp.getMapOperands();
  }

  // std::string currFactorStr = "";
  // std::string timeVarExpr = "0";
  // for(size_t idx = 0; idx < loopsInfo.size(); idx++){
  //   auto currIV = loopsInfo[idx].value;
  //   std::string currUBStr = "x" + std::to_string(nodeId) + "_" + std::to_string(idx);
  //   if(idx > 0){
  //     currFactorStr += " * x" + std::to_string(nodeId) + "_" + std::to_string(idx-1);
  //   }
  //   std::string newTerm = "";
  //   if(std::find(operands.begin(), operands.end(), currIV) != operands.end()){
  //     if(isFirst) continue;
  //     newTerm = "(" + currUBStr + " - 1)" + currFactorStr;
  //   }else{
  //     if(isa<AffineLoadOp>(op)) continue;
  //     else if(isa<AffineStoreOp>(op)) {
  //       newTerm = "(" + currUBStr + " - 1)" + currFactorStr;
  //     }
  //   }
  //   timeVarExpr += " + " + newTerm;
  // }
  auto ctx = op->getContext();
  AffineExpr currFactor = getAffineConstantExpr(1, ctx);
  AffineExpr timeVarExpr = getAffineConstantExpr(0, ctx);
  for(size_t idx = 0; idx < loopsInfo.size(); idx++){
    auto currIV = loopsInfo[idx].value;
    AffineExpr currUB = getAffineSymbolExpr(idx, ctx);
    if(idx > 0){
      currFactor = currFactor * getAffineSymbolExpr(idx-1, ctx);
    }
    AffineExpr newTerm;
    if(std::find(operands.begin(), operands.end(), currIV) != operands.end()){
      if(isFirst) continue;
      newTerm = (currUB - 1) * currFactor;
    }else{
      if(isa<AffineLoadOp>(op)) continue;
      else if(isa<AffineStoreOp>(op)) {
        newTerm = (currUB - 1) * currFactor;
      }
    }
    timeVarExpr = timeVarExpr + newTerm;
  }
  AffineExprInfo info;
  info.expr = timeVarExpr;
  // llvm::dbgs() << info.expr << "\n";
  return info;


}

AffineExprInfo getAccessTime(Operation *op, const LoopBandInfo &loopBandInfo,
                                bool isFirst, MLIRContext *ctx) {
  unsigned bandSize = loopBandInfo.size();
  AffineExpr expr = getAffineConstantExpr(0, ctx);

  SmallVector<Value, 4> operands;
  if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
    operands = loadOp.getMapOperands();
  } else if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
    operands = storeOp.getMapOperands();
  }

  AffineExpr currFactor = getAffineConstantExpr(1, ctx);
  // Linearization: i0 + i1 * TC0 + i2 * (TC0 * TC1) + ...
  for (unsigned i = 0; i < bandSize; ++i) {
    const LoopInfo &loop = loopBandInfo[i];
    auto idx = loop.originalIndex;
    AffineExpr tcSym = getAffineSymbolExpr(i, ctx);

    bool isRelevant =
        (std::find(operands.begin(), operands.end(), loop.iv) != operands.end());

    AffineExpr ivVal;
    if (isRelevant) {
      ivVal = isFirst ? getAffineConstantExpr(0, ctx) : (tcSym - 1);
    } else {
      ivVal = isa<AffineLoadOp>(op) ? getAffineConstantExpr(0, ctx) : (tcSym - 1);
    }

    expr = expr + ivVal * currFactor;
    currFactor = currFactor * tcSym;
  }

  AffineExprInfo info;
  info.expr = expr;
  // Note: exprStr is left empty as Equations.cpp uses the symbolic expression directly.
  return info;
}

AffineMap getMinimalAccessPattern(Operation *op){
  SmallVector<Value, 4> mapOperands;
  AffineMap accessMap;
  if(auto loadOp = dyn_cast<AffineLoadOp>(op)){
    mapOperands = loadOp.getMapOperands();
    accessMap = loadOp.getAffineMap();
  }else if(auto storeOp = dyn_cast<AffineStoreOp>(op)){
    mapOperands = storeOp.getMapOperands();
    accessMap = storeOp.getAffineMap();
  }else{
    assert(false && "op is not a load or store operation");
  }

  if(accessMap.getNumSymbols() > 0){
    accessMap = fixAccessMap(accessMap);
  }

  AffineLoopBand band;
  auto innerMostForOp = op->getParentOfType<AffineForOp>();
  getLoopBandFromInnermost(innerMostForOp, band);

  SmallVector<AffineForOp, 4> relevantLoops; 
  SmallVector<Value, 4> relevantOperands;
  for(auto forOp : band){
    auto iv = forOp.getInductionVar();
    if(std::find(mapOperands.begin(), mapOperands.end(), iv) != mapOperands.end()){
      relevantLoops.push_back(forOp);
      relevantOperands.push_back(iv);
    }
  }

  SmallVector<unsigned, 4> orderingIndices2;
  for(auto operandPair1 : llvm::enumerate(mapOperands)){
    for(auto operandPair2 : llvm::enumerate(relevantOperands)){
      if(operandPair1.value() == operandPair2.value()){
        orderingIndices2.push_back(operandPair2.index());
      }
    }
  }

  auto map2 = AffineMap::getPermutationMap(orderingIndices2, op->getContext());
  auto outMap = accessMap.compose(map2);

  SmallVector<unsigned, 4> locations;
  for(auto expr : outMap.getResults()){
    expr.walk([&](AffineExpr e){
      if(e.getKind() == AffineExprKind::DimId){
        locations.push_back(e.cast<AffineDimExpr>().getPosition());
      }
    });
  }

  if(locations.size() != relevantOperands.size()){
    SmallVector<unsigned, 4> newLocations;
    for(auto loc : locations){
      if(std::find(newLocations.begin(), newLocations.end(), loc) == newLocations.end()){
        newLocations.push_back(loc);
      }
    }
    locations = newLocations;
  }

  return AffineMap::getPermutationMap(locations, op->getContext());
}

SmallVector<std::pair<size_t, LoopInfo>> getRelevantLoops(
  unsigned idx,
  LoopBandInfo &loopBandInfo,
  Value carrierIV
) {
  SmallVector<std::pair<size_t, LoopInfo>> relevantLoops;
  for(unsigned i = idx; i < loopBandInfo.size(); i++){
    auto loopInfo = loopBandInfo[i];
    auto iv = loopInfo.iv;
    if(iv == carrierIV){
      relevantLoops.push_back({i, loopInfo});
    }
  }
  return relevantLoops;
}

SmallVector<SmallVector<std::pair<size_t, LoopInfo>>> getRelevantLoopsForAllCarriers(
  unsigned idx,
  LoopBandInfo &loopBandInfo,
  SmallVectorImpl<Value> &carrierIVs
) {
  SmallVector<SmallVector<std::pair<size_t, LoopInfo>>, 4> allRelevantLoops;
  for(auto carrierIV : carrierIVs){
    SmallVector<std::pair<size_t, LoopInfo>> relevantLoops = getRelevantLoops(idx, loopBandInfo, carrierIV);
    allRelevantLoops.push_back(relevantLoops);
  }
  return allRelevantLoops;
}

SmallVector<AffineExprInfo> getBufferSizes(
  const SmallVectorImpl<SmallVector<std::pair<size_t, LoopInfo>>> &irrelevantLoopsForAllCarriers, 
  MLIRContext *ctx
) {
  SmallVector<AffineExprInfo> bufferSizes;
  for (const auto &irrelevantLoops : irrelevantLoopsForAllCarriers) {
    SmallVector<unsigned, 4> ivOperands;
    SmallVector<unsigned, 4> ubOperands;
    SmallVector<unsigned, 4> symbols;
    SmallVector<std::string, 4> ubStrs;
    for (const auto &loopInfo : llvm::enumerate(irrelevantLoops)) {
      symbols.push_back(loopInfo.index());
      ubOperands.push_back(loopInfo.value().first);
      ubStrs.push_back(loopInfo.value().second.name);
    }
    bufferSizes.push_back(packExpr(productOfSymbols(ctx, symbols), ivOperands, ubOperands, ubStrs));
  }
  if(bufferSizes.empty()) {
    AffineExprInfo bufferSize = {
      "1",
      getAffineConstantExpr(1, ctx),
      {},
      {}
    };
    bufferSizes.push_back(bufferSize);
  }
  return bufferSizes;
}

SmallVector<AffineExprInfo> getAccessIndices(
  const SmallVectorImpl<SmallVector<std::pair<size_t, LoopInfo>>> &irrelevantLoopsForAllCarriers,
  MLIRContext *ctx
) {
  SmallVector<AffineExprInfo> accessIndices;
  for (auto irrelevantLoops : irrelevantLoopsForAllCarriers) {
    accessIndices.push_back(getMixedRadixExprInfo(irrelevantLoops));
  }
  if(accessIndices.empty()) {
    AffineExprInfo accessIndex = {
      "0",
      getAffineConstantExpr(0, ctx),
      {},
      {}
    };
    accessIndices.push_back(accessIndex);
  }
  return accessIndices;
}

SmallVector<AffineExprInfo> getFifoSizes(
  const SmallVectorImpl<SmallVector<std::pair<size_t, LoopInfo>>> &irrelevantLoopsForAllCarriers, 
  MLIRContext *ctx
) {
  SmallVector<AffineExprInfo> fifoSizes;
  for (const auto &irrelevantLoops : irrelevantLoopsForAllCarriers) {
    SmallVector<unsigned, 4> ivOperands;
    SmallVector<unsigned, 4> ubOperands;
    SmallVector<unsigned, 4> symbols;
    SmallVector<std::string, 4> ubStrs;
    uint skippedLoopsCount = 0;
    for (const auto &loopInfo : llvm::enumerate(irrelevantLoops)) {
      if(loopInfo.value().second.level != 2){
        skippedLoopsCount++;
        continue; 
      }
      symbols.push_back(loopInfo.index() - skippedLoopsCount);
      ubOperands.push_back(loopInfo.value().first);
      ubStrs.push_back(loopInfo.value().second.name);
    }
    fifoSizes.push_back(packExpr(productOfSymbols(ctx, symbols), ivOperands, ubOperands, ubStrs));
  }
  if(fifoSizes.empty()) {
    AffineExprInfo fifoSize = {
      "1",
      getAffineConstantExpr(1, ctx),
      {},
      {}
    };
    fifoSizes.push_back(fifoSize);
  }
  return fifoSizes;
}

SmallVector<AffineExprInfo> getFifoAccessIndices(
  const SmallVectorImpl<SmallVector<std::pair<size_t, LoopInfo>>> &irrelevantLoopsForAllCarriers,
  MLIRContext *ctx
) {
  SmallVector<AffineExprInfo> fifoAccessIndices;
  for (auto irrelevantLoops : irrelevantLoopsForAllCarriers) {
    SmallVector<std::pair<size_t, LoopInfo>> filteredIrrelevantLoops;
    for(auto loopInfoPair : irrelevantLoops){
      if(loopInfoPair.second.level == 2){
        filteredIrrelevantLoops.push_back(loopInfoPair);
      }
    }
    fifoAccessIndices.push_back(getMixedRadixExprInfo(filteredIrrelevantLoops));
  }
  if(fifoAccessIndices.empty()) {
    AffineExprInfo accessIndex = {
      "0",
      getAffineConstantExpr(0, ctx),
      {},
      {}
    };
    fifoAccessIndices.push_back(accessIndex);
  }
  return fifoAccessIndices;
}

SmallVector<AffineExprInfo> getGuardLHS(const SmallVectorImpl<SmallVector<std::pair<size_t, LoopInfo>>> &relevantLoopsForAllCarriers) {
  SmallVector<AffineExprInfo> guardLHS;
  for (auto relevantLoops : relevantLoopsForAllCarriers) {
    guardLHS.push_back(getMixedRadixExprInfo(relevantLoops));
  }
  return guardLHS;
}

SmallVector<AffineExprInfo> getGuardRHS(const SmallVectorImpl<SmallVector<std::pair<size_t, LoopInfo>>> &relevantLoopsForAllCarriers, MLIRContext *ctx) {
  SmallVector<AffineExprInfo> guardRHS;
  for (const auto &relevantLoops : relevantLoopsForAllCarriers) {
    SmallVector<unsigned, 4> ivOperands;
    SmallVector<unsigned, 4> ubOperands;
    SmallVector<unsigned, 4> symbols;
    SmallVector<std::string, 4> ubStrs;
    for (const auto &loopInfo : llvm::enumerate(relevantLoops)) {
      symbols.push_back(loopInfo.index());
      ubOperands.push_back(loopInfo.value().first);
      ubStrs.push_back(loopInfo.value().second.name);
    }
    guardRHS.push_back(packExpr(productOfSymbols(ctx, symbols) - getAffineConstantExpr(1, ctx), ivOperands, ubOperands, ubStrs));
  }
  return guardRHS;
}

AffineExprInfo getTotalReuseFactor(
  unsigned idx,
  SmallVectorImpl<Value> &reuseDims,
  LoopBandInfo &loopBandInfo,
  const MemrefInfo &memrefInfo
) {
  SmallVector<std::string, 4> ubStrs;
  for(auto dim : memrefInfo.shape){
    ubStrs.push_back(std::to_string(dim));
  }
  for(int j = (int)idx - 1; j >= 0; j--){
    auto loopInfo = loopBandInfo[j];
    auto iv = loopInfo.iv;
    if(llvm::is_contained(reuseDims, iv)){
      ubStrs.push_back(loopInfo.name); 
    }
  }
  return packExpr(nullptr, {}, {}, ubStrs);
}

static bool equalBufferSizes(const llvm::SmallVector<AffineExprInfo> &a,
                             const llvm::SmallVector<AffineExprInfo> &b) {
  if (a.size() != b.size())
    return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i].exprStr != b[i].exprStr)
      return false;
  return true;
}

SingleMemrefCandidates
bestPerReuseLevel(const SingleMemrefCandidates &raw) {
  SingleMemrefCandidates best;
  llvm::StringMap<unsigned> posByReuse; 

  for (const auto &cand : raw) {
    llvm::StringRef key = cand.reuseFactor.exprStr;
    auto it = posByReuse.find(key);
    if (it == posByReuse.end()) {
      posByReuse[key] = best.size();
      best.push_back(cand);
    } else {
      best[it->second] = cand;
    }
  }
  return best;
}

SingleMemrefCandidates
filterCandidatesMonotoneStrict(const SingleMemrefCandidates &raw) {
  SingleMemrefCandidates best = bestPerReuseLevel(raw);
  SingleMemrefCandidates filtered;
  llvm::SmallVector<llvm::SmallVector<AffineExprInfo>> seenBuffers;

  auto alreadySeen = [&](const llvm::SmallVector<AffineExprInfo> &buf) {
    for (const auto &b : seenBuffers)
      if (equalBufferSizes(b, buf))
        return true;
    return false;
  };

  for (const auto &cand : best) {
    if (alreadySeen(cand.bufferSizes))
      continue; 
    seenBuffers.push_back(cand.bufferSizes);
    filtered.push_back(cand);
  }

  return filtered;
}

// ---------- Functions to collect loop band info ----------
void collectLoopBandsInfo(SmallVectorImpl<AffineForOp> &forOps, AllLoopBandsInfo &allLoopBandsInfo) {
  for(auto forOpPair : llvm::enumerate(forOps)){
    auto idx = forOpPair.index();
    auto forOp = forOpPair.value();
    AffineLoopBand band;
    LoopBandInfo bandInfo;
    getLoopBandFromOutermost(forOp, band);
    int depth = 0;
    for(auto forOp : band){
      LoopInfo loopInfo;
      loopInfo.iv = forOp.getInductionVar();
      loopInfo.depth = depth++;
      loopInfo.lb = forOp.getConstantLowerBound();
      loopInfo.ub = forOp.getConstantUpperBound();
      loopInfo.step = forOp.getStepAsInt();
      bandInfo.push_back(loopInfo);
    }
    allLoopBandsInfo.push_back(bandInfo);
  }
  LLVM_DEBUG(
    llvm::dbgs() << "Number of loop bands: " << allLoopBandsInfo.size() << "\n";
  );
}

// ---------- get min and max loop stats across all loop bands ----------
void getMinAndMaxLoopStats(SmallVectorImpl<LoopBandInfo> &allLoopBandsInfo, LoopBandInfo &minBandInfo, LoopBandInfo &maxBandInfo) {
  for(auto loopBandInfo : allLoopBandsInfo){
    if(loopBandInfo.size() < minBandInfo.size()){
      minBandInfo = loopBandInfo;
    }
    if(loopBandInfo.size() > maxBandInfo.size()){
      maxBandInfo = loopBandInfo;
    }
  }
}

// ---------- tile each loop band into two levels ----------
void tileLoopsTwoLevels(SmallVectorImpl<LoopBandInfo> &allLoopBandsInfo) {
  for(auto &loopBandInfo : allLoopBandsInfo){
    LoopBandInfo tiledBandInfo;
    for(auto loopInfoPair : llvm::enumerate(loopBandInfo)){
      auto idx = loopInfoPair.index();
      auto loopInfo = loopInfoPair.value();
      LoopInfo newLoopInfo;
      newLoopInfo.depth = loopInfo.depth;
      newLoopInfo.level = 0;
      newLoopInfo.iv = loopInfo.iv;
      newLoopInfo.lb = loopInfo.lb;
      newLoopInfo.ub = loopInfo.ub;
      newLoopInfo.step = loopInfo.step;
      newLoopInfo.name = "(" + std::to_string(loopInfo.ub) + " / (Z" + std::to_string(idx) + " * X" + std::to_string(idx)+"))";
      tiledBandInfo.push_back(newLoopInfo);
    }
    for(auto loopInfoPair : llvm::enumerate(loopBandInfo)){
      auto idx = loopInfoPair.index();
      auto loopInfo = loopInfoPair.value();
      LoopInfo newLoopInfo;
      newLoopInfo.depth = loopBandInfo.size() + loopInfo.depth;
      newLoopInfo.level = 1;
      newLoopInfo.iv = loopInfo.iv;
      newLoopInfo.lb = 0;
      newLoopInfo.ub = 0;
      newLoopInfo.step = loopInfo.step;
      newLoopInfo.name = "Z" + std::to_string(idx);
      tiledBandInfo.push_back(newLoopInfo);
    }
    for(auto loopInfoPair : llvm::enumerate(loopBandInfo)){
      auto idx = loopInfoPair.index();
      auto loopInfo = loopInfoPair.value();
      LoopInfo newLoopInfo;
      newLoopInfo.depth = 2*loopBandInfo.size() + loopInfo.depth;
      newLoopInfo.level = 2;
      newLoopInfo.iv = loopInfo.iv;
      newLoopInfo.lb = 0;
      newLoopInfo.ub = 0;
      newLoopInfo.step = loopInfo.step;
      newLoopInfo.name = "X" + std::to_string(idx);
      tiledBandInfo.push_back(newLoopInfo);
    }
    loopBandInfo = tiledBandInfo;
  }
}

// ---------- add dummy loop info to each loop band ----------
void addDummyLoopInfo(SmallVectorImpl<LoopBandInfo> &allLoopBandsInfo){
  for(auto &loopInfo : allLoopBandsInfo){
    LoopInfo dummyInfo;
    dummyInfo.iv = nullptr;
    dummyInfo.level = -1;
    dummyInfo.lb = -1;
    dummyInfo.ub = -1;
    dummyInfo.step = -1;
    dummyInfo.name = "dummy";
    // push the front
    loopInfo.insert(loopInfo.begin(), dummyInfo);
  }
}

// ---------- Collect memref info used in a loop ----------
void getLoopMemRefs(AffineForOp forOp, SmallVectorImpl<MemrefInfo> &memrefs){
  SetVector<Value> memrefValues;
  forOp.walk([&](Operation *op) {
    if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
      memrefValues.insert(loadOp.getMemRef());
    } else if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
      memrefValues.insert(storeOp.getMemRef());
    }
  });
  LLVM_DEBUG(
    llvm::dbgs() << "Number of memrefs: " << memrefValues.size() << "\n";
  );
  for(auto memrefPair : llvm::enumerate(memrefValues.getArrayRef())){
    auto idx = memrefPair.index();
    auto memrefValue = memrefPair.value();
    MemrefInfo memrefInfo;
    memrefInfo.name = "memref_" + std::to_string(idx);
    memrefInfo.memref = memrefValue;
    memrefInfo.isLoad = false;
    memrefInfo.isStore = false;
    // check if the memref is loaded or stored
    for(auto user : memrefValue.getUsers()){
      if(isa<AffineLoadOp>(user)){
        memrefInfo.isLoad = true;
        memrefInfo.loadOp = cast<AffineLoadOp>(user);
      }
      if(isa<AffineStoreOp>(user)){
        memrefInfo.isStore = true;
        memrefInfo.storeOp = cast<AffineStoreOp>(user);
      }
    }
    // get the memref IVs
    SetVector<Value> memrefIVs;
    for(auto user : memrefValue.getUsers()){
      if(auto loadOp = dyn_cast<AffineLoadOp>(user)){
        if(loadOp.getMapOperands() == memrefInfo.memrefIVs){
          continue;
        }
        for(auto operand : loadOp.getMapOperands()){
          memrefInfo.memrefIVs.push_back(operand);
        }
      }
      if(auto storeOp = dyn_cast<AffineStoreOp>(user)){
        if(storeOp.getMapOperands() == memrefInfo.memrefIVs){
          continue;
        }
        for(auto operand : storeOp.getMapOperands()){
          memrefInfo.memrefIVs.push_back(operand);
        }
      }
    }
    // get the memref shape
    if(auto memrefType = memrefValue.getType().dyn_cast<MemRefType>()){
      for(auto dim : memrefType.getShape()){
        memrefInfo.shape.push_back(dim);
      }
    }
    if (memrefInfo.shape.empty()) {
      memrefInfo.shape.push_back(1); // handle scalar case
    }
    memrefs.push_back(memrefInfo);
  }

}

// ---------- Main function to get all candidates ----------
AllMemrefCandidates getAllCandidates(
  LoopBandInfo &loopBandInfo, 
  SmallVectorImpl<MemrefInfo> &memrefs,
  bool fullDataReuse
){
  AllMemrefCandidates allMemrefCandidates;
  for (const auto &memref : memrefs) {
    SingleMemrefCandidates singleMemrefCandidates;
    StringRef memrefName = memref.name;
    auto memrefIvs = memref.memrefIVs;
    
    SmallVector<Value> reuseDims; // loops that do NOT index this memref
    SmallVector<Value> otherDims; // loops that DO index this memref

    for (const auto &loopInfo : loopBandInfo) {
      if (loopInfo.level != 0) continue; // keep if you truly want only the outer band
      Value iv = loopInfo.iv;
      if (llvm::is_contained(memrefIvs, iv))
        otherDims.push_back(iv);   // in Python: "if d in memref_idx_dims" → other_dims
      else
        reuseDims.push_back(iv);   // in Python: "if d not in memref_idx_dims" → reuse_dims
    }

    for(auto loopInfoPair : llvm::enumerate(loopBandInfo)){
      auto idx = loopInfoPair.index();
      auto loopInfo = loopInfoPair.value();
      auto iv = loopInfo.iv;

      if(idx > (2*(loopBandInfo.size()/3) + 1)){
        continue;
      }

      auto relevantLoopsForEachCarrier = getRelevantLoopsForAllCarriers(idx, loopBandInfo, reuseDims);
      auto irrelevantLoopsForEachCarrier = getRelevantLoopsForAllCarriers(idx, loopBandInfo, otherDims);

      auto bufferSizes = getBufferSizes(irrelevantLoopsForEachCarrier, memref.memref.getContext());
      auto accessIndices = getAccessIndices(irrelevantLoopsForEachCarrier, memref.memref.getContext());
      
      auto fifoSizes = getFifoSizes(irrelevantLoopsForEachCarrier, memref.memref.getContext());
      auto fifoAccessIndices = getFifoAccessIndices(irrelevantLoopsForEachCarrier, memref.memref.getContext());
      
      auto guardLHSs = getGuardLHS(relevantLoopsForEachCarrier);
      auto guardRHSs = getGuardRHS(relevantLoopsForEachCarrier, memref.memref.getContext());
      auto reuseFactor = getTotalReuseFactor(idx, reuseDims, loopBandInfo, memref);

      MemrefCandidateInfo candidateInfo;
      candidateInfo.memref = memref.memref;
      candidateInfo.storeOp = memref.storeOp;
      candidateInfo.loadOp = memref.loadOp;
      candidateInfo.isLoad = memref.isLoad;
      candidateInfo.isStore = memref.isStore;
      candidateInfo.bufferSizes = bufferSizes;
      candidateInfo.accessIndices = accessIndices;
      candidateInfo.fifoSizes = fifoSizes;
      candidateInfo.fifoAccessIndices = fifoAccessIndices;
      candidateInfo.reuseFactor = reuseFactor;
      candidateInfo.guardLHS = guardLHSs;
      candidateInfo.guardRHS = guardRHSs;
      SmallVector<int64_t, 4> nominalValues;
      for (auto &loop : loopBandInfo) {
        if (loop.level == -1) continue;
        nominalValues.push_back(loop.ub);
      }
      if(memref.storeOp){
        candidateInfo.storeFirstAccessTime = getAccessTime(memref.storeOp, loopBandInfo, true, memref.memref.getContext());
        candidateInfo.storeLastAccessTime = getAccessTime(memref.storeOp, loopBandInfo, false, memref.memref.getContext());
        candidateInfo.storeFirstAccessTimeEval = evaluateAffineExpr(candidateInfo.storeFirstAccessTime.expr, nominalValues, memref.memref.getContext());
        candidateInfo.storeLastAccessTimeEval = evaluateAffineExpr(candidateInfo.storeLastAccessTime.expr, nominalValues, memref.memref.getContext());
      }
      if(memref.loadOp){
        candidateInfo.loadFirstAccessTime = getAccessTime(memref.loadOp, loopBandInfo, true, memref.memref.getContext());
        candidateInfo.loadLastAccessTime = getAccessTime(memref.loadOp, loopBandInfo, false, memref.memref.getContext());
        candidateInfo.loadFirstAccessTimeEval = evaluateAffineExpr(candidateInfo.loadFirstAccessTime.expr, nominalValues, memref.memref.getContext());
        candidateInfo.loadLastAccessTimeEval = evaluateAffineExpr(candidateInfo.loadLastAccessTime.expr, nominalValues, memref.memref.getContext());
      }

      singleMemrefCandidates.push_back(candidateInfo);
    }

    // Filter after collection
    // TODO: filtering is not complete yet
    SingleMemrefCandidates filtered = filterCandidatesMonotoneStrict(singleMemrefCandidates);
    if(fullDataReuse){
      // get the last candidate only for full data reuse to eliminate the first load
      MemrefCandidateInfo candidateInfo = filtered.front();
      SingleMemrefCandidates singleCandidate = {candidateInfo};
      allMemrefCandidates.push_back({memrefName.str(), std::move(singleCandidate)});
    }else{
      if(memref.isLoad && memref.isStore){
        // get the last candidate only for full data reuse to eliminate the first load
        MemrefCandidateInfo candidateInfo = filtered.front();
        SingleMemrefCandidates singleCandidate = {candidateInfo};
        allMemrefCandidates.push_back({memrefName.str(), std::move(singleCandidate)});
      }else{
        allMemrefCandidates.push_back({memrefName.str(), std::move(filtered)});
      }
    }
  }
  return allMemrefCandidates;
}

MemrefCandidateInfo getMemrefCandidateInfo(
  Operation *op, 
  LoopBandInfo &loopBandInfo,
  SmallVectorImpl<unsigned> &permutation,
  bool fullDataReuse
) {
  MLIRContext *ctx = op->getContext();
  MemrefCandidateInfo candidateInfo;
  Value memrefValue;
  SmallVector<Value, 4> memrefIvs;
  
  if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
    memrefValue = loadOp.getMemRef();
    memrefIvs = loadOp.getMapOperands();
    candidateInfo.loadOp = loadOp;
    candidateInfo.isLoad = true;
    candidateInfo.isStore = false;
  } else if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
    memrefValue = storeOp.getMemRef();
    memrefIvs = storeOp.getMapOperands();
    candidateInfo.storeOp = storeOp;
    candidateInfo.isLoad = false;
    candidateInfo.isStore = true;
  } else {
    assert(false && "op is not a load or store operation");
  }
  candidateInfo.memref = memrefValue;

  SmallVector<Value> reuseDims;
  SmallVector<Value> otherDims;
  for (const auto &loopInfo : loopBandInfo) {
    if (loopInfo.level != 0) continue;
    Value iv = loopInfo.iv;
    if (llvm::is_contained(memrefIvs, iv))
      otherDims.push_back(iv);
    else
      reuseDims.push_back(iv);
  }

  // We only consider level 0 (the outermost/original loops) for now.
  // In the future, we might want to iterate over idx like in getAllCandidates.
  unsigned idx = 0; // Outermost level or no reuse

  auto relevantLoopsForEachCarrier = getRelevantLoopsForAllCarriers(idx, loopBandInfo, reuseDims);
  auto irrelevantLoopsForEachCarrier = getRelevantLoopsForAllCarriers(idx, loopBandInfo, otherDims);

  candidateInfo.bufferSizes = getBufferSizes(irrelevantLoopsForEachCarrier, ctx);
  candidateInfo.accessIndices = getAccessIndices(irrelevantLoopsForEachCarrier, ctx);
  candidateInfo.fifoSizes = getFifoSizes(irrelevantLoopsForEachCarrier, ctx);
  candidateInfo.fifoAccessIndices = getFifoAccessIndices(irrelevantLoopsForEachCarrier, ctx);
  candidateInfo.guardLHS = getGuardLHS(relevantLoopsForEachCarrier);
  candidateInfo.guardRHS = getGuardRHS(relevantLoopsForEachCarrier, ctx);
  
  // Create a pseudo-MemrefInfo for reuse factor calculation
  MemrefInfo memref;
  memref.memref = memrefValue;
  if (auto mt = memrefValue.getType().dyn_cast<MemRefType>()) {
    for (auto d : mt.getShape()) memref.shape.push_back(d);
  }
  if (memref.shape.empty()) memref.shape.push_back(1);
  candidateInfo.reuseFactor = getTotalReuseFactor(idx, reuseDims, loopBandInfo, memref);


  SmallVector<int64_t, 4> ubs;
  for (auto &loop : loopBandInfo) {
    if (loop.level == -1) continue; // skip dummy
    ubs.push_back(loop.ub);
  }


  // reverse nominal values
  std::reverse(ubs.begin(), ubs.end());

  // SmallVector<unsigned> perm(permutation.begin(), permutation.end());
  // std::reverse(perm.begin(), perm.end());
  

  // print permuation 
  // llvm::dbgs() << "permutation: ";
  // for (auto &p : permutation) {
  //   llvm::dbgs() << p << " ";
  // }
  // llvm::dbgs() << ", inverse permutation: ";
  // for (auto &p : invPerm) {
  //   llvm::dbgs() << p << " ";
  // }
  // llvm::dbgs() << "\n";
  // llvm::dbgs() << "ubs: ";
  // for (auto &ub : ubs) {
  //   llvm::dbgs() << ub << " ";
  // }
  // llvm::dbgs() << "\n";

  // SmallVector<int64_t, 4> permVals;
  // // SmallVector<int64_t, 4> invPermVals;
  // for(int i = 0; i < ubs.size(); i++){
  //   permVals.push_back(ubs[perm[i]]);
  // }
  // for(int i = 0; i < ubs.size(); i++){
  //   invPermVals.push_back(ubs[inversePerm(permutation)[i]]);
  // }

  SmallVector<int64_t, 4> vals = ubs;
  // std::reverse(vals.begin(), vals.end());

  // Access times
  if (candidateInfo.isStore) {
    candidateInfo.storeFirstAccessTime = getTimeFunc(op, true);
    candidateInfo.storeLastAccessTime = getTimeFunc(op, false);
    candidateInfo.storeFirstAccessTimeEval = evaluateAffineExpr(candidateInfo.storeFirstAccessTime.expr, vals, ctx);

    // llvm::dbgs() << "Expr: " << candidateInfo.storeFirstAccessTime.expr << "\n";
    // llvm::dbgs() << "Inputs: ";
    // for (auto &val : vals) {
    //   llvm::dbgs() << val << " ";
    // }
    // llvm::dbgs() << "\n";
    // llvm::dbgs() << "Eval: " << candidateInfo.storeFirstAccessTimeEval << "\n";
    candidateInfo.storeLastAccessTimeEval = evaluateAffineExpr(candidateInfo.storeLastAccessTime.expr, vals, ctx);

    // legacy
    candidateInfo.storeFATimeLegacyEval = getTimeFunction(op, true);
    candidateInfo.storeLATimeLegacyEval = getTimeFunction(op, false);
    candidateInfo.storeFATimeLegacy = getVariableTimeFunction(idx, op, true);
    candidateInfo.storeLATimeLegacy = getVariableTimeFunction(idx, op, false);

    // llvm::dbgs() << "sf = \"" << candidateInfo.storeFirstAccessTime.expr << "\"\n";
    // llvm::dbgs() << "sf_legacy = \"" << candidateInfo.storeFATimeLegacy << "\"\n";

    // llvm::dbgs() << "sl = \"" << candidateInfo.storeLastAccessTime.expr << "\"\n";
    // llvm::dbgs() << "sl_legacy = \"" << candidateInfo.storeLATimeLegacy << "\"\n";

  } else {
    candidateInfo.loadFirstAccessTime = getTimeFunc(op, true);
    candidateInfo.loadLastAccessTime = getTimeFunc(op, false);
    candidateInfo.loadFirstAccessTimeEval = evaluateAffineExpr(candidateInfo.loadFirstAccessTime.expr, vals, ctx);
    candidateInfo.loadLastAccessTimeEval = evaluateAffineExpr(candidateInfo.loadLastAccessTime.expr, vals, ctx);

    // legacy
    candidateInfo.loadFATimeLegacyEval = getTimeFunction(op, true);
    candidateInfo.loadLATimeLegacyEval = getTimeFunction(op, false);
    candidateInfo.loadFATimeLegacy = getVariableTimeFunction(idx, op, true);
    candidateInfo.loadLATimeLegacy = getVariableTimeFunction(idx, op, false);

    // llvm::dbgs() << "lf = \"" << candidateInfo.loadFirstAccessTime.expr << "\"\n";
    // llvm::dbgs() << "lf_legacy = \"" << candidateInfo.loadFATimeLegacy << "\"\n";

    // llvm::dbgs() << "ll = \"" << candidateInfo.loadLastAccessTime.expr << "\"\n";
    // llvm::dbgs() << "ll_legacy = \"" << candidateInfo.loadLATimeLegacy << "\"\n";
  }

  candidateInfo.accessMap = getMinimalAccessPattern(op);

  return candidateInfo;
}

} // namespace streamhls
} // namespace mlir
