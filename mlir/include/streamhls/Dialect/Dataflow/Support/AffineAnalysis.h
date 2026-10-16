
#ifndef STREAMHLS_AFFINEANALYSIS_H
#define STREAMHLS_AFFINEANALYSIS_H

#include "mlir/IR/Value.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "llvm/ADT/SmallVector.h"
#include "streamhls/Dialect/Dataflow/Support/Utils.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/IntegerSet.h"
#include "llvm/ADT/SmallVector.h"
#include "streamhls/Dialect/Dataflow/Support/Utils.h"

#include <string>
#include <vector>

namespace mlir { namespace streamhls {

using namespace affine;

/// === Common types you already use ===
struct LoopInfo {
  Value iv;
  int depth = 0;     // position in band
  int level = 0;     // 0: outer/original, 1: Tm, 2: Tu (after tiling)
  int64_t lb = 0, ub = 0, step = 1;
  std::string name;
  int originalIndex = -1; // Index in the original outermost loop band
};
using LoopBandInfo      = llvm::SmallVector<LoopInfo, 3>;
using AllLoopBandsInfo  = llvm::SmallVector<LoopBandInfo, 4>;
struct AffineExprInfo {
  std::string exprStr;
  AffineExpr expr;
  llvm::SmallVector<unsigned, 4> ivOperands; // indices into tiled band
  llvm::SmallVector<unsigned, 4> ubOperands; // indices into tiled band (as symbols)
};

struct MemrefCandidateInfo{
  Value memref;
  SmallVector<AffineExprInfo> bufferSizes;
  SmallVector<AffineExprInfo> accessIndices;
  SmallVector<AffineExprInfo> fifoSizes;
  SmallVector<AffineExprInfo> fifoAccessIndices;
  AffineExprInfo reuseFactor;
  SmallVector<AffineExprInfo> guardLHS;
  SmallVector<AffineExprInfo> guardRHS;
  AffineExprInfo storeFirstAccessTime;
  AffineExprInfo storeLastAccessTime;
  AffineExprInfo loadFirstAccessTime;
  AffineExprInfo loadLastAccessTime;
  int64_t storeFirstAccessTimeEval;
  int64_t storeLastAccessTimeEval;
  int64_t loadFirstAccessTimeEval;
  int64_t loadLastAccessTimeEval;

  // legacy fields
  std::string storeFATimeLegacy;
  std::string storeLATimeLegacy;
  std::string loadFATimeLegacy;
  std::string loadLATimeLegacy;
  uint64_t storeFATimeLegacyEval;
  uint64_t storeLATimeLegacyEval;
  uint64_t loadFATimeLegacyEval;
  uint64_t loadLATimeLegacyEval;


  AffineStoreOp storeOp;
  AffineLoadOp loadOp;
  AffineMap accessMap;
  bool isStore, isLoad;
};

using SingleMemrefCandidates = SmallVector<MemrefCandidateInfo, 4>;

using AllMemrefCandidates = SmallVector<std::pair<std::string, SingleMemrefCandidates>, 4>;

using AllCandidates = std::vector<AllMemrefCandidates>;

struct MemrefInfo{
  std::string name;
  Value memref;
  Operation *op;
  bool isStore, isLoad;
  AffineLoadOp loadOp;
  AffineStoreOp storeOp;
  SmallVector<Value, 4> memrefIVs;
  SmallVector<int64_t, 4> shape;
};

// induction variable struct
struct IV {
  Value value;
  int64_t lb;
  int64_t ub;
  int64_t tripCount;
};

// core analytic functions for performance modeling
int64_t evaluateAffineExpr(AffineExpr expr, ArrayRef<int64_t> values, MLIRContext *ctx);
SmallVector<unsigned> inversePerm(const SmallVectorImpl<unsigned> &permutation);

AffineMap fixAccessMap(AffineMap map);

// ---------- Functions to collect loop band info ----------
void collectLoopBandsInfo(SmallVectorImpl<AffineForOp> &forOps, AllLoopBandsInfo &allLoopBandsInfo);
// ---------- get min and max loop stats across all loop bands ----------
void getMinAndMaxLoopStats(SmallVectorImpl<LoopBandInfo> &allLoopBandsInfo, LoopBandInfo &minBandInfo, LoopBandInfo &maxBandInfo);
// ---------- tile each loop band into two levels ----------
void tileLoopsTwoLevels(SmallVectorImpl<LoopBandInfo> &allLoopBandsInfo);
// ---------- add dummy loop info to each loop band ----------
void addDummyLoopInfo(SmallVectorImpl<LoopBandInfo> &allLoopBandsInfo);
// ---------- Collect memref info used in a loop ----------
void getLoopMemRefs(AffineForOp forOp, SmallVectorImpl<MemrefInfo> &memrefs);
// ---------- Main function to get all candidates ----------
AllMemrefCandidates getAllCandidates(LoopBandInfo &loopBandInfo, SmallVectorImpl<MemrefInfo> &memrefs, bool fullDataReuse);

uint64_t getTimeFunction(Operation *op, bool isFirst);

std::string getVariableTimeFunction(unsigned nodeId, 
                                    const SmallVectorImpl<unsigned> &permutation, 
                                    Operation *op, bool isFirst);
std::string getVariableTimeFunction(unsigned nodeId, Operation *op, bool isFirst);

AffineExprInfo getAccessTime(
  Operation *op,
  const LoopBandInfo &loopBandInfo,
  bool isFirst,
  MLIRContext *ctx
);

AffineExprInfo getTimeFunc(
  Operation *op,
  const SmallVectorImpl<unsigned> &permutation, 
  bool isFirst
);

AffineMap getMinimalAccessPattern(Operation *op);

SmallVector<std::pair<size_t, LoopInfo>> getRelevantLoops(
  unsigned idx,
  LoopBandInfo &loopBandInfo,
  Value carrierIV
);

SmallVector<SmallVector<std::pair<size_t, LoopInfo>>> getRelevantLoopsForAllCarriers(
  unsigned idx,
  LoopBandInfo &loopBandInfo,
  SmallVectorImpl<Value> &carrierIVs
);

SmallVector<AffineExprInfo> getBufferSizes(
  const SmallVectorImpl<SmallVector<std::pair<size_t, LoopInfo>>> &irrelevantLoopsForAllCarriers, 
  MLIRContext *ctx
);

SmallVector<AffineExprInfo> getAccessIndices(
  const SmallVectorImpl<SmallVector<std::pair<size_t, LoopInfo>>> &irrelevantLoopsForAllCarriers,
  MLIRContext *ctx
);

SmallVector<AffineExprInfo> getFifoSizes(
  const SmallVectorImpl<SmallVector<std::pair<size_t, LoopInfo>>> &irrelevantLoopsForAllCarriers, 
  MLIRContext *ctx
);

SmallVector<AffineExprInfo> getFifoAccessIndices(
  const SmallVectorImpl<SmallVector<std::pair<size_t, LoopInfo>>> &irrelevantLoopsForAllCarriers,
  MLIRContext *ctx
);

SmallVector<AffineExprInfo> getGuardLHS(const SmallVectorImpl<SmallVector<std::pair<size_t, LoopInfo>>> &relevantLoopsForAllCarriers);

SmallVector<AffineExprInfo> getGuardRHS(const SmallVectorImpl<SmallVector<std::pair<size_t, LoopInfo>>> &relevantLoopsForAllCarriers, MLIRContext *ctx);

AffineExprInfo getTotalReuseFactor(
  unsigned idx,
  SmallVectorImpl<Value> &reuseDims,
  LoopBandInfo &loopBandInfo,
  const MemrefInfo &memrefInfo
);

SingleMemrefCandidates
bestPerReuseLevel(const SingleMemrefCandidates &raw);

SingleMemrefCandidates
filterCandidatesMonotoneStrict(const SingleMemrefCandidates &raw);

MemrefCandidateInfo getMemrefCandidateInfo(Operation *op, LoopBandInfo &loopBandInfo, SmallVectorImpl<unsigned> &permutation, bool fullDataReuse);

} // namespace streamhls
} // namespace mlir

#endif // STREAMHLS_AFFINEANALYSIS_H
