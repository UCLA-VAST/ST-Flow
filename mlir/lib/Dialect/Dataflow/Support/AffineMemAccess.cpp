#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
// #include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Affine/LoopUtils.h"

#include "streamhls/Dialect/Dataflow/Support/AffineMemAccess.h"
#include "streamhls/Dialect/Dataflow/Support/AffineAnalysis.h"
#include "streamhls/Dialect/Dataflow/Support/Utils.h"

#include <optional>

#define DEBUG_TYPE "streamhls-affine-mem-access"

using namespace mlir;
using namespace streamhls;
using namespace dataflow;
using namespace presburger;


static bool pinIVFromIfEquality(affine::AffineIfOp ifOp, Value iv, int64_t &pinnedVal) {
  // Match extremely simple sets like affine_set<(d0) : (d0 == C)>
  IntegerSet set = ifOp.getIntegerSet();
  auto ops = ifOp.getOperands(); // dims + symbols mapping

  // Find index where operand == iv and that index is a DIM.
  int dimIdx = -1;
  for (auto [i, v] : llvm::enumerate(ops)) {
    if (v == iv && i < static_cast<int>(set.getNumDims())) { dimIdx = i; break; }
  }
  if (dimIdx < 0) return false;

  // Look through equalities; detect pattern: d_dimIdx + c == 0  -> pinned = -c
  auto eqFlags = set.getEqFlags(); // ArrayRef<bool> true for equality
  for (auto it : llvm::enumerate(set.getConstraints())) {
    AffineExpr e = it.value();
    if (!eqFlags[it.index()]) continue;               // only equalities
    // Extract constant and coefficient of dimIdx.
    // For constant sets emitted by frontend, constraints are linear:
    //   sum_i a_i * d_i + sum_j b_j * s_j + c == 0
    // We only accept a_dimIdx == 1 and all other dims/syms zero.
    // Use a tiny walker: e = e1 + e2 + ... with possible ConstantExpr.
    // Build (coef, const) pair for the target dim.
    int64_t coef = 0;
    int64_t cst = 0;
    bool bad = false;

    std::function<void(AffineExpr)> decompose = [&](AffineExpr ex) {
      if (auto d = ex.dyn_cast<AffineDimExpr>()) {
        coef += (d.getPosition() == dimIdx) ? 1 : 0;
        if (d.getPosition() != dimIdx) bad = true;
        return;
      }
      if (auto s = ex.dyn_cast<AffineSymbolExpr>()) { bad = true; return; }
      if (auto c = ex.dyn_cast<AffineConstantExpr>()) { cst += c.getValue(); return; }
      if (auto bin = ex.dyn_cast<AffineBinaryOpExpr>()) {
        if (bin.getKind() == AffineExprKind::Add) {
          decompose(bin.getLHS()); decompose(bin.getRHS()); return;
        }
        // We do not accept mul/div/mod here in the "simple" path.
        bad = true; return;
      }
      bad = true;
    };
    decompose(e);

    if (!bad && coef == 1) {
      // equality: d_dimIdx + cst == 0  -> d_dimIdx == -cst
      pinnedVal = -cst;
      return true;
    }
  }
  return false;
}

// Collect enclosing affine.for ops (outer -> inner)
static void collectEnclosingFors(Operation *op, SmallVector<affine::AffineForOp> &fors) {
  for (Operation *p = op->getParentOp(); p; p = p->getParentOp())
    if (auto f = dyn_cast<affine::AffineForOp>(p))
      fors.push_back(f);
  llvm::reverse(fors);
}

// Find if an IV is pinned by any surrounding affine.if
static bool isIVPinned(Value iv, Operation *op, int64_t &pinnedVal) {
  for (Operation *p = op->getParentOp(); p; p = p->getParentOp())
    if (auto iff = dyn_cast<affine::AffineIfOp>(p))
      if (pinIVFromIfEquality(iff, iv, pinnedVal)) return true;
  return false;
}

// Given constants-only affine.for, return (lb, ub, step, tripCount).
static std::tuple<int64_t,int64_t,int64_t,int64_t>
getConstLoopInfo(affine::AffineForOp forOp) {
  auto lbm = forOp.getLowerBoundMap();
  auto ubm = forOp.getUpperBoundMap();
  assert(lbm.getNumResults() == 1 && ubm.getNumResults() == 1 &&
         "expected single-result bounds");
  auto lbConst = lbm.getResult(0).dyn_cast<AffineConstantExpr>();
  auto ubConst = ubm.getResult(0).dyn_cast<AffineConstantExpr>();
  assert(lbConst && ubConst && "expected constant bounds (per assumption)");
  int64_t lb = lbConst.getValue();
  int64_t ub = ubConst.getValue();
  int64_t step = forOp.getStepAsInt();
  assert(step > 0 && "expected positive constant step");
  int64_t trips = (ub - lb + step - 1) / step;
  if (trips < 0) trips = 0;
  return {lb, ub, step, trips};
}

// Build row-major lexicographic linearization map over dims of extents[].
static AffineMap buildRowMajorLinearMap(MLIRContext *ctx, ArrayRef<int64_t> extents) {
  unsigned n = extents.size();
  SmallVector<AffineExpr> dims;
  dims.reserve(n);
  for (unsigned i = 0; i < n; ++i) dims.push_back(getAffineDimExpr(i, ctx));

  // stride for dim i = product of extents[j] for j > i
  SmallVector<int64_t> stride(n, 1);
  for (int i = (int)n - 2; i >= 0; --i) stride[i] = stride[i+1] * extents[i+1];

  AffineExpr lin = getAffineConstantExpr(0, ctx);
  for (unsigned i = 0; i < n; ++i) {
    if (extents[i] == 1) continue;
    lin = lin + dims[i] * getAffineConstantExpr(stride[i], ctx);
  }
  return AffineMap::get(n, /*numSymbols=*/0, ArrayRef<AffineExpr>{lin}, ctx);
}

// namespace mlir {
// namespace dataflow {
// // Kind of memory access
// enum class AccessKind : uint32_t {
//   load = 0,
//   store = 1,
// };

// ::std::optional<AccessKind> symbolizeAccessKind(uint32_t);
// ::llvm::StringRef stringifyAccessKind(AccessKind);
// ::std::optional<AccessKind> symbolizeAccessKind(::llvm::StringRef);
// inline constexpr unsigned getMaxEnumValForAccessKind() {
//   return 1;
// }


// inline ::llvm::StringRef stringifyEnum(AccessKind enumValue) {
//   return stringifyAccessKind(enumValue);
// }

// template <typename EnumType>
// ::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

// template <>
// inline ::std::optional<AccessKind> symbolizeEnum<AccessKind>(::llvm::StringRef str) {
//   return symbolizeAccessKind(str);
// }
// } // namespace dataflow
// } // namespace mlir


// Public utility: build AccessInfoAttr for a given affine.load/store.
FailureOr<AccessInfoAttr> mlir::streamhls::buildAccessInfoForAffineAccess(Operation *accessOp) {
  MLIRContext *ctx = accessOp->getContext();

  // 0) Determine kind
  AccessKind kind;
  if (isa<affine::AffineLoadOp>(accessOp)) {
    kind = AccessKind::load;
  } else if (isa<affine::AffineStoreOp>(accessOp)) {
    kind = AccessKind::store;
  } else {
    return failure();
  }

  // 1) Gather enclosing loops, figure out which IVs are **varying** (not pinned).
  SmallVector<affine::AffineForOp> loops;
  collectEnclosingFors(accessOp, loops);

  struct DimInfo { int64_t lb, ub, step, trips; };
  SmallVector<DimInfo> varying;  varying.reserve(loops.size());

  for (auto forOp : loops) {
    auto iv = forOp.getInductionVar();
    int64_t pinnedVal;
    if (isIVPinned(iv, accessOp, pinnedVal)) {
      // IV is fixed by an equality guard: does not contribute to lexicographic dims.
      continue;
    }
    auto [lb, ub, step, trips] = getConstLoopInfo(forOp);
    // Normalize to [0, trips) logical coordinate (we don’t expose lb/step in dims).
    varying.push_back({/*lb=*/0, /*ub=*/trips, /*step=*/1, /*trips=*/trips});
  }

  // 2) If no varying loops, this is a scalar access executed exactly once.
  if (varying.empty()) {
    auto map = AffineMap::get(/*dims=*/0, /*symbols=*/0,
                              ArrayRef<AffineExpr>{getAffineConstantExpr(0, ctx)}, ctx);
    int64_t num = 1;
    SmallVector<std::pair<int64_t,int64_t>> bounds; // empty
    return AccessInfoAttr::get(ctx, kind, map, bounds, num);
  }

  // 3) Build bounds array and num_access
  SmallVector<int64_t> extents;
  extents.reserve(varying.size());
  SmallVector<std::pair<int64_t,int64_t>> bounds;
  bounds.reserve(varying.size());
  int64_t numAccess = 1;
  for (auto &d : varying) {
    extents.push_back(d.trips);
    bounds.emplace_back(d.lb, d.ub);
    numAccess *= d.trips;
  }

  // 4) Build row-major linearization over d0..d{n-1}
  auto accessMap = buildRowMajorLinearMap(ctx, extents);

  // 5) Package into the dialect attribute
  return AccessInfoAttr::get(ctx, kind, accessMap, bounds, numAccess);
}



void mlir::streamhls::getRelevantAndIrrelevantEnclosingAffineOps(Operation &op,
                                         SmallVectorImpl<Operation *> *relevantOps,
                                         SmallVectorImpl<Operation *> *irrelevantOps
                                         ) {
  relevantOps->clear();
  irrelevantOps->clear();
  Operation *currOp = op.getParentOp();

  // Traverse up the hierarchy collecting all `affine.for`, `affine.if`, and
  // affine.parallel operations.
  while (currOp) {
    if (isa<AffineIfOp, AffineForOp, AffineParallelOp>(currOp)){
      // check if any of the indices are used in the currOp
      bool relevant = false;
      if (isa<AffineForOp>(currOp)){
        auto loopIV = cast<AffineForOp>(currOp).getInductionVar();
        for(auto user : loopIV.getUsers()) {
          if(user == &op) {
            relevant = true;
            break;
          }
        }
      }
      // else if(isa<AffineIfOp>(currOp)){
      //   auto ifOp = cast<AffineIfOp>(currOp);
      //   for(auto operand : ifOp.getOperands()){
      //     for(auto user : operand.getUsers()){
      //       if(user == &op) {
      //         relevant = true;
      //         break;
      //       }
      //     }
      //   }
      // }
      if(relevant)
        relevantOps->push_back(currOp);
      else
        irrelevantOps->push_back(currOp);
    }
    currOp = currOp->getParentOp();
  }

  std::reverse(relevantOps->begin(), relevantOps->end());
  std::reverse(irrelevantOps->begin(), irrelevantOps->end());
}

void mlir::streamhls::getEnclosingAffineOps(Operation &op,
                                         SmallVectorImpl<Operation *> *ops) {
  ops->clear();
  Operation *currOp = op.getParentOp();

  // Traverse up the hierarchy collecting all `affine.for`, `affine.if`, and
  // affine.parallel operations.
  while (currOp) {
    if (isa<AffineIfOp, AffineForOp, AffineParallelOp>(currOp))
      ops->push_back(currOp);
    currOp = currOp->getParentOp();
  }
  std::reverse(ops->begin(), ops->end());
}

// Builds a system of constraints with dimensional variables corresponding to
// the loop IVs of the forOps appearing in that order. Any symbols founds in
// the bound operands are added as symbols in the system. Returns failure for
// the yet unimplemented cases.
// TODO: Handle non-unit steps through local variables or stride information in
// FlatAffineValueConstraints. (For eg., by using iv - lb % step = 0 and/or by
// introducing a method in FlatAffineValueConstraints
// setExprStride(ArrayRef<int64_t> expr, int64_t stride)
LogicalResult mlir::streamhls::getIndexSet(MutableArrayRef<Operation *> ops,
                                        FlatAffineValueConstraints *domain) {
  SmallVector<Value, 4> indices;
  SmallVector<Operation *, 8> loopOps;
  size_t numDims = 0;
  for (Operation *op : ops) {
    if (!isa<AffineForOp, AffineIfOp, AffineParallelOp>(op)) {
      LLVM_DEBUG(llvm::dbgs() << "getIndexSet only handles affine.for/if/"
                                 "parallel ops");
      return failure();
    }
    if (AffineForOp forOp = dyn_cast<AffineForOp>(op)) {
      loopOps.push_back(forOp);
      // An AffineForOp retains only 1 induction variable.
      numDims += 1;
    } else if (AffineParallelOp parallelOp = dyn_cast<AffineParallelOp>(op)) {
      loopOps.push_back(parallelOp);
      numDims += parallelOp.getNumDims();
    }
  }
  extractInductionVars(loopOps, indices);
  // Reset while associating Values in 'indices' to the domain.
  *domain = FlatAffineValueConstraints(numDims, /*numSymbols=*/0,
                                       /*numLocals=*/0, indices);
  for (Operation *op : ops) {
    // Add constraints from forOp's bounds.
    if (AffineForOp forOp = dyn_cast<AffineForOp>(op)) {
      if (failed(domain->addAffineForOpDomain(forOp)))
        return failure();
    } else if (auto ifOp = dyn_cast<AffineIfOp>(op)) {
      domain->addAffineIfOpDomain(ifOp);
    } else if (auto parallelOp = dyn_cast<AffineParallelOp>(op))
      if (failed(domain->addAffineParallelOpDomain(parallelOp)))
        return failure();
  }
  return success();
}

/// Computes the iteration domain for 'op' and populates 'indexSet', which
/// encapsulates the constraints involving loops surrounding 'op' and
/// potentially involving any Function symbols. The dimensional variables in
/// 'indexSet' correspond to the loops surrounding 'op' from outermost to
/// innermost.
LogicalResult mlir::streamhls::getOpIndexSet(Operation *op,
                                   FlatAffineValueConstraints *indexSet,
                                   SmallVectorImpl<Operation *> *irrelevantOps,
                                   bool relevantOnly /*only for load/store ops*/) {
  // if(relevantOnly){
  //   SmallVector<Operation *, 4> relevantOps;
  //   if(isa<AffineReadOpInterface>(op) || isa<AffineWriteOpInterface>(op)){
  //     getRelevantAndIrrelevantEnclosingAffineOps(*op, &relevantOps, irrelevantOps);
  //     return streamhls::getIndexSet(relevantOps, indexSet);
  //   } else {
  //     assert(false && "only for load/store ops");
  //     return failure();
  //   }
  // } else {
  SmallVector<Operation *, 4> ops;
  getEnclosingAffineOps(*op, &ops);
  LLVM_DEBUG(llvm::dbgs() << "getOpIndexSet\n");
  return streamhls::getIndexSet(ops, indexSet);
  // }
}

// MemRefAccess implementation
// Constructs  MemRefAccess populating it with the memref, its indices and
// opinst from 'loadOrStoreOpInst'.
streamhls::MemRefAccess::MemRefAccess(Operation *loadOrStoreOpInst) {
  if (auto loadOp = dyn_cast<AffineReadOpInterface>(loadOrStoreOpInst)) {
    memref = loadOp.getMemRef();
    opInst = loadOrStoreOpInst;
    llvm::append_range(indices, loadOp.getMapOperands());
  } else {
    assert(isa<AffineWriteOpInterface>(loadOrStoreOpInst) &&
           "Affine read/write op expected");
    auto storeOp = cast<AffineWriteOpInterface>(loadOrStoreOpInst);
    opInst = loadOrStoreOpInst;
    memref = storeOp.getMemRef();
    llvm::append_range(indices, storeOp.getMapOperands());
  }
}

unsigned streamhls::MemRefAccess::getRank() const {
  return cast<MemRefType>(memref.getType()).getRank();
}

bool streamhls::MemRefAccess::isStore() const {
  return isa<AffineWriteOpInterface>(opInst);
}

LogicalResult streamhls::MemRefAccess::getAccessRelation(IntegerRelation &rel) const {
  // Create set corresponding to domain of access.
  FlatAffineValueConstraints domain;
  if (failed(getOpIndexSet(opInst, &domain)))
    return failure();

  // Get access relation from access map.
  AffineValueMap accessValueMap;
  getAccessMap(&accessValueMap);
  if (failed(getRelationFromMap(accessValueMap, rel)))
    return failure();

  // Merge and align domain ids of `rel` with ids of `domain`. Since the domain
  // of the access map is a subset of the domain of access, the domain ids of
  // `rel` are guranteed to be a subset of ids of `domain`.
  unsigned inserts = 0;
  for (unsigned i = 0, e = domain.getNumDimVars(); i < e; ++i) {
    const Identifier domainIdi = Identifier(domain.getValue(i));
    const Identifier *findBegin = rel.getIds(VarKind::SetDim).begin() + i;
    const Identifier *findEnd = rel.getIds(VarKind::SetDim).end();
    const Identifier *itr = std::find(findBegin, findEnd, domainIdi);
    if (itr != findEnd) {
      rel.swapVar(i, i + std::distance(findBegin, itr));
    } else {
      ++inserts;
      rel.insertVar(VarKind::SetDim, i);
      rel.setId(VarKind::SetDim, i, domainIdi);
    }
  }

  // Append domain constraints to `rel`.
  IntegerRelation domainRel = domain;
  if (rel.getSpace().isUsingIds() && !domainRel.getSpace().isUsingIds())
    domainRel.resetIds();
  domainRel.appendVar(VarKind::Range, accessValueMap.getNumResults());
  domainRel.mergeAndAlignSymbols(rel);
  domainRel.mergeLocalVars(rel);
  rel.append(domainRel);

  rel.convertVarKind(VarKind::SetDim, 0, accessValueMap.getNumDims() + inserts,
                     VarKind::Domain);

  return success();
}
// LogicalResult streamhls::MemRefAccess::getRelevantAccessRelation(FlatAffineRelation &rel, SmallVectorImpl<Operation *> *irrelevantOps) const {
//   // Get access relation from access map.
//   AffineValueMap accessValueMap;
//   getAccessMap(&accessValueMap);

//   if (failed(getRelationFromMap(accessValueMap, rel)))
//     return failure();
//   // LLVM_DEBUG(
//   //   llvm::dbgs() << "accessRel: ";
//   //   rel.print(llvm::dbgs());
//   //   llvm::dbgs() << "\n";
//   // );
//   // Create set corresponding to domain of access.
//   FlatAffineValueConstraints domain;
//   if (failed(getOpIndexSet(opInst, &domain, irrelevantOps, /*relevantOnly=*/true)))
//     return failure();
//   // LLVM_DEBUG(
//   //   llvm::dbgs() << "domain: ";
//   //   domain.print(llvm::dbgs());
//   //   llvm::dbgs() << "\n";
//   // );
//   FlatAffineRelation domainRel(rel.getNumDomainDims(), /*numRangeDims=*/0,
//                                domain);

//   // Merge and align domain ids of `ret` and ids of `domain`. Since the domain
//   // of the access map is a subset of the domain of access, the domain ids of
//   // `ret` are guranteed to be a subset of ids of `domain`.
//   for (unsigned i = 0, e = domain.getNumDimVars(); i < e; ++i) {
//     unsigned loc;
//     if (rel.findVar(domain.getValue(i), &loc)) {
//       rel.swapVar(i, loc);
//     } else {
//       rel.insertDomainVar(i);
//       rel.setValue(i, domain.getValue(i));
//     }
//   }

//   // Append domain constraints to `rel`.
//   domainRel.appendRangeVar(rel.getNumRangeDims());
//   domainRel.mergeSymbolVars(rel);
//   domainRel.mergeLocalVars(rel);
//   rel.append(domainRel);

//   return success();
// }

// Populates 'accessMap' with composition of AffineApplyOps reachable from
// indices of MemRefAccess.
void streamhls::MemRefAccess::getAccessMap(AffineValueMap *accessMap) const {
  // Get affine map from AffineLoad/Store.
  AffineMap map;
  if (auto loadOp = dyn_cast<AffineReadOpInterface>(opInst))
    map = loadOp.getAffineMap();
  else
    map = cast<AffineWriteOpInterface>(opInst).getAffineMap();

  SmallVector<Value, 8> operands(indices.begin(), indices.end());
  fullyComposeAffineMapAndOperands(&map, &operands);
  map = simplifyAffineMap(map);
  canonicalizeMapAndOperands(&map, &operands);
  accessMap->reset(map, operands);
}

// my utils

static FlatAffineRelation createTimeIVsRelation(Operation* operation, bool isRelevantFlag = false){
  // Create set corresponding to domain of access.
  AffineStoreOp storeOp;
  AffineLoadOp loadOp;
  if(!isa<AffineStoreOp, AffineLoadOp>(operation)){
    assert(false && "operation is not a store or load");
  }else{
    if(isa<AffineStoreOp>(operation)){
      storeOp = cast<AffineStoreOp>(operation);
    }
    if(isa<AffineLoadOp>(operation)){
      loadOp = cast<AffineLoadOp>(operation);
    }
  }
  AffineLoopBand band;
  auto innerMostForOp = operation->getParentOfType<AffineForOp>();
  auto outerMostForOp = getLoopBandFromInnermost(innerMostForOp, band);
  // SmallVector<Operation*, 4> relevantOps;
  // SmallVector<Operation*, 4> irrelevantOps;
  // getRelevantAndIrrelevantEnclosingAffineOps(*operation, &relevantOps, &irrelevantOps);
  // operation->dump();
  // operation->dump();
  // llvm::dbgs() << "relevantOps.size(): " << relevantOps.size() << "\n";
  // llvm::dbgs() << "irrelevantOps.size(): " << irrelevantOps.size() << "\n";
  if(!isPerfectlyNested(band)) // TODO: add more checks
    assert(false && "band is not perfectly nested");

  SmallVector<IV, 4> ivs;
  // SmallPtrSet<AffineForOp, 4> visitedforOps;
  // for(auto op : relevantOps){
  //   if(auto forOp = dyn_cast<AffineForOp>(op)){
  //     // if forOp is vitisted skip it
  //     if(visitedforOps.count(forOp) > 0)
  //       continue;
  //   }else if(auto ifOp = dyn_cast<AffineIfOp>(op)){
  //     auto cond = ifOp.getIntegerSet();
  //     auto operands = ifOp.getOperands();
  //     for(auto operandPair : llvm::enumerate(operands)){
  //       auto idx = operandPair.index();
  //       auto operand = operandPair.value();
  //       auto isEq = cond.isEq(idx);
  //       // we need to check which loop this operand is dependent on
  //       auto loop = operand.getParentBlock()->getParentOp();
  //       if(!isa<AffineForOp>(loop)){
  //         assert(false && "only for/if ops are allowed");
  //       }
  //       auto forOp = cast<AffineForOp>(loop);
  //       visitedforOps.insert(forOp);
  //       if(isEq)
  //         continue;
  //       auto expr = cond.getConstraint(idx);
  //       if(!isa<AffineBinaryOpExpr>(expr))
  //         continue;
  //       auto binaryExpr = cast<AffineBinaryOpExpr>(expr);
  //       auto rhs = 
  //       auto iv = IV{forOp.getInductionVar(), lb, ub, tripCount};
  //       llvm::dbgs() << "lb: " << lb << ", ub: " << ub << ", tripCount: " << tripCount << "\n";
  //       ivs.push_back(iv);
  //     }
  //   }else {
  //     assert(false && "only for/if ops are allowed");
  //   }
  // }
  for(auto forOp : band){
    // check if the iv is relevant
    auto iter = forOp.getInductionVar();
    if(isRelevantFlag){
      auto isRelevant = false;
      for(auto op : iter.getUsers()){
        if(op == operation){
          isRelevant = true;
          break;
        }
      }
      if(!isRelevant)
        continue;
    }
    if(!forOp.hasConstantBounds())
      assert(false && "bounds are not constant");
    auto lb = forOp.getConstantLowerBound();
    if(lb != 0)
      assert(false && "lower bound is not 0");
    auto ub = forOp.getConstantUpperBound();
    auto tripCount = ub - lb;
    auto step = forOp.getStep();
    auto iv = IV{forOp.getInductionVar(), lb, ub, tripCount};
    // llvm::dbgs() << "lb: " << lb << ", ub: " << ub << ", tripCount: " << tripCount << "\n";
    ivs.push_back(iv);
  }
  // create relation
  // creating timestamp equation
  SmallVector<int64_t, 4> timeEq = {-1};
  auto idx = 0;
  for(auto iv : ivs){
    auto var = 1;
    // iterate over ivs starting from current iv to the end of ivs
    for(auto i = idx + 1; i < ivs.size(); i++){
      var *= ivs[i].tripCount;
    }
    idx++;
    timeEq.push_back(var);
  }
  timeEq.push_back(0);
  SmallVector<SmallVector<int64_t, 4>, 4> inequalities;

  // creating constraints
  auto colIdx = 0;
  for(auto iv : ivs){
    SmallVector<int64_t, 4> lbConstraint = {0}; // for time col
    SmallVector<int64_t, 4> ubConstraint = {0}; // for time col
    for(auto i = 0; i < ivs.size(); i++){
      if(i == colIdx){
        lbConstraint.push_back(1);
        ubConstraint.push_back(-1);
      }else{
        lbConstraint.push_back(0);
        ubConstraint.push_back(0);
      }
    }
    lbConstraint.push_back(iv.lb);
    ubConstraint.push_back(iv.ub-1);
    inequalities.push_back(lbConstraint);
    inequalities.push_back(ubConstraint);
    colIdx++;
  }
  // time bounds
  auto bandTripCount = 1;
  for(auto iv : ivs){
    bandTripCount *= iv.tripCount;
  }
  SmallVector<int64_t, 4> lbConstraint = {1}; // for time col
  SmallVector<int64_t, 4> ubConstraint = {-1}; // for time col
  for(auto i = 0; i < ivs.size(); i++){
    lbConstraint.push_back(0);
    ubConstraint.push_back(0);
  }
  lbConstraint.push_back(0);
  ubConstraint.push_back(bandTripCount-1);
  inequalities.push_back(lbConstraint);
  inequalities.push_back(ubConstraint);

  // create relation
  FlatAffineRelation relation(
    (unsigned) inequalities.size(), // numInequalities
    (unsigned) timeEq.size(), // numEqualities
    (unsigned) 1 + ivs.size() + 1, // cols (time + ivs + const)
    (unsigned) 1, // domain size (time)
    (unsigned) ivs.size(), // range size (ivs)
    (unsigned) 0, // numSymbols
    (unsigned) 0 // numLocals
  );
  relation.setSpace(presburger::PresburgerSpace::getRelationSpace(1,(unsigned) ivs.size(),0,0));
  
  colIdx = 1;
  for(auto iv : ivs){
    relation.setValue(colIdx, ivs[colIdx-1].value);
    colIdx++;
  }
  relation.addEquality(timeEq);
  for(auto ineq : inequalities){
    relation.addInequality(ineq);
  }
  return relation;
}


// this function removes irrelevant ivs from the access relation
// if the iv is constant with respect to the memory access, it is removed
void streamhls::getRelevantAccessRelation(
  FlatAffineRelation &access,
  SmallVector<Value, 4> *irrelevantIVs,
  SmallVector<int64_t, 4> *irrelevantLBs,
  SmallVector<int64_t, 4> *irrelevantUBs
){
  // access.dump();
  // llvm::dbgs() << "access.getNumDomainDims(): " << access.getNumDomainDims() << "\n"; 
  SmallVector<unsigned> equalitiesToRemove;
  // skip the ivs, and check the range if it does not contain any non-zero values
  for(unsigned row = 0; row < access.getNumEqualities(); row++){
    auto numDomain = access.getNumDomainDims();
    auto numCols = access.getNumCols();
    bool nonZero = false;
    for(unsigned col = numDomain; col < numCols - 1; col++){
      LLVM_DEBUG(
        // llvm::dbgs() << "access.atIneq64(row, col): " << access.atEq64(row, col) << "\n";
        // llvm::dbgs() << "access.atIneq64("<< row << ", " << col << "): " << access.atEq64(row, col) << "\n";
      );
      if(access.atEq64(row, col) != 0){
        nonZero = true;
        break;
      }
    }
    if(!nonZero){
      // LLVM_DEBUG(
      //   llvm::dbgs() << "Zero row: " << row << "\n";
      // );
      equalitiesToRemove.push_back(row);
    }
  }
  // LLVM_DEBUG(
  //   llvm::dbgs() << "equalitiesToRemove.size(): " << equalitiesToRemove.size() << "\n";
  // );
  for(auto row : equalitiesToRemove){
    access.removeEquality(row);
    // reduce all equalitiesToRemove by 1
    for(auto &r : equalitiesToRemove){
      if(r > row){
        r--;
      }
    }
  }
  // access.dump();
  // SmallVector<Value, 4> unitSizeIVs;
  // check if any column is all zeros for the equalities
  // llvm::dbgs() << "Cols: " << access.getNumDimAndSymbolVars() << "\n";
  // llvm::dbgs() << "Rows: " << access.getNumEqualities() << "\n";
  // llvm::dbgs() << "Num rows: " << access.getNumConstraints() << " | " << access.getNumEqualities() + access.getNumInequalities() << "\n";
  for(unsigned col=0; col<access.getNumDimAndSymbolVars(); col++){
    bool nonRelevant = true;
    // bool unitSize = false;
    for(unsigned row=0; row<access.getNumEqualities(); row++){
      // a hack to remove irrelevant ivs FIXME: remove this hack
      // llvm::dbgs() << access.getEquality(row)[col] << " ";
      if( access.getEquality(row)[col] != 0){
        nonRelevant = false;
      }
    }
    // llvm::dbgs() << "\n";

    SmallVector<unsigned, 4> lbIndices;
    SmallVector<unsigned, 4> ubIndices;
    access.getLowerAndUpperBoundIndices(col, &lbIndices, &ubIndices);
    // llvm::dbgs() << "nonRelevant: " << nonRelevant << "\n";
    // llvm::dbgs() << "lbIndices: ";
    // for(auto lb : lbIndices){
    //   llvm::dbgs() << lb << " ";
    // }
    // llvm::dbgs() << "\n";
    // llvm::dbgs() << "ubIndices: ";
    // for(auto ub : ubIndices){
    //   llvm::dbgs() << ub << " ";
    // }
    // llvm::dbgs() << "\n";
    if(lbIndices.size() >= 1 && ubIndices.size() == 1){
      int64_t lb = access.atIneq64(lbIndices[0], access.getNumCols()-1);
      int64_t ub = access.atIneq64(ubIndices[0], access.getNumCols()-1);
      if(lbIndices.size() == 2){
        lb = -access.atIneq64(lbIndices[1], access.getNumCols()-1);
      }
      // if(lb == 0 && ub == 0){
      //   unitSize = true;
      // }
      // llvm::dbgs() << "lb: " << lb << "\n";
      if(nonRelevant){
        irrelevantLBs->push_back(lb);
        irrelevantUBs->push_back(ub);
        irrelevantIVs->push_back(access.getValue(col));
      }
    } else {
      // LLVM_DEBUG(
      //   llvm::dbgs() << "lbIndices.size() != 1 || ubIndices.size() != 1\n"
      // );
      // assert (lbIndices.size() == 1 && ubIndices.size() == 1 && "lbIndices.size() != 1 || ubIndices.size() != 1");
    }
  }
  // llvm::dbgs() << "\n";
  for(auto var : *irrelevantIVs){
    unsigned pos;
    if(access.findVar(var, &pos)){
      access.removeVar(pos);
    }
  }

  access.removeRedundantConstraints();
  // access.dump();
}

static presburger::SymbolicLexOpt findSymbolicIntegerLexOpt(const presburger::PresburgerRelation &rel,
                                                bool isMin) {
  presburger::SymbolicLexOpt result(rel.getSpace());
  presburger::PWMAFunction &lexopt = result.lexopt;
  presburger::PresburgerSet &unboundedDomain = result.unboundedDomain;
  for (const presburger::IntegerRelation &cs : rel.getAllDisjuncts()) {
    presburger::SymbolicLexOpt s(rel.getSpace());
    if (isMin) {
      s = cs.findSymbolicIntegerLexMin();
      lexopt = lexopt.unionLexMin(s.lexopt);
    } else {
      s = cs.findSymbolicIntegerLexMax();
      lexopt = lexopt.unionLexMax(s.lexopt);
    }
    unboundedDomain = unboundedDomain.intersect(s.unboundedDomain);
  }
  return result;
}

static uint64_t getAccessLexOpt(presburger::IntegerRelation &rel, SmallVectorImpl<int64_t> &elementIndices, bool isMin = false){
  // presburger::PresburgerRelation newRel = rel.computeReprWithOnlyDivLocals();
  // newRel.inverse();
  // presburger::SymbolicLexOpt lexOpt = findSymbolicIntegerLexOpt(newRel, isMin);
  // std::optional<SmallVector<presburger::MPInt, 8>> output = lexOpt.lexopt.valueAt(elementIndices);
  // if(output.has_value()){
  //   if(output.value().size() == 1){
  //     presburger::MPInt val = output.value()[0];
  //     return int64FromMPInt(val);
  //   } else {
  //     assert(false && "output.value().size() != 1");
  //   }
  // }else{
  //   assert(false && "could't find lexmax");
  // }
  return 0;
}

// std::string streamhls::getVariableTimeFunction(unsigned nodeId, Operation *op, bool isFirst){
//   if(!isa<AffineLoadOp, AffineStoreOp>(op)){
//     assert(false && "op is not a load or store operation");
//   }
//   AffineLoopBand band;
//   auto innerMostForOp = op->getParentOfType<AffineForOp>();
//   getLoopBandFromInnermost(innerMostForOp, band);
//   std::stringstream equation;
//   for(auto forOpPair : llvm::enumerate(band)){
//     auto idx = forOpPair.index();
//     auto forOp = forOpPair.value();
//     if(isFirst && idx != (band.size() - 1))
//       equation << "0 * ";
//     else if(!isFirst && idx != (band.size() - 1))
//       equation << "(x" << nodeId << "_" << idx << " - 1) * ";
//     else{
//       if(isa<AffineLoadOp>(op))
//         equation << "0";
//       else if(isa<AffineStoreOp>(op))
//         equation << "(x" << nodeId << "_" << idx << " - 1)";
//     }
//     for(unsigned i = idx + 1; i < band.size(); i++){
//       equation <<"x" << nodeId << "_" << i;
//       if(i != band.size() - 1)
//         equation << " * ";
//     }
//     if(idx != band.size() - 1)
//       equation << " + ";
//   }
//   return equation.str();
//   // // SmallVector<Operation*, 4> relevantOps;
//   // // SmallVector<Operation*, 4> irrelevantOps;
//   // // getRelevantAndIrrelevantEnclosingAffineOps(*op, &relevantOps, &irrelevantOps);
//   // SmallVector<IV, 4> loopsInfo;
//   // for(auto forOp : band){
//   //   if(!forOp.hasConstantBounds())
//   //     assert(false && "getVariableTimeFunction: bounds are not constant");
//   //   auto lb = forOp.getConstantLowerBound();
//   //   auto ub = forOp.getConstantUpperBound();
//   //   auto tripCount = ub - lb;
//   //   auto step = forOp.getStep();
//   //   // llvm::dbgs() << "lb: " << lb << ", ub: " << ub << ", tripCount: " << tripCount << "\n";
//   //   IV info = {forOp.getInductionVar(), lb, ub, tripCount};
//   //   loopsInfo.push_back(info);
//   // }
//   // // reverse tripCounts
//   // std::reverse(loopsInfo.begin(), loopsInfo.end());
//   // // map inputs
//   // // array shape
//   // // ArrayRef<int64_t> shape;
//   // SmallVector<Value, 4> operands;
//   // AffineMap accessMap;
//   // if(auto loadOp = dyn_cast<AffineLoadOp>(op)){
//   //   operands = loadOp.getMapOperands();
//   //   accessMap = loadOp.getAffineMap();
//   // }else if(auto storeOp = dyn_cast<AffineStoreOp>(op)){
//   //   operands = storeOp.getMapOperands();
//   //   accessMap = storeOp.getAffineMap();
//   // }
//   // // llvm::dbgs() << "operands.size(): " << operands.size() << "\n";
//   // // accessMap.dump();
//   // // for(auto iv : loopsInfo){
//   // //   llvm::dbgs() << "lb: " << iv.lb << ", ub: " << iv.ub << ", tripCount: " << iv.tripCount << "\n";
//   // // }
//   // // traverse loopsInfo in reverse order
//   // int64_t currFactor = 1;
//   // AffineExpr timeExpr = getAffineConstantExpr(0, op->getContext());
//   // auto relevantOpIdx = 0;
//   // for(size_t idx = 0; idx < loopsInfo.size(); idx++){
//   //   auto currIV = loopsInfo[idx].value;
//   //   auto currUB = loopsInfo[idx].ub;
//   //   auto prevTripCount = 0;
//   //   // auto prevUB = 0;
//   //   if(idx > 0){
//   //     prevTripCount = loopsInfo[idx-1].tripCount;
//   //     // prevUB = loopsInfo[idx-1].ub;
//   //   }else{
//   //     prevTripCount = 1;
//   //     // prevUB = 1;
//   //   }
//   //   currFactor *= prevTripCount;
//   //   // if iv.value in operands
//   //   if(std::find(operands.begin(), operands.end(), currIV) != operands.end()){
//   //     // add iv.value to the map
//   //     // add currFactor to the map
//   //     // AffineExpr ivExpr = getAffineDimExpr(relevantOpIdx, op->getContext());
//   //     AffineExpr ivExpr;
//   //     if(isFirst)
//   //       ivExpr = getAffineConstantExpr(0, op->getContext());
//   //     else
//   //       ivExpr = getAffineConstantExpr(currUB - 1, op->getContext());
//   //     timeExpr = timeExpr + ivExpr * currFactor;
//   //     relevantOpIdx++;
//   //   }else{
//   //     // add 0 to the map if load
//   //     if(isa<AffineLoadOp>(op))
//   //       timeExpr = timeExpr + getAffineConstantExpr(0, op->getContext()) * currFactor;
//   //     else if(isa<AffineStoreOp>(op))
//   //       timeExpr = timeExpr + getAffineConstantExpr(currUB - 1, op->getContext()) * currFactor;
//   //   }
//   // }
//   // // timeExpr.dump();
//   // uint64_t timeVal = -1;
//   // if(timeExpr.getKind() == AffineExprKind::Constant){
//   //   // cast to uint64_t
//   //   timeVal = timeExpr.cast<AffineConstantExpr>().getValue();
//   // }
//   // if(timeVal == -1){
//   //   assert(false && "timeExpr.getKind() != AffineExprKind::Constant");
//   // }
//   // return "X";
//   // timeExpr.dump();
//   // create map
//   // return AffineMap::get(relevantOpIdx, 0, timeExpr);
// }
// uint64_t streamhls::evalTimeFunctionAt(AffineMap map, ArrayRef<int64_t> indices){
//   if(map.getNumResults() != 1){
//     assert(false && "map.getNumResults() != 1");
//   }
//   auto expr = map.getResult(0);
//   auto context = map.getContext();
//   auto time = expr.replaceDims(getAffineConstantExprs(indices, context));
//   if(time.getKind() != AffineExprKind::Constant)
//     assert(false && "time.getKind() != AffineExprKind::Constant");
//   uint64_t timeVal = time.cast<AffineConstantExpr>().getValue();
//   return timeVal;
// }
// AffineMap streamhls::getAccessPattern(Operation *op){
//   if(!isa<AffineLoadOp, AffineStoreOp>(op)){
//     assert(false && "op is not a load or store operation");
//   }
//   SmallVector<Operation*, 4> relevantOps;
//   SmallVector<Operation*, 4> irrelevantOps;
//   getRelevantAndIrrelevantEnclosingAffineOps(*op, &relevantOps, &irrelevantOps);

//   // handle load or store
//   SmallVector<Value, 4> operands;
//   if(auto loadOp = dyn_cast<AffineLoadOp>(op)){
//     operands = loadOp.getMapOperands();
//   }else if(auto storeOp = dyn_cast<AffineStoreOp>(op)){
//     operands = storeOp.getMapOperands();
//   }


//   SmallVector<unsigned, 4> perm;
//   for(auto operandPair : llvm::enumerate(operands)){
//     auto operand = operandPair.value();
//     for(auto relevantOpPair : llvm::enumerate(relevantOps)){
//       auto loopIdx = relevantOpPair.index();
//       auto relevantOp = relevantOpPair.value();
//       auto forOp = dyn_cast<AffineForOp>(relevantOp);
//       if(forOp){
//         if(operand == forOp.getInductionVar()){
//           perm.push_back(loopIdx);
//         }
//       }
//     }
//   }
//   // for(auto num : perm){
//   //   llvm::dbgs() << num << " ";
//   // }
//   // llvm::dbgs() << "\n\n";
//   return AffineMap::getPermutationMap(perm, op->getContext());
// }



AffineMap streamhls::getAccessPattern(Operation *op){
  if(!isa<AffineLoadOp, AffineStoreOp>(op)){
    assert(false && "op is not a load or store operation");
  }

  // handle load or store
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


  // AffineMap simplifiedAccessMap;
  AffineLoopBand band;
  auto innerMostForOp = op->getParentOfType<AffineForOp>();
  getLoopBandFromInnermost(innerMostForOp, band); // 0 is outermost loop
  if(accessMap.getNumSymbols() > 0){
    accessMap = fixAccessMap(accessMap);
  }
  // accessMap.dump();
  // for(auto dim = 0; dim < accessMap.getNumDims(); dim++){
  //   auto expr = getAffineDimExpr(dim, op->getContext());
  //   llvm::dbgs() << accessMap.getResultPosition(expr) << " ";
  // }
  // llvm::dbgs() << "\n";
  // get loops relevant to the access
  SmallVector<AffineForOp, 4> relevantLoops; 
  SmallVector<Value, 4> relevantOperands;
  for(auto forOp : band){
    auto iv = forOp.getInductionVar();
    // if iv is in mapOperands, add to relevantLoops
    if(std::find(mapOperands.begin(), mapOperands.end(), iv) != mapOperands.end()){
      relevantLoops.push_back(forOp);
      relevantOperands.push_back(iv);
    }
  }
  if(relevantLoops.size() != mapOperands.size()){
    assert(false && "relevantLoops.size() != mapOperands.size()");
  }

  // create correct ordering map
  SmallVector<unsigned, 4> orderingIndices1;
  SmallVector<Value, 4> orderingOperands1;
  orderingIndices1.resize(relevantLoops.size());
  orderingOperands1.resize(relevantLoops.size());
  SmallVector<unsigned, 4> orderingIndices2;
  SmallVector<Value, 4> orderingOperands2;
  // unsigned i = 0;
  // for(auto operand : mapOperands){
  //   auto idx = std::find(relevantOperands.begin(), relevantOperands.end(), operand) - relevantOperands.begin();
  //   orderingIndices1[idx] = i++;
  //   orderingIndices2.push_back(idx);
  // }
  for(auto operandPair1 : llvm::enumerate(mapOperands)){
    for(auto operandPair2 : llvm::enumerate(relevantOperands)){
      if(operandPair1.value() == operandPair2.value()){
        orderingIndices1[operandPair2.index()] = operandPair1.index();
        orderingOperands1[operandPair2.index()] = operandPair1.value();
        orderingIndices2.push_back(operandPair2.index());
        orderingOperands2.push_back(operandPair2.value());
      }
    }
  }
  // llvm::dbgs() << "orderingIndices1: ";
  // for(auto idx : orderingIndices1){
  //   llvm::dbgs() << idx << " ";
  // }
  // llvm::dbgs() << "\n";
  // llvm::dbgs() << "orderingIndices2: ";
  // for(auto idx : orderingIndices2){
  //   llvm::dbgs() << idx << " ";
  // }
  // llvm::dbgs() << "\n";
  // auto map1 = AffineMap::getPermutationMap(orderingIndices1, op->getContext());
  auto map2 = AffineMap::getPermutationMap(orderingIndices2, op->getContext());

  // map2.dump();
  // accessMap.dump();
  // accessMap.compose(map1).dump();
  // accessMap.compose(map2).dump();
  // map1.compose(accessMap).dump();
  // map2.compose(accessMap).dump();
  // SmallVector<unsigned, 4> perm;
  // for(auto operandPair1 : llvm::enumerate(relevantOperands)){
  //   for(auto operandPair2 : llvm::enumerate(orderingOperands2)){
  //     if(operandPair1.value() == operandPair2.value()){
  //       perm.push_back(operandPair2.index());
  //     }
  //   }
  // }
  // llvm::dbgs() << "perm: ";
  // for(auto idx : perm){
  //   llvm::dbgs() << idx << " ";
  // }
  // llvm::dbgs() << "\n";
  // accessMap.dump();
  
  SmallVector<unsigned, 4> perm;
  SmallVector<unsigned, 4> perm2;
  SmallVector<unsigned, 4> perm3;

  perm2.resize(relevantOperands.size());
  perm3.resize(relevantOperands.size());

  // auto newOrdering = inversePerm(orderingIndices2);
  for(auto idx = 0; idx < orderingIndices2.size(); idx++){
    // auto operand = operandPair.value();
    auto operand = relevantOperands[orderingIndices2[idx]];
    for(auto relevantOpPair : llvm::enumerate(relevantOperands)){
      auto loopIdx = relevantOpPair.index();
      auto relevantOperand = relevantOpPair.value();
      if(operand == relevantOperand){
        perm.push_back(loopIdx);
        perm2[loopIdx] = orderingIndices2[idx]; //identity
        perm3[orderingIndices2[idx]] = loopIdx;
      }
    }
  }
  auto newMap = AffineMap::getPermutationMap(perm, op->getContext());
  auto newInverseMap = AffineMap::getPermutationMap(inversePerm(perm), op->getContext());
  // auto newMap3 = AffineMap::getPermutationMap(perm3, op->getContext());
  // newMap.dump();

  // newMap3.dump();
  // newInverseMap.dump();
  // map2.compose(accessMap).dump();
  // if(newMap != map2.compose(accessMap)){
  //   llvm::dbgs() << "====================================================================\n";
  //   newMap.dump();
  //   map2.compose(accessMap).dump();
  // }
  // accessMap.dump();
  // map2.dump();
  // accessMap.compose(map2).dump();
  // accessMap.dump();
  auto outMap = accessMap.compose(map2);
  // outMap.dump();
  // SmallVector<unsigned, 4> locations;
  // for(auto expr : outMap.getResults()){
  //   expr.walk([&](AffineExpr e){
  //     if(e.getKind() == AffineExprKind::DimId){
  //       locations.push_back(e.cast<AffineDimExpr>().getPosition());
  //     }
  //   });
  // }
  // // for(auto loc : locations){
  // //   llvm::dbgs() << loc << " ";
  // // }
  // // llvm::dbgs() << "\n";
  // if(locations.size() != relevantOperands.size()){
  //   assert(false && "locations.size() != relevantOperands.size()");
  // }
  return outMap;
}



LogicalResult streamhls::AccessInfo::getInfoForPerformanceModel(MLIRContext *context){
  if(failed(access.getAccessRelation(relation))){
    LLVM_DEBUG(
      llvm::dbgs() << "access.getAccessRelation failed\n"
    );
    return failure();
  }
  if(failed(getAllInfo(context))){
    LLVM_DEBUG(
      llvm::dbgs() << "getAllInfo failed\n"
    );
    return failure();
  }
  LLVM_DEBUG(
    llvm::dbgs() << "getAllInfo success\n"
  );
  // get time index relations
  auto timeIndexRel = getTimeIndexRelation();
  if(timeIndexRel.isEmpty()){
    LLVM_DEBUG(
      llvm::dbgs() << "timeIndexRelation is empty\n"
    );
    return failure();
  }
  timeIndexRelation = &timeIndexRel;
  auto timeIndexRelevantRel = getTimeIndexRelevantRelation();
  if(timeIndexRelevantRel.isEmpty()){
    LLVM_DEBUG(
      llvm::dbgs() << "timeIndexRelevantRel is empty\n"
    );
    return failure();
  }
  timeIndexRelevantRelation = &timeIndexRelevantRel;
  // get trip count
  auto innerMostForOp = access.opInst->getParentOfType<AffineForOp>();
  getLoopBandFromInnermost(innerMostForOp, loopBand);
  tripCount = getLoopNestIterations(loopBand);
  LLVM_DEBUG(
    llvm::dbgs() << "tripCount: " << tripCount << "\n";
  );
  // get first and last indices
  SmallVector<int64_t, 4> firstElementIndices;
  SmallVector<int64_t, 4> lastElementIndices;
  auto shape = access.memref.getType().cast<MemRefType>().getShape();
  for(unsigned i = 0; i < shape.size(); i++){
    firstElementIndices.push_back(0);
    lastElementIndices.push_back(shape[i] - 1);
  }
  // get first and last element time
  firstElementTime = getAccessLexOpt(timeIndexRel, firstElementIndices, true);
  lastElementTime = getAccessLexOpt(timeIndexRel, lastElementIndices, false);
  LLVM_DEBUG(
    llvm::dbgs() << "firstElementTime: " << firstElementTime << "\n";
    llvm::dbgs() << "lastElementTime: " << lastElementTime << "\n";
  );
  return success();
}
LogicalResult streamhls::AccessInfo::getAllInfo(MLIRContext *context){
  if(failed(access.getAccessRelation(relation))){
    LLVM_DEBUG(
      llvm::dbgs() << "access.getAccessRelation failed\n"
    );
    return failure();
  }
  relevantRelation = FlatAffineRelation(relation);
  getRelevantAccessRelation(
    relevantRelation, 
    &irrelevantIVs,
    &irrelevantLBs,
    &irrelevantUBs
  );
  unsigned idx = 0;
  for(auto iv : irrelevantIVs){
    ifOperands.push_back(iv);
    ifExprs.push_back(
      // rewriter.getAffineDimExpr(idx) -
      mlir::getAffineDimExpr(idx, context) -
      (access.isStore()? irrelevantUBs[idx] : irrelevantLBs[idx])
    );
    ifEqFlags.push_back(true);
    idx++;
  }
  return success();
}

presburger::IntegerRelation streamhls::AccessInfo::getTimeIndexRelation(){
  auto timeIVsRelation = cast<presburger::IntegerRelation>(
    createTimeIVsRelation(access.opInst)
  );
  auto IVsIndexRelation = cast<presburger::IntegerRelation>(
    relation
  );
  if((timeIVsRelation.getSpace().getNumRangeVars() == IVsIndexRelation.getSpace().getNumDomainVars())){
    timeIVsRelation.compose(IVsIndexRelation);
    timeIVsRelation.removeRedundantConstraints();
    timeIVsRelation.removeRedundantInequalities();
    timeIVsRelation.removeTrivialRedundancy();
    return timeIVsRelation;
  } else {
    // return empty relation
    return presburger::IntegerRelation(
      presburger::PresburgerSpace::getRelationSpace(0,0,0,0)
    );
  }
}
presburger::IntegerRelation streamhls::AccessInfo::getTimeIndexRelevantRelation(){
  // get time index relation
  auto timeIVsRelation = cast<presburger::IntegerRelation>(
    createTimeIVsRelation(access.opInst, true)
  );
  auto IVsIndexRelation = cast<presburger::IntegerRelation>(
    relevantRelation
  );
  if((timeIVsRelation.getSpace().getNumRangeVars() == IVsIndexRelation.getSpace().getNumDomainVars())){
    timeIVsRelation.compose(IVsIndexRelation);
    timeIVsRelation.removeRedundantConstraints();
    timeIVsRelation.removeRedundantInequalities();
    timeIVsRelation.removeTrivialRedundancy();
    return timeIVsRelation;
  } else {
    // return empty relation
    return presburger::IntegerRelation(
      presburger::PresburgerSpace::getRelationSpace(0,0,0,0)
    );
  }
}
presburger::PresburgerSpace streamhls::AccessInfo::getSpace(){
  return relation.getSpace();
}
presburger::PresburgerSpace streamhls::AccessInfo::getRelevantSpace(){
  return relevantRelation.getSpace();
}

LogicalResult streamhls::reorderOps(
  SmallVector<AffineLoadOp, 4> loadOps,
  SmallVector<AffineStoreOp, 4> storeOps,
  StoreLoadStoreLoadOps &ops
){
  auto loadOp1 = loadOps[0];
  auto loadOp2 = loadOps[1];
  auto storeOp1 = storeOps[0];
  auto storeOp2 = storeOps[1];
  if(loadOp1->getBlock() == storeOp1->getBlock()){
    ops.storeOp = storeOp2;
    ops.loopStoreOp = storeOp1;
    ops.loopLoadOp = loadOp1;
    ops.loadOp = loadOp2;
    return success();
  }else if(loadOp1->getBlock() == storeOp2->getBlock()){
    ops.storeOp = storeOp1;
    ops.loopStoreOp = storeOp2;
    ops.loopLoadOp = loadOp1;
    ops.loadOp = loadOp2;
    return success();
  }else if(loadOp2->getBlock() == storeOp1->getBlock()){
    ops.storeOp = storeOp2;
    ops.loopStoreOp = storeOp1;
    ops.loopLoadOp = loadOp2;
    ops.loadOp = loadOp1;
    return success();
  }else if(loadOp2->getBlock() == storeOp2->getBlock()){
    ops.storeOp = storeOp1;
    ops.loopStoreOp = storeOp2;
    ops.loopLoadOp = loadOp2;
    ops.loadOp = loadOp1;
    return success();
  } else{
    return failure();
  }
}

streamhls::FifoAccess::FifoAccess(Operation *readOrWriteOpInst) {
  if (auto readOp = dyn_cast<StreamReadOp>(readOrWriteOpInst)) {
    fifo = readOp.getChannel();
    opInst = readOrWriteOpInst;
  } else {
    assert(isa<StreamWriteOp>(readOrWriteOpInst) &&
           "Fifo read/write op expected");
    auto writeOp = cast<StreamWriteOp>(readOrWriteOpInst);
    fifo = writeOp.getChannel();
    opInst = readOrWriteOpInst;
  }
}
static FlatAffineRelation createTimeFifoRelation(Operation* operation){
  // Create set corresponding to domain of access.
  StreamReadOp streamRead;
  StreamWriteOp streamWrite;
  if(!isa<StreamReadOp, StreamWriteOp>(operation)){
    assert(false && "operation is not a read or write");
  }else{
    if(isa<StreamReadOp>(operation)){
      streamRead = cast<StreamReadOp>(operation);
    }
    if(isa<StreamWriteOp>(operation)){
      streamWrite = cast<StreamWriteOp>(operation);
    }
  }
  AffineLoopBand band;
  auto innerMostForOp = operation->getParentOfType<AffineForOp>();
  auto outerMostForOp = getLoopBandFromInnermost(innerMostForOp, band);

  if(!isPerfectlyNested(band)) // TODO: add more checks
    assert(false && "band is not perfectly nested");

  SmallVector<IV, 4> ivs;
  for(auto forOp : band){
    // check if the iv is relevant
    auto iter = forOp.getInductionVar();
    if(!forOp.hasConstantBounds())
      assert(false && "bounds are not constant");
    auto lb = forOp.getConstantLowerBound();
    if(lb != 0)
      assert(false && "lower bound is not 0");
    auto ub = forOp.getConstantUpperBound();
    auto tripCount = ub - lb;
    auto step = forOp.getStep();
    auto iv = IV{forOp.getInductionVar(), lb, ub, tripCount};
    ivs.push_back(iv);
  }
  // create relation
  // creating timestamp equation
  SmallVector<int64_t, 4> timeEq = {-1};
  auto idx = 0;
  for(auto iv : ivs){
    auto var = 1;
    // iterate over ivs starting from current iv to the end of ivs
    for(auto i = idx + 1; i < ivs.size(); i++){
      var *= ivs[i].tripCount;
    }
    idx++;
    timeEq.push_back(var);
  }
  timeEq.push_back(0);
  SmallVector<SmallVector<int64_t, 4>, 4> inequalities;

  // creating constraints
  auto colIdx = 0;
  for(auto iv : ivs){
    SmallVector<int64_t, 4> lbConstraint = {0}; // for time col
    SmallVector<int64_t, 4> ubConstraint = {0}; // for time col
    for(auto i = 0; i < ivs.size(); i++){
      if(i == colIdx){
        lbConstraint.push_back(1);
        ubConstraint.push_back(-1);
      }else{
        lbConstraint.push_back(0);
        ubConstraint.push_back(0);
      }
    }
    lbConstraint.push_back(iv.lb);
    ubConstraint.push_back(iv.ub-1);
    inequalities.push_back(lbConstraint);
    inequalities.push_back(ubConstraint);
    colIdx++;
  }
  // time bounds
  auto bandTripCount = 1;
  for(auto iv : ivs){
    bandTripCount *= iv.tripCount;
  }
  SmallVector<int64_t, 4> lbConstraint = {1}; // for time col
  SmallVector<int64_t, 4> ubConstraint = {-1}; // for time col
  for(auto i = 0; i < ivs.size(); i++){
    lbConstraint.push_back(0);
    ubConstraint.push_back(0);
  }
  lbConstraint.push_back(0);
  ubConstraint.push_back(bandTripCount-1);
  inequalities.push_back(lbConstraint);
  inequalities.push_back(ubConstraint);

  // create relation
  FlatAffineRelation relation(
    (unsigned) inequalities.size(), // numInequalities
    (unsigned) timeEq.size(), // numEqualities
    (unsigned) 1 + ivs.size() + 1, // cols (time + ivs + const)
    (unsigned) 1, // domain size (time)
    (unsigned) ivs.size(), // range size (ivs)
    (unsigned) 0, // numSymbols
    (unsigned) 0 // numLocals
  );
  relation.setSpace(presburger::PresburgerSpace::getRelationSpace(1,(unsigned) ivs.size(),0,0));
  
  colIdx = 1;
  for(auto iv : ivs){
    relation.setValue(colIdx, ivs[colIdx-1].value);
    colIdx++;
  }
  relation.addEquality(timeEq);
  for(auto ineq : inequalities){
    relation.addInequality(ineq);
  }
  return relation;
}

LogicalResult streamhls::FifoAccess::getFirstOutputTime(int64_t &lexmin) const {
// Create set corresponding to domain of access.
  FlatAffineValueConstraints domain;
  if (failed(getOpIndexSet(opInst, &domain)))
    return failure();
  IntegerRelation timeRel = cast<IntegerRelation>(domain);

  IntegerRelation timeFifoRel = cast<IntegerRelation>(createTimeFifoRelation(opInst));
  // timeRel.dump();
  timeFifoRel.inverse();
  // timeFifoRel.dump();
  timeRel.compose(timeFifoRel);
  // timeRel.dump();
  auto lexMin = timeRel.findIntegerLexMin().getOptimumIfBounded();
  if(lexMin.value().size() != 1)
    return failure();
  auto vec = getInt64Vec(lexMin.value());
  lexmin = vec[0];
  return success();
  
  // for(auto val : vec){
  //   LLVM_DEBUG(
  //     llvm::dbgs() << val << " "
  //   );
  // }
  // LLVM_DEBUG(
  //   llvm::dbgs() << "\n"
  // );
  
  // // Get access relation from access map.
  // AffineValueMap accessValueMap;
  // getAccessMap(&accessValueMap);
  // if (failed(getRelationFromMap(accessValueMap, rel)))
  //   return failure();

  // FlatAffineRelation domainRel(rel.getNumDomainDims(), /*numRangeDims=*/0,
  //                              domain);

  // // Merge and align domain ids of `ret` and ids of `domain`. Since the domain
  // // of the access map is a subset of the domain of access, the domain ids of
  // // `ret` are guranteed to be a subset of ids of `domain`.
  // for (unsigned i = 0, e = domain.getNumDimVars(); i < e; ++i) {
  //   unsigned loc;
  //   if (rel.findVar(domain.getValue(i), &loc)) {
  //     rel.swapVar(i, loc);
  //   } else {
  //     rel.insertDomainVar(i);
  //     rel.setValue(i, domain.getValue(i));
  //   }
  // }

  // // Append domain constraints to `rel`.
  // domainRel.appendRangeVar(rel.getNumRangeDims());
  // domainRel.mergeSymbolVars(rel);
  // domainRel.mergeLocalVars(rel);
  // rel.append(domainRel);

  // // TODO: Make sure setting space is always correct
  // rel.setSpace(presburger::PresburgerSpace::getRelationSpace(
  //     rel.getNumDomainDims(), // domain size
  //     rel.getNumRangeDims(), // range size
  //     0, // num symbols
  //     rel.getNumLocalVars() // num locals
  //   ));
  // return success();
  
}

static FlatAffineRelation createTimeMemRefRelation(Operation* operation){
  // Create set corresponding to domain of access.
  AffineStoreOp storeOp;
  AffineLoadOp loadOp;
  if(!isa<AffineStoreOp, AffineLoadOp>(operation)){
    assert(false && "operation is not a store or load");
  }else{
    if(isa<AffineStoreOp>(operation)){
      storeOp = cast<AffineStoreOp>(operation);
    }
    if(isa<AffineLoadOp>(operation)){
      loadOp = cast<AffineLoadOp>(operation);
    }
  }
  AffineLoopBand band;
  auto innerMostForOp = operation->getParentOfType<AffineForOp>();
  auto outerMostForOp = getLoopBandFromInnermost(innerMostForOp, band);

  if(!isPerfectlyNested(band)) // TODO: add more checks
    assert(false && "band is not perfectly nested");

  SmallVector<IV, 4> ivs;
  for(auto forOp : band){
    // check if the iv is relevant
    auto iter = forOp.getInductionVar();
    if(!forOp.hasConstantBounds())
      assert(false && "bounds are not constant");
    auto lb = forOp.getConstantLowerBound();
    if(lb != 0)
      assert(false && "lower bound is not 0");
    auto ub = forOp.getConstantUpperBound();
    auto tripCount = ub - lb;
    auto step = forOp.getStep();
    auto iv = IV{forOp.getInductionVar(), lb, ub, tripCount};
    ivs.push_back(iv);
  }
  // create relation
  // creating timestamp equation
  SmallVector<int64_t, 4> timeEq = {-1};
  auto idx = 0;
  for(auto iv : ivs){
    auto var = 1;
    // iterate over ivs starting from current iv to the end of ivs
    for(auto i = idx + 1; i < ivs.size(); i++){
      var *= ivs[i].tripCount;
    }
    idx++;
    timeEq.push_back(var);
  }
  timeEq.push_back(0);
  SmallVector<SmallVector<int64_t, 4>, 4> inequalities;

  // creating constraints
  auto colIdx = 0;
  for(auto iv : ivs){
    SmallVector<int64_t, 4> lbConstraint = {0}; // for time col
    SmallVector<int64_t, 4> ubConstraint = {0}; // for time col
    for(auto i = 0; i < ivs.size(); i++){
      if(i == colIdx){
        lbConstraint.push_back(1);
        ubConstraint.push_back(-1);
      }else{
        lbConstraint.push_back(0);
        ubConstraint.push_back(0);
      }
    }
    lbConstraint.push_back(iv.lb);
    ubConstraint.push_back(iv.ub-1);
    inequalities.push_back(lbConstraint);
    inequalities.push_back(ubConstraint);
    colIdx++;
  }
  // time bounds
  auto bandTripCount = 1;
  for(auto iv : ivs){
    bandTripCount *= iv.tripCount;
  }
  SmallVector<int64_t, 4> lbConstraint = {1}; // for time col
  SmallVector<int64_t, 4> ubConstraint = {-1}; // for time col
  for(auto i = 0; i < ivs.size(); i++){
    lbConstraint.push_back(0);
    ubConstraint.push_back(0);
  }
  lbConstraint.push_back(0);
  ubConstraint.push_back(bandTripCount-1);
  inequalities.push_back(lbConstraint);
  inequalities.push_back(ubConstraint);

  // create relation
  FlatAffineRelation relation(
    (unsigned) inequalities.size(), // numInequalities
    (unsigned) timeEq.size(), // numEqualities
    (unsigned) 1 + ivs.size() + 1, // cols (time + ivs + const)
    (unsigned) 1, // domain size (time)
    (unsigned) ivs.size(), // range size (ivs)
    (unsigned) 0, // numSymbols
    (unsigned) 0 // numLocals
  );
  relation.setSpace(presburger::PresburgerSpace::getRelationSpace(1,(unsigned) ivs.size(),0,0));
  
  colIdx = 1;
  for(auto iv : ivs){
    relation.setValue(colIdx, ivs[colIdx-1].value);
    colIdx++;
  }
  relation.addEquality(timeEq);
  for(auto ineq : inequalities){
    relation.addInequality(ineq);
  }
  return relation;
}

LogicalResult streamhls::MemRefAccess::getFirstOutputTime(int64_t &lexmin) const {
// Create set corresponding to domain of access.
  FlatAffineValueConstraints domain;
  if (failed(getOpIndexSet(opInst, &domain)))
    return failure();
  IntegerRelation timeRel = cast<IntegerRelation>(domain);

  IntegerRelation timeFifoRel = cast<IntegerRelation>(createTimeMemRefRelation(opInst));
  // timeRel.dump();
  timeFifoRel.inverse();
  // timeFifoRel.dump();
  timeRel.compose(timeFifoRel);
  // timeRel.dump();
  auto lexMin = timeRel.findIntegerLexMin().getOptimumIfBounded();
  if(lexMin.value().size() != 1)
    return failure();
  auto vec = getInt64Vec(lexMin.value());
  lexmin = vec[0];
  return success();
}