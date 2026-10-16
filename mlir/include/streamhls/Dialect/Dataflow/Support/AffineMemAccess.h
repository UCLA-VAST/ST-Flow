
#ifndef STREAMHLS_AFFINEMEMACCESS_H
#define STREAMHLS_AFFINEMEMACCESS_H
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Analysis/Presburger/PWMAFunction.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "streamhls/Dialect/Dataflow/IR/Dataflow.h"

#include <optional>

namespace mlir { namespace streamhls {

using namespace affine;
using namespace dataflow;

FailureOr<AccessInfoAttr> buildAccessInfoForAffineAccess(Operation *accessOp);

void getRelevantAndIrrelevantEnclosingAffineOps(Operation &op,
                                         SmallVectorImpl<Operation *> *relevantOps,
                                         SmallVectorImpl<Operation *> *irrelevantOps
                                         );
void getEnclosingAffineOps(Operation &op,
                                         SmallVectorImpl<Operation *> *ops);

/// Builds a system of constraints with dimensional variables corresponding to
/// the loop IVs of the forOps and AffineIfOp's operands appearing in
/// that order. Bounds of the loop are used to add appropriate inequalities.
/// Constraints from the index sets of AffineIfOp are also added. Any symbols
/// founds in the bound operands are added as symbols in the system. Returns
/// failure for the yet unimplemented cases. `ops` accepts both AffineForOp and
/// AffineIfOp.
//  TODO: handle non-unit strides.
LogicalResult getIndexSet(MutableArrayRef<Operation *> ops,
                          FlatAffineValueConstraints *domain);

/// Computes the iteration domain for 'op' and populates 'indexSet', which
/// encapsulates the constraints involving loops surrounding 'op' and
/// potentially involving any Function symbols. The dimensional variables in
/// 'indexSet' correspond to the loops surrounding 'op' from outermost to
/// innermost.
LogicalResult getOpIndexSet(Operation *op,
                                   FlatAffineValueConstraints *indexSet,
                                   SmallVectorImpl<Operation *> *irrelevantOps = nullptr,
                                   bool relevantOnly = false /*only for load/store ops*/);

struct FifoAccess {
  Value fifo;
  Operation *opInst;
  SmallVector<Value, 4> indices;

  /// Constructs a FifoAccess from a load or store operation.
  explicit FifoAccess(Operation *opInst);
  LogicalResult getFirstOutputTime(int64_t &lexmin) const;
};

/// Encapsulates a memref load or store access information.
struct MemRefAccess {
  Value memref;
  Operation *opInst;
  SmallVector<Value, 4> indices;

  /// Constructs a MemRefAccess from a load or store operation.
  // TODO: add accessors to standard op's load, store, DMA op's to return
  // MemRefAccess, i.e., loadOp->getAccess(), dmaOp->getRead/WriteAccess.
  explicit MemRefAccess(Operation *opInst);
  LogicalResult getFirstOutputTime(int64_t &lexmin) const;

  // Returns the rank of the memref associated with this access.
  unsigned getRank() const;
  // Returns true if this access is of a store op.
  bool isStore() const;

  /// Creates an access relation for the access. An access relation maps
  /// elements of an iteration domain to the element(s) of an array domain
  /// accessed by that iteration of the associated statement through some array
  /// reference. For example, given the MLIR code:
  ///
  /// affine.for %i0 = 0 to 10 {
  ///   affine.for %i1 = 0 to 10 {
  ///     %a = affine.load %arr[%i0 + %i1, %i0 + 2 * %i1] : memref<100x100xf32>
  ///   }
  /// }
  ///
  /// The access relation, assuming that the memory locations for %arr are
  /// represented as %m0, %m1 would be:
  ///
  ///   (%i0, %i1) -> (%m0, %m1)
  ///   %m0 = %i0 + %i1
  ///   %m1 = %i0 + 2 * %i1
  ///   0  <= %i0 < 10
  ///   0  <= %i1 < 10
  ///
  /// Returns failure for yet unimplemented/unsupported cases (see docs of
  /// mlir::getIndexSet and mlir::getRelationFromMap for these cases).
  LogicalResult getAccessRelation(mlir::presburger::IntegerRelation &accessRel) const;

  /// get access relation of relevant loop IVs
  LogicalResult getRelevantAccessRelation(FlatAffineRelation &rel, SmallVectorImpl<Operation *> *irrelevantOps) const;

  /// Populates 'accessMap' with composition of AffineApplyOps reachable from
  /// 'indices'.
  void getAccessMap(AffineValueMap *accessMap) const;

  /// Equal if both affine accesses can be proved to be equivalent at compile
  /// time (considering the memrefs, their respective affine access maps  and
  /// operands). The equality of access functions + operands is checked by
  /// subtracting fully composed value maps, and then simplifying the difference
  /// using the expression flattener.
  /// TODO: this does not account for aliasing of memrefs.
  bool operator==(const MemRefAccess &rhs) const;
  bool operator!=(const MemRefAccess &rhs) const { return !(*this == rhs); }
};

// my utils
void getRelevantAccessRelation(
  FlatAffineRelation &access,
  SmallVector<Value, 4> *irrelevantIVs,
  SmallVector<int64_t, 4> *irrelevantLBs,
  SmallVector<int64_t, 4> *irrelevantUBs
);

struct StoreLoadStoreLoadOps{
  AffineStoreOp storeOp;
  AffineStoreOp loopStoreOp;
  AffineLoadOp loopLoadOp;
  AffineLoadOp loadOp;
};

struct AccessInfo{
  streamhls::MemRefAccess access;
  FlatAffineRelation relation;
  FlatAffineRelation relevantRelation;
  presburger::IntegerRelation* timeIndexRelation;
  presburger::IntegerRelation* timeIndexRelevantRelation;
  SmallVector<Value, 4> irrelevantIVs;
  SmallVector<int64_t, 4> irrelevantLBs;
  SmallVector<int64_t, 4> irrelevantUBs;
  SmallVector<AffineExpr, 4> ifExprs;
  SmallVector<bool, 4> ifEqFlags;
  SmallVector<Value, 4> ifOperands;
  AccessInfo(Operation *loadOrStoreOp) : access(loadOrStoreOp){};
  LogicalResult getAllInfo(MLIRContext *context);
  presburger::IntegerRelation getTimeIndexRelation();
  presburger::IntegerRelation getTimeIndexRelevantRelation();
  presburger::PresburgerSpace getSpace();
  presburger::PresburgerSpace getRelevantSpace();
  // data for loop permutation and performance model
  uint64_t tripCount;
  uint64_t firstElementTime;
  uint64_t lastElementTime;
  SmallVector<AffineForOp, 6> loopBand;
  LogicalResult getInfoForPerformanceModel(MLIRContext *context);
};

LogicalResult reorderOps(
  SmallVector<AffineLoadOp, 4> loadOps,
  SmallVector<AffineStoreOp, 4> storeOps,
  StoreLoadStoreLoadOps &ops
);

AffineMap getAccessPattern(Operation *op);
} // namespace streamhls
} // namespace mlir

#endif // STREAMHLS_AFFINEMEMACCESS_H