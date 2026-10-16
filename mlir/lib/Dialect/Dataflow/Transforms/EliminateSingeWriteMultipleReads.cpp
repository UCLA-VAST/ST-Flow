#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "streamhls/Dialect/Dataflow/Transforms/Passes.h"
#include "streamhls/Dialect/Dataflow/Support/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;
using namespace affine;
using namespace streamhls;
using namespace dataflow;

#define DEBUG_TYPE "streamhls-eliminate-single-write-multiple-reads"


namespace {

// static bool isAllowedPureOp(Operation *op) {
//   // Allow affine.load/store, affine.apply, arith ops, casts, constants, etc.
//   if (isa<affine::AffineLoadOp, affine::AffineStoreOp, affine::AffineApplyOp>(op))
//     return true;
//   if (op->getDialect() && op->getDialect()->getNamespace() == "arith")
//     return true;
//   if (isa<arith::ConstantOp, arith::IndexCastOp, arith::TruncFOp, arith::ExtFOp>(op))
//     return true;

//   // Allow terminators in affine.for bodies.
//   if (op->hasTrait<OpTrait::IsTerminator>())
//     return true;

//   // Otherwise require it to be side-effect-free.
//   return MemoryEffectOpInterface::hasNoEffect(op);
// }

// Return the outermost affine.for in the (perfect) chain containing `op`.
// If not inside any affine.for, returns null.
static affine::AffineForOp findOutermostFor(Operation *op) {
  affine::AffineForOp cur = op->getParentOfType<affine::AffineForOp>();
  if (!cur) return nullptr;
  while (auto parent = cur->getParentOfType<affine::AffineForOp>())
    cur = parent;
  return cur;
}

// Check that `root` is a perfectly-nested band down to a leaf that contains `leafOp`,
// and that the band contains only allowed ops; also check that the only memref written
// by affine.store inside the band is `targetMemref`.
// Returns success if safe to erase the band.
static LogicalResult isSafeToEraseBand(affine::AffineForOp root,
                                      Value targetMemref) {
  if (!root) return failure();

  bool sawStore = false;
  bool writesOtherMemref = false;

  root.walk([&](Operation *op) {
    // if (!isAllowedPureOp(op)) {
    //   LLVM_DEBUG({
    //     llvm::dbgs() << "[DeadStoreElim] band not safe: has disallowed op: ";
    //     op->print(llvm::dbgs());
    //     llvm::dbgs() << "\n";
    //   });
    //   writesOtherMemref = true; // force failure via early stop logic below
    //   return WalkResult::interrupt();
    // }

    if (auto st = dyn_cast<affine::AffineStoreOp>(op)) {
      sawStore = true;
      if (st.getMemref() != targetMemref) {
        writesOtherMemref = true;
        LLVM_DEBUG({
          llvm::dbgs() << "[DeadStoreElim] band writes other memref -> not erasing loops\n";
        });
      }
    }
    return WalkResult::advance();
  });

  if (!sawStore) {
    LLVM_DEBUG(llvm::dbgs() << "[DeadStoreElim] band not safe: no stores\n");
    return failure();
  }
  if (writesOtherMemref) return failure();
  return success();
}

// Erase the whole perfectly-nested band (outermost loop) if it contains only stores to `targetMemref`.
static void eraseBand(PatternRewriter &rewriter, affine::AffineForOp root) {
  LLVM_DEBUG({
    llvm::dbgs() << "[DeadStoreElim] erasing band rooted at: ";
    root->print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });
  rewriter.eraseOp(root);
}

struct ElideAllocOnlyStored final : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp alloc,
                                PatternRewriter &rewriter) const override {
    Value buf = alloc.getResult();

    // Gather users.
    SmallVector<affine::AffineStoreOp, 8> stores;
    SmallVector<Operation*, 4> otherUsers;

    for (Operation *u : buf.getUsers()) {
      if (auto st = dyn_cast<affine::AffineStoreOp>(u)) {
        if (st.getMemref() == buf) stores.push_back(st);
        else otherUsers.push_back(u); // shouldn't happen, but keep conservative
        continue;
      }
      if (isa<memref::DeallocOp>(u)) continue; // ok
      // Any load means it's not dead.
      if (isa<affine::AffineLoadOp>(u)) {
        LLVM_DEBUG({
          llvm::dbgs() << "[DeadStoreElim] no match: alloc has affine.load user\n";
          llvm::dbgs() << "  alloc: "; alloc->print(llvm::dbgs()); llvm::dbgs() << "\n";
        });
        return failure();
      }
      otherUsers.push_back(u);
    }

    if (!otherUsers.empty()) {
      LLVM_DEBUG({
        llvm::dbgs() << "[DeadStoreElim] no match: alloc has non-store/dealloc users: "
                     << otherUsers.size() << "\n";
        llvm::dbgs() << "  alloc: "; alloc->print(llvm::dbgs()); llvm::dbgs() << "\n";
      });
      return failure();
    }

    if (stores.empty()) {
      // Truly unused alloc. Let other DCE handle it, or remove here.
      LLVM_DEBUG({
        llvm::dbgs() << "[DeadStoreElim] removing unused alloc (no stores)\n";
        llvm::dbgs() << "  alloc: "; alloc->print(llvm::dbgs()); llvm::dbgs() << "\n";
      });
      rewriter.eraseOp(alloc);
      return success();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "[DeadStoreElim] candidate: alloc has only stores, store count="
                   << stores.size() << "\n";
      llvm::dbgs() << "  alloc: "; alloc->print(llvm::dbgs()); llvm::dbgs() << "\n";
    });

    // Try to erase the loop band(s) that perform the stores.
    // Simplification: require all stores to be under the same outermost affine.for root.
    affine::AffineForOp root = findOutermostFor(stores.front());
    if (!root) {
      LLVM_DEBUG(llvm::dbgs() << "[DeadStoreElim] cannot erase loops: store not in affine.for\n");
    } else {
      for (auto st : stores) {
        if (findOutermostFor(st) != root) {
          LLVM_DEBUG(llvm::dbgs() << "[DeadStoreElim] cannot erase loops: stores in multiple bands\n");
          root = nullptr;
          break;
        }
      }

      if (root && succeeded(isSafeToEraseBand(root, buf))) {
        // If we erase the band, it will erase the stores (and maybe deallocs inside).
        eraseBand(rewriter, root);
      } else {
        LLVM_DEBUG(llvm::dbgs() << "[DeadStoreElim] not erasing loops (safety check failed)\n");
        // If we didn't erase loops, we must erase stores individually before alloc removal.
        for (auto st : stores) {
          LLVM_DEBUG({
            llvm::dbgs() << "[DeadStoreElim] erasing store: ";
            st->print(llvm::dbgs());
            llvm::dbgs() << "\n";
          });
          rewriter.eraseOp(st);
        }
      }
    }

    // Remove dealloc users (if any) and then remove the alloc.
    SmallVector<memref::DeallocOp, 4> deallocs;
    for (Operation *u : llvm::make_early_inc_range(buf.getUsers()))
      if (auto d = dyn_cast<memref::DeallocOp>(u))
        deallocs.push_back(d);

    for (auto d : deallocs) {
      LLVM_DEBUG({
        llvm::dbgs() << "[DeadStoreElim] erasing dealloc: ";
        d->print(llvm::dbgs());
        llvm::dbgs() << "\n";
      });
      rewriter.eraseOp(d);
    }

    LLVM_DEBUG({
      llvm::dbgs() << "[DeadStoreElim] erasing alloc: ";
      alloc->print(llvm::dbgs());
      llvm::dbgs() << "\n";
    });
    rewriter.eraseOp(alloc);
    return success();
  }
};

} // namespace
//===----------------------------------------------------------------------===//
// Data-movement elimination patterns (restrictive but robust)
// - Eliminates full copies into alloc() by forwarding loads back to the source.
// - Eliminates transpose/permutation buffers by rewriting dst loads -> src loads
//   with permuted indices, then DCE removes the buffer + copy loops.
//
// Assumptions (kept intentionally simple):
// - Copy materialization is an affine perfectly-nested loop band whose innermost
//   body contains exactly ONE AffineStoreOp.
// - The stored value is an AffineLoadOp.
// - Both load/store use identity affine maps (i.e., map is identity and indices
//   are explicit operands).
// - The store memref is a memref.alloc() result (the temporary buffer).
// - The temp buffer has NO other stores (besides the copy store we match).
//
// Notes:
// - This file intentionally does NOT handle array_of_streams.
// - This file avoids building memref.copy; it directly forwards loads and relies
//   on DCE/canonicalize to erase dead allocs/loops.
//
// Plug-in:
//   populateElideDataMovementPatterns(patterns);
//   (and then applyPatternsAndFoldGreedily)
//
//===----------------------------------------------------------------------===//

namespace {
void debugOp(Operation *op) {
  LLVM_DEBUG({
    llvm::dbgs() << "  op: ";
    op->print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });
}

/// Common utils
///----------------------------------------------------------------------------

static bool isIdentityMap(AffineMap m) {
  if (!m) return false;
  return m.isIdentity();
}

/// Return true if `v` is (exactly) an induction var of some affine.for.
static bool isAffineIV(Value v) { return v && v.isa<BlockArgument>(); }

/// Collect a perfectly-nested band starting at `root` (inclusive).
/// Returns empty if `root` isn't an AffineForOp.
static SmallVector<affine::AffineForOp, 8>
getPerfectlyNestedBand(affine::AffineForOp root) {
  SmallVector<affine::AffineForOp, 8> band;
  if (!root) return band;

  affine::AffineForOp cur = root;
  while (cur) {
    band.push_back(cur);

    // Must have exactly one non-terminator op that is an AffineForOp,
    // otherwise stop.
    Operation *single = nullptr;
    for (Operation &op : *cur.getBody()) {
      if (op.hasTrait<OpTrait::IsTerminator>()) continue;
      if (!single) single = &op;
      else return band; // not perfectly nested beyond this point
    }

    auto next = single ? dyn_cast_or_null<affine::AffineForOp>(single) : nullptr;
    if (!next) break;
    cur = next;
  }
  return band;
}

/// Find the outermost affine.for containing `op` by walking parents while the
/// parent has exactly one non-terminator op (the child).
static affine::AffineForOp findOutermostPerfectNest(Operation *op) {
  auto curFor = op->getParentOfType<affine::AffineForOp>();
  if (!curFor) return nullptr;

  affine::AffineForOp outer = curFor;
  while (auto parent = outer->getParentOfType<affine::AffineForOp>()) {
    // check "perfect nesting": parent's body contains exactly one non-terminator op
    Operation *single = nullptr;
    for (Operation &p : *parent.getBody()) {
      if (p.hasTrait<OpTrait::IsTerminator>()) continue;
      if (!single) single = &p;
      else { parent = nullptr; break; }
    }
    if (!parent) break;
    if (single != outer.getOperation()) break;
    outer = parent;
  }
  return outer;
}

/// Check innermost body has exactly one AffineStoreOp (and nothing else besides terminator).
static affine::AffineStoreOp getSingleStoreInInnermost(affine::AffineForOp innermost) {
  if (!innermost) return nullptr;

  Operation *single = nullptr;
  for (Operation &op : *innermost.getBody()) {
    if (op.hasTrait<OpTrait::IsTerminator>()) continue;
    if (!single) single = &op;
    else return nullptr;
  }
  return single ? dyn_cast<affine::AffineStoreOp>(single) : nullptr;
}

static bool isPurePermMap(AffineMap map) {
  if (!map) return false;

  llvm::SmallDenseSet<unsigned, 16> seen;

  // Each result must be exactly a dim or symbol expr; no arithmetic.
  for (AffineExpr e : map.getResults()) {
    unsigned key = 0;
    if (auto d = dyn_cast<AffineDimExpr>(e)) {
      key = d.getPosition(); // dims: 0..numDims-1
    } else if (auto s = dyn_cast<AffineSymbolExpr>(e)) {
      key = map.getNumDims() + s.getPosition(); // symbols: numDims+...
    } else {
      return false;
    }
    if (!seen.insert(key).second) return false; // must be 1-1
  }

  return true;
}

/// A matched copy/permutation materialization:
///   dst[ storeIdxs ] = src[ loadIdxs ]
struct CopyLike {
  affine::AffineForOp root;
  affine::AffineStoreOp store;
  affine::AffineLoadOp load;
  Value dst; // memref (usually alloc)
  Value src; // memref
  SmallVector<unsigned, 8> invPerm; // maps srcDim -> dstDim (so srcIdx[d] = dstIdx[invPerm[d]])
};

/// Try match:
///   affine.for ... { affine.store (affine.load %src[...]) , %dst[...] }
/// with identity maps, and where indices are a permutation of each other.
///
/// Computes invPerm such that for any read of dst with indices dstIdx:
///   srcIdx[d] = dstIdx[invPerm[d]]
///
/// Returns failure if not matched.
static LogicalResult matchCopyOrPermFromStore(affine::AffineStoreOp st,
                                              CopyLike &out) {
  if (!st) {
    LLVM_DEBUG(llvm::dbgs() << "[match] fail: no store\n");
    return failure();
  }

  auto ld = st.getValueToStore().getDefiningOp<affine::AffineLoadOp>();
  if (!ld) {
    LLVM_DEBUG(llvm::dbgs() << "[match] fail: store value not affine.load\n");
    return failure();
  }

  if (!isIdentityMap(st.getAffineMap()) ||
      !isIdentityMap(ld.getAffineMap())) {
    LLVM_DEBUG(llvm::dbgs() << "[match] fail: non-identity affine map\n");
    return failure();
  }

  // Minimal safety: reject non-trivial index expressions (floordiv/mod/etc).
  // Your current permutation logic uses getMapOperands(), which is unsafe otherwise.
  if (!isPurePermMap(st.getAffineMap()) || !isPurePermMap(ld.getAffineMap())) {
    LLVM_DEBUG(llvm::dbgs() << "[match] fail: non-pure permutation map\n");
    return failure();
  }

  Value dst = st.getMemref();
  Value src = ld.getMemref();

  if (!dst.getDefiningOp<memref::AllocOp>()) {
    LLVM_DEBUG(llvm::dbgs() << "[match] fail: dst is not memref.alloc\n");
    return failure();
  }

  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  auto srcTy = dyn_cast<MemRefType>(src.getType());
  if (!dstTy || !srcTy || dstTy.getRank() != srcTy.getRank()) {
    LLVM_DEBUG(llvm::dbgs() << "[match] fail: rank/type mismatch\n");
    return failure();
  }

  unsigned rank = dstTy.getRank();
  auto stIdxs = st.getMapOperands();
  auto ldIdxs = ld.getMapOperands();

  if (stIdxs.size() != rank || ldIdxs.size() != rank) {
    LLVM_DEBUG(llvm::dbgs() << "[match] fail: index count mismatch\n");
    return failure();
  }

  for (Value v : stIdxs)
    if (!isAffineIV(v)) {
      LLVM_DEBUG(llvm::dbgs() << "[match] fail: store index not IV\n");
      return failure();
    }

  for (Value v : ldIdxs)
    if (!isAffineIV(v)) {
      LLVM_DEBUG(llvm::dbgs() << "[match] fail: load index not IV\n");
      return failure();
    }

  SmallVector<int, 8> dstToSrc(rank, -1);
  for (unsigned d = 0; d < rank; ++d) {
    for (unsigned s = 0; s < rank; ++s) {
      if (stIdxs[d] == ldIdxs[s]) {
        dstToSrc[d] = s;
        break;
      }
    }
    if (dstToSrc[d] < 0) {
      LLVM_DEBUG(llvm::dbgs() << "[match] fail: index not permutation\n");
      return failure();
    }
  }

  out.invPerm.assign(rank, 0);
  for (unsigned d = 0; d < rank; ++d)
    out.invPerm[dstToSrc[d]] = d;

  out.store = st;
  out.load  = ld;
  out.dst   = dst;
  out.src   = src;
  out.root  = findOutermostPerfectNest(st.getOperation());

  if (!out.root) {
    LLVM_DEBUG(llvm::dbgs() << "[match] fail: no perfect nest root\n");
    return failure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "[match] SUCCESS: copy/permutation detected\n";
    llvm::dbgs() << "  src: "; src.print(llvm::dbgs()); llvm::dbgs() << "\n";
    llvm::dbgs() << "  dst: "; dst.print(llvm::dbgs()); llvm::dbgs() << "\n";
    llvm::dbgs() << "  invPerm: ";
    for (auto p : out.invPerm) llvm::dbgs() << p << " ";
    llvm::dbgs() << "\n";
  });

  return success();
}


/// Return true if `memref` has any store other than `exceptStore`.
static bool hasOtherStores(Value memref, affine::AffineStoreOp exceptStore) {
  for (OpOperand &use : memref.getUses()) {
    auto st = dyn_cast<affine::AffineStoreOp>(use.getOwner());
    if (!st) continue;
    if (st != exceptStore) return true;
  }
  return false;
}

/// Rewrite all affine.load from `dst` into affine.load from `src` with permuted indices.
/// Only handles identity-map loads.
static LogicalResult rewriteDstLoadsToSrc(PatternRewriter &rewriter,
                                         Value dst,
                                         Value src,
                                         ArrayRef<unsigned> invPerm) {
  LLVM_DEBUG(llvm::dbgs() << "[rewrite] rewriting loads of dst -> src\n");

  SmallVector<affine::AffineLoadOp, 16> loads;
  for (OpOperand &use : dst.getUses())
    if (auto ld = dyn_cast<affine::AffineLoadOp>(use.getOwner()))
      loads.push_back(ld);

  LLVM_DEBUG(llvm::dbgs() << "  loads found: " << loads.size() << "\n");

  for (auto ld : loads) {
    // *** MINIMAL SAFETY FIX ***
    // Only rewrite dst loads that are identity-map loads, since we remap by operands.
    if (!isIdentityMap(ld.getAffineMap())) {
      LLVM_DEBUG(llvm::dbgs() << "[rewrite] fail: dst load has non-identity map\n");
      return failure();
    }
    rewriter.setInsertionPoint(ld);
    auto dstIdxs = ld.getMapOperands();

    SmallVector<Value, 8> srcIdxs(invPerm.size());
    for (unsigned i = 0; i < invPerm.size(); ++i)
      srcIdxs[i] = dstIdxs[invPerm[i]];

    auto newLd = rewriter.create<affine::AffineLoadOp>(
        ld.getLoc(), src,
        rewriter.getMultiDimIdentityMap(invPerm.size()), srcIdxs);

    LLVM_DEBUG({
      llvm::dbgs() << "  replaced load:\n";
      debugOp(ld);
    });

    rewriter.replaceOp(ld, newLd.getResult());
  }

  return success();
}


/// Try erase the whole copy band if it became dead (no remaining uses of dst stores).
static void tryEraseDeadCopyBand(PatternRewriter &rewriter, CopyLike &info) {
  // After rewriting loads, the store might be dead. If the dst (alloc) has no uses, erase.
  if (!info.dst.use_empty()) return;

  // Erase the entire root loop; DCE/canonicalize can handle the rest,
  // but we can be aggressive here.
  rewriter.eraseOp(info.root);

  // If alloc is still around and unused, erase it.
  if (auto a = info.dst.getDefiningOp<memref::AllocOp>()) {
    if (a->use_empty())
      rewriter.eraseOp(a);
  }
}

/// Patterns
///----------------------------------------------------------------------------

/// (1) Elide copy/permutation buffer materialization:
///     alloc_dst + perfect nest store(load(src))  ==> rewrite loads of dst to loads of src
///     using the permutation derived from the store indices.
struct ElideAffineCopyOrPermIntoAlloc final : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp alloc,
                              PatternRewriter &rewriter) const override {
  LLVM_DEBUG({
    llvm::dbgs() << "\n[pattern] Try ElideAffineCopyOrPermIntoAlloc\n";
    debugOp(alloc);
  });

  Value dst = alloc.getResult();

  affine::AffineStoreOp st = nullptr;
  for (OpOperand &use : dst.getUses()) {
    st = dyn_cast<affine::AffineStoreOp>(use.getOwner());
    if (st) break;
  }

  if (!st) {
    LLVM_DEBUG(llvm::dbgs() << "  fail: no store to alloc\n");
    return failure();
  }

  CopyLike info;
  if (failed(matchCopyOrPermFromStore(st, info))) {
    LLVM_DEBUG(llvm::dbgs() << "  fail: store did not match copy/permutation\n");
    return failure();
  }

  if (hasOtherStores(dst, info.store)) {
    LLVM_DEBUG(llvm::dbgs() << "  fail: multiple stores to alloc\n");
    return failure();
  }

  if (failed(rewriteDstLoadsToSrc(rewriter, info.dst, info.src, info.invPerm))) {
    LLVM_DEBUG(llvm::dbgs() << "  fail: rewrite failed\n");
    return failure();
  }

  LLVM_DEBUG(llvm::dbgs() << "  rewrite SUCCESS\n");

  tryEraseDeadCopyBand(rewriter, info);

  LLVM_DEBUG(llvm::dbgs() << "  erase attempt done\n");
  return success();
}
};

/// (2) Elide trivial alloc->alloc identity copies (common after bufferization):
/// Matches:
///   allocB
///   perfect nest: store (load allocA) -> allocB with identity permutation
/// Then rewrites loads of allocB -> loads of allocA and removes the copy.
///
/// NOTE: This is already covered by (1), but keeping it explicit is useful if you
/// later relax matchCopyOrPermFromStore and want a “fast path”.
struct ElideAffineIdentityCopyIntoAlloc final : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp alloc,
                                PatternRewriter &rewriter) const override {
    Value dst = alloc.getResult();

    affine::AffineStoreOp st = nullptr;
    for (OpOperand &use : dst.getUses()) {
      st = dyn_cast<affine::AffineStoreOp>(use.getOwner());
      if (st && st.getMemref() == dst) break;
      st = nullptr;
    }
    if (!st) return failure();

    CopyLike info;
    if (failed(matchCopyOrPermFromStore(st, info)))
      return failure();

    // Require identity permutation.
    for (unsigned i = 0; i < info.invPerm.size(); ++i)
      if (info.invPerm[i] != i) return failure();

    if (hasOtherStores(dst, info.store)) return failure();

    if (failed(rewriteDstLoadsToSrc(rewriter, info.dst, info.src, info.invPerm)))
      return failure();

    for (OpOperand &use : dst.getUses()) {
      Operation *u = use.getOwner();
      if (isa<affine::AffineLoadOp>(u)) continue;
      if (isa<affine::AffineStoreOp>(u)) continue;
      return failure();
    }

    tryEraseDeadCopyBand(rewriter, info);
    return success();
  }
};

/// Registration
///----------------------------------------------------------------------------

} // namespace

/// Call this from your pass:
///   RewritePatternSet patterns(&getContext());
///   populateElideDataMovementPatterns(patterns);
///   (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
static void populateElideDataMovementPatterns(RewritePatternSet &patterns) {
  patterns.add<ElideAffineCopyOrPermIntoAlloc>(patterns.getContext());
  patterns.add<ElideAffineIdentityCopyIntoAlloc>(patterns.getContext());
}


namespace {

static AffineForOp getOutermostFor(Operation *op) {
  AffineForOp outer;
  for (Operation *cur = op; cur; cur = cur->getParentOp())
    if (auto f = dyn_cast<AffineForOp>(cur))
      outer = f;
  return outer;
}

/// Check if an alloc is in a memory module context.
/// Memory modules are functions that deal with off-chip buffers (have buffer.info
/// attributes with type="off-chip" or "weights" on their arguments).
static bool isInMemoryModuleContext(memref::AllocOp allocOp) {
  // Find the containing function
  Operation *parent = allocOp->getParentOp();
  func::FuncOp func = nullptr;
  while (parent) {
    if (auto f = dyn_cast<func::FuncOp>(parent)) {
      func = f;
      break;
    }
    parent = parent->getParentOp();
  }
  
  if (!func) {
    LLVM_DEBUG(
    	llvm::dbgs() << "[eliminate-sw-mr]     No containing function found\n";
    );
    return false;
  }
  
  LLVM_DEBUG(
  	llvm::dbgs() << "[eliminate-sw-mr]     Checking function: @" << func.getSymName() << "\n";
  );
  
  // Check if any function argument has buffer.info with type="off-chip" or "weights"
  for (unsigned i = 0; i < func.getNumArguments(); ++i) {
    if (auto attr = func.getArgAttr(i, "buffer.info")) {
      if (auto dict = dyn_cast<DictionaryAttr>(attr)) {
        if (auto typeAttr = dict.getAs<StringAttr>("type")) {
          StringRef type = typeAttr.getValue();
          LLVM_DEBUG(
          	llvm::dbgs() << "[eliminate-sw-mr]       Arg " << i << " buffer.info.type = \"" << type << "\"\n";
          );
          if (type.equals_insensitive("off-chip") || 
              type.equals_insensitive("offchip") ||
              type.equals_insensitive("weights")) {
            LLVM_DEBUG(
            	llvm::dbgs() << "[eliminate-sw-mr]       -> MEMORY MODULE (arg " << i << " is off-chip/weights)\n";
            );
            return true;
          }
        }
      }
    }
  }
  
  // Also check if the alloc itself has buffer.info with off-chip type
  if (auto attr = allocOp->getAttrOfType<DictionaryAttr>("buffer.info")) {
    if (auto typeAttr = attr.getAs<StringAttr>("type")) {
      StringRef type = typeAttr.getValue();
      LLVM_DEBUG(
      	llvm::dbgs() << "[eliminate-sw-mr]       Alloc buffer.info.type = \"" << type << "\"\n";
      );
      if (type.equals_insensitive("off-chip") || 
          type.equals_insensitive("offchip") ||
          type.equals_insensitive("weights")) {
        LLVM_DEBUG(
        	llvm::dbgs() << "[eliminate-sw-mr]       -> MEMORY MODULE (alloc is off-chip/weights)\n";
        );
        return true;
      }
    }
  }
  
  LLVM_DEBUG(
  	llvm::dbgs() << "[eliminate-sw-mr]       -> COMPUTE MODULE (no off-chip/weights found)\n";
  );
  return false;
}

static memref::AllocOp cloneAllocLike(memref::AllocOp oldAlloc,
                                      PatternRewriter &rewriter) {
  auto newAlloc = rewriter.create<memref::AllocOp>(oldAlloc.getLoc(),
                                                  oldAlloc.getType());
  if (auto a = oldAlloc->getAttrOfType<IntegerAttr>("alignment"))
    newAlloc->setAttr("alignment", a);
  return newAlloc;
}

/// Match a memref.alloc that has exactly one affine.store into it and multiple
/// affine.load from it in different consumer loop-nests. Rewrite by duplicating
/// the alloc and cloning the producer loop-nest per consumer loop-nest.
struct EliminateSingleWriteMultipleReadsPattern
    : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    Value buf = allocOp.getResult();

    // Collect store/load users.
    SmallVector<AffineStoreOp, 4> stores;
    SmallVector<AffineLoadOp, 16> loads;
    for (Operation *u : buf.getUsers()) {
      if (auto st = dyn_cast<AffineStoreOp>(u))
        stores.push_back(st);
      else if (auto ld = dyn_cast<AffineLoadOp>(u))
        loads.push_back(ld);
      else
        return failure(); // under your assumptions, ignore other ops by failing
    }
    LLVM_DEBUG(
    	llvm::dbgs() << "[eliminate-sw-mr] alloc: " << allocOp << "\n";
    );
    LLVM_DEBUG(
    	llvm::dbgs() << "[eliminate-sw-mr]   stores: " << stores.size()
                 << ", loads: " << loads.size() << "\n\n";
    );
    
    // Exactly one writer and more than one reader.
    if (stores.size() != 1 || loads.size() <= 1)
      return failure();

    AffineStoreOp storeOp = stores.front();
    AffineForOp prodOuter = getOutermostFor(storeOp);
    if (!prodOuter)
      return failure();

    // Group loads by consumer outermost loop-nest (outermost affine.for).
    llvm::DenseMap<Operation *, SmallVector<AffineLoadOp, 8>> consumerToLoads;
    for (AffineLoadOp ld : loads) {
      AffineForOp consOuter = getOutermostFor(ld);
      if (!consOuter)
        return failure();
      consumerToLoads[consOuter.getOperation()].push_back(ld);
    }

    // If all loads belong to the same consumer loop-nest, no duplication needed.
    if (consumerToLoads.size() <= 1)
      return failure();

    LLVM_DEBUG(llvm::dbgs() << "[eliminate-sw-mr] duplicating alloc: " << allocOp
                            << "\n");

    // For each consumer loop-nest:
    //  - create new alloc
    //  - clone producer loop-nest mapping old buf -> new buf
    //  - rewrite uses of old buf inside that consumer loop-nest to new buf
    for (auto &kv : consumerToLoads) {
      Operation *consumerOuter = kv.first;

      // Insert (new alloc + cloned producer nest) right before the consumer nest.
      rewriter.setInsertionPoint(consumerOuter);

      memref::AllocOp newAlloc = cloneAllocLike(allocOp, rewriter);
      Value newBuf = newAlloc.getResult();

      IRMapping map;
      map.map(buf, newBuf);
      rewriter.clone(*prodOuter.getOperation(), map);

      // Rewrite uses inside this consumer nest only.
      buf.replaceUsesWithIf(newBuf, [&](OpOperand &use) {
        Operation *user = use.getOwner();
        return (consumerOuter->isProperAncestor(user) || user == consumerOuter);
      });
    }

    // If the original buffer is dead, remove the original producer nest + alloc.
    if (buf.use_empty()) {
      LLVM_DEBUG(llvm::dbgs() << "[eliminate-sw-mr] erasing original producer + alloc\n");
      rewriter.eraseOp(prodOuter);
      rewriter.eraseOp(allocOp);
    }

    return success();
  }
};

struct EliminateSingleWriteMultipleReadsInPlacePattern
    : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    Value buf = allocOp.getResult();

    // Collect store/load users.
    SmallVector<AffineStoreOp, 4> stores;
    SmallVector<AffineLoadOp, 16> loads;
    for (Operation *u : buf.getUsers()) {
      if (auto st = dyn_cast<AffineStoreOp>(u))
        stores.push_back(st);
      else if (auto ld = dyn_cast<AffineLoadOp>(u))
        loads.push_back(ld);
      else
        return failure(); // keep your "only loads/stores" assumption
    }

    // Exactly one writer and >1 reader.
    if (stores.size() != 1 || loads.size() <= 1)
      return failure();

    AffineStoreOp storeOp = stores.front();
    AffineForOp prodOuter = getOutermostFor(storeOp);
    if (!prodOuter)
      return failure();

    // Group loads by consumer outermost loop-nest.
    llvm::DenseMap<Operation *, SmallVector<AffineLoadOp, 8>> consumerToLoads;
    consumerToLoads.reserve(loads.size());
    for (AffineLoadOp ld : loads) {
      AffineForOp consOuter = getOutermostFor(ld);
      if (!consOuter)
        return failure();

      // If a load is in the same outermost nest as the producer, this "in-place"
      // duplication becomes ambiguous (you'd need intra-nest selective rewrites).
      if (consOuter == prodOuter)
        return failure();

      consumerToLoads[consOuter.getOperation()].push_back(ld);
    }

    // If all loads are in one consumer nest, nothing to do.
    if (consumerToLoads.size() <= 1)
      return failure();

    // Deterministic order (avoid DenseMap nondeterminism).
    SmallVector<Operation *, 8> consumers;
    consumers.reserve(consumerToLoads.size());
    for (auto &kv : consumerToLoads)
      consumers.push_back(kv.first);

    llvm::sort(consumers, [](Operation *a, Operation *b) {
      // Prefer earlier in the same block; otherwise fall back to stable pointer order.
      if (a->getBlock() == b->getBlock())
        return a->isBeforeInBlock(b);
      return a < b;
    });

    // Choose one consumer to keep using the original buffer (first in order),
    // and create one new buffer per remaining consumer.
    Operation *keepConsumer = consumers.front();

    // Create new allocs right after the old alloc (dominates all loops).
    rewriter.setInsertionPointAfter(allocOp);

    DenseMap<Operation *, Value> consumerToBuf;
    consumerToBuf[keepConsumer] = buf;

    SmallVector<Value, 8> extraBufs;
    extraBufs.reserve(consumers.size() - 1);

    for (Operation *c : llvm::drop_begin(consumers, 1)) {
      memref::AllocOp newAlloc = cloneAllocLike(allocOp, rewriter);
      Value newBuf = newAlloc.getResult();
      consumerToBuf[c] = newBuf;
      extraBufs.push_back(newBuf);
    }

    // Inside the original producer nest, duplicate ONLY the store op to fill
    // the duplicated buffers. Do NOT clone the whole loop nest.
    //
    // Insert immediately after the original affine.store.
    rewriter.setInsertionPointAfter(storeOp);
    for (Value newBuf : extraBufs) {
      IRMapping map;
      map.map(buf, newBuf);
      (void)rewriter.clone(*storeOp.getOperation(), map);
    }

    // Rewrite uses of the old buffer inside each consumer nest to its assigned buffer.
    // (Only rewrite within that consumer loop-nest.)
    for (Operation *c : consumers) {
      Value target = consumerToBuf.lookup(c);
      if (target == buf)
        continue;

      buf.replaceUsesWithIf(target, [&](OpOperand &use) {
        Operation *user = use.getOwner();
        // Restrict to this consumer nest.
        return (c->isProperAncestor(user) || user == c);
      });
    }

    // Optional cleanup: if the original buffer ended up unused (rare, because
    // producer still writes it), erase it + its store. In most cases, keep it.
    // (Do nothing here to stay conservative.)

    return success();
  }
};


} // namespace

namespace {
// Handles the pattern:
//
//   alloc
//   (producer nest #0)  store to alloc
//   (producer nest #1)  store to alloc
//   ... (possibly more producer nests) ...
//   (consumer nest #0)  load alloc ...
//   (consumer nest #1)  load alloc ...
//   ... (multiple consumer nests) ...
//
// Rewrite:
//   - Create one extra buffer per extra consumer nest.
//   - Clone *each* producer store op for each extra buffer (right after the
//     corresponding original store op).
//   - Rewrite loads inside each consumer nest to use its assigned buffer.
//
// We explicitly DO NOT touch cases like: store in loop1, load in loop2, store in loop2, load in loop3.
// (i.e., interleaved producer/consumer phases). We only match when all stores happen in one consecutive
// “producer phase”, followed by a “consumer phase”.
struct EliminateConsecutiveMultiWriteMultipleReadsInPlacePattern
    : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    Value buf = allocOp.getResult();

    LLVM_DEBUG(llvm::dbgs()
               << "\n[ECMWMR] Visiting alloc: " << allocOp << "\n");

    // Collect store/load users only.
    SmallVector<AffineStoreOp, 8> stores;
    SmallVector<AffineLoadOp, 16> loads;
    for (Operation *u : buf.getUsers()) {
      if (auto st = dyn_cast<AffineStoreOp>(u)) {
        stores.push_back(st);
      } else if (auto ld = dyn_cast<AffineLoadOp>(u)) {
        loads.push_back(ld);
      } else {
        LLVM_DEBUG(llvm::dbgs()
                   << "[ECMWMR]   Non-affine user found, bailing: "
                   << *u << "\n");
        return failure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "[ECMWMR]   #stores = " << stores.size()
                   << ", #loads = " << loads.size() << "\n";
    });

    if (stores.size() < 2 || loads.size() <= 1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[ECMWMR]   Not enough stores or loads, skipping\n");
      return failure();
    }

    // === Producer outer loops ===
    llvm::SmallDenseSet<Operation *, 8> producerOuterSet;
    SmallVector<Operation *, 8> producerOuters;

    for (AffineStoreOp st : stores) {
      AffineForOp outer = getOutermostFor(st);
      if (!outer) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[ECMWMR]   Store without outer loop, bailing\n");
        return failure();
      }
      if (producerOuterSet.insert(outer.getOperation()).second) {
        producerOuters.push_back(outer.getOperation());
        LLVM_DEBUG(llvm::dbgs()
                   << "[ECMWMR]   Producer outer: " << *outer << "\n");
      }
    }

    if (producerOuters.size() < 2) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[ECMWMR]   Only one producer nest, skipping\n");
      return failure();
    }

    // === Consumer outer loops ===
    llvm::DenseMap<Operation *, SmallVector<AffineLoadOp, 8>> consumerToLoads;
    llvm::SmallDenseSet<Operation *, 8> consumerOuterSet;
    SmallVector<Operation *, 8> consumerOuters;

    for (AffineLoadOp ld : loads) {
      AffineForOp outer = getOutermostFor(ld);
      if (!outer) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[ECMWMR]   Load without outer loop, bailing\n");
        return failure();
      }

      Operation *o = outer.getOperation();

      if (producerOuterSet.contains(o)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[ECMWMR]   Load inside producer nest, rejecting "
                      "interleaved store/load case\n");
        return failure();
      }

      consumerToLoads[o].push_back(ld);
      if (consumerOuterSet.insert(o).second) {
        consumerOuters.push_back(o);
        LLVM_DEBUG(llvm::dbgs()
                   << "[ECMWMR]   Consumer outer: " << *outer << "\n");
      }
    }

    if (consumerOuters.size() <= 1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[ECMWMR]   Single consumer nest, skipping\n");
      return failure();
    }

    auto sortByBlockOrder = [](SmallVectorImpl<Operation *> &ops) {
      llvm::sort(ops, [](Operation *a, Operation *b) {
        if (a->getBlock() == b->getBlock())
          return a->isBeforeInBlock(b);
        return a < b;
      });
    };

    sortByBlockOrder(producerOuters);
    sortByBlockOrder(consumerOuters);

    LLVM_DEBUG({
      llvm::dbgs() << "[ECMWMR]   Producer order:\n";
      for (Operation *p : producerOuters)
        llvm::dbgs() << "     - " << *p << "\n";

      llvm::dbgs() << "[ECMWMR]   Consumer order:\n";
      for (Operation *c : consumerOuters)
        llvm::dbgs() << "     - " << *c << "\n";
    });

    // Producers must be consecutive
    for (unsigned i = 0; i + 1 < producerOuters.size(); ++i) {
      if (producerOuters[i]->getNextNode() != producerOuters[i + 1]) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[ECMWMR]   Producer nests not consecutive, bailing\n");
        return failure();
      }
    }

    Operation *lastProducer = producerOuters.back();
    for (Operation *c : consumerOuters) {
      if (!lastProducer->isBeforeInBlock(c)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[ECMWMR]   Consumer appears before last producer, "
                      "rejecting interleaved case\n");
        return failure();
      }
    }

    LLVM_DEBUG(llvm::dbgs()
               << "[ECMWMR]   Matched consecutive producers followed by "
                  "multiple consumers\n");

    // === Buffer duplication ===
    Operation *keepConsumer = consumerOuters.front();
    rewriter.setInsertionPointAfter(allocOp);

    llvm::DenseMap<Operation *, Value> consumerToBuf;
    consumerToBuf[keepConsumer] = buf;

    SmallVector<Value, 8> extraBufs;
    for (Operation *c : llvm::drop_begin(consumerOuters, 1)) {
      memref::AllocOp newAlloc = cloneAllocLike(allocOp, rewriter);
      Value newBuf = newAlloc.getResult();
      consumerToBuf[c] = newBuf;
      extraBufs.push_back(newBuf);

      LLVM_DEBUG(llvm::dbgs()
                 << "[ECMWMR]   Created duplicate buffer " << newBuf
                 << " for consumer " << *c << "\n");
    }

    // === Clone each store for each new buffer ===
    for (AffineStoreOp st : stores) {
      rewriter.setInsertionPointAfter(st);
      for (Value newBuf : extraBufs) {
        IRMapping map;
        map.map(buf, newBuf);
        rewriter.clone(*st.getOperation(), map);

        LLVM_DEBUG(llvm::dbgs()
                   << "[ECMWMR]   Cloned store for buffer " << newBuf
                   << " after store " << st << "\n");
      }
    }

    // === Rewrite consumer loads ===
    for (Operation *c : consumerOuters) {
      Value target = consumerToBuf.lookup(c);
      if (target == buf)
        continue;

      LLVM_DEBUG(llvm::dbgs()
                 << "[ECMWMR]   Rewriting loads in consumer "
                 << *c << " to buffer " << target << "\n");

      buf.replaceUsesWithIf(target, [&](OpOperand &use) {
        Operation *user = use.getOwner();
        return (c->isProperAncestor(user) || user == c);
      });
    }

    LLVM_DEBUG(llvm::dbgs()
               << "[ECMWMR]   Rewrite successful\n");

    return success();
  }
};


}

/// Unified pattern that chooses between original (for memory modules) and 
/// in-place (for compute modules) patterns based on context.
struct EliminateSingleWriteMultipleReadsUnifiedPattern
    : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    Value buf = allocOp.getResult();

    LLVM_DEBUG(
    	llvm::dbgs() << "[eliminate-sw-mr] ========================================\n";
    );
    LLVM_DEBUG(
    	llvm::dbgs() << "[eliminate-sw-mr] Processing alloc: ";
    );
    LLVM_DEBUG(allocOp.print(llvm::dbgs()););
    LLVM_DEBUG(
    	llvm::dbgs() << "\n";
    );

    // Collect store/load users.
    SmallVector<AffineStoreOp, 4> stores;
    SmallVector<AffineLoadOp, 16> loads;
    for (Operation *u : buf.getUsers()) {
      if (auto st = dyn_cast<AffineStoreOp>(u))
        stores.push_back(st);
      else if (auto ld = dyn_cast<AffineLoadOp>(u))
        loads.push_back(ld);
      else {
        LLVM_DEBUG(
        	llvm::dbgs() << "[eliminate-sw-mr]   Skipping: has non-store/load users\n";
        );
        return failure();
      }
    }
    
    LLVM_DEBUG(
    	llvm::dbgs() << "[eliminate-sw-mr]   Found " << stores.size() << " store(s), " << loads.size() << " load(s)\n";
    );
    
    // Exactly one writer and more than one reader.
    if (stores.size() != 1 || loads.size() <= 1) {
      LLVM_DEBUG(
      	llvm::dbgs() << "[eliminate-sw-mr]   Skipping: stores=" << stores.size() << " (need 1), loads=" << loads.size() << " (need >1)\n";
      );
      return failure();
    }

    AffineStoreOp storeOp = stores.front();
    AffineForOp prodOuter = getOutermostFor(storeOp);
    if (!prodOuter) {
      LLVM_DEBUG(
      	llvm::dbgs() << "[eliminate-sw-mr]   Skipping: store not in affine.for loop\n";
      );
      return failure();
    }

    LLVM_DEBUG(
    	llvm::dbgs() << "[eliminate-sw-mr]   Producer loop: ";
    );
    LLVM_DEBUG(prodOuter.print(llvm::dbgs()););
    LLVM_DEBUG(
    	llvm::dbgs() << "\n";
    );

    // Group loads by consumer outermost loop-nest.
    llvm::DenseMap<Operation *, SmallVector<AffineLoadOp, 8>> consumerToLoads;
    for (AffineLoadOp ld : loads) {
      AffineForOp consOuter = getOutermostFor(ld);
      if (!consOuter) {
        LLVM_DEBUG(
        	llvm::dbgs() << "[eliminate-sw-mr]   Skipping: load not in affine.for loop\n";
        );
        return failure();
      }
      consumerToLoads[consOuter.getOperation()].push_back(ld);
    }

    LLVM_DEBUG(
    	llvm::dbgs() << "[eliminate-sw-mr]   Found " << consumerToLoads.size() << " distinct consumer loop nest(s)\n";
    );
    for (auto &kv : consumerToLoads) {
      LLVM_DEBUG(
      	llvm::dbgs() << "[eliminate-sw-mr]     Consumer loop: ";
      );
      LLVM_DEBUG(kv.first->print(llvm::dbgs()););
      LLVM_DEBUG(
      	llvm::dbgs() << " (" << kv.second.size() << " load(s))\n";
      );
    }

    // If all loads belong to the same consumer loop-nest, no duplication needed.
    if (consumerToLoads.size() <= 1) {
      LLVM_DEBUG(
      	llvm::dbgs() << "[eliminate-sw-mr]   Skipping: all loads in same consumer loop\n";
      );
      return failure();
    }

    // Check if this is a memory module context
    bool isMemoryModule = isInMemoryModuleContext(allocOp);
    LLVM_DEBUG(
    	llvm::dbgs() << "[eliminate-sw-mr]   Module type: " << (isMemoryModule ? "MEMORY" : "COMPUTE") << "\n";
    );
    
    if (isMemoryModule) {
      // Use original pattern: duplicate loops for memory modules
      LLVM_DEBUG(
      	llvm::dbgs() << "[eliminate-sw-mr]   Using PATTERN: ORIGINAL (duplicate loops for memory modules)\n";
      );
      
      // For each consumer loop-nest (excluding the producer loop itself):
      //  - create new alloc
      //  - clone producer loop-nest mapping old buf -> new buf
      //  - rewrite uses of old buf inside that consumer loop-nest to new buf
      int duplicateCount = 0;
      for (auto &kv : consumerToLoads) {
        Operation *consumerOuter = kv.first;
        
        // Skip if this consumer is the producer loop itself - those loads stay with original buffer
        if (consumerOuter == prodOuter.getOperation()) {
          LLVM_DEBUG(
          	llvm::dbgs() << "[eliminate-sw-mr]     Skipping producer loop itself (loads stay with original buffer)\n";
          );
          continue;
        }

        LLVM_DEBUG(
        	llvm::dbgs() << "[eliminate-sw-mr]     Duplicating producer loop for consumer loop: ";
        );
        LLVM_DEBUG(consumerOuter->print(llvm::dbgs()););
        LLVM_DEBUG(
        	llvm::dbgs() << "\n";
        );

        // Insert (new alloc + cloned producer nest) right before the consumer nest.
        rewriter.setInsertionPoint(consumerOuter);

        memref::AllocOp newAlloc = cloneAllocLike(allocOp, rewriter);
        Value newBuf = newAlloc.getResult();
        duplicateCount++;

        IRMapping map;
        map.map(buf, newBuf);
        rewriter.clone(*prodOuter.getOperation(), map);

        // Rewrite uses inside this consumer nest only.
        int replacedUses = 0;
        buf.replaceUsesWithIf(newBuf, [&](OpOperand &use) {
          Operation *user = use.getOwner();
          bool shouldReplace = (consumerOuter->isProperAncestor(user) || user == consumerOuter);
          if (shouldReplace) replacedUses++;
          return shouldReplace;
        });
        LLVM_DEBUG(
        	llvm::dbgs() << "[eliminate-sw-mr]       Replaced " << replacedUses << " use(s) in consumer loop\n";
        );
      }

      LLVM_DEBUG(
      	llvm::dbgs() << "[eliminate-sw-mr]   Created " << duplicateCount << " duplicate buffer(s)\n";
      );

      // If the original buffer is dead, remove the original producer nest + alloc.
      if (buf.use_empty()) {
        LLVM_DEBUG(
        	llvm::dbgs() << "[eliminate-sw-mr]   Erasing original producer loop and alloc (no longer used)\n";
        );
        rewriter.eraseOp(prodOuter);
        rewriter.eraseOp(allocOp);
      } else {
        LLVM_DEBUG(
        	llvm::dbgs() << "[eliminate-sw-mr]   Keeping original buffer (still in use)\n";
        );
      }
    } else {
      // Use in-place pattern: duplicate allocs and stores, but not loops
      LLVM_DEBUG(
      	llvm::dbgs() << "[eliminate-sw-mr]   Using PATTERN: IN-PLACE (duplicate allocs/stores, not loops)\n";
      );
      
      // Check if any load is in the same outermost nest as the producer
      bool hasLoadInProducerNest = false;
      for (AffineLoadOp ld : loads) {
        AffineForOp consOuter = getOutermostFor(ld);
        if (consOuter == prodOuter) {
          hasLoadInProducerNest = true;
          LLVM_DEBUG(
          	llvm::dbgs() << "[eliminate-sw-mr]   Found load(s) in producer loop nest\n";
          );
          break;
        }
      }
      
      // If a load is in the same outermost nest as the producer, we cannot use
      // the in-place pattern. Fall back to the original pattern (duplicate loops)
      // even for compute modules, as this is necessary for correctness.
      if (hasLoadInProducerNest) {
        LLVM_DEBUG(
        	llvm::dbgs() << "[eliminate-sw-mr]   Falling back to ORIGINAL pattern (loads in producer nest)\n";
        );
        
        // Use original pattern: duplicate loops (but not for the producer loop itself)
        int duplicateCount = 0;
        for (auto &kv : consumerToLoads) {
          Operation *consumerOuter = kv.first;
          
          // Skip if this consumer is the producer loop itself - those loads stay with original buffer
          if (consumerOuter == prodOuter.getOperation()) {
            LLVM_DEBUG(
            	llvm::dbgs() << "[eliminate-sw-mr]     Skipping producer loop itself (loads stay with original buffer)\n";
            );
            continue;
          }

          LLVM_DEBUG(
          	llvm::dbgs() << "[eliminate-sw-mr]     Duplicating producer loop for consumer loop: ";
          );
          LLVM_DEBUG(consumerOuter->print(llvm::dbgs()););
          LLVM_DEBUG(
          	llvm::dbgs() << "\n";
          );

          // Insert (new alloc + cloned producer nest) right before the consumer nest.
          rewriter.setInsertionPoint(consumerOuter);

          memref::AllocOp newAlloc = cloneAllocLike(allocOp, rewriter);
          Value newBuf = newAlloc.getResult();
          duplicateCount++;

          IRMapping map;
          map.map(buf, newBuf);
          rewriter.clone(*prodOuter.getOperation(), map);

          // Rewrite uses inside this consumer nest only.
          int replacedUses = 0;
          buf.replaceUsesWithIf(newBuf, [&](OpOperand &use) {
            Operation *user = use.getOwner();
            bool shouldReplace = (consumerOuter->isProperAncestor(user) || user == consumerOuter);
            if (shouldReplace) replacedUses++;
            return shouldReplace;
          });
          LLVM_DEBUG(
          	llvm::dbgs() << "[eliminate-sw-mr]       Replaced " << replacedUses << " use(s) in consumer loop\n";
          );
        }

        LLVM_DEBUG(
        	llvm::dbgs() << "[eliminate-sw-mr]   Created " << duplicateCount << " duplicate buffer(s)\n";
        );

        // If the original buffer is dead, remove the original producer nest + alloc.
        if (buf.use_empty()) {
          LLVM_DEBUG(
          	llvm::dbgs() << "[eliminate-sw-mr]   Erasing original producer loop and alloc (no longer used)\n";
          );
          rewriter.eraseOp(prodOuter);
          rewriter.eraseOp(allocOp);
        } else {
          LLVM_DEBUG(
          	llvm::dbgs() << "[eliminate-sw-mr]   Keeping original buffer (still in use)\n";
          );
        }
        
        LLVM_DEBUG(
        	llvm::dbgs() << "[eliminate-sw-mr] ========================================\n";
        );
        return success();
      }

      // Deterministic order (avoid DenseMap nondeterminism).
      SmallVector<Operation *, 8> consumers;
      consumers.reserve(consumerToLoads.size());
      for (auto &kv : consumerToLoads)
        consumers.push_back(kv.first);

      llvm::sort(consumers, [](Operation *a, Operation *b) {
        if (a->getBlock() == b->getBlock())
          return a->isBeforeInBlock(b);
        return a < b;
      });

      // Choose one consumer to keep using the original buffer (first in order),
      // and create one new buffer per remaining consumer.
      Operation *keepConsumer = consumers.front();
      LLVM_DEBUG(
      	llvm::dbgs() << "[eliminate-sw-mr]   Keeping original buffer for consumer: ";
      );
      LLVM_DEBUG(keepConsumer->print(llvm::dbgs()););
      LLVM_DEBUG(
      	llvm::dbgs() << "\n";
      );

      // Create new allocs right after the old alloc (dominates all loops).
      rewriter.setInsertionPointAfter(allocOp);

      DenseMap<Operation *, Value> consumerToBuf;
      consumerToBuf[keepConsumer] = buf;

      SmallVector<Value, 8> extraBufs;
      extraBufs.reserve(consumers.size() - 1);

      for (Operation *c : llvm::drop_begin(consumers, 1)) {
        memref::AllocOp newAlloc = cloneAllocLike(allocOp, rewriter);
        Value newBuf = newAlloc.getResult();
        consumerToBuf[c] = newBuf;
        extraBufs.push_back(newBuf);
        LLVM_DEBUG(
        	llvm::dbgs() << "[eliminate-sw-mr]     Created new buffer for consumer: ";
        );
        LLVM_DEBUG(c->print(llvm::dbgs()););
        LLVM_DEBUG(
        	llvm::dbgs() << "\n";
        );
      }

      // Inside the original producer nest, duplicate ONLY the store op to fill
      // the duplicated buffers. Do NOT clone the whole loop nest.
      LLVM_DEBUG(
      	llvm::dbgs() << "[eliminate-sw-mr]   Duplicating store op " << extraBufs.size() << " time(s) in producer loop\n";
      );
      rewriter.setInsertionPointAfter(storeOp);
      for (Value newBuf : extraBufs) {
        IRMapping map;
        map.map(buf, newBuf);
        (void)rewriter.clone(*storeOp.getOperation(), map);
      }

      // Rewrite uses of the old buffer inside each consumer nest to its assigned buffer.
      for (Operation *c : consumers) {
        Value target = consumerToBuf.lookup(c);
        if (target == buf)
          continue;

        int replacedUses = 0;
        buf.replaceUsesWithIf(target, [&](OpOperand &use) {
          Operation *user = use.getOwner();
          bool shouldReplace = (c->isProperAncestor(user) || user == c);
          if (shouldReplace) replacedUses++;
          return shouldReplace;
        });
        LLVM_DEBUG(
        	llvm::dbgs() << "[eliminate-sw-mr]     Replaced " << replacedUses << " use(s) in consumer: ";
        );
        LLVM_DEBUG(c->print(llvm::dbgs()););
        LLVM_DEBUG(
        	llvm::dbgs() << "\n";
        );
      }
      
      LLVM_DEBUG(
      	llvm::dbgs() << "[eliminate-sw-mr]   Created " << extraBufs.size() << " duplicate buffer(s)\n";
      );
    }

    LLVM_DEBUG(
    	llvm::dbgs() << "[eliminate-sw-mr] ========================================\n";
    );
    return success();
  }
};

//===----------------------------------------------------------------------===//
// DeadAllocCopyElim: remove undef→undef copy loops from lowering
//
// Matches a common affine-lowering artifact where an uninitialized temporary
// buffer is copied into another uninitialized buffer:
//
//   %src = memref.alloc() : memref<...>
//   %dst = memref.alloc() : memref<...>
//   affine.for ... {
//     %v = affine.load %src[...] : memref<...>   // undefined
//     affine.store %v, %dst[...] : memref<...>  // still undefined
//   }
//
// If both %src and %dst are memref.alloc values that have not been written to
// before the copy loop, then the loop copies undefined values into an undefined
// destination. Removing the loop preserves semantics: %dst remains undefined
// for the same index set as before.
//
// The pass erases the pure copy loop and erases %src if it becomes dead.
// No assumption is made about later overwrites of %dst; partial overwrites
// are allowed, since copying undefined data into an undefined buffer is
// semantically useless regardless of subsequent writes.
//===----------------------------------------------------------------------===//
namespace {
static bool matchSingleCopyBody(AffineForOp loop,
                               Value srcMemref,
                               AffineLoadOp &outLoad,
                               AffineStoreOp &outStore) {
  outLoad = nullptr;
  outStore = nullptr;

  for (Operation &op : *loop.getBody()) {
    if (isa<AffineYieldOp>(op))
      continue;

    if (auto ld = dyn_cast<AffineLoadOp>(op)) {
      if (ld.getMemRef() != srcMemref) {
        LLVM_DEBUG(llvm::dbgs() << "[elim] load reads wrong src\n");
        return false;
      }
      if (outLoad)
        return false;
      outLoad = ld;
      continue;
    }

    if (auto st = dyn_cast<AffineStoreOp>(op)) {
      if (outStore)
        return false;
      outStore = st;
      continue;
    }

    LLVM_DEBUG(llvm::dbgs()
               << "[elim] non-copy op in loop body: " << op << "\n");
    return false;
  }

  if (!outLoad || !outStore) {
    LLVM_DEBUG(llvm::dbgs()
               << "[elim] missing load or store in copy loop\n");
    return false;
  }

  if (outStore.getValueToStore() != outLoad.getResult()) {
    LLVM_DEBUG(llvm::dbgs()
               << "[elim] store does not store loaded value\n");
    return false;
  }

  if (outLoad.getAffineMap() != outStore.getAffineMap() ||
      outLoad.getMapOperands() != outStore.getMapOperands()) {
    LLVM_DEBUG(llvm::dbgs()
               << "[elim] load/store index mismatch\n");
    return false;
  }

  return true;
}
static bool hasWriteToMemrefInRange(Operation *start,
                                   Operation *end,
                                   Value memref) {
  Block *blk = start->getBlock();
  if (blk != end->getBlock()) {
    LLVM_DEBUG(llvm::dbgs()
               << "[elim] start/end not in same block → conservative write\n");
    return true;
  }

  for (Operation *op = start->getNextNode(); op && op != end;
       op = op->getNextNode()) {
    if (auto st = dyn_cast<AffineStoreOp>(op)) {
      if (st.getMemRef() == memref)
        return true;
    } else if (auto st = dyn_cast<memref::StoreOp>(op)) {
      if (st.getMemref() == memref)
        return true;
    } else if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
      if (iface.hasEffect<MemoryEffects::Write>()) {
        for (Value v : op->getOperands())
          if (v == memref)
            return true;
      }
    }
  }
  return false;
}
struct EliminateUndefToUndefCopyPattern
    : OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp srcAlloc,
                                PatternRewriter &rewriter) const override {
    Value src = srcAlloc.getResult();

    LLVM_DEBUG(llvm::dbgs()
               << "\n[elim] considering src alloc: " << src << "\n");

    // src must have exactly one load, no stores
    AffineLoadOp onlyLd = nullptr;
    for (Operation *user : src.getUsers()) {
      if (isa<AffineStoreOp>(user) || isa<memref::StoreOp>(user)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[elim] src alloc has store user → skip\n");
        return failure();
      }
      auto ld = dyn_cast<AffineLoadOp>(user);
      if (!ld) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[elim] src alloc has non-affine.load user\n");
        return failure();
      }
      if (onlyLd) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[elim] src alloc has multiple loads → skip\n");
        return failure();
      }
      onlyLd = ld;
    }

    if (!onlyLd)
      return failure();

    auto copyLoop = onlyLd->getParentOfType<AffineForOp>();
    if (!copyLoop) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[elim] load not inside affine.for → skip\n");
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs()
               << "[elim] candidate copy loop at "
               << copyLoop.getLoc() << "\n");

    AffineLoadOp bodyLd;
    AffineStoreOp bodySt;
    if (!matchSingleCopyBody(copyLoop, src, bodyLd, bodySt)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[elim] loop is not a pure copy → skip\n");
      return failure();
    }

    Value dst = bodySt.getMemRef();
    auto dstAlloc = dst.getDefiningOp<memref::AllocOp>();
    if (!dstAlloc) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[elim] dst is not memref.alloc → skip\n");
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs()
               << "[elim] copy dst memref: " << dst << "\n");

    // // src undefined?
    // if (hasWriteToMemrefInRange(srcAlloc, copyLoop, src)) {
    //   LLVM_DEBUG(llvm::dbgs()
    //              << "[elim] src written before loop → skip\n");
    //   return failure();
    // }

    // // dst undefined?
    // if (hasWriteToMemrefInRange(dstAlloc, copyLoop, dst)) {
    //   LLVM_DEBUG(llvm::dbgs()
    //              << "[elim] dst written before loop → skip\n");
    //   return failure();
    // }

    LLVM_DEBUG(llvm::dbgs()
               << "[elim] eliminating undef→undef copy loop\n");

    rewriter.eraseOp(copyLoop);

    if (src.use_empty()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[elim] erasing dead src alloc\n");
      rewriter.eraseOp(srcAlloc);
    }

    return success();
  }
};


} // end namespace

namespace {
struct EliminateSingleWriteMultipleReads
    : public EliminateSingleWriteMultipleReadsBase<EliminateSingleWriteMultipleReads> {
  EliminateSingleWriteMultipleReads() = default;
  EliminateSingleWriteMultipleReads(
    std::string argEliminationType
  ){
    eliminationType = argEliminationType;
  }
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = func.getContext();

    LLVM_DEBUG(llvm::dbgs() << "\n[ESWMR] running on func: " << func.getName() << "\n");
    LLVM_DEBUG(llvm::dbgs() << "[ESWMR] eliminationType=" << eliminationType << "\n");

    if (eliminationType == "skip") return;

    // Tonight-friendly pipeline:
    //  - Run stable data-movement elision first
    //  - Then run in-place hazard-avoidance patterns
    //  - Then cleanup
    //
    // "test": run the full chain.
    // "inplace": run only the hazard-avoidance subset.
    if (eliminationType == "inplace") {
      RewritePatternSet patterns(ctx);
      patterns.add<EliminateSingleWriteMultipleReadsInPlacePattern>(ctx);
      patterns.add<EliminateConsecutiveMultiWriteMultipleReadsInPlacePattern>(ctx);
      (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
      return;
    }

    if (eliminationType == "conservative") {
      {
        RewritePatternSet patterns(ctx);
        // patterns.add<ElideAffineCopyOrPermIntoAlloc>(ctx);
        patterns.add<ElideAffineIdentityCopyIntoAlloc>(ctx);
        (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
      }
      {
        RewritePatternSet patterns(ctx);
        patterns.add<EliminateUndefToUndefCopyPattern>(ctx);
        patterns.add<EliminateSingleWriteMultipleReadsInPlacePattern>(ctx);
        patterns.add<EliminateConsecutiveMultiWriteMultipleReadsInPlacePattern>(ctx);
        (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
      }
      {
        RewritePatternSet patterns(ctx);
        patterns.add<ElideAllocOnlyStored>(ctx);
        (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
      }
      return;
    }

    if (eliminationType == "aggressive") {
      {
        RewritePatternSet patterns(ctx);
        patterns.add<ElideAffineCopyOrPermIntoAlloc>(ctx);
        patterns.add<ElideAffineIdentityCopyIntoAlloc>(ctx);
        (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
      }
      {
        RewritePatternSet patterns(ctx);
        patterns.add<EliminateUndefToUndefCopyPattern>(ctx);
        patterns.add<EliminateSingleWriteMultipleReadsInPlacePattern>(ctx);
        patterns.add<EliminateConsecutiveMultiWriteMultipleReadsInPlacePattern>(ctx);
        (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
      }
      {
        RewritePatternSet patterns(ctx);
        patterns.add<ElideAllocOnlyStored>(ctx);
        (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
      }
      return;
    }

    // Fallback: keep old assert behavior.
    assert(false && "[ESWMR] Unknown elimination type");
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> streamhls::createEliminateSingleWriteMultipleReadsPass(
  std::string eliminationType
) {
  return std::make_unique<EliminateSingleWriteMultipleReads>(eliminationType);
}

ModuleOp streamhls::eliminateSingleWriteMultipleReads(ModuleOp &module, std::string eliminationType) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::dataflow::DataflowDialect, mlir::func::FuncDialect,
                  mlir::arith::ArithDialect, mlir::tensor::TensorDialect,
                  mlir::affine::AffineDialect, mlir::math::MathDialect,
                  mlir::memref::MemRefDialect>();
  PassManager pm = PassManager::on<ModuleOp>(module.getContext());

  pm.getDependentDialects(registry); // check if this is necessary

  // eliminate single write multiple reads
  pm.nest<func::FuncOp>().addPass(streamhls::createEliminateSingleWriteMultipleReadsPass(eliminationType));
  pm.addPass(mlir::createCanonicalizerPass());
  if (failed(pm.run(module)))
    assert(false && "Stream-HLS Eliminate Single Write Multiple Reads failed");

  return module;
}