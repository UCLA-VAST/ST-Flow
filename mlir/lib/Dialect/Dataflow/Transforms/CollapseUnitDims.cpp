//===- CollapseUnitDimsPass.cpp ---------------------------------*- C++ -*-===//
//
// Collapse unit (size-1) dimensions of memrefs and ranked tensors,
// and update affine / ArrayOfStreams accesses accordingly.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "streamhls/Dialect/Dataflow/IR/Dataflow.h"
#include "streamhls/Dialect/Dataflow/Transforms/Passes.h"

using namespace mlir;
using namespace streamhls;
using namespace dataflow;

#define DEBUG_TYPE "streamhls-collapse-unit-dims"

namespace {

//===----------------------------------------------------------------------===//
// Type helpers
//===----------------------------------------------------------------------===//

// Collapse unit dims for MemRefType (static shape, identity layout only).
static FailureOr<MemRefType> collapseUnitDims(MemRefType ty) {
  if (!ty.hasStaticShape())
    return failure(); // keep it simple
  if (!ty.getLayout().isIdentity())
    return failure(); // identity only

  SmallVector<int64_t> newShape;
  SmallVector<unsigned> keptIdx;
  for (auto it : llvm::enumerate(ty.getShape())) {
    if (it.value() != 1) {
      newShape.push_back(it.value());
      keptIdx.push_back(it.index());
    }
  }
  // If all were 1 → rank-0 memref.
  if (newShape.empty())
    return MemRefType::get({}, ty.getElementType());

  return MemRefType::get(newShape, ty.getElementType());
}

// Collapse unit dims for RankedTensorType (static shape only).
static FailureOr<RankedTensorType> collapseUnitDims(RankedTensorType ty) {
  if (!ty.hasStaticShape())
    return failure(); // keep it simple for now

  SmallVector<int64_t> newShape;
  newShape.reserve(ty.getRank());
  for (int64_t d : ty.getShape())
    if (d != 1)
      newShape.push_back(d);

  // If all dims were 1 → rank-0 tensor (scalar tensor).
  if (newShape.empty())
    return RankedTensorType::get(/*shape=*/{}, ty.getElementType());

  return RankedTensorType::get(newShape, ty.getElementType());
}

// Compute kept result positions (dims whose size != 1) – memref version.
static inline SmallVector<unsigned> keptResultPositions(MemRefType oldTy) {
  SmallVector<unsigned> kept;
  kept.reserve(oldTy.getRank());
  for (unsigned i = 0, e = oldTy.getRank(); i < e; ++i)
    if (oldTy.getDimSize(i) != 1)
      kept.push_back(i);
  return kept;
}

// Same for ranked tensors.
static inline SmallVector<unsigned> keptResultPositions(RankedTensorType oldTy) {
  SmallVector<unsigned> kept;
  kept.reserve(oldTy.getRank());
  for (unsigned i = 0, e = oldTy.getRank(); i < e; ++i)
    if (oldTy.getDimSize(i) != 1)
      kept.push_back(i);
  return kept;
}

//===----------------------------------------------------------------------===//
// Affine map + operand collapsing
//===----------------------------------------------------------------------===//

// Build a collapsed affine map+operands by keeping only the results at
// `keptResIdx`. Operands are remapped and unused dims/symbols are pruned.
static LogicalResult buildCollapsedMapAndOperands(
    AffineMap oldMap, ValueRange oldMapOperands, ArrayRef<unsigned> keptResIdx,
    unsigned newRank, AffineMap &newMap,
    SmallVectorImpl<Value> &newMapOperands, MLIRContext *ctx) {
  if (oldMap.getNumResults() == 0) {
    // Scalar base already — nothing to keep; ensure new rank is also 0.
    if (newRank != 0)
      return failure();
    newMap = AffineMap::get(oldMap.getNumDims(), oldMap.getNumSymbols(), {}, ctx);
    newMapOperands.clear();
    return success();
  }

  // Collect kept result expressions in-order.
  SmallVector<AffineExpr> keptExprs;
  keptExprs.reserve(keptResIdx.size());
  for (unsigned r : keptResIdx) {
    if (r >= oldMap.getNumResults())
      return failure();
    keptExprs.push_back(oldMap.getResult(r));
  }

  // Build interim map with same (dims, symbols), fewer results.
  AffineMap interim =
      AffineMap::get(oldMap.getNumDims(), oldMap.getNumSymbols(), keptExprs, ctx);

  // Start with original operands (dims first, then symbols).
  newMapOperands.assign(oldMapOperands.begin(), oldMapOperands.end());

  // Canonicalize/prune unused dims/symbols from map and operands.
  affine::canonicalizeMapAndOperands(&interim, &newMapOperands);

  // The interim map now references exactly the dims/symbols appearing in keptExprs.
  // It must have #results == new base rank.
  if (interim.getNumResults() != newRank)
    return failure();

  newMap = interim;
  return success();
}

//===----------------------------------------------------------------------===//
// Memref affine access rewriting
//===----------------------------------------------------------------------===//

// Rewrite an affine.load/store on `oldMemref` to use `newMemref` and a collapsed
// affine map+operands.
static LogicalResult rewriteAffineAccessWithMap(Operation *op, Value oldMemref,
                                                Value newMemref, MemRefType oldTy,
                                                MemRefType newTy,
                                                PatternRewriter &rewriter) {
  auto *ctx = rewriter.getContext();

  if (auto load = dyn_cast<affine::AffineLoadOp>(op)) {
    if (load.getMemRef() != oldMemref)
      return failure();

    AffineMap oldMap = load.getAffineMap();
    SmallVector<unsigned> kept = keptResultPositions(oldTy);

    AffineMap newMap;
    SmallVector<Value> newOps;
    if (failed(buildCollapsedMapAndOperands(oldMap, load.getMapOperands(), kept,
                                            newTy.getRank(), newMap, newOps,
                                            ctx)))
      return failure();

    rewriter.setInsertionPoint(load);
    auto repl = rewriter.create<affine::AffineLoadOp>(load.getLoc(), newMemref,
                                                      newMap, newOps);
    load.replaceAllUsesWith(repl.getResult());
    rewriter.eraseOp(load);
    return success();
  }

  if (auto store = dyn_cast<affine::AffineStoreOp>(op)) {
    if (store.getMemRef() != oldMemref)
      return failure();

    AffineMap oldMap = store.getAffineMap();
    SmallVector<unsigned> kept = keptResultPositions(oldTy);

    AffineMap newMap;
    SmallVector<Value> newOps;
    if (failed(buildCollapsedMapAndOperands(oldMap, store.getMapOperands(), kept,
                                            newTy.getRank(), newMap, newOps,
                                            ctx)))
      return failure();

    rewriter.setInsertionPoint(store);
    rewriter.create<affine::AffineStoreOp>(store.getLoc(), store.getValue(),
                                           newMemref, newMap, newOps);
    rewriter.eraseOp(store);
    return success();
  }

  return failure();
}

// Rebuild an affine.load/store `op` to use `baseMem` (new arg type) and a
// collapsed map/operands computed from the *old* access map + kept results.
static LogicalResult rewriteAffineAccessOnCollapsedArg(Operation *op,
                                                       Value baseMem,
                                                       MemRefType oldTy,
                                                       MemRefType newTy,
                                                       PatternRewriter &rewriter) {
  auto *ctx = rewriter.getContext();

  if (auto load = dyn_cast<affine::AffineLoadOp>(op)) {
    AffineMap oldMap = load.getAffineMap();
    SmallVector<unsigned> kept = keptResultPositions(oldTy);
    AffineMap newMap;
    SmallVector<Value> newOps;
    if (failed(buildCollapsedMapAndOperands(oldMap, load.getMapOperands(), kept,
                                            newTy.getRank(), newMap, newOps,
                                            ctx)))
      return failure();

    rewriter.setInsertionPoint(load);
    auto nl = rewriter.create<affine::AffineLoadOp>(load.getLoc(), baseMem,
                                                    newMap, newOps);
    load.replaceAllUsesWith(nl.getResult());
    rewriter.eraseOp(load);
    return success();
  }

  if (auto store = dyn_cast<affine::AffineStoreOp>(op)) {
    AffineMap oldMap = store.getAffineMap();
    SmallVector<unsigned> kept = keptResultPositions(oldTy);
    AffineMap newMap;
    SmallVector<Value> newOps;
    if (failed(buildCollapsedMapAndOperands(oldMap, store.getMapOperands(), kept,
                                            newTy.getRank(), newMap, newOps,
                                            ctx)))
      return failure();

    rewriter.setInsertionPoint(store);
    rewriter.create<affine::AffineStoreOp>(store.getLoc(), store.getValue(),
                                           baseMem, newMap, newOps);
    rewriter.eraseOp(store);
    return success();
  }

  return failure();
}

// Rewrite all uses of a specific BlockArgument whose memref type changed
// from oldTy→newTy.
static void rewriteUsesForCollapsedArg(BlockArgument barg, MemRefType oldTy,
                                       MemRefType newTy,
                                       PatternRewriter &rewriter) {
  SmallVector<Operation *> worklist(barg.user_begin(), barg.user_end());

  while (!worklist.empty()) {
    Operation *u = worklist.pop_back_val();

    // Follow memref.cast chains to leaves; rewrite loads/stores there.
    if (auto c = dyn_cast<memref::CastOp>(u)) {
      for (Operation *uu : c.getResult().getUsers())
        worklist.push_back(uu);
      continue;
    }

    if (succeeded(
            rewriteAffineAccessOnCollapsedArg(u, barg, oldTy, newTy, rewriter)))
      continue;

    // Unknown/unsupported user; skip. (Subview/view would need dedicated logic.)
  }

  // After rewriting leaf users, remove trivial casts on this arg that became no-ops.
  SmallVector<memref::CastOp> deadCasts;
  for (Operation *u : barg.getUsers())
    if (auto c = dyn_cast<memref::CastOp>(u))
      if (c.getSource().getType() == c.getType())
        deadCasts.push_back(c);

  for (auto c : deadCasts) {
    c.getResult().replaceAllUsesWith(c.getSource());
    c.erase();
  }
}

//===----------------------------------------------------------------------===//
// ArrayOfStreams (RankedTensor) access rewriting
//===----------------------------------------------------------------------===//

static LogicalResult rewriteAofsAccessOnCollapsedTensor(
    Operation *op, Value baseTensor, RankedTensorType oldTy,
    RankedTensorType newTy, PatternRewriter &rewriter) {
  auto *ctx = rewriter.getContext();

  if (auto read = dyn_cast<ArrayOfStreamsReadOp>(op)) {
    if (read.getChannels() != baseTensor)
      return failure();

    AffineMap oldMap = read.getAffineMapAttr().getAffineMap();
    SmallVector<unsigned> kept = keptResultPositions(oldTy);

    AffineMap newMap;
    SmallVector<Value> newOps;
    if (failed(buildCollapsedMapAndOperands(oldMap, read.getIndices(), kept,
                                            newTy.getRank(), newMap, newOps,
                                            ctx)))
      return failure();

    rewriter.setInsertionPoint(read);
    auto newRead = rewriter.create<ArrayOfStreamsReadOp>(
        read.getLoc(),
        read.getResult().getType(), // result type unchanged (e.g., f32)
        baseTensor, newOps, AffineMapAttr::get(newMap));

    read.replaceAllUsesWith(newRead.getResult());
    rewriter.eraseOp(read);
    return success();
  }

  if (auto write = dyn_cast<ArrayOfStreamsWriteOp>(op)) {
    if (write.getChannels() != baseTensor)
      return failure();

    AffineMap oldMap = write.getAffineMapAttr().getAffineMap();
    SmallVector<unsigned> kept = keptResultPositions(oldTy);

    AffineMap newMap;
    SmallVector<Value> newOps;
    if (failed(buildCollapsedMapAndOperands(oldMap, write.getIndices(), kept,
                                            newTy.getRank(), newMap, newOps,
                                            ctx)))
      return failure();

    rewriter.setInsertionPoint(write);
    rewriter.create<ArrayOfStreamsWriteOp>(
        write.getLoc(), baseTensor, write.getValue(), newOps,
        AffineMapAttr::get(newMap));

    rewriter.eraseOp(write);
    return success();
  }

  return failure();
}

// Rewrite all uses of a tensor block-argument whose type changed from
// oldTy→newTy. Currently only handles AoS ops (+ optional tensor.cast).
static void rewriteUsesForCollapsedTensorArg(BlockArgument barg,
                                             RankedTensorType oldTy,
                                             RankedTensorType newTy,
                                             PatternRewriter &rewriter) {
  SmallVector<Operation *> worklist(barg.user_begin(), barg.user_end());

  while (!worklist.empty()) {
    Operation *u = worklist.pop_back_val();

    if (auto c = dyn_cast<tensor::CastOp>(u)) {
      for (Operation *uu : c.getResult().getUsers())
        worklist.push_back(uu);
      continue;
    }

    if (succeeded(rewriteAofsAccessOnCollapsedTensor(u, barg, oldTy, newTy,
                                                     rewriter)))
      continue;

    // Unknown / unsupported user — left as-is. Extend as needed.
  }
}

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

// Collapse unit dims for memref.alloc.
struct CollapseAllocPattern : OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp alloc,
                                PatternRewriter &rewriter) const override {
    auto oldTy = alloc.getType();
    auto newTyOr = collapseUnitDims(oldTy);
    if (failed(newTyOr))
      return failure();
    auto newTy = *newTyOr;
    if (newTy == oldTy)
      return failure();

    // New alloc.
    rewriter.setInsertionPoint(alloc);
    auto newAlloc = rewriter.create<memref::AllocOp>(alloc.getLoc(), newTy);

    // Rewrite all affine load/store users.
    SmallVector<Operation *> users(alloc->user_begin(), alloc->user_end());
    for (Operation *u : users)
      (void)rewriteAffineAccessWithMap(u, alloc.getResult(),
                                       newAlloc.getResult(), oldTy, newTy,
                                       rewriter);

    // Replace any remaining uses that don't encode indices (shouldn’t exist in
    // your IR).
    alloc.getResult().replaceAllUsesWith(newAlloc.getResult());
    rewriter.eraseOp(alloc);
    return success();
  }
};

// Collapse unit dims for ArrayOfStreams producers (dataflow.aofs) in-place.
// We mutate the result type and then rewrite all AoS accesses to match.
struct CollapseAofsPattern : OpRewritePattern<ArrayOfStreamsOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ArrayOfStreamsOp op,
                                PatternRewriter &rewriter) const override {
    auto oldTy = op.getChannels().getType().dyn_cast<RankedTensorType>();
    if (!oldTy)
      return failure();

    auto newTyOr = collapseUnitDims(oldTy);
    if (failed(newTyOr) || *newTyOr == oldTy)
      return failure();
    auto newTy = *newTyOr;

    // Update the result type in-place.
    Value channels = op.getResult();
    channels.setType(newTy);

    // Rewrite AoS accesses on this value.
    SmallVector<Operation *> users(channels.user_begin(), channels.user_end());
    for (Operation *u : users)
      (void)rewriteAofsAccessOnCollapsedTensor(u, channels, oldTy, newTy,
                                               rewriter);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Function-argument collapsing (in-place)
//===----------------------------------------------------------------------===//

// In-place collapse of unit dims for function arguments.
// Also updates loads/stores and AoS ops that (possibly through cast) access
// those args.
//
// IMPORTANT: this assumes there are no func.call sites, or that you handle
// them elsewhere. In your current usage, you said there are no calls.
static LogicalResult collapseFuncArgsInPlace(func::FuncOp func) {
  auto oldFty = func.getFunctionType();
  MLIRContext *ctx = func.getContext();

  // Compute new input types and which args collapse.
  SmallVector<Type> newInputs;
  newInputs.reserve(oldFty.getNumInputs());
  SmallVector<bool> collapse;
  collapse.reserve(oldFty.getNumInputs());

  SmallVector<MemRefType> oldArgMemTy(oldFty.getNumInputs(), {});
  SmallVector<MemRefType> newArgMemTy(oldFty.getNumInputs(), {});
  SmallVector<RankedTensorType> oldArgTensorTy(oldFty.getNumInputs(), {});
  SmallVector<RankedTensorType> newArgTensorTy(oldFty.getNumInputs(), {});

  for (auto i : llvm::seq<unsigned>(0, oldFty.getNumInputs())) {
    Type t = oldFty.getInput(i);

    if (auto mt = dyn_cast<MemRefType>(t)) {
      oldArgMemTy[i] = mt;
      if (auto nt = collapseUnitDims(mt); succeeded(nt) && *nt != mt) {
        collapse.push_back(true);
        newInputs.push_back(*nt);
        newArgMemTy[i] = *nt;
        continue;
      }
      collapse.push_back(false);
      newInputs.push_back(t);
      newArgMemTy[i] = mt;
      continue;
    }

    if (auto tt = dyn_cast<RankedTensorType>(t)) {
      oldArgTensorTy[i] = tt;
      if (auto nt = collapseUnitDims(tt); succeeded(nt) && *nt != tt) {
        collapse.push_back(true);
        newInputs.push_back(*nt);
        newArgTensorTy[i] = *nt;
        continue;
      }
      collapse.push_back(false);
      newInputs.push_back(t);
      newArgTensorTy[i] = tt;
      continue;
    }

    // Non-memref, non-ranked-tensor: leave as-is.
    collapse.push_back(false);
    newInputs.push_back(t);
  }

  if (llvm::none_of(collapse, [](bool b) { return b; }))
    return success();

  // Update the function type in place.
  auto newFty = FunctionType::get(ctx, newInputs, oldFty.getResults());
  func.setType(newFty);

  // Update entry block argument types in place.
  PatternRewriter rewriter(ctx);
  Block &entry = func.front();
  for (auto i : llvm::seq<unsigned>(0, newFty.getNumInputs())) {
    if (collapse[i]) {
      auto barg = entry.getArgument(i);
      barg.setType(newInputs[i]);
    }
  }

  // Rewrite uses for each collapsed arg.
  for (auto i : llvm::seq<unsigned>(0, newFty.getNumInputs())) {
    if (!collapse[i])
      continue;
    auto barg = entry.getArgument(i);

    if (oldArgMemTy[i]) {
      rewriteUsesForCollapsedArg(barg, oldArgMemTy[i], newArgMemTy[i], rewriter);
      continue;
    }

    if (oldArgTensorTy[i]) {
      rewriteUsesForCollapsedTensorArg(barg, oldArgTensorTy[i],
                                       newArgTensorTy[i], rewriter);
      continue;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

struct CollapseUnitDims : public CollapseUnitDimsBase<CollapseUnitDims> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    // 1) Collapse allocs and aofs inside the module.
    RewritePatternSet patterns(ctx);
    patterns.add<CollapseAllocPattern, CollapseAofsPattern>(ctx);
    GreedyRewriteConfig cfg;
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns), cfg)))
      signalPassFailure();

    // 2) Collapse function arguments in place.
    // You stated there are no func.call sites in your use cases,
    // so we don't rewrite call sites here.
    module.walk([&](func::FuncOp f) {
      (void)collapseFuncArgsInPlace(f);
    });
  }
};

} // namespace

std::unique_ptr<Pass> streamhls::createCollapseUnitDimsPass() {
  return std::make_unique<CollapseUnitDims>();
}