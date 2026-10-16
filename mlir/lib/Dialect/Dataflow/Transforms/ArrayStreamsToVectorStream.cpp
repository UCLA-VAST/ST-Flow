//===- ArrayStreamsToVectorStream.cpp ---------------------------*- C++ -*-===//
//
// 1) Type conversion for function args/calls:
//      tensor<LANES x !dataflow.stream<Elem, depth>>
//        → !dataflow.stream<vector<LANES x Elem>, depth>
//
// 2) Access lowering (no tensor semantics anymore):
//    dataflow.aofs_read  → dataflow.stream_read + dataflow.vector_load
//    dataflow.aofs_write → dataflow.empty_vector + dataflow.vector_store
//                          + dataflow.stream_write
//
// Additionally, when hoisting stream_read / stream_write across unrolled loops,
// we reuse the original affine.if guard:
//   - We clone the nearest enclosing affine.if at the new insertion point.
//   - For any IVs that now go out of scope, we plug:
//       * loop LB when placing BEFORE the unrolled loops,
//       * loop UB when placing AFTER the unrolled loops.
//   - Inner loops must have constant bounds; otherwise we error.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "streamhls/Dialect/Dataflow/Transforms/Passes.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace affine;
using namespace streamhls;
using namespace dataflow;

#define DEBUG_TYPE "streamhls-array-streams-to-vector-stream"

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Convert 1D AoS type:
///   tensor<LANES x !dataflow.stream<Elem, depth>>
/// → !dataflow.stream<vector<LANES x Elem>, depth>
static Type convertAOSToVectorStream(Type t) {
  auto rt = t.dyn_cast<RankedTensorType>();
  if (!rt || rt.getRank() != 1)
    return t;

  auto elemStream = rt.getElementType().dyn_cast<StreamType>();
  if (!elemStream)
    return t;

  int64_t lanes   = rt.getDimSize(0);
  Type   scalar   = elemStream.getElementType();
  auto vecTy = VectorType::get({lanes}, scalar);
  return StreamType::get(t.getContext(), vecTy);
}

struct AOSToVectorStreamTypeConverter : public TypeConverter {
  AOSToVectorStreamTypeConverter() {
    addConversion([](Type t) { return t; });

    addConversion([](RankedTensorType rt) -> Type {
      return convertAOSToVectorStream(rt);
    });

    // Unrealized casts for materialization.
    addArgumentMaterialization(
        [](OpBuilder &b, Type type, ValueRange inputs, Location loc) {
          return b.create<UnrealizedConversionCastOp>(loc, type, inputs)
              .getResult(0);
        });
    addSourceMaterialization(
        [](OpBuilder &b, Type type, ValueRange inputs, Location loc) {
          return b.create<UnrealizedConversionCastOp>(loc, type, inputs)
              .getResult(0);
        });
    addTargetMaterialization(
        [](OpBuilder &b, Type type, ValueRange inputs, Location loc) {
          return b.create<UnrealizedConversionCastOp>(loc, type, inputs)
              .getResult(0);
        });
  }
};

/// Extract vector<LANES x Elem> from original AoS type.
static VectorType getLaneVectorTypeFromAOSType(Type aosType) {
  auto rt = aosType.dyn_cast<RankedTensorType>();
  if (!rt || rt.getRank() != 1)
    return nullptr;

  auto streamTy = rt.getElementType().dyn_cast<StreamType>();
  if (!streamTy)
    return nullptr;

  int64_t lanes = rt.getDimSize(0);
  Type elemTy   = streamTy.getElementType();
  return VectorType::get({lanes}, elemTy);
}

/// Get affine map attr from op or fall back to identity(rank).
static AffineMapAttr getMapAttr(Operation *op,
                                PatternRewriter &rewriter,
                                unsigned rank) {
  if (auto mapAttr = op->getAttrOfType<AffineMapAttr>("map"))
    return mapAttr;
  return AffineMapAttr::get(rewriter.getMultiDimIdentityMap(rank));
}

/// Find nearest ancestor affine.for whose IV is used in the index operands.
/// (Old behavior helper; still used as fallback.)
static AffineForOp findOuterMostIndexedLoop(Operation *userOp,
                                            ValueRange indexOperands) {
  llvm::SmallPtrSet<Value, 4> indexSet;
  indexSet.insert(indexOperands.begin(), indexOperands.end());

  if (indexSet.empty())
    return AffineForOp();

  AffineForOp outerMost;
  Operation *cur = userOp;

  while (auto forOp = dyn_cast_or_null<AffineForOp>(cur->getParentOp())) {
    Value iv = forOp.getInductionVar();
    if (!indexSet.contains(iv))
      break;

    outerMost = forOp;
    indexSet.erase(iv);
    cur = forOp;
  }

  return outerMost;
}

/// Find the nearest ancestor affine.for that carries the "pipeline" loop
/// directive (your `loop_directive` attribute).
static AffineForOp findPipelineRootLoop(Operation *userOp) {
  Operation *cur = userOp->getParentOp();
  while (cur) {
    if (auto forOp = dyn_cast<AffineForOp>(cur)) {
      if (forOp->hasAttr("loop_directive"))
        return forOp;
    }
    cur = cur->getParentOp();
  }
  return AffineForOp();
}

/// Collect affine.for loops strictly between `userOp` and `insertionOwner`.
/// These are the "inner" loops we are hoisting across.
static SmallVector<AffineForOp>
collectInnerLoopsBetween(Operation *userOp, Operation *insertionOwner) {
  SmallVector<AffineForOp> loops;
  if (!insertionOwner)
    return loops;

  Operation *cur = userOp->getParentOp();
  while (cur && cur != insertionOwner) {
    if (auto forOp = dyn_cast<AffineForOp>(cur))
      loops.push_back(forOp);
    cur = cur->getParentOp();
  }
  return loops;
}

/// Find the nearest enclosing AffineIfOp for `userOp`.
static AffineIfOp findNearestEnclosingAffineIf(Operation *userOp) {
  Operation *cur = userOp->getParentOp();
  while (cur) {
    if (auto ifOp = dyn_cast<AffineIfOp>(cur))
      return ifOp;
    cur = cur->getParentOp();
  }
  return AffineIfOp();
}

/// Clone the nearest enclosing AffineIfOp at the current insertion point,
/// adjusting operands for IVs from `innerLoops`.
///
/// - If `afterLoops == false` (pre-loop placement), replace inner IVs by LB.
/// - If `afterLoops == true`  (post-loop placement), replace inner IVs by UB.
///
/// Returns the new AffineIfOp if a guard existed, or a null AffineIfOp if no
/// affine.if was found. Emits an error on non-constant bounds.
static FailureOr<AffineIfOp>
cloneGuardingAffineIf(Operation *userOp,
                      ArrayRef<AffineForOp> innerLoops,
                      PatternRewriter &rewriter,
                      bool afterLoops) {
  AffineIfOp sourceIf = findNearestEnclosingAffineIf(userOp);
  if (!sourceIf)
    return AffineIfOp(); // no guard to clone

  Location loc = userOp->getLoc();

  // Pre-check: inner loops must have constant bounds.
  for (AffineForOp forOp : innerLoops) {
    if (!forOp.hasConstantLowerBound() || !forOp.hasConstantUpperBound()) {
      return userOp->emitError()
             << "ArrayStreamsToVectorStream requires constant lower/upper "
                "bounds on inner unrolled loops when cloning affine.if guards";
    }
  }

  auto operands = sourceIf.getOperands();
  SmallVector<Value, 8> newOperands;
  newOperands.reserve(operands.size());

  for (Value operand : operands) {
    bool replaced = false;
    for (AffineForOp inner : innerLoops) {
      if (operand == inner.getInductionVar()) {
        // Choose LB or UB.
        int64_t c = afterLoops ? inner.getConstantUpperBound() - 1
                               : inner.getConstantLowerBound();
        auto cst =
            rewriter.create<arith::ConstantIndexOp>(loc, c);
        newOperands.push_back(cst);
        replaced = true;
        break;
      }
    }
    if (!replaced)
      newOperands.push_back(operand);
  }

  // Reuse the same IntegerSet; operands are patched.
  auto newIf = rewriter.create<AffineIfOp>(
      loc, sourceIf.getIntegerSet(), newOperands,
      /*withElseRegion=*/false);

  return newIf;
}

//===----------------------------------------------------------------------===//
// ArrayOfStreamsOp → StreamOp (type-level conversion)
//===----------------------------------------------------------------------===//

struct ArrayOfStreamsOpConversion
    : public OpConversionPattern<ArrayOfStreamsOp> {
  using OpConversionPattern<ArrayOfStreamsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArrayOfStreamsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto oldTy =
        op.getChannels().getType().dyn_cast<RankedTensorType>();
    if (!oldTy)
      return failure();

    Type newTy = getTypeConverter()->convertType(oldTy);
    if (!newTy || newTy == oldTy)
      return failure();

    auto streamTy = newTy.dyn_cast<StreamType>();
    if (!streamTy)
      return failure();

    unsigned depth = op.getDepth();

    auto newOp =
        rewriter.create<StreamOp>(op.getLoc(), streamTy, depth);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// aofs_read → stream_read (hoisted + guarded) + vector_affine_load
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// aofs_read → hoisted empty_vector + (guarded) stream_read + vector_assign
//             + vector_affine_load at original site
//===----------------------------------------------------------------------===//

struct AofsReadToVectorLoadPattern
    : public OpConversionPattern<ArrayOfStreamsReadOp> {
  using OpConversionPattern<ArrayOfStreamsReadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArrayOfStreamsReadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "[AOS->Vector] Rewrite aofs_read: " << op << "\n");

    Type aosType = op.getChannels().getType();
    auto vecTy   = getLaneVectorTypeFromAOSType(aosType);
    if (!vecTy) {
      LLVM_DEBUG(llvm::dbgs() << "  Not a 1D AoS of streams; skip\n");
      return failure();
    }

    Value streamVal = adaptor.getChannels();
    auto streamTy   = streamVal.getType().dyn_cast<StreamType>();
    if (!streamTy || !streamTy.getElementType().isa<VectorType>()) {
      LLVM_DEBUG(llvm::dbgs()
          << "  Converted channel is not stream<vector<...>>; skip\n");
      return failure();
    }

    ValueRange indices = adaptor.getIndices();
    Location loc       = op.getLoc();

    // Decide where to hoist the stream_read (same logic as before).
    AffineForOp pipelineRoot = findPipelineRootLoop(op.getOperation());
    AffineForOp outerIndexed = findOuterMostIndexedLoop(op.getOperation(),
                                                        indices);

    Operation *insertionOwner = nullptr;
    if (pipelineRoot)
      insertionOwner = pipelineRoot.getOperation();
    else if (outerIndexed)
      insertionOwner = outerIndexed.getOperation();

    // Collect inner loops we are hoisting across.
    auto innerLoops = collectInnerLoopsBetween(op.getOperation(),
                                               insertionOwner);

    // This is the "buffer" vector handle we will use for all loads.
    Value vecBase;

    {
      OpBuilder::InsertionGuard guard(rewriter);

      if (auto forOp = dyn_cast_or_null<AffineForOp>(insertionOwner)) {
        rewriter.setInsertionPointToStart(forOp.getBody());
      } else if (insertionOwner) {
        rewriter.setInsertionPoint(insertionOwner);
      } else {
        rewriter.setInsertionPoint(op);
      }

      // 1) Create the empty vector at the hoisted level.
      LLVM_DEBUG(llvm::dbgs()
          << "  Inserting dataflow.empty_vector at hoisted level\n");
      vecBase = rewriter.create<EmptyVectorOp>(loc, vecTy).getResult();

      // 2) Clone guarding affine.if (if any), patching IVs with LB.
      auto guardOr =
          cloneGuardingAffineIf(op.getOperation(), innerLoops, rewriter,
                                /*afterLoops=*/false);
      if (failed(guardOr))
        return failure();

      AffineIfOp newIf = *guardOr;

      if (newIf) {
        LLVM_DEBUG(llvm::dbgs()
            << "  Inserting stream_read+vector_assign inside cloned affine.if guard\n");
        OpBuilder ifBuilder = OpBuilder::atBlockBegin(newIf.getThenBlock());
        Value vecToken =
            ifBuilder.create<StreamReadOp>(loc, vecTy, streamVal).getResult();
        // Copy vecToken into vecBase (persistent handle).
        ifBuilder.create<VectorAssignOp>(loc, vecBase, vecToken);
      } else {
        LLVM_DEBUG(llvm::dbgs() << "  Inserting stream_read+vector_assign without guard\n");
        Value vecToken =
            rewriter.create<StreamReadOp>(loc, vecTy, streamVal).getResult();
        rewriter.create<VectorAssignOp>(loc, vecBase, vecToken);
      }
    }

    // 3) Replace aofs_read with dataflow.vector_load at the original location,
    //    using vecBase as the vector operand.
    rewriter.setInsertionPoint(op);
    auto mapAttr = getMapAttr(op.getOperation(), rewriter, indices.size());

    LLVM_DEBUG(llvm::dbgs()
        << "  Replacing aofs_read with dataflow.vector_load using vecBase\n");

    auto load = rewriter.create<VectorAffineLoadOp>(
        loc, op.getResult().getType(), vecBase, indices, mapAttr);

    rewriter.replaceOp(op, load.getResult());
    return success();
  }
};

struct AofsWriteToVectorStorePattern
    : public OpConversionPattern<ArrayOfStreamsWriteOp> {
  using OpConversionPattern<ArrayOfStreamsWriteOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArrayOfStreamsWriteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "[AOS->Vector] Rewrite aofs_write: " << op << "\n");

    Type aosType = op.getChannels().getType();
    auto vecTy   = getLaneVectorTypeFromAOSType(aosType);
    if (!vecTy) {
      LLVM_DEBUG(llvm::dbgs() << "  Not a 1D AoS of streams; skip\n");
      return failure();
    }

    Value streamVal = adaptor.getChannels();
    auto streamTy   = streamVal.getType().dyn_cast<StreamType>();
    if (!streamTy || !streamTy.getElementType().isa<VectorType>()) {
      LLVM_DEBUG(llvm::dbgs()
          << "  Converted channel is not stream<vector<...>>; skip\n");
      return failure();
    }

    ValueRange indices = adaptor.getIndices();
    Location loc       = op.getLoc();

    // Where to hoist the vector storage / stream_write:
    AffineForOp pipelineRoot = findPipelineRootLoop(op.getOperation());
    AffineForOp outerIndexed = findOuterMostIndexedLoop(op.getOperation(),
                                                        indices);

    Operation *insertionOwner = nullptr;
    if (pipelineRoot)
      insertionOwner = pipelineRoot.getOperation();
    else if (outerIndexed)
      insertionOwner = outerIndexed.getOperation();

    auto innerLoops = collectInnerLoopsBetween(op.getOperation(),
                                               insertionOwner);

    // 1) Allocate the vector storage (empty_vector) at the start of the owner.
    Value vecStorage;
    {
      OpBuilder::InsertionGuard guard(rewriter);

      if (auto forOp = dyn_cast_or_null<AffineForOp>(insertionOwner)) {
        rewriter.setInsertionPointToStart(forOp.getBody());
      } else if (insertionOwner) {
        rewriter.setInsertionPoint(insertionOwner);
      } else {
        rewriter.setInsertionPoint(op);
      }

      LLVM_DEBUG(llvm::dbgs()
          << "  Inserting dataflow.empty_vector at chosen loop/owner level\n");

      vecStorage = rewriter.create<EmptyVectorOp>(loc, vecTy).getResult();
    }

    // 2) Insert stream_write at the *end* of the owner loop body (before
    //    its terminator), optionally under cloned guard with IVs replaced by UB.
    {
      OpBuilder::InsertionGuard guard(rewriter);
      if (auto forOp = dyn_cast_or_null<AffineForOp>(insertionOwner)) {
        // Inside the pipelined loop body, just before affine.yield.
        Operation *term = forOp.getBody()->getTerminator();
        rewriter.setInsertionPoint(term);
      } else if (insertionOwner) {
        rewriter.setInsertionPointAfter(insertionOwner);
      } else {
        rewriter.setInsertionPointAfter(op);
      }

      auto guardOr =
          cloneGuardingAffineIf(op.getOperation(), innerLoops, rewriter,
                                /*afterLoops=*/true);
      if (failed(guardOr))
        return failure();

      AffineIfOp newIf = *guardOr;
      if (newIf) {
        LLVM_DEBUG(llvm::dbgs()
            << "  Inserting stream_write inside cloned affine.if guard\n");
        OpBuilder ifBuilder = OpBuilder::atBlockBegin(newIf.getThenBlock());
        ifBuilder.create<StreamWriteOp>(loc, streamVal, vecStorage);
      } else {
        LLVM_DEBUG(llvm::dbgs() << "  Inserting stream_write without guard\n");
        rewriter.create<StreamWriteOp>(loc, streamVal, vecStorage);
      }
    }

    // 3) Replace aofs_write with vector_affine_store at original location.
    rewriter.setInsertionPoint(op);
    auto mapAttr = getMapAttr(op.getOperation(), rewriter, indices.size());

    LLVM_DEBUG(llvm::dbgs()
        << "  Replacing aofs_write with dataflow.vector_store\n");

    rewriter.create<VectorAffineStoreOp>(
        loc, vecStorage, adaptor.getValue(), indices, mapAttr);

    rewriter.eraseOp(op);
    return success();
  }
};


//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

namespace {

struct ArrayStreamsToVectorStreamPass
    : public ArrayStreamsToVectorStreamPassBase<
          ArrayStreamsToVectorStreamPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    LLVM_DEBUG(llvm::dbgs()
        << "\n=== Running ArrayStreamsToVectorStreamPass ===\n");

    AOSToVectorStreamTypeConverter typeConverter;

    ConversionTarget target(*ctx);
    target.addLegalDialect<dataflow::DataflowDialect,
                           func::FuncDialect,
                           arith::ArithDialect,
                           affine::AffineDialect,
                           tensor::TensorDialect,
                           memref::MemRefDialect,
                           math::MathDialect,
                           vector::VectorDialect>();

    target.addLegalOp<UnrealizedConversionCastOp>();

    // AoS creators/users are legal only if their channel types no longer
    // need conversion (i.e., already stream<vector<...>>).
    target.addDynamicallyLegalOp<ArrayOfStreamsOp>(
        [&](ArrayOfStreamsOp op) {
          Type t     = op.getChannels().getType();
          Type convT = typeConverter.convertType(t);
          return (!convT || convT == t);
        });

    target.addDynamicallyLegalOp<ArrayOfStreamsReadOp>(
        [&](ArrayOfStreamsReadOp op) {
          Type t     = op.getChannels().getType();
          Type convT = typeConverter.convertType(t);
          return (!convT || convT == t);
        });

    target.addDynamicallyLegalOp<ArrayOfStreamsWriteOp>(
        [&](ArrayOfStreamsWriteOp op) {
          Type t     = op.getChannels().getType();
          Type convT = typeConverter.convertType(t);
          return (!convT || convT == t);
        });

    // Func signatures and calls must be updated by the TypeConverter.
    target.addDynamicallyLegalOp<func::FuncOp>(
        [&](func::FuncOp f) {
          return typeConverter.isSignatureLegal(f.getFunctionType());
        });

    target.addDynamicallyLegalOp<func::CallOp>(
        [&](func::CallOp c) {
          return typeConverter.isLegal(c);
        });

    RewritePatternSet patterns(ctx);

    // 1) Function signatures + call sites.
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);

    // 2) AoS builder → StreamOp.
    patterns.add<ArrayOfStreamsOpConversion>(typeConverter, ctx);

    // 3) Access lowering for aofs_read / aofs_write.
    patterns.add<AofsReadToVectorLoadPattern,
                 AofsWriteToVectorStorePattern>(typeConverter, ctx);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      LLVM_DEBUG(llvm::dbgs() << "[Pass] ERROR: Partial conversion failed!\n");
      signalPassFailure();
      return;
    }

    LLVM_DEBUG(llvm::dbgs()
        << "[Pass] SUCCESS: AoS → vector-stream conversion completed.\n");
  }
};

} // end anonymous namespace

std::unique_ptr<Pass> streamhls::createArrayStreamsToVectorStreamPass() {
  return std::make_unique<ArrayStreamsToVectorStreamPass>();
}

ModuleOp streamhls::arrayStreamsToVectorStream(ModuleOp &module) {
  MLIRContext *ctx = module.getContext();
  PassManager pm(ctx);
  pm.addPass(streamhls::createArrayStreamsToVectorStreamPass());

  if (failed(pm.run(module))) {
    module.emitError("ArrayStreamsToVectorStream pass failed");
  }
  return module;
}
