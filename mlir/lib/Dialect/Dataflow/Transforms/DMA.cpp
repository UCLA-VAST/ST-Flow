#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#include "streamhls/Dialect/Dataflow/Transforms/Passes.h"
#include "streamhls/Dialect/Dataflow/Transforms/MergeKernels/Transform.h"

using namespace mlir;
using namespace affine;
using namespace dataflow;
using namespace streamhls;

#define DEBUG_TYPE "streamhls-dma"

namespace {


static MemRefType makePackedMemRefType(MLIRContext *ctx, int64_t N, Type elemTy) {
  (void)ctx;
  return MemRefType::get({N}, elemTy);
}

/// Max AXI (or HLS stream) width in bits for one beat.
static constexpr int64_t kMaxStreamBitsPerBeat = 512;

/// How to split a wide vector type into <=512-bit beats for streaming.
struct VectorSerialization {
  int64_t numBeats = 1;     /// S — packed memref has logicalLength * numBeats elements
  int64_t lanesPerBeat = 1; /// L — scalar lanes per beat (e.g. 16 x f32 @ 512b)
};

static int64_t nextPowerOfTwo(int64_t n) {
  if (n <= 0)
    return 1;
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n |= n >> 32;
  n++;
  return n;
}

static VectorSerialization getVectorSerialization(Type t) {
  VectorSerialization vs;
  auto vecTy = dyn_cast<VectorType>(t);
  if (!vecTy)
    return vs;
  Type baseTy = vecTy.getElementType();
  int64_t bitWidth = 32;
  if (baseTy.isIntOrFloat())
    bitWidth = baseTy.getIntOrFloatBitWidth();

  int64_t W = vecTy.getNumElements();
  int64_t totalBits = W * bitWidth;
  if (totalBits <= kMaxStreamBitsPerBeat) {
    /// Single beat: widen W to next power of two for AXI compatibility.
    vs.lanesPerBeat = nextPowerOfTwo(W);
    vs.numBeats = 1;
    return vs;
  }
  int64_t L = kMaxStreamBitsPerBeat / bitWidth;
  assert(L > 0 && "scalar element wider than kMaxStreamBitsPerBeat");
  vs.lanesPerBeat = L;
  vs.numBeats = (W + L - 1) / L; /// ceil(W / L)
  return vs;
}

static int64_t getSerializationFactor(Type t) {
  return getVectorSerialization(t).numBeats;
}

/// Guard: lanesPerBeat * j + k < W  <=>  W - 1 - lanesPerBeat*j - k >= 0
static void emitPartialBeatGuardedXfer(
    OpBuilder &b, Location loc, MLIRContext *ctx, int64_t W,
    int64_t lanesPerBeat, Value j, Value k,
    llvm::function_ref<void(OpBuilder &)> emitThen) {
  AffineExpr d0 = getAffineDimExpr(0, ctx);
  AffineExpr d1 = getAffineDimExpr(1, ctx);
  AffineExpr ge0 = getAffineConstantExpr(W - 1, ctx) -
                   getAffineConstantExpr(lanesPerBeat, ctx) * d0 - d1;
  IntegerSet iset = IntegerSet::get(2, 0, {ge0}, {false});
  auto ifOp =
      b.create<AffineIfOp>(loc, iset, ValueRange{j, k}, /*withElseRegion=*/false);
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(&ifOp.getThenRegion().front());
  emitThen(b);
}


static void setFuncArgType(func::FuncOp f, unsigned argIdx, Type newTy) {
  // 1) Update function_type attribute.
  auto oldType = f.getFunctionType();
  SmallVector<Type> inputs(oldType.getInputs().begin(), oldType.getInputs().end());
  inputs[argIdx] = newTy;
  auto newFnType = FunctionType::get(f.getContext(), inputs, oldType.getResults());
  f->setAttr(f.getFunctionTypeAttrName(), TypeAttr::get(newFnType));

  // 2) Update the entry block argument type.
  f.getBody().getArgument(argIdx).setType(newTy);
}

static void updateTopArgType(func::FuncOp top, unsigned topArgIdx, Type newType) {
  setFuncArgType(top, topArgIdx, newType);
}

static bool collectStreamWriteOutputs(func::FuncOp f,
                                      SmallVectorImpl<unsigned> &streamArgIdxs,
                                      SmallVectorImpl<dataflow::StreamWriteOp> &writeOps) {
  streamArgIdxs.clear();
  writeOps.clear();

  // For each argument that is a stream, find a StreamWriteOp that writes to it.
  for (unsigned i = 0; i < f.getNumArguments(); ++i) {
    auto st = dyn_cast<dataflow::StreamType>(f.getArgument(i).getType());
    if (!st) continue;

    dataflow::StreamWriteOp found = nullptr;
    f.walk([&](dataflow::StreamWriteOp op) {
      if (!found && op.getChannel() == f.getArgument(i))
        found = op;
    });

    if (found) {
      streamArgIdxs.push_back(i);
      writeOps.push_back(found);
    }
  }

  return !writeOps.empty();
}

// Multi-output rewrite: packed memref -> N vector loads -> write to all output streams.
static LogicalResult rewriteReadModuleToLinearDMA_MultiOut(func::FuncOp f,
                                                          unsigned memrefArgIdx,
                                                          ArrayRef<unsigned> streamArgIdxs,
                                                          int64_t N,
                                                          Type elemTy,
                                                          Attribute loopDirective) {
  if (N <= 0 || streamArgIdxs.empty())
    return failure();

  VectorSerialization ser = getVectorSerialization(elemTy);
  int64_t S = ser.numBeats;
  Type narrowTy = elemTy;

  auto vecTy = dyn_cast<VectorType>(elemTy);
  int64_t W = vecTy ? vecTy.getNumElements() : 1;

  if (vecTy && (S > 1 || ser.lanesPerBeat != W)) {
    narrowTy = VectorType::get({ser.lanesPerBeat}, vecTy.getElementType());
  }

  auto packedTy = makePackedMemRefType(f.getContext(), N * S, narrowTy);
  setFuncArgType(f, memrefArgIdx, packedTy);

  Block &entry = f.getBody().front();
  Operation *term = entry.getTerminator();

  SmallVector<Operation*> erase;
  for (Operation &op : llvm::make_early_inc_range(entry.getOperations())) {
    if (&op == term) continue;
    erase.push_back(&op);
  }
  for (Operation *op : erase) op->erase();

  OpBuilder b(&entry, entry.begin());
  Location loc = f.getLoc();

  auto forOp = b.create<AffineForOp>(loc, 0, N);
  if (loopDirective)
    forOp->setAttr("loop_directive", loopDirective);

  b.setInsertionPointToStart(forOp.getBody());
  Value iv     = forOp.getInductionVar();
  Value packed = f.getArgument(memrefArgIdx);

  if (S == 1 && (ser.lanesPerBeat == W)) {
    Value v = b.create<AffineLoadOp>(loc, packed, ValueRange{iv}).getResult();
    for (unsigned sIdx : streamArgIdxs) {
      Value stream = f.getArgument(sIdx);
      b.create<dataflow::StreamWriteOp>(loc, stream, v);
    }
  } else {
    // Serialization logic (Read narrow, assemble wide)
    auto vecTy = cast<VectorType>(elemTy);
    int64_t lanesPerBeat = ser.lanesPerBeat;
    int64_t W = vecTy.getNumElements();

    Value wideVec = b.create<dataflow::EmptyVectorOp>(loc, elemTy).getResult();

    auto innerFor = b.create<AffineForOp>(loc, 0, S);
    b.setInsertionPointToStart(innerFor.getBody());
    Value j = innerFor.getInductionVar();

    // idx = iv * S + j
    auto amap = AffineMap::get(2, 0, b.getAffineDimExpr(0) * S + b.getAffineDimExpr(1), f.getContext());
    Value idx = b.create<AffineApplyOp>(loc, amap, ValueRange{iv, j});

    Value narrowVal = b.create<AffineLoadOp>(loc, packed, ValueRange{idx}).getResult();

    auto identityMap = AffineMap::get(1, 0, b.getAffineDimExpr(0), f.getContext());
    auto identityMapAttr = AffineMapAttr::get(identityMap);

    // globalIdx = j * lanesPerBeat + k
    auto gmap = AffineMap::get(
        2, 0,
        b.getAffineDimExpr(0) * lanesPerBeat + b.getAffineDimExpr(1),
        f.getContext());

    // Transfer narrowVal (lanesPerBeat lanes, last beat partial) into wideVec
    auto kFor = b.create<AffineForOp>(loc, 0, lanesPerBeat);
    b.setInsertionPointToStart(kFor.getBody());
    Value k = kFor.getInductionVar();
    Value gIdx = b.create<AffineApplyOp>(loc, gmap, ValueRange{j, k});

    emitPartialBeatGuardedXfer(b, loc, f.getContext(), W, lanesPerBeat, j, k,
                               [&](OpBuilder &bb) {
                                 Value valBit =
                                     bb
                                         .create<dataflow::VectorAffineLoadOp>(
                                             loc, vecTy.getElementType(),
                                             narrowVal, ValueRange{k},
                                             identityMapAttr)
                                         .getResult();
                                 bb.create<dataflow::VectorAffineStoreOp>(
                                     loc, wideVec, valBit, ValueRange{gIdx},
                                     identityMapAttr);
                               });

    b.setInsertionPointAfter(innerFor);
    for (unsigned sIdx : streamArgIdxs) {
      Value stream = f.getArgument(sIdx);
      b.create<dataflow::StreamWriteOp>(loc, stream, wideVec);
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Attach original_layout into buffer.info on a func argument
// - Captures the *current* memref type layout (incl. affine map) before DMA rewrite
// - Stores it under:  arg_attrs[i]["buffer.info"]["original_layout"]
//
// Format:
//   original_layout = {
//     memref_type   = <TypeAttr memref<...>>,
//     shape         = <ArrayAttr<i64>>,
//     elem_type     = <TypeAttr>,
//     rank          = <I64IntegerAttr>,
//     layout_map    = <AffineMapAttr> (only if non-identity),
//     memory_space  = <Attribute> (only if present)
//   }
//===----------------------------------------------------------------------===//

static DictionaryAttr getFuncArgAttrsDict(func::FuncOp f, unsigned argIdx) {
  auto opt = f.getArgAttrs();
  if (!opt.has_value())
    return nullptr;
  ArrayAttr arr = opt.value();
  if (argIdx >= arr.size())
    return nullptr;
  return dyn_cast_or_null<DictionaryAttr>(arr[argIdx]);
}

static void setFuncArgAttrsDict(func::FuncOp f, unsigned argIdx, DictionaryAttr dict) {
  MLIRContext *ctx = f.getContext();
  auto opt = f.getArgAttrs();
  SmallVector<Attribute> attrs;

  if (opt.has_value()) {
    auto arr = opt.value();
    attrs.assign(arr.begin(), arr.end());
  } else {
    attrs.assign(f.getNumArguments(), DictionaryAttr::get(ctx));
  }

  if (attrs.size() < f.getNumArguments())
    attrs.resize(f.getNumArguments(), DictionaryAttr::get(ctx));

  attrs[argIdx] = dict ? Attribute(dict) : Attribute(DictionaryAttr::get(ctx));
  f.setAllArgAttrs(ArrayAttr::get(ctx, attrs));
}

static DictionaryAttr getOrCreateSubDict(OpBuilder &b, DictionaryAttr parent,
                                        StringRef key) {
  if (parent) {
    if (auto sub = parent.getAs<DictionaryAttr>(key))
      return sub;
  }
  return DictionaryAttr::get(b.getContext());
}

static DictionaryAttr setSubDict(OpBuilder &b, DictionaryAttr parent,
                                 StringRef key, DictionaryAttr sub) {
  SmallVector<NamedAttribute> kvs;
  if (parent)
    kvs.assign(parent.getValue().begin(), parent.getValue().end());

  bool replaced = false;
  for (auto &na : kvs) {
    if (na.getName() == key) {
      na.setValue(sub);
      replaced = true;
      break;
    }
  }
  if (!replaced)
    kvs.push_back(b.getNamedAttr(key, sub));
  return DictionaryAttr::get(b.getContext(), kvs);
}

static DictionaryAttr setDictAttr(OpBuilder &b, DictionaryAttr dict,
                                  StringRef key, Attribute value) {
  SmallVector<NamedAttribute> kvs;
  if (dict)
    kvs.assign(dict.getValue().begin(), dict.getValue().end());

  bool replaced = false;
  for (auto &na : kvs) {
    if (na.getName() == key) {
      na.setValue(value);
      replaced = true;
      break;
    }
  }
  if (!replaced)
    kvs.push_back(b.getNamedAttr(key, value));
  return DictionaryAttr::get(b.getContext(), kvs);
}

static DictionaryAttr buildOriginalLayoutDict(OpBuilder &b, MemRefType mr) {
  MLIRContext *ctx = b.getContext();

  SmallVector<NamedAttribute> fields;

  // Full memref type (most faithful)
  fields.push_back(b.getNamedAttr("memref_type", TypeAttr::get(mr)));

  // Shape + elem type + rank (convenience for host codegen)
  SmallVector<Attribute> shapeAttrs;
  shapeAttrs.reserve(mr.getRank());
  for (int64_t d : mr.getShape())
    shapeAttrs.push_back(b.getI64IntegerAttr(d));
  fields.push_back(b.getNamedAttr("shape", ArrayAttr::get(ctx, shapeAttrs)));
  fields.push_back(b.getNamedAttr("elem_type", TypeAttr::get(mr.getElementType())));
  fields.push_back(b.getNamedAttr("rank", b.getI64IntegerAttr(mr.getRank())));

  // Layout map, if any (non-identity)
  if (!mr.getLayout().isIdentity()) {
    // Normalize to AffineMapAttr when possible.
    if (auto mapAttr = dyn_cast<AffineMapAttr>(mr.getLayout()))
      fields.push_back(b.getNamedAttr("layout_map", mapAttr));
    else
      fields.push_back(b.getNamedAttr("layout", mr.getLayout()));
  }

  // Memory space if present
  if (Attribute ms = mr.getMemorySpace())
    fields.push_back(b.getNamedAttr("memory_space", ms));

  return DictionaryAttr::get(ctx, fields);
}

// Main utility: attach original_layout under buffer.info if not already present.
// Returns true if it modified the IR.
static bool attachOriginalLayoutToBufferInfo(func::FuncOp top, unsigned argIdx) {
  MLIRContext *ctx = top.getContext();
  OpBuilder b(ctx);

  Type argTy = top.getArgument(argIdx).getType();
  auto mr = dyn_cast<MemRefType>(argTy);
  if (!mr)
    return false;

  // Get arg dict
  DictionaryAttr argDict = getFuncArgAttrsDict(top, argIdx);
  if (!argDict)
    argDict = DictionaryAttr::get(ctx);

  // Get buffer.info subdict (create if missing)
  DictionaryAttr bi = getOrCreateSubDict(b, argDict, "buffer.info");

  // If already has original_layout, do nothing
  if (bi.getAs<DictionaryAttr>("original_layout"))
    return false;

  // Build + attach
  DictionaryAttr orig = buildOriginalLayoutDict(b, mr);
  bi = setDictAttr(b, bi, "original_layout", orig);

  // Write back
  argDict = setSubDict(b, argDict, "buffer.info", bi);
  setFuncArgAttrsDict(top, argIdx, argDict);
  return true;
}

// Convenience: call this once at the start of rewriteMemoryModulesToDMA().
static void attachAllOriginalLayoutsToTop(func::FuncOp top) {
  for (unsigned i = 0; i < top.getNumArguments(); ++i) {
    (void)attachOriginalLayoutToBufferInfo(top, i);
  }
}

}

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static DictionaryAttr getFuncArgBufferInfo(func::FuncOp f, unsigned argIdx) {
  auto opt = f.getArgAttrs(); // std::optional<ArrayAttr> in your build
  if (!opt.has_value())
    return nullptr;

  ArrayAttr arr = opt.value();
  if (argIdx >= arr.size())
    return nullptr;

  auto dict = dyn_cast_or_null<DictionaryAttr>(arr[argIdx]);
  if (!dict)
    return nullptr;

  return dict.getAs<DictionaryAttr>("buffer.info");
}

static bool isOffChipROorWO(func::FuncOp top, unsigned argIdx, bool &isReadOnly) {
  DictionaryAttr bi = getFuncArgBufferInfo(top, argIdx);
  if (!bi)
    return false;

  auto typeAttr = bi.getAs<StringAttr>("type");
  auto ioAttr   = bi.getAs<StringAttr>("io");
  if (!typeAttr || !ioAttr)
    return false;

  if (typeAttr.getValue() != "off-chip" && typeAttr.getValue() != "weights")
    return false;

  if (ioAttr.getValue() == "read-only") {
    isReadOnly = true;
    return true;
  }
  if (ioAttr.getValue() == "write-only") {
    isReadOnly = false;
    return true;
  }
  return false;
}

static Type getStreamElementType(Type t) {
  auto st = dyn_cast<dataflow::StreamType>(t);
  if (!st) return Type();
  return st.getElementType(); // can be f32 OR vector<...>
}

static bool isReadModuleCandidate(func::FuncOp f, unsigned streamArgIdx,
                                  dataflow::StreamWriteOp &writeOpOut) {
  writeOpOut = nullptr;
  f.walk([&](dataflow::StreamWriteOp op) {
    if (op.getChannel() == f.getArgument(streamArgIdx))
      writeOpOut = op;
  });
  return (bool)writeOpOut;
}

static bool isWriteModuleCandidate(func::FuncOp f, unsigned streamArgIdx,
                                   dataflow::StreamReadOp &readOpOut) {
  readOpOut = nullptr;
  f.walk([&](dataflow::StreamReadOp op) {
    if (op.getChannel() == f.getArgument(streamArgIdx))
      readOpOut = op;
  });
  return (bool)readOpOut;
}

// Returns product of constant tripcounts of all enclosing affine.for's.
// Also returns the first loop_directive attribute found while walking up.
static int64_t productOfEnclosingConstantAffineFors(Operation *anchor,
                                                    SmallVectorImpl<AffineForOp> &loopsOut,
                                                    Attribute &loopDirectiveOut) {
  loopsOut.clear();
  loopDirectiveOut = Attribute();

  Operation *cur = anchor;
  while (cur) {
    if (auto forOp = dyn_cast<AffineForOp>(cur)) {
      loopsOut.push_back(forOp);
      if (!loopDirectiveOut && forOp->hasAttr("loop_directive"))
        loopDirectiveOut = forOp->getAttr("loop_directive");
    }
    cur = cur->getParentOp();
    if (!cur || isa<func::FuncOp>(cur))
      break;
  }

  if (loopsOut.empty())
    return -1;

  int64_t prod = 1;
  for (AffineForOp forOp : loopsOut) {
    int64_t lb = forOp.getConstantLowerBound();
    int64_t ub = forOp.getConstantUpperBound(); // int64_t in your build
    int64_t trip = ub - lb;
    if (trip <= 0)
      return -1;

    if (prod > (INT64_MAX / trip))
      return -1;
    prod *= trip;
  }
  return prod;
}


// Clears the entry block (except terminator) and replaces with linear DMA loop.
static LogicalResult rewriteReadModuleToLinearDMA(func::FuncOp f,
                                                  unsigned memrefArgIdx,
                                                  unsigned streamArgIdx,
                                                  int64_t N,
                                                  Type elemTy,
                                                  Attribute loopDirective) {
  return rewriteReadModuleToLinearDMA_MultiOut(f, memrefArgIdx, {streamArgIdx}, N, elemTy, loopDirective);
}

static LogicalResult rewriteWriteModuleToLinearDMA(func::FuncOp f,
                                                   unsigned memrefArgIdx,
                                                   unsigned streamArgIdx,
                                                   int64_t N,
                                                   Type elemTy,
                                                   Attribute loopDirective) {
  if (N <= 0)
    return failure();

  VectorSerialization ser = getVectorSerialization(elemTy);
  int64_t S = ser.numBeats;
  Type narrowTy = elemTy;

  auto vecTy = dyn_cast<VectorType>(elemTy);
  int64_t W = vecTy ? vecTy.getNumElements() : 1;

  if (vecTy && (S > 1 || ser.lanesPerBeat != W)) {
    narrowTy = VectorType::get({ser.lanesPerBeat}, vecTy.getElementType());
  }

  auto packedTy = makePackedMemRefType(f.getContext(), N * S, narrowTy);
  setFuncArgType(f, memrefArgIdx, packedTy);

  Block &entry = f.getBody().front();
  Operation *term = entry.getTerminator();

  SmallVector<Operation*> erase;
  for (Operation &op : llvm::make_early_inc_range(entry.getOperations())) {
    if (&op == term) continue;
    erase.push_back(&op);
  }
  for (Operation *op : erase) op->erase();

  OpBuilder b(&entry, entry.begin());
  Location loc = f.getLoc();

  auto forOp = b.create<AffineForOp>(loc, 0, N);
  if (loopDirective)
    forOp->setAttr("loop_directive", loopDirective);

  b.setInsertionPointToStart(forOp.getBody());
  Value iv = forOp.getInductionVar();
  Value packed = f.getArgument(memrefArgIdx);
  Value stream = f.getArgument(streamArgIdx);

  if (S == 1 && (ser.lanesPerBeat == W)) {
    Value v = b.create<dataflow::StreamReadOp>(loc, elemTy, stream).getResult();
    b.create<AffineStoreOp>(loc, v, packed, ValueRange{iv});
  } else {
    // Disassemble wide into narrow vectors (<=512b per stored beat)
    Value wideVec = b.create<dataflow::StreamReadOp>(loc, elemTy, stream).getResult();
    auto vecTy = cast<VectorType>(elemTy);
    int64_t lanesPerBeat = ser.lanesPerBeat;
    int64_t W = vecTy.getNumElements();

    auto innerFor = b.create<AffineForOp>(loc, 0, S);
    b.setInsertionPointToStart(innerFor.getBody());
    Value j = innerFor.getInductionVar();

    Value narrowVec = b.create<dataflow::EmptyVectorOp>(loc, narrowTy).getResult();

    auto identityMap = AffineMap::get(1, 0, b.getAffineDimExpr(0), f.getContext());
    auto identityMapAttr = AffineMapAttr::get(identityMap);

    auto gmap = AffineMap::get(
        2, 0,
        b.getAffineDimExpr(0) * lanesPerBeat + b.getAffineDimExpr(1),
        f.getContext());

    auto kFor = b.create<AffineForOp>(loc, 0, lanesPerBeat);
    b.setInsertionPointToStart(kFor.getBody());
    Value k = kFor.getInductionVar();
    Value gIdx = b.create<AffineApplyOp>(loc, gmap, ValueRange{j, k});

    emitPartialBeatGuardedXfer(b, loc, f.getContext(), W, lanesPerBeat, j, k,
                               [&](OpBuilder &bb) {
                                 Value valBit =
                                     bb
                                         .create<dataflow::VectorAffineLoadOp>(
                                             loc, vecTy.getElementType(),
                                             wideVec, ValueRange{gIdx},
                                             identityMapAttr)
                                         .getResult();
                                 bb.create<dataflow::VectorAffineStoreOp>(
                                     loc, narrowVec, valBit, ValueRange{k},
                                     identityMapAttr);
                               });

    // idx = iv * S + j
    b.setInsertionPointAfter(kFor);
    auto amap = AffineMap::get(2, 0, b.getAffineDimExpr(0) * S + b.getAffineDimExpr(1), f.getContext());
    Value idx = b.create<AffineApplyOp>(loc, amap, ValueRange{iv, j});
    b.create<AffineStoreOp>(loc, narrowVec, packed, ValueRange{idx});
  }

  return success();
}
} // namespace

namespace {
//===----------------------------------------------------------------------===//
// Host laytout transformation utils
//===----------------------------------------------------------------------===//

// static Value buildLinearIndexFromLoops(OpBuilder &b, Location loc,
//                                        ArrayRef<AffineForOp> loops) {
//   // loops are collected bottom-up in productOfEnclosingConstantAffineFors;
//   // we want outermost->innermost for linearization.
//   SmallVector<AffineForOp> ordered(loops.begin(), loops.end());
//   std::reverse(ordered.begin(), ordered.end());

//   Value idx = b.create<arith::ConstantIndexOp>(loc, 0);
//   for (AffineForOp forOp : ordered) {
//     int64_t lb = forOp.getConstantLowerBound();
//     int64_t ub = forOp.getConstantUpperBound();
//     int64_t trip = ub - lb;

//     Value iv = forOp.getInductionVar();
//     Value iv0 = iv;
//     if (lb != 0) {
//       Value lbv = b.create<arith::ConstantIndexOp>(loc, lb);
//       iv0 = b.create<arith::SubIOp>(loc, iv, lbv);
//     }

//     Value tripV = b.create<arith::ConstantIndexOp>(loc, trip);
//     idx = b.create<arith::MulIOp>(loc, idx, tripV);
//     idx = b.create<arith::AddIOp>(loc, idx, iv0);
//   }
//   return idx;
// }
static Value buildAffineLinearIndexFromLoops(OpBuilder &b, Location loc,
                                             ArrayRef<AffineForOp> loops) {
  // loops were collected bottom-up; linearization should be outermost->innermost.
  SmallVector<AffineForOp> ordered(loops.begin(), loops.end());
  std::reverse(ordered.begin(), ordered.end());

  // idx = 0 (affine constant)
  Value idx = b.create<affine::AffineApplyOp>(
      loc,
      AffineMap::get(/*dimCount=*/0, /*symCount=*/0,
                    b.getAffineConstantExpr(0),
                    b.getContext()),
      ValueRange{});

  for (AffineForOp forOp : ordered) {
    int64_t lb = forOp.getConstantLowerBound();
    int64_t ub = forOp.getConstantUpperBound();
    int64_t trip = ub - lb;

    Value iv = forOp.getInductionVar();

    // iv0 = iv - lb  (affine)
    Value iv0 = iv;
    if (lb != 0) {
      auto m = AffineMap::get(/*dimCount=*/1, /*symCount=*/0,
                              b.getAffineDimExpr(0) - b.getAffineConstantExpr(lb),
                              b.getContext());
      iv0 = b.create<affine::AffineApplyOp>(loc, m, ValueRange{iv});
    }

    // idx = idx * trip + iv0   (affine)
    // map: (d0, d1) -> d0 * trip + d1
    auto m = AffineMap::get(/*dimCount=*/2, /*symCount=*/0,
                            b.getAffineDimExpr(0) * trip + b.getAffineDimExpr(1),
                            b.getContext());
    idx = b.create<affine::AffineApplyOp>(loc, m, ValueRange{idx, iv0});
  }

  return idx;
}
// Prune a multi-output read module (host clone) down to a single output stream,
// and aggressively delete the dead "vector construction" subgraph for the
// removed outputs by:
//   1) For each StreamWriteOp to delete:
//        - find its written value
//        - if it is produced by dataflow.empty_vector, erase ALL users of that
//          empty_vector result (except the StreamWriteOp itself), then erase
//          the empty_vector op, then erase the StreamWriteOp.
//        - otherwise, just erase the StreamWriteOp.
//   2) Drop unused function arguments (rebuild signature + entry block).
//
// IMPORTANT: call this BEFORE you change any stream arg types, so getChannel()
// stays valid.
static void pruneMultiOutHostPackerToSingleOutAndCleanup(func::FuncOp f,
                                                        unsigned primaryStreamArgIdx) {
  Block &entry = f.getBody().front();
  Value primaryChan = entry.getArgument(primaryStreamArgIdx);

  // ---- Helper: drop unused func args by rebuilding entry block/signature.
  auto dropUnusedFuncArgs = [&](func::FuncOp fn) {
    Block &oldEntry = fn.getBody().front();

    llvm::SmallBitVector keep(fn.getNumArguments(), false);
    for (unsigned i = 0; i < fn.getNumArguments(); ++i)
      if (!oldEntry.getArgument(i).use_empty())
        keep.set(i);

    if (keep.all())
      return;

    SmallVector<Type> newInputs;
    newInputs.reserve(keep.count());
    SmallVector<unsigned> oldToNew(fn.getNumArguments(), (unsigned)-1);
    for (unsigned i = 0, j = 0; i < fn.getNumArguments(); ++i) {
      if (!keep.test(i))
        continue;
      newInputs.push_back(oldEntry.getArgument(i).getType());
      oldToNew[i] = j++;
    }

    auto oldFnTy = fn.getFunctionType();
    auto newFnTy = FunctionType::get(fn.getContext(), newInputs, oldFnTy.getResults());
    fn->setAttr(fn.getFunctionTypeAttrName(), TypeAttr::get(newFnTy));

    auto *newEntry = new Block();
    for (unsigned i = 0; i < fn.getNumArguments(); ++i) {
      if (!keep.test(i))
        continue;
      newEntry->addArgument(oldEntry.getArgument(i).getType(), fn.getLoc());
    }

    fn.getBody().getBlocks().insert(fn.getBody().begin(), newEntry);
    newEntry->getOperations().splice(newEntry->end(), oldEntry.getOperations());

    for (unsigned i = 0; i < fn.getNumArguments(); ++i) {
      if (!keep.test(i))
        continue;
      unsigned j = oldToNew[i];
      oldEntry.getArgument(i).replaceAllUsesWith(newEntry->getArgument(j));
    }

    oldEntry.erase();
  };

  // ---- Collect StreamWriteOps that are NOT writing to primaryChan.
  SmallVector<dataflow::StreamWriteOp> toDeleteWrites;
  f.walk([&](dataflow::StreamWriteOp op) {
    if (op.getChannel() != primaryChan)
      toDeleteWrites.push_back(op);
  });

  // ---- For each redundant stream_write:
  //      if it writes a value produced by dataflow.empty_vector, erase that
  //      empty_vector and all its users (except this stream_write), then erase stream_write.
  for (auto w : toDeleteWrites) {
    Value vecVal = w.getValue();

    Operation *def = vecVal.getDefiningOp();
    // Recognize dataflow.empty_vector without depending on the C++ op class name.
    bool isEmptyVector = def && def->getName().getStringRef() == "dataflow.empty_vector";

    if (isEmptyVector) {
      // Erase ALL users of the empty_vector result, except the stream_write we handle last.
      // (This deletes vector_store chains that mutate/fill the vector.)
      SmallVector<Operation *> users;
      for (Operation *u : vecVal.getUsers())
        if (u != w.getOperation())
          users.push_back(u);

      // Erase users first (they may include other writes/ops); order doesn't matter if
      // they don't depend on each other. If you do have dependencies, MLIR will assert;
      // in your pattern (vector_store then stream_write), this is safe.
      for (Operation *u : users)
        u->erase();

      // Now empty_vector should be unused except by `w` (which we erase next).
      // Erase empty_vector now or after erasing w; either is fine as long as uses are gone.
      // We'll erase after w to avoid any unexpected remaining use.
    }

    // Finally erase the redundant stream_write.
    w.erase();

    // If it was an empty_vector producer, erase it once its uses are gone.
    if (isEmptyVector) {
      if (def && def->use_empty())
        def->erase();
      // If not use_empty(), leave it; something else still uses it.
    }
  }

  // ---- Drop unused args (now that extra stream writes are removed).
  dropUnusedFuncArgs(f);

  MLIRContext *ctx = f.getContext();

  // Save required symbol attrs
  StringAttr symName = f.getSymNameAttr();
  StringAttr symVis  = f->getAttrOfType<StringAttr>("sym_visibility");

  // Drop ALL attributes
  f->setAttrs(DictionaryAttr::get(ctx));

  // Restore required attrs
  if (symName)
    f.setSymNameAttr(symName);
  if (symVis)
    f->setAttr("sym_visibility", symVis);

  // Restore function_type (mandatory)
  f->setAttr(f.getFunctionTypeAttrName(),
              TypeAttr::get(f.getFunctionType()));

  // Drop arg_attrs entirely (host doesn’t need them)
  f->removeAttr("arg_attrs");

}

static func::FuncOp cloneAsHostPackerRO(ModuleOp m,
                                        func::FuncOp src,
                                        unsigned memrefArgIdx,
                                        unsigned streamArgIdx,
                                        MemRefType packedTy,
                                        dataflow::StreamWriteOp writeOp,
                                        ArrayRef<AffineForOp> loopsAroundWrite,
                                        StringRef newName) {
  // Clone whole func
  auto cloned = cast<func::FuncOp>(src->clone());
  pruneMultiOutHostPackerToSingleOutAndCleanup(cloned, streamArgIdx);
  cloned.setSymName(newName);
  m.push_back(cloned);

  // New signature: (orig_memref, packed_memref) -> ()
  // Keep original memref arg type; replace stream arg with packed memref.
  auto oldFnTy = cloned.getFunctionType();
  SmallVector<Type> inputs(oldFnTy.getInputs().begin(), oldFnTy.getInputs().end());
  inputs[streamArgIdx] = packedTy;
  auto newFnTy = FunctionType::get(m.getContext(), inputs, oldFnTy.getResults());
  cloned->setAttr(cloned.getFunctionTypeAttrName(), TypeAttr::get(newFnTy));
  cloned.getBody().getArgument(streamArgIdx).setType(packedTy);

  // Find the corresponding StreamWriteOp in the cloned body:
  dataflow::StreamWriteOp clonedWrite = nullptr;
  cloned.walk([&](dataflow::StreamWriteOp op) { clonedWrite = op; });

  if (!clonedWrite)
    return cloned;

  // Build linear index at the write site using the enclosing loops of *the cloned op*.
  // Re-discover loops around clonedWrite to avoid mapping headaches.
  SmallVector<AffineForOp> loops;
  Attribute loopDirective;
  (void)productOfEnclosingConstantAffineFors(clonedWrite.getOperation(), loops, loopDirective);

  OpBuilder b(clonedWrite);
  Location loc = clonedWrite.getLoc();
  Value lin = buildAffineLinearIndexFromLoops(b, loc, loops);

  Value packed = cloned.getArgument(streamArgIdx);
  Value vec = clonedWrite.getValue();
  Type elemTy = vec.getType();
  VectorSerialization ser = getVectorSerialization(elemTy);
  int64_t S = ser.numBeats;

  auto vecTy = dyn_cast<VectorType>(elemTy);
  int64_t W = vecTy ? vecTy.getNumElements() : 1;

  if (S == 1 && (ser.lanesPerBeat == W)) {
    b.create<AffineStoreOp>(loc, vec, packed, ValueRange{lin});
  } else {
    auto vecTy = cast<VectorType>(elemTy);
    int64_t lanesPerBeat = ser.lanesPerBeat;
    int64_t W = vecTy.getNumElements();

    auto innerFor = b.create<AffineForOp>(loc, 0, S);
    b.setInsertionPointToStart(innerFor.getBody());
    Value j = innerFor.getInductionVar();

    Type narrowTy = packedTy.getElementType();
    Value narrowVec = b.create<dataflow::EmptyVectorOp>(loc, narrowTy).getResult();

    auto identityMap = AffineMap::get(1, 0, b.getAffineDimExpr(0), m.getContext());
    auto identityMapAttr = AffineMapAttr::get(identityMap);

    auto gmap = AffineMap::get(
        2, 0,
        b.getAffineDimExpr(0) * lanesPerBeat + b.getAffineDimExpr(1),
        m.getContext());

    auto kFor = b.create<AffineForOp>(loc, 0, lanesPerBeat);
    b.setInsertionPointToStart(kFor.getBody());
    Value k = kFor.getInductionVar();
    Value gIdx = b.create<AffineApplyOp>(loc, gmap, ValueRange{j, k});

    emitPartialBeatGuardedXfer(b, loc, m.getContext(), W, lanesPerBeat, j, k,
                               [&](OpBuilder &bb) {
                                 Value valBit =
                                     bb
                                         .create<dataflow::VectorAffineLoadOp>(
                                             loc, vecTy.getElementType(), vec,
                                             ValueRange{gIdx}, identityMapAttr)
                                         .getResult();
                                 bb.create<dataflow::VectorAffineStoreOp>(
                                     loc, narrowVec, valBit, ValueRange{k},
                                     identityMapAttr);
                               });

    b.setInsertionPointAfter(kFor);
    // idx = lin * S + j
    auto amap = AffineMap::get(2, 0, b.getAffineDimExpr(0) * S + b.getAffineDimExpr(1), m.getContext());
    Value idx = b.create<AffineApplyOp>(loc, amap, ValueRange{lin, j});
    b.create<AffineStoreOp>(loc, narrowVec, packed, ValueRange{idx});
  }
  clonedWrite.erase();

  return cloned;
}

static func::FuncOp cloneAsHostUnpackerWO(ModuleOp m,
                                          func::FuncOp src,
                                          unsigned memrefArgIdx,
                                          unsigned streamArgIdx,
                                          MemRefType packedTy,
                                          dataflow::StreamReadOp readOp,
                                          StringRef newName) {
  auto cloned = cast<func::FuncOp>(src->clone());
  cloned.setSymName(newName);
  m.push_back(cloned);

  // New signature: (packed_memref, orig_memref) -> ()
  // Replace stream arg type with packed memref.
  auto oldFnTy = cloned.getFunctionType();
  SmallVector<Type> inputs(oldFnTy.getInputs().begin(), oldFnTy.getInputs().end());
  inputs[streamArgIdx] = packedTy;
  auto newFnTy = FunctionType::get(m.getContext(), inputs, oldFnTy.getResults());
  cloned->setAttr(cloned.getFunctionTypeAttrName(), TypeAttr::get(newFnTy));
  cloned.getBody().getArgument(streamArgIdx).setType(packedTy);

  // Find the StreamReadOp in the cloned body
  dataflow::StreamReadOp clonedRead = nullptr;
  cloned.walk([&](dataflow::StreamReadOp op) { clonedRead = op; });
  if (!clonedRead)
    return cloned;

  // Compute linear index at the read site
  SmallVector<AffineForOp> loops;
  Attribute loopDirective;
  (void)productOfEnclosingConstantAffineFors(clonedRead.getOperation(), loops, loopDirective);

  OpBuilder b(clonedRead);
  Location loc = clonedRead.getLoc();
  Value lin = buildAffineLinearIndexFromLoops(b, loc, loops);

  Value packed = cloned.getArgument(streamArgIdx);
  Type elemTy = clonedRead.getResult().getType();
  VectorSerialization ser = getVectorSerialization(elemTy);
  int64_t S = ser.numBeats;

  auto vecTy = dyn_cast<VectorType>(elemTy);
  int64_t W = vecTy ? vecTy.getNumElements() : 1;

  if (S == 1 && (ser.lanesPerBeat == W)) {
    Value v = b.create<AffineLoadOp>(loc, packed, ValueRange{lin});
    clonedRead.getResult().replaceAllUsesWith(v);
  } else {
    auto vecTy = cast<VectorType>(elemTy);
    int64_t lanesPerBeat = ser.lanesPerBeat;
    int64_t W = vecTy.getNumElements();
    Value wideVec = b.create<dataflow::EmptyVectorOp>(loc, vecTy).getResult();

    auto innerFor = b.create<AffineForOp>(loc, 0, S);
    b.setInsertionPointToStart(innerFor.getBody());
    Value j = innerFor.getInductionVar();

    // idx = lin * S + j
    auto amap = AffineMap::get(2, 0, b.getAffineDimExpr(0) * S + b.getAffineDimExpr(1), m.getContext());
    Value idx = b.create<AffineApplyOp>(loc, amap, ValueRange{lin, j});
    Value narrowVec = b.create<AffineLoadOp>(loc, packed, ValueRange{idx}).getResult();

    auto identityMap = AffineMap::get(1, 0, b.getAffineDimExpr(0), m.getContext());
    auto identityMapAttr = AffineMapAttr::get(identityMap);

    auto gmap = AffineMap::get(
        2, 0,
        b.getAffineDimExpr(0) * lanesPerBeat + b.getAffineDimExpr(1),
        m.getContext());

    auto kFor = b.create<AffineForOp>(loc, 0, lanesPerBeat);
    b.setInsertionPointToStart(kFor.getBody());
    Value k = kFor.getInductionVar();
    Value gIdx = b.create<AffineApplyOp>(loc, gmap, ValueRange{j, k});

    emitPartialBeatGuardedXfer(b, loc, m.getContext(), W, lanesPerBeat, j, k,
                               [&](OpBuilder &bb) {
                                 Value valBit =
                                     bb
                                         .create<dataflow::VectorAffineLoadOp>(
                                             loc, vecTy.getElementType(),
                                             narrowVec, ValueRange{k},
                                             identityMapAttr)
                                         .getResult();
                                 bb.create<dataflow::VectorAffineStoreOp>(
                                     loc, wideVec, valBit, ValueRange{gIdx},
                                     identityMapAttr);
                               });

    clonedRead.getResult().replaceAllUsesWith(wideVec);
  }
  clonedRead.erase();

  return cloned;
}

} // namespace

std::pair<ModuleOp, ModuleOp> streamhls::rewriteMemoryModulesToDMA(ModuleOp &m) {
  if (!m)
    return {m, ModuleOp()};

  auto top = m.lookupSymbol<func::FuncOp>("module_top");
  if (!top)
    return {m, ModuleOp()};

  // Capture original memref layouts before DMA port rewriting.
  attachAllOriginalLayoutsToTop(top);
    
  LLVM_DEBUG(llvm::dbgs() << "[DMA] Found module_top with " << top.getNumArguments()
               << " args\n");
  auto host = ModuleOp::create(m.getLoc());

  for (unsigned argIdx = 0; argIdx < top.getNumArguments(); ++argIdx) {
    bool isRO = false;
    if (!isOffChipROorWO(top, argIdx, isRO))
      continue;

    auto bi = getFuncArgBufferInfo(top, argIdx);
    auto nameAttr = bi ? bi.getAs<StringAttr>("name") : StringAttr();
    auto ioAttr   = bi ? bi.getAs<StringAttr>("io")   : StringAttr();
    LLVM_DEBUG(llvm::dbgs() << "[DMA] Candidate top arg #" << argIdx
                 << " name=" << (nameAttr ? nameAttr.getValue() : "<none>")
                 << " io=" << (ioAttr ? ioAttr.getValue() : "<none>")
                 << " (RO=" << (isRO ? "true" : "false") << ")\n");

    BlockArgument topArg = top.getArgument(argIdx);

    // Collect ALL calls that use this top argument
    SmallVector<std::pair<func::CallOp, unsigned>> targetCalls;
    top.walk([&](func::CallOp call) {
      LLVM_DEBUG(llvm::dbgs() << "[DMA]  Checking call to @" << call.getCallee() << " with " << call.getNumOperands() << " operands\n");
      for (auto [i, opnd] : llvm::enumerate(call.getOperands())) {
        LLVM_DEBUG(llvm::dbgs() << "[DMA]    Operand " << i << ": " << opnd << " (topArg=" << topArg << ")\n");
        if (opnd == topArg) {
          LLVM_DEBUG(llvm::dbgs() << "[DMA]    MATCH! Adding call to @" << call.getCallee() << " operandPos=" << i << "\n");
          targetCalls.push_back({call, (unsigned)i});
        }
      }
    });

    if (targetCalls.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "[DMA]  No direct call use found for top arg #" << argIdx
                   << "\n");
      continue;
    }

    LLVM_DEBUG(llvm::dbgs() << "[DMA]  Found " << targetCalls.size() << " calls using top arg #" << argIdx << "\n");

    Attribute loopDirective;
    int64_t N = -1;
    MemRefType packedTy;
    std::string portName = (nameAttr ? nameAttr.getValue().str() : ("arg" + std::to_string(argIdx)));
    bool transformedAny = false;

    // Process all calls that use this argument
    for (auto [targetCall, operandPos] : targetCalls) {
      LLVM_DEBUG(llvm::dbgs() << "[DMA]  Processing call to @" << targetCall.getCallee()
                   << " operandPos=" << operandPos << "\n");

      auto callee = m.lookupSymbol<func::FuncOp>(targetCall.getCallee());
      if (!callee) {
        LLVM_DEBUG(llvm::dbgs() << "[DMA]  Could not resolve callee @" << targetCall.getCallee()
                     << "\n");
        continue;
      }

      LLVM_DEBUG(llvm::dbgs() << "[DMA]  Callee resolved: @" << callee.getSymName()
                   << " (#args=" << callee.getNumArguments() << ")\n");

      if (callee.getNumArguments() < 2) {
        LLVM_DEBUG(llvm::dbgs() << "[DMA]  Skip @" << callee.getSymName()
                     << " (expected >=2 args)\n");
        continue;
      }

      unsigned memrefArgIdx = operandPos;
      unsigned streamArgIdx = (memrefArgIdx == 0) ? 1 : 0;

      LLVM_DEBUG(llvm::dbgs() << "[DMA]  Indices: memrefArgIdx=" << memrefArgIdx
                   << " streamArgIdx=" << streamArgIdx << "\n");

      Type elemTy = getStreamElementType(callee.getArgument(streamArgIdx).getType());
        if (!elemTy) {
        LLVM_DEBUG(llvm::dbgs() << "[DMA]  Skip @" << callee.getSymName()
                     << " (stream arg is not dataflow.stream<...>)\n");
        continue;
      }

      LLVM_DEBUG(llvm::dbgs() << "[DMA]  Stream element type: " << elemTy << "\n");
      if (isRO) {
        // Collect ALL output streams (args that are written via stream_write).
        SmallVector<unsigned> outStreamArgIdxs;
        SmallVector<dataflow::StreamWriteOp> outWriteOps;
        if (!collectStreamWriteOutputs(callee, outStreamArgIdxs, outWriteOps)) {
          LLVM_DEBUG(llvm::dbgs() << "[DMA]  Skip @" << callee.getSymName()
                      << " (no StreamWriteOp writing to any stream arg)\n");
          continue;
        }

        // Verify all output streams have the same vector element type.
        Type commonElemTy;
        for (unsigned sIdx : outStreamArgIdxs) {
          Type elemTy = getStreamElementType(callee.getArgument(sIdx).getType());
          if (!elemTy) {
            LLVM_DEBUG(llvm::dbgs() << "[DMA]  Skip @" << callee.getSymName()
                        << " (stream arg #" << sIdx << " is not stream<...>)\n");
            commonElemTy = Type();
            break;
          }
          if (!commonElemTy)
            commonElemTy = elemTy;
          else if (commonElemTy != elemTy) {
            LLVM_DEBUG(llvm::dbgs() << "[DMA]  Skip @" << callee.getSymName()
                        << " (mismatched vector types across outputs: "
                        << commonElemTy << " vs " << elemTy << ")\n");
            commonElemTy = Type();
            break;
          }
        }
        if (!commonElemTy)
          continue;

        // Compute N from the first write op, and require all writes agree.
        Attribute localLoopDirective;
        SmallVector<AffineForOp> loops0;
        int64_t localN0 =
            productOfEnclosingConstantAffineFors(outWriteOps.front().getOperation(),
                                                loops0, localLoopDirective);
        if (localN0 <= 0) {
          LLVM_DEBUG(llvm::dbgs() << "[DMA]  Skip @" << callee.getSymName()
                      << " (failed to compute constant N)\n");
          continue;
        }

        for (auto w : outWriteOps) {
          SmallVector<AffineForOp> loopsTmp;
          Attribute tmpDirective;
          int64_t nTmp =
              productOfEnclosingConstantAffineFors(w.getOperation(), loopsTmp, tmpDirective);
          if (nTmp != localN0) {
            LLVM_DEBUG(llvm::dbgs() << "[DMA]  Skip @" << callee.getSymName()
                        << " (multi-out writes have different N: " << localN0
                        << " vs " << nTmp << ")\n");
            localN0 = -1;
            break;
          }
        }
        if (localN0 <= 0)
          continue;

        // Use first N we compute, or verify they're all the same across different call sites.
        if (N < 0) {
          N = localN0;
          elemTy = commonElemTy;
          VectorSerialization serPack = getVectorSerialization(elemTy);
          int64_t S = serPack.numBeats;
          Type narrowTy = elemTy;

          auto vecTy = dyn_cast<VectorType>(elemTy);
          int64_t W = vecTy ? vecTy.getNumElements() : 1;

          if (vecTy && (S > 1 || serPack.lanesPerBeat != W)) {
            narrowTy =
                VectorType::get({serPack.lanesPerBeat}, vecTy.getElementType());
          }
          packedTy = makePackedMemRefType(m.getContext(), N * S, narrowTy);
          loopDirective = localLoopDirective;
          LLVM_DEBUG(llvm::dbgs() << "[DMA]  Computed N=" << N << " S=" << S
                      << " packedTy=" << packedTy
                      << " (#outStreams=" << outStreamArgIdxs.size() << ")\n");
        } else if (N != localN0) {
          LLVM_DEBUG(llvm::dbgs() << "[DMA]  Warning: @" << callee.getSymName()
                      << " has different N=" << localN0 << " (expected " << N << ")\n");
          continue;
        } else {
          // Keep elemTy/packedTy from earlier.
        }

        // Optional host packer (kept as-is: you can improve later)
        if (!transformedAny) {
          std::string fnName = "__host_pack_" + portName;
          // NOTE: existing cloneAsHostPackerRO assumes one stream arg; keep using the first.
          (void)cloneAsHostPackerRO(host, callee, memrefArgIdx, outStreamArgIdxs.front(),
                                    packedTy, outWriteOps.front(), loops0, fnName);
        }

        if (failed(rewriteReadModuleToLinearDMA_MultiOut(callee, memrefArgIdx,
                                                        outStreamArgIdxs,
                                                        N, elemTy, loopDirective))) {
          LLVM_DEBUG(llvm::dbgs() << "[DMA]  Rewrite failed for @" << callee.getSymName()
                      << "\n");
          continue;
        }

        transformedAny = true;
      } else {
        dataflow::StreamReadOp readOp;
        if (!isWriteModuleCandidate(callee, streamArgIdx, readOp)) {
          LLVM_DEBUG(llvm::dbgs() << "[DMA]  Skip @" << callee.getSymName()
                       << " (no StreamReadOp using channel arg)\n");
          continue;
        }

        LLVM_DEBUG(llvm::dbgs() << "[DMA]  Matched WRITE module @" << callee.getSymName()
                     << "\n");

        SmallVector<AffineForOp> loops;
        int64_t localN = productOfEnclosingConstantAffineFors(readOp.getOperation(), loops,
                                                 loopDirective);
        if (localN <= 0) {
          LLVM_DEBUG(llvm::dbgs() << "[DMA]  Skip @" << callee.getSymName()
                       << " (failed to compute constant N)\n");
          continue;
        }

        // Use the first N we compute, or verify they're all the same
        if (N < 0) {
          N = localN;
          VectorSerialization serUnpack = getVectorSerialization(elemTy);
          int64_t S = serUnpack.numBeats;
          Type narrowTy = elemTy;

          auto vecTy = dyn_cast<VectorType>(elemTy);
          int64_t W = vecTy ? vecTy.getNumElements() : 1;

          if (vecTy && (S > 1 || serUnpack.lanesPerBeat != W)) {
            narrowTy =
                VectorType::get({serUnpack.lanesPerBeat}, vecTy.getElementType());
          }
          packedTy = makePackedMemRefType(m.getContext(), N * S, narrowTy);
          LLVM_DEBUG(llvm::dbgs() << "[DMA]  Computed N=" << N << " S=" << S
                       << " packedTy=" << packedTy << "\n");
        } else if (N != localN) {
          LLVM_DEBUG(llvm::dbgs() << "[DMA]  Warning: @" << callee.getSymName()
                       << " has different N=" << localN << " (expected " << N << ")\n");
          continue;
        }

        LLVM_DEBUG(llvm::dbgs() << "[DMA]  Rewriting WRITE module @" << callee.getSymName()
                     << " -> linear DMA loop\n");

        // Create host unpacker function (only for first call)
        if (!transformedAny) {
          std::string fnName = "__host_unpack_" + portName;
          (void)cloneAsHostUnpackerWO(host, callee, memrefArgIdx, streamArgIdx,
                                      packedTy, readOp, fnName);
        }

        if (failed(rewriteWriteModuleToLinearDMA(callee, memrefArgIdx, streamArgIdx,
                                                 N, elemTy, loopDirective))) {
          LLVM_DEBUG(llvm::dbgs() << "[DMA]  Rewrite failed for @" << callee.getSymName()
                       << "\n");
          continue;
        }

        transformedAny = true;
      }
    }

    if (transformedAny && N > 0) {
      updateTopArgType(top, argIdx, packedTy);
      LLVM_DEBUG(llvm::dbgs() << "[DMA]  Updated module_top arg #" << argIdx
                   << " to " << packedTy << "\n");
    }
    LLVM_DEBUG(llvm::dbgs() << "[DMA]  Done processing top arg #" << argIdx << "\n\n");
  }
  simplifyAffineOps(host);
  return {m, host};
}

