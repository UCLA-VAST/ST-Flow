//===- ConvertMemrefArgsToDataflow.cpp -------------------------*- C++ -*-===//
// Replace func memref arguments with dataflow.stream and single uses with
// dataflow.stream_{read,write}, while extracting access patterns.
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/IntegerSet.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include "streamhls/Dialect/Dataflow/Transforms/Passes.h"
#include "streamhls/Dialect/Dataflow/Transforms/ConverterInsertion.h"
#include "streamhls/Dialect/Dataflow/Support/Utils.h"

using namespace mlir;
using namespace streamhls;
using namespace dataflow;

#define DEBUG_TYPE "streamhls-fifo-conversion"

namespace {

mlir::LogicalResult convertMemrefArgsToFIFOsImpl(mlir::func::FuncOp func, AffineLoopBand &band, const DenseSet<unsigned> &argsToConvert){
  auto &entry = func.getBody().front();

  // Collect candidate arg indices first (memref only), excluding off-chip/weights.
  SmallVector<unsigned> memrefArgIdxs;
  for (unsigned idx : argsToConvert) {
    if (idx < entry.getNumArguments() && isa<MemRefType>(entry.getArgument(idx).getType())) {
      memrefArgIdxs.push_back(idx);
    }
  }
  if (memrefArgIdxs.empty()) return success();

  auto fType = func.getFunctionType();
  SmallVector<Type> newInputs(fType.getInputs().begin(), fType.getInputs().end());

  SmallVector<DictionaryAttr> newArgDicts(func.getNumArguments());
  for (unsigned i = 0, e = func.getNumArguments(); i < e; ++i)
    newArgDicts[i] = func.getArgAttrDict(i);

  struct Plan {
    unsigned idx;
    Operation *use;    // the unique affine.load OR affine.store
    bool isLoad;
    bool useArrayOfStreams;
  };
  SmallVector<Plan> plans;

  MLIRContext *ctx = func.getContext();

  // --------------------------------------------------------------------------
  // First pass: decide per-argument replacement type (stream vs array-of-streams)
  // --------------------------------------------------------------------------
  for (unsigned idx : memrefArgIdxs) {
    BlockArgument arg = entry.getArgument(idx);

    SmallVector<Operation*> matchingUses;
    SmallVector<Operation*> unsupportedUses;
    for (Operation *user : arg.getUsers()) {
      if (auto ld = dyn_cast<affine::AffineLoadOp>(user)) {
        if (ld.getMemRef() == arg) matchingUses.push_back(user);
      } else if (auto st = dyn_cast<affine::AffineStoreOp>(user)) {
        if (st.getMemRef() == arg) matchingUses.push_back(user);
      } else {
        unsupportedUses.push_back(user);
      }
    }

    if (!unsupportedUses.empty()) {
      llvm::errs() << "\n[ConvertFuncMemrefsToFIFOs] ERROR: memref argument #" << idx 
                   << " has " << unsupportedUses.size() << " unsupported user(s):\n";
      for (Operation *user : unsupportedUses) {
        llvm::errs() << "  - Operation: " << user->getName() << "\n";
        llvm::errs() << "    Location: ";
        user->getLoc().print(llvm::errs());
        llvm::errs() << "\n";
        llvm::errs() << "    Operation dump:\n";
        user->print(llvm::errs());
        llvm::errs() << "\n";
      }
      llvm::errs() << "  Supported operations: affine.load, affine.store\n";
      func.emitError() << "memref argument #" << idx
                       << " has unsupported user: " << unsupportedUses.front()->getName();
      return failure();
    }

    if (matchingUses.empty()) continue;

    for (Operation *u : matchingUses) {
      bool isLoad = isa<affine::AffineLoadOp>(u);
      auto mty = cast<MemRefType>(arg.getType());
      Type elemTy = mty.getElementType();
      bool useArrayOfStreams = false;

      // Check for fifo-info on the load/store.
      if (auto fifoInfo = u->getAttrOfType<DictionaryAttr>("fifo-info")) {
        auto shapeAttr = fifoInfo.getAs<ArrayAttr>("shape");
        if (!shapeAttr) {
          func.emitError() << "fifo-info on op does not contain 'shape' ArrayAttr";
          return failure();
        }

        SmallVector<int64_t, 4> fifoShape;
        fifoShape.reserve(shapeAttr.size());
        for (Attribute a : shapeAttr) {
          auto ia = mlir::dyn_cast<IntegerAttr>(a);
          if (!ia) {
            func.emitError() << "fifo-info.shape must be an array of IntegerAttr";
            return failure();
          }
          fifoShape.push_back(ia.getInt());
        }

        bool allOnes = llvm::all_of(fifoShape, [](int64_t v) { return v == 1; });
        auto streamTy = dataflow::StreamType::get(ctx, elemTy);
        auto arrayOfStreamsTy = RankedTensorType::get(fifoShape, streamTy);

        if (!allOnes){
          newInputs[idx] = arrayOfStreamsTy;
          useArrayOfStreams = true;
        }else{
          newInputs[idx] = streamTy;
        }
      } else {
        auto streamTy = dataflow::StreamType::get(ctx, elemTy);
        newInputs[idx] = streamTy;
      }
      LLVM_DEBUG(llvm::dbgs() << "Converting arg #" << idx << " to " << newInputs[idx] << "\n");
      plans.push_back(Plan{idx, u, isLoad, useArrayOfStreams});
    }
  }

  // Update function type and argument types.
  auto newFuncType =
      FunctionType::get(func.getContext(), newInputs, fType.getResults());

  func.setType(newFuncType);

  for (auto &p : plans) {
    BlockArgument arg = entry.getArgument(p.idx);
    arg.setType(newInputs[p.idx]);
    
    DictionaryAttr dict = newArgDicts[p.idx];
    if (dict) {
      if (auto ioAttr = dict.getAs<StringAttr>("io")) {
        if (ioAttr.getValue() == "unknown") {
          NamedAttrList attrs(dict);
          attrs.set("io", StringAttr::get(ctx, p.isLoad ? "read-only" : "write-only"));
          dict = DictionaryAttr::get(ctx, attrs);
        }
      }
    }
    func.setArgAttrs(p.idx, dict ? dict : DictionaryAttr::get(ctx));
  }

  // --------------------------------------------------------------------------
  // Second pass: rewrite the unique uses.
  //  * For ArrayOfStreams, reconstruct IVs from fifo-info.indices (reverse of setFifoAttr)
  //    and build an identity affine map of rank = #indices.
  // --------------------------------------------------------------------------
  OpBuilder builder(func);

  for (auto &p : plans) {
    BlockArgument chanArg = entry.getArgument(p.idx);
    builder.setInsertionPoint(p.use);

    if (p.useArrayOfStreams) {
      // Array-of-streams case: use ArrayOfStreamsReadOp / ArrayOfStreamsWriteOp.
     auto arrayTy = mlir::cast<RankedTensorType>(chanArg.getType());
     auto streamTy = mlir::cast<dataflow::StreamType>(arrayTy.getElementType());
      Type elemTy = streamTy.getElementType();

      // Get fifo-info again.
      auto fifoInfo = p.use->getAttrOfType<DictionaryAttr>("fifo-info");
      assert(fifoInfo && "Expected fifo-info for array-of-streams plan");

      // indices : ArrayAttr of i64, each is a position into the reversed loop band.
      auto indicesAttr = fifoInfo.getAs<ArrayAttr>("indices");
      if (!indicesAttr) {
        func.emitError() << "fifo-info on op does not contain 'indices' ArrayAttr";
        return failure();
      }

      // Rebuild the same loop band used in setFifoAttr.
      auto innermostLoop = p.use->getParentOfType<affine::AffineForOp>();
      assert(innermostLoop && "Expected op inside an AffineForOp for fifo-info");

      // get loop indices from fifo-info.indices
      SmallVector<unsigned, 4> loopIndices;
      for(Attribute a : indicesAttr) {
        auto ia = mlir::dyn_cast<IntegerAttr>(a);
        assert(ia && "fifo-info.indices must be an array of IntegerAttr");
        loopIndices.push_back(static_cast<unsigned>(ia.getInt()));
      }
      SmallVector<Value, 4> indexVals;
      indexVals.reserve(indicesAttr.size());
      for (unsigned loopIdx : loopIndices) {
        assert(loopIdx < band.size() && "fifo-info.indices has invalid loop index");
        indexVals.push_back(band[loopIdx].getInductionVar());
      }

      // Identity affine map: (d0, ..., d_{n-1}) -> (d0, ..., d_{n-1}),
      // where n = indexVals.size().
      unsigned nDims = indexVals.size();
      SmallVector<AffineExpr, 4> exprs;
      exprs.reserve(nDims);
      for (unsigned i = 0; i < nDims; ++i)
        exprs.push_back(getAffineDimExpr(i, ctx));
      auto map = AffineMap::get(/*dimCount=*/nDims, /*symbolCount=*/0, exprs, ctx);
      auto mapAttr = AffineMapAttr::get(map);

      if (p.isLoad) {
        auto read = builder.create<dataflow::ArrayOfStreamsReadOp>(
            p.use->getLoc(), elemTy, chanArg,
            /*indices=*/ValueRange(indexVals), mapAttr);
        read->setAttrs(p.use->getAttrDictionary());
        read->setAttr("map", mapAttr);
        p.use->getResult(0).replaceAllUsesWith(read.getResult());
        p.use->erase();
      } else {
        auto store = cast<affine::AffineStoreOp>(p.use);
        Value val = store.getValue();
        auto write = builder.create<dataflow::ArrayOfStreamsWriteOp>(
            p.use->getLoc(), chanArg, val,
            /*indices=*/ValueRange(indexVals), mapAttr);
        write->setAttrs(p.use->getAttrDictionary());
        write->setAttr("map", mapAttr);
        p.use->erase();
      }
    } else {
      // Single-stream case: original behavior.
      auto streamTy = mlir::cast<dataflow::StreamType>(chanArg.getType());
      Type elemTy = streamTy.getElementType();

      if (p.isLoad) {
        auto read = builder.create<dataflow::StreamReadOp>(
            p.use->getLoc(), elemTy, chanArg);
        read->setAttrs(p.use->getAttrDictionary());
        p.use->getResult(0).replaceAllUsesWith(read.getResult());
        p.use->erase();
      } else {
        auto store = cast<affine::AffineStoreOp>(p.use);
        Value val = store.getValue();
        auto write = builder.create<dataflow::StreamWriteOp>(
            p.use->getLoc(), chanArg, val);
        write->setAttrs(p.use->getAttrDictionary());
        p.use->erase();
      }
    }
  }

  return success();
}

AffineForOp getFirstAffineForInFunc(func::FuncOp func) {
  AffineForOp firstFor; // default-constructed = null

  func.walk([&](AffineForOp forOp) {
    if (!firstFor)
      firstFor = forOp;
  });

  return firstFor; // check with: if (!firstFor) { ... }
}

static uint64_t calculateEdgeDepth(DFG &dfg, DFG::Edge &edge, unsigned srcId, unsigned dstId) {
  auto *srcNode = dfg.getNode(srcId);
  auto *dstNode = dfg.getNode(dstId);
  if (!srcNode || !dstNode) return 4096;

  // Rule 5: self-looping edges
  bool isSelfLoop = (srcId == dstId);
  
  // Rule 6: Input to converter node
  bool isInputToConverter = (dstNode->kind == DFG::NodeKind::CONVERTER);
  bool isDstCompute = (dstNode->kind == DFG::NodeKind::COMPUTE);
  // Rule 7: output edge of converter node makes it act as non-reduction node
  bool effectiveSrcIsReduction = (srcNode->kind == DFG::NodeKind::CONVERTER) ? false : srcNode->isReduction;
  bool effectiveDstIsReduction = dstNode->isReduction;

  auto getProduct = [](ArrayRef<int64_t> shape) {
    int64_t p = 1;
    for (auto s : shape) p *= s;
    return (uint64_t)p;
  };

  uint64_t ratio = 1;
  if (auto mt = edge.value.getType().dyn_cast<MemRefType>()) {
    if (auto rtt = edge.resolvedType.dyn_cast<RankedTensorType>()) {
      auto memShape = mt.getShape();
      auto aosShape = rtt.getShape();
      for (unsigned i = 0; i < std::min(memShape.size(), aosShape.size()); ++i) {
        ratio *= (uint64_t)std::ceil(float(memShape[i]) / float(aosShape[i]));
      }
    } else {
      ratio = getProduct(mt.getShape());
    }
  }

  uint64_t depth = std::min(ratio, (uint64_t)64);
  // add 5% safety for ration
  // ratio = ratio + 0.05*ratio;
  if (isSelfLoop) depth = ratio;
  else if (isDstCompute) depth = ratio;
  else if (effectiveSrcIsReduction && effectiveDstIsReduction) depth = ratio; //std::min(ratio, (uint64_t)1024);
  else if (!effectiveSrcIsReduction && effectiveDstIsReduction) depth = ratio;
  
  LLVM_DEBUG(llvm::dbgs() << "calculateEdgeDepth: src=" << srcId << " dst=" << dstId 
                          << " isReduction=" << effectiveSrcIsReduction << "->" << effectiveDstIsReduction 
                          << " isSelfLoop=" << isSelfLoop << " isConverter=" << isInputToConverter 
                          << " ratio=" << ratio << " finalDepth=" << depth << "\n");

  return depth;
}

} // end anonymous namespace

namespace mlir {
namespace streamhls {
ModuleOp convertMemrefArgsToFIFOs(ModuleOp &module, const DenseSet<unsigned> &argsToConvert){
  // Iterate over all functions in the module.
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    auto firstFor = getFirstAffineForInFunc(func);
    if(!firstFor){
      continue; 
    }
    AffineLoopBand band;
    getLoopBandFromInnermost(firstFor, band);
    if (failed(convertMemrefArgsToFIFOsImpl(func, band, argsToConvert))) {
      assert(false && "Failed to convert memref args to dataflow streams");
    }
  }
  return module;
}

void mlir::streamhls::convertFuncMemrefsToFIFOsDFG(DFG &dfg) {
  resolveDFGEdges(dfg);

  for (auto &nodePair : dfg.nodes) {
    auto &node = nodePair.second;
    if (!node.hwModule) continue;

    auto funcOps = node.hwModule.getOps<func::FuncOp>();
    if (funcOps.begin() == funcOps.end()) continue;
    func::FuncOp func = *funcOps.begin();

    DenseSet<unsigned> argsToConvert;
    
    auto processEdge = [&](DFG::Edge &edge, StringRef io) {
      if (edge.kind == DFG::EdgeKind::STREAM || edge.kind == DFG::EdgeKind::AOS) {
        std::string bufferName = getMemrefBufferName(edge.value);
        if (!bufferName.empty()) {
          StringAttr bufferNameAttr = StringAttr::get(node.hwModule.getContext(), bufferName);
          int idx = getArgIdxByBufferName(node.hwModule, func, bufferNameAttr, io);
          if (idx != -1) {
            LLVM_DEBUG(llvm::dbgs() << "Node " << node.id << ": marking arg #" << idx << " (" << bufferName << ", " << io << ") for conversion\n");
            argsToConvert.insert((unsigned)idx);
          }
        }
      }
    };

    if (dfg.outEdges.count(node.id)) {
      for (auto &edge : dfg.outEdges[node.id])
        processEdge(edge, "write-only");
    }
    if (dfg.inEdges.count(node.id)) {
      for (auto &edge : dfg.inEdges[node.id])
        processEdge(edge, "read-only");
    }

    // Always create a new fifoModule to maintain the expected pipeline state
    node.fifoModule = convertMemrefArgsToFIFOs(node.hwModule, argsToConvert);
    node.hwModule = node.fifoModule;

    // Update the resolvedType of the edges
    node.fifoModule.walk([&](func::FuncOp func) {
      auto updateEdgeType = [&](DFG::Edge &edge, StringRef io) {
        std::string bufferName = getMemrefBufferName(edge.value);
        if (!bufferName.empty()) {
          StringAttr bufferNameAttr = StringAttr::get(node.fifoModule.getContext(), bufferName);
          int idx = getArgIdxByBufferName(node.fifoModule, func, bufferNameAttr, io);
          if (idx != -1) {
            edge.resolvedType = func.getArgument(idx).getType();
            if (isa<RankedTensorType>(edge.resolvedType))
              edge.kind = DFG::EdgeKind::AOS;
            else if (isa<dataflow::StreamType>(edge.resolvedType))
              edge.kind = DFG::EdgeKind::STREAM;
            LLVM_DEBUG(llvm::dbgs() << "Edge " << edge.id << ": updated type to " << edge.resolvedType << " (io=" << io << ")\n");
          }
        }
      };

      if (dfg.outEdges.count(node.id)) {
        for (auto &edge : dfg.outEdges[node.id]) {
          updateEdgeType(edge, "write-only");
          edge.depth = calculateEdgeDepth(dfg, edge, node.id, edge.id);
        }
      }
      if (dfg.inEdges.count(node.id)) {
        for (auto &edge : dfg.inEdges[node.id]) {
          updateEdgeType(edge, "read-only");
          edge.depth = calculateEdgeDepth(dfg, edge, edge.id, node.id);
        }
      }
    });
  }
}

} // namespace streamhls
} // namespace mlir
