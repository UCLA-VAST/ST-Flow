//===- ConverterInsertion.cpp ------------------------------------*- C++ -*-===//
// Logic for creating and inserting converter nodes in the DFG.
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
#include "streamhls/Dialect/Dataflow/IR/Dataflow.h"

using namespace mlir;
using namespace streamhls;
using namespace dataflow;

#define DEBUG_TYPE "streamhls-fifo-conversion"

namespace mlir {
namespace streamhls {

int getArgIdxByBufferName(ModuleOp module, func::FuncOp func, StringAttr bufferName, StringRef io) {
  auto checkArg = [&](unsigned i, DictionaryAttr bufInfo) -> bool {
    if (bufInfo.getAs<StringAttr>("name") != bufferName) return false;
    StringRef portIO = bufInfo.getAs<StringAttr>("io").getValue();
    if (io.empty() || portIO == io) return true;
    return false;
  };

  // 1. Check direct name on arguments (handles non-merged or first variant)
  for (unsigned i = 0; i < func.getNumArguments(); ++i) {
    if (auto bufInfo = func.getArgAttrOfType<DictionaryAttr>(i, "buffer.info")) {
      if (checkArg(i, bufInfo)) return i;
    }
  }

  // 2. Check merge.variants for other variants
  if (auto variants = module->getAttrOfType<ArrayAttr>("merge.variants")) {
    for (auto variantAttr : variants) {
      auto variant = mlir::cast<DictionaryAttr>(variantAttr);
      auto bufferInfoList = variant.getAs<ArrayAttr>("buffer_info_list");
      if (bufferInfoList) {
        for (unsigned i = 0; i < bufferInfoList.size(); ++i) {
          auto bufInfoWrapper = mlir::cast<DictionaryAttr>(bufferInfoList[i]);
          auto bufInfo = bufInfoWrapper.getAs<DictionaryAttr>("buffer.info");
          if (bufInfo && checkArg(i, bufInfo)) return (int)i;
        }
      }
    }
  }
  return -1;
}

namespace {

static bool isRealAOS(DictionaryAttr fifoInfo) {
  if (!fifoInfo) return false;
  auto shape = fifoInfo.getAs<ArrayAttr>("shape");
  if (!shape) return false;
  for (auto s : shape) {
    if (mlir::cast<IntegerAttr>(s).getInt() > 1) return true;
  }
  return false;
}



static ModuleOp createConverterModule(MLIRContext *ctx, StringRef name,
                                      AccessInfoAttr srcInfo, AccessInfoAttr dstInfo,
                                      DictionaryAttr srcFifoInfo, DictionaryAttr dstFifoInfo,
                                      Type elementType, StringAttr bufferName) {
  OpBuilder builder(ctx);
  auto module = ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module.getBody());

  // Determine stream types
  Type inStreamType;
  if (isRealAOS(srcFifoInfo)) {
    auto shape = srcFifoInfo.getAs<ArrayAttr>("shape");
    SmallVector<int64_t> shapeVec;
    for (auto s : shape)
      shapeVec.push_back(mlir::cast<IntegerAttr>(s).getInt());
    inStreamType = RankedTensorType::get(shapeVec, dataflow::StreamType::get(ctx, elementType));
  } else {
    inStreamType = dataflow::StreamType::get(ctx, elementType);
  }

  Type outStreamType;
  if (isRealAOS(dstFifoInfo)) {
    auto shape = dstFifoInfo.getAs<ArrayAttr>("shape");
    SmallVector<int64_t> shapeVec;
    for (auto s : shape)
      shapeVec.push_back(mlir::cast<IntegerAttr>(s).getInt());
    outStreamType = RankedTensorType::get(shapeVec, dataflow::StreamType::get(ctx, elementType));
  } else {
    outStreamType = dataflow::StreamType::get(ctx, elementType);
  }

  auto funcType = builder.getFunctionType({inStreamType, outStreamType}, {});
  auto func = builder.create<func::FuncOp>(builder.getUnknownLoc(), name, funcType);
  if (bufferName) {
    auto dictRead = DictionaryAttr::get(ctx, {builder.getNamedAttr("name", bufferName),
                                               builder.getNamedAttr("io", builder.getStringAttr("read-only"))});
    auto dictWrite = DictionaryAttr::get(ctx, {builder.getNamedAttr("name", bufferName),
                                                builder.getNamedAttr("io", builder.getStringAttr("write-only"))});
    func.setArgAttr(0, "buffer.info", dictRead);
    func.setArgAttr(1, "buffer.info", dictWrite);
  }
  auto *block = func.addEntryBlock();
  builder.setInsertionPointToStart(block);

  // 1. Alloc 1D buffer
  auto memrefType = MemRefType::get({srcInfo.getNumAccess()}, elementType);
  auto alloc = builder.create<memref::AllocOp>(builder.getUnknownLoc(), memrefType);

  // 2. Loop 1: Inbound (FIFO -> Buffer)
  {
    auto srcBounds = srcInfo.getAccessBounds();
    SmallVector<int64_t> srcLBs, srcUBs, srcSteps;
    for (auto bAttr : srcBounds) {
      auto b = mlir::cast<AccessBoundAttr>(bAttr);
      srcLBs.push_back(b.getLower());
      srcUBs.push_back(b.getUpper());
      srcSteps.push_back(1);
    }
    buildAffineLoopNest(builder, builder.getUnknownLoc(), srcLBs, srcUBs, srcSteps,
                        [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
      Value val;
      if (isRealAOS(srcFifoInfo)) {
        auto indices = srcFifoInfo.getAs<ArrayAttr>("indices");
        unsigned rank = indices.size();
        SmallVector<Value> indexVals;
        for (unsigned i = 0; i < rank; ++i) {
          if (ivs.size() >= rank)
            indexVals.push_back(ivs[ivs.size() - rank + i]);
          else
            indexVals.push_back(nestedBuilder.create<arith::ConstantIndexOp>(loc, 0));
        }
        auto map = nestedBuilder.getMultiDimIdentityMap(indexVals.size());
        val = nestedBuilder.create<ArrayOfStreamsReadOp>(loc, elementType, block->getArgument(0), indexVals, map).getResult();
      } else {
        val = nestedBuilder.create<StreamReadOp>(loc, elementType, block->getArgument(0)).getResult();
      }
      nestedBuilder.create<AffineStoreOp>(loc, val, alloc, srcInfo.getAccessMap().getAffineMap(), ivs);
    });
    // Add pipeline pragma to the rank-th loop
    if (auto outerMost = dyn_cast<AffineForOp>(block->getOperations().back())) {
      unsigned rank = srcBounds.size() / 3;
      unsigned d = 1;
      AffineForOp current = outerMost;
      while (d < rank && current) {
        current = dyn_cast<AffineForOp>(current.getBody()->getOperations().front());
        d++;
      }
      if (current) setLoopDirective(current, true, 1, false, false);
    }
  }

  // 3. Loop 2: Outbound (Buffer -> FIFO)
  {
    auto dstBounds = dstInfo.getAccessBounds();
    SmallVector<int64_t> dstLBs, dstUBs, dstSteps;
    for (auto bAttr : dstBounds) {
      auto b = mlir::cast<AccessBoundAttr>(bAttr);
      dstLBs.push_back(b.getLower());
      dstUBs.push_back(b.getUpper());
      dstSteps.push_back(1);
    }
    buildAffineLoopNest(builder, builder.getUnknownLoc(), dstLBs, dstUBs, dstSteps,
                        [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
      auto val = nestedBuilder.create<AffineLoadOp>(loc, elementType, alloc, ivs, dstInfo.getAccessMap().getAffineMap());
      if (isRealAOS(dstFifoInfo)) {
        auto indices = dstFifoInfo.getAs<ArrayAttr>("indices");
        unsigned rank = indices.size();
        SmallVector<Value> indexVals;
        for (unsigned i = 0; i < rank; ++i) {
          if (ivs.size() >= rank)
            indexVals.push_back(ivs[ivs.size() - rank + i]);
          else
            indexVals.push_back(nestedBuilder.create<arith::ConstantIndexOp>(loc, 0));
        }
        auto map = nestedBuilder.getMultiDimIdentityMap(indexVals.size());
        nestedBuilder.create<ArrayOfStreamsWriteOp>(loc, block->getArgument(1), val, indexVals, map);
      } else {
        nestedBuilder.create<StreamWriteOp>(loc, block->getArgument(1), val);
      }
    });
    // Add pipeline pragma to the rank-th loop
    if (auto outerMost = dyn_cast<AffineForOp>(block->getOperations().back())) {
      unsigned rank = dstBounds.size() / 3;
      unsigned d = 1;
      AffineForOp current = outerMost;
      while (d < rank && current) {
        current = dyn_cast<AffineForOp>(current.getBody()->getOperations().front());
        d++;
      }
      if (current) setLoopDirective(current, true, 1, false, false);
    }
  }

  builder.create<func::ReturnOp>(builder.getUnknownLoc());
  return module;
}

static AccessInfoAttr getAccessInfo(DFG::Node &node, Value memref, StringRef io = "") {
  if (!node.hwModule)
    return nullptr;
  
  std::string bufferName = getMemrefBufferName(memref);
  if (bufferName.empty())
    return nullptr;

  StringAttr bufferNameAttr = StringAttr::get(node.hwModule.getContext(), bufferName);

  auto funcOps = node.hwModule.getOps<func::FuncOp>();
  if (funcOps.begin() == funcOps.end())
    return nullptr;
  func::FuncOp func = *funcOps.begin();

  if (auto variants = node.hwModule->getAttrOfType<ArrayAttr>("merge.variants")) {
    for (auto variantAttr : variants) {
       auto variant = mlir::cast<DictionaryAttr>(variantAttr);
       auto accessInfoList = variant.getAs<ArrayAttr>("access_info_list");
       auto bufferInfoList = variant.getAs<ArrayAttr>("buffer_info_list");
       if (accessInfoList && bufferInfoList && accessInfoList.size() == bufferInfoList.size()) {
         for (unsigned i = 0; i < bufferInfoList.size(); ++i) {
           auto bufInfoWrapper = mlir::cast<DictionaryAttr>(bufferInfoList[i]);
           auto bufInfo = bufInfoWrapper.getAs<DictionaryAttr>("buffer.info");
           if (bufInfo && bufInfo.getAs<StringAttr>("name") == bufferNameAttr) {
               StringRef portIO = bufInfo.getAs<StringAttr>("io").getValue();
              if (io.empty() || portIO == io) {
                auto accInfoWrapper = mlir::cast<DictionaryAttr>(accessInfoList[i]);
                return accInfoWrapper.getAs<AccessInfoAttr>("access.info");
              }
           }
         }
       }
     }
  }

  int idx = getArgIdxByBufferName(node.hwModule, func, bufferNameAttr, io);
  if (idx != -1)
    return func.getArgAttrOfType<AccessInfoAttr>(idx, "dataflow.access_info");

  return nullptr;
}

static bool areCompatible(AccessInfoAttr a, AccessInfoAttr b) {
  if (!a || !b) return false;
  // Patterns must match
  if (a.getAccessMap() != b.getAccessMap()) return false;
  if (a.getAccessBounds() != b.getAccessBounds()) return false;
  if (a.getNumAccess() != b.getNumAccess()) return false;
  return true;
}

} // end anonymous namespace

void resolveDFGEdges(DFG &dfg) {
  struct ConverterTask {
    unsigned srcId;
    unsigned dstId;
    Value value;
    AffineStoreOp srcOp;
    AffineLoadOp dstOp;
    AccessInfoAttr srcInfo;
    AccessInfoAttr dstInfo;
    DictionaryAttr srcFifoInfo;
    DictionaryAttr dstFifoInfo;
  };
  SmallVector<ConverterTask> tasks;

  for (auto &nodePair : dfg.nodes) {
    auto &srcNode = nodePair.second;
    auto srcId = srcNode.id;

    if (dfg.outEdges.count(srcId)) {
      for (auto &edge : dfg.outEdges[srcId]) {
        auto dstId = edge.id;
        auto it = dfg.nodes.find(dstId);
        if (it == dfg.nodes.end()) continue;
        auto &dstNode = it->second;
        
        // Default to MEMREF unless proven otherwise
        edge.kind = DFG::EdgeKind::MEMREF;

        auto srcInfo = getAccessInfo(srcNode, edge.value, "write-only");
        auto dstInfo = getAccessInfo(dstNode, edge.value, "read-only");
        auto srcFifoInfo = edge.srcFifoInfo;
        auto dstFifoInfo = edge.dstFifoInfo;
        LLVM_DEBUG(llvm::dbgs() << "srcInfo: " << srcInfo << "\ndstInfo: " << dstInfo << "\n";);
        bool areComp = areCompatible(srcInfo, dstInfo);
        LLVM_DEBUG(llvm::dbgs() << "areCompatible: " << areComp << "\n";);
        if (areComp) {
          bool realAOS = isRealAOS(srcFifoInfo) || isRealAOS(dstFifoInfo);
          edge.kind = realAOS ? DFG::EdgeKind::AOS : DFG::EdgeKind::STREAM;
          LLVM_DEBUG(llvm::dbgs() << "Edge " << srcId << " -> " << dstId << ": " 
                                  << (realAOS ? "AOS" : "STREAM") << " (compatible)\n");
          // Also update the corresponding inEdge
          if (dfg.inEdges.count(dstId)) {
            for (auto &inEdge : dfg.inEdges[dstId]) {
               if (inEdge.id == srcId && inEdge.value == edge.value) {
                 inEdge.kind = edge.kind;
                 break;
               }
            }
          }
        } else if (srcInfo && dstInfo) {
          // Incompatible MEMREF edge: needs a converter
          LLVM_DEBUG(llvm::dbgs() << "Edge " << srcId << " -> " << dstId << ": INCOMPATIBLE (needs converter)\n");
          tasks.push_back({srcId, dstId, edge.value, (affine::AffineStoreOp)edge.srcOp,
                           (affine::AffineLoadOp)edge.dstOp, srcInfo, dstInfo,
                           srcFifoInfo, dstFifoInfo});
        }
      }
    }
  }

  // Insert converters
  unsigned nextConverterId = 0;
  for (auto &nodePair : dfg.nodes)
    if (nodePair.first >= nextConverterId) nextConverterId = nodePair.first + 1;

  for (auto &task : tasks) {
    auto &srcNode = dfg.nodes[task.srcId];
    
    // 1. Create unique name
    std::string bufName = getMemrefBufferName(task.value);
    if (bufName.empty()) bufName = "buf";
    std::string converterName = "converter_" + bufName + "_" + std::to_string(task.srcId) + "_to_" + std::to_string(task.dstId);

    // 2. Extract FIFO info
    auto srcFifoInfo = task.srcFifoInfo;
    auto dstFifoInfo = task.dstFifoInfo;
    auto elementType = mlir::cast<MemRefType>(task.value.getType()).getElementType();

    // 3. Create ModuleOp
    StringAttr bufferNameAttr = getMemrefBufferName(task.value).empty() ? nullptr : StringAttr::get(srcNode.hwModule->getContext(), getMemrefBufferName(task.value));
    LLVM_DEBUG(llvm::dbgs() << "Creating converter module: " << converterName << "\n");
    auto converterModule = createConverterModule(srcNode.hwModule->getContext(), converterName,
                                                  task.srcInfo, task.dstInfo,
                                                  srcFifoInfo, dstFifoInfo, elementType, bufferNameAttr);

    // 4. Create Node
    unsigned converterId = nextConverterId++;
    DFG::Node converterNode(converterId, nullptr);
    converterNode.hwModule = converterModule;
    converterNode.type = "converter";
    converterNode.kind = DFG::NodeKind::CONVERTER;
    dfg.nodes.insert({converterId, converterNode});

    // 5. Connect src -> converter -> dst
    // Remove the old edge
    dfg.removeEdge(task.srcId, task.dstId, task.value);

    // Add new edges
    dfg.addEdge(task.srcId, converterId, task.srcOp, nullptr, task.value);
    // Since we just added it, we need to set its kind to STREAM
    if (dfg.outEdges.count(task.srcId)) {
      for (auto &e : dfg.outEdges[task.srcId]) {
        if (e.id == converterId && e.value == task.value) {
          e.kind = isRealAOS(srcFifoInfo) ? DFG::EdgeKind::AOS : DFG::EdgeKind::STREAM;
        }
      }
    }
    if (dfg.inEdges.count(converterId)) {
      for (auto &e : dfg.inEdges[converterId]) {
        if (e.id == task.srcId && e.value == task.value) {
          e.kind = isRealAOS(srcFifoInfo) ? DFG::EdgeKind::AOS : DFG::EdgeKind::STREAM;
        }
      }
    }

    dfg.addEdge(converterId, task.dstId, nullptr, task.dstOp, task.value);
    if (dfg.outEdges.count(converterId)) {
      for (auto &e : dfg.outEdges[converterId]) {
        if (e.id == task.dstId && e.value == task.value) {
          e.kind = isRealAOS(dstFifoInfo) ? DFG::EdgeKind::AOS : DFG::EdgeKind::STREAM;
        }
      }
    }
    if (dfg.inEdges.count(task.dstId)) {
      for (auto &e : dfg.inEdges[task.dstId]) {
        if (e.id == converterId && e.value == task.value) {
          e.kind = isRealAOS(dstFifoInfo) ? DFG::EdgeKind::AOS : DFG::EdgeKind::STREAM;
        }
      }
    }
  }
}

} // namespace streamhls
} // namespace mlir
