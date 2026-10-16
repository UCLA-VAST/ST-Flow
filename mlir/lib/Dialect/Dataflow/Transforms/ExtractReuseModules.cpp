// #include "PassDetail.h"

#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/MemoryBuffer.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringMap.h"
#include "streamhls/Dialect/Dataflow/Transforms/Passes.h"
#include "streamhls/Dialect/Dataflow/Support/Utils.h"
#include "streamhls/Dialect/Dataflow/Support/DFG.h"
#include "streamhls/Dialect/Dataflow/Transforms/MergeKernels/Utils.h"
#include "streamhls/Dialect/Dataflow/Transforms/MergeKernels/Transform.h"


#define DEBUG_TYPE "streamhls-extract-reuse-modules"

using namespace mlir;
using namespace streamhls;

namespace {

struct ReuseNodeInfo {
  int node_id;
  int start_time;
  StringRef hw_name;
  int dsps;
  int memory;
};

constexpr StringLiteral kInfoKey("buffer.info");

bool isDefinedOutside(Value v, Region &scope) {
  if (Operation *def = v.getDefiningOp())
    return !scope.isAncestor(def->getParentRegion());
  auto *ownerBlock = v.cast<BlockArgument>().getOwner();
  return !scope.isAncestor(ownerBlock->getParent());
}

bool isTopFuncBlockArg(Value v) {
  if (!v.isa<BlockArgument>()) return false;
  auto ba = v.cast<BlockArgument>();
  auto *block = ba.getOwner();
  auto *parentOp = block->getParentOp();
  return isa_and_nonnull<func::FuncOp>(parentOp) && block->isEntryBlock();
}

/// Read the classification {name,type} from the standardized "buffer.info" attr.
std::pair<std::string, std::string> getNameAndTypeFromAnnotation(Value v) {
  // Block argument? Check the arg attribute.
  if (auto ba = v.dyn_cast<BlockArgument>()) {
    auto func = cast<func::FuncOp>(ba.getOwner()->getParentOp());
    if (auto dict = func.getArgAttrOfType<DictionaryAttr>(ba.getArgNumber(), kInfoKey)) {
      auto nameAttr = dict.getAs<StringAttr>("name");
      auto typeAttr = dict.getAs<StringAttr>("type");
      if (nameAttr && typeAttr) return {nameAttr.str(), typeAttr.str()};
    }
    // Fallbacks
    std::string name = "arg" + std::to_string(ba.getArgNumber());
    return {name, "off-chip"};
  }

  // Defined by an op?
  Operation *def = v.getDefiningOp();
  if (!def) return {"<unknown>", "off-chip"};

  if (auto dict = def->getAttrOfType<DictionaryAttr>(kInfoKey)) {
    auto nameAttr = dict.getAs<StringAttr>("name");
    auto typeAttr = dict.getAs<StringAttr>("type");
    if (nameAttr && typeAttr) return {nameAttr.str(), typeAttr.str()};
  }

  // Heuristic defaults by op kind if annotation missing.
  if (isa<memref::GetGlobalOp>(def)) {
    auto g = cast<memref::GetGlobalOp>(def);
    return {g.getName().str(), "weights"};
  }
  if (isa<memref::AllocOp, memref::AllocaOp>(def)) {
    return {"buf", "local-buffer"}; // conservative fallback
  }

  return {"<unknown>", "off-chip"};
}

/// Collect *memref* deps used by the loop-nest root.
struct Dep {
  Value v;                  // the memref-like SSA value used
  std::string name;         // from annotation
  std::string type;         // "off-chip" | "local-buffer" | "shared-buffer" | "weights"
};

void collectMemrefDepsClassified(
    Operation *rootOfNest,
    SmallVectorImpl<Dep> &orderedDeps,
    DenseSet<Value> &depSet) {

  rootOfNest->walk([&](Operation *op) {
    auto pushMem = [&](Value m) {
      if (!m || !isa<MemRefType>(m.getType())) return;
      if (depSet.contains(m)) return;
      // Only consider top func args, allocs, get_global
      if (!(isTopFuncBlockArg(m) ||
            m.getDefiningOp<memref::AllocOp>() ||
            m.getDefiningOp<memref::GetGlobalOp>()))
        return;

      auto [nm, tp] = getNameAndTypeFromAnnotation(m);
      orderedDeps.push_back(Dep{m, nm, tp});
      depSet.insert(m);
    };

    if (auto ld = dyn_cast<affine::AffineLoadOp>(op)) {
      pushMem(ld.getMemRef());
      return;
    }
    if (auto st = dyn_cast<affine::AffineStoreOp>(op)) {
      pushMem(st.getMemRef());
      return;
    }
    if (auto ld = dyn_cast<memref::LoadOp>(op)) {
      pushMem(ld.getMemref());
      return;
    }
    if (auto st = dyn_cast<memref::StoreOp>(op)) {
      pushMem(st.getMemref());
      return;
    }
  });
}

/// Also collect constants referenced by the nest.
void collectExternalConstants(
    Operation *rootOfNest,
    SmallVectorImpl<arith::ConstantOp> &orderedConsts,
    DenseSet<Operation*> &constSet) {

  Region &nestRegion = *rootOfNest->getParentRegion();

  rootOfNest->walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      if (auto cst = operand.getDefiningOp<arith::ConstantOp>()) {
        if (nestRegion.isAncestor(cst->getParentRegion())) {
          if (constSet.insert(cst).second)
            orderedConsts.push_back(cst);
        }
      }
    }
  });
}

/// Build a standalone module containing a function for a single loop nest.
ModuleOp affineForOpToModule(AffineForOp &forOp, int node_id) {
  MLIRContext *ctx = forOp.getContext();
  auto loc = forOp.getLoc();

  // 1) Collect classified deps.
  SmallVector<Dep, 8> orderedDeps;
  DenseSet<Value> depSet;
  collectMemrefDepsClassified(forOp.getOperation(), orderedDeps, depSet);

  SmallVector<arith::ConstantOp, 8> orderedConsts;
  DenseSet<Operation*> constSet;
  collectExternalConstants(forOp.getOperation(), orderedConsts, constSet);

  // Partition deps: args vs internal materialization.
  SmallVector<Dep, 8> argDeps;
  SmallVector<Dep, 8> internalizeDeps; // locals & weights defined outside the nest

  Region &nestRegion = *forOp->getParentRegion();
  for (const Dep &d : orderedDeps) {
    bool outside = isDefinedOutside(d.v, nestRegion);
    if (d.type == "off-chip" || d.type == "shared-buffer" || d.type == "weights") {
      argDeps.push_back(d);
    } else if ((d.type == "local-buffer")) {
      internalizeDeps.push_back(d);
    }
  }

  // 2) Create new module + function type with argDeps only.
  OpBuilder topBuilder(ctx);
  ModuleOp newModule = ModuleOp::create(UnknownLoc::get(ctx));

  SmallVector<Type, 8> argTypes;
  argTypes.reserve(argDeps.size());
  for (const Dep &d : argDeps)
    argTypes.push_back(d.v.getType());

  auto funcType = FunctionType::get(ctx, argTypes, {});
  func::FuncOp newFunc =
      func::FuncOp::create(loc, "forOpFunc_" + std::to_string(node_id), funcType);
  newModule.push_back(newFunc);

  // Preserve buffer.info
  for (unsigned i = 0; i < argDeps.size(); ++i) {
    const auto &d = argDeps[i];
    StringRef ioKind = "unknown";
    
    // Attempt to retrieve ioKind from original attributes
    DictionaryAttr dictAttr = nullptr;
    if (auto ba = d.v.dyn_cast<BlockArgument>()) {
        auto func = cast<func::FuncOp>(ba.getOwner()->getParentOp());
        dictAttr = func.getArgAttrOfType<DictionaryAttr>(ba.getArgNumber(), kInfoKey);
    } else if (Operation *def = d.v.getDefiningOp()) {
        dictAttr = def->getAttrOfType<DictionaryAttr>(kInfoKey);
    }

    if (dictAttr) {
        if (auto ioAttr = dictAttr.getAs<StringAttr>("io"))
            ioKind = ioAttr.getValue();
    }

    if (ioKind == "unknown") {
      // Infer from use-sites inside the forOp
      forOp.walk([&](Operation *op) {
        if (auto ld = dyn_cast<affine::AffineLoadOp>(op)) {
          if (ld.getMemRef() == d.v) ioKind = "read-only";
        } else if (auto st = dyn_cast<affine::AffineStoreOp>(op)) {
          if (st.getMemRef() == d.v) ioKind = "write-only";
        } else if (auto ld = dyn_cast<memref::LoadOp>(op)) {
          if (ld.getMemref() == d.v) ioKind = "read-only";
        } else if (auto st = dyn_cast<memref::StoreOp>(op)) {
          if (st.getMemref() == d.v) ioKind = "write-only";
        }
      });
    }
    
    auto dict = makeInfoAttr(ctx, d.name, d.type, ioKind);
    newFunc.setArgAttr(i, kInfoKey, dict);
  }

  // 3) Entry block & mapping.
  Block *entry = newFunc.addEntryBlock();
  IRMapping mapper;

  for (auto it : llvm::zip(argDeps, entry->getArguments())) {
    const Dep &d = std::get<0>(it);
    Value newArg = std::get<1>(it);
    mapper.map(d.v, newArg);
  }

  // 4) Clone constants.
  OpBuilder b(entry, entry->begin());
  for (arith::ConstantOp cst : orderedConsts) {
    auto newCst = b.create<arith::ConstantOp>(cst.getLoc(), cst.getType(), cst.getValueAttr());
    mapper.map(cst.getResult(), newCst.getResult());
  }

  // 5) Materialize local-buffers and weights.
  for (const Dep &d : internalizeDeps) {
    Operation *def = d.v.getDefiningOp();
    if (auto alloc = dyn_cast<memref::AllocOp>(def)) {
      auto memTy = cast<MemRefType>(alloc.getType());
      if (memTy.hasStaticShape() && alloc.getDynamicSizes().empty() && !alloc.getAlignment()) {
        auto newAlloc = b.create<memref::AllocOp>(alloc.getLoc(), memTy);
        mapper.map(d.v, newAlloc.getResult());
      } else {
        Operation *cloned = b.clone(*alloc.getOperation(), mapper);
        mapper.map(d.v, cloned->getResult(0));
      }
      continue;
    }
    if (auto alloca = dyn_cast<memref::AllocaOp>(def)) {
      auto memTy = cast<MemRefType>(alloca.getType());
      if (memTy.hasStaticShape() && alloca.getDynamicSizes().empty() && !alloca.getAlignment()) {
        auto newAlloca = b.create<memref::AllocaOp>(alloca.getLoc(), memTy);
        mapper.map(d.v, newAlloca.getResult());
      } else {
        Operation *cloned = b.clone(*alloca.getOperation(), mapper);
        mapper.map(d.v, cloned->getResult(0));
      }
      continue;
    }

    DenseSet<StringRef> copiedGlobals;
    if (auto g = dyn_cast<memref::GetGlobalOp>(def)) {
      FlatSymbolRefAttr symRef = g.getNameAttr();
      StringRef symName = symRef.getValue();

      if (copiedGlobals.insert(symName).second) {
        ModuleOp srcModule = forOp->getParentOfType<ModuleOp>();
        memref::GlobalOp srcGlobal = srcModule.lookupSymbol<memref::GlobalOp>(symName);
        if (srcGlobal) {
          OpBuilder topBeforeFunc(newModule.getBody(), Block::iterator(newFunc));
          (void)topBeforeFunc.clone(*srcGlobal.getOperation());
        }
      }
      auto newG = b.create<memref::GetGlobalOp>(g.getLoc(), g.getType(), symName);
      mapper.map(d.v, newG.getResult());
      continue;
    }
    
    if (def) {
      Operation *cloned = b.clone(*def, mapper);
      if (!cloned->getResults().empty())
        mapper.map(d.v, cloned->getResult(0));
    }
  }

  // 6) Clone the loop nest.
  b.setInsertionPointToEnd(entry);
  (void)b.clone(*forOp.getOperation(), mapper);

  b.create<func::ReturnOp>(loc);
  return newModule;
}

LogicalResult tileMemoryModule(ModuleOp &module, SmallVector<int64_t> &tiles) {
 
  TilingSolution tilingSol;
  for(size_t i = 0; i < tiles.size(); ++i){
    tilingSol.l1TilingFactors.push_back(1);
    tilingSol.l2TilingFactors.push_back(tiles[i]);
  }

  AffineLoopBand band;
  module->walk([&](AffineForOp forOp) { band.push_back(forOp); });
  std::reverse(band.begin(), band.end());
  assert(band.size() == tiles.size() && "ExtractModule: Tiling size mismatch");
  func::FuncOp func = *module.getOps<func::FuncOp>().begin();
  MLIRContext *ctx = module.getContext();
  OpBuilder b(ctx);
  AffineForOp topLoop = apply2LevelTiling(band[0], b, tilingSol);
  simplifyAffineOps(module);
  auto attachFifoInfo = [&](Operation *op, Value mem) {
    auto ba = mem.dyn_cast<BlockArgument>();
    if (!ba || ba.getOwner() != &func.front()) return;

    if (auto dict = func.getArgAttrOfType<DictionaryAttr>(ba.getArgNumber(), kInfoKey)) {
      if (auto tyAttr = dict.getAs<StringAttr>("type")) {
        StringRef ty = tyAttr.getValue();
        if (ty == "off-chip" || ty == "weights") return;
      }
    }

    auto innermostLoop = op->getParentOfType<AffineForOp>();
    if (!innermostLoop) return;

    AffineLoopBand postTileBand;
    getLoopBandFromInnermost(innermostLoop, postTileBand);
    if (postTileBand.size() < tiles.size()) return;

    SmallVector<Value, 4> indicesVals;
    indicesVals.reserve(tiles.size());
    for (AffineForOp loop : llvm::ArrayRef(postTileBand).take_back(tiles.size())) {
      indicesVals.push_back(loop.getInductionVar());
    }

    (void)setFifoAttr(ctx, op, tiles, indicesVals);
  };

  func.walk([&](Operation *op) {
    if (auto ld = dyn_cast<AffineLoadOp>(op))
      attachFifoInfo(op, ld.getMemRef());
    else if (auto st = dyn_cast<AffineStoreOp>(op))
      attachFifoInfo(op, st.getMemRef());
  });

  
  // --------------------------------------------------------------------------
  // NEW: Attach single-variant metadata for this tiled memory module
  // --------------------------------------------------------------------------
  {
    OpBuilder b(ctx);
    mlir::streamhls::AffineLoopBand tiledBand;
    getLoopBandFromOutermost(topLoop, tiledBand);
    DictionaryAttr variantDict =
        buildVariantMetadataForFunc(b, /*srcFunc=*/func,
                                       /*mergedFunc=*/func,
                                       /*band=*/tiledBand);

    module->setAttr("merge.variants", b.getArrayAttr(variantDict));
  }

  return success();
}


} // end anonymous namespace

namespace mlir {
namespace streamhls {

DFG extractMergedDFG(ModuleOp &module, HWMapping hwMapping, bool enableTiling) {
  auto funcs = module.getOps<func::FuncOp>();
  if (funcs.empty()) {
    llvm::errs() << "[extractMergedDFG] ERROR: No functions found.\n";
    assert(false && "No functions found");
  }
  func::FuncOp func = *funcs.begin();
  Block *block = &func.front();
  DFG graph(*block);
  if (!graph.init())
    return graph;

  // --- Build hw mapping and merge graph ---
  DFG mergedGraph(*block);
  if (!mergedGraph.init())
    return mergedGraph;

  mergedGraph.mergeNodesByHwMapping(hwMapping.mapping);

  // --- Process nodes for extraction and tiling ---
  auto topoOrder = graph.topologicalSort();
  for (unsigned nodeId : topoOrder) {
    DFG::Node *node = graph.getNode(nodeId);
    if (!node) continue;

    auto forOp = cast<AffineForOp>(node->op);
    auto newModule = affineForOpToModule(forOp, nodeId);

    // Attach wrapped module to merged node
    for (auto &mergedNodePair : mergedGraph.nodes) {
      auto &mergedNode = mergedNodePair.second;
      for (auto &mForOp : mergedNode.forOps) {
        if (mForOp.getOperation() == forOp.getOperation()) {
          mergedNode.wrappedModules.push_back(newModule);
          // Set initial hwModule (especially for communication nodes)
          if (!mergedNode.hwModule)
            mergedNode.hwModule = newModule;
          break;
        }
      }
    }
  }
  mergedGraph.populateNodeCandidates();
  return mergedGraph;
}

std::string printDFGNodes(DFG &graph) {
  std::string data;
  llvm::raw_string_ostream os(data);
  graph.printNodes(os);
  return data;
}

std::string printDFGEdges(DFG &graph) {
  std::string data;
  llvm::raw_string_ostream os(data);
  graph.printEdges(os);
  return data;
}

std::string printDFGAsDot(DFG &graph) {
  std::string data;
  llvm::raw_string_ostream os(data);
  graph.printAsDot(os);
  return data;
}

ModuleOp generateSystemModule(DFG &graph, ModuleOp templateModule) {
  return graph.generateSystemModule(templateModule);
}

} // namespace streamhls
} // namespace mlir