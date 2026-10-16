

#include <optional>
#include <functional>
#include <string>
#include <regex>
#include <iostream>
#include <memory>
#include <array>
#include <stdexcept>
#include <random>
#include <chrono>
#include <ctime>
#include <map>
#include <utility>
#include <numeric> // for std::gcd
#include <algorithm>

#include "llvm/ADT/DenseSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/MapVector.h"

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/Debug.h"

#include "streamhls/Dialect/Dataflow/Support/DFG.h"
#include "streamhls/Dialect/Dataflow/Support/AffineAnalysis.h"
#include "streamhls/Dialect/Dataflow/Support/Utils.h"
#include "streamhls/Dialect/Dataflow/Support/Equations.h"
#include "streamhls/Dialect/Dataflow/Support/PerformanceModel.h"
#include "streamhls/Dialect/Dataflow/Transforms/MergeKernels/Utils.h"
#include "streamhls/Dialect/Dataflow/Transforms/MergeKernels/Merging.h"

using namespace mlir;
using namespace streamhls;
using namespace dataflow;
using namespace affine;

using Node = DFG::Node;

#define DEBUG_TYPE "streamhls-dfg"


std::pair<bool, Attribute> mlir::streamhls::isInitializationLoopNest(AffineForOp forOp) {
  AffineLoopBand band;
  getLoopBandFromOutermost(forOp, band);
  if (band.empty())
    return {false, Attribute()};

  // Innermost loop in the (assumed) perfectly nested band.
  AffineForOp innermost = band.back();

  Block &body = *innermost.getBody();

  // Consider only non-terminator ops; affine.for bodies end with affine.yield.
  auto nonTermOps = body.without_terminator();

  // Condition 1: exactly one non-terminator operation in the innermost loop.
  if (std::distance(nonTermOps.begin(), nonTermOps.end()) != 1)
    return {false, Attribute()};

  Operation &innerOp = *nonTermOps.begin();

  // Condition 2: that single op must be an affine.store.
  auto store = dyn_cast<AffineStoreOp>(&innerOp);
  if (!store)
    return {false, Attribute()};

  // Condition 3: stored value must be a constant.
  Attribute initAttr;
  if (!matchPattern(store.getValue(), m_Constant(&initAttr)))
    return {false, Attribute()};

  // Condition 4: The constant must be zero.
  bool isZero = false;
  if (auto floatAttr = initAttr.dyn_cast<FloatAttr>()) {
    isZero = floatAttr.getValue().isZero();
  } else if (auto intAttr = initAttr.dyn_cast<IntegerAttr>()) {
    isZero = intAttr.getValue().isZero();
  }

  // if (!isZero) {
  //   llvm::errs() << "Error: Initialization loop nest must initialize to constant zero. Found: " << initAttr << "\n";
  //   llvm::report_fatal_error("Invalid initialization value");
  // }

  return {true, initAttr};
}

// Exact ceilDiv for non-negative ints.
static inline int64_t ceilDiv64(int64_t num, int64_t den) {
  assert(den > 0 && "denominator must be > 0");
  if (num <= 0) return 0;
  return (num + den - 1) / den;
}

static std::optional<int64_t> getConstTripCount(AffineForOp forOp) {
  if (!forOp.hasConstantLowerBound() || !forOp.hasConstantUpperBound())
    return std::nullopt;
  int64_t lb = forOp.getConstantLowerBound();
  int64_t ub = forOp.getConstantUpperBound();
  int64_t step = forOp.getStep().getSExtValue();
  if (ub <= lb) return int64_t(0);
  return ceilDiv64(ub - lb, step);
}

// Trip count for the whole band rooted at 'root' (product of per-loop TCs).
static std::optional<int64_t> getBandTripCount(AffineForOp root) {
  AffineLoopBand band;
  getLoopBandFromOutermost(root, band);
  int64_t prod = 1;
  for (AffineForOp l : band) {
    auto tc = getConstTripCount(l);
    if (!tc.has_value()) return std::nullopt;
    prod *= *tc;
  }
  return prod;
}

struct FlopTally {
  int64_t add = 0, sub = 0, mul = 0, div = 0, fma = 0, other = 0;
  // Classical FLOP accounting: add/sub/mul/div = 1, fma = 2.
  int64_t totalScalar() const { return add + sub + mul + div + 2 * fma; }
};

// Count floating-point operations exactly once per dynamic iteration.
static FlopTally countFlopsPerIter(mlir::Operation *scope) {
  FlopTally t;
  scope->walk([&](mlir::Operation *op) {
    using namespace mlir;
    using namespace mlir::arith;
    using namespace mlir::math;

    if (isa<AddFOp>(op))      { t.add++; return; }
    if (isa<SubFOp>(op))      { t.sub++; return; }
    if (isa<MulFOp>(op))      { t.mul++; return; }
    if (isa<DivFOp>(op))      { t.div++; return; }
    // arith FMA (if present in your MLIR version)
#if defined(MLIR_ARITH_HAS_FMA) || 1
    if (op->getName().getStringRef() == "arith.fma") { t.fma++; return; }
#endif
    // Transcendentals (optionally account these as 'other'; not counted in totalScalar)
    if (isa<ExpOp, Exp2Op, LogOp, Log10Op, SinOp, CosOp, TanOp, SqrtOp, PowFOp>(op)) {
      t.other++; return;
    }
  });
  return t;
}

// Initializes the data dependence graph by walking operations in `block`.
// - Nodes: outermost loops of perfectly-nested affine loop bands that contain
//          at least one AffineLoadOp or AffineStoreOp.
// - Edges: for each memref, connect every store-node to every load-node that
//          accesses the same memref (multi-producer / multi-consumer supported).
// - Stream types are ignored. No node merging is performed here.
bool DFG::init() {
  LLVM_DEBUG(llvm::dbgs() << "--- Initializing DFG (multi-producer/multi-consumer) ---\n");

  nodes.clear();
  inEdges.clear();
  outEdges.clear();
  memrefEdgeCount.clear();
  edges.clear();
  nextNodeId = 0;

  // Map outermost loop-band header -> node id.
  DenseMap<AffineForOp, unsigned> loopToNodeId;

  // Helper: get outermost for-op (band[0]) from any op inside a perfectly nested band.
  auto getOutermostBandFor = [&](Operation *innerOp) -> AffineForOp {
    if (!innerOp) return AffineForOp();
    auto innermostFor = innerOp->getParentOfType<AffineForOp>();
    if (!innermostFor) return AffineForOp();
    AffineLoopBand band;
    getLoopBandFromInnermost(innermostFor, band);
    if (band.empty()) return AffineForOp();
    return band.front(); // outermost
  };

  // Record of accesses grouped by memref.
  struct AccessRec {
    Value memref;
    Operation *op;     // AffineLoadOp or AffineStoreOp (Operation*)
    unsigned nodeId;   // owner node
  };

  DenseMap<Value, SmallVector<AccessRec, 4>> loadsByMemref;
  DenseMap<Value, SmallVector<AccessRec, 4>> storesByMemref;

  // Also keep per-node lists to fill Node.loads / Node.stores.
  SmallVector<std::pair<unsigned, AffineLoadOp>, 8> nodeLoads;
  SmallVector<std::pair<unsigned, AffineStoreOp>, 8> nodeStores;

  // Ensure a node exists for the outermost band header; return its id.
  auto ensureNodeFor = [&](AffineForOp outerFor, Operation *anyInnerOp) -> unsigned {
    auto it = loopToNodeId.find(outerFor);
    if (it != loopToNodeId.end()) return it->second;

    // Create node
    unsigned id = nextNodeId++;
    Node node(id, outerFor);

    // Gather band (outer->inner) and compute trip count.
    AffineLoopBand band;
    getLoopBandFromInnermost(anyInnerOp->getParentOfType<AffineForOp>(), band);
    // trip count for the band (falls back to 0 on failure)
    node.tripCount = getLoopNestIterations(band);
    // Save the band header (outermost) for convenience
    node.forOps.push_back(outerFor);

    // Populate performance model fields
    auto tally = countFlopsPerIter(outerFor.getOperation());
    node.arithWeight = tally.totalScalar();

    for (AffineForOp l : band) {
      if (auto tc = getConstTripCount(l)) {
        node.loopsUBs.push_back(*tc);
        // node.maxTripCounts.push_back(*tc);
      } else {
        node.loopsUBs.push_back(-1);
        // node.maxTripCounts.push_back(0);
      }
    }

    // get node type by collecting loop depth and type of arithmetic operations
    // llvm::dbgs() << "node: " << id << " has: \n";
    SmallVector<std::string> arithOps;
    SmallVector<std::string> memrefRanks;
    // set to identify the number of unique memrefs
    llvm::SmallDenseSet<Value> uniqueMemrefs;
    band[0]->walk([&](Operation *op) {
      auto dialect = op->getDialect();
      if (dialect->getNamespace() == "arith" || dialect->getNamespace() == "math") {
        auto opNameWithDialect = op->getName().getStringRef().str();
        arithOps.push_back(opNameWithDialect);
      }
      if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
        auto memrefType = loadOp.getMemRefType();
        auto shape = memrefType.getShape();
        auto rankStr = std::to_string(memrefType.getRank());
        uniqueMemrefs.insert(loadOp.getMemRef());
        memrefRanks.push_back(rankStr);
      } else if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
        auto memrefType = storeOp.getMemRefType();
        auto shape = memrefType.getShape();
        auto rankStr = std::to_string(memrefType.getRank());
        uniqueMemrefs.insert(storeOp.getMemRef());
        memrefRanks.push_back(rankStr);
      }
    });
    node.kind = arithOps.empty() ? NodeKind::MEMORY : NodeKind::COMPUTE;
    arithOps.push_back("depth_" + std::to_string(band.size()));
    // join the arithops with "_"
    std::string arithOpsStr = llvm::join(arithOps, "_") + "_memranks_" + llvm::join(memrefRanks, "_") + "_memrefs_" + std::to_string(uniqueMemrefs.size());
    node.type = arithOpsStr;
    node.bandSize = band.size();
    nodes.insert({id, node});
    loopToNodeId.insert({outerFor, id});
    return id;
  };

  // Walk to collect all loads/stores and assign them to nodes.
  block.walk([&](Operation *op) {
    if (auto load = dyn_cast<AffineLoadOp>(op)) {
      auto outerFor = getOutermostBandFor(op);
      if (!outerFor) return; // ignore non-loop loads for this project’s assumption
      
      // Skip initialization nodes
      if (isInitializationLoopNest(outerFor).first) return;

      unsigned nid = ensureNodeFor(outerFor, op);
      nodeLoads.emplace_back(nid, load);
      loadsByMemref[load.getMemRef()].push_back({load.getMemRef(), op, nid});
    } else if (auto store = dyn_cast<AffineStoreOp>(op)) {
      auto outerFor = getOutermostBandFor(op);
      if (!outerFor) return; // ignore non-loop stores
      
      // Skip initialization nodes
      if (isInitializationLoopNest(outerFor).first) return;

      unsigned nid = ensureNodeFor(outerFor, op);
      nodeStores.emplace_back(nid, store);
      storesByMemref[store.getMemRef()].push_back({store.getMemRef(), op, nid});
    }
    // NOTE: Stream/FIFO ops are intentionally ignored in this init() variant.
  });

  // Populate per-node load/store lists (visible in DFG::Node).
  for (auto &p : nodeLoads) {
    auto *n = getNode(p.first);
    if (n) n->loads.push_back(p.second);
  }
  for (auto &p : nodeStores) {
    auto *n = getNode(p.first);
    if (n) n->stores.push_back(p.second);
  }

  // Build edges: for each memref, connect each producer node to each consumer node.
  // We add one edge per (store op, load op) pair across *different* nodes.
  // (If you prefer one edge per node pair per memref, you can collapse duplicates below.)
  for (auto &kv : storesByMemref) {
    Value mem = kv.first;
    auto &prods = kv.second;

    auto loadsIt = loadsByMemref.find(mem);
    if (loadsIt == loadsByMemref.end()) continue; // no consumers
    auto &cons = loadsIt->second;

    for (const auto &s : prods) {
      auto srcStore = cast<AffineStoreOp>(s.op);
      for (const auto &l : cons) {
        if (s.nodeId == l.nodeId) {
          // Intra-node flow stays internal; skip creating a graph edge.
          continue;
        }
        auto dstLoad = cast<AffineLoadOp>(l.op);

        // (Optional, stricter) Filter with dependence analysis:
        //   Use affine::MemRefAccess & affine::checkMemrefAccessDependence(...)
        //   to keep only RAW dependences. Example skeleton you can enable later:
        //
        //   MemRefAccess srcAcc(srcStore);
        //   MemRefAccess dstAcc(dstLoad);
        //   auto depRes = checkMemrefAccessDependence(srcAcc, dstAcc, /*loopDepth*/ std::numeric_limits<unsigned>::max());
        //   if (!depRes || depRes.value == DependenceResult::NoDependence) continue;

        // Add graph edge (multi-producer / multi-consumer allowed).
        addEdge(s.nodeId, l.nodeId, srcStore, dstLoad, mem);
        edges.push_back({s.nodeId, l.nodeId, mem});
      }
    }
  }

  // (Optional) Collapse duplicate node-pair edges per memref if desired:
  //   You can dedup by (srcId, dstId, mem) and keep the first (op pair).

  // Done.
  return true;
}


// Merge node 'srcId' into node 'dstId' by moving all edges from 'srcId' to
Node *DFG::mergeNodes(unsigned srcId, unsigned dstId, bool force) {
  Node *srcNode = getNode(srcId);
  Node *dstNode = getNode(dstId);
  if (!force) {
    assert(srcNode->bandSize == dstNode->bandSize &&
           "Cannot merge nodes with different band sizes");
  }

  // Move all edges from 'srcId' to 'dstId'.
  SmallVector<memref::AllocOp, 2> allocOps;
  if (outEdges.count(srcId) > 0) {
    SmallVector<Edge, 2> oldOutEdges = outEdges[srcId];
    for (auto &outEdge : oldOutEdges) {
      removeEdge(srcId, outEdge.id, outEdge.value);
      
      unsigned newDstId = (outEdge.id == srcId) ? dstId : outEdge.id;
      if (newDstId != dstId) {
        addEdge(dstId, newDstId, outEdge.value, outEdge.kind, outEdge.resolvedType,
                outEdge.srcOp, outEdge.dstOp, outEdge.srcFifoInfo, outEdge.dstFifoInfo);
      } else {
        // Self-loop (internal buffer or internal stream)
        addEdge(dstId, dstId, outEdge.value, outEdge.kind, outEdge.resolvedType,
                outEdge.srcOp, outEdge.dstOp, outEdge.srcFifoInfo, outEdge.dstFifoInfo);
        if (outEdge.value && outEdge.value.getDefiningOp<memref::AllocOp>()) {
          auto allocOp = outEdge.value.getDefiningOp<memref::AllocOp>();
          if (!llvm::is_contained(dstNode->allocOps, allocOp)) {
            dstNode->allocOps.push_back(allocOp);
          }
        }
      }
    }
  }
  if (inEdges.count(srcId) > 0) {
    SmallVector<Edge, 2> oldInEdges = inEdges[srcId];
    for (auto &inEdge : oldInEdges) {
      removeEdge(inEdge.id, srcId, inEdge.value);

      unsigned newSrcId = (inEdge.id == srcId) ? dstId : inEdge.id;
      if (newSrcId != dstId) {
        addEdge(newSrcId, dstId, inEdge.value, inEdge.kind, inEdge.resolvedType,
                inEdge.srcOp, inEdge.dstOp, inEdge.srcFifoInfo, inEdge.dstFifoInfo);
      } else {
        // Self-loop (internal buffer or internal stream)
        addEdge(dstId, dstId, inEdge.value, inEdge.kind, inEdge.resolvedType,
                inEdge.srcOp, inEdge.dstOp, inEdge.srcFifoInfo, inEdge.dstFifoInfo);
        if (inEdge.value && inEdge.value.getDefiningOp<memref::AllocOp>()) {
          auto allocOp = inEdge.value.getDefiningOp<memref::AllocOp>();
          if (!llvm::is_contained(dstNode->allocOps, allocOp)) {
            dstNode->allocOps.push_back(allocOp);
          }
        }
      }
    }
  }
  // Move all internal ops from 'srcId' to 'dstId'.
  for (auto allocOp : srcNode->allocOps) {
    if (!llvm::is_contained(dstNode->allocOps, allocOp)) {
      dstNode->allocOps.push_back(allocOp);
    }
  }
  dstNode->forOps.append(srcNode->forOps.begin(), srcNode->forOps.end());
  dstNode->wrappedModules.append(srcNode->wrappedModules.begin(),
                                 srcNode->wrappedModules.end());
  NodeKind newKind;
  if (dstNode->kind == NodeKind::COMPUTE && srcNode->kind == NodeKind::COMPUTE) {
    newKind = NodeKind::COMPUTE;
  } else if (dstNode->kind != srcNode->kind) {
    newKind = NodeKind::SUPER_NODE;
  } else {
    newKind = NodeKind::UNKNOWN;
  }
  if (force)
    newKind = NodeKind::SUPER_NODE;
  dstNode->kind = newKind;

  // Aggregate performance model fields
  dstNode->arithWeight += srcNode->arithWeight;
  
  // if (dstNode->minTripCounts.empty()) {
  //   dstNode->minTripCounts = srcNode->minTripCounts;
  // } else if (!srcNode->minTripCounts.empty()) {
  //   unsigned minSize = std::min(dstNode->minTripCounts.size(), srcNode->minTripCounts.size());
  //   for (unsigned i = 0; i < minSize; ++i) {
  //     dstNode->minTripCounts[i] = std::min(dstNode->minTripCounts[i], srcNode->minTripCounts[i]);
  //   }
  // }

  // if (dstNode->maxTripCounts.empty()) {
  //   dstNode->maxTripCounts = srcNode->maxTripCounts;
  // } else if (!srcNode->maxTripCounts.empty()) {
  //   unsigned minSize = std::min(dstNode->maxTripCounts.size(), srcNode->maxTripCounts.size());
  //   for (unsigned i = 0; i < minSize; ++i) {
  //     dstNode->maxTripCounts[i] = std::max(dstNode->maxTripCounts[i], srcNode->maxTripCounts[i]);
  //   }
  // }
  // reverse the order of ops in dstNode
  // std::reverse(dstNode->ops.begin(), dstNode->ops.end());
  // // alloc ops need to be always first
  // std::stable_sort(dstNode->ops.begin(), dstNode->ops.end(), [](Operation *a, Operation *b){
  //   return isa<memref::AllocOp>(a) && !isa<memref::AllocOp>(b);
  // });
  // Remove 'srcId' from the graph.
  removeNode(srcId);
  return dstNode;
}

bool DFG::mergeNodesByHwMapping(
    const llvm::StringMap<SmallVector<unsigned>> &hwNameToNodes) {

  // 1) Compute topological order *before* any merging so we can sort forOps
  //    inside each super-node in dependency order.
  SmallVector<unsigned> topoOrder = topologicalSort();
  DenseMap<unsigned, unsigned> topoRank;
  for (unsigned idx = 0; idx < topoOrder.size(); ++idx)
    topoRank[topoOrder[idx]] = idx;

  LLVM_DEBUG(llvm::dbgs() << "--- mergeNodesByHwMapping ---\n");
  LLVM_DEBUG(llvm::dbgs() << "Topological order: ");
  LLVM_DEBUG(for (unsigned id : topoOrder) llvm::dbgs() << id << " ");
  LLVM_DEBUG(llvm::dbgs() << "\n");

  // 2) Build a nodeId → keepId remapping for all groups, and collect which
  //    nodes belong to which group (so we can detect inter-group edges).
  DenseMap<unsigned, unsigned> nodeToKeepId;
  for (const auto &entry : hwNameToNodes) {
    SmallVector<unsigned> groupIds(entry.getValue().begin(),
                                   entry.getValue().end());
    if (groupIds.size() <= 1) {
      nodeToKeepId[groupIds[0]] = groupIds[0];
      continue;
    }

    // Sort by topo rank — first element becomes the keepId.
    llvm::sort(groupIds, [&](unsigned a, unsigned b) {
      return topoRank.lookup(a) < topoRank.lookup(b);
    });

    unsigned keepId = groupIds[0];
    if (getNode(keepId)->kind == NodeKind::COMPUTE) {
      for (unsigned id : groupIds)
        nodeToKeepId[id] = keepId;
    }
  }

  // 3) Snapshot all edges *before* merging. mergeNodes() drops alloc-valued
  //    edges (by design), so inter-group RAW-dependency edges would be lost.
  //    We save them and re-add the inter-group ones after merging.
  struct SavedEdge {
    unsigned srcId, dstId;
    Value value;
    AffineStoreOp srcOp;
    AffineLoadOp dstOp;
  };
  SmallVector<SavedEdge> savedEdges;
  for (const auto &pair : inEdges) {
    unsigned dstId = pair.first;
    for (const auto &e : pair.second) {
      savedEdges.push_back({e.id, dstId, e.value, (AffineStoreOp)e.srcOp, (AffineLoadOp)e.dstOp});
    }
  }

  // 4) Perform the actual group merges.
  for (const auto &entry : hwNameToNodes) {
    StringRef hwName = entry.getKey();
    SmallVector<unsigned> groupIds(entry.getValue().begin(),
                                   entry.getValue().end());
    if (groupIds.size() <= 1)
      continue;

    llvm::sort(groupIds, [&](unsigned a, unsigned b) {
      return topoRank.lookup(a) < topoRank.lookup(b);
    });

    LLVM_DEBUG({
      llvm::dbgs() << "Merging group \"" << hwName << "\" nodes: [";
      for (unsigned i = 0; i < groupIds.size(); ++i) {
        if (i) llvm::dbgs() << ", ";
        llvm::dbgs() << groupIds[i];
      }
      llvm::dbgs() << "]\n";
    });

    unsigned keepId = groupIds[0];
    if (getNode(keepId)->kind != NodeKind::COMPUTE) {
      LLVM_DEBUG(llvm::dbgs() << "Skipping merge for communication group \"" << hwName << "\"\n");
      continue;
    }

    for (unsigned i = 1; i < groupIds.size(); ++i) {
      mergeNodes(groupIds[i], keepId);
    }

    // Set merged node type to the hw group name (strip _unit_N suffix).
    Node *superNode = getNode(keepId);
    if (superNode) {
      std::string baseType = hwName.str();
      auto unitPos = baseType.rfind("_unit_");
      if (unitPos != std::string::npos)
        baseType = baseType.substr(0, unitPos);
      superNode->type = baseType;
    }
  }

  // 5) Re-add inter-group edges that were dropped by mergeNodes().
  //    Retarget node IDs using the remap table.
  for (const auto &se : savedEdges) {
    unsigned newSrc = nodeToKeepId.count(se.srcId) ? nodeToKeepId[se.srcId]
                                                    : se.srcId;
    unsigned newDst = nodeToKeepId.count(se.dstId) ? nodeToKeepId[se.dstId]
                                                    : se.dstId;
    // Skip self-loops.
    if (newSrc == newDst) continue;
    
    // Check if this edge already exists (avoid duplicates).
    bool alreadyExists = false;
    if (inEdges.count(newDst)) {
      for (const auto &existing : inEdges[newDst]) {
        if (existing.id == newSrc && existing.value == se.value) {
          alreadyExists = true;
          break;
        }
      }
    }
    if (!alreadyExists)
      addEdge(newSrc, newDst, se.srcOp, se.dstOp, se.value);
  }

  // 6) Also update the flat edges vector for any code that uses it.
  for (auto &e : edges) {
    if (nodeToKeepId.count(e.srcId)) e.srcId = nodeToKeepId[e.srcId];
    if (nodeToKeepId.count(e.dstId)) e.dstId = nodeToKeepId[e.dstId];
  }
  // Remove self-loop edges.
  edges.erase(
      std::remove_if(edges.begin(), edges.end(),
                     [](const MyEdge &e) { return e.srcId == e.dstId; }),
      edges.end());

  LLVM_DEBUG(llvm::dbgs() << "After merging: " << nodes.size() << " nodes, "
                          << edges.size() << " edges\n");
  return true;
}

static void printVector(SmallVectorImpl<unsigned> &v){
  for(auto i : v){
    LLVM_DEBUG(llvm::dbgs() << i << " ");
  }
  LLVM_DEBUG(llvm::dbgs() << "\n");
}
static void printVector(SmallVectorImpl<int64_t> &v){
  for(auto i : v){
    LLVM_DEBUG(llvm::dbgs() << i << " ");
  }
  LLVM_DEBUG(llvm::dbgs() << "\n");
}

// induction variable struct
struct IV{
  Value value;
  int64_t lb;
  int64_t ub;
  int64_t tripCount;
};

void DFG::topologicalSortUtil(unsigned currId, std::set<unsigned> &visited, std::stack<unsigned> &stack){
  // mark currId as visited
  visited.insert(currId);
  // for each edge from currId
  for(auto edge : outEdges[currId]){
    // if edge.dstId is not visited
    if(visited.find(edge.id) == visited.end()){
      // call recursive DFS
      topologicalSortUtil(edge.id, visited, stack);
    }
  }
  // push currId to stack
  stack.push(currId);
}

SmallVector<unsigned> DFG::topologicalSort(){
  // visited
  std::set<unsigned> visited;
  // stack
  std::stack<unsigned> stack;
  // for each node
  for(auto node : nodes){
    // if node is not visited
    if(visited.find(node.first) == visited.end()){
      // call recursive DFS
      topologicalSortUtil(node.first, visited, stack);
    }
  }
  // create vector from stack
  SmallVector<unsigned> sorted;
  while(!stack.empty()){
    sorted.push_back(stack.top());
    stack.pop();
  }
  return sorted;
}

void DFG::virtualTopologicalSortUtil(VirtualNode curr, std::set<VirtualNode> &visited, std::stack<VirtualNode> &stack) {
  visited.insert(curr);

  // 1. Intra-node sequential edge: i -> i+1
  Node *node = getNode(curr.nodeId);
  if (node && curr.loopIdx + 1 < node->forOps.size()) {
    VirtualNode next{curr.nodeId, curr.loopIdx + 1};
    if (visited.find(next) == visited.end()) {
      virtualTopologicalSortUtil(next, visited, stack);
    }
  }

  // 2. Inter-node edges from current virtual node
  if (outEdges.count(curr.nodeId)) {
    for (auto &edge : outEdges[curr.nodeId]) {
      // Find srcLoopIdx of this edge
      int srcIdx = -1;
      if (edge.srcOp) {
        srcIdx = getLoopIndex(curr.nodeId, edge.srcOp);
      } else if (curr.loopIdx == 0) {
        // For StreamOp edges which might not have srcOp, assume loop 0 for now as fallback
        // (This might need more care for merged nodes with streams)
        srcIdx = 0; 
      }

      if (srcIdx == (int)curr.loopIdx) {
        // This edge starts from current virtual node
        int dstIdx = 0;
        if (edge.dstOp) {
          dstIdx = getLoopIndex(edge.id, edge.dstOp);
        }
        if (dstIdx != -1) {
          VirtualNode next{edge.id, (unsigned)dstIdx};
          if (visited.find(next) == visited.end()) {
            virtualTopologicalSortUtil(next, visited, stack);
          }
        }
      }
    }
  }

  stack.push(curr);
}

SmallVector<DFG::VirtualNode> DFG::virtualTopologicalSort() {
  std::set<VirtualNode> visited;
  std::stack<VirtualNode> stack;
  // Prioritize lower loop indices and topological order if possible
  for (auto &nodePair : nodes) {
    unsigned nodeId = nodePair.first;
    for (unsigned i = 0; i < nodePair.second.forOps.size(); ++i) {
      VirtualNode vn{nodeId, i};
      if (visited.find(vn) == visited.end()) {
        virtualTopologicalSortUtil(vn, visited, stack);
      }
    }
  }
  SmallVector<VirtualNode> sorted;
  while (!stack.empty()) {
    sorted.push_back(stack.top());
    stack.pop();
  }
  return sorted;
}

int DFG::getLoopIndex(unsigned nodeId, Operation *op) {
  Node *node = getNode(nodeId);
  if (!node || node->forOps.empty()) return -1;
  
  auto currForOp = op->getParentOfType<AffineForOp>();
  if (!currForOp) return -1;
  
  // Find the band header
  AffineLoopBand band;
  getLoopBandFromInnermost(currForOp, band);
  if (band.empty()) return -1;
  auto bandHeader = band.front();

  for (unsigned i = 0; i < node->forOps.size(); ++i) {
    if (node->forOps[i] == bandHeader) return (int)i;
  }
  return -1;
}



static OperationName getOpsBetweenLoadAndStore(
  AffineLoadOp loadOp, AffineStoreOp storeOp) {
  auto currValue = loadOp.getResult();
  SmallVector<Operation *, 8> opsBetweenLoadAndStore;
  for(auto user : currValue.getUsers()){
    // user->dump();
    opsBetweenLoadAndStore.push_back(user);
  }
  if(opsBetweenLoadAndStore.size() == 1){
    return opsBetweenLoadAndStore[0]->getName();
  }else{
    assert(false && "opsBetweenLoadAndStore.size() != 1");
  }
}

static uint64_t getForLoopII(AffineForOp forOp){
  // Collect all load and store ops in loop nest rooted at 'forOp'.
  SmallVector<AffineStoreOp, 1> storeOps;
  SmallVector<AffineLoadOp, 1> loadOps;
  auto walkResult = forOp.walk([&](Operation *op) -> WalkResult {
    if (auto readOp = dyn_cast<AffineReadOpInterface>(op)) {
      loadOps.push_back(cast<AffineLoadOp>(op));
    } else if (auto writeOp = dyn_cast<AffineWriteOpInterface>(op)) {
      storeOps.push_back(cast<AffineStoreOp>(op));
    } 
    return WalkResult::advance();
  });
  auto II = 1;
  for(auto storeOp : storeOps){
    auto storeMemref = storeOp.getMemRef();
    auto storeBlock = storeOp->getBlock();
    auto storeAffineMap = storeOp.getAffineMap();
    auto storeMapOperands = storeOp.getMapOperands();
    for(auto loadOp : loadOps){
      auto loadMemref = loadOp.getMemRef();
      auto loadBlock = loadOp->getBlock();
      auto loadAffineMap = loadOp.getAffineMap();
      auto loadMapOperands = loadOp.getMapOperands();
      // to make sure the store and load are in the same block and have the same memref
      if(storeMemref == loadMemref && storeBlock == loadBlock){
        auto innerMostLoop = storeOp->getParentOfType<AffineForOp>();
        auto innerMostIV = innerMostLoop.getInductionVar();
        SmallVector<Operation *, 8> opsBetweenLoadAndStore;
        auto opName = getOpsBetweenLoadAndStore(cast<AffineLoadOp>(loadOp), cast<AffineStoreOp>(storeOp));
        auto latency = (opName.getStringRef() == "arith.addf") ? 4 : 2;
        II = latency;
        for(auto operand : storeMapOperands){
          if(operand == innerMostIV){
            II = 1;
          }
        }
      }
    }
  }
  if(II == -1) {
    assert(false && "II == -1");
  }
  return II;
}

static uint64_t getForLoopII(AffineForOp forOp, SmallVectorImpl<unsigned>&permMap){
  // permute loop 
  AffineLoopBand band;
  getLoopBandFromOutermost(forOp, band);
  auto newRoot = band[permuteLoops(band, permMap)];
  band.clear();
  getLoopBandFromOutermost(newRoot, band);
  auto II = getForLoopII(newRoot);
  // undo the permutation
  newRoot = band[permuteLoops(band, inversePerm(permMap))];
  band.clear();
  getLoopBandFromOutermost(newRoot, band);
  return II;
}
static uint64_t factorial(unsigned int n) {
    std::vector<unsigned int> numbers(n);
    std::iota(numbers.begin(), numbers.end(), 1);
    return std::accumulate(numbers.begin(), numbers.end(), 1ull, std::multiplies<unsigned long long>());
}

static bool isSubset(SmallVectorImpl<unsigned>& group, SmallVectorImpl<unsigned>& permMap){
  for(auto elem : group){
    if(std::find(permMap.begin(), permMap.end(), elem) == permMap.end()){
      return false;
    }
  }
  return true;
}

// the group is not ordered and the permMap is not ordered
static bool isContiguousSubset(SmallVectorImpl<unsigned>& group, SmallVectorImpl<unsigned>& permMap){
  if(group.size() == 0){
    return false;
  }
  if(permMap.size() == 0){
    return false;
  }
  auto firstElem = group[0];
  auto firstElemIdx = std::find(permMap.begin(), permMap.end(), firstElem);
  if(firstElemIdx == permMap.end()){
    return false;
  }
  for(unsigned i = 1; i < group.size(); i++){
    auto elem = group[i];
    auto elemIdx = std::find(permMap.begin(), permMap.end(), elem);
    if(elemIdx == permMap.end()){
      return false;
    }
    if(elemIdx - firstElemIdx != i){
      return false;
    }
  }
  return true;
}
static uint64_t getNumberOfDivisors(uint64_t num){
  uint64_t count = 0;
  for(uint64_t i = 1; i <= num; i++){
    if(num % i == 0){
      count++;
    }
  }
  return count;
}

static uint64_t getNumberOfDivisorTilingFactors(AffineLoopBand band){
  uint64_t numDivisors = 1;
  for(auto loop : band){
    auto tripCount = loop.getConstantUpperBound();
    numDivisors *= getNumberOfDivisors(tripCount);
  }
  return numDivisors;
}

LogicalResult DFG::populateNodeInfo(bool enablePermutations){
  unsigned varIdx = 0;
  uint64_t permDesignSpaceSize = 1;
  uint64_t tilingDesignSpaceSize = 1;
  initializeTilingVars();
  populateTilingVarsWithOnes();
  
  for(auto& nodePair : nodes){
    auto id = nodePair.first;
    if (id >= 10000) continue; 

    auto& node = nodePair.second;
    auto forOp = dyn_cast<AffineForOp>(node.op);
    if(!forOp){
      assert(false && "node is not a for op");
    }
    forOp->walk([&](Operation *op) {
      if(isa<arith::AddFOp, arith::SubFOp>(op)){
        node.DSP_factor += 2;
      }
      if(isa<arith::MulFOp>(op)){
        node.DSP_factor += 3;
      }
      // erf op
      if(isa<math::ErfOp>(op)){
        node.DSP_factor += 25;
      }
    });

    // collect stores and loads
    auto inputs = inEdges[id];
    auto outputs = outEdges[id];
    for(auto inEdge : inputs){
      auto value = inEdge.value;
      if (!value) continue;
      for(auto user : value.getUsers()){
        if(auto loadOp = dyn_cast<AffineLoadOp>(user)){
          node.loads.push_back(loadOp);
        }
      }
    }
    for(auto outEdge : outputs){
      auto value = outEdge.value;
      if (!value) continue;
      for(auto user : value.getUsers()){
        if(auto storeOp = dyn_cast<AffineStoreOp>(user)){
          node.stores.push_back(storeOp);
        }
      }
    }


    LLVM_DEBUG(
      llvm::dbgs() << "node: " << id << " has " << node.loads.size() << " loads and " << node.stores.size() << " stores\n";
    );
    // // for each store and each load, print the operation
    // for(auto loadOp : node.loads){
    //     llvm::dbgs() << "loadOp: " << loadOp << "\n";
    // }
    // for(auto storeOp : node.stores){
    //     llvm::dbgs() << "storeOp: " << storeOp << "\n";
    // }

    // 1. Map operations to their parent forOps
    DenseMap<Operation *, AffineForOp> opToParentFor;
    for (auto fOp : node.forOps) {
      fOp.walk([&](Operation *inner) {
        if (isa<AffineLoadOp, AffineStoreOp>(inner)) {
          opToParentFor[inner] = fOp;
        }
      });
    }

    forOp = node.forOps.front();
    AffineLoopBand probeBand;
    getLoopBandFromOutermost(forOp, probeBand);
    node.bandSize = probeBand.size();

    SmallVector<std::string> arithOps;
    forOp->walk([&](Operation *op) {
      auto dialect = op->getDialect();
      if (dialect->getNamespace() == "arith" || dialect->getNamespace() == "math") {
        auto opNameWithDialect = op->getName().getStringRef().str();
        arithOps.push_back(opNameWithDialect);
      }
    });
    arithOps.push_back("depth_" + std::to_string(probeBand.size()));
    node.type = llvm::join(arithOps, "_");
    node.isReduction = isReductionLoopBand(probeBand);

    // original permutation map
    SmallVector<unsigned> permMap;
    for (unsigned i = 0; i < probeBand.size(); i++) {
      permMap.push_back(i);
    }
    Node* currNode = getNode(id);
    auto allPerms = std::vector<SmallVector<unsigned>>();
    if(enablePermutations){
      do {
        allPerms.push_back(permMap);
      } while (std::next_permutation(permMap.begin(), permMap.end()));
    }else{
      allPerms.push_back(permMap);
    }

    permDesignSpaceSize *= allPerms.size();
    tilingDesignSpaceSize *= getNumberOfDivisorTilingFactors(probeBand);

    int permIdx = 0;
    currNode->candidates.permutationCandidates.resize(allPerms.size());

    SmallVector<LoopBandInfo> loopBandInfos;
    for (auto fOp : node.forOps) {
      AffineLoopBand band;
      getLoopBandFromOutermost(fOp, band);
      LoopBandInfo loopBandInfo;
      int depth = 0;
      for(auto forLoop : band){
        LoopInfo loopInfo;
        loopInfo.iv = forLoop.getInductionVar();
        loopInfo.depth = depth++;
        loopInfo.lb = forLoop.getConstantLowerBound();
        loopInfo.ub = forLoop.getConstantUpperBound();
        loopInfo.step = forLoop.getStepAsInt();
        loopInfo.level = 0; 
        loopInfo.originalIndex = depth - 1; 
        loopInfo.name = "x" + std::to_string(id) + "_" + std::to_string(loopInfo.originalIndex);
        loopBandInfo.push_back(loopInfo);
      }
      loopBandInfos.push_back(loopBandInfo);
    }

    for(auto permMap : allPerms){
      auto inversePermMap = inversePerm(permMap);
      currNode->candidates.permutationCandidates[permIdx] = PermutationMemrefCandidates();
      currNode->candidates.permutationCandidates[permIdx].permutation = permMap;
      currNode->candidates.permutationCandidates[permIdx].II = 1;
      int loopIdx = 0;
      for (auto fOp : node.forOps) {
        AffineLoopBand band;
        getLoopBandFromOutermost(fOp, band);

        // do the permutation
        auto newRoot = band[permuteLoops(band, permMap)];
        band.clear();
        getLoopBandFromOutermost(newRoot, band);

        // Update II if needed (SUS)
        int candidateII = getForLoopII(newRoot);
        if (candidateII > currNode->candidates.permutationCandidates[permIdx].II)
          currNode->candidates.permutationCandidates[permIdx].II = candidateII;

        // Populate maps for each load and store operation belonging to this forOp
        // auto loopBandInfo = loopBandInfos[loopIdx];
        LoopBandInfo loopBandInfo;
        int depth = 0;
        for(auto forLoop : band){
          LoopInfo loopInfo;
          loopInfo.iv = forLoop.getInductionVar();
          loopInfo.depth = depth++;
          loopInfo.lb = forLoop.getConstantLowerBound();
          loopInfo.ub = forLoop.getConstantUpperBound();
          loopInfo.step = forLoop.getStepAsInt();
          loopInfo.level = 0; 
          loopInfo.originalIndex = depth - 1; 
          loopInfo.name = "x" + std::to_string(id) + "_" + std::to_string(loopInfo.originalIndex);
          loopBandInfo.push_back(loopInfo);
        }
        for(auto loadOp : currNode->loads){
          if (opToParentFor[loadOp] == fOp){
            currNode->candidates.permutationCandidates[permIdx].loadsMap[loadOp] = getMemrefCandidateInfo(loadOp, loopBandInfo, permMap, true);
          }
        }
        for(auto storeOp : currNode->stores){
          if (opToParentFor[storeOp] == fOp){
            currNode->candidates.permutationCandidates[permIdx].storesMap[storeOp] = getMemrefCandidateInfo(storeOp, loopBandInfo, permMap, true);
          }
        }

        // undo the permutation for this band
        newRoot = band[permuteLoops(band, inversePermMap)];
        loopIdx++;
      }
      permIdx++;
    } 
    currNode->bandSize = probeBand.size();
    varIdx += allPerms.size();
  }
  // llvm::dbgs() << "NumberOfVariables: " << varIdx << "\n";
  // llvm::dbgs() << "Permutation DesignSpaceSize: " << permDesignSpaceSize << "\n";
  // llvm::dbgs() << "Parallelization DesignSpaceSize: " << tilingDesignSpaceSize << "\n";
  // llvm::dbgs() << "Total DesignSpaceSize: " << permDesignSpaceSize * tilingDesignSpaceSize << "\n";
  return success();
}

bool DFG::createRootNode(){
  SmallVector<unsigned, 3> srcNodeIds;
  for(auto nodePair : nodes){
    auto id = nodePair.first;
    auto& node = nodePair.second;
    if(inEdges[id].size() == 0){
      srcNodeIds.push_back(id);
    }
  }
  Node root(10000, nullptr);
  root.candidates.permutationCandidates.push_back(PermutationMemrefCandidates());
  nodes.insert({root.id, root});
  for(auto dstId : srcNodeIds){
    if (!hasEdge(root.id, dstId, nullptr)) {
      outEdges[root.id].push_back({dstId, nullptr});
      inEdges[dstId].push_back({root.id, nullptr});
    }
  }
  return true;
}

bool DFG::createSinkNode(){
  SmallVector<unsigned, 3> sinkNodeIds;
  for(auto nodePair : nodes){
    auto id = nodePair.first;
    auto& node = nodePair.second;
    if(outEdges[id].size() == 0){
      sinkNodeIds.push_back(id);
    }
  }
  Node sink(10001, nullptr);
  sink.tripCount = 0;
  nodes.insert({sink.id, sink});
  for(auto srcId : sinkNodeIds){
    if (!hasEdge(srcId, sink.id, nullptr)) {
      outEdges[srcId].push_back({sink.id, nullptr});
      inEdges[sink.id].push_back({srcId, nullptr});
    }
  }
  return true;
}

void DFG::initializeTilingVars() {
  struct UF {
    std::vector<unsigned> parent;
    unsigned find(unsigned x) {
      if (parent[x] == x)
        return x;
      return parent[x] = find(parent[x]);
    }
    void unite(unsigned a, unsigned b) {
      a = find(a);
      b = find(b);
      if (a != b)
        parent[a] = b;
    }
  };

  UF uf;
  std::map<std::pair<unsigned, unsigned>, unsigned> loopToId;
  std::vector<std::pair<unsigned, unsigned>> idToLoop;

  auto getLid = [&](unsigned nodeId, unsigned level) {
    auto p = std::make_pair(nodeId, level);
    if (loopToId.count(p))
      return loopToId[p];
    unsigned id = idToLoop.size();
    loopToId[p] = id;
    idToLoop.push_back(p);
    uf.parent.push_back(id);
    return id;
  };

  // Map: (memref, dimension) -> list of UF LIDs
  DenseMap<std::pair<Value, unsigned>, std::vector<unsigned>> memrefDimToLids;

  for (auto &it : nodes) {
    Node &node = it.second;
    for (auto forOp : node.forOps) {
      AffineLoopBand band;
      getLoopBandFromOutermost(forOp, band);

      DenseMap<Value, unsigned> ivToLevel;
      for (unsigned i = 0; i < band.size(); ++i) {
        ivToLevel[band[i].getInductionVar()] = i;
      }

      forOp.walk([&](Operation *op) {
        auto handleAccess = [&](Value memref, AffineMap map,
                                ValueRange operands) {
          if (!memref)
            return;
          for (unsigned d = 0; d < map.getNumResults(); ++d) {
            AffineExpr expr = map.getResult(d);
            if (auto dimExpr = expr.dyn_cast<AffineDimExpr>()) {
              Value iv = operands[dimExpr.getPosition()];
              if (ivToLevel.count(iv)) {
                unsigned level = ivToLevel[iv];
                unsigned lid = getLid(node.id, level);
                memrefDimToLids[{memref, d}].push_back(lid);
              }
            }
          }
        };

        if (auto load = dyn_cast<AffineLoadOp>(op)) {
          handleAccess(load.getMemRef(), load.getAffineMap(),
                       load.getMapOperands());
        } else if (auto store = dyn_cast<AffineStoreOp>(op)) {
          handleAccess(store.getMemRef(), store.getAffineMap(),
                       store.getMapOperands());
        }
      });
    }
  }

  // Rule 1: Unite LIDs that share the same (memref, dimension)
  for (auto &entry : memrefDimToLids) {
    const auto &lids = entry.second;
    for (size_t i = 1; i < lids.size(); ++i) {
      uf.unite(lids[0], lids[i]);
    }
  }

  // Assign "X_#" names to each root
  std::map<unsigned, std::string> rootToName;
  unsigned nextVarId = 0;

  // Clear existing tilingVars
  tilingVars.clear();

  // We want a deterministic order for variable naming. Sort by (nodeId, level).
  for (auto const &[loop, lid] : loopToId) {
    unsigned root = uf.find(lid);
    if (rootToName.count(root) == 0) {
      std::string name = "X_" + std::to_string(nextVarId++);
      rootToName[root] = name;
      tilingVars[name] = -1;
    }
  }

  // Populate Node::candidates.tilingCandidates and tilingVars with trip counts
  for (auto &it : nodes) {
    Node &node = it.second;
    node.candidates.tilingCandidates.clear();
    if (node.forOps.empty()) continue;
    
    AffineLoopBand band;
    getLoopBandFromOutermost(node.forOps[0], band);

    for (unsigned l = 0; l < node.bandSize; ++l) {
      if (loopToId.count({node.id, l})) {
        unsigned lid = loopToId[{node.id, l}];
        unsigned root = uf.find(lid);
        std::string varName = rootToName[root];
        node.candidates.tilingCandidates.push_back(varName);

        // Populate tilingVars with trip counts (assuming they are fixed per group)
        if (l < band.size()) {
          auto tc = getConstTripCount(band[l]);
          if (tc.has_value()) {
            tilingVars[varName] = -1;
          }
        }
      } else {
        node.candidates.tilingCandidates.push_back("");
      }
    }
  }
}

static int64_t getGcd(int64_t a, int64_t b) {
  while (b) {
    a %= b;
    std::swap(a, b);
  }
  return a;
}

void DFG::populateTilingVarsRandomly() {
  std::map<std::string, int64_t> varToGcd;

  for (auto const &it : nodes) {
    const Node &node = it.second;
    if (node.forOps.empty())
      continue;

    AffineLoopBand band;
    getLoopBandFromOutermost(node.forOps[0], band);

    for (unsigned i = 0; i < node.candidates.tilingCandidates.size(); ++i) {
      std::string varName = node.candidates.tilingCandidates[i];
      if (varName.empty())
        continue;
      if (i >= band.size())
        continue;

      auto tripCount = getConstTripCount(band[i]);
      if (!tripCount.has_value())
        continue;
      int64_t tc = tripCount.value();

      if (varToGcd.count(varName) == 0) {
        varToGcd[varName] = tc;
      } else {
        varToGcd[varName] = getGcd(varToGcd[varName], tc);
      }
    }
  }

  std::mt19937 rng(time(0));

  for (auto &it : tilingVars) {
    std::string varName = it.first;
    if (varToGcd.count(varName) == 0)
      continue;

    int64_t commonGcd = varToGcd[varName];
    if (commonGcd <= 0) {
      it.second = 1;
      continue;
    }

    std::vector<int64_t> divisors;
    for (int64_t i = 1; i * i <= commonGcd; ++i) {
      if (commonGcd % i == 0) {
        divisors.push_back(i);
        if (i * i != commonGcd) {
          divisors.push_back(commonGcd / i);
        }
      }
    }

    if (divisors.empty()) {
      it.second = 1;
    } else {
      std::uniform_int_distribution<size_t> dist(0, divisors.size() - 1);
      it.second = divisors[dist(rng)];
    }
  }
  updateNodeTilingFactors();
}

void DFG::populateTilingVarsWithOnes() {
  for (auto &it : tilingVars) {
    it.second = 1;
  }
  updateNodeTilingFactors();
}

void DFG::updateNodeTilingFactors() {
  for (auto &it : nodes) {
    Node &node = it.second;
    node.solution.tilingFactors.clear();
    for (const std::string &varName : node.candidates.tilingCandidates) {
      if (!varName.empty() && tilingVars.count(varName)) {
        node.solution.tilingFactors.push_back(tilingVars[varName]);
      } else {
        assert(false && "failed updateNodeTilingFactors: varName not found in tilingVars");
      }
    }
  }
}
  
static void permuteModuleLoops(ModuleOp mod, const SmallVector<unsigned> &permMap) {
  if (!mod) return;
  AffineForOp modForOp;
  func::FuncOp modFuncOp;
  mod.walk([&](AffineForOp forOp) {
    modForOp = forOp;
  });
  mod.walk([&](func::FuncOp funcOp) {
    modFuncOp = funcOp;
  });
  AffineLoopBand band;
  getLoopBandFromOutermost(modForOp, band);
  auto newRoot = band[permuteLoops(band, permMap)];
  band.clear();
  getLoopBandFromOutermost(newRoot, band);
  
  // OpBuilder b(mod.getContext());
  // DictionaryAttr variantDict =
  //     buildVariantMetadataForFunc(b, /*srcFunc=*/modFuncOp,
  //                                     /*mergedFunc=*/modFuncOp,
  //                                     /*band=*/band);

  // mod->setAttr("merge.variants", b.getArrayAttr(variantDict));
}

bool DFG::applyNodePermutations(){
  for(auto& nodePair : nodes){
    auto& node = nodePair.second;
    if(node.id == 10000 || node.id == 10001)
      continue;
    // llvm::dbgs() << "Applying permutation for node: " << node.id << "\n";
    if(auto forOp = dyn_cast_or_null<AffineForOp>(node.op)){
      AffineLoopBand band;
      getLoopBandFromOutermost(forOp, band);
      LLVM_DEBUG(
        llvm::dbgs() << "band size: " << band.size() << "\n";
      );
      auto permIdx = node.solution.permIdx;
      auto& permMap = node.candidates.permutationCandidates[permIdx].permutation;
      if(permMap.size() > 0){
        auto newRoot = band[permuteLoops(band, permMap)];
        band.clear();
        getLoopBandFromOutermost(newRoot, band);
        // set the new loop band
        node.op = band[0];

        // Update forOps
        for (auto &fOp : node.forOps) {
          if (fOp == forOp) {
            fOp = band[0];
            break;
          }
        }

        // Propagate to wrapped modules and hwModule
        for (auto& mod : node.wrappedModules) {
          permuteModuleLoops(mod, permMap);
        }
        // permuate solution.tilingFactors vector based on permMap
        SmallVector<unsigned, 4> permutedTilingFactors(permMap.size());
        SmallVector<std::string, 4> permutedTilingVars(permMap.size());
        assert(permMap.size() == node.candidates.tilingCandidates.size() && "permMap size does not match tilingCandidates size");
        assert(permMap.size() == node.solution.tilingFactors.size() && "permMap size does not match tilingFactors size");
        for (unsigned i = 0; i < permMap.size(); ++i) {
          permutedTilingVars[permMap[i]] = node.candidates.tilingCandidates[i];
          permutedTilingFactors[permMap[i]] = node.solution.tilingFactors[i];
        }
        node.solution.tilingFactors.clear();
        node.solution.tilingFactors.append(permutedTilingFactors.begin(), permutedTilingFactors.end());
        node.candidates.tilingCandidates.clear();
        node.candidates.tilingCandidates.append(permutedTilingVars.begin(), permutedTilingVars.end());
        
        
        // first module of wrapped modules is the hwModule, so no need to permute hwModule separately
        // if (node.hwModule) {
        //   permuteModuleLoops(node.hwModule, permMap);
        // }else{
        //   assert(false && "node has no hwModule");
        // }
      }
    } else {
      LLVM_DEBUG(
        llvm::dbgs() << "node is not a for op or is null\n";
      );
    }
  }
  return true;
}

// Returns the graph node for 'id'.
Node *DFG::getNode(unsigned id) {
  auto it = nodes.find(id);
  assert(it != nodes.end());
  return &it->second;
}

// Adds a node with 'op' to the graph and returns its unique identifier.
unsigned DFG::addNode(Operation *op) {
  Node node(nextNodeId++, op);
  nodes.insert({node.id, node});
  return node.id;
}

// Remove node 'id' (and its associated edges) from graph.
void DFG::removeNode(unsigned id) {
  // Remove each edge in 'inEdges[id]'.
  if (inEdges.count(id) > 0) {
    SmallVector<Edge, 2> oldInEdges = inEdges[id];
    for (auto &inEdge : oldInEdges) {
      removeEdge(inEdge.id, id, inEdge.value);
    }
  }
  // Remove each edge in 'outEdges[id]'.
  if (outEdges.count(id) > 0) {
    SmallVector<Edge, 2> oldOutEdges = outEdges[id];
    for (auto &outEdge : oldOutEdges) {
      removeEdge(id, outEdge.id, outEdge.value);
    }
  }
  // Erase remaining node state.
  inEdges.erase(id);
  outEdges.erase(id);
  nodes.erase(id);
}

// Returns true iff there is an edge from node 'srcId' to node 'dstId' which
// is for 'value' if non-null, or for any value otherwise. Returns false
// otherwise.
bool DFG::hasEdge(unsigned srcId, unsigned dstId,
                                    Value value) {
  if (outEdges.count(srcId) == 0 || inEdges.count(dstId) == 0) {
    return false;
  }
  bool hasOutEdge = llvm::any_of(outEdges[srcId], [=](Edge &edge) {
    return edge.id == dstId && (!value || edge.value == value);
  });
  bool hasInEdge = llvm::any_of(inEdges[dstId], [=](Edge &edge) {
    return edge.id == srcId && (!value || edge.value == value);
  });
  return hasOutEdge && hasInEdge;
}

// Adds an edge from node 'srcId' to node 'dstId' for 'value'.
void DFG::addEdge(unsigned srcId, unsigned dstId,
                                    Value value) {
  if (!hasEdge(srcId, dstId, value)) {
    EdgeKind kind = EdgeKind::MEMREF;
    if (isa<dataflow::StreamType>(value.getType()))
      kind = EdgeKind::STREAM;
    else if (isa<RankedTensorType>(value.getType()) && 
             isa<dataflow::StreamType>(mlir::cast<RankedTensorType>(value.getType()).getElementType()))
      kind = EdgeKind::AOS;
    outEdges[srcId].push_back({dstId, value, nullptr, nullptr, kind, value.getType(), 2, nullptr, nullptr});
    inEdges[dstId].push_back({srcId, value, nullptr, nullptr, kind, value.getType(), 2, nullptr, nullptr});
    if (isa<MemRefType>(value.getType()))
      memrefEdgeCount[value]++;
  }
}

// Adds an edge from node 'srcId' to node 'dstId' for 'value'.
void DFG::addEdge(
  unsigned srcId, unsigned dstId, 
  AffineStoreOp srcOp, AffineLoadOp dstOp,
  Value value
) {
  EdgeKind kind = EdgeKind::MEMREF;
  if (isa<dataflow::StreamType>(value.getType()))
    kind = EdgeKind::STREAM;
  else if (isa<RankedTensorType>(value.getType()) && 
           isa<dataflow::StreamType>(mlir::cast<RankedTensorType>(value.getType()).getElementType()))
    kind = EdgeKind::AOS;

  addEdge(srcId, dstId, value, kind, value.getType(), srcOp, dstOp, nullptr, nullptr);
}

void DFG::addEdge(unsigned srcId, unsigned dstId, Value value, EdgeKind kind,
                   Type resolvedType, AffineStoreOp srcOp,
                   AffineLoadOp dstOp, DictionaryAttr srcFifoInfo,
                   DictionaryAttr dstFifoInfo) {
  if (!hasEdge(srcId, dstId, value)) {
    outEdges[srcId].push_back({dstId, value, srcOp, dstOp, kind, resolvedType, 2, srcFifoInfo, dstFifoInfo});
    inEdges[dstId].push_back({srcId, value, srcOp, dstOp, kind, resolvedType, 2, srcFifoInfo, dstFifoInfo});
    if (value && isa<MemRefType>(value.getType()))
      memrefEdgeCount[value]++;
  }
}

// Removes an edge from node 'srcId' to node 'dstId' for 'value'.
void DFG::removeEdge(unsigned srcId, unsigned dstId,
                                       Value value) {
  assert(inEdges.count(dstId) > 0);
  assert(outEdges.count(srcId) > 0);
  if (value && isa<MemRefType>(value.getType())) {
    assert(memrefEdgeCount.count(value) > 0);
    memrefEdgeCount[value]--;
  }
  // Remove 'srcId' from 'inEdges[dstId]'.
  for (auto *it = inEdges[dstId].begin(); it != inEdges[dstId].end(); ++it) {
    if ((*it).id == srcId && (*it).value == value) {
      inEdges[dstId].erase(it);
      break;
    }
  }
  // Remove 'dstId' from 'outEdges[srcId]'.
  for (auto *it = outEdges[srcId].begin(); it != outEdges[srcId].end(); ++it) {
    if ((*it).id == dstId && (*it).value == value) {
      outEdges[srcId].erase(it);
      break;
    }
  }
}

// Returns true if there is a path in the dependence graph from node 'srcId'
// to node 'dstId'. Returns false otherwise. `srcId`, `dstId`, and the
// operations that the edges connected are expected to be from the same block.
bool DFG::hasDependencePath(unsigned srcId, unsigned dstId) {
  // Worklist state is: <node-id, next-output-edge-index-to-visit>
  SmallVector<std::pair<unsigned, unsigned>, 4> worklist;
  worklist.push_back({srcId, 0});
  Operation *dstOp = getNode(dstId)->op;
  // Run DFS traversal to see if 'dstId' is reachable from 'srcId'.
  while (!worklist.empty()) {
    auto &idAndIndex = worklist.back();
    // Return true if we have reached 'dstId'.
    if (idAndIndex.first == dstId)
      return true;
    // Pop and continue if node has no out edges, or if all out edges have
    // already been visited.
    if (outEdges.count(idAndIndex.first) == 0 ||
        idAndIndex.second == outEdges[idAndIndex.first].size()) {
      worklist.pop_back();
      continue;
    }
    // Get graph edge to traverse.
    Edge edge = outEdges[idAndIndex.first][idAndIndex.second];
    // Increment next output edge index for 'idAndIndex'.
    ++idAndIndex.second;
    // Add node at 'edge.id' to the worklist. We don't need to consider
    // nodes that are "after" dstId in the containing block; one can't have a
    // path to `dstId` from any of those nodes.
    bool afterDst = dstOp->isBeforeInBlock(getNode(edge.id)->op);
    if (!afterDst && edge.id != idAndIndex.first)
      worklist.push_back({edge.id, 0});
  }
  return false;
}

// // Returns the input edge count for node 'id' and 'memref' from src nodes
// // which access 'memref' with a store operation.
// unsigned DFG::getIncomingMemRefAccesses(unsigned id,
//                                                           Value memref) {
//   unsigned inEdgeCount = 0;
//   if (inEdges.count(id) > 0)
//     for (auto &inEdge : inEdges[id])
//       if (inEdge.value == memref) {
//         Node *srcNode = getNode(inEdge.id);
//         // Only count in edges from 'srcNode' if 'srcNode' accesses 'memref'
//         if (srcNode->getStoreOpCount(memref) > 0)
//           ++inEdgeCount;
//       }
//   return inEdgeCount;
// }

// Returns the output edge count for node 'id' and 'memref' (if non-null),
// otherwise returns the total output edge count from node 'id'.
unsigned DFG::getOutEdgeCount(unsigned id, Value memref) {
  unsigned outEdgeCount = 0;
  if (outEdges.count(id) > 0)
    for (auto &outEdge : outEdges[id])
      if (!memref || outEdge.value == memref)
        ++outEdgeCount;
  return outEdgeCount;
}

/// Return all nodes which define SSA values used in node 'id'.
void DFG::gatherDefiningNodes(
    unsigned id, DenseSet<unsigned> &definingNodes) {
  for (DFG::Edge edge : inEdges[id])
    // By definition of edge, if the edge value is a non-memref value,
    // then the dependence is between a graph node which defines an SSA value
    // and another graph node which uses the SSA value.
    if (edge.value && !isa<MemRefType>(edge.value.getType()))
      definingNodes.insert(edge.id);
}

// Computes and returns an insertion point operation, before which the
// the fused <srcId, dstId> loop nest can be inserted while preserving
// dependences. Returns nullptr if no such insertion point is found.
Operation *
DFG::getFusedLoopNestInsertionPoint(unsigned srcId,
                                                      unsigned dstId) {
  if (outEdges.count(srcId) == 0)
    return getNode(dstId)->op;

  // Skip if there is any defining node of 'dstId' that depends on 'srcId'.
  DenseSet<unsigned> definingNodes;
  gatherDefiningNodes(dstId, definingNodes);
  if (llvm::any_of(definingNodes,
                   [&](unsigned id) { return hasDependencePath(srcId, id); })) {
    LLVM_DEBUG(llvm::dbgs()
               << "Can't fuse: a defining op with a user in the dst "
                  "loop has dependence from the src loop\n");
    return nullptr;
  }

  // Build set of insts in range (srcId, dstId) which depend on 'srcId'.
  SmallPtrSet<Operation *, 2> srcDepInsts;
  for (auto &outEdge : outEdges[srcId])
    if (outEdge.id != dstId)
      srcDepInsts.insert(getNode(outEdge.id)->op);

  // Build set of insts in range (srcId, dstId) on which 'dstId' depends.
  SmallPtrSet<Operation *, 2> dstDepInsts;
  for (auto &inEdge : inEdges[dstId])
    if (inEdge.id != srcId)
      dstDepInsts.insert(getNode(inEdge.id)->op);

  Operation *srcNodeInst = getNode(srcId)->op;
  Operation *dstNodeInst = getNode(dstId)->op;

  // Computing insertion point:
  // *) Walk all operation positions in Block operation list in the
  //    range (src, dst). For each operation 'op' visited in this search:
  //   *) Store in 'firstSrcDepPos' the first position where 'op' has a
  //      dependence edge from 'srcNode'.
  //   *) Store in 'lastDstDepPost' the last position where 'op' has a
  //      dependence edge to 'dstNode'.
  // *) Compare 'firstSrcDepPos' and 'lastDstDepPost' to determine the
  //    operation insertion point (or return null pointer if no such
  //    insertion point exists: 'firstSrcDepPos' <= 'lastDstDepPos').
  SmallVector<Operation *, 2> depInsts;
  std::optional<unsigned> firstSrcDepPos;
  std::optional<unsigned> lastDstDepPos;
  unsigned pos = 0;
  for (Block::iterator it = std::next(Block::iterator(srcNodeInst));
       it != Block::iterator(dstNodeInst); ++it) {
    Operation *op = &(*it);
    if (srcDepInsts.count(op) > 0 && firstSrcDepPos == std::nullopt)
      firstSrcDepPos = pos;
    if (dstDepInsts.count(op) > 0)
      lastDstDepPos = pos;
    depInsts.push_back(op);
    ++pos;
  }

  if (firstSrcDepPos.has_value()) {
    if (lastDstDepPos.has_value()) {
      if (*firstSrcDepPos <= *lastDstDepPos) {
        // No valid insertion point exists which preserves dependences.
        return nullptr;
      }
    }
    // Return the insertion point at 'firstSrcDepPos'.
    return depInsts[*firstSrcDepPos];
  }
  // No dependence targets in range (or only dst deps in range), return
  // 'dstNodInst' insertion point.
  return dstNodeInst;
}

// Updates edge mappings from node 'srcId' to node 'dstId' after fusing them,
// taking into account that:
//   *) if 'removeSrcId' is true, 'srcId' will be removed after fusion,
//   *) memrefs in 'privateMemRefs' has been replaced in node at 'dstId' by a
//      private memref.
void DFG::updateEdges(unsigned srcId, unsigned dstId,
                                        const DenseSet<Value> &privateMemRefs,
                                        bool removeSrcId) {
  // For each edge in 'inEdges[srcId]': add new edge remapping to 'dstId'.
  if (inEdges.count(srcId) > 0) {
    SmallVector<Edge, 2> oldInEdges = inEdges[srcId];
    for (auto &inEdge : oldInEdges) {
      // Add edge from 'inEdge.id' to 'dstId' if it's not a private memref.
      if (privateMemRefs.count(inEdge.value) == 0)
        addEdge(inEdge.id, dstId, inEdge.value);
    }
  }
  // For each edge in 'outEdges[srcId]': remove edge from 'srcId' to 'dstId'.
  // If 'srcId' is going to be removed, remap all the out edges to 'dstId'.
  if (outEdges.count(srcId) > 0) {
    SmallVector<Edge, 2> oldOutEdges = outEdges[srcId];
    for (auto &outEdge : oldOutEdges) {
      // Remove any out edges from 'srcId' to 'dstId' across memrefs.
      if (outEdge.id == dstId)
        removeEdge(srcId, outEdge.id, outEdge.value);
      else if (removeSrcId) {
        addEdge(dstId, outEdge.id, outEdge.value);
        removeEdge(srcId, outEdge.id, outEdge.value);
      }
    }
  }
  // Remove any edges in 'inEdges[dstId]' on 'oldMemRef' (which is being
  // replaced by a private memref). These edges could come from nodes
  // other than 'srcId' which were removed in the previous step.
  if (inEdges.count(dstId) > 0 && !privateMemRefs.empty()) {
    SmallVector<Edge, 2> oldInEdges = inEdges[dstId];
    for (auto &inEdge : oldInEdges)
      if (privateMemRefs.count(inEdge.value) > 0)
        removeEdge(inEdge.id, dstId, inEdge.value);
  }
}

// Update edge mappings for nodes 'sibId' and 'dstId' to reflect fusion
// of sibling node 'sibId' into node 'dstId'.
void DFG::updateEdges(unsigned sibId, unsigned dstId) {
  // For each edge in 'inEdges[sibId]':
  // *) Add new edge from source node 'inEdge.id' to 'dstNode'.
  // *) Remove edge from source node 'inEdge.id' to 'sibNode'.
  if (inEdges.count(sibId) > 0) {
    SmallVector<Edge, 2> oldInEdges = inEdges[sibId];
    for (auto &inEdge : oldInEdges) {
      addEdge(inEdge.id, dstId, inEdge.value);
      removeEdge(inEdge.id, sibId, inEdge.value);
    }
  }

  // For each edge in 'outEdges[sibId]' to node 'id'
  // *) Add new edge from 'dstId' to 'outEdge.id'.
  // *) Remove edge from 'sibId' to 'outEdge.id'.
  if (outEdges.count(sibId) > 0) {
    SmallVector<Edge, 2> oldOutEdges = outEdges[sibId];
    for (auto &outEdge : oldOutEdges) {
      addEdge(dstId, outEdge.id, outEdge.value);
      removeEdge(sibId, outEdge.id, outEdge.value);
    }
  }
}

// // Adds ops in 'loads' and 'stores' to node at 'id'.
// void DFG::addToNode(
//     unsigned id, const SmallVectorImpl<Operation *> &loads,
//     const SmallVectorImpl<Operation *> &stores) {
//   Node *node = getNode(id);
//   llvm::append_range(node->loads, loads);
//   llvm::append_range(node->stores, stores);
// }

static std::optional<int64_t> getPerfectNestTripCount(AffineForOp root) {
  using namespace mlir;
  int64_t prod = 1;
  AffineForOp cur = root;

  while (true) {
    // Per-loop TC (requires constant bounds)
    if (!cur.hasConstantLowerBound() || !cur.hasConstantUpperBound())
      return std::nullopt;

    int64_t lb = cur.getConstantLowerBound();
    int64_t ub = cur.getConstantUpperBound();
    int64_t step = cur.getStep().getSExtValue(); // fix for APInt -> int64_t
    if (ub <= lb) return int64_t(0);

    auto thisTc = (ub - lb + step - 1) / step; // ceilDiv for positives
    prod *= thisTc;

    // Check for a single non-yield op that is an inner AffineForOp
    AffineForOp inner = nullptr;
    int nonYieldCount = 0;
    for (Operation &op : *cur.getBody()) {
      if (!isa<affine::AffineYieldOp>(op)) {
        ++nonYieldCount;
        if (auto next = dyn_cast<AffineForOp>(op))
          inner = next;
      }
    }

    if (nonYieldCount == 1 && inner)
      cur = inner; // keep descending
    else
      break; // not perfectly nested any further
  }

  return prod;
}

void DFG::print(raw_ostream &os) const {
  os << "\nMemRefDependenceGraph\n";
  os << "\nNodes:\n";
  for (const auto &idAndNode : nodes) {
    os << "Node: " << idAndNode.first << "\n";
    auto it = inEdges.find(idAndNode.first);
    if (it != inEdges.end()) {
      for (const auto &e : it->second)
        os << "  InEdge: " << e.id << " " << e.value << "\n";
    }
    it = outEdges.find(idAndNode.first);
    if (it != outEdges.end()) {
      for (const auto &e : it->second)
        os << "  OutEdge: " << e.id << " " << e.value << "\n";
    }
  }
}

void DFG::populateNodeCandidates(){
  for(auto &nodePair : nodes){
    populateCandidatesForNode(nodePair.second);
  }
}

void DFG::populateCandidatesForNode(Node &node){
  if(node.wrappedModules.empty()) return;
  
  auto [funcOps, forOps] = gatherFuncsAndForOps(node.wrappedModules);
  if (forOps.empty()) return;

  // collect loop bands info
  AllLoopBandsInfo allLoopBandsInfo;
  collectLoopBandsInfo(forOps, allLoopBandsInfo);

  // Tile the loops
  tileLoopsTwoLevels(allLoopBandsInfo);

  // add dummy loop info
  addDummyLoopInfo(allLoopBandsInfo);
  
  // For each loop band (e.g. stencil kernel), populate candidates
  for(auto loopBandInfoAndForOp : llvm::zip(allLoopBandsInfo, forOps)){
    auto loopBandInfo = std::get<0>(loopBandInfoAndForOp);
    auto forOp = std::get<1>(loopBandInfoAndForOp);
    SmallVector<MemrefInfo, 4> memrefs;
    getLoopMemRefs(forOp, memrefs);
    auto currLoopCandidates = getAllCandidates(loopBandInfo, memrefs, true);
    // node.candidates.memrefCandidates.push_back(currLoopCandidates);
  }

  // NOTE: In the new structure, we prefer populateNodeInfo which handles all permutations.
  // This legacy helper is kept for dot printing but updated to use the new maps.
}



void DFG::printNodes(raw_ostream &os) const {
  os << "\n=== DFG Nodes Information ===\n";
  os << "Global Tiling Variables: {";
  bool firstGlobal = true;
  for (const auto &it : tilingVars) {
    if (!firstGlobal)
      os << ", ";
    os << it.first << ": " << it.second;
    firstGlobal = false;
  }
  os << "}\n";
  for (const auto &idAndNode : nodes) {
    const auto &node = idAndNode.second;
    std::string kindStr;
    switch (node.kind) {
    case NodeKind::COMPUTE:    kindStr = "Compute"; break;
    case NodeKind::MEMORY:     kindStr = "Memory"; break;
    case NodeKind::CONVERTER:  kindStr = "Converter"; break;
    case NodeKind::SUPER_NODE: kindStr = "SuperNode"; break;
    case NodeKind::UNKNOWN:    kindStr = "Unknown"; break;
    }

    os << "Node ID: " << node.id << "\n"
       << "  Classification: " << kindStr << "\n"
       << "  isReduction: " << node.isReduction << "\n"
       << "  Type: " << node.type << "\n"
       << "  Arithmetic Weight: " << node.arithWeight << "\n"
       << "  Band Size: " << node.bandSize << "\n"
       << "  Trip Count: " << node.tripCount << "\n";
    os << "  Tiling Variables: [";
    for (unsigned i = 0; i < node.candidates.tilingCandidates.size(); ++i) {
      if (i > 0)
        os << ", ";
      os << "\"" << node.candidates.tilingCandidates[i] << "\"";
    }
    os << "]\n";

    os << "  ForOps (" << node.forOps.size() << "):\n";
    for (auto forOp : node.forOps) {
      forOp.print(os);
      os << "\n";
    }
    
    os << "  Wrapped Modules (" << node.wrappedModules.size() << "):\n";
    for (auto mod : node.wrappedModules) {
      mod.print(os);
      os << "\n";
    }    

    os << "  Wrapped Modules (" << node.wrappedModules.size() << "):\n";
    for (auto module : node.wrappedModules) {
      if (auto nameAttr = module->getAttrOfType<StringAttr>("sym_name")) {
        os << "    - Module: " << nameAttr.getValue() << "\n";
      } else {
        os << "    - Module: <unnamed>\n";
      }
      module.walk([&](func::FuncOp func) {
        if (!func.isDeclaration()) {
          os << "      - Func: " << func.getSymName() << "\n";
        }
      });
    }
    if (node.hwModule) {
      os << "  HW Module:\n";
      node.hwModule->print(os);
    }
    if (!node.candidates.permutationCandidates.empty()){
      os << "candidates: [\n";
      for (auto &permCandidate : node.candidates.permutationCandidates){
          os << "  Permutation: [";
          for (unsigned i = 0; i < permCandidate.permutation.size(); ++i) {
            if (i > 0) os << ", ";
            os << permCandidate.permutation[i];
          }
          os << "], Inverse Permutation: [";
          for (unsigned i = 0; i < permCandidate.permutation.size(); ++i) {
            if (i > 0) os << ", ";
            os << inversePerm(permCandidate.permutation)[i];
          }
          os << "], II: " << permCandidate.II << "\n";
          
          os << "    Loads:\n";
          for (auto &it : permCandidate.loadsMap) {
            auto &candidate = it.second;
            os << "      Load: " << candidate.memref << "\n";
            os << "        Buffer Sizes: ";
            for (auto &bs : candidate.bufferSizes) os << bs.exprStr << " ";
            os << "\n";
            os << "        Access Indices: ";
            for (auto &ai : candidate.accessIndices) os << ai.exprStr << " ";
            os << "\n";
            os << "        Access Map: " << candidate.accessMap << "\n";
            os << "        (" << candidate.loadFirstAccessTimeEval << "): ";
            candidate.loadFirstAccessTime.expr.print(os);
            os << " (First Access Time)\n";
            os << "        (" << candidate.loadFATimeLegacyEval << "): ";
            os << candidate.loadFATimeLegacy << " (First Access Time (Legacy))\n";
            os << "        (" << candidate.loadLastAccessTimeEval << "): ";
            candidate.loadLastAccessTime.expr.print(os);
            os << " (Last Access Time)\n";
            os << "        (" << candidate.loadLATimeLegacyEval << "): ";
            os << candidate.loadLATimeLegacy << " (Last Access Time (Legacy))\n";
          }

          os << "    Stores:\n";
          for (auto &it : permCandidate.storesMap) {
            auto &candidate = it.second;
            os << "      Store: " << candidate.memref << "\n";
            os << "        Buffer Sizes: ";
            for (auto &bs : candidate.bufferSizes) os << bs.exprStr << " ";
            os << "\n";
            os << "        Access Indices: ";
            for (auto &ai : candidate.accessIndices) os << ai.exprStr << " ";
            os << "\n";
            os << "        Access Map: " << candidate.accessMap << "\n";
            os << "        (" << candidate.storeFirstAccessTimeEval << "): ";
            candidate.storeFirstAccessTime.expr.print(os);
            os << " (First Access Time) \n";
            os << "        (" << candidate.storeFATimeLegacyEval << "): ";
            os << candidate.storeFATimeLegacy << " (First Access Time (Legacy))\n";
            os << "        (" << candidate.storeLastAccessTimeEval << "): ";
            candidate.storeLastAccessTime.expr.print(os);
            os << " (Last Access Time) \n";
            os << "        (" << candidate.storeLATimeLegacyEval << "): ";
            os << candidate.storeLATimeLegacy << " (Last Access Time (Legacy))\n";
          }
      }
      os << "]\n";
    }
    os << "===================================\n";
  }
}

void DFG::printEdges(raw_ostream &os) const {
  os << "\n=== DFG Edges Information ===\n";
  for (const auto &idAndEdges : outEdges) {
    unsigned srcId = idAndEdges.first;
    for (const auto &edge : idAndEdges.second) {
      os << "Source ID: " << srcId << " -> Dest ID: " << edge.id << "\n";
      os << "  Value: ";
      edge.value.print(os);
      os << "\n";
      os << "  Kind: " << (edge.kind == EdgeKind::STREAM ? "STREAM" : (edge.kind == EdgeKind::AOS ? "AOS" : "MEMREF")) << "\n";
      if (edge.resolvedType)
        edge.resolvedType.print(os);
      else
        os << "None";
      os << "\n";
      os << "  Src FIFO Info: ";
      if (edge.srcFifoInfo) edge.srcFifoInfo.print(os);
      else os << "None";
      os << "\n";
      os << "  Dst FIFO Info: ";
      if (edge.dstFifoInfo) edge.dstFifoInfo.print(os);
      else os << "None";
      os << "\n";
      os << "-----------------------------------\n";
    }
  }
  os << "===================================\n";
}

// namespace {

// Try to compute constant #elements for a memref type.
static std::optional<int64_t> getStaticNumElements(mlir::MemRefType mt) {
  if (!mt.hasStaticShape()) return std::nullopt;
  int64_t n = 1;
  for (int64_t d : mt.getShape()) {
    if (d < 0) return std::nullopt;
    n *= d;
  }
  return n;
}

static std::optional<int64_t>
getBytesPerElement(mlir::Type elemTy, const mlir::DataLayout *DL) {
  using namespace mlir;

  auto ceilDiv8 = [](unsigned w) -> int64_t {
    return int64_t((w + 7) / 8); // ceil(bits/8), always >= 1 when w > 0
  };

  if (auto it = dyn_cast<IntegerType>(elemTy))
    return ceilDiv8(it.getWidth());

  if (auto ft = dyn_cast<FloatType>(elemTy))
    return ceilDiv8(ft.getWidth());

  if (isa<IndexType>(elemTy)) {
    if (DL) return int64_t(std::max<int64_t>(1, DL->getTypeSize(elemTy)));
    return int64_t(8); // fallback for index when no DataLayout
  }

  if (DL) return int64_t(std::max<int64_t>(1, DL->getTypeSize(elemTy)));

  // Unknown without DL
  return std::nullopt;
}

// Follow view/cast ops to the underlying base buffer so we don’t double-count.
static mlir::Value getBaseBuffer(mlir::Value v) {
  using namespace mlir;
  while (true) {
    if (auto c = v.getDefiningOp<memref::CastOp>())              { v = c.getSource(); continue; }
    if (auto s = v.getDefiningOp<memref::SubViewOp>())           { v = s.getSource(); continue; }
    if (auto r = v.getDefiningOp<memref::ReinterpretCastOp>())   { v = r.getSource(); continue; }
    if (auto vi = dyn_cast_or_null<ViewLikeOpInterface>(v.getDefiningOp())) {
      v = vi.getViewSource(); continue;
    }
    return v; // reached alloc/alloca/global/func-arg or non-view op
  }
}

// Return a canonical pointer key for deduping the underlying buffer:
//  - For memref.get_global: the unique memref.global op pointer
//  - Otherwise: the (stripped) base SSA value pointer
static void *getCanonicalBufferKey(mlir::Value v) {
  using namespace mlir;

  // Strip views/casts first (your existing helper is fine)
  Value base = getBaseBuffer(v);

  // If it's a get_global, map to the defining memref.global op (stable identity)
  if (auto getg = base.getDefiningOp<memref::GetGlobalOp>()) {
    // Lookup the symbol; nearest symbol table is fine
    if (Operation *symScope = SymbolTable::getNearestSymbolTable(getg))
      if (auto global = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
              getg, getg.getNameAttr()))
        return global.getOperation(); // same pointer for all uses of this global
    // Fallback to the get_global result if lookup somehow fails
    return base.getAsOpaquePointer();
  }

  // Block arguments / alloc / alloca: SSA value pointer is stable for dedupe
  return base.getAsOpaquePointer();
}

// Sum of unique array footprints (bytes) used under `scope`.
// - Dedupes per base buffer
// - Static shapes contribute to knownBytes; dynamic dims bump unknownCount
static void sumArrayFootprint(mlir::Operation *scope,
                              int64_t &knownBytes,
                              int64_t &unknownCount,
                              const mlir::DataLayout *DL = nullptr) {
  // FIXME: It can count many times the same array.
  using namespace mlir;
  knownBytes = 0;
  unknownCount = 0;
  llvm::SmallPtrSet<void*, 32> seen;

  scope->walk([&](Operation *op) {
    auto processVal = [&](Value v) {
      auto mt = dyn_cast<MemRefType>(v.getType());
      if (!mt) return;

      void *key = getCanonicalBufferKey(v);
      if (!seen.insert(key).second) return; // already counted

      auto nOpt   = getStaticNumElements(mt);
      auto bpeOpt = getBytesPerElement(mt.getElementType(), DL);
      if (nOpt && bpeOpt) {
        knownBytes += (*nOpt);// * (*bpeOpt);
      } else {
        unknownCount += 1;
      }
    };

    // Count **uses** only (operands). This avoids double counting producers.
    for (Value v : op->getOperands()) processVal(v);

    // If you *also* want to include locally produced buffers, uncomment:
    // for (Value v : op->getResults())  processVal(v);
  });
}

// } // namespace

void DFG::printAsDot(raw_ostream &os) const {
  os << "digraph DFG {\n";
  os << "  rankdir=LR;\n";

  for (const auto &idAndNode : nodes) {
    const auto &node = idAndNode.second;
    unsigned id = idAndNode.first;

    // Metadata calculations for scheduling tools
    int maxTC = 0, minTC = 0, avgTC = 0;
    int64_t flop = 0, flopsPerIter = 0;
    int64_t footprintBytes = 0, footprintUnknown = 0;
    std::string opName = "";

    if (node.op) {
      sumArrayFootprint(node.op, footprintBytes, footprintUnknown, /*DL*/ nullptr);
      if (auto attr = node.op->getAttrOfType<StringAttr>("op_name")) {
        opName = attr.getValue().str();
      }
      if (auto forOp = dyn_cast<AffineForOp>(node.op)) {
        auto tally = countFlopsPerIter(forOp.getOperation());
        flopsPerIter = tally.totalScalar();
        if (auto tcOpt = getBandTripCount(forOp)) {
          maxTC = minTC = avgTC = *tcOpt;
          flop = flopsPerIter * (*tcOpt);
        } else {
          flop = flopsPerIter;
        }
      }
    }

    // Node styling based on classification
    std::string color = "#1e88e5";      // Default Blue
    std::string fillcolor = "#e3f2fd";  // Default Light Blue
    switch (node.kind) {
    case NodeKind::SUPER_NODE:
      color = "#ffa726";      // Orange
      fillcolor = "#fff3e0";  // Light Orange
      break;
    case NodeKind::COMPUTE:
      color = "#e53935";      // Red
      fillcolor = "#ffebee";  // Light Red
      break;
    case NodeKind::MEMORY:
      color = "#1e88e5";      // Blue
      fillcolor = "#e3f2fd";  // Light Blue
      break;
    case NodeKind::CONVERTER:
      color = "#8e24aa";      // Purple
      fillcolor = "#f3e5f5";  // Light Purple
      break;
    case NodeKind::UNKNOWN:
    default:
      color = "#212121";      // Black/Grey
      fillcolor = "#eeeeee";  // Light Grey
      break;
    }
    
    // Visual Label construction
    std::string labelStr;
    llvm::raw_string_ostream label(labelStr);
    label << "ID: " << id;
    if (!node.wrappedModules.empty()) {
      label << "\\nPatterns: ";
      SmallVector<std::string, 4> funcNames;
      for (auto mod : node.wrappedModules) {
        for (auto func : mod.getOps<func::FuncOp>()) {
          funcNames.push_back(func.getName().str());
        }
      }
      label << llvm::join(funcNames, ", ");
    }
    label.flush();

    // Print node on a SINGLE LINE with minimal formatting for fragile mlg.py parser
    os << "  " << id << " [label=\"" << labelStr 
       << "\",color=\"" << color 
       << "\",fillcolor=\"" << fillcolor 
       << "\",style=\"filled,rounded\",shape=box,fontname=\"Helvetica\",fontsize=10,fontcolor=\"#212121\""
       << ",isRed=" << (node.kind == NodeKind::COMPUTE ? "true" : "false")
       << ",fctName=\"\",op_name=\"" << opName << "\""
       << ",max_tc=\"" << maxTC << "\",min_tc=\"" << minTC << "\",avg_tc=\"" << avgTC << "\""
       << ",type=\"" << node.type << "\",flop=\"" << flop << "\",flops_per_iter=\"" << flopsPerIter << "\""
       << ",footprint_bytes=\"" << footprintBytes << "\",footprint_unknown=\"" << footprintUnknown << "\"];\n";

    // Edges
    if (inEdges.count(id)) {
      for (const auto &e : inEdges.at(id)) {
        std::string edgeColor = "black";
        std::string bufferName = "";

        if (e.value) {
          if (e.kind == EdgeKind::STREAM) edgeColor = "#4caf50";     // Green
          else if (e.kind == EdgeKind::AOS) edgeColor = "#2196f3";    // Blue
          else if (e.kind == EdgeKind::MEMREF) edgeColor = "#f44336"; // Red

          if (e.value && e.value.getDefiningOp()) {
            auto op = e.value.getDefiningOp();
            if (auto attr = op->getAttrOfType<DictionaryAttr>("buffer.info")) {
              if (auto nameAttr = attr.get("name").dyn_cast_or_null<StringAttr>()) {
                bufferName = nameAttr.getValue().str();
              }
            }
          }
        }

        os << "  " << e.id << " -> " << id << " [label=\"" << bufferName << " (d=" << e.depth << ")\",color=\"" << edgeColor << "\"];\n";
      }
    }
  }
  os << "}\n";
}

void DFG::mergeSuperNodesDFG() {
  LLVM_DEBUG(llvm::dbgs() << "--- mergeSuperNodesDFG ---\n");

  // 1) Identify MEMREF edges and compute In-Degrees for Topological Sort
  SmallVector<std::pair<unsigned, unsigned>, 8> memrefEdges;
  DenseMap<unsigned, unsigned> inDegree;
  for (auto const &nodePair : nodes) inDegree[nodePair.first] = 0;

  for (auto const &pair : outEdges) {
    unsigned srcId = pair.first;
    for (auto &edge : pair.second) {
      if (edge.kind == EdgeKind::MEMREF) {
        memrefEdges.push_back({srcId, edge.id});
        inDegree[edge.id]++;
      }
    }
  }

  // 2) Global Topological Sort (Kahn's Algorithm)
  // This gives us the correct "execution order" for all nodes.
  std::vector<unsigned> topoOrder;
  std::queue<unsigned> q;
  for (auto const &nodePair : nodes) {
    if (inDegree[nodePair.first] == 0) q.push(nodePair.first);
  }

  while (!q.empty()) {
    unsigned u = q.front(); q.pop();
    topoOrder.push_back(u);
    if (outEdges.count(u)) {
      for (auto &edge : outEdges[u]) {
        if (edge.kind == EdgeKind::MEMREF) {
          if (--inDegree[edge.id] == 0) q.push(edge.id);
        }
      }
    }
  }

  // Map each ID to its position in the topological sort for quick lookup
  DenseMap<unsigned, int> topoPos;
  for (int i = 0; i < (int)topoOrder.size(); ++i) topoPos[topoOrder[i]] = i;

  // 3) Group nodes via DSU (Same as before)
  DenseMap<unsigned, unsigned> parent;
  std::function<unsigned(unsigned)> find_root = [&](unsigned i) -> unsigned {
    if (parent.count(i) == 0 || parent[i] == i) return parent[i] = i;
    return parent[i] = find_root(parent[i]);
  };
  auto unite = [&](unsigned i, unsigned j) {
    unsigned rootI = find_root(i);
    unsigned rootJ = find_root(j);
    if (rootI != rootJ) parent[rootI] = rootJ;
  };
  for (auto &pair : memrefEdges) unite(pair.first, pair.second);

  std::map<unsigned, std::vector<unsigned>> groups;
  for (auto const &nodePair : nodes) {
    groups[find_root(nodePair.first)].push_back(nodePair.first);
  }

  // 4) Merge groups following the Topological Order
  for (auto &groupPair : groups) {
    auto &groupIds = groupPair.second;
    if (groupIds.size() <= 1) continue;

    // SORT the IDs within this group based on their global topological position
    std::sort(groupIds.begin(), groupIds.end(), [&](unsigned a, unsigned b) {
      return topoPos[a] < topoPos[b];
    });

    // The earliest node in the topological order becomes the SuperNode representative
    unsigned keepId = groupIds[0];
    Node *keepNode = getNode(keepId);
    
    // Seed constituentModules with keepId's own module
    if (keepNode->constituentModules.empty() && keepNode->hwModule) {
      keepNode->constituentModules.push_back(keepNode->hwModule);
    }

    // Merge the rest in order. mergeNodes usually appends modules to the end.
    for (size_t i = 1; i < groupIds.size(); ++i) {
      unsigned srcId = groupIds[i];
      mergeNodes(srcId, keepId, /*force=*/true);
    }
    
    keepNode->kind = NodeKind::SUPER_NODE;
    keepNode->type = keepNode->type + "_super";
  }
}

namespace {
// Helper to get buffer name from AllocOp or BlockArgument
static StringAttr getBufferName(Value memref) {
  if (memref && memref.getDefiningOp<memref::AllocOp>()) {
    auto allocOp = memref.getDefiningOp<memref::AllocOp>();
    if (auto attr = allocOp->getAttrOfType<DictionaryAttr>("buffer.info")) {
      return attr.getAs<StringAttr>("name");
    }
  } else if (memref && isa<BlockArgument>(memref)) {
    auto arg = cast<BlockArgument>(memref);
    if (auto func = dyn_cast<func::FuncOp>(arg.getOwner()->getParentOp())) {
      if (auto attr = func.getArgAttrOfType<DictionaryAttr>(arg.getArgNumber(), "buffer.info")) {
        return attr.getAs<StringAttr>("name");
      }
    }
  }
  return nullptr;
}

static func::FuncOp getEntryFunction(ModuleOp mod) {
  func::FuncOp entry;
  for (auto f : mod.getOps<func::FuncOp>()) {
    entry = f;
  }
  return entry;
}
} // namespace

void DFG::wrapSuperNodesDFG() {
  MLIRContext *ctx = block.getParentOp()->getContext();
  OpBuilder b(ctx);

  for (auto &nodePair : nodes) {
    auto &node = nodePair.second;
    if (!node.op || node.kind != NodeKind::SUPER_NODE || node.constituentModules.empty()) continue;

    Location loc = node.op->getLoc();
    ModuleOp superModule = ModuleOp::create(loc);
    
    // 1) Collect constituent functions.
    SmallVector<func::FuncOp, 4> constituentFuncs;
    for (auto mod : node.constituentModules) {
      if (auto f = getEntryFunction(mod)) {
        constituentFuncs.push_back(f);
      }
    }

    // 2) Identify internal memrefs/streams (allocOps IF used by constituent functions).
    llvm::DenseSet<StringRef> usedBufferNames;
    for (auto f : constituentFuncs) {
      for (unsigned i = 0; i < f.getNumArguments(); ++i) {
        if (auto attr = f.getArgAttrOfType<DictionaryAttr>(i, "buffer.info")) {
          if (auto nameAttr = attr.getAs<StringAttr>("name")) {
            usedBufferNames.insert(nameAttr.getValue());
          }
        }
      }
    }
    llvm::StringMap<Type> internalBuffers;
    for (auto allocOp : node.allocOps) {
      if (auto nameAttr = getBufferName(allocOp.getResult())) {
        StringRef name = nameAttr.getValue();
        if (usedBufferNames.count(name)) {
          Type targetType = allocOp.getResult().getType();
          // If there is a self-loop edge for this buffer, use its resolvedType (e.g. if it became a stream)
          if (outEdges.count(node.id)) {
            for (auto &edge : outEdges[node.id]) {
              if (edge.id == node.id && getBufferName(edge.value) == nameAttr) {
                if (edge.resolvedType) {
                  targetType = edge.resolvedType;
                  break;
                }
              }
            }
          }
          internalBuffers[name] = targetType;
        }
      }
    }

    // 3) Collect external arguments and their types.
    llvm::MapVector<StringRef, std::pair<Type, DictionaryAttr>> externalArgs;
    
    for (auto f : constituentFuncs) {
      for (unsigned i = 0; i < f.getNumArguments(); ++i) {
        auto attr = f.getArgAttrOfType<DictionaryAttr>(i, "buffer.info");
        if (!attr) continue;
        auto nameAttr = attr.getAs<StringAttr>("name");
        if (!nameAttr) continue;
        
        StringRef name = nameAttr.getValue();
        if (internalBuffers.count(name)) continue; // skip internal
        
        if (externalArgs.count(name) == 0) {
          externalArgs[name] = {f.getArgument(i).getType(), attr};
        }
      }
    }

    // 7) Clone ALL functions from constituent modules.
    SymbolTable symbolTable(superModule);
    for (auto mod : node.constituentModules) {
      for (auto f : mod.getOps<func::FuncOp>()) {
        auto clonedF = f.clone();
        symbolTable.insert(clonedF);
      }
    }

    // 4) Create the top function signature.
    SmallVector<Type> wrapperArgTypes;
    for (auto &it : externalArgs) {
      wrapperArgTypes.push_back(it.second.first);
    }
    
    std::string wrapperName = "top_super_node_" + std::to_string(node.id);
    auto wrapperTy = b.getFunctionType(wrapperArgTypes, {});
    
    b.setInsertionPointToEnd(superModule.getBody());
    auto wrapper = b.create<func::FuncOp>(loc, wrapperName, wrapperTy);
    wrapper.setVisibility(SymbolTable::Visibility::Public);
    auto *entryBlock = wrapper.addEntryBlock();
    b.setInsertionPointToStart(entryBlock);

    // 5) Apply buffer.info attributes to wrapper arguments.
    unsigned argIdx = 0;
    for (auto &it : externalArgs) {
      wrapper.setArgAttr(argIdx++, "buffer.info", it.second.second);
    }

    // 6) Allocate internal memrefs or streams.
    llvm::StringMap<Value> internalVals;
    for (auto &it : internalBuffers) {
      if (isa<MemRefType>(it.second)) {
        internalVals[it.first()] = b.create<memref::AllocOp>(loc, cast<MemRefType>(it.second));
      } else if (auto tensorTy = dyn_cast<RankedTensorType>(it.second)) {
        internalVals[it.first()] = b.create<dataflow::ArrayOfStreamsOp>(loc, tensorTy, b.getI32IntegerAttr(2));
      } else if (isa<dataflow::StreamType>(it.second)) {
        internalVals[it.first()] = b.create<dataflow::StreamOp>(loc, it.second, b.getI32IntegerAttr(2));
      }
    }

    // 8) Build calls for entry functions.
    for (auto mod : node.constituentModules) {
      auto entryFunc = getEntryFunction(mod);
      if (!entryFunc) continue;

      // Find the cloned version of entryFunc in superModule
      auto clonedEntryF = superModule.lookupSymbol<func::FuncOp>(entryFunc.getName());
      assert(clonedEntryF && "Cloned entry function not found");

      SmallVector<Value> callArgs;
      for (unsigned i = 0; i < clonedEntryF.getNumArguments(); ++i) {
        auto attr = clonedEntryF.getArgAttrOfType<DictionaryAttr>(i, "buffer.info");
        if (!attr) {
          assert(false && "Argument missing buffer.info");
        }
        StringRef name = attr.getAs<StringAttr>("name").getValue();
        
        if (internalVals.count(name)) {
          callArgs.push_back(internalVals[name]);
        } else {
          unsigned extIdx = 0;
          bool found = false;
          for (auto &it : externalArgs) {
            if (it.first == name) {
              found = true;
              break;
            }
            extIdx++;
          }
          if (!found) {
              llvm::errs() << "Could not find argument " << name << " in externalArgs\n";
              assert(false);
          }
          callArgs.push_back(entryBlock->getArgument(extIdx));
        }
      }
      b.create<func::CallOp>(loc, clonedEntryF, callArgs);
    }

    b.create<func::ReturnOp>(loc);
    node.hwModule = superModule;
  }
}

static void applyTopLevelArrayPartition(DFG::Node &node, Value topArg, Value childArg) {
  auto func = dyn_cast<func::FuncOp>(childArg.getParentBlock()->getParentOp());
  if (!func) return;

  auto type = topArg.getType().cast<MemRefType>();
  unsigned rank = type.getRank();
  SmallVector<int64_t> factors(rank, 1);
  SmallVector<dataflow::PartitionKind> kinds(rank, dataflow::PartitionKind::CYCLIC);

  // 1. Collect and sort all loops by depth
  SmallVector<AffineForOp> allLoops;
  func.walk([&](AffineForOp forOp) {
    if (forOp.hasConstantUpperBound())
      allLoops.push_back(forOp);
  });
  
  std::sort(allLoops.begin(), allLoops.end(), [](AffineForOp a, AffineForOp b) {
    auto getDepth = [](AffineForOp f) {
      int d = 0;
      Operation *curr = f.getOperation();
      while (curr && (curr = curr->getParentOp())) {
        if (isa<AffineForOp>(curr)) d++;
      }
      return d;
    };
    return getDepth(a) > getDepth(b);
  });

  // 2. Find the memory access operation for this argument
  Operation *memOp = nullptr;
  func.walk([&](Operation *op) {
    if (auto load = dyn_cast<AffineLoadOp>(op)) {
      if (load.getMemRef() == childArg) {
        memOp = op; return WalkResult::interrupt();
      }
    }
    if (auto store = dyn_cast<AffineStoreOp>(op)) {
      if (store.getMemRef() == childArg) {
        memOp = op; return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  if (memOp) {
    // 3. Match innermost loops to dimensions via the AffineMap
    AffineMap map;
    SmallVector<Value> indices;
    if (auto load = dyn_cast<AffineLoadOp>(memOp)) {
        map = load.getAffineMap();
        indices.append(load.getIndices().begin(), load.getIndices().end());
    } else if (auto store = dyn_cast<AffineStoreOp>(memOp)) {
        map = store.getAffineMap();
        indices.append(store.getIndices().begin(), store.getIndices().end());
    }

    for (unsigned d = 0; d < std::min((size_t)rank, (size_t)map.getNumResults()); ++d) {
      AffineExpr expr = map.getResult(d);
      
      // Look for the deepest loop IV used in this result expression
      for (auto loop : allLoops) {
        Value iv = loop.getInductionVar();
        bool ivFound = false;
        
        expr.walk([&](AffineExpr e) {
          if (auto dimExpr = e.dyn_cast<AffineDimExpr>()) {
            if (dimExpr.getPosition() < indices.size() && 
                indices[dimExpr.getPosition()] == iv) {
              ivFound = true;
            }
          }
        });
        
        if (ivFound) {
          factors[d] = loop.getConstantUpperBound();
          break; // Deepest loop found for this dimension
        }
      }
    }
  }

  bool anyPartition = false;
  for (auto f : factors) if (f > 1) anyPartition = true;
  if (!anyPartition) return;

  applyArrayPartition(topArg, factors, kinds, true);
  applyArrayPartition(childArg, factors, kinds, true);
}


ModuleOp DFG::generateSystemModule(ModuleOp templateModule) {

  MLIRContext *ctx = templateModule.getContext();
  OpBuilder b(ctx);
  Location loc = templateModule.getLoc();

  // 1) Create New Module
  ModuleOp systemModule = ModuleOp::create(loc);
  b.setInsertionPointToEnd(systemModule.getBody());

  // 2) Top-Level Signature: Find the main function in templateModule (typically @forward)
  func::FuncOp templateFunc = nullptr;
  for (auto f : templateModule.getOps<func::FuncOp>()) {
    if (f.getSymName() == "forward") {
      templateFunc = f;
      break;
    }
  }
  if (!templateFunc) templateFunc = *templateModule.getOps<func::FuncOp>().begin();
  assert(templateFunc && "Template function not found");

  func::FuncOp topFunc = b.create<func::FuncOp>(loc, "module_top", templateFunc.getFunctionType());
  topFunc.setVisibility(SymbolTable::Visibility::Public);
  // Copy attributes (including buffer.info for arguments)
  for (unsigned i = 0; i < templateFunc.getNumArguments(); ++i) {
    if (auto attr = templateFunc.getArgAttr(i, "buffer.info"))
      topFunc.setArgAttr(i, "buffer.info", attr);
  }

  Block *entry = topFunc.addEntryBlock();
  b.setInsertionPointToStart(entry);
  auto returnOp = b.create<func::ReturnOp>(loc);
  b.setInsertionPoint(returnOp);



  // 3) FIFO Allocation: Iterate through all DFG edges and create streams
  llvm::DenseMap<const Edge *, Value> edgeToStream;
  for (auto &nodeEdges : outEdges) {
    for (auto &edge : nodeEdges.second) {
      // Allow self-loops (internal to super-node) because they might be exposed as arguments
      if ((edge.kind == EdgeKind::STREAM || edge.kind == EdgeKind::AOS) && edge.resolvedType) {
        Type streamTy = edge.resolvedType;
        if (auto aofsTy = dyn_cast<RankedTensorType>(streamTy)) {
          auto aofsOp = b.create<ArrayOfStreamsOp>(loc, aofsTy, b.getI32IntegerAttr(edge.depth));
          edgeToStream[&edge] = aofsOp.getResult();
        } else if (isa<dataflow::StreamType>(streamTy)) {
          // Default depth of 2 for single streams if not specified
          auto streamOp = b.create<StreamOp>(loc, streamTy, b.getI32IntegerAttr(edge.depth));
          edgeToStream[&edge] = streamOp.getResult();
        }
      }
    }
  }

  // 4) Component Function Cloning (at Module Level)
  SymbolTable systemSymbolTable(systemModule);
  OpBuilder moduleBuilder(systemModule.getBodyRegion());
  llvm::DenseMap<unsigned, func::FuncOp> nodeEntryFuncs;

  for (auto &nodePair : nodes) {
    Node &node = nodePair.second;
    unsigned nodeId = nodePair.first;
    if (!node.hwModule) continue;

    func::FuncOp nodeEntry = nullptr;
    for (auto f : node.hwModule.getOps<func::FuncOp>()) {
      func::FuncOp clonedFunc = cast<func::FuncOp>(moduleBuilder.clone(*f.getOperation()));
      clonedFunc.setVisibility(SymbolTable::Visibility::Private);
      
      std::string newName = clonedFunc.getSymName().str();
      if (systemSymbolTable.lookup(newName) && systemSymbolTable.lookup(newName) != clonedFunc) {
        newName = newName + "_node" + std::to_string(nodeId);
        clonedFunc.setSymName(newName);
      }
      nodeEntry = clonedFunc;
    }
    nodeEntryFuncs[nodeId] = nodeEntry;
  }

  // 4.5) Update top-level arg types to match padded node entry functions.
  //      After tiling-padding, node functions may expect larger memref shapes
  //      (e.g. 129x129 instead of 128x128). We propagate the max shape per
  //      top-level argument so that func.call operand types match.
  {
    // Map top-level arg name → max shape seen across all node entry funcs.
    llvm::StringMap<SmallVector<int64_t>> maxShapeByName;
    for (auto &nodePair : nodes) {
      unsigned nodeId = nodePair.first;
      if (!nodeEntryFuncs.count(nodeId)) continue;
      func::FuncOp entryFunc = nodeEntryFuncs[nodeId];
      for (unsigned i = 0; i < entryFunc.getNumArguments(); ++i) {
        auto attr = entryFunc.getArgAttrOfType<DictionaryAttr>(i, "buffer.info");
        if (!attr) continue;
        auto typeAttr = attr.getAs<StringAttr>("type");
        if (!typeAttr) continue;
        StringRef bufType = typeAttr.getValue();
        if (bufType != "off-chip" && bufType != "weights") continue;
        auto nameAttr = attr.getAs<StringAttr>("name");
        if (!nameAttr) continue;
        std::string bufName = nameAttr.str();
        auto memTy = dyn_cast<MemRefType>(entryFunc.getArgument(i).getType());
        if (!memTy || !memTy.hasStaticShape()) continue;
        auto &maxShape = maxShapeByName[bufName];
        if (maxShape.empty()) {
          maxShape.assign(memTy.getShape().begin(), memTy.getShape().end());
        } else {
          for (unsigned d = 0; d < memTy.getRank(); ++d)
            maxShape[d] = std::max(maxShape[d], memTy.getDimSize(d));
        }
      }
    }
    // Apply the max shapes to top-level function arguments.
    for (auto topArg : entry->getArguments()) {
      auto topAttr = topFunc.getArgAttrOfType<DictionaryAttr>(
          topArg.getArgNumber(), "buffer.info");
      if (!topAttr) continue;
      auto nameAttr = topAttr.getAs<StringAttr>("name");
      if (!nameAttr) continue;
      auto it = maxShapeByName.find(nameAttr.str());
      if (it == maxShapeByName.end()) continue;
      auto memTy = dyn_cast<MemRefType>(topArg.getType());
      if (!memTy) continue;
      SmallVector<int64_t> &maxShape = it->second;
      if (SmallVector<int64_t>(memTy.getShape()) == maxShape) continue;

      // Record the original (pre-padding) shape so host code can size
      // DMA transfers to match the user's actual buffer dimensions.
      SmallVector<Attribute> origShapeDims;
      for (auto d : memTy.getShape())
        origShapeDims.push_back(b.getI64IntegerAttr(d));
      SmallVector<Attribute> paddedShapeDims;
      for (auto d : maxShape)
        paddedShapeDims.push_back(b.getI64IntegerAttr(d));
      auto origLayoutDict = b.getDictionaryAttr({
          b.getNamedAttr("shape", b.getArrayAttr(origShapeDims))});
      auto paddedLayoutDict = b.getDictionaryAttr({
          b.getNamedAttr("shape", b.getArrayAttr(paddedShapeDims))});
      auto newTy = MemRefType::get(maxShape, memTy.getElementType(),
                                   memTy.getLayout(), memTy.getMemorySpace());
      topArg.setType(newTy);

      // Update buffer.info with original_layout and padded_layout
      SmallVector<NamedAttribute> newBufAttrs(topAttr.getValue());
      newBufAttrs.push_back(
          b.getNamedAttr("original_layout", origLayoutDict));
      newBufAttrs.push_back(
          b.getNamedAttr("padded_layout", paddedLayoutDict));
      topFunc.setArgAttr(topArg.getArgNumber(), "buffer.info",
                         b.getDictionaryAttr(newBufAttrs));

      LLVM_DEBUG(llvm::dbgs() << "[generateSystemModule] Updated top-level arg '"
                               << nameAttr.getValue() << "' type to " << newTy
                               << " (original: " << memTy << ")\n");
    }
    // Rebuild top-level function type signature.
    SmallVector<Type> newArgTypes;
    for (auto arg : entry->getArguments())
      newArgTypes.push_back(arg.getType());
    topFunc.setType(FunctionType::get(ctx, newArgTypes, {}));
  }

  // 5) Node Call Generation (inside Top-Level Function)
  for (auto &nodePair : nodes) {
    Node &node = nodePair.second;
    unsigned nodeId = nodePair.first;
    if (!node.hwModule || !nodeEntryFuncs.count(nodeId)) continue;

    func::FuncOp entryFunc = nodeEntryFuncs[nodeId];
    LLVM_DEBUG(llvm::dbgs() << "--- Generating call for Node: " << nodeId << " (" << entryFunc.getSymName() << ") ---\n");

    // B) Prepare arguments for the call
    SmallVector<Value> callArgs;
    for (unsigned i = 0; i < entryFunc.getNumArguments(); ++i) {
      auto attr = entryFunc.getArgAttrOfType<DictionaryAttr>(i, "buffer.info");
      assert(attr && "Node function argument missing buffer.info");
      
      std::string bufName = "";
      if (auto nameAttr = attr.getAs<StringAttr>("name"))
        bufName = nameAttr.str();
        
      std::string bufType = "shared-buffer"; // Default to FIFO
      if (auto typeAttr = attr.getAs<StringAttr>("type"))
        bufType = typeAttr.str();

      Value mappedVal = nullptr;
      if (bufType == "off-chip" || bufType == "weights") {
        // Match with top-level arguments
        for (auto topArg : entry->getArguments()) {
          auto topAttr = topFunc.getArgAttrOfType<DictionaryAttr>(topArg.getArgNumber(), "buffer.info");
          if (topAttr && topAttr.getAs<StringAttr>("name").str() == bufName) {
            mappedVal = topArg;
            break;
          }
        }
        if (mappedVal) {
          LLVM_DEBUG(llvm::dbgs() << "    Mapped off-chip/weights arg to top-level arg\n");
          applyTopLevelArrayPartition(node, mappedVal, entryFunc.getArgument(i));
        } else {
          LLVM_DEBUG(llvm::dbgs() << "    FAILED to map off-chip/weights arg to top-level arg\n");
        }

      } else {
        // FIFO: Find the edge
        std::string io = "unknown";
        if (auto ioAttr = attr.getAs<StringAttr>("io"))
          io = ioAttr.str();

        for (const auto &outEdgePair : outEdges) {
          unsigned src = outEdgePair.first;
          for (const auto &edge : outEdgePair.second) {
            unsigned dst = edge.id;
            
            // Check if buffer name matches
            if (getBufferName(edge.value) && getBufferName(edge.value).str() == bufName) {
              // Disambiguate based on IO
              if (io == "read-only") {
                if (dst == nodeId) { // Incoming edge
                  mappedVal = edgeToStream[&edge];
                }
              } else if (io == "write-only") {
                if (src == nodeId) { // Outgoing edge
                  mappedVal = edgeToStream[&edge];
                }
              } else {
                // Fallback for "unknown" or if we can't decide: 
                // check if it's either src or dst (legacy behavior)
                if (src == nodeId || dst == nodeId) {
                   mappedVal = edgeToStream[&edge];
                }
              }
              if (mappedVal) break;
            }
          }
          if (mappedVal) break;
        }

        if (mappedVal) {
          LLVM_DEBUG(llvm::dbgs() << "    Mapped FIFO arg to stream\n");
        } else {
          llvm::errs() << "    FAILED to map FIFO arg: " << bufName << " (IO: " << io << ") for Node: " << nodeId << "\n";
          llvm::errs() << "    Checked outEdges for all nodes:\n";
          for (const auto &outEdgePair : outEdges) {
            unsigned src = outEdgePair.first;
            for (const auto &edge : outEdgePair.second) {
              StringRef name = "";
              if (auto n = getBufferName(edge.value)) name = n.getValue();
              llvm::errs() << "      Edge: src=" << src << " -> dst=" << edge.id << ", buf=" << name << "\n";
            }
          }
        }
      }
      assert(mappedVal && "Could not map node argument to top-level or FIFO");
      callArgs.push_back(mappedVal);
    }

    // C) Emit Call
    b.create<func::CallOp>(loc, entryFunc, callArgs);
  }

  return systemModule;
}

int64_t DFG::getLatency(const DSEOptions &options) const {
  return getPerformanceMetrics(options).latency;
}

int64_t DFG::getTotalMemory(const DSEOptions &options) const {
  return getPerformanceMetrics(options).memory;
}

int64_t DFG::getDSPs(const DSEOptions &options) const {
  return getPerformanceMetrics(options).dsps;
}

int64_t DFG::getCost(const DSEOptions &options) const {
  return getPerformanceMetrics(options).cost;
}

SolveResult DFG::getPerformanceMetrics(const DSEOptions &options) const {
  PerformanceModel model;
  createCombinedPerformanceModel(model, const_cast<DFG&>(*this), options);
  model.finalize();

  // Start with a default solution (handles default unrolling=1 and first permutation=1)
  auto vars = generateDefaultSolution(model);
  
  // Overwrite with current permutation solution
  auto pSol = getPermutationSolution();
  for (auto const& [nodeId, permIdx] : pSol) {
    // Clear all b_nodeId_i to 0 first (generateDefaultSolution might have picked i=0)
    std::string prefix = "b" + std::to_string(nodeId) + "_";
    for (const auto& v : model.variables) {
      if (v.name.compare(0, prefix.size(), prefix) == 0) {
        vars[v.name] = 0;
      }
    }
    std::string varName = prefix + std::to_string(permIdx);
    vars[varName] = 1;
  }

  // Overwrite with current tiling solution
  auto tSol = getTilingSolution();
  for (auto const& [varName, val] : tSol) {
    vars[varName] = val;
  }

  // Resolve Z variables (trip counts) from X variables (tiling factors)
  for (const auto& g : model.tilingGroups) {
    if (vars.count(g.xVar) && vars.at(g.xVar) != 0) {
      vars[g.zVar] = g.totalTC / vars.at(g.xVar);
    }
  }

  runAssignments(model, vars);
  SolveResult result;
  result.success = is_valid(model, vars);
  result.populate(vars);
  return result;
}

std::map<unsigned, unsigned> DFG::getPermutationSolution() const {
  std::map<unsigned, unsigned> pSolution;
  for (const auto &it : nodes) {
    pSolution[it.first] = it.second.solution.permIdx;
  }
  return pSolution;
}

std::map<std::string, int64_t> DFG::getTilingSolution() const {
  std::map<std::string, int64_t> tSolution;
  for (const auto &it : tilingVars) {
    tSolution[it.first] = it.second;
  }
  return tSolution;
}

bool DFG::updatePermutationSolution(const SolveResult &result) {
  bool solutionFound = false;
  std::regex pattern("b([0-9]+)_([0-9]+)");
  for (auto const &[name, val] : result.variables) {
    std::smatch match;
    if (std::regex_search(name, match, pattern) && val == 1) {
      unsigned nodeId = std::stoi(match[1].str());
      unsigned permIdx = std::stoi(match[2].str());
      auto *node = getNode(nodeId);
      if (node) {
        node->solution.permIdx = permIdx;
        solutionFound = true;
      }
    }
  }
  return solutionFound;
}

bool DFG::updatePermutationSolution(const std::map<unsigned, unsigned> &solution) {
  for (auto const &[nodeId, permIdx] : solution) {
    auto *node = getNode(nodeId);
    if (node) {
      node->solution.permIdx = permIdx;
    }
  }
  return true;
}

bool DFG::updateUnrollingSolution(const SolveResult &result) {
  bool solutionFound = false;
  for (auto const &[name, val] : result.variables) {
    if (name.substr(0, 2) == "X_") {
      tilingVars[name] = val;
      solutionFound = true;
    }
  }
  if (solutionFound) {
    updateNodeTilingFactors();
  }
  return solutionFound;
}

bool DFG::updateUnrollingSolution(const std::map<std::string, int64_t> &solution) {
  bool solutionFound = false;
  for (auto const &[name, val] : solution) {
    if (name.substr(0, 2) == "X_") {
      tilingVars[name] = val;
      solutionFound = true;
    }
  }
  if (solutionFound) {
    updateNodeTilingFactors();
  }
  return solutionFound;
}

void DFG::printFixedPermutationSolution() {
  llvm::dbgs() << "\tfixed_permutation_solution: {";
  bool first = true;
  for (const auto &nodePair : nodes) {
    const auto &node = nodePair.second;
    if (node.candidates.permutationCandidates.empty())
      continue;
    if (!first) llvm::dbgs() << ", ";
    llvm::dbgs() << node.id << ": " << node.solution.permIdx;
    first = false;
  }
  llvm::dbgs() << "}\n";
}

void DFG::printFixedUnrollingSolution() {
  llvm::dbgs() << "\tfixed_unrolling_solution: {";
  bool first = true;
  for (const auto &pair : tilingVars) {
    if (!first) llvm::dbgs() << ", ";
    llvm::dbgs() << pair.first << ": " << pair.second;
    first = false;
  }
  llvm::dbgs() << "}\n";
}
namespace {
bool isTopFuncBlockArg(Value v) {
  if (!v.isa<BlockArgument>()) return false;
  auto ba = v.cast<BlockArgument>();
  auto *block = ba.getOwner();
  auto *parentOp = block->getParentOp();
  return isa_and_nonnull<func::FuncOp>(parentOp) && block->isEntryBlock();
}

std::pair<std::string, std::string> getNameAndTypeFromAnnotation(Value v) {
  if (auto ba = v.dyn_cast<BlockArgument>()) {
    auto func = cast<func::FuncOp>(ba.getOwner()->getParentOp());
    if (auto dict = func.getArgAttrOfType<DictionaryAttr>(ba.getArgNumber(), "buffer.info")) {
      auto nameAttr = dict.getAs<StringAttr>("name");
      auto typeAttr = dict.getAs<StringAttr>("type");
      if (nameAttr && typeAttr) return {nameAttr.str(), typeAttr.str()};
    }
    return {"arg" + std::to_string(ba.getArgNumber()), "off-chip"};
  }
  Operation *def = v.getDefiningOp();
  if (!def) return {"<unknown>", "off-chip"};
  if (auto dict = def->getAttrOfType<DictionaryAttr>("buffer.info")) {
    auto nameAttr = dict.getAs<StringAttr>("name");
    auto typeAttr = dict.getAs<StringAttr>("type");
    if (nameAttr && typeAttr) return {nameAttr.str(), typeAttr.str()};
  }
  if (isa<memref::GetGlobalOp>(def)) {
    return {cast<memref::GetGlobalOp>(def).getName().str(), "weights"};
  }
  if (isa<memref::AllocOp, memref::AllocaOp>(def)) {
    return {"buf", "local-buffer"};
  }
  return {"<unknown>", "off-chip"};
}

int getArgIdx(AffineForOp forOp, Value memref) {
  SmallVector<Value, 8> orderedDeps;
  DenseSet<Value> depSet;
  forOp.walk([&](Operation *op) {
    auto pushMem = [&](Value m) {
      if (!m || !isa<MemRefType>(m.getType())) return;
      if (!depSet.insert(m).second) return;
      if (!(isTopFuncBlockArg(m) || m.getDefiningOp<memref::AllocOp>() || m.getDefiningOp<memref::GetGlobalOp>()))
        return;
      auto [nm, tp] = getNameAndTypeFromAnnotation(m);
      if (tp == "off-chip" || tp == "shared-buffer" || tp == "weights") {
        orderedDeps.push_back(m);
      }
    };
    if (auto ld = dyn_cast<AffineLoadOp>(op)) pushMem(ld.getMemRef());
    else if (auto st = dyn_cast<AffineStoreOp>(op)) pushMem(st.getMemRef());
    else if (auto ld = dyn_cast<memref::LoadOp>(op)) pushMem(ld.getMemref());
    else if (auto st = dyn_cast<memref::StoreOp>(op)) pushMem(st.getMemref());
  });
  for (int i = 0; i < (int)orderedDeps.size(); ++i) {
    if (orderedDeps[i] == memref) return i;
  }
  return -1;
}
} // namespace

void DFG::populateFifoInfo() {
  for (auto &nodePair : nodes) {
    Node &node = nodePair.second;
    if (!node.hwModule) continue;
    
    func::FuncOp hwFunc = *node.hwModule.getOps<func::FuncOp>().begin();
    
    auto processEdges = [&](SmallVectorImpl<Edge> &nodeEdges, bool isInput) {
      for (auto &edge : nodeEdges) {
        if (edge.kind != EdgeKind::MEMREF) continue;
        
        Operation *originalOp = isInput ? edge.dstOp.getOperation() : edge.srcOp.getOperation();
        if (!originalOp) continue;
        
        AffineForOp targetFor;
        for (auto forOp : node.forOps) {
          if (forOp->isAncestor(originalOp)) {
            targetFor = forOp;
            break;
          }
        }
        if (!targetFor) continue;
        
        int argIdx = getArgIdx(targetFor, edge.value);
        if (argIdx == -1) continue;
        
        assert(argIdx < (int)hwFunc.getNumArguments() && "argIdx out of range");
        Value hwArg = hwFunc.getArgument(argIdx);
        
        for (auto user : hwArg.getUsers()) {
          if (isInput) {
            if (auto loadOp = dyn_cast<AffineLoadOp>(user)) {
              if (auto attr = loadOp->getAttrOfType<DictionaryAttr>("fifo-info")) {
                edge.dstFifoInfo = attr;
              }
            }
          } else {
            if (auto storeOp = dyn_cast<AffineStoreOp>(user)) {
              if (auto attr = storeOp->getAttrOfType<DictionaryAttr>("fifo-info")) {
                edge.srcFifoInfo = attr;
              }
            }
          }
        }
      }
    };
    
    if (inEdges.count(node.id)) processEdges(inEdges[node.id], true);
    if (outEdges.count(node.id)) processEdges(outEdges[node.id], false);
  }
  
  // Sync the copies across nodes
  for (auto &nodePair : nodes) {
    unsigned nid = nodePair.first;
    for (auto &edge : inEdges[nid]) {
      unsigned srcId = edge.id;
      if (outEdges.count(srcId)) {
        for (auto &outEdge : outEdges[srcId]) {
          if (outEdge.id == nid && outEdge.value == edge.value) {
            outEdge.dstFifoInfo = edge.dstFifoInfo;
            edge.srcFifoInfo = outEdge.srcFifoInfo;
          }
        }
      }
    }
  }
}
