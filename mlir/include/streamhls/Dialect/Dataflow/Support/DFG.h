#ifndef MLIR_STREAMHLS_DIALECT_DATAFLOW_SUPPORT_DFG_H
#define MLIR_STREAMHLS_DIALECT_DATAFLOW_SUPPORT_DFG_H

#include "mlir/Analysis/FlatLinearValueConstraints.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "streamhls/Dialect/Dataflow/Support/PerformanceModel.h"
#include "mlir/IR/IntegerSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Analysis/Presburger/PWMAFunction.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Analysis/Presburger/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallBitVector.h"
#include <memory>
#include <optional>
#include <string>
#include <fstream>

#include <queue>
#include <set>
#include <stack>
#include <map>
#include <vector>

#include "streamhls/Dialect/Dataflow/IR/Dataflow.h"
#include "streamhls/Dialect/Dataflow/Support/AffineAnalysis.h"


namespace mlir { namespace streamhls {
using namespace affine;
using namespace dataflow;
struct SolveResult;
struct HWMapping {
  std::string hwMappingStrategy; // "no_reuse" or "max_reuse"
  llvm::StringMap<SmallVector<unsigned>> mapping; // hwName -> nodeIds
};


/// Returns true if the loop nest is a perfectly nested band that performs
/// initialization (strictly constant zero store). Returns the initialization 
/// value attribute if successful.
std::pair<bool, Attribute> isInitializationLoopNest(AffineForOp forOp);


// structure to represent dataflow graph
// Nodes are:
// Store/Load operations
// FIFO read/write operations
// Edges are:
// buffers or FIFOs
struct DFG {
public:

  struct PermutationMemrefCandidates {
    SmallVector<unsigned> permutation;
    unsigned II = 1;

    DenseMap<AffineLoadOp, MemrefCandidateInfo> loadsMap;
    DenseMap<AffineStoreOp, MemrefCandidateInfo> storesMap;
  };

  
  enum class NodeKind { UNKNOWN, COMPUTE, MEMORY, CONVERTER, SUPER_NODE };
    
  struct VirtualNode {
    unsigned nodeId;
    unsigned loopIdx;
    bool operator<(const VirtualNode& other) const {
      return std::tie(nodeId, loopIdx) < std::tie(other.nodeId, other.loopIdx);
    }
    bool operator==(const VirtualNode& other) const {
      return nodeId == other.nodeId && loopIdx == other.loopIdx;
    }
  };


  struct Node {
    // The unique identifier of this node in the graph.
    unsigned id;
    // The top-level statement which is (or contains) a load/store.
    Operation *op;

    std::string type;

    uint64_t tripCount = 0;

    bool isReduction = false;

    uint64_t DSP_factor = 0;

    SmallVector<AffineForOp, 4> forOps;
    SmallVector<memref::AllocOp, 4> allocOps;

    // Wrapped modules: one ModuleOp per forOp, created by affineForOpToModule.
    // Populated during extractModules() so that the merged super-node carries
    // both the original loop nests and their standalone module representations.
    SmallVector<ModuleOp, 4> wrappedModules;

    // The result of merging all wrappedModules into a single module.
    ModuleOp mergedModule = nullptr;

    // The result after converting function memrefs to FIFOs.
    ModuleOp fifoModule = nullptr;

    // The result after applying hardware reuse.
    ModuleOp reuseModule = nullptr;

    // The canonical hardware module after all transformations.
    ModuleOp hwModule = nullptr;
    
    // List of load operations.
    SmallVector<AffineLoadOp, 4> loads;
    // List of store op insts.
    SmallVector<AffineStoreOp, 4> stores;
    // // List of FIFO read operations.
    // SmallVector<Operation *, 4> fifoReads;
    // // List of FIFO write operations.
    // SmallVector<Operation *, 4> fifoWrites;
    
    // Performance model fields
    int64_t arithWeight = 0;
    SmallVector<int64_t, 4> loopsUBs;
    // SmallVector<int64_t, 4> maxTripCounts;
    unsigned bandSize = 0;
    NodeKind kind = NodeKind::UNKNOWN;

    // For super-nodes: the modules that were merged into this one.
    SmallVector<ModuleOp, 4> constituentModules;

    struct NodeCandidates {
      SmallVector<PermutationMemrefCandidates, 4> permutationCandidates;
      SmallVector<std::string, 4> tilingCandidates;
    };
    NodeCandidates candidates;

    struct NodeSolution {
      unsigned permIdx = 0;
      SmallVector<unsigned, 4> tilingFactors;
    };
    NodeSolution solution;

    Node() : id(0), op(nullptr), kind(NodeKind::UNKNOWN) {}
    Node(unsigned id, Operation *op) : id(id), op(op), kind(NodeKind::UNKNOWN) {}

  };

  enum class EdgeKind { STREAM, AOS, MEMREF };

  // Edge represents a data dependence between nodes in the graph.
  struct Edge {
    // The id of the node at the other end of the edge.
    // If this edge is stored in Edge = Node.inEdges[i], then
    // 'Node.inEdges[i].id' is the identifier of the source node of the edge.
    // If this edge is stored in Edge = Node.outEdges[i], then
    // 'Node.outEdges[i].id' is the identifier of the dest node of the edge.
    unsigned id;
    // The SSA value on which this edge represents a dependence.
    // If the value is a memref, then the dependence is between graph nodes
    // which contain accesses to the same memref 'value'. If the value is a
    // non-memref value, then the dependence is between a graph node which
    // defines an SSA value and another graph node which uses the SSA value
    // (e.g. a constant or load operation defining a value which is used inside
    // a loop nest).
    Value value;
    AffineStoreOp srcOp = nullptr;
    AffineLoadOp dstOp = nullptr;

    EdgeKind kind = EdgeKind::MEMREF;
    Type resolvedType = nullptr;
    uint64_t depth = 2;
    DictionaryAttr srcFifoInfo = nullptr;
    DictionaryAttr dstFifoInfo = nullptr;
  };
  struct MyEdge {
    unsigned srcId;
    unsigned dstId;

    Value value;
  };

  SmallVector<MyEdge, 4> edges;

  // Map from node id to Node.
  DenseMap<unsigned, Node> nodes;
  // Map from node id to list of input edges.
  DenseMap<unsigned, SmallVector<Edge, 2>> inEdges;
  // Map from node id to list of output edges.
  DenseMap<unsigned, SmallVector<Edge, 2>> outEdges;
  // Map from memref to a count on the dependence edges associated with that
  // memref.
  DenseMap<Value, unsigned> memrefEdgeCount;

  // The next unique identifier to use for newly created graph nodes.
  unsigned nextNodeId = 0;

  std::map<std::string, int64_t> tilingVars;

  void initializeTilingVars();

  void populateTilingVarsRandomly();

  void populateTilingVarsWithOnes();

  void updateNodeTilingFactors();

  std::map<unsigned, unsigned> getPermutationSolution() const;
  std::map<std::string, int64_t> getTilingSolution() const;

  int64_t getLatency(const DSEOptions &options = DSEOptions()) const;
  int64_t getTotalMemory(const DSEOptions &options = DSEOptions()) const;
  int64_t getDSPs(const DSEOptions &options = DSEOptions()) const;
  int64_t getCost(const DSEOptions &options = DSEOptions()) const;
  SolveResult getPerformanceMetrics(const DSEOptions &options = DSEOptions()) const;

  bool updatePermutationSolution(const SolveResult &result);
  bool updateUnrollingSolution(const SolveResult &result);
  bool updatePermutationSolution(const std::map<unsigned, unsigned> &solution);
  bool updateUnrollingSolution(const std::map<std::string, int64_t> &solution);

  DFG(Block &block) : block(block) {}

  /// Generate the top-level HLS dataflow module.
  /// \param templateModule  the module providing the top-level function signature.
  /// \return the generated system module.
  ModuleOp generateSystemModule(ModuleOp templateModule);

  // Initializes the dependence graph based on operations in `block'.
  // Returns true on success, false otherwise.
  bool init();

  Node *mergeNodes(unsigned srcId, unsigned dstId, bool force = false);

  /// Merge nodes that share the same hw_name in the mapping.
  /// Groups nodes by hwName and iteratively merges each group into a single
  /// super-node. ForOps in the resulting super-nodes are ordered by the
  /// topological sort of the original graph.
  /// \param nodeToHwName  nodeId → hwName mapping (from node_to_hw_mapping.json)
  /// \return true on success
  bool mergeNodesByHwMapping(
      const llvm::StringMap<SmallVector<unsigned>> &hwNameToNodes);

  /// Group nodes by MEMREF edges and merge them into super-nodes.
  void mergeSuperNodesDFG();
  /// Generate wrappers for super-nodes.
  void wrapSuperNodesDFG();

  // topological sort
  SmallVector<unsigned> topologicalSort();
  void topologicalSortUtil(unsigned currId, std::set<unsigned> &visited, std::stack<unsigned> &stack);

  SmallVector<VirtualNode> virtualTopologicalSort();
  void virtualTopologicalSortUtil(VirtualNode curr, std::set<VirtualNode> &visited, std::stack<VirtualNode> &stack);

  int getLoopIndex(unsigned nodeId, Operation *op);


  LogicalResult populateNodeInfo(bool enablePermutations);

  bool createRootNode();

  bool createSinkNode();

  bool applyNodePermutations();

  // Returns the graph node for 'id'.
  Node *getNode(unsigned id);

  // Adds a node with 'op' to the graph and returns its unique identifier.
  unsigned addNode(Operation *op);

  // Remove node 'id' (and its associated edges) from graph.
  void removeNode(unsigned id);

  // Returns true if node 'id' writes to any memref which escapes (or is an
  // argument to) the block. Returns false otherwise.
  bool writesToLiveInOrEscapingMemrefs(unsigned id);

  // Returns true iff there is an edge from node 'srcId' to node 'dstId' which
  // is for 'value' if non-null, or for any value otherwise. Returns false
  // otherwise.
  bool hasEdge(unsigned srcId, unsigned dstId, Value value = nullptr);

  // Adds an edge from node 'srcId' to node 'dstId' for 'value'.
  void addEdge(unsigned srcId, unsigned dstId, Value value);
  void addEdge(unsigned srcId, unsigned dstId, AffineStoreOp srcOp,
               AffineLoadOp dstOp, Value value);
  void addEdge(unsigned srcId, unsigned dstId, Value value, EdgeKind kind,
               Type resolvedType, AffineStoreOp srcOp = nullptr,
               AffineLoadOp dstOp = nullptr,
               DictionaryAttr srcFifoInfo = nullptr,
               DictionaryAttr dstFifoInfo = nullptr);
  // Removes an edge from node 'srcId' to node 'dstId' for 'value'.
  void removeEdge(unsigned srcId, unsigned dstId, Value value);

  // Returns true if there is a path in the dependence graph from node 'srcId'
  // to node 'dstId'. Returns false otherwise. `srcId`, `dstId`, and the
  // operations that the edges connected are expected to be from the same block.
  bool hasDependencePath(unsigned srcId, unsigned dstId);

  // Returns the input edge count for node 'id' and 'memref' from src nodes
  // which access 'memref' with a store operation.
  unsigned getIncomingMemRefAccesses(unsigned id, Value memref);

  // Returns the output edge count for node 'id' and 'memref' (if non-null),
  // otherwise returns the total output edge count from node 'id'.
  unsigned getOutEdgeCount(unsigned id, Value memref = nullptr);

  /// Return all nodes which define SSA values used in node 'id'.
  void gatherDefiningNodes(unsigned id, DenseSet<unsigned> &definingNodes);

  // Computes and returns an insertion point operation, before which the
  // the fused <srcId, dstId> loop nest can be inserted while preserving
  // dependences. Returns nullptr if no such insertion point is found.
  Operation *getFusedLoopNestInsertionPoint(unsigned srcId, unsigned dstId);

  // Updates edge mappings from node 'srcId' to node 'dstId' after fusing them,
  // taking into account that:
  //   *) if 'removeSrcId' is true, 'srcId' will be removed after fusion,
  //   *) memrefs in 'privateMemRefs' has been replaced in node at 'dstId' by a
  //      private memref.
  void updateEdges(unsigned srcId, unsigned dstId,
                   const DenseSet<Value> &privateMemRefs, bool removeSrcId);

  // Update edge mappings for nodes 'sibId' and 'dstId' to reflect fusion
  // of sibling node 'sibId' into node 'dstId'.
  void updateEdges(unsigned sibId, unsigned dstId);

  // Adds ops in 'loads' and 'stores' to node at 'id'.
  void addToNode(unsigned id, const SmallVectorImpl<Operation *> &loads,
                 const SmallVectorImpl<Operation *> &stores);

  void print(raw_ostream &os) const;
  void populateNodeCandidates();
  void populateFifoInfo();
  void populateCandidatesForNode(Node &node);

  void printNodes(raw_ostream &os) const;
  void printEdges(raw_ostream &os) const;
  void printFixedPermutationSolution();
  void printFixedUnrollingSolution();
  void printAsDot(raw_ostream &os) const;

  void dump() const { print(llvm::errs()); }

  /// The block for which this graph is created to perform fusion.
  Block &block;
};
} // namespace streamhls
} // namespace mlir

#endif // MLIR_STREAMHLS_DIALECT_DATAFLOW_SUPPORT_DFG_H