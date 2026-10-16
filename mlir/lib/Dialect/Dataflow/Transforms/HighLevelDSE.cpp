#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Pass/PassManager.h"


#include "streamhls/Dialect/Dataflow/Transforms/Passes.h"
#include "streamhls/Dialect/Dataflow/Support/Utils.h"
#include "streamhls/Dialect/Dataflow/Support/DFG.h"

#include <cstdlib>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <array>

using namespace mlir;
using namespace streamhls;
using namespace dataflow;

#define DEBUG_TYPE "streamhls-dse"

static void generateNoReuseMapping(DFG &graph, HWMapping &result) {
  // 1-1 mapping: Each node gets its own hardware unit
  for (auto &nodePair : graph.nodes) {
    unsigned id = nodePair.first;
    std::string hwName = "node_" + std::to_string(id);
    result.mapping[hwName].push_back(id);
  }
}

static void generateMaxReuseMapping(DFG &graph, HWMapping &result) {
  // Group nodes by their type string, but preserve topological order
  llvm::StringMap<SmallVector<unsigned>> typeToNodes;
  auto topoOrder = graph.topologicalSort();
  for (unsigned id : topoOrder) {
    DFG::Node *node = graph.getNode(id);
    if (!node) continue;
    typeToNodes[node->type].push_back(id);
  }
  
  // Assign hardware names based on types
  for (auto &entry : typeToNodes) {
    std::string hwName = entry.first().str() + "_unit";
    result.mapping[hwName] = entry.second;
  }
}

static bool isNodeSimilar(DFG::Node *n1, DFG::Node *n2, const DSEOptions &options) {
  // If either node has no trip count, we can't reliably compare intensity.
  if (n1->tripCount == 0 || n2->tripCount == 0) return true;

  // Primary check: ratio of total trip counts (arithmetic intensity proxy).
  double ratio = (double)std::max(n1->tripCount, n2->tripCount) /
                 std::min(n1->tripCount, n2->tripCount);
  if (ratio > options.balancedMappingSimilarityThreshold) return false;

  // Secondary check: individual loop upper bounds similarity.
  // Since they have the same 'type', bandSize should match.
  if (n1->loopsUBs.size() != n2->loopsUBs.size()) return false;
  for (size_t i = 0; i < n1->loopsUBs.size(); ++i) {
    if (n1->loopsUBs[i] <= 0 || n2->loopsUBs[i] <= 0) continue;
    double ubRatio = (double)std::max(n1->loopsUBs[i], n2->loopsUBs[i]) /
                      std::min(n1->loopsUBs[i], n2->loopsUBs[i]);
    if (ubRatio > options.balancedMappingUBThreshold) return false;
  }
  return true;
}

static void generateBalancedMapping(DFG &graph, HWMapping &result, const DSEOptions &options) {
  // Balanced mapping: merge nodes of the same type ONLY if they are sequentially dependent.
  // This preserves parallelism while saving resources for serial chains.
  auto topoOrder = graph.topologicalSort();
  
  // type -> list of clusters (each cluster is a SmallVector of node IDs)
  llvm::StringMap<SmallVector<SmallVector<unsigned>>> typeToClusters;

  for (unsigned nodeId : topoOrder) {
    DFG::Node *node = graph.getNode(nodeId);
    if (!node || node->kind != DFG::NodeKind::COMPUTE) continue;
    
    std::string type = node->type;
    auto &clusters = typeToClusters[type];
    bool assigned = false;

    for (auto &cluster : clusters) {
      bool isSequential = true;
      for (unsigned existingId : cluster) {
        // Since we process in topo order, the only way existingId is NOT a producer
        // of nodeId is if they are parallel.
        LLVM_DEBUG(llvm::dbgs() << "Node " << nodeId << " depends on " << existingId << "? " << graph.hasDependencePath(existingId, nodeId) << "\n";);
        if (!graph.hasDependencePath(existingId, nodeId)) {
          isSequential = false;
          break;
        }
      }

      if (isSequential) {
        // Only merge if nodes have similar intensity/structure
        if (isNodeSimilar(node, graph.getNode(cluster.back()), options)) {
          cluster.push_back(nodeId);
          assigned = true;
          break;
        }
      }
    }

    if (!assigned) {
      SmallVector<unsigned> newCluster;
      newCluster.push_back(nodeId);
      clusters.push_back(newCluster);
    }
  }

  // Process non-compute nodes (e.g., MEMORY, CONVERTER) as no_reuse for now
  for (auto &nodePair : graph.nodes) {
    unsigned id = nodePair.first;
    DFG::Node *node = &nodePair.second;
    if (node->kind != DFG::NodeKind::COMPUTE) {
      std::string hwName = "node_" + std::to_string(id);
      result.mapping[hwName].push_back(id);
    }
  }

  // Assign hardware names for clustered compute nodes
  for (auto &typeEntry : typeToClusters) {
    std::string type = typeEntry.first().str();
    int clusterIdx = 0;
    for (auto &cluster : typeEntry.second) {
      std::string hwName = type + "_unit_" + std::to_string(clusterIdx++);
      result.mapping[hwName] = cluster;
    }
  }
}

HWMapping streamhls::HardwareMappingDSE(
  ModuleOp &module,
  std::string projectPath,
  const DSEOptions &options,
  std::string hwMappingStrategy /* no_reuse, max_reuse */
) {
  HWMapping result;
  result.hwMappingStrategy = hwMappingStrategy;

  auto funcs = module.getOps<func::FuncOp>();
  if (funcs.empty()) {
    llvm::errs() << "[HardwareMappingDSE] ERROR: No functions found in module.\n";
    return result;
  }
  auto func = *funcs.begin();
  Block* block = &func.front();
  DFG graph(*block);
  if (!graph.init()) {
    llvm::errs() << "[HardwareMappingDSE] ERROR: DFG init failed.\n";
    return result;
  }

  // --- DOT file generation for debugging ---
  std::string dotFilePath = projectPath + "/02_dfg/00_init.dot";
  std::error_code stdErr;
  // Ensure directory exists
  std::filesystem::path dirPath = std::filesystem::path(dotFilePath).parent_path();
  if (!dirPath.empty() && !std::filesystem::exists(dirPath)) {
      std::filesystem::create_directories(dirPath);
  }
  
  llvm::raw_fd_ostream graphFile(dotFilePath, stdErr);
  if (!stdErr) {
    graph.printAsDot(graphFile);
    graphFile.flush();
    llvm::outs() << "[HardwareMappingDSE] Graph DOT file written to: " << dotFilePath << "\n";
  } else {
    llvm::errs() << "[HardwareMappingDSE] WARNING: Failed to write DOT file: " << stdErr.message() << "\n";
  }
  
  for(auto node : graph.nodes){
    LLVM_DEBUG(llvm::dbgs() << "Node " << node.first << " " << node.second.type << " " << node.second.tripCount << "\n";);
  }
  // --- Generate mapping ---
  if (hwMappingStrategy == "no_reuse") {
    generateNoReuseMapping(graph, result);
  } else if (hwMappingStrategy == "max_reuse") {
    generateMaxReuseMapping(graph, result);
  } else if (hwMappingStrategy == "balanced") {
    generateBalancedMapping(graph, result, options);
  } else {
    llvm::errs() << "[HardwareMappingDSE] WARNING: Unknown strategy '" << hwMappingStrategy << "'. Returning empty mapping.\n";
  }

  return result;
}

HWMapping streamhls::runHighLevelDSE(
  ModuleOp &module,
  std::string projectPath,
  std::string mlgPath,
  const DSEOptions &options,
  std::string hwMappingStrategy /* no_reuse, max_reuse */
) {
  return HardwareMappingDSE(module, projectPath, options, hwMappingStrategy);
}
