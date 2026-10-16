#include "streamhls/Dialect/Dataflow/Support/Equations.h"
#include "streamhls/Dialect/Dataflow/Support/Utils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include <sstream>
#include <algorithm>
#include <unordered_set>
#include <fstream>

#define DEBUG_TYPE "performance-equations"

namespace mlir {
namespace streamhls {

namespace {
SymbolicExpr toSymbolic(AffineExpr expr, const std::vector<std::string> &symbols, PerformanceModel &model) {
  if (!expr) {
    assert(false && "Null AffineExpr");
  }
  // return SymbolicExpr((int64_t)0);
  if (auto cst = dyn_cast<AffineConstantExpr>(expr))
    return SymbolicExpr(cst.getValue());
  if (auto sym = dyn_cast<AffineSymbolExpr>(expr)) {
    unsigned idx = sym.getPosition();
    if (idx < symbols.size()) return model.var(symbols[idx]);
    return SymbolicExpr((int64_t)0);
  }
  if (auto bin = dyn_cast<AffineBinaryOpExpr>(expr)) {
    auto lhs = toSymbolic(bin.getLHS(), symbols, model);
    auto rhs = toSymbolic(bin.getRHS(), symbols, model);
    switch (expr.getKind()) {
      case AffineExprKind::Add: return lhs + rhs;
      case AffineExprKind::Mul: return lhs * rhs;
      case AffineExprKind::Mod: return lhs - (lhs / rhs) * rhs;
      case AffineExprKind::FloorDiv: return lhs / rhs;
      case AffineExprKind::CeilDiv: return (lhs + rhs - SymbolicExpr((int64_t)1)) / rhs;
      default: return lhs + rhs; // Default to add for now if unknown
    }
  }
  expr.dump();
  assert(false && "Unknown AffineExpr kind");
  // return SymbolicExpr((int64_t)0);
}

int64_t getMemRefSize(Value value) {
  if (!value) return 0;
  if (auto memrefType = value.getType().dyn_cast<MemRefType>()) {
    if (memrefType.hasStaticShape()) {
      int64_t size = 1;
      for (int64_t dim : memrefType.getShape()) {
        size *= dim;
      }
      return size;
    }
  }
  return 0;
}


void declarePermutationVariables(PerformanceModel &model, DFG &dfg) {
  for (auto &nodePair : dfg.nodes) {
    const DFG::Node &node = nodePair.second;
    if (node.id >= 10000) continue;

    std::vector<std::string> varNames;
    unsigned numCandidates = node.candidates.permutationCandidates.size();
    if (numCandidates > 0) {
      for (unsigned i = 0; i < numCandidates; ++i) {
        std::string varName = "b" + std::to_string(node.id) + "_" + std::to_string(i);
        model.addVariable({varName, "binary", std::nullopt, std::nullopt, {}, true, ""});
        varNames.push_back(varName);
      }
      model.addOneHotGroup("G_perm_" + std::to_string(node.id), varNames);
    }
  }
}

void declareTilingVariables(PerformanceModel &model, DFG &dfg, const DSEOptions &options) {
    if (options.sharedTilingStrategy == "max" && options.enforceDivisorOnlyTiling) {
        llvm::errs() << "Warning: 'max' tiling strategy is not supported with 'enforceDivisorOnlyTiling'. Falling back to 'min' divisor logic for shared X range.\n";
    }

    // First, collect domains/bounds for shared X variables.
    std::unordered_map<std::string, std::vector<int64_t>> sharedXDivisors;
    std::unordered_map<std::string, int64_t> sharedXUpperBounds;
    std::vector<std::string> sharedXVariables;

    for (auto const &it : dfg.nodes) {
        const DFG::Node &node = it.second;
        if (node.id >= 10000 || node.forOps.empty()) continue;
        AffineForOp forOp = dyn_cast_or_null<AffineForOp>(node.op);
        if (!forOp) continue;
        AffineLoopBand band;
        getLoopBandFromOutermost(forOp, band);

        for (unsigned i = 0; i < band.size(); ++i) {
            if (i >= node.candidates.tilingCandidates.size()) continue;
            std::string xVar = node.candidates.tilingCandidates[i];
            if (xVar.empty()) continue;

            int64_t ub = (int64_t)band[i].getConstantUpperBound();
            std::vector<int64_t> currentDivisors;
            if (options.enforceDivisorOnlyTiling) {
                for (int64_t d = 1; d <= ub; ++d) {
                    if (ub % d == 0 && (uint64_t)(ub / d) >= options.minTimeLoopSize) {
                        currentDivisors.push_back(d);
                    }
                }
            }

            if (std::find(sharedXVariables.begin(), sharedXVariables.end(), xVar) == sharedXVariables.end()) {
                sharedXVariables.push_back(xVar);
                sharedXUpperBounds[xVar] = ub;
                if (options.enforceDivisorOnlyTiling) {
                    sharedXDivisors[xVar] = currentDivisors;
                    std::sort(sharedXDivisors[xVar].begin(), sharedXDivisors[xVar].end());
                }
            } else {
                if (options.sharedTilingStrategy == "max") {
                    sharedXUpperBounds[xVar] = std::max(sharedXUpperBounds[xVar], ub);
                } else {
                    sharedXUpperBounds[xVar] = std::min(sharedXUpperBounds[xVar], ub);
                }

                if (options.enforceDivisorOnlyTiling) {
                    std::sort(currentDivisors.begin(), currentDivisors.end());
                    if (options.sharedTilingStrategy == "max") {
                        // Union with existing divisors
                        std::vector<int64_t> result;
                        std::set_union(sharedXDivisors[xVar].begin(), sharedXDivisors[xVar].end(),
                                             currentDivisors.begin(), currentDivisors.end(),
                                             std::back_inserter(result));
                        sharedXDivisors[xVar] = result;
                    } else {
                        // Intersect with existing divisors
                        std::vector<int64_t> intersection;
                        std::set_intersection(sharedXDivisors[xVar].begin(), sharedXDivisors[xVar].end(),
                                             currentDivisors.begin(), currentDivisors.end(),
                                             std::back_inserter(intersection));
                        sharedXDivisors[xVar] = intersection;
                    }
                }
            }
        }
    }

    // Declare shared X variables
    for (const std::string &xVar : sharedXVariables) {
        int64_t xUb = sharedXUpperBounds[xVar];
        if (options.enforceDivisorOnlyTiling) {
            auto &divs = sharedXDivisors[xVar];
            if (divs.empty()) divs.push_back(1);
            model.addVariable({xVar, "integer", 1, xUb, divs, true, ""});
        } else {
            model.addVariable({xVar, "integer", 1, xUb, {}, true, ""});
        }
    }

    // Declare per-node-kernel Z variables and tiling constraints
    for (auto const &it : dfg.nodes) {
        const DFG::Node &node = it.second;
        if (node.id >= 10000 || node.forOps.empty()) continue;

        for (unsigned l = 0; l < node.forOps.size(); ++l) {
            AffineForOp forOp = node.forOps[l];
            AffineLoopBand band;
            getLoopBandFromOutermost(forOp, band);

            for (unsigned i = 0; i < band.size(); ++i) {
                if (i >= node.candidates.tilingCandidates.size()) continue;
                std::string xVar = node.candidates.tilingCandidates[i];
                if (xVar.empty()) continue;

                int64_t ub = (int64_t)band[i].getConstantUpperBound();
                int64_t targetUb = (options.sharedTilingStrategy == "max") ? 
                                    sharedXUpperBounds[xVar] : ub;

                std::string zVar = "Z_" + std::to_string(node.id) + "_" + std::to_string(l) + "_" + std::to_string(i);

                // Tiling Variable Bounds: local ub for latency, targetUb (group max) for hardware limit
                int64_t lbHost = std::min((int64_t)options.minTimeLoopSize, ub);
                model.addVariable({zVar, "integer", lbHost, ub, {}, true, ""});

                if (options.enforceDivisorOnlyTiling) {
                    model.addConstraint("c_tiling_" + std::to_string(node.id) + "_" + std::to_string(l) + "_" + std::to_string(i),
                                        model.var(xVar) * model.var(zVar), Operator::Equal, SymbolicExpr(ub));
                    // Register relationship for C++ evaluator using local ub (dynamic latency)
                    model.addTilingGroup({xVar, zVar, ub, lbHost});
                } else {
                    // Padding-aware relation: ub <= x*z <= ub + x - 1
                    // This forces Z = ceil(ub / X), ensuring local (dynamic) latency.
                    // The variable itself is still allowed up to targetUb for hardware compatibility.
                    model.addConstraint("c_tiling_lb_" + std::to_string(node.id) + "_" + std::to_string(l) + "_" + std::to_string(i),
                                        model.var(xVar) * model.var(zVar), Operator::GreaterEqual, SymbolicExpr(ub));
                    
                    if(options.sharedTilingStrategy == "max"){
                      model.addConstraint("c_tiling_ub_" + std::to_string(node.id) + "_" + std::to_string(l) + "_" + std::to_string(i),
                                        model.var(xVar) * model.var(zVar), Operator::LessEqual,
                                        model.var(xVar) + SymbolicExpr(targetUb) - SymbolicExpr((int64_t)1));
                      // Register relationship for C++ evaluator using group ub (global latency)
                      model.addTilingGroup({xVar, zVar, ub, lbHost});
                    }else{
                      model.addConstraint("c_tiling_ub_" + std::to_string(node.id) + "_" + std::to_string(l) + "_" + std::to_string(i),
                                        model.var(xVar) * model.var(zVar), Operator::LessEqual,
                                        model.var(xVar) + SymbolicExpr(ub) - SymbolicExpr((int64_t)1));
                      // Register relationship for C++ evaluator using local ub (dynamic latency)
                      model.addTilingGroup({xVar, zVar, ub, lbHost});
                    }
                }
                
            }
        }
    }
}

} // namespace

void declareTilingParameters(PerformanceModel &model, DFG &dfg, const std::map<std::string, int64_t>& solution) {
  // 1. Declare X parameters from the provided solution or default to 1
  // We need to know which X variables are expected by the nodes in the DFG.
  for (auto const &it : dfg.nodes) {
    const DFG::Node &node = it.second;
    if (node.id >= 10000 || node.forOps.empty()) continue;
    for (const std::string &xVar : node.candidates.tilingCandidates) {
      if (xVar.empty()) continue;
      int64_t xVal = 1;
      auto solIt = solution.find(xVar);
      if (solIt != solution.end()) {
        xVal = solIt->second;
      } else {
        // Fallback to dfg.tilingVars if not in explicit solution
        auto dfgIt = dfg.tilingVars.find(xVar);
        if (dfgIt != dfg.tilingVars.end()) {
          xVal = dfgIt->second;
        }
      }
      auto it_param = std::find_if(model.parameters.begin(), model.parameters.end(),
                                  [&](const Parameter& p) { return p.name == xVar; });
      if (it_param == model.parameters.end()) {
        model.addParameter({xVar, "integer", xVal});
      }
    }
  }

  // 2. Derive and declare Z parameters: Z = ceil(UB / X)
  for (auto const &it : dfg.nodes) {
    const DFG::Node &node = it.second;
    if (node.id >= 10000 || node.forOps.empty()) continue;

    for (unsigned l = 0; l < node.forOps.size(); ++l) {
      AffineForOp forOp = node.forOps[l];
      AffineLoopBand band;
      getLoopBandFromOutermost(forOp, band);

      for (unsigned i = 0; i < band.size(); ++i) {
        if (i >= node.candidates.tilingCandidates.size()) continue;
        std::string xVar = node.candidates.tilingCandidates[i];
        if (xVar.empty()) continue;

        int64_t ub = (int64_t)band[i].getConstantUpperBound();
        std::string zVar = "Z_" + std::to_string(node.id) + "_" + std::to_string(l) + "_" + std::to_string(i);

        int64_t xVal = 1;
        auto it_p = std::find_if(model.parameters.begin(), model.parameters.end(),
                                [&](const Parameter& p) { return p.name == xVar; });
        if (it_p != model.parameters.end()) {
          xVal = it_p->value;
        }
        
        if (xVal <= 0) xVal = 1;
        // Always use local 'ub' for parameters (unpadding/dynamic latency)
        int64_t zVal = (ub + xVal - 1) / xVal; // ceilDiv
        model.addParameter({zVar, "integer", zVal});
      }
    }
  }
}



void declarePermutationParameters(PerformanceModel &model, DFG &dfg, const std::map<unsigned, unsigned>& solution) {
  for (auto &nodePair : dfg.nodes) {
    const DFG::Node &node = nodePair.second;
    if (node.id >= 10000) continue;

    unsigned permIdx = node.solution.permIdx;
    auto solIt = solution.find(node.id);
    if (solIt != solution.end()) {
      permIdx = solIt->second;
    }

    unsigned numCandidates = node.candidates.permutationCandidates.size();
    if (numCandidates > 0) {
      for (unsigned i = 0; i < numCandidates; ++i) {
        std::string varName = "b" + std::to_string(node.id) + "_" + std::to_string(i);
        int64_t val = (i == permIdx) ? 1 : 0;
        model.addParameter({varName, "integer", val});
      }
    }
  }
}

void declareDSPConstraints(PerformanceModel &model, DFG &dfg, int64_t dspLimit) {
  SymbolicExpr totalDSPs = SymbolicExpr((int64_t)0);

  for (auto const &it : dfg.nodes) {
    const DFG::Node &node = it.second;
    if (node.id >= 10000)
      continue;
    if (node.kind != DFG::NodeKind::COMPUTE)
      continue;

    SymbolicExpr nodeDSP = SymbolicExpr((int64_t)node.DSP_factor);
    bool hasTiling = false;

    // Multiply by all tiling factors X associated with this node
    for (const std::string &xVar : node.candidates.tilingCandidates) {
      if (!xVar.empty()) {
        nodeDSP = nodeDSP * model.var(xVar);
        hasTiling = true;
      }
    }

    if (hasTiling || node.DSP_factor > 0) {
      std::string dspVar = "DSP_" + std::to_string(node.id);
      LLVM_DEBUG(llvm::dbgs() << "Node " << node.id << " DSP factor: " << node.DSP_factor << "\n");
      model.addAssignment(dspVar, nodeDSP, "integer", true);
      totalDSPs = totalDSPs + model.var(dspVar);
    }
  }

  model.addAssignment("totalDSPs", totalDSPs, "integer", true);
  model.addConstraint("c_total_dsp", model.var("totalDSPs"), Operator::LessEqual, SymbolicExpr(dspLimit));
}

void declareObjective(PerformanceModel &model, DFG &dfg, const DSEOptions &options) {
  SymbolicExpr maxLw = SymbolicExpr((int64_t)0);
  bool first = true;

  auto vTopo = dfg.virtualTopologicalSort();
  for (auto const &vn : vTopo) {
    if (vn.nodeId >= 10000) continue;
    
    bool hasSuccessor = false;
    // 1. Physical intra-node successor
    DFG::Node *node = dfg.getNode(vn.nodeId);
    if (vn.loopIdx + 1 < node->forOps.size()) {
      hasSuccessor = true;
    }
    
    // 2. Inter-node successor
    if (!hasSuccessor && dfg.outEdges.count(vn.nodeId)) {
      for (const auto &edge : dfg.outEdges[vn.nodeId]) {
        if (edge.id < 10000 && dfg.getLoopIndex(vn.nodeId, edge.srcOp) == (int)vn.loopIdx) {
          hasSuccessor = true;
          break;
        }
      }
    }

    if (!hasSuccessor) {
      std::string varName = "lw" + std::to_string(vn.nodeId) + "_lp" + std::to_string(vn.loopIdx);
      if (first) {
        maxLw = model.var(varName);
        first = false;
      } else {
        maxLw = max(maxLw, model.var(varName));
      }
    }
  }

  model.addAssignment("latency", maxLw, "integer", true);

  // Normalization Factors
  uint64_t maxL = 0;
  for (auto const &it : dfg.nodes) {
    if (it.first < 10000) {
      maxL += it.second.tripCount;
    }
  }
  int64_t maxLatency = (maxL > 0) ? (int64_t)maxL : 1;

  uint64_t maxM = 0;
  for (auto const &it : dfg.inEdges) {
    if (it.first >= 10000) continue;
    for (const auto &inEdge : it.second) {
      if (inEdge.id >= 10000) continue;
      maxM += getMemRefSize(inEdge.value);
    }
  }
  int64_t maxMemory = (maxM > 0) ? (int64_t)maxM : 1;

  int64_t wL = options.latencyMemoryRatio;
  int64_t wM = 100 - options.latencyMemoryRatio;

  int64_t coeffL = wL * maxMemory;
  int64_t coeffM = wM * maxLatency;

  // Use GCD to simplify coefficients
  int64_t common = std::gcd(coeffL, coeffM);
  if (common > 0) {
    coeffL /= common;
    coeffM /= common;
  }

  // Scale down if still too large to prevent overflow (max coeff 1M)
  int64_t maxCoeff = std::max(coeffL, coeffM);
  if (maxCoeff > options.objectiveScalingFactor) {
    int64_t scale = maxCoeff / options.objectiveScalingFactor;
    coeffL = std::max((int64_t)1, coeffL / scale);
    coeffM = std::max((int64_t)1, coeffM / scale);
  }

  model.addAssignment("combined_cost", 
      SymbolicExpr(coeffL) * model.var("latency") + 
      SymbolicExpr(coeffM) * model.var("totalMemory"), "integer", true);

  model.setObjective(model.var("combined_cost"));
}


void declareStartTimeConstraints(PerformanceModel &model, DFG &dfg, unsigned nodeId, unsigned loopIdx) {
  DFG::Node *node = dfg.getNode(nodeId);
  std::vector<SymbolicExpr> arrivalTimes;
  std::string currSuffix = std::to_string(nodeId) + "_lp" + std::to_string(loopIdx);

  // 1. Intra-node sequential dependency
  if (loopIdx > 0) {
    arrivalTimes.push_back(model.var("lw" + std::to_string(nodeId) + "_lp" + std::to_string(loopIdx - 1)));
  }

  if (dfg.inEdges.count(nodeId)) {
    for (const auto &inEdge : dfg.inEdges[nodeId]) {
      // Filter edges that point to THIS specific loop index
      int dstIdx = 0;
      if (inEdge.dstOp) {
        dstIdx = dfg.getLoopIndex(nodeId, inEdge.dstOp);
      }
      if (dstIdx != (int)loopIdx) continue;

      SymbolicExpr edgeArrival = SymbolicExpr((int64_t)0);
      DFG::Node *srcNode = dfg.getNode(inEdge.id);

      if (srcNode->id >= 10000) {
        // Root node always has arrival time 0
        edgeArrival = SymbolicExpr((int64_t)0);
      } else {
        int srcIdx = 0;
        if (inEdge.srcOp) {
          srcIdx = dfg.getLoopIndex(inEdge.id, inEdge.srcOp);
        }
        std::string srcVarSuffix = std::to_string(srcNode->id) + "_lp" + std::to_string(srcIdx);

        for (unsigned pSrc = 0; pSrc < srcNode->candidates.permutationCandidates.size(); ++pSrc) {
          std::string bSrc = "b" + std::to_string(srcNode->id) + "_" + std::to_string(pSrc);
          const auto &srcPerm = srcNode->candidates.permutationCandidates[pSrc];
          
          AffineMap srcAccessMap;
          if (inEdge.srcOp && srcPerm.storesMap.count(inEdge.srcOp)) {
            srcAccessMap = srcPerm.storesMap.lookup(inEdge.srcOp).accessMap;
          }

          for (unsigned pCurr = 0; pCurr < node->candidates.permutationCandidates.size(); ++pCurr) {
            std::string bCurr = "b" + std::to_string(node->id) + "_" + std::to_string(pCurr);
            const auto &currPerm = node->candidates.permutationCandidates[pCurr];
            
            AffineMap currAccessMap;
            if (inEdge.dstOp && currPerm.loadsMap.count(inEdge.dstOp)) {
              currAccessMap = currPerm.loadsMap.lookup(inEdge.dstOp).accessMap;
            }

            SymbolicExpr select = model.var(bSrc) * model.var(bCurr);
            if (srcAccessMap && currAccessMap && srcAccessMap == currAccessMap) {
              edgeArrival = edgeArrival + select * model.var("fw" + srcVarSuffix);
            } else {
              edgeArrival = edgeArrival + select * model.var("lw" + srcVarSuffix);
            }
          }
        }
      }
      arrivalTimes.push_back(edgeArrival);
    }
  }

  SymbolicExpr stVal = SymbolicExpr((int64_t)0);
  if (!arrivalTimes.empty()) {
    stVal = arrivalTimes[0];
    for (size_t i = 1; i < arrivalTimes.size(); ++i) {
      stVal = max(stVal, arrivalTimes[i]);
    }
  }
  model.addAssignment("st" + currSuffix, stVal);
}


void declareFirstWriteTimeConstraints(PerformanceModel &model, DFG &dfg, unsigned nodeId, unsigned loopIdx) {
  DFG::Node *node = dfg.getNode(nodeId);
  std::string currSuffix = std::to_string(nodeId) + "_lp" + std::to_string(loopIdx);
  SymbolicExpr fwVal = model.var("st" + currSuffix);

  for (unsigned p = 0; p < node->candidates.permutationCandidates.size(); ++p) {
    std::string bVar = "b" + std::to_string(nodeId) + "_" + std::to_string(p);
    const auto &perm = node->candidates.permutationCandidates[p];
    
    // Mapping symbols in getAccessTime to model variables
    std::vector<std::string> symbols;
    for (unsigned i = 0; i < node->bandSize; ++i) {
      auto permIdx = inversePerm(perm.permutation)[i];
      symbols.push_back("Z_" + std::to_string(nodeId) + "_" + std::to_string(loopIdx) + "_" + std::to_string(permIdx));
    }
    std::reverse(symbols.begin(), symbols.end());

    // Find the first store of THIS specific kernel
    AffineStoreOp targetStore;
    for (auto storeOp : node->stores) {
      if (dfg.getLoopIndex(nodeId, storeOp) == (int)loopIdx) {
        targetStore = storeOp;
        break;
      }
    }

    if (targetStore && perm.storesMap.count(targetStore)) {
      SymbolicExpr nodeAccessTime = toSymbolic(perm.storesMap.lookup(targetStore).storeFirstAccessTime.expr, symbols, model);
      // only print for node 4
      // if(nodeId == 4){
      //   // change format for python eval example:
      //   // print("Expr: s0 - 1")
      //   // print(f"Inputs: {Z_4_0_0} {Z_4_0_1} {Z_4_0_2}") 
      //   // print(f"Eval: {(Z_4_0_0 + -1)}")
      //   llvm::dbgs() << "print(\"Expr: " << perm.storesMap.lookup(targetStore).storeFirstAccessTime.expr << "\")\n";
      //   llvm::dbgs() << "print(f\"Inputs: ";
      //   for(auto &s : symbols){
      //     llvm::dbgs() << "{" << s << "} ";
      //   }
      //   llvm::dbgs() << "\")\n";
      //   llvm::dbgs() << "print(f\"Eval: {(" << nodeAccessTime.toAMPL() << ")}\")\n\n";
      // }
      fwVal = fwVal + model.var(bVar) * nodeAccessTime * SymbolicExpr((int64_t)perm.II);
    }
  }

  model.addAssignment("fw" + currSuffix, fwVal);
}


void declareLastWriteTimeConstraints(PerformanceModel &model, DFG &dfg, unsigned nodeId, unsigned loopIdx) {
  DFG::Node *node = dfg.getNode(nodeId);
  std::string currSuffix = std::to_string(nodeId) + "_lp" + std::to_string(loopIdx);

  // 1. Relative Last Read (rlr) and Epilogues (ep) per input edge
  if (dfg.inEdges.count(nodeId)) {
    for (const auto &inEdge : dfg.inEdges[nodeId]) {
      // Filter edges that point to THIS specific loop index
      int dstIdx = 0;
      if (inEdge.dstOp) {
        dstIdx = dfg.getLoopIndex(nodeId, inEdge.dstOp);
      }
      if (dstIdx != (int)loopIdx) continue;

      int srcIdx = 0;
      if (inEdge.srcOp) {
        srcIdx = dfg.getLoopIndex(inEdge.id, inEdge.srcOp);
      }

      std::string edgeSuffix = std::to_string(inEdge.id) + "_lp" + std::to_string(srcIdx) + 
                               "_" + std::to_string(nodeId) + "_lp" + std::to_string(loopIdx);
      
      SymbolicExpr rlrVal = model.var("st" + currSuffix);
      SymbolicExpr epVal = SymbolicExpr((int64_t)0);

      for (unsigned p = 0; p < node->candidates.permutationCandidates.size(); ++p) {
        std::string bVar = "b" + std::to_string(nodeId) + "_" + std::to_string(p);
        const auto &perm = node->candidates.permutationCandidates[p];
        SymbolicExpr ii = SymbolicExpr((int64_t)perm.II);

        std::vector<std::string> symbols;
        for (unsigned i = 0; i < node->bandSize; ++i) { 
          auto permIdx = inversePerm(perm.permutation)[i];
          symbols.push_back("Z_" + std::to_string(nodeId) + "_" + std::to_string(loopIdx) + "_" + std::to_string(permIdx));
        }
        std::reverse(symbols.begin(), symbols.end());

        // rlr calculation
        if (inEdge.dstOp && perm.loadsMap.count(inEdge.dstOp)) {
          SymbolicExpr alr = toSymbolic(perm.loadsMap.lookup(inEdge.dstOp).loadLastAccessTime.expr, symbols, model);
          rlrVal = rlrVal + model.var(bVar) * alr * ii;
        }

        // ep calculation: alw - alr
        AffineStoreOp targetStore;
        for (auto storeOp : node->stores) {
          if (dfg.getLoopIndex(nodeId, storeOp) == (int)loopIdx) {
            targetStore = storeOp;
            break;
          }
        }
        if (targetStore && perm.storesMap.count(targetStore) && inEdge.dstOp && perm.loadsMap.count(inEdge.dstOp)) {
          SymbolicExpr alw = toSymbolic(perm.storesMap.lookup(targetStore).storeLastAccessTime.expr, symbols, model);
          SymbolicExpr alr = toSymbolic(perm.loadsMap.lookup(inEdge.dstOp).loadLastAccessTime.expr, symbols, model);
          epVal = epVal + model.var(bVar) * (alw - alr) * ii;
        }
      }
      model.addAssignment("rlr" + edgeSuffix, rlrVal);
      model.addAssignment("ep" + edgeSuffix, epVal);

      // 2. Dependency logic
      SymbolicExpr srcLw = (inEdge.id >= 10000) ? SymbolicExpr((int64_t)0) : 
                                                 model.var("lw" + std::to_string(inEdge.id) + "_lp" + std::to_string(srcIdx));
      model.addAssignment("lt" + edgeSuffix, max(model.var("rlr" + edgeSuffix), srcLw));
      model.addAssignment("lw" + edgeSuffix, model.var("lt" + edgeSuffix) + model.var("ep" + edgeSuffix));
    }
  }

  // 3. Final node lw = max of all incoming lw candidates AND sequential predecessors
  SymbolicExpr lwNodeVal = SymbolicExpr((int64_t)0);
  std::vector<SymbolicExpr> candidates;

  // sequential predecessor in same node
  if (loopIdx > 0) {
    for (unsigned p = 0; p < node->candidates.permutationCandidates.size(); ++p) {
      std::string bVar = "b" + std::to_string(nodeId) + "_" + std::to_string(p);
      const auto &perm = node->candidates.permutationCandidates[p];
      
      std::vector<std::string> symbols;
      for (unsigned i = 0; i < node->bandSize; ++i) { 
        auto permIdx = inversePerm(perm.permutation)[i];
        symbols.push_back("Z_" + std::to_string(nodeId) + "_" + std::to_string(loopIdx) + "_" + std::to_string(permIdx));
      }
      std::reverse(symbols.begin(), symbols.end());

      AffineStoreOp targetStore;
      for (auto storeOp : node->stores) {
        if (dfg.getLoopIndex(nodeId, storeOp) == (int)loopIdx) {
          targetStore = storeOp;
          break;
        }
      }
      if (targetStore && perm.storesMap.count(targetStore)) {
        SymbolicExpr alw = toSymbolic(perm.storesMap.lookup(targetStore).storeLastAccessTime.expr, symbols, model);
        SymbolicExpr prevLw = model.var("lw" + std::to_string(nodeId) + "_lp" + std::to_string(loopIdx - 1));
        candidates.push_back(prevLw + model.var(bVar) * alw * SymbolicExpr((int64_t)perm.II));
      }
    }
  }

  if (dfg.inEdges.count(nodeId)) {
    for (const auto &inEdge : dfg.inEdges[nodeId]) {
      int dstIdx = (inEdge.dstOp) ? dfg.getLoopIndex(nodeId, inEdge.dstOp) : 0;
      if (dstIdx == (int)loopIdx) {
        int srcIdx = (inEdge.srcOp) ? dfg.getLoopIndex(inEdge.id, inEdge.srcOp) : 0;
        std::string edgeSuffix = std::to_string(inEdge.id) + "_lp" + std::to_string(srcIdx) + 
                                 "_" + std::to_string(nodeId) + "_lp" + std::to_string(loopIdx);
        candidates.push_back(model.var("lw" + edgeSuffix));
      }
    }
  }

  if (!candidates.empty()) {
    lwNodeVal = candidates[0];
    for (size_t i = 1; i < candidates.size(); ++i) {
      lwNodeVal = max(lwNodeVal, candidates[i]);
    }
  } else {
    // For root nodes or nodes with no inputs/seq-predecessors, calculate lw from own write epilogue
    for (unsigned p = 0; p < node->candidates.permutationCandidates.size(); ++p) {
        std::string bVar = "b" + std::to_string(nodeId) + "_" + std::to_string(p);
        const auto &perm = node->candidates.permutationCandidates[p];

        std::vector<std::string> symbols;
        for (unsigned i = 0; i < node->bandSize; ++i) { 
          auto permIdx = inversePerm(perm.permutation)[i];
          symbols.push_back("Z_" + std::to_string(nodeId) + "_" + std::to_string(loopIdx) + "_" + std::to_string(permIdx));
        }
        std::reverse(symbols.begin(), symbols.end());

        AffineStoreOp targetStore;
        for (auto storeOp : node->stores) {
          if (dfg.getLoopIndex(nodeId, storeOp) == (int)loopIdx) {
            targetStore = storeOp;
            break;
          }
        }
        if (targetStore && perm.storesMap.count(targetStore)) {
          SymbolicExpr alw = toSymbolic(perm.storesMap.lookup(targetStore).storeLastAccessTime.expr, symbols, model);
          lwNodeVal = lwNodeVal + model.var(bVar) * alw * SymbolicExpr((int64_t)perm.II);
        }
    }
  }
  
  model.addAssignment("lw" + currSuffix, lwNodeVal);
}

void calculateMemoryFootprint(PerformanceModel &model, DFG &dfg) {
  SymbolicExpr totalMemory = SymbolicExpr((int64_t)0);
  for (auto const &it : dfg.inEdges) {
    unsigned nodeId = it.first;
    if (nodeId >= 10000) continue;
    DFG::Node *node = dfg.getNode(nodeId);

    for (const auto &inEdge : it.second) {
      if (inEdge.id >= 10000) continue;
      DFG::Node *srcNode = dfg.getNode(inEdge.id);

      int64_t footprint = getMemRefSize(inEdge.value);
      SymbolicExpr edgeMemory = SymbolicExpr((int64_t)0);

      for (unsigned pSrc = 0; pSrc < srcNode->candidates.permutationCandidates.size(); ++pSrc) {
        std::string bSrc = "b" + std::to_string(srcNode->id) + "_" + std::to_string(pSrc);
        const auto &srcPerm = srcNode->candidates.permutationCandidates[pSrc];
        AffineMap srcAccessMap;
        if (inEdge.srcOp && srcPerm.storesMap.count(inEdge.srcOp)) {
          srcAccessMap = srcPerm.storesMap.lookup(inEdge.srcOp).accessMap;
        }

        for (unsigned pCurr = 0; pCurr < node->candidates.permutationCandidates.size(); ++pCurr) {
          std::string bCurr = "b" + std::to_string(node->id) + "_" + std::to_string(pCurr);
          const auto &currPerm = node->candidates.permutationCandidates[pCurr];
          AffineMap currAccessMap;
          if (inEdge.dstOp && currPerm.loadsMap.count(inEdge.dstOp)) {
            currAccessMap = currPerm.loadsMap.lookup(inEdge.dstOp).accessMap;
          }

          SymbolicExpr select = model.var(bSrc) * model.var(bCurr);
          if (!(srcAccessMap && currAccessMap && srcAccessMap == currAccessMap)) {
            edgeMemory = edgeMemory + select * SymbolicExpr(footprint);
          }
        }
      }
      std::string bufferName = streamhls::getMemrefBufferName(inEdge.value);
      std::string edgeName = "footprint_" + std::to_string(inEdge.id) + "_to_" + std::to_string(nodeId);
      if (!bufferName.empty()) {
        edgeName += "_" + bufferName;
      }
      model.addAssignment(edgeName, edgeMemory, "integer", false);
      totalMemory = totalMemory + model.var(edgeName);
    }
  }
  model.addAssignment("totalMemory", totalMemory, "integer", true);
}

void createLoopPermutationPerformanceModel(PerformanceModel &model, DFG &dfg, const DSEOptions &options, const std::map<std::string, int64_t>& tilingSolution) {
  
  declareTilingParameters(model, dfg, tilingSolution);
  declarePermutationVariables(model, dfg);
  calculateMemoryFootprint(model, dfg);

  auto vTopo = dfg.virtualTopologicalSort();
  for (auto const &vn : vTopo) {
    if (vn.nodeId >= 10000) continue;
    declareStartTimeConstraints(model, dfg, vn.nodeId, vn.loopIdx);
    declareLastWriteTimeConstraints(model, dfg, vn.nodeId, vn.loopIdx);
    declareFirstWriteTimeConstraints(model, dfg, vn.nodeId, vn.loopIdx);
  }

  declareObjective(model, dfg, options);
}

void createLoopUnrollingPerformanceModel(PerformanceModel &model, DFG &dfg, const DSEOptions &options, const std::map<unsigned, unsigned>& permutationSolution) {
  
  declarePermutationParameters(model, dfg, permutationSolution);
  declareTilingVariables(model, dfg, options);
  declareDSPConstraints(model, dfg, options.dspLimit);
  calculateMemoryFootprint(model, dfg);

  auto vTopo = dfg.virtualTopologicalSort();
  for (auto const &vn : vTopo) {
    if (vn.nodeId >= 10000) continue;
    declareStartTimeConstraints(model, dfg, vn.nodeId, vn.loopIdx);
    declareLastWriteTimeConstraints(model, dfg, vn.nodeId, vn.loopIdx);
    declareFirstWriteTimeConstraints(model, dfg, vn.nodeId, vn.loopIdx);
  }

  declareObjective(model, dfg, options);
}

void createCombinedPerformanceModel(PerformanceModel &model, DFG &dfg, const DSEOptions &options) {

  declarePermutationVariables(model, dfg);
  declareTilingVariables(model, dfg, options);
  declareDSPConstraints(model, dfg, options.dspLimit);
  calculateMemoryFootprint(model, dfg);

  auto vTopo = dfg.virtualTopologicalSort();
  for (auto const &vn : vTopo) {
    if (vn.nodeId >= 10000) continue;
    declareStartTimeConstraints(model, dfg, vn.nodeId, vn.loopIdx);
    declareLastWriteTimeConstraints(model, dfg, vn.nodeId, vn.loopIdx);
    declareFirstWriteTimeConstraints(model, dfg, vn.nodeId, vn.loopIdx);
  }

  declareObjective(model, dfg, options);
}

void createLatencyPerformanceModel(PerformanceModel &model, DFG &dfg, 
                                   const std::map<unsigned, unsigned>& permSolution,
                                   const std::map<std::string, int64_t>& tilingSolution) {
    
    declareTilingParameters(model, dfg, tilingSolution);
    declarePermutationParameters(model, dfg, permSolution);
    calculateMemoryFootprint(model, dfg);

    auto vTopo = dfg.virtualTopologicalSort();
    for (auto const &vn : vTopo) {
      if (vn.nodeId >= 10000) continue;
      declareStartTimeConstraints(model, dfg, vn.nodeId, vn.loopIdx);
      declareLastWriteTimeConstraints(model, dfg, vn.nodeId, vn.loopIdx);
      declareFirstWriteTimeConstraints(model, dfg, vn.nodeId, vn.loopIdx);
    }
    DSEOptions defaultOptions;
    declareObjective(model, dfg, defaultOptions);
}

} // namespace streamhls
} // namespace mlir