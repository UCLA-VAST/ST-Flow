#include "streamhls/Dialect/Dataflow/Support/LoopPermutationModel.h"
#include "streamhls/Dialect/Dataflow/Support/Utils.h"
#include "streamhls/Dialect/Dataflow/Support/Equations.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Path.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Debug.h"
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <ctime>


#define DEBUG_TYPE "loop-permutation-model"

namespace mlir {
namespace streamhls {

bool setFixedPermutations(DFG &dfg, const DSEOptions &options) {
  LLVM_DEBUG(llvm::dbgs() << "--- Setting Fixed Permutations ---\n");
  for (auto &nodePair : dfg.nodes) {
    auto &node = nodePair.second;
    if (node.candidates.permutationCandidates.empty())
      continue;

    unsigned nodeId = nodePair.first;
    if (options.fixedPermutationSolution.count(nodeId)) {
      unsigned permIdx = options.fixedPermutationSolution.at(nodeId);
      if (permIdx >= node.candidates.permutationCandidates.size()) {
        llvm::errs() << "Error: Fixed permutation index " << permIdx 
                     << " for node " << nodeId << " is out of bounds (max " 
                     << node.candidates.permutationCandidates.size() - 1 << ")\n";
        return false;
      }
      node.solution.permIdx = permIdx;
      llvm::dbgs() << "Node " << nodeId << " fixed permutation index: " << permIdx << "\n";
    } else {
      // Default to 0 if not specified? Or error?
      // User said: "If the solution is invalid, throw an error."
      // Maybe not being in the map is an error if 'fixed' is selected.
      llvm::errs() << "Error: No fixed permutation index provided for node " << nodeId << "\n";
      return false;
    }
  }
  return true;
}

SolveResult findRandomPermutationSolution(DFG &dfg, const DSEOptions &options) {
  LLVM_DEBUG(llvm::dbgs() << "--- Finding Random Permutations ---\n");
  
  dfg.createRootNode();
  dfg.createSinkNode();

  PerformanceModel model;
  createLoopPermutationPerformanceModel(model, dfg, options, dfg.getTilingSolution());
  model.finalize();

  llvm::dbgs() << "DSE Search Method: [Random Search]\n";
  dfg.getPerformanceMetrics(options).printReport(llvm::dbgs(), "Initial Permutation");
  auto result = runRandomSearch(model, options.permutationRandomConfig);
  // dfg.updatePermutationSolution(result);

  // if (result.success) {
  //   result.printReport(llvm::dbgs(), "Final Permutation");
  //   dfg.printFixedPermutationSolution();
  // } else {
  //   llvm::errs() << "Error: No valid random permutation solution found.\n";
  // }

  dfg.removeNode(10000); // Root
  dfg.removeNode(10001); // Sink

  return result;
}

SolveResult runLoopPermutationMINLP(DFG &dfg, llvm::StringRef projectPath, const DSEOptions &options) {
    LLVM_DEBUG(llvm::dbgs() << "--- Running Loop Permutation MINLP ---\n");

    // 1. Create helper nodes for modeling if they don't exist
    dfg.createRootNode();
    dfg.createSinkNode();

    PerformanceModel model;
    createLoopPermutationPerformanceModel(model, dfg, options, dfg.getTilingSolution());
    model.finalize();
    
    llvm::dbgs() << "DSE Search Method: [MINLP (AMPL)]\n";
    dfg.getPerformanceMetrics(options).printReport(llvm::dbgs(), "Initial Permutation");

    // 5. Write and Solve
    SmallString<128> modPath(projectPath);
    llvm::sys::path::append(modPath, "loop_permutation.mod");
    std::string modPathStr = modPath.str().str();
    writeAMPLModel(model, modPathStr, options.permutationAMPLSolver);

    SolveResult result = solvePerformanceModel(modPathStr, options.permutationTimeout);
    if (result.success && result.cost != -1) {
      result.printReport(llvm::dbgs(), "AMPL Permutation");
    } else {
        llvm::errs() << "No solution found in solver output:\n" << result.fullOutput << "\n";
        // Cleanup modeling nodes
        dfg.removeNode(10000); // Root
        dfg.removeNode(10001); // Sink
        return result;
    }

    // 6. Update DFG with results
    // bool solutionFound = dfg.updatePermutationSolution(result);

    // if (solutionFound) {
    //     dfg.getPerformanceMetrics(options).printReport(llvm::dbgs(), "Final Permutation");
    //     dfg.printFixedPermutationSolution();
    // }

    // 8. Cleanup modeling nodes
    dfg.removeNode(10000); // Root
    dfg.removeNode(10001); // Sink

    return result;
}

SolveResult runLoopPermutationSA(DFG &dfg, const DSEOptions &options) {
    LLVM_DEBUG(llvm::dbgs() << "--- Running Loop Permutation SA DSE ---\n");

    dfg.createRootNode();
    dfg.createSinkNode();

    PerformanceModel model;
    createLoopPermutationPerformanceModel(model, dfg, options, dfg.getTilingSolution());
    model.finalize();
    llvm::dbgs() << "DSE Search Method: [SA]\n";
    llvm::dbgs() << "Objective: " << model.objective.expressionStr << "\n";
    dfg.getPerformanceMetrics(options).printReport(llvm::dbgs(), "Initial Permutation [SA]");

    auto result = runSimulatedAnnealing(model, options.permutationSAConfig);
    // dfg.updatePermutationSolution(result);

    result.printReport(llvm::dbgs(), "Final Permutation [SA]");
    // dfg.printFixedPermutationSolution();

    dfg.removeNode(10000); // Root
    dfg.removeNode(10001); // Sink

    return result;
}

SolveResult runLoopPermutationGA(DFG &dfg, const DSEOptions &options) {
    LLVM_DEBUG(llvm::dbgs() << "--- Running Loop Permutation GA DSE ---\n");

    dfg.createRootNode();
    dfg.createSinkNode();

    PerformanceModel model;
    createLoopPermutationPerformanceModel(model, dfg, options, dfg.getTilingSolution());
    model.finalize();
    llvm::dbgs() << "DSE Search Method: [GA]\n";
    llvm::dbgs() << "Objective: " << model.objective.expressionStr << "\n";
    dfg.getPerformanceMetrics(options).printReport(llvm::dbgs(), "Initial Permutation [GA]");

    auto result = runGeneticAlgorithm(model, options.permutationGAConfig);
    // dfg.updatePermutationSolution(result);

    result.printReport(llvm::dbgs(), "Final Permutation [GA]");
    // dfg.printFixedPermutationSolution();

    dfg.removeNode(10000); // Root
    dfg.removeNode(10001); // Sink

    return result;
}

} // namespace streamhls
} // namespace mlir
