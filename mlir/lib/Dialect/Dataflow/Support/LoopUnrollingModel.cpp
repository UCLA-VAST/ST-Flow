#include "streamhls/Dialect/Dataflow/Support/LoopUnrollingModel.h"
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

#define DEBUG_TYPE "loop-unrolling-model"

namespace mlir {
namespace streamhls {


bool setFixedUnrollingFactors(DFG &dfg, const DSEOptions &options) {
  LLVM_DEBUG(llvm::dbgs() << "--- Setting Fixed Unrolling Factors ---\n");
  SolveResult result;
  result.success = true;
  for (auto const &pair : options.fixedUnrollingSolution) {
    result.variables[pair.first] = pair.second;
  }
  if (!dfg.updateUnrollingSolution(result)) {
    llvm::errs() << "Error: Invalid fixed unrolling solution provided.\n";
    return false;
  }
  return true;
}

SolveResult findRandomUnrollingSolution(DFG &dfg, const DSEOptions &options) {
  LLVM_DEBUG(llvm::dbgs() << "--- Finding Random Unrolling Factors ---\n");
  
  dfg.createRootNode();
  dfg.createSinkNode();

  PerformanceModel model;
  createLoopUnrollingPerformanceModel(model, dfg, options, dfg.getPermutationSolution());
  model.finalize();

  llvm::dbgs() << "DSE Search Method: [Random Search]\n";
  dfg.getPerformanceMetrics(options).printReport(llvm::dbgs(), "Initial Unrolling");
  auto result = runRandomSearch(model, options.unrollingRandomConfig);
  // dfg.updateUnrollingSolution(result);

  // if (result.success) {
  //   result.printReport(llvm::dbgs(), "Final Unrolling");
  //   dfg.printFixedUnrollingSolution();
  // } else {
  //   llvm::errs() << "Error: No valid random unrolling solution found.\n";
  // }

  dfg.removeNode(10000); // Root
  dfg.removeNode(10001); // Sink

  return result;
}

bool setUnrollingFactorsToOnes(DFG &dfg){
  LLVM_DEBUG(llvm::dbgs() << "--- Setting Unrolling Factors to Ones ---\n");
  llvm::dbgs() << "Initial Loop Unrolling Latency: " << dfg.getLatency() << "\n";
  dfg.populateTilingVarsWithOnes();
  llvm::dbgs() << "Final Loop Unrolling Latency: " << dfg.getLatency() << "\n";
  return true;
}

SolveResult runLoopUnrollingMINLP(DFG &dfg, std::string projectPath, const DSEOptions &options) {
    LLVM_DEBUG(llvm::dbgs() << "--- Running Loop Unrolling MINLP ---\n");

    // 1. Create helper nodes for modeling
    dfg.createRootNode();
    dfg.createSinkNode();

    PerformanceModel model;
    createLoopUnrollingPerformanceModel(model, dfg, options, dfg.getPermutationSolution());
    model.finalize();
 
    llvm::dbgs() << "DSE Search Method: [MINLP (AMPL)]\n";
    llvm::dbgs() << "Objective: " << model.objective.expressionStr << "\n";
    dfg.getPerformanceMetrics(options).printReport(llvm::dbgs(), "Initial Unrolling");

    // 7. Write and Solve
    SmallString<128> modPath(projectPath);
    llvm::sys::path::append(modPath, "loop_unrolling.mod");
    std::string modPathStr = modPath.str().str();
    writeAMPLModel(model, modPathStr, options.unrollingAMPLSolver);

    SolveResult result = solvePerformanceModel(modPathStr, options.unrollingTimeout);
    if (result.success && result.cost != -1) {
        result.printReport(llvm::dbgs(), "AMPL Unrolling");
    } else {
        llvm::errs() << "No solution found in solver output:\n" << result.fullOutput << "\n";
        // Cleanup modeling nodes (do not apply unrolling/parallelization here)
        dfg.removeNode(10000); // Root
        dfg.removeNode(10001); // Sink
        return result;
    }

    // 8. Update DFG with results for tiling variables (X_#)
    // dfg.updateUnrollingSolution(result);
    
    // if (result.success) {
    //     dfg.getPerformanceMetrics(options).printReport(llvm::dbgs(), "Final Unrolling");
    //     dfg.printFixedUnrollingSolution();
    // }

    // 10. Cleanup modeling nodes (do not apply unrolling/parallelization here)
    dfg.removeNode(10000); // Root
    dfg.removeNode(10001); // Sink

    return result;
}

SolveResult runLoopUnrollingSA(DFG &dfg, const DSEOptions &options) {
    LLVM_DEBUG(llvm::dbgs() << "--- Running Loop Unrolling SA DSE ---\n");

    dfg.createRootNode();
    dfg.createSinkNode();

    PerformanceModel model;
    createLoopUnrollingPerformanceModel(model, dfg, options, dfg.getPermutationSolution());
    model.finalize();
 
    llvm::dbgs() << "DSE Search Method: [SA]\n";
    llvm::dbgs() << "Objective: " << model.objective.expressionStr << "\n";
    dfg.getPerformanceMetrics(options).printReport(llvm::dbgs(), "Initial Unrolling [SA]");

    auto result = runSimulatedAnnealing(model, options.unrollingSAConfig);
    // dfg.updateUnrollingSolution(result);

    result.printReport(llvm::dbgs(), "Final Unrolling [SA]");
    // dfg.printFixedUnrollingSolution();

    dfg.removeNode(10000); // Root
    dfg.removeNode(10001); // Sink

    return result;
}


SolveResult runLoopUnrollingGA(DFG &dfg, const DSEOptions &options) {
    LLVM_DEBUG(llvm::dbgs() << "--- Running Loop Unrolling GA DSE ---\n");

    dfg.createRootNode();
    dfg.createSinkNode();

    PerformanceModel model;
 
    createLoopUnrollingPerformanceModel(model, dfg, options, dfg.getPermutationSolution());
    model.finalize();
 
 
    llvm::dbgs() << "DSE Search Method: [GA]\n";
    llvm::dbgs() << "Objective: " << model.objective.expressionStr << "\n";
    dfg.getPerformanceMetrics(options).printReport(llvm::dbgs(), "Initial Unrolling [GA]");

    auto result = runGeneticAlgorithm(model, options.unrollingGAConfig);
    // dfg.updateUnrollingSolution(result);

    result.printReport(llvm::dbgs(), "Final Unrolling [GA]");
    // dfg.printFixedUnrollingSolution();

    dfg.removeNode(10000); // Root
    dfg.removeNode(10001); // Sink

    return result;
}

} // namespace streamhls
} // namespace mlir
