#include "streamhls/Dialect/Dataflow/Support/LoopCombinedModel.h"
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

#define DEBUG_TYPE "loop-combined-model"

namespace mlir {
namespace streamhls {

SolveResult runLoopCombinedMINLP(DFG &dfg, std::string projectPath, const DSEOptions &options) {
    LLVM_DEBUG(llvm::dbgs() << "--- Running Loop Combined MINLP ---\n");

    // 1. Create helper nodes for modeling
    dfg.createRootNode();
    dfg.createSinkNode();

    PerformanceModel model;
    createCombinedPerformanceModel(model, dfg, options);
    model.finalize();
 
    // Initial latency
    llvm::dbgs() << "DSE Search Method: [MINLP (AMPL)]\n";
    dfg.getPerformanceMetrics(options).printReport(llvm::dbgs(), "Initial Combined");

    // Write and Solve
    SmallString<128> modPath(projectPath);
    llvm::sys::path::append(modPath, "loop_combined.mod");
    std::string modPathStr = modPath.str().str();
    writeAMPLModel(model, modPathStr, options.combinedAMPLSolver);

    SolveResult result = solvePerformanceModel(modPathStr, options.combinedTimeout);
    if (result.success && result.cost != -1) {
        result.printReport(llvm::dbgs(), "AMPL Combined");
    } else {
        llvm::errs() << "No valid solution found in solver output:\n" << result.fullOutput << "\n";
        // Cleanup modeling nodes
        dfg.removeNode(10000); // Root
        dfg.removeNode(10001); // Sink
        return result;
    }

    // Update DFG with results for both permutation and tiling variables
    // dfg.updatePermutationSolution(result);
    // dfg.updateUnrollingSolution(result);
    
    // dfg.getPerformanceMetrics(options).printReport(llvm::dbgs(), "Final Combined");
    // dfg.printFixedPermutationSolution();
    // dfg.printFixedUnrollingSolution();

    // Cleanup modeling nodes
    dfg.removeNode(10000); // Root
    dfg.removeNode(10001); // Sink

    return result;
}

SolveResult runLoopCombinedSA(DFG &dfg, const DSEOptions &options) {
    LLVM_DEBUG(llvm::dbgs() << "--- Running Loop Combined SA DSE ---\n");

    dfg.createRootNode();
    dfg.createSinkNode();

    PerformanceModel model;
    createCombinedPerformanceModel(model, dfg, options);
    model.finalize();
 
    llvm::dbgs() << "DSE Search Method: [SA]\n";
    dfg.getPerformanceMetrics(options).printReport(llvm::dbgs(), "Initial Combined [SA]");

    auto result = runSimulatedAnnealing(model, options.combinedSAConfig);
    // dfg.updatePermutationSolution(result);
    // dfg.updateUnrollingSolution(result);

    result.printReport(llvm::dbgs(), "Final Combined [SA]");
    // dfg.printFixedPermutationSolution();
    // dfg.printFixedUnrollingSolution();

    dfg.removeNode(10000); // Root
    dfg.removeNode(10001); // Sink

    return result;
}

SolveResult runLoopCombinedGA(DFG &dfg, const DSEOptions &options) {
    LLVM_DEBUG(llvm::dbgs() << "--- Running Loop Combined GA DSE ---\n");

    dfg.createRootNode();
    dfg.createSinkNode();

    PerformanceModel model;
    createCombinedPerformanceModel(model, dfg, options);
    model.finalize();
 
    llvm::dbgs() << "DSE Search Method: [GA]\n";
    dfg.getPerformanceMetrics(options).printReport(llvm::dbgs(), "Initial Combined [GA]");

    auto result = runGeneticAlgorithm(model, options.combinedGAConfig);
    // dfg.updatePermutationSolution(result);
    // dfg.updateUnrollingSolution(result);

    result.printReport(llvm::dbgs(), "Final Combined [GA]");
    // dfg.printFixedPermutationSolution();
    // dfg.printFixedUnrollingSolution();

    dfg.removeNode(10000); // Root
    dfg.removeNode(10001); // Sink

    return result;
}

SolveResult findRandomCombinedSolution(DFG &dfg, const DSEOptions &options) {
  LLVM_DEBUG(llvm::dbgs() << "--- Finding Random Combined Solution ---\n");
  
  dfg.createRootNode();
  dfg.createSinkNode();

  PerformanceModel model;
  createCombinedPerformanceModel(model, dfg, options);
  model.finalize();

  llvm::dbgs() << "DSE Search Method: [Random Search]\n";
  dfg.getPerformanceMetrics(options).printReport(llvm::dbgs(), "Initial Combined");
  auto result = runRandomSearch(model, options.combinedRandomConfig);
  
  // bool permutationUpdated = dfg.updatePermutationSolution(result);
  // bool unrollingUpdated = dfg.updateUnrollingSolution(result);

  // if (result.success) {
  //   result.printReport(llvm::dbgs(), "Final Combined");
  //   dfg.printFixedPermutationSolution();
  //   dfg.printFixedUnrollingSolution();
  // } else {
  //   llvm::errs() << "Error: No valid random combined solution found.\n";
  // }

  dfg.removeNode(10000); // Root
  dfg.removeNode(10001); // Sink

  return result;
}

} // namespace streamhls
} // namespace mlir
