#ifndef MLIR_STREAMHLS_DIALECT_DATAFLOW_SUPPORT_LOOPCOMBINEDMODEL_H
#define MLIR_STREAMHLS_DIALECT_DATAFLOW_SUPPORT_LOOPCOMBINEDMODEL_H

#include "streamhls/Dialect/Dataflow/Support/DFG.h"
#include "streamhls/Dialect/Dataflow/Support/PerformanceModel.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace streamhls {

/// Runs a solver-based MINLP to optimize both permutation and unrolling factors simultaneously.
SolveResult runLoopCombinedMINLP(DFG &dfg, std::string projectPath, const DSEOptions &options);

/// Runs the combined DSE using Simulated Annealing.
SolveResult runLoopCombinedSA(DFG &dfg, const DSEOptions &options);

/// Runs the combined DSE using Genetic Algorithm.
SolveResult runLoopCombinedGA(DFG &dfg, const DSEOptions &options);

/// Finds a random valid combined solution for the DFG.
SolveResult findRandomCombinedSolution(DFG &dfg, const DSEOptions &options);

} // namespace streamhls
} // namespace mlir

#endif // MLIR_STREAMHLS_DIALECT_DATAFLOW_SUPPORT_LOOPCOMBINEDMODEL_H
