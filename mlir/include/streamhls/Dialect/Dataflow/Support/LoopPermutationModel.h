#ifndef MLIR_STREAMHLS_DIALECT_DATAFLOW_SUPPORT_LOOPPERMUTATIONMODEL_H
#define MLIR_STREAMHLS_DIALECT_DATAFLOW_SUPPORT_LOOPPERMUTATIONMODEL_H

#include "streamhls/Dialect/Dataflow/Support/DFG.h"
#include "streamhls/Dialect/Dataflow/Support/PerformanceModel.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace streamhls {

/// Runs the loop permutation MINLP.
SolveResult runLoopPermutationMINLP(DFG &dfg, llvm::StringRef projectPath, const DSEOptions &options);

/// Sets the permutation of all loops in the DFG to fixed values from DSEOptions.
bool setFixedPermutations(DFG &dfg, const DSEOptions &options);

/// Finds random valid permutations for the DFG.
SolveResult findRandomPermutationSolution(DFG &dfg, const DSEOptions &options);

/// Runs the loop permutation DSE using Simulated Annealing.
SolveResult runLoopPermutationSA(DFG &dfg, const DSEOptions &options);

/// Runs the loop permutation DSE using Genetic Algorithm.
SolveResult runLoopPermutationGA(DFG &dfg, const DSEOptions &options);

} // namespace streamhls
} // namespace mlir

#endif // MLIR_STREAMHLS_DIALECT_DATAFLOW_SUPPORT_LOOPPERMUTATIONMODEL_H
