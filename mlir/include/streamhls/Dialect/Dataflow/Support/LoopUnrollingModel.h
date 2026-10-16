#ifndef MLIR_STREAMHLS_DIALECT_DATAFLOW_SUPPORT_LOOPUNROLLINGMODEL_H
#define MLIR_STREAMHLS_DIALECT_DATAFLOW_SUPPORT_LOOPUNROLLINGMODEL_H

#include "streamhls/Dialect/Dataflow/Support/DFG.h"
#include "streamhls/Dialect/Dataflow/Support/PerformanceModel.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace streamhls {

/// Sets the unrolling factors of all loops in the DFG to fixed values from DSEOptions.
bool setFixedUnrollingFactors(DFG &dfg, const DSEOptions &options);

/// Finds random valid unrolling factors for the DFG.
SolveResult findRandomUnrollingSolution(DFG &dfg, const DSEOptions &options);

/// Runs a solver-based MINLP to find optimal unrolling factors.
SolveResult runLoopUnrollingMINLP(DFG &dfg, std::string projectPath, const DSEOptions &options);

/// Sets the unrolling factors of all loops in the DFG to 1.
bool setUnrollingFactorsToOnes(DFG &dfg);

/// Runs the loop unrolling DSE using Simulated Annealing.
SolveResult runLoopUnrollingSA(DFG &dfg, const DSEOptions &options);

/// Runs the loop unrolling DSE using Genetic Algorithm.
SolveResult runLoopUnrollingGA(DFG &dfg, const DSEOptions &options);

} // namespace streamhls
} // namespace mlir

#endif // MLIR_STREAMHLS_DIALECT_DATAFLOW_SUPPORT_LOOPUNROLLINGMODEL_H
