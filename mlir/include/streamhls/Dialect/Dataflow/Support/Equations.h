#ifndef MLIR_STREAMHLS_DIALECT_DATAFLOW_SUPPORT_EQUATIONS_H
#define MLIR_STREAMHLS_DIALECT_DATAFLOW_SUPPORT_EQUATIONS_H

#include "streamhls/Dialect/Dataflow/Support/DFG.h"
#include "streamhls/Dialect/Dataflow/Support/PerformanceModel.h"
#include <optional>
#include <string>
#include <map>

namespace mlir {
namespace streamhls {

void createLoopPermutationPerformanceModel(PerformanceModel &model, DFG &dfg, const DSEOptions &options, const std::map<std::string, int64_t>& tilingSolution = {});
void createLoopUnrollingPerformanceModel(PerformanceModel &model, DFG &dfg, const DSEOptions &options, const std::map<unsigned, unsigned>& permutationSolution = {});
void createLatencyPerformanceModel(PerformanceModel &model, DFG &dfg, 
                                   const std::map<unsigned, unsigned>& permSolution,
                                   const std::map<std::string, int64_t>& tilingSolution);
void createCombinedPerformanceModel(PerformanceModel &model, DFG &dfg, const DSEOptions &options);

} // namespace streamhls
} // namespace mlir

#endif // MLIR_STREAMHLS_DIALECT_DATAFLOW_SUPPORT_EQUATIONS_H
