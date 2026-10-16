#pragma once
#include "streamhls/Dialect/Dataflow/Transforms/MergeKernels/DataTypes.h"

namespace mlir {
namespace streamhls {
// ----------- Extract detailed memref info from AMPL result -----------
void getMemRefInfo(
  OpBuilder &builder,
  const std::string &result,
  AffineLoopBand &allLoops,
  const AllCandidates &loopCandidates,
  const LoopBandInfo &maxBandInfo,
  MemrefSolutions &memrefSolutions
);
/// ----------- Gather functions and their outermost loops from modules -----------
void getTilingFactors(
  const std::string &result,
  TilingSolution &tilingSolution
);

void setFixedTilingFactors(
  const std::string &result,
  const unsigned fixedTm,
  const unsigned fixedTu,
  TilingSolution &tilingSolution
);

void printMemRefSolutions(const MemrefSolutions &S);

} // end namespace streamhls
} // end namespace mlir