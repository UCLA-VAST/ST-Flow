#include "streamhls/Dialect/Dataflow/Transforms/MergeKernels/Utils.h"
#include "streamhls/Dialect/Dataflow/Transforms/MergeKernels/Candidates.h"

using namespace mlir;
using namespace streamhls;

#define DEBUG_TYPE "streamhls-merge-kernels-candidates"

// Implementations moved to AffineAnalysis.cpp