#ifndef STREAMHLS_TRANSFORMS_PASSES_H
#define STREAMHLS_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
// #include "streamhls/Dialect/Dataflow/IR/Dataflow.h"
#include "streamhls/InitAllDialects.h"

#include <map>
#include <memory>
#include <string>
#include "llvm/ADT/MapVector.h"
#include "streamhls/Dialect/Dataflow/Support/DFG.h"

namespace mlir {
namespace streamhls {
class DFG;
struct HWMapping;
struct DSEOptions;
  
enum CreateSubviewMode { Point, Reduction };

using VarType = std::variant<bool, int64_t, float, std::string, std::vector<int64_t>>;
using MergingData = std::tuple<SmallVector<ModuleOp>, SmallVector<int64_t>, std::string>;
using MergingGroups = std::map<std::string, MergingData>;
using OrderedModuleMap = llvm::MapVector<std::string, SmallVector<ModuleOp>, std::map<std::string, unsigned>>;


void registerTransformsPasses();

std::unique_ptr<Pass> createRemoveRedundantMemRefsPass();
std::unique_ptr<Pass> createRemoveRedundantOpsPass();
std::unique_ptr<Pass> createRemoveLoopsOfUnitIterPass();
std::unique_ptr<Pass> createLowerSubviewToAffinePass();
std::unique_ptr<Pass> createLowerTensorConcatToSlicesPass();
std::unique_ptr<Pass> createBufferizeFuncArgsPass();
std::unique_ptr<Pass> createStencilDataReusePass();
std::unique_ptr<Pass> createEliminateSingleWriteMultipleReadsPass(std::string eliminationType = "skip");
HWMapping HardwareMappingDSE(
    ModuleOp &module,
    std::string projectPath = ".",
    const DSEOptions &options = {},
    std::string hwMappingStrategy = "no_reuse"
);
std::unique_ptr<Pass> createCollapseUnitDimsPass();
std::unique_ptr<Pass> createArrayStreamsToVectorStreamPass();

// ScaleHLS
std::unique_ptr<Pass> createLowerCopyToAffinePass(bool internalCopyOnly = true);


ModuleOp mergeKernels(SmallVectorImpl<ModuleOp> &modules, std::unordered_map<std::string, VarType> &options);
ModuleOp reuseHardware(ModuleOp &module);
HWMapping runHighLevelDSE(ModuleOp &module, std::string projectPath, std::string mlgPath, const DSEOptions &options, std::string hwMappingStrategy /* no_reuse, max_reuse*/);
void mergeKernelsDFG(DFG &dfg, std::unordered_map<std::string, VarType> &options);
DFG extractMergedDFG(ModuleOp &module, HWMapping hwMapping, bool enableTiling = true);
std::string printDFGNodes(DFG &graph);
std::string printDFGEdges(DFG &graph);
std::string printDFGAsDot(DFG &graph);
void reuseHardwareDFG(DFG &dfg);
ModuleOp generateSystemModule(DFG &graph, ModuleOp templateModule);
ModuleOp annotateBuffers(ModuleOp &module, std::string projectPath);
ModuleOp annotateLoops(ModuleOp &module);
ModuleOp binarizeWeights(ModuleOp &module, llvm::StringRef weightBinsPath);
ModuleOp annotateMemrefs(ModuleOp &module);
ModuleOp lowerLinalgToAffine(ModuleOp &module, unsigned debugPoint = 0);
ModuleOp bufferizeFuncArgs(ModuleOp &module);
ModuleOp convertMemrefArgsToFIFOs(ModuleOp &module, const DenseSet<unsigned> &argsToConvert);
void convertFuncMemrefsToFIFOsDFG(DFG &dfg);
ModuleOp setPortWidthsToOne(ModuleOp &module);
ModuleOp eleminateUnitDimsAndIters(ModuleOp &module);
ModuleOp flattenArrayOfStreams(ModuleOp &module);
ModuleOp arrayStreamsToVectorStream(ModuleOp &module);
ModuleOp eliminateSingleWriteMultipleReads(ModuleOp &module, std::string eliminationType = "skip");
std::pair<ModuleOp, ModuleOp> rewriteMemoryModulesToDMA(ModuleOp &module);
ModuleOp raiseInnerLoopByOneLevel(ModuleOp &module);

#define GEN_PASS_CLASSES
#include "streamhls/Dialect/Dataflow/Transforms/Passes.h.inc"

// #define GEN_PASS_REGISTRATION
// #include "streamhls/Dialect/Dataflow/Transforms/Passes.h.inc"

} // namespace streamhls
} // namespace mlir

#endif // STREAMHLS_TRANSFORMS_PASSES_H