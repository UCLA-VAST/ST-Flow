#include "streamhls/Dialect/Dataflow/Transforms/MergeKernels/DataTypes.h"
#include "streamhls/Dialect/Dataflow/Transforms/MergeKernels/Utils.h"
#include "streamhls/Dialect/Dataflow/Transforms/MergeKernels/Candidates.h"
#include "streamhls/Dialect/Dataflow/Transforms/MergeKernels/Equations.h"
#include "streamhls/Dialect/Dataflow/Transforms/MergeKernels/Merging.h"
#include "streamhls/Dialect/Dataflow/Transforms/MergeKernels/Solution.h"
#include "streamhls/Dialect/Dataflow/Transforms/MergeKernels/Transform.h"
#include "streamhls/Dialect/Dataflow/Transforms/Passes.h"

#include <math.h>
#include <regex>
#include <sstream>
#include <string>

#define DEBUG_TYPE "streamhls-merge-kernels"

using namespace mlir;
using namespace streamhls;

namespace {
  
void createPerformanceModel(
  SmallVectorImpl<AffineForOp> &forOps, 
  std::unordered_map<std::string, VarType> &options,
  AllCandidates& loopCandidates,
  LoopBandInfo &maxBandInfo
) {
  // collect loop bands info
  AllLoopBandsInfo allLoopBandsInfo;
  collectLoopBandsInfo(forOps, allLoopBandsInfo);


  LoopBandInfo minDummy = allLoopBandsInfo[0];
  maxBandInfo = allLoopBandsInfo[0];
  getMinAndMaxLoopStatsMerge(allLoopBandsInfo, minDummy, maxBandInfo);

  // Tile the loops
  tileLoopsTwoLevels(allLoopBandsInfo);

  // add dummy loop info
  addDummyLoopInfo(allLoopBandsInfo);
  unsigned count = 0;
  for(auto loopBandInfoAndForOp : llvm::zip(allLoopBandsInfo, forOps)){
    auto loopBandInfo = std::get<0>(loopBandInfoAndForOp);
    auto forOp = std::get<1>(loopBandInfoAndForOp);
    SmallVector<MemrefInfo, 4> memrefs;
    getLoopMemRefs(forOp, memrefs);
    // if (count == 0){
    //   for(auto memref : memrefs){
    //     // struct MemrefInfo{
    //     //   std::string name;
    //     //   Value memref;
    //     //   bool isStore, isLoad;
    //     //   SmallVector<Value, 4> memrefIVs;
    //     //   SmallVector<int64_t, 4> shape;
    //     // };
    //     llvm::dbgs() << "Memref: " << memref.name << "\n";
    //     llvm::dbgs() << "  Shape: ";
    //     for(auto dim : memref.shape){
    //       llvm::dbgs() << dim << " ";
    //     }
    //     llvm::dbgs() << "\n";
    //     llvm::dbgs() << "  Induction Vars: ";
    //     for(auto iv : memref.memrefIVs){
    //       llvm::dbgs() << iv << " ";
    //     }

    //     llvm::dbgs() << "\n";
    //     llvm::dbgs() << "  isLoad: " << memref.isLoad << ", isStore: " << memref.isStore << "\n";
    //   }
    // }
    auto currLoopCandidates = getAllCandidates(loopBandInfo, memrefs, std::get<bool>(options["full_data_reuse"]));
    loopCandidates.push_back(currLoopCandidates);
    LLVM_DEBUG(
      for(auto memrefCandidatePair : llvm::enumerate(currLoopCandidates)){
        auto memrefIdx = memrefCandidatePair.index();
        auto memrefName = memrefCandidatePair.value().first;
        auto singleMemrefCandidates = memrefCandidatePair.value().second;
        llvm::dbgs() << "Memref " << memrefName << " has " << singleMemrefCandidates.size() << " candidates.\n";
      }
      llvm::dbgs() << "\n";
    );
    // if (count == 0){
    //     if(memrefName == "memref_2"){
    //       // candidateInfo.isLoad = memref.isLoad;
    //       // candidateInfo.isStore = memref.isStore;
    //       // candidateInfo.bufferSizes = bufferSizes;
    //       // candidateInfo.accessIndices = accessIndices;
    //       // candidateInfo.reuseFactor = reuseFactor;
    //       // candidateInfo.guardLHS = guardLHSs;
    //       // candidateInfo.guardRHS = guardRHSs;
    //       for(auto candidatePair : llvm::enumerate(singleMemrefCandidates)){
    //         auto candidateIdx = candidatePair.index();
    //         auto candidateInfo = candidatePair.value();
    //         llvm::dbgs() << " ===  Candidate " << candidateIdx << ":\n";
    //         llvm::dbgs() << "isLoad: " << candidateInfo.isLoad << ", isStore: " << candidateInfo.isStore << "\n";
    //         llvm::dbgs() << "Buffer Sizes:\n";
    //         for(auto bufferSize : candidateInfo.bufferSizes){
    //           llvm::dbgs() << "\texpr: " << bufferSize.expr << "\n";
    //           llvm::dbgs() << "\texprStr: " << bufferSize.exprStr << "\n";
    //         }
    //         llvm::dbgs() << "\n";
    //         llvm::dbgs() << "Access Indices:\n";
    //         for(auto accessIndex : candidateInfo.accessIndices){
    //           llvm::dbgs() << "\texpr: " << accessIndex.expr << "\n";
    //           llvm::dbgs() << "\texprStr: " << accessIndex.exprStr << "\n";
    //         }
    //         llvm::dbgs() << "\n";
    //         llvm::dbgs() << "Reuse Factor:\n";
    //         llvm::dbgs() << "\texpr: " << candidateInfo.reuseFactor.expr << "\n";
    //         llvm::dbgs() << "\texprStr: " << candidateInfo.reuseFactor.exprStr << "\n";
    //         llvm::dbgs() << "\n";
    //         llvm::dbgs() << "Guard LHSs:\n";
    //         for(auto guardLHS : candidateInfo.guardLHS){
    //           llvm::dbgs() << "\texpr: " << guardLHS.expr << "\n";
    //           llvm::dbgs() << "\texprStr: " << guardLHS.exprStr << "\n";
    //         }
    //         llvm::dbgs() << "\n";
    //         llvm::dbgs() << "Guard RHSs:\n";
    //         for(auto guardRHS : candidateInfo.guardRHS){
    //           llvm::dbgs() << "\texpr: " << guardRHS.expr << "\n";
    //           llvm::dbgs() << "\texprStr: " << guardRHS.exprStr << "\n";
    //         }
    //         llvm::dbgs() << "\n";
    //       }
    //     }
    //   }
    //   llvm::dbgs() << "\n";
    // }
    count++;
  }
  // createPerformanceModelFromCandidates(loopCandidates, allLoopBandsInfo, options);
}

void remapMemrefSolutions(MemrefSolutions &solutions,
                                 const IRMapping &map) {
  auto remapVals = [&](SmallVector<Value> &vals) {
    for (Value &v : vals)
      v = map.lookupOrDefault(v);
  };

  for (auto &S : solutions) {
    // Memref itself:
    if (S.memref)
      S.memref = map.lookupOrDefault(S.memref);

    // Any operands that may contain original IVs / args / memrefs:
    remapVals(S.mapOperands);
    remapVals(S.fifoMapOperands);
    remapVals(S.loadSetOperands);
    remapVals(S.storeSetOperands);

    // If you later add other Value fields, remap them here too.
  }
}

ModuleOp mergeForLoopKernels(
  SmallVectorImpl<ModuleOp> &modules,
  const AllCandidates &loopCandidates,
  const LoopBandInfo &maxLoopInfo,
  const std::string &solutionStr,
  OpBuilder &builder,
  StringRef mergedFuncName,
  std::vector<int64_t> tilingFactors
) {
  // auto module = builder.create<ModuleOp>(loc);
  // auto newForOp = mergeForLoops(
  //   module,
  //   builder,
  //   forOps,
  //   funcOps,
  //   mergedFuncName
  // );

  // module.dump();
  // newForOp.dump();
  
  auto [module, newForOp, map] = mergeKernels(modules, builder, mergedFuncName.str());
  // module.dump();
  // newForOp.dump();
  
  TilingSolution tilingSolution;
  MemrefSolutions memrefSolutions;
  // // tile the loops
  // // getTilingFactors(solutionStr, tilingSolution);
  // if(tilingFactors.size() == 0){
  //   getTilingFactors(solutionStr, tilingSolution);
  //   // const unsigned fixedTm = 1;
  //   // const unsigned fixedTu = 1;
  //   // setFixedTilingFactors(solutionStr, fixedTm, fixedTu, tilingSolution);
  // }else{
  for(auto tf : tilingFactors){
    tilingSolution.l1TilingFactors.push_back(1); // default value
    tilingSolution.l2TilingFactors.push_back(tf); 
  }
  // }
  newForOp = apply2LevelTiling(newForOp, builder, tilingSolution);

  simplifyAffineOps(module);
  
  AffineLoopBand allLoops;
  getLoopBandFromOutermost(newForOp, allLoops);
  
  getMemRefInfo(
    builder,
    solutionStr, 
    allLoops, 
    loopCandidates,
    maxLoopInfo,
    memrefSolutions
  );

  printMemRefSolutions(memrefSolutions);
  remapMemrefSolutions(memrefSolutions, map);

  // for(auto memrefSolutionPair : llvm::enumerate(memrefSolutions)){
  //   auto memrefIdx = memrefSolutionPair.index();
  //   auto &memrefSolution = memrefSolutionPair.value();
  //   llvm::dbgs() << "Memref Solution " << memrefIdx << ":\n";
  //   llvm::dbgs() << "isLoad: " << memrefSolution.isLoad << ", isStore: " << memrefSolution.isStore << "\n\n";
  // }

  applyBufferingStrategy(
    module,
    newForOp,
    builder,
    memrefSolutions
  );

  updateLoopBounds(
    newForOp,
    maxLoopInfo
  );

  return module;
}

inline void stripTrailingCR(std::string &s) {
    if (!s.empty() && s.back() == '\r') s.pop_back();
}

std::string fixRlLevel(const std::string &solution,
                       const std::string &memrefName,
                       unsigned rl) {
    // Matches lines like: "memref_2_rl_1 = 0"
    // ^\s*memref_2_rl_(\d+)\s*=\s*\d+\s*$
    const std::regex re("^\\s*(" + memrefName + ")_rl_(\\d+)\\s*=\\s*\\d+\\s*$");

    std::istringstream in(solution);
    std::ostringstream out;
    std::string line;

    while (std::getline(in, line)) {
        stripTrailingCR(line); // handle Windows CRLF input
        std::smatch m;
        if (std::regex_match(line, m, re)) {
            unsigned idx = static_cast<unsigned>(std::stoul(m[2].str()));
            out << memrefName << "_rl_" << idx << " = " << (idx == rl ? 1 : 0) << "\n";
        } else {
            out << line << "\n";
        }
    }
    return out.str();
}

/// Compute the next multiple of `factor` that is >= `bound`.
static int64_t nextMultiple(int64_t bound, int64_t factor) {
  assert(factor > 0 && "tiling factor must be positive");
  return ((bound + factor - 1) / factor) * factor;
}

/// Pad loop bounds and memref argument shapes so that every loop dimension
/// is evenly divisible by the corresponding tiling factor.
/// This allows non-divisor tiling factors (e.g. factor 3 on dim 128 → 129).
/// Uses IV-based access pattern analysis to only pad the exact memref
/// dimensions that correspond to padded loop induction variables.
static void padModuleForTiling(ModuleOp mod,
                               ArrayRef<int64_t> tilingFactors) {
  func::FuncOp func = *mod.getOps<func::FuncOp>().begin();

  // 1. Collect the outermost loop band.
  SmallVector<affine::AffineForOp, 4> band;
  affine::AffineForOp outer;
  func.walk([&](affine::AffineForOp fo) {
    if (!fo->getParentOfType<affine::AffineForOp>())
      if (!outer) outer = fo;
  });
  if (!outer) return;
  auto cur = outer;
  while (cur) {
    band.push_back(cur);
    affine::AffineForOp next;
    cur.getBody()->walk([&](affine::AffineForOp inner) {
      if (inner->getParentOp() == cur) next = inner;
    });
    cur = next;
  }
  if (band.size() != tilingFactors.size()) return;

  // 2. Pad loop upper bounds and record which IVs were padded.
  DenseMap<Value, int64_t> ivToPaddedUB; // IV → new padded UB
  for (unsigned i = 0; i < band.size(); ++i) {
    if (!band[i].hasConstantUpperBound()) continue;
    int64_t origUB = band[i].getConstantUpperBound();
    int64_t padded = nextMultiple(origUB, tilingFactors[i]);
    if (padded != origUB) {
      ivToPaddedUB[band[i].getInductionVar()] = padded;
      band[i].setConstantUpperBound(padded);
      LLVM_DEBUG({
        llvm::dbgs() << "[pad] Loop dim " << i << ": " << origUB
                     << " → " << padded
                     << " (factor=" << tilingFactors[i] << ")\n";
      });
    }
  }
  if (ivToPaddedUB.empty()) return;

  // 3. Update memref argument types using IV-to-dimension access analysis.
  //    For each memref arg, find an affine.load or affine.store that uses it,
  //    then examine the access map to determine which loop IVs index which
  //    memref dimensions. Only pad dimensions that map to padded IVs.
  for (auto arg : func.getArguments()) {
    auto memTy = dyn_cast<MemRefType>(arg.getType());
    if (!memTy || !memTy.hasStaticShape()) continue;

    SmallVector<int64_t> newShape(memTy.getShape());
    bool changed = false;

    // Find a load or store that accesses this memref arg.
    func.walk([&](Operation *op) {
      SmallVector<Value> mapOperands;
      AffineMap accessMap;
      Value memref;

      if (auto load = dyn_cast<affine::AffineLoadOp>(op)) {
        memref = load.getMemRef();
        mapOperands.append(load.getMapOperands().begin(),
                           load.getMapOperands().end());
        accessMap = load.getAffineMap();
      } else if (auto store = dyn_cast<affine::AffineStoreOp>(op)) {
        memref = store.getMemRef();
        mapOperands.append(store.getMapOperands().begin(),
                           store.getMapOperands().end());
        accessMap = store.getAffineMap();
      } else {
        return WalkResult::advance();
      }

      if (memref != arg) return WalkResult::advance();

      // For each result dimension of the access map, check if a padded IV
      // contributes to it. If so, set that memref dim to the padded UB.
      for (unsigned d = 0; d < accessMap.getNumResults() && d < newShape.size();
           ++d) {
        AffineExpr expr = accessMap.getResult(d);
        expr.walk([&](AffineExpr e) {
          if (auto dimExpr = dyn_cast<AffineDimExpr>(e)) {
            unsigned pos = dimExpr.getPosition();
            if (pos < mapOperands.size()) {
              auto it = ivToPaddedUB.find(mapOperands[pos]);
              if (it != ivToPaddedUB.end()) {
                newShape[d] = it->second;
                changed = true;
              }
            }
          }
        });
      }
      return WalkResult::interrupt(); // One access op is enough
    });

    if (changed) {
      auto newTy = MemRefType::get(newShape, memTy.getElementType(),
                                   memTy.getLayout(), memTy.getMemorySpace());
      arg.setType(newTy);
      LLVM_DEBUG({
        llvm::dbgs() << "[pad] Updated arg type to " << newTy << "\n";
      });
    }
  }
  // Update func type signature to match block arg types.
  SmallVector<Type> argTypes;
  for (auto arg : func.getArguments())
    argTypes.push_back(arg.getType());
  func.setType(FunctionType::get(func.getContext(), argTypes, {}));
}

/// Pad all modules in the list for the given tiling factors.
static void padModulesForTiling(SmallVectorImpl<ModuleOp> &modules,
                                ArrayRef<int64_t> tilingFactors) {
  for (ModuleOp mod : modules)
    padModuleForTiling(mod, tilingFactors);
}

} // end anonymous namespace

namespace mlir {
namespace streamhls {
ModuleOp mergeKernels(SmallVectorImpl<ModuleOp> &modules, std::unordered_map<std::string, VarType> &options) {
  // if (modules.size() < 2) {
  //   llvm::errs() << "Error: There are less than two modules to merge\n";
  //   return nullptr;
  // }
  LLVM_DEBUG(
    llvm::dbgs() << "Merging: " << std::get<std::string>(options["module_name"]) << " of "<< modules.size() << " modules.\n";
  );
  auto [funcOps, forOps] = gatherFuncsAndForOps(modules);

  AllCandidates loopCandidates;
  LoopBandInfo maxLoopBand;
  createPerformanceModel(
    forOps, 
    options,
    loopCandidates,
    maxLoopBand
  );

  // auto projectPath = std::get<std::string>(options["project_path"]);
  // std::string fileName = projectPath + "/kernel_merging.mod";
  auto solutionStr = ""; //solvePerformanceModel(fileName);

  // // initialize load/store buffers with zero
  // modules = initializeFullReuseBuffersWithZero(modules);

  // mergeForLoopKernels and return a merged module based on the solution
  MLIRContext *context = modules[0].getContext();
  OpBuilder builder(context);

  auto newModule = mergeForLoopKernels(
    modules,
    loopCandidates,
    maxLoopBand,
    solutionStr,
    builder,
    StringRef(std::get<std::string>(options["module_name"])),
    std::get<std::vector<int64_t>>(options["tiling_factors"])
  );

  // attach per function meta variant data
  attachVariantMetadata(newModule, funcOps);

  return newModule;
}



LogicalResult tileMemoryModule(ModuleOp &module, SmallVectorImpl<int64_t> &tiles) {
 
  TilingSolution tilingSol;
  for(size_t i = 0; i < tiles.size(); ++i){
    tilingSol.l1TilingFactors.push_back(1);
    tilingSol.l2TilingFactors.push_back(tiles[i]);
  }

  AffineLoopBand band;
  module->walk([&](AffineForOp forOp) { band.push_back(forOp); });
  std::reverse(band.begin(), band.end());
  assert(band.size() == tiles.size() && "ExtractModule: Tiling size mismatch");
  func::FuncOp func = *module.getOps<func::FuncOp>().begin();
  MLIRContext *ctx = module.getContext();
  OpBuilder b(ctx);
  AffineForOp topLoop = apply2LevelTiling(band[0], b, tilingSol);
  simplifyAffineOps(module);
  auto attachFifoInfo = [&](Operation *op, Value mem) {
    auto ba = mem.dyn_cast<BlockArgument>();
    if (!ba || ba.getOwner() != &func.front()) return;

    if (auto dict = func.getArgAttrOfType<DictionaryAttr>(ba.getArgNumber(), "buffer.info")) {
      if (auto tyAttr = dict.getAs<StringAttr>("type")) {
        StringRef ty = tyAttr.getValue();
        if (ty == "off-chip" || ty == "weights") return;
      }
    }

    auto innermostLoop = op->getParentOfType<AffineForOp>();
    if (!innermostLoop) return;

    AffineLoopBand postTileBand;
    getLoopBandFromInnermost(innermostLoop, postTileBand);
    if (postTileBand.size() < tiles.size()) return;

    SmallVector<Value, 4> indicesVals;
    indicesVals.reserve(tiles.size());
    for (AffineForOp loop : llvm::ArrayRef(postTileBand).take_back(tiles.size())) {
      indicesVals.push_back(loop.getInductionVar());
    }

    (void)setFifoAttr(ctx, op, tiles, indicesVals);
  };

  func.walk([&](Operation *op) {
    if (auto ld = dyn_cast<AffineLoadOp>(op))
      attachFifoInfo(op, ld.getMemRef());
    else if (auto st = dyn_cast<AffineStoreOp>(op))
      attachFifoInfo(op, st.getMemRef());
  });

  
  // --------------------------------------------------------------------------
  // NEW: Attach single-variant metadata for this tiled memory module
  // --------------------------------------------------------------------------
  {
    OpBuilder b(ctx);
    mlir::streamhls::AffineLoopBand tiledBand;
    getLoopBandFromOutermost(topLoop, tiledBand);
    DictionaryAttr variantDict =
        buildVariantMetadataForFunc(b, /*srcFunc=*/func,
                                       /*mergedFunc=*/func,
                                       /*band=*/tiledBand);

    module->setAttr("merge.variants", b.getArrayAttr(variantDict));
  }

  return success();
}

void mergeKernelsDFG(DFG &dfg, std::unordered_map<std::string, VarType> &options) {
  if(!dfg.applyNodePermutations()){
    llvm::errs() << "Error: Failed to apply node permutations\n";
    return;
  }
  for (auto &nodePair : dfg.nodes) {
    auto &node = nodePair.second;
    // Convert node tilingFactors from unsigned to int64_t
    std::vector<int64_t> tilingFactors;
    for (auto tv : node.candidates.tilingCandidates){
      tilingFactors.push_back(dfg.tilingVars[tv]);
    }
    if (node.kind == DFG::NodeKind::COMPUTE && !node.wrappedModules.empty()) {
      // Pad loop bounds & memref shapes so tiling factors divide evenly.
      padModulesForTiling(node.wrappedModules, tilingFactors);
      options["module_name"] = "merged_node_" + std::to_string(node.id);
      options["tiling_factors"] = tilingFactors;
      node.mergedModule = mergeKernels(node.wrappedModules, options);
      node.hwModule = node.mergedModule;
    }else{
      // Pad the memory module before tiling for consistent streaming.
      SmallVector<ModuleOp> hwModVec = {node.hwModule};
      padModulesForTiling(hwModVec, tilingFactors);
      SmallVector<int64_t> tiles(tilingFactors.begin(), tilingFactors.end());
      tileMemoryModule(node.hwModule, tiles);
    }
  }
  dfg.populateFifoInfo();
}

} // namespace streamhls
} // namespace mlir