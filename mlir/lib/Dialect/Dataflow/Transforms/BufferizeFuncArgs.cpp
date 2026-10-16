#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/IR/IntegerSet.h"


#include "streamhls/Dialect/Dataflow/Transforms/Passes.h"
#include "streamhls/Dialect/Dataflow/Support/Utils.h"
#include "streamhls/Dialect/Dataflow/Support/AffineMemAccess.h"
#include "streamhls/Dialect/Dataflow/Support/DFG.h"


using namespace mlir;
using namespace streamhls;
using namespace dataflow;

#define DEBUG_TYPE "streamhls-bufferize-func-args"


namespace {

static affine::AffineForOp parentForOfIV(Value v) {
  if (auto barg = dyn_cast<BlockArgument>(v))
    return dyn_cast_or_null<affine::AffineForOp>(barg.getOwner()->getParentOp());
  return nullptr;
}

static unsigned loopDepth(affine::AffineForOp forOp) {
  unsigned d = 0;
  Operation *cur = forOp.getOperation();
  while (auto p = cur->getParentOfType<affine::AffineForOp>()) {
    ++d;
    cur = p.getOperation();
  }
  return d;
}

// Try to interpret load indices as a permutation of surrounding IVs.
// Returns:
// - canonicalIVs: the IVs sorted by loop nesting (outer->inner)
// - perm: for each canonical dim i, perm[i] = position in loadIndices where canonicalIVs[i] appears
// - invPerm: inverse mapping so you can reorder loadIndices back to canonical order
static LogicalResult getPermutationFromLoad(affine::AffineLoadOp load,
                                           SmallVectorImpl<Value> &canonicalIVs,
                                           SmallVectorImpl<unsigned> &perm,
                                           SmallVectorImpl<unsigned> &invPerm) {
  canonicalIVs.clear();
  perm.clear();
  invPerm.clear();

  auto idxs = load.getIndices();
  if (idxs.empty()) return failure();

  // Must be block arguments (IVs) and all distinct.
  llvm::SmallDenseSet<Value, 8> seen;
  SmallVector<std::pair<unsigned /*depth*/, Value>, 8> ivsWithDepth;
  ivsWithDepth.reserve(idxs.size());

  for (Value v : idxs) {
    if (!isa<BlockArgument>(v)) {
      LLVM_DEBUG({
        llvm::dbgs() << "[BufArgs] not a permutation: index not BlockArgument: ";
        v.print(llvm::dbgs()); llvm::dbgs() << "\n";
      });
      return failure();
    }
    if (!seen.insert(v).second) {
      LLVM_DEBUG(llvm::dbgs() << "[BufArgs] not a permutation: repeated IV\n");
      return failure();
    }
    affine::AffineForOp pfor = parentForOfIV(v);
    if (!pfor) {
      LLVM_DEBUG(llvm::dbgs() << "[BufArgs] not a permutation: IV not from affine.for\n");
      return failure();
    }
    ivsWithDepth.push_back({loopDepth(pfor), v});
  }

  // Sort by depth descending? We want outer->inner. Outer has *larger* depth? Actually outer has higher
  // "distance to top"? In loopDepth above, inner has larger depth. So outer->inner is increasing depth.
  llvm::sort(ivsWithDepth, [](auto &a, auto &b) { return a.first < b.first; });

  for (auto &p : ivsWithDepth) canonicalIVs.push_back(p.second);

  // Build perm: canonicalIVs[i] appears at position j in load idxs.
  perm.resize(canonicalIVs.size(), ~0u);
  invPerm.resize(canonicalIVs.size(), ~0u);

  for (unsigned i = 0; i < canonicalIVs.size(); ++i) {
    Value civ = canonicalIVs[i];
    bool found = false;
    for (unsigned j = 0; j < idxs.size(); ++j) {
      if (idxs[j] == civ) {
        perm[i] = j;
        found = true;
        break;
      }
    }
    if (!found) return failure();
  }

  // invPerm[j] = i s.t. perm[i] = j
  for (unsigned i = 0; i < perm.size(); ++i) {
    if (perm[i] >= perm.size()) return failure();
    invPerm[perm[i]] = i;
  }
  for (unsigned j = 0; j < invPerm.size(); ++j)
    if (invPerm[j] == ~0u) return failure();

  return success();
}

static bool samePermutationForLoad(affine::AffineLoadOp load,
                                   ArrayRef<unsigned> refPerm) {
  SmallVector<Value, 8> canonicalIVs;
  SmallVector<unsigned, 8> perm, invPerm;
  if (failed(getPermutationFromLoad(load, canonicalIVs, perm, invPerm)))
    return false;
  return perm == refPerm;
}

// New bufferization: build copy loops following consumer permutation and rewrite matching loads.
static LogicalResult bufferizeArgUsingConsumerPermutation(TypedValue<MemRefType> memref,
                                                         Block *block,
                                                         OpBuilder &builder) {
  Location loc = memref.getLoc();
  MemRefType ty = memref.getType();
  if (!ty.hasStaticShape()) {
    LLVM_DEBUG(llvm::dbgs() << "[BufArgs] skip: dynamic shape arg\n");
    return failure();
  }

  // Pick a representative consumer load.
  affine::AffineLoadOp repLoad = nullptr;
  for (Operation *u : memref.getUsers()) {
    if (auto l = dyn_cast<affine::AffineLoadOp>(u)) { repLoad = l; break; }
  }
  if (!repLoad) {
    LLVM_DEBUG(llvm::dbgs() << "[BufArgs] skip: arg has no affine.load users\n");
    return failure();
  }

  SmallVector<Value, 8> canonicalIVs;
  SmallVector<unsigned, 8> perm, invPerm;
  if (failed(getPermutationFromLoad(repLoad, canonicalIVs, perm, invPerm))) {
    LLVM_DEBUG({
      llvm::dbgs() << "[BufArgs] skip: representative load not a pure permutation: ";
      repLoad->print(llvm::dbgs()); llvm::dbgs() << "\n";
    });
    return failure();
  }

  if ((int64_t)perm.size() != ty.getRank()) {
    LLVM_DEBUG(llvm::dbgs() << "[BufArgs] skip: perm rank != memref rank\n");
    return failure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "[BufArgs] detected consumer permutation for arg: ";
    memref.print(llvm::dbgs());
    llvm::dbgs() << "\n  repLoad: ";
    repLoad->print(llvm::dbgs());
    llvm::dbgs() << "\n  perm = [";
    for (unsigned i = 0; i < perm.size(); ++i) {
      llvm::dbgs() << perm[i] << (i + 1 == perm.size() ? "" : ",");
    }
    llvm::dbgs() << "]\n";
  });

  // Create new buffer at block start.
  builder.setInsertionPoint(&block->front());
  auto newBuf = builder.create<memref::AllocOp>(loc, ty);

  // Build loop nest in canonical (dim0..dimN-1) order.
  SmallVector<int64_t, 8> lbs, ubs, steps;
  lbs.assign(ty.getRank(), 0);
  steps.assign(ty.getRank(), 1);
  for (int i = 0; i < ty.getRank(); ++i) ubs.push_back(ty.getDimSize(i));

  affine::buildAffineLoopNest(builder, loc, lbs, ubs, steps,
    [&](OpBuilder &b, Location loc, ValueRange ivs) {
      // srcIdxs = ivs permuted as consumer expects: srcIdxs[d] = ivs[perm[d]]? (perm maps canonical dim -> load position)
      // Here, ivs is canonical order. Consumer load uses indices in "load order" (positions).
      // srcIdxs in load order is: srcIdxs[j] = ivs[invPerm[j]]
      SmallVector<Value, 8> srcIdxs(ivs.size());
      for (unsigned j = 0; j < ivs.size(); ++j)
        srcIdxs[j] = ivs[invPerm[j]];

      auto v = b.create<affine::AffineLoadOp>(loc, memref, srcIdxs);
      b.create<affine::AffineStoreOp>(loc, v, newBuf, ivs);
    });

  // Rewrite matching loads: %memref[loadIdxs] -> %newBuf[canonicalIdxs]
  // canonicalIdxs = loadIdxs reordered back to canonical order: canonical[i] = loadIdxs[perm[i]]
  for (Operation *u : llvm::make_early_inc_range(memref.getUsers())) {
    auto load = dyn_cast<affine::AffineLoadOp>(u);
    if (!load) continue;
    if (!samePermutationForLoad(load, perm)) continue;

    SmallVector<Value, 8> loadIdxs(load.getIndices().begin(), load.getIndices().end());
    SmallVector<Value, 8> canonIdxs(loadIdxs.size());
    for (unsigned i = 0; i < loadIdxs.size(); ++i)
      canonIdxs[i] = loadIdxs[perm[i]];

    builder.setInsertionPoint(load);
    auto repl = builder.create<affine::AffineLoadOp>(load.getLoc(), newBuf, canonIdxs);
    load.replaceAllUsesWith(repl.getResult());
    load.erase();

    LLVM_DEBUG({
      llvm::dbgs() << "[BufArgs] rewrote one load to use bufferized arg\n";
    });
  }

  // Replace remaining uses of original memref (if any) except inside the copy loop we just created.
  // In practice, for "INPUT" args you likely want to route everything to newBuf now.
  memref.replaceAllUsesExcept(newBuf.getResult(), newBuf.getOperation());

  LLVM_DEBUG(llvm::dbgs() << "[BufArgs] bufferized arg with consumer-permutation copy\n");
  return success();
}

} // namespace

namespace {
  enum ArgType {
    INPUT, OUTPUT
  };
  static LogicalResult bufferizeArg(TypedValue<mlir::MemRefType> memref, Block* block, OpBuilder &builder){
    auto loc = builder.getUnknownLoc();
    // start of the block
    builder.setInsertionPoint(&block->front());
    auto newBuffer = builder.create<memref::AllocOp>(loc, memref.getType());
    SmallVector<int64_t> lbs;
    SmallVector<int64_t> ubs;
    SmallVector<int64_t> steps;
    for(int i = 0; i < memref.getType().getRank(); i++){
      lbs.push_back(0);
      ubs.push_back(memref.getType().getShape()[i]);
      steps.push_back(1);
    }
    affine::buildAffineLoopNest(builder, loc, lbs, ubs, steps,
      [&](OpBuilder &b, Location loc, ValueRange ivs) {
        // auto dtype = memref.getType().getElementType();
        // new load op
        auto loadOp = b.create<AffineLoadOp>(loc, memref, ivs);
        // new store op
        b.create<AffineStoreOp>(loc, loadOp, newBuffer, ivs);
        memref.replaceAllUsesExcept(newBuffer, loadOp);
    });
    // replace all uses of the original memref with the new buffer
    return success();
  }
}
namespace {
struct BufferizeFuncArgs : public BufferizeFuncArgsBase<BufferizeFuncArgs> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    auto context = func.getContext();
    OpBuilder builder(context);
    auto block = &func.front();
    SmallVector<std::pair<TypedValue<mlir::MemRefType>, ArgType>> memRefs;
    for(auto arg : func.getArguments()){
      // dynamic cast to TypedValue<MemRefType>
      if(auto memref = dyn_cast<TypedValue<mlir::MemRefType>>(arg)){
        bool isInput = false;
        bool isOutput = false;
        for(auto user : memref.getUsers()){
          if(isa<AffineLoadOp>(user)){
            isInput = true;
          }
          if(isa<AffineStoreOp>(user)){
            isOutput = true;
          }
        }
        if(isInput && isOutput){
          assert(false && "Top Func Argument is both INPUT and OUTPUT");
        } else if(isInput){
          memRefs.push_back({memref, ArgType::INPUT});
        } else if(isOutput){
          memRefs.push_back({memref, ArgType::OUTPUT});
        }
      } else {
        // assert(false && "Top Func Argument is not a MemRefType");
      }
    }

    for(auto pair : memRefs){
      auto memref = pair.first;
      auto argType = pair.second;
      if(argType == INPUT){
        if(succeeded(bufferizeArg(memref, block, builder))){
          LLVM_DEBUG(llvm::dbgs() << "Bufferized INPUT argument\n");
        }else{
          assert(false && "Failed to bufferize INPUT argument");
        }
      }
    }
    // (void)removeUnusedFuncArgs(func);
  }
};
} // namespace

std::unique_ptr<Pass> streamhls::createBufferizeFuncArgsPass() {
  return std::make_unique<BufferizeFuncArgs>();
}

ModuleOp streamhls::bufferizeFuncArgs(ModuleOp &module) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::dataflow::DataflowDialect, mlir::func::FuncDialect,
                  mlir::arith::ArithDialect, mlir::tensor::TensorDialect,
                  mlir::affine::AffineDialect, mlir::math::MathDialect,
                  mlir::memref::MemRefDialect>();
  PassManager pm = PassManager::on<ModuleOp>(module.getContext());

  pm.getDependentDialects(registry); // check if this is necessary
  pm.nest<func::FuncOp>().addPass(streamhls::createBufferizeFuncArgsPass());
  pm.addPass(mlir::createCanonicalizerPass());

  if (failed(pm.run(module)))
    assert(false && "Stream-HLS Func Arg Bufferization failed");

  return module;
}