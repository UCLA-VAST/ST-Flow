#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/AffineExpr.h"   // for Affine{Expr,BinaryOpExpr,DimExpr,ConstantExpr}
#include "mlir/IR/AffineMap.h"    // for AffineMap/AffineMapAttr
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "streamhls/Dialect/Dataflow/Transforms/Passes.h"
#include "streamhls/Dialect/Dataflow/Support/AffineMemAccess.h"


using namespace mlir;
using namespace streamhls;
using namespace affine;
// namespace {

// static void collectFixedIVsFromIf(AffineIfOp ifOp,
//                                   llvm::DenseSet<mlir::Value> &ivsFixedToConst) {

//   IntegerSet set = ifOp.getIntegerSet();
//   unsigned numDims = set.getNumDims();
//   auto operands = ifOp.getOperands(); // dims first, then symbols

//   auto markIfIV = [&](unsigned dimPos) {
//     if (dimPos >= numDims || dimPos >= operands.size())
//       return;
//     Value v = operands[dimPos];
//     if (auto ba = v.dyn_cast<BlockArgument>())
//       if (auto forOp =
//               dyn_cast_or_null<AffineForOp>(ba.getOwner()->getParentOp()))
//         ivsFixedToConst.insert(v);
//   };

//   for (unsigned i = 0, e = set.getNumConstraints(); i < e; ++i) {
//     if (!set.isEq(i))           // only equalities: expr == 0
//       continue;
//     AffineExpr expr = set.getConstraint(i);

//     // Match: d_j == 0
//     if (auto d = expr.dyn_cast<AffineDimExpr>()) {
//       markIfIV(d.getPosition());
//       continue;
//     }

//     // Match: d_j - c == 0, d_j + c == 0, or c + d_j == 0
//     if (auto bin = expr.dyn_cast<AffineBinaryOpExpr>()) {
//       auto L = bin.getLHS();
//       auto R = bin.getRHS();
//       auto lhsDim = L.dyn_cast<AffineDimExpr>();
//       auto rhsDim = R.dyn_cast<AffineDimExpr>();
//       bool rhsIsConst = R.isa<AffineConstantExpr>();
//       bool lhsIsConst = L.isa<AffineConstantExpr>();

//       switch (bin.getKind()) {
//       case AffineExprKind::Add:
//         if (lhsDim && rhsIsConst)
//           markIfIV(lhsDim.getPosition());
//         else if (rhsDim && lhsIsConst)
//           markIfIV(rhsDim.getPosition());
//         break;
//       // case AffineExprKind::Sub:
//       //   if (lhsDim && rhsIsConst)
//       //     markIfIV(lhsDim.getPosition());
//       //   break;
//       default:
//         break; // ignore other forms
//       }
//     }
//   }
// }


// /// Try to get a constant trip count for an affine.for. Returns -1 if unknown.
// static int64_t getConstTripCount(AffineForOp forOp) {
//   int64_t tripCount = forOp.getConstantUpperBound();
//   return tripCount >= 0 ? tripCount : -1;
// }

// /// Returns product of enclosing loops' trip counts around `op`.
// /// If an enclosing affine.if has an equality constraint fixing an IV,
// /// treat that IV's loop contribution as 1 (for simple d == c cases).
// /// If any loop has non-constant trip count, returns -1 (unknown).
// static int64_t computeNumAccesses(Operation *op) {
//   // Gather ancestor affine.for ops from inner to outer.
//   SmallVector<AffineForOp, 4> loops;
//   for (Operation *p = op->getParentOp(); p; p = p->getParentOp()) {
//     if (auto forOp = dyn_cast<AffineForOp>(p))
//       loops.push_back(forOp);
//   }

//   if (loops.empty())
//     return 1; // No loops around: executes once.

//   // Detect IVs fixed by enclosing affine.if (only simple equality sets).
//   // We mark IVs that are fixed, so their contribution is 1.
//   DenseSet<Value> ivsFixedToConst;

//   for (Operation *p = op->getParentOp(); p; p = p->getParentOp()) {
//     if (auto ifOp = dyn_cast<AffineIfOp>(p)) {
//       collectFixedIVsFromIf(ifOp, ivsFixedToConst);
//     }
//   }

//   // Multiply contributions.
//   int64_t prod = 1;
//   for (AffineForOp forOp : loops) {
//     int64_t tc = getConstTripCount(forOp);
//     if (tc < 0)
//       return -1; // unknown trip count
//     // If this forOp's IV is fixed by an enclosing if, contribution = 1
//     Value iv = forOp.getInductionVar();
//     if (ivsFixedToConst.contains(iv))
//       continue;
//     // Otherwise multiply.
//     // Guard overflow minimally (not expected in normal sizes).
//     if (prod > 0 && tc > 0) {
//       // Cheap overflow guard for 64-bit.
//       if (prod > (std::numeric_limits<int64_t>::max() / tc))
//         return -1;
//     }
//     prod *= tc;
//   }
//   return prod;
// }

// /// Annotate a single (and only) load/store user of a func-arg memref.
// static void annotateArgForSingleAccess(Value memrefArg,
//                                        Operation *theOnlyAccessOp) {
//   OpBuilder b(memrefArg.getContext());
//   DictionaryAttr accessInfo;

//   StringAttr typeAttr;
//   AffineMapAttr mapAttr;
//   int64_t numAccesses = computeNumAccesses(theOnlyAccessOp);

//   if (auto load = dyn_cast<AffineLoadOp>(theOnlyAccessOp)) {
//     typeAttr = b.getStringAttr("load");
//     mapAttr = AffineMapAttr::get(load.getAffineMap());
//   } else if (auto store = dyn_cast<AffineStoreOp>(theOnlyAccessOp)) {
//     typeAttr = b.getStringAttr("store");
//     mapAttr = AffineMapAttr::get(store.getAffineMap());
//   } else {
//     return; // Shouldn’t happen with our filter.
//   }

//   auto numAttr = b.getI64IntegerAttr(numAccesses);

//   auto dict = DictionaryAttr::get(b.getContext(),
//                                   ArrayRef<NamedAttribute>{
//                                       b.getNamedAttr("type", typeAttr),
//                                       b.getNamedAttr("num_accesses", numAttr),
//                                       b.getNamedAttr("access_map", mapAttr),
//                                   });
//   // Attach to the *function argument* as requested.
//   if (auto barg = memrefArg.dyn_cast<BlockArgument>()) {
//     auto func = dyn_cast<func::FuncOp>(barg.getOwner()->getParentOp());
//     if (!func)
//       return;
//     // Mutate argument attribute dictionary (merge with existing).
//     auto idx = barg.getArgNumber();
//     auto oldDict = func.getArgAttrDict(idx);
//     SmallVector<NamedAttribute, 8> merged;
//     if (oldDict)
//       merged.append(oldDict.begin(), oldDict.end());
//     // Overwrite if already present.
//     // Remove existing "access.info" if any:
//     llvm::erase_if(merged, [&](const NamedAttribute &na) {
//       return na.getName() == "access.info";
//     });
//     merged.emplace_back(b.getStringAttr("access.info"), dict);
//     func.setArgAttrs(idx, DictionaryAttr::get(b.getContext(), merged));
//   }
// }

// /// Main utility: annotate single-access func-arg memrefs.
// static LogicalResult annotateSingleAccessArgMemrefs(func::FuncOp func) {
//   // Collect memref-typed function arguments.
//   SmallVector<Value, 8> memrefArgs;
//   for (auto [idx, argType] : llvm::enumerate(func.getFunctionType().getInputs())) {
//     if (isa<MemRefType>(argType)) {
//       Value v = func.getBody().front().getArgument(idx);
//       memrefArgs.push_back(v);
//     }
//   }

//   // For each arg, count affine.load/affine.store that touch it.
//   for (Value arg : memrefArgs) {
//     // Walk users to find load/store ops using this exact memref value.
//     SmallVector<Operation *, 2> accesses;

//     // We only consider affine.load/store where the memref operand == arg.
//     func.walk([&](Operation *op) {
//       if (auto ld = dyn_cast<AffineLoadOp>(op)) {
//         if (ld.getMemRef() == arg)
//           accesses.push_back(op);
//       } else if (auto st = dyn_cast<AffineStoreOp>(op)) {
//         if (st.getMemRef() == arg)
//           accesses.push_back(op);
//       }
//     });

//     // Must be exactly one total (either load OR store).
//     if (accesses.size() != 1)
//       continue;

//     annotateArgForSingleAccess(arg, accesses.front());
//   }

//   return success();
// }

// } // namespace
namespace mlir {
namespace streamhls {

/// Annotate all eligible function arguments with dataflow.access_info.
/// - Only memref-typed args are considered.
/// - The arg must have exactly one user, which must be an affine.load/store whose
///   memref operand is the argument itself.
/// - On success, attaches "dataflow.access_info" to the argument’s attr dict.
///
/// NOTE: Mutates the module in place and returns it.
ModuleOp annotateMemrefs(ModuleOp &module) {

  module.walk([&](func::FuncOp func) {
    // Ignore external functions.
    if (func.isExternal()) return;

    Block &entry = func.getBody().front();
    for (auto it : llvm::enumerate(entry.getArguments())) {
      unsigned argIdx = it.index();
      BlockArgument arg = it.value();

      // Only consider memref arguments.
      auto memTy = dyn_cast<MemRefType>(arg.getType());
      if (!memTy) continue;

      // Must have exactly one user overall.
      if (!arg.hasOneUse()) {
        func.emitError() << "memref argument #" << argIdx
                         << " must have exactly one use (found "
                         << std::distance(arg.use_begin(), arg.use_end()) << ")";
        continue; // or early-return if you prefer hard failure
      }

      Operation *userOp =  arg.use_begin()->getOwner();  // Use has getOwner()
      // Check it's an affine.load/store and that the memref operand is the argument.
      Operation *accessOp = nullptr;
      if (auto ld = dyn_cast<affine::AffineLoadOp>(userOp)) {
        if (ld.getMemRef() == arg) accessOp = ld.getOperation();
      } else if (auto st = dyn_cast<affine::AffineStoreOp>(userOp)) {
        if (st.getMemRef() == arg) accessOp = st.getOperation();
      }
      if (!accessOp) {
        func.emitError() << "memref argument #" << argIdx
                         << " must be used by a single affine.load/store on itself";
        continue;
      }

      // Build the access info (assumes static bounds per your simplification).
      auto infoOr = mlir::streamhls::buildAccessInfoForAffineAccess(accessOp);
      if (failed(infoOr)) {
        accessOp->emitError() << "failed to build dataflow.access_info";
        continue;
      }

      // Attach as an argument attribute: "dataflow.access_info".
      func.setArgAttr(argIdx, "dataflow.access_info", *infoOr);
    }
  });

  return module;
} 
// ModuleOp annotateMemrefs(ModuleOp &module) {
//   llvm::dbgs() << "Starting memref annotation...\n";
//   // module.dump();
//   module.walk([&](func::FuncOp func) {
//     // func.dump();
//     (void)annotateSingleAccessArgMemrefs(func);
//   });
//   return module;
// }
}
}

