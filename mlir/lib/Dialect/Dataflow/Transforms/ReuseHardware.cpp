// #include "PassDetail.h"


// #include "mlir/CAPI/IR.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/InitAllDialects.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/MapVector.h"

#include "streamhls/Dialect/Dataflow/Transforms/Passes.h"
#include "streamhls/Dialect/Dataflow/Support/Utils.h"
#include <math.h>
#include <regex>
#include "mlir/IR/SymbolTable.h"

#define DEBUG_TYPE "streamhls-merge-kernels"


using namespace mlir;
using namespace streamhls;



namespace mlir {
namespace streamhls {

static SmallVector<SmallVector<int64_t>>
extractUbsFromMergeVariants(ModuleOp mod) {
  SmallVector<SmallVector<int64_t>> allUbs;
  if (auto arr = mod->getAttrOfType<ArrayAttr>("merge.variants")) {
    for (Attribute a : arr) {
      SmallVector<int64_t> ubs;
      if(auto dict = dyn_cast<DictionaryAttr>(a)){
        if(auto loopDict = dict.getAs<DictionaryAttr>("loop")){
          if(auto ubArr = loopDict.getAs<DenseI64ArrayAttr>("ubs")){
            ubs.assign(ubArr.asArrayRef().begin(), ubArr.asArrayRef().end());
          } else {
            assert(false && "UBs not found in merge.variants");
          }
        } else {
          assert(false && "loop dict not found in merge.variants");
        }
      } else {
        assert(false && "dict not found in merge.variants");
      }
      allUbs.push_back(std::move(ubs));
    }
  }
  return allUbs;
}


static func::FuncOp getCalleeFunction(ModuleOp mod) {
  for (auto f : mod.getOps<func::FuncOp>()) {
    return f; // return the first function found
  }
  return nullptr;
}

/// Build a wrapper right before the merged callee.
/// Wrapper type = (callee non-index inputs) repeated for each variant; no results.
/// Body: for each variant i, call callee(arg-slice_i..., const ub[0], const ub[1], ...).
// ModuleOp createWrapper(ModuleOp &mergedModule, StringRef wrapperFuncName) {
ModuleOp reuseHardware(ModuleOp &mergedModule) {
  // StringRef wrapperFuncName = "top_" + mergedModule.getName().value_or("wrapper");
  ModuleOp mod = mergedModule;
  OpBuilder topBuilder(mod.getContext());

  // 1) Read all variants’ UB vectors.
  SmallVector<SmallVector<int64_t>> allUbs = extractUbsFromMergeVariants(mod);
  unsigned ubArity = 0;
  if (!allUbs.empty()) {
    // Nothing to wrap — just return.
    ubArity = static_cast<unsigned>(allUbs.front().size());
  }else{
    allUbs.push_back({});
  }
  // const unsigned ubArity = static_cast<unsigned>(allUbs.front().size());

  // 2) Locate the merged callee
  //    assume there is only one function in the merged module and get it
  func::FuncOp callee = getCalleeFunction(mod);
  assert(callee && "Merged callee function not found");
  StringRef prefix = "top";
  StringRef mergedFuncName = callee.getSymName();
  std::string wrapperFuncNameStr = (prefix + "_" + mergedFuncName).str();
  StringRef wrapperFuncName = wrapperFuncNameStr;
  // 3) Split callee args into (nonIndexPrefix, ubSuffix).
  FunctionType calleeTy = cast<FunctionType>(callee.getFunctionType());
  unsigned nInputs = calleeTy.getNumInputs();
  assert(nInputs >= ubArity && "callee has fewer inputs than UB arity");
  unsigned nonIdxCount = nInputs - ubArity;

  SmallVector<Type> nonIdxArgTypes;
  nonIdxArgTypes.reserve(nonIdxCount);
  for (unsigned i = 0; i < nonIdxCount; ++i)
    nonIdxArgTypes.push_back(calleeTy.getInput(i));

  // 4) Build the wrapper function type: replicate nonIndex args per variant.
  SmallVector<Type> wrapperInputs;
  wrapperInputs.reserve(nonIdxCount * allUbs.size());
  for (size_t i = 0; i < allUbs.size(); ++i)
    wrapperInputs.append(nonIdxArgTypes.begin(), nonIdxArgTypes.end());

  auto wrapperTy = FunctionType::get(mod.getContext(), wrapperInputs, {});
  Location loc = UnknownLoc::get(mod.getContext());

  // 5) Create the wrapper right before the callee for deterministic order.
  OpBuilder mBuilder(mod);
  mBuilder.setInsertionPointAfter(callee); // insert directly after callee
  auto wrapper = mBuilder.create<func::FuncOp>(loc, wrapperFuncName, wrapperTy);
  wrapper.setVisibility(SymbolTable::Visibility::Public);

  // 5b) Propagate buffer.info attributes from callee parameters or merge.variants
  unsigned wrapperArgIdx = 0;
  auto variantsAttr = mod->getAttrOfType<ArrayAttr>("merge.variants");

  for (size_t vi = 0; vi < allUbs.size(); ++vi) {
    DictionaryAttr variantDict = nullptr;
    ArrayAttr bufInfoList = nullptr;
    if (variantsAttr && vi < variantsAttr.size()) {
      variantDict = dyn_cast<DictionaryAttr>(variantsAttr[vi]);
      if (variantDict)
        bufInfoList = variantDict.getAs<ArrayAttr>("buffer_info_list");
    }

    for (unsigned i = 0; i < nonIdxCount; ++i) {
      Attribute finalAttr = nullptr;
      // Try getting the buffer.info for this specific variant and argument
      if (bufInfoList && i < bufInfoList.size()) {
        if (auto wrapped = dyn_cast<DictionaryAttr>(bufInfoList[i])) {
          finalAttr = wrapped.get("buffer.info");
        }
      }
      // Fallback to callee's primary attribute if variant info is missing
      if (!finalAttr) {
        finalAttr = callee.getArgAttr(i, "buffer.info");
      }

      if (finalAttr) {
        wrapper.setArgAttr(wrapperArgIdx, "buffer.info", finalAttr);
      }
      wrapperArgIdx++;
    }
  }

  // 6) Build the entry block & IR builder.
  auto *entry = wrapper.addEntryBlock();
  OpBuilder b(entry, entry->begin());

  // Helper: constants for UBs (reused across calls when identical).
  auto getIndexConst = [&](int64_t v) -> Value {
    return b.create<arith::ConstantIndexOp>(loc, v);
  };

  // 7) Emit one call per variant: slice wrapper args and append UB constants.
  unsigned cursor = 0;
  for (size_t vi = 0; vi < allUbs.size(); ++vi) {
    // a) Take this variant's non-index operands from the wrapper's block args.
    SmallVector<Value> callArgs;
    callArgs.reserve(nInputs);
    for (unsigned ai = 0; ai < nonIdxCount; ++ai)
      callArgs.push_back(entry->getArgument(cursor + ai));
    cursor += nonIdxCount;

    // b) Append ub constants (index) in callee order.
    const auto &ubs = allUbs[vi];
    assert(ubs.size() == ubArity && "UB arity mismatch");
    for (unsigned ui = 0; ui < ubArity; ++ui)
      callArgs.push_back(getIndexConst(ubs[ui]));

    // c) Call merged callee.
    b.create<func::CallOp>(loc, callee.getSymName(), TypeRange{}, callArgs);
  }

  // 8) Return.
  b.create<func::ReturnOp>(loc);

  return mod;
}

void reuseHardwareDFG(DFG &dfg) {
  for (auto &nodePair : dfg.nodes) {
    auto &node = nodePair.second;
    if (node.fifoModule && node.kind == DFG::NodeKind::COMPUTE) {
      node.reuseModule = reuseHardware(node.fifoModule);
      node.hwModule = node.reuseModule;
    }
  }
}


} // namespace streamhls
} // namespace mlir