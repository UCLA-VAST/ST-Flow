#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LogicalResult.h"

#include "streamhls/Dialect/Dataflow/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace streamhls;
using namespace dataflow;

namespace {

/// Compute the flattened 1-D tensor type for an array-of-streams tensor. Only
/// static shapes are supported.
static FailureOr<RankedTensorType> getFlattenedArrayOfStreamsType(
    RankedTensorType type) {
  if (type.getRank() <= 1)
    return failure();

  if (!type.getElementType().isa<StreamType>())
    return failure();

  if (llvm::is_contained(type.getShape(), ShapedType::kDynamic))
    return failure();

  int64_t flatSize = 1;
  for (int64_t dim : type.getShape())
    flatSize *= dim;

  return RankedTensorType::get({flatSize}, type.getElementType(),
                               type.getEncoding());
}

/// Rewrites the affine access map on an ArrayOfStreams read/write to address
/// the flattened 1-D buffer. Keeps the same number of indices and collapses
/// multi-dimensional accesses into a single linearized expression.
static LogicalResult updateAccessMap(Operation *user,
                                     RankedTensorType originalType) {
  AffineMap map;
  // StringRef mapAttrName;
  if (auto read = dyn_cast<affine::AffineReadOpInterface>(user)) {
    map = read.getAffineMap();
    // mapAttrName = read.getMapAttrStrName();
  } else if (auto write = dyn_cast<affine::AffineWriteOpInterface>(user)) {
    map = write.getAffineMap();
    // mapAttrName = write.getMapAttrStrName();
  } else {
    return success();
  }

  if (!map)
    return user->emitError("array-of-streams access missing affine map");

  unsigned rank = originalType.getRank();
  if (map.getNumResults() != rank)
    return user->emitError("expected map results to match array-of-stream rank");

  ArrayRef<int64_t> shape = originalType.getShape();
  if (llvm::is_contained(shape, ShapedType::kDynamic))
    return user->emitError(
        "cannot flatten array-of-streams with dynamic dimensions");

  SmallVector<int64_t> strides(rank, 1);
  for (int i = rank - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * shape[i + 1];

  MLIRContext *ctx = user->getContext();
  AffineExpr linearExpr = getAffineConstantExpr(0, ctx);
  for (unsigned i = 0; i < rank; ++i) {
    AffineExpr term = map.getResult(i);
    if (strides[i] != 1)
      term = term * strides[i];
    linearExpr = linearExpr + term;
  }

  auto newMap =
      AffineMap::get(map.getNumDims(), map.getNumSymbols(), linearExpr, ctx);
  // user->setAttr(mapAttrName, AffineMapAttr::get(newMap));
  if (auto read = dyn_cast<ArrayOfStreamsReadOp>(user)) {
    read.setMap(newMap);
  } else if (auto write = dyn_cast<ArrayOfStreamsWriteOp>(user)) {
    write.setMap(newMap);
  }
  return success();
}

/// Flatten a single SSA value if it is a multi-dimensional array-of-streams and
/// update its users' affine maps.
static LogicalResult flattenArrayOfStreamsValue(Value value) {
  auto type = value.getType().dyn_cast<RankedTensorType>();
  if (!type)
    return success();

  auto streamType = type.getElementType().dyn_cast<StreamType>();
  if (!streamType)
    return success();

  if (type.getRank() <= 1)
    return success();

  auto flattenedType = getFlattenedArrayOfStreamsType(type);
  if (failed(flattenedType))
    return emitError(value.getLoc(),
                     "cannot flatten array-of-streams with dynamic or invalid "
                     "shape");

  for (Operation *user : value.getUsers())
    if (failed(updateAccessMap(user, type)))
      return failure();

  value.setType(*flattenedType);
  return success();
}

/// Update the function signature and call sites after argument/result types
/// were mutated in-place.
static void updateFunctionTypes(ModuleOp module, func::FuncOp funcOp) {
  FunctionType funcType = funcOp.getFunctionType();
  SmallVector<Type> inputTypes;
  for (BlockArgument arg : funcOp.getArguments())
    inputTypes.push_back(arg.getType());

  SmallVector<Type> resultTypes(funcType.getResults().begin(),
                                funcType.getResults().end());
  bool foundReturn = false;
  funcOp.walk([&](func::ReturnOp ret) {
    if (foundReturn)
      return;
    resultTypes.clear();
    for (Value operand : ret.getOperands())
      resultTypes.push_back(operand.getType());
    foundReturn = true;
  });

  bool inputsChanged =
      !std::equal(inputTypes.begin(), inputTypes.end(),
                  funcType.getInputs().begin(), funcType.getInputs().end());
  bool resultsChanged =
      !std::equal(resultTypes.begin(), resultTypes.end(),
                  funcType.getResults().begin(), funcType.getResults().end());

  if (!inputsChanged && !resultsChanged)
    return;

  funcOp.setType(
      FunctionType::get(funcOp.getContext(), inputTypes, resultTypes));

  if (auto uses = SymbolTable::getSymbolUses(funcOp, module))
    for (const auto &use : *uses)
      if (auto call = dyn_cast<func::CallOp>(use.getUser()))
        for (auto [result, type] : llvm::zip(call.getResults(), resultTypes))
          result.setType(type);
}

} // namespace

ModuleOp streamhls::flattenArrayOfStreams(ModuleOp &module) {
  LogicalResult flattenStatus = success();

  module.walk([&](Operation *op) {
    for (Value result : op->getResults())
      if (failed(flattenArrayOfStreamsValue(result)))
        flattenStatus = failure();

    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          if (failed(flattenArrayOfStreamsValue(arg)))
            flattenStatus = failure();
  });

  if (failed(flattenStatus)) {
    module.emitError("failed to flatten all array-of-streams values");
    return module;
  }

  module.walk([&](func::FuncOp funcOp) { updateFunctionTypes(module, funcOp); });
  return module;
}
