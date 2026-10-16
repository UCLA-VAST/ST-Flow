/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Modification: ScaleHLS
 * https://github.com/hanchenye/scalehls
 */

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"

#include "streamhls/Dialect/Dataflow/Support/Utils.h"

using namespace mlir;
using namespace streamhls;
using namespace dataflow;

std::string streamhls::getMemrefBufferName(Value memref) {
  if (!memref) return "";
  if (auto allocOp = memref.getDefiningOp<memref::AllocOp>()) {
    if (auto dict = allocOp->getAttrOfType<DictionaryAttr>("buffer.info")) {
      if (auto nameAttr = dict.get("name").dyn_cast_or_null<StringAttr>())
        return nameAttr.getValue().str();
    }
  } else if (auto arg = mlir::dyn_cast<BlockArgument>(memref)) {
    if (auto func = mlir::dyn_cast<func::FuncOp>(arg.getOwner()->getParentOp())) {
      if (auto dict = func.getArgAttrOfType<DictionaryAttr>(arg.getArgNumber(), "buffer.info")) {
        if (auto nameAttr = dict.get("name").dyn_cast_or_null<StringAttr>())
          return nameAttr.getValue().str();
      }
    }
  }
  return "";
}

#define DEBUG_TYPE "streamhls-utils"

// bool streamhls::getMinBufferSize(Operation* storeOrLoad, AffineLoopBand band, uint64_t &size) {
//   if(!isa<AffineLoadOp, AffineStoreOp>(storeOrLoad)){
//     return false;
//   }
//   SmallVector<Value, 4> loopIVs;
//   for(auto loop : band){
//     loopIVs.push_back(loop.getInductionVar());
//   }
//   Value memref;
//   if (auto loadOp = dyn_cast<AffineLoadOp>(storeOrLoad)) {
//     memref = loadOp.getMemRef();
//   } else if (auto storeOp = dyn_cast<AffineStoreOp>(storeOrLoad)) {
//     memref = storeOp.getMemRef();
//   }  
//   auto originalShape = memref.getType().cast<MemRefType>().getShape();
//   auto originalSize = 1;
//   for(auto dim : originalShape){
//     originalSize *= dim;
//   }
//   size = originalSize;
//   if (auto loadOp = dyn_cast<AffineLoadOp>(storeOrLoad)) {
//     auto mapOperands = loadOp.getMapOperands();
//     SmallVector<Value, 4> loopIVs;
//     for(auto loop : band){
//       loopIVs.push_back(loop.getInductionVar());
//     }
//     SmallVector<Value, 4> bufferIndices;
//     for(auto val : mapOperands){
//       bufferIndices.push_back(val);
//     }
//     Value irrelevantIV;
//     for(auto iv : loopIVs){
//       bool found = false;
//       for(auto val : bufferIndices){
//         if(val == iv){
//           found = true;
//           break;
//         }
//       }
//       if(!found){
//         irrelevantIV = iv;
//         break;
//       }
//     }
//     if(!irrelevantIV){
//       return false;
//     }
//     SmallVector<Value, 4> dimValsToKeep;
//     unsigned endIdx = 0;
//     for(auto iv : loopIVs){
//       if(iv == irrelevantIV){
//         break;
//       }
//       endIdx++;
//     }
//     for(unsigned i = endIdx; i < loopIVs.size(); i++){
//       dimValsToKeep.push_back(loopIVs[i]);
//     }
//     SmallVector<unsigned, 4> dimIdxsToKeep;
//     for(auto val : dimValsToKeep){
//       for(unsigned i = 0; i < bufferIndices.size(); i++){
//         if(bufferIndices[i] == val){
//           dimIdxsToKeep.push_back(i);
//         }
//       }
//     }

//     if(dimIdxsToKeep.size() == originalShape.size()){
//       return false;
//     }

//     SmallVector<int64_t, 4> newShape; 
//     SmallVector<unsigned, 4> remainDims;
//     for(auto idx : dimIdxsToKeep){
//       LLVM_DEBUG(
//         llvm::dbgs() << "Removing dimension " << idx << "\n";
//       );
//       newShape.push_back(originalShape[idx]);
//       remainDims.push_back(idx);
//     }
//     for(auto shp : originalShape){
//       LLVM_DEBUG(
//         llvm::dbgs() << shp << " ";
//       );
//     }
//     LLVM_DEBUG(
//       llvm::dbgs() << "\n";
//     );
//     for(auto shp : newShape){
//       LLVM_DEBUG(
//         llvm::dbgs() << shp << " ";
//       );
//     }
//     if(originalShape.size() == newShape.size()){
//       return false;
//     }
//     auto newSize = 1;
//     for(auto dim : newShape){
//       newSize *= dim;
//     }
//     size = newSize;
//     LLVM_DEBUG(
//       llvm::dbgs() << "Original size: " << originalSize << "\n";
//       llvm::dbgs() << "New size: " << newSize << "\n";
//     );
//   } else if (auto storeOp = dyn_cast<AffineStoreOp>(storeOrLoad)) {
//     auto mapOperands = storeOp.getMapOperands();
//     SmallVector<Value, 4> loopIVs;
//     for(auto loop : band){
//       loopIVs.push_back(loop.getInductionVar());
//     }
//     SmallVector<Value, 4> bufferIndices;
//     for(auto val : mapOperands){
//       bufferIndices.push_back(val);
//     }
//     Value irrelevantIV;
//     for(auto iv : loopIVs){
//       bool found = false;
//       for(auto val : bufferIndices){
//         if(val == iv){
//           found = true;
//           break;
//         }
//       }
//       if(!found){
//         irrelevantIV = iv;
//         break;
//       }
//     }
//     if(!irrelevantIV){
//       return false;
//     }
//     SmallVector<Value, 4> dimValsToKeep;
//     unsigned endIdx = 0;
//     for(auto iv : loopIVs){
//       if(iv == irrelevantIV){
//         break;
//       }
//       endIdx++;
//     }
//     for(unsigned i = endIdx; i < loopIVs.size(); i++){
//       dimValsToKeep.push_back(loopIVs[i]);
//     }
//     SmallVector<unsigned, 4> dimIdxsToKeep;
//     for(auto val : dimValsToKeep){
//       for(unsigned i = 0; i < bufferIndices.size(); i++){
//         if(bufferIndices[i] == val){
//           dimIdxsToKeep.push_back(i);
//         }
//       }
//     }

//     if(dimIdxsToKeep.size() == originalShape.size()){
//       return false;
//     }

//     SmallVector<int64_t, 4> newShape; 
//     SmallVector<unsigned, 4> remainDims;
//     for(auto idx : dimIdxsToKeep){
//       LLVM_DEBUG(
//         llvm::dbgs() << "Removing dimension " << idx << "\n";
//       );
//       newShape.push_back(originalShape[idx]);
//       remainDims.push_back(idx);
//     }
//     for(auto shp : originalShape){
//       LLVM_DEBUG(
//         llvm::dbgs() << shp << " ";
//       );
//     }
//     LLVM_DEBUG(
//       llvm::dbgs() << "\n";
//     );
//     for(auto shp : newShape){
//       LLVM_DEBUG(
//         llvm::dbgs() << shp << " ";
//       );
//     }
//     if(originalShape.size() == newShape.size()){
//       return false;
//     }
//     auto newSize = 1;
//     for(auto dim : newShape){
//       newSize *= dim;
//     }
//     size = newSize;
//     LLVM_DEBUG(
//       llvm::dbgs() << "Original size: " << originalSize << "\n";
//       llvm::dbgs() << "New size: " << newSize << "\n";
//     );
//   }
//   return true;
// }


LogicalResult streamhls::applyLoopTiling(AffineLoopBand &band, SmallVectorImpl<unsigned> &tileList,
                               bool loopNormalize, bool annotatePointLoop) {
  assert(!band.empty() && "no loops provided");
  if (!isPerfectlyNested(band))
    return failure();

  // If all tile sizes are one, we don't need to do anything but annotating all
  // loops as point loop.
  if (llvm::all_of(tileList, [](unsigned size) { return size == 1; })) {
    for (auto loop : band)
      if (annotatePointLoop)
        setReductionAttr(loop);
    return success();
  }

  // Record the original band size and attributes to make use of later.
  auto originalBandSize = band.size();
  SmallVector<std::pair<bool, bool>, 6> flags;
  for (auto loop : band)
    flags.push_back({hasParallelAttr(loop), hasReductionAttr(loop)});

  // Apply loop tiling.
  AffineLoopBand tiledBand;
  if (failed(mlir::affine::tilePerfectlyNested(band, tileList, &tiledBand)))
    return failure();

  // Get the tile loop band and point loop band.
  AffineLoopBand pointBand(std::next(tiledBand.begin(), originalBandSize),
                           tiledBand.end());
  tiledBand.resize(originalBandSize);

  // Annotate the required attributes.
  for (auto zip : llvm::zip(tiledBand, pointBand, flags)) {
    auto tileLoop = std::get<0>(zip);
    auto pointLoop = std::get<1>(zip);
    auto flag = std::get<2>(zip);

    // If a tile loop is parallel, the corresponding point loop should also be
    // a parallel loop.
    if (flag.first) {
      setParallelAttr(tileLoop);
      setParallelAttr(pointLoop);
    }

    // Re-annotate the point attribute to the tile loop if required.
    if (flag.second)
      setReductionAttr(tileLoop);

    // Annotate the point attribute to the point loop.
    if (annotatePointLoop)
      setReductionAttr(pointLoop);
  }

  // Always normalize point loop band.
  for (auto loop : pointBand)
    (void)normalizeAffineFor(loop);

  // Normalize tiled loop band if required.
  if (loopNormalize) {
    band.clear();
    for (auto loop : tiledBand)
      if (failed(promoteIfSingleIteration(loop))) {
        (void)normalizeAffineFor(loop);
        band.push_back(loop);
      }
  } else
    band = tiledBand;
  return success();
}



AffineMap streamhls::getSerializedAffineMap(ArrayRef<int64_t> shape, OpBuilder &builder){
  AffineExpr expr = builder.getAffineConstantExpr(0);
  for(unsigned i = 0; i < shape.size(); i++){
    if(i == shape.size() - 1){
      expr = expr + builder.getAffineDimExpr(i);
    }else{
      expr = expr + builder.getAffineDimExpr(i) * builder.getAffineConstantExpr(shape[i + 1]);
    }
  }
  return AffineMap::get(shape.size(), 0, expr);
}


AffineMap streamhls::getSerializedAffineMapFromBand(AffineLoopBand band, MLIRContext *context) {
  AffineExpr expr = getAffineConstantExpr(0, context);
  for(unsigned i = 0; i < band.size(); i++){
    auto tmpExpr = getAffineDimExpr(i, context);
    for(unsigned j = i + 1; j < band.size(); j++){
      assert(band[j].hasConstantBounds() && "loop bounds must be constant");
      assert(band[j].getConstantLowerBound() == 0 && "loop lower bound must be 0");
      auto ub = band[j].getConstantUpperBound();
      if(i < band.size() - 1)
        tmpExpr = tmpExpr * getAffineConstantExpr(ub, context);
    }
    expr = expr + tmpExpr;
  }
  return AffineMap::get(band.size(), 0, expr);
}

// Recursively replace floordiv(...) by ceildiv(...) inside an AffineExpr.
AffineExpr streamhls::floorToCeilDiv(AffineExpr e) {
  switch (e.getKind()) {
  case AffineExprKind::Add: {
    auto b = e.cast<AffineBinaryOpExpr>();
    return floorToCeilDiv(b.getLHS()) + floorToCeilDiv(b.getRHS());
  }
  case AffineExprKind::Mul: {
    auto b = e.cast<AffineBinaryOpExpr>();
    return floorToCeilDiv(b.getLHS()) * floorToCeilDiv(b.getRHS());
  }
  case AffineExprKind::Mod: {
    auto b = e.cast<AffineBinaryOpExpr>();
    auto lhs = floorToCeilDiv(b.getLHS());
    auto rhs = floorToCeilDiv(b.getRHS());
    return getAffineBinaryOpExpr(AffineExprKind::Mod, lhs, rhs);
  }
  case AffineExprKind::FloorDiv: {
    auto b = e.cast<AffineBinaryOpExpr>();
    auto lhs = floorToCeilDiv(b.getLHS());
    auto rhs = floorToCeilDiv(b.getRHS());
    // Prefer the convenience API when RHS is a positive constant.
    if (auto c = rhs.dyn_cast<AffineConstantExpr>()) {
      uint64_t d = static_cast<uint64_t>(c.getValue());
      return lhs.ceilDiv(d);
    }
    // Symbolic or non-constant RHS: use the generic binary-op builder.
    return getAffineBinaryOpExpr(AffineExprKind::CeilDiv, lhs, rhs);
  }
  case AffineExprKind::CeilDiv: {
    auto b = e.cast<AffineBinaryOpExpr>();
    auto lhs = floorToCeilDiv(b.getLHS());
    auto rhs = floorToCeilDiv(b.getRHS());
    return getAffineBinaryOpExpr(AffineExprKind::CeilDiv, lhs, rhs);
  }
  case AffineExprKind::Constant:
  case AffineExprKind::DimId:
  case AffineExprKind::SymbolId:
    return e; // leaves leaves unchanged
  }
  llvm_unreachable("unhandled AffineExprKind");
}

bool streamhls::getMinBufferSize(memref::AllocOp allocOp, AffineLoopBand band, uint64_t &size) {
  auto memRef = allocOp.getResult();
  auto originalShape = allocOp.getType().getShape();
  auto originalSize = 1;
  for(auto dim : originalShape){
    originalSize *= dim;
  }
  size = originalSize;
  SmallVector<Operation*, 4> loadAndStoreOps;
  SmallVector<Operation::operand_range, 4> mapOperands;
  for(auto user : memRef.getUsers()){
    if(isa<AffineLoadOp, AffineStoreOp>(user)){
      loadAndStoreOps.push_back(user);
      if(auto loadOp = dyn_cast<AffineLoadOp>(user)){
        mapOperands.push_back(loadOp.getMapOperands());
      }
      if(auto storeOp = dyn_cast<AffineStoreOp>(user)){
        mapOperands.push_back(storeOp.getMapOperands());
      }
    }
  }
  for(auto operands1 : mapOperands){
    for(auto operands2 : mapOperands){
      if(operands1.size() != operands2.size()){
        return false;
      }
      for(unsigned i = 0; i < operands1.size(); i++){
        if(operands1[i] != operands2[i]){
          return false;
        }
      }
    }
  }
  SmallVector<Value, 4> loopIVs;
  for(auto loop : band){
    loopIVs.push_back(loop.getInductionVar());
  }
  SmallVector<Value, 4> bufferIndices;
  for(auto val : mapOperands[0]){
    bufferIndices.push_back(val);
  }
  Value irrelevantIV;
  for(auto iv : loopIVs){
    bool found = false;
    for(auto val : bufferIndices){
      if(val == iv){
        found = true;
        break;
      }
    }
    if(!found){
      irrelevantIV = iv;
      break;
    }
  }
  if(!irrelevantIV){
    return false;
  }
  SmallVector<Value, 4> dimValsToKeep;
  unsigned endIdx = 0;
  for(auto iv : loopIVs){
    if(iv == irrelevantIV){
      break;
    }
    endIdx++;
  }
  for(unsigned i = endIdx; i < loopIVs.size(); i++){
    dimValsToKeep.push_back(loopIVs[i]);
  }
  SmallVector<unsigned, 4> dimIdxsToKeep;
  for(auto val : dimValsToKeep){
    for(unsigned i = 0; i < bufferIndices.size(); i++){
      if(bufferIndices[i] == val){
        dimIdxsToKeep.push_back(i);
      }
    }
  }

  if(dimIdxsToKeep.size() == originalShape.size()){
    return false;
  }

  SmallVector<int64_t, 4> newShape; 
  SmallVector<unsigned, 4> remainDims;
  for(auto idx : dimIdxsToKeep){
    LLVM_DEBUG(
      llvm::dbgs() << "Removing dimension " << idx << "\n";
    );
    newShape.push_back(originalShape[idx]);
    remainDims.push_back(idx);
  }
  for(auto shp : originalShape){
    LLVM_DEBUG(
      llvm::dbgs() << shp << " ";
    );
  }
  LLVM_DEBUG(
    llvm::dbgs() << "\n";
  );
  for(auto shp : newShape){
    LLVM_DEBUG(
      llvm::dbgs() << shp << " ";
    );
  }
  if(originalShape.size() == newShape.size()){
    return false;
  }
  auto newSize = 1;
  for(auto dim : newShape){
    newSize *= dim;
  }
  size = newSize;
  return true;
}
// MY Utils
// get the inner most loop band math or arith ops
void streamhls::getBandMathOrArithOps(
  Block *block, 
  llvm::SmallDenseMap<OperationName, int64_t> &mathOrArithOps
){
  for(auto &op : block->getOperations()){
    // if op belongs to math or arith dialect, add it to map
    if(op.getDialect()->getNamespace() == "math" || op.getDialect()->getNamespace() == "arith"){
      auto opName = op.getName();
      if(mathOrArithOps.find(opName) == mathOrArithOps.end())
        mathOrArithOps[opName] = 1;
      else
        mathOrArithOps[opName]++;
    }
  }
}


// total number of iterations of loop nest
int64_t streamhls::getLoopNestIterations(AffineLoopBand loopBand) {
  int64_t iterations = 1;
  for(auto loop : loopBand){
    // if loop doesn't have constant bounds, return -1
    if(loop.hasConstantBounds()){
      auto lb = loop.getConstantLowerBound();
      auto ub = loop.getConstantUpperBound();
      iterations *= (ub - lb);
    } else
      return -1;
  }
  return iterations;
}

// Function to get the body of the innermost loop.
Block *streamhls::getInnermostLoopBody(AffineForOp forOp) {
  auto body = forOp.getBody();
  while (true) {
    auto &ops = body->getOperations();
    if (ops.empty())
      break;

    auto &lastOp = ops.back();
    if (isa<AffineForOp>(lastOp))
      body = cast<AffineForOp>(lastOp).getBody();
    else
      break;
  }
  return body;
}

//===----------------------------------------------------------------------===//
// HLSCpp attribute utils
//===----------------------------------------------------------------------===//

/// Parse loop directives.
Attribute streamhls::getLoopDirective(Operation *op, std::string name) {
  return op->getAttr(name);
}

StringRef streamhls::getLoopName(AffineForOp &forOp) {
  if (forOp->hasAttr("loop_name"))
    return forOp->getAttr("loop_name").cast<StringAttr>().getValue();
  else
    return "";
}

void streamhls::setLoopName(AffineForOp &forOp, std::string loop_name) {
  forOp->setAttr("loop_name", StringAttr::get(forOp->getContext(), loop_name));
}

void streamhls::setStageName(AffineForOp &forOp, StringRef op_name) {
  forOp->setAttr("op_name", StringAttr::get(forOp->getContext(), op_name));
}

std::vector<std::string> streamhls::split_names(const std::string &arg_names) {
  std::stringstream ss(arg_names);
  std::vector<std::string> args;
  while (ss.good()) {
    std::string substr;
    getline(ss, substr, ',');
    args.push_back(substr);
  }
  return args;
}

/// Parse other attributes.
SmallVector<int64_t, 8> streamhls::getIntArrayAttrValue(Operation *op,
                                                  StringRef name) {
  SmallVector<int64_t, 8> array;
  if (auto arrayAttr = op->getAttrOfType<ArrayAttr>(name)) {
    for (auto attr : arrayAttr)
      if (auto intAttr = attr.dyn_cast<IntegerAttr>())
        array.push_back(intAttr.getInt());
      else
        return SmallVector<int64_t, 8>();
    return array;
  } else
    return SmallVector<int64_t, 8>();
}

bool streamhls::setIntAttr(SmallVector<AffineForOp, 6> &forOps,
                     const SmallVector<int, 6> &attr_arr,
                     const std::string attr_name) {
  assert(forOps.size() == attr_arr.size());
  unsigned cnt_loop = 0;
  for (AffineForOp newForOp : forOps) {
    newForOp->setAttr(
        attr_name,
        IntegerAttr::get(
            IntegerType::get(newForOp->getContext(), 32,
                             IntegerType::SignednessSemantics::Signless),
            attr_arr[cnt_loop]));
    cnt_loop++;
  }
  return true;
}

bool streamhls::setLoopNames(SmallVector<AffineForOp, 6> &forOps,
                       const SmallVector<std::string, 6> &nameArr) {
  assert(forOps.size() == nameArr.size());
  unsigned cnt_loop = 0;
  for (AffineForOp newForOp : forOps) {
    newForOp->setAttr("loop_name", StringAttr::get(newForOp->getContext(),
                                                   nameArr[cnt_loop]));
    cnt_loop++;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Memory and loop analysis utils
//===----------------------------------------------------------------------===//

LogicalResult streamhls::getStage(func::FuncOp &func, AffineForOp &forOp,
                            StringRef op_name) {
  for (auto rootForOp : func.getOps<AffineForOp>()) {
    if (op_name ==
        rootForOp->getAttr("op_name").cast<StringAttr>().getValue()) {
      forOp = rootForOp;
      return success();
    }
  }
  return failure();
}

// void recursivelyFindLoop(AffineForOp forOp, int depth, StringRef loop_name,
//                          AffineForOp &retForOp, int &retDepth,
//                          SmallVector<AffineForOp> &loops);

// void recursivelyFindLoopWithIf(AffineIfOp ifOp, int depth, StringRef loop_name,
//                                AffineForOp &retForOp, int &retDepth,
//                                SmallVector<AffineForOp> &loops) {
//   for (auto nextForOp : ifOp.getThenBlock()->getOps<AffineForOp>())
//     recursivelyFindLoop(nextForOp, depth + 1, loop_name, retForOp, retDepth,
//                         loops);
//   for (auto nextIfOp : ifOp.getThenBlock()->getOps<AffineIfOp>())
//     recursivelyFindLoopWithIf(nextIfOp, depth, loop_name, retForOp, retDepth,
//                               loops);
// }

// void recursivelyFindLoop(AffineForOp forOp, int depth, StringRef loop_name,
//                          AffineForOp &retForOp, int &retDepth,
//                          SmallVector<AffineForOp> &loops) {
//   loops.push_back(forOp);
//   if (getLoopName(forOp) == loop_name) {
//     retForOp = forOp;
//     retDepth = depth;
//     return;
//   }
//   for (auto nextForOp : forOp.getOps<AffineForOp>())
//     recursivelyFindLoop(nextForOp, depth + 1, loop_name, retForOp, retDepth,
//                         loops);
//   for (auto ifOp : forOp.getOps<AffineIfOp>())
//     recursivelyFindLoopWithIf(ifOp, depth, loop_name, retForOp, retDepth,
//                               loops);
// }

// int streamhls::getLoop(AffineForOp &forOp, StringRef loop_name) {
//   // return the axis id
//   AffineForOp currentLoop = forOp;
//   int cnt = -1;
//   SmallVector<AffineForOp> loops;
//   recursivelyFindLoop(currentLoop, 0, loop_name, forOp, cnt, loops);
//   return cnt;
// }

// void streamhls::getLoops(AffineForOp &forOp, SmallVector<AffineForOp> &forOpList) {
//   int cnt = -1;
//   recursivelyFindLoop(forOp, 0, "_placeholder_", forOp, cnt, forOpList);
// }

// bool streamhls::findContiguousNestedLoops(const AffineForOp &rootAffineForOp,
//                                     SmallVector<AffineForOp, 6> &resForOps,
//                                     SmallVector<StringRef, 6> &nameArr,
//                                     int depth, bool countReductionLoops) {
//   // depth = -1 means traverses all the inner loops
//   AffineForOp forOp = rootAffineForOp;
//   unsigned int sizeNameArr = nameArr.size();
//   if (sizeNameArr != 0)
//     depth = sizeNameArr;
//   else if (depth == -1)
//     depth = 0x3f3f3f3f;
//   resForOps.clear();
//   for (int i = 0; i < depth; ++i) {
//     if (!forOp) {
//       if (depth != 0x3f3f3f3f)
//         return false;
//       else // reach the inner-most loop
//         return true;
//     }

//     Attribute attr = forOp->getAttr("loop_name");
//     const StringRef curr_loop = attr.cast<StringAttr>().getValue();
//     if (sizeNameArr != 0 && curr_loop != nameArr[i])
//       return false;

//     if (forOp->hasAttr("reduction") == 1 && !countReductionLoops) {
//       i--;
//     } else {
//       resForOps.push_back(forOp);
//       if (sizeNameArr == 0)
//         nameArr.push_back(curr_loop);
//     }
//     Block &body = forOp.getRegion().front();
//     // if (body.begin() != std::prev(body.end(), 2)) // perfectly nested
//     //   break;

//     forOp = dyn_cast<AffineForOp>(&body.front());
//   }
//   return true;
// }

/// Collect all load and store operations in the block and return them in "map".
// void hcl::getMemAccessesMap(Block &block, MemAccessesMap &map) {
//   for (auto &op : block) {
//     if (isa<AffineReadOpInterface, AffineWriteOpInterface>(op))
//       map[MemRefAccess(&op).memref].push_back(&op);

//     else if (op.getNumRegions()) {
//       // Recursively collect memory access operations in each block.
//       for (auto &region : op.getRegions())
//         for (auto &block : region)
//           getMemAccessesMap(block, map);
//     }
//   }
// }

// Check if the lhsOp and rhsOp are in the same block. If so, return their
// ancestors that are located at the same block. Note that in this check,
// AffineIfOp is transparent.
std::optional<std::pair<Operation *, Operation *>>
streamhls::checkSameLevel(Operation *lhsOp, Operation *rhsOp) {
  // If lhsOp and rhsOp are already at the same level, return true.
  if (lhsOp->getBlock() == rhsOp->getBlock())
    return std::pair<Operation *, Operation *>(lhsOp, rhsOp);

  // Helper to get all surrounding AffineIfOps.
  auto getSurroundIfs =
      ([&](Operation *op, SmallVector<Operation *, 4> &nests) {
        nests.push_back(op);
        auto currentOp = op;
        while (true) {
          if (auto parentOp = currentOp->getParentOfType<AffineIfOp>()) {
            nests.push_back(parentOp);
            currentOp = parentOp;
          } else
            break;
        }
      });

  SmallVector<Operation *, 4> lhsNests;
  SmallVector<Operation *, 4> rhsNests;

  getSurroundIfs(lhsOp, lhsNests);
  getSurroundIfs(rhsOp, rhsNests);

  // If any parent of lhsOp and any parent of rhsOp are at the same level,
  // return true.
  for (auto lhs : lhsNests)
    for (auto rhs : rhsNests)
      if (lhs->getBlock() == rhs->getBlock())
        return std::pair<Operation *, Operation *>(lhs, rhs);

  return std::optional<std::pair<Operation *, Operation *>>();
}

/// Returns the number of surrounding loops common to 'loopsA' and 'loopsB',
/// where each lists loops from outer-most to inner-most in loop nest.
unsigned streamhls::getCommonSurroundingLoops(Operation *A, Operation *B,
                                        AffineLoopBand *band) {
  SmallVector<AffineForOp, 4> loopsA, loopsB;
  getAffineForIVs(*A, &loopsA);
  getAffineForIVs(*B, &loopsB);

  unsigned minNumLoops = std::min(loopsA.size(), loopsB.size());
  unsigned numCommonLoops = 0;
  for (unsigned i = 0; i < minNumLoops; ++i) {
    if (loopsA[i] != loopsB[i])
      break;
    ++numCommonLoops;
    if (band != nullptr)
      band->push_back(loopsB[i]);
  }
  return numCommonLoops;
}

// /// Calculate the upper and lower bound of "bound" if possible.
// std::optional<std::pair<int64_t, int64_t>>
// streamhls::getBoundOfAffineBound(AffineBound bound) {
//   auto boundMap = bound.getMap();
//   if (boundMap.isSingleConstant()) {
//     auto constBound = boundMap.getSingleConstantResult();
//     return std::pair<int64_t, int64_t>(constBound, constBound);
//   }

//   // For now, we can only handle one result affine bound.
//   if (boundMap.getNumResults() != 1)
//     return std::optional<std::pair<int64_t, int64_t>>();

//   auto context = boundMap.getContext();
//   SmallVector<int64_t, 4> lbs;
//   SmallVector<int64_t, 4> ubs;
//   for (auto operand : bound.getOperands()) {
//     // Only if the affine bound operands are induction variable, the calculation
//     // is possible.
//     if (!isAffineForInductionVar(operand))
//       return std::optional<std::pair<int64_t, int64_t>>();

//     // Only if the owner for op of the induction variable has constant bound,
//     // the calculation is possible.
//     auto ifOp = getForInductionVarOwner(operand);
//     if (!ifOp.hasConstantBounds())
//       return std::optional<std::pair<int64_t, int64_t>>();

//     auto lb = ifOp.getConstantLowerBound();
//     auto ub = ifOp.getConstantUpperBound();
//     auto step = ifOp.getStep();

//     lbs.push_back(lb);
//     ubs.push_back(ub - 1 - (ub - 1 - lb) % step);
//   }

//   // TODO: maybe a more efficient algorithm.
//   auto operandNum = bound.getNumOperands();
//   SmallVector<int64_t, 16> results;
//   for (unsigned i = 0, e = pow(2, operandNum); i < e; ++i) {
//     SmallVector<AffineExpr, 4> replacements;
//     for (unsigned pos = 0; pos < operandNum; ++pos) {
//       if (i >> pos % 2 == 0)
//         replacements.push_back(getAffineConstantExpr(lbs[pos], context));
//       else
//         replacements.push_back(getAffineConstantExpr(ubs[pos], context));
//     }
//     auto newExpr =
//         bound.getMap().getResult(0).replaceDimsAndSymbols(replacements, {});

//     if (auto constExpr = newExpr.dyn_cast<AffineConstantExpr>())
//       results.push_back(constExpr.getValue());
//     else
//       return std::optional<std::pair<int64_t, int64_t>>();
//   }

//   auto minmax = std::minmax_element(results.begin(), results.end());
//   return std::pair<int64_t, int64_t>(*minmax.first, *minmax.second);
// }

/// Return the layout map of "memrefType".
AffineMap streamhls::getLayoutMap(MemRefType memrefType) {
  // Check whether the memref has layout map.
  auto memrefMaps = memrefType.getLayout();
  if (memrefMaps.getAffineMap().isIdentity())
    return (AffineMap) nullptr;

  return memrefMaps.getAffineMap();
}

bool streamhls::isFullyPartitioned(MemRefType memrefType, int axis) {
  if (memrefType.getRank() == 0)
    return true;

  bool fullyPartitioned = false;
  if (auto layoutMap = getLayoutMap(memrefType)) {
    SmallVector<int64_t, 8> factors;
    getPartitionFactors(memrefType, &factors);

    // Case 1: Use floordiv & mod
    auto shapes = memrefType.getShape();
    if (axis == -1) // all the dimensions
      fullyPartitioned =
          factors == SmallVector<int64_t, 8>(shapes.begin(), shapes.end());
    else
      fullyPartitioned = factors[axis] == shapes[axis];

    // Case 2: Partition index is an identity function
    if (axis == -1) {
      bool flag = true;
      for (int64_t dim = 0; dim < memrefType.getRank(); ++dim) {
        auto expr = layoutMap.getResult(dim);
        if (!expr.isa<AffineDimExpr>()) {
          flag = false;
          break;
        }
      }
      fullyPartitioned |= flag;
    } else {
      auto expr = layoutMap.getResult(axis);
      fullyPartitioned |= expr.isa<AffineDimExpr>();
    }
  }

  return fullyPartitioned;
}

// Calculate partition factors through analyzing the "memrefType" and return
// them in "factors". Meanwhile, the overall partition number is calculated and
// returned as well.
int64_t streamhls::getPartitionFactors(MemRefType memrefType,
                                 SmallVector<int64_t, 8> *factors) {
  auto shape = memrefType.getShape();
  auto layoutMap = getLayoutMap(memrefType);
  int64_t accumFactor = 1;

  for (int64_t dim = 0; dim < memrefType.getRank(); ++dim) {
    int64_t factor = 1;

    if (layoutMap) {
      auto expr = layoutMap.getResult(dim);

      if (auto binaryExpr = expr.dyn_cast<AffineBinaryOpExpr>())
        if (auto rhsExpr = binaryExpr.getRHS().dyn_cast<AffineConstantExpr>()) {
          if (expr.getKind() == AffineExprKind::Mod)
            factor = rhsExpr.getValue();
          else if (expr.getKind() == AffineExprKind::FloorDiv)
            factor = (shape[dim] + rhsExpr.getValue() - 1) / rhsExpr.getValue();
        }
    }

    accumFactor *= factor;
    if (factors != nullptr)
      factors->push_back(factor);
  }

  return accumFactor;
}

/// This is method for finding the number of child loops which immediatedly
/// contained by the input operation.
static unsigned getChildLoopNum(Operation *op) {
  unsigned childNum = 0;
  for (auto &region : op->getRegions())
    for (auto &block : region)
      for (auto &op : block)
        if (isa<AffineForOp>(op))
          ++childNum;

  return childNum;
}

/// Get the whole loop band given the innermost loop and return it in "band".
static void getLoopBandFromInnermost(AffineForOp forOp, AffineLoopBand &band) {
  band.clear();
  AffineLoopBand reverseBand;

  auto currentLoop = forOp;
  while (true) {
    reverseBand.push_back(currentLoop);

    auto parentLoop = currentLoop->getParentOfType<AffineForOp>();
    if (!parentLoop)
      break;

    if (getChildLoopNum(parentLoop) == 1)
      currentLoop = parentLoop;
    else
      break;
  }

  band.append(reverseBand.rbegin(), reverseBand.rend());
}

/// Get the whole loop band given the outermost loop and return it in "band".
/// Meanwhile, the return value is the innermost loop of this loop band.
AffineForOp streamhls::getLoopBandFromOutermost(AffineForOp forOp,
                                          AffineLoopBand &band) {
  band.clear();
  auto currentLoop = forOp;
  while (true) {
    band.push_back(currentLoop);

    if (getChildLoopNum(currentLoop) == 1)
      currentLoop = *currentLoop.getOps<AffineForOp>().begin();
    else
      break;
  }
  return band.back();
}
AffineForOp streamhls::getLoopBandFromInnermost(AffineForOp forOp,
                                               AffineLoopBand &band) {
  band.clear();
  AffineLoopBand reverseBand;

  auto currentLoop = forOp;
  while (true) {
    reverseBand.push_back(currentLoop);

    auto parentLoop = currentLoop->getParentOfType<AffineForOp>();
    if (!parentLoop)
      break;

    if (getChildLoopNum(parentLoop) == 1)
      currentLoop = parentLoop;
    else
      break;
  }

  band.append(reverseBand.rbegin(), reverseBand.rend());
  return band.front();
}
/// Collect all loop bands in the "block" and return them in "bands". If
/// "allowHavingChilds" is true, loop bands containing more than 1 other loop
/// bands are also collected. Otherwise, only loop bands that contains no child
/// loops are collected.
void streamhls::getLoopBands(Block &block, AffineLoopBands &bands,
                       bool allowHavingChilds) {
  bands.clear();
  block.walk([&](AffineForOp loop) {
    auto childNum = getChildLoopNum(loop);

    if (childNum == 0 || (childNum > 1 && allowHavingChilds)) {
      AffineLoopBand band;
      getLoopBandFromInnermost(loop, band);
      bands.push_back(band);
    }
  });
}

void streamhls::getArrays(Block &block, SmallVectorImpl<Value> &arrays,
                    bool allowArguments) {
  // Collect argument arrays.
  if (allowArguments)
    for (auto arg : block.getArguments()) {
      if (arg.getType().isa<MemRefType>())
        arrays.push_back(arg);
    }

  // Collect local arrays.
  for (auto &op : block.getOperations()) {
    if (isa<memref::AllocaOp, memref::AllocOp>(op))
      arrays.push_back(op.getResult(0));
  }
}

// std::optional<unsigned> streamhls::getAverageTripCount(AffineForOp forOp) {
//   if (auto optionalTripCount = getConstantTripCount(forOp))
//     return optionalTripCount.value();
//   else {
//     // TODO: A temporary approach to estimate the trip count. For now, we take
//     // the average of the upper bound and lower bound of trip count as the
//     // estimated trip count.
//     auto lowerBound = getBoundOfAffineBound(forOp.getLowerBound());
//     auto upperBound = getBoundOfAffineBound(forOp.getUpperBound());

//     if (lowerBound && upperBound) {
//       auto lowerTripCount =
//           upperBound.value().second - lowerBound.value().first;
//       auto upperTripCount =
//           upperBound.value().first - lowerBound.value().second;
//       return (lowerTripCount + upperTripCount + 1) / 2;
//     } else
//       return std::optional<unsigned>();
//   }
// }

bool streamhls::checkDependence(Operation *A, Operation *B) {
  return true;
  // TODO: Fix the following
  //   AffineLoopBand commonLoops;
  //   unsigned numCommonLoops = getCommonSurroundingLoops(A, B, &commonLoops);

  //   // Traverse each loop level to find dependencies.
  //   for (unsigned depth = numCommonLoops; depth > 0; depth--) {
  //     // Skip all parallel loop level.
  //     if (auto loopAttr = getLoopDirective(commonLoops[depth - 1]))
  //       if (loopAttr.getParallel())
  //         continue;

  //     FlatAffineValueConstraints depConstrs;
  //     DependenceResult result = checkMemrefAccessDependence(
  //         MemRefAccess(A), MemRefAccess(B), depth, &depConstrs,
  //         /*dependenceComponents=*/nullptr);
  //     if (hasDependence(result))
  //       return true;
  //   }

  //   return false;
}

static bool gatherLoadOpsAndStoreOps(AffineForOp forOp,
                                     SmallVectorImpl<Operation *> &loadOps,
                                     SmallVectorImpl<Operation *> &storeOps) {
  bool hasIfOp = false;
  forOp.walk([&](Operation *op) {
    if (auto load = dyn_cast<AffineReadOpInterface>(op))
      loadOps.push_back(op);
    else if (auto load = dyn_cast<memref::LoadOp>(op))
      loadOps.push_back(op);
    else if (auto store = dyn_cast<AffineWriteOpInterface>(op))
      storeOps.push_back(op);
    else if (auto store = dyn_cast<memref::StoreOp>(op))
      storeOps.push_back(op);
    else if (isa<AffineIfOp>(op))
      hasIfOp = true;
  });
  return !hasIfOp;
}

bool streamhls::analyzeDependency(const AffineForOp &forOpA,
                            const AffineForOp &forOpB,
                            SmallVectorImpl<Dependency> &dependency) {
  SmallVector<Operation *, 4> readOpsA;
  SmallVector<Operation *, 4> writeOpsA;
  SmallVector<Operation *, 4> readOpsB;
  SmallVector<Operation *, 4> writeOpsB;

  if (!gatherLoadOpsAndStoreOps(forOpA, readOpsA, writeOpsA)) {
    return false;
  }

  if (!gatherLoadOpsAndStoreOps(forOpB, readOpsB, writeOpsB)) {
    return false;
  }

  DenseSet<Value> OpAReadMemRefs;
  DenseSet<Value> OpAWriteMemRefs;
  DenseSet<Value> OpBReadMemRefs;
  DenseSet<Value> OpBWriteMemRefs;

  for (Operation *op : readOpsA) {
    if (auto loadOp = dyn_cast<AffineReadOpInterface>(op)) {
      OpAReadMemRefs.insert(loadOp.getMemRef());
    } else if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      OpAReadMemRefs.insert(loadOp.getMemRef());
    }
  }

  for (Operation *op : writeOpsA) {
    if (auto storeOp = dyn_cast<AffineWriteOpInterface>(op)) {
      OpAWriteMemRefs.insert(storeOp.getMemRef());
    } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      OpAWriteMemRefs.insert(storeOp.getMemRef());
    }
  }

  for (Operation *op : readOpsB) {
    if (auto loadOp = dyn_cast<AffineReadOpInterface>(op)) {
      OpBReadMemRefs.insert(loadOp.getMemRef());
    } else if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      OpBReadMemRefs.insert(loadOp.getMemRef());
    }
  }

  for (Operation *op : writeOpsB) {
    if (auto storeOp = dyn_cast<AffineWriteOpInterface>(op)) {
      OpBWriteMemRefs.insert(storeOp.getMemRef());
    } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      OpBWriteMemRefs.insert(storeOp.getMemRef());
    }
  }

  for (Value memref : OpBReadMemRefs) {
    if (OpAWriteMemRefs.count(memref) > 0)
      dependency.push_back(Dependency::RAW);
    else if (OpAReadMemRefs.count(memref) > 0)
      dependency.push_back(Dependency::RAR);
  }

  for (Value memref : OpBWriteMemRefs) {
    if (OpAWriteMemRefs.count(memref) > 0)
      dependency.push_back(Dependency::WAW);
    else if (OpAReadMemRefs.count(memref) > 0)
      dependency.push_back(Dependency::WAR);
  }

  return true;
}

//===----------------------------------------------------------------------===//
// PtrLikeMemRefAccess Struct Definition
//===----------------------------------------------------------------------===//

PtrLikeMemRefAccess::PtrLikeMemRefAccess(Operation *loadOrStoreOpInst) {
  Operation *opInst = nullptr;
  SmallVector<Value, 4> indices;

  if (auto loadOp = dyn_cast<AffineReadOpInterface>(loadOrStoreOpInst)) {
    memref = loadOp.getMemRef();
    opInst = loadOrStoreOpInst;
    auto loadMemrefType = loadOp.getMemRefType();

    indices.reserve(loadMemrefType.getRank());
    for (auto index : loadOp.getMapOperands()) {
      indices.push_back(index);
    }
  } else {
    assert(isa<AffineWriteOpInterface>(loadOrStoreOpInst) &&
           "Affine read/write op expected");
    auto storeOp = cast<AffineWriteOpInterface>(loadOrStoreOpInst);
    opInst = loadOrStoreOpInst;
    memref = storeOp.getMemRef();
    auto storeMemrefType = storeOp.getMemRefType();

    indices.reserve(storeMemrefType.getRank());
    for (auto index : storeOp.getMapOperands()) {
      indices.push_back(index);
    }
  }

  // Get affine map from AffineLoad/Store.
  AffineMap map;
  if (auto loadOp = dyn_cast<AffineReadOpInterface>(opInst))
    map = loadOp.getAffineMap();
  else
    map = cast<AffineWriteOpInterface>(opInst).getAffineMap();

  SmallVector<Value, 8> operands(indices.begin(), indices.end());
  fullyComposeAffineMapAndOperands(&map, &operands);
  map = simplifyAffineMap(map);
  
  canonicalizeMapAndOperands(&map, &operands);

  accessMap.reset(map, operands);
}

bool PtrLikeMemRefAccess::operator==(const PtrLikeMemRefAccess &rhs) const {
  if (memref != rhs.memref || impl != rhs.impl){
    LLVM_DEBUG(llvm::dbgs() << "memref != rhs.memref || impl != rhs.impl\n");
    return false;
  }

  if (impl == rhs.impl && impl && rhs.impl)
    return true;

  AffineValueMap diff;
  AffineValueMap::difference(accessMap, rhs.accessMap, &diff);
  accessMap.getAffineMap().dump();
  rhs.accessMap.getAffineMap().dump();
  diff.getAffineMap().dump();
  return llvm::all_of(diff.getAffineMap().getResults(),
                      [](AffineExpr e) { return e == 0; });
}

// Returns the index of 'op' in its block.
inline static unsigned getBlockIndex(Operation &op) {
  unsigned index = 0;
  for (auto &opX : *op.getBlock()) {
    if (&op == &opX)
      break;
    ++index;
  }
  return index;
}

// Returns a string representation of 'sliceUnion'.
std::string
streamhls::getSliceStr(const mlir::affine::ComputationSliceState &sliceUnion) {
  std::string result;
  llvm::raw_string_ostream os(result);
  // Slice insertion point format [loop-depth, operation-block-index]
  unsigned ipd = mlir::affine::getNestingDepth(&*sliceUnion.insertPoint);
  unsigned ipb = getBlockIndex(*sliceUnion.insertPoint);
  os << "insert point: (" << std::to_string(ipd) << ", " << std::to_string(ipb)
     << ")";
  assert(sliceUnion.lbs.size() == sliceUnion.ubs.size());
  os << " loop bounds: ";
  for (unsigned k = 0, e = sliceUnion.lbs.size(); k < e; ++k) {
    os << '[';
    sliceUnion.lbs[k].print(os);
    os << ", ";
    sliceUnion.ubs[k].print(os);
    os << "] ";
  }
  return os.str();
}

// Value hcl::castInteger(OpBuilder builder, Location loc, Value input,
//                        Type srcType, Type tgtType, bool is_signed) {
//   int oldWidth = srcType.cast<IntegerType>().getWidth();
//   int newWidth = tgtType.cast<IntegerType>().getWidth();
//   Value casted;
//   if (newWidth < oldWidth) {
//     // trunc
//     casted = builder.create<arith::TruncIOp>(loc, tgtType, input);
//   } else if (newWidth > oldWidth) {
//     // extend
//     if (is_signed) {
//       casted = builder.create<arith::ExtSIOp>(loc, tgtType, input);
//     } else {
//       casted = builder.create<arith::ExtUIOp>(loc, tgtType, input);
//     }
//   } else {
//     casted = input;
//   }
//   return casted;
// }

// /* CastIntMemRef
//  * Allocate a new Int MemRef of target width and build a
//  * AffineForOp loop nest to load, cast, store the elements
//  * from oldMemRef to newMemRef.
//  */
// Value hcl::castIntMemRef(OpBuilder &builder, Location loc,
//                          const Value &oldMemRef, size_t newWidth, bool unsign,
//                          bool replace, const Value &dstMemRef) {
//   // If newWidth == oldWidth, no need to cast.
//   if (newWidth == oldMemRef.getType()
//                       .cast<MemRefType>()
//                       .getElementType()
//                       .cast<IntegerType>()
//                       .getWidth()) {
//     return oldMemRef;
//   }
//   // first, alloc new memref
//   MemRefType oldMemRefType = oldMemRef.getType().cast<MemRefType>();
//   Type newElementType = builder.getIntegerType(newWidth);
//   MemRefType newMemRefType =
//       oldMemRefType.clone(newElementType).cast<MemRefType>();
//   Value newMemRef;
//   if (!dstMemRef) {
//     newMemRef = builder.create<memref::AllocOp>(loc, newMemRefType);
//   }
//   // replace all uses
//   if (replace)
//     oldMemRef.replaceAllUsesWith(newMemRef);
//   // build loop nest
//   SmallVector<int64_t, 4> lbs(oldMemRefType.getRank(), 0);
//   SmallVector<int64_t, 4> steps(oldMemRefType.getRank(), 1);
//   size_t oldWidth =
//       oldMemRefType.getElementType().cast<IntegerType>().getWidth();
//   buildAffineLoopNest(
//       builder, loc, lbs, oldMemRefType.getShape(), steps,
//       [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
//         Value v = nestedBuilder.create<AffineLoadOp>(loc, oldMemRef, ivs);
//         Value casted;
//         if (newWidth < oldWidth) {
//           // trunc
//           casted =
//               nestedBuilder.create<arith::TruncIOp>(loc, newElementType, v);
//         } else if (newWidth > oldWidth) {
//           // extend
//           if (unsign) {
//             casted =
//                 nestedBuilder.create<arith::ExtUIOp>(loc, newElementType, v);
//           } else {
//             casted =
//                 nestedBuilder.create<arith::ExtSIOp>(loc, newElementType, v);
//           }
//         } else {
//           casted = v; // no cast happened
//         }
//         if (dstMemRef) {
//           nestedBuilder.create<AffineStoreOp>(loc, casted, dstMemRef, ivs);
//         } else {
//           nestedBuilder.create<AffineStoreOp>(loc, casted, newMemRef, ivs);
//         }
//       });
//   return newMemRef;
// }

// bool mlir::hcl::replace(std::string &str, const std::string &from,
//                         const std::string &to) {
//   size_t start_pos = str.find(from);
//   if (start_pos == std::string::npos)
//     return false;
//   str.replace(start_pos, from.length(), to);
//   return true;
// }

// Value mlir::hcl::castToF64(OpBuilder &rewriter, const Value &src,
//                            bool hasUnsignedAttr) {
//   Type t = src.getType();
//   Type I64 = rewriter.getIntegerType(64);
//   Type F64 = rewriter.getF64Type();
//   Value casted;
//   if (t.isa<IndexType>()) {
//     Type I32 = rewriter.getIntegerType(32);
//     Value intValue =
//         rewriter.create<arith::IndexCastOp>(src.getLoc(), I32, src);
//     return castToF64(rewriter, intValue, hasUnsignedAttr);
//   } else if (t.isa<IntegerType>()) {
//     size_t iwidth = t.getIntOrFloatBitWidth();
//     if (t.isUnsignedInteger() or hasUnsignedAttr) {
//       Value widthAdjusted;
//       if (iwidth < 64) {
//         widthAdjusted = rewriter.create<arith::ExtUIOp>(src.getLoc(), I64, src);
//       } else if (iwidth > 64) {
//         widthAdjusted =
//             rewriter.create<arith::TruncIOp>(src.getLoc(), I64, src);
//       } else {
//         widthAdjusted = src;
//       }
//       casted =
//           rewriter.create<arith::UIToFPOp>(src.getLoc(), F64, widthAdjusted);
//     } else { // signed and signless integer
//       Value widthAdjusted;
//       if (iwidth < 64) {
//         widthAdjusted = rewriter.create<arith::ExtSIOp>(src.getLoc(), I64, src);
//       } else if (iwidth > 64) {
//         widthAdjusted =
//             rewriter.create<arith::TruncIOp>(src.getLoc(), I64, src);
//       } else {
//         widthAdjusted = src;
//       }
//       casted =
//           rewriter.create<arith::SIToFPOp>(src.getLoc(), F64, widthAdjusted);
//     }
//   } else if (t.isa<FloatType>()) {
//     unsigned width = t.cast<FloatType>().getWidth();
//     if (width < 64) {
//       casted = rewriter.create<arith::ExtFOp>(src.getLoc(), F64, src);
//     } else if (width > 64) {
//       casted = rewriter.create<arith::TruncFOp>(src.getLoc(), F64, src);
//     } else {
//       casted = src;
//     }
//   } else if (t.isa<FixedType>()) {
//     unsigned width = t.cast<FixedType>().getWidth();
//     unsigned frac = t.cast<FixedType>().getFrac();
//     Value widthAdjusted;
//     if (width < 64) {
//       widthAdjusted = rewriter.create<arith::ExtSIOp>(src.getLoc(), I64, src);
//     } else if (width > 64) {
//       widthAdjusted = rewriter.create<arith::TruncIOp>(src.getLoc(), I64, src);
//     } else {
//       widthAdjusted = src;
//     }
//     Value srcF64 =
//         rewriter.create<arith::SIToFPOp>(src.getLoc(), F64, widthAdjusted);
//     Value const_frac = rewriter.create<arith::ConstantOp>(
//         src.getLoc(), F64, rewriter.getFloatAttr(F64, std::pow(2, frac)));
//     casted =
//         rewriter.create<arith::DivFOp>(src.getLoc(), F64, srcF64, const_frac);
//   } else if (t.isa<UFixedType>()) {
//     unsigned width = t.cast<UFixedType>().getWidth();
//     unsigned frac = t.cast<UFixedType>().getFrac();
//     Value widthAdjusted;
//     if (width < 64) {
//       widthAdjusted = rewriter.create<arith::ExtUIOp>(src.getLoc(), I64, src);
//     } else if (width > 64) {
//       widthAdjusted = rewriter.create<arith::TruncIOp>(src.getLoc(), I64, src);
//     } else {
//       widthAdjusted = src;
//     }
//     Value srcF64 =
//         rewriter.create<arith::UIToFPOp>(src.getLoc(), F64, widthAdjusted);
//     Value const_frac = rewriter.create<arith::ConstantOp>(
//         src.getLoc(), F64, rewriter.getFloatAttr(F64, std::pow(2, frac)));
//     casted =
//         rewriter.create<arith::DivFOp>(src.getLoc(), F64, srcF64, const_frac);
//   } else {
//     llvm::errs() << src.getLoc() << "could not cast value of type "
//                  << src.getType() << " to F64.\n";
//   }
//   return casted;
// }

// bool mlir::hcl::getEnv(const std::string &key, std::string &value) {
//   char *env = std::getenv(key.c_str());
//   if (env) {
//     value = env;
//     return true;
//   }
//   return false;
// }

// int mlir::hcl::getIndex(SmallVector<Operation *, 4> v, Operation *target) {
//   auto it = std::find(v.begin(), v.end(), target);

//   // If element was found
//   if (it != v.end()) {
//     int index = it - v.begin();
//     return index;
//   } else {
//     // If the element is not
//     // present in the vector
//     return -1;
//   }
// }

// bool chaseAffineApply(Value iv, Value target) {
//   for (auto &use : iv.getUses()) {
//     auto op = use.getOwner();
//     if (dyn_cast<AffineApplyOp>(op)) {
//       if (op->getResult(0) == target) {
//         return true;
//       } else {
//         return chaseAffineApply(op->getResult(0), target);
//       }
//     } else {
//       continue;
//     }
//   }
//   return false;
// };

// Find the which dimension of affine.store the
// loop induction variable operates on.
// e.g.
// for %i = 0; %i < 10; %i++
//   for %j = 0; %j < 10; %j++
//      %ii = affine.apply(%i) #some_map
//      affine.store %some_value %some_memref[%ii, %j]
// If we want to find the memref axis of %some_memref that
// %i operates on, the return result is 0.
// int mlir::streamhls::findMemRefAxisFromIV(AffineStoreOp store, Value iv) {
//   auto memrefRank = store.getMemRef().getType().cast<MemRefType>().getRank();
//   auto indices = store.getIndices();
//   for (int i = 0; i < memrefRank; i++) {
//     if (iv == indices[i]) {
//       // if it is a direct match
//       return i;
//     } else {
//       // try to chase down the affine.apply op
//       // see if any result of the affine.apply op
//       // matches indices[i].
//       // This essentially is a DFS search.
//       if (chaseAffineApply(iv, indices[i])) {
//         return i;
//       }
//     }
//   }
//   return -1;
// }


// ScaleHLS Utils

/// Wrap the operations in the block with dispatch op.
DispatchOp streamhls::dispatchBlock(Block *block) {
  if (!block->getOps<DispatchOp>().empty() ||
      !isa<func::FuncOp, AffineForOp>(block->getParentOp()))
    return DispatchOp();

  OpBuilder builder(block, block->begin());
  ValueRange returnValues(block->getTerminator()->getOperands());
  auto loc = builder.getUnknownLoc();
  auto dispatch = builder.create<DispatchOp>(loc, returnValues);

  auto &dispatchBlock = dispatch.getBody().emplaceBlock();
  builder.setInsertionPointToEnd(&dispatchBlock);
  builder.create<YieldOp>(loc, returnValues);

  auto &dispatchOps = dispatchBlock.getOperations();
  auto &parentOps = block->getOperations();
  dispatchOps.splice(dispatchBlock.begin(), parentOps,
                     std::next(parentOps.begin()), std::prev(parentOps.end()));
  block->getTerminator()->setOperands(dispatch.getResults());
  return dispatch;
}

/// Fuse the given operations into a new task. The new task will be created
/// before the first operation or last operation and each operation will be
/// inserted in order. This method always succeeds even if the resulting IR is
/// invalid.
TaskOp streamhls::fuseOpsIntoTask(ArrayRef<Operation *> ops,
                                 PatternRewriter &rewriter,
                                 bool insertToLastOp) {
  assert(!ops.empty() && "must fuse at least one op");
  llvm::SmallDenseSet<Operation *, 4> opsSet(ops.begin(), ops.end());

  // Collect output values. This is not sufficient and may lead to empty-used
  // outputs, which will be removed during canonicalization.
  llvm::SetVector<Value> outputValues;
  for (auto op : ops)
    for (auto result : op->getResults())
      if (llvm::any_of(result.getUsers(),
                       [&](Operation *user) { return !opsSet.count(user); }))
        outputValues.insert(result);

  // Create new graph task with all inputs and outputs.
  auto loc = rewriter.getUnknownLoc();
  if (!insertToLastOp)
    rewriter.setInsertionPoint(ops.front());
  else
    rewriter.setInsertionPoint(ops.back());
  auto task =
      rewriter.create<TaskOp>(loc, ValueRange(outputValues.getArrayRef()));
  auto taskBlock = rewriter.createBlock(&task.getBody());

  // Move each targeted op into the new graph task.
  rewriter.setInsertionPointToEnd(taskBlock);
  auto yield = rewriter.create<YieldOp>(loc, outputValues.getArrayRef());
  for (auto op : ops)
    op->moveBefore(yield);

  // Replace external output uses with the task results.
  unsigned idx = 0;
  for (auto output : outputValues)
    output.replaceUsesWithIf(task.getResult(idx++), [&](OpOperand &use) {
      return !task->isProperAncestor(use.getOwner());
    });

  // Inline all sub-tasks.
  for (auto subTask : llvm::make_early_inc_range(task.getOps<TaskOp>())) {
    auto &subTaskOps = subTask.getBody().front().getOperations();
    auto &taskOps = task.getBody().front().getOperations();
    taskOps.splice(subTask->getIterator(), subTaskOps, subTaskOps.begin(),
                   std::prev(subTaskOps.end()));
    rewriter.replaceOp(subTask, subTask.getYieldOp()->getOperands());
  }
  return task;
}

//===----------------------------------------------------------------------===//
// Linalg analysis utils
//===----------------------------------------------------------------------===//

bool streamhls::isElementwiseGenericOp(linalg::GenericOp op) {
  // All loops must be parallel loop.
  if (op.getNumParallelLoops() != op.getNumLoops())
    return false;

  for (auto valueMap : llvm::zip(op.getOperands(), op.getIndexingMapsArray())) {
    auto type = std::get<0>(valueMap).getType().dyn_cast<ShapedType>();
    auto map = std::get<1>(valueMap);

    // If the operand doens't have static shape, the index map must be identity.
    if (!type || !type.hasStaticShape()) {
      if (!map.isIdentity())
        return false;
      continue;
    }

    // Otherwise, each dimension must either have a size of one or have identity
    // access index.
    unsigned index = map.getNumDims() - type.getRank();
    for (auto shapeExpr : llvm::zip(type.getShape(), map.getResults())) {
      auto dimSize = std::get<0>(shapeExpr);
      auto expr = std::get<1>(shapeExpr);
      if (expr != getAffineDimExpr(index++, expr.getContext()) && dimSize != 1)
        return false;
    }
  }
  return true;
}

/// Move the loop body of AffineForOp 'src' from 'src' into the specified
/// location in destination's body, ignoring the terminator.
static void moveLoopBodyImpl(AffineForOp src, AffineForOp dest,
                             Block::iterator loc) {
  auto &ops = src.getBody()->getOperations();
  dest.getBody()->getOperations().splice(loc, ops, ops.begin(),
                                         std::prev(ops.end()));
}

/// Move the loop body of AffineForOp 'src' from 'src' to the start of dest
/// body.
static void moveLoopBody(AffineForOp src, AffineForOp dest) {
  moveLoopBodyImpl(src, dest, dest.getBody()->begin());
}

/// Constructs tiled loop nest, without setting the loop bounds and move the
/// body of the original loop nest to the tiled loop nest.
static void constructTiledLoopNest(MutableArrayRef<AffineForOp> origLoops,
                                   AffineForOp rootAffineForOp, unsigned width,
                                   MutableArrayRef<AffineForOp> tiledLoops) {
  Location loc = rootAffineForOp.getLoc();

  // The outermost among the loops as we add more..
  Operation *topLoop = rootAffineForOp.getOperation();
  AffineForOp innermostPointLoop;

  // Add intra-tile (or point) loops.
  for (unsigned i = 0; i < width; i++) {
    OpBuilder b(topLoop);
    // Loop bounds will be set later.
    AffineForOp pointLoop = b.create<AffineForOp>(loc, 0, 0);
    pointLoop.getBody()->getOperations().splice(
        pointLoop.getBody()->begin(), topLoop->getBlock()->getOperations(),
        topLoop);
    tiledLoops[2 * width - 1 - i] = pointLoop;
    topLoop = pointLoop.getOperation();
    if (i == 0)
      innermostPointLoop = pointLoop;
  }

  // Add tile space loops;
  for (unsigned i = width; i < 2 * width; i++) {
    OpBuilder b(topLoop);
    // Loop bounds will be set later.
    AffineForOp tileSpaceLoop = b.create<AffineForOp>(loc, 0, 0);
    tileSpaceLoop.getBody()->getOperations().splice(
        tileSpaceLoop.getBody()->begin(), topLoop->getBlock()->getOperations(),
        topLoop);
    tiledLoops[2 * width - i - 1] = tileSpaceLoop;
    topLoop = tileSpaceLoop.getOperation();
  }

  // Move the loop body of the original nest to the new one.
  moveLoopBody(origLoops.back(), innermostPointLoop);
}

/// Checks whether a loop nest is hyper-rectangular or not.
static LogicalResult
checkIfHyperRectangular(MutableArrayRef<AffineForOp> input) {
  FlatAffineValueConstraints cst;
  SmallVector<Operation *, 8> ops(input.begin(), input.end());
  // 0-d or 1-d is trivially hyper-rectangular.
  if (input.size() <= 1)
    return success();
  if (failed(mlir::affine::getIndexSet(ops, &cst))) {
    LLVM_DEBUG(llvm::dbgs() << "Index set computation failed!\n");
    return failure();
  }
  if (!cst.isHyperRectangular(0, input.size())) {
    LLVM_DEBUG(llvm::dbgs()
               << "Non-hyperrectangular nests not supported for tiling!\n");
    return failure();
  }
  return success();
}

/// Check if the input nest is supported for tiling and whether tiling would be
/// legal or not.
template <typename t>
static LogicalResult performPreTilingChecks(MutableArrayRef<AffineForOp> input,
                                            ArrayRef<t> tileSizes) {
  assert(input.size() == tileSizes.size() && "Too few/many tile sizes");

  if (llvm::any_of(input,
                   [](AffineForOp op) { return op.getNumResults() > 0; })) {
    LLVM_DEBUG(llvm::dbgs()
               << "Cannot tile nest where a loop has yield values\n");
    return failure();
  }

  // Check if the supplied `for` ops are all successively nested.
  if (!isPerfectlyNested(input)) {
    LLVM_DEBUG(llvm::dbgs() << "input loops not perfectly nested");
    return failure();
  }

  //  TODO: handle non hyper-rectangular spaces.
  if (failed(checkIfHyperRectangular(input)))
    return failure();

  return success();
}


/// Constructs and sets new loop bounds after tiling for the case of
/// hyper-rectangular index sets, where the bounds of one dimension do not
/// depend on other dimensions. Bounds of each dimension can thus be treated
/// independently, and deriving the new bounds is much simpler and faster
/// than for the case of tiling arbitrary polyhedral shapes.
static void
constructTiledIndexSetHyperRect(MutableArrayRef<AffineForOp> origLoops,
                                MutableArrayRef<AffineForOp> newLoops,
                                ArrayRef<unsigned> tileSizes) {
  assert(!origLoops.empty());
  assert(origLoops.size() == tileSizes.size());

  OpBuilder b(origLoops[0].getOperation());
  unsigned width = origLoops.size();

  // Bounds for tile space loops.
  for (unsigned i = 0; i < width; i++) {
    OperandRange newLbOperands = origLoops[i].getLowerBoundOperands();
    OperandRange newUbOperands = origLoops[i].getUpperBoundOperands();
    newLoops[i].setLowerBound(newLbOperands, origLoops[i].getLowerBoundMap());
    newLoops[i].setUpperBound(newUbOperands, origLoops[i].getUpperBoundMap());
    // If the step size of original loop is x and tileSize is y then after
    // tiling the tile space loops' step size becomes x*y.
    newLoops[i].setStep(tileSizes[i] * origLoops[i].getStepAsInt());
  }
  // Bounds for intra-tile loops.
  for (unsigned i = 0; i < width; i++) {
    int64_t largestDiv = getLargestDivisorOfTripCount(origLoops[i]);
    std::optional<uint64_t> mayBeConstantCount =
        getConstantTripCount(origLoops[i]);
    // The lower bound is just the tile-space loop.
    AffineMap lbMap = b.getDimIdentityMap();
    newLoops[width + i].setLowerBound(
        /*operands=*/newLoops[i].getInductionVar(), lbMap);
    // The step sizes of intra-tile loops is just the original loops' step size.
    newLoops[width + i].setStep(origLoops[i].getStepAsInt());

    // // Set the upper bound.
    // if (mayBeConstantCount && *mayBeConstantCount < tileSizes[i]) {
    //   // Trip count is less than the tile size: upper bound is lower bound +
    //   // trip count * stepSize.
    //   AffineMap ubMap = b.getSingleDimShiftAffineMap(
    //       *mayBeConstantCount * origLoops[i].getStepAsInt());
    //   newLoops[width + i].setUpperBound(
    //       /*operands=*/newLoops[i].getInductionVar(), ubMap);
    // } else if (largestDiv % tileSizes[i] != 0) {
    //   // Intra-tile loop ii goes from i to min(i + tileSize * stepSize, ub_i).
    //   // Construct the upper bound map; the operands are the original operands
    //   // with 'i' (tile-space loop) appended to it. The new upper bound map is
    //   // the original one with an additional expression i + tileSize * stepSize
    //   // appended.

    //   // Add dim operands from original upper bound.
    //   SmallVector<Value, 4> ubOperands;
    //   AffineBound ub = origLoops[i].getUpperBound();
    //   ubOperands.reserve(ub.getNumOperands() + 1);
    //   AffineMap origUbMap = ub.getMap();
    //   for (unsigned j = 0, e = origUbMap.getNumDims(); j < e; ++j)
    //     ubOperands.push_back(ub.getOperand(j));

    //   // Add dim operand for new loop upper bound.
    //   ubOperands.push_back(newLoops[i].getInductionVar());

    //   // Add symbol operands from original upper bound.
    //   for (unsigned j = 0, e = origUbMap.getNumSymbols(); j < e; ++j)
    //     ubOperands.push_back(ub.getOperand(origUbMap.getNumDims() + j));

    //   SmallVector<AffineExpr, 4> boundExprs;
    //   boundExprs.reserve(1 + origUbMap.getNumResults());
    //   AffineExpr dim = b.getAffineDimExpr(origUbMap.getNumDims());
    //   // The new upper bound map is the original one with an additional
    //   // expression i + tileSize * stepSize (of original loop) appended.
    //   boundExprs.push_back(dim + tileSizes[i] * origLoops[i].getStepAsInt());
    //   boundExprs.append(origUbMap.getResults().begin(),
    //                     origUbMap.getResults().end());
    //   AffineMap ubMap =
    //       AffineMap::get(origUbMap.getNumDims() + 1, origUbMap.getNumSymbols(),
    //                      boundExprs, b.getContext());
    //   newLoops[width + i].setUpperBound(/*operands=*/ubOperands, ubMap);
    // } else {
      // No need of the min expression.
      AffineExpr dim = b.getAffineDimExpr(0);
      AffineMap ubMap = AffineMap::get(
          1, 0, dim + tileSizes[i] * origLoops[i].getStepAsInt());
      newLoops[width + i].setUpperBound(newLoops[i].getInductionVar(), ubMap);
    // }
  }
}

LogicalResult streamhls::tilePerfectlyNested(
  MutableArrayRef<AffineForOp> input,
  ArrayRef<unsigned> tileSizes,
  SmallVectorImpl<AffineForOp> *tiledNest,
  bool allDivisors
) {
  if (input.empty())
    return success();

  if (failed(performPreTilingChecks(input, tileSizes)))
    return failure();

  MutableArrayRef<AffineForOp> origLoops = input;
  AffineForOp rootAffineForOp = origLoops[0];

  // Note that width is at least one since the band isn't empty.
  unsigned width = input.size();
  SmallVector<AffineForOp, 6> tiledLoops(2 * width);

  // Construct a tiled loop nest without setting their bounds. Bounds are
  // set later.
  constructTiledLoopNest(origLoops, rootAffineForOp, width, tiledLoops);

  SmallVector<Value, 8> origLoopIVs;
  extractForInductionVars(input, &origLoopIVs);

  // Set loop bounds for the tiled loop nest.
  constructTiledIndexSetHyperRect(origLoops, tiledLoops, tileSizes);

  // Replace original IVs with intra-tile loop IVs.
  for (unsigned i = 0; i < width; i++)
    origLoopIVs[i].replaceAllUsesWith(tiledLoops[i + width].getInductionVar());

  // Erase the old loop nest.
  rootAffineForOp.erase();

  if (tiledNest)
    *tiledNest = std::move(tiledLoops);

  return success();
}

LogicalResult streamhls::normalizeAffineForDivisorsOnly(
  AffineForOp op, 
  bool promoteSingleIter
) {
  if (promoteSingleIter && succeeded(promoteIfSingleIteration(op)))
    return success();

  // Check if the forop is already normalized.
  if (op.hasConstantLowerBound() && (op.getConstantLowerBound() == 0) &&
      (op.getStep() == 1))
    return success();

  // Check if the lower bound has a single result only. Loops with a max lower
  // bound can't be normalized without additional support like
  // affine.execute_region's. If the lower bound does not have a single result
  // then skip this op.
  if (op.getLowerBoundMap().getNumResults() != 1)
    return failure();

  Location loc = op.getLoc();
  OpBuilder opBuilder(op);
  int64_t origLoopStep = op.getStepAsInt();

  // Construct the new upper bound value map.
  AffineMap oldLbMap = op.getLowerBoundMap();
  // The upper bound can have multiple results. To use
  // AffineValueMap::difference, we need to have the same number of results in
  // both lower and upper bound maps. So, we just create a value map for the
  // lower bound with the only available lower bound result repeated to pad up
  // to the number of upper bound results.
  SmallVector<AffineExpr> lbExprs(op.getUpperBoundMap().getNumResults(),
                                  op.getLowerBoundMap().getResult(0));
  AffineValueMap lbMap(oldLbMap, op.getLowerBoundOperands());
  AffineMap paddedLbMap =
      AffineMap::get(oldLbMap.getNumDims(), oldLbMap.getNumSymbols(), lbExprs,
                     op.getContext());
  AffineValueMap paddedLbValueMap(paddedLbMap, op.getLowerBoundOperands());
  AffineValueMap ubValueMap(op.getUpperBoundMap(), op.getUpperBoundOperands());
  AffineValueMap newUbValueMap;
  // Compute the `upper bound - lower bound`.
  AffineValueMap::difference(ubValueMap, paddedLbValueMap, &newUbValueMap);
  (void)newUbValueMap.canonicalize();

  // Scale down the upper bound value map by the loop step.
  unsigned numResult = newUbValueMap.getNumResults();
  SmallVector<AffineExpr> scaleDownExprs(numResult);
  for (unsigned i = 0; i < numResult; ++i){
    scaleDownExprs[i] = opBuilder.getAffineDimExpr(i).floorDiv(origLoopStep);
  }
  // `scaleDownMap` is (d0, d1, ..., d_n) -> (d0 / step, d1 / step, ..., d_n /
  // step). Where `n` is the number of results in the upper bound map.
  AffineMap scaleDownMap =
      AffineMap::get(numResult, 0, scaleDownExprs, op.getContext());
  AffineMap newUbMap = scaleDownMap.compose(newUbValueMap.getAffineMap());
  // Set the newly create upper bound map and operands.
  op.setUpperBound(newUbValueMap.getOperands(), newUbMap);
  op.setLowerBound({}, opBuilder.getConstantAffineMap(0));
  op.setStep(1);
  // Calculate the Value of new loopIV. Create affine.apply for the value of
  // the loopIV in normalized loop.
  opBuilder.setInsertionPointToStart(op.getBody());
  // Construct an affine.apply op mapping the new IV to the old IV.
  AffineMap scaleIvMap =
      AffineMap::get(1, 0, -opBuilder.getAffineDimExpr(0) * origLoopStep);
  AffineValueMap scaleIvValueMap(scaleIvMap, ValueRange{op.getInductionVar()});
  AffineValueMap newIvToOldIvMap;
  AffineValueMap::difference(lbMap, scaleIvValueMap, &newIvToOldIvMap);
  (void)newIvToOldIvMap.canonicalize();
  auto newIV = opBuilder.create<AffineApplyOp>(
      loc, newIvToOldIvMap.getAffineMap(), newIvToOldIvMap.getOperands());
  op.getInductionVar().replaceAllUsesExcept(newIV->getResult(0), newIV);
  return success();
}
// bool streamhls::dummy() {
//   return true;
// }

/// Helper: build the attribute dictionary {name: <...>, type: <...>}
DictionaryAttr streamhls::makeInfoAttr(MLIRContext *ctx, StringRef name, StringRef type, StringRef io) {
  return DictionaryAttr::get(
      ctx, {NamedAttribute(StringAttr::get(ctx, "name"), StringAttr::get(ctx, name)),
            NamedAttribute(StringAttr::get(ctx, "type"), StringAttr::get(ctx, type)),
            NamedAttribute(StringAttr::get(ctx, "io"), StringAttr::get(ctx, io))});
}

/// Helper: build the attribute dictionary {name: <...>, type: <...>, access_order: <...>}
DictionaryAttr streamhls::makeInfoAttr(MLIRContext *ctx, StringRef name, StringRef type, StringRef io, int64_t accessOrder) {
  return DictionaryAttr::get(
      ctx, {NamedAttribute(StringAttr::get(ctx, "name"), StringAttr::get(ctx, name)),
            NamedAttribute(StringAttr::get(ctx, "type"), StringAttr::get(ctx, type)),
            NamedAttribute(StringAttr::get(ctx, "io"), StringAttr::get(ctx, io)),
            NamedAttribute(StringAttr::get(ctx, "access_order"), IntegerAttr::get(IntegerType::get(ctx, 64), accessOrder))});
}

LogicalResult streamhls::setFifoAttr(
    MLIRContext *ctx,
    Operation *op,
    SmallVectorImpl<int64_t> &shape,
    SmallVectorImpl<Value> &indicesVals) {

  auto i64Ty = IntegerType::get(ctx, 64);

  SmallVector<Attribute> shapeAttrs;
  shapeAttrs.reserve(shape.size());
  for (auto dim : shape)
    shapeAttrs.push_back(IntegerAttr::get(i64Ty, dim));
  auto shapeAttr = ArrayAttr::get(ctx, shapeAttrs);

  // Collect indices you want to store
  SmallVector<unsigned> indices;
  AffineForOp innermostLoop = op->getParentOfType<AffineForOp>();
  assert(innermostLoop && "Expected to be inside an AffineForOp");

  AffineLoopBand band;
  getLoopBandFromInnermost(innermostLoop, band);
  // std::reverse(band.begin(), band.end());

  for(auto loopPair : llvm::enumerate(band)) {
    auto idx = loopPair.index();
    Value iv = loopPair.value().getInductionVar();
    if (std::find(indicesVals.begin(), indicesVals.end(), iv) != indicesVals.end()) {
      indices.push_back(idx);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "indicesVals size: " << indices.size() << "\n");

  // Build "indices" = ArrayAttr of i64
  SmallVector<Attribute> indexAttrs;
  indexAttrs.reserve(indices.size());
  for (unsigned v : indices)
    indexAttrs.push_back(
        IntegerAttr::get(i64Ty, static_cast<int64_t>(v)));
  auto indicesAttr = ArrayAttr::get(ctx, indexAttrs);

  // Inner dictionary:
  // { shape = ..., indices = ... }
  auto fifoInfoDict = DictionaryAttr::get(
      ctx,
      {
          NamedAttribute(StringAttr::get(ctx, "shape"),   shapeAttr),
          NamedAttribute(StringAttr::get(ctx, "indices"), indicesAttr)
      });

  // Final attribute:
  // { fifo-info = { shape = ..., indices = ... } }
  op->setAttr("fifo-info", fifoInfoDict);
  return success();
}

bool streamhls::applyArrayPartition(Value array, SmallVectorImpl<int64_t> &factors,
                                  SmallVectorImpl<dataflow::PartitionKind> &kinds,
                                  bool updateFuncSignature) {
  // if(isa<BlockArgument>(array)){
  //   return false;
  // }
  auto builder = Builder(array.getContext());
  auto arrayType = array.getType().dyn_cast<MemRefType>();
  if (!arrayType || !arrayType.hasStaticShape()){
    return false;
  }

  if(((int64_t)factors.size() == (arrayType.getRank()-1) || (int64_t)kinds.size() != (arrayType.getRank()-1)) &&
      arrayType.getShape()[0] == 1
  ){
    factors.insert(factors.begin(), 1);
    kinds.insert(kinds.begin(), dataflow::PartitionKind::CYCLIC);
  }

  if(arrayType.getRank() != factors.size() || arrayType.getRank() != kinds.size())
    return false;

  // Walk through each dimension of the current memory.
  SmallVector<AffineExpr, 4> partitionIndices;
  SmallVector<AffineExpr, 4> addressIndices;
  // llvm::dbgs() << "Array type: " << arrayType << "\n";
  // llvm::dbgs() << "Factors size: " << factors.size() << "\n";
  // llvm::dbgs() << "Kinds size: " << kinds.size() << "\n";
  for (int64_t dim = 0; dim < arrayType.getRank(); ++dim) {
    auto kind = kinds[dim];
    auto factor = factors[dim];

    if (kind == PartitionKind::CYCLIC) {
      partitionIndices.push_back(builder.getAffineDimExpr(dim) % factor);
      addressIndices.push_back(builder.getAffineDimExpr(dim).floorDiv(factor));

    } else if (kind == PartitionKind::BLOCK) {
      auto blockFactor = (arrayType.getShape()[dim] + factor - 1) / factor;
      partitionIndices.push_back(
          builder.getAffineDimExpr(dim).floorDiv(blockFactor));
      addressIndices.push_back(builder.getAffineDimExpr(dim) % blockFactor);

    } else {
      partitionIndices.push_back(builder.getAffineConstantExpr(0));
      addressIndices.push_back(builder.getAffineDimExpr(dim));
    }
  }

  // Construct new layout map.
  partitionIndices.append(addressIndices.begin(), addressIndices.end());
  auto layoutMap = AffineMap::get(arrayType.getRank(), 0, partitionIndices,
                                  builder.getContext());

  // Construct new array type.
  auto newType =
      MemRefType::get(arrayType.getShape(), arrayType.getElementType(),
                      layoutMap, arrayType.getMemorySpace());

  // Set new type.
  array.setType(newType);

  if (updateFuncSignature)
    if (auto func =
            dyn_cast<func::FuncOp>(array.getParentBlock()->getParentOp())) {
      // Align function type with entry block argument types only if the array
      // is defined as an argument of the function.
      if (!array.getDefiningOp()) {
        auto resultTypes = func.front().getTerminator()->getOperandTypes();
        auto inputTypes = func.front().getArgumentTypes();
        func.setType(builder.getFunctionType(inputTypes, resultTypes));
      }
    }
  return true;
}

/// Build a partitioning layout affine map for a given memref type.
/// - `factors.size()` and `kinds.size()` must equal `arrayType.getRank()`.
/// - No mutation of input vectors; no function signature updates.
static FailureOr<AffineMap>
buildPartitionLayoutMap(MemRefType arrayType,
                        ArrayRef<int64_t> factors,
                        ArrayRef<dataflow::PartitionKind> kinds,
                        MLIRContext *ctx) {
  if (!arrayType || !arrayType.hasStaticShape())
    return failure();

  if (arrayType.getRank() != static_cast<int64_t>(factors.size()) ||
      arrayType.getRank() != static_cast<int64_t>(kinds.size()))
    return failure();

  Builder builder(ctx);

  SmallVector<AffineExpr, 4> partitionIndices;
  SmallVector<AffineExpr, 4> addressIndices;

  for (int64_t dim = 0; dim < arrayType.getRank(); ++dim) {
    auto kind = kinds[dim];
    auto factor = factors[dim];

    // Sanity: non-positive factors don't make sense.
    if (factor <= 0)
      return failure();

    auto d = builder.getAffineDimExpr(dim);
    switch (kind) {
    case dataflow::PartitionKind::CYCLIC:
      partitionIndices.push_back(d % factor);
      addressIndices.push_back(d.floorDiv(factor));
      break;

    case dataflow::PartitionKind::BLOCK: {
      auto dimSize = arrayType.getShape()[dim];
      auto blockFactor = (dimSize + factor - 1) / factor; // ceil-div
      partitionIndices.push_back(d.floorDiv(blockFactor));
      addressIndices.push_back(d % blockFactor);
      break;
    }

    case dataflow::PartitionKind::NONE:
    default:
      partitionIndices.push_back(builder.getAffineConstantExpr(0));
      addressIndices.push_back(d);
      break;
    }
  }

  // [partition dims..., address dims...]
  partitionIndices.append(addressIndices.begin(), addressIndices.end());

  AffineMap layoutMap =
      AffineMap::get(arrayType.getRank(), /*numSymbols=*/0, partitionIndices,
                     ctx);
  return layoutMap;
}

bool streamhls::applyArrayPartition(Value array,
                                    ArrayRef<int64_t> factors,
                                    ArrayRef<dataflow::PartitionKind> kinds) {
  auto arrayType = array.getType().dyn_cast<MemRefType>();
  if (!arrayType || !arrayType.hasStaticShape())
    return false;

  auto layoutOr = buildPartitionLayoutMap(arrayType, factors, kinds,
                                          array.getContext());
  if (failed(layoutOr))
    return false;

  auto layoutMap = *layoutOr;

  auto newType = MemRefType::get(arrayType.getShape(),
                                 arrayType.getElementType(),
                                 layoutMap,
                                 arrayType.getMemorySpace());
  array.setType(newType);
  return true;
}
