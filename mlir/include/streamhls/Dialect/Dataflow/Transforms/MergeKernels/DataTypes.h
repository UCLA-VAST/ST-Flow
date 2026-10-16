// #include "PassDetail.h"

#pragma once
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
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/SmallVector.h"

#include "streamhls/Dialect/Dataflow/Transforms/Passes.h"
#include "streamhls/Dialect/Dataflow/Support/Utils.h"
#include "streamhls/Dialect/Dataflow/Support/PerformanceModel.h"
#include "streamhls/Dialect/Dataflow/Support/AffineAnalysis.h"
#include <math.h>
#include <regex>

using namespace mlir;
using namespace streamhls;

struct TilingSolution{
  SmallVector<uint, 4> l1TilingFactors;
  SmallVector<uint, 4> l2TilingFactors;
};

struct MemRefSolution{
  Value memref;

  SmallVector<int64_t, 4> sizes;
  AffineMap map;
  SmallVector<Value> mapOperands;

  SmallVector<int64_t, 4> fifoSizes;
  AffineMap fifoMap;
  SmallVector<Value> fifoMapOperands;

  IntegerSet loadSet;
  SmallVector<Value> loadSetOperands;
  IntegerSet storeSet;
  SmallVector<Value> storeSetOperands;
  bool isStore, isLoad;
};

using MemrefSolutions = SmallVector<MemRefSolution, 4>;
