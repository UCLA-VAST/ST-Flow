#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "streamhls/Dialect/Dataflow/Transforms/Passes.h"


// include filesystem
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
using namespace mlir;
using namespace streamhls;

#define DEBUG_TYPE "streamhls-pass-pipelines"


namespace {
#define GEN_PASS_REGISTRATION
#include "streamhls/Dialect/Dataflow/Transforms/Passes.h.inc"
} // namespace


ModuleOp streamhls::lowerLinalgToAffine(ModuleOp &module, unsigned debugPoint) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::dataflow::DataflowDialect, mlir::func::FuncDialect,
                  mlir::arith::ArithDialect, mlir::tensor::TensorDialect,
                  mlir::affine::AffineDialect, mlir::math::MathDialect,
                  mlir::memref::MemRefDialect>();
  PassManager pm = PassManager::on<ModuleOp>(module.getContext());

  pm.getDependentDialects(registry); // check if this is necessary

  pm.addPass(streamhls::createRemoveRedundantOpsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  if(debugPoint == 1){
    if (failed(pm.run(module)))
      assert(false && "DBGS[1] Lowering Linalg to Affine failed after removing redundant ops");
    return module;
  }

  // Linalg optimization.
  pm.addPass(mlir::createLinalgElementwiseOpFusionPass()); // fuse layers of dnn
  pm.addPass(mlir::createConvertTensorToLinalgPass());
  pm.addPass(mlir::createCanonicalizerPass());
  if(debugPoint == 2){
    if (failed(pm.run(module)))
      assert(false && "DBGS[2] Lowering Linalg to Affine failed after Linalg optimization");
    return module;
  }

  // *** NEW: kill tensor.concat and simplify tensor ops ***
  pm.nest<func::FuncOp>().addPass(streamhls::createLowerTensorConcatToSlicesPass());
  // Optional, if available in your tree:
  // pm.addPass(tensor::createFoldTensorEmptyOpsPass());
  // pm.addPass(tensor::createTensorSimplifyPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Bufferization.
  pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());
  bufferization::OneShotBufferizationOptions oneShotBuffOptions;
  oneShotBuffOptions.copyBeforeWrite = true;
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(
    oneShotBuffOptions
  ));
  pm.addPass(func::createFuncBufferizePass());
  pm.addPass(bufferization::createBufferResultsToOutParamsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  if(debugPoint == 3){
    if (failed(pm.run(module)))
      assert(false && "DBGS[3] Lowering Linalg to Affine failed after Bufferization");
    return module;
  }

  // Linalg to Affine conversion.
  pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
  pm.addPass(memref::createFoldMemRefAliasOpsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  if(debugPoint == 4){
    if (failed(pm.run(module)))
      assert(false && "DBGS[4] Lowering Linalg to Affine failed after Linalg to Affine conversion");
    return module;
  }

  pm.nest<func::FuncOp>().addPass(streamhls::createLowerCopyToAffinePass());
  pm.addPass(memref::createFoldMemRefAliasOpsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  if(debugPoint == 5){
    if (failed(pm.run(module)))
      assert(false && "DBGS[5] Lowering Linalg to Affine failed after Lower Copy to Affine");
    return module;
  }

  pm.nest<func::FuncOp>().addPass(streamhls::createRemoveRedundantMemRefsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  if(debugPoint == 6){
    if (failed(pm.run(module)))
      assert(false && "DBGS[6] Lowering Linalg to Affine failed after Remove Redundant MemRefs");
    return module;
  }

  pm.addPass(streamhls::createCollapseUnitDimsPass());
  pm.addPass(mlir::createCanonicalizerPass());


  pm.nest<func::FuncOp>().addPass(streamhls::createRemoveLoopsOfUnitIterPass());
  pm.addPass(mlir::createCanonicalizerPass());

  if (failed(pm.run(module)))
    assert(false && "Lowering Linalg to Affine failed after full pipeline");

  return module;
}

ModuleOp streamhls::eleminateUnitDimsAndIters(ModuleOp &module) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::dataflow::DataflowDialect, mlir::func::FuncDialect,
                  mlir::arith::ArithDialect, mlir::tensor::TensorDialect,
                  mlir::affine::AffineDialect, mlir::math::MathDialect,
                  mlir::memref::MemRefDialect>();
  PassManager pm = PassManager::on<ModuleOp>(module.getContext());

  pm.getDependentDialects(registry); // check if this is necessary

  pm.addPass(streamhls::createCollapseUnitDimsPass());
  pm.addPass(mlir::createCanonicalizerPass());

  pm.nest<func::FuncOp>().addPass(streamhls::createRemoveLoopsOfUnitIterPass());
  pm.addPass(mlir::createCanonicalizerPass());

  if (failed(pm.run(module)))
    assert(false && "Elimination of Unit Dims and Iters failed");

  return module;
}

void streamhls::registerTransformsPasses(){
  registerStreamHLSPasses();
}