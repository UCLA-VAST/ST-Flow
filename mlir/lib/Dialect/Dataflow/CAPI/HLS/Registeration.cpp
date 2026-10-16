

 #include "streamhls-c/Dialect/Registeration.h"
 #include "streamhls/Dialect/Dataflow/Transforms/Passes.h"
 
 #include "mlir/Conversion/Passes.h"
 #include "mlir/Dialect/Affine/Passes.h"
 #include "mlir/Dialect/Arith/Transforms/Passes.h"
 #include "mlir/Dialect/Func/Transforms/Passes.h"
 #include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
 #include "mlir/Dialect/Linalg/Passes.h"
 #include "mlir/Dialect/MemRef/Transforms/Passes.h"
 #include "mlir/Transforms/Passes.h"
 
 #include "streamhls/InitAllPasses.h"
 #include "streamhls/Dialect/Dataflow/IR/Dataflow.h"

 #include "mlir/InitAllDialects.h"
 
 void streamhlsMlirRegisterAllDialects(MlirContext context) {
   mlir::DialectRegistry registry;
   registry.insert<mlir::dataflow::DataflowDialect, mlir::func::FuncDialect,
                   mlir::arith::ArithDialect, mlir::tensor::TensorDialect,
                   mlir::affine::AffineDialect, mlir::math::MathDialect,
                   mlir::memref::MemRefDialect, mlir::pdl::PDLDialect,
                   mlir::transform::TransformDialect>();
   unwrap(context)->appendDialectRegistry(registry);
   unwrap(context)->loadAllAvailableDialects();
 }
 
 void streamhlsMlirRegisterAllPasses() {
  //  // General passes
  //  mlir::registerTransformsPasses();
 
  //  // Conversion passes
  //  mlir::registerConversionPasses();
 
  //  // Dialect passes
  //  mlir::affine::registerAffinePasses();
  //  mlir::arith::registerArithPasses();
  //  mlir::LLVM::registerLLVMPasses();
  //  mlir::memref::registerMemRefPasses();
  //  mlir::registerLinalgPasses();
 
   mlir::streamhls::registerAllPasses();
 }