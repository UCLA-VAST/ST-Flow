//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Adapted from ScaleHLS (https://github.com/UIUC-ChenLab/scalehls) and modified
// for Stream-HLS and ST-Flow. See NOTICE.
//
//===----------------------------------------------------------------------===//

#ifndef STREAMHLS_INITALLDIALECTS_H
#define STREAMHLS_INITALLDIALECTS_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/AllInterfaces.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "streamhls/Dialect/Dataflow/IR/Dataflow.h"


namespace mlir {
namespace streamhls {

// Add all the STREAMHLS dialects to the provided registry.
inline void registerAllDialects(mlir::DialectRegistry &registry) {
  // clang-format off
  registry.insert<
    mlir::dataflow::DataflowDialect,
    mlir::func::FuncDialect,
    mlir::tosa::TosaDialect,
    mlir::tensor::TensorDialect,
    mlir::linalg::LinalgDialect,
    mlir::memref::MemRefDialect,
    mlir::bufferization::BufferizationDialect,
    mlir::affine::AffineDialect,
    mlir::math::MathDialect,
    mlir::arith::ArithDialect,
    mlir::vector::VectorDialect,
    mlir::scf::SCFDialect,
    mlir::LLVM::LLVMDialect,
    mlir::DLTIDialect,
    mlir::ml_program::MLProgramDialect,
    mlir::cf::ControlFlowDialect
  >();
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerAllDialectInterfaceImplementations(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  // clang-format on
}

} // namespace streamhls
} // namespace mlir

#endif // STREAMHLS_INITALLDIALECTS_H
