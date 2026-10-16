//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Adapted from ScaleHLS (https://github.com/UIUC-ChenLab/scalehls) and modified
// for Stream-HLS and ST-Flow. See NOTICE.
//
//===----------------------------------------------------------------------===//

#ifndef STREAMHLS_DIALECT_DATAFLOW_H
#define STREAMHLS_DIALECT_DATAFLOW_H

#include "mlir/IR/Dialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Analysis/Liveness.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "streamhls/Dialect/Dataflow/IR/DataflowDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "streamhls/Dialect/Dataflow/IR/DataflowTypes.h.inc"

#include "streamhls/Dialect/Dataflow/IR/DataflowEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "streamhls/Dialect/Dataflow/IR/DataflowAttributes.h.inc"

namespace mlir {
namespace dataflow {

#include "streamhls/Dialect/Dataflow/IR/DataflowInterfaces.h.inc"

enum class PartitionKind { CYCLIC, BLOCK, NONE };
enum class OperandKind { INPUT, OUTPUT, PARAM };
enum class MemoryKind {
  LUTRAM_1P = 1,
  LUTRAM_2P,
  LUTRAM_S2P,
  BRAM_1P,
  BRAM_2P,
  BRAM_S2P,
  BRAM_T2P,
  URAM_1P,
  URAM_2P,
  URAM_S2P,
  URAM_T2P,
  DRAM
};

bool isRam1P(MemoryKind kind);
bool isRam2P(MemoryKind kind);
bool isRamS2P(MemoryKind kind);
bool isRamT2P(MemoryKind kind);
bool isDram(MemoryKind kind);

namespace StreamEffects {
struct Instantiate : public MemoryEffects::Effect::Base<Instantiate> {};
struct Push : public MemoryEffects::Effect::Base<Push> {};
struct Pop : public MemoryEffects::Effect::Base<Pop> {};
} // namespace StreamEffects

/// Timing attribute utils.
TimingAttr getTiming(Operation *op);
void setTiming(Operation *op, TimingAttr timing);
void setTiming(Operation *op, int64_t begin, int64_t end, int64_t latency,
               int64_t interval);

/// Resource attribute utils.
ResourceAttr getResource(Operation *op);
void setResource(Operation *op, ResourceAttr resource);
void setResource(Operation *op, int64_t lut, int64_t dsp, int64_t bram);

/// Loop information attribute utils.
LoopInfoAttr getLoopInfo(Operation *op);
void setLoopInfo(Operation *op, LoopInfoAttr loopInfo);
void setLoopInfo(Operation *op, int64_t flattenTripCount, int64_t iterLatency,
                 int64_t minII);

/// Loop directives attribute utils.
LoopDirectiveAttr getLoopDirective(Operation *op);
void setLoopDirective(Operation *op, LoopDirectiveAttr loopDirective);
void setLoopDirective(Operation *op, bool pipeline, int64_t targetII,
                      bool dataflow, bool flatten);

/// Parrallel and Reduction loop attribute utils.
bool hasParallelAttr(Operation *op);
void setParallelAttr(Operation *op);
bool hasReductionAttr(Operation *op);
void setReductionAttr(Operation *op);
void setPEAttr(Operation *op);
bool hasPEAttr(Operation *op);
void setColAttr(Operation *op);
bool hasColAttr(Operation *op);
void setRowAttr(Operation *op);
bool hasRowAttr(Operation *op);
void setNamedAttr(Operation *op, StringRef name);
bool hasNamedAttr(Operation *op, StringRef name);
void setLoopIDAttr(Operation *op, int64_t id);
void setLoopLevelAttr(Operation *op, StringRef level);
StringRef getLoopLevelAttr(Operation *op);
int64_t getLoopIDAttr(Operation *op);
void setLoopTypeAttr(Operation *op, StringRef type);
StringRef getLoopTypeAttr(Operation *op);
bool hasLoopTypeAttr(Operation *op);
bool isReductionLoop(Operation *op);
bool isParallelLoop(Operation *op);
bool isReductionLoopBand(SmallVector<affine::AffineForOp, 6> band);
bool isParallelLoopBand(SmallVector<affine::AffineForOp, 6> band);
void setOpLoopAttr(Operation *op);
ArrayAttr getOpLoopAttr(Operation *op);
void setMemrefPortWidthAttr(Operation* op, unsigned width);
unsigned getMemrefPortWidthAttr(Operation* op);
bool hasMemrefPortWidthAttr(Operation* op);

/// Function directives attribute utils.
FuncDirectiveAttr getFuncDirective(Operation *op);
void setFuncDirective(Operation *op, FuncDirectiveAttr FuncDirective);
void setFuncDirective(Operation *op, bool pipeline, int64_t targetInterval,
                      bool dataflow);

/// Top and runtime function attribute utils.
bool hasTopFuncAttr(Operation *op);
void setTopFuncAttr(Operation *op);
bool hasRuntimeAttr(Operation *op);
void setRuntimeAttr(Operation *op);

// --------------------------------------------------------------------------
// dataflow.port_width<N>
// Represents number of vector lanes for a top-level port, used when
// converting to !dataflow.axivec<elem, N>.
// C++: ::mlir::dataflow::PortWidthAttr
// --------------------------------------------------------------------------

/// Attribute dictionary key.
extern const llvm::StringLiteral kPortWidthAttrName;

/// ------------------------------
/// Operation-level APIs
/// ------------------------------

/// Sets `dataflow.port_width` on an operation.
void setPortWidthAttr(mlir::Operation *op, unsigned lanes);

/// Gets `dataflow.port_width` on an operation (null if absent).
mlir::dataflow::PortWidthAttr getPortWidthAttr(mlir::Operation *op);

/// Returns true if operation has `dataflow.port_width`.
bool hasPortWidthAttr(mlir::Operation *op);


/// ------------------------------
/// Func argument-level APIs (by index)
/// ------------------------------

/// Sets `dataflow.port_width` on the argIndex-th argument of a func.
void setPortWidthAttr(mlir::func::FuncOp func, unsigned argIndex,
                      unsigned lanes);

/// Gets `dataflow.port_width` from the argIndex-th argument of a func.
mlir::dataflow::PortWidthAttr getPortWidthAttr(mlir::func::FuncOp func,
                                               unsigned argIndex);

/// Returns true if the argIndex-th argument has `dataflow.port_width`.
bool hasPortWidthAttr(mlir::func::FuncOp func, unsigned argIndex);


/// ------------------------------
/// BlockArgument APIs (optional)
/// ------------------------------

/// Sets port width on a BlockArgument of a func.
void setPortWidthAttr(mlir::BlockArgument arg, unsigned lanes);

/// Gets port width from a BlockArgument (null if absent).
mlir::dataflow::PortWidthAttr getPortWidthAttr(mlir::BlockArgument arg);

/// Returns true if the BlockArgument has port width.
bool hasPortWidthAttr(mlir::BlockArgument arg);


} // namespace dataflow
} // namespace mlir


#define GET_OP_CLASSES
#include "streamhls/Dialect/Dataflow/IR/Dataflow.h.inc"

#endif // STREAMHLS_DIALECT_DATAFLOW_H

