//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Adapted from ScaleHLS (https://github.com/UIUC-ChenLab/scalehls) and modified
// for Stream-HLS and ST-Flow. See NOTICE.
//
//===----------------------------------------------------------------------===//

#ifndef STREAMHLS_DIALECT_DATAFLOW_TD
#define STREAMHLS_DIALECT_DATAFLOW_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"


//===----------------------------------------------------------------------===//
// dataflow dialect definition.
//===----------------------------------------------------------------------===//

def DataflowDialect : Dialect {
  let name = "dataflow";
  let summary = "An HLS out-of-tree MLIR dialect";
  let description = [{
    This dialect contains dedicated operations, types, and attributes designed
    for representing HLS C++ specific structures and components in MLIR, while
    enabling comprehensive optimization for both performance and area.
  }];
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let cppNamespace = "::mlir::dataflow";
}

//===----------------------------------------------------------------------===//
// Base dataflow type, attribute, and operation definitions
//===----------------------------------------------------------------------===//

class DataflowType<string mnemonic, list<Trait> traits = []> : 
    TypeDef<DataflowDialect, mnemonic, traits>;

class DataflowAttr<string mnemonic, list<Trait> traits = []> : 
    AttrDef<DataflowDialect, mnemonic, traits>;

class DataflowOp<string mnemonic, list<Trait> traits = []> :
    Op<DataflowDialect, mnemonic, traits>;

include "streamhls/Dialect/Dataflow/IR/Attributes.td"
include "streamhls/Dialect/Dataflow/IR/Types.td"
include "streamhls/Dialect/Dataflow/IR/Ops.td"

#endif // STREAMHLS_DIALECT_DATAFLOW_TD
