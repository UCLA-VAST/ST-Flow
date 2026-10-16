//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Adapted from ScaleHLS (https://github.com/UIUC-ChenLab/scalehls) and modified
// for Stream-HLS and ST-Flow. See NOTICE.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_OPS
#define DATAFLOW_OPS

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.td"

include "mlir/IR/SymbolInterfaces.td"
include "streamhls/Dialect/Dataflow/IR/Interfaces.td"
include "mlir/IR/OpAsmInterface.td"

class StreamOf<list<Type> allowedTypes> : Type<And<[
    CPred<"$_self.isa<mlir::dataflow::StreamType>()">,
    Concat<"[](mlir::Type elementType) { return ",
      SubstLeaves<"$_self", "elementType", AnyTypeOf<allowedTypes>.predicate>,
      "; }($_self.cast<mlir::dataflow::StreamType>().getElementType())">
  ]>,
  "stream of " # AnyTypeOf<allowedTypes>.summary # " values",
  "mlir::dataflow::StreamType"
>;

class XRTBufferOf<list<Type> allowedTypes> : Type<And<[
    CPred<"$_self.isa<mlir::dataflow::XRTBufferType>()">,
    Concat<"[](mlir::Type elementType) { return ",
      SubstLeaves<"$_self", "elementType", AnyTypeOf<allowedTypes>.predicate>,
      "; }($_self.cast<mlir::dataflow::XRTBufferType>().getElementType())">
  ]>,
  "stream of " # AnyTypeOf<allowedTypes>.summary # " values",
  "mlir::dataflow::XRTBufferType"
>;

def AnyStream : StreamOf<[AnyType]>;

def ArrayOfStreams : RankedTensorOf<[StreamOf<[AnyType]>]>;

//===----------------------------------------------------------------------===//
// Stream Ops
//===----------------------------------------------------------------------===//

//===---------------------------------------===//
// single stream ops
//===---------------------------------------===//


def StreamOp : DataflowOp<"stream", [
      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Declare a stream channel";

  let arguments = (ins I32Attr:$depth);
  let results = (outs StreamOf<[AnyType]>:$channel);
  let assemblyFormat = "attr-dict `:` type($channel)";

  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins "mlir::Type":$channel, "int32_t":$depth),
        "build($_builder, $_state, channel, depth);">,
    OpBuilder<(ins "mlir::Type":$channel),
        "build($_builder, $_state, channel, 1);">
  ];
  // let extraClassDeclaration = [{
  //   // Returns the Tag MemRef for this DMA operation.
  //   Value getChannel() {
  //     return getOperand(0);
  //   }

  // }];
}

def StreamReadOp : DataflowOp<"stream_read", []> {
  let summary = "Read a stream channel";
  let description = [{
    Read/pop a value from a stream channel. The absence of the result indicates
    the popped value is dropped. Each stream channel can be read multiple times.
  }];

  let arguments = (ins StreamOf<[AnyType]>:$channel);
  let results = (outs Optional<AnyType>:$result);
  let assemblyFormat = [{
    $channel attr-dict `:` functional-type($channel, $result)
  }];

  let hasVerifier = 1;
}

def StreamWriteOp : DataflowOp<"stream_write", []> {
  let summary = "Write a stream channel";
  let description = [{
    Write/push a value to a stream channel. Each stream channel can only be
    written once.
  }];

  let arguments = (ins StreamOf<[AnyType]>:$channel, AnyType:$value);
  let assemblyFormat = [{
    $channel `,` $value attr-dict `:` type($channel) `,` type($value)
  }];

  let hasVerifier = 1;
}

//===---------------------------------------===//
// array of streams ops
//===---------------------------------------===//

def ArrayOfStreamsOp : DataflowOp<"aofs", []> { // may need to have memory effects
  let summary = "Declare an array of stream channels";
  let description = [{
    Declare an array of stream channels
  }];
  let arguments = (ins I32Attr:$depth);
  let results = (outs ArrayOfStreams:$channels);
  let builders = [
    OpBuilder<(ins "mlir::Type":$channels, "int32_t":$depth),
        "build($_builder, $_state, channels, depth);">,
    OpBuilder<(ins "mlir::Type":$channels),
        "build($_builder, $_state, channels, 1);">
  ];
  code extraClassDeclaration = [{
    /// Returns fifo element type
    Type getElementType() { return getChannels().getType().cast<RankedTensorType>().getElementType().cast<StreamType>().getElementType();}
  }];
  let assemblyFormat = "attr-dict `:` type($channels)";
}


def ArrayOfStreamsReadOp : DataflowOp<"aofs_read", [
  DeclareOpInterfaceMethods<AffineReadOpInterface>,
  DeclareOpInterfaceMethods<AffineMapAccessInterface>,
  MemRefsNormalizable
  ]>{
  let summary = "Read an array of stream channels";
  let description = [{
    Read/pop a value from an array of stream channels. The absence of the result indicates
    the popped value is dropped. Each stream channel can be read multiple times.
  }];

  let arguments = (ins ArrayOfStreams:$channels, Variadic<Index>:$indices, AffineMapAttr:$map);
  let results = (outs AnyType:$result);
  code extraClassDeclaration = [{
    /// Returns the operand index of the memref.
    unsigned getMemRefOperandIndex() { return 0; }

    void setMemRef(Value value) { setOperand(getMemRefOperandIndex(), value); }

    /// Returns the affine map used to index the memref for this operation.
    AffineMapAttr getAffineMapAttr() {
      return getProperties().map;
    }

    static StringRef getMapAttrStrName() { return "map"; }
  }];
  let assemblyFormat = [{
    $channels `[` $indices `]` attr-dict `:` functional-type($channels, $result)
  }];
}

def ArrayOfStreamsWriteOp : DataflowOp<"aofs_write", [
  DeclareOpInterfaceMethods<AffineWriteOpInterface>,
  DeclareOpInterfaceMethods<AffineMapAccessInterface>,
  MemRefsNormalizable
  ]>{
  let summary = "Write an array of stream channels";
  let description = [{
    Write/push a value to an array of stream channels. Each stream channel can only be
    written once.
  }];

  let arguments = (ins ArrayOfStreams:$channels, AnyType:$value, Variadic<Index>:$indices, AffineMapAttr:$map);
  // code extraClassDeclaration = [{
  //   /// Returns the operand index of the value to be stored.
  //   unsigned getStoredValOperandIndex() { return 0; }

  //   /// Returns the operand index of the memref.
  //   unsigned getMemRefOperandIndex() { return 1; }

  //   void setMemRef(Value value) { setOperand(getMemRefOperandIndex(), value); }

  //   /// get value to be stored
  //   Value getStoredValue() { return getOperand(getMemRefOperandIndex()); }

  //   /// Returns the affine map used to index the memref for this operation.
  //   AffineMapAttr getAffineMapAttr() {
  //     return getProperties().map;
  //   }

  //   static StringRef getMapAttrStrName() { return "map"; }
  // }];
    code extraClassDeclaration = [{
      /// Returns the operand index of the value to be stored.
      unsigned getStoredValOperandIndex() { return 1; }

      /// Returns the operand index of the "memref" (here: channels).
      unsigned getMemRefOperandIndex() { return 0; }

      void setMemRef(Value value) { setOperand(getMemRefOperandIndex(), value); }

      /// get value to be stored
      Value getStoredValue() { return getOperand(getStoredValOperandIndex()); }

      AffineMapAttr getAffineMapAttr() {
        return getProperties().map;
      }

      static StringRef getMapAttrStrName() { return "map"; }
  }];

  let assemblyFormat = [{
    $channels `[` $indices `]` `,` $value attr-dict `:` type($channels) `,` type($value)
  }];
}

//===----------------------------------------------------------------------===//
// Vector Ops
//===----------------------------------------------------------------------===//

def EmptyVectorOp : DataflowOp<"empty_vector"> {
  let summary = "Declare an empty vector";

  // let arguments = (ins Index:$size);
  let results = (outs AnyVector:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let builders = [
    OpBuilder<(ins "int64_t":$size, "Type":$elementType)>
  ];
}

def VectorAffineLoadOp : DataflowOp<"vector_load", [
    DeclareOpInterfaceMethods<AffineReadOpInterface>,
    DeclareOpInterfaceMethods<AffineMapAccessInterface>,
    MemRefsNormalizable
]>{
  let summary = "Load a value from a vector";
  let description = [{
    Load a value from a vector using affine maps. 
  }];

  let arguments = (ins AnyVector:$vector, Variadic<Index>:$indices, AffineMapAttr:$map);
  let results = (outs AnyType:$result);
  code extraClassDeclaration = [{
    /// Returns the operand index of the memref.
    unsigned getMemRefOperandIndex() { return 0; }

    void setMemRef(Value value) { setOperand(getMemRefOperandIndex(), value); }

    /// Returns the affine map used to index the memref for this operation.
    AffineMapAttr getAffineMapAttr() {
      return getProperties().map;
    }

    static StringRef getMapAttrStrName() { return "map"; }
  }];
  let assemblyFormat = [{
    $vector `[` $indices `]` attr-dict `:` functional-type($vector, $result)
  }];
}

def VectorAffineStoreOp : DataflowOp<"vector_store", [
  DeclareOpInterfaceMethods<AffineWriteOpInterface>,
  DeclareOpInterfaceMethods<AffineMapAccessInterface>,
  MemRefsNormalizable
  ]>{
  let summary = "Store a value to a vector";
  let description = [{
    Store a value to a vector using affine maps.
  }];
  let arguments = (ins AnyVector:$vector, AnyType:$value, Variadic<Index>:$indices, AffineMapAttr:$map);
  code extraClassDeclaration = [{
    /// Returns the operand index of the value to be stored.
    unsigned getStoredValOperandIndex() { return 1; }

    /// Returns the operand index of the memref.
    unsigned getMemRefOperandIndex() { return 0; }

    void setMemRef(Value value) { setOperand(getMemRefOperandIndex(), value); }

    /// get value to be stored
    Value getStoredValue() { return getOperand(getMemRefOperandIndex()); }

    /// Returns the affine map used to index the memref for this operation.
    AffineMapAttr getAffineMapAttr() {
      return getProperties().map;
    }

    static StringRef getMapAttrStrName() { return "map"; }
  }];
  let assemblyFormat = [{
    $vector `[` $indices `]` `,` $value attr-dict `:` type($vector) `,` type($value)
  }];
}

/// New: assign src vector into dst vector (side-effectful).
def VectorAssignOp : DataflowOp<"vector_assign"> {
  let summary = "Assign one vector to another";
  let description = [{
    Copy the contents of `src` into `dst`. Both must have the same vector type.
    Semantically this is a side-effectful operation on the dataflow vector
    handle, similar in spirit to a memref store.
  }];

  let arguments = (ins AnyVector:$dst, AnyVector:$src);

  // If you don't care about printing the type twice, this is the most
  // straightforward way to make both operands buildable.
  let assemblyFormat = [{
    $dst `,` $src attr-dict `:` type($dst) `,` type($src)
  }];
}


//===----------------------------------------------------------------------===//
// High Dataflow Operations from ScaleHLS
//===----------------------------------------------------------------------===//

def DispatchOp : DataflowOp<"dataflow.dispatch", [RecursiveMemoryEffects,
      SingleBlockImplicitTerminator<"YieldOp">,
      ParentOneOf<["func::FuncOp", "affine::AffineForOp"]>]> {
  let summary = "Represent a dataflow dispatch";
  let description = [{
    Dispatch op has a transparent region that contains a list of task ops to be
    dispatched. This op is designed to organize and manipulate task ops at a
    high level and will be lowered to schedule op for dataflow scheduling.
  }];

  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword ( `:` type($results)^ )? $body";

  let hasVerifier = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    /// Get the terminator yield op.
    YieldOp getYieldOp();
  }];
}

def TaskOp : DataflowOp<"dataflow.task", [
      DeclareOpInterfaceMethods<StageLikeInterface>, RecursiveMemoryEffects,
      SingleBlockImplicitTerminator<"YieldOp">, HasParent<"DispatchOp">]> {
  let summary = "Represent a dataflow task";
  let description = [{
    Task op has a transparent region that contains a list of ops to be executed
    in the task. This op is designed to represent the task partition at a high
    level and will be lowered to node op for dataflow scheduling.
  }];

  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword ( `:` type($results)^ )? $body";

  let hasVerifier = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    /// Get the parent dispatch op.
    DispatchOp getDispatchOp();

    /// Get the terminator yield op.
    YieldOp getYieldOp();
  }];
}

def AnyMemrefOrStream : Type<Or<[AnyMemRef.predicate, AnyStream.predicate]>,
    "memref or stream values">;

def NodeOp : DataflowOp<"dataflow.node", [
      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
      DeclareOpInterfaceMethods<StageLikeInterface>, IsolatedFromAbove,
      AffineScope, SingleBlock, NoTerminator, AttrSizedOperandSegments,
      HasParent<"DispatchOp">]> {
  let summary = "Represent a dataflow task";
  let description = [{
    Task op has a transparent region that contains a list of ops to be executed
    in the task. This op is designed to represent the task partition at a high
    level and will be lowered to node op for dataflow scheduling.
  }];

  let arguments = (ins Variadic<AnyMemrefOrStream>:$inputs,
    Variadic<AnyMemrefOrStream>:$outputs, Variadic<AnyType>:$params,
    I32ArrayAttr:$inputTaps, OptionalAttr<I32Attr>:$level);
    
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    `(` $inputs `)` `->` `(` $outputs `)` ( `[` $params^ `]` )? attr-dict `:`
    functional-type($inputs, $outputs) ( `[` type($params)^ `]` )? $body
  }];

  // let hasVerifier = 1;
  // let hasCanonicalizer = 1;
  let builders = [
    OpBuilder<(ins "mlir::ValueRange":$inputs, "mlir::ValueRange":$outputs,
        "mlir::ValueRange":$params, "ArrayRef<unsigned>":$inputTaps,
        "mlir::IntegerAttr":$level), [{
          auto newInputTaps = SmallVector<int32_t>(
            llvm::map_range(inputTaps, [](unsigned a) { return (int32_t)a; }));
          build($_builder, $_state, inputs, outputs, params,
                $_builder.getI32ArrayAttr(newInputTaps), level);
        }]>,

    OpBuilder<(ins "mlir::ValueRange":$inputs, "mlir::ValueRange":$outputs,
        "mlir::ValueRange":$params, "ArrayRef<unsigned>":$inputTaps), [{
          build($_builder, $_state, inputs, outputs, params, inputTaps, nullptr);
        }]>,
    OpBuilder<(ins "mlir::ValueRange":$inputs, "mlir::ValueRange":$outputs,
        "ArrayRef<unsigned>":$inputTaps), [{
          build($_builder, $_state, inputs, outputs, ValueRange(), inputTaps);
        }]>,

    OpBuilder<(ins "mlir::ValueRange":$inputs, "mlir::ValueRange":$outputs,
        "mlir::ValueRange":$params), [{
          build($_builder, $_state, inputs, outputs, params,
                SmallVector<unsigned>(inputs.size(), 0));
        }]>,
    OpBuilder<(ins "mlir::ValueRange":$inputs, "mlir::ValueRange":$outputs),
        "build($_builder, $_state, inputs, outputs, ValueRange());">
  ];

  let extraClassDeclaration = [{
    /// Get the parent dispatch op.
    DispatchOp getDispatchOp();

    /// Get the terminator yield op.
    YieldOp getYieldOp();

    /// Get input taps.
    void setInputTap(unsigned idx, unsigned tap);
    unsigned getInputTap(unsigned idx);
    SmallVector<unsigned> getInputTapsAsInt();

    /// Get the number of inputs, outputs, and params.
    unsigned getNumInputs();
    unsigned getNumOutputs();
    unsigned getNumParams();

    /// Get the type of operand: input, output, or param.
    OperandKind getOperandKind(OpOperand &operand);
    OperandKind getOperandKind(unsigned operandIdx);

    /// Get the input, output, and param arguments.
    iterator_range<Block::args_iterator> getInputArgs();
    iterator_range<Block::args_iterator> getOutputArgs();
    iterator_range<Block::args_iterator> getParamArgs();
  }];
}

def YieldOp : DataflowOp<"dataflow.yield", [NoMemoryEffect, ReturnLike, Terminator,
      ParentOneOf<["DispatchOp", "TaskOp", "NodeOp"]>]> {
  let summary = "Terminate and yield results of a dispatch or task op";

  let arguments = (ins Variadic<AnyType>:$results);
  let assemblyFormat = "$results attr-dict `:` type($results)";

  let builders = [OpBuilder<(ins), "build($_builder, $_state);">];
}

//===----------------------------------------------------------------------===//
// Miscellaneous operations from HeteroCL
//===----------------------------------------------------------------------===//

def PrintOp : DataflowOp<"print"> {
  let summary = "print value operation";
  let description = [{
    The "print" builtin operation prints a list of 
    values (variadic), and produces no results.
  }];

  // The print operation takes a list of values (variadic).
  let arguments = (ins StrAttr:$msg);
  let assemblyFormat = "$msg attr-dict";
}

def PrintMemRefOp : DataflowOp<"print_memref"> {
  let summary = "print memref operation";
  let description = [{
    The "print_memref" builtin operation prints a memref
    and produces no results. We define our own print_memref
    to support fixed-point type memrefs
  }];
  let arguments = (ins AnyMemRef:$input);
  let assemblyFormat = "`(` $input `)` attr-dict `:` type($input)";
}

//===----------------------------------------------------------------------===//
// Miscellaneous operations for Tapa
//===----------------------------------------------------------------------===//
def TAPATaskBeginOp : DataflowOp<"dataflow.TAPATaskBegin", [
      ParentOneOf<["func::FuncOp"]>]> {
  let summary = "TAPA task begin";
}
def TAPATaskEndOp : DataflowOp<"dataflow.TAPATaskEnd", [
      ParentOneOf<["func::FuncOp"]>]> {
  let summary = "TAPA task end";
}

//===----------------------------------------------------------------------===//
// XRT Operations
//===----------------------------------------------------------------------===//
// def XRTDeviceOp : DataflowOp<"xrt.device", [NoMemoryEffect]> {
//   let summary = "XRT device operation";
//   let description = [{
//     The "xrt.device" operation specifies the device to be used
//     for the following operations.
//   }];

//   let arguments = (ins I32Attr:$deviceId);
//   let results = (outs XRTDeviceType:$device);
//   let assemblyFormat = "$deviceId attr-dict `:` type($device)";
// }

// def XRTLoadBitStreamOp : DataflowOp<"xrt.load_bitstream", [NoMemoryEffect]> {
//   let summary = "XRT load bitstream operation";
//   let description = [{
//     The "xrt.load_bitstream" operation loads the bitstream
//     onto the specified device.
//   }];

//   let arguments = (ins XRTDeviceType:$device, StrAttr:$bitstream);
//   let results = (outs I32:$uuid);
//   let assemblyFormat = "$device `,` $bitstream attr-dict `:` type($uuid)";
// }

// def XRTKernelOp : DataflowOp<"xrt.kernel", [NoMemoryEffect]> {
//   let summary = "XRT kernel operation";
//   let description = [{
//     The "xrt.kernel" operation specifies the kernel to be used
//     for the following operations.
//   }];

//   let arguments = (ins XRTDeviceType:$device, I32:$uuid, StrAttr:$topModule, StrAttr:$mode);
//   let results = (outs XRTKernelType:$kernel);
//   let assemblyFormat = "$device `,` $uuid `,` $topModule `,` $mode attr-dict `:` type($kernel)";
// }

// def XRTBufferAllocOp : DataflowOp<"xrt.buffer_alloc", [NoMemoryEffect]> {
//   let summary = "XRT buffer allocation operation";
//   let description = [{
//     The "xrt.buffer_alloc" operation allocates a buffer on the
//     specified device.
//   }];
//   let arguments = (i ns XRTDeviceType:$device, XRTKernelType:$krnl, I32Attr:$groupId);
//   let results = (outs XRTBufferOf<[AnyType]>:$buffer);
//   let assemblyFormat = "$device `,` $krnl attr-dict `:` type($buffer)";
// }

// def XRTMapOp : DataflowOp<"xrt.map", [NoMemoryEffect]> {
//   let summary = "XRT map operation";
//   let description = [{
//     The "xrt.map" operation maps the buffer onto the specified
//     device.
//   }];
//   let arguments = (ins XRTBufferOf<[AnyType]>:$buffer);
//   let results = (outs XRTMapType:$result);
//   // let assemblyFormat = [{
//   //   $buffer attr-dict `:` type($result)
//   // }];

// }



// def MmapLoadOp : DataflowOp<"mmap_load", [
//     DeclareOpInterfaceMethods<AffineReadOpInterface>,
//     DeclareOpInterfaceMethods<AffineMapAccessInterface>,
//     MemRefsNormalizable,
//     SameVariadicOperandSize 
// ]>{
//   let summary = "Load a value from a mmap";
//   let description = [{
//     Load a value from a mmap using affine maps. 
//   }];
//   let arguments = (
//     ins AnyMemRef:$mmap, 
//     Variadic<Index>:$vector_indices, 
//     AffineMapAttr:$vector_map,
//     Variadic<Index>:$scalar_indices,
//     AffineMapAttr:$scalar_map
//   );
//   let results = (outs AnyType:$result);
//   // code extraClassDeclaration = [{
//   //   /// Returns the operand index of the memref.
//   //   unsigned getMemRefOperandIndex() { return 0; }

//   //   void setMemRef(Value value) { setOperand(getMemRefOperandIndex(), value); }

//   //   AffineMapAttr getAffineMapAttr() {
//   //     return getProperties().vector_map;
//   //   }

//   //   // AffineMapAttr getScalarAffineMapAttr() {
//   //   //   return getProperties().scalar_map;
//   //   // }

//   //   static StringRef getMapAttrStrName() { return "vector_map"; }
//   //   // static StringRef getScalarMapAttrStrName() { return "scalar_map"; }
//   // }];
//   let assemblyFormat = [{
//     $mmap `[` $vector_indices `]` `[` $scalar_indices `]` attr-dict `:` functional-type($mmap, $result)
//   }];
//   // let assemblyFormat = [{
//   //   $mmap `[` $vector_indices `]` attr-dict `:` functional-type($mmap, $result)
//   // }];
// }
// def StreamReadOp : DataflowOp<"stream_read", []> {
//   let summary = "Read a stream channel";
//   let description = [{
//     Read/pop a value from a stream channel. The absence of the result indicates
//     the popped value is dropped. Each stream channel can be read multiple times.
//   }];

//   let arguments = (ins StreamOf<[AnyType]>:$channel);
//   let results = (outs Optional<AnyType>:$result);
//   let assemblyFormat = [{
//     $channel attr-dict `:` functional-type($channel, $result)
//   }];

//   let hasVerifier = 1;
// }
// class MemAlloc<Resource resource, int stage = 0,
//                EffectRange range = PartialEffect>
//   : MemoryEffect<"::mlir::MemoryEffects::Allocate", resource, stage, range>;
// def MemAlloc : MemAlloc<DefaultResource, 0, PartialEffect>;

// // Any ranked memref whose element type is from the given `allowedTypes` list.
// class MemRefOf<list<Type> allowedTypes> :
//     ShapedContainerType<allowedTypes, IsMemRefTypePred, "memref",
//                         "::mlir::MemRefType">;
// def AnyMemRef : MemRefOf<[AnyType]>;

// class RankedTensorOf<list<Type> allowedTypes> :
//     ShapedContainerType<allowedTypes, IsRankedTensorTypePred, "tensor",
//                         "::mlir::RankedTensorType">;



// def ArrayOfStreamsOp : DataflowOp<"array_of_streams", [
//       DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
//   let summary = "Declare an array of stream channels";

//   let arguments = (ins Variadic<I32Attr>:$dims, I32Attr:$depth);
//   let results = (outs Variadic<StreamOf<[AnyType]>>:$channels);
//   let assemblyFormat = "attr-dict `:` type($channels)";

// }

// include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.td"

// def ArrayOfStreamsReadOp : DataflowOp<"array_of_streams_read", 
//       [DeclareOpInterfaceMethods<AffineReadOpInterface>,
//        DeclareOpInterfaceMethods<AffineMapAccessInterface>]> {
//   let summary = "Read an array of stream channels";
//   let description = [{
//     Read/pop a value from an array of stream channels. The absence of the result indicates
//     the popped value is dropped. Each stream channel can be read multiple times.
//   }];

  
// }

#endif // DATAFLOW_OPS