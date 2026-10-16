//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Adapted from ScaleHLS (https://github.com/UIUC-ChenLab/scalehls) and modified
// for Stream-HLS and ST-Flow. See NOTICE.
//
//===----------------------------------------------------------------------===//

#ifndef STREAMHLS_DIALECT_DATAFLOW_TYPES_TD
#define STREAMHLS_DIALECT_DATAFLOW_TYPES_TD

def StreamType : DataflowType<"Stream"> {
  let summary = "A stream type";
  let description = [{
    Represents a stream of any type that can be transfered between HLS modules.
  }];
  let mnemonic = "stream";
  let parameters = (ins AnyType :$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}

// New: AXI vector port type, modeling hls::vector<T, lanes>*.
def AXIVectorPortType : DataflowType<"AXIVectorPort"> {
  let summary = "AXI memory-mapped vector port";
  let description = [{
    Represents a memory-mapped AXI port whose payload is a vector of lanes of
    a scalar element type, e.g., hls::vector<float, 4>*.

    The total bitwidth (element bitwidth x lanes) should be a power-of-two
    width up to 512 bits (enforced in C++ verify()).
  }];
  // MLIR textual name: !dataflow.axivec<...>
  let mnemonic = "axivec";

  // Parameters: element type and number of lanes.
  let parameters = (ins
    AnyType   : $elementType,
    "unsigned": $lanes
  );

  // Syntax: !dataflow.axivec<f32, 4>
  let assemblyFormat = "`<` $elementType `,` $lanes `>`";

}
//===----------------------------------------------------------------------===//
// XRT Types
//===----------------------------------------------------------------------===//
// def XRTBufferType : DataflowType<"XRTBuffer"> {
//   let summary = "A buffer type";
//   let description = [{
//     Represents a buffer type that can be transfered between HLS modules.
//   }];
//   let mnemonic = "xrtBuffer";
//   let parameters = (ins "mlir::Type":$elementType, "unsigned":$sizeBytes);
//   let assemblyFormat = "`<` qualified($elementType) `,` $sizeBytes `>`";
// }

// def XRTDeviceType : DataflowType<"Device"> {
//   let summary = "A device type";
//   let description = [{
//     Represents a device type that can be transfered between HLS modules.
//   }];
//   let mnemonic = "device";
// }

// def XRTKernelType : DataflowType<"Kernel"> {
//   let summary = "A kernel type";
//   let description = [{
//     Represents a kernel type that can be transfered between HLS modules.
//   }];
//   let mnemonic = "kernel";
//   let parameters = (ins Variadic<AnyType>:$inputs);
//   let assemblyFormat = "`<` $inputs `>`";
// }

// def XRTMapType : DataflowType<"Map"> {
//   let summary = "A map type";
//   let description = [{
//     Represents a map type that can be transfered between HLS modules.
//   }];
//   let mnemonic = "map";
//   let parameters = (ins "mlir::Type":$elementType, "unsigned":$sizeBytes);
//   let assemblyFormat = "`<` qualified($elementType) `,` $sizeBytes `>`";
// }

#endif // STREAMHLS_DIALECT_DATAFLOW_TYPES_TD
