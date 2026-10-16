//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
//
//===----------------------------------------------------------------------===//

#ifndef STREAMHLS_DIALECT_HLS_ATTRIBUTES_TD
#define STREAMHLS_DIALECT_HLS_ATTRIBUTES_TD
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

// --------------------------------------------------------------------------
// Enum: dataflow access kind = {load, store} (integer-backed, no strings)
// Produces: ::mlir::dataflow::AccessKind  and  ::mlir::dataflow::AccessKindAttr
// --------------------------------------------------------------------------
def DataflowAccessKind : I32EnumAttr<"AccessKind", "Kind of memory access", [
  I32EnumAttrCase<"load",  0>,
  I32EnumAttrCase<"store", 1>
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "mlir::dataflow";
}

def DataflowIOKind : I32EnumAttr<"IOKind", "Kind of IO for top-level off-chip ports", [
  I32EnumAttrCase<"read_only",  0>,
  I32EnumAttrCase<"write_only", 1>,
  I32EnumAttrCase<"read_write", 2>,
  I32EnumAttrCase<"unknown",   3>
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "mlir::dataflow";
}

// // Wrap the enum as a dialect attribute (mnemonic "access_kind")
// def DataflowAccessKindAttr : EnumAttr<DataflowDialect, DataflowAccessKind, "access_kind">{
//   let cppNamespace = "mlir::dataflow";
// }
// --------------------------------------------------------------------------
// Element attribute for a single bound: [lower, upper)
// Printed form:  dataflow.access_bound<0, 256>
// C++ class:     ::mlir::dataflow::AccessBoundAttr
// --------------------------------------------------------------------------
def AccessBoundAttr : DataflowAttr<"AccessBound"> {
  let summary  = "Half-open bound [lower, upper) for a lexicographic dim";
  let mnemonic = "access_bound";               // printed name after `dataflow.`
  let parameters = (ins "int64_t":$lower, "int64_t":$upper);
  let assemblyFormat = "`<` $lower `,` $upper `>`";
  let cppNamespace = "::mlir::dataflow";
}
// --------------------------------------------------------------------------
// Main attribute: dataflow.access_info
//
// Fields:
//   - type          : load | store                      (AccessKindAttr)
//   - access_map    : affine_map<(d...)->(linear_idx)>
//   - access_bounds : array<dataflow.access_bound<lo,hi>>
//   - num_access    : i64
//
// Example (enum prints without quotes):
//   dataflow.access_info<load,
//                        affine_map<(d0, d1) -> (d0 * 256 + d1)>,
//                        [dataflow.access_bound<0, 2>, dataflow.access_bound<0, 256>],
//                        512>
// C++ class: ::mlir::dataflow::AccessInfoAttr
// --------------------------------------------------------------------------
def AccessInfoAttr : DataflowAttr<"AccessInfo"> {
  let summary  = "Lexicographic access info for a single affine load/store";
  let description = [{
    Captures the lexicographic execution order of a specific affine.load/store
    within its surrounding affine.for/affine.if nest (after removing IVs pinned
    by equality guards). The access_map is defined over new dims d0..d{n-1}
    whose ranges are given by access_bounds; num_access is the product of
    extents (upper - lower) across those dims.
  }];

  let parameters = (ins
    "mlir::dataflow::AccessKind":$kind,
    "mlir::AffineMapAttr":$access_map,           // <-- C++ type string
    "mlir::ArrayAttr":$access_bounds,            // <-- C++ type string (array of AccessBoundAttr)
    "int64_t":$num_access,                        // <-- primitive type (int64_t)
    "int64_t":$unroll_start_idx
  );

  let assemblyFormat = [{
    `<` $unroll_start_idx `,` $kind `,` $access_map `,` $access_bounds `,` $num_access `>`
  }];

  // *** Convenience builder: accepts enum + AffineMap + vector<pair> + int64_t ***
  let builders = [
    AttrBuilder<(ins
      "mlir::dataflow::AccessKind":$kind,
      "mlir::AffineMap":$map,
      "llvm::ArrayRef<std::pair<int64_t,int64_t>>":$bounds,
      "int64_t":$num_access
    ), [{
      llvm::SmallVector<mlir::Attribute> boundAttrs;
      boundAttrs.reserve(bounds.size());
      auto i64 = mlir::IntegerType::get(context, 64);
      for (auto &p : bounds) {
        boundAttrs.push_back(mlir::dataflow::AccessBoundAttr::get(
            context, p.first, p.second));
      }
      return mlir::dataflow::AccessInfoAttr::get(
        context,
        kind,
        mlir::AffineMapAttr::get(map),
        mlir::ArrayAttr::get(context, boundAttrs),
        num_access,
        -1 // default unroll_start_idx
      );
    }]>,
    AttrBuilder<(ins
      "mlir::dataflow::AccessKind":$kind,
      "mlir::AffineMap":$map,
      "llvm::ArrayRef<std::pair<int64_t,int64_t>>":$bounds,
      "int64_t":$num_access,
      "int64_t":$unroll_start_idx
    ), [{
      llvm::SmallVector<mlir::Attribute> boundAttrs;
      boundAttrs.reserve(bounds.size());
      auto i64 = mlir::IntegerType::get(context, 64);
      for (auto &p : bounds) {
        boundAttrs.push_back(mlir::dataflow::AccessBoundAttr::get(
            context, p.first, p.second));
      }
      return mlir::dataflow::AccessInfoAttr::get(
        context,
        kind,
        mlir::AffineMapAttr::get(map),
        mlir::ArrayAttr::get(context, boundAttrs),
        num_access,
        unroll_start_idx
      );
    }]>
  ];


  let mnemonic = "access_info"; 
  let cppNamespace = "::mlir::dataflow";
}

// --------------------------------------------------------------------------
// dataflow.port_width<N>
// Represents number of vector lanes for a top-level port, used when
// converting to !dataflow.axivec<elem, N>.
// C++: ::mlir::dataflow::PortWidthAttr
// --------------------------------------------------------------------------
def PortWidthAttr : DataflowAttr<"PortWidth"> {
  let summary  = "Number of lanes (vector width) for a top-level HLS port";
  let mnemonic = "port_width";      // prints as: dataflow.port_width<4>
  let cppNamespace = "::mlir::dataflow";

  // Single integer parameter: lanes
  let parameters = (ins "int64_t":$lanes);

  // Textual form: dataflow.port_width<4>
  let assemblyFormat = "`<` $lanes `>`";

  // Optional: convenience builder from unsigned
  let builders = [
    AttrBuilder<(ins "unsigned":$lanesVal), [{
      return mlir::dataflow::PortWidthAttr::get(
        context,
        static_cast<int64_t>(lanesVal)
      );
    }]>
  ];
}

// ==== Buffer kind enum: integer-backed, no strings ====
def DataflowBufferKind : I32EnumAttr<"BufferKind", "Kind of buffer", [
  I32EnumAttrCase<"shared_buffer", 0>,
  I32EnumAttrCase<"local_buffer",  1>,
  I32EnumAttrCase<"off_chip",      2>,
  I32EnumAttrCase<"weights",       3>
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::dataflow";
}

// def DataflowBufferKindAttr
//   : EnumAttr<DataflowDialect, DataflowBufferKind, "buffer_kind"> {
//   let cppNamespace = "::mlir::dataflow";
// }

// ==== dataflow.buffer_info { id: str, kind: enum } ====
// Prints like:
//   dataflow.buffer_info{ id = "__constant_64xf32", kind = weights }
def BufferInfoAttr : DataflowAttr<"BufferInfo"> {
  let mnemonic = "buffer_info";
  let summary  = "Buffer metadata (unique id + kind enum)";

  let parameters = (ins
    "mlir::StringAttr":$name,
    "mlir::dataflow::BufferKind":$kind
  );

  // dictionary-style text form
  let assemblyFormat = [{
    `{` `name` `=` $name `,` `kind` `=` $kind `}`
  }];

  // Optional: convenience builder accepting plain C++ types.
  let builders = [
    AttrBuilder<(ins
      "llvm::StringRef":$name,
      "mlir::dataflow::BufferKind":$kindEnum
    ), [{
      return mlir::dataflow::BufferInfoAttr::get(
        context,
        mlir::StringAttr::get(context, name),
        kindEnum
      );
    }]>
  ];

  let cppNamespace = "::mlir::dataflow";
}

def PortIOAttr : DataflowAttr<"PortIO"> {
  let mnemonic = "port_io";
  let summary  = "Port I/O metadata (unique id + kind enum)";

  let parameters = (ins
    "mlir::dataflow::IOKind":$io_kind
  );

  // dictionary-style text form
  let assemblyFormat = [{
    `{` `kind` `=` $io_kind `}`
  }];

  // // Optional: convenience builder accepting plain C++ types.
  // let builders = [
  //   AttrBuilder<(ins
  //     "mlir::dataflow::IOKind":$ioKind
  //   ), [{
  //     return mlir::dataflow::PortIOAttr::get(
  //       context,
  //       ioKind
  //     );
  //   }]>
  // ];

  let cppNamespace = "::mlir::dataflow";
}



def MM: I32EnumAttrCase<"MM", 0>;
def LITE: I32EnumAttrCase<"LITE", 1>;
def STREAM: I32EnumAttrCase<"STREAM", 2>;
def AxiKind: I32EnumAttr<"AxiKind", "Kind of AXI Port", [MM, LITE, STREAM]> {
  let cppNamespace = "mlir::dataflow";
}

def Resource : DataflowAttr<"Resource"> {
  let summary = "Resource utilization information";
  let description = [{
    This attribute includes the utilization of several important on-chip
    resources, such as LUT, DSP, BRAM, and URAM.
  }];

  let hasCustomAssemblyFormat = 1;
  let mnemonic = "r";
  let parameters = (ins "int64_t":$lut, "int64_t":$dsp, "int64_t":$bram );
}

def Timing : DataflowAttr<"Timing"> {
  let summary = "Timing information";
  let description = [{
    This attribute indicates the timing of operations, including scheduling
    begin and end cycles, latency, and interval.
  }];

  let hasCustomAssemblyFormat = 1;
  let mnemonic = "t";
  let parameters = (ins "int64_t":$begin, "int64_t":$end, "int64_t":$latency,
                        "int64_t":$interval);
}

def LoopInfo : DataflowAttr<"LoopInfo"> {
  let summary = "Loop information";
  let description = [{
    This attribute includes loop information used by estimator and optimizer,
    including flattened trip count, iteration latency, and minimum initial
    interval (II).
  }];

  let hasCustomAssemblyFormat = 1;
  let mnemonic = "l";
  let parameters = (ins "int64_t":$flattenTripCount, "int64_t":$iterLatency,
                        "int64_t":$minII);
}

def LoopDirective : DataflowAttr<"LoopDirective"> {
  let summary = "Aggregation of loop directives";
  let description = [{
    This attribute includes all loop directives, including loop pipelining,
    target initial interval (II), dataflow, and flattening.
  }];

  let hasCustomAssemblyFormat = 1;
  let mnemonic = "pipeline_ii";
  let parameters = (ins "bool":$pipeline, "int64_t":$targetII, "bool":$dataflow,
                        "bool":$flatten);
}

def FuncDirective : DataflowAttr<"FuncDirective"> {
  let summary = "Aggregation of function directives";
  let description = [{
    This attribute includes all function directives, including function
    pipelining, target interval, and dataflow.
  }];

  let hasCustomAssemblyFormat = 1;
  let mnemonic = "fd";
  let parameters = (ins "bool":$pipeline, "int64_t":$targetInterval,
                        "bool":$dataflow);
}

#endif // SCALEHLS_DIALECT_HLS_ATTRIBUTES_TD
