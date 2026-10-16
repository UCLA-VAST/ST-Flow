//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Adapted from ScaleHLS (https://github.com/UIUC-ChenLab/scalehls) and modified
// for Stream-HLS and ST-Flow. See NOTICE.
//
//===----------------------------------------------------------------------===//

#ifndef STREAMHLS_INITALLPASSES_H
#define STREAMHLS_INITALLPASSES_H

#include "streamhls/Dialect/Dataflow/Transforms/Passes.h"
#include "mlir/InitAllPasses.h"

namespace mlir {
namespace streamhls {

// Add all the STREAMHLS passes.
inline void registerAllPasses() {
  streamhls::registerTransformsPasses();
  // mlir::registerAllPasses();
}

} // namespace streamhls
} // namespace mlir

#endif // STREAMHLS_INITALLPASSES_H
