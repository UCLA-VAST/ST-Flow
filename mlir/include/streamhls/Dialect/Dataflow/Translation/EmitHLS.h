/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef STREAMHLS_TRANSLATION_EMITHLS_H
#define STREAMHLS_TRANSLATION_EMITHLS_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace streamhls {

LogicalResult emitHLS(ModuleOp module, llvm::raw_ostream &os, bool useHlsTask = false);
void registerEmitHLSTranslation();

} // namespace streamhls
} // namespace mlir

#endif // STREAMHLS_TRANSLATION_EMITHLS_H