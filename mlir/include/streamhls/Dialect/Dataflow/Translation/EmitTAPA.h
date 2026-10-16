/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef STREAMHLS_TRANSLATION_EMITTAPA_H
#define STREAMHLS_TRANSLATION_EMITTAPA_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace streamhls {

LogicalResult emitTAPA(ModuleOp module, llvm::raw_ostream &os);
void registerEmitTAPATranslation();

} // namespace streamhls
} // namespace mlir

#endif // STREAMHLS_TRANSLATION_EMITTAPA_H