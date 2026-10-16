

#include "streamhls/Dialect/Dataflow/Translation/EmitTAPA.h"
#include "streamhls-c/Dialect/Dataflow/Translation/EmitTapaHLS.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"

using namespace mlir;
using namespace streamhls;

MlirLogicalResult mlirEmitTapa(MlirModule module,
                                    MlirStringCallback callback,
                                    void *userData) {
  mlir::detail::CallbackOstream stream(callback, userData);
  return wrap(emitTAPA(unwrap(module), stream));
}