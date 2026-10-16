

#include "streamhls/Dialect/Dataflow/Translation/EmitHLS.h"
#include "streamhls-c/Dialect/Dataflow/Translation/EmitVitisHLS.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"

using namespace mlir;
using namespace streamhls;

MlirLogicalResult mlirEmitVitisHls(MlirModule module,
                                    MlirStringCallback callback,
                                    void *userData,
                                    bool useHlsTask) {
  mlir::detail::CallbackOstream stream(callback, userData);
  return wrap(emitHLS(unwrap(module), stream, useHlsTask));
}