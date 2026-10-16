#ifndef STFLOW_C_TRANSLATION_EMITTAPAHLS_H
#define STFLOW_C_TRANSLATION_EMITTAPAHLS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED MlirLogicalResult mlirEmitTapa(MlirModule module,
                                                    MlirStringCallback callback,
                                                    void *userData);

#ifdef __cplusplus
}
#endif

#endif // STFLOW_C_TRANSLATION_EMITTAPAHLS_H
