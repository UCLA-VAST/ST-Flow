
#ifndef STFLOW_C_TRANSLATION_EMITVIVADOHLS_H
#define STFLOW_C_TRANSLATION_EMITVIVADOHLS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED MlirLogicalResult mlirEmitVitisHls(MlirModule module,
                                                     MlirStringCallback callback,
                                                     void *userData,
                                                     bool useHlsTask);

#ifdef __cplusplus
}
#endif

#endif // STFLOW_C_TRANSLATION_EMITVIVADOHLS_H
