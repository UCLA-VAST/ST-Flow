 #ifndef STREAM_HLS_MLIR_C_REGISTRATION_H
 #define STREAM_HLS_MLIR_C_REGISTRATION_H
 
 #include "mlir/CAPI/IR.h"
 #include "mlir-c/IR.h"
 #include "mlir-c/Support.h"
 
 #ifdef __cplusplus
 extern "C" {
 #endif
 
 /** Registers all dialects with a context.
  * This is needed before creating IR for these Dialects.
  */
 MLIR_CAPI_EXPORTED void streamhlsMlirRegisterAllDialects(MlirContext context);
 
 /** Registers all passes for symbolic access with the global registry. */
 MLIR_CAPI_EXPORTED void streamhlsMlirRegisterAllPasses();
 
 #ifdef __cplusplus
 }
 #endif
 
 #endif // STREAM_HLS_MLIR_C_REGISTRATION_H