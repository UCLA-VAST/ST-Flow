 #ifndef STREAM_HLS_C_DIALECT_HLSCPP__H
 #define STREAM_HLS_C_DIALECT_HLSCPP__H
 
 #include "mlir-c/RegisterEverything.h"
 
 #ifdef __cplusplus
 extern "C" {
 #endif
 
 MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(STREAM_HLS, streamhls); //TODO: Check this
 
 #ifdef __cplusplus
 }
 #endif
 
 #endif // STREAM_HLS_C_DIALECT_HLSCPP__H
 