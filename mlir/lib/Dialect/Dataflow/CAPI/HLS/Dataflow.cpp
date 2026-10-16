

 #include "streamhls-c/Dialect/Dataflow/IR/Dataflow.h"

 #include "streamhls/Dialect/Dataflow/IR/Dataflow.h"
 #include "mlir/CAPI/Registration.h"
 
 MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(STREAM_HLS, streamhls, mlir::dataflow::DataflowDialect)