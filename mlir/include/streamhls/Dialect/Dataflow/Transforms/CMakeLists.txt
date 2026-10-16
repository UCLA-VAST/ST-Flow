set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name StreamHLS)
add_public_tablegen_target(MLIR_StreamHLS_TransformsIncGen)
