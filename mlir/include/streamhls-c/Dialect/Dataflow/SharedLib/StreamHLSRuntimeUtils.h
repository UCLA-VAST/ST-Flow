#ifndef STREAM_HLSC_SHARED_LIB_STREAM_HLS_RUNTIME_UTILS_H
#define STREAM_HLSC_SHARED_LIB_STREAM_HLS_RUNTIME_UTILS_H

#ifdef _WIN32
#ifndef STREAM_HLS_RUNTIME_UTILS_EXPORT
#ifdef allo_runtime_utils_EXPORTS
// We are building this library
#define STREAM_HLS_RUNTIME_UTILS_EXPORT __declspec(dllexport)
#define STREAM_HLS_RUNTIME_UTILS_DEFINE_FUNCTIONS
#else
// We are using this library
#define STREAM_HLS_RUNTIME_UTILS_EXPORT __declspec(dllimport)
#endif // allo_runtime_utils_EXPORTS
#endif // STREAM_HLS_RUNTIME_UTILS_EXPORT
#else  // _WIN32
// Non-windows: use visibility attributes.
#define STREAM_HLS_RUNTIME_UTILS_EXPORT __attribute__((visibility("default")))
#define STREAM_HLS_RUNTIME_UTILS_DEFINE_FUNCTIONS
#endif // _WIN32

#include <string>
#include "mlir/ExecutionEngine/CRunnerUtils.h"

extern "C" STREAM_HLS_RUNTIME_UTILS_EXPORT void
readMemrefI32(int64_t rank, void *ptr, char *str);

extern "C" STREAM_HLS_RUNTIME_UTILS_EXPORT void
readMemrefI64(int64_t rank, void *ptr, char *str);

extern "C" STREAM_HLS_RUNTIME_UTILS_EXPORT void
readMemrefF32(int64_t rank, void *ptr, char *str);

extern "C" STREAM_HLS_RUNTIME_UTILS_EXPORT void
readMemrefF64(int64_t rank, void *ptr, char *str);

extern "C" STREAM_HLS_RUNTIME_UTILS_EXPORT void
writeMemrefI32(int64_t rank, void *ptr, char *str);

extern "C" STREAM_HLS_RUNTIME_UTILS_EXPORT void
writeMemrefI64(int64_t rank, void *ptr, char *str);

extern "C" STREAM_HLS_RUNTIME_UTILS_EXPORT void
writeMemrefF32(int64_t rank, void *ptr, char *str);

extern "C" STREAM_HLS_RUNTIME_UTILS_EXPORT void
writeMemrefF64(int64_t rank, void *ptr, char *str);

#endif // STREAM_HLSC_SHARED_LIB_STREAM_HLS_RUNTIME_UTILS_H