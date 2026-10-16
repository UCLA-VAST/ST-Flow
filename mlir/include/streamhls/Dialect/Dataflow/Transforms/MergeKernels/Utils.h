#pragma once
#include "streamhls/Dialect/Dataflow/Transforms/MergeKernels/DataTypes.h"

namespace mlir {
namespace streamhls {
AffineExpr getConstOne(MLIRContext *ctx);
AffineExpr getConstZero(MLIRContext *ctx);
// ---------- get min and max loop stats across all loop bands ----------
void getMinAndMaxLoopStatsMerge(SmallVectorImpl<LoopBandInfo> &allLoopBandsInfo, LoopBandInfo &minBandInfo, LoopBandInfo &maxBandInfo);
/// Build the VariantMetadata dictionary for a single source function `srcFunc`.
/// This constructs:
///   - "loop"               : loopInfo dictionary (per-variant UBs, etc.)
///   - "buffer_info_list"   : per-argument buffer.info wrappers (if present)
///   - "access_info_list"   : per-argument access.info wrappers computed from
///                            the merged function's loop band.
///
/// `mergedFunc` is the function whose body contains the merged accesses, and
/// `band` is its outermost loop band.
DictionaryAttr buildVariantMetadataForFunc(OpBuilder &b,
                                                  func::FuncOp srcFunc,
                                                  func::FuncOp mergedFunc,
                                                  const AffineLoopBand &band);
// Build an ArrayAttr "merge.variants" on the merged module, each entry a
// DictionaryAttr describing one source kernel variant (its problem sizes)
// and the per-argument buffer.info attributes (if present).
void attachVariantMetadata(ModuleOp mergedModule, ArrayRef<func::FuncOp> funcs);
} // end namespace streamhls
} // end namespace mlir