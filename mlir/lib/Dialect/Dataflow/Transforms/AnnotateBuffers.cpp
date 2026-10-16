
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/IR/Builders.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "streamhls/Dialect/Dataflow/Transforms/Passes.h"
#include "streamhls/Dialect/Dataflow/Support/Utils.h"

#include <system_error>
#include <cstring>

#define DEBUG_TYPE "streamhls-annotate-buffers"

using namespace mlir;
using namespace streamhls;

// Debugging helpers
namespace {
static StringRef getBufferNameForAlloc(memref::AllocOp alloc,
                                      StringRef kAttrKey = "buffer.info") {
  if (auto dict = alloc->getAttrOfType<DictionaryAttr>(kAttrKey)) {
    if (auto n = dict.getAs<StringAttr>("name"))
      return n.getValue();
  }
  return "<unnamed_alloc>";
}

static void debugReportAllocLoadStoreCountsNoAliases(func::FuncOp func) {
  func.walk([&](memref::AllocOp alloc) {
    Value buf = alloc.getResult();
    int64_t numLoads = 0;
    int64_t numStores = 0;

    for (Operation *user : buf.getUsers()) {
      if (isa<affine::AffineLoadOp>(user))
        ++numLoads;
      else if (isa<affine::AffineStoreOp>(user))
        ++numStores;
    }

    StringRef name = getBufferNameForAlloc(alloc);

    if ((numLoads == 1 && numStores == 1) || (numLoads == 2 && numStores == 2)) {
      LLVM_DEBUG(llvm::dbgs() << "[AnnotateBuffers] INFO  alloc '" << name
                              << "': stores=" << numStores
                              << " loads=" << numLoads << "\n");
    } else if (numLoads == 0 || numStores == 0) {
      LLVM_DEBUG(llvm::dbgs() << "[AnnotateBuffers] ERROR alloc '" << name
                              << "': stores=" << numStores
                              << " loads=" << numLoads << "\n");
    } else {
      LLVM_DEBUG(llvm::dbgs() << "[AnnotateBuffers] WARN  alloc '" << name
                              << "': stores=" << numStores
                              << " loads=" << numLoads << "\n");
    }
  });
}
} // namespace

namespace {
/// Return the set of *distinct* enclosing affine.for ops for all uses of `v`.
static void collectDistinctEnclosingAffineFors(Value v,
                                               llvm::SmallPtrSetImpl<Operation *> &fors) {
  for (OpOperand &use : v.getUses()) {
    Operation *user = use.getOwner();
    if (auto forOp = user->getParentOfType<affine::AffineForOp>())
      fors.insert(forOp.getOperation());
  }
}

/// Try to extract a human-friendly name for a memref block argument.
static std::string nameForBlockArg(func::FuncOp f, BlockArgument arg) {
  // Try common arg-name attributes first.
  if (auto n = f.getArgAttrOfType<StringAttr>(arg.getArgNumber(), "llvm.name"))
    return n.str();
  if (auto n = f.getArgAttrOfType<StringAttr>(arg.getArgNumber(), "arg_name"))
    return n.str();

  // Fall back to "arg<idx>".
  return ("arg" + std::to_string(arg.getArgNumber()));
}


/// Write a DenseElementsAttr (weights) to a raw .bin file in little-endian.
/// Supports common element types; extend as needed.
static LogicalResult writeDenseToBin(DenseElementsAttr dense,
                                     llvm::StringRef filePath) {
  std::error_code ec;
  llvm::raw_fd_ostream os(filePath, ec, llvm::sys::fs::OF_None);
  if (ec)
    return failure();

  ShapedType st = dense.getType().cast<ShapedType>();
  Type elemTy = st.getElementType();

  if (auto it = elemTy.dyn_cast<IntegerType>()) {
    unsigned bw = it.getWidth();
    if (bw == 8) {
      for (auto v : dense.getValues<int8_t>())
        os.write(reinterpret_cast<const char *>(&v), sizeof(int8_t));
      return success();
    } else if (bw == 16) {
      // int16_t
      for (auto v : dense.getValues<int16_t>()) {
        llvm::support::ulittle16_t le((uint16_t)v);
        os.write(reinterpret_cast<const char*>(&le), sizeof(le));
      }
      return success();
    } else if (bw == 32) {
      // int32_t
      for (auto v : dense.getValues<int32_t>()) {
        llvm::support::ulittle32_t le((uint32_t)v);
        os.write(reinterpret_cast<const char*>(&le), sizeof(le));
      }
      return success();
    } else if (bw == 64) {
      // int64_t
      for (auto v : dense.getValues<int64_t>()) {
        llvm::support::ulittle64_t le((uint64_t)v);
        os.write(reinterpret_cast<const char*>(&le), sizeof(le));
      }
    }
  } else if (auto ft = elemTy.dyn_cast<FloatType>()) {
    unsigned bw = ft.getWidth();
    if (bw == 16) {
      // f16 (from APFloat bits)
      for (auto v : dense.getValues<APFloat>()) {
        llvm::APInt bits = v.bitcastToAPInt();
        uint16_t w = (uint16_t)bits.getZExtValue();
        llvm::support::ulittle16_t le(w);
        os.write(reinterpret_cast<const char*>(&le), sizeof(le));
      }
      return success();
    } else if (bw == 32) {
      // f32
      for (auto v : dense.getValues<float>()) {
        uint32_t w;
        std::memcpy(&w, &v, sizeof(float));
        llvm::support::ulittle32_t le(w);
        os.write(reinterpret_cast<const char*>(&le), sizeof(le));
      }
      return success();
    } else if (bw == 64) {
      // f64
      for (auto v : dense.getValues<double>()) {
        uint64_t w;
        std::memcpy(&w, &v, sizeof(double));
        llvm::support::ulittle64_t le(w);
        os.write(reinterpret_cast<const char*>(&le), sizeof(le));
      }
      return success();
    }
  }

  // Fallback: iterate generically over Attributes.
  for (Attribute a : dense.getValues<Attribute>()) {
    if (auto fa = a.dyn_cast<FloatAttr>()) {
      llvm::APInt bits = fa.getValue().bitcastToAPInt();
      const char *bytes = reinterpret_cast<const char *>(bits.getRawData());
      os.write(bytes, bits.getBitWidth() / 8);
    } else if (auto ia = a.dyn_cast<IntegerAttr>()) {
      llvm::APInt bits = ia.getValue();
      const char *bytes = reinterpret_cast<const char *>(bits.getRawData());
      os.write(bytes, bits.getBitWidth() / 8);
    } else {
      return failure();
    }
  }
  return success();
}

// /// Write a DenseResourceElementsAttr (weights) to a raw .bin file in little-endian.
// /// Supports common element types; extend as needed.
// static LogicalResult writeDenseResourceToBin(DenseResourceElementsAttr res,
//                                              llvm::StringRef filePath) {
//   // TODO: Implement similar to writeDenseToBin
// }
#include "llvm/Support/Endian.h"
#include <type_traits>
#include <cstring>

namespace {
template <typename T>
static inline void writeOneLE(llvm::raw_ostream &os, T v) {
  if constexpr (sizeof(T) == 1) {
    os.write(reinterpret_cast<const char *>(&v), 1);
  } else if constexpr (sizeof(T) == 2) {
    // 16-bit ints
    uint16_t w = static_cast<uint16_t>(v);
    llvm::support::ulittle16_t le(w);
    os.write(reinterpret_cast<const char *>(&le), sizeof(le));
  } else if constexpr (sizeof(T) == 4) {
    // 32-bit: ints or float bits
    uint32_t w;
    static_assert(std::is_trivially_copyable_v<T>, "must be POD");
    std::memcpy(&w, &v, 4);
    llvm::support::ulittle32_t le(w);
    os.write(reinterpret_cast<const char *>(&le), sizeof(le));
  } else if constexpr (sizeof(T) == 8) {
    // 64-bit: ints or double bits
    uint64_t w;
    static_assert(std::is_trivially_copyable_v<T>, "must be POD");
    std::memcpy(&w, &v, 8);
    llvm::support::ulittle64_t le(w);
    os.write(reinterpret_cast<const char *>(&le), sizeof(le));
  }
}

template <typename ResAttrT, typename ElemT>
static inline LogicalResult writeResArray(llvm::raw_fd_ostream &os, ResAttrT res) {
  if (auto arr = res.tryGetAsArrayRef()) {
    for (ElemT v : *arr) writeOneLE(os, v);
    return success();
  }
  return failure(); // resource blob not materialized
}
} // namespace

static LogicalResult writeDenseResourceToBin(DenseResourceElementsAttr res,
                                             llvm::StringRef filePath) {
  std::error_code ec;
  llvm::raw_fd_ostream os(filePath, ec, llvm::sys::fs::OF_None);
  if (ec) return failure();

  ShapedType st = res.getType().cast<ShapedType>();
  Type elemTy = st.getElementType();

  // Handle all supported resource-backed element specializations.
  // Signed integers
  if (auto e = dyn_cast<DenseI8ResourceElementsAttr>(res))
    return writeResArray<DenseI8ResourceElementsAttr,  int8_t>(os, e);
  if (auto e = dyn_cast<DenseI16ResourceElementsAttr>(res))
    return writeResArray<DenseI16ResourceElementsAttr, int16_t>(os, e);
  if (auto e = dyn_cast<DenseI32ResourceElementsAttr>(res))
    return writeResArray<DenseI32ResourceElementsAttr, int32_t>(os, e);
  if (auto e = dyn_cast<DenseI64ResourceElementsAttr>(res))
    return writeResArray<DenseI64ResourceElementsAttr, int64_t>(os, e);

  // Unsigned integers
  if (auto e = dyn_cast<DenseUI8ResourceElementsAttr>(res))
    return writeResArray<DenseUI8ResourceElementsAttr,  uint8_t>(os, e);
  if (auto e = dyn_cast<DenseUI16ResourceElementsAttr>(res))
    return writeResArray<DenseUI16ResourceElementsAttr, uint16_t>(os, e);
  if (auto e = dyn_cast<DenseUI32ResourceElementsAttr>(res))
    return writeResArray<DenseUI32ResourceElementsAttr, uint32_t>(os, e);
  if (auto e = dyn_cast<DenseUI64ResourceElementsAttr>(res))
    return writeResArray<DenseUI64ResourceElementsAttr, uint64_t>(os, e);

  // Floats
  if (auto e = dyn_cast<DenseF32ResourceElementsAttr>(res))
    return writeResArray<DenseF32ResourceElementsAttr, float>(os, e);
  if (auto e = dyn_cast<DenseF64ResourceElementsAttr>(res))
    return writeResArray<DenseF64ResourceElementsAttr, double>(os, e);

  // Bool (packed as 1 byte per element here)
  if (auto e = dyn_cast<DenseBoolResourceElementsAttr>(res))
    return writeResArray<DenseBoolResourceElementsAttr, bool>(os, e);

  // If we reach here, the element type is not one of the supported resource specializations.
  // You can add bf16/f16 when your drop exposes them as resource types.
  (void)elemTy;
  return failure();
}



/// Write either DenseElementsAttr or DenseResourceElementsAttr to little-endian .bin.
static LogicalResult writeElementsToBin(ElementsAttr elems,
                                        llvm::StringRef filePath) {
  // Fast path: DenseElementsAttr (existing code unchanged).
  if (auto dense = elems.dyn_cast<DenseElementsAttr>()) {
    return writeDenseToBin(dense, filePath);
  }else if (auto res = elems.dyn_cast<DenseResourceElementsAttr>()) {
    return writeDenseResourceToBin(res, filePath);
  }

  return failure(); // Not a dense elements kind.
}

/// Convenience: create/ensure directory.
static LogicalResult ensureDir(llvm::StringRef dir) {
  if (dir.empty()) return failure();
  if (llvm::sys::fs::exists(dir)) return success();
  return llvm::sys::fs::create_directories(dir) ? failure() : success();
}

// /// Collect all constant memref.globals with DenseElements initializers.
// /// Returns map: symbol -> (op, DenseElementsAttr)
// static llvm::DenseMap<StringRef, std::pair<memref::GlobalOp, DenseElementsAttr>>
// collectConstGlobals(ModuleOp module) {
//   llvm::DenseMap<StringRef, std::pair<memref::GlobalOp, DenseElementsAttr>> M;
//   module.walk([&](memref::GlobalOp g) {
//     // We only handle globals with DenseElementsAttr initializer.
//     if (!g.getInitialValue().has_value())
//       return;
//     Attribute init = *g.getInitialValue();
//     if (auto dense = init.dyn_cast<DenseElementsAttr>())
//       M.try_emplace(g.getSymName(), std::make_pair(g, dense));
//   });
//   return M;
// }

/// Returns: symbol -> (op, ElementsAttr)
static llvm::DenseMap<StringRef, std::pair<memref::GlobalOp, ElementsAttr>>
collectConstGlobals(ModuleOp module) {
  llvm::DenseMap<StringRef, std::pair<memref::GlobalOp, ElementsAttr>> M;
  module.walk([&](memref::GlobalOp g) {
    if (!g.getInitialValue().has_value()) return;
    Attribute init = *g.getInitialValue();
    if (auto el = init.dyn_cast<ElementsAttr>())
      M.try_emplace(g.getSymName(), std::make_pair(g, el));
  });
  return M;
}


} // namespace



namespace mlir {
namespace streamhls {

/// binarizeWeights:
/// 1) Dump each constant memref.global to <weightBinsPath>/<symbol>.bin
/// 2) For each func using memref.get_global @sym:
///    - add a new func arg of type memref<...> (same as the global)
///    - set buffer.info = {name = "<sym>", type = "weights"} on that arg
///    - replace the get_global result with that arg and erase the op
/// 3) Erase the processed memref.global ops from the module.
/// Returns the (possibly modified) module.
ModuleOp binarizeWeights(ModuleOp &module, llvm::StringRef weightBinsPath) {
  MLIRContext *ctx = module.getContext();
  const StringRef kAttrKey = "buffer.info";

  // 0) Collect globals we can binarize.
  auto globMap = collectConstGlobals(module);

  if (globMap.empty())
    return module;

  // Ensure destination directory exists.
  (void)ensureDir(weightBinsPath);

  // 1) Write each to disk as <sym>.bin
  // for (auto &it : globMap) {
  //   StringRef sym = it.first;
  //   DenseElementsAttr dense = it.second.second;

  //   llvm::SmallString<256> outPath(weightBinsPath);
  //   llvm::sys::path::append(outPath, (sym + ".bin").str());

  //   if (failed(writeDenseToBin(dense, outPath))) {
  //     // Non-fatal: skip file write errors, but continue rewriting IR.
  //     // You can choose to signal error instead.
  //     llvm::errs() << "Warning: failed to write weights for @" << sym
  //                  << " to " << outPath << "\n";
  //   }
  // }
  for (auto &it : globMap) {
    StringRef sym = it.first;
    ElementsAttr el = it.second.second;

    llvm::SmallString<256> outPath(weightBinsPath);
    llvm::sys::path::append(outPath, (sym + ".bin").str());

    if (failed(writeElementsToBin(el, outPath))) {
      llvm::errs() << "Warning: failed to write weights for @" << sym
                  << " to " << outPath << "\n";
    }
  }

  // 2) For each function, add args for distinct used globals and rewrite.
  module.walk([&](func::FuncOp func) {
    // Gather distinct symbol uses in deterministic order of appearance.
    llvm::SmallVector<std::pair<StringRef, MemRefType>, 4> usedSymsOrdered;
    llvm::DenseMap<StringRef, int> symToLocalIndex;

    func.walk([&](memref::GetGlobalOp get) {
      StringRef sym = get.getName(); // flat symbol name
      // Only rewrite if this symbol is one of the binarized globals.
      auto it = globMap.find(sym);
      if (it == globMap.end())
        return;

      if (!symToLocalIndex.count(sym)) {
        auto globalOp = it->second.first;
        auto ty = globalOp.getType().cast<MemRefType>();
        symToLocalIndex[sym] = (int)usedSymsOrdered.size();
        usedSymsOrdered.emplace_back(sym, ty);
      }
    });

    if (usedSymsOrdered.empty())
      return;

    // Append new arguments to entry block (values first, then adjust func type).
    Block &entry = func.front();
    OpBuilder b(func.getContext());
    Location loc = func.getLoc();

    // Existing inputs & results.
    FunctionType fty = func.getFunctionType();
    llvm::SmallVector<Type, 8> newInputs(fty.getInputs().begin(),
                                         fty.getInputs().end());
    llvm::SmallVector<Type, 4> results(fty.getResults().begin(),
                                       fty.getResults().end());

    // Map from symbol -> new BlockArgument value.
    llvm::DenseMap<StringRef, BlockArgument> symToArgVal;

    // Add block arguments first so we can use them in rewrites.
    for (auto &p : usedSymsOrdered) {
      StringRef sym = p.first;
      MemRefType ty = p.second;
      BlockArgument arg = entry.addArgument(ty, loc);
      symToArgVal[sym] = arg;

      // Prepare to update the function type too.
      newInputs.push_back(ty);
    }

    // Update function type.
    auto newFty = FunctionType::get(func.getContext(), newInputs, results);
    func.setType(newFty);

    // Attach buffer.info to each newly added argument.
    int baseIdx = (int)fty.getNumInputs();
    for (int i = 0; i < (int)usedSymsOrdered.size(); ++i) {
      StringRef sym = usedSymsOrdered[i].first;
      auto dict = makeInfoAttr(ctx, sym.str(), /*type*/ "weights", "read-only");
      func.setArgAttr(baseIdx + i, kAttrKey, dict);
      // // New PortIOAttr
      // func.setArgAttr(
      //   baseIdx + i, 
      //   "dataflow.port_io", 
      //   dataflow::PortIOAttr::get(ctx, dataflow::IOKind::read_only));
    }

    // Now replace all get_global uses with the corresponding new arg.
    llvm::SmallVector<Operation *, 8> toErase;
    func.walk([&](memref::GetGlobalOp get) {
      StringRef sym = get.getName();
      auto si = symToArgVal.find(sym);
      if (si == symToArgVal.end())
        return; // unrelated symbol
      Value repl = si->second;
      get.getResult().replaceAllUsesWith(repl);
      toErase.push_back(get.getOperation());
    });
    for (Operation *op : toErase)
      op->erase();
  });

  // 3) Erase the globals we handled (their users are gone now).
  //    If any still have users (e.g., not all rewritten), we keep them.
  llvm::SmallVector<memref::GlobalOp, 8> toEraseGlobals;
  for (auto &it : globMap) {
    memref::GlobalOp g = it.second.first;
    if (g.use_empty())
      toEraseGlobals.push_back(g);
  }
  for (auto g : toEraseGlobals)
    g.erase();

  return module;
}

ModuleOp annotateBuffers(ModuleOp &module, std::string projectPath) {
  MLIRContext *ctx = module.getContext();
  // All attributes land under this key.
  const StringRef kAttrKey = "buffer.info";

  // Track off-chip argument order for top function
  // Store as: (argument_index, name)
  llvm::SmallVector<std::pair<unsigned, std::string>> topOffChipArgOrder;
  llvm::SmallVector<std::pair<unsigned, std::string>> topAllArgOrder; // ALL arguments (on-chip + off-chip)
  bool topFuncFound = false;
  
  // Track argument order for golden reference function (e.g., "forward")
  // Store as: (argument_index, name)
  llvm::SmallVector<std::pair<unsigned, std::string>> goldenRefArgOrder;
  bool goldenRefFuncFound = false;

  module.walk([&](func::FuncOp func) {
    // Per-function numbering for unnamed allocs.
    int allocCounter = 0;

    // 1) Function arguments of memref type:
    //    - If they already have buffer.info, use it.
    //    - Otherwise create {name=<arg name>, type="off-chip"}.
    //    - For top function, track off-chip argument order by argument index
    //    - For golden reference function (e.g., "forward"), also track argument order
    bool isTopFunc = !topFuncFound && (func.getSymName() == "top" || func.getSymName().starts_with("top_"));
    if (isTopFunc) {
      topFuncFound = true;
      topOffChipArgOrder.clear(); // Clear in case we're re-processing
      topAllArgOrder.clear(); // Clear all args too
    }
    
    // Check if this is a golden reference function (e.g., "forward" or marked with attribute)
    bool isGoldenRefFunc = !goldenRefFuncFound && (
      func.getSymName() == "forward" || 
      func.getSymName() == "golden_reference" ||
      func->hasAttr("streamhls.golden_reference") ||
      func->hasAttr("golden_reference")
    );
    if (isGoldenRefFunc) {
      goldenRefFuncFound = true;
      goldenRefArgOrder.clear(); // Clear in case we're re-processing
    }
    
    for (BlockArgument arg : func.getArguments()) {
      if (!isa<MemRefType>(arg.getType()))
        continue;
      unsigned aidx = arg.getArgNumber();
      if (auto existing =
              func.getArgAttrOfType<DictionaryAttr>(aidx, kAttrKey)) {
        
        // Track ALL argument order for top function (by argument index)
        if (isTopFunc) {
          StringAttr nameAttr = existing.getAs<StringAttr>("name");
          StringAttr typeAttr = existing.getAs<StringAttr>("type");
          if (nameAttr) {
            topAllArgOrder.push_back({aidx, nameAttr.getValue().str()});
            // Also track off-chip separately
            if (typeAttr && (typeAttr.getValue() == "off-chip" || 
                             typeAttr.getValue() == "weights")) {
              topOffChipArgOrder.push_back({aidx, nameAttr.getValue().str()});
            }
          }
        }
        
        // Track ALL argument order for golden reference function
        if (isGoldenRefFunc) {
          StringAttr nameAttr = existing.getAs<StringAttr>("name");
          if (nameAttr) {
            goldenRefArgOrder.push_back({aidx, nameAttr.getValue().str()});
          }
        }
      } else {
        std::string name = nameForBlockArg(func, arg);
        // if arg has a single use
        bool isRead = false;
        bool isWrite = false;
        StringRef ioKind = "unknown";
        for(auto user : arg.getUsers()){
          // check if the user is an affine.load or affine.store
          if (isa<affine::AffineLoadOp>(user)){
            isRead = true;
          }else if (isa<affine::AffineStoreOp>(user)){
            isWrite = true;
          }
        }
        if(isRead && isWrite){
          // read-write
          ioKind = "read-write";
        }else if(isRead && !isWrite){
          // read-only
          ioKind = "read-only";
        }else if(!isRead && isWrite){
          // write-only
          ioKind = "write-only";
        }
        auto dict = makeInfoAttr(ctx, name, "off-chip", ioKind);
        func.setArgAttr(aidx, kAttrKey, dict);     // attach once

        // Track ALL argument order for top function (by argument index)
        if (isTopFunc) {
          topAllArgOrder.push_back({aidx, name});
          topOffChipArgOrder.push_back({aidx, name});
        }
        
        // Track ALL argument order for golden reference function (by argument index)
        if (isGoldenRefFunc) {
          goldenRefArgOrder.push_back({aidx, name});
        }
      }
    }

    // 2) memref.alloc results => classify by number of distinct enclosing affine.for uses
    func.walk([&](memref::AllocOp alloc) {
      Value buf = alloc.getResult();
      // Determine a unique, human-friendly name for this alloc.
      std::string name = "buf" + std::to_string(allocCounter++);

      llvm::SmallPtrSet<Operation *, 8> distinctFors;
      collectDistinctEnclosingAffineFors(buf, distinctFors);

      StringRef cls;
      if (distinctFors.size() <= 1)
        cls = "local-buffer";
      else
        cls = "shared-buffer";

      auto dict = makeInfoAttr(ctx, name, cls);
      // Attach to the alloc op itself.
      alloc->setAttr(kAttrKey, dict);
    });

    LLVM_DEBUG(debugReportAllocLoadStoreCountsNoAliases(func));

    // memref.get_global → weights
    func.walk([&](memref::GetGlobalOp g) {
      std::string name = g.getName().str(); // symbol reference like "__constant_256x784xf32"
      auto dict = makeInfoAttr(ctx, name, "weights");
      g->setAttr(kAttrKey, dict);
    });
  });
  
  return module;
}

ModuleOp setPortWidthsToOne(ModuleOp &module) {
  // Assume single top-level func.func in the module.
  for (auto func : module.getOps<func::FuncOp>()) {
    for (auto [idx, arg] : llvm::enumerate(func.getArguments())) {
      // Only annotate memref arguments.
      if (!isa<MemRefType>(arg.getType()))
        continue;
      // Attach dataflow.port_width<1>.
      setPortWidthAttr(func, idx, /*lanes=*/1);
    }
  }
  return module;
}

} // namespace streamhls
} // namespace mlir