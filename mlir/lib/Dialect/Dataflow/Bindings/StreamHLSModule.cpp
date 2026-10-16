/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "streamhls/Dialect/Dataflow/Bindings/StreamHLSModule.h"
#include "streamhls-c/Dialect/Dataflow/IR/Dataflow.h"
#include "streamhls-c/Dialect/Registeration.h"
#include "streamhls/Dialect/Dataflow/IR/Dataflow.h"
#include "streamhls/Dialect/Dataflow/Transforms/Passes.h"
#include "streamhls/Dialect/Dataflow/Support/LoopPermutationModel.h"
#include "streamhls/Dialect/Dataflow/Support/LoopUnrollingModel.h"
#include "streamhls/Dialect/Dataflow/Support/LoopCombinedModel.h"

// #include "streamhls/Dialect/Dataflow/Translation/EmitHLS.h"
#include "streamhls-c/Dialect/Dataflow/Translation/EmitVitisHLS.h"
#include "streamhls-c/Dialect/Dataflow/Translation/EmitTapaHLS.h"

// #include "streamhls/Dialect/Dataflow/Support/Utils.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm-c/ErrorHandling.h"
#include "llvm/Support/Signals.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <mutex>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace mlir::python::adaptors;

using namespace mlir;
using namespace mlir::python;
using namespace streamhls;

//===----------------------------------------------------------------------===//
// Customized Python classes
//===----------------------------------------------------------------------===//

// PybindUtils.h
class PyFileAccumulator {
public:
  PyFileAccumulator(const pybind11::object &fileObject, bool binary)
      : pyWriteFunction(fileObject.attr("write")), binary(binary) {}

  void *getUserData() { return this; }

  MlirStringCallback getCallback() {
    return [](MlirStringRef part, void *userData) {
      pybind11::gil_scoped_acquire acquire;
      PyFileAccumulator *accum = static_cast<PyFileAccumulator *>(userData);
      if (accum->binary) {
        // Note: Still has to copy and not avoidable with this API.
        pybind11::bytes pyBytes(part.data, part.length);
        accum->pyWriteFunction(pyBytes);
      } else {
        pybind11::str pyStr(part.data,
                            part.length); // Decodes as UTF-8 by default.
        accum->pyWriteFunction(pyStr);
      }
    };
  }

private:
  pybind11::object pyWriteFunction;
  bool binary;
};

//===----------------------------------------------------------------------===//
// Enable Debugging Support
//===----------------------------------------------------------------------===//

static std::once_flag LLVMDebugOnce;

static void enableLLVMDebugOnlyOnce(const std::vector<std::string> &debugTypes,
                                   bool enableAll = false) {
  std::call_once(LLVMDebugOnce, [&]() {
    std::vector<std::string> args;
    args.push_back("streamhls_py");

    if (enableAll || debugTypes.empty()) {
      args.push_back("-debug");
    } else {
      for (const auto &t : debugTypes) {
        if (!t.empty())
          args.push_back("-debug-only=" + t);
      }
    }

    std::vector<const char *> argv;
    argv.reserve(args.size());
    for (auto &a : args)
      argv.push_back(a.c_str());

    int argc = (int)argv.size();
    llvm::cl::ParseCommandLineOptions(argc, argv.data(), "StreamHLS LLVM Debug");
  });
}

//===----------------------------------------------------------------------===//
// StreamHLS Kernel Pipeline APIs
//===----------------------------------------------------------------------===//
static MlirModule lowerLinalgToAffine_api(MlirModule &mlir_mod, unsigned debugPoint = 0) {
  auto mod = unwrap(mlir_mod);
  return wrap(lowerLinalgToAffine(mod, debugPoint));
}

// //===----------------------------------------------------------------------===//
// // MergeKernels APIs
// //===----------------------------------------------------------------------===//
static MlirModule mergeKernels_api(std::vector<MlirModule> &mlir_mods, std::unordered_map<std::string, VarType> &options) {
  SmallVector<ModuleOp> mods;
  for (auto mlir_mod : mlir_mods) {
    mods.push_back(unwrap(mlir_mod));
  }
  return wrap(mergeKernels(mods, options));
}

static void mergeKernelsDFG_api(DFG &dfg, std::unordered_map<std::string, VarType> &options) {
  mergeKernelsDFG(dfg, options);
}

static void convertFuncMemrefsToFIFOsDFG_api(DFG &dfg) {
  convertFuncMemrefsToFIFOsDFG(dfg);
}

static void reuseHardwareDFG_api(DFG &dfg) {
  reuseHardwareDFG(dfg);
}

static void mergeSuperNodesDFG_api(DFG &dfg) {
  dfg.mergeSuperNodesDFG();
}

static void wrapSuperNodesDFG_api(DFG &dfg) {
  dfg.wrapSuperNodesDFG();
}

static MlirModule reuseHardware_api(MlirModule &mlir_mod) {
  ModuleOp mod = unwrap(mlir_mod);
  return wrap(reuseHardware(mod));
}

static HWMapping runHighLevelDSE_api(MlirModule &mlir_mod, std::string projectPath, std::string mlgPath, const DSEOptions &options, std::string mlgType = "no_reuse") {
  auto mod = unwrap(mlir_mod);
  return runHighLevelDSE(mod, projectPath, mlgPath, options, mlgType);
}

static DFG extractMergedDFG_api(
  MlirModule &mlir_mod,
  HWMapping hwMapping,
  bool enableTiling = true
) {
  auto mod = unwrap(mlir_mod);
  return extractMergedDFG(mod, hwMapping, enableTiling);
}

static HWMapping HardwareMappingDSE_api(
  MlirModule &mlir_mod,
  std::string projectPath,
  const DSEOptions &options,
  std::string mlgType = "no_reuse"
) {
  auto mod = unwrap(mlir_mod);
  return HardwareMappingDSE(mod, projectPath, options, mlgType);
}

static std::string printDFGNodes_api(DFG &graph) {
  return printDFGNodes(graph);
}

static std::string printDFGEdges_api(DFG &graph) {
  return printDFGEdges(graph);
}

static std::string printDFGAsDot_api(DFG &graph) {
  return printDFGAsDot(graph);
}

static MlirModule generateSystemModule_api(DFG &graph, MlirModule &mlir_template) {
  auto templateMod = unwrap(mlir_template);
  return wrap(generateSystemModule(graph, templateMod));
}

static MlirModule annotateBuffers_api(MlirModule &mlir_mod, std::string projectPath) {
  auto mod = unwrap(mlir_mod);
  return wrap(annotateBuffers(mod, projectPath));
}
static MlirModule annotateLoops_api(MlirModule &mlir_mod) {
  auto mod = unwrap(mlir_mod);
  return wrap(annotateLoops(mod));
}
static MlirModule setPortWidthsToOne_api(MlirModule &mlir_mod) {
  auto mod = unwrap(mlir_mod);
  return wrap(setPortWidthsToOne(mod));
}
static MlirModule binarizeWeights_api(MlirModule &mlir_mod, std::string weightBinsPath) {
  auto mod = unwrap(mlir_mod);
  llvm::StringRef pathRef(weightBinsPath);
  return wrap(binarizeWeights(mod, pathRef));
}

static MlirModule annotateMemrefs_api(MlirModule &mlir_mod) {
  auto mod = unwrap(mlir_mod);
  return wrap(annotateMemrefs(mod));
}

static MlirModule bufferizeFuncArgs_api(MlirModule &mlir_mod) {
  auto mod = unwrap(mlir_mod);
  return wrap(bufferizeFuncArgs(mod));
}

static MlirModule eliminateSingleWriteMultipleReads_api(MlirModule &mlir_mod, std::string eliminationType = "skip") {
  auto mod = unwrap(mlir_mod);
  return wrap(eliminateSingleWriteMultipleReads(mod, eliminationType));
}

static MlirModule convertMemrefArgsToFIFOs_api(MlirModule &mlir_mod) {
  auto mod = unwrap(mlir_mod);
  DenseSet<unsigned> allArgs;
  // If we want the old "convert everything" behavior, we'd need to gather arg indices.
  // For now, let's just make it compatible.
  return wrap(convertMemrefArgsToFIFOs(mod, allArgs));
}


static MlirModule eleminateUnitDimsAndIters_api(MlirModule &mlir_mod) {
  auto mod = unwrap(mlir_mod);
  return wrap(eleminateUnitDimsAndIters(mod));
}

static MlirModule flattenArrayOfStreams_api(MlirModule &mlir_mod) {
  auto mod = unwrap(mlir_mod);
  return wrap(flattenArrayOfStreams(mod));
}

static MlirModule arrayStreamsToVectorStream_api(MlirModule &mlir_mod) {
  auto mod = unwrap(mlir_mod);
  return wrap(arrayStreamsToVectorStream(mod));
}

static std::pair<MlirModule, MlirModule> rewriteMemoryModulesToDMA_api(MlirModule &mlir_mod) {
  ModuleOp mod = unwrap(mlir_mod);
  auto result = rewriteMemoryModulesToDMA(mod);
  return {wrap(result.first), wrap(result.second)};
}

static MlirModule raiseInnerLoopByOneLevel_api(MlirModule &mlir_mod) {
  auto mod = unwrap(mlir_mod);
  return wrap(raiseInnerLoopByOneLevel(mod));
}

static SolveResult runLoopPermutationMINLP_api(DFG &dfg, std::string projectPath, const DSEOptions &options) {
  return runLoopPermutationMINLP(dfg, projectPath, options);
}

static SolveResult findRandomUnrollingSolution_api(DFG &dfg, const DSEOptions &options) {
  return findRandomUnrollingSolution(dfg, options);
}

static SolveResult findRandomPermutationSolution_api(DFG &dfg, const DSEOptions &options) {
  return findRandomPermutationSolution(dfg, options);
}

static SolveResult findRandomCombinedSolution_api(DFG &dfg, const DSEOptions &options) {
  return findRandomCombinedSolution(dfg, options);
}

static bool setUnrollingFactorsToOnes_api(DFG &dfg) {
  setUnrollingFactorsToOnes(dfg);
  return true;
}

static SolveResult runLoopUnrollingMINLP_api(DFG &dfg, std::string projectPath, const DSEOptions &options) {
  return runLoopUnrollingMINLP(dfg, projectPath, options);
}

static SolveResult runLoopCombinedMINLP_api(DFG &dfg, std::string projectPath, const DSEOptions &options) {
  return runLoopCombinedMINLP(dfg, projectPath, options);
}

static bool setFixedPermutations_api(DFG &dfg, const DSEOptions &options) {
  return setFixedPermutations(dfg, options);
}

static bool setFixedUnrollingFactors_api(DFG &dfg, const DSEOptions &options) {
  return setFixedUnrollingFactors(dfg, options);
}
//===----------------------------------------------------------------------===//
// Emission APIs
//===----------------------------------------------------------------------===//

static bool emitVitisHls(MlirModule &mod, py::object fileObject, bool useHlsTask = false) {
  PyFileAccumulator accum(fileObject, false);
  py::gil_scoped_release();
  return mlirLogicalResultIsSuccess(
      mlirEmitVitisHls(mod, accum.getCallback(), accum.getUserData(), useHlsTask));
}

static bool emitTapaHLS(MlirModule &mod, py::object fileObject) {
  PyFileAccumulator accum(fileObject, false);
  py::gil_scoped_release();
  return mlirLogicalResultIsSuccess(
      mlirEmitTapa(mod, accum.getCallback(), accum.getUserData()));
}


//===----------------------------------------------------------------------===//
// Allo Python module definition
//===----------------------------------------------------------------------===//

PYBIND11_MODULE(_streamhls, m) {
  m.doc() = "Stream-HLS Python Native Extension";
  llvm::sys::PrintStackTraceOnErrorSignal(/*argv=*/"");
  LLVMEnablePrettyStackTrace();

  // register passes
  streamhlsMlirRegisterAllPasses();
  auto streamhls_m = m.def_submodule("streamhls");

  // register dialects
  streamhls_m.def(
      "register_dialect",
      [](MlirContext context) {
        MlirDialectHandle streamhls = mlirGetDialectHandle__streamhls__();
        mlir::DialectRegistry registry;
        unwrap(context)->appendDialectRegistry(registry);
        mlirDialectHandleRegisterDialect(streamhls, context);
        mlirDialectHandleLoadDialect(streamhls, context);
        streamhlsMlirRegisterAllDialects(context);
      },
      py::arg("context") = py::none());

  streamhls_m.def(
    "enable_llvm_debug_only",
    [](py::list debug_types) {
      std::vector<std::string> types;
      types.reserve(py::len(debug_types));
      for (auto item : debug_types)
        types.push_back(item.cast<std::string>());
      enableLLVMDebugOnlyOnce(types, /*enableAll=*/false);
    },
    py::arg("debug_types")
  );

  streamhls_m.def(
    "enable_llvm_debug_all",
    []() {
      enableLLVMDebugOnlyOnce(/*debugTypes=*/{}, /*enableAll=*/true);
    }
  );

  py::class_<mlir::streamhls::SAConfig>(streamhls_m, "SAConfig")
    .def(py::init<>())
    .def_readwrite("initial_temp", &mlir::streamhls::SAConfig::initialTemp)
    .def_readwrite("cooling_rate", &mlir::streamhls::SAConfig::coolingRate)
    .def_readwrite("iterations", &mlir::streamhls::SAConfig::iterations)
    .def_readwrite("initial_solution", &mlir::streamhls::SAConfig::initialSolution)
    .def_readwrite("csv_path", &mlir::streamhls::SAConfig::csvPath);

  py::class_<mlir::streamhls::GAConfig>(streamhls_m, "GAConfig")
    .def(py::init<>())
    .def_readwrite("pop_size", &mlir::streamhls::GAConfig::popSize)
    .def_readwrite("generations", &mlir::streamhls::GAConfig::generations)
    .def_readwrite("mutation_rate", &mlir::streamhls::GAConfig::mutationRate)
    .def_readwrite("crossover_rate", &mlir::streamhls::GAConfig::crossoverRate)
    .def_readwrite("init_method", &mlir::streamhls::GAConfig::initMethod)
    .def_readwrite("csv_path", &mlir::streamhls::GAConfig::csvPath);

  py::class_<mlir::streamhls::RandomConfig>(streamhls_m, "RandomConfig")
    .def(py::init<>())
    .def_readwrite("iterations", &mlir::streamhls::RandomConfig::iterations);

  py::class_<mlir::streamhls::SolveResult>(streamhls_m, "SolveResult")
    .def(py::init<>())
    .def_readwrite("success", &mlir::streamhls::SolveResult::success)
    .def_readwrite("latency", &mlir::streamhls::SolveResult::latency)
    .def_readwrite("memory", &mlir::streamhls::SolveResult::memory)
    .def_readwrite("dsps", &mlir::streamhls::SolveResult::dsps)
    .def_readwrite("cost", &mlir::streamhls::SolveResult::cost)
    .def_readwrite("variables", &mlir::streamhls::SolveResult::variables)
    .def_readwrite("full_output", &mlir::streamhls::SolveResult::fullOutput)
    .def("print_report", [](mlir::streamhls::SolveResult &self, std::string label) {
      self.printReport(llvm::dbgs(), label);
    });

  py::class_<mlir::streamhls::DSEOptions>(streamhls_m, "DSEOptions")
    .def(py::init<>())
    .def_readwrite("permutation_dse", &mlir::streamhls::DSEOptions::permutationDSE)
    .def_readwrite("unrolling_dse", &mlir::streamhls::DSEOptions::unrollingDSE)
    .def_readwrite("combined_dse", &mlir::streamhls::DSEOptions::combinedDSE)
    .def_readwrite("permutation_timeout", &mlir::streamhls::DSEOptions::permutationTimeout)
    .def_readwrite("unrolling_timeout", &mlir::streamhls::DSEOptions::unrollingTimeout)
    .def_readwrite("combined_timeout", &mlir::streamhls::DSEOptions::combinedTimeout)
    .def_readwrite("permutation_ampl_solver", &mlir::streamhls::DSEOptions::permutationAMPLSolver)
    .def_readwrite("unrolling_ampl_solver", &mlir::streamhls::DSEOptions::unrollingAMPLSolver)
    .def_readwrite("combined_ampl_solver", &mlir::streamhls::DSEOptions::combinedAMPLSolver)
    .def_readwrite("permutation_sa_config", &mlir::streamhls::DSEOptions::permutationSAConfig)
    .def_readwrite("unrolling_sa_config", &mlir::streamhls::DSEOptions::unrollingSAConfig)
    .def_readwrite("combined_sa_config", &mlir::streamhls::DSEOptions::combinedSAConfig)
    .def_readwrite("permutation_ga_config", &mlir::streamhls::DSEOptions::permutationGAConfig)
    .def_readwrite("unrolling_ga_config", &mlir::streamhls::DSEOptions::unrollingGAConfig)
    .def_readwrite("combined_ga_config", &mlir::streamhls::DSEOptions::combinedGAConfig)
    .def_readwrite("permutation_random_config", &mlir::streamhls::DSEOptions::permutationRandomConfig)
    .def_readwrite("unrolling_random_config", &mlir::streamhls::DSEOptions::unrollingRandomConfig)
    .def_readwrite("combined_random_config", &mlir::streamhls::DSEOptions::combinedRandomConfig)
    .def_readwrite("min_time_loop_size", &mlir::streamhls::DSEOptions::minTimeLoopSize)
    .def_readwrite("enforce_divisor_only_tiling", &mlir::streamhls::DSEOptions::enforceDivisorOnlyTiling)
    .def_readwrite("shared_tiling_strategy", &mlir::streamhls::DSEOptions::sharedTilingStrategy)
    .def_readwrite("dsp_limit", &mlir::streamhls::DSEOptions::dspLimit)
    .def_readwrite("memory_limit", &mlir::streamhls::DSEOptions::memoryLimit)
    .def_readwrite("latency_memory_ratio", &mlir::streamhls::DSEOptions::latencyMemoryRatio)
    .def_readwrite("fixed_permutation_solution", &mlir::streamhls::DSEOptions::fixedPermutationSolution)
    .def_readwrite("fixed_unrolling_solution", &mlir::streamhls::DSEOptions::fixedUnrollingSolution)
    .def_readwrite("balanced_mapping_similarity_threshold", &mlir::streamhls::DSEOptions::balancedMappingSimilarityThreshold)
    .def_readwrite("balanced_mapping_ub_threshold", &mlir::streamhls::DSEOptions::balancedMappingUBThreshold)
    .def_readwrite("objective_scaling_factor", &mlir::streamhls::DSEOptions::objectiveScalingFactor);

  py::class_<mlir::streamhls::HWMapping>(streamhls_m, "HWMapping")
    .def(py::init<>())
    .def_readwrite("hw_mapping_strategy", &mlir::streamhls::HWMapping::hwMappingStrategy)
    .def_readwrite("mapping", &mlir::streamhls::HWMapping::mapping);

  py::class_<mlir::streamhls::DFG::Node>(streamhls_m, "DFGNode")
    .def_readonly("id", &mlir::streamhls::DFG::Node::id)
    .def_readonly("type", &mlir::streamhls::DFG::Node::type)
    .def_readonly("is_reduction", &mlir::streamhls::DFG::Node::isReduction)
    .def_readonly("kind", &mlir::streamhls::DFG::Node::kind)
    .def_property_readonly("tiling_factors", [](mlir::streamhls::DFG::Node &self) {
      return self.solution.tilingFactors;
    })
    .def_property_readonly("perm_idx", [](mlir::streamhls::DFG::Node &self) {
      return self.solution.permIdx;
    })
    .def_property_readonly("merged_module", [](mlir::streamhls::DFG::Node &self) {
      return wrap(self.mergedModule);
    })
    .def_property_readonly("fifo_module", [](mlir::streamhls::DFG::Node &self) {
      return wrap(self.fifoModule);
    })
    .def_property_readonly("reuse_module", [](mlir::streamhls::DFG::Node &self) {
      return wrap(self.reuseModule);
    })
    .def_property_readonly("hw_module", [](mlir::streamhls::DFG::Node &self) {
      return wrap(self.hwModule);
    });

  auto edgeKind = py::enum_<mlir::streamhls::DFG::EdgeKind>(streamhls_m, "EdgeKind");
  edgeKind.value("STREAM", mlir::streamhls::DFG::EdgeKind::STREAM)
    .value("AOS", mlir::streamhls::DFG::EdgeKind::AOS)
    .value("MEMREF", mlir::streamhls::DFG::EdgeKind::MEMREF)
    .export_values();

  auto nodeKind = py::enum_<mlir::streamhls::DFG::NodeKind>(streamhls_m, "NodeKind");
  nodeKind.value("UNKNOWN", mlir::streamhls::DFG::NodeKind::UNKNOWN)
    .value("COMPUTE", mlir::streamhls::DFG::NodeKind::COMPUTE)
    .value("MEMORY", mlir::streamhls::DFG::NodeKind::MEMORY)
    .value("CONVERTER", mlir::streamhls::DFG::NodeKind::CONVERTER)
    .value("SUPER_NODE", mlir::streamhls::DFG::NodeKind::SUPER_NODE)
    .export_values();

  py::class_<mlir::streamhls::DFG::Edge>(streamhls_m, "DFGEdge")
    .def_readonly("id", &mlir::streamhls::DFG::Edge::id)
    .def_readonly("kind", &mlir::streamhls::DFG::Edge::kind)
    .def_property_readonly("value", [](mlir::streamhls::DFG::Edge &self) {
       return wrap(self.value);
    })
    .def_property_readonly("resolved_type", [](mlir::streamhls::DFG::Edge &self) {
       return wrap(self.resolvedType);
    });

  py::class_<mlir::streamhls::DFG>(streamhls_m, "DFG")
    .def(py::init<mlir::Block &>())
    .def_property_readonly("nodes", [](mlir::streamhls::DFG &self) {
      std::map<unsigned, mlir::streamhls::DFG::Node> result;
      for (auto &pair : self.nodes) {
        result[pair.first] = pair.second;
      }
      return result;
    })
    .def_property_readonly("in_edges", [](mlir::streamhls::DFG &self) {
      std::map<unsigned, std::vector<mlir::streamhls::DFG::Edge>> result;
      for (auto &pair : self.inEdges) {
        result[pair.first] = std::vector<mlir::streamhls::DFG::Edge>(pair.second.begin(), pair.second.end());
      }
      return result;
    })
    .def_property_readonly("out_edges", [](mlir::streamhls::DFG &self) {
      std::map<unsigned, std::vector<mlir::streamhls::DFG::Edge>> result;
      for (auto &pair : self.outEdges) {
        result[pair.first] = std::vector<mlir::streamhls::DFG::Edge>(pair.second.begin(), pair.second.end());
      }
      return result;
    })
    .def("populate_node_info", [](mlir::streamhls::DFG &self, bool enable_permutations) {
      return (bool)mlir::succeeded(self.populateNodeInfo(enable_permutations));
    }, py::arg("enable_permutations") = true)
    .def("get_permutation_solution", &mlir::streamhls::DFG::getPermutationSolution)
    .def("get_tiling_solution", &mlir::streamhls::DFG::getTilingSolution)
    .def("update_permutation_solution", py::overload_cast<const mlir::streamhls::SolveResult &>(&mlir::streamhls::DFG::updatePermutationSolution))
    .def("update_permutation_solution", py::overload_cast<const std::map<unsigned, unsigned> &>(&mlir::streamhls::DFG::updatePermutationSolution))
    .def("update_unrolling_solution", py::overload_cast<const mlir::streamhls::SolveResult &>(&mlir::streamhls::DFG::updateUnrollingSolution))
    .def("update_unrolling_solution", py::overload_cast<const std::map<std::string, int64_t> &>(&mlir::streamhls::DFG::updateUnrollingSolution));
  

  streamhls_m.def("lower_linalg_to_affine", &lowerLinalgToAffine_api);
  
  streamhls_m.def("emit_vitis_hls", &emitVitisHls);

  streamhls_m.def("emit_tapa_hls", &emitTapaHLS);


  streamhls_m.def("merge_kernels", &mergeKernels_api);
  streamhls_m.def("merge_kernels_dfg", &mergeKernelsDFG_api);
  streamhls_m.def("convert_func_memrefs_to_fifos_dfg", &convertFuncMemrefsToFIFOsDFG_api);
  streamhls_m.def("reuse_hardware_dfg", &reuseHardwareDFG_api);
  streamhls_m.def("merge_super_nodes_dfg", &mergeSuperNodesDFG_api);
  streamhls_m.def("wrap_super_nodes_dfg", &wrapSuperNodesDFG_api);

  streamhls_m.def("reuse_hardware", &reuseHardware_api);

  streamhls_m.def("run_high_level_dse", &runHighLevelDSE_api,
    py::arg("module"), py::arg("project_path"), py::arg("mlg_path"), py::arg("options"), py::arg("hw_mapping_strategy") = "no_reuse");

  streamhls_m.def("extract_merged_dfg", &extractMergedDFG_api);

  streamhls_m.def("hardware_mapping_dse", &HardwareMappingDSE_api,
    py::arg("module"), py::arg("project_path") = ".", py::arg("options"), py::arg("hw_mapping_strategy") = "no_reuse");

  streamhls_m.def("print_dfg_nodes", &printDFGNodes_api);
  streamhls_m.def("print_dfg_edges", &printDFGEdges_api);
  streamhls_m.def("print_dfg_as_dot", &printDFGAsDot_api);

  streamhls_m.def("generate_system_module", &generateSystemModule_api);

  streamhls_m.def("annotate_buffers", &annotateBuffers_api);
  streamhls_m.def("annotate_loops", &annotateLoops_api);

  streamhls_m.def("annotate_memrefs", &annotateMemrefs_api);

  streamhls_m.def("binarize_weights", &binarizeWeights_api);

  streamhls_m.def("bufferize_func_args", &bufferizeFuncArgs_api);

  streamhls_m.def("eliminate_single_write_multiple_reads", &eliminateSingleWriteMultipleReads_api);

  streamhls_m.def("convert_memref_args_to_fifos", &convertMemrefArgsToFIFOs_api);
  streamhls_m.def("convert_func_memrefs_to_fifos_dfg", &convertFuncMemrefsToFIFOsDFG_api);

  streamhls_m.def("set_port_widths_to_one", &setPortWidthsToOne_api);

  streamhls_m.def("eleminate_unit_dims_and_iters", &eleminateUnitDimsAndIters_api);

  streamhls_m.def("flatten_array_of_streams", &flattenArrayOfStreams_api);

  streamhls_m.def("array_streams_to_vector_stream", &arrayStreamsToVectorStream_api);

  streamhls_m.def("rewrite_memory_modules_to_dma", &rewriteMemoryModulesToDMA_api);

  streamhls_m.def("raise_inner_loop_by_one_level", &raiseInnerLoopByOneLevel_api);

  streamhls_m.def("run_loop_permutation_minlp", &runLoopPermutationMINLP_api);
  streamhls_m.def("find_random_permutation_solution", &findRandomPermutationSolution_api);
  streamhls_m.def("set_fixed_permutations", &setFixedPermutations_api);
  streamhls_m.def("run_loop_unrolling_minlp", &runLoopUnrollingMINLP_api);
  streamhls_m.def("find_random_unrolling_solution", &findRandomUnrollingSolution_api);
  streamhls_m.def("set_fixed_unrolling_factors", &setFixedUnrollingFactors_api);
  streamhls_m.def("set_unrolling_factors_to_ones", &setUnrollingFactorsToOnes_api);

  streamhls_m.def("run_loop_permutation_sa", [](DFG &dfg, const DSEOptions &options) {
    return runLoopPermutationSA(dfg, options);
  });
  streamhls_m.def("run_loop_permutation_ga", [](DFG &dfg, const DSEOptions &options) {
    return runLoopPermutationGA(dfg, options);
  });
  streamhls_m.def("run_loop_unrolling_sa", [](DFG &dfg, const DSEOptions &options) {
    return runLoopUnrollingSA(dfg, options);
  });
  streamhls_m.def("run_loop_unrolling_ga", [](DFG &dfg, const DSEOptions &options) {
    return runLoopUnrollingGA(dfg, options);
  });
  streamhls_m.def("run_loop_combined_minlp", &runLoopCombinedMINLP_api);
  streamhls_m.def("run_loop_combined_sa", [](DFG &dfg, const DSEOptions &options) {
    return runLoopCombinedSA(dfg, options);
  });
  streamhls_m.def("run_loop_combined_ga", [](DFG &dfg, const DSEOptions &options) {
    return runLoopCombinedGA(dfg, options);
  });
  streamhls_m.def("find_random_combined_solution", &findRandomCombinedSolution_api);
}