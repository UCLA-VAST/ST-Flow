// #include "streamhls/Dialect/Dataflow/Transforms/MergeKernels/Utils.h"
// #include "streamhls/Dialect/Dataflow/Transforms/MergeKernels/Equations.h"

// using namespace mlir;
// using namespace streamhls;

// #define DEBUG_TYPE "streamhls-merge-kernels-equations"

// namespace {
// void declarePortWidthVariables(
//   PerformanceModel &perfModel,
//   AllMemrefCandidates &memrefCandidates
// ){
//   perfModel.addComment(CommentType::Variable, "Port Width Variables:");
//   for(auto memrefCandidatePair : llvm::enumerate(memrefCandidates)){
//     auto memrefIdx = memrefCandidatePair.index();
//     auto memrefName = memrefCandidatePair.value().first;
//     auto singleMemrefCandidates = memrefCandidatePair.value().second;
//     // 5 binary indicator variables per candidate for 5 port widths
//     for(auto portWidth = 0; portWidth < 5; portWidth++){
//       auto portWidthWords = int(pow(2, portWidth));
//       auto portWidthVarName = memrefName + "_pw_" + std::to_string(portWidthWords);
//       perfModel.addVariable({
//         portWidthVarName,
//         "binary",
//         0,
//         1
//       });
//     }
//   }
//   perfModel.addComment(CommentType::Variable, "\n");
// }

// void portWidthVariablesConstraints(
//   PerformanceModel &perfModel,
//   AllMemrefCandidates &memrefCandidates
// ){
//   perfModel.addComment(CommentType::Constraint, "Port Width Variables Constraints:");
//   for(auto memrefCandidatePair : llvm::enumerate(memrefCandidates)){
//     auto memrefIdx = memrefCandidatePair.index();
//     auto memrefName = memrefCandidatePair.value().first;
//     auto singleMemrefCandidates = memrefCandidatePair.value().second;
//     SmallVector<std::string> portWidthVars;
//     // 5 binary indicator variables per candidate for 5 port widths
//     for(auto portWidth = 0; portWidth < 5; portWidth++){
//       auto portWidthWords = int(pow(2, portWidth));
//       auto portWidthVarName = memrefName + "_pw_" + std::to_string(portWidthWords);
//       portWidthVars.push_back(portWidthVarName);
//     }
//     // Add constraint: sum of port width vars == 1
//     std::string constName = memrefName + "_pw_const";
//     std::string lhs = "";
//     std::string rhs = "1";
//     for(auto portWidthVarPair : llvm::enumerate(portWidthVars)){
//       auto idx = portWidthVarPair.index();
//       if(idx == 0){
//         lhs += portWidthVarPair.value();
//       } else {
//         lhs += " + " + portWidthVarPair.value();
//       }
//     }
//     perfModel.addConstraint({
//       constName,
//       lhs,
//       rhs,
//       Operator::Equal
//     });
//   }
//   perfModel.addComment(CommentType::Constraint, "\n");
// }

// void declareReuseLevelVariables(
//   PerformanceModel &perfModel,
//   AllMemrefCandidates &memrefCandidates
// ){
//   perfModel.addComment(CommentType::Variable, "Reuse Level Variables:");
//   for(auto memrefCandidatePair : llvm::enumerate(memrefCandidates)){
//     auto memrefIdx = memrefCandidatePair.index();
//     auto memrefName = memrefCandidatePair.value().first;
//     auto singleMemrefCandidates = memrefCandidatePair.value().second;
//     SmallVector<std::string> reuseLevelVars;
//     for(auto candidatePair : llvm::enumerate(singleMemrefCandidates)){
//       auto candidateIdx = candidatePair.index();
//       auto candidateInfo = candidatePair.value();
//       auto reuseLevelVarName = memrefName + "_rl_" + std::to_string(candidateIdx);
//       perfModel.addVariable({
//         reuseLevelVarName,
//         "binary",
//         0,
//         1
//       });
//       reuseLevelVars.push_back(reuseLevelVarName);
//     }
//   }
//   perfModel.addComment(CommentType::Variable, "\n");
// }

// void reuseLevelVariablesConstraints(
//   PerformanceModel &perfModel,
//   AllMemrefCandidates &memrefCandidates
// ){
//   perfModel.addComment(CommentType::Constraint, "Reuse Level Variables Constraints :");
//   for(auto memrefCandidatePair : llvm::enumerate(memrefCandidates)){
//     auto memrefIdx = memrefCandidatePair.index();
//     auto memrefName = memrefCandidatePair.value().first;
//     auto singleMemrefCandidates = memrefCandidatePair.value().second;
//     SmallVector<std::string> reuseLevelVars;
//     for(auto candidatePair : llvm::enumerate(singleMemrefCandidates)){
//       auto candidateIdx = candidatePair.index();
//       auto reuseLevelVarName = memrefName + "_rl_" + std::to_string(candidateIdx);
//       reuseLevelVars.push_back(reuseLevelVarName);
//     }
//     // Add constraint: sum of reuse level vars == 1
//     std::string constName = memrefName + "_rl_const";
//     std::string lhs = "";
//     std::string rhs = "1";
//     for(auto reuseLevelVarPair : llvm::enumerate(reuseLevelVars)){
//       auto idx = reuseLevelVarPair.index();
//       if(idx == 0){
//         lhs += reuseLevelVarPair.value();
//       } else {
//         lhs += " + " + reuseLevelVarPair.value();
//       }
//     }
//     perfModel.addConstraint({
//       constName,
//       lhs,
//       rhs,
//       Operator::Equal
//     });
//   }
//   perfModel.addComment(CommentType::Constraint, "\n");
// }

// void declareTilingVariables(
//   PerformanceModel &perfModel,
//   LoopBandInfo &loopBandInfo
// ){
//   perfModel.addComment(CommentType::Variable, "Tiling Variables:");
//   for(auto loopInfoPair : llvm::enumerate(loopBandInfo)){
//     auto idx = loopInfoPair.index() - 1;
//     auto loopInfo = loopInfoPair.value();
//     if(loopInfo.level != 0){
//       continue;
//     }
//     auto tmVarName = "Tm" + std::to_string(idx);
//     perfModel.addVariable({
//       tmVarName,
//       "integer",
//       1,
//       loopInfo.ub
//     });
//     auto tuVarName = "Tu" + std::to_string(idx);
//     perfModel.addVariable({
//       tuVarName,
//       "integer",
//       1,
//       loopInfo.ub
//     });
//   }
//   perfModel.addComment(CommentType::Variable, "\n");
// }

// void tilingVariablesConstraints(
//   PerformanceModel &perfModel,
//   LoopBandInfo &loopBandInfo,
//   bool fixedTiling
// ){
//   perfModel.addComment(CommentType::Constraint, "Tiling Variable Bound Constraints:");
//   for(auto loopPair : llvm::enumerate(loopBandInfo)){
//     auto idx = loopPair.index() - 1;
//     if(loopPair.value().level != 0){
//       continue;
//     }
//     auto loopInfo = loopPair.value();
//     auto ub = loopInfo.ub;
//     // constraint for tiling factors
//     perfModel.addConstraint({
//       "tiling_factor_constraint" + std::to_string(idx),
//       "Tm" + std::to_string(idx) + " * " + "Tu" + std::to_string(idx),
//       std::to_string(ub),
//       Operator::LessEqual
//     });
//   }
//   perfModel.addComment(CommentType::Constraint, "\n");
//   perfModel.addComment(CommentType::Assignment, "Tiling Variable Divisor Constraints:");
//   for(auto loopPair : llvm::enumerate(loopBandInfo)){
//     auto idx = loopPair.index() - 1;
//     if(loopPair.value().level != 0){
//       continue;
//     }
//     auto loopInfo = loopPair.value();
//     auto ub = loopInfo.ub;
//     perfModel.addAssignment({
//       "Tu_divisor" + std::to_string(idx),
//       "integer",
//       "Tu_divisor" + std::to_string(idx) + " * Tu" + std::to_string(idx),
//       std::to_string(ub),
//       "=="
//     });
//     if(fixedTiling){
//       perfModel.addParameter({
//         "Tm" + std::to_string(idx),
//         "integer",
//         1
//       });
//     }else{
//       perfModel.addAssignment({
//         "Tm_divisor" + std::to_string(idx),
//         "integer",
//         "Tm_divisor" + std::to_string(idx) + " * Tm" + std::to_string(idx),
//         std::to_string(ub),
//         "=="
//       });
//     }
//   }
//   perfModel.addComment(CommentType::Assignment, "\n");
// }


// void dataVolumeEquations(
//   PerformanceModel &perfModel,
//   unsigned loopIdx,
//   AllMemrefCandidates &memrefCandidates
// ){
//   perfModel.addComment(CommentType::Assignment, "Data Volume Equations:");
//   std::string loopPrefix = "loop_" + std::to_string(loopIdx);
//   std::string lhsTotalDataVolume = loopPrefix + "_data_volume";
//   std::string rhsTotalDataVolume = "";
//   for(auto memrefCandidatePair : llvm::enumerate(memrefCandidates)){
//     auto memrefIdx = memrefCandidatePair.index();
//     auto memrefName = memrefCandidatePair.value().first;
//     auto singleMemrefCandidates = memrefCandidatePair.value().second;
//     std::string lhsMemref = loopPrefix + "_" + memrefName + "_data_volume";
//     std::string rhsMemref = "";
//     for(auto candidatePair : llvm::enumerate(singleMemrefCandidates)){
//       auto candidateIdx = candidatePair.index();
//       auto candidateInfo = candidatePair.value();
//       auto reuseLevelVarName = memrefName + "_rl_" + std::to_string(candidateIdx);
//       std::string lhsCandidate = loopPrefix + "_" + reuseLevelVarName + "_data_volume";
//       std::string rhsCandidate = reuseLevelVarName + " * " + candidateInfo.reuseFactor.exprStr;
//       perfModel.addAssignment({
//         lhsCandidate,
//         "float",
//         lhsCandidate,
//         rhsCandidate,
//         "=="
//       });
//       if(candidateIdx == 0){
//         rhsMemref += lhsCandidate;
//       } else {
//         rhsMemref += " + " + lhsCandidate;
//       }
//     }
//     perfModel.addAssignment({
//       lhsMemref,
//       "float",
//       lhsMemref,
//       rhsMemref,
//       "=="
//     });
//     if(memrefIdx == 0){
//       rhsTotalDataVolume += lhsMemref;
//     } else {
//       rhsTotalDataVolume += " + " + lhsMemref;
//     }
//   }
//   perfModel.addAssignment({
//     lhsTotalDataVolume,
//     "float",
//     lhsTotalDataVolume,
//     rhsTotalDataVolume,
//     "=="
//   });
//   perfModel.addComment(CommentType::Assignment, "\n");
// }


// void communicationLatencyPerMemref(
//   PerformanceModel &perfModel,
//   unsigned loopIdx,
//   AllMemrefCandidates &memrefCandidates
// ){
//   perfModel.addComment(CommentType::Assignment, "Communication Latency per Memref Equations:");
//   std::string loopPrefix = "loop_" + std::to_string(loopIdx);
//   for(auto memrefCandidatePair : llvm::enumerate(memrefCandidates)){
//     auto memrefIdx = memrefCandidatePair.index();
//     auto memrefName = memrefCandidatePair.value().first;
//     auto singleMemrefCandidates = memrefCandidatePair.value().second;
//     std::string lhsMemref = loopPrefix + "_" + memrefName + "_comm_latency";
//     std::string memrefDataVolume = loopPrefix + "_" + memrefName + "_data_volume";
//     std::string rhsMemref = "";
//     // 5 binary indicator variables per candidate for 5 port widths
//     for(auto portWidth = 0; portWidth < 5; portWidth++){
//       auto portWidthWords = int(pow(2, portWidth));
//       auto portWidthVarName = memrefName + "_pw_" + std::to_string(portWidthWords);
//       if(portWidth == 0){
//         rhsMemref += portWidthVarName +" * (" + memrefDataVolume + " / " + std::to_string(portWidthWords) + ")";
//       } else {
//         rhsMemref += " + " + portWidthVarName + " * (" + memrefDataVolume + " / " + std::to_string(portWidthWords) + ")";
//       }
//     }
//     perfModel.addAssignment({
//       lhsMemref,
//       "float",
//       lhsMemref,
//       rhsMemref,
//       "=="
//     });
//   }
//   perfModel.addComment(CommentType::Assignment, "\n");
// }

// void totalCommunicationLatencyEquation(
//   PerformanceModel &perfModel,
//   unsigned loopIdx,
//   AllMemrefCandidates &memrefCandidates
// ){
//   perfModel.addComment(CommentType::Assignment, "Total Communication Latency Equation:");
//   std::string loopPrefix = "loop_" + std::to_string(loopIdx);
//   std::string lhsTotalCommLatency = loopPrefix + "_comm_latency";
//   std::string rhsTotalCommLatency = "max(";
//   for(auto memrefCandidatePair : llvm::enumerate(memrefCandidates)){
//     auto memrefIdx = memrefCandidatePair.index();
//     auto memrefName = memrefCandidatePair.value().first;
//     std::string memrefCommLatency = loopPrefix + "_" + memrefName + "_comm_latency";
//     if(memrefIdx == 0){
//       rhsTotalCommLatency += memrefCommLatency;
//     } else {
//       rhsTotalCommLatency += ", " + memrefCommLatency;
//     }
//   }
//   rhsTotalCommLatency += ")";
//   perfModel.addAssignment({
//     lhsTotalCommLatency,
//     "float",
//     lhsTotalCommLatency,
//     rhsTotalCommLatency,
//     "==",
//     false,
//     nullptr,
//     true
//   });
//   perfModel.addComment(CommentType::Assignment, "\n");
// }


// void bufferSizeEquations(
//   PerformanceModel &perfModel,
//   unsigned loopIdx,
//   AllMemrefCandidates &memrefCandidates
// ){
//   perfModel.addComment(CommentType::Assignment, "Buffer Size Equations:");
//   std::string loopPrefix = "loop_" + std::to_string(loopIdx);
//   for(auto memrefCandidatePair : llvm::enumerate(memrefCandidates)){
//     auto memrefIdx = memrefCandidatePair.index();
//     auto memrefName = memrefCandidatePair.value().first;
//     auto singleMemrefCandidates = memrefCandidatePair.value().second;
//     for(auto candidatePair : llvm::enumerate(singleMemrefCandidates)){
//       auto candidateIdx = candidatePair.index();
//       auto candidateInfo = candidatePair.value();
//       auto reuseLevelVarName = memrefName + "_rl_" + std::to_string(candidateIdx);
//       std::string bufferSizeStr = "";
//       for(auto bufferSizePair : llvm::enumerate(candidateInfo.bufferSizes)){
//         auto bufIdx = bufferSizePair.index();
//         auto bufSizeInfo = bufferSizePair.value();
//         if(bufIdx == 0){
//           bufferSizeStr += bufSizeInfo.exprStr;
//         } else {
//           bufferSizeStr += " * " + bufSizeInfo.exprStr;
//         }
//       }
//       // llvm::dbgs() << bufferSizeStr << "\n";
//       std::string lhsCandidate = loopPrefix + "_" + reuseLevelVarName + "_buffer_size";
//       std::string rhsCandidate = reuseLevelVarName + " * (" + bufferSizeStr + ")";
//       perfModel.addAssignment({
//         lhsCandidate,
//         "float",
//         lhsCandidate,
//         rhsCandidate,
//         "=="
//       });
//     }
//   }
//   perfModel.addComment(CommentType::Assignment, "\n");
// }

// void bufferSizePerMemref(
//   PerformanceModel &perfModel,
//   unsigned loopIdx,
//   AllMemrefCandidates &memrefCandidates
// ){
//   perfModel.addComment(CommentType::Assignment, "Buffer Size per Memref Equations:");
//   std::string loopPrefix = "loop_" + std::to_string(loopIdx);
//   for(auto memrefCandidatePair : llvm::enumerate(memrefCandidates)){
//     auto memrefIdx = memrefCandidatePair.index();
//     auto memrefName = memrefCandidatePair.value().first;
//     auto singleMemrefCandidates = memrefCandidatePair.value().second;
//     std::string lhsMemref = loopPrefix + "_" + memrefName + "_buffer_size";
//     std::string rhsMemref = "";
//     for(auto candidatePair : llvm::enumerate(singleMemrefCandidates)){
//       auto candidateIdx = candidatePair.index();
//       auto candidateInfo = candidatePair.value();
//       auto reuseLevelVarName = memrefName + "_rl_" + std::to_string(candidateIdx);
//       std::string lhsCandidate = loopPrefix + "_" + reuseLevelVarName + "_buffer_size";
//       if(candidateIdx == 0){
//         rhsMemref += lhsCandidate;
//       } else {
//         rhsMemref += " + " + lhsCandidate;
//       }
//     }
//     perfModel.addAssignment({
//       lhsMemref,
//       "float",
//       lhsMemref,
//       rhsMemref,
//       "=="
//     });
//   }
//   perfModel.addComment(CommentType::Assignment, "\n");
// }

// void totalBufferSizeEquation(
//   PerformanceModel &perfModel,
//   unsigned loopIdx,
//   AllMemrefCandidates &memrefCandidates
// ){
//   perfModel.addComment(CommentType::Assignment, "Total Buffer Size Equation:");
//   std::string loopPrefix = "loop_" + std::to_string(loopIdx);
//   std::string lhsTotalBufferSize = loopPrefix + "_buffer_size";
//   std::string rhsTotalBufferSize = "";
//   for(auto memrefCandidatePair : llvm::enumerate(memrefCandidates)){
//     auto memrefIdx = memrefCandidatePair.index();
//     auto memrefName = memrefCandidatePair.value().first;
//     std::string memrefBufferSize = loopPrefix + "_" + memrefName + "_buffer_size";
//     if(memrefIdx == 0){
//       rhsTotalBufferSize += memrefBufferSize;
//     } else {
//       rhsTotalBufferSize += " + " + memrefBufferSize;
//     }
//   }
//   perfModel.addAssignment({
//     lhsTotalBufferSize,
//     "float",
//     lhsTotalBufferSize,
//     rhsTotalBufferSize,
//     "=="
//   });
//   perfModel.addComment(CommentType::Assignment, "\n");
// }


// void computationLatencyEquation(
//   PerformanceModel &perfModel,
//   unsigned loopIdx,
//   LoopBandInfo &loopBandInfo
// ){
//   perfModel.addComment(CommentType::Assignment, "Computation Latency Equation:");
//   std::string loopPrefix = "loop_" + std::to_string(loopIdx);
//   std::string lhsCompLatency = loopPrefix + "_comp_latency";
//   std::string rhsCompLatency = "(";
//   int64_t tripCount = 1;
//   std::string unrollFactor;
//   SmallVector<std::string> tripCounts;
//   SmallVector<std::string> unrollFactors;
//   for(auto loopInfoPair : llvm::enumerate(loopBandInfo)){
//     auto idx = loopInfoPair.index();
//     auto loopInfo = loopInfoPair.value();
//     if(loopInfo.level == 0){
//       tripCount *= loopInfo.ub;
//       tripCounts.push_back(std::to_string(loopInfo.ub));
//     }
//     if(loopInfo.level == 2){
//       unrollFactor += loopInfo.name;
//       if(idx != loopBandInfo.size() - 1){
//         unrollFactor += " * ";
//       }
//       unrollFactors.push_back(loopInfo.name);
//     }
//   }
//   unsigned t_idx = 0;
//   for(auto tripCountUnrollFactorPair : llvm::zip(tripCounts, unrollFactors)){
//     auto tripCount = std::get<0>(tripCountUnrollFactorPair);
//     auto unrollFactor = std::get<1>(tripCountUnrollFactorPair);
//     rhsCompLatency += "max((" + tripCount + " / " + unrollFactor + "), 1)";
//     rhsCompLatency += t_idx < tripCounts.size() - 1 ? " * " : ")";
//     t_idx++;
//   }
//   perfModel.addAssignment({
//     lhsCompLatency,
//     "float",
//     lhsCompLatency,
//     rhsCompLatency,
//     "==",
//     false,
//     nullptr,
//     true
//   });
//   perfModel.addComment(CommentType::Assignment, "\n");
// }

// void totalEquationsAndConstraints(
//   PerformanceModel &perfModel,
//   AllLoopBandsInfo &allLoopBandsInfo
// ){
//   perfModel.addComment(CommentType::Assignment, "Total Performance Equations:");
//   std::string totalLatencylhs = "total_latency";
//   std::string totalLatencyrhs = "";
//   std::string totalMemorylhs = "total_buffer_size";
//   std::string totalMemoryrhs = "max(";
//   std::string totalDataVolumelhs = "total_data_volume";
//   std::string totalDataVolumerhs = "(";
//   for(auto loopBandInfoPair : llvm::enumerate(allLoopBandsInfo)){
//     auto idx = loopBandInfoPair.index();
//     auto loopBandInfo = loopBandInfoPair.value();
//     totalLatencyrhs += "max(loop_" + std::to_string(idx) + "_comp_latency, " + "loop_" + std::to_string(idx) + "_comm_latency)";
//     totalLatencyrhs += idx < allLoopBandsInfo.size() - 1 ? " + " : "";
//     totalMemoryrhs += "loop_" + std::to_string(idx) + "_buffer_size";
//     totalMemoryrhs += idx < allLoopBandsInfo.size() - 1 ? ", " : ")";
//     totalDataVolumerhs += "loop_" + std::to_string(idx) + "_data_volume";
//     totalDataVolumerhs += idx < allLoopBandsInfo.size() - 1 ? " + " : ")";
//   }  
//   perfModel.addAssignment({
//     totalLatencylhs,
//     "float",
//     totalLatencylhs,
//     totalLatencyrhs,
//     "==",
//     false,
//     nullptr,
//     true
//   });
//   perfModel.addAssignment({
//     totalMemorylhs,
//     "float",
//     totalMemorylhs,
//     totalMemoryrhs,
//     "==",
//     false,
//     nullptr,
//     true
//   });
//   perfModel.addAssignment({
//     totalDataVolumelhs,
//     "float",
//     totalDataVolumelhs,
//     totalDataVolumerhs,
//     "==",
//     false,
//     nullptr,
//     true
//   });

//   // buffer size constraint
//   perfModel.addComment(CommentType::Constraint, "Total Buffer Size Constraint:");
//   perfModel.addConstraint({
//     "buffer_size_constraint",
//     "total_buffer_size",
//     "memory_avail",
//     Operator::LessEqual
//   });
//   perfModel.addComment(CommentType::Constraint, "\n");


//   SmallVector<std::string> dspTerms;
//   for(auto loopPair : llvm::enumerate(allLoopBandsInfo[0])){
//     auto idx = loopPair.index() - 1;
//     if(loopPair.value().level != 0){
//       continue;
//     }
//     dspTerms.push_back("Tu" + std::to_string(idx));
//   }
//   std::string totalDSPlhs = "total_dsp";
//   std::string totalDSPrhs = "";
//   for(auto dspTermPair : llvm::enumerate(dspTerms)){
//     auto termIdx = dspTermPair.index();
//     auto term = dspTermPair.value();
//     if(termIdx == 0){
//       totalDSPrhs += term;
//     } else {
//       totalDSPrhs += " * " + term;
//     }
//   }
//   perfModel.addAssignment({
//     "total_dsp",
//     "float",
//     totalDSPlhs,
//     totalDSPrhs,
//     "==",
//     false,
//     nullptr,
//     true
//   });
//   perfModel.addComment(CommentType::Constraint, "Total DSP Constraint:");
//   perfModel.addConstraint({
//     "dsp_constraint",
//     "total_dsp",
//     "DSP_avail",
//     Operator::LessEqual
//   });
//   perfModel.addComment(CommentType::Constraint, "\n");

//   perfModel.addComment(CommentType::Assignment, "\n");
// }
// } // end anonymous namespace

// namespace mlir{
// namespace streamhls {

// // ---------- Function to create performance model from candidates ----------
// void createPerformanceModelFromCandidates(
//   AllCandidates &loopCandidates,
//   AllLoopBandsInfo &allLoopBandsInfo,
//   std::unordered_map<std::string, VarType> &options
// ){
//   // create performance model
//   // U250 bw: 77 GB/s = 64 floats/cycle @ 300 MHz
//   // U280 bw: 460 GB/s = 384 floats/cycle @ 300 MHz
//   PerformanceModel perfModel;
//   perfModel.addParameter({"DSP_avail", "integer", std::get<int64_t>(options["DSP_avail"])});
//   perfModel.addParameter({"memory_avail", "integer", std::get<int64_t>(options["memory_avail"])});
//   perfModel.addParameter({"bw_avail", "integer", std::get<int64_t>(options["bw_avail"])});

//   declarePortWidthVariables(perfModel, loopCandidates[0]);
//   portWidthVariablesConstraints(perfModel, loopCandidates[0]);

//   declareReuseLevelVariables(perfModel, loopCandidates[0]);
//   reuseLevelVariablesConstraints(perfModel, loopCandidates[0]);

//   LoopBandInfo minBandInfo = allLoopBandsInfo[0];
//   LoopBandInfo maxDummy = allLoopBandsInfo[0];
//   getMinAndMaxLoopStatsMerge(allLoopBandsInfo, minBandInfo, maxDummy);
//   declareTilingVariables(perfModel, maxDummy);
//   tilingVariablesConstraints(perfModel, maxDummy, std::get<bool>(options["fixed_tiling"]));

//   for(auto loopMemrefCandidatesPair : llvm::enumerate(loopCandidates)){
//     auto loopIdx = loopMemrefCandidatesPair.index();
//     auto memrefCandidates = loopMemrefCandidatesPair.value();

//     perfModel.addComment(CommentType::Assignment, "==== Loop " + std::to_string(loopIdx) + " Equations ====\n");
//     // communication modeling equations
//     dataVolumeEquations(perfModel, loopIdx, memrefCandidates);
//     communicationLatencyPerMemref(perfModel, loopIdx, memrefCandidates);
//     totalCommunicationLatencyEquation(perfModel, loopIdx, memrefCandidates);

//     // buffer size modeling equations
//     bufferSizeEquations(perfModel, loopIdx, memrefCandidates);
//     bufferSizePerMemref(perfModel, loopIdx, memrefCandidates);
//     totalBufferSizeEquation(perfModel, loopIdx, memrefCandidates);

//     // computation latency modeling equations
//     computationLatencyEquation(perfModel, loopIdx, allLoopBandsInfo[loopIdx]);
//     perfModel.addComment(CommentType::Assignment, "\n");
//   }
  
//   totalEquationsAndConstraints(perfModel, allLoopBandsInfo);

//   // objective: minimize latency + alpha * memory
//   perfModel.addAssignment({
//     "score",
//     "float",
//     "score",
//     "total_latency + " + std::to_string(std::get<float>(options["memory_alpha"])) + "* total_data_volume",
//     "=="
//   });

//   perfModel.addComment(CommentType::Objective, "==== Objective ====\n");
//   perfModel.setObjective({
//     "score"
//   });

//   std::error_code stdError;
//   auto projectPath = std::get<std::string>(options["project_path"]);
//   std::string fileName = projectPath + "/kernel_merging.mod";
//   llvm::raw_fd_ostream amplModelFile(fileName, stdError);
//   if(!amplModelFile.has_error()) {
//     writeAMPLModel(perfModel, amplModelFile, std::get<bool>(options["display_vars"]), std::get<bool>(options["display_assignments"]));
//   } else {
//     llvm::errs() << "Error: Unable to open file for writing\n";
//   }
// }
// } // end namespace streamhls
// } // end namespace mlir