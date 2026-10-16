//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Adapted from ScaleHLS (https://github.com/UIUC-ChenLab/scalehls) and modified
// for Stream-HLS and ST-Flow. See NOTICE.
//
//===----------------------------------------------------------------------===//

// #include "mlir/Analysis/Liveness.h"
// #include "mlir/Dialect/Bufferization/IR/Bufferization.h"
// #include "mlir/Dialect/MemRef/IR/MemRef.h"
// #include "mlir/IR/DialectImplementation.h"
// #include "mlir/IR/IntegerSet.h"

#include "streamhls/Dialect/Dataflow/IR/Dataflow.h"

#include "streamhls/Dialect/Dataflow/IR/DataflowEnums.cpp.inc"

using namespace mlir;
using namespace dataflow;

//===----------------------------------------------------------------------===//
// Dataflow dialect
//===----------------------------------------------------------------------===//

void DataflowDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "streamhls/Dialect/Dataflow/IR/DataflowTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "streamhls/Dialect/Dataflow/IR/DataflowAttributes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "streamhls/Dialect/Dataflow/IR/Dataflow.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// StreamOp, StreamReadOp, and StreamWriteOp
//===----------------------------------------------------------------------===//

LogicalResult StreamOp::verify() {
  return success();
}

void StreamOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // effects.emplace_back(MemoryEffects::Allocate::get(), getChannel(),
  //                      SideEffects::DefaultResource::get());
}

LogicalResult StreamReadOp::verify() {
  if (getResult())
    if (getChannel().getType().cast<StreamType>().getElementType() !=
        getResult().getType())
      return emitOpError("result type doesn't align with channel type");
  return success();
}

// void StreamReadOp::getEffects(
//     SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
//         &effects) {
//   effects.emplace_back(MemoryEffects::Read::get(), getChannel(),
//                        SideEffects::DefaultResource::get());
// }

LogicalResult StreamWriteOp::verify() {
  if (getChannel().getType().cast<StreamType>().getElementType() !=
      getValue().getType())
    return emitOpError("value type doesn't align with channel type");
  return success();
}

// void StreamWriteOp::getEffects(
//     SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
//         &effects) {
//   effects.emplace_back(MemoryEffects::Write::get(), getChannel(),
//                        SideEffects::DefaultResource::get());
// }


//===----------------------------------------------------------------------===//
// High Dataflow Operations from ScaleHLS
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// DispatchOp
//===----------------------------------------------------------------------===//

namespace {
template <typename OpType>
struct SimplifyDispatchOrTaskOutputs : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    auto yield = op.getYieldOp();
    bool hasUnusedPort = false;

    // Identify output values that are used.
    SmallVector<Value, 4> usedOutputs;
    SmallVector<Value, 4> usedResults;
    for (auto result : op.getResults())
      if (result.use_empty()) {
        hasUnusedPort = true;
      } else {
        usedOutputs.push_back(yield.getOperand(result.getResultNumber()));
        usedResults.push_back(result);
      }

    // Construct new op with only used outputs.
    if (hasUnusedPort) {
      rewriter.setInsertionPoint(yield);
      rewriter.replaceOpWithNewOp<YieldOp>(yield, usedOutputs);

      rewriter.setInsertionPoint(op);
      auto newTask =
          rewriter.create<OpType>(op.getLoc(), ValueRange(usedOutputs));
      rewriter.inlineRegionBefore(op.getBody(), newTask.getBody(),
                                  newTask.getBody().end());
      for (auto t : llvm::zip(usedResults, newTask.getResults()))
        std::get<0>(t).replaceAllUsesWith(std::get<1>(t));

      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};
} // namespace

namespace {
template <typename OpType>
struct InlineDispatchOrTask : public OpRewritePattern<OpType> {
  InlineDispatchOrTask(MLIRContext *context,
                       llvm::function_ref<bool(OpType)> condition)
      : OpRewritePattern<OpType>(context), condition(condition) {}

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    if (condition(op)) {
      auto &ops = op.getBody().front().getOperations();
      auto &parentOps = op->getBlock()->getOperations();
      parentOps.splice(op->getIterator(), ops, ops.begin(),
                       std::prev(ops.end()));
      rewriter.replaceOp(op, op.getYieldOp()->getOperands());
      return success();
    }
    return failure();
  }

private:
  llvm::function_ref<bool(OpType)> condition;
};
} // namespace

void DispatchOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.add<SimplifyDispatchOrTaskOutputs<DispatchOp>>(context);
  results.add<InlineDispatchOrTask<DispatchOp>>(context, [](DispatchOp op) {
    return op.getOps<TaskOp>().empty() || llvm::hasSingleElement(op.getOps());
  });
}

LogicalResult DispatchOp::verify() {
  if (getResultTypes() != getYieldOp().getOperandTypes())
    return emitOpError("yield type doesn't align with result type");
  return success();
}

/// Get the terminator yield op.
YieldOp DispatchOp::getYieldOp() {
  return cast<YieldOp>(getBody().front().getTerminator());
}

//===----------------------------------------------------------------------===//
// TaskOp
//===----------------------------------------------------------------------===//

void TaskOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<SimplifyDispatchOrTaskOutputs<TaskOp>>(context);
  results.add<InlineDispatchOrTask<TaskOp>>(context, [](TaskOp op) {
    return llvm::hasSingleElement(op.getOps());
    // return llvm::hasSingleElement(op.getDispatchOp().getOps<TaskOp>()) ||
    //        llvm::hasSingleElement(op.getOps());
  });
}

LogicalResult TaskOp::verify() {
  if (getResultTypes() != getYieldOp().getOperandTypes())
    return emitOpError("yield type doesn't align with result type");
  return success();
}

/// Get the parent dispatch op.
DispatchOp TaskOp::getDispatchOp() {
  return (*this)->getParentOfType<DispatchOp>();
}

/// Get the terminator yield op.
YieldOp TaskOp::getYieldOp() {
  return cast<YieldOp>(getBody().front().getTerminator());
}

bool TaskOp::isLivein(Value value) {
  auto liveins = Liveness(*this).getLiveIn(&(*this).getBody().front());
  return liveins.count(value);
}

SmallVector<Value> TaskOp::getLiveins() {
  auto liveins = Liveness(*this).getLiveIn(&(*this).getBody().front());
  return {liveins.begin(), liveins.end()};
}

SmallVector<Operation *> TaskOp::getLiveinUsers(Value livein) {
  assert(isLivein(livein) && "invalid livein");
  auto users = llvm::make_filter_range(livein.getUsers(), [&](Operation *user) {
    return (*this)->isAncestor(user);
  });
  return {users.begin(), users.end()};
}

//===----------------------------------------------------------------------===//
// NodeOp
//===----------------------------------------------------------------------===//

namespace {
struct SimplifyNodeIOs : public OpRewritePattern<NodeOp> {
  using OpRewritePattern<NodeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(NodeOp node,
                                PatternRewriter &rewriter) const override {
    bool hasUnusedPort = false;

    // Identify input values that are used.
    llvm::SmallDenseSet<BlockArgument, 4> unusedArgs;
    SmallVector<Value, 4> usedInputs;
    SmallVector<int32_t, 4> usedInputTaps;
    SmallVector<Value, 4> usedOutputs;
    SmallVector<Value, 4> usedParams;
    for (auto arg : node.getBody().getArguments())
      if (arg.use_empty()) {
        hasUnusedPort = true;
        unusedArgs.insert(arg);
      } else {
        auto idx = arg.getArgNumber();
        if (node.getOperandKind(idx) == OperandKind::INPUT) {
          usedInputs.push_back(node.getOperand(idx));
          usedInputTaps.push_back(node.getInputTap(idx));
        } else if (node.getOperandKind(idx) == OperandKind::OUTPUT)
          usedOutputs.push_back(node.getOperand(idx));
        else
          usedParams.push_back(node.getOperand(idx));
      }
    node.getBody().front().eraseArguments(
        [&](BlockArgument arg) { return unusedArgs.count(arg); });

    // Construct new dataflow node.
    if (hasUnusedPort) {
      rewriter.setInsertionPoint(node);
      auto newNode = rewriter.create<NodeOp>(
          node.getLoc(), usedInputs, usedOutputs, usedParams,
          rewriter.getI32ArrayAttr(usedInputTaps), node.getLevelAttr());
      rewriter.inlineRegionBefore(node.getBody(), newNode.getBody(),
                                  newNode.getBody().end());
      rewriter.eraseOp(node);
      return success();
    }
    return failure();
  }
};
} // namespace

void NodeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  for (OpOperand &operand : getInputsMutable())
    effects.emplace_back(MemoryEffects::Read::get(), &operand,
                         SideEffects::DefaultResource::get());
  for (OpOperand &operand  : getOutputsMutable()) {
    effects.emplace_back(MemoryEffects::Read::get(), &operand,
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), &operand,
                         SideEffects::DefaultResource::get());
  }
}

/// Get the parent schedule op.
DispatchOp NodeOp::getDispatchOp() {
  return (*this)->getParentOfType<DispatchOp>();
}

/// Get the terminator yield op.
YieldOp NodeOp::getYieldOp() {
  return cast<YieldOp>(getBody().front().getTerminator());
}

/// Get input taps.
void NodeOp::setInputTap(unsigned idx, unsigned tap) {
  SmallVector<int32_t> newInputTaps(llvm::map_range(
      getInputTapsAsInt(), [](unsigned a) { return (int32_t)a; }));
  newInputTaps[idx] = tap;
  Builder builder(getContext());
  setInputTapsAttr(builder.getI32ArrayAttr(newInputTaps));
}
unsigned NodeOp::getInputTap(unsigned idx) {
  return getInputTaps()[idx].cast<IntegerAttr>().getInt();
}
SmallVector<unsigned> NodeOp::getInputTapsAsInt() {
  auto array = llvm::map_range(getInputTaps(), [](Attribute attr) {
    return attr.cast<IntegerAttr>().getInt();
  });
  return {array.begin(), array.end()};
}

/// Return the number of inputs, outputs, and params.
unsigned NodeOp::getNumInputs() {
  return getODSOperandIndexAndLength(0).second;
}
unsigned NodeOp::getNumOutputs() {
  return getODSOperandIndexAndLength(1).second;
}
unsigned NodeOp::getNumParams() {
  return getODSOperandIndexAndLength(2).second;
}

/// Get the type of operand: input, output, or param.
OperandKind NodeOp::getOperandKind(OpOperand &operand) {
  assert(operand.getOwner() == *this && "invalid operand");
  return getOperandKind(operand.getOperandNumber());
}
OperandKind NodeOp::getOperandKind(unsigned operandIdx) {
  if (operandIdx >= getODSOperandIndexAndLength(2).first)
    return OperandKind::PARAM;
  else if (operandIdx >= getODSOperandIndexAndLength(1).first)
    return OperandKind::OUTPUT;
  else
    return OperandKind::INPUT;
}

/// Get the input, output, and param arguments.
iterator_range<Block::args_iterator> NodeOp::getInputArgs() {
  auto range = getODSOperandIndexAndLength(0);
  return {std::next(getBody().args_begin(), range.first),
          std::next(getBody().args_begin(), range.first + range.second)};
}
iterator_range<Block::args_iterator> NodeOp::getOutputArgs() {
  auto range = getODSOperandIndexAndLength(1);
  return {std::next(getBody().args_begin(), range.first),
          std::next(getBody().args_begin(), range.first + range.second)};
}
iterator_range<Block::args_iterator> NodeOp::getParamArgs() {
  auto range = getODSOperandIndexAndLength(2);
  return {std::next(getBody().args_begin(), range.first),
          std::next(getBody().args_begin(), range.first + range.second)};
}

bool NodeOp::isLivein(Value value) {
  return value.isa<BlockArgument>() &&
         value.getParentRegion() == &(*this).getBody();
}

SmallVector<Value> NodeOp::getLiveins() {
  auto args = (*this).getBody().getArguments();
  return {args.begin(), args.end()};
}

SmallVector<Operation *> NodeOp::getLiveinUsers(Value livein) {
  assert(isLivein(livein) && "invalid livein");
  auto users = livein.getUsers();
  return {users.begin(), users.end()};
}

/// Loop directives attribute utils.
LoopDirectiveAttr dataflow::getLoopDirective(Operation *op) {
  return op->getAttrOfType<LoopDirectiveAttr>("loop_directive");
}
void dataflow::setLoopDirective(Operation *op, LoopDirectiveAttr loopDirective) {
  op->setAttr("loop_directive", loopDirective);
}
void dataflow::setLoopDirective(Operation *op, bool pipeline, int64_t targetII,
                           bool dataflow, bool flatten) {
  auto loopDirective = LoopDirectiveAttr::get(op->getContext(), pipeline,
                                              targetII, dataflow, flatten);
  setLoopDirective(op, loopDirective);
}

// Loop information attribute utils.
LoopInfoAttr dataflow::getLoopInfo(Operation *op) {
  return op->getAttrOfType<LoopInfoAttr>("loop_info");
}
void dataflow::setLoopInfo(Operation *op, LoopInfoAttr loopInfo) {
  op->setAttr("loop_info", loopInfo);
}
void dataflow::setLoopInfo(Operation *op, int64_t flattenTripCount, int64_t iterLatency,
                           int64_t minII) {
  auto loopInfo = LoopInfoAttr::get(op->getContext(), flattenTripCount, iterLatency,
                                         minII);
  setLoopInfo(op, loopInfo);
}

/// Parrallel and point loop attribute utils.
void dataflow::setParallelAttr(Operation *op) {
  op->setAttr("parallel", UnitAttr::get(op->getContext()));
}
bool dataflow::hasParallelAttr(Operation *op) {
  return op->hasAttrOfType<UnitAttr>("parallel");
}
void dataflow::setReductionAttr(Operation *op) {
  op->setAttr("reduction", UnitAttr::get(op->getContext()));
}
bool dataflow::hasReductionAttr(Operation *op) {
  return op->hasAttrOfType<UnitAttr>("reduction");
}

void dataflow::setPEAttr(Operation *op) {
  op->setAttr("PE", UnitAttr::get(op->getContext()));
}
bool dataflow::hasPEAttr(Operation *op) {
  return op->hasAttrOfType<UnitAttr>("PE");
}

void dataflow::setColAttr(Operation *op){
  op->setAttr("column", UnitAttr::get(op->getContext()));
}
bool dataflow::hasColAttr(Operation *op){
  return op->hasAttrOfType<UnitAttr>("column");
}
void dataflow::setRowAttr(Operation *op){
  op->setAttr("row", UnitAttr::get(op->getContext()));
}
bool dataflow::hasRowAttr(Operation *op){
  return op->hasAttrOfType<UnitAttr>("row");
}

void dataflow::setNamedAttr(Operation *op, StringRef name){
  op->setAttr(name, UnitAttr::get(op->getContext()));
}
bool dataflow::hasNamedAttr(Operation *op, StringRef name){
  return op->hasAttrOfType<UnitAttr>(name);
}

void dataflow::setLoopIDAttr(Operation *op, int64_t id){
  op->setAttr("loop_id", IntegerAttr::get(IntegerType::get(op->getContext(), 32), id));
}
int64_t dataflow::getLoopIDAttr(Operation *op){
  return op->getAttrOfType<IntegerAttr>("loop_id").getInt();
}

void dataflow::setLoopLevelAttr(Operation *op, StringRef level){
  assert(level == "inner" || level == "outer");
  op->setAttr("loop_level", StringAttr::get(op->getContext(), level));
}
StringRef dataflow::getLoopLevelAttr(Operation *op){
  assert(op->hasAttrOfType<StringAttr>("loop_level"));
  return op->getAttrOfType<StringAttr>("loop_level").getValue();
}

void dataflow::setLoopTypeAttr(Operation *op, StringRef type){
  assert(type == "parallel" || type == "reduction");
  op->setAttr("loop_type", StringAttr::get(op->getContext(), type));
}
StringRef dataflow::getLoopTypeAttr(Operation *op){
  assert(op->hasAttrOfType<StringAttr>("loop_type"));
  return op->getAttrOfType<StringAttr>("loop_type").getValue();
}
bool dataflow::hasLoopTypeAttr(Operation *op){
  return op->hasAttrOfType<StringAttr>("loop_type");
}
bool dataflow::isReductionLoop(Operation *op){
  return getLoopTypeAttr(op) == "reduction";
}
bool dataflow::isParallelLoop(Operation *op){
  return getLoopTypeAttr(op) == "parallel";
}
bool dataflow::isReductionLoopBand(SmallVector<affine::AffineForOp, 6> band){
  for(auto loop : band){
    if(getLoopTypeAttr(loop) == "reduction")
      return true;
  }
  return false;
}
bool dataflow::isParallelLoopBand(SmallVector<affine::AffineForOp, 6> band){
  for(auto loop : band){
    if(getLoopTypeAttr(loop) == "parallel")
      return true;
  }
  return false;
}
void dataflow::setOpLoopAttr(Operation *op){
  if(auto storeOp = dyn_cast<affine::AffineStoreOp>(op)){
    SmallVector<Attribute> loopIDs;
    for(auto operand : storeOp.getMapOperands()){
      if(isa<BlockArgument>(operand)){
        auto loop = operand.getParentBlock()->getParentOp();
        if(getLoopLevelAttr(loop) == "inner")
          loopIDs.push_back(IntegerAttr::get(IntegerType::get(op->getContext(), 32), getLoopIDAttr(loop)));
      }
    }
    ArrayAttr attr = ArrayAttr::get(op->getContext(), loopIDs);
    op->setAttr("opLoopInfo", attr);
  }else if(auto loadOp = dyn_cast<affine::AffineLoadOp>(op)){
    SmallVector<Attribute> loopIDs;
    for(auto operand : loadOp.getMapOperands()){
      if(isa<BlockArgument>(operand)){
        auto loop = operand.getParentBlock()->getParentOp();
        if(getLoopLevelAttr(loop) == "inner")
          loopIDs.push_back(IntegerAttr::get(IntegerType::get(op->getContext(), 32), getLoopIDAttr(loop)));
      }
    }
    ArrayAttr attr = ArrayAttr::get(op->getContext(), loopIDs);
    op->setAttr("opLoopInfo", attr);
  }else{
    emitError(op->getLoc(), "unsupported operation for loop attribute");
  }
}

ArrayAttr dataflow::getOpLoopAttr(Operation *op){
  return op->getAttrOfType<ArrayAttr>("opLoopInfo");
}

void dataflow::setMemrefPortWidthAttr(Operation* op, unsigned width){
  op->setAttr("portWidth", IntegerAttr::get(IntegerType::get(op->getContext(), 32), width));
}

unsigned dataflow::getMemrefPortWidthAttr(Operation* op){
  return op->getAttrOfType<IntegerAttr>("portWidth").getInt();
}

bool dataflow::hasMemrefPortWidthAttr(Operation* op){
  return op->hasAttrOfType<IntegerAttr>("portWidth");
}


/// Function directives attribute utils.
FuncDirectiveAttr dataflow::getFuncDirective(Operation *op) {
  return op->getAttrOfType<FuncDirectiveAttr>("func_directive");
}
void dataflow::setFuncDirective(Operation *op, FuncDirectiveAttr funcDirective) {
  op->setAttr("func_directive", funcDirective);
}
void dataflow::setFuncDirective(Operation *op, bool pipeline, int64_t targetInterval,
                           bool dataflow) {
  auto funcDirective = FuncDirectiveAttr::get(op->getContext(), pipeline,
                                              targetInterval, dataflow);
  setFuncDirective(op, funcDirective);
}

/// Top and runtime function attribute utils.
void dataflow::setTopFuncAttr(Operation *op) {
  op->setAttr("top_func", UnitAttr::get(op->getContext()));
}
bool dataflow::hasTopFuncAttr(Operation *op) {
  return op->hasAttrOfType<UnitAttr>("top_func");
}
void dataflow::setRuntimeAttr(Operation *op) {
  op->setAttr("runtime", UnitAttr::get(op->getContext()));
}
bool dataflow::hasRuntimeAttr(Operation *op) {
  return op->hasAttrOfType<UnitAttr>("runtime");
}

//===----------------------------------------------------------------------===//
// ResourceAttr
//===----------------------------------------------------------------------===//

Attribute ResourceAttr::parse(AsmParser &p, Type type) {
  StringRef lutKw, dspKw, bramKw;
  int64_t lut, dsp, bram;
  if (p.parseLess() || p.parseKeyword(&lutKw) || p.parseEqual() ||
      p.parseInteger(lut) || p.parseComma() || p.parseKeyword(&dspKw) ||
      p.parseEqual() || p.parseInteger(dsp) || p.parseComma() ||
      p.parseKeyword(&bramKw) || p.parseEqual() || p.parseInteger(bram) ||
      p.parseGreater())
    return Attribute();

  if (lutKw != "lut" || dspKw != "dsp" || bramKw != "bram")
    return Attribute();

  return ResourceAttr::get(p.getContext(), lut, dsp, bram);
}

void ResourceAttr::print(AsmPrinter &p) const {
  p << "<lut=" << getLut() << ", dsp=" << getDsp() << ", bram=" << getBram()
    << ">";
}

//===----------------------------------------------------------------------===//
// TimingAttr
//===----------------------------------------------------------------------===//

Attribute TimingAttr::parse(AsmParser &p, Type type) {
  int64_t begin, end, latency, interval;
  if (p.parseLess() || p.parseInteger(begin) || p.parseArrow() ||
      p.parseInteger(end) || p.parseComma() || p.parseInteger(latency) ||
      p.parseComma() || p.parseInteger(interval) || p.parseGreater())
    return Attribute();

  return TimingAttr::get(p.getContext(), begin, end, latency, interval);
}

void TimingAttr::print(AsmPrinter &p) const {
  p << "<" << getBegin() << " -> " << getEnd() << ", " << getLatency() << ", "
    << getInterval() << ">";
}

//===----------------------------------------------------------------------===//
// LoopInfoAttr
//===----------------------------------------------------------------------===//

Attribute LoopInfoAttr::parse(AsmParser &p, Type type) {
  StringRef flattenTripCountKw, iterLatencyKw, minIIKw;
  int64_t flattenTripCount, iterLatency, minII;
  if (p.parseLess() || p.parseKeyword(&flattenTripCountKw) || p.parseEqual() ||
      p.parseInteger(flattenTripCount) || p.parseComma() ||
      p.parseKeyword(&iterLatencyKw) || p.parseEqual() ||
      p.parseInteger(iterLatency) || p.parseComma() ||
      p.parseKeyword(&minIIKw) || p.parseEqual() || p.parseInteger(minII) ||
      p.parseGreater())
    return Attribute();

  if (flattenTripCountKw != "flattenTripCount" ||
      iterLatencyKw != "iterLatency" || minIIKw != "minII")
    return Attribute();

  return LoopInfoAttr::get(p.getContext(), flattenTripCount, iterLatency,
                           minII);
}

void LoopInfoAttr::print(AsmPrinter &p) const {
  p << "<flattenTripCount=" << getFlattenTripCount()
    << ", iterLatency=" << getIterLatency() << ", minII=" << getMinII() << ">";
}

//===----------------------------------------------------------------------===//
// LoopDirectiveAttr
//===----------------------------------------------------------------------===//

Attribute LoopDirectiveAttr::parse(AsmParser &p, Type type) {
  StringRef pipelineKw, targetIIKw, dataflowKw, flattenKw;
  StringRef pipeline, dataflow, flatten;
  int64_t targetII;
  if (p.parseLess() || p.parseKeyword(&pipelineKw) || p.parseEqual() ||
      p.parseKeyword(&pipeline) || p.parseComma() ||
      p.parseKeyword(&targetIIKw) || p.parseEqual() ||
      p.parseInteger(targetII) || p.parseComma() ||
      p.parseKeyword(&dataflowKw) || p.parseEqual() ||
      p.parseKeyword(&dataflow) || p.parseComma() ||
      p.parseKeyword(&flattenKw) || p.parseEqual() ||
      p.parseKeyword(&flatten) || p.parseGreater())
    return Attribute();

  if (pipelineKw != "pipeline" || targetIIKw != "targetII" ||
      dataflowKw != "dataflow" || flattenKw != "flatten")
    return Attribute();

  return LoopDirectiveAttr::get(p.getContext(), pipeline == "true", targetII,
                                dataflow == "true", flatten == "true");
}

void LoopDirectiveAttr::print(AsmPrinter &p) const {
  p << "<pipeline=" << getPipeline() << ", targetII=" << getTargetII()
    << ", dataflow=" << getDataflow() << ", flatten=" << getFlatten() << ">";
}

//===----------------------------------------------------------------------===//
// FuncDirectiveAttr
//===----------------------------------------------------------------------===//

Attribute FuncDirectiveAttr::parse(AsmParser &p, Type type) {
  StringRef pipelineKw, targetIntervalKw, dataflowKw;
  StringRef pipeline, dataflow;
  int64_t targetInterval;
  if (p.parseLess() || p.parseKeyword(&pipelineKw) || p.parseEqual() ||
      p.parseKeyword(&pipeline) || p.parseComma() ||
      p.parseKeyword(&targetIntervalKw) || p.parseEqual() ||
      p.parseInteger(targetInterval) || p.parseComma() ||
      p.parseKeyword(&dataflowKw) || p.parseEqual() ||
      p.parseKeyword(&dataflow) || p.parseGreater())
    return Attribute();

  if (pipelineKw != "pipeline" || targetIntervalKw != "targetInterval" ||
      dataflowKw != "dataflow")
    return Attribute();

  return FuncDirectiveAttr::get(p.getContext(), pipeline == "true",
                                targetInterval, dataflow == "true");
}

void FuncDirectiveAttr::print(AsmPrinter &p) const {
  p << "<pipeline=" << getPipeline()
    << ", targetInterval=" << getTargetInterval()
    << ", dataflow=" << getDataflow() << ">";
}

// --------------------------------------------------------------------------
// dataflow.port_width<N>
// Represents number of vector lanes for a top-level port, used when
// converting to !dataflow.axivec<elem, N>.
// C++: ::mlir::dataflow::PortWidthAttr
// --------------------------------------------------------------------------

const llvm::StringLiteral mlir::dataflow::kPortWidthAttrName("dataflow.port_width");

void mlir::dataflow::setPortWidthAttr(Operation *op, unsigned lanes) {
  auto attr = PortWidthAttr::get(op->getContext(), (int64_t)lanes);
  op->setAttr(kPortWidthAttrName, attr);
}

PortWidthAttr mlir::dataflow::getPortWidthAttr(Operation *op) {
  return op->getAttrOfType<PortWidthAttr>(kPortWidthAttrName);
}

bool mlir::dataflow::hasPortWidthAttr(Operation *op) {
  return (bool)getPortWidthAttr(op);
}


void mlir::dataflow::setPortWidthAttr(func::FuncOp func,
                                      unsigned argIndex, unsigned lanes) {
  auto attr = PortWidthAttr::get(func.getContext(), (int64_t)lanes);
  func.setArgAttr(argIndex, kPortWidthAttrName, attr);
}

PortWidthAttr mlir::dataflow::getPortWidthAttr(func::FuncOp func,
                                               unsigned argIndex) {
  return func.getArgAttrOfType<PortWidthAttr>(argIndex, kPortWidthAttrName);
}

bool mlir::dataflow::hasPortWidthAttr(func::FuncOp func,
                                      unsigned argIndex) {
  return (bool)getPortWidthAttr(func, argIndex);
}


void mlir::dataflow::setPortWidthAttr(BlockArgument arg, unsigned lanes) {
  auto func = dyn_cast<func::FuncOp>(arg.getOwner()->getParentOp());
  assert(func && "BlockArgument must belong to a func.func block");
  setPortWidthAttr(func, arg.getArgNumber(), lanes);
}

PortWidthAttr mlir::dataflow::getPortWidthAttr(BlockArgument arg) {
  auto func = dyn_cast<func::FuncOp>(arg.getOwner()->getParentOp());
  if (!func)
    return nullptr;
  return getPortWidthAttr(func, arg.getArgNumber());
}

bool mlir::dataflow::hasPortWidthAttr(BlockArgument arg) {
  return (bool)getPortWidthAttr(arg);
}


#include "streamhls/Dialect/Dataflow/IR/DataflowDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "streamhls/Dialect/Dataflow/IR/DataflowTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "streamhls/Dialect/Dataflow/IR/DataflowAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "streamhls/Dialect/Dataflow/IR/Dataflow.cpp.inc"