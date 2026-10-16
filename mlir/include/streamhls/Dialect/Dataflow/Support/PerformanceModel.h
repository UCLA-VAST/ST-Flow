#ifndef MLIR_STREAMHLS_DIALECT_DATAFLOW_SUPPORT_PERFORMANCEMODEL_H
#define MLIR_STREAMHLS_DIALECT_DATAFLOW_SUPPORT_PERFORMANCEMODEL_H

#include <iostream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <functional>
#include <string>
#include <algorithm>
#include <map>
#include <cassert>
#include <fstream>
#include <bit>
#include <memory>
#include "llvm/Support/raw_ostream.h"
// #include "streamhls/Dialect/Dataflow/Support/DFG.h"  <-- Removed to break circular dependency



namespace mlir { 
namespace streamhls {

struct ExprNode;
class SymbolicExpr {
public:
    std::shared_ptr<ExprNode> node;
    SymbolicExpr();
    SymbolicExpr(int64_t val);
    SymbolicExpr(std::string name, bool isParam);
    SymbolicExpr(std::shared_ptr<ExprNode> n) : node(n) {}

    SymbolicExpr operator+(const SymbolicExpr& other) const;
    SymbolicExpr operator-(const SymbolicExpr& other) const;
    SymbolicExpr operator*(const SymbolicExpr& other) const;
    SymbolicExpr operator/(const SymbolicExpr& other) const;

    std::string toAMPL() const;
    int64_t evaluate(const std::unordered_map<std::string, int64_t>& vars) const;
    std::vector<std::string> getDependencies() const;
};

SymbolicExpr max(SymbolicExpr a, SymbolicExpr b);
SymbolicExpr min(SymbolicExpr a, SymbolicExpr b);

struct GAConfig {
    int popSize = 50;
    int generations = 100;
    double mutationRate = 0.1;
    double crossoverRate = 0.8;
    std::string initMethod = "mixture"; // "random", "mixture"
    std::string csvPath = "";
};

struct RandomConfig {
    int iterations = 1;
};

struct SAConfig {
    double initialTemp = 1000.0;
    double coolingRate = 0.999;
    int iterations = 10000;
    std::string initialSolution = "default"; // "default" or "random"
    std::string csvPath = "";
};

struct DSEOptions {
    std::string permutationDSE = "sa";
    std::string unrollingDSE = "sa";
    std::string combinedDSE = "sa";
    std::string permutationAMPLSolver = "gecode";
    std::string unrollingAMPLSolver = "gecode";
    std::string combinedAMPLSolver = "gecode";
    int permutationTimeout = 10;
    int unrollingTimeout = 10;
    int combinedTimeout = 10;
    SAConfig permutationSAConfig;
    SAConfig unrollingSAConfig;
    SAConfig combinedSAConfig;
    GAConfig permutationGAConfig;
    GAConfig unrollingGAConfig;
    GAConfig combinedGAConfig;
    RandomConfig permutationRandomConfig;
    RandomConfig unrollingRandomConfig;
    RandomConfig combinedRandomConfig;
    int64_t dspLimit = 1024;
    int64_t memoryLimit = 1024; 
    int64_t minTimeLoopSize = 16;
    bool enforceDivisorOnlyTiling = true;
    std::string sharedTilingStrategy = "min";
    int64_t latencyMemoryRatio = 100;
    std::map<unsigned, unsigned> fixedPermutationSolution;
    std::map<std::string, int64_t> fixedUnrollingSolution;
    double balancedMappingSimilarityThreshold = 10.0;
    double balancedMappingUBThreshold = 5.0;
    int64_t objectiveScalingFactor = 1000;
};

enum class Operator {
    LessEqual, // "<="
    GreaterEqual, // ">="
    Equal // "=="
};


enum class CommentType {
    Variable,
    Parameter,
    Constraint,
    Assignment,
    Objective
};

struct Parameter {
    std::string name;
    std::string type;
    int64_t value;
    std::string comment = "";
};

struct Variable {
    std::string name;
    std::string type;
    std::optional<int64_t> lower;
    std::optional<int64_t> upper;
    std::vector<int64_t> discreteValues; // For tiling factors
    bool display = true;
    std::string comment = "";
};

struct VariableGroup {
    std::string name;
    std::vector<std::string> varNames;
    bool isOneHot = true;
};


std::ostream& operator<<(std::ostream& os, Operator op);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, Operator op);

struct Constraint {
    std::string name;
    SymbolicExpr lhsExpr;
    SymbolicExpr rhsExpr;
    std::string lhs; // String cache for AMPL
    std::string rhs; // String cache for AMPL
    Operator op; // e.g., "<=", ">=", "=="
    std::function<bool(std::unordered_map<std::string, int64_t>&)> evaluator;
    double penaltyWeight = 1e6;
    std::string comment = "";

    int64_t getViolation(const std::unordered_map<std::string, int64_t>& vars) const;
};

struct ObjConstraint {
  std::string name;
  std::string type;
  std::string lhs;
  std::string rhs;
  Operator op; // e.g., "<=", ">=", "=="
  std::function<void(std::unordered_map<std::string, int64_t>&)> evaluator;
};

struct Assignment {
    std::string name;
    std::string type;
    std::string lhs;
    std::string rhs;
    SymbolicExpr rhsExpr;
    std::string op; // e.g., "=", "<=", ">="
    bool isConstraint = false;
    std::function<void(std::unordered_map<std::string, int64_t>&)> evaluator;
    bool display = false;
    std::string comment = "";
};

struct Objective {
    SymbolicExpr expr;
    std::string expressionStr;
    std::function<int64_t(const std::unordered_map<std::string, int64_t>&)> evaluator;
    bool minimize = true;
    std::string comment = "";
};

class PerformanceModel;

struct SolveResult {
    bool success = false;
    int64_t latency = -1;  // Physical latency (cycles)
    int64_t memory = -1;   // Physical memory footprint (bytes)
    int64_t dsps = -1;     // Physical DSPs
    int64_t cost = -1;     // Optimization objective (combined cost)
    std::unordered_map<std::string, int64_t> variables;
    std::string fullOutput;

    void populate(const std::unordered_map<std::string, int64_t> &vars);
    void printReport(llvm::raw_ostream &os, llvm::StringRef label) const;
};


class PerformanceModel {
public:
    std::vector<Parameter> parameters;
    std::vector<Variable> variables;
    std::vector<Constraint> constraints;
    struct TilingGroup {
        std::string xVar;
        std::string zVar;
        int64_t totalTC;
        int64_t minZ;
    };
    std::vector<ObjConstraint> objConstraints;
    std::vector<Assignment> assignments;
    Objective objective;
    std::vector<TilingGroup> tilingGroups;
    std::vector<VariableGroup> groups;

    void addParameter(const Parameter& p) { parameters.push_back(p); }
    void addVariable(const Variable& v) { variables.push_back(v); }
    void addConstraint(const Constraint& c) { constraints.push_back(c); }
    void addObjConstraint(const ObjConstraint& a) { objConstraints.push_back(a); }
    void setObjective(const Objective& o) { objective = o; }
    void addTilingGroup(const TilingGroup& g) { tilingGroups.push_back(g); }
    void addAssignment(const Assignment& a) { assignments.push_back(a); }
    void addComment(CommentType type, llvm::StringRef comment);

    // Unified Symbolic Helpers
    SymbolicExpr var(std::string name);
    SymbolicExpr param(std::string name);
    void addAssignment(std::string name, SymbolicExpr expr, std::string type = "integer", bool display = false);
    void addConstraint(std::string name, SymbolicExpr lhs, Operator op, SymbolicExpr rhs);
    void setObjective(SymbolicExpr expr, bool minimize = true);
    void addOneHotGroup(std::string name, const std::vector<std::string>& varNames);

    void finalize(); // Sort assignments and validate
};

void writeAMPLModel(const PerformanceModel& model, const std::string& fileName, std::string solverName = "gecode", bool displayVars = true, bool displayAssignments = true);

SolveResult solvePerformanceModel(const std::string& fileName, int timeout = 10);

// Performance Model Evaluators
void runAssignments(const PerformanceModel& model, std::unordered_map<std::string, int64_t>& vars);
bool is_valid(const PerformanceModel& model, std::unordered_map<std::string, int64_t>& vars);
int64_t evaluate(const PerformanceModel& model, std::unordered_map<std::string, int64_t>& vars);

// Simulated Annealing Engine
std::unordered_map<std::string, int64_t> generateRandomSolution(const PerformanceModel& model);
std::unordered_map<std::string, int64_t> generateDefaultSolution(const PerformanceModel& model);
SolveResult 
runSimulatedAnnealing(const PerformanceModel& model, const SAConfig& config);
SolveResult 
runGeneticAlgorithm(const PerformanceModel& model, const GAConfig& config);
SolveResult 
runRandomSearch(const PerformanceModel& model, const RandomConfig& config);

}
}

#endif // MLIR_STREAMHLS_DIALECT_DATAFLOW_SUPPORT_PERFORMANCEMODEL_H