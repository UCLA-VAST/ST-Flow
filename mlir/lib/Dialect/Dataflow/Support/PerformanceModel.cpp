#include "streamhls/Dialect/Dataflow/Support/PerformanceModel.h"
#include "streamhls/Dialect/Dataflow/Support/DFG.h"
#include "streamhls/Dialect/Dataflow/Support/Utils.h"
#include <random>
#include <regex>
#include <sstream>
#include <unordered_set>
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>

#define DEBUG_TYPE "streamhls-performance-model"
// using namespace mlir;
// using namespace streamhls;
// using namespace dataflow;
// using namespace affine;

namespace mlir {
namespace streamhls {
using namespace affine;

static void printKV(llvm::raw_ostream &os, const std::unordered_map<std::string, int64_t> &vars, const std::string& prefix) {
    bool first = true;
    for (const auto& pair : vars) {
        if (pair.first.size() >= 2 && pair.first[0] == 'X' && pair.first[1] == '_') {
            if (!first) os << ", ";
            os << pair.first << "=" << pair.second;
            first = false;
        }
    }
}

//===----------------------------------------------------------------------===//
// Symbolic Expression System
//===----------------------------------------------------------------------===//

enum class ExprType { Constant, Variable, Parameter, Add, Sub, Mul, Div, Max, Min };

struct ExprNode {
    ExprType type;
    std::string name;
    int64_t value;
    std::shared_ptr<ExprNode> lhs, rhs;

    ExprNode(int64_t val) : type(ExprType::Constant), value(val), lhs(nullptr), rhs(nullptr) {}
    ExprNode(std::string n, bool isParam) : type(isParam ? ExprType::Parameter : ExprType::Variable), name(n), lhs(nullptr), rhs(nullptr) {}
    ExprNode(ExprType t, std::shared_ptr<ExprNode> l, std::shared_ptr<ExprNode> r) : type(t), lhs(l), rhs(r) {}

    std::string toAMPL() const {
        switch (type) {
            case ExprType::Constant: return std::to_string(value);
            case ExprType::Variable: 
            case ExprType::Parameter: return name;
            case ExprType::Add: return "(" + lhs->toAMPL() + " + " + rhs->toAMPL() + ")";
            case ExprType::Sub: return "(" + lhs->toAMPL() + " - " + rhs->toAMPL() + ")";
            case ExprType::Mul: return "(" + lhs->toAMPL() + " * " + rhs->toAMPL() + ")";
            case ExprType::Div: return "(" + lhs->toAMPL() + " / " + rhs->toAMPL() + ")";
            case ExprType::Max: return "max(" + lhs->toAMPL() + ", " + rhs->toAMPL() + ")";
            case ExprType::Min: return "min(" + lhs->toAMPL() + ", " + rhs->toAMPL() + ")";
        }
        return "";
    }

    int64_t evaluate(const std::unordered_map<std::string, int64_t>& vars) const {
        switch (type) {
            case ExprType::Constant: return value;
            case ExprType::Variable: 
            case ExprType::Parameter: 
                if (vars.count(name)) return vars.at(name);
                return 0;
            case ExprType::Add: return lhs->evaluate(vars) + rhs->evaluate(vars);
            case ExprType::Sub: return lhs->evaluate(vars) - rhs->evaluate(vars);
            case ExprType::Mul: return lhs->evaluate(vars) * rhs->evaluate(vars);
            case ExprType::Div: {
                int64_t den = rhs->evaluate(vars);
                return den == 0 ? 0 : lhs->evaluate(vars) / den;
            }
            case ExprType::Max: return std::max(lhs->evaluate(vars), rhs->evaluate(vars));
            case ExprType::Min: return std::min(lhs->evaluate(vars), rhs->evaluate(vars));
        }
        return 0;
    }

    void getDeps(std::unordered_set<std::string>& deps) const {
        if (type == ExprType::Variable) deps.insert(name);
        if (lhs) lhs->getDeps(deps);
        if (rhs) rhs->getDeps(deps);
    }
};

SymbolicExpr::SymbolicExpr() : node(nullptr) {}
SymbolicExpr::SymbolicExpr(int64_t val) : node(std::make_shared<ExprNode>(val)) {}
SymbolicExpr::SymbolicExpr(std::string name, bool isParam) : node(std::make_shared<ExprNode>(name, isParam)) {}

SymbolicExpr SymbolicExpr::operator+(const SymbolicExpr& other) const {
    return SymbolicExpr(std::make_shared<ExprNode>(ExprType::Add, node, other.node));
}
SymbolicExpr SymbolicExpr::operator-(const SymbolicExpr& other) const {
    return SymbolicExpr(std::make_shared<ExprNode>(ExprType::Sub, node, other.node));
}
SymbolicExpr SymbolicExpr::operator*(const SymbolicExpr& other) const {
    return SymbolicExpr(std::make_shared<ExprNode>(ExprType::Mul, node, other.node));
}
SymbolicExpr SymbolicExpr::operator/(const SymbolicExpr& other) const {
    return SymbolicExpr(std::make_shared<ExprNode>(ExprType::Div, node, other.node));
}

std::string SymbolicExpr::toAMPL() const { return node ? node->toAMPL() : "0"; }
int64_t SymbolicExpr::evaluate(const std::unordered_map<std::string, int64_t>& vars) const { return node ? node->evaluate(vars) : 0; }
std::vector<std::string> SymbolicExpr::getDependencies() const {
    std::unordered_set<std::string> deps;
    if (node) node->getDeps(deps);
    return std::vector<std::string>(deps.begin(), deps.end());
}

SymbolicExpr max(SymbolicExpr a, SymbolicExpr b) { return SymbolicExpr(std::make_shared<ExprNode>(ExprType::Max, a.node, b.node)); }
SymbolicExpr min(SymbolicExpr a, SymbolicExpr b) { return SymbolicExpr(std::make_shared<ExprNode>(ExprType::Min, a.node, b.node)); }

int64_t Constraint::getViolation(const std::unordered_map<std::string, int64_t>& vars) const {
    int64_t l = lhsExpr.evaluate(vars);
    int64_t r = rhsExpr.evaluate(vars);
    switch (op) {
        case Operator::LessEqual: return std::max<int64_t>(0, l - r);
        case Operator::GreaterEqual: return std::max<int64_t>(0, r - l);
        case Operator::Equal: return std::abs(l - r);
    }
    return 0;
}

//===----------------------------------------------------------------------===//
// Performance model helper functions
//===----------------------------------------------------------------------===//

std::ostream& operator<<(std::ostream& os, Operator op) {
  switch (op) {
      case Operator::LessEqual:    return os << "<=";
      case Operator::GreaterEqual: return os << ">=";
      case Operator::Equal:        return os << "=";
      default:                     return os << "?";
  }
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, Operator op) {
  switch (op) {
      case Operator::LessEqual:    return os << "<=";
      case Operator::GreaterEqual: return os << ">=";
      case Operator::Equal:        return os << "=";
      default:                     return os << "?";
  }
}

void PerformanceModel::addComment(CommentType type, llvm::StringRef comment) {
  switch(type){
    case CommentType::Variable:
      this->addVariable({"", "", std::nullopt, std::nullopt, {}, false, comment.str()});
      break;
    case CommentType::Parameter:
      this->addParameter({"", "", 0, comment.str()});
      break;
    case CommentType::Constraint:
      this->addConstraint({
        "",
        {}, {},
        "",
        "",
        Operator::LessEqual,
        nullptr,
        1e6,
        comment.str()
      });
      break;
    case CommentType::Assignment:
      this->addAssignment({
        "",
        "",
        "",
        "",
        {},
        "",
        false,
        nullptr,
        false,
        comment.str()
      });
      break;
    case CommentType::Objective:
      this->setObjective({
        {},
        "",
        nullptr,
        true,
        comment.str()
      });
      break;
  }
}

SymbolicExpr PerformanceModel::var(std::string name) { return SymbolicExpr(name, false); }
SymbolicExpr PerformanceModel::param(std::string name) { return SymbolicExpr(name, true); }

void PerformanceModel::addAssignment(std::string name, SymbolicExpr expr, std::string type, bool display) {
    Assignment a;
    a.name = name;
    a.lhs = name;
    a.rhsExpr = expr;
    a.rhs = expr.toAMPL();
    a.op = "=";
    a.type = type;
    a.display = display;
    a.evaluator = [name, expr](std::unordered_map<std::string, int64_t>& vars) {
        vars[name] = expr.evaluate(vars);
    };
    assignments.push_back(a);
}

void PerformanceModel::addConstraint(std::string name, SymbolicExpr lhs, Operator op, SymbolicExpr rhs) {
    Constraint c;
    c.name = name;
    c.lhsExpr = lhs;
    c.rhsExpr = rhs;
    c.lhs = lhs.toAMPL();
    c.rhs = rhs.toAMPL();
    c.op = op;
    c.evaluator = [lhs, rhs, op](std::unordered_map<std::string, int64_t>& vars) {
        int64_t l = lhs.evaluate(vars);
        int64_t r = rhs.evaluate(vars);
        switch (op) {
            case Operator::LessEqual: return l <= r;
            case Operator::GreaterEqual: return l >= r;
            case Operator::Equal: return l == r;
        }
        return false;
    };
    constraints.push_back(c);
}

void PerformanceModel::setObjective(SymbolicExpr expr, bool minimize) {
    objective.expr = expr;
    objective.expressionStr = expr.toAMPL();
    objective.minimize = minimize;
    objective.evaluator = [expr](const std::unordered_map<std::string, int64_t>& vars) {
        return expr.evaluate(vars);
    };
}

void PerformanceModel::addOneHotGroup(std::string name, const std::vector<std::string>& varNames) {
    groups.push_back({name, varNames, true});
}

void PerformanceModel::finalize() {
    // Topological sort of assignments
    if (assignments.empty()) return;

    std::vector<Assignment> sorted;
    std::unordered_set<std::string> readyVars;
    for (const auto& p : parameters) readyVars.insert(p.name);
    for (const auto& v : variables) {
        // Variables that are search variables (not assigned) are ready
        bool assigned = false;
        for (const auto& a : assignments) if (a.lhs == v.name) assigned = true;
        if (!assigned) readyVars.insert(v.name);
    }

    std::vector<Assignment> pending = assignments;
    while (!pending.empty()) {
        bool changed = false;
        for (auto it = pending.begin(); it != pending.end(); ) {
            auto deps = it->rhsExpr.getDependencies();
            bool allReady = true;
            for (const auto& d : deps) {
                if (readyVars.find(d) == readyVars.end()) { allReady = false; break; }
            }
            if (allReady) {
                readyVars.insert(it->lhs);
                sorted.push_back(*it);
                it = pending.erase(it);
                changed = true;
            } else {
                ++it;
            }
        }
        if (!changed) {
            break; 
        }
    }
    // Append remaining (circularly dependent or broken)
    for (const auto& a : pending) sorted.push_back(a);
    assignments = sorted;
}


//===----------------------------------------------------------------------===//
// Model writers/evaluators
//===----------------------------------------------------------------------===//
void SolveResult::populate(const std::unordered_map<std::string, int64_t> &vars) {
    this->variables = vars;
    
    if (vars.count("latency")) {
        this->latency = vars.at("latency");
    }
    if (vars.count("totalMemory")) {
        this->memory = vars.at("totalMemory");
    }
    if (vars.count("totalDSPs")) {
        this->dsps = vars.at("totalDSPs");
    }
    if (vars.count("combined_cost")) {
        this->cost = vars.at("combined_cost");
    }
}

void SolveResult::printReport(llvm::raw_ostream &os, llvm::StringRef label) const {
    std::string line = "+---------------------------+------------+------------+------------+------------+";
    os << label << "\n";
    os << line << "\n";
    os << "| Stage                     | Latency    | Memory     | DSPs       | Cost       |\n";
    os << line << "\n";
    
    // Format values to fit in the table
    auto formatVal = [](int64_t val) {
        std::string s = std::to_string(val);
        if (s.length() > 10) s = s.substr(0, 10);
        while (s.length() < 10) s += " ";
        return s;
    };

    std::string stageName = label.str();
    if (stageName.length() > 25) stageName = stageName.substr(0, 25);
    while (stageName.length() < 25) stageName += " ";

    os << "| " << stageName << " | " << formatVal(this->latency) << " | " 
       << formatVal(this->memory) << " | " << formatVal(this->dsps) << " | " 
       << formatVal(this->cost) << " |\n";
    os << line << "\n";
}

static void writeAMPLModelTmpForRealVals(const PerformanceModel& model, llvm::raw_fd_ostream& out) {
  
  out << "option solver gecode;\n";
//   out << "option gurobi_options 'lim:time=20 tech:logfile=gurobi.log qp:nonconvex=2';\n";
  
  for (const auto& p : model.parameters) {
    out << "param " << p.name << " := " << p.value << ";\n";
  }
  // out << "\n";
  // for (const auto& p : model.comments) {
  //   out << "# " << p.comment << "\n";
  // }
  out << "\n";
  for (const auto& v : model.variables) {
      // out << "var " << v.name << " " << v.type;
      out << "var " << v.name;
      if (v.lower)
          out << " >= " << *v.lower;
      if (v.upper)
          out << " <= " << *v.upper;
      out << ";\n";
  }
  out << "\n";


  for (const auto& c : model.objConstraints) {
      // out << "var " << a.name << ", " << a.type << ";\n";
      // out << "subject to " << a.name << "_def:\n\t" << a.lhs << " " << a.op << " " << a.rhs << ";\n";
      out << "subject to " << c.name << ":\n\t" << c.lhs << " " << c.op << " " << c.rhs << ";\n";
  }
  out << "\n";
  // out << (model.objective.minimize ? "minimize" : "maximize") << " obj: " << model.objective.expressionStr << ";\n";

  for (const auto& a : model.assignments){
    // out << "var " << a.name << ", " << a.type << ";\n";
    out << "var " << a.name <<";\n";
    out << "subject to " << a.name << "_def:\n\t" << a.lhs << " " << a.op << " " << a.rhs << ";\n";
  }
  out << "minimize cost: " << model.objective.expressionStr << ";\n";
  
  for (const auto& c : model.constraints) {
    out << "subject to " << c.name << ":\n\t" << c.lhs << " " << c.op << " " << c.rhs << ";\n";
  }
  out << "\n";
  
  out << "solve;\n";
  out << "display cost;\n";
  for (const auto& v : model.variables) {
    out << "display " << v.name << ";\n";
  }

}
void writeAMPLModel(const PerformanceModel& model, const std::string& fileName, std::string solverName, bool displayVars, bool displayAssignments) {
  std::error_code ec;
  llvm::raw_fd_ostream out(fileName, ec);
  if (ec) {
    llvm::errs() << "Could not open model file: " << fileName << "\n";
    return;
  }

  out << "option solver " << solverName << ";\n";
//   out << "option gurobi_options 'lim:time=60 tech:logfile=gurobi.log qp:nonconvex=2 mip:gap=0.0000000001';\n";
  for (const auto& p : model.parameters) {
    if(p.comment != ""){
      // Handle multi-line comments: prefix each line with #
      std::istringstream commentStream(p.comment);
      std::string line;
      while (std::getline(commentStream, line)) {
        out << "# " << line << "\n";
      }
    }
    // Skip parameters without a name (they are comment-only)
    if(p.name.empty())
      continue;
    out << "param " << p.name << " := " << p.value << ";\n";
  }
  out << "\n";
  out << "\n";
  for (const auto& v : model.variables) {
    if(v.comment != ""){
      // Handle multi-line comments: prefix each line with #
      std::istringstream commentStream(v.comment);
      std::string line;
      while (std::getline(commentStream, line)) {
        out << "# " << line << "\n";
      }
    }
    // Skip variables without a name (they are comment-only)
    if(v.name.empty())
      continue;
      
    if(v.type == "integer" || v.type == "binary")
      out << "var " << v.name << ", " << v.type;
    else
      out << "var " << v.name;

    if (v.lower)
      out << " >= " << *v.lower;
    if (v.upper)
      out << " <= " << *v.upper;
    out << ";\n";
  }
  out << "\n";


  for (const auto& c : model.objConstraints) {
      // out << "var " << a.name << ", " << a.type << ";\n";
      // out << "subject to " << a.name << "_def:\n\t" << a.lhs << " " << a.op << " " << a.rhs << ";\n";
        out << "subject to " << c.name << ":\n\t" << c.lhs << " " << c.op << " " << c.rhs << ";\n";
  }
  out << "\n";
  // out << (model.objective.minimize ? "minimize" : "maximize") << " obj: " << model.objective.expressionStr << ";\n";

  for (const auto& a : model.assignments){
    if(a.comment != ""){
      // Handle multi-line comments: prefix each line with #
      std::istringstream commentStream(a.comment);
      std::string line;
      while (std::getline(commentStream, line)) {
        out << "# " << line << "\n";
      }
    }
    // Skip assignments without a name (they are comment-only)
    if (a.name.empty())
      continue;
    if(a.type == "integer" || a.type == "binary")
      out << "var " << a.name << ", " << a.type << ";\n";
    else
      out << "var " << a.name << ";\n";

    out << "subject to " << a.name << "_def:\n\t" << a.lhs << " " << a.op << " " << a.rhs << ";\n";
  }
  for (const auto& c : model.constraints) {
    if(c.comment != ""){
      // Handle multi-line comments: prefix each line with #
      std::istringstream commentStream(c.comment);
      std::string line;
      while (std::getline(commentStream, line)) {
        out << "# " << line << "\n";
      }
    }
    // Skip constraints without a name (they are comment-only)
    out << "subject to " << c.name << ":\n\t" << c.lhs << " " << c.op << " " << c.rhs << ";\n";
  }
  out << "\n";
  
  // Handle variable groups (e.g. one-hot sum constraints)
  for (const auto& g : model.groups) {
    if (g.isOneHot && !g.varNames.empty()) {
      out << "subject to c_onehot_" << g.name << ":\n\t (";
      for (size_t i = 0; i < g.varNames.size(); ++i) {
        out << g.varNames[i] << (i == g.varNames.size() - 1 ? "" : " + ");
      }
      out << ") == 1;\n";
    }
  }
  out << "\n";
  
  out << "minimize cost: " << model.objective.expressionStr << ";\n";
  out << "solve;\n";
  if(displayVars){
    for (const auto& v : model.variables) {
      if(v.display)
        out << "display " << v.name << ";\n";
    }
  }
  if(displayAssignments){
    for (const auto& a : model.assignments) {
      if (a.display || 
          (a.name.size() >= 2 && (a.name.substr(0, 2) == "st" || 
                                 a.name.substr(0, 2) == "lw" || 
                                 a.name.substr(0, 2) == "fw"))) {
        out << "display " << a.name << ";\n";
      }
    }
    out << "display cost;\n";
}
}

SolveResult solvePerformanceModel(const std::string& fileName, int timeout) {
    char buffer[128];
    SolveResult result;
    std::string command = "timeout " + std::to_string(timeout * 60 + 5) + "s ampl " + fileName + " 2>&1";
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        llvm::errs() << "popen() failed!\n";
        return result;
    }
    while (fgets(buffer, sizeof buffer, pipe) != NULL) {
        result.fullOutput += buffer;
    }
    pclose(pipe);

    // Initial parsing for success
    if (result.fullOutput.find("infeasible") == std::string::npos && 
        result.fullOutput.find("error") == std::string::npos &&
        result.fullOutput.find("license") == std::string::npos &&
        result.fullOutput.find("No solution") == std::string::npos &&
        result.fullOutput.find("invalid") == std::string::npos) {
        result.success = true;
    }

    // Parse cost
    std::regex costPattern("cost = ([0-9]+)");
    std::smatch match;
    if (std::regex_search(result.fullOutput, match, costPattern)) {
        result.cost = std::stoll(match[1].str());
    } else {
        // If cost is not found, it's likely a failure even if no error string was matched
        result.success = false;
    }

    // Parse DSPs (looking for dsp = #, dsps = # or totalDSPs = #)
    std::regex dspPattern("totalDSPs = ([0-9]+)");
    if (std::regex_search(result.fullOutput, match, dspPattern)) {
        result.dsps = std::stoll(match[1].str());
    }

    // Parse variables (e.g., X_# = # or b#_# = #)
    std::regex varPattern("([a-zA-Z0-9_]+) = ([0-9]+)");
    std::istringstream iss(result.fullOutput);
    std::string line;
    while (std::getline(iss, line)) {
        if (std::regex_search(line, match, varPattern)) {
            result.variables[match[1].str()] = std::stoll(match[2].str());
        }
    }

    if (result.variables.count("latency")) {
        result.populate(result.variables);
    }

    return result;
}


// ==========================================================
// Performance Model Evaluators
// ==========================================================

void runAssignments(const PerformanceModel& model, std::unordered_map<std::string, int64_t>& vars) {
    for (const auto& a : model.assignments) {
        if (a.evaluator) {
            a.evaluator(vars);
        }
    }
}

bool is_valid(const PerformanceModel& model, std::unordered_map<std::string, int64_t>& vars) {
    runAssignments(model, vars);
    for (const auto& c : model.constraints) {
        if (c.evaluator && !c.evaluator(vars)) {
            LLVM_DEBUG(llvm::dbgs() << "    - Invalid: Constraint failed: " << (c.name.empty() ? "unnamed" : c.name) << "\n");
            return false;
        }
    }
    
    // Check variable bounds
    for (const auto& v : model.variables) {
        if (vars.count(v.name)) {
            int64_t val = vars.at(v.name);
            if (v.lower && val < *v.lower) {
                LLVM_DEBUG(llvm::dbgs() << "    - Invalid: Variable out of bounds: " << v.name << " (" << val << " < " << *v.lower << ")\n");
                return false;
            }
            if (v.upper && val > *v.upper) {
                LLVM_DEBUG(llvm::dbgs() << "    - Invalid: Variable out of bounds: " << v.name << " (" << val << " > " << *v.upper << ")\n");
                return false;
            }
        }
    }

    // Check one-hot groups
    for (const auto& group : model.groups) {
        if (group.isOneHot) {
            int count = 0;
            for (const auto& vn : group.varNames) {
                if (vars.count(vn) && vars.at(vn) == 1) count++;
            }
            if (count != 1) {
                LLVM_DEBUG(llvm::dbgs() << "    - Invalid: One-hot violation: " << group.name << " (count=" << count << ")\n");
                return false;
            }
        }
    }

    return true;
}

int64_t evaluate(const PerformanceModel& model, std::unordered_map<std::string, int64_t>& vars) {
    runAssignments(model, vars);
    int64_t cost = 0;
    if (model.objective.evaluator) {
        cost = model.objective.evaluator(vars);
    }
    
    // Add penalties for soft constraints
    double penalty = 0;
    for (const auto& c : model.constraints) {
        penalty += c.getViolation(vars) * c.penaltyWeight;
    }

    // Add penalties for one-hot violations
    for (const auto& group : model.groups) {
        if (group.isOneHot) {
            int count = 0;
            for (const auto& vn : group.varNames) {
                if (vars.count(vn) && vars.at(vn) == 1) count++;
            }
            if (count != 1) {
                penalty += std::abs(count - 1) * 1e9; // Large penalty
            }
        }
    }

    // Add penalties for out-of-bounds variables
    for (const auto& v : model.variables) {
        if (vars.count(v.name)) {
            int64_t val = vars.at(v.name);
            if (v.lower && val < *v.lower) penalty += (*v.lower - val) * 1e6;
            if (v.upper && val > *v.upper) penalty += (val - *v.upper) * 1e6;
        }
    }

    return cost + (int64_t)penalty;
}

// ==========================================================
// Simulated Annealing Engine
// ==========================================================

std::unordered_map<std::string, int64_t> generateRandomSolution(const PerformanceModel& model) {
    std::unordered_map<std::string, int64_t> vars;
    static std::random_device rd;
    static std::mt19937 gen(rd());

    for (const auto& p : model.parameters) {
        vars[p.name] = p.value;
    }

    std::unordered_set<std::string> handledVars;
    std::unordered_map<int, std::vector<std::string>> bVarGroups;

    // Use explicit VariableGroups for handledVars
    for (const auto& group : model.groups) {
        if (group.isOneHot) {
            for (const auto& v : group.varNames) {
                vars[v] = 0;
                handledVars.insert(v);
            }
            std::uniform_int_distribution<> dist(0, group.varNames.size() - 1);
            vars[group.varNames[dist(gen)]] = 1;
        }
    }

    // Fallback regex logic for backward compatibility or unplanned groups
    for (const auto& v : model.variables) {
        if (handledVars.count(v.name)) continue;
        std::regex bPattern("b([0-9]+)_([0-9]+)");
        std::smatch match;
        if (std::regex_match(v.name, match, bPattern)) {
            int nodeId = std::stoi(match[1].str());
            bVarGroups[nodeId].push_back(v.name);
            vars[v.name] = 0;
        }
    }
    for (auto& [nodeId, group] : bVarGroups) {
        if (handledVars.count(group[0])) continue;
        std::uniform_int_distribution<> dist(0, group.size() - 1);
        vars[group[dist(gen)]] = 1;
        for (const auto& vn : group) handledVars.insert(vn);
    }

    // Tiling variables with discrete domains
    for (const auto& v : model.variables) {
        if (handledVars.count(v.name)) continue;
        if (!v.discreteValues.empty()) {
            std::uniform_int_distribution<> dist(0, v.discreteValues.size() - 1);
            vars[v.name] = v.discreteValues[dist(gen)];
            handledVars.insert(v.name);
        }
    }

    // Handle legacy tiling groups
    for (const auto& g : model.tilingGroups) {
        if (handledVars.count(g.xVar)) continue;
        int64_t chosenX = 1;
        const Variable* xVarDef = nullptr;
        for (const auto& v : model.variables) {
            if (v.name == g.xVar) {
                xVarDef = &v;
                break;
            }
        }

        if (xVarDef && !xVarDef->discreteValues.empty()) {
            std::uniform_int_distribution<> dist(0, xVarDef->discreteValues.size() - 1);
            chosenX = xVarDef->discreteValues[dist(gen)];
        } else if (xVarDef && xVarDef->lower && xVarDef->upper) {
            int64_t lb = *xVarDef->lower;
            int64_t ub = *xVarDef->upper;
            if (lb > ub) std::swap(lb, ub);
            std::uniform_int_distribution<int64_t> dist(lb, ub);
            chosenX = dist(gen);
        } else {
            std::vector<int64_t> divisors;
            for (int64_t d = 1; d * d <= g.totalTC; ++d) {
                if (g.totalTC % d == 0) {
                    divisors.push_back(d);
                    if (d * d != g.totalTC) divisors.push_back(g.totalTC / d);
                }
            }
            std::uniform_int_distribution<> dist(0, divisors.size() - 1);
            chosenX = divisors[dist(gen)];
        }

        vars[g.xVar] = chosenX;
        vars[g.zVar] = std::max(g.minZ, (g.totalTC + chosenX - 1) / chosenX);
        handledVars.insert(g.xVar);
        handledVars.insert(g.zVar);
    }

    // Initialize remaining randomly
    for (const auto& v : model.variables) {
        if (handledVars.count(v.name)) continue;
        int64_t lb = v.lower.value_or(0);
        int64_t ub = v.upper.value_or(100);
        if (lb > ub) std::swap(lb, ub);
        std::uniform_int_distribution<int64_t> dist(lb, ub);
        vars[v.name] = dist(gen);
    }
    return vars;
}

std::unordered_map<std::string, int64_t> generateDefaultSolution(const PerformanceModel& model) {
    std::unordered_map<std::string, int64_t> vars;

    for (const auto& p : model.parameters) {
        vars[p.name] = p.value;
    }

    // Group binary variables by node
    std::unordered_map<int, std::vector<std::string>> bVarGroups;
    std::unordered_set<std::string> handledVars;

    for (const auto& v : model.variables) {
        std::regex bPattern("b([0-9]+)_([0-9]+)");
        std::smatch match;
        if (std::regex_match(v.name, match, bPattern)) {
            int nodeId = std::stoi(match[1].str());
            bVarGroups[nodeId].push_back(v.name);
            vars[v.name] = 0; // Initialize all to 0
            handledVars.insert(v.name);
        }
    }

    // For each binary group, pick the first one (index 0) to be 1
    for (auto& [nodeId, group] : bVarGroups) {
        std::sort(group.begin(), group.end());
        vars[group[0]] = 1;
    }

    // Handle tiling groups (default factor 1)
    for (const auto& g : model.tilingGroups) {
        vars[g.xVar] = 1;
        vars[g.zVar] = g.totalTC;
        handledVars.insert(g.xVar);
        handledVars.insert(g.zVar);
    }

    // Initialize others to lower bounds
    for (const auto& v : model.variables) {
        if (handledVars.count(v.name)) continue;
        vars[v.name] = v.lower.value_or(0);
    }
    return vars;
}

std::unordered_map<std::string, int64_t> generateNeighbor(
    const PerformanceModel& model,
    const std::unordered_map<std::string, int64_t>& current,
    const std::vector<std::string>& searchVars) {
    
    std::unordered_map<std::string, int64_t> neighbor = current;
    if (searchVars.empty()) return neighbor;

    static std::random_device rd;
    static std::mt19937 gen(rd());
    
    std::uniform_int_distribution<> varDist(0, searchVars.size() - 1);
    std::string varName = searchVars[varDist(gen)];

    // 1. Check if it is in an explicit group
    for (const auto& group : model.groups) {
        if (group.isOneHot) {
            auto it = std::find(group.varNames.begin(), group.varNames.end(), varName);
            if (it != group.varNames.end()) {
                for (const auto& vn : group.varNames) neighbor[vn] = 0;
                std::uniform_int_distribution<> dist(0, group.varNames.size() - 1);
                neighbor[group.varNames[dist(gen)]] = 1;
                return neighbor;
            }
        }
    }

    // 2. Check if it is a bVar (regex)
    std::regex bPattern("b([0-9]+)_([0-9]+)");
    std::smatch match;
    if (std::regex_match(varName, match, bPattern)) {
        int nodeId = std::stoi(match[1].str());
        std::vector<std::string> group;
        for (const auto& v : model.variables) {
            std::smatch m;
            if (std::regex_match(v.name, m, bPattern) && std::stoi(m[1].str()) == nodeId) {
                group.push_back(v.name);
                neighbor[v.name] = 0;
            }
        }
        std::uniform_int_distribution<> dist(0, group.size() - 1);
        neighbor[group[dist(gen)]] = 1;
        return neighbor;
    }

    // 3. Check if it is part of a tiling group
    for (const auto& g : model.tilingGroups) {
        if (varName == g.xVar || varName == g.zVar) {
            std::string xVar = g.xVar;
            int64_t chosenX = current.at(xVar);
            
            const Variable* xVarDef = nullptr;
            for (const auto& v : model.variables) {
                if (v.name == xVar) {
                    xVarDef = &v;
                    break;
                }
            }

            if (xVarDef && !xVarDef->discreteValues.empty()) {
                std::uniform_int_distribution<> dist(0, xVarDef->discreteValues.size() - 1);
                chosenX = xVarDef->discreteValues[dist(gen)];
            } else if (xVarDef && xVarDef->lower && xVarDef->upper) {
                int64_t lb = *xVarDef->lower;
                int64_t ub = *xVarDef->upper;
                if (lb > ub) std::swap(lb, ub);
                std::uniform_int_distribution<int64_t> dist(lb, ub);
                chosenX = dist(gen);
                LLVM_DEBUG(llvm::dbgs() << "Randomly chose X for variable " << xVar << " as " << chosenX << " with bounds (" << lb << ", " << ub << ").\n");
            } else {
                std::vector<int64_t> divisors;
                for (int64_t d = 1; d * d <= g.totalTC; ++d) {
                    if (g.totalTC % d == 0) {
                        divisors.push_back(d);
                        if (d * d != g.totalTC) divisors.push_back(g.totalTC / d);
                    }
                }
                std::uniform_int_distribution<> dist(0, divisors.size() - 1);
                chosenX = divisors[dist(gen)];
            }

            neighbor[xVar] = chosenX;
            for (const auto& tg : model.tilingGroups) {
                if (tg.xVar == xVar) {
                    neighbor[tg.zVar] = std::max(tg.minZ, (tg.totalTC + chosenX - 1) / chosenX);
                    LLVM_DEBUG(llvm::dbgs() << "Randomly chose Z for variable " << tg.zVar << " as " << neighbor[tg.zVar] << " with bounds (" << 1 << ", " << tg.totalTC << ").\n");
                }
            }
            return neighbor;
        }
    }

    // 4. Discrete domain variable
    for (const auto& v : model.variables) {
        if (v.name == varName && !v.discreteValues.empty()) {
            std::uniform_int_distribution<> dist(0, v.discreteValues.size() - 1);
            neighbor[varName] = v.discreteValues[dist(gen)];
            return neighbor;
        }
    }

    // 3. General variable perturb
    for (const auto& v : model.variables) {
        if (v.name == varName) {
            int64_t lb = v.lower.value_or(0);
            int64_t ub = v.upper.value_or(100);
            if (lb > ub) std::swap(lb, ub);
            
            if (v.type == "binary" || (lb == 0 && ub == 1)) {
                neighbor[varName] = 1 - current.at(varName);
            } else {
                std::uniform_int_distribution<int64_t> valDist(lb, ub);
                neighbor[varName] = valDist(gen);
            }
            break;
        }
    }
    return neighbor;
}

SolveResult 
runSimulatedAnnealing(const PerformanceModel& model, const SAConfig& config) {
    LLVM_DEBUG(llvm::dbgs() << "  Running Simulated Annealing...\n");
    // Identify search variables (those that aren't outputs of assignments)
    std::unordered_set<std::string> assignedVars;
    for (const auto& a : model.assignments) assignedVars.insert(a.lhs);
    std::vector<std::string> searchVars;
    for (const auto& v : model.variables) {
        if (assignedVars.find(v.name) == assignedVars.end() && !v.name.empty()) {
            searchVars.push_back(v.name);
        }
    }

    LLVM_DEBUG(
        llvm::dbgs() << "  SA: Search Variables: ";
        for (const auto& v : searchVars) llvm::dbgs() << v << " ";
        llvm::dbgs() << "\n";
        llvm::dbgs() << "  SA: Tiling Groups:\n";
        for (const auto& g : model.tilingGroups) {
            llvm::dbgs() << "    - " << g.xVar << ", " << g.zVar << " (Total TC: " << g.totalTC << ") Divisors: ";
            for (int64_t d = 1; d * d <= g.totalTC; ++d) {
                if (g.totalTC % d == 0) {
                    llvm::dbgs() << d << " ";
                    if (d * d != g.totalTC) llvm::dbgs() << g.totalTC / d << " ";
                }
            }
            llvm::dbgs() << "\n";
        }
    );

    llvm::dbgs() << "  SA: Generating initial solution...\n";
    std::unordered_map<std::string, int64_t> current;
    
    if (config.initialSolution == "default") {
        current = generateDefaultSolution(model);
        if (!is_valid(model, current)) {
            llvm::dbgs() << "  SA: Default solution invalid, falling back to random...\n";
            current = generateRandomSolution(model);
        }
    } else {
        current = generateRandomSolution(model);
    }

    int restartCount = 0;
    while (!is_valid(model, current) && config.iterations > 0) {
        current = generateRandomSolution(model);
        if (++restartCount % 100 == 0) {
            llvm::dbgs() << "  SA: Random search for valid initial solution (Try " << restartCount << ")...\n";
        }
    }
    llvm::dbgs() << "  SA: Initial solution found (Valid: " << is_valid(model, current) << ")\n";

    auto best = current;
    int64_t bestCost = evaluate(model, current);
    double T = config.initialTemp;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> probDist(0.0, 1.0);
    
    std::unique_ptr<llvm::raw_fd_ostream> csvFile = nullptr;
    if (!config.csvPath.empty()) {
        std::error_code ec;
        llvm::SmallString<128> csvDirPath(config.csvPath);
        llvm::sys::path::remove_filename(csvDirPath);
        if (!csvDirPath.empty()) {
            llvm::sys::fs::create_directories(csvDirPath);
        }
        csvFile = std::make_unique<llvm::raw_fd_ostream>(config.csvPath, ec, llvm::sys::fs::OF_None);
        if (!ec) {
            *csvFile << "iteration,cost\n";
        } else {
            llvm::errs() << "Warning: Could not open CSV file for SA logging: " << config.csvPath << "\n";
        }
    }

    for (int i = 0; i < config.iterations; ++i) {
        if (i % 1000 == 0) {
            llvm::dbgs() << "  SA Iteration " << i << "/" << config.iterations 
                          << " (Best Cost: " << bestCost << ", T: " << T << ")\n";
        }
        
        int64_t currentCost = evaluate(model, current);
        if (csvFile && (i % 100 == 0 || i == config.iterations - 1)) {
            *csvFile << i << "," << currentCost << "\n";
            csvFile->flush();
        }
        auto neighbor = generateNeighbor(model, current, searchVars);
        if (!is_valid(model, neighbor)) {
            LLVM_DEBUG(
                llvm::dbgs() << "    SA: Neighbor invalid! [";
                for (const auto& c : model.constraints) {
                    if (c.getViolation(neighbor) > 0) {
                        llvm::dbgs() << c.name << " (viol: " << c.getViolation(neighbor) << ") ";
                    }
                }
                llvm::dbgs() << "]\n";
            );
            continue;
        }

        LLVM_DEBUG(
            llvm::dbgs() << "    SA: Neighbor is valid. Checking content: [";
            printKV(llvm::dbgs(), neighbor, "X_");
            llvm::dbgs() << "]\n";
        );

        int64_t neighborCost = evaluate(model, neighbor);
        LLVM_DEBUG(
            llvm::dbgs() << "    SA: Neighbor cost: " << neighborCost << "\n";
        );

        int64_t delta = neighborCost - currentCost;
        LLVM_DEBUG(
            llvm::dbgs() << "    SA: delta: " << delta << "\n";
        );

        bool accept = false;
        if (delta < 0) accept = true;
        else accept = probDist(gen) < std::exp(-delta / T);

        if (accept) {
            LLVM_DEBUG(
              llvm::dbgs() << "    SA: Accepted move (Current: " << currentCost 
                           << ", Neighbor: " << neighborCost << ", Best: " << bestCost << ") [";
              printKV(llvm::dbgs(), neighbor, "X_");
              llvm::dbgs() << "]\n";
            );
            current = neighbor;
        }

        if (neighborCost < bestCost) {
            LLVM_DEBUG(
              llvm::dbgs() << "    SA: New Best Solution found! (Prev Best: " << bestCost 
                           << ", New Best: " << neighborCost << ") [";
              printKV(llvm::dbgs(), neighbor, "X_");
              llvm::dbgs() << "]\n";
            );
            best = neighbor;
            bestCost = neighborCost;
        }
        T *= config.coolingRate;
    }

    SolveResult result;
    result.success = is_valid(model, best);
    result.cost = bestCost;
    result.populate(best);
    return result;
}


SolveResult 
runGeneticAlgorithm(const PerformanceModel& model, const GAConfig& config) {
    LLVM_DEBUG(llvm::dbgs() << "  Running Genetic Algorithm...\n");
    std::unordered_set<std::string> assignedVars;
    for (const auto& a : model.assignments) assignedVars.insert(a.lhs);
    
    std::vector<std::string> searchVars;
    for (const auto& v : model.variables) {
        if (assignedVars.find(v.name) == assignedVars.end() && !v.name.empty()) {
            searchVars.push_back(v.name);
        }
    }

    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> probDist(0.0, 1.0);

    struct Individual {
        std::unordered_map<std::string, int64_t> vars;
        int64_t cost = 0;
        bool valid = false;
    };

    auto evaluateInd = [&](std::unordered_map<std::string, int64_t> vars) -> Individual {
        runAssignments(model, vars);
        bool valid = is_valid(model, vars);
        int64_t cost = evaluate(model, vars);
        return {vars, cost, valid};
    };

    llvm::dbgs() << "  GA: Initializing population with size " << config.popSize << "...\n";
    std::vector<Individual> population;
    for (int i = 0; i < config.popSize; ++i) {
        std::unordered_map<std::string, int64_t> vars;
        if (config.initMethod == "mixture" && i < config.popSize / 5) {
            vars = generateDefaultSolution(model);
        } else {
            vars = generateRandomSolution(model);
        }
        
        int attempts = 0;
        while (!is_valid(model, vars) && attempts < 100) {
            vars = generateRandomSolution(model);
            attempts++;
        }
        population.push_back(evaluateInd(vars));
    }

    auto getBest = [&](const std::vector<Individual>& pop) -> Individual {
        Individual best = pop[0];
        for (const auto& ind : pop) {
            if (ind.valid && (!best.valid || ind.cost < best.cost)) {
                best = ind;
            }
        }
        return best;
    };

    Individual globalBest = getBest(population);

    std::unique_ptr<llvm::raw_fd_ostream> csvFile = nullptr;
    if (!config.csvPath.empty()) {
        std::error_code ec;
        llvm::SmallString<128> csvDirPath(config.csvPath);
        llvm::sys::path::remove_filename(csvDirPath);
        if (!csvDirPath.empty()) {
            llvm::sys::fs::create_directories(csvDirPath);
        }
        csvFile = std::make_unique<llvm::raw_fd_ostream>(config.csvPath, ec, llvm::sys::fs::OF_None);
        if (!ec) {
            *csvFile << "generation,best_cost,avg_cost\n";
        } else {
            llvm::errs() << "Warning: Could not open CSV file for GA logging: " << config.csvPath << "\n";
        }
    }

    LLVM_DEBUG(
        llvm::dbgs() << "  GA: Search Variables: ";
        for (const auto& v : searchVars) llvm::dbgs() << v << " ";
        llvm::dbgs() << "\n";
        llvm::dbgs() << "  GA: Tiling Groups:\n";
        for (const auto& g : model.tilingGroups) {
            llvm::dbgs() << "    - " << g.xVar << ", " << g.zVar << " (Total TC: " << g.totalTC << ") Divisors: ";
            for (int64_t d = 1; d * d <= g.totalTC; ++d) {
                if (g.totalTC % d == 0) {
                    llvm::dbgs() << d << " ";
                    if (d * d != g.totalTC) llvm::dbgs() << g.totalTC / d << " ";
                }
            }
            llvm::dbgs() << "\n";
        }
    );

    auto tournamentSelection = [&](const std::vector<Individual>& pop, int k) -> Individual {
        std::uniform_int_distribution<> dist(0, pop.size() - 1);
        int bestIdx = dist(gen);
        for (int i = 1; i < k; ++i) {
            int idx = dist(gen);
            if (pop[idx].valid && (!pop[bestIdx].valid || pop[idx].cost < pop[bestIdx].cost)) {
                bestIdx = idx;
            }
        }
        return pop[bestIdx];
    };

    auto crossover = [&](const Individual& p1, const Individual& p2) -> Individual {
        std::unordered_map<std::string, int64_t> childVars = p1.vars;
        std::unordered_set<std::string> handledVars;

        // 1. Swap one-hot groups
        for (const auto& group : model.groups) {
            if (group.isOneHot) {
                if (probDist(gen) < 0.5) {
                    for (const auto& vn : group.varNames) {
                        childVars[vn] = p2.vars.at(vn);
                    }
                }
                for (const auto& vn : group.varNames) handledVars.insert(vn);
            }
        }

        // 2. Swap remaining independent variables
        for (const auto& varName : searchVars) {
            if (handledVars.count(varName)) continue;
            if (probDist(gen) < 0.5) {
                childVars[varName] = p2.vars.at(varName);
            }
        }
        return evaluateInd(childVars);
    };

    for (int g = 0; g < config.generations; ++g) {
        int64_t sumCost = 0;
        int validCount = 0;
        for (const auto& ind : population) {
            if (ind.valid) {
                sumCost += ind.cost;
                validCount++;
            }
        }
        double avgCost = validCount > 0 ? (double)sumCost / validCount : 0;

        if (csvFile && (g % 10 == 0 || g == config.generations - 1)) {
            *csvFile << g << "," << globalBest.cost << "," << avgCost << "\n";
            csvFile->flush();
        }

        if (g % 10 == 0 || g == config.generations - 1) {
            llvm::dbgs() << "  GA Generation " << g << "/" << config.generations 
                          << " (Best Cost: " << globalBest.cost 
                          << ", Avg Cost: " << avgCost 
                          << ", Valid: " << validCount << "/" << population.size() << ")\n";
        }

        std::vector<Individual> nextGen;
        // Elitism: carry over the best
        nextGen.push_back(globalBest);

        while (nextGen.size() < (size_t)config.popSize) {
            Individual child;
            if (probDist(gen) < config.crossoverRate) {
                Individual p1 = tournamentSelection(population, 3);
                Individual p2 = tournamentSelection(population, 3);
                child = crossover(p1, p2);
            } else {
                child = tournamentSelection(population, 3);
            }

            if (probDist(gen) < config.mutationRate) {
                std::unordered_map<std::string, int64_t> mutatedVars = generateNeighbor(model, child.vars, searchVars);
                child = evaluateInd(mutatedVars);
            }

            nextGen.push_back(child);
        }

        population = nextGen;
        Individual genBest = getBest(population);
        if (genBest.valid && (!globalBest.valid || genBest.cost < globalBest.cost)) {
            LLVM_DEBUG(
                llvm::dbgs() << "    GA: New Global Best Solution found! (Prev: " << globalBest.cost 
                             << ", New: " << genBest.cost << ") [";
                printKV(llvm::dbgs(), genBest.vars, "X_");
                llvm::dbgs() << "]\n";
            );
            globalBest = genBest;
        }
    }

    SolveResult result;
    result.success = globalBest.valid;
    result.cost = globalBest.cost;
    result.populate(globalBest.vars);
    return result;
}

SolveResult 
runRandomSearch(const PerformanceModel& model, const RandomConfig& config) {
    LLVM_DEBUG(llvm::dbgs() << "  Running Random Search (" << config.iterations << " iterations)...\n");
    
    SolveResult bestResult;
    bestResult.success = false;
    bestResult.latency = std::numeric_limits<int64_t>::max();

    for (int i = 0; i < config.iterations; ++i) {
        std::unordered_map<std::string, int64_t> current;
        // if (i == 0) {
        //     current = generateDefaultSolution(model);
        // } else {
        current = generateRandomSolution(model);
        // }
        
        int restartCount = 0;
        // Try up to 1000 times to get a valid random solution for this iteration
        // Skip random restarts if we already have a valid default solution
        while (!is_valid(model, current) && restartCount < 1000) {
            current = generateRandomSolution(model);
            restartCount++;
        }

        if (is_valid(model, current)) {
            int64_t currentCost = evaluate(model, current);
            if (!bestResult.success || currentCost < bestResult.cost) {
                bestResult.success = true;
                bestResult.cost = currentCost;
                bestResult.variables = current;
                if (current.count("totalDSPs")) {
                    bestResult.dsps = current.at("totalDSPs");
                }
            }
            // If iterations is 1, return the first valid one
            if (config.iterations == 1) break;
        }
    }

    if (bestResult.success) {
        LLVM_DEBUG(llvm::dbgs() << "  Random Search: Best cost found: " << bestResult.cost << "\n");
        // Random search already sets cost = bestCost (evaluate result)
        // populate will overwrite it with combined_cost if available
        bestResult.populate(bestResult.variables);
    } else {
        LLVM_DEBUG(llvm::dbgs() << "  Random Search: No valid solution found.\n");
    }

    return bestResult;
}

}
}


