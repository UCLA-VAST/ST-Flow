import re
from typing import List, Optional

VECTOR_CLASS = """
template<typename T, size_t N>
struct Vector {
  T data[N];
  Vector() {
    #pragma HLS inline
    #pragma HLS array_partition variable=data complete dim=1
  }

  T& operator[](int i) {
    #pragma HLS inline
    return data[i];
  }

  const T& operator[](int i) const {
    #pragma HLS inline
    return data[i];
  }

  // Support for assignment between Vectors
  Vector& operator=(const Vector& other) {
    #pragma HLS inline
    for (size_t i = 0; i < N; ++i) {
      #pragma HLS unroll
      data[i] = other.data[i];
    }
    return *this;
  }
};
"""

ERF_APPROX = """
#include <cmath>
// passed 10e-3
inline float erf_approx(float x){
  #pragma HLS inline
  const float a = 0.147f;
  const float s = (x < 0.0f) ? -1.0f : 1.0f;
  x = fabsf(x);
  float x2 = x * x;
  float num = (4.0f / 3.14159265358979323846f) + a * x2;
  float den = 1.0f + a * x2;
  float t = -x2 * (num / den);
  float y = 1.0f - expf(t);
  return s * sqrtf(y);
}
"""

def sanitize_ident(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    if re.match(r"^[0-9]", s):
        s = "_" + s
    return s

def extract_function_prototype(cpp_text: str, func_name: str) -> Optional[str]:
    lines = cpp_text.splitlines()
    # Match start: optional extern "C", void, followed by func_name
    start_re = re.compile(rf"^\s*(?:extern\s+\"C\"\s+)?void\s+{re.escape(func_name)}\s*\(")
    
    found_start = -1
    for i, line in enumerate(lines):
        if start_re.match(line):
            found_start = i
            break
            
    if found_start == -1:
        return None
        
    proto_lines = []
    paren_count = 0
    found_end = False
    
    for i in range(found_start, len(lines)):
        line = lines[i].rstrip()
        proto_lines.append(line)
        paren_count += line.count("(") - line.count(")")
        
        if paren_count == 0 and ")" in line:
            # We found the end of the argument list. 
            # Now we just need to end at the semicolon or open brace.
            last_line = proto_lines[-1]
            if "{" in last_line:
                proto_lines[-1] = last_line.split("{")[0].rstrip() + ";"
            elif not last_line.endswith(";"):
                proto_lines[-1] = last_line + ";"
            found_end = True
            break
            
    if not found_end:
        return None
        
    return "\n".join(proto_lines)

def extract_prototypes_by_prefix(cpp_text: str, prefixes: List[str]) -> List[str]:
    # Find all function names starting with any prefix
    names = set()
    for line in cpp_text.splitlines():
        # Look for "void <name>("
        m = re.search(r"void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", line)
        if m:
            nm = m.group(1)
            if any(nm.startswith(p) for p in prefixes):
                names.add(nm)
                
    protos = []
    for nm in sorted(names):
        p = extract_function_prototype(cpp_text, nm)
        if p:
            protos.append(p)
    return protos
