# Use current directory for the project root if not already set
if [ -z "$PRJ_ROOT" ]; then
    export PRJ_ROOT=$(pwd)
fi

# --- LLVM/MLIR Discovery ---
# 1. Use STFLOW_LLVM_DIR if set
# 2. Check for LLVM in ~/opt/llvm-stflow (default install location)
# 3. Check for LLVM 18 or 17 in standard locations
# 4. Fallback to local extern/llvm-project

if [ -z "$STFLOW_LLVM_DIR" ]; then
    # Check default install location
    if [ -d "$HOME/opt/llvm-stflow" ]; then
        STFLOW_LLVM_DIR="$HOME/opt/llvm-stflow"
        echo "Using default pre-built LLVM at $STFLOW_LLVM_DIR"
    else
        # Try to find a recent llvm-config that also has MLIR
        for ver in 20 19 18 17; do
            if command -v llvm-config-$ver >/dev/null 2>&1; then
                CANDIDATE_DIR=$(llvm-config-$ver --prefix)
                # Check if MLIR exists in this candidate
                if find "$CANDIDATE_DIR" -name "MLIRConfig.cmake" -print -quit | grep -q .; then
                    STFLOW_LLVM_DIR="$CANDIDATE_DIR"
                    echo "Found LLVM $ver with MLIR at $STFLOW_LLVM_DIR"
                    if [ "$ver" -lt 20 ]; then
                        echo "WARNING: ST-Flow is pinned to LLVM 20. LLVM $ver will not compile" >&2
                        echo "         mlir/. Install the prebuilt LLVM (see README.md, Step 1) or" >&2
                        echo "         set STFLOW_LLVM_DIR explicitly." >&2
                    fi
                    break
                fi
            fi
        done
    fi
fi

if [ -n "$STFLOW_LLVM_DIR" ]; then
    export LLVM_BUILD_DIR="$STFLOW_LLVM_DIR"
    # Check multiple possible locations for CMake files
    CMAKE_CANDIDATES=(
        "lib/cmake"
        "lib/llvm-17/lib/cmake"
        "lib/llvm-18/lib/cmake"
        "share/mlir/cmake"
        "build/lib/cmake"
    )
    
    for rel in "${CMAKE_CANDIDATES[@]}"; do
        if [ -d "$LLVM_BUILD_DIR/$rel/mlir" ]; then
            export MLIR_DIR="$LLVM_BUILD_DIR/$rel/mlir"
            export LLVM_DIR="$LLVM_BUILD_DIR/$rel/llvm"
            # If we found it in a 'build' dir, update LLVM_BUILD_DIR
            if [[ "$rel" == build* ]]; then
                export LLVM_BUILD_DIR="$LLVM_BUILD_DIR/build"
            fi
            break
        fi
    done
else
    # Fallback to local submodule
    export LLVM_PROJECT_PATH="$PRJ_ROOT/extern/llvm-project"
    export LLVM_BUILD_DIR="$LLVM_PROJECT_PATH/build"
    export MLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir"
    export LLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm"
fi

# Add MLIR/LLVM tools to path
if [ -d "$STFLOW_LLVM_DIR/bin" ]; then
    export PATH="$STFLOW_LLVM_DIR/bin:$PATH"
fi
export PATH="$PRJ_ROOT/mlir/build/bin:$PATH"
# The evaluation drivers (evrun, evplot, evtable, evcodo, ...) live in scripts/ and are
# invoked by name from PACT26_Artifacts/experiments.sh.
export PATH="$PRJ_ROOT/scripts:$PATH"
export pybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())" 2>/dev/null || echo "")

# Check if SLURM is available
if command -v srun >/dev/null 2>&1 && command -v sbatch >/dev/null 2>&1; then
    # Only load if modules are available
    if command -v ml >/dev/null 2>&1; then
        ml load ninja llvm cmake 2>/dev/null || true
        ml unload python py-pip 2>/dev/null || true
    fi
    echo "SLURM detected - IS_SLURM environment variable set"
else
    echo "SLURM not detected - running in local mode"
fi
