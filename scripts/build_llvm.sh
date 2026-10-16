#!/usr/bin/env bash
#
# Build LLVM/MLIR 20 from the pinned extern/llvm-project submodule (installation
# Tier 3). Use this only when the prebuilt release does not run on your system (for
# example glibc < 2.34). Expect 30-90 minutes on 16+ cores and ~30 GB of disk.
#
#   bash scripts/build_llvm.sh [install-dir]        # default: ~/opt/llvm-stflow
#
# then build ST-Flow against it:
#
#   STFLOW_LLVM_DIR=~/opt/llvm-stflow pip install .
#
# Environment:
#   NUM_THREADS   parallel jobs (default: half the cores; linking is memory-bound)
#   CC / CXX      compilers (default: clang/clang++ if found, else gcc/g++)
#   LLVM_PROJECTS LLVM projects to build (default: mlir; ST-Flow does not need clang)
#   LLVM_CMAKE_BUILD_DIR  build directory (default: extern/llvm-project/build; ~20 GB)
#
# Run it inside a Slurm allocation yourself if your cluster needs one, e.g.
#   srun -c32 --mem=64G bash scripts/build_llvm.sh

set -euo pipefail

PRJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_DIR="${1:-$HOME/opt/llvm-stflow}"
SRC_DIR="$PRJ_ROOT/extern/llvm-project"
BUILD_DIR="${LLVM_CMAKE_BUILD_DIR:-$SRC_DIR/build}"
NUM_THREADS="${NUM_THREADS:-$(( $(nproc) / 2 > 1 ? $(nproc) / 2 : 1 ))}"

PY="$(command -v python3 || command -v python)"
if ! "$PY" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))'; then
    echo "Error: activate the Python 3.12 environment you will install ST-Flow into" >&2
    echo "       (the MLIR Python bindings are built for that interpreter)." >&2
    exit 1
fi
# MLIR's Python bindings need these at configure time (pybind11 3.x cannot build them).
# CMake and Ninja come from pip too, so the build does not depend on whatever (possibly
# shimmed or outdated) versions the system PATH provides.
"$PY" -m pip install -q "pybind11>=2.10,<3" numpy PyYAML "cmake>=3.20" ninja
export PATH="$(dirname "$PY"):$PATH"

if [ ! -f "$SRC_DIR/llvm/CMakeLists.txt" ]; then
    echo "Fetching the pinned LLVM submodule (shallow)..."
    git -C "$PRJ_ROOT" submodule update --init --depth 1 extern/llvm-project
fi

: "${CC:=$(command -v clang || command -v gcc)}"
: "${CXX:=$(command -v clang++ || command -v g++)}"
LLD_FLAG=()
if command -v ld.lld >/dev/null 2>&1; then
    LLD_FLAG=(-DLLVM_ENABLE_LLD=ON)
fi

echo "LLVM source : $SRC_DIR"
echo "Install dir : $INSTALL_DIR"
echo "Compilers   : $CC / $CXX ${LLD_FLAG:+(lld)}"
echo "Parallelism : $NUM_THREADS"

cmake -G Ninja -S "$SRC_DIR/llvm" -B "$BUILD_DIR" \
    -DLLVM_ENABLE_PROJECTS="${LLVM_PROJECTS:-mlir}" \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_INSTALL_UTILS=ON \
    -DCMAKE_C_COMPILER="$CC" \
    -DCMAKE_CXX_COMPILER="$CXX" \
    "${LLD_FLAG[@]}" \
    -DLLVM_ENABLE_BINDINGS=OFF \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE="$PY" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"

cmake --build "$BUILD_DIR" --target install -j "$NUM_THREADS"

echo
echo "LLVM/MLIR installed to $INSTALL_DIR"
echo "Build ST-Flow against it with:  STFLOW_LLVM_DIR=$INSTALL_DIR pip install ."
