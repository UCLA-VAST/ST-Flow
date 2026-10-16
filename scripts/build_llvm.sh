#!/usr/bin/env bash

# Exit on error
set -e

PRJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_DIR="${1:-$HOME/opt/llvm-stflow}"
BUILD_DIR="$PRJ_ROOT/extern/llvm-project/build"

echo "Building LLVM/MLIR..."
echo "Install Directory: $INSTALL_DIR"
echo "Build Directory: $BUILD_DIR"

mkdir -p "$INSTALL_DIR"
mkdir -p "$BUILD_DIR"

cd "$BUILD_DIR"

# Check if SLURM is available for the configuration step
CONf_CMD=""
if command -v srun >/dev/null 2>&1; then
    CONF_CMD="srun --export=ALL"
fi

$CONF_CMD cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="clang;mlir" \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_INSTALL_UTILS=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_ENABLE_LLD=ON \
    -DLLVM_ENABLE_BINDINGS=OFF \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=$(which python) \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"

echo "Configuration complete. Starting build..."

# Run building with srun if available
if command -v srun >/dev/null 2>&1; then
    srun --export=ALL -c32 --mem=64GB ninja install
else
    # Fallback to local build if no Slurm
    num_cores=$(nproc)
    ninja -j$num_cores install
fi

echo "LLVM/MLIR installed to $INSTALL_DIR"
echo "You can now use it by setting: export STFLOW_LLVM_DIR=$INSTALL_DIR"
