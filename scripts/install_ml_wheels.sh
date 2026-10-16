#!/usr/bin/env bash
#
# Install ST-Flow's pinned ML front-end from a local directory of wheels.
#
#   bash scripts/install_ml_wheels.sh <dir>
#
# Use this when the wheels came from the Zenodo artifact record rather than from the
# GitHub release that requirements-ml.txt points at. <dir> must contain the three
# cp312 linux-x86_64 wheels shipped with the artifact:
#
#   torch-2.8.0.dev20250325+cpu-...whl
#   torchvision-0.22.0.dev20250325+cpu-...whl
#   torch_mlir-20251004.590-...whl
#
# These three move as a unit: torch_mlir hard-pins that exact torch nightly, and the
# nightly is no longer served by download.pytorch.org. Do not substitute versions.

set -euo pipefail

DIR="${1:-}"
if [ -z "$DIR" ]; then
    # Default to wheels/ beside the repository root, which is where the artifact's
    # instructions tell reviewers to unpack the wheels from the archive.
    DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)/wheels"
fi
if [ ! -d "$DIR" ]; then
    echo "Error: $DIR is not a directory." >&2
    echo "Usage: bash scripts/install_ml_wheels.sh [directory-containing-the-wheels]" >&2
    echo "       Defaults to ./wheels at the repository root." >&2
    exit 1
fi

py_tag=$(python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
if [ "$py_tag" != "cp312" ]; then
    echo "Error: these wheels are cp312-only; the active interpreter is $py_tag." >&2
    echo "       Activate a Python 3.12 environment first (see README.md)." >&2
    exit 1
fi

missing=0
wheels=()
for pkg in torch-2.8.0.dev20250325 torchvision-0.22.0.dev20250325 torch_mlir-20251004.590; do
    # shellcheck disable=SC2206
    found=($DIR/${pkg}*.whl)
    if [ ! -e "${found[0]}" ]; then
        echo "Missing: ${pkg}*.whl in $DIR" >&2
        missing=$((missing + 1))
    else
        wheels+=("${found[0]}")
    fi
done
[ "$missing" -eq 0 ] || exit 1

printf 'Installing:\n'; printf '  %s\n' "${wheels[@]}"
python -m pip install "${wheels[@]}"

python - <<'PY'
import torch, torchvision, torch_mlir
print(f"torch        {torch.__version__}")
print(f"torchvision  {torchvision.__version__}")
print("torch_mlir   ok")
assert torch.__version__.startswith("2.8.0.dev20250325"), "unexpected torch version"
PY
echo "ML front-end installed."
