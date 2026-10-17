# Developing ST-Flow

This guide covers building ST-Flow from source, running its test suite, and the details
behind its pinned toolchain. To use ST-Flow, `pip install st-flow` is enough; see the
[README](README.md#installation).

<!-- docs:build:start -->
## Building from source

There are three ways to install ST-Flow. Use the first one that works on your system;
all three end with the same installed `stflow` package.

| | Command | Needs | Time |
|---|---|---|---|
| **1. Prebuilt wheel** | `pip install st-flow` | Linux x86_64 with glibc ≥ 2.28 (Ubuntu 20.04+, Debian 10+, RHEL/Rocky 8+) | ~1 min |
| **2. From source, prebuilt LLVM** | `pip install .` | Linux x86_64 with GCC ≥ 12's C++ runtime (Ubuntu 22.04+, Debian 12+), plus a C/C++ toolchain | ~10–20 min |
| **3. From source, LLVM too** | `scripts/build_llvm.sh`, then `pip install .` | any Linux x86_64 with a C++17 compiler | 1–2 h |

All of them need **Python 3.12** (the pinned PyTorch wheels are `cp312`-only) built with
`bz2` support: `python -c "import bz2"` must work.

```bash
python3.12 -m venv ~/.venvs/stflow && source ~/.venvs/stflow/bin/activate
# or: pyenv virtualenv 3.12 stflow && pyenv activate stflow
```

### From source, with the prebuilt LLVM/MLIR

```bash
git clone https://github.com/UCLA-VAST/ST-Flow.git && cd ST-Flow
pip install .
pip install -r requirements-ml.txt
```

The build needs LLVM/MLIR 20 (Release + assertions, Python bindings), and looks for it in
this order:

1. `STFLOW_LLVM_DIR`, or `MLIR_DIR` / `LLVM_DIR`, if set;
2. `~/opt/llvm-stflow`;
3. a copy it downloaded before, in `~/.cache/stflow/llvm-stflow` (or under `$XDG_CACHE_HOME`);
4. otherwise it **downloads** the prebuilt release (1.3 GB) from the Zenodo record into
   that cache, checking its MD5 first. Set `STFLOW_FETCH_LLVM=OFF` to forbid this.

If the prebuilt LLVM cannot run on your system, the build stops at configure time with a
message saying so (on RHEL/Rocky 9, for example, whose C++ runtime is too old for it);
build LLVM from source instead, as below. The C++ build uses the clang shipped with that
LLVM if it can link on your system, and the default compiler otherwise. Set
`CMAKE_BUILD_PARALLEL_LEVEL` (for example `8`) to bound its parallelism on a
memory-limited machine.

While working on ST-Flow, `pip install -e .` gives an editable install; add
`-C cmake.build-type=Debug` for an assertions-enabled debug build of the C++ passes.

### Building LLVM/MLIR from source too

```bash
git clone https://github.com/UCLA-VAST/ST-Flow.git && cd ST-Flow
bash scripts/build_llvm.sh ~/opt/llvm-stflow     # fetches the pinned submodule; 1-2 h
pip install .                                    # finds ~/opt/llvm-stflow
pip install -r requirements-ml.txt
```

`scripts/build_llvm.sh` builds the `extern/llvm-project` submodule, pinned to the commit
the prebuilt release was made from, with the same configuration. It uses clang if found
and gcc otherwise; set `NUM_THREADS` to bound its parallelism. On a cluster, run it inside
an allocation, e.g. `srun -c32 --mem=64G bash scripts/build_llvm.sh`.

### Checking the installation

```bash
python -c "from stflow._mlir._mlir_libs import _streamhls; import stflow.main; print('ok')"
```

### Backend tools

```bash
source local_env.sh
```

This puts Vitis HLS, TAPA and AMPL on `PATH`. It finds each one itself and warns, rather
than failing, about anything missing, so a build-only setup can skip it. Override any of
them with `STFLOW_VITIS_HLS_DIR`, `STFLOW_AMPL_DIR`, or by putting the tools on `PATH`
yourself. Vitis HLS (validated with 2024.2) is needed for C-simulation, which is on by
default; AMPL with a MINLP solver for MINLP-based design-space exploration; XRT only to
run a generated bitstream on an FPGA board.

### Pinned ML toolchain

ST-Flow's front end requires `torch==2.8.0.dev20250325+cpu`, the matching torchvision, and
`torch_mlir==20251004.590`. That torch is a **nightly**, and `download.pytorch.org` keeps
only about two months of nightlies, so it can no longer be fetched from any package index;
the copies in the Zenodo record are the only surviving ones. Torch-MLIR pins that exact
torch build, so the three move as a unit. Do not upgrade one of them; moving off the
nightly means rebuilding Torch-MLIR and revalidating the front end.

`requirements-ml.txt` installs them straight from the Zenodo record. Without network
access, download the three `.whl` files from the record into `wheels/` and run:

```bash
bash scripts/install_ml_wheels.sh wheels/     # defaults to ./wheels if omitted
```

That script checks the interpreter is `cp312`, refuses to substitute versions, and
verifies the three imports afterwards.
<!-- docs:build:end -->

<!-- docs:dev:start -->
## Testing

```bash
pip install -e ".[test]"
source env.sh && source local_env.sh

pytest test/                                  # full suite, 66 tests
pytest test/03_polybench                      # one category
pytest test/03_polybench/test_03_gemm.py      # one file
pytest test/03_polybench -o addopts="-v -n11" # override the worker count
```

`pytest.ini` sets `addopts = -v -n10`, so every run uses `pytest-xdist`. Measured on a
20-core x86_64 workstation with 23 GB RAM, with C-simulation enabled:

| Directory | Tests | Wall clock (`-n4`) |
|---|---|---|
| `test/03_polybench` | 21 | ~1.5 min |
| `test/01_dnn` + `test/02_synthetic` | 42 | ~3.5 min |
| `test/00_transformer` | 3 | ~3 min (`-n2`) |

The full suite takes about 4 min at `-n11` and 7 min at `-n4` on that machine. On a host
with fewer cores, lower `-n` to roughly the core count: parallel C-simulation slows down
and becomes flaky when oversubscribed. Each test writes a `<...>.prj/` directory next to
its test file with the MLIR of every stage, DSE reports and logs; `test/*/clean*.sh`
removes them.

New tests go in the matching category folder under `test/`, named `test_*.py`. Each
category's `test_config.yaml` sets its configuration.

## Repository layout

| Path | Contents |
|---|---|
| `stflow/` | Python package: the compiler driver, DSE configuration and the Vitis/TAPA builders. |
| `mlir/` | C++ MLIR dialects, passes and the HLS C++ emitters. |
| `test/` | The test suite, by domain (transformers, DNNs, synthetic graphs, PolyBench). |
| `scripts/` | Build, plotting and analysis scripts. |
| `docs/` | Sources of the documentation website, plus the PACT '26 artifact appendix and checklist (`docs/ae/`). |
| `extern/` | Third-party code, including the pinned `llvm-project` submodule. |

ST-Flow is driven from Python; the compiler itself is C++ on LLVM/MLIR 20 with
pybind11 bindings, built with CMake and Ninja.

## Conventions

- **Environment.** `env.sh` finds LLVM/MLIR and detects Slurm; `local_env.sh` adds the
  backend tools (Vitis HLS, TAPA, AMPL) needed to *run* designs. Both are safe to source
  more than once.
- **Build.** `pyproject.toml` builds `mlir/` with scikit-build-core into
  `build/<wheel-tag>/` and installs the compiled bindings into the package as
  `stflow/_mlir`.
- **`.gitignore` is aggressive.** It ignores `*.txt`, `*.csv`, `run*`, `models*`,
  `experiments`, `paper` and more, so new files with those names are silently left
  untracked. Run `git check-ignore -v <path>` before assuming a file was committed.

## Troubleshooting

**`static assertion failed: def_property family does not currently support keep_alive`**
while compiling `IRCore.cpp`. You have pybind11 3.x. MLIR 20's Python bindings only build
against pybind11 2.x; `pyproject.toml` pins `pybind11>=2.10,<3`. After downgrading,
`rm -rf build/`, since the CMake cache holds the old pybind11 paths.

**`unrecognized arguments: -n10`** from pytest. `pytest-xdist` is missing; install the test
extras with `pip install -e ".[test]"`.

**Every test fails with `RuntimeError: Both FX export and TorchScript compilation failed`**
and the log shows `FX error: No module named '_bz2'`. Your Python 3.12 was built without
bzip2 support (common with `pyenv` when `libbz2-dev` was missing at build time), and
torch's FX export needs it. Install `libbz2-dev` (plus `liblzma-dev`), rebuild the
interpreter, and recreate the virtualenv; `python -c "import bz2"` must succeed.

**`This design was built without XRT, so it can only run in C-simulation`** when running
a bitstream. No XRT install was found when the design was built (ST-Flow looks at
`XILINX_XRT`, then `/opt/xilinx/xrt`). Install XRT, or point `XILINX_XRT` at it, and
rebuild the design.

**`ModuleNotFoundError: No module named 'torch'`** when importing `stflow`. The ML
toolchain is not installed; run `pip install -r requirements-ml.txt`.

**A C-simulation test fails, then passes when re-run alone.** Parallel C-simulation under
xdist is sensitive to load when there are more workers than cores. Re-run a failure on
its own before treating it as a real defect.

**MINLP DSE fails or hangs.** `ampl` must be an executable on `PATH`; the solver is run as
`timeout <N>s ampl <file>` from `mlir/lib/Dialect/Dataflow/Support/PerformanceModel.cpp`.
Either source `local_env.sh` or set `dse_options.permutation_dse` / `unrolling_dse` to a
heuristic (`sa`, `ga`, `random`).

**The build picks the wrong LLVM.** `env.sh` prefers `STFLOW_LLVM_DIR`, then
`~/opt/llvm-stflow`, then any system `llvm-config-19/18/17` that ships MLIR. A system LLVM
17/18 will *not* build this code. Check the line `env.sh` prints, and set
`STFLOW_LLVM_DIR` explicitly if it chose wrong.
<!-- docs:dev:end -->
