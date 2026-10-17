# ST-Flow: A Hardware Compiler for Automating Spatial-Temporal Dataflow Acceleration

[![Documentation](https://readthedocs.org/projects/st-flow/badge/?version=latest)](https://st-flow.readthedocs.io)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21881993.svg)](https://doi.org/10.5281/zenodo.21881993)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Documentation:** <https://st-flow.readthedocs.io> ·
**Paper artifact (PACT '26):** <https://doi.org/10.5281/zenodo.21881993>

<!-- docs:overview:start -->
## Project Overview

ST-Flow is an end-to-end hardware compiler designed to translate deep learning models and affine applications directly into highly optimized, FPGA-based streaming dataflow accelerators. It is built directly on top of the Multi-Level Intermediate Representation (MLIR) compiler infrastructure and targets high-performance execution of deep neural networks (DNNs). 

Instead of relying on standard spatial replication models (which map each kernel to a separate dedicated hardware unit and rapidly deplete on-chip resources), ST-Flow incorporates a reuse-aware compilation methodology. It automatically identifies structurally and temporally compatible kernels to safely time-multiplex shared compute engines. This process reclaims DSP slices and block memories while maintaining deadlock-free streaming concurrency under structural back-pressure.

---

## ST-Flow Core Features
ST-Flow outperforms prior automation approaches by breaking the limitations of purely spatial hardware layouts and strict divisibility bounds:
* **Spatial-Temporal Unified Mapping**: Generalizes the search space to support pure spatial mappings, pure temporal mappings, and hybrid spatial-temporal architectures.
* **Hybrid Mapping Heuristic**: Uses a dedicated clustering algorithm to group compatible compute nodes into shared hardware modules. Computes must share a structural signature (loop depth, operation sequence, tensor ranks) and exhibit matching arithmetic intensities to prevent latency padding or under-utilization.
* **Non-Divisor Tiling Strategies**: Decouples hardware unrolling from strict divisibility constraints. This supports:
  * `Div-Min`: Divisor-only tiling bounded by the smallest shared loop size.
  * `Non-Div-Min`: Flexible non-divisor tiling bounded by the smallest shared loop size.
  * `Non-Div-Max`: Flexible non-divisor tiling bounded by the maximum loop size to maximize computing parallelization at the cost of transient padding.
* **Multi-Engine Heuristic Search**: Beyond exact MINLP formulations, ST-Flow implements Simulated Annealing (SA) and Genetic Algorithms (GA). This enables fast DSE scaling to deeply nested graphs (such as GPT-2) where mathematical solvers time out.
* **Postprocessing Optimization**: Introduces layout converters for mismatched stream patterns, flattens multi-dimensional stream arrays into single handshaked vector paths, and optimizes Direct Memory Access (DMA) memory layouts to secure high-burst external memory bandwidth.

---

### Main Technologies
- **Python**: Main interface (`stflow/`); packaging in `pyproject.toml` (scikit-build-core).
- **C++**: MLIR dialects and passes implementation (`mlir/`).
- **LLVM / MLIR (v20)**: Core compiler infrastructure. The build is pinned to LLVM 20 —
  see [Building and Running](#building-and-running).
- **Pybind11**: For generating Python bindings for the C++ code.
- **CMake & Ninja**: Build system for the C++ MLIR components.
- **Pytest**: For unit and integration testing.

## Directory Structure

- `stflow/`: Python package containing the accelerator interface and builders.
- `mlir/`: C++ source code for MLIR dialects, passes, and tools.
- `test/`: Pytest suite containing tests across different domains (e.g., DNNs, polybench, synthetic).
- `extern/`: External dependencies, including `llvm-project`.
- `scripts/`: Utility scripts for building, plotting, and analysis.
- `docs/`: Sources of the documentation website, plus the PACT '26 artifact appendix and
  reproducibility checklist (`docs/ae/`).
<!-- docs:overview:end -->

## Artifact Evaluation

Reviewing the PACT '26 artifact? Start at **[ARTIFACT_EVALUATION.md](ARTIFACT_EVALUATION.md)**,
not here. It states which paper claims the artifact does and does not support, and gives a
tiered evaluation path — the first tier regenerates every figure and table in the paper
from the archived experiments in about 11 seconds, with no FPGA, HLS licence or solver.

The archived experiments (`PACT26_Artifacts/`, 5.6 GB) are too large for git and live in the
[Zenodo record](https://doi.org/10.5281/zenodo.21881993). From a clone of this repository:

```bash
curl -LO https://zenodo.org/records/21881993/files/st-flow-pact26.tar.gz
tar xzf st-flow-pact26.tar.gz --strip-components=1 st-flow-pact26/PACT26_Artifacts
pip install -r requirements.txt && source env.sh
bash scripts/ae_verify.sh          # expect: RESULT: PASS
```

The artifact appendix and reproducibility checklist are under [`docs/ae/`](docs/ae/).

## Building and Running
<!-- docs:install:start -->

ST-Flow installs in one of three ways. Start with Tier 1 and move down only if it does
not work on your system; every tier ends with the same installed `stflow` package.

| Tier | Command | Needs | Time |
|---|---|---|---|
| **1. Prebuilt wheel** | `pip install st-flow` | Linux x86_64 with glibc ≥ 2.28 (Ubuntu 20.04+, Debian 10+, RHEL/Rocky 8+) | ~1 min |
| **2. From source, prebuilt LLVM** | `pip install .` | Linux x86_64 with GCC ≥ 12's C++ runtime (Ubuntu 22.04+, Debian 12+), plus a C/C++ toolchain | ~10–20 min |
| **3. From source, LLVM too** | `scripts/build_llvm.sh`, then `pip install .` | any Linux x86_64 with a C++17 compiler | 1–2 h |

All tiers need **Python 3.12** (the pinned PyTorch wheels are `cp312`-only; not 3.11 or
3.13) built with `bz2` support (`python -c "import bz2"` must work). Running designs also
needs the backend tools: **Vitis HLS** (validated with 2024.2) for C-simulation, which
is on by default, and **AMPL** with a MINLP-capable solver for MINLP-based DSE.
**XRT** is only needed to run a generated bitstream on an FPGA board; without it, designs
are built for C-simulation only.

### Python environment

```bash
python3.12 -m venv ~/.venvs/stflow && source ~/.venvs/stflow/bin/activate
# or: pyenv virtualenv 3.12 stflow && pyenv activate stflow
```

### Tier 1 — prebuilt wheel

```bash
pip install st-flow
pip install -r https://raw.githubusercontent.com/UCLA-VAST/ST-Flow/main/requirements-ml.txt
```

The second line installs the pinned PyTorch front end (`torch`, `torchvision`,
`torch_mlir`) from the [Zenodo record](https://doi.org/10.5281/zenodo.21881993); see
[Pinned ML toolchain](#pinned-ml-toolchain) for why it cannot come from PyPI. If pip
reports that no wheel matches your platform, it falls back to building from source,
which is Tier 2.

The PyPI distribution is named `st-flow`; the Python package and the command are
`stflow`, as in every other tier.

### Tier 2 — from source, with the prebuilt LLVM/MLIR

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
continue with Tier 3. The C++ build uses the clang shipped with that LLVM if it can link
on your system, and the default compiler otherwise. To bound its parallelism on a memory-limited machine, set
`CMAKE_BUILD_PARALLEL_LEVEL` (for example `8`). For development, `pip install -e .`
gives an editable install; add `-C cmake.build-type=Debug` for an assertions-enabled
debug build of the C++ passes.

### Tier 3 — build LLVM/MLIR from source too

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

### Verify the installation

```bash
python -c "from stflow._mlir._mlir_libs import _streamhls; import stflow.main; print('ok')"
```

### Backend tools

```bash
source local_env.sh
```

This puts Vitis HLS, TAPA and AMPL on `PATH`. It auto-discovers each and warns (rather
than failing) about anything missing, so a build-only setup can skip it. Override any of
them with `STFLOW_VITIS_HLS_DIR`, `STFLOW_AMPL_DIR`, or by putting the tools on `PATH`
yourself.

### Testing

`pytest.ini` sets `addopts = -v -n10`, so every invocation runs under xdist. `pytest-xdist`
is in `requirements.txt` — without it pytest aborts on an unrecognized argument.

```bash
source env.sh && source local_env.sh

pytest test/                                  # full suite, 66 tests
pytest test/03_polybench                      # one category
pytest test/03_polybench/test_03_gemm.py      # one file
pytest test/03_polybench -o addopts="-v -n11" # override the worker count
```

Measured on the validation run (20-core x86_64 workstation with 23 GB RAM, `-n4`, csim enabled):

| Directory | Tests | Wall clock |
|---|---|---|
| `test/03_polybench` | 21 | ~1.5 min |
| `test/01_dnn` + `test/02_synthetic` | 42 | ~3.5 min |
| `test/00_transformer` | 3 | ~3 min (`-n2`) |

The full suite takes 3m42s at `-n11` and 6m52s at `-n4` on that machine. On a host with
fewer cores than workers, lower `-n` to roughly the core count: parallel csim is
load-sensitive and oversubscribing makes it slower and flakier — see Troubleshooting. Each test writes a `<...>.prj/` directory next to its test
file holding per-stage MLIR, DSE reports and logs; `test/*/clean*.sh` removes them.

### Pinned ML toolchain

ST-Flow's front-end requires `torch==2.8.0.dev20250325+cpu`, the matching torchvision, and
`torch_mlir==20251004.590`. That torch is a **nightly**, and `download.pytorch.org` retains
only about two months of nightlies, so it is no longer fetchable from any index — the
copies distributed with this artifact are the only surviving ones. Torch-MLIR hard-pins
that exact torch build, so the three move as a unit. Do not upgrade one of them; migrating
off the nightly means rebuilding Torch-MLIR and revalidating the front-end.

`requirements-ml.txt` installs them straight from the Zenodo record. Without network
access, download the three `.whl` files from the record into `wheels/` and run:

```bash
bash scripts/install_ml_wheels.sh wheels/     # defaults to ./wheels if omitted
```

That script checks the interpreter is `cp312`, refuses to substitute versions, and
verifies the three imports afterwards.

### Troubleshooting

**`static assertion failed: def_property family does not currently support keep_alive`**
while compiling `IRCore.cpp`. You have pybind11 3.x. MLIR 20's Python bindings only build
against pybind11 2.x; `requirements.txt` pins `pybind11>=2.10,<3`. After downgrading you
must `rm -rf build/` — the CMake cache holds the old pybind11 paths.

**`unrecognized arguments: -n10`** from pytest. `pytest-xdist` is missing; reinstall from
`requirements.txt`.

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
toolchain was not installed; run `bash scripts/install_ml_wheels.sh wheels/`.

**A csim test fails, then passes when re-run alone.** Parallel csim under xdist is
load-sensitive when the worker count exceeds the available cores.
Re-run a failure serially before treating it as a real defect — a failure that reproduces
serially is a genuine one.

**MINLP DSE fails or hangs.** `ampl` must be a bare executable on `PATH`; the solver is
invoked as `timeout <N>s ampl <file>` from
`mlir/lib/Dialect/Dataflow/Support/PerformanceModel.cpp`. Either source `local_env.sh` or
set `dse_options.permutation_dse` / `unrolling_dse` to a heuristic (`sa`, `ga`, `random`).

**The build picks the wrong LLVM.** `env.sh` prefers `STFLOW_LLVM_DIR`, then
`~/opt/llvm-stflow`, then any system `llvm-config-19/18/17` that ships MLIR. A system LLVM
17/18 will *not* build this code. Check the line `env.sh` prints, and set
`STFLOW_LLVM_DIR` explicitly if it chose wrong.

<!-- docs:install:end -->

<!-- docs:dev:start -->
## Development Conventions

- **Environment**: `env.sh` handles LLVM/MLIR discovery and detects SLURM automatically;
  `local_env.sh` layers on the backend tools (Vitis HLS, TAPA, AMPL) needed to *run*
  designs. Both are safe to source repeatedly.
- **Build Integration**: `pyproject.toml` builds `mlir/` with scikit-build-core into
  `build/<wheel-tag>/` and installs the compiled bindings into the package as
  `stflow/_mlir`. Use `pip install -e .` while developing; set
  `CMAKE_BUILD_PARALLEL_LEVEL` to bound the parallelism.
- **Testing**: Add new test cases to the appropriate category folder within the `test/`
  directory. Tests should be named with the `test_` prefix. Per-category policy lives in
  that category's `test_config.yaml`.
- **`.gitignore` is aggressive.** It ignores `*.txt`, `*.csv`, `run*`, `models*`,
  `experiments`, `paper` and more. New files with those names are silently untracked —
  `git check-ignore -v <path>` before assuming a file was committed.
<!-- docs:dev:end -->

## Citation

If you use ST-Flow in your research, please cite our PACT '26 paper (and, for the
evaluated artifact, its Zenodo record):

```bibtex
@inproceedings{basalama2026stflow,
  author    = {Basalama, Suhail and Pouget, St{\'e}phane and Cong, Jason},
  title     = {{ST-Flow}: A Hardware Compiler for Automating Spatial-Temporal Dataflow Acceleration},
  booktitle = {Proceedings of the International Conference on Parallel Architectures and Compilation Techniques},
  series    = {PACT '26},
  year      = {2026},
  month     = oct,
  address   = {Chicago, IL, USA},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3838684.3846865},
  isbn      = {979-8-4007-2915-7}
}

@misc{basalama2026stflow_artifact,
  author    = {Basalama, Suhail and Pouget, St{\'e}phane and Cong, Jason},
  title     = {{ST-Flow}: A Hardware Compiler for Spatial-Temporal Dataflow Acceleration ({PACT} '26 Artifact)},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21881993}
}
```


## Acknowledgments
<!-- docs:ack:start -->

ST-Flow builds on a line of open-source MLIR/HLS work. Its HLS/Dataflow dialect and
parts of its analysis and lowering are adopted from
[ScaleHLS](https://github.com/UIUC-ChenLab/scalehls); its HLS C++ emitter derives from
[HeteroCL](https://github.com/cornell-zhang/heterocl)'s; its Python bindings follow
[Allo](https://github.com/cornell-zhang/allo); and it generalizes
[Stream-HLS](https://github.com/UCLA-VAST/Stream-HLS) (FPGA '25), whose system modeling
and edge conversion passes it extends. We also thank the authors of
[HIDA](https://github.com/UIUC-ChenLab/ScaleHLS-HIDA) and [TAPA](https://github.com/rapidstream-org/rapidstream-tapa),
whose work on hierarchical dataflow and task-parallel HLS informed this one, and of
[Torch-MLIR](https://github.com/llvm/torch-mlir) and [MLIR/LLVM](https://mlir.llvm.org),
on which ST-Flow is built. The files derived from each project and their licenses are
listed in [NOTICE](NOTICE).

This work was partially supported by the National Science Foundation (NSF) under Award
No. 2533041 for NQVL, the [CDSC industrial partners](https://cdsc.ucla.edu/partners/),
the AMD HACC Programs, and the PRISM Center under the JUMP 2.0 Program.
<!-- docs:ack:end -->

## License

ST-Flow is released under the [Apache License 2.0](LICENSE). Third-party components and
their notices are listed in [NOTICE](NOTICE), with full license texts in
[`licenses/`](licenses/). The PACT '26 paper is published under CC BY 4.0.
