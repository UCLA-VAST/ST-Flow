# ST-Flow: A Hardware Compiler for Automating Spatial-Temporal Dataflow Acceleration

[![PyPI](https://img.shields.io/pypi/v/st-flow.svg)](https://pypi.org/project/st-flow/)
[![Documentation](https://readthedocs.org/projects/st-flow/badge/?version=latest)](https://st-flow.readthedocs.io)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21881993.svg)](https://doi.org/10.5281/zenodo.21881993)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Documentation:** <https://st-flow.readthedocs.io> ·
**Paper artifact (PACT '26):** <https://doi.org/10.5281/zenodo.21881993>

<!-- docs:overview:start -->
ST-Flow compiles PyTorch models and affine programs, through
[MLIR](https://mlir.llvm.org), into streaming dataflow accelerators for FPGAs. Instead of
giving every kernel its own hardware, it finds kernels that are structurally and
temporally compatible and time-multiplexes them onto shared compute engines, reclaiming
DSPs and on-chip memory while keeping the design streaming and deadlock-free.

- **Spatial, temporal or hybrid mapping** of kernels to hardware, chosen per design.
- **Non-divisor tiling**, so unroll factors are not limited to divisors of loop bounds.
- **Design-space exploration** with an exact MINLP model or with simulated annealing and
  genetic algorithms, which scale to large graphs such as GPT-2.
- **HLS C++ output** for Vitis HLS (and TAPA), checked against PyTorch by C-simulation.
<!-- docs:overview:end -->

## Installation
<!-- docs:install:start -->

ST-Flow needs Linux x86_64 with glibc ≥ 2.28 (Ubuntu 20.04+, Debian 10+, RHEL/Rocky 8+)
and **Python 3.12**.

```bash
python3.12 -m venv ~/.venvs/stflow && source ~/.venvs/stflow/bin/activate
pip install st-flow
pip install -r https://raw.githubusercontent.com/UCLA-VAST/ST-Flow/main/requirements-ml.txt
```

The second `pip` installs the PyTorch front end (`torch`, `torchvision`, `torch_mlir`).
ST-Flow is pinned to builds that are no longer on PyPI, so they are installed from the
[Zenodo record](https://doi.org/10.5281/zenodo.21881993). The package is named
`st-flow` on PyPI and `stflow` in Python.

To run the generated designs you also need:

- **Vitis HLS** (validated with 2024.2), for C-simulation, which is on by default.
  Source its `settings64.sh` before running ST-Flow.
- **AMPL** with a MINLP solver on `PATH`, only if you use MINLP-based design-space
  exploration. The heuristic engines (`sa`, `ga`) need nothing extra.
- **XRT**, only to run a generated bitstream on an FPGA board.

To build from source, for another platform or to work on ST-Flow itself, see
[DEVELOPMENT.md](DEVELOPMENT.md).
<!-- docs:install:end -->

## Quick start

```python
import torch
import torch.nn as nn
from stflow import generate_hardware, STFlowConfig

class GEMM(nn.Module):
    def forward(self, a, b, c_in):
        return (a @ b) * 0.5 + c_in * 0.1

inputs = (torch.randn(200, 240), torch.randn(240, 220), torch.randn(200, 220))

config = STFlowConfig.load()
config.hw_mapping_strategy = "balanced"   # no_reuse | balanced | max_reuse
generate_hardware("gemm", GEMM(), inputs, config)
```

This traces the model, explores the design space, emits an HLS dataflow design in a
project directory (here `500_gemm_comb_sa_balanced.prj/`) and checks it against PyTorch
with C-simulation. The [documentation](https://st-flow.readthedocs.io) covers the
options, the compilation flow and the generated files; every test under
[`test/`](test/) is another worked example.

## PACT '26 artifact

To reproduce the paper, start at [ARTIFACT_EVALUATION.md](ARTIFACT_EVALUATION.md). The
evaluated version is the git tag `PACT26`, identical to the
[Zenodo record](https://doi.org/10.5281/zenodo.21881993), which also holds the archived
experiments (5.6 GB). The guide's first tier regenerates every figure and table in the paper in
seconds, with no FPGA, HLS licence or solver.

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

ST-Flow builds on prior open-source work, and we thank the authors of
[ScaleHLS](https://github.com/UIUC-ChenLab/scalehls),
[HIDA](https://github.com/UIUC-ChenLab/ScaleHLS-HIDA),
[HeteroCL](https://github.com/cornell-zhang/heterocl),
[Allo](https://github.com/cornell-zhang/allo),
[Stream-HLS](https://github.com/UCLA-VAST/Stream-HLS),
[TAPA](https://github.com/rapidstream-org/rapidstream-tapa),
[Torch-MLIR](https://github.com/llvm/torch-mlir) and [LLVM/MLIR](https://mlir.llvm.org).
See [NOTICE](NOTICE) for the code adapted/extended/modified from these projects and its licenses.

This work was partially supported by the National Science Foundation (NSF) under Award
No. 2533041 for NQVL, the [CDSC industrial partners](https://cdsc.ucla.edu/partners/),
the AMD HACC Programs, and the PRISM Center under the JUMP 2.0 Program.
<!-- docs:ack:end -->

## License

ST-Flow is released under the [Apache License 2.0](LICENSE). Third-party components and
their notices are listed in [NOTICE](NOTICE), with full license texts in
[`licenses/`](licenses/). The PACT '26 paper is published under CC BY 4.0.
