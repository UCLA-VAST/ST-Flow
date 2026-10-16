# ST-Flow

**A hardware compiler for spatial-temporal dataflow acceleration.** ST-Flow lowers PyTorch
models and affine programs, through MLIR, into FPGA streaming dataflow accelerators, and
time-multiplexes compatible kernels onto shared compute engines to reclaim DSPs and BRAM
without giving up streaming concurrency.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} 🚀 Installation
:link: installation
:link-type: doc
Build the compiler against the prebuilt LLVM/MLIR 20 and the pinned PyTorch toolchain.
:::

:::{grid-item-card} 🧩 Using ST-Flow
:link: usage
:link-type: doc
Compile a PyTorch module to an HLS dataflow design with one call.
:::

:::{grid-item-card} ⚙️ Configuration
:link: configuration
:link-type: doc
Every `STFlowConfig` / `test_config.yaml` option, with its default.
:::

:::{grid-item-card} 📄 PACT '26 Artifact
:link: artifact-evaluation
:link-type: doc
Regenerate every figure and table in the paper from the archived experiments.
:::
::::

```{toctree}
:hidden:
:caption: Getting started

overview
installation
usage
```

```{toctree}
:hidden:
:caption: User guide

configuration
compilation-flow
experiments
```

```{toctree}
:hidden:
:caption: PACT '26

artifact-evaluation
```

```{toctree}
:hidden:
:caption: Project

development
citing
GitHub <https://github.com/UCLA-VAST/ST-Flow>
Zenodo record <https://doi.org/10.5281/zenodo.21881993>
```
