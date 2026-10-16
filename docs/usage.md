# Using ST-Flow

ST-Flow is driven from Python. You give it a `torch.nn.Module` and example inputs; it
traces the model through Torch-MLIR, runs design-space exploration (DSE), emits an HLS
dataflow design and, by default, checks that design against PyTorch with Vitis HLS
C-simulation.

## A minimal example

```python
import torch
import torch.nn as nn
from stflow import generate_hardware, STFlowConfig

class GEMM(nn.Module):
    def forward(self, a, b, c_in):
        return (a @ b) * 0.5 + c_in * 0.1

inputs = (torch.randn(200, 240), torch.randn(240, 220), torch.randn(200, 220))

config = STFlowConfig.load()              # test_config.yaml in the cwd, if present
config.hw_mapping_strategy = "balanced"   # no_reuse | balanced | max_reuse
generate_hardware("gemm", GEMM(), inputs, config)
```

This is `test/03_polybench/test_03_gemm.py` in miniature; every file under `test/` is a
worked example of the same call on a different model.

Before running it, set up the environment as described in {doc}`installation`:

```bash
source env.sh          # LLVM/MLIR, ST-Flow build
source local_env.sh    # Vitis HLS, AMPL (and TAPA) for csim and MINLP-based DSE
```

## Hardware mapping strategies

`hw_mapping_strategy` selects how kernels are mapped onto hardware:

| Strategy | Mapping |
|---|---|
| `no_reuse` | Pure spatial: every kernel gets its own dedicated hardware module. |
| `balanced` | Hybrid spatial-temporal: kernels of the same structural type share an engine only when they are sequentially dependent and their trip counts and loop bounds are within `balanced_mapping_similarity_threshold` / `balanced_mapping_ub_threshold` of each other; independent kernels stay parallel. |
| `max_reuse` | Maximal temporal sharing: all kernels of the same structural type (loop depth, operation sequence, tensor ranks) share one engine. |

The paper finds `balanced` gives the best area–delay trade-off.

## Tiling strategies

Unrolling (tiling) is controlled by two DSE options; the three configurations studied in
the paper are:

| Paper name | `enforce_divisor_only_tiling` | `shared_tiling_strategy` |
|---|---|---|
| `Div-Min` | `true` | `min` |
| `Non-Div-Min` | `false` | `min` |
| `Non-Div-Max` | `false` | `max` |

## DSE engines

Loop permutation and unrolling are each solved by a configurable engine
(`dse_options.permutation_dse`, `dse_options.unrolling_dse`):

- `minlp` — exact MINLP formulation solved through AMPL (needs `ampl` on `PATH`).
- `sa` — simulated annealing; scales to deep graphs such as GPT-2 where MINLP times out.
- `ga` — genetic algorithm; more stable run to run than SA.
- `random` — random search baseline.

Set `enable_combined_dse: true` to solve permutation and unrolling jointly with
`dse_options.combined_dse` instead. See {doc}`configuration` for every option.

## Choosing how far to go: `STFLOW_MODE`

`STFlowConfig.load()` applies an override from the `STFLOW_MODE` environment variable,
which is how the test suite and the Slurm drivers select a stage without editing configs:

| `STFLOW_MODE` | Effect |
|---|---|
| *(unset)* | Use the config as written (default: full DSE + C-simulation). |
| `csim` | Full DSE + C-simulation. |
| `csim_only` | Re-run C-simulation on an already-compiled design, skipping DSE. |
| `syn` | Vitis HLS synthesis of an already-compiled design. |
| `cosim` | Vitis HLS synthesis + RTL co-simulation of an already-compiled design. |
| `bitstream` | Bitstream generation for an already-compiled design (Vivado + a U55C platform). |

The "already-compiled" modes set `compile_only`: instead of re-running DSE they load
`03_top_modules/06_final_top.mlir` from the existing project directory, so run the
default (or `csim`) mode first.

## What gets generated

Each call creates a project directory named after the configuration,
`<DSP_limit>_<name>_<permutation_dse>_<unrolling_dse>_<strategy>.prj`
(for example `500_gemm_minlp_minlp_balanced.prj`). Inside it:

- `config.yaml` / `config.json` — the exact configuration used.
- `stflow.log` — the full Python and C++ log of the run.
- Intermediate MLIR for every stage of the flow (see {doc}`compilation-flow`).
- DSE reports, the generated HLS C++ kernel and host code, and a `Makefile` /
  `link_config.ini` for the Vitis bitstream flow.

`test/*/clean*.sh` removes the project directories under `test/`.
