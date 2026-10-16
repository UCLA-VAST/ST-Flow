# Configuration reference

All options live in `STFlowConfig` (`stflow/config.py`). They can be set in Python or in a
YAML file:

```python
from stflow import STFlowConfig
config = STFlowConfig.load("my_config.yaml")   # or STFlowConfig.load() / STFlowConfig()
config.dse_options.unrolling_dse = "sa"
```

`STFlowConfig.load()` with no path reads `test_config.yaml` from the current directory,
falling back to `vanilla_config.yaml`, then to the defaults below; unknown keys are
ignored. The `STFLOW_MODE` environment variable is applied on top (see {doc}`usage`).
The configuration a run actually used is written to `<project>.prj/config.yaml`.

A typical file, `test/03_polybench/test_config.yaml`:

```yaml
enable_unrolling: True
enable_permutation: True
enable_combined_dse: False
atol: 0.01
dse_options:
  permutation_dse: 'minlp'
  unrolling_dse: 'minlp'
  balanced_mapping_similarity_threshold: 1.1
  balanced_mapping_ub_threshold: 1.1
  latency_memory_ratio: 80
  min_time_loop_size: 10
  objective_scaling_factor: 10
```

## Top-level options

| Option | Default | Description |
|---|---|---|
| `hw_mapping_strategy` | `no_reuse` | `no_reuse`, `balanced` or `max_reuse` (see {doc}`usage`). |
| `target_tool` | `vitis` | Backend: `vitis` (Vitis HLS) or `tapa` (TAPA / RapidStream). |
| `csim` | `true` | Run C-simulation and compare against the PyTorch output. |
| `atol` | `1e-3` | Absolute tolerance for that comparison. |
| `run_vitis_hls_synth` | `false` | Run Vitis HLS synthesis. |
| `run_vitis_hls_cosim` | `false` | Run RTL co-simulation (implies synthesis). |
| `run_bitstream` | `false` | Generate an FPGA bitstream. |
| `compile_only` | `false` | Reuse the final top module of an existing project instead of re-running DSE. |
| `enable_dma` | `true` | Rewrite memory modules to DMA for burst off-chip access. |
| `enable_permutation` | `true` | Run loop-permutation DSE. |
| `enable_unrolling` | `true` | Run loop-unrolling (tiling) DSE. |
| `enable_combined_dse` | `true` | Solve permutation and unrolling jointly with `dse_options.combined_dse`. |
| `full_data_reuse` | `true` | Allow full on-chip data reuse when merging kernels. |
| `swmr_elimination` | `aggressive` | Single-write-multiple-read elimination: `skip`, `inplace`, `conservative` or `aggressive`. |
| `debug_types` | `['']` | LLVM `-debug-only` types to enable in the C++ passes. |
| `dse_options` | — | DSE options, below. |

## DSE options (`dse_options`)

| Option | Default | Description |
|---|---|---|
| `permutation_dse` | `sa` | Engine for loop permutation: `minlp`, `sa`, `ga` or `random`. |
| `unrolling_dse` | `sa` | Engine for loop unrolling. |
| `combined_dse` | `sa` | Engine for the joint problem when `enable_combined_dse` is set. |
| `permutation_timeout` / `unrolling_timeout` / `combined_timeout` | `60` | MINLP solver timeout, seconds. |
| `permutation_ampl_solver` / `unrolling_ampl_solver` / `combined_ampl_solver` | `gecode` | AMPL solver used by `minlp`. |
| `num_n_runs` | `1` | Independent DSE runs per design (the best is kept). |
| `DSP_limit` | `500` | DSP budget constraint; also prefixes the project directory name. |
| `Memory_limit` | `1` | Memory budget (passed to the C++ model). |
| `latency_memory_ratio` | `80` | Objective weighting, in percent: latency weight = ratio, memory weight = 100 − ratio. |
| `objective_scaling_factor` | `1000` | Upper bound on objective coefficients; larger coefficients are scaled down to keep the MINLP well-conditioned. |
| `min_time_loop_size` | `8` | Minimum trip count kept in the time (non-unrolled) loop when tiling. |
| `enforce_divisor_only_tiling` | `true` | Restrict unroll factors to divisors of the loop bound. |
| `shared_tiling_strategy` | `min` | Bound shared tiling by the `min` or `max` loop size (see the tiling table in {doc}`usage`). |
| `balanced_mapping_similarity_threshold` | `10.0` | `balanced` mapping: max ratio between two kernels' total trip counts. |
| `balanced_mapping_ub_threshold` | `5.0` | `balanced` mapping: max ratio between corresponding loop bounds. |
| `fixed_permutation_solution` / `fixed_unrolling_solution` | — | Bypass DSE with a fixed solution. |

### Heuristic engine settings

| Option | Fields (defaults) |
|---|---|
| `*_sa_config` | `initial_temp` (1000.0), `cooling_rate` (0.999), `iterations` (5000), `initial_solution` (`default`) |
| `*_ga_config` | `pop_size` (50), `generations` (500), `mutation_rate` (0.1), `crossover_rate` (0.8), `init_method` (`mixture`) |
| `*_random_config` | `iterations` (10) |

where `*` is `permutation`, `unrolling` or `combined`.
