# Running experiments on Slurm

The `scripts/ev*` drivers generate, submit and collect large experiment sweeps (the
paper's 400+ designs) on a Slurm cluster. They are research infrastructure: they assume a
Slurm scheduler with environment modules, Vitis/Vivado 2024.2 and, for on-board runs,
Alveo U55C nodes.

:::{important}
`slurm_env.sh` is specific to the cluster the paper was evaluated on. Before using it,
set `PRJ_ROOT` to your checkout and adjust the `ml load` line and the Slurm
account/partition options in `scripts/evrun` to your site.
:::

```{include} ../scripts/EVALUATION_GUIDE.md
:start-line: 1
```
