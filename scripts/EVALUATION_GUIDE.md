# Evaluation Guide

Commands to manage, run, and analyze experiments in the ST-Flow HLS environment.

## 0. Setup
Ensure you have sourced the environment from the project root:
```bash
source slurm_env.sh
```

## 1. Manage Evaluations (`evgen`)

### Generate/Update Folders
Populates the `evaluation/` directory with tests, project configs, and Slurm scripts.
```bash
evgen

# Update scripts ONLY (nodelists, resource limits)
# Use this when you only need to change Slurm parameters or node exclusions.
evgen --scripts-only
```

## 2. Running Experiments (`evrun`)

### Submit Jobs
Submit jobs for various stages of the pipeline.
```bash
# Standard DSE + CSIM
evrun submit --mode csim

# High Priority submission (-A suhailb account)
evrun submit --mode csim -p

# HLS Stages
evrun submit --mode syn
evrun submit --mode cosim

# Bitstream Generation (Runs 'make bitstream' in scratch)
evrun submit --mode gen_bit

# Hardware Execution (Runs bitstream on FPGA via pytest)
evrun submit --mode run_bit
```

### Advanced Filtering
You can filter by strategy (`-s`), benchmark (`-b`), or individual test (`-t`). All support comma-separated lists.
```bash
# Run SA and MINLP for DNN benchmarks containing 'mlp' in the test name
evrun submit -s "*sa*,*minlp*" -b "01_dnn" -t "*mlp*"
```

### Cancel Jobs
Stop active Slurm jobs using filters.
```bash
# Cancel only PENDING (queued) jobs for a specific benchmark
evrun cancel --pending -b "05_mot"
```

## 3. Monitoring & Reporting

### Status Dashboard
The dashboard provides a quick look at pass counts and active jobs.
```bash
# Summary view
evrun status

# Detailed view (includes Cycle Errors and DSP Errors)
evrun status -v

# failure filtering (e.g. show only projects where synthesis failed)
evrun status -f syn
```

### Data Collection
Consolidates all metrics from `stflow.csv` and Vitis/Bitstream reports into a single CSV.
```bash
# Generates study/data/all.csv
# This command automatically detects and prioritizes 'evaluation_bu'
evrun report
```

## 4. Visualization (`evplot`)

The `evplot` utility centralizes the generation of paper-ready figures.

### Benchmark Performance
Generates a wide categorical bar plot showing results across different hardware strategies and unrolling variants.
```bash
evplot bench 05_mot --cycles rtl_cycles
```

### Latency Validation
Compares ST-Flow predicted cycles vs. RTL simulation ground truth.
```bash
evplot latency
```

### Experiment Summary
Aggregates performance stats across multiple runs of the same configuration.
```bash
evplot summary
```

## 5. Key Metrics
- **Status Flags**: `BitGen` (XCLBIN exists), `BitRun` (Matched golden + latency file exists).
- **Accuracy**: `cycle_err_rtl (%)`, `dsp_err_vitis (%)`.
- **Resource Usage**: Predicted vs Vitis vs Bitstream for DSPs and Memory.
- **Latency**: Cycles across all simulation and implementation levels.
