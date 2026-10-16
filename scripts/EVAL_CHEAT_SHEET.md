# Evaluation Cheat Sheet

**Setup**: `source slurm_env.sh` (from project root)

---

## 1. Infrastructure (`evgen`)
Manages the `evaluation/` folder structure.

```sh
# (Re)generate all tests, configs, and Slurm scripts
evgen

# Update scripts ONLY (nodelists, limits); skips config updates
evgen --scripts-only
```

---

## 2. Execution & Monitoring (`evrun`)
Orchestrates Slurm jobs and consolidated reporting.

### Submission
```sh
# Run DSE + Simulation (use -p for high priority)
evrun submit --mode csim [-p]

# Fast Mode: Skip solver, run existing hardware
evrun submit --mode csim_only

# Run HLS Synthesis / Co-simulation
evrun submit --mode syn
evrun submit --mode cosim
# Filter by strategy or benchmark (supports comma-separated list)
evrun submit -s "*sa*,*minlp*" -b "01_dnn,03_tmp" -t "*3mm*"
# Hardware flow (Bitstream Gen -> Run on FPGA)
evrun submit --mode gen_bit
evrun submit --mode run_bit
```

### Filtering & Monitoring
```sh
# Filter by strategy, benchmark, or specific test
evrun status -s "*ga*" -b "01_dnn" -t "*3mm*"

# Cancel jobs (e.g. only pending jobs for a specific benchmark)
evrun cancel --pending -b "05_mot"

# View status (detailed)
evrun status -v

# Failure mode: Show only failures (with clickable stflow.log links)
evrun status -f [DSE|csim|syn|cosim|latency]
```

---

## 3. Reporting & Visualization
Collect data and generate paper-ready plots.

### Data Collection
```sh
# Consolidate all metrics into study/data/all.csv
# Automatically prioritizes evaluation_bu if present
evrun report
```

### Plotting (`evplot`)
```sh
# Plot benchmark comparison (Categorical)
evplot bench 05_mot --cycles rtl_cycles

# With filtering (e.g. only SA strategies for DNN tests)
evplot bench -s "*sa*" -b "01_dnn" -t "*3mm*"

# Plot prediction accuracy for specific configs
evplot latency -s "*minlp*" -c b_cycles

# Plot performance summary across filtered multi-runs
evplot summary -t "*diamond*"
```

---

## 4. Cleanup (`evclean`)
```sh
# Remove project folders and logs based on filters
evclean -s "*ga*" -b "01_dnn"
```
