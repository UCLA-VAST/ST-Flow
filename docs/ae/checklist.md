# Reproducibility Checklist — ST-Flow (PACT '26)

Answers to the [cTuning reproducibility checklist](https://github.com/ctuning/artifact-evaluation/blob/master/docs/checklist.md).
Full instructions: [`ARTIFACT_EVALUATION.md`](../../ARTIFACT_EVALUATION.md) — this file and
[`artifact_appendix.tex`](artifact_appendix.tex) are short-form derivatives of it. 

## Abstract

ST-Flow is a hardware compiler that lowers PyTorch models and affine programs to FPGA
streaming dataflow accelerators through MLIR, HLS synthesis and place-and-route. The
artifact contains the compiler plus the full archived record of the paper's experiments
(5.6 GB, 41,874 files) — generated sources, solver traces, HLS/implementation reports and
on-board logs. One command, `scripts/ae_verify.sh`, replays the analysis pipeline over
those archives and regenerates every figure and table in the paper in about 11 seconds,
needing only Python; the CSVs and tables come back byte-identical, the figure PDFs differ
only in embedded timestamp. Building the compiler and running its 66-test suite
additionally needs Vitis HLS 2024.2; reproducing the physical results needs a Xilinx
Alveo U55C. See the four-tier breakdown in `ARTIFACT_EVALUATION.md`.

**Badges requested:** Available, Functional, Reproduced.

## Checklist

- **Algorithm:** Spatial-temporal hardware mapping via structural/temporal kernel
  clustering; non-divisor loop tiling (`div-min`, `non-div-min`, `non-div-max`); design
  space exploration over a closed-form analytical latency/resource model by MINLP,
  simulated annealing or genetic algorithm.
- **Program:** No standard suite. Six DNN workloads (MLP-Mixer, RWKV channel-mix, TabM,
  SwiGLU, DIN, MNIST MLP), eight synthetic dataflow topologies (GEMM chains and
  reduction trees, deep and branch-merge MLPs, uniform and random loop bounds), seven
  Polybench kernels, and GPT-2 Tiny as a case study — all included as PyTorch modules
  under `test/`, none need downloading.
- **Compilation:** LLVM/MLIR 20 with Python bindings, Release plus assertions — provided
  prebuilt, since ST-Flow does not compile against distribution LLVM 17/18. CMake ≥ 3.20,
  Ninja, Python 3.12. Vitis HLS 2024.2 (proprietary, licensed) compiles the generated
  accelerators.
- **Transformations:** The ST-Flow compiler itself — an MLIR dialect with DFG solvers,
  transform passes and HLS/TAPA emission. Public, included, at the evaluated revision.
- **Binary:** No prebuilt ST-Flow binary; the prebuilt LLVM/MLIR 20 toolchain is
  included. Generated FPGA bitstreams (`.xclbin`) are not included (available upon request), but their
  implementation reports are.
- **Model / Data set:** Weights are seeded-random (`torch.manual_seed(42)`); inputs are
  synthetic tensors from the same fixed seed. Nothing is downloaded.
- **Run-time environment:** Linux x86_64, Python 3.12 (`cp312`-only pinned wheels).
  Key pins: `pybind11<3` (3.x can't compile the MLIR 20 bindings) and
  `matplotlib==3.10.5` (the version that produced the paper's figures). Vitis
  HLS/Vivado 2024.2 and XRT for synthesis and hardware; AMPL for MINLP. No root access
  required.
- **Hardware:** A Xilinx Alveo U55C at 300 MHz for bitstream generation and on-board
  execution (Tier 4 only). Tiers 1–2 need only a Linux workstation.
- **Run-time state:** The Tier 2 reproduction path is insensitive to machine load. The
  Tier 1 test suite runs HLS C-simulation in parallel and is load-sensitive — keep the
  pytest worker count at or below the core count and re-run any failure serially before
  treating it as real.
- **Metrics:** Cycle counts (analytical, Vitis min/max, RTL simulation, on-board); MAPE
  and Pearson correlation of prediction against RTL; speedup vs. divisor-only tiling;
  DSP/BRAM/URAM utilization; area-delay product; achieved clock frequency; DSE solution
  cost, wall-clock search time and coefficient of variation.
- **Output:** Four CSVs, seven figure PDFs and four LaTeX tables (Figures 3–6, Tables
  4–7), written to `PACT26_Artifacts/data/`. Reference copies ship alongside; `ae_verify.sh`
  diffs against them automatically. Validation statistics print to the console during
  regeneration.
- **Experiments:** 171 main-sweep designs (three tiling strategies × 14 workloads at three
  hardware-mapping strategies, plus Polybench and an extra DSP-budget sweep for the CODO
  comparison), 240 GPT-2 Tiny DSE runs (4 solvers × 20 repeats × 3 mappings), and 14
  augmented CODO baseline designs (9 more failed and are archived separately) — 425
  experiments in total. See the C1–C7 claims table in `ARTIFACT_EVALUATION.md` for what
  each one regenerates.
- **Disk space:** ~5.6 GB archive (~10 GB after extraction), ~5 GB for the LLVM/MLIR
  install, ~5 GB for the compiler build. Allow 20 GB.
- **Time to prepare workflow:** ~10 minutes for the archived-data path (clone + `pip
  install`); ~15–30 minutes more to build the compiler using the prebuilt LLVM.
- **Time to complete experiments:** Tier 2 (all figures/tables): ~11 s. Tier 1 (66-test
  suite): ~4 min at `-n11` on a 20-core host. Tier 3 (one design through HLS): minutes to
  ~1.5 hours per design. Tier 4 (bitstream + on-board): hours per design, days for the full sweep.
- **Publicly available?** Yes, as an Open Access archival record: https://doi.org/10.5281/zenodo.21881993.
  Self-contained: source, all archived experiments, the prebuilt LLVM/MLIR toolchain and
  the pinned Python wheels.
- **Code / Data licenses:** Apache License 2.0 (code); Creative Commons Attribution 4.0
  (archived experiment outputs).
- **Workflow framework:** No third-party framework, `pytest` for the functional suite, SLURM for
  the cluster sweep (not needed to evaluate the artifact).
- **Archived?** https://doi.org/10.5281/zenodo.21881993

## Installation

```bash
tar xzf st-flow-pact26.tar.gz && cd st-flow-pact26
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
source env.sh
```

Sufficient for Tier 2. To build the compiler (Tiers 1, 3, 4):

```bash
mkdir -p ~/opt && tar xzf llvm-stflow-ubuntu-latest.tar.gz -C ~/opt
bash scripts/install_ml_wheels.sh wheels/
source env.sh
export NUM_THREADS=6      # bounded on purpose: one job per core OOMs the build
python -m pip install -v -e .
```

## Evaluation

```bash
source env.sh
bash scripts/ae_verify.sh
```

Expected: `RESULT: PASS`, 15 outputs reproduced, 0 mismatches, ~11 seconds. Full
walkthrough, per-tier instructions and the C1–C7 claims table:
[`ARTIFACT_EVALUATION.md`](../../ARTIFACT_EVALUATION.md).

## Notes

Per-design DSE reports are named `reports/phoenix.csv` (the project's former name, ST-Flow
was renamed after the experiments ran); the reporting code accepts either name. CODO
baseline designs were built from the third-party CODO artifact and augmented only with
AXI-DMA interfaces; `scripts/codo/collect_codo_reports.py` re-derives `codo.csv` from
their archived Vitis reports, so Figure 5 is reproducible without the CODO toolchain
itself.
