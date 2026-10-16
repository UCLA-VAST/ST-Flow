# Compilation flow

`generate_hardware()` runs the stages below (`stflow/pipelines.py`). Each stage writes its
output into the project directory, so a run can be inspected step by step.

```text
PyTorch model ─► Torch-MLIR (linalg) ─► affine + bufferization ─► dataflow graph
      ─► hardware-mapping DSE ─► loop permutation / unrolling DSE ─► streaming top module
      ─► DMA / stream post-processing ─► Vitis HLS or TAPA C++ ─► csim / synth / cosim / bitstream
```

| # | Stage | Output in `<project>.prj/` |
|---|---|---|
| 1 | **Front-end.** Run the model in PyTorch for the golden output, then lower it with Torch-MLIR to `linalg`-on-tensors. | `01_pre_hldse_modules/01_input.mlir` |
| 2 | **Lower to affine** and bufferize. | `02_affine.mlir` |
| 3 | **Binarize weights** to files and turn them into function arguments. | `03_weights_binarized.mlir` |
| 4 | **Bufferize function arguments** into explicit off-chip reads/writes, and **eliminate single-write-multiple-read** buffers (`swmr_elimination`). | `04_…`, `05_bufferized.mlir`, `06_eliminated_swmr.mlir` |
| 5 | **Annotate** buffers and loops for the DSE. | `07_annotated.mlir` |
| 6 | **High-level DSE: hardware mapping.** Group kernels into shared engines according to `hw_mapping_strategy`. | `02_dfg/` |
| 7 | **Loop DSE.** Solve loop permutation and unrolling (tiling) on the merged dataflow graph with MINLP, SA, GA or random search. | `02_dfg/`, `reports/stflow.csv` |
| 8 | **Kernel merging, FIFO conversion and hardware reuse** on the dataflow graph: memrefs between kernels become streams, and shared engines are time-multiplexed. | `02_dfg/04_merged*`, `02_dfg/05_fifos*` |
| 9 | **System module generation.** | `03_top_modules/01_top.mlir` |
| 10 | **Post-processing.** Flatten arrays of streams, pack them into vector streams, and rewrite memory modules to DMA for burst off-chip access (`enable_dma`). | `03_top_modules/03_…`–`05_top_dma.mlir`, `06_final_top.mlir` |
| 11 | **Backend.** Emit Vitis HLS (or TAPA) C++ and host code, then run csim, synthesis, co-simulation or bitstream generation as configured. | `vitis/`, HLS project directories, `Makefile` |

Every `02_dfg/` step also writes the graph as Graphviz (`*.dot`) plus node and edge
listings, which is the quickest way to see what the mapping and DSE decided.

## Where the code lives

| Component | Location |
|---|---|
| Python driver and pipeline | `stflow/pipelines.py`, `stflow/main.py` |
| Configuration | `stflow/config.py` |
| Backends (Vitis HLS, TAPA) | `stflow/builders/`, `stflow/emitters/` |
| MLIR dialect and passes | `mlir/include/streamhls/`, `mlir/lib/Dialect/Dataflow/` |
| Hardware-mapping DSE | `mlir/lib/Dialect/Dataflow/Transforms/HighLevelDSE.cpp` |
| Performance model and solvers (MINLP/SA/GA) | `mlir/lib/Dialect/Dataflow/Support/` |
| HLS C++ emission | `mlir/lib/Dialect/Dataflow/Translation/` |
