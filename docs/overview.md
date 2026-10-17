# Overview

```{include} ../README.md
:start-after: <!-- docs:overview:start -->
:end-before: <!-- docs:overview:end -->
```

## Why spatial-temporal

Spatial dataflow compilers map each kernel to its own dedicated hardware unit, which
quickly exhausts on-chip resources on deep networks. ST-Flow's reuse-aware compilation
instead identifies kernels that can safely share a compute engine and time-multiplexes
them, while keeping the design deadlock-free under structural back-pressure.

## Features

- **Spatial-temporal unified mapping.** The search space covers pure spatial, pure
  temporal and hybrid spatial-temporal architectures.
- **Hybrid mapping heuristic.** A clustering algorithm groups compatible compute nodes
  into shared hardware modules. Nodes must share a structural signature (loop depth,
  operation sequence, tensor ranks) and have matching arithmetic intensities, to avoid
  latency padding or under-utilization.
- **Non-divisor tiling.** Hardware unrolling is decoupled from strict divisibility:
  - `Div-Min`: divisor-only tiling bounded by the smallest shared loop size.
  - `Non-Div-Min`: non-divisor tiling bounded by the smallest shared loop size.
  - `Non-Div-Max`: non-divisor tiling bounded by the largest loop size, maximizing
    parallelism at the cost of transient padding.
- **Multi-engine search.** Besides the exact MINLP formulation, ST-Flow implements
  simulated annealing (SA) and genetic algorithms (GA), which scale to deeply nested
  graphs such as GPT-2 where mathematical solvers time out.
- **Post-processing.** Layout converters for mismatched stream patterns, flattening of
  multi-dimensional stream arrays into single handshaked vector paths, and DMA memory
  layouts that sustain high-burst external memory bandwidth.

See {doc}`usage` for how these map onto configuration options, and {doc}`development`
for the repository layout.
