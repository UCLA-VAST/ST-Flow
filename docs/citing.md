# Citing ST-Flow

If you use ST-Flow in your research, please cite the PACT '26 paper:

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

The evaluated artifact (source, archived experiments, prebuilt LLVM/MLIR and pinned
wheels) is archived on Zenodo:

> Basalama, S., Pouget, S., & Cong, J. (2026). *ST-Flow: A Hardware Compiler for
> Spatial-Temporal Dataflow Acceleration (PACT '26 Artifact).* Zenodo.
> <https://doi.org/10.5281/zenodo.21881993>

The paper is available at <https://doi.org/10.1145/3838684.3846865> (CC BY 4.0).

## License

ST-Flow is released under the
[Apache License 2.0](https://github.com/UCLA-VAST/ST-Flow/blob/main/LICENSE).
Third-party components (ScaleHLS-, HeteroCL- and Allo-derived files, Stream-HLS,
vendored XRT) and their notices are listed in
[NOTICE](https://github.com/UCLA-VAST/ST-Flow/blob/main/NOTICE), with full license
texts in [`licenses/`](https://github.com/UCLA-VAST/ST-Flow/tree/main/licenses).

## Acknowledgments

```{include} ../README.md
:start-after: <!-- docs:ack:start -->
:end-before: <!-- docs:ack:end -->
```

ST-Flow is developed at the [UCLA VAST Lab](https://vast.cs.ucla.edu).
