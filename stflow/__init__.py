from . import _mlir
from .main import (
  enable_llvm_debug_only,
  lower_linalg_to_affine,
  build_hls,
  build_tapa,
  build_vitis,
  run_high_level_dse,
  extract_merged_dfg,

  run_loop_permutation_minlp,
  run_loop_permutation_sa,
  run_loop_permutation_ga,
  run_loop_unrolling_minlp,
  run_loop_unrolling_sa,
  run_loop_unrolling_ga,
  run_loop_combined_minlp,
  run_loop_combined_sa,
  run_loop_combined_ga,
  find_random_permutation_solution,
  set_fixed_permutations,
  find_random_unrolling_solution,
  find_random_combined_solution,
  set_fixed_unrolling_factors,
  set_unrolling_factors_to_ones,
  DSEOptions,
  SAConfig,
  GAConfig,
  
  generate_system_module,
  annotate_buffers,
  annotate_loops,
  binarize_weights,
  bufferize_func_args,
  eliminate_single_write_multiple_reads,
  flatten_array_of_streams,
  array_streams_to_vector_stream,
  eleminate_unit_dims_and_iters,
  rewrite_memory_modules_to_dma,
  raise_inner_loop_by_one_level,
  convert_torch_to_mlir,
  print_to_file,
  merge_kernels_dfg,
  convert_func_memrefs_to_fifos_dfg,
  reuse_hardware_dfg,
  merge_super_nodes_dfg,
  wrap_super_nodes_dfg,
  print_dfg_dot,
  print_dfg_nodes,
  print_dfg_edges
)

from .pipelines import generate_hardware, e2e_pipeline

from .host import (
 codegen_host
)
from .host_tapa import (
 codegen_host_tapa,
 generate_link_config
)

from .utils import invoke_mlir_parser

from .config import STFlowConfig


