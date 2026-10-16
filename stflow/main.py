
from ._mlir.dialects import streamhls as streamhls_d
# from .emit import StreamHLSModule
# from .schedule import customize_python
from .utils import convert_torch_to_mlir_fx, convert_torch_to_mlir_torchscript
from .tapa_build import build_tapa
from .vitis_build import build_vitis

from collections.abc import Callable
import io
from typing import Union
import os

import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

DSEOptions = streamhls_d.DSEOptions
SAConfig = streamhls_d.SAConfig
GAConfig = streamhls_d.GAConfig

# def parse_python(
#   fn: Union[Callable, str],
#   instantiate: list = None
# ):
#   """
#   Parse a Python function or a string containing Python code into an MLIR module.
#   Args:
#       fn (Callable or str): The Python function or code to be parsed.
#       instantiate (list): List of types to instantiate in the function.
#   Returns:
#       MLIR module: The parsed MLIR module.
#   """
#   return customize_python(fn, instantiate=instantiate).module

def build_hls(
    module, 
    top_func_name,
    project,
    platform="vitis_hls",
    compile_only=False,
    mode="csim",
    ext_libs=[],
    configs=[],
    func_args=[],
    wrap_io=False
):
  """
  Build the given MLIR module into a StreamHLS project.
  Args:
      module (MLIR module): The MLIR module to be built.
      project (str): Path to the project directory.
      top_func_name (str): Name of the top function in the module.
  Returns:
      StreamHLSModule: The built StreamHLS module.
  """
  pass
#   return StreamHLSModule(
#     module,
#     top_func_name=top_func_name,
#     platform=platform,
#     compile_only=compile_only,
#     mode=mode,
#     project=project,
#     ext_libs=ext_libs,
#     configs=configs,
#     func_args=func_args,
#     wrap_io=wrap_io,
#   )

# Debugging Utilities
def enable_llvm_debug_only(debug_types : list):
  """
  Enable LLVM debug output for specific debug types.
  Args:
      debug_types (list): List of debug types to enable.
  """
  streamhls_d.enable_llvm_debug_only(debug_types)

# StreamHLS Passes and Pipelines
def lower_linalg_to_affine(module, debug_point=0):
  """
  Lower Linalg operations in the given MLIR module to Affine operations.
  Args:
      module (MLIR module): The MLIR module to be processed.
  Returns:
      MLIR module: The processed MLIR module with Linalg operations lowered to Affine.
  """  
  return streamhls_d.lower_linalg_to_affine(module, debug_point)

def merge_kernels(modules, options):
  """
  Merge multiple kernels into a single kernel.
  Args:
      modules (list): List of MLIR module objects to be merged.
      options (dict): Options for merging kernels.
    Returns:
      MLIR module: Merged MLIR module.
  """  
  return streamhls_d.merge_kernels(modules, options)

def merge_kernels_new(modules, options):
  """
  Merge multiple kernels into a single kernel.
  Args:
      modules (list): List of MLIR module objects to be merged.
      options (dict): Options for merging kernels.
    Returns:
      MLIR module: Merged MLIR module.
  """  
  return streamhls_d.merge_kernels_new(modules, options)

def reuse_hardware(module):
  """
  Reuse hardware resources in the given MLIR module.
  Args:
      module (MLIR module): The MLIR module to be processed.
  Returns:
      MLIR module: The processed MLIR module with reused hardware.
  """
  return streamhls_d.reuse_hardware(module)

def run_high_level_dse(module, project_path, options, hw_mapping_strategy="no_reuse"):
  """Run high-level DSE on the DFG.
  
  Args:
      module (ir.Module): The input MLIR module.
      project_path (str): The path to the project directory.
      options (DSEOptions): The DSE options.
      hw_mapping_strategy (str): The type of DSE mapping (no_reuse, max_reuse).

  Returns:
      HWMapping: The hardware mapping result.
  """
  return streamhls_d.run_high_level_dse(module, project_path, "", options, hw_mapping_strategy)

def extract_modules(module, project_path, enable_tiling=True):
  return streamhls_d.extract_modules(module, project_path, enable_tiling)

def extract_merged_dfg(module, hw_mapping, enable_tiling=True):
  return streamhls_d.extract_merged_dfg(module, hw_mapping, enable_tiling)

def print_dfg_nodes(graph):
  return streamhls_d.print_dfg_nodes(graph)

def print_dfg_edges(graph):
  return streamhls_d.print_dfg_edges(graph)

def print_dfg_dot(dfg):
  return streamhls_d.print_dfg_as_dot(dfg)

def run_loop_permutation_minlp(graph, project_path, options):
  return streamhls_d.run_loop_permutation_minlp(graph, project_path, options)

def run_loop_permutation_sa(graph, options):
  return streamhls_d.run_loop_permutation_sa(graph, options)

def run_loop_permutation_ga(graph, options):
  return streamhls_d.run_loop_permutation_ga(graph, options)

def run_loop_unrolling_minlp(graph, project_path, options):
  return streamhls_d.run_loop_unrolling_minlp(graph, project_path, options)

def run_loop_unrolling_sa(graph, options):
  return streamhls_d.run_loop_unrolling_sa(graph, options)

def run_loop_unrolling_ga(graph, options):
  return streamhls_d.run_loop_unrolling_ga(graph, options)

def run_loop_combined_minlp(graph, project_path, options):
  return streamhls_d.run_loop_combined_minlp(graph, project_path, options)

def run_loop_combined_sa(graph, options):
  return streamhls_d.run_loop_combined_sa(graph, options)

def run_loop_combined_ga(graph, options):
  return streamhls_d.run_loop_combined_ga(graph, options)

def find_random_permutation_solution(graph, options):
  return streamhls_d.find_random_permutation_solution(graph, options)

def set_fixed_permutations(graph, options):
  return streamhls_d.set_fixed_permutations(graph, options)

def find_random_unrolling_solution(graph, options):
  return streamhls_d.find_random_unrolling_solution(graph, options)

def set_fixed_unrolling_factors(graph, options):
  return streamhls_d.set_fixed_unrolling_factors(graph, options)
def find_random_combined_solution(graph, options):
  return streamhls_d.find_random_combined_solution(graph, options)

def set_unrolling_factors_to_ones(graph):
  return streamhls_d.set_unrolling_factors_to_ones(graph)

def generate_system_module(graph, template_module):
  return streamhls_d.generate_system_module(graph, template_module)

def annotate_buffers(module, project_path):
  return streamhls_d.annotate_buffers(module, project_path)

def annotate_loops(module):
  return streamhls_d.annotate_loops(module)

def annotate_memrefs(module):
  return streamhls_d.annotate_memrefs(module)

def set_port_widths_to_one(module):
  return streamhls_d.set_port_widths_to_one(module)

def binarize_weights(module, weightBinsPath):
  return streamhls_d.binarize_weights(module, weightBinsPath)

def bufferize_func_args(module):
  return streamhls_d.bufferize_func_args(module)

def eliminate_single_write_multiple_reads(module, eliminationType="skip"):
  return streamhls_d.eliminate_single_write_multiple_reads(module, eliminationType)

def convert_memref_args_to_fifos(module):
  return streamhls_d.convert_memref_args_to_fifos(module)

def convert_func_memrefs_to_fifos_dfg(graph):
  return streamhls_d.convert_func_memrefs_to_fifos_dfg(graph)

def flatten_array_of_streams(module):
  return streamhls_d.flatten_array_of_streams(module)

def array_streams_to_vector_stream(module):
  return streamhls_d.array_streams_to_vector_stream(module)

def eleminate_unit_dims_and_iters(module):
  return streamhls_d.eleminate_unit_dims_and_iters(module)

def rewrite_memory_modules_to_dma(module):
  return streamhls_d.rewrite_memory_modules_to_dma(module)

def raise_inner_loop_by_one_level(module):
  return streamhls_d.raise_inner_loop_by_one_level(module)

def split_modules(modules_dict):
  compute_modules_to_be_merged_dict = {}
  single_memory_modules_list = []
  for key, (mods, tiling_v, kind) in modules_dict.items():
    # print(f"Module extracted: {key} with {len(mods)} kernels, tiling={tiling_v}, type={kind}")
    if kind == "compute":
      name = key.replace(".", "_")
      compute_modules_to_be_merged_dict[name] = (mods, tiling_v)
    elif kind == "memory": #i.e. memory modules, no merging
      if len(mods) != 1:
        raise ValueError(f"Expected single memory module for key {key}, but got {len(mods)} modules.")
      for i, mod in enumerate(mods):
        single_memory_modules_list.append((mod, tiling_v))

  return compute_modules_to_be_merged_dict, single_memory_modules_list

def merge_kernels_dfg(dfg, options):
  return streamhls_d.merge_kernels_dfg(dfg, options)

def convert_func_memrefs_to_fifos_dfg(dfg):
  return streamhls_d.convert_func_memrefs_to_fifos_dfg(dfg)

def merge_super_nodes_dfg(dfg):
  return streamhls_d.merge_super_nodes_dfg(dfg)

def wrap_super_nodes_dfg(dfg):
  return streamhls_d.wrap_super_nodes_dfg(dfg)

def reuse_hardware_dfg(dfg):
  return streamhls_d.reuse_hardware_dfg(dfg)

def convert_torch_to_mlir(model: nn.Module, inputs, output_type="linalg-on-tensors"):
    """
    Unified entry: try FX path first, then TorchScript fallback with clear error if both fail.
    """
    # Try FX
    try:
        return convert_torch_to_mlir_fx(model, inputs, output_type=output_type)
    except Exception as e_fx:
        logger.info("[torch-mlir] FX export failed; trying TorchScript fallback…")
        # Try TorchScript
        try:
            return convert_torch_to_mlir_torchscript(model, inputs, output_type=output_type)
        except Exception as e_ts:
            raise RuntimeError(
                "Both FX export and TorchScript compilation failed.\n"
                f"FX error: {e_fx}\n"
                f"TorchScript error: {e_ts}"
            )

def print_to_file(module, file_path: str):
    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    # If it's a Module object, convert to string
    if not isinstance(module, str):
        # We can use the C++ print_to_file_api if available or just str()
        # Looking at ph.print_to_file implementation:
        module_text = str(module)
    else:
        module_text = module

    with open(file_path, "w") as f:
        f.write(module_text)
