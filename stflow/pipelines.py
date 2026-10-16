import os
import sys
import logging
import yaml
import json
from typing import Optional, Union
import subprocess
import numpy as np
import torch
# import stflow  # Removed in favor of relative imports
import re
import time
import multiprocessing as mp
import traceback
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed

from .vitis_build import build_vitis, run_vitis_hls
from .tapa_build import build_tapa
from .main import (
    lower_linalg_to_affine,
    build_hls,
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
    print_to_file,
    merge_kernels_dfg,
    convert_func_memrefs_to_fifos_dfg,
    reuse_hardware_dfg,
    print_dfg_dot,
    print_dfg_nodes,
    print_dfg_edges,
    extract_modules,
    enable_llvm_debug_only,
    convert_torch_to_mlir,
)
from .host_tapa import generate_link_config
from .utils import invoke_mlir_parser, get_bitstream_report, reset_fpga
from .config import STFlowConfig, DSEConfig

# Configure logging
logger = logging.getLogger(__name__)

def ensure_config(config: Union[dict, STFlowConfig]) -> STFlowConfig:
    if isinstance(config, STFlowConfig):
        return config
    return STFlowConfig.from_dict(config or {})

def setup_logging(prj_name: str, console: bool = True, use_file: bool = True):
    """Configures logging. If use_file=True, writes to stflow.log in the project directory."""
    log_file = os.path.join(prj_name, "stflow.log")
    if use_file:
        os.makedirs(prj_name, exist_ok=True)
    
    # Create file handler (overwrite old logs for new runs)
    fh = None
    if use_file:
        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.INFO)
    
    # Create console handler (keeping it as requested only for Level 0/1)
    ch = None
    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if fh:
        fh.setFormatter(formatter)
    if ch:
        ch.setFormatter(formatter)
    
    # Clear existing handlers to avoid duplicates in case of repeated calls
    root_logger = logging.getLogger('stflow')
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    root_logger.setLevel(logging.INFO)
    if fh:
        root_logger.addHandler(fh)
    if ch:
        root_logger.addHandler(ch)
    
    # Disable propagation to prevent root logger from also handling messages.
    # This is critical when stdout/stderr are redirected to avoid duplicate entries.
    root_logger.propagate = False

def print_dfg_info(dfg, step, dfg_dir):
    """Helper to dump DFG nodes, edges, and dot files."""
    dfg_nodes = print_dfg_nodes(dfg)
    dfg_edges = print_dfg_edges(dfg)
    dfg_dot = print_dfg_dot(dfg)
    print_to_file(dfg_dot, os.path.join(dfg_dir, f"{step}.dot"))
    print_to_file(dfg_nodes, os.path.join(dfg_dir, f"{step}_nodes.log"))
    print_to_file(dfg_edges, os.path.join(dfg_dir, f"{step}_edges.log"))

# fix random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Makefile template for Vitis 2024.2 bitstream generation.
VITIS_MAKEFILE_TEMPLATE = """\
# Vitis 2024.2 Bitstream Makefile

PLATFORM := {platform}
TOP := {top_func_name}
SRC_DIR := vitis/src
KERNEL_SRC := $(SRC_DIR)/kernel.cpp

BUILD_DIR := build
XO := $(BUILD_DIR)/$(TOP).$(PLATFORM).xo
XCLBIN := $(BUILD_DIR)/$(TOP).$(PLATFORM).xclbin

VPP := v++
VPP_FLAGS := --platform $(PLATFORM) --save-temps --temp_dir $(BUILD_DIR) --log_dir $(BUILD_DIR) --report_dir $(BUILD_DIR) --kernel_frequency 300

# Default target
all: bitstream

# Compilation to .xo
$(XO): $(KERNEL_SRC)
	mkdir -p $(BUILD_DIR)
	$(VPP) $(VPP_FLAGS) -c -t $(TARGET) -k $(TOP) -I $(SRC_DIR) -o $@ $<

# Linking to .xclbin
$(XCLBIN): $(XO)
	mkdir -p $(BUILD_DIR)
	$(VPP) $(VPP_FLAGS) -l -t $(TARGET) --config link_config.ini -o $@ $<

# Targets
bitstream: TARGET := hw
bitstream: $(XCLBIN)

hw_emu: TARGET := hw_emu
hw_emu: $(XCLBIN)
"""

# SLURM batch script template for Vitis bitstream generation.
VITIS_BATCH_TEMPLATE = """\
#!/bin/bash
#SBATCH -J {prj_name}
#SBATCH -c4
#SBATCH --mem=32GB
#SBATCH -o %j.log
#SBATCH -q low
#SBATCH -p batch
#SBATCH --exclude=c04

# Define directories
ORIGINAL_DIR=$(pwd)
USER_NAME=$(whoami)
SCRATCH_DIR="/scratch/${{USER_NAME}}/${{SLURM_JOB_ID}}"

echo "Job ${{SLURM_JOB_ID}} started on $(hostname)"
echo "Original directory: ${{ORIGINAL_DIR}}"
echo "Scratch directory: ${{SCRATCH_DIR}}"

# 1. Create scratch directory
mkdir -p "${{SCRATCH_DIR}}"

# 2. Copy project files to scratch
echo "Copying files to scratch..."
cp -r ./* "${{SCRATCH_DIR}}/"

# 3. Enter scratch directory
cd "${{SCRATCH_DIR}}"

# 4. Load tools
module load xilinx/vivado/2024.2

# 5. Build and run
echo "Running synthesis and bitstream generation..."
make bitstream -j4

# 6. Copy results back
echo "Copying results back to original directory..."
mkdir -p "${{ORIGINAL_DIR}}/build"
cp -r build/* "${{ORIGINAL_DIR}}/build/"

# 7. Cleanup scratch
echo "Cleaning up scratch..."
rm -rf "${{SCRATCH_DIR}}"

echo "Job finished successfully."
"""


def flatten_nested_list_or_tuple(nested):
    """Flattens a nested list or tuple into a single list."""
    flat_list = []
    for item in nested:
        if isinstance(item, (list, tuple)):
            flat_list.extend(flatten_nested_list_or_tuple(item))
        else:
            flat_list.append(item)
    return flat_list


def generate_build_scripts(prj_name, top_func_name, build_options=None):

    # Generate Makefile
    # Build options control Vitis version and default make goal
    build_options = build_options or {}
    makefile_content = VITIS_MAKEFILE_TEMPLATE.format(
		platform="xilinx_u55c_gen3x16_xdma_3_202210_1",
		top_func_name=top_func_name,
	)
    print_to_file(makefile_content, f"{prj_name}/Makefile")
    
    # Generate batch script
    print_to_file(VITIS_BATCH_TEMPLATE.format(prj_name=prj_name), f"{prj_name}/gen_bitstream.batch")

    # Generate link_config.ini
    kernel_file = f"{prj_name}/vitis/src/kernel.cpp"
    if not os.path.exists(kernel_file):
        kernel_file = f"{prj_name}/tapa/src/kernel.cpp"
    if not os.path.exists(kernel_file):
        kernel_file = f"{prj_name}/kernel.cpp"
        
    if os.path.exists(kernel_file):
        with open(kernel_file, "r") as kf:
            kernel_code = kf.read()
        link_config = generate_link_config(top_func_name, kernel_code)
        print_to_file(link_config, f"{prj_name}/link_config.ini")
    else:
        logger.warning(f"Could not find kernel.cpp in {prj_name} for link_config.ini generation.")

def pre_hldse_pipeline(
        prj_name: str, 
        module, 
        test_config: STFlowConfig = None,
    ):
    """
    Runs the ST-Flow-based pipeline with minimal changes.
    Keeps intermediate files for debugging.
    """
    intermediate_dir = f"{prj_name}/01_pre_hldse_modules"
    os.makedirs(intermediate_dir, exist_ok=True)

    # single-write-multiple-reads elimination: aggressive | conservative (for GPT2) | inplace 
    swmr_elimination = test_config.swmr_elimination
    enable_pre_bufferize_elimination = False if swmr_elimination == 'inplace' else True

    print_to_file(str(module), f"{intermediate_dir}/01_input.mlir")
    # 1) Lower linalg → affine + bufferization
    mod = lower_linalg_to_affine(module)
    print_to_file(str(mod), f"{intermediate_dir}/02_affine.mlir")

    # 2) Write weights (bin) and make them function args
    mod = binarize_weights(mod, f"{prj_name}/weight_bins")
    print_to_file(str(mod), f"{intermediate_dir}/03_weights_binarized.mlir")

    if enable_pre_bufferize_elimination:
        # 4) Eliminate single write multiple reads
        mod = eliminate_single_write_multiple_reads(mod, eliminationType=swmr_elimination)
        print_to_file(str(mod), f"{intermediate_dir}/04_pre_bufferized_{swmr_elimination}_eliminated_swmr.mlir")
    
    # 3) Bufferize function args (generate off-chip reads/writes)
    mod = bufferize_func_args(mod)
    print_to_file(str(mod), f"{intermediate_dir}/05_bufferized.mlir")

    # 4) Eliminate single write multiple reads
    mod = eliminate_single_write_multiple_reads(mod, eliminationType="inplace")
    print_to_file(str(mod), f"{intermediate_dir}/06_eliminated_swmr.mlir")
    
    # 5) Annotate buffers + loops for later passes
    mod = annotate_buffers(mod, prj_name)
    mod = annotate_loops(mod)
    print_to_file(str(mod), f"{intermediate_dir}/07_annotated.mlir")

    return mod

def run_dse_flow(
    dfg, 
    prj_name: str, 
    test_config: STFlowConfig, 
    dse_opts, 
    enable_tiling: bool, 
    dfg_dir: str
):
    """Encapsulates the design space exploration flow."""
    dse_opts_raw = test_config.dse_options
    num_n_runs = dse_opts_raw.num_n_runs

    # --- 1) Internal Helpers ---

    def print_best_stats(label, res):
        """Prints a formatted summary of the best solution found."""
        if not res or not res.success:
            logger.warning("[%s] %s failed to find a valid solution.", prj_name, label)
            return
            
        headers = ["Stage", "Latency", "Memory", "DSPs", "Cost"]
        row = [label, str(res.latency), str(res.memory), str(res.dsps), str(res.cost)]
        col_widths = [max(len(h), len(r)) + 2 for h, r in zip(headers, row)]
        
        sep = "+" + "+".join(["-" * w for w in col_widths]) + "+"
        header_row = "|" + "|".join([f" {h:<{w-2}} " for h, w in zip(headers, col_widths)]) + "|"
        value_row = "|" + "|".join([f" {r:<{w-2}} " for r, w in zip(row, col_widths)]) + "|"
        
        logger.info("[%s] BEST %s Solution found:", prj_name, label)
        logger.info(sep)
        logger.info(header_row)
        logger.info(sep)
        logger.info(value_row)
        logger.info(sep)

    def _run_solver_stage(stage, dfg_obj, run_idx):
        """Unified helper to execute a specific DSE stage with the configured solver."""
        if stage == 'permutation':
            solver = dse_opts.permutation_dse
            sa_cfg = dse_opts.permutation_sa_config
            ga_cfg = dse_opts.permutation_ga_config
            run_funcs = {
                'sa': lambda: run_loop_permutation_sa(dfg_obj, dse_opts),
                'ga': lambda: run_loop_permutation_ga(dfg_obj, dse_opts),
                'random': lambda: find_random_permutation_solution(dfg_obj, dse_opts),
                'minlp': lambda: run_loop_permutation_minlp(dfg_obj, prj_name, dse_opts)
            }
            update_func = dfg_obj.update_permutation_solution
        elif stage == 'unrolling':
            solver = dse_opts.unrolling_dse
            sa_cfg = dse_opts.unrolling_sa_config
            ga_cfg = dse_opts.unrolling_ga_config
            run_funcs = {
                'sa': lambda: run_loop_unrolling_sa(dfg_obj, dse_opts),
                'ga': lambda: run_loop_unrolling_ga(dfg_obj, dse_opts),
                'random': lambda: find_random_unrolling_solution(dfg_obj, dse_opts),
                'minlp': lambda: run_loop_unrolling_minlp(dfg_obj, prj_name, dse_opts)
            }
            update_func = dfg_obj.update_unrolling_solution
        elif stage == 'combined':
            solver = dse_opts.combined_dse
            sa_cfg = dse_opts.combined_sa_config
            ga_cfg = dse_opts.combined_ga_config
            run_funcs = {
                'sa': lambda: run_loop_combined_sa(dfg_obj, dse_opts),
                'ga': lambda: run_loop_combined_ga(dfg_obj, dse_opts),
                'random': lambda: find_random_combined_solution(dfg_obj, dse_opts),
                'minlp': lambda: run_loop_combined_minlp(dfg_obj, prj_name, dse_opts)
            }
            # Combined update affects both
            def update_func(res):
                dfg_obj.update_permutation_solution(res)
                dfg_obj.update_unrolling_solution(res)
        else:
            raise ValueError(f"Unknown DSE stage: {stage}")

        # Handle 'fixed' solver separately
        if solver == 'fixed':
            if stage == 'permutation':
                dse_opts.fixed_permutation_solution = test_config.dse_options.fixed_permutation_solution
                if not set_fixed_permutations(dfg_obj, dse_opts):
                    raise RuntimeError(f"[{prj_name}] Failed to set fixed permutations")
            elif stage == 'unrolling':
                dse_opts.fixed_unrolling_solution = test_config.dse_options.fixed_unrolling_solution
                if not set_fixed_unrolling_factors(dfg_obj, dse_opts):
                    raise RuntimeError(f"[{prj_name}] Failed to set fixed unrolling factors")
            return None

        # Execute selected solver
        if solver not in run_funcs:
            # Default to MINLP/AMPL for unrecognized solvers
            res = run_funcs['minlp']()
        else:
            # Set CSV path for SA/GA before running
            if solver == 'sa':
                sa_cfg.csv_path = os.path.join(prj_name, "dse", f"sa_{stage}_{run_idx + 1}.csv")
            elif solver == 'ga':
                ga_cfg.csv_path = os.path.join(prj_name, "dse", f"ga_{stage}_{run_idx + 1}.csv")
            
            res = run_funcs[solver]()
        logger.info("[%s] %s solution found: %s", prj_name, stage, res)
        if res and res.success and res.cost != -1:
            update_func(res)
        elif solver != 'fixed':
            logger.error("[%s] Failed to find %s solution using %s", prj_name, stage, solver)
            # Export failure report before crashing
            export_stflow_report(prj_name, res)
            raise RuntimeError(f"[{prj_name}] Failed to find {stage} solution using {solver}")
        
        return res
    
    # --- 2) Main DSE Flow Preparation ---

    is_combined = test_config.enable_combined_dse
    enable_permutation = test_config.enable_permutation
    enable_unrolling = enable_tiling  # Use existing convention

    # Final bypass check for decoupled flow
    if not is_combined and not enable_permutation and not enable_unrolling:
        logger.info("[%s] Both permutation and unrolling are disabled. Skipping DSE loop.", prj_name)
        set_unrolling_factors_to_ones(dfg)
        return

    # --- 3) Execution ---

    if is_combined:
        # --- Combined DSE Flow ---
        logger.info("[%s] Running Loop Combined %s DSE (n_runs=%d)...", 
                    prj_name, dse_opts.combined_dse.upper(), num_n_runs)
        
        best_res = None
        for i in range(num_n_runs):
            if num_n_runs > 1:
                logger.info("[%s] Combined Run %d/%d...", prj_name, i + 1, num_n_runs)
                
            res = _run_solver_stage('combined', dfg, i)
            
            if res and res.success and (best_res is None or res.cost < best_res.cost):
                best_res = res
            
            export_stflow_report(
                prj_name, res, 
                filename=f"{dse_opts_raw.combined_dse}_combined_{i+1}_final.csv",
                subdirectory="dse"
            )

        if best_res and best_res.success:
            print_best_stats("Combined", best_res)
            # Ensure DFG reflects best solution
            dfg.update_permutation_solution(best_res)
            dfg.update_unrolling_solution(best_res)
        
        print_dfg_info(dfg, "02_post_combined", dfg_dir)
        export_stflow_report(prj_name, best_res)

    else:
        # --- Decoupled DSE Flow (Permutation + Unrolling Pairs) ---
        logger.info("[%s] Running Decoupled DSE (n_runs=%d)...", prj_name, num_n_runs)
        
        # Capture initial state for fresh resets in each run
        initial_p_map = dfg.get_permutation_solution()
        initial_t_map = dfg.get_tiling_solution()

        best_overall_res = None # Refers to unroll res if enabled, else permutation res
        best_results_pair = (None, None) # (res_p, res_u)
        best_pair_p_map = initial_p_map
        best_pair_t_map = initial_t_map
        best_cost = float('inf')

        for i in range(num_n_runs):
            if num_n_runs > 1:
                logger.info("[%s] Decoupled Run %d/%d...", prj_name, i + 1, num_n_runs)
                
            # Reset DFG to start each pair fresh
            dfg.update_permutation_solution(initial_p_map)
            dfg.update_unrolling_solution(initial_t_map)

            # Stage 1: Permutation
            res_p = None
            if enable_permutation:
                res_p = _run_solver_stage('permutation', dfg, i)
                if res_p:
                    export_stflow_report(
                        prj_name, res_p,
                        filename=f"{dse_opts_raw.permutation_dse}_perm_{i+1}_final.csv",
                        subdirectory="dse"
                    )

            # Stage 2: Unrolling
            res_u = None
            if enable_unrolling:
                res_u = _run_solver_stage('unrolling', dfg, i)
                if res_u:
                    export_stflow_report(
                        prj_name, res_u,
                        filename=f"{dse_opts_raw.unrolling_dse}_unroll_{i+1}_final.csv",
                        subdirectory="dse"
                    )
            else:
                set_unrolling_factors_to_ones(dfg)

            # --- Evaluate and Capture Pair based on "Final Solution" logic ---
            # Final solution refers to unroll solution if enable_unrolling is True, 
            # or permutation solution if enable_unrolling is False.
            current_target_res = res_u if enable_unrolling else res_p
            current_cost = float('inf')
            
            if current_target_res and current_target_res.success and current_target_res.cost != -1:
                current_cost = current_target_res.cost
            
            if current_cost != float('inf') and current_cost < best_cost:
                best_cost = current_cost
                best_overall_res = current_target_res
                best_results_pair = (res_p, res_u)
                best_pair_p_map = dfg.get_permutation_solution()
                best_pair_t_map = dfg.get_tiling_solution()
        
        # Final atomic update with the best found pair
        logger.info("[%s] Finalizing DFG with the best discovered solution pair (cost: %d)", prj_name, best_cost)
        dfg.update_permutation_solution(best_pair_p_map)
        dfg.update_unrolling_solution(best_pair_t_map)

        # Log individual members of the best pair if they exist
        best_p, best_u = best_results_pair
        if best_p:
            print_best_stats("Best Permutation Pair Member", best_p)
            print_dfg_info(dfg, "02_post_perm", dfg_dir)

        if best_u:
            print_best_stats("Best Unrolling Pair Member", best_u)
            print_dfg_info(dfg, "03_post_unroll", dfg_dir)
        elif not enable_unrolling and enable_permutation:
            # If only permutation was active, the "final" state is post-permutation
            print_dfg_info(dfg, "02_post_perm", dfg_dir)
        
        # Export the definitive overall result report
        export_stflow_report(prj_name, best_overall_res)
        return best_overall_res

def export_stflow_report(prj_name, res, filename="stflow.csv", subdirectory="reports", csim_success=None):
    """Saves DSE results to prj_name/subdirectory/filename."""
    target_dir = os.path.join(prj_name, subdirectory)
    os.makedirs(target_dir, exist_ok=True)
    csv_path = os.path.join(target_dir, filename)

    # --- 1) Load Existing Data if available ---
    existing_data = {}
    if os.path.exists(csv_path):
        try:
            import csv
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_data = row
                    break
        except Exception:
            pass

    # --- 2) Determine new values, falling back to existing ones if res is None ---
    if res is None:
        # If we have existing data, keep success and metrics
        cost = existing_data.get('cost', -1)
        latency = existing_data.get('latency', -1)
        memory = existing_data.get('memory', -1)
        dsps = existing_data.get('dsps', -1)
        success = existing_data.get('success', 'False').lower() == 'true'
    else:
        cost = getattr(res, 'cost', -1)
        latency = getattr(res, 'latency', -1)
        memory = getattr(res, 'memory', -1)
        dsps = getattr(res, 'dsps', -1)
        success = getattr(res, 'success', False)
    
    # Determine CSIM status
    if csim_success is not None:
        csim_str = str(csim_success)
    else:
        # Keep existing CSIM status if it was already True
        csim_str = existing_data.get('csim_success', 'N/A')

    # --- 3) Write Report ---
    try:
        with open(csv_path, "w") as f:
            f.write("cost,latency,memory,dsps,success,csim_success\n")
            f.write(f"{cost},{latency},{memory},{dsps},{success},{csim_str}\n")
        logger.info("[%s] ST-Flow DSE report written to %s", prj_name, csv_path)
    except Exception as e:
        logger.error("[%s] Failed to write ST-Flow DSE report: %s", prj_name, str(e))


def dfg_based_pipeline(
    prj_name: str, 
    module, 
    enable_tiling: bool = True,
    test_config: STFlowConfig = None
):
    # create dfg folder
    dfg_dir = f"{prj_name}/02_dfg"
    os.makedirs(dfg_dir, exist_ok=True)
    
    final_top_path = f"{prj_name}/03_top_modules/06_final_top.mlir"
    host_layout_transforms_path = f"{prj_name}/03_top_modules/host_layout_transforms.mlir"

    kernel_merging_options = {

        "project_path": prj_name,
        "DSP_avail": 1024,
        "memory_avail": 1024 * 1024,
        "bw_avail": 64,
        "display_vars": True,
        "display_assignments": True,
        "memory_alpha": 0.0001,
        "fixed_tiling": False,
        "full_data_reuse": test_config.full_data_reuse
    }
    
    # 6) High-Level DSE
    logger.info("[%s] Running High-Level DSE with hw_mapping_strategy=%s...", prj_name, test_config.hw_mapping_strategy)
    dse_opts = test_config.dse_options.to_cpp()
    hw_mapping = run_high_level_dse(module, prj_name, dse_opts, hw_mapping_strategy=test_config.hw_mapping_strategy)


    # 7) Extract Merged DFG
    logger.info("Extracting merged DFG...")
    dfg = extract_merged_dfg(module, hw_mapping, enable_tiling)
    
    if(not dfg.populate_node_info(True)):
        raise RuntimeError(f"[{prj_name}] Failed to populate node info")

    print_dfg_info(dfg, "01_hw", dfg_dir)
    
    dse_opts_raw = test_config.dse_options
    dse_opts = dse_opts_raw.to_cpp()

    # 7) Optimization Flows
    best_res = run_dse_flow(dfg, prj_name, test_config, dse_opts, enable_tiling, dfg_dir)

    # 8) High-Level DSE
    logger.info("Running Kernel Merging based on merged DFG...")
    merge_kernels_dfg(dfg, kernel_merging_options)

    print_dfg_info(dfg, "04_merged", dfg_dir)
    
    # 9) Convert Func Memrefs to FIFOs via DFG
    logger.info("Converting memrefs to FIFOs via DFG...")
    convert_func_memrefs_to_fifos_dfg(dfg)

    print_dfg_info(dfg, "05_fifos", dfg_dir)
    
    # 10) Reuse Hardware via DFG
    logger.info("Applying hardware reuse via DFG...")
    reuse_hardware_dfg(dfg)


    # 13) Print DFG as DOT for visualization
    top_modules_dir = f"{prj_name}/03_top_modules"
    os.makedirs(top_modules_dir, exist_ok=True)

    top_module = generate_system_module(dfg, module)
    print_to_file(str(top_module), f"{top_modules_dir}/01_top.mlir")

    # # FIXME: Has an issue
    # top_module = eleminate_unit_dims_and_iters(top_module)
    # print_to_file(str(top_module), f"{top_modules_dir}/02_top_no_unit_dims_iters.mlir")


    # run_vitis_hls_synth = test_config.run_vitis_hls_synth
    # if run_vitis_hls_synth:
    #     logger.info("[%s] Running Pre-DMA Vitis HLS synthesis...", prj_name)
    #     run_vitis_hls(
    #         top_module, 
    #         invoke_mlir_parser("module {}"), 
    #         top_func_name="module_top", 
    #         project=prj_name,
    #         dir_name="vitis_pre_dma",
    #         use_hls_task=True
    #     )

    if test_config.enable_dma or test_config.target_tool == 'tapa':
        top_module = flatten_array_of_streams(top_module)
        print_to_file(str(top_module), f"{top_modules_dir}/03_top_flattened.mlir")

        top_module = array_streams_to_vector_stream(top_module)
        print_to_file(str(top_module), f"{top_modules_dir}/04_top_vectorized.mlir")

    # 12) Rewrite memory modules to use DMA
    if test_config.enable_dma:
        top_module, host_layout_transforms = rewrite_memory_modules_to_dma(top_module)
        print_to_file(str(top_module), f"{top_modules_dir}/05_top_dma.mlir")
    else:
        host_layout_transforms = invoke_mlir_parser("module {}")
    print_to_file(str(top_module), final_top_path)
    print_to_file(str(host_layout_transforms), host_layout_transforms_path)

    return top_module, host_layout_transforms, best_res

def e2e_pipeline(
    prj_name: str, 
    module, 
    inputs=None, 
    golden_output=None,
    enable_tiling: bool = True,
    test_config: STFlowConfig = None
):
    """
    Runs the ST-Flow + Stream-HLS pipeline with hardware reuse.
    """

    # Ensure build_options is a dict
    compile_only = test_config.compile_only
    run_bitstream = test_config.run_bitstream
    run_csim = test_config.csim
    run_vitis_hls_synth = test_config.run_vitis_hls_synth
    run_vitis_hls_cosim = test_config.run_vitis_hls_cosim

    # Step 1: Pre-HLDSE pipeline
    if not logging.getLogger('stflow').hasHandlers():
        setup_logging(prj_name)

    logger.info("[%s] Running pre-HLDSE pipeline...", prj_name)
    module = pre_hldse_pipeline(
        prj_name, 
        module, 
        test_config=test_config
    )

    # Step 2: Post-HLDSE pipeline
    final_top_path = f"{prj_name}/03_top_modules/06_final_top.mlir"
    host_layout_transforms_path = f"{prj_name}/03_top_modules/host_layout_transforms.mlir"
    stflow_report_path = os.path.join(prj_name, "reports", "stflow.csv")

    if compile_only:
        logger.info("[%s] Compile only mode: loading existing final top module from %s", prj_name, final_top_path)
            
        with open(final_top_path, "r") as f:
            top_module_ir = f.read()
        top_module = invoke_mlir_parser(top_module_ir)
        
        if os.path.exists(host_layout_transforms_path):
            logger.info("[%s] Loading existing host layout transforms from %s", prj_name, host_layout_transforms_path)
            with open(host_layout_transforms_path, "r") as f:
                host_layout_transforms_ir = f.read()
            host_layout_transforms = invoke_mlir_parser(host_layout_transforms_ir)
        else:
            host_layout_transforms = invoke_mlir_parser("module {}")
    else:
        logger.info("[%s] Running post-HLDSE pipeline...", prj_name)
        top_module, host_layout_transforms, best_res = dfg_based_pipeline(
            prj_name, 
            module, 
            enable_tiling=enable_tiling, 
            test_config=test_config
        )

    # Step 3: Build the project with the selected tool
    target_tool = test_config.target_tool
    if target_tool == 'tapa':
        logger.info("[%s] Building TAPA host callable...", prj_name)
        host_callable = build_tapa(
            top_mod=top_module,
            utils_mod=host_layout_transforms,
            top_func_name="module_top",
            project=prj_name,
            compile_only=compile_only,
            enable_dma=test_config.enable_dma,
        )
    elif target_tool == 'vitis':
        from .emitters.vitis import emit_vitis_sources
        if not test_config.compile_only:
            logger.info("[%s] Generating Vitis traditional design (for bitstream/synth)...", prj_name)
            emit_vitis_sources(
                top_mod=top_module,
                utils_mod=host_layout_transforms,
                top_func_name="module_top",
                project_dir=prj_name,
                dir_name="vitis",
                use_hls_task=False
            )
        logger.info("[%s] Building Vitis HLS task-based host callable (for csim)...", prj_name)
        host_callable = build_vitis(
            top_mod=top_module,
            utils_mod=host_layout_transforms,
            top_func_name="module_top",
            project=prj_name,
            compile_only=compile_only,
            use_hls_task=True,
            dir_name="vitis_csim",
            enable_dma=test_config.enable_dma,
        )
    else:
        raise ValueError(f"Unsupported target tool: {target_tool}")

    generate_build_scripts(
        prj_name, "module_top", build_options=test_config
    )

    if run_csim:

        logger.info("[%s] Host built successfully. Callable: %s", prj_name, host_callable)

        # Run CSIM or Bitstream
        if run_bitstream:
            bitstream_default_path = f"{prj_name}/build/module_top.xilinx_u55c_gen3x16_xdma_3_202210_1.xclbin"
            if os.path.exists(bitstream_default_path):
                bitstream = bitstream_default_path
                logger.info("[%s] INFO: Bitstream found at %s. Running FPGA execution...", prj_name, bitstream)
                # generate bitstream report into prj/reports/bitstream.csv
                # reset_fpga()
                get_bitstream_report(prj_name)
            else:
                # generate bitstream report into prj/reports/bitstream.csv (fallback logic)
                get_bitstream_report(prj_name)
                logger.error("[%s] ERROR: No bitstream found at %s. Cannot run FPGA execution.", prj_name, bitstream_default_path)
                raise FileNotFoundError(f"No bitstream found at {bitstream_default_path}.")
        else:
            logger.info("[%s] INFO: Running CSIM execution...", prj_name)
            bitstream = ""
        
        csim_passed = False
        try:
            run_host_callable(host_callable, inputs, golden_output, test_config.atol, bitstream=bitstream)
            csim_passed = True
        except Exception as e:
            logger.error("[%s] CSIM execution failed: %s", prj_name, str(e))
            csim_passed = False
            # Re-raise to ensure the job status reflects failure if needed, 
            # though we want to record the report first.
            # actually, let's not re-raise yet so we can finish other tasks if any
        
        # Update the report with CSIM status
        res_to_report = best_res if 'best_res' in locals() else None
        export_stflow_report(prj_name, res_to_report, csim_success=csim_passed)

        if bitstream != '':
            get_bitstream_report(prj_name)
        
        if not csim_passed:
            raise RuntimeError(f"CSIM failed for project {prj_name}")


    if run_vitis_hls_synth or run_vitis_hls_cosim:
        logger.info("[%s] Running Post DMA Vitis HLS synthesis/cosim...", prj_name)
        run_vitis_hls(
            top_module,
            host_layout_transforms,
            "module_top",
            prj_name,
            dir_name="vitis_post_dma",
            use_hls_task=False,
            run_cosim=run_vitis_hls_cosim
        )


# -------------------------
# Entry Point
# -------------------------

@contextmanager
def redirect_stdout_stderr_to_file(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Saved original sys streams
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    # Use explicit OS-level file descriptors 1 and 2 to ensure we catch 
    # C++ output (e.g., llvm::dbgs()) which may not be aware of sys.stdout wrappers.
    stdout_fd = 1
    stderr_fd = 2

    saved_stdout = os.dup(stdout_fd)
    saved_stderr = os.dup(stderr_fd)
    
    # Flush Python buffers before redirection to avoid ordered issues
    sys.stdout.flush()
    sys.stderr.flush()

    # Use 'w' to overwrite the log file for new runs
    with open(path, "w") as f:
        # Dual redirection: 
        # 1. OS level for C++
        os.dup2(f.fileno(), stdout_fd)
        os.dup2(f.fileno(), stderr_fd)
        
        # 2. Python level for logging/print
        sys.stdout = f
        sys.stderr = f

        try:
            yield
        finally:
            # Flush again before restoring to ensure all content is written to the file
            sys.stdout.flush()
            sys.stderr.flush()

            # Restore Python streams
            sys.stdout = old_stdout
            sys.stderr = old_stderr

            # Restore OS FDs
            os.dup2(saved_stdout, stdout_fd)
            os.dup2(saved_stderr, stderr_fd)
            os.close(saved_stdout)
            os.close(saved_stderr)

def generate_hardware(
    prj_name: str,
    model,
    inputs,
    test_config: Union[dict, STFlowConfig] = None
):
    test_config = ensure_config(test_config)
    # append solver types to prj_name
    if test_config.enable_combined_dse:
        prj_name += f"_comb_{test_config.dse_options.combined_dse}"
    else:
        prj_name += f"_{test_config.dse_options.permutation_dse}_{test_config.dse_options.unrolling_dse}"
    
    prj_name += f"_{test_config.hw_mapping_strategy}"
    prj_name = f'{test_config.dse_options.DSP_limit}_{prj_name}'
    curr_prj_name = f"{prj_name}.prj"
    log_path = os.path.join(curr_prj_name, "stflow.log")

    os.makedirs(curr_prj_name, exist_ok=True)
    # dump config to file
    with open(os.path.join(curr_prj_name, "config.yaml"), "w") as f:
        yaml.dump(test_config.dict(), f, indent=2)

    with open(os.path.join(curr_prj_name, "config.json"), "w") as f:
        json.dump(test_config.dict(), f, indent=2)

    with redirect_stdout_stderr_to_file(log_path):
        # We redirect EVERYTHING to stdout first, then logging uses stdout.
        # This ensures Python and C++ output follow the same buffer/stream.
        setup_logging(curr_prj_name, console=True, use_file=False)
        
        logger.info("[generate_hardware] starting project: %s", curr_prj_name)
        enable_llvm_debug_only(test_config.debug_types)

        with torch.no_grad():
            out = model(*inputs)
            if not isinstance(out, tuple):
                out_golden = (out.detach().numpy(),)
            else:
                out_golden = tuple(o.detach().numpy() for o in out)

        mlir_text = convert_torch_to_mlir(
            model, inputs, output_type="linalg-on-tensors"
        )
        module = invoke_mlir_parser(mlir_text)

        e2e_pipeline(
            curr_prj_name,
            module,
            inputs=inputs,
            golden_output=out_golden,
            enable_tiling=test_config.enable_unrolling,
            test_config=test_config
        )

        logger.info("[generate_hardware] done")



def run_host_callable(host_callable, inputs, golden_output, atol, bitstream=""):
    logger.info("Running host callable with the following inputs and outputs:")
    for i, input in enumerate(inputs):
        logger.info("Input %d: type=%s shape=%s", i, type(input), input.shape)
    
    # Golden output is a list of Tensors/Arrays
    for i, golden in enumerate(golden_output):
        logger.info("Output %d: type=%s shape=%s", i, type(golden), golden.shape)

    # Convert everything to standard list of arguments for the callable
    # Note: the callable handles its own mapping
    args = []
    for input in inputs:
        args.append(input)
    
    # Prepare output containers
    output_containers = []
    for golden in golden_output:
        # Clone to avoid corruption
        if hasattr(golden, 'numpy'): # torch tensor
            output_containers.append(golden.detach().cpu().numpy().copy())
        else:
            output_containers.append(golden.copy())
    
    for i, arg in enumerate(args + output_containers):
        logger.info("  - Arg %d: type=%s shape=%s", i, type(arg), arg.shape)

    logger.info("Executing host binary...")
    start_time = time.time()
    
    try:
        # INVOKE
        host_callable(*(args + output_containers), bitstream=bitstream)
    finally:
        # Release FPGA device if applicable
        if bitstream:
            host_callable.release()
    
    end_time = time.time()
    logger.info("Host simulation total finished in %.4fs", end_time - start_time)

    # Verify results
    for i, (out, golden) in enumerate(zip(output_containers, golden_output)):
        if hasattr(golden, 'numpy'):
            golden_np = golden.detach().cpu().numpy()
        else:
            golden_np = golden
        
        logger.info("Kernel Output (first 5): %s", out.flatten()[:5])
        logger.info("Golden (first 5): %s", golden_np.flatten()[:5])
        
        if not np.allclose(out, golden_np, atol=atol):
            logger.error("Pipeline correctness test failed for output %d!", i)
            # Find the first mismatch
            mismatch_idx = np.where(~np.isclose(out, golden_np, atol=atol))
            first_idx = tuple(idx[0] for idx in mismatch_idx)
            logger.error("Mismatch at %s: kernel=%f, golden=%f", first_idx, out[first_idx], golden_np[first_idx])
            raise RuntimeError(f"Pipeline correctness test failed for output {i}!")
        else:
            logger.info("✅ Pipeline correctness test for output %d passed!", i)
