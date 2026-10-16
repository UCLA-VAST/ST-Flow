import os
import subprocess
import logging
import re
from typing import Optional
from ._mlir.ir import Module
from .emitters.vitis import emit_vitis_sources
from .builders.vitis import VitisBuilder, VitisCallable, parse_vitis_hls_report

logger = logging.getLogger(__name__)

HLS_TCL_SCRIPT = """\
open_project {prj_name}.vitis
set_top {top_func_name}
add_files src/kernel.cpp -cflags "-DALLOW_EMPTY_HLS_STREAM_READS" -csimflags "-DALLOW_EMPTY_HLS_STREAM_READS"
add_files src/kernel.h
add_files -tb src/kernel_tb.cpp
open_solution "solution1" -flow_target vitis
set_part {part}
create_clock -period 3.33 -name default_clock
config_compile -unsafe_math_optimizations
csynth_design
{extra_commands}
exit
"""


def parse_vitis_hls_cosim_report(rpt_path):
    """
    Parses module_top_cosim.rpt to extract latency.
    Format:
    |   Verilog|      Pass|         786529|         786529|         786529|             NA|             NA|             NA|                786529|
    """
    if not os.path.exists(rpt_path):
        return None, None
    
    with open(rpt_path, "r") as f:
        for line in f:
            if "Verilog" in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                # parts[0]: Verilog, parts[1]: Status, parts[2]: min latency, etc.
                try:
                    status = parts[1]
                    latency = int(parts[2])
                    return latency, status
                except (ValueError, IndexError):
                    continue
    return None, None


def report_vitis_hls(prj_name, dir_name="vitis"):
  report_path = f"{prj_name}/{dir_name}/{prj_name}.vitis/solution1/syn/report/module_top_csynth.xml"
  cosim_rpt_path = f"{prj_name}/{dir_name}/{prj_name}.vitis/solution1/sim/report/module_top_cosim.rpt"
  
  report_metrics = {}
  if os.path.exists(report_path):
      report_metrics = parse_vitis_hls_report(report_path)
      
      # Try to add cosim latency and status if available
      cosim_latency, cosim_status = parse_vitis_hls_cosim_report(cosim_rpt_path)
      if cosim_latency is not None:
          report_metrics["cosim_latency"] = cosim_latency
          report_metrics["cosim_status"] = cosim_status

      # Format combined resource strings
      dsps_str = f"{report_metrics['dsps']} ({report_metrics.get('dsps_pct', 'N/A')})"
      brams_str = f"{report_metrics['brams_18k']} ({report_metrics.get('brams_pct', 'N/A')})"
      urams_str = f"{report_metrics['urams']} ({report_metrics.get('urams_pct', 'N/A')})"
      
      headers = ["Min Latency", "Max Latency"]
      values = [str(report_metrics.get("min_cycles", "N/A")), str(report_metrics.get("max_cycles", "N/A"))]
      
      if "cosim_latency" in report_metrics:
          headers.append("Cosim Latency")
          values.append(str(report_metrics["cosim_latency"]))
          headers.append("Cosim Status")
          values.append(str(report_metrics["cosim_status"]))
      
      headers.extend(["DSPs", "BRAMs (18K)", "URAMs", "Est. CP (ns)"])
      values.extend([dsps_str, brams_str, urams_str, str(report_metrics.get("estimated_clock_period_ns", "N/A"))])

      logger.info(f"[{prj_name}] Vitis HLS Report Metrics:")
      col_widths = [max(len(h), len(v)) + 2 for h, v in zip(headers, values)]
      
      sep = "+" + "+".join(["-" * w for w in col_widths]) + "+"
      header_row = "|" + "|".join([f" {h:<{w-2}} " for h, w in zip(headers, col_widths)]) + "|"
      value_row = "|" + "|".join([f" {v:<{w-2}} " for v, w in zip(values, col_widths)]) + "|"
      
      logger.info(sep)
      logger.info(header_row)
      logger.info(sep)
      logger.info(value_row)
      logger.info(sep)

      # print to csv file directly in reports/ folder
      reports_dir = os.path.join(prj_name, "reports")
      os.makedirs(reports_dir, exist_ok=True)
      csv_path = os.path.join(reports_dir, f"{dir_name}.csv")
      
      try:
          with open(csv_path, "w") as f:
              # print header
              f.write(",".join(report_metrics.keys()) + "\n")
              # print values
              f.write(",".join(str(v) for v in report_metrics.values()) + "\n")
          logger.info(f"[{prj_name}] Vitis HLS report metrics written to {csv_path}")
      except Exception as e:
          logger.error(f"[{prj_name}] Failed to write Vitis HLS report: {str(e)}")

  else:
      logger.warning(f"[{prj_name}] WARNING: Vitis HLS report not found at {report_path}")

def _get_vitis_version():
    """
    Determines the Vitis version using the XILINX_HLS environment variable.
    Raises RuntimeError if XILINX_HLS is not set.
    """
    xilinx_hls = os.environ.get("XILINX_HLS")
    if not xilinx_hls:
        raise RuntimeError("XILINX_HLS environment variable not set. Please source the Vitis environment.")
    
    # Extract version from path, e.g., /opt/xilinx/Vitis_HLS/2025.1
    # Look for 4 digits followed by a dot and 1 or more digits
    match = re.search(r"(\d{4}\.\d+)", xilinx_hls)
    if match:
        return float(match.group(1))
    
    # Fallback to a default if parsing fails but variable is set
    return 2024.1

def run_vitis_hls_synthesis(prj_name, dir_name="vitis"):
    logger.info(f"[{prj_name}] Running Vitis HLS (Synthesis/Cosim) in {dir_name}...")
    
    version = _get_vitis_version()
    if version >= 2025.1:
        vitis_cmd = "vitis-run --tcl hls_script.tcl"
    else:
        vitis_cmd = "vitis_hls hls_script.tcl"

    cmd = f"cd {prj_name}/{dir_name} && {vitis_cmd} && cd -"
    # run using process
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"[{prj_name}] Vitis HLS execution failed!")
        logger.error("STDOUT: %s", result.stdout)
        logger.error("STDERR: %s", result.stderr)
        report_vitis_hls(prj_name, dir_name)
        raise RuntimeError(f"Vitis HLS execution failed for project {prj_name}")
    logger.info(f"[{prj_name}] Vitis HLS execution completed.")


def run_vitis_hls(
    top_mod: Module,
    utils_mod: Module,
    top_func_name: str,
    project: str,
    dir_name: str = "vitis",
    use_hls_task: bool = False,
    run_cosim: bool = False
):
    kernel_path, host_path, arg_infos = emit_vitis_sources(
        top_mod=top_mod,
        utils_mod=utils_mod,
        top_func_name=top_func_name,
        project_dir=project,
        dir_name=dir_name,
        use_hls_task=use_hls_task
    )
    
    # Synthesis tcl
    extra_commands = "cosim_design" if run_cosim else ""
    tcl = HLS_TCL_SCRIPT.format(
        prj_name=project, 
        top_func_name=top_func_name, 
        part="xcu55c-fsvh2892-2L-e",
        extra_commands=extra_commands
    )
    tcl_path = os.path.join(project, dir_name, "hls_script.tcl")
    os.makedirs(os.path.dirname(tcl_path), exist_ok=True)
    with open(tcl_path, "w") as f: f.write(tcl)

    run_vitis_hls_synthesis(project, dir_name)
    report_vitis_hls(project, dir_name)

def build_vitis(
    top_mod: Module,
    utils_mod: Module,
    top_func_name: str,
    project: str,
    compile_only: bool = False,
    use_hls_task: bool = False,
    dir_name: str = "vitis",
    enable_dma: bool = False,
) -> VitisCallable:
    if not compile_only:
        kernel_path, host_path, arg_infos = emit_vitis_sources(
            top_mod=top_mod,
            utils_mod=utils_mod,
            top_func_name=top_func_name,
            project_dir=project,
            dir_name=dir_name,
            use_hls_task=use_hls_task
        )
    else:
        from .core.arg_info import collect_dma_arg_info
        from .utils import find_func_in_module
        top_func = find_func_in_module(top_mod, top_func_name)
        arg_infos = collect_dma_arg_info(top_func)
        src_dir = os.path.join(project, dir_name, "src")
        kernel_path = os.path.join(src_dir, "kernel.cpp")
        host_path = os.path.join(src_dir, "dma_host_funcs.cpp")

    builder = VitisBuilder(project, top_func_name, arg_infos, enable_dma=enable_dma)
    return builder.build(kernel_path, host_path)