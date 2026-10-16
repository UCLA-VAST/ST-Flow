import os
import subprocess
import sysconfig
import logging
from typing import List

logger = logging.getLogger(__name__)

# XRT is only needed to run a bitstream on a board. The generated wrapper guards all of
# its XRT code with this macro, so C-simulation builds and runs without XRT installed.
_XRT_MACRO = "STFLOW_WITH_XRT"
import xml.etree.ElementTree as ET
from .base import BaseBuilder, BaseCallable
from ..core.arg_info import ArgInfo

def _get_text(root, path):
    n = root.find(path)
    if n is None or n.text is None:
        return None
    t = n.text.strip()
    return None if t == "" else t

def _to_int(s):
    if s is None:
        return None
    s = s.strip()
    if s.lower() == "undef":
        return None
    return int(s)

def _to_float(s):
    if s is None:
        return None
    s = s.strip()
    if s.lower() == "undef":
        return None
    return float(s)

def parse_vitis_hls_report(xml_path):
    root = ET.parse(xml_path).getroot()
    min_cycles = _to_int(_get_text(root, "./PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency"))
    max_cycles = _to_int(_get_text(root, "./PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency"))
    dsps = _to_int(_get_text(root, "./AreaEstimates/Resources/DSP"))
    brams = _to_int(_get_text(root, "./AreaEstimates/Resources/BRAM_18K"))
    urams = _to_int(_get_text(root, "./AreaEstimates/Resources/URAM"))
    dsps_avail = _to_int(_get_text(root, "./AreaEstimates/AvailableResources/DSP"))
    brams_avail = _to_int(_get_text(root, "./AreaEstimates/AvailableResources/BRAM_18K"))
    urams_avail = _to_int(_get_text(root, "./AreaEstimates/AvailableResources/URAM"))

    def pct(used, avail):
        if used is None or avail in (None, 0):
            return None
        return 100.0 * used / avail

    return {
        "min_cycles": min_cycles,
        "max_cycles": max_cycles,
        "dsps": dsps,
        "brams_18k": brams,
        "urams": urams,
        "dsps_pct": f"{pct(dsps, dsps_avail):.2f}%",
        "brams_pct": f"{pct(brams, brams_avail):.2f}%",
        "urams_pct": f"{pct(urams, urams_avail):.2f}%",
        "estimated_clock_period_ns": _to_float(_get_text(root, "./PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod")),
    }

class VitisCallable(BaseCallable):
    def __call__(self, *args, bitstream: str = ""):
        return super().__call__(*args, bitstream=bitstream)

class VitisBuilder(BaseBuilder):
    def __init__(
        self,
        project_dir: str,
        top_func_name: str,
        arg_infos: List[ArgInfo],
        enable_dma: bool = False,
    ):
        super().__init__(project_dir, top_func_name, arg_infos, "vitis", enable_dma=enable_dma)

    def generate_wrapper(self) -> str:
        lines = []
        emit = lines.append
        self.generate_common_wrapper_parts(emit)
        
        emit(f"#ifdef {_XRT_MACRO}")
        emit("#include <xrt/xrt_kernel.h>")
        emit("#include <xrt/xrt_bo.h>")
        emit("#include <xrt/xrt_device.h>")
        emit("#endif")
        emit("#include <chrono>")
        emit("#include <fstream>")
        emit("#include <stdexcept>")
        emit("")
        
        emit(f"#ifdef {_XRT_MACRO}")
        emit("struct XRTContext {")
        emit("  xrt::device device;")
        emit("  xrt::uuid uuid;")
        emit("  xrt::kernel kernel;")
        emit("};")
        emit("static XRTContext* g_xrt = nullptr;")
        emit("#endif")
        emit("")
        
        self.emit_wrapper_signature(emit)
        self.emit_buffer_setup(emit)
        self.emit_packing_logic(emit)
        
        emit("  if (bitstream.empty()) {")
        emit("    // CSIM Mode")
        emit(f"    {self.top_func_name}(")
        for i, a in enumerate(self.arg_infos):
            cast = ""
            if len(a.physical_shape) > 1:
                 tail = "[" + "][".join(str(s) for s in a.physical_shape[1:]) + "]"
                 cast = f"reinterpret_cast<float (*){tail}>"
            emit(f"      {cast}(p_packed_{a.index}.data())" + ("," if i < len(self.arg_infos)-1 else ""))
        emit("    );")
        emit("  } else {")
        emit(f"#ifndef {_XRT_MACRO}")
        emit('    throw std::runtime_error("This design was built without XRT, so it can only run in '
             'C-simulation. Install XRT (or set XILINX_XRT) and rebuild to run a bitstream.");')
        emit("#else")
        emit("    // XRT Mode")
        emit("    if (g_xrt == nullptr) {")
        emit("      for (int i = 0; i < 8; ++i) {")
        emit("        try {")
        emit("          g_xrt = new XRTContext();")
        emit("          g_xrt->device = xrt::device(i);")
        emit("          g_xrt->uuid = g_xrt->device.load_xclbin(bitstream);")
        emit("          g_xrt->kernel = xrt::kernel(g_xrt->device, g_xrt->uuid, \"module_top\");")
        emit("          std::cout << \"Successfully initialized device \" << i << std::endl;")
        emit("          break;")
        emit("        } catch (const std::exception& e) {")
        emit("          delete g_xrt;")
        emit("          g_xrt = nullptr;")
        emit("          std::cerr << \"Failed to initialize device \" << i << \": \" << e.what() << std::endl;")
        emit("          continue;")
        emit("        }")
        emit("      }")
        emit("      if (g_xrt == nullptr) throw std::runtime_error(\"Could not find a valid FPGA device after checking indices 0-7.\");")
        emit("    }")
        emit("")
        for a in self.arg_infos:
            emit(f"    xrt::bo bo_{a.index}(g_xrt->device, p_packed_{a.index}.size() * sizeof({a.packed_elem_cpp}), g_xrt->kernel.group_id({a.index}));")
            if a.role in ("input", "weight", "rw_buf"):
                emit(f"    bo_{a.index}.write(p_packed_{a.index}.data());")
                emit(f"    bo_{a.index}.sync(XCL_BO_SYNC_BO_TO_DEVICE);")
        emit("")
        emit("    double total_time = 0;")
        emit("    for (int i = 0; i < 11; ++i) {")
        emit("      auto start = std::chrono::high_resolution_clock::now();")
        emit("      auto run = g_xrt->kernel(")
        for i, a in enumerate(self.arg_infos):
            emit(f"        bo_{a.index}" + ("," if i < len(self.arg_infos)-1 else ""))
        emit("      );")
        emit("      run.wait();")
        emit("      auto end = std::chrono::high_resolution_clock::now();")
        emit("      std::chrono::duration<double, std::milli> elapsed = end - start;")
        emit("      std::cout << \"Run \" << i << \": \" << elapsed.count() << \" ms\" << (i == 0 ? \" (warm-up)\" : \"\") << std::endl;")
        emit("      if (i > 0) total_time += elapsed.count();")
        emit("    }")
        emit('    std::cout << "Kernel execution time: " << total_time / 10.0 << " ms" << std::endl;')
        emit(f'    std::ofstream latency_file("{self.project_dir}/reports/bitstream_latency.txt");')
        emit("    if (latency_file.is_open()) {")
        emit("        latency_file << total_time / 10.0 << std::endl;")
        emit("        latency_file.close();")
        emit("    }")
        emit("")
        for a in self.arg_infos:
            if a.role in ("output", "rw_buf"):
                emit(f"    bo_{a.index}.sync(XCL_BO_SYNC_BO_FROM_DEVICE);")
                emit(f"    bo_{a.index}.read(p_packed_{a.index}.data());")
        emit("#endif")
        emit("  }")
        emit("")
        
        self.emit_unpacking_logic(emit)
        emit("}")
        emit("")
        self.emit_pybind_module(emit)
        
        return "\n".join(lines)

    def emit_pybind_module(self, emit):
        exposed = [a for a in self.arg_infos if a.role in ("input", "output")]
        emit(f"PYBIND11_MODULE({self.lib_name}, m) {{")
        emit('  m.def("release", []() {')
        emit(f"#ifdef {_XRT_MACRO}")
        emit('    if (g_xrt) {')
        emit('      delete g_xrt;')
        emit('      g_xrt = nullptr;')
        emit('    }')
        emit("#endif")
        emit('  });')
        args = [f'py::arg("arg{a.index}")' for a in exposed] + ['py::arg("bitstream") = ""']
        emit(f'  m.def("{self.top_func_name}", &{self.lib_name}, {", ".join(args)});')
        emit("}")

    def build(self, kernel_path: str, host_path: str) -> VitisCallable:
        wrapper_cpp = os.path.join(self.output_dir, f"{self.lib_name}.cpp")
        with open(wrapper_cpp, "w") as f:
            f.write(self.generate_wrapper())
            
        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ""
        so_path = os.path.join(self.output_dir, f"{self.lib_name}{ext_suffix}")
        
        def get_path(env_name, fallbacks, desc):
            path = os.environ.get(env_name)
            if path:
                if os.path.exists(path):
                    return path
                # A stale value (e.g. a cluster module pointing at a tool that is not
                # installed on this node) must not shadow a working fallback.
                logger.warning(f"Warning: {env_name} is set to {path}, but this path does not exist on this node; trying the defaults.")

            for p in fallbacks:
                if os.path.exists(p):
                    return os.path.abspath(p)
            
            raise RuntimeError(
                f"{env_name} environment variable not set and could not find {desc} in common locations. "
                f"Please set {env_name} in slurm_env.sh or local_env.sh."
            )

        hls_path = get_path("XILINX_HLS", [
            "/opt/xilinx/Vitis_HLS/2024.1", 
            "/opt/xilinx/Vitis_HLS/2024.2", 
            "/opt/xilinx/tools/Vitis/2024.2",
            "/tools/Xilinx/Vitis_HLS/2024.1",
            "/tools/Xilinx/Vitis_HLS/2024.2"
        ], "HLS")

        # XRT is optional: with it the wrapper can also run a bitstream on a board;
        # without it, it is built for C-simulation only.
        try:
            xrt_path = get_path("XILINX_XRT", ["/opt/xilinx/xrt", "/opt/xilinx/XRT"], "XRT")
        except RuntimeError:
            xrt_path = None
            logger.info("XRT not found: building the host wrapper for C-simulation only.")


        src_dir = os.path.dirname(kernel_path)
        cmd = ["g++", "-shared", "-std=c++17", "-fPIC"]
        cmd += subprocess.check_output(["python3", "-m", "pybind11", "--includes"], text=True).strip().split()
        if hls_path:
            cmd += [f"-I{hls_path}/include"]
        cmd += [f"-I{src_dir}", f"-I{self.output_dir}"]
        if xrt_path:
            cmd += [f"-D{_XRT_MACRO}", f"-I{xrt_path}/include"]
        cmd += [kernel_path, host_path, wrapper_cpp]
        cmd += ["-o", so_path, "-O3"]
        if xrt_path:
            cmd += [f"-L{xrt_path}/lib", "-lxrt_coreutil", f"-Wl,-rpath,{xrt_path}/lib"]
        
        self.run_gpp(cmd)
        return VitisCallable(so_path, self.lib_name, self.top_func_name, self.output_dir)
