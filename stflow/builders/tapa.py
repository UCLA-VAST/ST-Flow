import os
import subprocess
import sysconfig
import logging
from typing import List

logger = logging.getLogger(__name__)
from .base import BaseBuilder, BaseCallable
from ..core.arg_info import ArgInfo

class TapaCallable(BaseCallable):
    def __call__(self, *args, bitstream: str = ""):
        return super().__call__(*args, bitstream=bitstream)

class TapaBuilder(BaseBuilder):
    def __init__(
        self,
        project_dir: str,
        top_func_name: str,
        arg_infos: List[ArgInfo],
        enable_dma: bool = False,
    ):
        super().__init__(project_dir, top_func_name, arg_infos, "tapa", enable_dma=enable_dma)

    def generate_wrapper(self) -> str:
        lines = []
        emit = lines.append
        self.generate_common_wrapper_parts(emit)
        
        emit("#include <tapa.h>")
        emit("#include <hls_vector.h>")
        emit("")
        
        self.emit_wrapper_signature(emit)
        self.emit_buffer_setup(emit)
        self.emit_packing_logic(emit)
        
        emit("  // invoke kernel\n")
        emit("  tapa::invoke(")
        emit(f"    {self.top_func_name}, bitstream,")
        
        invoke_terms = []
        for a in self.arg_infos:
            if len(a.physical_shape) > 1:
                tail = "[" + "][".join(str(s) for s in a.physical_shape[1:]) + "]"
                invoke_terms.append(f"    reinterpret_cast<float (*){tail}>(p_packed_{a.index}.data())")
            else:
                if a.role in ("input", "weight"):
                    invoke_terms.append(f"    tapa::read_only_mmap<{a.packed_elem_cpp}>(p_packed_{a.index})")
                elif a.role == "output":
                    invoke_terms.append(f"    tapa::write_only_mmap<{a.packed_elem_cpp}>(p_packed_{a.index})")
                else:
                    invoke_terms.append(f"    tapa::read_write_mmap<{a.packed_elem_cpp}>(p_packed_{a.index})")
        
        for i, t in enumerate(invoke_terms):
            emit(t + ("," if i + 1 < len(invoke_terms) else ""))
        emit("  );")
        emit("")
        
        self.emit_unpacking_logic(emit)
        emit("}")
        emit("")
        self.emit_pybind_module(emit)
        
        return "\n".join(lines)

    def build(self, kernel_path: str, host_path: str) -> TapaCallable:
        wrapper_cpp = os.path.join(self.output_dir, f"{self.lib_name}.cpp")
        with open(wrapper_cpp, "w") as f:
            f.write(self.generate_wrapper())
            
        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ""
        so_path = os.path.join(self.output_dir, f"{self.lib_name}{ext_suffix}")
        
        def get_path(env_name, fallbacks, desc):
            path = os.environ.get(env_name)
            if path:
                if not os.path.exists(path):
                    logger.warning(f"Warning: {env_name} is set to {path}, but this path does not exist on this node.")
                return path
            
            for p in fallbacks:
                # expand user
                p_exp = os.path.expanduser(p)
                if os.path.exists(p_exp):
                    return os.path.abspath(p_exp)
            
            raise RuntimeError(
                f"{env_name} environment variable not set and could not find {desc} in common locations. "
                "Please set it in slurm_env.sh or local_env.sh."
            )

        hls_path = get_path("XILINX_HLS", [
            "/opt/xilinx/Vitis_HLS/2024.1", 
            "/opt/xilinx/Vitis_HLS/2024.2", 
            "/tools/Xilinx/Vitis_HLS/2024.1",
            "/tools/Xilinx/Vitis_HLS/2024.2"
        ], "HLS")

        tapa_path = get_path("XILINX_TAPA", [
            "~/.rapidstream-tapa/usr",
            "/usr",
            "/usr/local"
        ], "TAPA")
                    
        src_dir = os.path.dirname(kernel_path)
        
        cmd = ["g++", "-shared", "-std=c++17", "-fPIC"]
        cmd += subprocess.check_output(["python3", "-m", "pybind11", "--includes"], text=True).strip().split()
        if hls_path:
            cmd += [f"-I{hls_path}/include"]
        
        cmd += [f"-I{tapa_path}/include"]
        cmd += [f"-I{src_dir}", f"-I{self.output_dir}"]
        cmd += [kernel_path, host_path, wrapper_cpp]
        cmd += ["-o", so_path, "-O3"]
        cmd += [f"-L{tapa_path}/lib", "-Wl,--no-as-needed", "-ltapa", "-lfrt", "-lglog", "-lgflags", "-lyaml-cpp", "-ltinyxml2", "-lOpenCL", "-lcontext", "-lthread", "-Wl,--as-needed"]

        cmd += [f"-Wl,-rpath,{tapa_path}/lib"]

        
        self.run_gpp(cmd)
        return TapaCallable(so_path, self.lib_name, self.top_func_name, self.output_dir)
