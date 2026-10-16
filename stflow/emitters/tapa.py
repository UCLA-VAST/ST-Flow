import os
import io
from typing import List, Tuple
from .._mlir.ir import Context, Location, Module
from .._mlir.dialects import streamhls as streamhls_d
from ..utils import find_func_in_module
from ..core.arg_info import ArgInfo, collect_dma_arg_info
from .common import write_text, emit_kernel_h, emit_dma_host_funcs_h, generate_default_host_funcs

def emit_tapa_sources(
    top_mod: Module,
    utils_mod: Module,
    top_func_name: str,
    project_dir: str,
) -> Tuple[str, str, List[ArgInfo]]:
    src_dir = os.path.join(project_dir, "tapa", "src")
    os.makedirs(src_dir, exist_ok=True)

    with Context() as ctx, Location.unknown():
        streamhls_d.register_dialect(ctx)
        top_func = find_func_in_module(top_mod, top_func_name)
        if top_func is None:
            raise RuntimeError(f"Could not find {top_func_name} in module")
        
        arg_infos = collect_dma_arg_info(top_func)

        # kernel.cpp (tapa)
        buf = io.StringIO()
        streamhls_d.emit_tapa_hls(top_mod, buf)
        kernel_cpp = buf.getvalue()
        kernel_path = os.path.join(src_dir, "kernel.cpp")
        write_text(kernel_path, kernel_cpp)

        # kernel.h
        kernel_h = emit_kernel_h(kernel_cpp, top_func_name, is_tapa=True)
        write_text(os.path.join(src_dir, "kernel.h"), kernel_h)

        # dma_host_funcs.cpp
        buf = io.StringIO()
        streamhls_d.emit_vitis_hls(utils_mod, buf, False)
        host_cpp = buf.getvalue()

        # Fallback to default
        if not host_cpp or "__host_pack_" not in host_cpp:
             host_cpp = generate_default_host_funcs(arg_infos)
        
        host_path = os.path.join(src_dir, "dma_host_funcs.cpp")
        write_text(host_path, host_cpp)

        # dma_host_funcs.h
        host_h = emit_dma_host_funcs_h(host_cpp)
        write_text(os.path.join(src_dir, "dma_host_funcs.h"), host_h)

        return kernel_path, host_path, arg_infos
