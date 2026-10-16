import os
import io
from typing import List, Tuple
from .._mlir.ir import Context, Location, Module
from .._mlir.dialects import streamhls as streamhls_d
from ..utils import find_func_in_module
from ..core.arg_info import ArgInfo, collect_dma_arg_info
from .common import write_text, emit_kernel_h, emit_dma_host_funcs_h, generate_default_host_funcs

def _emit_vitis_tb_cpp(arg_infos: List[ArgInfo], top_func_name: str) -> str:
    lines = [
        '#include <cstdint>', 
        '#include "kernel.h"',
        "",
        "int main() {",
    ]
    # # Simple TB for functional check
    # for arg in arg_infos:
    #     total_elems = 1
    #     for d in arg.physical_shape:
    #         total_elems *= int(d)
    #     lines.append(f"  static {arg.packed_elem_cpp} {arg.name}[{total_elems}];")

    # Simple TB for functional check (use full array dimensions)
    for arg in arg_infos:
        dims = "[" + "][".join(str(d) for d in arg.physical_shape) + "]"
        lines.append(f"  static {arg.packed_elem_cpp} {arg.name}{dims};")

    call_args = ", ".join([arg.name for arg in arg_infos])
    lines.append(f"  {top_func_name}({call_args});")
    lines.append("  return 0;")
    lines.append("}")
    return "\n".join(lines)

def emit_vitis_sources(
    top_mod: Module,
    utils_mod: Module,
    top_func_name: str,
    project_dir: str,
    dir_name: str = "vitis",
    use_hls_task: bool = False
) -> Tuple[str, str, List[ArgInfo]]:
    src_dir = os.path.join(project_dir, dir_name, "src")
    os.makedirs(src_dir, exist_ok=True)

    with Context() as ctx, Location.unknown():
        streamhls_d.register_dialect(ctx)
        top_func = find_func_in_module(top_mod, top_func_name)
        if top_func is None:
            raise RuntimeError(f"Could not find {top_func_name} in module")
        
        arg_infos = collect_dma_arg_info(top_func)

        # kernel.cpp
        buf = io.StringIO()
        streamhls_d.emit_vitis_hls(top_mod, buf, use_hls_task)
        kernel_cpp = buf.getvalue()

        # Fix include order for Vitis 2024+
        if "#include <hls_stream.h>" in kernel_cpp and "#include <hls_task.h>" in kernel_cpp:
            kernel_cpp = kernel_cpp.replace("#include <hls_stream.h>", "// hls_stream.h moved")
            kernel_cpp = kernel_cpp.replace("#include <hls_task.h>", "#include <hls_task.h>\n#include <hls_stream.h>")

        kernel_path = os.path.join(src_dir, "kernel.cpp")
        write_text(kernel_path, kernel_cpp)

        # kernel.h
        kernel_h = emit_kernel_h(kernel_cpp, top_func_name, is_tapa=False)
        write_text(os.path.join(src_dir, "kernel.h"), kernel_h)

        # kernel_tb.cpp
        tb_cpp = _emit_vitis_tb_cpp(arg_infos, top_func_name)
        write_text(os.path.join(src_dir, "kernel_tb.cpp"), tb_cpp)

        # dma_host_funcs.cpp (from utils_mod)
        buf = io.StringIO()
        streamhls_d.emit_vitis_hls(utils_mod, buf, False)
        host_cpp = buf.getvalue()

        # Fix include order for Vitis 2024+
        if "#include <hls_stream.h>" in host_cpp and "#include <hls_task.h>" in host_cpp:
            host_cpp = host_cpp.replace("#include <hls_stream.h>", "// hls_stream.h moved")
            host_cpp = host_cpp.replace("#include <hls_task.h>", "#include <hls_task.h>\n#include <hls_stream.h>")

        # Fallback to default if empty or headers-only
        if not host_cpp or "__host_pack_" not in host_cpp:
             host_cpp = generate_default_host_funcs(arg_infos)
        
        host_path = os.path.join(src_dir, "dma_host_funcs.cpp")
        write_text(host_path, host_cpp)

        # dma_host_funcs.h
        host_h = emit_dma_host_funcs_h(host_cpp)
        write_text(os.path.join(src_dir, "dma_host_funcs.h"), host_h)

        return kernel_path, host_path, arg_infos
