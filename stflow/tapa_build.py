import os
from typing import Optional
from ._mlir.ir import Module
from .emitters.tapa import emit_tapa_sources
from .builders.tapa import TapaBuilder, TapaCallable

def build_tapa(
    top_mod: Module,
    utils_mod: Module,
    top_func_name: str,
    project: str,
    compile_only: bool = False,
    enable_dma: bool = False,
) -> TapaCallable:
    """
    Standardized entry point for TAPA build flow.
    """
    dir_name = "tapa"
    if not compile_only:
        kernel_path, host_path, arg_infos = emit_tapa_sources(
            top_mod=top_mod,
            utils_mod=utils_mod,
            top_func_name=top_func_name,
            project_dir=project,
        )
    else:
        from .core.arg_info import collect_dma_arg_info
        from .utils import find_func_in_module
        top_func = find_func_in_module(top_mod, top_func_name)
        arg_infos = collect_dma_arg_info(top_func)
        src_dir = os.path.join(project, dir_name, "src")
        kernel_path = os.path.join(src_dir, "kernel.cpp")
        host_path = os.path.join(src_dir, "dma_host_funcs.cpp")

    builder = TapaBuilder(project, top_func_name, arg_infos, enable_dma=enable_dma)
    return builder.build(kernel_path, host_path)
