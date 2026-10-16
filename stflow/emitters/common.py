import os
import io
from typing import List
from ..core.arg_info import ArgInfo
from ..core.cpp_utils import extract_function_prototype, extract_prototypes_by_prefix, VECTOR_CLASS, ERF_APPROX

def write_text(path: str, s: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)

def emit_kernel_h(kernel_cpp: str, top_func: str, is_tapa: bool = True) -> str:
    p = extract_function_prototype(kernel_cpp, top_func)
    if not p:
        raise RuntimeError(f"Could not find prototype for {top_func} in kernel.cpp")

    sync_task_logic = ""
    if not is_tapa:
        sync_task_logic = """
#ifndef __SYNTHESIS__
#include <thread>
namespace hls {
  struct SyncTask {
    std::thread t;
    template<typename F, typename... Args>
    SyncTask(F f, Args&... args) : t([f, &args...]() { f(args...); }) {}
    ~SyncTask() { if(t.joinable()) t.join(); }
  };
}
#define hls_thread_local
#define task SyncTask
#endif
"""

    return (
        "#ifndef KERNEL_H\n#define KERNEL_H\n\n"
        + ("#include <tapa.h>\n" if is_tapa else "")
        + sync_task_logic + "\n"
        + VECTOR_CLASS + "\n\n"
        + ERF_APPROX + "\n\n"
        + "  " + p.replace("\n", "\n  ")
        + "\n\n"
        "#endif // KERNEL_H\n"
    )

def emit_dma_host_funcs_h(host_cpp: str) -> str:
    protos = extract_prototypes_by_prefix(
        host_cpp, prefixes=["__host_pack_", "__host_unpack_"]
    )
    if not protos:
        # Fallback to empty OR just return empty header
        return "#ifndef DMA_HOST_FUNCS_H\n#define DMA_HOST_FUNCS_H\n\n#endif\n"

    return (
        "#ifndef DMA_HOST_FUNCS_H\n#define DMA_HOST_FUNCS_H\n\n"
        + "\n\n".join("  " + p.replace("\n", "\n  ") for p in protos)
        + "\n\n"
        "#endif // DMA_HOST_FUNCS_H\n"
    )

def generate_default_host_funcs(arg_infos: List[ArgInfo]) -> str:
    # Generic memcpy-based packing if specialized ones aren't available
    def _elem_count(shape, fallback: int) -> int:
        if not shape:
            return fallback
        n = 1
        for d in shape:
            n *= int(d)
        return n

    lines = [
        '#include "dma_host_funcs.h"',
        '#include <cstring>',
        ""
    ]
    for arg in arg_infos:
        original_shape = tuple(getattr(arg, "original_shape", ()) or ())
        physical_shape = tuple(getattr(arg, "physical_shape", ()) or ())
        original_elems = _elem_count(original_shape, arg.packed_depth)
        can_index_map = (
            arg.packed_elem_cpp == "float"
            and original_shape
            and physical_shape
            and len(original_shape) == len(physical_shape)
        )

        if arg.role == "input" or arg.role == "weight":
            # __host_pack_argName(logical*, packed*)
            lines.append(f'void __host_pack_{arg.name}(float* src, {arg.packed_elem_cpp}* dst) {{')
            if can_index_map and original_shape != physical_shape:
                lines.append(f"  std::memset(dst, 0, {arg.packed_depth} * sizeof(float));")
                phys_strides = []
                stride = 1
                for d in reversed(physical_shape):
                    phys_strides.append(stride)
                    stride *= int(d)
                phys_strides = list(reversed(phys_strides))
                lines.append(f"  for (size_t lin = 0; lin < {original_elems}; ++lin) {{")
                lines.append("    size_t rem = lin;")
                lines.append("    size_t phys = 0;")
                for dim in range(len(original_shape) - 1, -1, -1):
                    lines.append(f"    size_t i{dim} = rem % {int(original_shape[dim])};")
                    lines.append(f"    rem /= {int(original_shape[dim])};")
                    lines.append(f"    phys += i{dim} * {int(phys_strides[dim])};")
                lines.append("    dst[phys] = src[lin];")
                lines.append("  }")
            else:
                size_bytes = f"{original_elems} * sizeof(float)"
                lines.append(f"  std::memcpy(dst, src, {size_bytes});")
            lines.append("}")
        elif arg.role == "output":
            lines.append(f'void __host_unpack_{arg.name}({arg.packed_elem_cpp}* src, float* dst) {{')
            if can_index_map and original_shape != physical_shape:
                phys_strides = []
                stride = 1
                for d in reversed(physical_shape):
                    phys_strides.append(stride)
                    stride *= int(d)
                phys_strides = list(reversed(phys_strides))
                lines.append(f"  for (size_t lin = 0; lin < {original_elems}; ++lin) {{")
                lines.append("    size_t rem = lin;")
                lines.append("    size_t phys = 0;")
                for dim in range(len(original_shape) - 1, -1, -1):
                    lines.append(f"    size_t i{dim} = rem % {int(original_shape[dim])};")
                    lines.append(f"    rem /= {int(original_shape[dim])};")
                    lines.append(f"    phys += i{dim} * {int(phys_strides[dim])};")
                lines.append("    dst[lin] = src[phys];")
                lines.append("  }")
            else:
                size_bytes = f"{original_elems} * sizeof(float)"
                lines.append(f"  std::memcpy(dst, src, {size_bytes});")
            lines.append("}")
        lines.append("")
    return "\n".join(lines)
