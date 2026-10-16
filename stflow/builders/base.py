import os
import sys
import math
import importlib
import subprocess
import sysconfig
import logging

logger = logging.getLogger(__name__)
from typing import List, Optional, Tuple
from ..core.arg_info import ArgInfo
from ..core.cpp_utils import sanitize_ident

class BaseCallable:
    def __init__(self, so_path: str, lib_name: str, top_func_name: str, module_dir: str):
        self.so_path = so_path
        self.lib_name = lib_name
        self.top_func_name = top_func_name
        self.module_dir = module_dir
        self._mod = None

    def _load(self):
        if self._mod is not None:
            return
        # Ensure we can find the module
        sys.path.append(self.module_dir)
        try:
             # Force reload or import
             if self.lib_name in sys.modules:
                 del sys.modules[self.lib_name]
             self._mod = importlib.import_module(self.lib_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load pybind module {self.lib_name} from {self.module_dir}: {e}")

    def __call__(self, *args, **kwargs):
        self._load()
        fn = getattr(self._mod, self.top_func_name)
        return fn(*args, **kwargs)

    def release(self):
        if self._mod is not None and hasattr(self._mod, "release"):
            self._mod.release()

class BaseBuilder:
    def __init__(
        self,
        project_dir: str,
        top_func_name: str,
        arg_infos: List[ArgInfo],
        tool: str,
        enable_dma: bool = False,
    ):
        self.project_dir = project_dir
        self.top_func_name = top_func_name
        self.arg_infos = arg_infos
        self.tool = tool
        self.enable_dma = enable_dma
        self.lib_name = f"py{tool}_{sanitize_ident(os.path.basename(project_dir))}_{sanitize_ident(top_func_name)}"
        self.output_dir = os.path.join(project_dir, tool, "pybind")
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _shape_prod(shape: Tuple[int, ...]) -> int:
        n = 1
        for d in shape:
            n *= int(d)
        return n

    def _get_ndarray_cast(self, shape: Tuple[int, ...]) -> str:
        if len(shape) <= 1:
            return ""
        tail = "[" + "][".join(str(s) for s in shape[1:]) + "]"
        return f"float (*){tail}"

    def _get_strides(self, shape: Tuple[int, ...]) -> List[int]:
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * int(shape[i + 1])
        return strides

    def _emit_nd_copy(self, emit, src_shape, dst_shape, src_ptr, dst_ptr, is_unpack=False):
        rank_src = len(src_shape)
        rank_dst = len(dst_shape)
        
        # Determine the rank to loop over. Usually original_shape rank.
        rank = rank_src
        src_strides = self._get_strides(src_shape)
        
        for i in range(rank):
            emit(f"  " * (i + 1) + f"for (size_t __i{i} = 0; __i{i} < {src_shape[i]}; ++__i{i}) {{")
        
        src_idx = " + ".join([f"__i{i} * {src_strides[i]}" for i in range(rank)]) if rank > 0 else "0"
        
        if rank_dst == rank:
            dst_strides = self._get_strides(dst_shape)
            dst_idx = " + ".join([f"__i{i} * {dst_strides[i]}" for i in range(rank)])
        else:
            # If ranks differ, we assume one is a flattened version of the other.
            # We use the same flat index 'src_idx'. 
            # This only works if the multi-dim one has no hidden padding relative to the flat one.
            dst_idx = src_idx

        if is_unpack:
            emit(f"  " * (rank + 1) + f"{src_ptr}[{src_idx}] = {dst_ptr}[{dst_idx}];")
        else:
            emit(f"  " * (rank + 1) + f"{dst_ptr}[{dst_idx}] = {src_ptr}[{src_idx}];")

        for i in range(rank - 1, -1, -1):
            emit(f"  " * (i + 1) + f"}}")

    def _emit_dma_pack_host_view(self, emit, a: ArgInfo, base_ptr: str) -> str:
        """Return C expression for the first argument of __host_pack_*."""
        r_orig = a.original_shape
        r_padded = a.padded_shape
        # logger.info(f"Arg {a.name}: r_orig={r_orig}, r_padded={r_padded}")
        emit(f"  // Arg {a.name}: r_orig={r_orig}, r_padded={r_padded}")

        cast_type = self._get_ndarray_cast(r_padded)
        cast_str = f"reinterpret_cast<{cast_type}>" if cast_type else ""

        if r_orig == r_padded:
            return f"{cast_str}({base_ptr})" if cast_str else base_ptr

        # Manual padding needed
        tmp = f"__pphx_pad_{a.index}"
        total_padded = self._shape_prod(r_padded)
        emit(f"  std::vector<float> {tmp}({total_padded}, 0.f);")
        self._emit_nd_copy(emit, r_orig, r_padded, base_ptr, f"{tmp}.data()", is_unpack=False)

        return f"{cast_str}({tmp}.data())" if cast_str else f"{tmp}.data()"

    def _emit_dma_unpack_gather(self, emit, a: ArgInfo, base_ptr: str, tmp_name: str) -> None:
        r_orig = a.original_shape
        r_padded = a.padded_shape

        if r_orig == r_padded:
            return

        self._emit_nd_copy(emit, r_orig, r_padded, base_ptr, f"{tmp_name}.data()", is_unpack=True)

    def generate_common_wrapper_parts(self, emit):
        exposed = [a for a in self.arg_infos if a.role in ("input", "output")]
        
        emit(f"// Auto-generated by ST-Flow {self.tool} builder\n")
        emit("#include <pybind11/pybind11.h>")
        emit("#include <pybind11/numpy.h>")
        emit("#include <pybind11/stl.h>")
        emit("#include <vector>")
        emit("#include <string>")
        emit("#include <fstream>")
        emit("#include <stdexcept>")
        emit("#include <iostream>")
        emit("#include <cstring>")
        emit("")
        emit('#include "kernel.h"')
        emit('#include "dma_host_funcs.h"')
        emit("")
        emit("namespace py = pybind11;")
        emit("")
        
        # Load bin helper
        emit("static void __stflow_load_bin(const std::string& path, void* dst, size_t nbytes) {")
        emit("  std::ifstream fin(path, std::ios::binary);")
        emit('  if (!fin) throw std::runtime_error("failed to open: " + path);')
        emit("  fin.read(reinterpret_cast<char*>(dst), nbytes);")
        emit('  if (!fin) throw std::runtime_error("failed to read: " + path);')
        emit("}")
        emit("")

    def emit_wrapper_signature(self, emit):
        exposed = [a for a in self.arg_infos if a.role in ("input", "output")]
        emit(f"void {self.lib_name}(")
        for k, a in enumerate(exposed):
            comma = "," if k < len(exposed) - 1 else ","
            emit(f"  py::array_t<float, py::array::c_style | py::array::forcecast> arg{a.index}{comma}")
        emit('  const std::string& bitstream = ""')
        emit(") {")
        emit("")

    def emit_buffer_setup(self, emit):
        exposed = [a for a in self.arg_infos if a.role in ("input", "output")]
        for a in exposed:
            emit(f"  auto buf{a.index} = arg{a.index}.request();")
            emit(f"  float* p_arg{a.index} = reinterpret_cast<float*>(buf{a.index}.ptr);")
            if a.role == "output":
                cast_type = self._get_ndarray_cast(a.original_shape)
                if not cast_type:
                    emit(f"  auto* p_arg{a.index}_nd = p_arg{a.index};")
                else:
                    emit(f"  auto* p_arg{a.index}_nd = reinterpret_cast<{cast_type}>(p_arg{a.index});")
        emit("")
        for a in self.arg_infos:
            emit(f"  std::vector<{a.packed_elem_cpp}> p_packed_{a.index}({a.packed_depth});")
        emit("")

    def emit_packing_logic(self, emit):
        for a in self.arg_infos:
            # print(
            #     f"Arg {a.index}: original_shape: {a.original_shape}, padded_shape: {a.padded_shape}, "
            #     f"physical_shape: {a.physical_shape}"
            # )
            if a.role == "input":
                if self.enable_dma:
                    expr = self._emit_dma_pack_host_view(emit, a, f"p_arg{a.index}")
                    emit(f"  __host_pack_{a.name}({expr}, p_packed_{a.index}.data());")
                else:
                    # When DMA is disabled, __host_pack takes float*
                    emit(f"  __host_pack_{a.name}(p_arg{a.index}, p_packed_{a.index}.data());")
            elif a.role == "weight":
                total = self._shape_prod(a.original_shape)
                emit(f"  std::vector<float> w_{a.index}({total});")
                emit(
                    f'  __stflow_load_bin("{self.project_dir}/weight_bins/{a.name}.bin", '
                    f"w_{a.index}.data(), w_{a.index}.size() * sizeof(float));"
                )
                if self.enable_dma:
                    expr = self._emit_dma_pack_host_view(emit, a, f"w_{a.index}.data()")
                    emit(f"  __host_pack_{a.name}({expr}, p_packed_{a.index}.data());")
                else:
                    # When DMA is disabled, __host_pack takes float*
                    emit(f"  __host_pack_{a.name}(w_{a.index}.data(), p_packed_{a.index}.data());")
            emit("")

    def emit_unpacking_logic(self, emit):
        for a in self.arg_infos:
            if a.role != "output":
                continue
            if self.enable_dma:
                r_orig = a.original_shape
                r_padded = a.padded_shape
                cast_type = self._get_ndarray_cast(r_padded)
                cast_str = f"reinterpret_cast<{cast_type}>" if cast_type else ""
                
                if r_orig == r_padded:
                    arg_expr = f"{cast_str}(p_arg{a.index})" if cast_str else f"p_arg{a.index}"
                    emit(f"  __host_unpack_{a.name}(p_packed_{a.index}.data(), {arg_expr});")
                else:
                    tmp = f"__pphx_upad_{a.index}"
                    total_padded = self._shape_prod(r_padded)
                    emit(f"  std::vector<float> {tmp}({total_padded}, 0.f);")
                    tmp_expr = f"{cast_str}({tmp}.data())" if cast_str else f"{tmp}.data()"
                    emit(f"  __host_unpack_{a.name}(p_packed_{a.index}.data(), {tmp_expr});")
                    self._emit_dma_unpack_gather(emit, a, f"p_arg{a.index}", tmp)
            else:
                # When DMA is disabled, __host_unpack takes float*
                emit(f"  __host_unpack_{a.name}(p_packed_{a.index}.data(), p_arg{a.index});")
        emit("")

    def emit_pybind_module(self, emit):
        exposed = [a for a in self.arg_infos if a.role in ("input", "output")]
        emit(f"PYBIND11_MODULE({self.lib_name}, m) {{")
        args = [f'py::arg("arg{a.index}")' for a in exposed] + ['py::arg("bitstream") = ""']
        emit(f'  m.def("{self.top_func_name}", &{self.lib_name}, {", ".join(args)});')
        emit("}")

    def run_gpp(self, cmd):
        logger.info("Running command: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
