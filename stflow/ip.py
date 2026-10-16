import os
import sys
import importlib
import subprocess
import traceback
import time
import logging

logger = logging.getLogger(__name__)

# https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html
allo2c_type = {
    "float32": "float",
    "float64": "double",
    "int1": "bool",
    "int8": "int8_t",
    "int16": "int16_t",
    "int32": "int",
    "int64": "int64_t",
    "int128": "ap_int<128>",
    # bitwidth larger than 64 is not supported by numpy+pybind11
    "uint1": "bool",
    "uint8": "uint8_t",
    "uint16": "uint16_t",
    "uint32": "unsigned int",
    "uint64": "uint64_t",
    "uint128": "ap_uint<128>",
}

c2allo_type = {v: k for k, v in allo2c_type.items()}
c2allo_type["int32_t"] = "int32"
c2allo_type["uint32_t"] = "uint32"


class StreamHLSIPModule():
    def __init__(
        self, top, platform, headers, impls, signature, project, args_with_attrs, include_paths=None, link_hls=True
    ):
        self.top = top
        self.platform = platform
        self.headers = headers
        self.project = project
        self.abs_path = self.project #os.path.dirname(traceback.extract_stack()[-2].filename)
        self.temp_path = os.path.join(self.abs_path, "pybind")
        os.makedirs(self.temp_path, exist_ok=True)
        abs_impls = []
        for impl in impls:
            abs_impls.append(os.path.join(self.abs_path, impl))
        self.impls = abs_impls
        if include_paths is None:
            include_paths = []
        self.include_paths = include_paths + [self.abs_path]
        if link_hls:
            if hls_path := os.environ.get("XILINX_HLS"):
                # Adjust to the path of Vitis include directory
                self.include_paths.append(
                    hls_path + "/include"
                )
            else:
                raise RuntimeError(
                    "Please set source Vivato/Vitis settings64.sh or install Vivado/Vitis HLS and add it to your PATH"
                )
        
        # Parse signature
        arg_types = signature
        self.args = []
        for arg_type in arg_types:
            arg_type = arg_type.strip()
            if "[" not in arg_type or "[]" in arg_type:
                # scalar
                self.args.append((arg_type.split("[")[0], ()))
                continue
            ele_type = arg_type.split("[")[0].strip()
            # pylint: disable=eval-used
            shape = eval(f'({arg_type.split("[")[1].split("]")[0].strip()},)')
            self.args.append((ele_type, shape))

        self.args_with_attrs = []

        if len(arg_types) != len(args_with_attrs):
            raise RuntimeError("[ip.py] Argument types and attributes length mismatch!")

        for arg_type, arg_attrs in zip(arg_types, args_with_attrs):
            arg_type = arg_type.strip()
            if "[" not in arg_type or "[]" in arg_type:
                # scalar - extract type and add to args_with_attrs
                ele_type = arg_type.split("[")[0].strip() if "[" in arg_type else arg_type.strip()
                # For scalars, we need to infer name/kind/io from arg_attrs if available
                # If arg_attrs is None or incomplete, use defaults
                if arg_attrs and len(arg_attrs) >= 5:
                    arg_name = arg_attrs[2] if len(arg_attrs) > 2 else f"arg{len(self.args_with_attrs)}"
                    arg_kind = arg_attrs[3] if len(arg_attrs) > 3 else "off-chip"
                    arg_io = arg_attrs[4] if len(arg_attrs) > 4 else "read-only"
                else:
                    arg_name = f"arg{len(self.args_with_attrs)}"
                    arg_kind = "off-chip"
                    arg_io = "read-only"
                self.args_with_attrs.append((ele_type, (), arg_name, arg_kind, arg_io))
                continue
            ele_type = arg_type.split("[")[0].strip()
            # Ex Attr: ('f32', [128, 128], 'arg0', 'off-chip', 'read-only')
            shape = arg_attrs[1]
            arg_name = arg_attrs[2]
            arg_kind = arg_attrs[3]  # off-chip / weight
            arg_io = arg_attrs[4]  # read-only / write-only / read-write
            self.args_with_attrs.append((ele_type, shape, arg_name, arg_kind, arg_io))

        self.lib_name = f"py{self.top}_{self.project.split('.')[0]}_{self.platform}"
        self.c_wrapper_file = os.path.join(self.temp_path, f"{self.lib_name}.cpp")
    
    def _generate_pybind11_wrapper_legacy(self):
        # --------------------------------------------------------------
        # 1. Classify arguments
        # --------------------------------------------------------------
        # Each entry in args_with_attrs:
        #   (ele_type, shape, arg_name, arg_kind, arg_io)
        #
        # arg_kind ∈ {"off-chip", "weights"}
        # arg_io   ∈ {"read-only", "write-only", "read-write"}
        #
        # We preserve argument order in module_top.
        # --------------------------------------------------------------

        classified = []  # will hold dicts describing each arg

        for i, (ele_type, shape, arg_name, arg_kind, arg_io) in enumerate(self.args_with_attrs):
            entry = {
                "index": i,
                "ele_type": ele_type,
                "shape": shape,
                "name": arg_name,
                "kind": arg_kind,
                "io": arg_io,
            }
            if arg_kind == "weights":
                entry["role"] = "weight"
            else:
                if arg_io == "read-only":
                    entry["role"] = "input"
                elif arg_io == "write-only":
                    entry["role"] = "output"
                elif arg_io == "read-write":
                    entry["role"] = "rw_buf"
                else:
                    # Fallback for unexpected/unknown IO annotations: treat as
                    # an internal read-write buffer so that at least the kernel
                    # call is well-formed and the wrapper generation does not
                    # crash with missing 'role'.
                    entry["role"] = "rw_buf"
            classified.append(entry)

        # Collect exposed args (inputs/outputs only) preserving original order
        exposed_args = [c for c in classified if c["role"] in ("input", "output")]

        out = []
        emit = out.append

        # --------------------------------------------------------------
        # 2. Emit headers
        # --------------------------------------------------------------
        emit("// Auto-generated by ST-Flow\n")
        emit("#include <iostream>")
        emit("#include <fstream>")
        emit("#include <vector>")
        emit("#include <pybind11/numpy.h>")
        emit("#include <pybind11/pybind11.h>")
        if self.platform == "tapa":
            emit("#include <tapa.h>")
        for header in self.headers:
            emit(f'#include "{header}"')
        emit("\nnamespace py = pybind11;\n")

        # --------------------------------------------------------------
        # 3. Generate function signature: preserve original order
        # --------------------------------------------------------------
        emit(f"void {self.lib_name}(")
        sig_parts = []
        for idx, c in enumerate(exposed_args):
            i = c["index"]
            ele = c["ele_type"]
            resolved = allo2c_type[ele]
            is_scalar = len(c["shape"]) == 0
            if is_scalar:
                sig_parts.append(f"  {resolved} arg{i}")
            else:
                sig_parts.append(f"  py::array_t<{resolved}> &arg{i}")
        for idx, part in enumerate(sig_parts):
            comma = "," if idx < len(sig_parts) - 1 else ""
            emit(f"{part}{comma}")
        emit(") {")

        # --------------------------------------------------------------
        # 4. Process input/output buffers
        # --------------------------------------------------------------
        emit("")

        # Process array arguments (request buffers)
        for c in exposed_args:
            if len(c["shape"]) == 0:  # Skip scalars
                continue
            i = c["index"]
            emit(f"  py::buffer_info buf{i} = arg{i}.request();")

        emit("")

        ptr_vars = {}  # index → variable name passed to module_top

        # Process exposed arguments (arrays and scalars)
        for c in exposed_args:
            i = c["index"]
            ele = c["ele_type"]
            resolved = allo2c_type[ele]
            shape = c["shape"]
            
            if len(shape) == 0:
                # Scalar: pass directly
                ptr_vars[i] = f"arg{i}"
            else:
                # Array: process buffer
                emit(f"  {resolved} *p_arg{i} = ({resolved}*) buf{i}.ptr;")
                if len(shape) <= 1:
                    ptr_vars[i] = f"p_arg{i}"
                else:
                    tail = "[" + "][".join(str(s) for s in shape[1:]) + "]"
                    emit(f"  {resolved} (*p_arg{i}_nd){tail} = "
                        f"reinterpret_cast<{resolved} (*){tail}>(p_arg{i});")
                    ptr_vars[i] = f"p_arg{i}_nd"

        # --------------------------------------------------------------
        # 5. Handle weights: load from binary files based on arg_name
        # --------------------------------------------------------------
        emit("\n  // Load weights")
        for c in classified:
            if c["role"] != "weight":
                continue

            i = c["index"]
            ele = c["ele_type"]
            resolved = allo2c_type[ele]
            shape = c["shape"]
            arg_name = c["name"]

            # Scalars can't be weights, so shape should always be non-empty
            if len(shape) == 0:
                raise RuntimeError(f"[ip.py] Weight argument {i} ({arg_name}) cannot be a scalar")

            total_elems = 1
            for x in shape:
                total_elems *= x

            emit(f"  std::vector<{resolved}> weight_{i}({total_elems});")
            emit(f'  std::ifstream fin_{i}("{self.abs_path}/weight_bins/{arg_name}.bin", std::ios::binary);')
            emit(f"  fin_{i}.read(reinterpret_cast<char*>(weight_{i}.data()), "
                f"{total_elems} * sizeof({resolved}));")
            emit(f"  {resolved} *p_arg{i} = weight_{i}.data();")

            if len(shape) <= 1:
                ptr_vars[i] = f"p_arg{i}"
            else:
                tail = "[" + "][".join(str(s) for s in shape[1:]) + "]"
                emit(f"  {resolved} (*p_arg{i}_nd){tail} = "
                    f"reinterpret_cast<{resolved} (*){tail}>(p_arg{i});")
                ptr_vars[i] = f"p_arg{i}_nd"

        # --------------------------------------------------------------
        # 6. Read-write buffers: allocate empty buffers internally
        # --------------------------------------------------------------
        emit("\n  // Allocate read-write buffers")
        for c in classified:
            if c["role"] != "rw_buf":
                continue
            
            # Skip scalars - they're already handled above
            if len(c["shape"]) == 0:
                continue

            i = c["index"]
            ele = c["ele_type"]
            resolved = allo2c_type[ele]
            shape = c["shape"]

            total_elems = 1
            for x in shape:
                total_elems *= x

            emit(f"  std::vector<{resolved}> rwbuf_{i}({total_elems});")
            emit(f"  {resolved} *p_arg{i} = rwbuf_{i}.data();")

            if len(shape) <= 1:
                ptr_vars[i] = f"p_arg{i}"
            else:
                tail = "[" + "][".join(str(s) for s in shape[1:]) + "]"
                emit(f"  {resolved} (*p_arg{i}_nd){tail} = "
                    f"reinterpret_cast<{resolved} (*){tail}>(p_arg{i});")
                ptr_vars[i] = f"p_arg{i}_nd"

        # --------------------------------------------------------------
        # 7. Construct call to module_top preserving original arg order
        # --------------------------------------------------------------
        emit("\n  // Call kernel")
        ordered_args = ", ".join(ptr_vars[i] for i in range(len(classified)))
        emit(f"  {self.top}({ordered_args});")

        emit("}\n")

        # --------------------------------------------------------------
        # 8. Pybind module
        # --------------------------------------------------------------
        emit(f"PYBIND11_MODULE({self.lib_name}, m) {{")
        # Build argument list for pybind def (preserve order)
        pybind_args = []
        for c in exposed_args:
            idx = c["index"]
            pybind_args.append(f'py::arg("arg{idx}")')
        if pybind_args:
            sig_str = ", " + ", ".join(pybind_args)
        else:
            sig_str = ""
        emit(f'  m.def("{self.top}", &{self.lib_name}{sig_str}, "{self.top} wrapper");')
        emit("}")

        # Write file
        with open(self.c_wrapper_file, "w") as f:
            f.write("\n".join(out))

        return self.c_wrapper_file

    def generate_pybind11_wrapper(self):
        # --------------------------------------------------------------
        # Helpers to parse packed + original_layout from args_with_attrs
        # args_with_attrs entries are tuples:
        #   (dtype, shape, name, kind, io)
        #
        # After DMA rewrite, dtype/shape are the packed memref element + shape,
        # but original_layout is present in MLIR; python side doesn't see it.
        #
        # Therefore we must reconstruct "logical" shapes using the naming rules:
        #   - for off-chip read-only:   __host_pack_<name>(logical, packed)
        #   - for off-chip write-only:  __host_unpack_<name>(packed, logical)
        #   - weights: load bin name=<name>, then pack via __host_pack_<name>
        #
        # We will use a conservative approach:
        #   logical shape comes from the *original* pipeline inputs (what python passes)
        #   -> that's exactly what you already have in args_with_attrs BEFORE DMA.
        #
        # In your current setup you are calling build_hls(top_dma,...)
        # but StreamHLSModule.args_with_attrs comes from get_func_args_with_attrs(self.func),
        # which for dma_top.mlir contains packed types/shapes and "name/kind/io".
        #
        # To get logical shapes for pack/unpack we need original_layout.
        #
        # Minimal change: extend get_func_args_with_attrs to also return original_layout.shape
        # as arg_attrs[...]['buffer.info']['original_layout']['shape'] if present.
        #
        # This generator expects args_with_attrs entries to be:
        #   (packed_ele_dtype_str, packed_shape, name, kind, io, logical_shape or None)
        # --------------------------------------------------------------

        if self.platform != "tapa":
            # keep your existing generator for non-tapa
            return self._generate_pybind11_wrapper_legacy()

        # ---- classify ----
        classified = []
        for i, tup in enumerate(self.args_with_attrs):
            # Expected new tuple shape: (dtype, shape, name, kind, io, logical_shape)
            if len(tup) == 5:
                raise RuntimeError(
                    "[ip.py] For tapa DMA wrapper, args_with_attrs must include logical_shape "
                    "(from buffer.info.original_layout.shape)."
                )
            ele_type, packed_shape, arg_name, arg_kind, arg_io, logical_shape = tup
            entry = {
                "index": i,
                "packed_ele_type": ele_type,        # e.g. "vector<4xf32>" or "f32"
                "packed_shape": packed_shape,       # e.g. (4096,) or (8192,)
                "logical_shape": tuple(logical_shape) if logical_shape else tuple(),
                "name": arg_name,
                "kind": arg_kind,                   # "off-chip" or "weights"
                "io": arg_io,                       # "read-only"/"write-only"/"read-write"
            }
            if arg_kind == "weights":
                entry["role"] = "weight"
            else:
                if arg_io == "read-only":
                    entry["role"] = "input"
                elif arg_io == "write-only":
                    entry["role"] = "output"
                else:
                    entry["role"] = "rw_buf"
            classified.append(entry)

        # Python-visible args: inputs + outputs only (off-chip)
        io_args = [c for c in classified if c["role"] in ("input", "output")]

        out = []
        emit = out.append

        # ---- headers ----
        emit("// Auto-generated by ST-Flow\n")
        emit("#include <pybind11/pybind11.h>")
        emit("#include <pybind11/numpy.h>")
        emit("#include <pybind11/stl.h>")
        emit("")
        emit("#include <tapa.h>")
        emit("#include <hls_vector.h>")
        emit("")
        emit("#include <vector>")
        emit("#include <string>")
        emit("#include <fstream>")
        emit("#include <stdexcept>")
        emit("")
        for header in self.headers:
            emit(f'#include "{header}"')
        # If dma_host_funcs.cpp is compiled as TU, no need to include it here.
        # If you converted to header, include it:
        # emit(f'#include "{self.project}/dma_host_funcs.h"')
        emit("")
        emit("namespace py = pybind11;")
        emit("")
        emit("static void __allo_load_bin(const std::string& path, void* dst, size_t nbytes) {")
        emit("  std::ifstream fin(path, std::ios::binary);")
        emit('  if (!fin) throw std::runtime_error("failed to open: " + path);')
        emit("  fin.read(reinterpret_cast<char*>(dst), nbytes);")
        emit('  if (!fin) throw std::runtime_error("failed to read: " + path);')
        emit("}")
        emit("")

        # ---- function signature ----
        emit(f"void {self.lib_name}(")
        for idx, c in enumerate(io_args):
            # only float supported in your current example; keep mapping anyway
            resolved = allo2c_type.get("float32", "float")  # logical tensors are f32 in your flow
            comma = "," if idx < len(io_args) - 1 else ","
            emit(f"  py::array_t<{resolved}, py::array::c_style | py::array::forcecast> arg{c['index']}{comma}")
        emit('  const std::string& bitstream = ""')
        emit(") {")
        emit("")

        # ---- request buffers + logical pointer casts ----
        for c in io_args:
            i = c["index"]
            emit(f"  auto buf{i} = arg{i}.request();")
        emit("")

        # logical ptr vars (based on logical_shape)
        for c in io_args:
            i = c["index"]
            lshape = c["logical_shape"]
            emit(f"  float* p_arg{i} = reinterpret_cast<float*>(buf{i}.ptr);")
            if len(lshape) <= 1:
                emit(f"  auto* p_arg{i}_nd = p_arg{i};")
            else:
                tail = "[" + "][".join(str(s) for s in lshape[1:]) + "]"
                emit(f"  auto* p_arg{i}_nd = reinterpret_cast<float (*){tail}>(p_arg{i});")
        emit("")

        # ---- allocate packed buffers for ALL args ----
        # map MLIR packed ele type string -> C++ element type
        def packed_elem_cpp(packed_ele_type: str) -> str:
            # examples:
            #  - "vector<4xf32>" -> "hls::vector<float, 4>"
            #  - "vector<8xf32>" -> "hls::vector<float, 8>"
            #  - "f32" -> "float"
            t = packed_ele_type.strip()
            if t.startswith("vector<") and "xf32" in t:
                n = t.split("<")[1].split("x")[0]  # "4" in vector<4xf32>
                return f"hls::vector<float, {n}>"
            if t == "f32":
                return "float"
            raise RuntimeError(f"[ip.py] Unsupported packed element type: {packed_ele_type}")

        for c in classified:
            i = c["index"]
            elem_cpp = packed_elem_cpp(c["packed_ele_type"])
            # packed_shape is typically (N,) for memref<Nx...>
            if len(c["packed_shape"]) != 1:
                raise RuntimeError(f"[ip.py] Expected 1D packed memref for arg{i}, got {c['packed_shape']}")
            N = int(c["packed_shape"][0])
            emit(f"  std::vector<{elem_cpp}> p_packed_{i}({N});")
        emit("")

        # ---- pack inputs ----
        for c in classified:
            if c["role"] != "input":
                continue
            i = c["index"]
            name = c["name"]
            emit(f"  // pack input: {name}")
            emit(f"  __host_pack_{name}(p_arg{i}_nd, p_packed_{i}.data());")
            emit("")

        # ---- load + pack weights ----
        for c in classified:
            if c["role"] != "weight":
                continue
            i = c["index"]
            name = c["name"]
            lshape = c["logical_shape"]
            if not lshape:
                raise RuntimeError(f"[ip.py] weight {name} missing logical_shape (original_layout.shape)")
            total = 1
            for d in lshape:
                total *= int(d)
            emit(f"  // load + pack weight: {name}")
            emit(f"  std::vector<float> w_{i}({total});")
            emit(f'  __allo_load_bin("{self.abs_path}/weight_bins/{name}.bin", w_{i}.data(), w_{i}.size() * sizeof(float));')
            if len(lshape) <= 1:
                emit(f"  __host_pack_{name}(w_{i}.data(), p_packed_{i}.data());")
            else:
                tail = "[" + "][".join(str(s) for s in lshape[1:]) + "]"
                emit(f"  auto* w_{i}_nd = reinterpret_cast<float (*){tail}>(w_{i}.data());")
                emit(f"  __host_pack_{name}(w_{i}_nd, p_packed_{i}.data());")
            emit("")

        # ---- invoke ----
        emit("  // invoke kernel (csim if bitstream == \"\")")
        emit("  tapa::invoke(")
        emit(f"    {self.top}, bitstream,")

        invoke_terms = []
        for c in classified:
            i = c["index"]
            elem_cpp = packed_elem_cpp(c["packed_ele_type"])
            var = f"p_packed_{i}"
            if c["role"] in ("input", "weight"):
                invoke_terms.append(f"    tapa::read_only_mmap<{elem_cpp}>({var})")
            elif c["role"] == "output":
                invoke_terms.append(f"    tapa::write_only_mmap<{elem_cpp}>({var})")
            else:
                invoke_terms.append(f"    tapa::read_write_mmap<{elem_cpp}>({var})")

        # join with commas
        for k, t in enumerate(invoke_terms):
            comma = "," if k < len(invoke_terms) - 1 else ""
            emit(t + comma)
        emit("  );")
        emit("")

        # ---- unpack outputs ----
        for c in classified:
            if c["role"] != "output":
                continue
            i = c["index"]
            name = c["name"]
            emit(f"  // unpack output: {name}")
            emit(f"  __host_unpack_{name}(p_packed_{i}.data(), p_arg{i}_nd);")
            emit("")

        emit("}")
        emit("")

        # ---- pybind module ----
        emit(f"PYBIND11_MODULE({self.lib_name}, m) {{")
        sig = ", ".join([f'py::arg("arg{c["index"]}")' for c in io_args] + ['py::arg("bitstream") = ""'])
        emit(f"  m.def(\"{self.top}\", &{self.lib_name}, {sig});")
        emit("}")

        with open(self.c_wrapper_file, "w") as f:
            f.write("\n".join(out))

        return self.c_wrapper_file


    def compile_pybind11(self):
        # Avoid recompiling if the shared library already exists.
        import sysconfig
        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ""
        built_so = os.path.join(self.temp_path, f"{self.lib_name}{ext_suffix}")

        self.generate_pybind11_wrapper()
        cmd = "g++ -shared -std=c++17 -fPIC"
        cmd += " `python3 -m pybind11 --includes` "
        cmd += " ".join(
            ["-I" + (path if path != "" else ".") for path in self.include_paths]
        )
        srcs = self.impls + [self.c_wrapper_file]
        cmd += " " + " ".join(srcs)
        cmd += (
            f" -o {self.temp_path}/{self.lib_name}`python3-config --extension-suffix`"
        )
        if self.platform == "tapa":
            cmd += " -ltapa -lfrt -lglog -lgflags"#" -lcontext -lthread -ltinyxml2 -lOpenCL"
        with open(os.path.join(self.temp_path, "build_cmd.sh"), "w", encoding="utf-8") as f:
            f.write(cmd + "\n")
        logger.info(cmd)
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            logger.error("=== g++ failed with output ===")
            logger.error(exc.stdout)
            raise RuntimeError(
                f"Failed to compile pybind wrapper for {self.lib_name}!"
            ) from exc
        return built_so

    def generate_mlir_c_wrapper(self):
        out_str = "// Auto-generated by ST-Flow\n\n"
        # Add headers
        out_str += "#include <iostream>\n"
        out_str += '#include "mlir/ExecutionEngine/CRunnerUtils.h"\n'
        for header in self.headers:
            out_str += f'#include "{header}"\n'
        out_str += "\n"
        # Generate function interface
        unranked_memrefs = []
        for i, (arg_type, arg_shape) in enumerate(self.args):
            if len(arg_shape) > 0:
                unranked_memrefs.append(f"int64_t rank_{i}, void *ptr_{i}")
            else:
                unranked_memrefs.append(f"{allo2c_type.get(arg_type)} in{i}")
        unranked_memrefs_str = ", ".join(unranked_memrefs)
        out_str += f'extern "C" void {self.lib_name}({unranked_memrefs_str}) {{\n'
        in_ptrs = []
        for i, (arg_type, arg_shape) in enumerate(self.args):
            if len(arg_shape) == 0:  # scalar
                in_ptrs.append(f"in{i}")
                continue
            resolved_type = allo2c_type.get(arg_type)
            out_str += f"  UnrankedMemRefType<{resolved_type}> in{i} = {{rank_{i}, ptr_{i}}};\n"
            out_str += f"  DynamicMemRefType<{resolved_type}> ranked_in{i}(in{i});\n"
            out_str += f"  {resolved_type} *in{i}_ptr = ({resolved_type} *)ranked_in{i}.data;\n"
            if len(arg_shape) == 1:
                in_ptrs.append(f"in{i}_ptr")
            else:
                tail_shape = "[" + "][".join([str(s) for s in arg_shape[1:]]) + "]"
                out_str += f"  {resolved_type} (*in{i}_nd){tail_shape} = "
                out_str += (
                    f"reinterpret_cast<{resolved_type} (*){tail_shape}>(in{i}_ptr);\n"
                )
                in_ptrs.append(f"in{i}_nd")
        # Call library function
        out_str += f"  {self.top}({', '.join(in_ptrs)});\n"
        out_str += "}\n"
        with open(self.c_wrapper_file, "w", encoding="utf-8") as f:
            f.write(out_str)
        return self.c_wrapper_file

    def compile_shared_lib(self):
        # Used in direct function call in an ST-Flow kernel
        self.generate_mlir_c_wrapper()
        if os.system("which llvm-config >> /dev/null") != 0:
            raise RuntimeError("Please install LLVM and add it to your PATH")
        cmd = "g++ -c -std=c++14 -fpic "
        # suppose the build directory is under llvm-project
        self.include_paths.append(
            "/".join(os.popen("which llvm-config").read().split("/")[:-3])
            + "/mlir/include"
        )
        cmd += " ".join(
            ["-I" + (path if path != "" else ".") for path in self.include_paths]
        )
        srcs = self.impls + [self.c_wrapper_file]
        obj_files = []
        for src in srcs:
            subcmd = cmd
            subcmd += " " + src
            obj = f"{self.temp_path}/{src.split('/')[-1]}.o"
            subcmd += " -o " + obj
            logger.info(subcmd)
            try:
                subprocess.check_output(subcmd, shell=True)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"Failed to compile {src.split('/')[-1]}.o!"
                ) from exc
            obj_files.append(obj)
        cmd = f"g++ -shared -o {self.temp_path}/lib{self.top}.so " + " ".join(obj_files)
        logger.info(cmd)
        try:
            subprocess.check_output(cmd, shell=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Failed to compile {self.top}.so!") from exc
        return f"{self.temp_path}/lib{self.top}.so"
    def csim(self, *args):
        self.compile_pybind11()
        sys.path.append(self.temp_path)
        logger.info(self.lib_name)
        self.lib = importlib.import_module(f"{self.lib_name}")
        return getattr(self.lib, f"{self.top}")(*args)
    
    def __call__(self, *args):
        built_so = self.compile_pybind11()
        sys.path.append(self.temp_path)
        # Reuse already-imported module if present.
        if f"{self.lib_name}" in sys.modules:
            self.lib = sys.modules[f"{self.lib_name}"]
        else:
            self.lib = importlib.import_module(f"{self.lib_name}")
        return getattr(self.lib, f"{self.top}")(*args)
