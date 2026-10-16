from ._mlir.ir import (
    MemRefType,
    RankedTensorType,
    IntegerType,
    IndexType,
    F16Type,
    F32Type,
    F64Type,
    BF16Type,
    VectorType
)
from ._mlir.dialects import (
    func as func_d,
    streamhls as streamhls_d
)

import numpy as np
import ml_dtypes
import textwrap
from contextlib import contextmanager


import torch
import torch.nn as nn
from torch_mlir import fx as fx_importer
import torch_mlir

from ._mlir.ir import (
    Context,
    Location,
    Module,
    UnitAttr,
)
import re
INDENT = -1

np_supported_types = {
    "bf16": ml_dtypes.bfloat16,
    "f16": np.float16,
    "f32": np.float32,
    "f64": np.float64,
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
    "i64": np.int64,
    "ui1": np.bool_,
    "ui8": np.uint8,
    "ui16": np.uint16,
    "ui32": np.uint32,
    "ui64": np.uint64,
}


def invoke_mlir_parser(mod: str):
    with Context() as ctx, Location.unknown():
        streamhls_d.register_dialect(ctx)
        module = Module.parse(str(mod), ctx)
    return module

def format_str(s, indent=4, strip=True):
    if INDENT != -1:
        # global context
        indent = INDENT
    if strip:
        return textwrap.indent(textwrap.dedent(s).strip(), " " * indent) + "\n"
    return textwrap.indent(s, " " * indent) + "\n"

def find_func_in_module(module, func_name):
    for op in module.body.operations:
        if isinstance(op, func_d.FuncOp) and op.name.value == func_name:
            return op
    return None

def get_clostest_pow2(n):
    # .bit_length() is a Python method
    return 1 << (n - 1).bit_length()
def get_dtype_and_shape_from_type(dtype):
    # memref<...>
    if MemRefType.isinstance(dtype):
        dtype = MemRefType(dtype)
        shape = dtype.shape
        ele_type, _ = get_dtype_and_shape_from_type(dtype.element_type)
        return ele_type, shape

    # tensor<...>
    if RankedTensorType.isinstance(dtype):
        dtype = RankedTensorType(dtype)
        shape = dtype.shape
        ele_type, _ = get_dtype_and_shape_from_type(dtype.element_type)
        return ele_type, shape

    # vector<...>
    if VectorType.isinstance(dtype):
        vt = VectorType(dtype)
        # Return a stable string encoding that your downstream can parse.
        # Example: "vector<4xf32>" or "vector<8xf32>"
        elem_str, _ = get_dtype_and_shape_from_type(vt.element_type)
        # vt.shape is the vector lane shape, usually (4,) or (8,)
        lanes = 1
        for d in vt.shape:
            lanes *= int(d)
        return f"vector<{lanes}x{elem_str}>", tuple()

    # scalars
    if IndexType.isinstance(dtype):
        return "index", tuple()
    if IntegerType.isinstance(dtype):
        return str(IntegerType(dtype)), tuple()
    if F16Type.isinstance(dtype):
        return str(F16Type(dtype)), tuple()
    if F32Type.isinstance(dtype):
        return str(F32Type(dtype)), tuple()
    if F64Type.isinstance(dtype):
        return str(F64Type(dtype)), tuple()
    if BF16Type.isinstance(dtype):
        return str(BF16Type(dtype)), tuple()

    raise RuntimeError(f"Unsupported type: {dtype}")


def get_signed_type_by_hint(dtype, hint):
    if hint == "u" and (dtype.startswith("i") or dtype.startswith("fixed")):
        return "u" + dtype
    return dtype

def get_func_inputs_outputs_with_attrs(func):
    inputs = []
    outputs = []
    inputs_outputs = []
    arg_attrs = None
    for attr_p in func.attributes:
        if attr_p.name == "arg_attrs":
            arg_attrs = attr_p.attr
    if not arg_attrs: 
        raise RuntimeError("No arg_attrs found in function attributes")
    
    for i, (arg, arg_attr) in enumerate(zip(func.arguments, arg_attrs)):
        dtype, shape = get_dtype_and_shape_from_type(arg.type)
        buf_info = arg_attr['buffer.info']
        name = buf_info['name'].value
        kind = buf_info['type'].value
        io = buf_info['io'].value
        if io == 'read-only':
            inputs.append((dtype, shape, name, kind))
        elif io == 'write-only':
            outputs.append((dtype, shape, name, kind))
        elif io == 'read-write':  
            inputs_outputs.append((dtype, shape, name, kind))
        elif io == 'unknown': # TODO to remove
            inputs_outputs.append((dtype, shape, name, kind))
        else:
            raise RuntimeError(f"Unsupported io kind: {io}")
    return inputs, outputs, inputs_outputs

# def get_func_args_with_attrs(func):
#     args = []
#     arg_attrs = None
#     for attr_p in func.attributes:
#         if attr_p.name == "arg_attrs":
#             arg_attrs = attr_p.attr
#     if not arg_attrs: 
#         raise RuntimeError("No arg_attrs found in function attributes")
    
#     for i, (arg, arg_attr) in enumerate(zip(func.arguments, arg_attrs)):
#         dtype, shape = get_dtype_and_shape_from_type(arg.type)
#         buf_info = arg_attr['buffer.info']
#         name = buf_info['name'].value
#         kind = buf_info['type'].value
#         io = buf_info['io'].value
#         args.append((dtype, shape, name, kind, io))
#     return args
def get_func_args_with_attrs(func):
    args = []
    arg_attrs = None
    for attr_p in func.attributes:
        if attr_p.name == "arg_attrs":
            arg_attrs = attr_p.attr
    if not arg_attrs:
        raise RuntimeError("No arg_attrs found in function attributes")

    for i, (arg, arg_attr) in enumerate(zip(func.arguments, arg_attrs)):
        dtype, shape = get_dtype_and_shape_from_type(arg.type)

        buf_info = arg_attr["buffer.info"]
        name = buf_info["name"].value
        kind = buf_info["type"].value
        io = buf_info["io"].value

        logical_shape = None
        if "original_layout" in buf_info:
            ol = buf_info["original_layout"]
            if "shape" in ol:
                logical_shape = [int(x) for x in ol["shape"]]
        if logical_shape is None:
            logical_shape = list(shape)

        args.append((dtype, shape, name, kind, io, logical_shape))
    return args

# -------------------------
#  Torch-MLIR Utilities
# -------------------------
def _map_output_type_fx(s: str):
    """Map user string to FX OutputType enum."""
    table = {
        "linalg-on-tensors": fx_importer.OutputType.LINALG_ON_TENSORS,
        "tosa": fx_importer.OutputType.TOSA,
        "raw": fx_importer.OutputType.RAW,
    }
    return table.get(s, fx_importer.OutputType.LINALG_ON_TENSORS)


def _map_output_type_ts(s: str):
    """Map user string to TorchScript OutputType enum."""
    table = {
        "linalg-on-tensors": torch_mlir.torchscript.OutputType.LINALG_ON_TENSORS,
        "tosa": torch_mlir.torchscript.OutputType.TOSA,
        "raw": torch_mlir.torchscript.OutputType.RAW,
    }
    return table.get(s, torch_mlir.torchscript.OutputType.LINALG_ON_TENSORS)


def convert_torch_to_mlir_fx(model: nn.Module, inputs, output_type="linalg-on-tensors"):
    """
    Preferred path for new torch-mlir:
      torch.export.export(...)  →  torch_mlir.fx.export_and_import(...)
    Returns MLIR Module Op in text form.
    """
    model.eval()
    # torch.export requires real tensors (not FakeTensors)
    exported = torch.export.export(model, tuple(inputs))
    mlir_module = fx_importer.export_and_import(
        exported,
        func_name="forward",
        output_type=_map_output_type_fx(output_type),
    )
    return mlir_module.operation.get_asm()


def convert_torch_to_mlir_torchscript(model: nn.Module, inputs, output_type="linalg-on-tensors"):
    """
    Fallback path when FX export struggles: TorchScript trace → torch_mlir.torchscript.compile
    """
    model.eval()
    with torch.no_grad():
        traced = torch.jit.trace(model, tuple(inputs), strict=False)
        traced = torch.jit.freeze(traced)
    mlir_module = torch_mlir.torchscript.compile(
        traced,
        tuple(inputs),
        output_type=_map_output_type_ts(output_type),
    )
    return mlir_module.operation.get_asm()

import re
def _sanitize_ident(s: str) -> str:
    # valid C/C++ identifier-ish: letters/digits/underscore only
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    if re.match(r"^[0-9]", s):
        s = "_" + s
    return s

def get_bitstream_report(prj_path):
    import csv
    import os
    import re

    # 1. Resource utilization (prj/build/link/imp/impl_1_kernel_util_*.rpt)
    rpt_fallbacks = [
        ("impl_1_kernel_util_routed.rpt", "routed"),
        ("impl_1_kernel_util_placed.rpt", "placed"),
        ("impl_1_kernel_util_synthed.rpt", "synth")
    ]
    util_dir = os.path.join(prj_path, "build/link/imp")
    
    info = {
        "lut_abs": -1, "lut_pct": -1,
        "reg_abs": -1, "reg_pct": -1,
        "bram_abs": -1, "bram_pct": -1,
        "uram_abs": -1, "uram_pct": -1,
        "dsp_abs": -1, "dsp_pct": -1,
        "bitstream_freq_mhz": -1,
        "report_type": "none"
    }

    # Search for utilization report
    rpt_path = ""
    for rpt, rtype in rpt_fallbacks:
        p = os.path.join(util_dir, rpt)
        if os.path.exists(p):
            rpt_path = p
            info["report_type"] = rtype
            break
    
    # Check if xclbin exists to upgrade status to 'bitstream'
    # Actually, we should check if the .xclbin.info exists
    build_dir = os.path.join(prj_path, "build")
    xclbin_info_path = ""
    if os.path.exists(build_dir):
        for f in os.listdir(build_dir):
            if f.endswith(".xclbin.info"):
                xclbin_info_path = os.path.join(build_dir, f)
                info["report_type"] = "bitstream"
                break

    # Parse details if rpt_path exists
    if rpt_path:
        with open(rpt_path, "r") as f:
            for line in f:
                if "| module_top" in line:
                    parts = [p.strip() for p in line.split("|")]
                    def parse_cell(cell):
                        # Extract "34046" and "2.90"
                        m = re.match(r"(\d+)\s*\[\s*([\d\.]+)%\]", cell)
                        if m:
                            return int(m.group(1)), float(m.group(2))
                        return -1, -1

                    info["lut_abs"], info["lut_pct"] = parse_cell(parts[2])
                    info["reg_abs"], info["reg_pct"] = parse_cell(parts[4])
                    info["bram_abs"], info["bram_pct"] = parse_cell(parts[5])[0]*2, parse_cell(parts[5])[1]
                    info["uram_abs"], info["uram_pct"] = parse_cell(parts[6])
                    info["dsp_abs"], info["dsp_pct"] = parse_cell(parts[7])
                    break

    # Parse frequency
    if xclbin_info_path and os.path.exists(xclbin_info_path):
        with open(xclbin_info_path, "r") as f:
            content = f.read()
            # Look for DATA_CLK frequency
            match = re.search(r"Name:\s+DATA_CLK.*?Frequency:\s+(\d+)\s+MHz", content, re.DOTALL)
            if match:
                info["bitstream_freq_mhz"] = int(match.group(1))

    # 4. Latency (prj/reports/bitstream_latency.txt)
    info["bitstream_latency_ms"] = -1
    info["bitstream_latency_cycles"] = -1
    latency_file = os.path.join(prj_path, "reports/bitstream_latency.txt")
    if os.path.exists(latency_file):
        with open(latency_file, "r") as f:
            try:
                latency_ms = float(f.read().strip())
                info["bitstream_latency_ms"] = latency_ms
                
                # Calculate cycles: Cycles = MHz * ms * 1000
                if info["bitstream_freq_mhz"] != -1:
                    info["bitstream_latency_cycles"] = int(latency_ms * info["bitstream_freq_mhz"] * 1000)
            except ValueError:
                pass

    # Save to CSV
    reports_dir = os.path.join(prj_path, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    csv_path = os.path.join(reports_dir, "bitstream.csv")
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=info.keys())
        writer.writeheader()
        writer.writerow(info)
        
    return info

def reset_fpga():
    """Resets the FPGA using xbutil."""
    import subprocess
    import logging
    logger = logging.getLogger(__name__)
    
    # Standard command for Xilinx 2021.1+ (XRT 2.9+)
    # xbutil reset --device <device> --force
    cmd = ["xbutil", "reset", "-d", "--force"]
    
    logger.info("Resetting FPGA using command: %s", " ".join(cmd))
    try:
        # Run the command with a timeout to prevent hanging if xbutil itself deadlocks
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            logger.info("FPGA reset successfully.")
        else:
            logger.warning("FPGA reset failed (return code %d): %s", result.returncode, result.stderr.strip())
    except subprocess.TimeoutExpired:
        logger.error("FPGA reset timed out after 30 seconds.")
    except FileNotFoundError:
        logger.warning("xbutil command not found. Skipping FPGA reset.")
    except Exception as e:
        logger.error("Unexpected error during FPGA reset: %s", str(e))