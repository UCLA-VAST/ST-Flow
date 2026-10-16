from dataclasses import dataclass
from typing import List, Tuple
from .._mlir.ir import (
    MemRefType,
    VectorType,
    F32Type,
    IntegerType
)
from ..utils import get_dtype_and_shape_from_type

@dataclass
class ArgInfo:
    index: int
    name: str                 # buffer.info.name
    kind: str                 # buffer.info.type: "off-chip" | "weights" | ...
    io: str                   # buffer.info.io: read-only/write-only/read-write
    role: str                 # "input"|"output"|"weight"|"rw_buf"
    packed_depth: int         # product(memref shape) — on-device layout
    packed_elem_cpp: str      # "float" or "Vector<float, W>"
    original_shape: Tuple[int, ...]  # buffer.info.original_layout.shape (host / logical tensor)
    padded_shape: Tuple[int, ...]    # buffer.info.padded_layout.shape (host view for DMA pack/unpack)
    physical_shape: Tuple[int, ...]  # actual memref shape from MLIR type
    logical_elem: str = "f32"        # usually "f32" (currently assumed)

    @property
    def logical_shape(self) -> Tuple[int, ...]:
        """Backward-compatible alias for original_layout shape."""
        return self.original_shape

def _mlir_type_to_cpp_packed_elem(t) -> str:
    # Supports f32 and vector<Wx f32>
    if VectorType.isinstance(t):
        vt = VectorType(t)
        if not F32Type.isinstance(vt.element_type):
            raise RuntimeError(f"Only vector<...xf32> supported, got: {vt}")
        lanes = 1
        for d in vt.shape:
            lanes *= int(d)
        return f"Vector<float, {lanes}>"
    # scalar
    if F32Type.isinstance(t):
        return "float"
    elif IntegerType.isinstance(t) and t.width == 64:
        return "int64_t"
    elif IntegerType.isinstance(t) and t.width == 32:
        return "int32_t"
    return "float" # Fallback

def collect_dma_arg_info(func) -> List[ArgInfo]:
    arg_infos = []
    arg_attrs = None
    for attr in func.attributes:
        if attr.name == "arg_attrs":
            arg_attrs = attr.attr
            break
    if not arg_attrs:
        raise RuntimeError("No arg_attrs found in top function")

    for i, (arg, attrs) in enumerate(zip(func.arguments, arg_attrs)):
        if "buffer.info" not in attrs:
            continue
        buf_info = attrs["buffer.info"]
        
        name = buf_info["name"].value
        kind = buf_info["type"].value
        io = buf_info["io"].value
        
        dtype_str, physical_shape = get_dtype_and_shape_from_type(arg.type)

        original_shape = None
        if "original_layout" in buf_info:
            ol = buf_info["original_layout"]
            if "shape" in ol:
                original_shape = tuple(int(x) for x in ol["shape"])
        if original_shape is None:
            original_shape = physical_shape

        padded_shape = None
        if "padded_layout" in buf_info:
            pl = buf_info["padded_layout"]
            if "shape" in pl:
                padded_shape = tuple(int(x) for x in pl["shape"])
        if padded_shape is None:
            padded_shape = original_shape

        if MemRefType.isinstance(arg.type):
            mt = MemRefType(arg.type)
            packed_elem_cpp = _mlir_type_to_cpp_packed_elem(mt.element_type)
        else:
            packed_elem_cpp = _mlir_type_to_cpp_packed_elem(arg.type)
        packed_depth = 1
        for d in physical_shape:
            packed_depth *= int(d)

        role = "rw_buf"
        if kind == "weights":
            role = "weight"
        elif io == "read-only":
            role = "input"
        elif io == "write-only":
            role = "output"
        # print(
        #     f"Arg {i}: original_shape: {original_shape}, padded_shape: {padded_shape}, "
        #     f"physical_shape: {physical_shape}"
        # )
        arg_infos.append(ArgInfo(
            index=i,
            name=name,
            kind=kind,
            io=io,
            role=role,
            packed_depth=packed_depth,
            packed_elem_cpp=packed_elem_cpp,
            original_shape=original_shape,
            padded_shape=padded_shape,
            physical_shape=physical_shape,
            logical_elem="f32",
        ))
    return arg_infos
