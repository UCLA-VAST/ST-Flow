import os

SRC_DIR = "src"
PYBIND_DIR = "pybind"

def get_tool_src_dir(project_dir: str, tool: str) -> str:
    return os.path.join(project_dir, tool, SRC_DIR)

def get_tool_pybind_dir(project_dir: str, tool: str) -> str:
    return os.path.join(project_dir, tool, PYBIND_DIR)
