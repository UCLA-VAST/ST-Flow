# Copyright ST-Flow authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import subprocess
import shutil
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools import find_packages


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        # Ensure CMake is installed
        try:
            subprocess.check_call(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        # Call the build process for each extension
        for ext in self.extensions:
            print(ext)
            self.build_extension(ext)

    def build_extension(self, ext):
        # Retrieve MLIR_DIR and LLVM_DIR from environment
        mlir_dir = os.environ.get("MLIR_DIR")
        llvm_dir = os.environ.get("LLVM_DIR")
        
        # If not set, try to derive from LLVM_BUILD_DIR (legacy behavior)
        if not mlir_dir or not llvm_dir:
            llvm_build_dir = os.environ.get("LLVM_BUILD_DIR")
            if not llvm_build_dir:
                raise RuntimeError("Either (MLIR_DIR and LLVM_DIR) or LLVM_BUILD_DIR must be set")
            
            if not mlir_dir:
                mlir_dir = os.path.join(llvm_build_dir, "lib", "cmake", "mlir")
            if not llvm_dir:
                llvm_dir = os.path.join(llvm_build_dir, "lib", "cmake", "llvm")
        
        pybind11_DIR = os.environ.get("pybind11_DIR")
        if not pybind11_DIR:
            # Try to get it from python
            try:
                import pybind11
                pybind11_DIR = pybind11.get_cmake_dir()
            except ImportError:
                raise RuntimeError("pybind11_DIR environment variable is not set and pybind11 is not installed")
        
        cmake_args = [
            f"-DMLIR_DIR={mlir_dir}",
            f"-DLLVM_DIR={llvm_dir}",
            "-DLLVM_TARGETS_TO_BUILD=host",
            "-DLLVM_ENABLE_ASSERTIONS=ON",
            "-DCMAKE_BUILD_TYPE=DEBUG",
            "-DCMAKE_C_COMPILER=clang",
            "-DCMAKE_CXX_COMPILER=clang++",
            "-DLLVM_ENABLE_LLD=ON",
            "-DMLIR_ENABLE_BINDINGS_PYTHON=ON",
            f"-DPython3_EXECUTABLE={sys.executable}",
        ]

        build_temp = os.path.join(ext.sourcedir, "build")
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        if os.system("which ninja >> /dev/null") == 0:
            subprocess.run(
                ["cmake", "-G Ninja", ext.sourcedir] + cmake_args,
                cwd=build_temp,
                check=True,
            )
            if NUM_THREADS := os.environ.get("NUM_THREADS"):
                subprocess.run(
                    ["ninja", f"-j{NUM_THREADS}"], cwd=build_temp, check=True
                )
            else:
                subprocess.run(["ninja"], cwd=build_temp, check=True)
        elif os.system("which make >> /dev/null") == 0:
            subprocess.run(
                ["cmake", "-G Unix Makefiles", ext.sourcedir] + cmake_args,
                cwd=build_temp,
                check=True,
            )
            if NUM_THREADS := os.environ.get("NUM_THREADS"):
                subprocess.run(["make", f"-j{NUM_THREADS}"], cwd=build_temp, check=True)
            else:
                subprocess.run(["make", "-j"], cwd=build_temp, check=True)
        else:
            raise RuntimeError(f"Unsupported could not find ninja or make")
        
        # Create symlink after successful build
        self.create_mlir_symlink(ext.sourcedir)

    def create_mlir_symlink(self, sourcedir):
        """Create symlink from mlir/build/tools/streamhls/_mlir to stflow/_mlir."""
        # Define source and target paths
        source_path = os.path.join(sourcedir, "build", "tools", "streamhls", "_mlir")
        target_path = os.path.join("stflow", "_mlir")
        
        # Make sure the source path exists
        if not os.path.exists(source_path):
            print(f"Warning: Source path {source_path} does not exist. Skipping symlink creation.")
            return
        
        # Make sure the stflow directory exists
        stflow_dir = "stflow"
        if not os.path.exists(stflow_dir):
            os.makedirs(stflow_dir)
            print(f"Created directory: {stflow_dir}")
        
        # Remove existing target if it exists
        if os.path.exists(target_path) or os.path.islink(target_path):
            if os.path.islink(target_path):
                os.unlink(target_path)
                print(f"Removed existing symlink: {target_path}")
            else:
                if os.path.isdir(target_path):
                    shutil.rmtree(target_path)
                else:
                    os.remove(target_path)
                print(f"Removed existing file/directory: {target_path}")
        
        # Create the symlink
        try:
            # Convert to absolute paths for the symlink
            abs_source_path = os.path.abspath(source_path)
            abs_target_path = os.path.abspath(target_path)
            
            os.symlink(abs_source_path, abs_target_path)
            print(f"Created symlink: {abs_target_path} -> {abs_source_path}")
        except OSError as e:
            print(f"Failed to create symlink: {e}")
            # Fallback: try to copy instead of symlink
            try:
                if os.path.isdir(source_path):
                    shutil.copytree(source_path, target_path)
                else:
                    shutil.copy2(source_path, target_path)
                print(f"Created copy instead of symlink: {target_path}")
            except Exception as copy_e:
                print(f"Failed to create copy as fallback: {copy_e}")


def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


if __name__ == "__main__":
    with open("README.md", encoding="utf-8") as fp:
        long_description = fp.read()

    setup(
        name="stflow",
        description="ST-Flow",
        version="0.0",
        author="ST-Flow Community",
        long_description=long_description,
        long_description_content_type="text/markdown",
        setup_requires=[],
        install_requires=parse_requirements("requirements.txt"),
        packages=find_packages(),
        ext_modules=[CMakeExtension("mlir", sourcedir="mlir")],
        cmdclass={"build_ext": CMakeBuild},
        url="https://github.com/UCLA-VAST/ST-Flow",
        python_requires=">=3.12",
        classifiers=[
            "Programming Language :: Python :: 3.12",
            "Topic :: Scientific/Engineering",
            "Topic :: System :: Hardware",
            "Operating System :: OS Independent",
        ],
        zip_safe=True,
    )
