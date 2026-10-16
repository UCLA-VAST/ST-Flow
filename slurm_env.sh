#!/bin/bash
ml load xilinx/vivado/2024.2
# export XILINX_VITIS="/opt/xilinx/tools/Vitis/2024.2"
# export XILINX_HLS="/opt/xilinx/tools/Vitis/2024.2"
# if [ -d "/opt/xilinx/xrt" ]; then
#   export XILINX_XRT="/opt/xilinx/xrt"
# else
#   export XILINX_XRT="$(pwd)/extern/xrt"
# fi
export PRJ_ROOT="/home/suhailb/Projects/Stream-HLS"
export PATH="$PRJ_ROOT/scripts:$PATH"
export PATH="$PATH:$HOME/.rapidstream-tapa/usr/bin"
export PATH="$PATH:$HOME/.rapidstream"
export LD_LIBRARY_PATH="$HOME/.rapidstream-tapa/usr/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$HOME/.rapidstream-tapa/usr/lib:$LIBRARY_PATH"
export CPATH="$HOME/.rapidstream-tapa/usr/include:$CPATH"
export TAPA_LDFLAGS="-lyaml-cpp -ltinyxml2 -lOpenCL -lcontext -lthread -Wl,--no-as-needed -Wl,--as-needed"