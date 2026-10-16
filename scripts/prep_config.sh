#!/bin/bash

MODE=$1
VANILLA_CONFIG="vanilla_config.yaml"
TARGET_CONFIG="test_config.yaml"

if [ ! -f "$VANILLA_CONFIG" ]; then
    echo "Error: $VANILLA_CONFIG not found in $(pwd)"
    exit 1
fi

case $MODE in
    "csim")
        echo "Preparing $TARGET_CONFIG for CSIM mode..."
        cat "$VANILLA_CONFIG" > "$TARGET_CONFIG"
        ;;
    "csim_only")
        echo "Preparing $TARGET_CONFIG for Fast CSIM (compile_only) mode..."
        echo "compile_only: True" > "$TARGET_CONFIG"
        cat "$VANILLA_CONFIG" >> "$TARGET_CONFIG"
        ;;
    "syn")
        echo "Preparing $TARGET_CONFIG for Synthesis mode..."
        echo "run_vitis_hls_synth: True" > "$TARGET_CONFIG"
        echo "compile_only: True" >> "$TARGET_CONFIG"
        cat "$VANILLA_CONFIG" >> "$TARGET_CONFIG"
        ;;
    "cosim")
        echo "Preparing $TARGET_CONFIG for Co-Simulation mode..."
        echo "run_vitis_hls_synth: True" > "$TARGET_CONFIG"
        echo "run_vitis_hls_cosim: True" >> "$TARGET_CONFIG"
        echo "compile_only: True" >> "$TARGET_CONFIG"
        cat "$VANILLA_CONFIG" >> "$TARGET_CONFIG"
        ;;
    "bitstream")
        echo "Preparing $TARGET_CONFIG for Bitstream mode..."
        echo "compile_only: True" > "$TARGET_CONFIG"
        echo "run_bitstream: True" >> "$TARGET_CONFIG"
        cat "$VANILLA_CONFIG" >> "$TARGET_CONFIG"
        ;;
    *)
        echo "Usage: $0 {csim|syn|cosim|bitstream}"
        exit 1
        ;;
esac

echo "✅ Prepared $TARGET_CONFIG for $MODE mode."
