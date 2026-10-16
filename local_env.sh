# Backend-tool environment for running (not just building) ST-Flow designs:
# Vitis HLS for csim/synthesis, TAPA for the TAPA backend, and AMPL for the MINLP solver.
#
# Usage:
#   source env.sh          # compiler build environment (LLVM/MLIR discovery)
#   source local_env.sh    # backend tools, for running the test suite
#
# Every path below can be overridden by exporting the corresponding variable first.
# Nothing here is required to *build* ST-Flow — only to run designs through csim,
# synthesis, or a MINLP-based DSE.

# --- Project root -------------------------------------------------------------
# Derive it from this script's location so any clone works. env.sh sets this too;
# only fill it in if it is still unset.
if [ -z "$PRJ_ROOT" ]; then
    export PRJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
fi

# --- Vitis HLS ----------------------------------------------------------------
# Needed for csim (STFlowConfig.csim, on by default), synthesis and cosim.
if [ -z "$STFLOW_VITIS_HLS_DIR" ]; then
    for candidate in /tools/Xilinx/Vitis_HLS/* /opt/Xilinx/Vitis_HLS/*; do
        if [ -f "$candidate/settings64.sh" ]; then
            STFLOW_VITIS_HLS_DIR="$candidate"
        fi
    done
fi
if [ -n "$STFLOW_VITIS_HLS_DIR" ] && [ -f "$STFLOW_VITIS_HLS_DIR/settings64.sh" ]; then
    source "$STFLOW_VITIS_HLS_DIR/settings64.sh"
    echo "Using Vitis HLS at $STFLOW_VITIS_HLS_DIR"
else
    echo "WARNING: Vitis HLS not found. csim / synthesis / cosim will fail." >&2
    echo "         Set STFLOW_VITIS_HLS_DIR to a Vitis_HLS/<version> directory." >&2
fi

# --- TAPA / RapidStream -------------------------------------------------------
# Only needed when target_tool = 'tapa'.
for dir in "$HOME/.rapidstream-tapa/usr/bin" "$HOME/.rapidstream"; do
    [ -d "$dir" ] && export PATH="$PATH:$dir"
done

# --- AMPL ---------------------------------------------------------------------
# The MINLP solver path shells out to a bare `ampl` on PATH
# (mlir/lib/Dialect/Dataflow/Support/PerformanceModel.cpp). Any test whose
# dse_options select 'minlp' fails without it.
if ! command -v ampl >/dev/null 2>&1; then
    if [ -z "$STFLOW_AMPL_DIR" ]; then
        for candidate in "$HOME/tools/ampl.linux-intel64" "$HOME/ampl.linux-intel64" "$HOME/ampl"; do
            [ -x "$candidate/ampl" ] && STFLOW_AMPL_DIR="$candidate" && break
        done
    fi
    if [ -n "$STFLOW_AMPL_DIR" ] && [ -x "$STFLOW_AMPL_DIR/ampl" ]; then
        export PATH="$PATH:$STFLOW_AMPL_DIR"
        echo "Using AMPL at $STFLOW_AMPL_DIR"
    else
        echo "WARNING: 'ampl' not on PATH. MINLP-based DSE will fail." >&2
        echo "         Set STFLOW_AMPL_DIR, or select a heuristic solver (sa/ga/random)." >&2
    fi
fi
