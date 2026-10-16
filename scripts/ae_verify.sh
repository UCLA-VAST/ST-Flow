#!/usr/bin/env bash
#
# Artifact-evaluation check for ST-Flow (PACT'26).
#
# Regenerates every paper figure and table from the archived experiment outputs into
# ae_verify_out/ at the repository root, then compares the result against the reference
# copy that ships in PACT26_Artifacts/data. Nothing under PACT26_Artifacts/data is
# modified, and ae_verify_out/ is gitignored.
#
#   source env.sh
#   bash scripts/ae_verify.sh [output-dir]
#
# Expected result: CSVs and .tex tables byte-identical; PDF figures identical in content
# with a differing embedded creation timestamp (reported as "content identical").
# Takes about a minute. Requires no Vitis, no FPGA, no SLURM and no solver.

set -uo pipefail

if [ -z "${PRJ_ROOT:-}" ]; then
    echo "Error: PRJ_ROOT is not set. Run 'source env.sh' from the repository root." >&2
    exit 1
fi

REF="$PRJ_ROOT/PACT26_Artifacts/data"
OUT="${1:-$PRJ_ROOT/ae_verify_out}"

# Output from a previous run has to go: a file the regeneration fails to produce would
# otherwise survive and compare identical, turning a broken run into a false PASS. Only
# ever clear a directory this script created, identified by the marker it drops.
MARKER=".stflow-ae-output-dir"
if [ -e "$OUT" ]; then
    if [ ! -d "$OUT" ]; then
        echo "Error: $OUT exists and is not a directory." >&2; exit 1
    fi
    if [ ! -f "$OUT/$MARKER" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
        echo "Error: $OUT is not empty and was not created by this script." >&2
        echo "Remove it yourself or pass a different output directory." >&2
        exit 1
    fi
    rm -rf "${OUT:?}"/* "${OUT:?}"/.[!.]* 2>/dev/null
fi
mkdir -p "$OUT" || { echo "Error: cannot create output directory $OUT" >&2; exit 1; }
touch "$OUT/$MARKER"

echo "Reference : $REF"
echo "Regenerate: $OUT"

OUT_PATH="$OUT" bash "$PRJ_ROOT/PACT26_Artifacts/experiments.sh" > "$OUT/regenerate.log" 2>&1
gen_status=$?
if [ "$gen_status" -ne 0 ]; then
    echo "WARNING: regeneration reported $gen_status failed step(s); see $OUT/regenerate.log" >&2
fi

fail=0
pass=0

echo
echo "=== CSVs and LaTeX tables (expect byte-identical) ==="
for rel in main.csv gpt2.csv codo.csv gpt2_dse_traces.csv \
           tables/summary_table_bitstream.tex tables/gpt2_dse_stats.tex \
           tables/onboard_results.tex tables/robustness.tex; do
    if [ ! -f "$OUT/$rel" ]; then
        printf '  %-42s NOT GENERATED\n' "$rel"; fail=$((fail+1)); continue
    fi
    if cmp -s "$OUT/$rel" "$REF/$rel"; then
        printf '  %-42s identical\n' "$rel"; pass=$((pass+1))
    else
        printf '  %-42s DIFFERS\n' "$rel"; fail=$((fail+1))
    fi
done

echo
echo "=== Figures (expect content identical; PDF timestamp may differ) ==="
for f in "$REF"/figures/*.pdf; do
    rel="figures/$(basename "$f")"
    if [ ! -f "$OUT/$rel" ]; then
        printf '  %-42s NOT GENERATED\n' "$(basename "$f")"; fail=$((fail+1)); continue
    fi
    if cmp -s "$OUT/$rel" "$f"; then
        printf '  %-42s identical\n' "$(basename "$f")"; pass=$((pass+1)); continue
    fi
    # A PDF whose only differences fall at or after the /CreationDate entry differs
    # solely in its metadata timestamp (and the xref offsets that timestamp shifts).
    verdict=$(python3 - "$OUT/$rel" "$f" <<'PY'
import sys
new, ref = (open(p, 'rb').read() for p in sys.argv[1:3])
meta = new.find(b'/CreationDate')
first = next((i for i, (a, b) in enumerate(zip(new, ref)) if a != b), min(len(new), len(ref)))
print("content identical (timestamp only)" if meta != -1 and first >= meta else "CONTENT DIFFERS")
PY
)
    printf '  %-42s %s\n' "$(basename "$f")" "$verdict"
    case "$verdict" in *"CONTENT DIFFERS"*) fail=$((fail+1));; *) pass=$((pass+1));; esac
done

echo
echo "$pass output(s) reproduced, $fail mismatch(es)."
[ "$fail" -eq 0 ] && echo "RESULT: PASS" || echo "RESULT: FAIL"
echo "Regenerated outputs kept in $OUT"
exit "$fail"
