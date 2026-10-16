#!/usr/bin/env python3
"""Collect Vitis HLS reports from the augmented CODO baseline designs into one CSV.

This is the input to Figure 5 (ST-Flow vs. CODO). It walks a CODO designs tree

    <designs>/<benchmark>/<test>/hls/<test>.prj.vitis/solution1/

and pulls each design's synthesis estimates, C-simulation status and RTL co-simulation
cycle count out of the raw tool reports:

    solution1/syn/report/csynth.xml          resources, latency estimates, clock period
    solution1/csim/report/*_csim.log         C-simulation status
    solution1/sim/report/*_cosim.rpt         RTL co-simulation status and cycles
    solution1/sim/report/verilog/lat.rpt     RTL cycle count (fallback)

Designs that never synthesized live in a sibling failed/ tree and are not collected;
the paper reports them qualitatively in Section 5.5 instead.

Usage:
    collect_codo_reports.py -i <designs-dir> [-o <out.csv>]

Invoked via `evcodo --designs-dir <dir> report -o <out.csv>`.
"""

import argparse
import csv
import os
import re
import sys
import xml.etree.ElementTree as ET

FIELDS = [
    "benchmark", "test_name",
    "syn_status", "csim_status", "cosim_status",
    "v_min_cycles", "v_max_cycles", "rtl_cycles",
    "v_dsps", "v_brams", "v_urams", "v_mem", "v_freq",
]


def _int(text, default=-1):
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return default


def read_csynth(solution_dir):
    """Resources, latency estimates and achieved frequency from csynth.xml."""
    out = {
        "syn_status": "Fail",
        "v_min_cycles": -1, "v_max_cycles": -1,
        "v_dsps": -1, "v_brams": -1, "v_urams": -1, "v_mem": -1, "v_freq": -1.0,
    }
    path = os.path.join(solution_dir, "syn", "report", "csynth.xml")
    if not os.path.exists(path):
        return out
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return out

    res = root.find(".//AreaEstimates/Resources")
    if res is not None:
        out["v_dsps"] = _int(res.findtext("DSP"))
        out["v_brams"] = _int(res.findtext("BRAM_18K"))
        out["v_urams"] = _int(res.findtext("URAM"))
        if out["v_brams"] >= 0 and out["v_urams"] >= 0:
            # Memory-equivalent score used throughout the paper: a URAM is worth
            # 16 BRAM_18K tiles (Section 5.4).
            out["v_mem"] = out["v_brams"] + 16 * out["v_urams"]

    lat = root.find(".//PerformanceEstimates/SummaryOfOverallLatency")
    if lat is not None:
        out["v_min_cycles"] = _int(lat.findtext("Best-caseLatency"))
        out["v_max_cycles"] = _int(lat.findtext("Worst-caseLatency"))

    # The first EstimatedClockPeriod in the document is the top-level one; the
    # per-submodule summaries that follow it must not override it.
    period = root.findtext(".//SummaryOfTimingAnalysis/EstimatedClockPeriod")
    if period is None:
        for elem in root.iter("EstimatedClockPeriod"):
            period = elem.text
            break
    try:
        p = float(period)
        if p > 0:
            out["v_freq"] = round(1000.0 / p, 2)
    except (TypeError, ValueError):
        pass

    out["syn_status"] = "Pass"
    return out


def read_csim(solution_dir):
    report_dir = os.path.join(solution_dir, "csim", "report")
    if not os.path.isdir(report_dir):
        return "N/A"
    for name in sorted(os.listdir(report_dir)):
        if not name.endswith(".log"):
            continue
        with open(os.path.join(report_dir, name), errors="replace") as f:
            text = f.read()
        if "CSim done with 0 errors" in text:
            return "Pass"
        if "CSim failed" in text or "ERROR" in text:
            return "Fail"
    return "N/A"


def read_cosim(solution_dir):
    """RTL co-simulation status and cycle count."""
    status, cycles = "N/A", -1
    report_dir = os.path.join(solution_dir, "sim", "report")

    if os.path.isdir(report_dir):
        for name in sorted(os.listdir(report_dir)):
            if not name.endswith("_cosim.rpt"):
                continue
            with open(os.path.join(report_dir, name), errors="replace") as f:
                for line in f:
                    # | Verilog | Pass | min | avg | max | ... | total |
                    if "Verilog" not in line or "|" not in line:
                        continue
                    cols = [c.strip() for c in line.strip().strip("|").split("|")]
                    if len(cols) < 3 or cols[0] != "Verilog":
                        continue
                    status = cols[1]
                    cycles = _int(cols[2])
                    break
            if status != "N/A":
                break

    if cycles < 0:
        lat = os.path.join(report_dir, "verilog", "lat.rpt")
        if os.path.exists(lat):
            with open(lat, errors="replace") as f:
                m = re.search(r'\$MAX_LATENCY\s*=\s*"(\d+)"', f.read())
            if m:
                cycles = int(m.group(1))
                if status == "N/A":
                    status = "Pass"
    return status, cycles


def collect(designs_dir):
    rows = []
    for benchmark in sorted(os.listdir(designs_dir)):
        bench_dir = os.path.join(designs_dir, benchmark)
        if not os.path.isdir(bench_dir):
            continue
        for test in sorted(os.listdir(bench_dir)):
            hls_dir = os.path.join(bench_dir, test, "hls")
            if not os.path.isdir(hls_dir):
                continue
            solution = os.path.join(hls_dir, f"{test}.prj.vitis", "solution1")
            if not os.path.isdir(solution):
                # Fall back to whatever *.prj.vitis the design actually carries.
                candidates = sorted(
                    d for d in os.listdir(hls_dir) if d.endswith(".prj.vitis")
                )
                if not candidates:
                    continue
                solution = os.path.join(hls_dir, candidates[0], "solution1")
                if not os.path.isdir(solution):
                    continue

            row = {"benchmark": benchmark, "test_name": test}
            row.update(read_csynth(solution))
            row["csim_status"] = read_csim(solution)
            row["cosim_status"], row["rtl_cycles"] = read_cosim(solution)
            rows.append(row)
            print(f" Processing {benchmark} / {test}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--input", required=True, help="CODO designs directory")
    ap.add_argument("-o", "--output", help="output CSV (default: stdout)")
    args = ap.parse_args()

    if not os.path.isdir(args.input):
        sys.exit(f"Error: {args.input} is not a directory")

    rows = collect(args.input)
    if not rows:
        sys.exit(f"Error: no CODO designs found under {args.input}")

    out = open(args.output, "w", newline="") if args.output else sys.stdout
    try:
        # Default \r\n terminator, matching the CSVs the other collectors emit.
        writer = csv.DictWriter(out, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in FIELDS} for r in rows)
    finally:
        if args.output:
            out.close()
            print(f"\n Consolidated {len(rows)} CODO designs into {args.output}")


if __name__ == "__main__":
    main()
