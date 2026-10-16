import os
import csv
import glob
import re
import xml.etree.ElementTree as ET

prj_root = os.environ.get("PRJ_ROOT", os.getcwd())

def safe_float(val, default=-1.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def safe_int(val, default=-1):
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default

# The per-design DSE report used to be written as reports/phoenix.csv, under the
# project's former name. It was renamed to stflow.csv after the PACT'26 experiments
# were run, so every archived design in PACT26_Artifacts/ carries the old name. Read
# whichever exists, newest name first: without this the p_* (ST-Flow predicted)
# columns come back empty and the model-validation figure plots ST-Flow as NaN.
STFLOW_REPORT_NAMES = ('stflow.csv', 'phoenix.csv')


def find_stflow_report(prj_path):
    """Returns the path to a design's DSE report, or None if it has none."""
    for name in STFLOW_REPORT_NAMES:
        candidate = os.path.join(prj_path, 'reports', name)
        if os.path.exists(candidate):
            return candidate
    return None


def get_stflow_data(prj_path):
    """Parses the DSE report (stflow.csv, or legacy phoenix.csv) for a design."""
    csv_file = find_stflow_report(prj_path)
    data = {
        'p_status': 'Fail',
        'p_cycles': -1,
        'p_dsps': -1,
        'p_mem': -1,
        'p_cost': -1.0
    }
    
    if csv_file:
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # DSE is considered successful if the solver reports success OR if csim passed
                    dse_success = row.get('success', '').strip().lower() == 'true'
                    csim_success = row.get('csim_success', '').strip().lower() == 'true'
                    
                    data['dse_success'] = dse_success
                    data['csim_success'] = csim_success
                    data['p_status'] = 'Pass' if (dse_success or csim_success) else 'Fail'
                    data['p_cycles'] = safe_int(row.get('latency', -1))
                    data['p_dsps'] = safe_int(row.get('dsps', -1))
                    data['p_mem'] = safe_int(row.get('memory', -1))
                    data['p_cost'] = safe_float(row.get('cost', -1.0))
                    break
        except Exception:
            pass
    return data

def get_vitis_data(prj_path):
    """Parses vitis_post_dma.csv and returns report data."""
    csv_file = os.path.join(prj_path, 'reports', 'vitis_post_dma.csv')
    data = {
        'v_status': 'N/A',
        'v_min_cycles': -1,
        'v_max_cycles': -1,
        'v_dsps': -1,
        'v_mem': -1,
        'v_freq': -1.0,
        'v_cosim_status': 'N/A',
        'rtl_cycles': -1
    }
    
    # Check if synthesis log exists to determine if it ran
    log_file = os.path.join(prj_path, 'vitis_post_dma', 'vitis_hls.log')
    if os.path.exists(log_file):
        data['v_status'] = 'Running/Failed'
    
    if os.path.exists(csv_file):
        data['v_status'] = 'Pass'
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data['v_min_cycles'] = safe_int(row.get('min_cycles', -1))
                    data['v_max_cycles'] = safe_int(row.get('max_cycles', -1))
                    data['v_dsps'] = safe_int(row.get('dsps', -1))
                    brams = safe_int(row.get('brams_18k', 0))
                    urams = safe_int(row.get('urams', 0))
                    data['v_mem'] = brams + (16 * urams)
                    
                    period = safe_float(row.get('estimated_clock_period_ns', 0))
                    if period > 0:
                        data['v_freq'] = round(1000.0 / period, 2)
                    
                    data['v_cosim_status'] = row.get('cosim_status', 'Pass' if safe_int(row.get('cosim_latency', -1)) > 0 else 'N/A')
                    data['rtl_cycles'] = safe_int(row.get('cosim_latency', -1))
                    break
        except Exception:
            pass
    return data

def get_bitstream_data(prj_path):
    """Parses bitstream.csv and checks for hardware execution artifacts."""
    csv_file = os.path.join(prj_path, 'reports', 'bitstream.csv')
    data = {
        'b_status': 'N/A',
        'b_cycles': -1,
        'b_dsps': -1,
        'b_mem': -1,
        'b_freq': -1.0,
        'bitgen_success': False,
        'bitrun_success': False,
    }
    
    # BitGen Success: Existence of .xclbin in build/
    if glob.glob(os.path.join(prj_path, "build", "*.xclbin")):
        data['bitgen_success'] = True

    # BitRun Success Part 1: Existence of bitstream_latency.txt with valid float
    latency_file = os.path.join(prj_path, 'reports', 'bitstream_latency.txt')
    if os.path.exists(latency_file):
        try:
            with open(latency_file, 'r') as f:
                val = f.read().strip()
                if val:
                    data['bitrun_latency_ms'] = safe_float(val, -1.0)
                    if data['bitrun_latency_ms'] > 0:
                        data['bitrun_success_latency'] = True
        except Exception:
            pass

    # Auto-generate or refresh report if missing or outdated
    build_dir = os.path.join(prj_path, "build")
    should_refresh = not os.path.exists(csv_file)
    if not should_refresh and os.path.exists(build_dir):
        csv_mtime = os.path.getmtime(csv_file)
        # Check if any .rpt in build/link/imp is newer
        imp_dir = os.path.join(build_dir, "link/imp")
        if os.path.exists(imp_dir):
            for f in os.listdir(imp_dir):
                if f.endswith(".rpt"):
                    if os.path.getmtime(os.path.join(imp_dir, f)) > csv_mtime:
                        should_refresh = True
                        break

    if should_refresh and os.path.exists(build_dir):
        try:
            get_bitstream_report(prj_path)
        except Exception:
            pass

    if os.path.exists(csv_file):
        data['b_status'] = 'Pass'
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data['b_status'] = row.get('report_type', 'Pass')
                    data['b_cycles'] = safe_int(row.get('bitstream_latency_cycles', -1))
                    data['b_dsps'] = safe_int(row.get('dsp_abs', -1))
                    brams = safe_int(row.get('bram_abs', 0))
                    urams = safe_int(row.get('uram_abs', 0))
                    data['b_mem'] = brams + (16 * urams)
                    data['b_freq'] = safe_float(row.get('bitstream_freq_mhz', -1.0))
                    break
        except Exception:
            pass
    return data

def get_full_project_report(prj_path):
    """Aggregates all metrics and calculates errors."""
    report = {}
    report.update(get_stflow_data(prj_path))
    report.update(get_vitis_data(prj_path))
    report.update(get_bitstream_data(prj_path))
    
    # BitRun success: latency file exists AND csim_success is True (results matched)
    report['bitrun_success'] = report.get('bitrun_success_latency', False) and report.get('csim_success', False)

    # Calculate Discrepancies
    # DSP Error
    if report['v_dsps'] > 0:
        report['dsp_err_vitis'] = round((report['p_dsps'] - report['v_dsps']) / report['v_dsps'] * 100, 2)
        report['dsp_diff_vitis'] = abs(report['p_dsps'] - report['v_dsps'])
    else:
        report['dsp_err_vitis'] = None
        report['dsp_diff_vitis'] = None
        
    if report['b_dsps'] > 0:
        report['dsp_err_bit'] = round((report['p_dsps'] - report['b_dsps']) / report['b_dsps'] * 100, 2)
    else:
        report['dsp_err_bit'] = None
        
    # Cycle Error (Model vs RTL)
    if report['rtl_cycles'] > 0:
        report['cycle_err_rtl'] = round((report['p_cycles'] - report['rtl_cycles']) / report['rtl_cycles'] * 100, 2)
    else:
        report['cycle_err_rtl'] = None
        
    # Cycle Error (Model vs Bitstream)
    if report['b_cycles'] > 0:
        report['cycle_err_bit'] = round((report['p_cycles'] - report['b_cycles']) / report['b_cycles'] * 100, 2)
    else:
        report['cycle_err_bit'] = None
        
    return report

def get_bitstream_report(prj_path):
    # Resource utilization (prj/build/link/imp/impl_1_kernel_util_*.rpt)
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
                    # Count BRAM_18K as the unit (BRAM_36K * 2)
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

    # Latency (prj/reports/bitstream_latency.txt)
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
