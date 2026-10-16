import os
import csv
import argparse
import glob
import re
import fnmatch
from report_utils import get_full_project_report

def main():
    prj_root = os.environ.get("PRJ_ROOT", os.getcwd())

    parser = argparse.ArgumentParser(description="Collect and analyze all evaluation results.")
    parser.add_argument("-i", "--input", type=str, default=None, help="Input evaluation directory")
    parser.add_argument("-o", "--output", type=str, default="all.csv", help="Output CSV file")
    parser.add_argument("-v", "--detailed", action="store_true", help="Generate detailed per-benchmark reports")
    parser.add_argument("-x", "--exclude", type=str, default=None, help="Exclude tests matching these patterns (comma-separated)")
    args = parser.parse_args()

    # Default output directory is always the current directory
    root = os.getcwd()
    os.makedirs(root, exist_ok=True)
    
    if args.input:
        eval_dir = args.input
    else:
        # Default to 'evaluation' in the current directory
        eval_dir = os.path.join(os.getcwd(), "evaluation")
    
    if not os.path.exists(eval_dir):
        print(f"Error: Evaluation directory not found at {eval_dir}.")
        return

    all_results = []
    
    # Column ordering for the CSV
    fieldnames = [
        'config', 'benchmark', 'test_name',
        'test', 'dsp_lim', 'p_solver', 'u_solver', 'u_strategy', 'hw_strategy',
        'p_status', 'v_status', 'b_status',
        'p_cycles', 'rtl_cycles', 'b_cycles', 'v_min_cycles', 'v_max_cycles',
        'cycle_err_rtl', 'cycle_err_bit',
        'p_dsps', 'v_dsps', 'b_dsps', 'dsp_err_vitis', 'dsp_err_bit',
        'p_mem', 'v_mem', 'b_mem',
        'v_freq', 'b_freq', 'p_cost'
    ]

    # Regex for breaking down test_name
    # Pattern: {DSP}_{TEST}_{SOLVER1}_{SOLVER2}_{STRATEGY} 
    # OR {DSP}_{TEST}_comb_{SOLVER}_{STRATEGY}
    hw_strategies = ["no_reuse", "max_reuse", "balanced"]
    hw_strategy_pattern = "|".join(hw_strategies)
    regex = rf"^(\d+)_+(.*)_([^_]+)_([^_]+)_({hw_strategy_pattern})$"

    # Extraction for u_strategy from config_name
    u_strategies = ["ndiv_min", "ndiv_max", "div_min", "div", "ndiv"]
    def get_u_strategy(config):
        for s in u_strategies:
            if s in config:
                return s
        return "N/A"

    def matches_filter(name, pattern_str):
        if not pattern_str: return False
        patterns = pattern_str.split(',')
        return any(fnmatch.fnmatch(name, p.strip()) for p in patterns)

    print(f"Scanning {eval_dir} for projects...")
    
    # Iterate through strategy folders
    for config_name in sorted(os.listdir(eval_dir)):
        strat_path = os.path.join(eval_dir, config_name)
        if not os.path.isdir(strat_path): continue
        
        # Iterate through benchmark folders
        for bench in sorted(os.listdir(strat_path)):
            bench_path = os.path.join(strat_path, bench)
            if not os.path.isdir(bench_path): continue
            
            bench_results = []
            # Find all .prj folders
            for prj in sorted(glob.glob(os.path.join(bench_path, "*.prj"))):
                test_name = os.path.basename(prj).replace(".prj", "")
                
                if args.exclude and matches_filter(test_name, args.exclude):
                    print(f" Skipping excluded project: {test_name}")
                    continue

                print(f" Processing {config_name} / {bench} / {test_name}")
                
                report = get_full_project_report(prj)
                
                # Add metadata
                report['config'] = config_name
                report['benchmark'] = bench
                report['test_name'] = test_name
                
                # Extract u_strategy from config_name
                report['u_strategy'] = get_u_strategy(config_name)

                # Extract components from test_name
                match = re.match(regex, test_name)
                if match:
                    dsp_lim, base_test, part3, part4, hw_strategy = match.groups()
                    report['dsp_lim'] = dsp_lim
                    report['test'] = base_test
                    report['hw_strategy'] = hw_strategy
                    
                    if part3 == 'comb':
                        report['p_solver'] = 'comb'
                        report['u_solver'] = part4
                    else:
                        report['p_solver'] = part3
                        report['u_solver'] = part4
                    
                    # Update config to include solver info for easier filtering (e.g. minlp_minlp_div_min_500)
                    report['config'] = f"{report['p_solver']}_{report['u_solver']}_{config_name}"
                else:
                    # Fallback for unexpected names
                    report['dsp_lim'] = 'N/A'
                    report['test'] = 'N/A'
                    report['hw_strategy'] = 'N/A'
                    report['p_solver'] = 'N/A'
                    report['u_solver'] = 'N/A'
                    report['config'] = config_name
                
                bench_results.append(report)
            
            if bench_results:
                all_results.extend(bench_results)
                
                # Save detailed report if requested
                if args.detailed:
                    bench_data_dir = os.path.join(root, config_name, bench)
                    os.makedirs(bench_data_dir, exist_ok=True)
                    bench_csv_path = os.path.join(bench_data_dir, "results.csv")
                    with open(bench_csv_path, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                        writer.writeheader()
                        writer.writerows(bench_results)
                    print(f"  Saved detailed report to {bench_csv_path}")

    if not all_results:
        print("No project results found.")
        return

    # If output path is absolute or contains a directory separator, use it as is
    if os.path.isabs(args.output) or os.path.dirname(args.output):
        output_path = args.output
    else:
        output_path = os.path.join(root, args.output)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n\u2705 Consolidated {len(all_results)} projects into {output_path}")

if __name__ == "__main__":
    main()
