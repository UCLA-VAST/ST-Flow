import os
import csv
import glob
import re
import argparse

def main():
    parser = argparse.ArgumentParser(description="Collect GPT2 DSE traces.")
    parser.add_argument("-i", "--input", type=str, default=None, help="Input evaluation directory")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    prj_root = os.environ.get("PRJ_ROOT", os.getcwd())
    
    if args.input:
        eval_dir = args.input
    else:
        eval_dir = os.path.join(prj_root, "evaluation_dse")

    if args.output:
        out_dir = args.output
    else:
        out_dir = os.path.join(prj_root, "study/data/gpt2_dse")
    
    os.makedirs(out_dir, exist_ok=True)

    configs = [
        "20_sa_comb_gpt_1000",
        "21_sa_sa_gpt_1000",
        "22_ga_comb_gpt_1000",
        "23_ga_ga_gpt_1000"
    ]

    all_traces = []

    print(f"Collecting GPT2 DSE traces from {eval_dir}...")

    for config in configs:
        config_path = os.path.join(eval_dir, config)
        if not os.path.exists(config_path):
            print(f" Warning: Config directory {config} not found.")
            continue

        # Determine solver and type from config name
        solver = "SA" if "sa" in config else "GA"
        is_comb = "comb" in config
        
        # Iterate through benchmark runs r00, r01, ...
        for bench in sorted(os.listdir(config_path)):
            if not bench.startswith("00_transformer_r"):
                continue
            
            run_match = re.search(r"r(\d+)", bench)
            if not run_match:
                continue
            run_id = int(run_match.group(1))
            
            bench_path = os.path.join(config_path, bench)
            # Iterate through all project folders (balanced, no_reuse, max_reuse).
            # sorted(): glob returns raw directory order, which varies between machines
            # and filesystems and would reorder the rows of gpt2_dse_traces.csv.
            for prj_path in sorted(glob.glob(os.path.join(bench_path, "*.prj"))):
                # Extract hw_strategy from project name
                # e.g. 1000_02_gpt2_tiny_comb_sa_balanced.prj
                prj_name = os.path.basename(prj_path)
                hw_strategy = "N/A"
                for s in ["balanced", "no_reuse", "max_reuse"]:
                    if s in prj_name:
                        hw_strategy = s
                        break

                dse_dir = os.path.join(prj_path, "dse")
                if not os.path.exists(dse_dir):
                    continue

                if is_comb:
                    # Combined trace
                    trace_file = os.path.join(dse_dir, f"{solver.lower()}_combined_1.csv")
                    if os.path.exists(trace_file):
                        with open(trace_file, 'r') as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                iteration = row.get('iteration') or row.get('generation')
                                cost = row.get('cost') or row.get('best_cost')
                                if iteration is not None and cost is not None:
                                    all_traces.append({
                                        'config': config,
                                        'solver': solver,
                                        'type': 'combined',
                                        'run': run_id,
                                        'hw_strategy': hw_strategy,
                                        'stage': 'combined',
                                        'iteration': int(float(iteration)),
                                        'cost': float(cost)
                                    })
                else:
                    # Separate traces: permutation and unrolling
                    perm_file = os.path.join(dse_dir, f"{solver.lower()}_permutation_1.csv")
                    unroll_file = os.path.join(dse_dir, f"{solver.lower()}_unrolling_1.csv")
                    
                    if os.path.exists(perm_file):
                        with open(perm_file, 'r') as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                iteration = row.get('iteration') or row.get('generation')
                                cost = row.get('cost') or row.get('best_cost')
                                if iteration is not None and cost is not None:
                                    all_traces.append({
                                        'config': config,
                                        'solver': solver,
                                        'type': 'separate',
                                        'run': run_id,
                                        'hw_strategy': hw_strategy,
                                        'stage': 'permutation',
                                        'iteration': int(float(iteration)),
                                        'cost': float(cost)
                                    })
                    
                    if os.path.exists(unroll_file):
                        with open(unroll_file, 'r') as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                iteration = row.get('iteration') or row.get('generation')
                                cost = row.get('cost') or row.get('best_cost')
                                if iteration is not None and cost is not None:
                                    all_traces.append({
                                        'config': config,
                                        'solver': solver,
                                        'type': 'separate',
                                        'run': run_id,
                                        'hw_strategy': hw_strategy,
                                        'stage': 'unrolling',
                                        'iteration': int(float(iteration)),
                                        'cost': float(cost)
                                    })

    if not all_traces:
        print("No traces found.")
        return

    # Save consolidated CSV
    out_file = os.path.join(out_dir, "gpt2_dse_traces.csv")
    fieldnames = ['config', 'solver', 'type', 'run', 'hw_strategy', 'stage', 'iteration', 'cost']
    with open(out_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_traces)

    print(f" Successfully collected {len(all_traces)} trace points into {out_file}")

if __name__ == "__main__":
    main()
