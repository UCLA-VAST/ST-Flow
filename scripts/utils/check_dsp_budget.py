#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
import fnmatch
from scipy.stats import gmean

def matches_filter(name, pattern_str):
    if not pattern_str: return True
    patterns = pattern_str.split(',')
    return any(fnmatch.fnmatch(str(name), p.strip()) for p in patterns)

def main():
    parser = argparse.ArgumentParser(description="Check if DSP budget was respected across experiments.")
    parser.add_argument("--csv", type=str, default="study/data/all.csv", help="Input results CSV")
    parser.add_argument("--limit", type=int, default=500, help="Target DSP limit to analyze")
    parser.add_argument("-s", "--filter", type=str, help="Filter by config pattern")
    parser.add_argument("-b", "--bench", type=str, help="Filter by benchmark pattern")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {args.csv}")
        return

    # Basic cleaning
    df.columns = [c.strip() for c in df.columns]
    
    # Filtering similar to evrun/evtable
    if args.filter:
        df = df[df['config'].apply(lambda x: matches_filter(x, args.filter))]
    if args.bench:
        df = df[df['benchmark'].apply(lambda x: matches_filter(x, args.bench))]

    # Ensure dsp_lim is numeric
    df['dsp_lim'] = pd.to_numeric(df['dsp_lim'], errors='coerce')
    
    # Filter for successful DSE and the target budget
    # p_status is DSE status. We only care about cases where the tool finished.
    mask = (df['p_status'] == 'Pass') & (df['dsp_lim'] == args.limit)
    df_budget = df[mask].copy()
    
    if df_budget.empty:
        available_limits = df[df['p_status'] == 'Pass']['dsp_lim'].unique()
        print(f"No successful experiments found with DSP limit {args.limit}")
        print(f"Available limits in CSV: {available_limits}")
        return

    print(f"\n============================================================")
    print(f" DSP Budget Respect Summary (Limit: {args.limit})")
    print(f" Analyzed {len(df_budget)} experiments")
    print(f"============================================================\n")
    
    cols = [
        ('p_dsps', 'ST-Flow (DSE)'),
        ('v_dsps', 'Vitis HLS (Syn)'),
        ('b_dsps', 'Vivado (Bitstream)')
    ]
    
    for col, label in cols:
        # Filter for valid data (non-zero and non-NaN)
        data = df_budget[df_budget[col] > 0][col].dropna()
        if data.empty:
            print(f"[{label}]\n  No valid data available.\n")
            continue
            
        avg = data.mean()
        worst = data.max()
        try:
            # gmean requires all values to be positive
            geo = gmean(data)
        except:
            geo = np.nan
            
        # Utilization %
        util_avg = (avg / args.limit) * 100
        util_worst = (worst / args.limit) * 100
        
        # Difference from limit (Positive means under budget, Negative means over budget)
        diff_avg = args.limit - avg
        diff_worst = args.limit - worst
        
        print(f"[{label}]")
        print(f"  Average:    {avg:7.1f} ({util_avg:5.1f}% util, Margin: {diff_avg:7.1f})")
        print(f"  Geo Mean:   {geo:7.1f}")
        print(f"  Worst Case: {worst:7.1f} ({util_worst:5.1f}% util, Margin: {diff_worst:7.1f})")
        
        # Check if budget was ever exceeded
        exceeded = data[data > args.limit]
        if not exceeded.empty:
            print(f"  [!] OVER BUDGET: {len(exceeded)} cases (Max: {exceeded.max():.1f})")
        else:
            print(f"  [OK] Budget respected in all cases.")
        print("-" * 60)

if __name__ == "__main__":
    main()
