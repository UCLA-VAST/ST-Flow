import pandas as pd
import numpy as np
import os
from ..plot.theme import FPGA_LIMITS, get_label

def gmean(series):
    clean = series.dropna()
    return np.exp(np.log(clean[clean > 0]).mean()) if not clean.empty else np.nan

def generate_ppa_summary_table(df, use_bitstream=False):
    # Select columns
    prefix = 'b' if use_bitstream else 'v'
    cycle_col = 'b_cycles' if use_bitstream else 'rtl_cycles'
    dsp_col = f'{prefix}_dsps'
    mem_col = f'{prefix}_mem'
    
    # Filter for valid cycles
    df = df[df[cycle_col] > 0].copy()
    
    # Calculate Area and ADP only for valid resources
    valid_res_mask = (df[dsp_col] > 0) & (df[mem_col] > 0)
    df['area'] = np.nan
    df['adp'] = np.nan
    
    df.loc[valid_res_mask, 'area'] = (df.loc[valid_res_mask, dsp_col] / FPGA_LIMITS['dsp']) + \
                                    (df.loc[valid_res_mask, mem_col] / FPGA_LIMITS['mem_equiv'])
    
    # ADP only valid if both cycles and area are valid
    valid_adp_mask = valid_res_mask & (df[cycle_col] > 0)
    df.loc[valid_adp_mask, 'adp'] = df.loc[valid_adp_mask, cycle_col] * df.loc[valid_adp_mask, 'area']
    
    configs = ['div_min', 'ndiv_min', 'ndiv_max']
    hw_order = ['no_reuse', 'balanced', 'max_reuse']
    
    eval_type = "On-Board" if use_bitstream else "RTL/Vitis"

    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append(f"\\caption{{{eval_type} Aggregate PPA Breakdown (Geo. Means)}}")
    latex.append("\\begin{tabular}{l l | c c c | c}")
    latex.append("\\hline")
    latex.append("Config & Strategy & Delay & Area & ADP & Imp. \\\\")
    latex.append("\\hline")
    
    for config in configs:
        conf_df = df[df['u_strategy'] == config]
        if conf_df.empty: continue
        
        # Calculate geometric means for all strategies in this config to find best Delay, Area and ADP
        strategy_gmeans_adp = [gmean(conf_df[conf_df['hw_strategy'] == hw]['adp']) for hw in hw_order]
        strategy_gmeans_delay = [gmean(conf_df[conf_df['hw_strategy'] == hw][cycle_col]) for hw in hw_order]
        strategy_gmeans_area = [gmean(conf_df[conf_df['hw_strategy'] == hw]['area']) for hw in hw_order]
        
        worst_adp = np.nanmax(strategy_gmeans_adp) if not np.all(np.isnan(strategy_gmeans_adp)) else 0
        min_delay = np.nanmin(strategy_gmeans_delay) if not np.all(np.isnan(strategy_gmeans_delay)) else 0
        min_area = np.nanmin(strategy_gmeans_area) if not np.all(np.isnan(strategy_gmeans_area)) else 0
        
        for i, hw in enumerate(hw_order):
            hw_df = conf_df[conf_df['hw_strategy'] == hw]
            if hw_df.empty: continue
            
            d_gm = gmean(hw_df[cycle_col])
            a_gm = gmean(hw_df['area'])
            adp_gm = gmean(hw_df['adp'])
            imp = worst_adp / adp_gm if adp_gm > 0 and worst_adp > 0 else 0
            
            # Identify best metrics in each group
            is_best_adp = (adp_gm == np.nanmin(strategy_gmeans_adp)) if not np.isnan(adp_gm) else False
            is_best_delay = (d_gm == min_delay) if not np.isnan(d_gm) else False
            is_best_area = (a_gm == min_area) if not np.isnan(a_gm) else False
            
            conf_label = get_label(config, 'config', tex=True) if i == 0 else ""
            hw_label = get_label(hw, 'strat', tex=True)
            
            delay_str = f"{d_gm:,.0f}"
            if is_best_delay:
                delay_str = f"\\textbf{{{delay_str}}}"
            
            area_str = f"{a_gm:.3f}"
            if is_best_area:
                area_str = f"\\textbf{{{area_str}}}"
                
            line = f"{conf_label} & {hw_label} & "
            line += f"{delay_str} & {area_str} & {adp_gm:.0f} & "
            if is_best_adp:
                line += f"\\textbf{{{imp:.2f}x}} \\\\"
            else:
                line += f"{imp:.2f}x \\\\"
            latex.append(line)
        latex.append("\\hline")
        
    latex.append("\\end{tabular}")
    label = "tab:ppa_summary_bitstream" if use_bitstream else "tab:ppa_summary_vitis"
    latex.append(f"\\label{{{label}}}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)

def generate_ppa_table(df, config_label=None, use_bitstream=False):
    # Select columns
    prefix = 'b' if use_bitstream else 'v'
    cycle_col = 'b_cycles' if use_bitstream else 'rtl_cycles'
    dsp_col = f'{prefix}_dsps'
    mem_col = f'{prefix}_mem'
    
    # Filter for valid cycles
    df = df[df[cycle_col] > 0].copy()

    # Calculate Area and ADP only for valid resources
    valid_res_mask = (df[dsp_col] > 0) & (df[mem_col] > 0)
    df['area'] = np.nan
    df['adp'] = np.nan
    
    df.loc[valid_res_mask, 'area'] = (df.loc[valid_res_mask, dsp_col] / FPGA_LIMITS['dsp']) + \
                                    (df.loc[valid_res_mask, mem_col] / FPGA_LIMITS['mem_equiv'])
    
    # ADP only valid if both cycles and area are valid
    valid_adp_mask = valid_res_mask & (df[cycle_col] > 0)
    df.loc[valid_adp_mask, 'adp'] = df.loc[valid_adp_mask, cycle_col] * df.loc[valid_adp_mask, 'area']
    
    # Group by Test and Strategy
    pivot_df = df.pivot_table(index=['benchmark', 'test'], 
                              columns='hw_strategy', 
                              values=['rtl_cycles', 'area', 'adp'])
    
    # Reorder columns to group by strategy
    hw_order = ['no_reuse', 'balanced', 'max_reuse']
    latex = []
    latex.append("\\begin{table*}[t]")
    latex.append("\\centering")
    latex.append("\\caption{PPA Comparison across Hardware Strategies (Config: " + (config_label.replace('_', '\\_') if config_label else "All") + ")}")
    
    col_def = "l l | c c | c c c | c c c"
    latex.append("\\begin{tabular}{" + col_def + "}")
    latex.append("\\hline")
    latex.append(" & & \\multicolumn{2}{c|}{No Reuse} & \\multicolumn{3}{c|}{Balanced} & \\multicolumn{3}{c}{Max Reuse} \\\\")
    latex.append("Bench & Test & Delay & Area & Delay & Area & Imp. & Delay & Area & Imp. \\\\")
    latex.append("\\hline")
    
    prev_bench = None
    for (bench, test), row in pivot_df.iterrows():
        # Clean names
        bench_label = get_label(bench, 'bench') if bench != prev_bench else ""
        test_label = get_label(test, 'test')
        prev_bench = bench
        
        # Extract values
        base_delay = row[('rtl_cycles', 'no_reuse')]
        base_area = row[('area', 'no_reuse')]
        base_adp = row[('adp', 'no_reuse')]
        
        bal_delay = row[('rtl_cycles', 'balanced')]
        bal_area = row[('area', 'balanced')]
        bal_adp = row[('adp', 'balanced')]
        
        max_delay = row[('rtl_cycles', 'max_reuse')]
        max_area = row[('area', 'max_reuse')]
        max_adp = row[('adp', 'max_reuse')]
        
        # Calculate Improvements (relative to the WORST one in this row)
        adps = [v for v in [base_adp, bal_adp, max_adp] if v > 0]
        worst_adp = max(adps) if adps else 0
        best_adp = min(adps) if adps else 0
        
        base_imp = worst_adp / base_adp if base_adp > 0 else 0
        bal_imp = worst_adp / bal_adp if bal_adp > 0 else 0
        max_imp = worst_adp / max_adp if max_adp > 0 else 0
        
        def format_imp(val, adp):
            if val == 0: return "-"
            if adp == best_adp and len(adps) > 1: return f"\\textbf{{{val:.2f}x}}"
            return f"{val:.2f}x"

        line = f"{bench_label} & {test_label} & "
        line += f"{base_delay:,.0f} & {base_area:.3f} & "
        line += f"{bal_delay:,.0f} & {bal_area:.3f} & {format_imp(bal_imp, bal_adp)} & "
        line += f"{max_delay:,.0f} & {max_area:.3f} & {format_imp(max_imp, max_adp)} \\\\"
        latex.append(line)
        
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\label{tab:ppa_detailed}")
    latex.append("\\end{table*}")
    
    return "\n".join(latex)

def generate_gpt2_onboard_table(df):
    # Filter for GPT2
    df = df[df['test'].str.contains('gpt2', case=False)].copy()
    # Filter for passed bitstream and execution
    # Looking at the data, b_status is 'bitstream' for successful on-board runs
    df = df[df['b_status'] == 'bitstream'].copy()
    
    # We want one row per hw_strategy
    # If multiple exist, take the one with the best (lowest) b_cycles
    hw_order = ['no_reuse', 'balanced', 'max_reuse']
    
    def get_solver_key(row):
        config = str(row['config']).lower()
        if 'minlp' in config: return 'minlp'
        if 'sa_sa' in config: return 'sa_sa'
        if 'sa_comb' in config or 'comb_sa' in config: return 'sa_comb'
        if 'ga_ga' in config: return 'ga_ga'
        if 'ga_comb' in config or 'comb_ga' in config: return 'ga_comb'
        return None
    
    rows = []
    for hw in hw_order:
        hw_df = df[df['hw_strategy'] == hw]
        if hw_df.empty: continue
        
        # Take the best run
        best_run = hw_df.loc[hw_df['b_cycles'].idxmin()]
        
        # Use consistent LaTeX macros from theme.py
        strategy_label = get_label(hw, 'strat', tex=True)
        
        solver_key = get_solver_key(best_run)
        method_label = get_label(solver_key, 'solver', tex=True) if solver_key else "Unknown"
        
        dsp = best_run['b_dsps']
        bram = best_run['b_mem']
        cycles = best_run['b_cycles']
        freq = best_run['b_freq']
        
        rows.append({
            'Strategy': strategy_label,
            'Method': method_label,
            'DSP': f"{int(dsp):,}",
            'BRAM': f"{int(bram):,}",
            'FPGA Cyc.': f"${cycles/10000:.0f} \\times 10^4$",
            'Freq.': f"{freq:.0f} MHz"
        })
    
    # Generate LaTeX
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("  \\small")
    latex.append("  \\centering")
    latex.append("  \\caption{On-board FPGA Evaluation and Resource Usage}")
    latex.append("  \\label{tab:onboard_results}")
    latex.append("  \\begin{tabular}{llcccr}")
    latex.append("    \\toprule")
    latex.append("    \\textbf{Strategy} & \\textbf{Method} & \\textbf{DSP} & \\textbf{BRAM} & \\textbf{FPGA Cyc.} & \\textbf{Freq.} \\\\")
    latex.append("    \\midrule")
    
    for r in rows:
        latex.append(f"    {r['Strategy']} & {r['Method']} & {r['DSP']} & {r['BRAM']} & {r['FPGA Cyc.']} & {r['Freq.']} \\\\")
        
    latex.append("    \\bottomrule")
    latex.append("  \\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)

def generate_robustness_table(df, benchmarks=("01_dnn", "02_synthetic")):
    """Table 7: bitstream generation and on-board reliability.

    Derived entirely from the b_status / b_cycles columns of main.csv:

      b_status == 'placed'      placement succeeded, routing failed
      b_status == 'bitstream'   an xclbin was produced
      b_cycles > 0              the bitstream also executed on the FPGA

    Anything that never reached placement would appear as some other b_status and is
    counted in the Placement column, so the three failure columns plus Success always
    sum to Total.
    """
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("  \\small")
    latex.append("  \\centering")
    latex.append("  \\caption{Bitstream Generation and On-board Reliability}")
    latex.append("  \\label{tab:robustness}")
    latex.append("  \\begin{tabular}{lcc|ccc}")
    latex.append("    \\hline")
    latex.append("    \\multirow{2}{*}{\\textbf{Category}} & \\multirow{2}{*}{\\textbf{Total}} & "
                 "\\multirow{2}{*}{\\textbf{Success}} & \\multicolumn{3}{c}{\\textbf{Failure}} \\\\")
    latex.append("    \\cline{4-6}")
    latex.append("    & & &  \\textbf{Placement} & \\textbf{Routing} & \\textbf{On-board} \\\\")
    latex.append("    \\hline")

    totals = dict(total=0, success=0, place=0, route=0, board=0)
    for bench in benchmarks:
        sub = df[df["benchmark"] == bench]
        if sub.empty:
            continue
        bit = sub[sub["b_status"] == "bitstream"]
        route = int((sub["b_status"] == "placed").sum())
        success = int((bit["b_cycles"] > 0).sum())
        board = int((bit["b_cycles"] <= 0).sum())
        place = int(len(sub) - len(bit) - route)
        latex.append(f"    {get_label(bench, 'bench', tex=True)} & {len(sub)} & {success} & "
                     f"{place} & {route} & {board} \\\\")
        totals["total"] += len(sub)
        totals["success"] += success
        totals["place"] += place
        totals["route"] += route
        totals["board"] += board

    latex.append("    \\hline")
    latex.append(f"    \\textbf{{Total}} & \\textbf{{{totals['total']}}} & \\textbf{{{totals['success']}}} & "
                 f"\\textbf{{{totals['place']}}} & \\textbf{{{totals['route']}}} & "
                 f"\\textbf{{{totals['board']}}} \\\\")
    latex.append("    \\hline")
    latex.append("  \\end{tabular}")
    latex.append("\\end{table}")
    return "\n".join(latex) + "\n"


def generate_gpt2_dse_stats_table(df):
    # Process separate runs: shift unrolling iterations (needed for correctly identifying the final state)
    def get_abs_iteration(row):
        if row['stage'] == 'unrolling':
            offset = 10000 if row['solver'] == 'SA' else 1000
            return row['iteration'] + offset
        return row['iteration']

    df['abs_iteration'] = df.apply(get_abs_iteration, axis=1)

    # Get baseline costs (max cost at iteration 0 for each strategy)
    baselines = df[df['abs_iteration'] == 0].groupby('hw_strategy')['cost'].max().to_dict()

    # Get final state for each run
    final_df = df.sort_values('abs_iteration').groupby(['solver', 'type', 'hw_strategy', 'run']).last().reset_index()
    
    hw_strategies = ["no_reuse", "balanced", "max_reuse"]
    solvers = ["SA", "GA"]
    mtypes = ["combined", "separate"]
    
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("  \\small")
    latex.append("  \\centering")
    latex.append("  \\caption{GPT-2 DSE Comparison (Costs in $10^6$)}")
    latex.append("  \\label{tab:dse_stats}")
    latex.append("  \\begin{tabular}{llcccc}")
    latex.append("    \\toprule")
    latex.append("    \\textbf{Strat.} & \\textbf{Method} & \\textbf{Best} & \\textbf{Mean} & \\textbf{CV} & \\textbf{Imp.} \\\\")
    latex.append("    \\midrule")
    
    for hw_strat in hw_strategies:
        baseline = baselines.get(hw_strat, 1e9)
        strat_label = get_label(hw_strat, 'strat', tex=True)
        
        # We'll use \multirow for the strategy
        first_row = True
        
        for solver in solvers:
            for mtype in mtypes:
                subset = final_df[(final_df['solver'] == solver) & 
                                  (final_df['type'] == mtype) & 
                                  (final_df['hw_strategy'] == hw_strat)]
                
                if subset.empty: continue
                
                best = subset['cost'].min()
                mean = subset['cost'].mean()
                std = subset['cost'].std()
                cv = (std / mean * 100) if mean > 0 else 0
                imp = baseline / mean if mean > 0 else 0
                
                # Get consistent method label
                # SA + separate -> sa_sa
                # SA + combined -> sa_comb
                solver_key = f"{solver.lower()}_{'comb' if mtype == 'combined' else solver.lower()}"
                method_label = get_label(solver_key, 'solver', tex=True)
                
                if first_row:
                    latex.append(f"    \\multirow{{4}}{{*}}{{{strat_label}}}")
                    first_row = False
                
                latex.append(f"    & {method_label:<6} & {best/1e6:>6.2f} & {mean/1e6:>6.2f} & {cv:>6.1f}\\% & {imp:>6.1f}$\\times$ \\\\")
        
        if hw_strat != hw_strategies[-1]:
            latex.append("    \\midrule")
            
    latex.append("    \\bottomrule")
    latex.append("  \\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("-s", "--strategy", help="Filter by u_strategy (e.g. minlp_ndiv_min)")
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv)
    df.columns = [c.strip() for c in df.columns]
    print(generate_ppa_table(df, config_filter=args.strategy))
