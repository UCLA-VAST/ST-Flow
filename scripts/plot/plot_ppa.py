import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from .theme import (FS_TITLE, FS_AXIS, FS_TICK, FS_ANNOT, FS_LEGEND, 
                    STRAT_COLORS, FPGA_LIMITS, apply_base_theme, 
                    get_strat_color, get_label)

def draw_ppa_panel(ax, df, tests, hw_order, x, width, metric_col, label_name, cycle_col, normalized=False, baselines=None, log_scale=False):
    apply_base_theme(ax)
    ax.set_facecolor('#fdfdfd')
    
    floor = 0
    if log_scale:
        ax.set_yscale('log')
        floor = 1e-1 # Small floor for log scale visibility
    
    for j, hw in enumerate(hw_order):
        hw_data = []
        missing_mask = []
        for test in tests:
            row_data = df[(df['test'] == test) & (df['hw_strategy'] == hw)]
            if not row_data.empty:
                val = row_data[metric_col].values[0]
                cycle_val = row_data[cycle_col].values[0]
                
                if cycle_val > 0:
                    if normalized and baselines and test in baselines and baselines[test] > 0:
                        hw_data.append(baselines[test] / val if val > 0 else 0)
                    else:
                        hw_data.append(val if val > 0 else 0)
                    missing_mask.append(False)
                else:
                    hw_data.append(0)
                    missing_mask.append(True)
            else:
                hw_data.append(0)
                missing_mask.append(True)
        
        pos = x + (j - 1) * width
        color = get_strat_color(hw)
        
        plot_data = [max(v, floor) if v > 0 else 0 for v in hw_data]
        
        if log_scale:
            bars = ax.bar(pos, [v-floor if v > 0 else 0 for v in plot_data], width, 
                          bottom=floor, label=hw, color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
        else:
            bars = ax.bar(pos, plot_data, width, label=hw, color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Mark missing data with an X
        for k, is_missing in enumerate(missing_mask):
            if is_missing:
                ax.text(pos[k], floor*1.5 if log_scale else 0.05, 'X', 
                        ha='center', va='bottom', color='red', fontsize=FS_ANNOT, fontweight='bold')

        # Bar labels (only if normalized or for small values)
        if normalized or metric_col == 'area_score':
            ax.bar_label(bars, fmt='%.2f', rotation=90, padding=3, fontsize=FS_ANNOT)

    ax.set_title(label_name, fontsize=FS_TITLE-2, fontweight='bold', loc='left')
    ax.set_ylabel(label_name, fontsize=FS_AXIS)
    
    if normalized:
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([get_label(t, 'test') for t in tests], fontsize=FS_TICK, rotation=30, ha='right')

def plot(df, output_dir, use_bitstream=False, log_scale=False, ext="png", output_name=None):
    # Select columns based on vitis or bitstream
    prefix = 'b' if use_bitstream else 'v'
    cycle_col = 'b_cycles' if use_bitstream else 'rtl_cycles'
    dsp_col = f'{prefix}_dsps'
    mem_col = f'{prefix}_mem'
    
    # Filter for successful projects (at least passed Vitis)
    df = df[df['rtl_cycles'] > 0].copy()
    if df.empty:
        print(f"No successful projects found with valid RTL cycles.")
        return

    # To prevent skewed geometric means, we only calculate Area/ADP for valid runs.
    # If a run is missing resources (marked -1), it should not pull down the GM.
    valid_res_mask = (df[dsp_col] > 0) & (df[mem_col] > 0)
    
    # Calculate Area Score and ADP
    df['area_score'] = np.nan
    df['adp'] = np.nan
    
    # Apply calculation only where resources are valid
    df.loc[valid_res_mask, 'area_score'] = (df.loc[valid_res_mask, dsp_col] / FPGA_LIMITS['dsp']) + \
                                          (df.loc[valid_res_mask, mem_col] / FPGA_LIMITS['mem_equiv'])
    
    # ADP is only valid if both cycles and area are valid
    valid_adp_mask = valid_res_mask & (df[cycle_col] > 0)
    df.loc[valid_adp_mask, 'adp'] = df.loc[valid_adp_mask, cycle_col] * df.loc[valid_adp_mask, 'area_score']
    
    hw_order = ['no_reuse', 'balanced', 'max_reuse']
    benchmarks = sorted(df['benchmark'].unique())
    
    # Pre-calculate layout for wide plot
    all_tests = []
    all_x_pos = []
    current_x_offset = 0
    separators = []
    
    for i, bench in enumerate(benchmarks):
        bench_df = df[df['benchmark'] == bench]
        tests = sorted(bench_df['test'].unique())
        
        for t in tests:
            all_tests.append(t)
            all_x_pos.append(current_x_offset)
            current_x_offset += 1
            
        if i < len(benchmarks) - 1:
            separators.append(current_x_offset - 0.5)

    x = np.array(all_x_pos)
    width = 0.28
    
    # Create 3-panel figure
    fig, axes = plt.subplots(3, 1, figsize=(max(18, len(all_tests)*1.4), 10), sharex=True)
    
    # Row 1: Delay (Cycles)
    draw_ppa_panel(axes[0], df, all_tests, hw_order, x, width, cycle_col, 'Delay (Cycles)', cycle_col, log_scale=log_scale)
    
    # Row 2: Area (Normalized Score)
    draw_ppa_panel(axes[1], df, all_tests, hw_order, x, width, 'area_score', 'Area Score (DSP+Mem)', cycle_col, log_scale=False)
    
    # Row 3: ADP (Lower is better)
    draw_ppa_panel(axes[2], df, all_tests, hw_order, x, width, 'adp', 'Area-Delay Product (ADP)', cycle_col, log_scale=log_scale)
    
    # Add vertical separators and tighten limits
    for ax in axes:
        ax.set_xlim(x[0] - 0.8, x[-1] + 0.8)
        for sep in separators:
            ax.axvline(x=sep, color='#999999', linestyle='--', linewidth=1, alpha=0.5)

    mode_str = "Bitstream" if use_bitstream else "Vitis-Estimated"
    bench_str = ", ".join([get_label(b, 'bench') for b in benchmarks])
    fig.suptitle(f"PPA Analysis ({mode_str}): {bench_str}", fontsize=FS_TITLE+2, fontweight='bold', y=0.98)
    
    # Shared Legend
    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, [get_label(l, 'strat') for l in labels_leg], title='Hardware Strategies', 
               loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3, frameon=True, facecolor='white', fontsize=FS_LEGEND)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93], h_pad=1.2)
    
    os.makedirs(output_dir, exist_ok=True)
    bench_suffix = "_".join(benchmarks) if len(benchmarks) <= 2 else "combined"
    log_suffix = "_log" if log_scale else ""
    
    if output_name:
        filename = f"{output_name}{log_suffix}.{ext}"
    else:
        filename = f"ppa_{prefix}_{bench_suffix}{log_suffix}.{ext}"
        
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"PPA plot saved to {output_path}")

    # --- Summary Analysis ---
    print("\n" + "="*40)
    print("PPA STRATEGY SUMMARY (Geometric Means)")
    print("="*40)
    
    summary_data = []
    for hw in hw_order:
        hw_df = df[df['hw_strategy'] == hw]
        if hw_df.empty: continue
        
        # Using Geometric Mean for ratios/products is generally better
        def gmean(series):
            clean = series.dropna()
            return np.exp(np.log(clean[clean > 0]).mean()) if not clean.empty else np.nan

        summary_data.append({
            'Strategy': get_label(hw, 'strat'),
            'Delay': gmean(hw_df[cycle_col]),
            'Area': gmean(hw_df['area_score']),
            'ADP': gmean(hw_df['adp'])
        })
    
    summary_df = pd.DataFrame(summary_data)
    if not summary_df.empty:
        # Find best strategy (lowest ADP)
        best_idx = summary_df['ADP'].idxmin()
        best_strat = summary_df.loc[best_idx, 'Strategy']
        
        # Calculate improvements relative to the WORST strategy for context
        worst_adp = summary_df['ADP'].max()
        summary_df['Improvement'] = (worst_adp / summary_df['ADP']).apply(lambda x: f"{x:.2f}x")
        
        print(summary_df.to_string(index=False))
        print("-" * 40)
        print(f"OVERALL BEST STRATEGY: {best_strat}")
        print("="*40 + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--bitstream", action="store_true")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--ext", type=str, default="png")
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv)
    df.columns = [c.strip() for c in df.columns]
    plot(df, args.out, use_bitstream=args.bitstream, log_scale=args.log, ext=args.ext)
