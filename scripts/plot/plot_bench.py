import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from .theme import FS_TITLE, FS_AXIS, FS_TICK, FS_ANNOT, FS_LEGEND, STRAT_COLORS, apply_base_theme, get_strat_color, get_label

def draw_bench_panel(ax, df_label, tests, hw_order, x, width, cycles_col, label, normalized, baselines, show_xlabel=True, log_scale=False):
    apply_base_theme(ax)
    ax.set_facecolor('#fdfdfd')
    
    for j, hw in enumerate(hw_order):
        hw_data = []
        for test in tests:
            row_data = df_label[(df_label['test'] == test) & (df_label['hw_strategy'] == hw)]
            if not row_data.empty:
                val = row_data[cycles_col].values[0]
                if val <= 0:
                    val = 0
                
                if normalized:
                    base = baselines.get(test, 0)
                    if val > 0 and base > 0:
                        hw_data.append(base / val) # Speedup
                    else:
                        hw_data.append(0)
                else:
                    hw_data.append(val)
            else:
                hw_data.append(0)
        
        pos = x + (j - 1) * width
        color = get_strat_color(hw)
        
        floor = 0.1 if (normalized or log_scale) else 0
        plot_data = [max(v, floor) if v > 0 else 0 for v in hw_data]
        
        if log_scale:
            heights = [v - floor if v > 0 else 0 for v in plot_data]
            bars = ax.bar(pos, heights, width, label=hw, color=color, alpha=0.8, edgecolor='white', linewidth=0.5, bottom=floor)
        else:
            bars = ax.bar(pos, plot_data, width, label=hw, color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Bar labels (only if not too many tests)
        if len(tests) <= 15:
            fmt = '%.2f' if normalized else '%.0f'
            if log_scale:
                labels = [fmt % v if v > 0 else "" for v in hw_data]
                ax.bar_label(bars, labels=labels, rotation=90, padding=3, fontsize=FS_ANNOT)
            else:
                ax.bar_label(bars, fmt=fmt, rotation=90, padding=3, fontsize=FS_ANNOT)

    ax.set_title(f"Configuration: {get_label(label, 'config')}", fontsize=FS_TITLE-2, fontweight='bold', loc='left')
    
    if normalized:
        y_label = f"Speedup (vs {get_label('div_min', 'config')})"
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    else:
        y_label = get_label(cycles_col, 'column')
    
    ax.set_ylabel(y_label, fontsize=FS_AXIS)
    
    if log_scale:
        ax.set_yscale('log')
        if normalized:
            ax.set_ylim(bottom=0.1)
    elif normalized:
        # Standardize range to [0, 2] with 1.0 in the middle
        ax.set_ylim(0, 2)

    ax.set_xticks(x)
    if show_xlabel:
        ax.set_xticklabels([get_label(t, 'test') for t in tests], fontsize=FS_TICK, rotation=30, ha='right')
    else:
        ax.set_xticklabels([])

def plot(df, output_dir, cycles_col='p_cycles', normalized=False, log_scale=False, ext="png", output_name=None):
    # Filter for successful DSE
    df = df[df['p_status'] == 'Pass'].copy()
    df[cycles_col] = pd.to_numeric(df[cycles_col], errors='coerce')
    df = df[df[cycles_col] > 0].copy()
    
    if df.empty:
        print(f"No successful projects found with valid cycles ({cycles_col})")
        return

    df['div_label'] = df['u_strategy'].apply(lambda x: get_label(x, 'config'))
    
    hw_order = ['no_reuse', 'balanced', 'max_reuse']
    target_labels = ['Div-Min', 'Non-Div-Min', 'Non-Div-Max']
    
    # Pre-calculate baselines (min cycles for Div-Min per test)
    baselines = {}
    div_min_df = df[df['div_label'] == 'Div-Min']
    for test in df['test'].unique():
        test_df = div_min_df[div_min_df['test'] == test]
        if not test_df.empty:
            # Only consider positive cycle counts for the baseline
            valid_baselines = test_df[test_df[cycles_col] > 0][cycles_col]
            if not valid_baselines.empty:
                baselines[test] = valid_baselines.min()
            else:
                print(f"Warning: No valid baseline cycles found for test {test} in Div-Min. Plots will be empty.")
                baselines[test] = 0
    
    # Determine categorical order
    df['div_label'] = pd.Categorical(df['div_label'], categories=target_labels, ordered=True)
    df['hw_strategy'] = pd.Categorical(df['hw_strategy'], categories=hw_order, ordered=True)
    
    benchmarks = sorted(df['benchmark'].unique())
    
    # Pre-calculate layout for wide plot
    all_tests = []
    all_x_pos = []
    current_x_offset = 0
    separators = []
    bench_baselines = {} # test -> baseline
    
    for i, bench in enumerate(benchmarks):
        bench_df = df[df['benchmark'] == bench]
        tests = sorted(bench_df['test'].unique())
        
        for t in tests:
            all_tests.append(t)
            all_x_pos.append(current_x_offset)
            # Find baseline for this test (from Div-Min)
            t_df = bench_df[(bench_df['test'] == t) & (bench_df['div_label'] == 'Div-Min')]
            if not t_df.empty:
                bench_baselines[t] = t_df[cycles_col].min()
            else:
                bench_baselines[t] = 0
                
            current_x_offset += 1
            
        if i < len(benchmarks) - 1:
            # The current_x_offset is already at X_last + 1.0.
            # We want the separator exactly in the middle of that 1.0 unit gap.
            separators.append(current_x_offset - 0.5)

    x = np.array(all_x_pos)
    width = 0.25
    
    # Create 3-panel figure
    fig, axes = plt.subplots(3, 1, figsize=(max(20, len(all_tests)*1.4), 8.5), sharex=True)
    
    for i, label in enumerate(target_labels):
        df_label = df[df['div_label'] == label]
        draw_bench_panel(axes[i], df_label, all_tests, hw_order, x, width, cycles_col, label, 
                         normalized, bench_baselines, show_xlabel=(i == 2), log_scale=log_scale)
        
        # Tighten X-axis limits to match benchmark spacing
        axes[i].set_xlim(all_x_pos[0] - 0.8, all_x_pos[-1] + 0.8)
        
        # Add vertical separators
        for sep in separators:
            axes[i].axvline(x=sep, color='#999999', linestyle='--', linewidth=1, alpha=0.5)

    mode_str = "Speedup Analysis" if normalized else "Performance Analysis"
    bench_str = ", ".join([get_label(b, 'bench') for b in benchmarks])
    # fig.suptitle(f"Comparison {mode_str}: {bench_str}", fontsize=FS_TITLE+2, fontweight='bold', y=0.97)
    
    # Shared Legend
    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, [get_label(l, 'strat') for l in labels_leg], title='Hardware Mapping Strategies', 
               loc='upper center', bbox_to_anchor=(0.5, 0.91), ncol=3, frameon=True, 
               facecolor='white', fontsize=FS_LEGEND + 2, title_fontsize=FS_LEGEND + 4)
    
    plt.tight_layout(rect=[0, 0, 1, 0.91], h_pad=1.0)
    
    os.makedirs(output_dir, exist_ok=True)
    mode_str = "speedup" if normalized else "perf"
    bench_suffix = "_".join(benchmarks) if len(benchmarks) <= 2 else "combined"
    
    if output_name:
        filename = f"{output_name}.{ext}"
    else:
        filename = f"ndiv_{bench_suffix}_{mode_str}.{ext}"
        
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Div vs Non-Div plot saved to {output_path}")

    # Summary Statistics
    print("\n" + "="*60)
    summary_col = 'rtl_cycles' if 'rtl_cycles' in df.columns else cycles_col
    print(f"Speedup Summary: ({summary_col} Div-Min / {summary_col} Non-Div)")
    print("="*60)
    
    summary_labels = target_labels[1:] # Non-Div-Min, Non-Div-Max
    total_ratios = {label: [] for label in summary_labels}
    all_speedups = []
    
    for hw in hw_order:
        print(f"\nHardware Mapping Strategy: {get_label(hw, 'strat')}")
        df_hw = df[df['hw_strategy'] == hw]
        df_baseline = df_hw[df_hw['div_label'] == 'Div-Min']
        
        for label in summary_labels:
            ratios = []
            df_target = df_hw[df_hw['div_label'] == label]
            
            for test in all_tests:
                row_t = df_target[df_target['test'] == test]
                row_b = df_baseline[df_baseline['test'] == test]
                
                if not row_t.empty and not row_b.empty:
                    val = row_t[summary_col].values[0]
                    base = row_b[summary_col].values[0]
                    if val > 0 and base > 0:
                        speedup = base / val
                        ratios.append(speedup)
                        all_speedups.append({
                            'test': test,
                            'hw': hw,
                            'config': label,
                            'speedup': speedup
                        })
            
            if ratios:
                avg = np.mean(ratios)
                geo = np.exp(np.mean(np.log(ratios)))
                min_val = np.min(ratios)
                max_val = np.max(ratios)
                total_ratios[label].extend(ratios)
                print(f"  {label:15}: Avg = {avg:.4f}, Geo = {geo:.4f}, Min = {min_val:.4f}, Max = {max_val:.4f}")
            else:
                print(f"  {label:15}: No data")

    print("\n" + "-"*60)
    print("TOTAL SPEEDUP SUMMARY (Across all HW strategies)")
    print("-"*60)
    for label in summary_labels:
        ratios = total_ratios[label]
        if ratios:
            avg = np.mean(ratios)
            geo = np.exp(np.mean(np.log(ratios)))
            min_val = np.min(ratios)
            max_val = np.max(ratios)
            print(f"  {label:15}: Total Avg = {avg:.4f}, Geo Mean = {geo:.4f}, Min = {min_val:.4f}, Max = {max_val:.4f}")
    
    print("\n" + "*"*60)
    print(f"TOP 5 SPEEDUP CASES:")
    all_speedups.sort(key=lambda x: x['speedup'], reverse=True)
    for i, item in enumerate(all_speedups[:5]):
        print(f"  {i+1}. {item['speedup']:.4f}x - {get_label(item['test'], 'test')} "
              f"({get_label(item['hw'], 'strat')} + {item['config']})")
    print("*"*60 + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bench", type=str, default="*")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--cycles", type=str, default="rtl_cycles")
    parser.add_argument("-n", "--normalized", action="store_true")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--ext", type=str, default="png")
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv)
    df.columns = [c.strip() for c in df.columns]
    
    if args.bench != "*":
        df = df[df['benchmark'] == args.bench]
        
    plot(df, args.out, cycles_col=args.cycles, normalized=args.normalized, log_scale=args.log, ext=args.ext)
