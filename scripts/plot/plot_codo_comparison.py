import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from .theme import FS_TITLE, FS_AXIS, FS_TICK, FS_ANNOT, FS_LEGEND, STRAT_COLORS, apply_base_theme, get_strat_color, get_label

def draw_codo_plot(ax, df_bench_all, df_bench_codo, tests, groups, num_bars_per_test, width, x, cycles_col, normalized, show_all, show_xlabel=True, log_scale=False):
    apply_base_theme(ax)
    ax.set_facecolor('#fdfdfd')
    
    # Pre-calculate Codo baseline for speedup
    codo_baselines = {}
    for test in tests:
        subset = df_bench_codo[df_bench_codo['test_name'] == test]
        valid_vals = subset[cycles_col][subset[cycles_col] > 0]
        if not valid_vals.empty:
            codo_baselines[test] = valid_vals.min()
        else:
            codo_baselines[test] = 0

    tab10 = plt.cm.get_cmap('tab10')

    # Plot framework groups
    for i, group in enumerate(groups):
        config_data = []
        
        if show_all:
            c_name, s_name = group
            label = f"{get_label(c_name, 'config')} ({get_label(s_name, 'strat')})"
            color = get_strat_color(s_name)
            if color == '#999999': # Fallback
                color = tab10(i)
        else:
            c_name = group
            label = get_label(c_name, 'config')
            color = tab10(i % 10)

        bar_colors = []

        for test in tests:
            if show_all:
                subset = df_bench_all[(df_bench_all['config'] == c_name) & (df_bench_all['hw_strategy'] == s_name) & (df_bench_all['test'] == test)]
            else:
                subset = df_bench_all[(df_bench_all['config'] == c_name) & (df_bench_all['test'] == test)]
            
            min_val = 1
            valid_vals = subset[cycles_col][subset[cycles_col] >= min_val]
            
            if not valid_vals.empty:
                idx = valid_vals.idxmin()
                abs_val = valid_vals.loc[idx]
                strat = subset.loc[idx, 'hw_strategy']
                bar_colors.append(get_strat_color(strat))
                
                if normalized:
                    codo_base = codo_baselines.get(test, 0)
                    if abs_val > 0 and codo_base > 0:
                        val = codo_base / abs_val
                    else:
                        val = 0
                else:
                    val = abs_val
            else:
                val = 0
                bar_colors.append('#999999')
            
            config_data.append(val)
        
        pos = x + (i - (num_bars_per_test-1)/2) * width
        floor = 0.1 if normalized else 1
        if log_scale:
            config_data_plot = [max(v, floor) for v in config_data]
            heights = [v - floor for v in config_data_plot]
            bars = ax.bar(pos, heights, width, label=label, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=0.5, bottom=floor)
        else:
            bars = ax.bar(pos, config_data, width, label=label, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Add 'x' for missing data
        for j, val in enumerate(config_data):
            if val == 0:
                ax.text(pos[j], floor if log_scale else 0, 'x', ha='center', va='bottom', color='red', fontsize=14, fontweight='bold')
        
        # Bar labels
        fmt = '%.2f' if normalized else '%.0f'
        if log_scale:
            labels = [fmt % v if v > 0 else "" for v in config_data]
            ax.bar_label(bars, labels=labels, rotation=90, padding=3, fontsize=FS_ANNOT)
        else:
            ax.bar_label(bars, fmt=fmt, rotation=90, padding=3, fontsize=FS_ANNOT)

    if not normalized:
        # Plot Codo baseline as absolute bars
        codo_vals = [codo_baselines[t] for t in tests]
        i = len(groups)
        pos = x + (i - (num_bars_per_test-1)/2) * width
        
        if log_scale:
            codo_vals_plot = [max(v, floor) for v in codo_vals]
            heights = [v - floor for v in codo_vals_plot]
            bars = ax.bar(pos, heights, width, label='Codo Baseline', color='#333333', bottom=floor)
        else:
            bars = ax.bar(pos, codo_vals, width, label='Codo Baseline', color='#333333')

        for j, val in enumerate(codo_vals):
            if val == 0:
                ax.text(pos[j], floor if log_scale else 0, 'x', ha='center', va='bottom', color='red', fontsize=14, fontweight='bold')
        
        if log_scale:
            labels = ['%.0f' % v if v > 0 else "" for v in codo_vals]
            ax.bar_label(bars, labels=labels, rotation=90, padding=3, fontsize=FS_ANNOT, color='#333333')
        else:
            ax.bar_label(bars, fmt='%.0f', rotation=90, padding=3, fontsize=FS_ANNOT, color='#333333')
    else:
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Codo Baseline (1.0)')
    
    if log_scale:
        ax.set_yscale('log')
        if normalized:
            ax.set_ylim(bottom=floor)
        else:
            # Set a reasonable floor for cycles
            ax.set_ylim(bottom=floor)

    # Formatting
    bench = df_bench_all['benchmark'].iloc[0]
    title_suffix = "(Normalized Speedup)" if normalized else "(Absolute Cycles)"
    ax.set_title(f"Framework vs Codo Comparison: {get_label(bench, 'bench')} {title_suffix}", fontsize=FS_TITLE, fontweight='bold')
    ax.set_ylabel("Speedup (Codo/FW)" if normalized else cycles_col.replace('_', ' ').title(), fontsize=FS_AXIS)
    
    if show_xlabel:
        ax.set_xlabel('Test Name', fontsize=FS_AXIS)
    else:
        ax.set_xlabel('', fontsize=FS_AXIS)

    ax.set_xticks(x)
    ax.set_xticklabels([get_label(t, 'test') for t in tests], fontsize=FS_TICK, rotation=30, ha='right')

def plot(df_all, df_codo, output_dir, cycles_col='rtl_cycles', normalized=False, show_all=False, log_scale=False, ext="png"):
    df_all.columns = [c.strip() for c in df_all.columns]
    df_codo.columns = [c.strip() for c in df_codo.columns]
    benchmarks = sorted(df_all['benchmark'].unique())
    num_bench = len(benchmarks)
    
    os.makedirs(output_dir, exist_ok=True)
    mode_str = "speedup" if normalized else cycles_col
    output_path_base = os.path.join(output_dir, f"comparison_{mode_str}")

    # Combined plot setup
    fig_all, axes_all = plt.subplots(num_bench, 1, figsize=(18, 4 * num_bench + 2), squeeze=False)

    for i, bench in enumerate(benchmarks):
        df_bench_all = df_all[df_all['benchmark'] == bench]
        df_bench_codo = df_codo[df_codo['benchmark'] == bench]
        if df_bench_all.empty: continue
        
        tests = sorted(df_bench_all['test'].unique())
        if show_all:
            groups = sorted(df_bench_all[['config', 'hw_strategy']].drop_duplicates().values.tolist(), key=lambda x: (x[0], x[1]))
        else:
            groups = sorted(df_bench_all['config'].unique())
            
        num_groups = len(groups)
        num_bars_per_test = num_groups + (0 if normalized else 1)
        x = np.arange(len(tests))
        width = 0.8 / max(1, num_bars_per_test)

        # 1. Individual Plot
        fig_single, ax_single = plt.subplots(figsize=(18, 5)) # Increased height slightly
        draw_codo_plot(ax_single, df_bench_all, df_bench_codo, tests, groups, num_bars_per_test, width, x, cycles_col, normalized, show_all, show_xlabel=True, log_scale=log_scale)
        
        handles, labels = ax_single.get_legend_handles_labels()
        from matplotlib.patches import Patch
        strat_handles = [Patch(color=c, label=get_label(s, 'strat')) for s, c in STRAT_COLORS.items()]
        ax_single.legend(handles + strat_handles, labels + [get_label(s, 'strat') for s in STRAT_COLORS], 
                         title="Configs & Strategies", loc='upper left', bbox_to_anchor=(1, 1), fontsize=FS_LEGEND)
        
        plt.tight_layout()
        plot_path = f"{output_path_base}_{bench}.{ext}"
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close(fig_single)
        print(f"Comparison plot saved to {plot_path}")

        # 2. Combined Plot
        draw_codo_plot(axes_all[i, 0], df_bench_all, df_bench_codo, tests, groups, num_bars_per_test, width, x, cycles_col, normalized, show_all, show_xlabel=(i == num_bench - 1), log_scale=log_scale)

    # Shared Legend for Combined Plot
    handles, labels = axes_all[0, 0].get_legend_handles_labels()
    from matplotlib.patches import Patch
    strat_handles = [Patch(color=c, label=get_label(s, 'strat')) for s, c in STRAT_COLORS.items()]
    fig_all.legend(handles + strat_handles, labels + [get_label(s, 'strat') for s in STRAT_COLORS], 
                   title='Configs & Strategies', loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=min(6, len(labels) + 3), frameon=True, facecolor='white', fontsize=FS_LEGEND)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94], h_pad=0.2)
    combined_path = f"{output_path_base}_combined.{ext}"
    plt.savefig(combined_path, bbox_inches='tight')
    plt.close(fig_all)
    print(f"Combined comparison plot saved to {combined_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--all_csv", type=str, required=True)
    parser.add_argument("--codo_csv", type=str, required=True)
    parser.add_argument("--out", type=str, default="figs")
    parser.add_argument("--cycles", type=str, default="rtl_cycles")
    parser.add_argument("-n", "--normalized", action="store_true")
    parser.add_argument("--show_all", action="store_true")
    parser.add_argument("--log", action="store_true", help="Use log scale for y-axis")
    args = parser.parse_args()
    df_all = pd.read_csv(args.all_csv)
    df_codo = pd.read_csv(args.codo_csv)
    plot(df_all, df_codo, args.out, cycles_col=args.cycles, normalized=args.normalized, show_all=args.show_all, log_scale=args.log)
