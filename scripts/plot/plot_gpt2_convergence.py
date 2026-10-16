import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from .theme import FS_TITLE, FS_AXIS, FS_TICK, FS_ANNOT, FS_LEGEND, STRAT_COLORS, apply_base_theme, get_strat_color, get_label
FS_TITLE  = FS_TITLE  * 1.3
FS_AXIS   = FS_AXIS   * 1.3
FS_TICK   = FS_TICK   * 1.3
FS_ANNOT  = FS_ANNOT  * 1.3
FS_LEGEND = FS_LEGEND * 1.3
def main():
    parser = argparse.ArgumentParser(description="Plot GPT-2 DSE convergence traces.")
    parser.add_argument("-i", "--input", type=str, default="study/data/gpt2_dse/gpt2_dse_traces.csv", help="Input traces CSV")
    parser.add_argument("-o", "--output", type=str, default="study/figs/gpt2_convergence.pdf", help="Output plot file")
    args = parser.parse_args()

    prj_root = os.environ.get("PRJ_ROOT", os.getcwd())
    csv_path = os.path.join(prj_root, args.input)
    
    if not os.path.exists(csv_path):
        print(f"Error: Traces file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Process separate runs: shift unrolling iterations
    # We'll map stage labels to absolute iterations
    # permutation: 0-9999
    # unrolling: 10000-19999
    def get_abs_iteration(row):
        if row['stage'] == 'unrolling':
            offset = 10000 if row['solver'] == 'SA' else 1000
            return row['iteration'] + offset
        return row['iteration']

    df['abs_iteration'] = df.apply(get_abs_iteration, axis=1)

    # Create 4 horizontal subplots with shared Y axis
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True, figsize=(24, 7))
    
    # Define combinations
    plot_configs = [
        {"solver": "SA", "type": "combined", "ax": ax1, "title": "SA: Combined Search", "offset": 10000},
        {"solver": "SA", "type": "separate", "ax": ax2, "title": "SA: Separate Search", "offset": 10000},
        {"solver": "GA", "type": "combined", "ax": ax3, "title": "GA: Combined Search", "offset": 1000},
        {"solver": "GA", "type": "separate", "ax": ax4, "title": "GA: Separate Search", "offset": 1000},
    ]

    hw_strategies = ["no_reuse", "balanced", "max_reuse"]

    # --- Calculate Baseline Costs ---
    # We use the max cost at iteration 0 across all configs for each strategy
    # to find the "true" initial design cost before any optimization.
    baselines = df[df['abs_iteration'] == 0].groupby('hw_strategy')['cost'].max().to_dict()

    # --- Calculate Global Statistics for Annotations ---
    all_stats = df.groupby(['solver', 'type', 'hw_strategy'])['cost'].min().reset_index()
    global_min_row = all_stats.loc[all_stats['cost'].idxmin()]
    global_min_cost = global_min_row['cost']
    best_config_name = f"{global_min_row['solver']} {global_min_row['type'].capitalize()} ({get_label(global_min_row['hw_strategy'], 'strat')})"
    
    final_costs = df[df['abs_iteration'] == df['abs_iteration'].max()].groupby('hw_strategy')['cost'].mean()
    if not final_costs.empty:
        max_strat_gap = (final_costs.max() / final_costs.min())
        gap_text = f"Max Strategy Gap: {max_strat_gap:.1f}x"
    else:
        gap_text = ""

    for cfg in plot_configs:
        ax = cfg["ax"]
        apply_base_theme(ax)
        ax.set_title(cfg["title"], fontsize=FS_TITLE, fontweight='bold', pad=25)
        ax.set_xlabel("Iterations", fontsize=FS_AXIS)
        if ax == ax1:
            ax.set_ylabel("Cost (Log Scale)", fontsize=FS_AXIS + 4)
        
        # Filter data for this plot
        sub_df = df[(df['solver'] == cfg['solver']) & (df['type'] == cfg['type'])]
        
        for hw_strat in hw_strategies:
            color = get_strat_color(hw_strat)
            label = get_label(hw_strat, 'strat')
            
            config_df = sub_df[sub_df['hw_strategy'] == hw_strat]
            if config_df.empty: continue
            
            # Group by iteration to get stats
            stats = config_df.groupby('abs_iteration')['cost'].agg(['mean', 'min', 'max']).reset_index()
            
            # FORCE BASELINE ALIGNMENT:
            # Ensure all curves start at the same initial cost for this strategy
            baseline = baselines.get(hw_strat)
            if baseline is not None and stats.iloc[0]['abs_iteration'] == 0:
                stats.loc[stats['abs_iteration'] == 0, ['mean', 'min', 'max']] = baseline
            
            # Plot average line
            ax.plot(stats['abs_iteration'], stats['mean'], color=color, linewidth=2.5, 
                    label=label, zorder=3)
            
            # Cloud/Shaded area
            ax.fill_between(stats['abs_iteration'], stats['min'], stats['max'], 
                            facecolor=color, alpha=0.15, zorder=2)

            # Special Notation: Mark the BEST point
            if cfg['solver'] == global_min_row['solver'] and cfg['type'] == global_min_row['type'] and hw_strat == global_min_row['hw_strategy']:
                best_iter = stats.loc[stats['min'] == global_min_cost, 'abs_iteration'].iloc[0]
                ax.scatter(best_iter, global_min_cost, color='gold', marker='*', s=200, 
                           edgecolor='black', linewidth=1, zorder=10, label="Global Best")
                ax.annotate("Global Best", xy=(best_iter, global_min_cost), xytext=(20, 20),
                            textcoords="offset points", arrowprops=dict(arrowstyle="->", color='black', alpha=0.6),
                            fontsize=FS_ANNOT, fontweight='bold')

        # Global Baseline Line
        ax.axhline(y=global_min_cost, color='#2c3e50', linestyle='--', linewidth=0.8, alpha=0.4, zorder=1)

        # Log scale
        ax.set_yscale('log')
        
        # Vertical divider for separate runs
        if cfg["type"] == "separate":
            offset = cfg["offset"]
            ax.axvline(x=offset, color='#95a5a6', linestyle=':', linewidth=1, alpha=0.6)
            ax.text(offset -  (offset*0.2), ax.get_ylim()[0]*1.2, "Phase Transition", 
                    rotation=90, verticalalignment='bottom', color='#7f8c8d', fontsize=FS_ANNOT)

        ax.grid(True, which="both", ls="-", alpha=0.1)

    # # Add gap summary in the last plot
    # ax4.text(0.95, 0.05, f"Best: {best_config_name}\n{gap_text}", 
    #          transform=ax4.transAxes, ha='right', va='bottom', 
    #          bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='#bdc3c7'),
    #          fontsize=FS_ANNOT)

    # Add a single shared legend at the top
    handles, labels = ax1.get_legend_handles_labels()
    
    fig.legend(handles, labels, loc='upper center', ncol=3, 
               frameon=True, facecolor='white', framealpha=0.9, 
               fontsize=FS_LEGEND + 4, bbox_to_anchor=(0.5, 0.85))

    plt.tight_layout()
    plt.subplots_adjust(top=0.82)
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Convergence plot saved to {args.output}")

if __name__ == "__main__":
    main()
