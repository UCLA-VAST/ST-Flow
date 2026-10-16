import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from .theme import FS_TITLE, FS_AXIS, FS_TICK, FS_ANNOT, FS_LEGEND, STRAT_COLORS, apply_base_theme, get_strat_color, get_label

def plot(df, output_dir, cycles_col='p_cycles', ext="png", output_name=None):
    os.makedirs(output_dir, exist_ok=True)
    if output_name:
        output_path = os.path.join(output_dir, f"{output_name}.{ext}")
    else:
        output_path = os.path.join(output_dir, f"summary_{cycles_col}.{ext}")

    # Clean column names
    df.columns = [c.strip() for c in df.columns]

    # Filter for Pass and valid metric
    df[cycles_col] = pd.to_numeric(df[cycles_col], errors='coerce')
    df = df[(df['p_status'] == 'Pass') & (df[cycles_col] > 0)].copy()

    if df.empty:
        print(f"No passed experiments found to plot for {cycles_col}.")
        return

    # Create a unique experiment ID (Solver Pair + Strategy)
    df['config_label'] = df['p_solver'] + "_" + df['u_solver'] + " (" + df['hw_strategy'].apply(lambda x: get_label(x, 'strat')) + ")"
    
    # Aggregate stats across runs
    agg_df = df.groupby('config_label')[cycles_col].agg(['min', 'max', 'mean', 'count']).reset_index()
    
    # Sort by minimum latency (best performance first)
    agg_df = agg_df.sort_values(by='min', ascending=False) # Plotting top-down

    # Setup fig and ax
    fig, ax = plt.subplots(figsize=(12, max(4, 0.8 * len(agg_df))))
    
    # Setup aesthetics
    apply_base_theme(ax)
    DEFAULT_COLOR = '#888888'
    
    # Map colors to bars based on which strategy is in the config name
    bar_colors = []
    for config in agg_df['config_label']:
        color = DEFAULT_COLOR
        for strat in STRAT_COLORS:
            if strat in config:
                color = STRAT_COLORS[strat]
                break
        bar_colors.append(color)

    # Draw bars for the BEST (MIN) latency
    error_config = [np.zeros(len(agg_df)), agg_df['max'] - agg_df['min']]

    bars = ax.barh(
        agg_df['config_label'], 
        agg_df['min'], 
        xerr=error_config,
        color=bar_colors,
        edgecolor='white',
        linewidth=0.8,
        alpha=0.85,
        capsize=5,
        error_kw={'ecolor': 'black', 'lw': 1, 'alpha': 0.6}
    )

    # Log scale if range is large
    if agg_df['max'].max() / agg_df['min'].min() > 10:
        ax.set_xscale('log')
        ax.set_xlabel(f'{cycles_col} (Log Scale, Lower is Better)', fontsize=FS_AXIS, labelpad=15)
    else:
        ax.set_xlabel(f'{cycles_col} (Lower is Better)', fontsize=FS_AXIS, labelpad=15)

    # Formatting
    ax.set_title(f'Performance Summary across Multi-Run Experiments\nMetric: {cycles_col}', fontsize=FS_TITLE, fontweight='bold', pad=25)
    ax.set_ylabel('Experiment Configuration (Solvers & Strategy)', fontsize=FS_AXIS, labelpad=15)
    
    # Custom Grid
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, which='both', color='gray', linestyle='dashed', alpha=0.3)
    ax.yaxis.grid(False)

    # Hide spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Add numeric labels on/near bars
    for i, (idx, row) in enumerate(agg_df.iterrows()):
        val = row['min']
        count = int(row['count'])
        label = f"Best: {val:,} (in {count} runs)"
        ax.text(val, i, f"  {label}", va='center', fontsize=FS_ANNOT, fontweight='bold', color='#333333')

    # Legend for colors
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=color, lw=4, label=get_label(strat, 'strat'))
                       for strat, color in STRAT_COLORS.items()]
    ax.legend(handles=legend_elements, loc='lower right', title='Mapping Strategy', 
               frameon=True, facecolor='white', fontsize=FS_LEGEND)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Summary plot generated: {output_path}")
