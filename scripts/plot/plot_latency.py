import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse
from .theme import FS_TITLE, FS_AXIS, FS_TICK, FS_ANNOT, FS_LEGEND, STRAT_COLORS, apply_base_theme, get_strat_color, get_label

def plot_latency(csv_path, output_path, metric='p_cycles'):
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Strategies to plot in requested order
    target_strategies = ['no_reuse', 'balanced', 'max_reuse']
    colors = ['#FF6B6B', '#4D96FF', '#6BCB77'] # Vibrant palette: Redish, Bluish, Greenish

    # Load data
    df = pd.read_csv(csv_path)
    # Clean column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    # Convert metric to numeric and filter out invalid data
    df[metric] = pd.to_numeric(df[metric], errors='coerce')
    df = df[df[metric] > 0].copy()

    # Use the new columns
    # Include solvers in project name to distinguish variants (e.g., comb_ga vs ga_ga)
    df['project'] = df['test'] + "_" + df['p_solver'] + "_" + df['u_solver']
    df['strategy'] = df['strategy']
    df['folder'] = df['benchmark']

    # Group by folder
    folders = df['folder'].unique()

    for folder in folders:
        folder_df = df[df['folder'] == folder].copy()
        
        # Pivot to get strategies as columns: project becomes index
        pivot_df = folder_df.pivot(index='project', columns='strategy', values=metric)
        
        # Reorder columns to match target strategies, only if they exist in data
        available_strategies = [s for s in target_strategies if s in pivot_df.columns]
        if not available_strategies:
            continue
        pivot_df = pivot_df[available_strategies]
        
        # Normalize by min latency of the available strategies for each project
        min_latencies = pivot_df.min(axis=1)
        normalized_df = pivot_df.div(min_latencies, axis=0)
        
        # Plotting with Rich Aesthetics
        try:
            plt.style.use('seaborn-v0_8-muted')
        except:
            plt.style.use('ggplot') # Fallback
            
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Create the bar plot
        # Use custom colors mapping strategies to hex codes
        color_map = {strat: colors[i] for i, strat in enumerate(target_strategies)}
        current_colors = [color_map[s] for s in available_strategies]
        
        normalized_df.plot(
            kind='bar', 
            ax=ax, 
            width=0.8, 
            color=current_colors,
            edgecolor='white',
            linewidth=0.5
        )
        
        # Log scale for Y-axis
        # ax.set_yscale('log')
        
        # Formatting
        metric_title = "ST-Flow Latency" if metric == 'p_cycles' else "RTL Latency"
        fig.suptitle(f"Multi-Strategy Latency Comparison: {get_label(folder, 'bench')}", fontsize=FS_TITLE+2, fontweight='bold', y=0.98)
    
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color=STRAT_COLORS[s], lw=4, label=get_label(s, 'strat')) for s in available_strategies]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3, 
                   frameon=True, facecolor='white', fontsize=FS_LEGEND)

        ax.set_ylabel(f'Normalized {metric_title} (Log Scale, Lower is Better)', fontsize=12, labelpad=10)
        ax.set_xlabel('Project Variant', fontsize=12, labelpad=10)
        
        # Add a horizontal line at 1.0 (baseline)
        ax.axhline(y=1.0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
        
        # Custom ticks and grid
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, which='both', color='gray', linestyle='dashed', alpha=0.2)
        ax.xaxis.grid(False)
        
        plt.xticks(rotation=30, ha='right', fontsize=10)
        
        # Set major y-ticks for log scale if needed
        from matplotlib.ticker import LogLocator, ScalarFormatter
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        
        # Apply base theme
        apply_base_theme(ax)
        
        # Remove top/right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add labels on top of bars for clarity
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                label = f'{height:.1f}x' if height >= 10 else f'{height:.2f}'
                ax.annotate(label, 
                            (p.get_x() + p.get_width() / 2., height), 
                            ha = 'center', va = 'center', 
                            xytext = (0, 7), 
                            textcoords = 'offset points',
                            fontsize=8,
                            alpha=0.8)

        plt.tight_layout()
        
        # Save as PDF
        plot_name = output_path
        plot_name = plot_name.replace('.pdf', f'_{folder}_{metric}.pdf')
        plt.savefig(plot_name, bbox_inches='tight')
        plt.close()
        print(f"Successfully generated plot: {plot_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot latency comparison.')
    parser.add_argument('-i','--input', type=str, required=True, help='Path to CSV file.')
    parser.add_argument('-o','--output', type=str, required=True, help='Path to output directory (filename prefix).')
    parser.add_argument('-m','--metric', type=str, default='p_cycles', choices=['p_cycles', 'rtl_cycles'], help='Metric to plot.')
    args = parser.parse_args()
    
    plot_latency(args.input, args.output, args.metric)
