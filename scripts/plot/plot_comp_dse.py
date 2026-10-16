import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from .theme import FS_TITLE, FS_AXIS, FS_TICK, FS_ANNOT, FS_LEGEND, STRAT_COLORS, apply_base_theme, get_strat_color, get_label

def draw_dse_panel(ax, df_label, tests, hw_order, methods, x, width, cycles_col, label, baselines, show_xlabel=True, log_scale=False):
    apply_base_theme(ax)
    ax.set_facecolor('#fdfdfd')
    
    num_hw = len(hw_order)
    num_methods = len(methods)
    total_bars = num_hw * num_methods
    
    # Calculate a safe bar width that avoids overlap
    # width is the total fraction of the x-interval [0, 1] dedicated to one test
    bar_width = (width / total_bars) * 0.85
    
    # Search Method Alphas (to distinguish SA, SA-Comb, GA, GA-Comb)
    # Using a list to match the methods order
    alphas = [0.35, 0.55, 0.75, 1.0] 
    hatches = ['', '...', '', '///'] # Added some subtle hatching for better B&W/accessibility
    
    for i, hw in enumerate(hw_order):
        color = get_strat_color(hw)
        for j, method in enumerate(methods):
            method_data = []
            for test in tests:
                row = df_label[(df_label['test'] == test) & 
                               (df_label['hw_strategy'] == hw) & 
                               (df_label['search_method'] == method)]
                if not row.empty:
                    val = row[cycles_col].values[0]
                    base = baselines.get((test, hw), 0)
                    if val > 0 and base > 0:
                        method_data.append(base / val) # Speedup relative to MINLP
                    else:
                        method_data.append(0)
                else:
                    method_data.append(0)
            
            # Global index of this bar within the test cluster
            k = i * num_methods + j
            # Center the cluster around x
            pos = x + (k - (total_bars - 1) / 2) * (width / total_bars)
            
            floor = 0.1 if log_scale else 0
            plot_data = [max(v, floor) if v > 0 else 0 for v in method_data]
            
            bars = ax.bar(pos, plot_data, bar_width, 
                         color=color, alpha=alphas[j], hatch=hatches[j],
                         edgecolor='black', linewidth=0.3)
            
    ax.set_title(f"Tiling Strategy: {get_label(label, 'config')}", fontsize=FS_TITLE-2, fontweight='bold', loc='left')
    ax.set_ylabel("Speedup (vs MINLP)", fontsize=FS_AXIS)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylim(bottom=0.1)

    if show_xlabel:
        ax.set_xticklabels([get_label(t, 'test') for t in tests], fontsize=FS_TICK, rotation=30, ha='right')
    else:
        ax.set_xticklabels([])

def plot(benchmark, df, output_dir, cycles_col='p_cycles', log_scale=False, ext="png", output_name=None):
    # 1. Pre-process Data
    df = df[df['benchmark'] == benchmark].copy()
    df = df[df['p_status'] == 'Pass']
    
    # Map configurations to search methods
    def map_search_method(row):
        config = row['config'].lower()
        if 'comp_' in config or 'minlp' in config: return 'MINLP'
        if 'sa_sa' in config: return 'SA'
        if 'sa_comb' in config or 'comb_sa' in config: return 'SA-Comb'
        if 'ga_ga' in config: return 'GA'
        if 'ga_comb' in config or 'comb_ga' in config: return 'GA-Comb'
        return 'Unknown'
    
    df['search_method'] = df.apply(map_search_method, axis=1)
    
    df['div_label'] = df['u_strategy'].apply(lambda x: get_label(x, 'config'))
    
    target_labels = ['Div-Min', 'Non-Div-Min', 'Non-Div-Max']
    hw_order = ['no_reuse', 'balanced', 'max_reuse']
    methods_to_plot = ['SA', 'SA-Comb', 'GA', 'GA-Comb']
    
    # 2. Calculate Baselines (MINLP per test and per hw_strategy and per tiling)
    baselines = {}
    for tiling in target_labels:
        tiling_df = df[(df['div_label'] == tiling) & (df['search_method'] == 'MINLP')]
        for test in df['test'].unique():
            for hw in hw_order:
                val = tiling_df[(tiling_df['test'] == test) & (tiling_df['hw_strategy'] == hw)][cycles_col]
                if not val.empty and val.iloc[0] > 0:
                    baselines[(test, hw, tiling)] = val.iloc[0]
    
    tests = sorted(df[df['search_method'] != 'MINLP']['test'].unique())
    if not tests:
        print(f"No stochastic DSE results found for benchmark {benchmark}")
        return

    x = np.arange(len(tests))
    width = 0.8 # Total width for all 12 bars per test
    
    fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)
    
    for i, label in enumerate(target_labels):
        df_label = df[df['div_label'] == label]
        current_baselines = { (k[0], k[1]): v for k, v in baselines.items() if k[2] == label }
        
        draw_dse_panel(axes[i], df_label, tests, hw_order, methods_to_plot, x, width, 
                       cycles_col, label, current_baselines, show_xlabel=(i == 2), log_scale=log_scale)

    fig.suptitle(f"Search Method Comparison (Normalized to MINLP): {get_label(benchmark, 'bench')}", 
                 fontsize=FS_TITLE+2, fontweight='bold', y=0.98)
    
    # --- Dual Legend Construction ---
    from matplotlib.patches import Patch
    
    # Strategy Legend (Colors)
    strat_legends = [Patch(facecolor=STRAT_COLORS[hw], label=get_label(hw, 'strat')) for hw in hw_order]
    
    # Method Legend (Alpha/Hatch)
    alphas = [0.35, 0.55, 0.75, 1.0]
    hatches = ['', '...', '', '///']
    method_legends = [Patch(facecolor='grey', alpha=alphas[i], hatch=hatches[i], label=methods_to_plot[i]) 
                      for i in range(len(methods_to_plot))]
    
    # Add combined legend
    leg1 = fig.legend(handles=strat_legends, loc='upper left', bbox_to_anchor=(0.1, 0.96), 
                      title='Hardware Strategy', ncol=1, frameon=True, fontsize=FS_LEGEND)
    leg2 = fig.legend(handles=method_legends, loc='upper right', bbox_to_anchor=(0.9, 0.96), 
                      title='Search Method', ncol=4, frameon=True, fontsize=FS_LEGEND)
    
    fig.text(0.5, 0.01, "Bars are clustered by Hardware Strategy (No-Reuse, Balanced, Max-Reuse). Within each cluster, bars represent search methods (SA, SA-Comb, GA, GA-Comb).", 
             ha='center', fontsize=FS_AXIS, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93], h_pad=2.0)
    
    os.makedirs(output_dir, exist_ok=True)
    if output_name:
        filename = f"{output_name}.{ext}"
    else:
        filename = f"comp_dse_{benchmark}.{ext}"
        
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"DSE Comparison plot saved to {output_path}")
    
    # 3. Print Stats
    print(f"\n--- Statistics for {benchmark} ---")
    stats = []
    for tiling in target_labels:
        for method in methods_to_plot:
            for hw in hw_order:
                m_df = df[(df['div_label'] == tiling) & (df['search_method'] == method) & (df['hw_strategy'] == hw)]
                if m_df.empty: continue
                
                speedups = []
                for _, row in m_df.iterrows():
                    test = row['test']
                    base = baselines.get((test, hw, tiling), 0)
                    val = row[cycles_col]
                    if val > 0 and base > 0:
                        speedups.append(base / val)
                
                if speedups:
                    stats.append({
                        'Tiling': tiling,
                        'Method': method,
                        'HW': hw,
                        'Avg Speedup': np.mean(speedups),
                        'Min Speedup': np.min(speedups),
                        'Max Speedup': np.max(speedups),
                        'Count': len(speedups)
                    })
    
    stats_df = pd.DataFrame(stats)
    if not stats_df.empty:
        print(stats_df.to_string(index=False))
        summary = stats_df.groupby(['Tiling', 'Method'])['Avg Speedup'].mean().unstack()
        print("\nSummary (Average Speedup across HW strategies):")
        print(summary)
    else:
        print("No statistics available.")
