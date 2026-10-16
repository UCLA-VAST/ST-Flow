import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from .theme import FS_TITLE, FS_AXIS, FS_TICK, FS_ANNOT, FS_LEGEND, STRAT_COLORS, apply_base_theme, get_label

FS_TITLE  = 15
FS_AXIS   = 14
FS_TICK   = 10
FS_ANNOT  = 9
FS_LEGEND = 12

def draw_validation_plot(ax, plot_df, folder, baseline, strat_colors, strat_order, width, tick_labels, tick_positions=None, show_xlabel=True, show_ylabel=True):
    # Plot bars by strategy
    for strat in strat_order:
        strat_data = plot_df[plot_df['hw_strategy'] == strat]
        if strat_data.empty: continue
        
        color = strat_colors.get(strat, '#CCCCCC')
        
        # ST-Flow Bars (Solid)
        ax.bar(strat_data['x_pos'] - width/2, strat_data['ST-Flow'], width, 
               color=color, edgecolor='white', linewidth=0.5)
        
        # Vitis Bars (Transparent/Light)
        ax.bar(strat_data['x_pos'] + width/2, strat_data['Vitis_Base'], width, 
               color=color, alpha=0.3, edgecolor=color, linewidth=1.0)
        
        # Error bars for Vitis (only if range is > 0.05)
        v_diff = (strat_data['Vitis_Max'] - strat_data['Vitis_Base']).fillna(0)
        mask = v_diff > 0.05
        if mask.any():
            ax.errorbar(strat_data['x_pos'][mask] + width/2, strat_data['Vitis_Base'][mask], 
                        yerr=[np.zeros(mask.sum()), v_diff[mask]], 
                        fmt='none', ecolor='#333333', capsize=3, elinewidth=0.8, alpha=0.6)

    # Baseline horizontal line at 1.0 (Ideal)
    ax.axhline(y=1.0, color='#333333', linestyle='--', linewidth=1.5, alpha=0.8)

    # Formatting
    apply_base_theme(ax)
    # ax.set_title(f'Accuracy - {get_label(folder, "bench")}', fontsize=FS_TITLE, fontweight='bold', pad=10)
    if show_ylabel:
        ax.set_ylabel(f'Ratio to {get_label(baseline, "column")}', fontsize=FS_AXIS)
    else:
        ax.set_ylabel('', fontsize=FS_AXIS)
    # if show_xlabel:
    #     ax.set_xlabel('Test Name', fontsize=FS_AXIS)
    # else:
    #     ax.set_xlabel('', fontsize=FS_AXIS)

    # X-axis ticks
    if tick_positions is None:
        tick_positions = [i*4 + 1 for i in range(len(tick_labels))]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([get_label(t, 'test') for t in tick_labels], rotation=30, ha='right', fontsize=FS_TICK)
    
    # Tighten margins
    if len(tick_positions) > 0:
        ax.set_xlim(min(tick_positions) - 2, max(tick_positions) + 2)

    # Log scale logic
    max_ratio = max(plot_df['ST-Flow'].max(), plot_df['Vitis_Max'].max())
    if max_ratio > 2:
        ax.set_yscale('log')
        from matplotlib.ticker import ScalarFormatter
        ax.yaxis.set_major_formatter(ScalarFormatter())

    # Annotate bars
    for _, row in plot_df.iterrows():
        x_p = row['x_pos']
        # ST-Flow
        val = row['ST-Flow']
        if not np.isnan(val) and val > 0:
            label = f'{val:.2f}' if val < 10 else f'{val:.1f}'
            ax.annotate(label, (x_p - width/2, 1.1), ha='center', va='bottom', rotation=90, 
                        xytext=(0, 2), textcoords='offset points', fontsize=FS_ANNOT)
        
        # Vitis
        v_base, v_max = row['Vitis_Base'], row['Vitis_Max']
        if not np.isnan(v_base) and v_base > 0:
            is_range = abs(v_max - v_base) > 0.05
            label = f'[{v_base:.2f}, {v_max:.2f}]' if is_range else f'{v_base:.2f}'
            
            # Logic to avoid overlap: if it's a horizontal range label, it might 
            # collide with the vertical ST-Flow label. Push it slightly higher.
            y_pos = max(v_max, 1.1) if is_range else (1.1 if v_max < 1.1 else v_max)
            y_off = 2 if not is_range else 7
            
            # If the horizontal range label is close to the ST-Flow label (at 1.1),
            # push it up just enough to clear the vertical text.
            if is_range and abs(y_pos - 1.1) < 0.9:
                y_off = 20 
            
            ax.annotate(label, (x_p + width/2, y_pos), 
                        ha='center', va='bottom' if not is_range else 'center', 
                        rotation=90 if not is_range else 0,
                        xytext=(0, y_off), 
                        textcoords='offset points', fontsize=FS_ANNOT)

def plot(df, output_dir, baseline="rtl_cycles", ext="png", output_name=None):
    os.makedirs(output_dir, exist_ok=True)
    output_path_base = os.path.join(output_dir, f"validation_{baseline}")

    # Clean column names
    df.columns = [c.strip() for c in df.columns]

    # Filter for valid baseline data
    df[baseline] = pd.to_numeric(df[baseline], errors='coerce')
    df = df[df[baseline] > 0].copy()
    if df.empty:
        print(f"No valid baseline data ({baseline}) found for plotting.")
        return

    # Normalize metrics
    valid_metrics = ['p_cycles', 'v_min_cycles', 'v_max_cycles']
    for m in valid_metrics:
        df[m] = pd.to_numeric(df[m], errors='coerce')
        df[f'{m}_norm'] = df[m] / df[baseline]

    # Project identification
    df['folder'] = df['benchmark']

    # Group by folder
    folders = df['folder'].unique()
    num_folders = len(folders)
    
    # Combined plot setups
    fig_vert, axes_vert = plt.subplots(num_folders, 1, figsize=(18, 4 * num_folders), squeeze=False)
    
    # Pre-calculate layout for wide plot
    strat_order = ['no_reuse', 'balanced', 'max_reuse']
    strat_colors = STRAT_COLORS
    
    wide_plot_dfs = []
    all_tick_labels = []
    all_tick_positions = []
    current_x_offset = 0
    separators = []
    
    for i, folder in enumerate(folders):
        folder_df = df[df['folder'] == folder].copy()
        
        # Prepare for plotting
        plot_df = folder_df.copy()
        plot_df['ST-Flow'] = plot_df['p_cycles_norm'].apply(lambda x: x if x > 0 else np.nan)
        plot_df['Vitis_Base'] = plot_df['v_min_cycles_norm'].apply(lambda x: x if x > 0 else np.nan)
        plot_df['Vitis_Max'] = plot_df['v_max_cycles_norm'].apply(lambda x: x if x > 0 else np.nan)
        
        plot_df['strat_idx'] = plot_df['hw_strategy'].apply(lambda x: strat_order.index(x) if x in strat_order else 99)
        plot_df = plot_df.sort_values(['test', 'strat_idx'])
        
        unique_tests = sorted(plot_df['test'].unique())
        test_to_idx = {t: i for i, t in enumerate(unique_tests)}
        plot_df['x_pos'] = plot_df.apply(lambda row: test_to_idx[row['test']] * 4 + row['strat_idx'], axis=1)
        
        width = 0.42

        # 1. Plot on individual figure
        fig_single, ax_single = plt.subplots(figsize=(18, 4))
        draw_validation_plot(ax_single, plot_df, folder, baseline, strat_colors, strat_order, width, unique_tests)
        
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=STRAT_COLORS['no_reuse'], lw=4, label=get_label('no_reuse', 'strat')),
            Line2D([0], [0], color=STRAT_COLORS['balanced'], lw=4, label=get_label('balanced', 'strat')),
            Line2D([0], [0], color=STRAT_COLORS['max_reuse'], lw=4, label=get_label('max_reuse', 'strat')),
            Line2D([0], [0], marker='s', color='w', label='ST-Flow (Estimate)', markerfacecolor='#666666', markersize=10),
            Line2D([0], [0], marker='s', color='w', label='Vitis (Range)', markerfacecolor='#CCCCCC', markersize=10),
        ]
        ax_single.legend(handles=legend_elements, title='Strategy & Estimator', 
                         bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=FS_LEGEND, title_fontsize=FS_LEGEND)
        
        plt.tight_layout()
        plot_name = f"{output_path_base}_{folder}.{ext}"
        plt.savefig(plot_name, bbox_inches='tight')
        plt.close(fig_single)
        print(f"Generated validation plot: {plot_name}")

        # 2. Plot on combined vertical figure
        draw_validation_plot(axes_vert[i, 0], plot_df, folder, baseline, strat_colors, strat_order, width, unique_tests, show_xlabel=(i == num_folders - 1))

        # 3. Prepare data for combined wide figure
        wide_df = plot_df.copy()
        wide_df['x_pos'] += current_x_offset
        wide_plot_dfs.append(wide_df)
        
        all_tick_labels.extend(unique_tests)
        all_tick_positions.extend([p + current_x_offset for p in [idx*4 + 1 for idx in range(len(unique_tests))]])
        
        current_x_offset += len(unique_tests) * 4 + 1 # Add a small gap between benchmarks
        if i < num_folders - 1:
            separators.append(current_x_offset - 1.0)

        # Calculate Per-Folder Accuracy Metrics
        def calc_mape(norm_series):
            valid = norm_series.dropna()
            return (abs(1 - valid)).mean() * 100 if not valid.empty else np.nan
        
        def calc_corr(df, col1, col2):
            if col1 == col2: return 1.0
            valid = df[[col1, col2]].dropna()
            if len(valid) < 2: return np.nan
            # Use iloc to select by position in case of duplicate column names
            return valid.iloc[:, 0].corr(valid.iloc[:, 1])

        mape_p = calc_mape(plot_df['ST-Flow'])
        mape_v_min = calc_mape(plot_df['Vitis_Base'])
        mape_v_max = calc_mape(plot_df['Vitis_Max'])
        
        corr_p = calc_corr(plot_df, 'p_cycles', baseline)
        corr_v_min = calc_corr(plot_df, 'v_min_cycles', baseline)
        corr_v_max = calc_corr(plot_df, 'v_max_cycles', baseline)
        
        print(f"  Accuracy Summary for {folder}:")
        print(f"    - ST-Flow:   MAPE = {mape_p:6.2f}%, Corr = {corr_p:.4f}")
        print(f"    - Vitis Min: MAPE = {mape_v_min:6.2f}%, Corr = {corr_v_min:.4f}")
        print(f"    - Vitis Max: MAPE = {mape_v_max:6.2f}%, Corr = {corr_v_max:.4f}")

    # 4. Finalize combined figures
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=STRAT_COLORS['no_reuse'], lw=4, label=get_label('no_reuse', 'strat')),
        Line2D([0], [0], color=STRAT_COLORS['balanced'], lw=4, label=get_label('balanced', 'strat')),
        Line2D([0], [0], color=STRAT_COLORS['max_reuse'], lw=4, label=get_label('max_reuse', 'strat')),
        Line2D([0], [0], marker='s', color='w', label='ST-Flow (Estimate)', markerfacecolor='#666666', markersize=10),
        Line2D([0], [0], marker='s', color='w', label='Vitis (Range Estimate)', markerfacecolor='#CCCCCC', markersize=10),
    ]
    
    # Finalize Vertical
    fig_vert.legend(handles=legend_elements, title='Strategy & Estimator', 
                   loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=5, 
                   frameon=True, facecolor='white', fontsize=FS_LEGEND)
    plt.tight_layout(rect=[0, 0, 1, 0.94], h_pad=0.2)
    
    if output_name:
        vert_plot_name = os.path.join(output_dir, f"{output_name}_vertical.{ext}")
    else:
        vert_plot_name = f"{output_path_base}_vertical.{ext}"
        
    fig_vert.savefig(vert_plot_name, bbox_inches='tight')
    plt.close(fig_vert)
    print(f"Generated combined vertical validation plot: {vert_plot_name}")

    # Finalize Wide
    fig_wide, ax_wide = plt.subplots(figsize=(max(18, len(all_tick_labels)*0.6), 4))
    full_wide_df = pd.concat(wide_plot_dfs)
    draw_validation_plot(ax_wide, full_wide_df, "All Benchmarks", baseline, strat_colors, strat_order, width, 
                         all_tick_labels, tick_positions=all_tick_positions)
    
    # Add vertical separators between benchmarks
    for sep in separators:
        ax_wide.axvline(x=sep, color='#999999', linestyle='--', linewidth=1, alpha=0.5)
        
    ax_wide.legend(handles=legend_elements, title='Strategy & Estimator', 
                   loc='upper left', bbox_to_anchor=(0.4, 1.2), ncol=5, 
                   frameon=True, facecolor='white', fontsize=FS_LEGEND, title_fontsize=FS_LEGEND)
    
    plt.tight_layout()
    
    if output_name:
        wide_plot_name = os.path.join(output_dir, f"{output_name}_wide.{ext}")
    else:
        wide_plot_name = f"{output_path_base}_wide.{ext}"
        
    plt.savefig(wide_plot_name, bbox_inches='tight')
    plt.close(fig_wide)
    print(f"Generated combined wide validation plot: {wide_plot_name}")

    # Global MAPE Summary
    print("\n" + "="*40)
    print("GLOBAL VALIDATION SUMMARY")
    print("="*40)
    
    def calc_global_mape(col):
        valid = df[df[col] > 0][f'{col}_norm'].dropna()
        return (abs(1 - valid)).mean() * 100 if not valid.empty else np.nan

    def calc_global_corr(col):
        if col == baseline: return 1.0
        valid = df[[col, baseline]].dropna()
        valid = valid[valid[col] > 0]
        if len(valid) < 2: return np.nan
        # Use iloc to select by position in case of duplicate column names
        return valid.iloc[:, 0].corr(valid.iloc[:, 1])

    g_mape_p = calc_global_mape('p_cycles')
    g_mape_v_min = calc_global_mape('v_min_cycles')
    g_mape_v_max = calc_global_mape('v_max_cycles')

    g_corr_p = calc_global_corr('p_cycles')
    g_corr_v_min = calc_global_corr('v_min_cycles')
    g_corr_v_max = calc_global_corr('v_max_cycles')

    print(f"ST-Flow   - Aggregate MAPE: {g_mape_p:6.2f}%, Corr: {g_corr_p:.4f}")
    print(f"Vitis Min - Aggregate MAPE: {g_mape_v_min:6.2f}%, Corr: {g_corr_v_min:.4f}")
    print(f"Vitis Max - Aggregate MAPE: {g_mape_v_max:6.2f}%, Corr: {g_corr_v_max:.4f}")
    print("="*40 + "\n")
