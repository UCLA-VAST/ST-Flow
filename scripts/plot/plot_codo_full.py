import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from .theme import apply_base_theme, get_strat_color, STRAT_COLORS, FS_TITLE, FS_AXIS, FS_TICK, FS_ANNOT, FS_LEGEND, get_label
FS_TITLE  = 19
FS_AXIS   = 16
FS_TICK   = 16
FS_ANNOT  = 16
FS_LEGEND = 16
FS_LABEL  = 16
def plot(df, df_codo, output_dir, u_strategies=None, normalized=False, show_all=False, log_scale=False):
    if u_strategies is None:
        u_strategies = ['div_min', 'ndiv_min']
    
    # Filter benchmarks
    target_benchmarks = ['01_dnn', '02_synthetic', '03_polybench']
    df = df[df['benchmark'].isin(target_benchmarks)].copy()
    df_codo = df_codo[df_codo['benchmark'].isin(target_benchmarks)].copy()
    
    # Filter out CODO tests that exceed hardware DSP capacity (9024)
    # df_codo = df_codo[df_codo['v_dsps'] <= 9024.0].copy()

    # PPA Constants (from paper/tex)
    # Alveo U55C capacities
    DSP_CAP = 9024.0
    BRAM_CAP = 4032.0
    URAM_CAP = 960.0
    MEM_CAP = BRAM_CAP + 16.0 * URAM_CAP

    def calc_ppa(row):
        rtl_cycles = row.get('rtl_cycles', -1)
        if pd.isna(rtl_cycles) or rtl_cycles <= 0:
            return np.nan
        
        # Resources
        dsps = row.get('v_dsps', 0)
        # Handle different column names between CODO and ST-Flow
        # CODO uses v_brams, v_urams. ST-Flow uses v_mem for BRAM.
        brams = row.get('v_brams', row.get('v_bram', row.get('v_mem', 0)))
        urams = row.get('v_urams', row.get('v_uram', 0))
        
        area_score = (dsps / DSP_CAP) + ((brams + 16.0 * urams) / MEM_CAP)
        return rtl_cycles * area_score

    # Prepare data structure for plotting
    plot_data = []
    
    # Get all unique tests across both dataframes for the target benchmarks
    all_tests = []
    for bench in target_benchmarks:
        phx_tests = df[df['benchmark'] == bench]['test'].unique()
        codo_tests = set(df_codo[df_codo['benchmark'] == bench]['test_name'].unique())
        # Only include tests that are available in the CODO baseline
        combined_tests = sorted([t for t in phx_tests if t in codo_tests])
        for t in combined_tests:
            all_tests.append((bench, t))
    
    for bench, test in all_tests:
        # Diagnostic printing for 05_k2mm as requested
        is_k2mm = (test == '01_atax')
        if is_k2mm:
            print(f"\n[PPA Details for {test} ({bench})]")

        # CODO baseline
        codo_row = df_codo[(df_codo['benchmark'] == bench) & (df_codo['test_name'] == test)]
        if codo_row.empty:
            codo_lat = np.nan
            codo_ppa = np.nan
        else:
            codo_row = codo_row.iloc[0]
            codo_lat = codo_row['rtl_cycles'] if codo_row['rtl_cycles'] > 0 else np.nan
            codo_ppa = calc_ppa(codo_row)
            if is_k2mm:
                dsps = codo_row.get('v_dsps', 0)
                brams = codo_row.get('v_brams', 0)
                urams = codo_row.get('v_urams', 0)
                area_score = (dsps / DSP_CAP) + ((brams + 16.0 * urams) / MEM_CAP)
                print(f"  CODO: Latency={codo_lat}, DSPs={dsps}, BRAMs={brams}, URAMs={urams}, AreaScore={area_score:.4f}, PPA={codo_ppa:.2e}")

        entry = {
            'test': test,
            'benchmark': bench,
            'codo_lat': codo_lat,
            'codo_ppa': codo_ppa
        }

        for u_strat in u_strategies:
            strat_df = df[(df['benchmark'] == bench) & (df['test'] == test) & (df['u_strategy'] == u_strat)]
            if strat_df.empty:
                entry[f'{u_strat}_lat'] = np.nan
                entry[f'{u_strat}_ppa'] = np.nan
                entry[f'{u_strat}_strat'] = None
            else:
                # Pick best hw_strategy based on rtl_cycles
                valid_df = strat_df[strat_df['rtl_cycles'] > 0]
                if valid_df.empty:
                    entry[f'{u_strat}_lat'] = 0 # Mark as failed/missing
                    entry[f'{u_strat}_ppa'] = 0
                    entry[f'{u_strat}_strat'] = None
                else:
                    best_row = valid_df.loc[valid_df['rtl_cycles'].idxmin()]
                    entry[f'{u_strat}_lat'] = best_row['rtl_cycles']
                    entry[f'{u_strat}_ppa'] = calc_ppa(best_row)
                    entry[f'{u_strat}_strat'] = best_row['hw_strategy']
                    
                    if is_k2mm:
                        dsps = best_row.get('v_dsps', 0)
                        brams = best_row.get('v_mem', 0) # ST-Flow uses v_mem for BRAM
                        urams = 0
                        area_score = (dsps / DSP_CAP) + ((brams + 16.0 * urams) / MEM_CAP)
                        print(f"  ST-Flow ({u_strat}, {best_row['hw_strategy']}): Latency={best_row['rtl_cycles']}, DSPs={dsps}, BRAMs={brams}, AreaScore={area_score:.4f}, PPA={entry[f'{u_strat}_ppa']:.2e}")
        
        plot_data.append(entry)

    # [DEBUG] ST-Flow Test Information (Remove after debugging)
    print("\n" + "-"*110)
    print(f"{'DEBUG: ST-Flow Test Information & Selection':^110}")
    print("-"*110)
    print(f"{'Test':<20} | {'U-Strat':<10} | {'Best HW-Strat':<15} | {'P-Cycles':>12} | {'RTL-Cycles':>12} | {'P-Solv':<8} | {'U-Solv':<8}")
    print("-"*110)
    plotted_tests = [t[1] for t in all_tests]
    debug_df = df[df['test'].isin(plotted_tests) & df['u_strategy'].isin(u_strategies)].copy()
    for t in plotted_tests:
        for us in u_strategies:
            subset = debug_df[(debug_df['test'] == t) & (debug_df['u_strategy'] == us)]
            valid = subset[subset['rtl_cycles'] > 0]
            if not valid.empty:
                best_idx = valid['rtl_cycles'].idxmin()
                for _, row in subset.sort_values('hw_strategy').iterrows():
                    mark = " [PICKED]" if row.name == best_idx else ""
                    p_cyc = row.get('p_cycles', 0)
                    r_cyc = row.get('rtl_cycles', 0)
                    # Handle cases where rtl_cycles might be NaN or 0 but still in subset
                    r_cyc_str = f"{r_cyc:>12.0f}" if not pd.isna(r_cyc) and r_cyc > 0 else f"{'FAILED':>12}"
                    print(f"{row['test']:<20} | {row['u_strategy']:<10} | {row['hw_strategy']:<15} | {p_cyc:>12.0f} | {r_cyc_str} | {row.get('p_solver','N/A'):<8} | {row.get('u_solver','N/A'):<8}{mark}")
            print("-" * 110)
    print("\n")

    # Convert to DataFrame for easier plotting
    p_df = pd.DataFrame(plot_data)
    
    # Normalization
    if normalized:
        for u_strat in u_strategies:
            # Latency Speedup: CODO / ST-Flow
            p_df[f'{u_strat}_lat_norm'] = p_df['codo_lat'] / p_df[f'{u_strat}_lat']
            # PPA Improvement: CODO / ST-Flow (since PPA is a product, lower is better)
            p_df[f'{u_strat}_ppa_norm'] = p_df['codo_ppa'] / p_df[f'{u_strat}_ppa']
        
        # CODO bars will be 1.0
        p_df['codo_lat_plot'] = 1.0
        p_df['codo_ppa_plot'] = 1.0
        
        # Replace inf with 0 for failed designs (where phx_lat was 0)
        p_df.replace([np.inf, -np.inf], 0, inplace=True)
    else:
        p_df['codo_lat_plot'] = p_df['codo_lat']
        p_df['codo_ppa_plot'] = p_df['codo_ppa']
        for u_strat in u_strategies:
            p_df[f'{u_strat}_lat_plot'] = p_df[f'{u_strat}_lat']
            p_df[f'{u_strat}_ppa_plot'] = p_df[f'{u_strat}_ppa']

    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(14, len(p_df)*0.8), 8), sharex=True)
    
    x = np.arange(len(p_df))
    
    num_phx = len(u_strategies)
    total_bars = num_phx if normalized else (num_phx + 1)
    
    # Dynamic width and gap
    width = 0.8 / (total_bars + 1)
    gap = 0.02
    
    # Calculate all offsets centered around x
    all_offsets = [(i - (total_bars - 1)/2) * (width + gap) for i in range(total_bars)]
    
    if normalized:
        codo_offset = None
        phx_offsets = all_offsets
    else:
        codo_offset = all_offsets[0]
        phx_offsets = all_offsets[1:]
        
    codo_color = '#7f8c8d' # Gray for CODO
    
    # Hatch mapping
    HATCH_MAP = {
        'div_min': None,
        'ndiv_min': '///',
        'ndiv_max': 'xxx'
    }
        
    # Plot Latency
    if not normalized:
        ax1.bar(x + codo_offset, p_df['codo_lat_plot'], width, label='CODO', color=codo_color, edgecolor='none', linewidth=0)
    else:
        # Add baseline line for normalized plot
        ax1.axhline(y=1.0, color='red', linestyle='-', linewidth=1.5, alpha=0.8, zorder=0)

    for i, u_strat in enumerate(u_strategies):
        val_col = f'{u_strat}_lat_norm' if normalized else f'{u_strat}_lat_plot'
        vals = p_df[val_col]
        offset = phx_offsets[i]
        hatch = HATCH_MAP.get(u_strat)
        
        colors = [get_strat_color(s) for s in p_df[f'{u_strat}_strat']]
        
        for j, val in enumerate(vals):
            if not pd.isna(val) and val > 0:
                ax1.bar(x[j] + offset, val, width, color=colors[j], edgecolor='white', linewidth=0.3, hatch=hatch)
            elif val == 0 or (pd.isna(val) and p_df.loc[j, f'{u_strat}_strat'] is None and not pd.isna(p_df.loc[j, 'test'])):
                if p_df.loc[j, f'{u_strat}_lat'] == 0:
                    ax1.text(x[j] + offset, 0.1, 'X', ha='center', va='bottom', color='red', fontweight='bold', fontsize=FS_ANNOT)

    # Plot PPA
    if not normalized:
        ax2.bar(x + codo_offset, p_df['codo_ppa_plot'], width, label='CODO', color=codo_color, edgecolor='none', linewidth=0)
    else:
        # Add baseline line for normalized plot
        ax2.axhline(y=1.0, color='red', linestyle='-', linewidth=1.5, alpha=0.8, zorder=0)

    for i, u_strat in enumerate(u_strategies):
        val_col = f'{u_strat}_ppa_norm' if normalized else f'{u_strat}_ppa_plot'
        vals = p_df[val_col]
        offset = phx_offsets[i]
        hatch = HATCH_MAP.get(u_strat)
        
        colors = [get_strat_color(s) for s in p_df[f'{u_strat}_strat']]
        
        for j, val in enumerate(vals):
            if not pd.isna(val) and val > 0:
                ax2.bar(x[j] + offset, val, width, color=colors[j], edgecolor='white', linewidth=0.3, hatch=hatch)
            elif val == 0:
                ax2.text(x[j] + offset, 0.1, 'X', ha='center', va='bottom', color='red', fontweight='bold', fontsize=FS_ANNOT)

    # Formatting
    apply_base_theme(ax1)
    apply_base_theme(ax2)
    
    if log_scale:
        ax1.set_yscale('log')
        ax2.set_yscale('log')
        
    ax1.set_ylabel('Latency Speedup' if normalized else 'RTL Cycles', fontsize=FS_AXIS)
    ax2.set_ylabel('PPA Improvement' if normalized else 'ADP (Cycles * Area)', fontsize=FS_AXIS)
    
    ax1.set_title('Latency Comparison: CODO vs ST-Flow', fontsize=FS_TITLE, fontweight='bold')
    ax2.set_title('PPA Comparison: CODO vs ST-Flow', fontsize=FS_TITLE, fontweight='bold')
    
    # X-axis labels
    ax2.set_xticks(x)
    # Use labels from theme.py
    labels = [get_label(row['test'], 'test') for _, row in p_df.iterrows()]
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=FS_TICK)
    
    # Add vertical lines between benchmarks
    current_bench = None
    for i, row in p_df.iterrows():
        if current_bench is not None and row['benchmark'] != current_bench:
            ax1.axvline(x=i-0.5, color='black', linestyle='--', alpha=0.3)
            ax2.axvline(x=i-0.5, color='black', linestyle='--', alpha=0.3)
        current_bench = row['benchmark']

    # Custom Legend - Row-based Layout (4 Columns)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    # Flattened Legend
    legend_elements = []
    
    # 1. CODO Baseline
    baseline = Line2D([0], [0], color='red', lw=1.5, label='CODO Baseline') if normalized else Patch(facecolor=codo_color, edgecolor='none', label='CODO')
    legend_elements.append(baseline)
    
    # 2. Hardware Mapping Strategies
    legend_elements.append(Patch(facecolor=STRAT_COLORS['no_reuse'], edgecolor='none', label=get_label('no_reuse', 'strat')))
    legend_elements.append(Patch(facecolor=STRAT_COLORS['balanced'], edgecolor='none', label=get_label('balanced', 'strat')))
    legend_elements.append(Patch(facecolor=STRAT_COLORS['max_reuse'], edgecolor='none', label=get_label('max_reuse', 'strat')))
    
    # # 3. Tiling Strategies
    # for u_strat in u_strategies:
    #     h = HATCH_MAP.get(u_strat)
    #     legend_elements.append(Patch(facecolor='lightgray', edgecolor='white', hatch=h, 
    #                                  label=get_label(u_strat, 'config'), linewidth=0.5))

    ax1.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.80),
               fontsize=FS_LEGEND-2, frameon=True, ncol=len(legend_elements), 
               handletextpad=0.4, columnspacing=0.8, facecolor='white', edgecolor='none')
    
    # Annotation for U-Strategies
    # We can add a text label for div_min and ndiv_min bars if needed, or just leave it to the colors/legend
    # But wait, the two ST-Flow bars are div_min and ndiv_min. 
    # Maybe I should add a small indicator or just use the label.
    # The user said "two 'u_strategy' ... shown next to each other".
    # I'll add a label to the first one of each group.
    
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    suffix = "_normalized" if normalized else "_absolute"
    out_path = os.path.join(output_dir, f"codo_full_comparison{suffix}.pdf")
    plt.savefig(out_path, bbox_inches='tight')
    plt.savefig(out_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    print(f"Plot saved to {out_path}")

    # --- Summary Statistics ---
    def geo_mean(series):
        # Filter for positive valid numbers
        vals = series[series > 0].dropna().values
        if len(vals) == 0:
            return np.nan
        return np.exp(np.log(vals).mean())

    print("\n" + "="*65)
    print(f"{'GEO MEAN SUMMARY vs CODO':^65}")
    print("="*65)
    
    for u_strat in u_strategies:
        # Calculate ratios for all rows
        p_df[f'{u_strat}_lat_ratio'] = p_df['codo_lat'] / p_df[f'{u_strat}_lat']
        p_df[f'{u_strat}_ppa_ratio'] = p_df['codo_ppa'] / p_df[f'{u_strat}_ppa']
        
        # Replace inf with nan (where phx was 0)
        p_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        print(f"\n[Strategy: {get_label(u_strat, 'config')}]")
        print(f"{'Benchmark':<20} | {'Lat Speedup':>15} | {'PPA Imp.':>15}")
        print("-" * 65)
        
        for bench in target_benchmarks:
            bench_df = p_df[p_df['benchmark'] == bench]
            lat_gm = geo_mean(bench_df[f'{u_strat}_lat_ratio'])
            ppa_gm = geo_mean(bench_df[f'{u_strat}_ppa_ratio'])
            
            lat_str = f"{lat_gm:.2f}x" if not np.isnan(lat_gm) else "N/A"
            ppa_str = f"{ppa_gm:.2f}x" if not np.isnan(ppa_gm) else "N/A"
            print(f"{get_label(bench, 'bench'):<20} | {lat_str:>15} | {ppa_str:>15}")
            
        # Total
        total_lat_gm = geo_mean(p_df[f'{u_strat}_lat_ratio'])
        total_ppa_gm = geo_mean(p_df[f'{u_strat}_ppa_ratio'])
        lat_str = f"{total_lat_gm:.2f}x" if not np.isnan(total_lat_gm) else "N/A"
        ppa_str = f"{total_ppa_gm:.2f}x" if not np.isnan(total_ppa_gm) else "N/A"
        print("-" * 65)
        print(f"{'AGGREGATE TOTAL':<20} | {lat_str:>15} | {ppa_str:>15}")
    print("="*65 + "\n")

if __name__ == "__main__":
    # This script is intended to be called from evplot, but we can add a standalone mode for testing
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--codo", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--u_strategies", type=str, default="div_min,ndiv_min")
    parser.add_argument("-n", "--normalized", action="store_true")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv)
    df_codo = pd.read_csv(args.codo)
    u_strats = args.u_strategies.split(',')
    plot(df, df_codo, args.out, u_strategies=u_strats, normalized=args.normalized, log_scale=args.log)
