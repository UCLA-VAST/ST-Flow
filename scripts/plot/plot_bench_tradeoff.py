import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from .theme import FS_TITLE, FS_AXIS, FS_TICK, FS_ANNOT, FS_LEGEND, STRAT_COLORS, apply_base_theme, get_strat_color, get_label

def draw_efficiency_panel(ax, df_label, tests, cycles_col, res_col, label):
    apply_base_theme(ax)
    ax.set_facecolor('#fdfdfd')
    
    speedups = []
    savings = []
    valid_tests = []
    
    for test in tests:
        test_df = df_label[df_label['test'] == test]
        
        # Get data for the three strategies
        bal = test_df[test_df['hw_strategy'] == 'balanced']
        no_r = test_df[test_df['hw_strategy'] == 'no_reuse']
        max_r = test_df[test_df['hw_strategy'] == 'max_reuse']
        
        if bal.empty or no_r.empty or max_r.empty:
            continue
            
        # 1. Speedup over Max Reuse (Baseline = slowest design)
        c_bal = bal[cycles_col].values[0]
        c_max = max_r[cycles_col].values[0]
        if c_bal > 0 and c_max > 0:
            speedups.append(c_max / c_bal)
        else:
            speedups.append(0)
            
        # 2. Resource Savings over No Reuse (Baseline = most expensive design)
        r_bal = bal[res_col].values[0]
        r_no = no_r[res_col].values[0]
        if r_no > 0:
            sav = (r_no - r_bal) / r_no * 100
            savings.append(max(0, sav)) # Ensure positive savings
        else:
            savings.append(0)
            
        valid_tests.append(test)

    if not valid_tests:
        ax.text(0.5, 0.5, "Insufficient data for comparison", ha='center')
        return

    x = np.arange(len(valid_tests))
    width = 0.35
    
    # Left Axis: Speedup
    color_speed = STRAT_COLORS.get('max_reuse', '#4D96FF') # Blue for speedup (since it's vs max_reuse)
    bars1 = ax.bar(x - width/2, speedups, width, label='Speedup (vs Max-Reuse)', color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_ylabel("Speedup (Multiplier)", fontsize=FS_AXIS, color='#2980b9', fontweight='bold')
    ax.tick_params(axis='y', labelcolor='#2980b9')
    
    # Right Axis: Savings
    ax2 = ax.twinx()
    color_sav = STRAT_COLORS.get('balanced', '#6BCB77') # Green for savings (efficiency)
    bars2 = ax2.bar(x + width/2, savings, width, label='DSP Savings (vs No-Reuse)', color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel("Resource Savings (%)", fontsize=FS_AXIS, color='#27ae60', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#27ae60')
    ax2.set_ylim(0, 110) # Savings is 0-100%
    
    ax.set_title(f"Configuration: {label}", fontsize=FS_TITLE-2, fontweight='bold', loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_tests, fontsize=FS_TICK, rotation=25, ha='right')
    
    # Add labels on top of bars
    ax.bar_label(bars1, fmt='%.1fx', padding=3, fontsize=FS_ANNOT-1, color='#2980b9')
    ax2.bar_label(bars2, fmt='%.0f%%', padding=3, fontsize=FS_ANNOT-1, color='#27ae60')

def plot(benchmark, df, output_dir, cycles_col='p_cycles', res_col='v_dsps', ext="png", output_name=None):
    # Filter for specified benchmark and successful DSE
    df = df[df['benchmark'] == benchmark]
    df = df[df['p_status'] == 'Pass']
    df = df[df['p_cycles'] > 0]
    
    if df.empty:
        print(f"No successful projects found for benchmark {benchmark}")
        return

    # Strategy mapping
    df['div_label'] = df['u_strategy'].apply(lambda x: get_label(x, 'config'))
    
    target_labels = ['Div-Min', 'Non-Div-Min', 'Non-Div-Max']
    df = df[df['div_label'].isin(target_labels)]
    df['div_label'] = pd.Categorical(df['div_label'], categories=target_labels, ordered=True)
    
    tests = sorted(df['test'].unique())
    
    # Create 3-panel figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))
    
    for i, label in enumerate(target_labels):
        df_label = df[df['div_label'] == label]
        draw_efficiency_panel(axes[i], df_label, tests, cycles_col, res_col, label)

    res_name = res_col.replace('_', ' ').replace('v_', '').upper()
    fig.suptitle(f"Resource/Performance Tradeoff: {get_label(benchmark, 'bench')}", fontsize=FS_TITLE+2, fontweight='bold', y=0.98)
    
    # Combined Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', edgecolor='black', alpha=0.8, label='Speedup (Cycles_Max / Cycles_Bal)'),
        Patch(facecolor='#2ecc71', edgecolor='black', alpha=0.8, label=f'Savings % ((Res_No - Res_Bal) / Res_No)')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2, 
               frameon=True, facecolor='white', fontsize=FS_LEGEND, shadow=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93], h_pad=3.0)
    
    os.makedirs(output_dir, exist_ok=True)
    if output_name:
        filename = f"{output_name}.{ext}"
    else:
        res_short = res_col.replace('v_', '')
        filename = f"tradeoff_{benchmark}_{res_short}.{ext}"
        
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Efficiency plot saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", type=str)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--cycles", type=str, default="rtl_cycles")
    parser.add_argument("--res", type=str, default="v_dsps")
    parser.add_argument("--ext", type=str, default="png")
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv)
    df.columns = [c.strip() for c in df.columns]
    plot(args.benchmark, df, args.out, cycles_col=args.cycles, res_col=args.res, ext=args.ext)
