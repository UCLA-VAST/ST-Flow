# Hardware Strategy Colors
STRAT_COLORS = {
    'no_reuse':  '#4D96FF', # Red-ish
    'balanced':  '#6BCB77', # Green-ish
    'max_reuse': '#FF6B6B', # Blue-ish
}

# Default color if strategy is unknown
DEFAULT_COLOR = '#999999'

# Font Size Constants (Tune these to scale plot text)
FS_TITLE  = 15
FS_AXIS   = 12
FS_TICK   = 12
FS_ANNOT  = 12
FS_LEGEND = 12
FS_LABEL  = 12

# FPGA Resource Limits (U250 or similar)
FPGA_LIMITS = {
    'bram': 4032,
    'dsp':  9024,
    'uram': 960,
}

# Formula: num_brams + 16 * num_urams
FPGA_LIMITS['mem_equiv'] = FPGA_LIMITS['bram'] + 16 * FPGA_LIMITS['uram']

def get_strat_color(strat):
    """Returns the color for a given hardware strategy."""
    return STRAT_COLORS.get(strat, DEFAULT_COLOR)

# Label Mapping Dictionaries (for Plots)
STRAT_MAP = {
    'no_reuse':  'Spatial',
    'balanced':  'Hybrid',
    'max_reuse': 'Temporal',
}

SOLVER_MAP = {
    'minlp':   'MINLP',
    'sa_sa':   'SA-S',
    'sa_comb': 'SA-C',
    'ga_ga':   'GA-S',
    'ga_comb': 'GA-C',
}

# LaTeX Macro Mapping Dictionaries (for Tables)
STRAT_MAP_TEX = {
    'no_reuse':  '\\snoreuse',
    'balanced':  '\\sbalanced',
    'max_reuse': '\\smaxreuse',
}

SOLVER_MAP_TEX = {
    'minlp':   '\\solminlp',
    'sa_sa':   '\\solsasa',
    'sa_comb': '\\solsacomb',
    'ga_ga':   '\\solgaga',
    'ga_comb': '\\solgacomb',
}

BENCH_MAP = {
    '01_dnn':       'DNN',
    '02_synthetic': 'Synthetic',
    '03_polybench': 'PolyBench',
    '04_machsuite': 'MachSuite',
    '05_mot':       'MOT',
}

CONFIG_MAP = {
    'div':      'Div',
    'div_min':  'Div-Min',
    'ndiv':     'Non-Div',
    'ndiv_min': 'Non-Div-Min',
    'ndiv_max': 'Non-Div-Max',
}

CONFIG_MAP_TEX = {
    'div':      '\\divstrat',
    'div_min':  '\\divmin',
    'ndiv':     '\\ndivstrat',
    'ndiv_min': '\\ndivmin',
    'ndiv_max': '\\ndivmax',
}

TEST_MAP = {
    '01_mlp_mixer': 'MLP-Mixer',
    '02_rwkv_channel_mix': 'RWKV-CM',
    '03_tabm': 'TabM',
    '04_swiglu': 'SwiGLU',
    '06_din': 'DIN',
    '07_mnist_mlp': 'MNIST',

    '01_uniform_mm_seq': '7MM-Seq Reg.',
    '02_random_mm_seq': '7MM-Seq Rand.',
    '03_uniform_mm_tree': '7MM-Tree Reg.',
    '04_random_mm_tree': '7MM-Tree Rand.',
    '05_uniform_mlp_seq': 'MLP-Seq Reg.',
    '06_random_mlp_seq': 'MLP-Seq Rand.',
    '07_uniform_mlp_diamond': 'MLP-Diamond Reg.',
    '08_random_mlp_diamond': 'MLP-Diamond Rand.',

    '01_atax': 'Atax',
    '02_bicg': 'Bicg',
    '03_gemm': 'Gemm',
    '04_gesummv': 'Gesummv',
    '05_k2mm': '2MM',
    '06_k3mm': '3MM',
    '07_mvt': 'MVT',
}

COLUMN_MAP = {
    'rtl_cycles':    'RTL Cycles',
    'p_cycles':      'ST-Flow Cycles',
    'b_cycles':      'Bitstream Cycles',
    'v_min_cycles':  'Vitis Min Cycles',
    'v_max_cycles':  'Vitis Max Cycles',
    'p_dsps':        'DSPs',
    'v_dsps':        'Vitis DSPs',
    'b_dsps':        'Bitstream DSPs',
    'p_mem':         'BRAMs',
    'v_mem':         'Vitis BRAMs',
    'b_mem':         'Bitstream BRAMs',
    'v_freq':        'Vitis Freq (MHz)',
    'b_freq':        'Bitstream Freq (MHz)',
    'p_cost':        'DSE Cost',
    'adp':           'Area-Delay Product',
    'area':          'Area Score',
}

def get_label(val, map_type, tex=False):
    """
    Centralized function to map internal strings to display labels.
    map_type: 'strat', 'bench', 'config', 'test', 'column'
    tex: If True, returns LaTeX macros for tables.
    """
    if not isinstance(val, str): return str(val)
    
    if map_type == 'strat':
        return (STRAT_MAP_TEX if tex else STRAT_MAP).get(val, val)
    elif map_type == 'bench':
        return BENCH_MAP.get(val, val)
    elif map_type == 'config':
        return (CONFIG_MAP_TEX if tex else CONFIG_MAP).get(val, val)
    elif map_type == 'test':
        return TEST_MAP.get(val, val)
    elif map_type == 'column':
        return COLUMN_MAP.get(val, val)
    elif map_type == 'solver':
        return (SOLVER_MAP_TEX if tex else SOLVER_MAP).get(val, val)
    
    return val

def apply_base_theme(ax):
    """Applies common aesthetic settings to a matplotlib axis."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=FS_TICK)
