from dataclasses import dataclass, field, asdict
from typing import List, Optional, Union, Any
import os
import yaml

@dataclass
class SAConfig:
    initial_temp: float = 1000.0
    cooling_rate: float = 0.999
    iterations: int = 5000
    initial_solution: str = 'default'

    def dict(self):
        return asdict(self)

    def to_cpp(self, cpp_obj):
        cpp_obj.initial_temp = self.initial_temp
        cpp_obj.cooling_rate = self.cooling_rate
        cpp_obj.iterations = self.iterations
        cpp_obj.initial_solution = self.initial_solution
        return cpp_obj

@dataclass
class GAConfig:
    pop_size: int = 50
    generations: int = 500
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    init_method: str = 'mixture'

    def dict(self):
        return asdict(self)

    def to_cpp(self, cpp_obj):
        cpp_obj.pop_size = self.pop_size
        cpp_obj.generations = self.generations
        cpp_obj.mutation_rate = self.mutation_rate
        cpp_obj.crossover_rate = self.crossover_rate
        cpp_obj.init_method = self.init_method
        return cpp_obj

@dataclass
class RandomConfig:
    iterations: int = 10

    def dict(self):
        return asdict(self)

    def to_cpp(self, cpp_obj):
        cpp_obj.iterations = self.iterations
        return cpp_obj

@dataclass
class DSEConfig:
    latency_memory_ratio: int = 80
    num_n_runs: int = 1
    permutation_dse: str = 'sa'
    unrolling_dse: str = 'sa'
    combined_dse: str = 'sa'
    permutation_timeout: int = 60
    unrolling_timeout: int = 60
    combined_timeout: int = 60
    permutation_ampl_solver: str = 'gecode'
    unrolling_ampl_solver: str = 'gecode'
    combined_ampl_solver: str = 'gecode'
    balanced_mapping_similarity_threshold: float = 10.0
    balanced_mapping_ub_threshold: float = 5.0
    objective_scaling_factor: int = 1000
    min_time_loop_size: int = 8
    enforce_divisor_only_tiling: bool = True
    shared_tiling_strategy: str = "min"
    DSP_limit: int = 500
    Memory_limit: int = 1
    
    fixed_unrolling_solution: Optional[dict] = None
    fixed_permutation_solution: Optional[dict] = None
    
    permutation_random_config: RandomConfig = field(default_factory=RandomConfig)
    unrolling_random_config: RandomConfig = field(default_factory=RandomConfig)
    combined_random_config: RandomConfig = field(default_factory=RandomConfig)
    
    permutation_sa_config: SAConfig = field(default_factory=SAConfig)
    unrolling_sa_config: SAConfig = field(default_factory=SAConfig)
    combined_sa_config: SAConfig = field(default_factory=SAConfig)
    
    permutation_ga_config: GAConfig = field(default_factory=GAConfig)
    unrolling_ga_config: GAConfig = field(default_factory=GAConfig)
    combined_ga_config: GAConfig = field(default_factory=GAConfig)

    def dict(self):
        return asdict(self)

    def to_cpp(self):
        from ._mlir.dialects import streamhls as streamhls_d
        cpp_opts = streamhls_d.DSEOptions()
        
        cpp_opts.permutation_dse = self.permutation_dse
        cpp_opts.unrolling_dse = self.unrolling_dse
        cpp_opts.combined_dse = self.combined_dse
        cpp_opts.permutation_timeout = self.permutation_timeout
        cpp_opts.unrolling_timeout = self.unrolling_timeout
        cpp_opts.combined_timeout = self.combined_timeout
        cpp_opts.permutation_ampl_solver = self.permutation_ampl_solver
        cpp_opts.unrolling_ampl_solver = self.unrolling_ampl_solver
        cpp_opts.combined_ampl_solver = self.combined_ampl_solver
        cpp_opts.dsp_limit = self.DSP_limit
        cpp_opts.memory_limit = self.Memory_limit
        cpp_opts.min_time_loop_size = self.min_time_loop_size
        cpp_opts.enforce_divisor_only_tiling = self.enforce_divisor_only_tiling
        cpp_opts.shared_tiling_strategy = self.shared_tiling_strategy
        cpp_opts.latency_memory_ratio = self.latency_memory_ratio
        cpp_opts.balanced_mapping_similarity_threshold = self.balanced_mapping_similarity_threshold
        cpp_opts.balanced_mapping_ub_threshold = self.balanced_mapping_ub_threshold
        cpp_opts.objective_scaling_factor = self.objective_scaling_factor

        if self.fixed_permutation_solution:
            cpp_opts.fixed_permutation_solution = self.fixed_permutation_solution
        if self.fixed_unrolling_solution:
            cpp_opts.fixed_unrolling_solution = self.fixed_unrolling_solution

        # SA Configs
        self.permutation_sa_config.to_cpp(cpp_opts.permutation_sa_config)
        self.unrolling_sa_config.to_cpp(cpp_opts.unrolling_sa_config)
        self.combined_sa_config.to_cpp(cpp_opts.combined_sa_config)

        # GA Configs
        self.permutation_ga_config.to_cpp(cpp_opts.permutation_ga_config)
        self.unrolling_ga_config.to_cpp(cpp_opts.unrolling_ga_config)
        self.combined_ga_config.to_cpp(cpp_opts.combined_ga_config)

        # Random Configs
        self.permutation_random_config.to_cpp(cpp_opts.permutation_random_config)
        self.unrolling_random_config.to_cpp(cpp_opts.unrolling_random_config)
        self.combined_random_config.to_cpp(cpp_opts.combined_random_config)

        return cpp_opts

    @classmethod
    def from_dict(cls, data: dict) -> 'DSEConfig':
        if not data:
            return cls()
        
        # Handle random configs
        p_random = RandomConfig(**data.get('permutation_random_config', {}))
        u_random = RandomConfig(**data.get('unrolling_random_config', {}))
        c_random = RandomConfig(**data.get('combined_random_config', {}))
        
        # Handle SA configs
        p_sa = SAConfig(**data.get('permutation_sa_config', {}))
        u_sa = SAConfig(**data.get('unrolling_sa_config', {}))
        c_sa = SAConfig(**data.get('combined_sa_config', {}))
        
        # Handle GA configs
        p_ga = GAConfig(**data.get('permutation_ga_config', {}))
        u_ga = GAConfig(**data.get('unrolling_ga_config', {}))
        c_ga = GAConfig(**data.get('combined_ga_config', {}))
        
        # Base fields
        base_fields = {
            k: v for k, v in data.items() 
            if k not in [
                'permutation_random_config', 'unrolling_random_config', 'combined_random_config',
                'permutation_sa_config', 'unrolling_sa_config', 'combined_sa_config',
                'permutation_ga_config', 'unrolling_ga_config', 'combined_ga_config',
                'latency_weight' # Backward compatibility if present
            ]
        }
        
        # Backward compatibility for latency_weight
        if 'latency_weight' in data and 'latency_memory_ratio' not in data:
            base_fields['latency_memory_ratio'] = data['latency_weight']

        # Filter fields not in DSEConfig
        valid_fields = {k: v for k, v in base_fields.items() if k in cls.__dataclass_fields__}

        return cls(
            **valid_fields,
            permutation_random_config=p_random,
            unrolling_random_config=u_random,
            combined_random_config=c_random,
            permutation_sa_config=p_sa,
            unrolling_sa_config=u_sa,
            combined_sa_config=c_sa,
            permutation_ga_config=p_ga,
            unrolling_ga_config=u_ga,
            combined_ga_config=c_ga
        )

@dataclass
class STFlowConfig:
    hw_mapping_strategy: str = 'no_reuse'
    target_tool: str = 'vitis'
    atol: float = 1e-3
    run_vitis_hls_synth: bool = False
    run_vitis_hls_cosim: bool = False
    compile_only: bool = False
    csim: bool = True
    enable_dma: bool = True
    enable_unrolling: bool = True
    enable_permutation: bool = True
    enable_combined_dse: bool = True
    dse_options: DSEConfig = field(default_factory=DSEConfig)
    full_data_reuse: bool = True
    swmr_elimination: str = 'aggressive'
    timeout_s: int = 360000
    debug_types: List[str] = field(default_factory=lambda: [''])
    
    # Other potential fields mentioned in pipelines.py
    run_bitstream: bool = False

    def dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'STFlowConfig':
        if not data:
            return cls()
        
        dse_dict = data.get('dse_options', {})
        dse_options = DSEConfig.from_dict(dse_dict)
        
        base_fields = {k: v for k, v in data.items() if k != 'dse_options'}
        
        # Filter fields not in STFlowConfig
        valid_fields = {k: v for k, v in base_fields.items() if k in cls.__dataclass_fields__}
        
        return cls(**valid_fields, dse_options=dse_options)

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> 'STFlowConfig':
        """
        Loads config from a file and applies environment variable overrides.
        Prioritizes: config_path > test_config.yaml > vanilla_config.yaml
        """
        if not config_path:
            if os.path.exists("test_config.yaml"):
                config_path = "test_config.yaml"
            elif os.path.exists("vanilla_config.yaml"):
                config_path = "vanilla_config.yaml"
        
        data = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f) or {}
        
        config = cls.from_dict(data)
        
        # Apply environment variable overrides from STFLOW_MODE
        mode = os.environ.get("STFLOW_MODE")
        if mode == "csim":
            config.csim = True
        elif mode == "csim_only":
            config.csim = True
            config.compile_only = True
        elif mode == "syn":
            config.run_vitis_hls_synth = True
            config.compile_only = True
            config.csim = False
        elif mode == "cosim":
            config.run_vitis_hls_synth = True
            config.run_vitis_hls_cosim = True
            config.compile_only = True
            config.csim = False
        elif mode == "bitstream":
            config.run_bitstream = True
            config.compile_only = True
            
        return config
