import torch
import torch.nn as nn
import os

from stflow import generate_hardware, STFlowConfig

# fix random seed for reproducibility
torch.manual_seed(42)

class RWKVChannelMixBlock(nn.Module):
    def __init__(self, d_model=64, intermediate_size=256):
        super().__init__()
        # In RWKV, Channel Mix is essentially a gated MLP
        self.key_fc = nn.Linear(d_model, intermediate_size, bias=False)
        self.receptance_fc = nn.Linear(d_model, d_model, bias=False)
        self.value_fc = nn.Linear(intermediate_size, d_model, bias=False)
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (num_tokens, d_model)
        
        # 1. Compute Receptance gate
        # Receptance acts as a gate for the output
        r = self.receptance_fc(x)
        r = self.sigmoid(r)
        
        # 2. Compute Key (intermediate representation)
        k = self.key_fc(x)
        k = torch.square(self.relu(k)) # Quadratic ReLU is common in RWKV
        
        # 3. Compute Value and gate it
        v = self.value_fc(k)
        
        # Output is the element-wise product of receptance and value
        out = r * v
        
        return out

def init_tests():
    prj_name = "02_rwkv_channel_mix"
    num_tokens = 64
    d_model = 128
    intermediate_size = 256
    
    A = torch.randn(num_tokens, d_model)
    inputs = (A,)
    model = RWKVChannelMixBlock(
        d_model=d_model, 
        intermediate_size=intermediate_size
    )
    return prj_name, model, inputs

def test_rwkv_no_reuse():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'no_reuse'
    generate_hardware(prj_name, model, inputs, test_config)

def test_rwkv_balanced():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'balanced'
    generate_hardware(prj_name, model, inputs, test_config)

def test_rwkv_max_reuse():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'max_reuse'
    generate_hardware(prj_name, model, inputs, test_config)

if __name__ == "__main__":
    test_rwkv_no_reuse()
    test_rwkv_balanced()
    test_rwkv_max_reuse()
