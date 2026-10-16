import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from stflow import generate_hardware, STFlowConfig

# fix random seed for reproducibility
torch.manual_seed(42)

class SwiGLUBlock(nn.Module):
    def __init__(self, d_model=64, d_ffn=256):
        super().__init__()
        # In SwiGLU, we have two projections for the gating part
        # and one projection for the output
        self.w1 = nn.Linear(d_model, d_ffn, bias=False)
        self.v = nn.Linear(d_model, d_ffn, bias=False)
        self.w2 = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x):
        # x shape: (batch_size, d_model)
        
        # 1. Compute gated linear unit components
        # Swish(xW1) * xV
        # SiLU is the standard implementation of Swish used in LLMs
        gate = self.w1(x)
        gate = F.silu(gate)
        
        val = self.v(x)
        
        # 2. Element-wise multiplication
        h = gate * val
        
        # 3. Final projection
        out = self.w2(h)
        
        return out

def init_tests():
    prj_name = "04_swiglu"
    batch_size = 64
    d_model = 128
    d_ffn = 256
    
    A = torch.randn(batch_size, d_model)
    inputs = (A,)
    model = SwiGLUBlock(
        d_model=d_model, 
        d_ffn=d_ffn
    )
    return prj_name, model, inputs

def test_swiglu_no_reuse():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'no_reuse'
    generate_hardware(prj_name, model, inputs, test_config)

def test_swiglu_balanced():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'balanced'
    generate_hardware(prj_name, model, inputs, test_config)

def test_swiglu_max_reuse():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'max_reuse'
    generate_hardware(prj_name, model, inputs, test_config)

if __name__ == "__main__":
    test_swiglu_no_reuse()
    test_swiglu_balanced()
    test_swiglu_max_reuse()
