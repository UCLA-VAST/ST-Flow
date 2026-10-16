import torch
import torch.nn as nn
import os

from stflow import generate_hardware, STFlowConfig

# fix random seed for reproducibility
torch.manual_seed(42)

class TabMBlock(nn.Module):
    def __init__(self, d_in=32, d_model=128, num_heads=4, d_out=16):
        super().__init__()
        # Shared first layer
        self.fc_shared = nn.Linear(d_in, d_model, bias=False)
        self.act = nn.ReLU()
        
        # Multiple heads (flattened representation)
        # In a real TabM, these might be more efficient, 
        # but here we use distinct heads for hardware testing.
        self.heads = nn.ModuleList([
            nn.Linear(d_model, d_out, bias=False) for _ in range(num_heads)
        ])
        
        self.num_heads = num_heads

    def forward(self, x):
        # x shape: (batch_size, d_in) -> (32, 32)
        
        # 1. Shared projection
        h = self.fc_shared(x)
        h = self.act(h)
        
        # 2. Parallel heads
        # To keep it "flattened" and avoid complex list comprehensions 
        # that might confuse some compilers, we sum them explicitly or use a loop
        # that the stflow compiler can preferably handle.
        # But for "minimal flattened", let's just do them.
        
        out_sum = self.heads[0](h)
        for i in range(1, self.num_heads):
            out_sum = out_sum + self.heads[i](h)
            
        # 3. Ensemble Average
        out = out_sum / self.num_heads
        
        return out

def init_tests():
    prj_name = "03_tabm"
    batch_size = 64
    d_in = 128
    d_model = 256
    num_heads = 8
    d_out = 128
    
    A = torch.randn(batch_size, d_in)
    inputs = (A,)
    model = TabMBlock(
        d_in=d_in, 
        d_model=d_model, 
        num_heads=num_heads, 
        d_out=d_out
    )
    return prj_name, model, inputs

def test_tabm_no_reuse():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'no_reuse'
    generate_hardware(prj_name, model, inputs, test_config)

def test_tabm_balanced():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'balanced'
    generate_hardware(prj_name, model, inputs, test_config)

def test_tabm_max_reuse():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'max_reuse'
    generate_hardware(prj_name, model, inputs, test_config)

if __name__ == "__main__":
    test_tabm_no_reuse()
    test_tabm_balanced()
    test_tabm_max_reuse()
