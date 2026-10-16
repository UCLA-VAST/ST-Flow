import torch
import torch.nn as nn
import os

from stflow import generate_hardware, STFlowConfig

# fix random seed for reproducibility
torch.manual_seed(42)

class DINMLPBlock(nn.Module):
    def __init__(self, d_in=128, d_hidden_wide=256, d_hidden_narrow=64):
        super().__init__()
        # Bottleneck structure: Wide -> Narrow -> Wide
        self.fc1 = nn.Linear(d_in, d_hidden_wide, bias=False)
        self.fc2 = nn.Linear(d_hidden_wide, d_hidden_narrow, bias=False)
        self.fc3 = nn.Linear(d_hidden_narrow, d_hidden_wide, bias=False)
        self.fc4 = nn.Linear(d_hidden_wide, d_in, bias=False)
        
        self.act = nn.ReLU() # DIN often uses Dice or PReLU, but ReLU is fine for DSE testing

    def forward(self, x):
        # x shape: (batch_size, d_in) -> (32, 128)
        
        # Layer 1: Expand
        x = self.fc1(x)
        x = self.act(x)
        
        # Layer 2: Bottleneck (Narrow)
        x = self.fc2(x)
        x = self.act(x)
        
        # Layer 3: Expand back
        x = self.fc3(x)
        x = self.act(x)
        
        # Layer 4: Final projection
        out = self.fc4(x)
        
        return out

def init_tests():
    prj_name = "06_din"
    batch_size = 16
    d_in = 64
    d_hidden_wide = 1024
    d_hidden_narrow = 64
    
    A = torch.randn(batch_size, d_in)
    inputs = (A,)
    model = DINMLPBlock(
        d_in=d_in, 
        d_hidden_wide=d_hidden_wide, 
        d_hidden_narrow=d_hidden_narrow
    )
    return prj_name, model, inputs

def test_din_no_reuse():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'no_reuse'
    generate_hardware(prj_name, model, inputs, test_config)

def test_din_balanced():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'balanced'
    generate_hardware(prj_name, model, inputs, test_config)

def test_din_max_reuse():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'max_reuse'
    generate_hardware(prj_name, model, inputs, test_config)

if __name__ == "__main__":
    test_din_no_reuse()
    test_din_balanced()
    test_din_max_reuse()
