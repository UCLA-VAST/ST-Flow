import torch
import torch.nn as nn
import sys

from stflow import generate_hardware, STFlowConfig
# fix random seed for reproducibility
torch.manual_seed(42)

class ATAX(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, x):
        tmp = A @ x
        y = A.T @ tmp
        return y

def init_tests():
    prj_name = "01_atax"
    # Problem size from data.py: (390, 410)
    M, N = 390, 410
    A = torch.randn(M, N)
    x = torch.randn(N)
    inputs = (A, x)
    return prj_name, ATAX(), inputs

def test_atax_no_reuse():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'no_reuse'
    test_config.swmr_elimination = 'conservative'
    generate_hardware(prj_name, model, inputs, test_config)

def test_atax_balanced():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'balanced'
    test_config.swmr_elimination = 'conservative'
    generate_hardware(prj_name, model, inputs, test_config)

def test_atax_max_reuse():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'max_reuse'
    test_config.swmr_elimination = 'conservative'
    generate_hardware(prj_name, model, inputs, test_config)

if __name__ == "__main__":
    test_atax_no_reuse()
    test_atax_balanced()
    test_atax_max_reuse()
