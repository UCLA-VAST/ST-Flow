import torch
import torch.nn as nn
import sys

from stflow import generate_hardware, STFlowConfig
# fix random seed for reproducibility
torch.manual_seed(42)

class BICG(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A1, A2, r, p):
        s = r @ A1
        q = A2 @ p
        return s, q

def init_tests():
    prj_name = "02_bicg"
    # Problem size from data.py: (410, 390)
    M, N = 410, 390
    A1 = torch.randn(M, N)
    A2 = torch.randn(M, N)
    r = torch.randn(M)
    p = torch.randn(N)
    inputs = (A1, A2, r, p)
    return prj_name, BICG(), inputs

def test_bicg_no_reuse():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'no_reuse'
    generate_hardware(prj_name, model, inputs, test_config)

def test_bicg_balanced():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'balanced'
    generate_hardware(prj_name, model, inputs, test_config)

def test_bicg_max_reuse():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'max_reuse'
    generate_hardware(prj_name, model, inputs, test_config)

if __name__ == "__main__":
    test_bicg_no_reuse()
    test_bicg_balanced()
    test_bicg_max_reuse()
