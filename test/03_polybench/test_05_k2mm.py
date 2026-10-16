import torch
import torch.nn as nn
import sys

from stflow import generate_hardware, STFlowConfig
# fix random seed for reproducibility
torch.manual_seed(42)

class K2MM(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.1
        self.beta = 0.5

    def forward(self, a, b, c, d_in):
        tmp = a @ b
        tmp2 = tmp @ c
        d = tmp2 * self.beta + d_in * self.alpha
        return d

def init_tests():
    prj_name = "05_k2mm"
    # Problem size from data.py: (180, 210, 190, 220)
    # randTensor(180, 210), # A
    # randTensor(210, 190), # B
    # randTensor(190, 220), # C
    # randTensor(180, 220)  # D
    # So NI=180, NK=210, NJ=190, NL=220
    NI, NK, NJ, NL = 180, 210, 190, 220
    A = torch.randn(NI, NK)
    B = torch.randn(NK, NJ)
    C = torch.randn(NJ, NL)
    D = torch.randn(NI, NL)
    inputs = (A, B, C, D)
    return prj_name, K2MM(), inputs

def test_k2mm_no_reuse():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'no_reuse'
    generate_hardware(prj_name, model, inputs, test_config)

def test_k2mm_balanced():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'balanced'
    generate_hardware(prj_name, model, inputs, test_config)

def test_k2mm_max_reuse():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'max_reuse'
    generate_hardware(prj_name, model, inputs, test_config)

if __name__ == "__main__":
    test_k2mm_no_reuse()
    test_k2mm_balanced()
    test_k2mm_max_reuse()
