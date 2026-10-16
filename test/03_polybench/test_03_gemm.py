import torch
import torch.nn as nn
import sys

from stflow import generate_hardware, STFlowConfig
# fix random seed for reproducibility
torch.manual_seed(42)

class GEMM(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.1
        self.beta = 0.5

    def forward(self, a, b, c_in):
        c = a @ b
        c = c * self.beta + c_in * self.alpha
        return c

def init_tests():
    prj_name = "03_gemm"
    # Problem size from data.py: (200, 240, 220) -> NI, NK, NJ
    NI, NK, NJ = 200, 240, 220
    A = torch.randn(NI, NK)
    B = torch.randn(NK, NJ)
    C = torch.randn(NI, NJ)
    inputs = (A, B, C)
    return prj_name, GEMM(), inputs

def test_gemm_no_reuse():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'no_reuse'
    generate_hardware(prj_name, model, inputs, test_config)

def test_gemm_balanced():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'balanced'
    generate_hardware(prj_name, model, inputs, test_config)

def test_gemm_max_reuse():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'max_reuse'
    generate_hardware(prj_name, model, inputs, test_config)

if __name__ == "__main__":
    test_gemm_no_reuse()
    test_gemm_balanced()
    test_gemm_max_reuse()
