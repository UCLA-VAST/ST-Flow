import torch
import torch.nn as nn
import sys

from stflow import generate_hardware, STFlowConfig
# fix random seed for reproducibility
torch.manual_seed(42)

class K3MM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b, c, d):
        tmp1 = a @ b
        tmp2 = c @ d
        e = tmp1 @ tmp2
        return e

def init_tests():
    prj_name = "06_k3mm"
    # Problem size from data.py: (180, 200, 190, 220, 210)
    # randTensor(180, 200), # a
    # randTensor(200, 190), # b
    # randTensor(190, 220), # c
    # randTensor(220, 210)  # d
    # So NI=180, NJ=200, NK=190, NL=220, NM=210
    NI, NJ, NK, NL, NM = 180, 200, 190, 220, 210
    A = torch.randn(NI, NJ)
    B = torch.randn(NJ, NK)
    C = torch.randn(NK, NL)
    D = torch.randn(NL, NM)
    inputs = (A, B, C, D)
    return prj_name, K3MM(), inputs

def test_k3mm_no_reuse():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'no_reuse'
    generate_hardware(prj_name, model, inputs, test_config)

def test_k3mm_balanced():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'balanced'
    generate_hardware(prj_name, model, inputs, test_config)

def test_k3mm_max_reuse():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'max_reuse'
    generate_hardware(prj_name, model, inputs, test_config)

if __name__ == "__main__":
    test_k3mm_no_reuse()
    test_k3mm_balanced()
    test_k3mm_max_reuse()
