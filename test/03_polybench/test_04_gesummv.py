import torch
import torch.nn as nn
import sys

from stflow import generate_hardware, STFlowConfig
# fix random seed for reproducibility
torch.manual_seed(42)

class GESUMMV(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B, x):
        tmp = A @ x
        y = B @ x
        y = 1.5 * tmp + 1.2 * y
        return y

def init_tests():
    prj_name = "04_gesummv"
    # Problem size from data.py: (250, 250) -> N=250
    N = 250
    A = torch.randn(N, N)
    B = torch.randn(N, N)
    x = torch.randn(N)
    inputs = (A, B, x)
    return prj_name, GESUMMV(), inputs

def test_gesummv_no_reuse():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'no_reuse'
    generate_hardware(prj_name, model, inputs, test_config)

def test_gesummv_balanced():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'balanced'
    generate_hardware(prj_name, model, inputs, test_config)

def test_gesummv_max_reuse():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'max_reuse'
    generate_hardware(prj_name, model, inputs, test_config)

if __name__ == "__main__":
    test_gesummv_no_reuse()
    test_gesummv_balanced()
    test_gesummv_max_reuse()
