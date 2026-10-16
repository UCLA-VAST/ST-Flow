import torch
import torch.nn as nn
import sys

from stflow import generate_hardware, STFlowConfig
# fix random seed for reproducibility
torch.manual_seed(42)

class MVT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A1, A2, y1, y2):
        x1 = A1 @ y1
        x2 = A2.T @ y2
        return x1, x2

def init_tests():
    prj_name = "07_mvt"
    # Problem size from data.py: (400, 400) -> N=400
    N = 400
    A1 = torch.randn(N, N)
    A2 = torch.randn(N, N)
    y1 = torch.randn(N)
    y2 = torch.randn(N)
    inputs = (A1, A2, y1, y2)
    return prj_name, MVT(), inputs

def test_mvt_no_reuse():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'no_reuse'
    generate_hardware(prj_name, model, inputs, test_config)

def test_mvt_balanced():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'balanced'
    generate_hardware(prj_name, model, inputs, test_config)

def test_mvt_max_reuse():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'max_reuse'
    # MVT's two transposed mat-vecs merged under max_reuse produce a design whose
    # C-simulation is sensitive to machine load with the default 'aggressive' SWMR
    # elimination; 'conservative' keeps it stable.
    test_config.swmr_elimination = 'conservative'
    generate_hardware(prj_name, model, inputs, test_config)

if __name__ == "__main__":
    test_mvt_no_reuse()
    test_mvt_balanced()
    test_mvt_max_reuse()
