import torch
import torch.nn as nn
import sys

from stflow import generate_hardware, STFlowConfig

# fix random seed for reproducibility
torch.manual_seed(42)

class MMTree(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B, C, D, E, F, G, H):
        t1 = A @ B
        t2 = C @ D
        t3 = E @ F
        t4 = G @ H
        t5 = t1 @ t2
        t6 = t3 @ t4
        t7 = t5 @ t6
        return t7

def init_tests():
    prj_name = "03_uniform_mm_tree"
    I, J, K, L, M, N, O, P, Q = 128, 128, 128, 128, 128, 128, 128, 128, 128
    A = torch.randn(I, J)
    B = torch.randn(J, K)
    C = torch.randn(K, L)
    D = torch.randn(L, M)
    E = torch.randn(M, N)
    F = torch.randn(N, O)
    G = torch.randn(O, P)
    H = torch.randn(P, Q)
    inputs = (A, B, C, D, E, F, G, H)
    return prj_name, MMTree(), inputs

def test_uniform_mm_tree_no_reuse():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'no_reuse'
    test_config.atol = 1000
    generate_hardware(prj_name, model, inputs, test_config)

def test_uniform_mm_tree_balanced():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'balanced'
    test_config.atol = 1000
    generate_hardware(prj_name, model, inputs, test_config)

def test_uniform_mm_tree_max_reuse():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'max_reuse'
    test_config.atol = 1000
    generate_hardware(prj_name, model, inputs, test_config)

if __name__ == "__main__":
    test_uniform_mm_tree_no_reuse()
    test_uniform_mm_tree_balanced()
    test_uniform_mm_tree_max_reuse()