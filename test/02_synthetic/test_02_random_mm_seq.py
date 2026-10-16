import torch
import torch.nn as nn
import sys

from stflow import generate_hardware, STFlowConfig

# fix random seed for reproducibility
torch.manual_seed(42)


class MMSeq(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B, C, D, E, F, G, H):
      t1 = A @ B
      print(f'Gemm1 dims: {A.shape[0]} x {A.shape[1]} x {B.shape[1]} = {A.shape[0] * A.shape[1] * B.shape[1]}')
      t2 = t1 @ C
      print(f'Gemm2 dims: {t1.shape[0]} x {t1.shape[1]} x {C.shape[1]} = {t1.shape[0] * t1.shape[1] * C.shape[1]}')
      t3 = t2 @ D
      print(f'Gemm3 dims: {t2.shape[0]} x {t2.shape[1]} x {D.shape[1]} = {t2.shape[0] * t2.shape[1] * D.shape[1]}')
      t4 = t3 @ E
      print(f'Gemm4 dims: {t3.shape[0]} x {t3.shape[1]} x {E.shape[1]} = {t3.shape[0] * t3.shape[1] * E.shape[1]}')
      t5 = t4 @ F
      print(f'Gemm5 dims: {t4.shape[0]} x {t4.shape[1]} x {F.shape[1]} = {t4.shape[0] * t4.shape[1] * F.shape[1]}')
      t6 = t5 @ G
      print(f'Gemm6 dims: {t5.shape[0]} x {t5.shape[1]} x {G.shape[1]} = {t5.shape[0] * t5.shape[1] * G.shape[1]}')
      t7 = t6 @ H
      print(f'Gemm7 dims: {t6.shape[0]} x {t6.shape[1]} x {H.shape[1]} = {t6.shape[0] * t6.shape[1] * H.shape[1]}')
      return t7

def init_tests():
    prj_name = "02_random_mm_seq"
    factor = 1
    I, J, K, L, M, N, O, P, Q = 100, 200, 190, 180, 170, 160, 150, 140, 130
    A = torch.randn(I*factor, J*factor)
    B = torch.randn(J*factor, K*factor)
    C = torch.randn(K*factor, L*factor)
    D = torch.randn(L*factor, M*factor)
    E = torch.randn(M*factor, N*factor)
    F = torch.randn(N*factor, O*factor)
    G = torch.randn(O*factor, P*factor)
    H = torch.randn(P*factor, Q*factor)
    inputs = (A, B, C, D, E, F, G, H)
    return prj_name, MMSeq(), inputs

def test_random_mm_seq_no_reuse():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'no_reuse'
    test_config.atol = 1000
    generate_hardware(prj_name, model, inputs, test_config)

def test_random_mm_seq_balanced():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'balanced'
    test_config.atol = 1000
    generate_hardware(prj_name, model, inputs, test_config)

def test_random_mm_seq_max_reuse():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'max_reuse'
    test_config.atol = 1000
    generate_hardware(prj_name, model, inputs, test_config)

if __name__ == "__main__":
    _, model, inputs = init_tests()
    model.eval()
    out = model(*inputs)
    # test_random_mm_seq_no_reuse()
    # test_random_mm_seq_balanced()
    # test_random_mm_seq_max_reuse()