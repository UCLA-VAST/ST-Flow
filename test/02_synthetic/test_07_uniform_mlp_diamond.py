import torch
import torch.nn as nn
import os

from stflow import generate_hardware, STFlowConfig

# fix random seed for reproducibility
torch.manual_seed(42)
class MLPDiamond(nn.Module):
    def __init__(self, p_large=256, p_small=64):
        super().__init__()
        # left branch
        self.fc_l1_1 = nn.Linear(p_large, p_large, bias=False)
        self.fc_l1_2 = nn.Linear(p_large, p_large, bias=False)
        self.fc_l1_3 = nn.Linear(p_large, p_large, bias=False)

        self.fc_l_inter_0 = nn.Linear(p_large, p_small, bias=False)

        self.fc_l2_1 = nn.Linear(p_small, p_small, bias=False)
        self.fc_l2_2 = nn.Linear(p_small, p_small, bias=False)
        self.fc_l2_3 = nn.Linear(p_small, p_small, bias=False)

        self.fc_l_inter_1 = nn.Linear(p_small, p_large, bias=False)

        # right branch
        self.fc_r_inter_0 = nn.Linear(p_large, p_small, bias=False)

        self.fc_r1_1 = nn.Linear(p_small, p_small, bias=False)
        self.fc_r1_2 = nn.Linear(p_small, p_small, bias=False)
        self.fc_r1_3 = nn.Linear(p_small, p_small, bias=False)

        self.fc_r_inter_1 = nn.Linear(p_small, p_large, bias=False)

        self.fc_r2_1 = nn.Linear(p_large, p_large, bias=False)
        self.fc_r2_2 = nn.Linear(p_large, p_large, bias=False)
        self.fc_r2_3 = nn.Linear(p_large, p_large, bias=False)


    def forward(self, x):
        
        
        # left branch
        x_l = self.fc_l1_1(x)
        x_l = self.fc_l1_2(x_l)
        x_l = self.fc_l1_3(x_l)
        
        # Layer 2
        x_l = self.fc_l2_1(x_l)
        x_l = self.fc_l2_2(x_l)
        x_l = self.fc_l2_3(x_l)
        
        # right branch
        x_r = self.fc_r1_1(x)
        x_r = self.fc_r1_2(x_r)
        x_r = self.fc_r1_3(x_r)
        
        x_r = self.fc_r2_1(x_r)
        x_r = self.fc_r2_2(x_r)
        x_r = self.fc_r2_3(x_r)
        
        return x_l + x_r

def init_tests():
    prj_name = "07_uniform_mlp_diamond"
    batch_size = 1
    p_large = 256
    p_small = 256
    
    A = torch.randn(batch_size, p_large)
    inputs = (A,)
    model = MLPDiamond(p_large, p_small)
    return prj_name, model, inputs

def test_uniform_mlp_diamond_no_reuse():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'no_reuse'
    generate_hardware(prj_name, model, inputs, test_config)

def test_uniform_mlp_diamond_balanced():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'balanced'
    generate_hardware(prj_name, model, inputs, test_config)

def test_uniform_mlp_diamond_max_reuse():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'max_reuse'
    generate_hardware(prj_name, model, inputs, test_config)

if __name__ == "__main__":
    pass
