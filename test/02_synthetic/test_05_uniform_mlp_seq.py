import torch
import torch.nn as nn
import os

from stflow import generate_hardware, STFlowConfig

# fix random seed for reproducibility
torch.manual_seed(42)

class MLPSeq(nn.Module):
    def __init__(self, p_large=256, p_small=64):
        super().__init__()
        # Traditional MNIST MLP structure
        self.fc_1_1 = nn.Linear(p_large, p_large, bias=False)
        self.fc_1_2 = nn.Linear(p_large, p_large, bias=False)
        self.fc_1_3 = nn.Linear(p_large, p_large, bias=False)
        self.fc_1_4 = nn.Linear(p_large, p_large, bias=False)

        self.fc_inter_1 = nn.Linear(p_large, p_small, bias=False)

        self.fc_2_1 = nn.Linear(p_small, p_small, bias=False)
        self.fc_2_2 = nn.Linear(p_small, p_small, bias=False)
        self.fc_2_3 = nn.Linear(p_small, p_small, bias=False)
        self.fc_2_4 = nn.Linear(p_small, p_small, bias=False)

    def forward(self, x):
        # x shape: (batch_size, 784)
        
        # Layer 1
        x = self.fc_1_1(x)
        x = self.fc_1_2(x)
        x = self.fc_1_3(x)
        
        x = self.fc_inter_1(x)
        # Layer 2
        x = self.fc_2_1(x)
        x = self.fc_2_2(x)
        x = self.fc_2_3(x)
        
        return x

def init_tests():
    prj_name = "05_uniform_mlp_seq"
    batch_size = 1
    p_large = 256
    p_small = 256
    
    A = torch.randn(batch_size, p_large)
    inputs = (A,)
    model = MLPSeq(p_large, p_small)
    return prj_name, model, inputs

def test_uniform_mlp_seq_no_reuse():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'no_reuse'
    generate_hardware(prj_name, model, inputs, test_config)

def test_uniform_mlp_seq_balanced():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'balanced'
    generate_hardware(prj_name, model, inputs, test_config)

def test_uniform_mlp_seq_max_reuse():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'max_reuse'
    generate_hardware(prj_name, model, inputs, test_config)

if __name__ == "__main__":
    test_uniform_mlp_seq_no_reuse()
    test_uniform_mlp_seq_balanced()
    test_uniform_mlp_seq_max_reuse()
