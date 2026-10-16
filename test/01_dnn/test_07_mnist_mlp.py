import torch
import torch.nn as nn
import os

from stflow import generate_hardware, STFlowConfig

# fix random seed for reproducibility
torch.manual_seed(42)

class MNISTMLPBlock(nn.Module):
    def __init__(self, d_in=784, d_h1=512, d_h2=256, d_out=10):
        super().__init__()
        # Traditional MNIST MLP structure
        self.fc1 = nn.Linear(d_in, d_h1, bias=False)
        self.fc2 = nn.Linear(d_h1, d_h2, bias=False)
        self.fc3 = nn.Linear(d_h2, d_out, bias=False)
        self.act = nn.ReLU()

    def forward(self, x):
        # x shape: (batch_size, 784)
        
        # Layer 1
        x = self.fc1(x)
        x = self.act(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.act(x)
        
        # Layer 3: Output logits
        out = self.fc3(x)
        
        return out

def init_tests():
    prj_name = "07_mnist_mlp"
    batch_size = 32
    d_in = 784
    d_h1 = 512
    d_h2 = 256
    d_out = 10
    
    A = torch.randn(batch_size, d_in)
    inputs = (A,)
    model = MNISTMLPBlock(
        d_in=d_in, 
        d_h1=d_h1, 
        d_h2=d_h2, 
        d_out=d_out
    )
    return prj_name, model, inputs

def test_mnist_mlp_no_reuse():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'no_reuse'
    generate_hardware(prj_name, model, inputs, test_config)

def test_mnist_mlp_balanced():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'balanced'
    generate_hardware(prj_name, model, inputs, test_config)

def test_mnist_mlp_max_reuse():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'max_reuse'
    generate_hardware(prj_name, model, inputs, test_config)

if __name__ == "__main__":
    test_mnist_mlp_no_reuse()
    test_mnist_mlp_balanced()
    test_mnist_mlp_max_reuse()
