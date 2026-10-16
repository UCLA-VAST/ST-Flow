import torch
import torch.nn as nn
import sys

from stflow import generate_hardware, STFlowConfig

# fix random seed for reproducibility
torch.manual_seed(42)

class MLPMixerBlock(nn.Module):
    def __init__(self, num_tokens=16, d_model=32, token_expand=64, channel_expand=128):
        super().__init__()
        # Token-mixing MLP (Operates across tokens/rows)
        self.token_fc1 = nn.Linear(num_tokens, token_expand, bias=False)
        self.token_act = nn.ReLU()
        self.token_fc2 = nn.Linear(token_expand, num_tokens, bias=False)
        
        # Channel-mixing MLP (Operates across channels/columns)
        self.channel_fc1 = nn.Linear(d_model, channel_expand, bias=False)
        self.channel_act = nn.ReLU()
        self.channel_fc2 = nn.Linear(channel_expand, d_model, bias=False)

    def forward(self, x):
        # Input x shape: (num_tokens, d_model) -> (16, 32)
        
        # --- TOKEN MIXING ---
        # Transpose to (d_model, num_tokens) for token-wise FC
        t = x.transpose(0, 1) 
        t = self.token_fc1(t)
        t = self.token_act(t)
        t = self.token_fc2(t)
        # Transpose back to (num_tokens, d_model)
        t = t.transpose(0, 1)
        
        # First Residual Connection
        x = x + t
        
        # --- CHANNEL MIXING ---
        # FC operates on the last dimension (d_model) naturally
        c = self.channel_fc1(x)
        c = self.channel_act(c)
        c = self.channel_fc2(c)
        
        # Second Residual Connection
        out = x + c
        
        return out

def init_tests():
    prj_name = "01_mlp_mixer"
    num_tokens = 64
    d_model = 128
    token_expand = 256
    channel_expand = 512
    A = torch.randn(num_tokens, d_model)
    inputs = (A,)
    model = MLPMixerBlock(
        num_tokens=num_tokens, 
        d_model=d_model, 
        token_expand=token_expand, 
        channel_expand=channel_expand
    )
    return prj_name, model, inputs

def test_mlp_mixer_no_reuse():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'no_reuse'
    generate_hardware(prj_name, model, inputs, test_config)

def test_mlp_mixer_balanced():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'balanced'
    generate_hardware(prj_name, model, inputs, test_config)

def test_mlp_mixer_max_reuse():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'max_reuse'
    generate_hardware(prj_name, model, inputs, test_config)

if __name__ == "__main__":
    test_mlp_mixer_no_reuse()
    test_mlp_mixer_balanced()
    test_mlp_mixer_max_reuse()
