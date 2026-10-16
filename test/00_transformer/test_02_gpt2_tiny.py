# test_05_transformer.py
import numpy as np
import torch
import torch.nn as nn

from stflow import generate_hardware, STFlowConfig


# -------------------------
#  Model definition
#  - Separate Q/K/V projections (keeps matmuls separate)
# -------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,T,Dh]
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,T,Dh]
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,T,Dh]
        # return q, k, v
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.d_head ** -0.5)  # [B,H,T,T]
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
        # works with manual fix up to here
        probs = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, v)
        # return ctx # works up to here
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, D)
        final_output = self.out(ctx)
        return final_output


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=True)
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))



class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, eps: float = 1e-5):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=eps)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model, eps=eps)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# -------------------------
#  Tests
# -------------------------

def init_tests():
    B_dim, T_dim, D_dim = 1, 64, 128
    n_heads = 4
    d_ff = 256
    prj_name = "02_gpt2_tiny"
    model = TransformerBlock(d_model=D_dim, n_heads=n_heads, d_ff=d_ff).eval()
    x = np.random.rand(B_dim, T_dim, D_dim).astype(np.float32)
    inputs = [torch.from_numpy(x)]
    return prj_name, model, inputs

def test_gpt2_no_reuse():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'no_reuse'
    generate_hardware(prj_name, model, inputs, test_config)

def test_gpt2_balanced():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'balanced'
    generate_hardware(prj_name, model, inputs, test_config)

def test_gpt2_max_reuse():
    prj_name, model, inputs = init_tests()
    test_config = STFlowConfig.load()
    test_config.hw_mapping_strategy = 'max_reuse'
    generate_hardware(prj_name, model, inputs, test_config)