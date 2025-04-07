# models/dt_hcam.py

import torch.nn as nn
from .decision_transformer import DecisionTransformer
from .hcam_attention import HCAMAttention

class DTHCAM(DecisionTransformer):

    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_size=128,
        n_layers=3,
        n_heads=4,
        max_length=50,
        chunk_size=8,
        num_memory_slots=16,
    ):
        super().__init__(state_dim, act_dim, hidden_size, n_layers, n_heads, max_length)

        self.blocks = nn.ModuleList([
            _HCAMBlock(hidden_size, n_heads, chunk_size, num_memory_slots) 
            for _ in range(n_layers)
        ])

class _HCAMBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, chunk_size, num_memory_slots):
        super().__init__()
        self.hcam = HCAMAttention(embed_dim, num_heads, chunk_size, num_memory_slots)

        self.ln_1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.ln_2 = nn.LayerNorm(embed_dim)

    def forward(self, x, attn_mask=None):
        attn_out = self.hcam(x)
        x = self.ln_1(x + attn_out)

        mlp_out = self.mlp(x)
        x = self.ln_2(x + mlp_out)

        return x

