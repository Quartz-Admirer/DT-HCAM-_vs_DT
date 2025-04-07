# models/hcam_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class HCAMAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, chunk_size=8, num_memory_slots=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.num_memory_slots = num_memory_slots

        self.local_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln_local = nn.LayerNorm(embed_dim)

        self.memory = nn.Parameter(torch.zeros(num_memory_slots, embed_dim))
        nn.init.xavier_uniform_(self.memory)

        self.cross_attn_chunk = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln_cross_chunk = nn.LayerNorm(embed_dim)

        self.cross_attn_mem = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln_cross_mem = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.ln_mlp = nn.LayerNorm(embed_dim)

    def forward(self, x):
        bsz, seq_len, embed_dim = x.shape 

        if seq_len % self.chunk_size != 0:
             raise ValueError(f"HCAMAttention: Input sequence length {seq_len} is not divisible by chunk_size {self.chunk_size}. Check dataset padding.")

        n_chunks = seq_len // self.chunk_size

        x_chunked = x.view(bsz, n_chunks, self.chunk_size, self.embed_dim)
        x_chunked_2d = x_chunked.reshape(bsz * n_chunks, self.chunk_size, self.embed_dim)
        
        attn_local_out, _ = self.local_attn(x_chunked_2d, x_chunked_2d, x_chunked_2d)
        x_chunked_2d = x_chunked_2d + attn_local_out
        x_chunked_2d = self.ln_local(x_chunked_2d)

        all_chunks = x_chunked_2d
        memory = self.memory.unsqueeze(0).expand(bsz * n_chunks, -1, -1)

        cross_chunk_out, _ = self.cross_attn_chunk(all_chunks, memory, memory)
        cross_chunk_out = all_chunks + cross_chunk_out
        cross_chunk_out = self.ln_cross_chunk(cross_chunk_out)

        cross_mem_out, _ = self.cross_attn_mem(memory, cross_chunk_out, cross_chunk_out)
        cross_mem_out = memory + cross_mem_out
        cross_mem_out = self.ln_cross_mem(cross_mem_out)

        mlp_input = cross_chunk_out
        mlp_out = self.mlp(mlp_input)
        x_chunked_2d = mlp_input + mlp_out
        x_chunked_2d = self.ln_mlp(x_chunked_2d)

        x_out = x_chunked_2d.reshape(bsz, seq_len, self.embed_dim)

        return x_out

