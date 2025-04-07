# models/decision_transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.ln_2 = nn.LayerNorm(embed_dim)

    def forward(self, x, attn_mask=None):
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.ln_1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.ln_2(x + mlp_out)
        return x

class DecisionTransformer(nn.Module):
    def __init__(
        self, state_dim, act_dim, hidden_size=128,
        n_layers=3, n_heads=4, max_length=150
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.max_length = max_length


        self.state_embedding = nn.Linear(state_dim, hidden_size)
        self.action_embedding = nn.Embedding(act_dim, hidden_size)
        self.return_embedding = nn.Linear(1, hidden_size)

        self.pos_embedding = nn.Parameter(torch.zeros(1, max_length * 3, hidden_size))

        self.blocks = nn.ModuleList([
            AttentionBlock(hidden_size, n_heads) for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(hidden_size)
        self.action_head = nn.Linear(hidden_size, act_dim)

    def forward(self, states, actions, returns_to_go, timesteps=None):
        batch_size, seq_len, _ = states.shape
        if returns_to_go.dim() == 2:
             if returns_to_go.shape[0] == batch_size and returns_to_go.shape[1] == seq_len:
                 returns_to_go = returns_to_go.unsqueeze(-1)
             else:
                  raise ValueError(f"Unexpected returns_to_go shape: {returns_to_go.shape}")


        state_tokens = self.state_embedding(states)
        return_tokens = self.return_embedding(returns_to_go)

        actions_long = actions.long()
        action_mask = (actions_long != -1)
        safe_actions = torch.where(action_mask, actions_long, torch.zeros_like(actions_long))
        action_tokens = self.action_embedding(safe_actions)
        action_tokens = action_tokens * action_mask.unsqueeze(-1).float()

        attn_mask = torch.triu(
            torch.ones((seq_len * 3, seq_len * 3), device=states.device, dtype=torch.bool),
            diagonal=1
        )

        if not (return_tokens.shape[1] == state_tokens.shape[1] == action_tokens.shape[1] == seq_len):
             raise ValueError("Sequence length mismatch between R, S, A tokens")

        r_flat = return_tokens.reshape(-1, self.hidden_size)
        s_flat = state_tokens.reshape(-1, self.hidden_size)
        a_flat = action_tokens.reshape(-1, self.hidden_size)

        interleaved_flat = torch.stack([r_flat, s_flat, a_flat], dim=1)
        token_sequence = interleaved_flat.reshape(batch_size, seq_len * 3, self.hidden_size)

        current_seq_len_x3 = seq_len * 3
        if self.pos_embedding.shape[1] < current_seq_len_x3:
             raise ValueError(f"Positional embedding table size ({self.pos_embedding.shape[1]}) is too small for required sequence length {current_seq_len_x3}")
        pos_emb = self.pos_embedding[:, :current_seq_len_x3, :]
        x = token_sequence + pos_emb
  
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
 
        x = self.ln_f(x)
        state_outputs = x[:, 1::3, :]
        action_preds = self.action_head(state_outputs)

        return action_preds