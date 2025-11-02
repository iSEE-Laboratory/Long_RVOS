import torch.nn as nn
import copy
import torch

class TemporalBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoderLayer(d_model, n_heads, dropout=dropout)

    def forward(self, hs, tgt):
        hs  = self.norm(hs + self.dropout(self.self_attn(hs, hs, hs)[0]))
        tgt = self.decoder(tgt, hs)
        return hs, tgt

class TemporalDecoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layer, dropout):
        super().__init__()
        self.n_layer = n_layer
        self.layer = nn.ModuleList([copy.deepcopy(TemporalBlock(d_model, n_heads, dropout)) for _ in range(n_layer)])
        self.norm = nn.ModuleList([copy.deepcopy(nn.LayerNorm(d_model)) for _ in range(n_layer)])

    def forward(self, hs, tgt):
        for i in range(self.n_layer):
            hs, tgt = self.layer[i](hs, tgt)
            hs = self.norm[i](hs + tgt)
        return hs