"""Module 2 of WH-ESPC: multi-sequence CT feature extraction.

A stack of transformer layers in which the feed-forward sub-layer of each
layer is replaced by the sparse Mixture-of-Experts (see ``moe.py``). The MoE
uses dual-stage gated routing with historical-memory load balancing and
softsign regularization, allowing the module to adaptively extract critical
information from multi-sequence CT features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe import SparseMoE


class MultiHeadAttention(nn.Module):
    """Standard scaled dot-product multi-head self-attention."""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v)
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, embed_dim)
        return self.fc_out(attention_output)


class MoETransformerLayer(nn.Module):
    """Transformer layer: multi-head self-attention + sparse MoE."""

    def __init__(self, embed_dim, num_heads, num_experts, top_k,
                 capacity_factor=1.0, alpha=0.1, window_size=3):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.moe = SparseMoE(embed_dim, num_experts, top_k, capacity_factor,
                             alpha, window_size)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.norm1(x + self.self_attn(x))
        x = self.norm2(x + self.moe(x))
        return x


class MoETransformer(nn.Module):
    """Stack of MoE transformer layers operating on slice embeddings."""

    def __init__(self, embed_dim, num_heads, num_layers, num_experts, top_k,
                 capacity_factor=1.0, alpha=0.1, window_size=3):
        super().__init__()
        self.layers = nn.ModuleList([
            MoETransformerLayer(embed_dim, num_heads, num_experts, top_k,
                                capacity_factor, alpha, window_size)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        """Args:
            x: slice embeddings of shape ``[batch, num_slices, embed_dim]``.

        Returns:
            Refined features of the same shape.
        """
        for layer in self.layers:
            x = layer(x)
        return x
