# -*- coding: utf-8 -*-
"""WH-ESPCA Module 2: MoE-Transformer with noisy top-k routing and
dynamic bias adjustment (historical-memory load balancing).

Identical in behaviour to the single-encoder WH-ESPC `moe.py` /
`transformer.py`; kept self-contained so this update drops in cleanly.
"""

import math
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        return self.weight * (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + self.eps) + self.bias


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.head_dim), dim=-1)
        return self.out_proj((attn @ v).transpose(1, 2).reshape(B, S, D))


class Expert(nn.Module):
    def __init__(self, dim=512, hidden_multiplier=4, drop_p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * hidden_multiplier), nn.GELU(),
                                 nn.Linear(dim * hidden_multiplier, dim), nn.Dropout(drop_p))

    def forward(self, x):
        return self.net(x)


class DynamicBiasAdjuster:
    """Historical-memory load balancing for the router.

    Biases are NOT updated by gradient descent. A sliding window of recent
    per-expert loads is kept; each training forward pass shifts the bias by
    ``alpha * gain * softsign(avg_load - load)``, suppressing overloaded
    experts and encouraging idle ones. Bounded by softsign.
    """

    def __init__(self, num_experts, window_size=3, alpha=0.1, gain=8.0):
        self.alpha = alpha
        self.gain = gain
        self.history = deque(maxlen=2 * window_size + 1)
        self.biases = None

    def update(self, load):
        load = load.detach()
        if self.biases is None:
            self.biases = torch.zeros_like(load)
        self.history.append(load)
        avg_load = torch.stack(list(self.history)).mean(dim=0)
        self.biases = self.biases + self.alpha * self.gain * F.softsign(avg_load - load)

    def apply(self, logits):
        return logits if self.biases is None else logits + self.biases


class SparseMoE(nn.Module):
    """Noisy top-k router + 1 shared expert + E routed experts (capacity-constrained)
    + dynamic bias adjustment (updated in training mode only)."""

    def __init__(self, dim=512, num_experts=16, top_k=4, capacity_factor=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.clean_proj = nn.Linear(dim, num_experts)
        self.noise_proj = nn.Linear(dim, num_experts)
        self.shared_expert = Expert(dim)
        self.experts = nn.ModuleList([Expert(dim) for _ in range(num_experts)])
        self.bias_adjuster = DynamicBiasAdjuster(num_experts)

    def forward(self, x):            # x: [B, S, D]
        B, S, D = x.shape
        T = B * S
        flat = x.reshape(T, D)

        # stage 1: noisy top-k routing -> load estimate
        clean_logits = self.clean_proj(flat)
        noise = torch.randn_like(clean_logits) * F.softplus(self.noise_proj(flat))
        noisy_logits = clean_logits + noise
        _, top_idx = noisy_logits.topk(self.top_k, dim=-1)
        if self.training:
            load = F.one_hot(top_idx, self.num_experts).float().mean(dim=(0, 1))
            self.bias_adjuster.update(load)

        # stage 2: bias-corrected re-routing
        adj_logits = self.bias_adjuster.apply(clean_logits)
        probs = F.softmax(adj_logits, dim=-1)
        gating, sel = probs.topk(self.top_k, dim=-1)
        gating = gating / gating.sum(dim=-1, keepdim=True)

        out = self.shared_expert(flat)
        capacity = max(1, math.ceil(T * self.top_k / self.num_experts * self.capacity_factor))
        for e in range(self.num_experts):
            tok, slot = torch.nonzero(sel == e, as_tuple=True)
            if tok.numel() == 0:
                continue
            if tok.numel() > capacity:
                tok, slot = tok[:capacity], slot[:capacity]
            expert_out = self.experts[e](flat[tok]) * gating[tok, slot].unsqueeze(-1)
            out = out.index_add(0, tok, expert_out)
        return out.view(B, S, D)


class MoETransformerLayer(nn.Module):
    def __init__(self, embed_dim=512, num_heads=4, num_experts=16, top_k=4):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = CustomLayerNorm(embed_dim)
        self.moe = SparseMoE(embed_dim, num_experts, top_k)
        self.norm2 = CustomLayerNorm(embed_dim)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        return self.norm2(x + self.moe(x))


class MoETransformer(nn.Module):
    def __init__(self, embed_dim=512, num_heads=4, num_layers=2, num_experts=16, top_k=4):
        super().__init__()
        self.layers = nn.ModuleList(
            [MoETransformerLayer(embed_dim, num_heads, num_experts, top_k) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
