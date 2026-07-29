"""Sparse Mixture-of-Experts with dual-stage gated routing and load balancing.

This module implements the core of the multi-sequence CT feature extraction
module of WH-ESPC:

1. A noisy top-k router produces the first-stage routing probabilities and a
   per-expert load estimate.
2. A dynamic bias adjuster traces the routing load over a historical window
   and shifts the router logits with a softsign-regularized update, which
   balances the load across experts.
3. The bias-adjusted logits are re-normalized and re-sparsified (second-stage
   gating), and each token is dispatched to its selected experts, plus one
   shared expert that processes every token.
"""

from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """A single expert: a position-wise feed-forward network."""

    def __init__(self, n_embd, dropout=0.1, hidden_multiplier=4, activation=nn.GELU()):
        super().__init__()
        self.hidden_dim = hidden_multiplier * n_embd
        self.net = nn.Sequential(
            nn.Linear(n_embd, self.hidden_dim),
            activation,
            nn.Linear(self.hidden_dim, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class NoisyTopkRouter(nn.Module):
    """First-stage gating: noisy top-k routing over experts."""

    def __init__(self, n_embed, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        # Linear layer for the raw expert scores.
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        # Linear layer that controls the scale of the exploration noise.
        self.noise_linear = nn.Linear(n_embed, num_experts)

    def forward(self, x):
        # Raw expert logits plus softplus-scaled Gaussian exploration noise.
        logits = self.topkroute_linear(x)
        noise_logits = self.noise_linear(x)
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise
        # Keep only the top-k experts and re-normalize into a sparse
        # probability distribution.
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices


class DynamicBiasAdjuster:
    """Historical-memory load balancer for the MoE router.

    The per-expert routing load is traced over a sliding window of the last
    ``2 * window_size + 1`` observations. The bias of each expert is updated
    towards the historical average load with a softsign-regularized step, so
    that overloaded experts are penalized and underloaded experts are
    encouraged without unbounded bias drift.

    Note: the biases are updated explicitly (not by gradient descent) and are
    therefore excluded from the model ``state_dict``.
    """

    def __init__(self, num_experts, alpha=0.1, window_size=3, scale=8.0):
        self.num_experts = num_experts
        self.alpha = alpha            # step size of the bias update
        self.window_size = window_size
        self.scale = scale            # gain of the softsign regularization
        self.biases = torch.nn.Parameter(torch.zeros(num_experts), requires_grad=False)
        self.load_history = deque(maxlen=2 * window_size + 1)

    def update(self, load):
        """Trace the current load and update the routing biases."""
        self.load_history.append(load.detach().clone())
        load_tensor = torch.stack(list(self.load_history))
        avg_load = torch.mean(load_tensor, dim=0)
        # Softsign regularization keeps the update bounded in [-scale, scale].
        delta = self.scale * F.softsign(avg_load - load)
        self.biases.data += self.alpha * delta.to(self.biases.device)

    def apply(self, logits):
        """Add the current biases to the router logits."""
        if self.biases.device != logits.device:
            self.biases = self.biases.to(logits.device)
        return logits + self.biases


class SparseMoE(nn.Module):
    """Sparse MoE with dual-stage gated routing and a shared expert.

    Args:
        n_embed: token embedding dimension.
        num_experts: number of routed experts (excluding the shared expert).
        top_k: number of experts each token is routed to.
        capacity_factor: multiplier on the per-expert token capacity.
        alpha: step size of the dynamic bias adjuster.
        window_size: historical tracing window of the bias adjuster.
    """

    def __init__(self, n_embed, num_experts, top_k, capacity_factor=1.0,
                 alpha=0.1, window_size=3):
        super().__init__()
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
        # Shared expert that processes every token.
        self.expert_share = Expert(n_embed)
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.num_experts = num_experts
        self.dynamic_bias_adjuster = DynamicBiasAdjuster(
            num_experts, alpha=alpha, window_size=window_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # ---- Stage 1: noisy top-k routing ----
        gating_output, _ = self.router(x)
        flat_x = x.reshape(-1, x.size(-1))
        flat_gating_output = gating_output.reshape(-1, gating_output.size(-1))

        # Per-expert load estimate, used by the load-balancing strategy.
        load = flat_gating_output.sum(dim=0)
        if self.training:
            self.dynamic_bias_adjuster.update(load)

        # ---- Stage 2: bias-adjusted re-routing ----
        adjusted_logits = self.dynamic_bias_adjuster.apply(
            self.router.topkroute_linear(flat_x))
        adjusted_gating_output = F.softmax(adjusted_logits, dim=-1)
        adjusted_gating_output = adjusted_gating_output.reshape(batch_size, seq_len, -1)
        top_k_logits, indices = adjusted_gating_output.topk(self.top_k, dim=-1)
        zeros = torch.full_like(adjusted_gating_output, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)

        # ---- Expert dispatch ----
        updates = torch.zeros_like(flat_x)
        # The shared expert contributes to every token.
        updates += self.expert_share(flat_x)

        # Per-expert token capacity.
        capacity = int((batch_size * seq_len * self.top_k / self.num_experts)
                       * self.capacity_factor)

        for i, expert in enumerate(self.experts):
            # Tokens routed to expert i in the second-stage gating.
            expert_mask = (indices == i).any(dim=-1)
            selected_indices = torch.nonzero(expert_mask.view(-1)).squeeze(-1)
            limited_indices = selected_indices[:capacity]
            if limited_indices.numel() > 0:
                expert_output = expert(flat_x[limited_indices])
                # Weight each expert output by its first-stage routing score.
                gating_scores = flat_gating_output[limited_indices, i].unsqueeze(1)
                updates.index_add_(0, limited_indices, expert_output * gating_scores)

        return updates.view(batch_size, seq_len, -1)
