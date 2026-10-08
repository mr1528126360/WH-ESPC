# -*- coding: utf-8 -*-
"""WH-ESPCA Module 3: LSTM sequence decoder + learnable gated late fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMDecoder(nn.Module):
    """LSTM along the slice axis; last time step -> FC head -> softmax.
    Also exposes the stream's hidden representation for the fusion gate."""

    def __init__(self, embed_dim=512, h_nodes=64, fc_dim=32, drop_p=0.5, num_classes=4):
        super().__init__()
        self.LSTM = nn.LSTM(input_size=embed_dim, hidden_size=h_nodes,
                            num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(h_nodes, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)
        self.fc2 = nn.Linear(fc_dim, num_classes)
        self.drop_p = drop_p

    def forward(self, x, return_hidden=False):
        self.LSTM.flatten_parameters()
        RNN_out, _ = self.LSTM(x)
        x = self.fc1(RNN_out[:, -1, :])
        h = self.bn(F.relu(x))
        h_d = F.dropout(h, p=self.drop_p, training=self.training)
        out = F.softmax(self.fc2(h_d), dim=1)
        return (out, h_d) if return_hidden else out


class GatedFusion(nn.Module):
    """Sample-adaptive learnable gated fusion — a learnable generalisation of
    USweA's fixed per-class accuracy weighting.

    Gate weights are computed per sample from the concatenated stream hidden
    states: g = softmax(W [h_1; ...; h_M]) in the probability simplex, and the
    final posterior is the convex combination sum_m g_m * p_m.
    """

    def __init__(self, stream_hidden_dim=32, n_streams=3):
        super().__init__()
        self.gate = nn.Linear(stream_hidden_dim * n_streams, n_streams)

    def forward(self, probs, hiddens):
        g = F.softmax(self.gate(torch.cat(hiddens, dim=1)), dim=1)     # [B, M]
        return (g.unsqueeze(-1) * torch.stack(probs, dim=1)).sum(dim=1)
