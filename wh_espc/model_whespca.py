# -*- coding: utf-8 -*-
"""WH-ESPCA: multi-encoder dynamic-bias MoE sequence network.

Three frozen heterogeneous encoders (CaFormer-B36 / ResNet-152 / Swin-B)
-> per-stream FC projection head + MoE-Transformer + LSTM decoder
-> sample-adaptive gated weighted fusion.
"""

import torch
import torch.nn as nn

from .encoders import FCProjectionHead
from .moe_transformer import MoETransformer
from .decoder_whespca import LSTMDecoder, GatedFusion


class WHESPCA(nn.Module):
    """Full WH-ESPCA model operating on pre-extracted backbone features.

    Inputs: list of per-stream features [B, T, D_m] (from the frozen
    encoders; see wh_espc.encoders.extract_features).
    Output: class probabilities [B, K].
    """

    def __init__(self, in_dims=(768, 2048, 1024), num_classes=4, drop_p=0.5,
                 embed_dim=512, num_heads=4, num_layers=2, num_experts=16, top_k=4):
        super().__init__()
        n = len(in_dims)
        self.fc_encoders = nn.ModuleList([FCProjectionHead(d, embed_dim=embed_dim, drop_p=drop_p)
                                          for d in in_dims])
        self.transformers = nn.ModuleList([MoETransformer(embed_dim, num_heads, num_layers,
                                                          num_experts, top_k) for _ in range(n)])
        self.decoders = nn.ModuleList([LSTMDecoder(embed_dim, drop_p=drop_p,
                                                   num_classes=num_classes) for _ in range(n)])
        self.fusion = GatedFusion(stream_hidden_dim=32, n_streams=n)

    def forward(self, xs):           # xs: list of [B, T, D_m]
        probs, hiddens = [], []
        for fc, tr, dec, x in zip(self.fc_encoders, self.transformers, self.decoders, xs):
            p, h = dec(tr(fc(x)), return_hidden=True)
            probs.append(p)
            hiddens.append(h)
        return self.fusion(probs, hiddens)
