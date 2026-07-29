"""WH-ESPC: full model definition.

The model chains the three key modules:

1. ``CaFormerEncoder``  — pre-trained CaFormer feature extraction module.
2. ``MoETransformer``   — multi-sequence CT feature extraction module based on
   a sparse MoE with dual-stage gated routing, historical-memory load
   balancing and softsign regularization.
3. ``LSTMDecoder``      — classification decision module processing the slice
   sequence along the axial Z-axis (lung apex -> lung base).
"""

import torch
import torch.nn as nn

from .encoder import CaFormerEncoder
from .transformer import MoETransformer
from .decoder import LSTMDecoder


class WHESPC(nn.Module):
    """WH-ESPC model for multi-sequence chest CT classification."""

    def __init__(self,
                 # Module 1: feature extraction
                 backbone_name="caformer_b36.sail_in22k",
                 backbone_weights=None,
                 backbone_feature_dim=768,
                 fc_hidden1=1024,
                 fc_hidden2=768,
                 embed_dim=512,
                 dropout=0.3,
                 # Module 2: MoE transformer
                 num_heads=4,
                 num_layers=2,
                 num_experts=16,
                 top_k=4,
                 capacity_factor=1.0,
                 alpha=0.1,
                 window_size=3,
                 # Module 3: sequence classification
                 lstm_layers=1,
                 lstm_hidden=64,
                 lstm_fc_dim=32,
                 bidirectional=False,
                 num_classes=3):
        super().__init__()
        self.encoder = CaFormerEncoder(
            model_name=backbone_name,
            backbone_weights=backbone_weights,
            backbone_feature_dim=backbone_feature_dim,
            fc_hidden1=fc_hidden1,
            fc_hidden2=fc_hidden2,
            embed_dim=embed_dim,
            drop_p=dropout,
        )
        self.transformer = MoETransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k,
            capacity_factor=capacity_factor,
            alpha=alpha,
            window_size=window_size,
        )
        self.decoder = LSTMDecoder(
            embed_dim=embed_dim,
            hidden_layers=lstm_layers,
            hidden_nodes=lstm_hidden,
            fc_dim=lstm_fc_dim,
            drop_p=dropout,
            num_classes=num_classes,
            bidirectional=bidirectional,
        )

    def forward(self, x):
        """Args:
            x: input scan of shape ``[batch, num_slices, C, H, W]``.

        Returns:
            probs: class probabilities of shape ``[batch, num_classes]``.
            features: penultimate representation ``[batch, lstm_fc_dim]``.
        """
        features = self.encoder(x)       # [batch, num_slices, embed_dim]
        features = self.transformer(features)
        probs, features = self.decoder(features)
        return probs, features

    def trainable_parameters(self):
        """Parameters optimized during training.

        The frozen CaFormer backbone is excluded; the encoder projection
        head, the MoE transformer and the LSTM decoder are trainable.
        """
        modules = [self.encoder.fc1, self.encoder.bn1,
                   self.encoder.fc2, self.encoder.bn2,
                   self.encoder.fc3, self.transformer, self.decoder]
        params = []
        for module in modules:
            params += list(module.parameters())
        return params
