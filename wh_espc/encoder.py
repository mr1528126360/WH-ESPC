"""Module 1 of WH-ESPC: CaFormer-based per-slice feature extraction.

A pre-trained CaFormer (or any compatible timm model) is used as a frozen
image feature extractor. Each axial CT slice is embedded independently, and a
small trainable fully-connected head projects the backbone features into the
latent embedding space shared by the downstream modules.
"""

import os

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class CaFormerEncoder(nn.Module):
    """Extract a latent embedding for every slice of a CT scan.

    Args:
        model_name: timm model identifier, e.g. ``caformer_b36.sail_in22k``.
        backbone_weights: optional path to a local pre-trained checkpoint
            (``.bin``). If the file does not exist, weights are downloaded
            from the Hugging Face Hub instead.
        backbone_feature_dim: output feature dimension of the backbone
            (768 for ``caformer_b36.sail_in22k``).
        fc_hidden1: hidden size of the first projection layer.
        fc_hidden2: hidden size of the second projection layer.
        embed_dim: latent embedding dimension fed to the MoE transformer.
        drop_p: dropout probability (kept for interface compatibility).
    """

    def __init__(self, model_name="caformer_b36.sail_in22k", backbone_weights=None,
                 backbone_feature_dim=768, fc_hidden1=1024, fc_hidden2=768,
                 embed_dim=512, drop_p=0.3):
        super().__init__()

        if backbone_weights and os.path.isfile(backbone_weights):
            backbone = timm.create_model(
                model_name, pretrained=True, num_classes=0,
                pretrained_cfg_overlay=dict(file=backbone_weights))
        else:
            backbone = timm.create_model(model_name, pretrained=True, num_classes=0)

        # The backbone is used as a frozen feature extractor.
        for param in backbone.parameters():
            param.requires_grad_(False)
        self.backbone = backbone

        # Trainable projection head.
        self.fc1 = nn.Linear(backbone_feature_dim, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2)
        self.fc3 = nn.Linear(fc_hidden2, embed_dim)
        self.drop_p = drop_p

    def forward(self, x):
        """Embed every slice of the input scan.

        Args:
            x: input tensor of shape ``[batch, num_slices, C, H, W]``.

        Returns:
            Tensor of shape ``[batch, num_slices, embed_dim]``.
        """
        embed_seq = []
        for t in range(x.size(1)):
            with torch.no_grad():
                feat = self.backbone(x[:, t])
            h = F.relu(self.bn1(self.fc1(feat)))
            h = F.relu(self.bn2(self.fc2(h)))
            embed_seq.append(self.fc3(h))
        # Stack along the slice axis: [batch, num_slices, embed_dim].
        return torch.stack(embed_seq, dim=1)
