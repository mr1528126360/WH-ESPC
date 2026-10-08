# -*- coding: utf-8 -*-
"""WH-ESPCA Module 1: multi-encoder feature extraction.

Three frozen ImageNet-pretrained backbones (CaFormer-B36 / ResNet-152 /
Swin-B) extract per-slice embeddings; a trainable FC projection head maps
each stream into a shared 512-d latent space.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# encoder registry: key -> (timm model name, default weight file, input size)
ENCODER_REGISTRY = {
    "caformer":  ("caformer_b36.sail_in22k",      "caformer_b36.sail_in22k.bin", 224),
    "resnet152": ("resnet152.tv_in1k",            "resnet152.tv_in1k.bin", 224),
    "swin":      ("swin_base_patch4_window7_224", "swin_base_patch4_window7_224.bin", 224),
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FrozenBackboneEncoder(nn.Module):
    """Frozen timm backbone; per-slice global-pooled feature extraction."""

    def __init__(self, key, weights_path=None, num_classes=0):
        super().__init__()
        model_name, default_bin, self.img_size = ENCODER_REGISTRY[key]
        path = weights_path or os.path.join("checkpoints", default_bin)
        kwargs = dict(pretrained=True, num_classes=num_classes)
        if os.path.exists(path):
            kwargs["pretrained_cfg_overlay"] = dict(file=path)
        self.backbone = timm.create_model(model_name, **kwargs)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        cfg = self.backbone.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg.get("mean", IMAGENET_MEAN)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg.get("std", IMAGENET_STD)).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x):            # x: [B*, 3, S, S] in [0,1] -> [B*, D]
        return self.backbone((x - self.mean) / self.std)


class FCProjectionHead(nn.Module):
    """in_dim -> 1024 (BN,ReLU,Dropout) -> 768 (BN,ReLU,Dropout) -> embed_dim."""

    def __init__(self, in_dim, fc_hidden1=1024, fc_hidden2=768, embed_dim=512, drop_p=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2)
        self.fc3 = nn.Linear(fc_hidden2, embed_dim)
        self.drop = nn.Dropout(drop_p)

    def forward(self, x):            # x: [B, T, in_dim] -> [B, T, embed_dim]
        B, T, D = x.shape
        x = x.reshape(B * T, D)
        x = self.drop(F.relu(self.bn1(self.fc1(x))))
        x = self.drop(F.relu(self.bn2(self.fc2(x))))
        return self.fc3(x).view(B, T, -1)


def frame_indices(n_available, n_wanted):
    """Evenly-spaced slice sampling; padded with the last slice if insufficient."""
    if n_available >= n_wanted:
        return np.clip(np.round(np.linspace(0, n_available - 1, n_wanted)).astype(int),
                       0, n_available - 1)
    return np.array(list(range(n_available)) + [n_available - 1] * (n_wanted - n_available))


@torch.no_grad()
def extract_features(enc_key, scans, n_frames, device, weights_path=None,
                     cache_path=None, batch_size=32):
    """Extract per-scan sequence features with a frozen backbone.

    scans: list of (scan_id, label, [sorted image paths])
    returns float16 ndarray [N, n_frames, D]
    """
    from PIL import Image
    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path)
    encoder = FrozenBackboneEncoder(enc_key, weights_path).to(device)
    S = encoder.img_size
    N = len(scans)
    feats = []
    for i in range(0, N, batch_size):
        batch = np.zeros((min(batch_size, N - i), n_frames, 3, S, S), dtype=np.uint8)
        for j, (_, _, files) in enumerate(scans[i:i + batch_size]):
            for t, fi in enumerate(frame_indices(len(files), n_frames)):
                img = Image.open(files[fi]).convert("RGB").resize((S, S), Image.BILINEAR)
                batch[j, t] = np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)
        x = torch.from_numpy(batch).to(device).float() / 255.0
        f = encoder(x.view(-1, 3, S, S))
        feats.append(f.view(-1, n_frames, f.shape[-1]).half().cpu().numpy())
    arr = np.concatenate(feats, axis=0).astype(np.float16)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, arr)
    return arr
