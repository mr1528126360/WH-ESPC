"""Module 3 of WH-ESPC: sequential classification decision along the Z-axis.

The refined slice features are processed by an LSTM along the axial Z-axis
(from lung apex to lung base), capturing the spatial anatomical continuity
across contiguous CT slices. The representation at the final time step is
fed to a small fully-connected classifier.

The original experiments use a unidirectional LSTM; set
``bidirectional=True`` for the Bi-LSTM variant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMDecoder(nn.Module):
    """Classify a sequence of slice embeddings.

    Args:
        embed_dim: dimension of the input slice embeddings.
        hidden_layers: number of stacked LSTM layers.
        hidden_nodes: LSTM hidden size (per direction).
        fc_dim: hidden size of the classifier head.
        drop_p: dropout probability before the output layer.
        num_classes: number of target categories.
        bidirectional: whether to use a Bi-LSTM.
    """

    def __init__(self, embed_dim=512, hidden_layers=1, hidden_nodes=64,
                 fc_dim=32, drop_p=0.3, num_classes=3, bidirectional=False):
        super().__init__()
        self.drop_p = drop_p
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_nodes,
            num_layers=hidden_layers,
            batch_first=True,           # input shape: (batch, time_step, input_size)
            bidirectional=bidirectional,
        )

        self.bn = nn.BatchNorm1d(fc_dim)
        self.fc1 = nn.Linear(hidden_nodes * self.num_directions, fc_dim)
        self.fc2 = nn.Linear(fc_dim, num_classes)

    def forward(self, x):
        """Args:
            x: slice features of shape ``[batch, num_slices, embed_dim]``.

        Returns:
            probs: class probabilities of shape ``[batch, num_classes]``
                (softmax output, kept for compatibility with the original
                training procedure).
            features: the ``fc_dim``-dimensional representation before the
                output layer, useful for visualization (e.g. Grad-CAM/t-SNE).
        """
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x)

        # Use the representation at the last time step (lung base).
        h = self.fc1(lstm_out[:, -1, :])
        h = F.relu(h)
        h = self.bn(h)
        features = F.dropout(h, p=self.drop_p, training=self.training)
        logits = self.fc2(features)
        probs = F.softmax(logits, dim=1)
        return probs, features
