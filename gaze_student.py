from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class ViTGazeStudent(nn.Module):
    """
    ViT-B/16 student trained to reproduce the unsupervised paper-based
    teacher's spatial gaze distribution.

    Input:
        [B, 3, 224, 224]

    Output:
        [B, 1, 14, 14] logits
    """

    def __init__(self, dropout: float = 0.2):
        super().__init__()

        self.vit = models.vit_b_16(
            weights=models.ViT_B_16_Weights.DEFAULT
        )

        dim = self.vit.hidden_dim

        self.gaze_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.vit._process_input(images)  # [B, 196, 768]
        batch_size = x.shape[0]

        cls = self.vit.class_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.vit.encoder(x)              # [B, 197, 768]

        patches = x[:, 1:, :]                # [B, 196, 768]
        logits = self.gaze_head(patches).squeeze(-1)

        n = int(logits.shape[1] ** 0.5)
        return logits.reshape(batch_size, 1, n, n)


def gaze_distribution(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert logits to a spatial probability distribution whose cells sum to 1.
    """
    b = logits.shape[0]
    probs = torch.softmax(logits.reshape(b, -1), dim=1)
    return probs.reshape_as(logits)
