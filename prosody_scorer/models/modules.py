import torch
import torch.nn as nn


# Attention pooling
class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, 1),
            nn.GELU(),
        )

    def forward(self, x, attn, mask):
        w = self.attention(attn).float()
        w[mask == 0] = float("-inf")
        w = torch.softmax(w, 1)
        x = torch.sum(w * x, dim=1)
        return x
