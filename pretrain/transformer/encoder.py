import torch.nn as nn
import torch


class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm()

    def forward(self, x, mask):
        return self.norm(x)