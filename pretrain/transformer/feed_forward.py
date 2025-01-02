import torch
import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(
            self,
            d_model: int,
            d_ff: int,
            dropout_prob: float = 0.1,
            activation=nn.ReLU(),
            is_gated: bool = True,
            bias1: bool = True,
            bias2: bool = True,
            bias_gate: bool = True
    ):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=bias1)
        self.linear2 = nn.Linear(d_ff, d_model, bias=bias2)
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = activation
        self.is_gated = is_gated
        if is_gated:
            self.linear_v = nn.Linear(d_model, d_ff, bias=bias_gate)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: shape (batch_size, seq_len, d_model)
        Return:
            out: shape (batch_size, seq_len, d_model). The output of a multihead attention layer
        """
        g = self.activation(self.linear1(x))
        if self.is_gated:
            x = g * self.linear_v(x)
        else:
            x = g
        x = self.dropout(x)
        return self.layer2(x)
