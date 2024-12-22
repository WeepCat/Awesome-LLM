import torch
import torch.nn as nn
import math


def get_sinusoidal_positional_encoding(d_model: int, max_len: int = 4096):
    # Empty encodings vectors
    encodings = torch.zeros(max_len, d_model)
    # Position indexes
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    # $2 * i$
    two_i = torch.arange(0, d_model, 2, dtype=torch.float32)
    # $10000^{\frac{2i}{d_{model}}}$
    div_term = torch.exp(two_i * -(math.log(10000.0) / d_model))
    # $PE_{p,2i} = sin\Bigg(\frac{p}{10000^{\frac{2i}{d_{model}}}}\Bigg)$
    encodings[:, 0::2] = torch.sin(position * div_term)
    # $PE_{p,2i + 1} = cos\Bigg(\frac{p}{10000^{\frac{2i}{d_{model}}}}\Bigg)$
    encodings[:, 1::2] = torch.cos(position * div_term)
    # Add batch dimension
    encodings = encodings.unsqueeze(1).requires_grad_(False)
    return encodings


class SinusoidalPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout_prob: float, max_len: int = 4096):
        super().__init__()
        self.register_buffer('positional_encoding', get_sinusoidal_positional_encoding(d_model, max_len), False)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor):
        pe = self.positional_encoding[:x.shape[0]].detach().requires_grad_(False)
        return self.dropout(x + pe)


class LearnedPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout_prob: float, max_len: int = 4096):
        super().__init__()
        self.positional_encoding = nn.Parameter(torch.zeros(max_len, 1, d_model), requires_grad=True)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor):
        pe = self.positional_encoding[:x.shape[0]]
        return self.dropout(x + pe)


class RelativePositionEncoding(nn.Module):

    def __init__(self, d_model: int, max_relative_position: int):
        super().__init__()
        self.max_relative_position = max_relative_position
        self.positional_encoding = nn.Parameter(torch.zeros(max_relative_position * 2 + 1, d_model), requires_grad=True)

    def forward(self, length_q: int, length_k: int):
        range_vec_q = torch.arange(length_q).cuda()
        range_vec_k = torch.arange(length_k).cuda()
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.positional_encoding[final_mat]
        return embeddings
