import math
from typing import Optional, List
import torch
from torch import nn


class PrepareForMultiHeadAttention(nn.Module):
    """
    approx a linear transformation
    """

    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        self.heads = heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        head_shape = x.shape[:-1]
        x = self.linear(x)
        x = x.view(*head_shape, self.heads, self.d_k) # bs, seq_len, heads, d_k
        return x


class MultiHeadAttention(nn.Module):

    def __init__(
            self,
            d_model: int,
            heads: int,
            dropout_prob: float = 0.1,
            bias: bool = True
    ):
        super().__init__()
        self.d_k = d_model // heads
        self.heads = heads
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias)
        self.dropout = nn.Dropout(dropout_prob)
        self.scale = 1 / math.sqrt(self.d_k)
        self.softmax = nn.Softmax(dim=-1)
        self.output = nn.Linear(d_model, d_model)
        self.attn = None

    def get_score(self, query: torch.Tensor, key: torch.Tensor):
        score = torch.matmul(query, key.transpose(-2, -1))
        return score

    def prepare_mask(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]):
        assert len(mask.shape) == 2 or len(mask.shape) == 3
        if mask.shape == 2:
            assert mask.shape[0] == query_shape[1]
            assert mask.shape[1] == key_shape[1]
            mask = mask.unsqueeze(0)
        else:
            assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
            assert mask.shape[1] == query_shape[1]
            assert mask.shape[2] == key_shape[1]
        return mask

    def forward(self, *,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        """
        Args:
            query: shape (batch_size, seq_len, d_model)
            key: shape (batch_size, seq_len, d_model)
            value: shape (batch_size, seq_len, d_model)
            mask: shape (batch_size, seq_len, seq_len). Since we assume all data use a same mask, so
                  here the shape also equals to (1, seq_len, seq_len)

        Return:
            out: shape (batch_size, seq_len, d_model). The output of a multihead attention layer
        """
        seq_len, batch_size, _ = query.shape
        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        scores = self.get_score(query, key)
        scores *= self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = self.softmax(scores)
        attn = self.dropout(attn)
        x = torch.matmul(attn, value)
        self.attn = attn.detach()

        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.output(x)


