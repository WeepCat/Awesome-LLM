import torch.nn as nn
import torch
from attention import MultiHeadAttention
from feed_forward import FeedForward
from copy import deepcopy


def clone(model: nn.Module, n: int):
    return nn.ModuleList([deepcopy(model) for _ in range(n)])


class TransformerLayer(nn.Module):
    def __init__(self, *,
                 d_model: int,
                 self_attn: MultiHeadAttention,
                 src_attn: MultiHeadAttention = None,
                 feed_forward: FeedForward,
                 dropout_prob: float):
        super().__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_self_attn = nn.LayerNorm([d_model])
        if self.src_attn is not None:
            self.norm_src_attn = nn.LayerNorm([d_model])
        self.norm_ff = nn.LayerNorm([d_model])
        self.d_model = d_model

    def forward(self, *,
                x: torch.Tensor,
                mask: torch.Tensor,
                src: torch.Tensor = None,
                src_mask: torch.Tensor = None):
        z = self.norm_self_attn(x)
        self_attn = self.self_attn(query=z, key=z, value=z, mask=mask)
        x = x + self.dropout(self_attn)

        if src is not None:
            z = self.norm_src_attn(x)
            src_attn = self.src_attn(query=z, key=src, value=src, mask=src_mask)
            x = x + self.dropout(src_attn)

        z = self.norm_ff(x)
        ff = self.feed_forward(z)
        x = x + self.dropout(ff)
        return x


class Encoder(nn.Module):
    
    def __init__(self, layer: TransformerLayer, n_layers: int):
        super().__init__()
        self.layers = clone(layer, n_layers)
        self.norm = nn.LayerNorm([layer.d_model])

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x=x, mask=mask)
        return self.norm(x)

    def __call__(self, x: torch.Tensor, mask: torch.Tensor):
        return self.forward(x=x, mask=mask)


class Decoder(nn.Module):

    def __init__(self, layer: TransformerLayer, n_layers: int):
        super().__init__()
        self.layers = clone(layer, n_layers)
        self.norm = nn.LayerNorm([layer.d_model])

    def forward(self, x: torch.Tensor, tgt_mask: torch.Tensor,
                src: torch.Tensor, src_mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x=x, mask=tgt_mask, src=src, src_mask=src_mask)
        return self.norm(x)

    def __call__(self, x: torch.Tensor, tgt_mask: torch.Tensor, src: torch.Tensor, src_mask: torch.Tensor):
        return self.forward(x=x, tgt_mask=tgt_mask, src=src, src_mask=src_mask)


class Generator(nn.Module):

    def __init__(self, n_vocab: int, d_model: int):
        super().__init__()
        self.projection = nn.Linear(d_model, n_vocab)

    def forward(self, x: torch.Tensor):
        return self.projection(x)

    def __call__(self, x: torch.Tensor):
        return self.forward(x)


class Transformer(nn.Module):
    def __init__(self,
                 src_embed: nn.Module,
                 tgt_embed: nn.Module,
                 positional_encoding: nn.Module,
                 encoder: Encoder,
                 decoder: Decoder,
                 generator: Generator):
        super().__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.pe = positional_encoding
        self.encoder = encoder
        self.decoder = decoder,
        self.generator = generator

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        src = self.pe(self.src_embed(src))
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, tgt: torch.Tensor, tgt_mask: torch.Tensor, src: torch.Tensor, src_mask: torch.Tensor):
        tgt = self.pe(self.tgt_embed(tgt))
        return self.decoder(tgt, tgt_mask, src, src_mask)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        src = self.encode(src, src_mask)
        return self.decode(src, src_mask, tgt, tgt_mask)





