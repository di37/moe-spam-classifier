"""
A tiny BERT-style encoder (Devlin et al., 2019) whose feed-forward blocks are mixture-of-experts
layers, trained from scratch to tell spam from ham.

  BertConfig    the sizes that define a model (saved with checkpoints)
  EncoderBlock  self-attention, then a MoE feed-forward layer; each has a LayerNorm in front
                and a residual connection around it
  TinyBert      token + position embeddings, a stack of blocks, a classifier on the final [CLS]
  Output        what a forward pass returns: spam logits, load-balancing loss, each layer's
                Routing, and on request each layer's attention weights

Kept apart from BERT on purpose: whole words instead of word pieces, no sentence-pair segment
embeddings, LayerNorm before each sublayer rather than after (it trains more reliably from
scratch on little data), no pooler, and no masked-language-model pre-training: the model learns
from the ham/spam labels alone.
"""
from dataclasses import dataclass

import torch
from torch import nn

from moe import MoELayer
from vocab import CLS, PAD


@dataclass(frozen=True)
class BertConfig:
    vocab_size: int
    max_len: int = 64
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    n_experts: int = 3
    d_ff: int = 256
    dropout: float = 0.1


@dataclass(frozen=True)
class Output:
    logits: torch.Tensor    # [B] spam logit of each message
    aux_loss: torch.Tensor  # load-balancing loss, averaged over the layers
    routing: tuple          # one moe.Routing per layer
    attention: tuple = None  # with need_weights=True: per layer, [B, L, L] averaged over heads


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_experts, d_ff, dropout):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.moe = MoELayer(d_model, n_experts, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, need_weights=False):
        h = self.norm_attn(x)
        # key_padding_mask is True where a key must be *ignored*: the padding.
        attended, weights = self.attn(h, h, h, key_padding_mask=~mask, need_weights=need_weights)
        x = x + self.dropout(attended)
        moe_out, aux, routing = self.moe(self.norm_ffn(x), mask)
        return x + self.dropout(moe_out), aux, routing, weights


class TinyBert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokens = nn.Embedding(config.vocab_size, config.d_model, padding_idx=PAD)
        self.positions = nn.Embedding(config.max_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            EncoderBlock(config.d_model, config.n_heads, config.n_experts, config.d_ff,
                         config.dropout)
            for _ in range(config.n_layers)
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, 1)

    def forward(self, ids, need_weights=False):
        """
        ids [B, L] from vocab.encode: [CLS] first, [PAD] at the end.
        need_weights: also return each layer's attention weights (for drawing; a bit slower).
        """
        if ids.shape[1] > self.config.max_len:
            raise ValueError(f"sequences are {ids.shape[1]} long; this model's max_len is "
                             f"{self.config.max_len}")
        if not (ids[:, 0] == CLS).all():
            raise ValueError("every row must start with [CLS]; encode texts with vocab.encode")
        mask = ids != PAD
        x = self.dropout(self.tokens(ids) + self.positions(torch.arange(ids.shape[1])))
        aux_losses, routing, attention = [], [], []
        for block in self.blocks:
            x, aux, layer_routing, weights = block(x, mask, need_weights)
            aux_losses.append(aux)
            routing.append(layer_routing)
            attention.append(weights)
        # Like BERT, the whole message is summed up by the final [CLS] token.
        logits = self.classifier(self.norm(x[:, 0])).squeeze(-1)
        return Output(logits=logits, aux_loss=torch.stack(aux_losses).mean(),
                      routing=tuple(routing),
                      attention=tuple(attention) if need_weights else None)
