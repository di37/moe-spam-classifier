"""A tiny BERT-style encoder whose feed-forward blocks are mixture-of-experts layers.

Devlin et al., "BERT", 2019. Kept apart from BERT on purpose: whole words instead of word
pieces, no sentence-pair segment embeddings, LayerNorm before each sublayer rather than after
(it trains more reliably from scratch on little data), no pooler, and no masked-language-model
pre-training: the model learns from the ham/spam labels alone.
"""

from dataclasses import dataclass

import torch
from torch import nn

from moe_spam.data.tokenizer import CLS, PAD
from moe_spam.data.vocab import MAX_LEN
from moe_spam.model.moe import MoELayer

N_LAYERS, N_EXPERTS = 2, 3  # the default shape: the one every video is drawn for


@dataclass(frozen=True)
class BertConfig:
    """The sizes that define a model; saved with every checkpoint.

    Attributes:
        vocab_size: The number of tokens in the vocabulary.
        max_len: The longest sequence the position embeddings cover, ``[CLS]`` included.
        d_model: The width of every token vector.
        n_heads: Attention heads per layer.
        n_layers: Encoder blocks.
        n_experts: Experts per MoE layer (1 makes a plain dense BERT).
        d_ff: The hidden width of each expert.
        dropout: The dropout rate during training.
    """

    vocab_size: int
    max_len: int = MAX_LEN
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = N_LAYERS
    n_experts: int = N_EXPERTS
    d_ff: int = 256
    dropout: float = 0.1


@dataclass(frozen=True)
class Output:
    """What one forward pass returns.

    Attributes:
        logits: ``[B]`` the spam logit of each message.
        aux_loss: The load-balancing loss, averaged over the layers.
        routing: One ``Routing`` per layer.
        attention: With ``need_weights=True``, per layer the ``[B, L, L]`` attention weights,
            averaged over the heads; otherwise None.
    """

    logits: torch.Tensor
    aux_loss: torch.Tensor
    routing: tuple
    attention: tuple | None = None


class EncoderBlock(nn.Module):
    """One encoder block: self-attention, then a MoE feed-forward layer.

    Each has a LayerNorm in front and a residual connection around it.

    Attributes:
        norm_attn: The LayerNorm before attention.
        attn: Multi-head self-attention.
        norm_ffn: The LayerNorm before the MoE layer.
        moe: The mixture-of-experts feed-forward layer.
        dropout: Dropout on both sublayers' outputs.
    """

    def __init__(self, d_model, n_heads, n_experts, d_ff, dropout):
        """Builds the block; the arguments are the ``BertConfig`` fields of the same names."""
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.moe = MoELayer(d_model, n_experts, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, need_weights=False):
        """Runs the block.

        Args:
            x: ``[B, L, d_model]`` tokens.
            mask: ``[B, L]`` bool, True for real tokens.
            need_weights: Whether to return the attention weights.

        Returns:
            ``(x, aux_loss, routing, weights)``; ``weights`` is None unless asked for.
        """
        h = self.norm_attn(x)
        # key_padding_mask is True where a key must be *ignored*: the padding.
        attended, weights = self.attn(h, h, h, key_padding_mask=~mask, need_weights=need_weights)
        x = x + self.dropout(attended)
        moe_out, aux, routing = self.moe(self.norm_ffn(x), mask)
        return x + self.dropout(moe_out), aux, routing, weights


class TinyBert(nn.Module):
    """Token and position embeddings, a stack of encoder blocks, a classifier on ``[CLS]``.

    Attributes:
        config: The model's sizes.
        tokens: Token embeddings (``[PAD]`` is all zeros).
        positions: Learned position embeddings.
        blocks: The encoder blocks.
        norm: The final LayerNorm, on ``[CLS]``.
        classifier: ``Linear(d_model, 1)``: the spam logit.
    """

    def __init__(self, config):
        """Builds the model described by ``config`` (a ``BertConfig``)."""
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
        """Scores messages.

        Args:
            ids: ``[B, L]`` token ids from ``Vocab.encode``: ``[CLS]`` first, ``[PAD]`` last.
            need_weights: Whether to also return each layer's attention weights (for drawing;
                a little slower).

        Returns:
            The ``Output`` of the pass.

        Raises:
            ValueError: If the rows are longer than ``max_len`` or do not start with ``[CLS]``.
        """
        if ids.shape[1] > self.config.max_len:
            raise ValueError(f"sequences are {ids.shape[1]} long; this model's max_len is "
                             f"{self.config.max_len}")
        if not (ids[:, 0] == CLS).all():
            raise ValueError("every row must start with [CLS]; encode texts with Vocab.encode")
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
