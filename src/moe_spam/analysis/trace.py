"""Following one message through a trained model, keeping what each step did to it."""

from dataclasses import dataclass

import numpy as np
import torch

from moe_spam.data.tokenizer import PAD, TOKENIZER
from moe_spam.model.modes import evaluating


@dataclass(frozen=True)
class MessageTrace:
    """What each step of the model did to one message.

    Attributes:
        text: The message.
        words: What to print for each token, ``[CLS]`` first.
        tokens: The token each word became: itself, ``[NUM]``, ``[NUM5+]`` or ``[UNK]``.
        embeddings: ``[n, d_model]`` token + position embedding of each token.
        cls_attention: Per layer, ``[n]``: how much ``[CLS]`` takes from each token (sums to
            1, averaged over the heads).
        experts: Per layer, the expert (0-based) that processed each token.
        cls_probs: Per layer, ``[n_experts]``: the router's probabilities for ``[CLS]``.
        p_spam: The model's spam probability for the message.
    """

    text: str
    words: tuple
    tokens: tuple
    embeddings: np.ndarray
    cls_attention: tuple
    experts: tuple
    cls_probs: tuple
    p_spam: float

    @property
    def d_model(self):
        """The width of the model's token vectors."""
        return self.embeddings.shape[1]

    @property
    def n_experts(self):
        """The number of experts per layer."""
        return len(self.cls_probs[0])


class MessageTracer:
    """Runs single messages through a model, in eval mode, and records every step.

    Attributes:
        model: The trained ``TinyBert``.
        vocab: Its vocabulary.
    """

    def __init__(self, model, vocab, tokenizer=TOKENIZER):
        """Creates a tracer for ``model`` and its ``vocab``."""
        self.model = model
        self.vocab = vocab
        self.tokenizer = tokenizer

    @torch.no_grad()
    def trace(self, text):
        """Returns the ``MessageTrace`` of ``text`` (cut to the model's ``max_len``)."""
        ids = self.vocab.encode([text], self.model.config.max_len)
        n = int((ids != PAD).sum())
        with evaluating(self.model):
            out = self.model(ids, need_weights=True)
        embeddings = self.model.tokens(ids[0, :n]) + self.model.positions(torch.arange(n))
        return MessageTrace(
            text=text,
            words=("[CLS]", *self.tokenizer.pieces(text))[:n],
            tokens=tuple(self.vocab.words[i] for i in ids[0, :n].tolist()),
            embeddings=embeddings.numpy(),
            cls_attention=tuple(w[0, 0, :n].numpy().copy() for w in out.attention),
            experts=tuple(tuple(r.expert[0, :n].tolist()) for r in out.routing),
            cls_probs=tuple(r.probs[0, 0].numpy().copy() for r in out.routing),
            p_spam=float(torch.sigmoid(out.logits[0])),
        )
