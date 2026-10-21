"""
Follow single messages through a trained model, for the architecture video.

  MessageTrace   what each step of the model did to one message: its tokens, their
                 embeddings, what [CLS] attended to, which expert ran each token, p(spam)
  trace_message  run one message through the model and keep all of that
  Pick           an example message, its trace, and how typical its [CLS] routing is
  preference     how good an example a message makes (smaller is better)
  pick_messages  choose one spam and one ham example from a list of (test) messages
"""
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

from vocab import PAD, UNK, encode, pieces, tokenize

MIN_TOKENS = 8   # examples shorter than this show too little of what attention does
MAX_TOKENS = 24  # longer ones do not fit on screen, so they are never picked


@contextmanager
def evaluating(model):
    """Eval mode (no dropout) inside the block, and the model's own mode back after it."""
    was_training = model.training
    model.eval()
    try:
        yield model
    finally:
        model.train(was_training)


@dataclass(frozen=True)
class MessageTrace:
    text: str
    words: tuple             # what to print for each token, [CLS] first
    tokens: tuple            # the token each word became: itself, [NUM], [NUM5+] or [UNK]
    embeddings: np.ndarray   # [n, d_model] token + position embedding of each token
    cls_attention: tuple     # per layer: [n] how much [CLS] takes from each token (sums to 1)
    experts: tuple           # per layer: the expert (0-based) that processed each token
    cls_probs: tuple         # per layer: [n_experts] the router's probabilities for [CLS]
    p_spam: float


@torch.no_grad()
def trace_message(model, vocab, text):
    ids = encode(vocab, [text], model.config.max_len)
    n = int((ids != PAD).sum())
    with evaluating(model):
        out = model(ids, need_weights=True)
    positions = torch.arange(n)
    embeddings = model.tokens(ids[0, :n]) + model.positions(positions)
    return MessageTrace(
        text=text,
        words=("[CLS]", *pieces(text))[:n],
        tokens=tuple(vocab.words[i] for i in ids[0, :n].tolist()),
        embeddings=embeddings.numpy(),
        cls_attention=tuple(w[0, 0, :n].numpy().copy() for w in out.attention),
        experts=tuple(tuple(r.expert[0, :n].tolist()) for r in out.routing),
        cls_probs=tuple(r.probs[0, 0].numpy().copy() for r in out.routing),
        p_spam=float(torch.sigmoid(out.logits[0])),
    )


@dataclass(frozen=True)
class Pick:
    text: str
    trace: MessageTrace
    same: int   # messages with this label whose last-layer [CLS] goes to the same expert...
    total: int  # ...out of this many messages with this label


def preference(wrong, unusual, n_tokens, n_punctuation, n_chars):
    """
    Sort key for example messages, smaller first: one the model gets right, whose
    last-layer [CLS] goes where most of its label goes, long enough to show what attention
    does, with little punctuation (it reads like a sentence), and short, so it fits.
    """
    return (wrong, unusual, n_tokens < MIN_TOKENS, n_punctuation, n_chars)


def _punctuation(tokens):
    return sum(not t[0].isalnum() and not t.startswith("[") for t in tokens)


@torch.no_grad()
def pick_messages(model, vocab, messages, fits=None):
    """
    {"spam": Pick, "ham": Pick}: for each label, the most preferred message with no [UNK]
    token and at most MAX_TOKENS tokens. fits(trace) can turn down a message that would not
    fit on screen; the next one in order is tried.
    """
    if not messages:
        raise ValueError("no messages to pick from")
    with evaluating(model):
        out = model(encode(vocab, [m.text for m in messages], model.config.max_len))
    cls_expert = out.routing[-1].expert[:, 0]
    says_spam = out.logits > 0
    picks = {}
    for label, is_spam in (("spam", True), ("ham", False)):
        rows = [i for i, m in enumerate(messages) if m.is_spam == is_spam]
        if not rows:
            raise ValueError(f"no {label} messages to pick from")
        usual = torch.bincount(cls_expert[rows], minlength=model.config.n_experts).argmax()
        tokens = {i: tokenize(messages[i].text) for i in rows}
        candidates = sorted(
            (i for i in rows if 0 < len(tokens[i]) <= MAX_TOKENS
             and all(vocab.id(t) != UNK for t in tokens[i])),
            key=lambda i: preference(bool(says_spam[i]) != is_spam, bool(cls_expert[i] != usual),
                                     len(tokens[i]), _punctuation(tokens[i]),
                                     len(messages[i].text)))
        traces = (trace_message(model, vocab, messages[i].text) for i in candidates)
        trace = next((t for t in traces if fits is None or fits(t)), None)
        if trace is None:
            raise ValueError(f"no {label} message is short enough, with known words, to show")
        expert = trace.experts[-1][0]
        same = int((cls_expert[rows] == expert).sum())
        picks[label] = Pick(trace.text, trace, same, len(rows))
    return picks
