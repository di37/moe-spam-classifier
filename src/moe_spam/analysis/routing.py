"""The numbers behind the routing equations, read from a trained model.

For one token ``x`` an MoE layer computes::

    p = softmax(W_router · x)     i = argmax p     y = p_i · FFN_i(x)

and training adds ``aux = n_experts · Σ f_i · P_i`` to keep the experts evenly used.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from moe_spam.data.tokenizer import PAD, TOKENIZER
from moe_spam.model.modes import evaluating


@dataclass(frozen=True)
class RouterStep:
    """One token's trip through one MoE layer.

    Attributes:
        layer: The layer, from 0.
        position: The token's place in the message (0 is ``[CLS]``).
        word: The token as printed.
        x: ``[d_model]``: the MoE layer's input for the token (after attention and LayerNorm).
        weights: ``[n_experts, d_model]``: the router's weights, ``W_router``.
        logits: ``[n_experts]``: ``W_router · x``.
        probs: ``[n_experts]``: ``p``, the softmax of the logits.
        expert: ``i``, the one expert that runs (0-based).
        hidden: ``[d_ff]``: expert ``i``'s hidden layer, after the GELU.
        ffn_out: ``[d_model]``: ``FFN_i(x)``.
        y: ``[d_model]``: the layer's output for the token, ``p_i · FFN_i(x)``.
        loss_slope: ``dL/dp_i``, how fast the message's loss changes with ``p_i``. Training
            raises ``p_i`` where it is negative and lowers it where it is positive.
    """

    layer: int
    position: int
    word: str
    x: np.ndarray
    weights: np.ndarray
    logits: np.ndarray
    probs: np.ndarray
    expert: int
    hidden: np.ndarray
    ffn_out: np.ndarray
    y: np.ndarray
    loss_slope: float

    @property
    def p_i(self):
        """The probability of the expert that runs."""
        return float(self.probs[self.expert])


@dataclass(frozen=True)
class LayerBalance:
    """How evenly one layer spreads the real tokens of a set of messages.

    Attributes:
        load: ``[n_experts]``: ``f_i``, the share of tokens each expert gets.
        mean_probs: ``[n_experts]``: ``P_i``, the router's mean probability for each expert.
        aux: The load-balancing loss, ``n_experts · Σ f_i · P_i``: 1 when even, up to
            ``n_experts`` when one expert gets everything.
    """

    load: np.ndarray
    mean_probs: np.ndarray
    aux: float


class RouterProbe:
    """Reads the router's arithmetic out of a model, in eval mode.

    Attributes:
        model: The model (a ``TinyBert``).
        vocab: Its vocabulary.
    """

    def __init__(self, model, vocab, tokenizer=TOKENIZER):
        """Creates a probe for ``model`` and its ``vocab``."""
        self.model = model
        self.vocab = vocab
        self.tokenizer = tokenizer

    def step(self, text, is_spam, layer, position):
        """Follows one token of a message through one MoE layer.

        Args:
            text: The message.
            is_spam: Its label, for the loss that ``loss_slope`` is the slope of.
            layer: The layer, from 0.
            position: The token, from 0 (``[CLS]``).

        Returns:
            The ``RouterStep``.

        Raises:
            ValueError: If the layer or the position does not exist.
        """
        ids = self._encode([text])
        n = int((ids != PAD).sum())
        if not 0 <= layer < len(self.model.blocks):
            raise ValueError(f"no layer {layer}: the model has {len(self.model.blocks)}")
        if not 0 <= position < n:
            raise ValueError(f"no position {position}: the message has {n} tokens")
        moe = self.model.blocks[layer].moe
        with evaluating(self.model), torch.no_grad():
            x = self._moe_input(ids, layer)[0, position]
            logits = moe.router(x)
            probs = F.softmax(logits, dim=-1)
            expert = int(probs.argmax())
            first, gelu, second = moe.experts[expert]
            hidden = gelu(first(x))
            ffn_out = second(hidden)
        p_i = float(probs[expert])
        return RouterStep(
            layer=layer, position=position, word=("[CLS]", *self.tokenizer.pieces(text))[position],
            x=x.numpy(), weights=moe.router.weight.detach().numpy().copy(),
            logits=logits.numpy(), probs=probs.numpy(), expert=expert, hidden=hidden.numpy(),
            ffn_out=ffn_out.numpy(), y=(p_i * ffn_out).numpy(),
            loss_slope=self._loss_slope(ids, is_spam, layer, position) / p_i)

    def clearest(self, text, layer):
        """Returns the position of the word the router is surest about.

        ``[CLS]`` and punctuation are left out: a word makes a clearer example.

        Raises:
            ValueError: If the message has no words.
        """
        ids = self._encode([text])
        n = int((ids != PAD).sum())
        tokens = self.tokenizer.tokenize(text)[:n - 1]
        words = [k + 1 for k, token in enumerate(tokens)
                 if not self.tokenizer.is_punctuation(token)]
        if not words:
            raise ValueError("the message has no words")
        with evaluating(self.model), torch.no_grad():
            surest = self.model(ids).routing[layer].probs[0].max(dim=-1).values
        return max(words, key=lambda k: float(surest[k]))

    def probs(self, text, layer):
        """Returns ``[n, n_experts]``: the router's probabilities for every token of ``text``.

        Raises:
            ValueError: If the layer does not exist.
        """
        if not 0 <= layer < len(self.model.blocks):
            raise ValueError(f"no layer {layer}: the model has {len(self.model.blocks)}")
        ids = self._encode([text])
        n = int((ids != PAD).sum())
        with evaluating(self.model), torch.no_grad():
            return self.model(ids).routing[layer].probs[0, :n].numpy().copy()

    def balance(self, texts):
        """Returns one ``LayerBalance`` per layer, over the real tokens of ``texts``."""
        with evaluating(self.model), torch.no_grad():
            routing = self.model(self._encode(texts)).routing
        return tuple(LayerBalance(load=r.load().numpy(),
                                  mean_probs=r.probs[r.mask].mean(dim=0).numpy(),
                                  aux=float(r.balance_loss()))
                     for r in routing)

    def _encode(self, texts):
        return self.vocab.encode(texts, self.model.config.max_len)

    def _moe_input(self, ids, layer):
        """Runs the model and returns what ``layer``'s MoE layer received."""
        caught = []
        hook = self.model.blocks[layer].moe.register_forward_pre_hook(
            lambda module, args: caught.append(args[0]))
        try:
            self.model(ids)
        finally:
            hook.remove()
        return caught[0]

    def _loss_slope(self, ids, is_spam, layer, position):
        """Returns ``dL/dg``, where ``g`` scales one token's MoE output (``g = 1`` as is).

        Since ``y = p_i · FFN_i(x)``, scaling ``y`` by ``g`` is scaling ``p_i``, so
        ``dL/dp_i = dL/dg / p_i``.
        """
        gate = torch.ones((), requires_grad=True)
        picked = torch.zeros(ids.shape + (1,), dtype=torch.bool)
        picked[0, position] = True

        def scale(module, args, out):
            return (torch.where(picked, out[0] * gate, out[0]), *out[1:])

        hook = self.model.blocks[layer].moe.register_forward_hook(scale)
        try:
            with evaluating(self.model), torch.enable_grad():
                logits = self.model(ids).logits
                loss = F.binary_cross_entropy_with_logits(logits, torch.tensor([float(is_spam)]))
                (slope,) = torch.autograd.grad(loss, gate)
        finally:
            hook.remove()
        return float(slope)
