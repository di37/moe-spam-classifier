"""A mixture-of-experts feed-forward layer with top-1 routing, as in the Switch Transformer.

Fedus, Zoph and Shazeer, "Switch Transformers", 2021. Each real token's output is
``gate * expert(x)``, where ``expert`` is the router's argmax and ``gate`` its probability.
The argmax has no gradient, so the gate is how the task loss reaches the router. Padding
never reaches an expert and never counts towards the load.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class Routing:
    """What a router decided for one batch.

    Attributes:
        probs: ``[B, L, E]`` the router's softmax over the experts.
        expert: ``[B, L]`` long, the top-1 expert that processed each token.
        mask: ``[B, L]`` bool, True for real tokens and False for padding.
    """

    probs: torch.Tensor
    expert: torch.Tensor
    mask: torch.Tensor

    @property
    def n_experts(self):
        """The number of experts the router chooses between."""
        return self.probs.shape[-1]

    def load(self):
        """Returns ``[E]``: the share of real tokens sent to each expert (sums to 1).

        Raises:
            ValueError: If the batch has no real tokens.
        """
        if not self.mask.any():
            raise ValueError("the mask has no real tokens")
        counts = torch.bincount(self.expert[self.mask], minlength=self.n_experts)
        return counts.float() / counts.sum()

    def balance_loss(self):
        """Returns the Switch Transformer load-balancing loss, over the real tokens only.

        The loss is ``n_experts * sum_i f_i * P_i``, where ``f_i`` is the share of tokens
        whose top-1 is expert ``i`` and ``P_i`` the mean router probability for expert ``i``.
        It is 1 when tokens and probability are spread evenly, and ``n_experts`` when every
        token goes to one expert with certainty. ``f_i`` comes from an argmax and carries no
        gradient, so the router learns through ``P_i``, by lowering busy experts' probability.
        """
        return self.n_experts * (self.load() * self.probs[self.mask].mean(dim=0)).sum()


class MoELayer(nn.Module):
    """A Transformer feed-forward block split into experts, with a router in front.

    There is no capacity limit, so no token is ever dropped.

    Attributes:
        n_experts: The number of experts.
        router: ``Linear(d_model, n_experts)`` without bias: one score per expert.
        experts: The expert networks, each ``Linear → GELU → Linear``.
    """

    def __init__(self, d_model, n_experts, d_ff, router_std=0.02):
        """Builds the router and the experts.

        Args:
            d_model: The width of the tokens.
            n_experts: The number of experts.
            d_ff: The hidden width of each expert.
            router_std: The standard deviation of the router's initial weights. Not zero:
                a zero router ties every expert, and argmax breaks the tie the same way for
                every token.

        Raises:
            ValueError: If ``n_experts`` is below 1.
        """
        super().__init__()
        if n_experts < 1:
            raise ValueError(f"n_experts must be at least 1, got {n_experts}")
        self.n_experts = n_experts
        self.router = nn.Linear(d_model, n_experts, bias=False)
        nn.init.normal_(self.router.weight, std=router_std)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
            for _ in range(n_experts)
        )

    def forward(self, x, mask):
        """Routes every real token to one expert.

        Args:
            x: ``[B, L, d_model]`` tokens.
            mask: ``[B, L]`` bool, True for real tokens.

        Returns:
            ``(out, aux_loss, routing)``: the ``[B, L, d_model]`` output (zero at pads), the
            load-balancing loss and the ``Routing``.

        Raises:
            ValueError: If ``mask`` is not a bool tensor of shape ``[B, L]``.
        """
        if mask.dtype != torch.bool or mask.shape != x.shape[:2]:
            raise ValueError(f"mask must be a bool tensor of shape {tuple(x.shape[:2])}, "
                             f"got {mask.dtype} of shape {tuple(mask.shape)}")
        # Zero the pads first, so that nothing, not even their routing, depends on their values.
        x = x.masked_fill(~mask[..., None], 0.0)
        probs = F.softmax(self.router(x), dim=-1)
        gate, expert = probs.max(dim=-1)

        # Work on a flat list of tokens: each expert runs only on the real tokens it won,
        # and its gated output is added back at those rows. Pads stay zero.
        flat_x, flat_gate = x.reshape(-1, x.shape[-1]), gate.reshape(-1)
        flat_out = torch.zeros_like(flat_x)
        for i, ffn in enumerate(self.experts):
            rows = torch.nonzero((mask & (expert == i)).reshape(-1)).squeeze(1)
            gated = flat_gate[rows, None] * ffn(flat_x[rows])
            # Under autocast the expert may answer in lower precision than x; match x's dtype.
            flat_out = flat_out.index_add(0, rows, gated.to(flat_out.dtype))

        routing = Routing(probs=probs, expert=expert, mask=mask)
        return flat_out.reshape(x.shape), routing.balance_loss(), routing
