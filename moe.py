"""
A Mixture-of-Experts feed-forward layer with top-1 routing, as in the Switch Transformer
(Fedus et al., 2021), small enough to read in one sitting.

  Routing              what the router decided for one batch: its probabilities, the expert
                       each token went to, and which tokens are real
  load_balancing_loss  the Switch auxiliary loss, which pushes the router to spread tokens
                       evenly over the experts
  MoELayer             a router in front of n_experts small FFNs; each real token is processed
                       by exactly one of them

Each real token's output is gate * expert(x), where expert is the router's argmax and gate
is its probability. The argmax itself has no gradient, so the gate is how the task loss
reaches the router. Padding never reaches an expert and never counts towards the load.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def _expert_share(expert, mask, n_experts):
    """The share of real tokens whose top-1 is each expert: [n_experts], summing to 1."""
    if not mask.any():
        raise ValueError("the mask has no real tokens")
    counts = torch.bincount(expert[mask], minlength=n_experts)
    return counts.float() / counts.sum()


@dataclass(frozen=True)
class Routing:
    probs: torch.Tensor   # [B, L, E] router softmax over experts
    expert: torch.Tensor  # [B, L] long, the top-1 expert that processed each token
    mask: torch.Tensor    # [B, L] bool, True = real token, False = padding

    def load(self):
        """[E] share of real tokens sent to each expert (sums to 1)."""
        return _expert_share(self.expert, self.mask, self.probs.shape[-1])


def load_balancing_loss(probs, expert, mask, n_experts):
    """
    The Switch Transformer auxiliary loss, over the real tokens only:

        n_experts * sum_i f_i * P_i

    f_i is the share of tokens whose top-1 is expert i and P_i the mean router probability
    for expert i. It is 1 when tokens and probability are spread evenly, and n_experts when
    every token goes to one expert with certainty. f_i comes from an argmax and carries no
    gradient, so the router learns through P_i: lowering the probability of busy experts.
    """
    f = _expert_share(expert, mask, n_experts)
    p = probs[mask].mean(dim=0)
    return n_experts * (f * p).sum()


class MoELayer(nn.Module):
    """
    The feed-forward block of a Transformer layer, split into n_experts FFNs with a router
    in front. There is no capacity limit, so no token is ever dropped.
    """

    def __init__(self, d_model, n_experts, d_ff, router_std=0.02):
        super().__init__()
        if n_experts < 1:
            raise ValueError(f"n_experts must be at least 1, got {n_experts}")
        self.n_experts = n_experts
        self.router = nn.Linear(d_model, n_experts, bias=False)
        # Small random weights, not zeros: a zero router gives every expert the same
        # probability, and argmax breaks that tie the same way for every token.
        nn.init.normal_(self.router.weight, std=router_std)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
            for _ in range(n_experts)
        )

    def forward(self, x, mask):
        """x [B, L, d_model], mask [B, L] bool -> (out [B, L, d_model], aux loss, Routing)."""
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

        aux = load_balancing_loss(probs, expert, mask, self.n_experts)
        return flat_out.reshape(x.shape), aux, Routing(probs=probs, expert=expert, mask=mask)
