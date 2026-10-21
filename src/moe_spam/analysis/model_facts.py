"""A trained model's sizes and parameter counts: what runs, and what is only stored."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelFacts:
    """The numbers a diagram of the model is annotated with, counted on the model itself.

    Attributes:
        vocab_size: Tokens in the vocabulary.
        max_len: The longest message, in tokens, ``[CLS]`` included.
        d_model: The width of every token vector.
        n_heads: Attention heads per layer.
        n_layers: Encoder blocks.
        n_experts: Experts per MoE layer.
        d_ff: The hidden width of one expert.
        total_params: Every parameter of the model.
        expert_params: The parameters of one expert in one layer.
        router_params: The parameters of one layer's router.
    """

    vocab_size: int
    max_len: int
    d_model: int
    n_heads: int
    n_layers: int
    n_experts: int
    d_ff: int
    total_params: int
    expert_params: int
    router_params: int

    @classmethod
    def of(cls, model):
        """Returns the facts of a ``TinyBert``, counting its parameters."""
        config, moe = model.config, model.blocks[0].moe
        return cls(vocab_size=config.vocab_size, max_len=config.max_len,
                   d_model=config.d_model, n_heads=config.n_heads, n_layers=config.n_layers,
                   n_experts=config.n_experts, d_ff=config.d_ff,
                   total_params=_count(model), expert_params=_count(moe.experts[0]),
                   router_params=_count(moe.router))

    @property
    def moe_params(self):
        """The parameters of one MoE layer: every expert and the router."""
        return self.n_experts * self.expert_params + self.router_params

    @property
    def active_params(self):
        """The parameters one token goes through: the experts it is not sent to are left out."""
        return self.total_params - (self.n_experts - 1) * self.expert_params * self.n_layers


def _count(module):
    return sum(p.numel() for p in module.parameters())
