"""Counted and measured cost of three feed-forward layers that see the same tokens.

What is counted is the work each layer does and the weights it holds; what is measured is
how long it takes.

- ``dense``: one network, ``d_model → d_ff → d_model`` (this project's dense model).
- ``wide``: one network ``n_experts`` times wider: as many weights as the MoE layer.
- ``moe``: ``n_experts`` networks of width ``d_ff``, and a router that sends each token to one.

Work and weights are counted; time is measured, for many tokens at once (throughput), one
message at a time (latency), and whole trained models.
"""

import json
import statistics
import time
from dataclasses import asdict, dataclass, field

import torch
from torch import nn

from moe_spam.model.bert import BertConfig
from moe_spam.model.modes import evaluating
from moe_spam.model.moe import MoELayer
from moe_spam.records import RecordStore

LAYERS = ("dense", "wide", "moe")
RECORD = "inference_benchmark"  # results/<RECORD>.json
# (d_model, d_ff, n_experts): this project's layer, and one closer to a real model's
SIZES = {"ours": (BertConfig.d_model, BertConfig.d_ff, BertConfig.n_experts),
         "large": (1024, 4096, 8)}


@dataclass(frozen=True)
class LayerCost:
    """What one layer holds and does, counted exactly.

    Attributes:
        name: ``dense``, ``wide`` or ``moe``.
        weights: Numbers held in memory.
        macs_per_token: Multiply-adds to run one token through the layer.
    """

    name: str
    weights: int
    macs_per_token: int

    @classmethod
    def for_sizes(cls, d_model, d_ff, n_experts):
        """Returns the costs of ``(dense, wide, moe)`` at these sizes.

        The MoE layer runs one expert and its router for each token. The router is one
        ``d_model × n_experts`` matrix, without bias.
        """
        ffn_macs = 2 * d_model * d_ff
        router = d_model * n_experts
        return (cls("dense", _ffn_weights(d_model, d_ff), ffn_macs),
                cls("wide", _ffn_weights(d_model, n_experts * d_ff), n_experts * ffn_macs),
                cls("moe", n_experts * _ffn_weights(d_model, d_ff) + router, ffn_macs + router))


@dataclass(frozen=True)
class Benchmark:
    """Times the three layers, and whole models, with fixed settings.

    Attributes:
        sizes: ``{name: (d_model, d_ff, n_experts)}`` to time the layers at.
        batch_tokens: Tokens per batch for throughput.
        reps: Timed runs per throughput measurement.
        warmup: Untimed runs first.
        latency_tokens: Tokens in the single message timed for latency.
        latency_reps: Timed runs per latency measurement (single messages are fast and noisy).
        latency_warmup: Untimed runs first, for latency.
    """

    sizes: dict = field(default_factory=lambda: dict(SIZES))
    batch_tokens: int = 8192
    reps: int = 20
    warmup: int = 3
    latency_tokens: int = 24
    latency_reps: int = 300
    latency_warmup: int = 10

    @staticmethod
    def build_layers(d_model, d_ff, n_experts, seed=0):
        """Returns ``(dense, wide, moe)`` in eval mode, with random weights.

        Speed does not depend on the weights' values.
        """
        torch.manual_seed(seed)
        return (_ffn(d_model, d_ff).eval(), _ffn(d_model, n_experts * d_ff).eval(),
                MoELayer(d_model, n_experts, d_ff).eval())

    @torch.no_grad()
    def time_layers(self, d_model, d_ff, n_experts, n_tokens, reps, warmup, seed=0):
        """Times the three layers on one batch of random tokens.

        Args:
            d_model: The width of a token vector.
            d_ff: The hidden width of the dense layer and of each expert.
            n_experts: The number of experts (the wide layer is this many times wider).
            n_tokens: Tokens in the batch.
            reps: Timed runs; the median is kept.
            warmup: Untimed runs first.
            seed: Seed of the random weights and tokens.

        Returns:
            ``{layer: median milliseconds}`` for one pass of the batch.
        """
        dense, wide, moe = self.build_layers(d_model, d_ff, n_experts, seed)
        x = torch.randn(1, n_tokens, d_model, generator=torch.Generator().manual_seed(seed))
        mask = torch.ones(1, n_tokens, dtype=torch.bool)
        runs = {"dense": lambda: dense(x), "wide": lambda: wide(x), "moe": lambda: moe(x, mask)}
        return {name: _median_ms(run, reps, warmup) for name, run in runs.items()}

    def layers(self):
        """Times the three layers at every size.

        Returns:
            ``{size name: its sizes, the counted costs, milliseconds per 1,000 tokens in
            batches of batch_tokens (throughput), and microseconds for one latency_tokens
            message (latency)}``.
        """
        results = {}
        for name, (d_model, d_ff, n_experts) in self.sizes.items():
            ms = self.time_layers(d_model, d_ff, n_experts, self.batch_tokens, self.reps,
                                  self.warmup)
            one = self.time_layers(d_model, d_ff, n_experts, self.latency_tokens,
                                   self.latency_reps, self.latency_warmup)
            results[name] = {
                "d_model": d_model, "d_ff": d_ff, "n_experts": n_experts,
                "batch_tokens": self.batch_tokens, "latency_tokens": self.latency_tokens,
                "ms_per_1k_tokens": {k: v * 1000 / self.batch_tokens for k, v in ms.items()},
                "latency_us": {k: v * 1000 for k, v in one.items()},
                "costs": {c.name: asdict(c)
                          for c in LayerCost.for_sizes(d_model, d_ff, n_experts)},
            }
        return results

    @torch.no_grad()
    def models(self, models, one_message, batch):
        """Times whole models, in eval mode.

        Args:
            models: ``{name: model}``.
            one_message: ``[1, n]`` ids of one message, without padding.
            batch: ``[m, max_len]`` ids, run as one batch.

        Returns:
            ``{name: {"one_message_ms": ..., "ms_per_1k_messages": ...}}``.
        """
        timings = {}
        for name, model in models.items():
            with evaluating(model):
                one = _median_ms(lambda m=model: m(one_message), self.latency_reps,
                                 self.latency_warmup)
                whole = _median_ms(lambda m=model: m(batch), self.reps, self.warmup)
            timings[name] = {"one_message_ms": one,
                             "ms_per_1k_messages": whole * 1000 / len(batch)}
        return timings

    def record(self, layers, models=None, test_messages=None, store=None):
        """Saves layer and model timings with the machine they ran on.

        Args:
            layers: What ``layers()`` returned.
            models: What ``models()`` returned, if anything.
            test_messages: How many messages the model batch had.
            store: Where to save (``results/`` by default).

        Returns:
            The saved record.
        """
        store = store or RecordStore()
        saved = {"sizes": layers, "models": models, "test_messages": test_messages,
                 "reps": self.reps, "latency_reps": self.latency_reps,
                 "environment": store.environment()}
        store.write_json(RECORD, saved)
        return saved



def saved_timings(store=None):
    """Returns the saved benchmark record, or None if there is none or it cannot be used.

    A record can be used if it reads as JSON and has layer timings at every size in
    ``SIZES``; anything else is measured again.
    """
    store = store or RecordStore()
    try:
        record = store.read_json(RECORD)
    except (OSError, json.JSONDecodeError):
        return None
    sizes = record.get("sizes") if isinstance(record, dict) else None
    return record if isinstance(sizes, dict) and set(SIZES) <= set(sizes) else None


def _ffn_weights(d_model, d_ff):
    return 2 * d_model * d_ff + d_ff + d_model  # two weight matrices and their biases


def _ffn(d_model, d_ff):
    return nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))


def _median_ms(run, reps, warmup):
    for _ in range(warmup):
        run()
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)
    return 1000 * statistics.median(times)
