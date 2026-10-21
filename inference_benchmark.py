"""
How much work a feed-forward layer does per token, and how long it takes, for three layers
that see the same tokens:

  dense  one feed-forward network, d_model -> d_ff -> d_model (this project's dense model)
  wide   one network n_experts times wider: as many weights as the MoE layer
  moe    n_experts networks of width d_ff, and a router that sends each token to one of them

    python inference_benchmark.py

  LayerCost     weights held and multiply-adds per token (counted, not measured)
  layer_costs   those costs for the three layers at given sizes
  build_layers  the three layers themselves, with random weights
  time_layers   the median time each takes to run on the same tokens
  benchmark     both, at this project's size and at a large size: many tokens at once
                (throughput) and one 24-token message at a time (latency)
  time_models   the same two timings for whole trained classifiers

Writes results/inference_benchmark.json, which inference_video.py draws.
"""
import os
import statistics
import time
from dataclasses import asdict, dataclass

import torch
from torch import nn

from moe import MoELayer
from run_log import RESULTS_DIR, environment_info, write_json
from walkthrough import evaluating

LAYERS = ("dense", "wide", "moe")
# (d_model, d_ff, n_experts): this project's layer, and one closer to a real model's
SIZES = {"ours": (64, 256, 3), "large": (1024, 4096, 8)}
N_TOKENS = 8192       # tokens per batch for throughput
REPS = 20
WARMUP = 3
LATENCY_TOKENS = 24   # one SMS, like the example in the videos
LATENCY_REPS = 300    # single messages are fast and noisy: time many
LATENCY_WARMUP = 10
RECORD = os.path.join(RESULTS_DIR, "inference_benchmark.json")


@dataclass(frozen=True)
class LayerCost:
    name: str
    weights: int         # numbers held in memory
    macs_per_token: int  # multiply-adds to run one token through the layer


def _ffn_weights(d_model, d_ff):
    return 2 * d_model * d_ff + d_ff + d_model  # two weight matrices and their biases


def layer_costs(d_model, d_ff, n_experts):
    """(dense, wide, moe). The MoE layer runs one expert and its router for each token."""
    ffn_macs = 2 * d_model * d_ff
    router = d_model * n_experts  # the router is one d_model x n_experts matrix, no bias
    return (LayerCost("dense", _ffn_weights(d_model, d_ff), ffn_macs),
            LayerCost("wide", _ffn_weights(d_model, n_experts * d_ff), n_experts * ffn_macs),
            LayerCost("moe", n_experts * _ffn_weights(d_model, d_ff) + router,
                      ffn_macs + router))


def _ffn(d_model, d_ff):
    return nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))


def build_layers(d_model, d_ff, n_experts, seed=0):
    """(dense, wide, moe) in eval mode, with random weights (speed does not depend on them)."""
    torch.manual_seed(seed)
    return (_ffn(d_model, d_ff).eval(), _ffn(d_model, n_experts * d_ff).eval(),
            MoELayer(d_model, n_experts, d_ff).eval())


def _median_ms(run, reps, warmup=WARMUP):
    for _ in range(warmup):
        run()
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)
    return 1000 * statistics.median(times)


@torch.no_grad()
def time_layers(d_model, d_ff, n_experts, n_tokens=N_TOKENS, reps=REPS, seed=0,
                warmup=WARMUP):
    """{layer: median milliseconds to run all n_tokens through it once}, as one batch."""
    dense, wide, moe = build_layers(d_model, d_ff, n_experts, seed)
    x = torch.randn(1, n_tokens, d_model, generator=torch.Generator().manual_seed(seed))
    mask = torch.ones(1, n_tokens, dtype=torch.bool)
    runs = {"dense": lambda: dense(x), "wide": lambda: wide(x), "moe": lambda: moe(x, mask)}
    return {name: _median_ms(run, reps, warmup) for name, run in runs.items()}


@torch.no_grad()
def time_models(models, one_message, batch, reps_one=LATENCY_REPS, reps_batch=REPS):
    """
    {name: {"one_message_ms", "ms_per_1k_messages"}} for whole models, in eval mode.
    one_message: ids [1, n] with no padding; batch: ids [m, max_len], run as one batch.
    """
    timings = {}
    for name, model in models.items():
        with evaluating(model):
            one = _median_ms(lambda m=model: m(one_message), reps_one, LATENCY_WARMUP)
            whole = _median_ms(lambda m=model: m(batch), reps_batch)
        timings[name] = {"one_message_ms": one, "ms_per_1k_messages": whole * 1000 / len(batch)}
    return timings


def benchmark(sizes=SIZES, n_tokens=N_TOKENS, reps=REPS):
    """
    {size name: its sizes, the counted costs, the measured ms per 1,000 tokens in batches of
    n_tokens (throughput), and the measured microseconds for one LATENCY_TOKENS message}.
    """
    results = {}
    for name, (d_model, d_ff, n_experts) in sizes.items():
        ms = time_layers(d_model, d_ff, n_experts, n_tokens, reps)
        one = time_layers(d_model, d_ff, n_experts, LATENCY_TOKENS, LATENCY_REPS,
                          warmup=LATENCY_WARMUP)
        results[name] = {
            "d_model": d_model, "d_ff": d_ff, "n_experts": n_experts,
            "batch_tokens": n_tokens, "latency_tokens": LATENCY_TOKENS,
            "ms_per_1k_tokens": {k: v * 1000 / n_tokens for k, v in ms.items()},
            "latency_us": {k: v * 1000 for k, v in one.items()},
            "costs": {c.name: asdict(c) for c in layer_costs(d_model, d_ff, n_experts)},
        }
    return results


def record(results, models=None, test_messages=None):
    """Save benchmark() and time_models() results with the machine they ran on; returns them."""
    saved = {"sizes": results, "models": models, "test_messages": test_messages,
             "reps": REPS, "latency_reps": LATENCY_REPS, "environment": environment_info()}
    write_json(RECORD, saved)
    return saved


def main():
    results = benchmark()
    record(results)
    for name, r in results.items():
        many = "  ".join(f"{k} {v:.3f} ms" for k, v in r["ms_per_1k_tokens"].items())
        one = "  ".join(f"{k} {v:,.0f} us" for k, v in r["latency_us"].items())
        print(f"{name}: d_model {r['d_model']}, d_ff {r['d_ff']}, {r['n_experts']} experts\n"
              f"  many tokens: {many} per 1,000 tokens\n"
              f"  one message ({r['latency_tokens']} tokens): {one}")


if __name__ == "__main__":
    main()
