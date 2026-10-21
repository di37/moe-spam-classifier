import os

import pytest
import torch
from PIL import Image

from bert import BertConfig, TinyBert
from frames import FIGURE_PX
from inference_benchmark import LAYERS, layer_costs, time_layers, time_models
from inference_video import beats, compare, render_inference, tokens_done
from vocab import Vocab, encode
from walkthrough import trace_message


def test_layer_costs_count_work_and_weights_exactly():
    dense, wide, moe = layer_costs(d_model=64, d_ff=256, n_experts=3)
    assert [c.name for c in (dense, wide, moe)] == list(LAYERS)
    assert dense.macs_per_token == 2 * 64 * 256 == 32_768
    assert wide.macs_per_token == 3 * dense.macs_per_token
    assert moe.macs_per_token == dense.macs_per_token + 64 * 3  # one expert plus the router
    assert dense.weights == 2 * 64 * 256 + 256 + 64
    assert moe.weights == 3 * dense.weights + 64 * 3
    assert wide.weights == 2 * 64 * 768 + 768 + 64


def test_layer_costs_match_the_real_layers():
    from inference_benchmark import build_layers
    dense, wide, moe = build_layers(d_model=16, d_ff=32, n_experts=4)
    costs = layer_costs(d_model=16, d_ff=32, n_experts=4)
    for layer, cost in zip((dense, wide, moe), costs, strict=True):
        assert sum(p.numel() for p in layer.parameters()) == cost.weights


def test_time_layers_measures_every_layer():
    timings = time_layers(d_model=16, d_ff=32, n_experts=3, n_tokens=64, reps=3)
    assert set(timings) == set(LAYERS)
    assert all(ms > 0 for ms in timings.values())


def test_time_models_times_one_message_and_a_batch():
    vocab = Vocab.build(["win cash now", "see you at lunch"], min_count=1)
    models = {}
    for name, experts in (("dense", 1), ("moe", 3)):
        torch.manual_seed(0)
        models[name] = TinyBert(BertConfig(vocab_size=len(vocab), max_len=8, d_model=16,
                                           n_heads=2, n_experts=experts, d_ff=32)).train()
    one = encode(vocab, ["win cash now"], 8)[:, :4]
    batch = encode(vocab, ["win cash now", "see you at lunch"] * 4, 8)
    timings = time_models(models, one, batch, reps_one=3, reps_batch=3)
    assert set(timings) == {"dense", "moe"}
    for t in timings.values():
        assert t["one_message_ms"] > 0 and t["ms_per_1k_messages"] > 0
    assert all(model.training for model in models.values())  # mode given back


def test_compare_says_faster_or_slower_with_the_factor():
    assert compare(2.0, 1.0) == "2.0× slower"
    assert compare(1.0, 4.0) == "4.0× faster"
    assert compare(1.0, 1.0) == "as fast"


def test_tokens_done_moves_from_none_to_all():
    assert tokens_done(0.0, 24) == 0
    assert tokens_done(1.0, 24) == 24
    steps = [tokens_done(t / 100, 24) for t in range(101)]
    assert steps == sorted(steps)


def test_render_writes_every_frame(tmp_path):
    vocab = Vocab.build(["win cash now call 09061701461", "see you at lunch"], min_count=1)
    torch.manual_seed(0)
    model = TinyBert(BertConfig(vocab_size=len(vocab), max_len=32, d_model=16, n_heads=2,
                                d_ff=32)).eval()
    trace = trace_message(model, vocab, "win cash now call 09061701461")
    measured = {
        "sizes": {"ours": {"d_model": 64, "d_ff": 256, "n_experts": 3, "batch_tokens": 8192,
                           "latency_tokens": 24,
                           "ms_per_1k_tokens": {"dense": 0.1, "wide": 0.4, "moe": 0.42},
                           "latency_us": {"dense": 20.0, "wide": 56.0, "moe": 1400.0}},
                  "large": {"d_model": 1024, "d_ff": 4096, "n_experts": 8, "batch_tokens": 8192,
                            "latency_tokens": 24,
                            "ms_per_1k_tokens": {"dense": 7.0, "wide": 60.0, "moe": 11.0},
                            "latency_us": {"dense": 1200.0, "wide": 4700.0, "moe": 12000.0}}},
        "models": {"dense": {"one_message_ms": 1.4, "ms_per_1k_messages": 85.0},
                   "moe": {"one_message_ms": 3.0, "ms_per_1k_messages": 97.0}},
        "test_messages": 1027,
    }
    folder = tmp_path / "frames"
    n = render_inference(trace, measured, str(folder), fps=2)
    files = sorted(os.listdir(folder))
    assert len(files) == n == sum(f + h for _, f, h in beats(fps=2))
    with Image.open(folder / files[-1]) as img:
        assert img.size == FIGURE_PX


def test_render_needs_three_experts(tmp_path):
    vocab = Vocab.build(["win cash now"], min_count=1)
    model = TinyBert(BertConfig(vocab_size=len(vocab), max_len=8, d_model=16, n_heads=2,
                                n_experts=2, d_ff=32)).eval()
    trace = trace_message(model, vocab, "win cash now")
    with pytest.raises(ValueError, match="3 experts"):
        render_inference(trace, {}, str(tmp_path / "frames"), fps=2)
