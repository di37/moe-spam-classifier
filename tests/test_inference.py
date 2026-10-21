import os

import pytest
import torch
from PIL import Image

from moe_spam.analysis.benchmark import LAYERS, RECORD, Benchmark, LayerCost
from moe_spam.analysis.trace import MessageTracer
from moe_spam.data.vocab import Vocab
from moe_spam.model.bert import BertConfig, TinyBert
from moe_spam.records import RecordStore
from moe_spam.viz.inference.lanes import amount
from moe_spam.viz.inference.script import BEATS, captions, compare
from moe_spam.viz.inference.video import InferenceVideo, tokens_done
from moe_spam.viz.style import FIGURE_PX


def tiny_trace(n_experts=3):
    vocab = Vocab.build(["win cash now call 09061701461", "see you at lunch"], min_count=1)
    torch.manual_seed(0)
    model = TinyBert(BertConfig(vocab_size=len(vocab), max_len=32, d_model=16, n_heads=2,
                                n_experts=n_experts, d_ff=32)).eval()
    return MessageTracer(model, vocab).trace("win cash now call 09061701461")


def test_layer_costs_count_work_and_weights_exactly():
    dense, wide, moe = LayerCost.for_sizes(d_model=64, d_ff=256, n_experts=3)
    assert [c.name for c in (dense, wide, moe)] == list(LAYERS)
    assert dense.macs_per_token == 2 * 64 * 256 == 32_768
    assert wide.macs_per_token == 3 * dense.macs_per_token
    assert moe.macs_per_token == dense.macs_per_token + 64 * 3  # one expert plus the router
    assert dense.weights == 2 * 64 * 256 + 256 + 64
    assert moe.weights == 3 * dense.weights + 64 * 3
    assert wide.weights == 2 * 64 * 768 + 768 + 64


def test_layer_costs_match_the_real_layers():
    layers = Benchmark.build_layers(d_model=16, d_ff=32, n_experts=4)
    costs = LayerCost.for_sizes(d_model=16, d_ff=32, n_experts=4)
    for layer, cost in zip(layers, costs, strict=True):
        assert sum(p.numel() for p in layer.parameters()) == cost.weights


def test_time_layers_measures_every_layer(fast_benchmark):
    timings = fast_benchmark.time_layers(d_model=16, d_ff=32, n_experts=3, n_tokens=64,
                                         reps=3, warmup=1)
    assert set(timings) == set(LAYERS)
    assert all(ms > 0 for ms in timings.values())


def test_layers_and_record_save_throughput_latency_and_costs(tmp_path, fast_benchmark):
    layers = fast_benchmark.layers()
    assert set(layers) == {"ours", "large"}
    tiny = layers["ours"]
    assert (tiny["d_model"], tiny["batch_tokens"], tiny["latency_tokens"]) == (16, 64, 4)
    assert set(tiny["ms_per_1k_tokens"]) == set(tiny["latency_us"]) == set(LAYERS)
    assert tiny["costs"]["moe"]["macs_per_token"] == 2 * 16 * 32 + 16 * 3
    store = RecordStore(tmp_path)
    saved = fast_benchmark.record(layers, test_messages=8, store=store)
    assert store.read_json(RECORD)["sizes"] == saved["sizes"]
    assert saved["models"] is None and saved["test_messages"] == 8


def test_models_times_one_message_and_a_batch(fast_benchmark):
    vocab = Vocab.build(["win cash now", "see you at lunch"], min_count=1)
    models = {}
    for name, experts in (("dense", 1), ("moe", 3)):
        torch.manual_seed(0)
        models[name] = TinyBert(BertConfig(vocab_size=len(vocab), max_len=8, d_model=16,
                                           n_heads=2, n_experts=experts, d_ff=32)).train()
    one = vocab.encode(["win cash now"], 8)[:, :4]
    batch = vocab.encode(["win cash now", "see you at lunch"] * 4, 8)
    timings = fast_benchmark.models(models, one, batch)
    assert set(timings) == {"dense", "moe"}
    for t in timings.values():
        assert t["one_message_ms"] > 0 and t["ms_per_1k_messages"] > 0
    assert all(model.training for model in models.values())  # mode given back


def test_compare_says_faster_or_slower_with_the_factor():
    assert compare(2.0, 1.0) == "2.0× slower"
    assert compare(1.0, 4.0) == "4.0× faster"
    assert compare(1.0, 1.0) == "as fast"


def test_amount_shortens_large_numbers():
    assert (amount(1_234_567), amount(32_768), amount(192)) == ("1.23 M", "32.8 k", "192")


def test_captions_without_timings_cover_the_counted_scenes_only():
    trace = tiny_trace()
    costs = {c.name: c for c in LayerCost.for_sizes(trace.d_model, 32, 3)}
    counted = captions(trace, costs, None, 3)
    assert {"setup", "stream", "totals"} <= set(counted)
    assert "models" not in counted


def test_tokens_done_moves_from_none_to_all():
    assert tokens_done(0.0, 24) == 0
    assert tokens_done(1.0, 24) == 24
    steps = [tokens_done(t / 100, 24) for t in range(101)]
    assert steps == sorted(steps)


def test_render_writes_every_frame(tmp_path, measured):
    folder = tmp_path / "frames"
    n = InferenceVideo(tiny_trace(), measured, fps=2).render(folder)
    files = sorted(os.listdir(folder))
    assert len(files) == n == sum(sum(beat.frames(fps=2)) for beat in BEATS)
    with Image.open(folder / files[-1]) as img:
        assert img.size == FIGURE_PX


def test_render_needs_three_experts(tmp_path, measured):
    with pytest.raises(ValueError, match="3 experts"):
        InferenceVideo(tiny_trace(n_experts=2), measured, fps=2).render(tmp_path / "frames")
