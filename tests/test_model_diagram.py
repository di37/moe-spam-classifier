import os
import textwrap

import numpy as np
import pytest
import torch
from PIL import Image

from moe_spam.analysis.model_facts import ModelFacts
from moe_spam.analysis.routing import RouterProbe
from moe_spam.analysis.trace import MessageTracer
from moe_spam.data.vocab import Vocab
from moe_spam.model.bert import BertConfig, TinyBert
from moe_spam.viz.model_diagram.script import (
    BEATS,
    SAMPLE_CHARS,
    SCENE_NAMES,
    captions,
    ffn_notes,
    sample,
    savings,
    stack_notes,
    title,
)
from moe_spam.viz.model_diagram.stack import Stack
from moe_spam.viz.model_diagram.video import CAPTION_CHARS, ModelDiagramVideo
from moe_spam.viz.style import FIGURE_PX

TEXTS = ["win a prize call 0800 now", "see you at lunch", "win cash now call 09061701461"]
SPAM = TEXTS[2]


def story(n_experts=3):
    """The video's inputs, from a tiny untrained model."""
    vocab = Vocab.build(TEXTS, min_count=1)
    torch.manual_seed(0)
    model = TinyBert(BertConfig(vocab_size=len(vocab), max_len=32, d_model=16, n_heads=2,
                                n_experts=n_experts, d_ff=32)).eval()
    trace = MessageTracer(model, vocab).trace(SPAM)
    return ModelFacts.of(model), trace, RouterProbe(model, vocab).probs(SPAM, 0)


def test_title_names_the_model_and_its_size():
    facts = story()[0]
    assert title(facts) == f"Tiny BERT + Mixture of Experts ({round(facts.total_params / 1000)}K)"


def test_notes_and_savings_use_the_models_own_numbers():
    facts = story()[0]
    notes = stack_notes(facts)
    assert f"{facts.vocab_size:,} tokens" in notes["tokens"]
    assert f"{facts.max_len} tokens" in notes["context"]
    assert f"of {facts.d_model}" in notes["embed"]
    assert f"({facts.n_heads} heads)" in notes["attn"]
    assert ffn_notes(facts)["first"] == f"{facts.d_model} → {facts.d_ff}"
    heading, bullets = savings(facts)
    assert heading == "Resource savings:"
    assert f"{facts.total_params:,}" in bullets[0]
    assert f"{facts.active_params:,}" in bullets[2]


def test_captions_fit_one_line_and_sample_is_cut():
    facts, trace, _ = story()
    texts = captions(facts, trace)
    assert list(texts) == list(SCENE_NAMES) == [b.step for b in BEATS]
    assert all(len(textwrap.wrap(c, CAPTION_CHARS)) == 1 for c in texts.values())
    long = sample("x" * 100)
    assert len(long) == SAMPLE_CHARS + 2 and long.endswith("…”")
    assert sample("pay $5 now, $10 later") == r"“pay \$5 now, \$10 later”"


def test_path_goes_up_twice_and_ends_at_the_output():
    names = [name for *_, name in Stack.path()]
    assert names[0] == "input" and names[-1] == "p"
    assert names.count("moe") == 2 and names.count("attn") == 2


def test_render_writes_every_frame(tmp_path):
    folder = tmp_path / "frames"
    n = ModelDiagramVideo(*story(), fps=2).render(folder)
    files = sorted(os.listdir(folder))
    assert len(files) == n == sum(sum(beat.frames(fps=2)) for beat in BEATS)
    with Image.open(folder / files[-1]) as img:
        assert img.size == FIGURE_PX
    first, last = (np.asarray(Image.open(folder / f)) for f in (files[0], files[-1]))
    assert not np.array_equal(first, last)


def test_render_needs_three_experts(tmp_path):
    with pytest.raises(ValueError, match="3 experts"):
        ModelDiagramVideo(*story(n_experts=2), fps=2).render(tmp_path / "frames")
