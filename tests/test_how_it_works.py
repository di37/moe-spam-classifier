import os

import numpy as np
import pytest
import torch
from PIL import Image

from moe_spam.analysis.routing import RouterProbe
from moe_spam.analysis.trace import MessageTracer
from moe_spam.data.vocab import Vocab
from moe_spam.model.bert import BertConfig, TinyBert
from moe_spam.viz.how_it_works.flowchart import PIECES, Flowchart
from moe_spam.viz.how_it_works.script import BEATS, SCENE_NAMES, captions, equation_values, legend
from moe_spam.viz.how_it_works.video import HowItWorksVideo
from moe_spam.viz.style import FIGURE_PX

TEXTS = ["win a prize call 0800 now", "see you at lunch", "win cash now call 09061701461"]
SPAM = TEXTS[2]


def story(n_experts=3):
    """The video's inputs, from a tiny untrained model."""
    vocab = Vocab.build(TEXTS, min_count=1)
    torch.manual_seed(0)
    model = TinyBert(BertConfig(vocab_size=len(vocab), max_len=32, d_model=16, n_heads=2,
                                n_experts=n_experts, d_ff=32)).eval()
    probe = RouterProbe(model, vocab)
    trace = MessageTracer(model, vocab).trace(SPAM)
    step = probe.step(SPAM, True, 0, probe.clearest(SPAM, 0))
    return trace, step, probe.balance(TEXTS), len(TEXTS)


def test_path_goes_through_both_layers_and_their_experts():
    path = Flowchart.path((2, 0))
    names = [name for *_, name in path]
    assert names[0] == "text" and names[-1] == "clf"
    assert names.count("router") == 2 and "loop" in names
    first, second = (y for _, y, name in path if name == "experts")
    assert first < second  # expert 3 is drawn below expert 1
    lane = [x for x, _, name in path if name in ("loop", "final") and x > 74.5]
    assert lane  # out of the experts by the side lane, not across the other boxes


def test_locate_moves_from_start_to_end_along_the_path():
    path = Flowchart.path((0, 1))
    assert Flowchart.locate(path, 0)[:2] == pytest.approx(path[0][:2])
    assert Flowchart.locate(path, 1)[:2] == pytest.approx(path[-1][:2])
    assert Flowchart.locate(path, 1)[2] == "clf"
    marks = [Flowchart.progress(path, k) for k in range(len(path))]
    assert marks[0] == 0 and marks[-1] == pytest.approx(1) and marks == sorted(marks)


def test_script_has_a_title_and_caption_for_every_scene():
    trace, step, balance, n = story()
    texts = captions(trace, step, balance, n)
    assert [b.step for b in BEATS] == list(SCENE_NAMES) == list(texts)
    assert f"{step.p_i:.2f}" in texts["argmax"]
    assert ("raises" if step.loss_slope < 0 else "lowers") in texts["gradient"]
    assert len(legend(trace, step)) == len(PIECES)


def test_equation_values_use_the_real_numbers():
    trace, step, balance, _ = story()
    softmax = equation_values("softmax", step, balance)
    assert softmax[1] == "= [" + ", ".join(f"{p:.2f}" for p in step.probs) + "]"
    assert equation_values("argmax", step, balance)[0] == f"= expert {step.expert + 1}"
    assert f"{balance[1].aux:.3f}" in equation_values("balance", step, balance)[1]
    assert equation_values("build", step, balance) == ()
    assert equation_values("gradient", step, balance) == (f"dL/dp_i = {step.loss_slope:+.1e}",)


def test_render_writes_every_frame(tmp_path):
    folder = tmp_path / "frames"
    n = HowItWorksVideo(*story(), fps=2).render(folder)
    files = sorted(os.listdir(folder))
    assert len(files) == n == sum(sum(beat.frames(fps=2)) for beat in BEATS)
    with Image.open(folder / files[-1]) as img:
        assert img.size == FIGURE_PX
    first, last = (np.asarray(Image.open(folder / f)) for f in (files[0], files[-1]))
    assert not np.array_equal(first, last)


def test_render_needs_three_experts(tmp_path):
    with pytest.raises(ValueError, match="3 experts"):
        HowItWorksVideo(*story(n_experts=2), fps=2).render(tmp_path / "frames")
