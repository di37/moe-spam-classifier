import dataclasses
import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from PIL import Image

from moe_spam.analysis.routing import RouterProbe
from moe_spam.analysis.trace import MessageTracer
from moe_spam.data.vocab import Vocab
from moe_spam.model.bert import BertConfig, TinyBert
from moe_spam.viz.how_it_works.flowchart import PIECES, Flowchart
from moe_spam.viz.how_it_works.script import (
    BALANCING,
    BEATS,
    ROUTING,
    SCENE_NAMES,
    SCENE_TITLES,
    captions,
    equation_values,
    legend,
    plain,
    scientific,
)
from moe_spam.viz.how_it_works.video import CAPTION_CHARS, TITLE, HowItWorksVideo
from moe_spam.viz.motion import locate, progress
from moe_spam.viz.style import FIGURE_PX, MATH_FONT

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
    assert locate(path, 0)[:2] == pytest.approx(path[0][:2])
    assert locate(path, 1)[:2] == pytest.approx(path[-1][:2])
    assert locate(path, 1)[2] == "clf"
    marks = [progress(path, k) for k in range(len(path))]
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
    assert softmax[1] == "$= [" + ", ".join(f"{p:.2f}" for p in step.probs) + "]$"
    assert equation_values("argmax", step, balance)[0] == f"$i = {step.expert + 1}$"
    assert f"{balance[1].aux:.3f}" in equation_values("balance", step, balance)[1]
    assert equation_values("build", step, balance) == ()
    gradient = equation_values("gradient", step, balance)[0]
    assert gradient.endswith(f"= {scientific(step.loss_slope)}$")


def test_scientific_writes_a_power_of_ten():
    assert scientific(-7.66e-6) == r"-7.7 \times 10^{-6}"
    assert scientific(0.0012) == r"+1.2 \times 10^{-3}"


def test_plain_keeps_dollar_signs_out_of_the_math():
    assert plain("win $5 or $10") == r"win \$5 or \$10"


@pytest.mark.parametrize("slope", [-7.66e-6, 0.25])  # the trained model's tiny one, and not
def test_captions_fit_in_two_lines_without_splitting_an_equation(slope):
    trace, step, balance, n = story()
    step = dataclasses.replace(step, word="congratulations", loss_slope=slope)
    for caption in captions(trace, step, balance, 1027).values():
        lines = textwrap.wrap(caption, CAPTION_CHARS)
        assert len(lines) <= 2
        assert all(line.count("$") % 2 == 0 for line in lines)


def test_every_equation_is_valid_math():
    trace, step, balance, n = story()
    texts = [*ROUTING, *BALANCING, *SCENE_TITLES.values(),
             *captions(trace, step, balance, n).values(),
             *(line for scene in SCENE_NAMES for line in equation_values(scene, step, balance)),
             *(text for _, text in legend(trace, step))]
    assert all(text.startswith("$") and text.endswith("$") for text in (*ROUTING, *BALANCING))
    figure = plt.figure()
    for text in texts:
        figure.text(0, 0, text, math_fontfamily=MATH_FONT)
    figure.canvas.draw()  # raises on any malformed math
    plt.close(figure)


def test_title_names_the_model():
    assert TITLE == "How Tiny BERT + Mixture of Experts works"


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
