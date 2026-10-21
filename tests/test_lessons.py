import os
import textwrap

import numpy as np
import pytest
from PIL import Image

from moe_spam.analysis.benchmark import LayerCost
from moe_spam.training.runs import RunHistory, RunSummary
from moe_spam.viz.balancing.script import BEATS as BALANCING_BEATS
from moe_spam.viz.balancing.script import BalancingStory, share_range
from moe_spam.viz.balancing.script import captions as balancing_captions
from moe_spam.viz.balancing.video import CAPTION_CHARS as BALANCING_CHARS
from moe_spam.viz.balancing.video import BalancingVideo
from moe_spam.viz.style import FIGURE_PX
from moe_spam.viz.why_moe.diagram import chip_positions
from moe_spam.viz.why_moe.script import BEATS as WHY_BEATS
from moe_spam.viz.why_moe.script import WhyMoeStory
from moe_spam.viz.why_moe.script import captions as why_captions
from moe_spam.viz.why_moe.video import CAPTION_CHARS as WHY_CHARS
from moe_spam.viz.why_moe.video import WhyMoeVideo


def why_story(n_experts=3):
    return WhyMoeStory(
        d_model=64, d_ff=256, n_experts=n_experts,
        costs={c.name: c for c in LayerCost.for_sizes(64, 256, n_experts)},
        words=("[CLS]", "win", "a", "prize", "now"), experts=(0, 2, 1, 2, 0),
        shares=np.full((3, n_experts), 1 / n_experts), group_labels=("[CLS]", "words", "numbers"),
        results=(("bag-of-words", 16, 0.984), ("dense Tiny BERT", 15, 0.985),
                 ("Tiny BERT + MoE", 17, 0.983)))


def summary(name, aux, layer2, errors):
    return RunSummary(name, 3, aux, 0, 1 - errors / 1027, errors, 0.93,
                      ((0.4, 0.3, 0.3), layer2))


def history(final):
    steps = np.arange(0, 801, 100)
    start = np.full(3, 1 / 3)
    loads = np.array([start + (np.array(final) - start) * k / 8 for k in range(len(steps))])
    return RunHistory(steps=steps, loads=(loads, loads),
                      aux=np.r_[np.nan, np.linspace(1.0, 3 * (np.array(final) ** 2).sum(), 8)])


def balancing_story():
    balanced = tuple(summary(f"moe{s}", 0.01, (0.36, 0.33, 0.31), 13 + s) for s in range(5))
    unbalanced = tuple(summary(f"noaux{s}", 0.0, (0.1, 0.8 - 0.02 * s, 0.1 + 0.02 * s), 15 + s)
                       for s in range(5))
    return BalancingStory(balanced, unbalanced, history((0.36, 0.33, 0.31)),
                          history((0.1, 0.8, 0.1)), 800)


def test_chips_fit_the_block_and_keep_their_order():
    xs = chip_positions(["[CLS]", "congratulations", "you've", "won"] * 4)
    assert xs == sorted(xs) and len(xs) < 16
    assert xs[0] > 0 and xs[-1] < 80


def test_why_moe_captions_use_the_counts_and_fit():
    texts = why_captions(why_story())
    assert "99,456" in texts["tradeoff"] and "32,960" in texts["tradeoff"]
    assert "15, by the dense Tiny BERT" in texts["verdict"]
    assert all(len(textwrap.wrap(c, WHY_CHARS)) <= 2 for c in texts.values())


def test_why_moe_render_writes_every_frame(tmp_path):
    n = WhyMoeVideo(why_story(), fps=2).render(tmp_path / "frames")
    assert n == len(os.listdir(tmp_path / "frames")) == sum(sum(b.frames(fps=2))
                                                           for b in WHY_BEATS)
    with Image.open(tmp_path / "frames" / sorted(os.listdir(tmp_path / "frames"))[-1]) as img:
        assert img.size == FIGURE_PX
    with pytest.raises(ValueError, match="3 experts"):
        WhyMoeVideo(why_story(n_experts=4), fps=2).render(tmp_path / "frames2")


def test_balancing_captions_report_the_runs():
    story = balancing_story()
    texts = balancing_captions(story)
    assert "80%" in texts["collapse"]
    assert share_range(story.unbalanced, 2) == "72–80%"
    assert "13–17 with the loss, 15–19 without" in texts["accuracy"]
    assert all(len(textwrap.wrap(c, BALANCING_CHARS)) <= 2 for c in texts.values())


def test_balancing_render_writes_every_frame(tmp_path):
    n = BalancingVideo(balancing_story(), fps=2).render(tmp_path / "frames")
    assert n == len(os.listdir(tmp_path / "frames")) == sum(sum(b.frames(fps=2))
                                                           for b in BALANCING_BEATS)
