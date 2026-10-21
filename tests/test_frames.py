import math
import os

import numpy as np
import pytest
from PIL import Image

from moe_spam.training.history import FrameState, History, TokenRow
from moe_spam.viz.canvas import Canvas
from moe_spam.viz.panels import layout_tokens
from moe_spam.viz.style import EXPERT_COLORS, FIGURE_PX, INK, ink_on
from moe_spam.viz.ternary import Ternary
from moe_spam.viz.training.frames import TrainingFrames, balancing_label
from moe_spam.viz.video import Beat, FrameWriter, Video, ease, numbered_titles

GROUPS = ("[CLS]", "words", "numbers")


def frame_files(folder):
    return sorted(f for f in os.listdir(folder) if f.endswith(".jpg"))


def random_state(rng, n=60, n_experts=3, concentration=1.0):
    probs = rng.dirichlet(np.full(n_experts, concentration), size=n)
    shares = rng.dirichlet(np.ones(n_experts), size=len(GROUPS))
    shares[1] = np.nan  # a group with no tokens in this frame
    examples = (
        TokenRow("ham", ("[CLS]", "see", "you", "at", "lunch", "?"), (0, 1, 1, 2, 1, 0), 0.01),
        TokenRow("spam", ("[CLS]", "win", "a", "£", "900", "prize", "!", "call", "09061701461"),
                 (2, 0, 1, 2, 2, 0, 1, 0, 2), 0.993),
    )
    return FrameState(cls_probs=probs, is_spam=rng.random(n) < 0.2, wrong=rng.random(n) < 0.05,
                      examples=examples, group_shares=shares)


def log_step(history, step, n_experts=3):
    history.log(step, {"train BCE": 0.5 / (step + 1), "test BCE": 0.4},
                {"spam missed": 12 - step // 10, "ham flagged": 0},
                [np.full(n_experts, 1 / n_experts), np.eye(n_experts)[0]], {"aux loss": 1.02})


def test_ternary_puts_one_hot_on_corners_and_uniform_in_the_centre():
    assert np.allclose(Ternary.xy(np.eye(3)), Ternary.CORNERS)
    centre = Ternary.xy(np.full((1, 3), 1 / 3))[0]
    assert np.allclose(centre, Ternary.CENTRE)
    assert np.allclose(centre, [0.5, math.sqrt(3) / 6])


def test_ternary_rejects_other_expert_counts():
    with pytest.raises(ValueError, match="3 experts"):
        Ternary.xy(np.full((2, 4), 0.25))


def test_ternary_kite_is_an_experts_territory():
    kite = np.array(Ternary.kite(0))
    assert np.allclose(kite[0], Ternary.CORNERS[0])
    assert np.allclose(kite[2], Ternary.CENTRE)
    assert np.allclose(kite[1], (Ternary.CORNERS[0] + Ternary.CORNERS[1]) / 2)


def test_ternary_view_fits_every_frame_or_shows_the_whole_triangle():
    near = np.array([[0.60, 0.20, 0.20]])
    nearer = np.array([[0.40, 0.30, 0.30]])
    reach = np.abs(Ternary.xy(near) - Ternary.CENTRE).max()
    assert Ternary.view([nearer, near]) == pytest.approx(1.2 * reach)
    assert Ternary.view([np.full((1, 3), 1 / 3)]) == Ternary.MIN_VIEW
    assert Ternary.view([near, np.eye(3)]) is None
    # A view reaching below the triangle's base would leave an empty strip: show it all.
    farther = np.array([[0.65, 0.175, 0.175]])
    assert 1.2 * np.abs(Ternary.xy(farther) - Ternary.CENTRE).max() > Ternary.CENTRE[1]
    assert Ternary.view([farther]) is None


def test_ternary_view_needs_at_least_one_frame():
    with pytest.raises(ValueError, match="no frames"):
        Ternary.view([])


def test_layout_tokens_wraps_at_the_line_width():
    # "hello" fills cols 0-4, a space, "world" 6-10; "again" does not fit in 12 chars.
    assert layout_tokens(["hello", "world", "again"], line_chars=12) == ((0, 0), (0, 6), (1, 0))


def test_layout_tokens_gives_an_overlong_word_its_own_line():
    assert layout_tokens(["a", "x" * 20, "b"], line_chars=8) == ((0, 0), (1, 0), (2, 0))


def test_ink_on_picks_the_more_readable_text_color():
    assert ink_on("#fcfcfb") == INK
    assert ink_on("#0d366b") == "white"
    assert all(ink_on(c) in (INK, "white") for c in EXPERT_COLORS)


def test_history_rows_flatten_losses_errors_extras_and_load():
    history = History()
    log_step(history, 0)
    log_step(history, 10)
    rows = history.rows()
    assert [r["step"] for r in rows] == [0, 10]
    assert rows[1]["spam missed"] == 11
    assert rows[0]["layer 2 load expert 1"] == 1.0
    assert rows[0]["layer 2 load expert 2"] == 0.0
    assert rows[0]["train BCE"] == 0.5
    assert rows[0]["aux loss"] == 1.02


def test_keep_needs_the_step_logged_first():
    history = History()
    with pytest.raises(RuntimeError, match="log"):
        history.keep(0, random_state(np.random.default_rng(0)))
    log_step(history, 0)
    with pytest.raises(RuntimeError, match="log"):
        history.keep(10, random_state(np.random.default_rng(0)))


@pytest.mark.parametrize("concentration", [1.0, 200.0])  # whole triangle, then zoomed in
def test_training_frames_render_one_fixed_size_frame_per_kept_step(tmp_path, concentration):
    history = History()
    rng = np.random.default_rng(0)
    for step in (0, 50, 100):
        log_step(history, step)
        history.keep(step, random_state(rng, concentration=concentration))
    log_step(history, 110)  # logged but not kept: no frame
    folder = tmp_path / "train"
    folder.mkdir()
    (folder / "0009999.jpg").write_bytes(b"old")
    (folder / "notes.log").write_text("keep me")
    frames = TrainingFrames(history, total_steps=100, n_spam=12, n_ham=48, group_labels=GROUPS,
                            aux_weight=0.01)
    assert frames.render(folder) == 3
    assert frame_files(folder) == ["0000000.jpg", "0000050.jpg", "0000100.jpg"]
    assert (folder / "notes.log").exists()  # stale frames go, nothing else
    with Image.open(folder / "0000100.jpg") as img:
        assert img.size == FIGURE_PX


def test_balancing_label_says_whether_the_run_balanced_its_experts():
    assert balancing_label(0.01) == "with load balancing (aux weight 0.01)"
    assert balancing_label(0.5) == "with load balancing (aux weight 0.5)"
    assert balancing_label(0.0) == "without load balancing (aux weight 0)"


def test_training_frames_title_names_the_load_balancing(tmp_path, monkeypatch):
    history = History()
    log_step(history, 0)
    history.keep(0, random_state(np.random.default_rng(0)))
    titles = []
    monkeypatch.setattr(Canvas, "title", lambda self, title, right=None: titles.append(
        (title, right)))
    for aux_weight in (0.01, 0.0):
        TrainingFrames(history, total_steps=10, n_spam=12, n_ham=48, group_labels=GROUPS,
                       aux_weight=aux_weight).render(tmp_path / str(aux_weight))
    assert [t.split("|")[-1].strip() for t, _ in titles] == [
        "with load balancing (aux weight 0.01)", "without load balancing (aux weight 0)"]
    assert {right for _, right in titles} == {"step 0 of 10"}


def test_canvas_saves_on_success_only(tmp_path):
    with Canvas(tmp_path / "ok.jpg") as canvas:
        canvas.title("title", "right")
        canvas.heading(0.3, 0.8, "heading")
        canvas.caption(0.3, 0.7, "a caption " * 30, chars=40)
    with Image.open(tmp_path / "ok.jpg") as img:
        assert img.size == FIGURE_PX
    with pytest.raises(RuntimeError), Canvas(tmp_path / "bad.jpg"):
        raise RuntimeError("drawing failed")
    assert not (tmp_path / "bad.jpg").exists()


def test_beats_ease_and_numbered_titles():
    assert Beat("a", 1.0, 0.5).frames(fps=10) == (10, 5)
    assert Beat("a", 0.0, 0.0).frames(fps=10) == (1, 0)  # always one animated frame
    assert ease(-1) == 0 and ease(2) == 1 and ease(0.5) == 0.5
    beats = (Beat("x", 1, 0), Beat("y", 1, 0), Beat("y2", 1, 0), Beat("z", 1, 0))
    titles = numbered_titles(beats, {"x": "X", "y": "Y", "y2": "Y", "z": "Z"})
    assert titles == {"x": "1. X", "y": "2. Y", "y2": "2. Y", "z": "3. Z"}


def test_frame_writer_holds_by_copying_the_last_frame(tmp_path):
    writer = FrameWriter(tmp_path / "f")
    writer.clear()
    writer.path(0).write_bytes(b"frame")
    assert writer.hold(0, 2) == 3
    assert writer.path(2).read_bytes() == b"frame"


class TwoPartVideo(Video):
    name = "test"

    def segments(self):
        return [("first", (Beat("a", 0.2, 0.1),)), ("second", (Beat("b", 0.1, 0.0),))]

    def draw(self, canvas, context, beat, t):
        canvas.title(f"{context} {beat.step} {t:.2f}")


def test_video_renders_every_segment_with_its_holds(tmp_path):
    n = TwoPartVideo(fps=10).render(tmp_path / "frames")
    assert n == 2 + 1 + 1  # a: 2 frames + 1 held, b: 1 frame
    assert frame_files(tmp_path / "frames") == [f"{i:07d}.jpg" for i in range(4)]
