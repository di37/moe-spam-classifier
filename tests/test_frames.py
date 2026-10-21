import math
import os

import numpy as np
import pytest
from PIL import Image

from frames import FIGURE_PX, FrameState, TokenRow, TrainingRecorder
from panels import (
    CENTRE,
    CORNERS,
    EXPERT_COLORS,
    INK,
    MIN_VIEW,
    ink_on,
    layout_tokens,
    ternary_view,
    ternary_xy,
)

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


def make_recorder(folder, n_experts=3):
    frame_dir = None if folder is None else str(folder)
    return TrainingRecorder(frame_dir, "test run", total_steps=100, n_experts=n_experts,
                            n_spam=12, n_ham=48, group_labels=GROUPS)


def log_step(rec, step, n_experts=3):
    rec.log(step, {"train BCE": 0.5 / (step + 1), "test BCE": 0.4},
            {"spam missed": 12 - step // 10, "ham flagged": 0},
            [np.full(n_experts, 1 / n_experts), np.eye(n_experts)[0]], {"aux loss": 1.02})


def test_ternary_puts_one_hot_on_corners_and_uniform_in_the_centre():
    assert np.allclose(ternary_xy(np.eye(3)), CORNERS)
    centre = ternary_xy(np.full((1, 3), 1 / 3))[0]
    assert np.allclose(centre, CORNERS.mean(axis=0))
    assert np.allclose(centre, [0.5, math.sqrt(3) / 6])


def test_ternary_rejects_other_expert_counts():
    with pytest.raises(ValueError, match="3 experts"):
        ternary_xy(np.full((2, 4), 0.25))


def test_layout_tokens_wraps_at_the_line_width():
    # "hello" fills cols 0-4, a space, "world" 6-10; "again" does not fit in 12 chars.
    assert layout_tokens(["hello", "world", "again"], line_chars=12) == ((0, 0), (0, 6), (1, 0))


def test_layout_tokens_gives_an_overlong_word_its_own_line():
    assert layout_tokens(["a", "x" * 20, "b"], line_chars=8) == ((0, 0), (1, 0), (2, 0))


def test_ink_on_picks_the_more_readable_text_color():
    assert ink_on("#fcfcfb") == INK
    assert ink_on("#0d366b") == "white"
    assert all(ink_on(c) in (INK, "white") for c in EXPERT_COLORS)


@pytest.mark.parametrize("concentration", [1.0, 200.0])  # whole triangle, then zoomed in
def test_recorder_renders_one_fixed_size_frame_per_kept_step(tmp_path, concentration):
    rec = make_recorder(tmp_path / "train")
    rng = np.random.default_rng(0)
    for step in (0, 50, 100):
        log_step(rec, step)
        rec.keep(step, random_state(rng, concentration=concentration))
    log_step(rec, 110)  # logged but not kept: no frame
    assert frame_files(tmp_path / "train") == []
    assert rec.render() == 3
    assert frame_files(tmp_path / "train") == ["0000000.jpg", "0000050.jpg", "0000100.jpg"]
    with Image.open(tmp_path / "train" / "0000100.jpg") as img:
        assert img.size == FIGURE_PX


def test_ternary_view_fits_every_frame_or_shows_the_whole_triangle():
    near = np.array([[0.60, 0.20, 0.20]])
    nearer = np.array([[0.40, 0.30, 0.30]])
    reach = np.abs(ternary_xy(near) - CENTRE).max()
    assert ternary_view([nearer, near]) == pytest.approx(1.2 * reach)
    assert ternary_view([np.full((1, 3), 1 / 3)]) == MIN_VIEW
    assert ternary_view([near, np.eye(3)]) is None
    # A view reaching below the triangle's base would leave an empty strip: show it all.
    farther = np.array([[0.65, 0.175, 0.175]])
    assert 1.2 * np.abs(ternary_xy(farther) - CENTRE).max() > CENTRE[1]
    assert ternary_view([farther]) is None


def test_recorder_clears_stale_frames_only(tmp_path):
    folder = tmp_path / "train"
    folder.mkdir()
    (folder / "0009999.jpg").write_bytes(b"old")
    (folder / "notes.log").write_text("keep me")
    make_recorder(folder)
    assert frame_files(folder) == []
    assert (folder / "notes.log").exists()


def test_recorder_history_rows_flatten_losses_errors_and_load(tmp_path):
    rec = make_recorder(tmp_path / "train")
    log_step(rec, 0)
    log_step(rec, 10)
    rows = rec.history_rows()
    assert [r["step"] for r in rows] == [0, 10]
    assert rows[1]["spam missed"] == 11
    assert rows[0]["layer 2 load expert 1"] == 1.0
    assert rows[0]["layer 2 load expert 2"] == 0.0
    assert rows[0]["train BCE"] == 0.5
    assert rows[0]["aux loss"] == 1.02


def test_history_only_recorder_logs_but_does_not_draw(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rec = make_recorder(None)
    log_step(rec, 0)
    assert len(rec.history_rows()) == 1
    assert os.listdir(tmp_path) == []
    with pytest.raises(RuntimeError, match="history only"):
        rec.keep(0, random_state(np.random.default_rng(0)))


def test_keep_needs_the_step_logged_first(tmp_path):
    rec = make_recorder(tmp_path / "train")
    with pytest.raises(RuntimeError, match="log"):
        rec.keep(0, random_state(np.random.default_rng(0)))
    log_step(rec, 0)
    with pytest.raises(RuntimeError, match="log"):
        rec.keep(10, random_state(np.random.default_rng(0)))
