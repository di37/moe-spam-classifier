import pytest

from moe_spam.viz.motion import locate, locate_in_steps, progress, token_progress

PATH = [(0.0, 0.0, "a"), (10.0, 0.0, "b"), (10.0, 30.0, "c")]


def test_locate_moves_at_one_speed_along_the_path():
    assert locate(PATH, 0)[:2] == (0.0, 0.0)
    assert locate(PATH, 0.25)[:2] == pytest.approx((10.0, 0.0))  # 10 of 40 units
    assert locate(PATH, 1) == (10.0, 30.0, "c")
    assert progress(PATH, 1) == pytest.approx(0.25)


def test_locate_in_steps_gives_every_step_the_same_time_and_eases_into_each_stop():
    assert locate_in_steps(PATH, 0.5) == (10.0, 0.0, "b")  # one of two steps done
    assert locate_in_steps(PATH, 1.0) == (10.0, 30.0, "c")
    x_early, _, piece = locate_in_steps(PATH, 0.05)
    assert x_early < 1.0 and piece == "b"  # slow out of a stop, heading for the next
    ys = [locate_in_steps(PATH, t / 100)[1] for t in range(50, 101)]
    assert ys == sorted(ys)


def test_token_progress_is_slow_then_fast():
    assert token_progress(0.5, 24) == 4
    assert token_progress(1.0, 24) == 24
