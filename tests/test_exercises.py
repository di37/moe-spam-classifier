import os

import pytest
from PIL import Image

from moe_spam.analysis.model_facts import ModelFacts
from moe_spam.training.runs import RunSummary
from moe_spam.viz.exercises.script import BEATS, closest, exercises, three_way
from moe_spam.viz.exercises.video import ExercisesVideo
from moe_spam.viz.style import FIGURE_PX


def run(name, experts=3, aux=0.01, errors=17, layer2=(0.36, 0.33, 0.31)):
    return RunSummary(name, experts=experts, aux_weight=aux, seed=0, accuracy=1 - errors / 1027,
                      errors=errors, spam_f1=0.93, loads=((0.38, 0.33, 0.29), layer2))


def facts(experts):
    expert, router = 33088, 64 * experts
    total = 311809 - 2 * 64 + 2 * router + 2 * (experts - 1) * expert
    return ModelFacts(3241, 64, 64, 4, 2, experts, 256, total, expert, router)


RUNS = {"moe": run("moe"), "moe_noaux": run("moe_noaux", aux=0.0, errors=18,
                                             layer2=(0.18, 0.80, 0.02)),
        "moe_auxweight0.001": run("moe_auxweight0.001", aux=0.001, errors=19,
                                  layer2=(0.18, 0.76, 0.06)),
        "moe_auxweight0.1": run("moe_auxweight0.1", aux=0.1, errors=18),
        "moe_auxweight1.0": run("moe_auxweight1.0", aux=1.0, errors=15),
        "dense": run("dense", experts=1, errors=15, layer2=(1.0,)),
        "moe_experts2": run("moe_experts2", experts=2, errors=17, layer2=(0.51, 0.49)),
        "moe_experts4": run("moe_experts4", experts=4, errors=13, layer2=(0.31, 0.25, 0.25, 0.19)),
        "moe_experts8": run("moe_experts8", experts=8, errors=15, layer2=(0.125,) * 8)}
FACTS = {k: facts(k) for k in (1, 2, 3, 4, 8)}
TIMINGS = {"dense": {"one_message_ms": 1.44}, "moe": {"one_message_ms": 2.95}}


def test_closest_and_three_way_pick_the_right_option():
    assert closest(0.80, (1 / 3, 0.5, 0.8)) == 2
    assert closest(0.36, (1 / 3, 0.5, 0.8)) == 0
    assert three_way(15, 17, absolute=3) == 1      # within 3: about the same
    assert three_way(21, 17, absolute=3) == 2      # more
    assert three_way(2.95, 1.44, relative=0.05) == 2
    assert three_way(1.0, 1.02, relative=0.05) == 1


def test_exercises_answer_from_the_runs():
    by_key = {e.key: e for e in exercises(RUNS, FACTS, TIMINGS)}
    assert list(by_key) == ["collapse", "tiny", "strong", "params", "accuracy", "speed"]
    assert by_key["collapse"].options[by_key["collapse"].answer] == "about 80%"
    assert by_key["tiny"].answer == 2                  # 76%: it still collapses
    assert by_key["strong"].answer == 1                # 15 against 17: about the same
    assert by_key["params"].answer == 0                # a token uses about the same
    assert by_key["accuracy"].answer == 1
    assert by_key["speed"].answer == 2                 # slower
    params = by_key["params"]
    assert params.series[0].values == tuple(FACTS[k].total_params for k in (1, 2, 3, 4, 8))
    assert params.series[1].values == tuple(FACTS[k].active_params for k in (1, 2, 3, 4, 8))
    assert "80%" in by_key["collapse"].takeaway


def test_a_changed_run_changes_the_answer():
    balanced = {**RUNS, "moe_auxweight0.001": run("moe_auxweight0.001", aux=0.001,
                                                    layer2=(0.36, 0.33, 0.31))}
    tiny = next(e for e in exercises(balanced, FACTS, TIMINGS) if e.key == "tiny")
    assert tiny.answer == 0


def test_render_writes_every_frame(tmp_path):
    video = ExercisesVideo(exercises(RUNS, FACTS, TIMINGS)[:2], fps=2)
    n = video.render(tmp_path / "frames")
    files = sorted(os.listdir(tmp_path / "frames"))
    assert len(files) == n == 2 * sum(sum(beat.frames(fps=2)) for beat in BEATS)
    with Image.open(tmp_path / "frames" / files[-1]) as img:
        assert img.size == FIGURE_PX


def test_exercises_need_their_runs():
    with pytest.raises(KeyError):
        exercises({}, FACTS, TIMINGS)
