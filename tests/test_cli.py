import os
import runpy
import shutil
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

import moe_spam.cli.training_commands as training_commands
import moe_spam.cli.video_commands as video_commands
from moe_spam.analysis.benchmark import RECORD
from moe_spam.cli.main import COMMANDS, build_parser, main, run
from moe_spam.data.vocab import Vocab
from moe_spam.errors import MoeSpamError
from moe_spam.model.checkpoint import Checkpoint
from moe_spam.records import RecordStore

MOE = "checkpoints/moe_steps2.pt"


def train_both():
    """Trains a 2-step MoE and dense model on the synthetic data, as the videos need them."""
    main(["train", "--steps", "2", "--no-video"])
    main(["train", "--steps", "2", "--experts", "1", "--no-video"])
    shutil.copy("checkpoints/dense_steps2.pt", "checkpoints/dense.pt")


def test_every_command_has_a_subparser():
    _, commands = build_parser()
    assert list(commands) == [c.name for c in COMMANDS]
    assert set(commands) == {"train", "baselines", "benchmark", "architecture-video",
                             "inference-video", "how-it-works-video", "encode-video"}


def test_the_command_line_exits_with_status_zero(tiny_project):
    tmp = tiny_project / "frames"
    tmp.mkdir()
    Image.fromarray(np.zeros((45, 81, 3), dtype=np.uint8)).save(tmp / "0.jpg")
    script = os.path.join(os.path.dirname(sys.executable), "moe-spam")
    done = subprocess.run([script, "encode-video", str(tmp), "out"], capture_output=True,
                          text=True)
    assert done.returncode == 0, done.stderr
    assert main(["train", "--steps", "2", "--no-video"]) == 0


def test_python_dash_m_runs_the_cli(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["moe-spam", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("moe_spam", run_name="__main__")
    assert exit_info.value.code == 0


@pytest.mark.parametrize("bad", [["train", "--batch-size", "0"],
                                 ["architecture-video", "--fps", "0"],
                                 ["encode-video", "f", "o", "--gif-every", "0"],
                                 ["encode-video", "f", "o", "--fps", "0"],
                                 []])
def test_bad_command_lines_exit_with_usage(bad):
    with pytest.raises(SystemExit):
        main(bad)


def test_baselines_prints_every_run(tiny_project, capsys):
    main(["train", "--steps", "2", "--no-video"])
    main(["baselines"])
    table = capsys.readouterr().out
    assert "bag-of-words logreg" in table and "moe_steps2" in table
    assert "steps" in RecordStore().read_json("baselines")


def test_benchmark_prints_and_records_the_timings(tiny_project, monkeypatch, capsys,
                                                  fast_benchmark):
    monkeypatch.setattr(training_commands, "Benchmark", lambda: fast_benchmark)
    main(["benchmark"])
    out = capsys.readouterr().out
    assert "ours: d_model 16, d_ff 32, 3 experts" in out and "one message (4 tokens)" in out
    assert set(RecordStore().read_json(RECORD)["sizes"]) == {"ours", "large"}


def test_architecture_video_draws_both_messages(tiny_project, capsys):
    train_both()
    main(["architecture-video", MOE, "--fps", "1", "--no-video"])
    out = capsys.readouterr().out
    assert "spam: " in out and "ham: " in out
    assert len(os.listdir("frames/architecture")) > 0
    assert not os.path.exists("media/architecture.mp4")


def test_architecture_video_needs_two_layers_of_three_experts(tiny_project, capsys):
    main(["train", "--steps", "2", "--experts", "2", "--no-video"])
    with pytest.raises(MoeSpamError, match="n_experts"):
        run(["architecture-video", "checkpoints/moe_experts2_steps2.pt"])
    assert main(["architecture-video", "checkpoints/moe_experts2_steps2.pt"]) == 1
    assert "moe-spam: error: " in capsys.readouterr().err  # a message, not a traceback


def test_how_it_works_video_draws_the_diagram_and_the_equations(tiny_project, capsys):
    main(["train", "--steps", "2", "--no-video"])
    main(["how-it-works-video", MOE, "--fps", "1", "--no-video"])
    out = capsys.readouterr().out
    assert "message: " in out and "token: " in out
    assert len(os.listdir("frames/how_it_works")) > 0


def test_how_it_works_video_needs_two_layers_of_three_experts(tiny_project):
    main(["train", "--steps", "2", "--experts", "2", "--no-video"])
    with pytest.raises(MoeSpamError, match="n_experts"):
        run(["how-it-works-video", "checkpoints/moe_experts2_steps2.pt"])


def test_inference_video_reuses_saved_timings(tiny_project, capsys, measured):
    train_both()
    RecordStore().write_json(RECORD, measured)
    main(["inference-video", MOE, "--fps", "1", "--no-video", "--reuse"])
    assert "message: " in capsys.readouterr().out
    assert len(os.listdir("frames/inference")) > 0
    assert RecordStore().read_json(RECORD) == measured  # not measured again


def test_inference_video_measures_without_saved_timings(tiny_project, monkeypatch, capsys,
                                                        fast_benchmark):
    train_both()
    monkeypatch.setattr(video_commands, "Benchmark", lambda: fast_benchmark)
    main(["inference-video", MOE, "--fps", "1", "--no-video", "--reuse"])
    assert "measuring" in capsys.readouterr().out
    saved = RecordStore().read_json(RECORD)
    assert set(saved["models"]) == {"dense", "moe"}
    assert saved["test_messages"] > 0


def test_inference_video_needs_models_with_one_vocabulary(tiny_project):
    train_both()
    dense = Checkpoint.load("checkpoints/dense.pt")
    other = Vocab((*dense.vocab.words[:-1], "zzz"))
    Checkpoint(dense.model, other).save("checkpoints/dense.pt")
    with pytest.raises(MoeSpamError, match="different vocabularies"):
        run(["inference-video", MOE, "--no-video"])


def test_inference_video_explains_a_missing_or_wrong_dense_model(tiny_project):
    main(["train", "--steps", "2", "--no-video"])
    with pytest.raises(MoeSpamError, match="train --experts 1 --no-video"):
        run(["inference-video", MOE, "--no-video"])
    shutil.copy(MOE, "checkpoints/dense.pt")  # an MoE model where the dense one should be
    with pytest.raises(MoeSpamError, match="n_experts"):
        run(["inference-video", MOE, "--no-video"])


@pytest.mark.parametrize("saved", ["{not json", '{"sizes": {}, "models": {"moe": {}}}'])
def test_inference_video_measures_again_when_saved_timings_are_unusable(
        tiny_project, monkeypatch, fast_benchmark, saved):
    train_both()
    monkeypatch.setattr(video_commands, "Benchmark", lambda: fast_benchmark)
    RecordStore().path(RECORD, ".json").parent.mkdir(exist_ok=True)
    RecordStore().path(RECORD, ".json").write_text(saved)
    main(["inference-video", MOE, "--fps", "1", "--no-video", "--reuse"])
    assert set(RecordStore().read_json(RECORD)["sizes"]) == {"ours", "large"}


def test_benchmark_keeps_the_saved_model_timings(tiny_project, monkeypatch, fast_benchmark,
                                                 measured):
    RecordStore().write_json(RECORD, measured)
    monkeypatch.setattr(training_commands, "Benchmark", lambda: fast_benchmark)
    main(["benchmark"])
    saved = RecordStore().read_json(RECORD)
    assert saved["models"] == measured["models"]
    assert saved["test_messages"] == measured["test_messages"]
    assert saved["sizes"]["ours"]["d_model"] == 16  # the layers were measured again


def test_encode_video_reports_a_missing_folder(tmp_path, capsys):
    assert main(["encode-video", str(tmp_path / "missing"), str(tmp_path / "out")]) == 1
    assert "not found" in capsys.readouterr().err


def test_a_bad_number_says_what_it_should_be(capsys):
    with pytest.raises(SystemExit):
        main(["train", "--lr", "abc"])
    assert "invalid number value: 'abc'" in capsys.readouterr().err


def test_encode_video_turns_frames_into_a_video(tmp_path):
    folder = tmp_path / "frames"
    folder.mkdir()
    for i in range(3):
        Image.fromarray(np.full((45, 81, 3), 60 * i, dtype=np.uint8)).save(folder / f"{i}.jpg")
    main(["encode-video", str(folder), str(tmp_path / "out"), "--fps", "2", "--gif-every", "1"])
    assert (tmp_path / "out.mp4").stat().st_size > 0
    assert (tmp_path / "out.gif").stat().st_size > 0
