import os

import pytest
import torch

from moe_spam.analysis.examples import ExampleFinder
from moe_spam.cli.main import build_parser, main, run
from moe_spam.cli.training_commands import TrainCommand
from moe_spam.data.dataset import MAX_LEN, SpamData
from moe_spam.data.vocab import Vocab
from moe_spam.errors import MoeSpamError
from moe_spam.model.bert import BertConfig, TinyBert
from moe_spam.model.checkpoint import Checkpoint
from moe_spam.records import RecordStore
from moe_spam.training.config import TrainingConfig
from moe_spam.training.evaluator import Evaluator
from moe_spam.training.trainer import Trainer, WarmupLinearDecay


def parse_train(argv):
    parser, commands = build_parser()
    args = parser.parse_args(["train", *argv])
    return TrainCommand.config(args, commands["train"][1])


def test_defaults_live_in_the_config():
    config = parse_train([])
    assert config == TrainingConfig()
    assert (config.steps, config.experts, config.aux_weight, config.frame_every) == (800, 3, 0.01,
                                                                                      5)
    assert set(config.as_record()) == {"steps", "batch_size", "lr", "experts", "aux_weight",
                                       "frame_every", "seed", "no_video"}


def test_video_needs_three_experts():
    with pytest.raises(SystemExit):
        parse_train(["--experts", "2"])
    assert parse_train(["--experts", "2", "--no-video"]).experts == 2
    with pytest.raises(ValueError, match="3 experts"):
        TrainingConfig(experts=2)


@pytest.mark.parametrize("bad", [["--aux-weight", "-1"], ["--lr", "0"], ["--steps", "0"],
                                 ["--lr", "nan"], ["--aux-weight", "inf"]])
def test_rejects_bad_values(bad):
    with pytest.raises(SystemExit):
        parse_train(bad)


@pytest.mark.parametrize("bad", [{"steps": 0}, {"lr": float("nan")}, {"aux_weight": -1.0}])
def test_config_checks_its_own_values(bad):
    with pytest.raises(ValueError):
        TrainingConfig(**bad)


def test_run_names():
    assert TrainingConfig().run_name == "moe"
    assert TrainingConfig(aux_weight=0).run_name == "moe_noaux"
    assert TrainingConfig(experts=1, no_video=True).run_name == "dense"
    assert TrainingConfig(seed=2).run_name == "moe_seed2"
    # every other changed setting is named too, so different runs never share files
    assert TrainingConfig(experts=2, no_video=True).run_name == "moe_experts2"
    assert TrainingConfig(aux_weight=0.5, steps=50).run_name == "moe_auxweight0.5_steps50"
    assert TrainingConfig(aux_weight=0, seed=1, frame_every=2).run_name == "moe_noaux_seed1"


def test_lr_warms_up_then_decays_to_zero():
    factor = WarmupLinearDecay(total_steps=1000, warmup=100)
    assert factor(0) == pytest.approx(0.01)
    assert factor(99) == pytest.approx(1.0)
    assert factor(550) == pytest.approx(0.5)
    assert factor(1000) == 0.0
    assert WarmupLinearDecay(total_steps=10, warmup=100)(9) == pytest.approx(1.0)  # short runs


def test_first_examples_are_short_known_messages_and_a_spam_with_a_number(synthetic_messages):
    messages = synthetic_messages
    finder = ExampleFinder(Vocab.build([m.text for m in messages], min_count=1))
    ham, spam = finder.first(messages)
    assert not messages[ham].is_spam
    assert messages[spam].is_spam
    with pytest.raises(MoeSpamError, match="example"):
        finder.first([m for m in messages if not m.is_spam])


def test_train_command_writes_frames_video_checkpoint_and_record(tiny_project):
    final = run(["train", "--steps", "16", "--frame-every", "8"])
    assert sorted(os.listdir("frames/training_moe_steps16")) == [
        "0000000.jpg", "0000008.jpg", "0000016.jpg"]
    assert os.path.getsize("media/training_moe_steps16.mp4") > 0
    assert os.path.getsize("media/training_moe_steps16.gif") > 0
    store = RecordStore()
    record = store.read_json("training_moe_steps16")
    assert record["final_metrics"] == final
    assert record["frames"] == 3
    assert record["args"]["steps"] == 16
    assert {"accuracy", "spam F1", "errors", "layer 1 load", "train BCE"} <= set(final)
    assert {"drawing_frames", "video_encoding", "total"} <= set(record["wall_clock_s"])
    assert Checkpoint.load("checkpoints/moe_steps16.pt").model.config.n_experts == 3
    rows = store.read_csv("training_moe_steps16")
    assert [r["step"] for r in rows] == ["0", "8", "16"]
    assert rows[0]["batch BCE"] == "nan"  # step 0 is the untrained model: no batch yet


def test_same_seed_same_history_and_no_frames_without_video(tiny_project):
    args = ["train", "--steps", "10", "--frame-every", "5", "--no-video"]
    store = RecordStore()
    main(args)
    first = store.read_csv("training_moe_steps10")
    main(args)
    assert store.read_csv("training_moe_steps10") == first
    assert not os.path.exists("frames")
    assert store.read_json("training_moe_steps10")["frames"] == 0
    assert main([*args, "--seed", "1"]) == 0  # an exit status, not the metrics
    assert store.read_csv("training_moe_steps10_seed1") != first


def test_dense_run_trains_without_a_router_choice(tiny_project, synthetic_messages):
    config = TrainingConfig(experts=1, steps=5, frame_every=5, no_video=True)
    run = Trainer(config, data=SpamData.prepare(synthetic_messages)).run()
    assert run.final_metrics["layer 1 load"] == [1.0]
    assert run.name == "training_dense_steps5"


def test_evaluate_leaves_the_model_in_the_mode_it_found_it(synthetic_messages):
    data = SpamData.prepare(synthetic_messages)
    examples = ExampleFinder(data.vocab).first(data.test_messages)
    model = TinyBert(BertConfig(vocab_size=len(data.vocab), max_len=MAX_LEN, d_model=16,
                                n_heads=2, d_ff=32)).train()
    evaluation = Evaluator(data, examples).evaluate(model)
    assert model.training
    assert evaluation.state.cls_probs.shape == (len(data.test_ids), 3)
    assert evaluation.state.cls_probs.base is None or (
        evaluation.state.cls_probs.base.size == evaluation.state.cls_probs.size)
    metrics = evaluation.metrics
    assert metrics["errors"] == metrics["spam missed"] + metrics["ham flagged"]
    assert torch.is_tensor(data.test_ids)
