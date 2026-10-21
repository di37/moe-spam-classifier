import os
import random

import numpy as np
import pytest
import torch

import dataset
from bert import BertConfig, TinyBert
from checkpoint import load_checkpoint
from dataset import MAX_LEN, prepare_data
from run_log import read_csv, read_json
from sms_data import Message
from train_moe import (
    evaluate,
    group_shares,
    lr_factor,
    parse_args,
    pick_examples,
    run_name,
    train,
)
from vocab import Vocab

HAM_WORDS = ("see", "you", "at", "lunch", "tomorrow", "ok", "thanks", "call", "me", "later",
             "home", "now")
SPAM_WORDS = ("win", "free", "prize", "cash", "claim", "txt", "now", "urgent", "call", "stop")


def synthetic_messages(n=240, seed=0):
    """Ham and spam built from two small word lists; spam ends with a 5-digit short code."""
    rng = random.Random(seed)
    messages = []
    for i in range(n):
        is_spam = i % 4 == 0
        pool = SPAM_WORDS if is_spam else HAM_WORDS
        words = [rng.choice(pool) for _ in range(rng.randint(8, 12))]
        tail = f" {rng.randint(10000, 99999)}" if is_spam else ""
        messages.append(Message(is_spam, " ".join(words) + tail))
    return messages


@pytest.fixture
def tiny_project(tmp_path, monkeypatch):
    """Run in an empty folder, on synthetic messages instead of the real dataset."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dataset, "load_messages", lambda: synthetic_messages())
    return tmp_path


def test_defaults():
    args = parse_args([])
    assert (args.steps, args.experts, args.aux_weight, args.frame_every) == (800, 3, 0.01, 5)


def test_video_needs_three_experts():
    with pytest.raises(SystemExit):
        parse_args(["--experts", "2"])
    assert parse_args(["--experts", "2", "--no-video"]).experts == 2


@pytest.mark.parametrize("bad", [["--aux-weight", "-1"], ["--lr", "0"], ["--steps", "0"],
                                 ["--lr", "nan"], ["--aux-weight", "inf"]])
def test_rejects_bad_values(bad):
    with pytest.raises(SystemExit):
        parse_args(bad)


def test_run_names():
    assert run_name(parse_args([])) == "moe"
    assert run_name(parse_args(["--aux-weight", "0"])) == "moe_noaux"
    assert run_name(parse_args(["--experts", "1", "--no-video"])) == "dense"
    assert run_name(parse_args(["--seed", "2"])) == "moe_seed2"
    # every other changed setting is named too, so different runs never share files
    assert run_name(parse_args(["--experts", "2", "--no-video"])) == "moe_experts2"
    assert run_name(parse_args(["--aux-weight", "0.5", "--steps", "50"])) == (
        "moe_auxweight0.5_steps50")
    assert run_name(parse_args(["--aux-weight", "0", "--seed", "1", "--frame-every", "2"])) == (
        "moe_noaux_seed1")


def test_lr_warms_up_then_decays_to_zero():
    factor = lr_factor(total_steps=1000, warmup=100)
    assert factor(0) == pytest.approx(0.01)
    assert factor(99) == pytest.approx(1.0)
    assert factor(550) == pytest.approx(0.5)
    assert factor(1000) == 0.0
    assert lr_factor(total_steps=10, warmup=100)(9) == pytest.approx(1.0)  # short runs


def test_group_shares_rows_sum_to_one_and_empty_groups_are_nan():
    groups = torch.tensor([0, 0, 0, 2, 2])
    experts = torch.tensor([0, 1, 1, 2, 2])
    shares = group_shares(groups, experts, n_groups=3, n_experts=3)
    assert np.allclose(shares[0], [1 / 3, 2 / 3, 0])
    assert np.isnan(shares[1]).all()
    assert np.allclose(shares[2], [0, 0, 1])


def test_pick_examples_finds_short_known_messages_and_a_spam_with_a_number():
    messages = synthetic_messages()
    vocab = Vocab.build([m.text for m in messages], min_count=1)
    ham, spam = pick_examples(messages, vocab)
    assert not messages[ham].is_spam
    assert messages[spam].is_spam
    with pytest.raises(ValueError, match="example"):
        pick_examples([m for m in messages if not m.is_spam], vocab)


def test_short_run_writes_frames_video_checkpoint_and_record(tiny_project):
    final = train(parse_args(["--steps", "16", "--frame-every", "8"]))
    assert sorted(os.listdir("frames/training_moe_steps16")) == ["0000000.jpg", "0000008.jpg",
                                                                 "0000016.jpg"]
    assert os.path.getsize("media/training_moe_steps16.mp4") > 0
    assert os.path.getsize("media/training_moe_steps16.gif") > 0
    record = read_json("results/training_moe_steps16.json")
    assert record["final_metrics"] == final
    assert record["frames"] == 3
    assert {"accuracy", "spam F1", "errors", "layer 1 load", "train BCE"} <= set(final)
    assert {"drawing_frames", "video_encoding", "total"} <= set(record["wall_clock_s"])
    model, vocab = load_checkpoint("checkpoints/moe_steps16.pt")
    assert model.config.n_experts == 3
    rows = read_csv("results/training_moe_steps16.csv")
    assert [r["step"] for r in rows] == ["0", "8", "16"]
    assert rows[0]["batch BCE"] == "nan"  # step 0 is the untrained model: no batch yet


def test_same_seed_same_history_and_no_frames_without_video(tiny_project):
    args = ["--steps", "10", "--frame-every", "5", "--no-video"]
    train(parse_args(args))
    first = read_csv("results/training_moe_steps10.csv")
    train(parse_args(args))
    assert read_csv("results/training_moe_steps10.csv") == first
    assert not os.path.exists("frames")
    assert read_json("results/training_moe_steps10.json")["frames"] == 0
    train(parse_args([*args, "--seed", "1"]))
    assert read_csv("results/training_moe_steps10_seed1.csv") != first


def test_dense_run_trains_without_a_router_choice(tiny_project):
    final = train(parse_args(["--experts", "1", "--steps", "5", "--frame-every", "5",
                              "--no-video"]))
    assert final["layer 1 load"] == [1.0]


def test_evaluate_leaves_the_model_in_the_mode_it_found_it(tiny_project):
    data = prepare_data()
    model = TinyBert(BertConfig(vocab_size=len(data.vocab), max_len=MAX_LEN, d_model=16,
                                n_heads=2, d_ff=32)).train()
    metrics, state, loads = evaluate(model, data, pick_examples(data.test_messages, data.vocab))
    assert model.training
    assert state.cls_probs.shape == (len(data.test_ids), 3)
    assert state.cls_probs.base is None or state.cls_probs.base.size == state.cls_probs.size
    assert metrics["errors"] == metrics["spam missed"] + metrics["ham flagged"]


def test_prepare_data_learns_only_from_the_training_half(tiny_project):
    data = prepare_data()
    assert len(data.train_ids) + len(data.test_ids) == len(dataset.dedupe(synthetic_messages()))
    assert data.test_ids.shape[1] == MAX_LEN
    assert data.test_labels.tolist() == [float(m.is_spam) for m in data.test_messages]
