import os

import numpy as np
import pytest
import torch
from PIL import Image

from moe_spam.data.vocab import Vocab
from moe_spam.errors import MoeSpamError
from moe_spam.model.bert import BertConfig, TinyBert
from moe_spam.model.checkpoint import Checkpoint
from moe_spam.records import RecordStore
from moe_spam.viz.encoder import VideoEncoder


def small_checkpoint(**sizes):
    vocab = Vocab.build(["win a prize now", "see you at lunch"], min_count=1)
    torch.manual_seed(0)
    config = {"vocab_size": len(vocab), "max_len": 16, "d_model": 16, "n_heads": 2, "d_ff": 32,
              **sizes}
    return Checkpoint(TinyBert(BertConfig(**config)).eval(), vocab)


def test_checkpoint_round_trip(tmp_path):
    saved = small_checkpoint()
    path = tmp_path / "ckpt" / "moe.pt"
    saved.save(path)
    loaded = Checkpoint.load(path)
    ids = saved.vocab.encode(["win a prize", "see you"], max_len=16)
    with torch.no_grad():
        assert torch.equal(loaded.model(ids).logits, saved.model(ids).logits)
    assert loaded.vocab == saved.vocab
    assert loaded.model.config == saved.model.config
    assert not loaded.model.training


def test_require_checks_the_sizes_a_command_needs():
    checkpoint = small_checkpoint()
    assert checkpoint.require(n_layers=2, n_experts=3) is checkpoint
    with pytest.raises(MoeSpamError, match="n_experts"):
        checkpoint.require(n_experts=4)


def test_load_explains_a_missing_file(tmp_path):
    with pytest.raises(MoeSpamError, match="not found"):
        Checkpoint.load(tmp_path / "nope.pt")


def test_load_refuses_foreign_files(tmp_path):
    garbage = tmp_path / "garbage.pt"
    garbage.write_bytes(b"not a checkpoint")
    with pytest.raises(MoeSpamError, match="could not read"):
        Checkpoint.load(garbage)
    other = tmp_path / "other.pt"
    torch.save({"model": {}}, other)
    with pytest.raises(MoeSpamError, match="missing"):
        Checkpoint.load(other)


def test_load_refuses_weights_that_do_not_fit_the_config(tmp_path):
    path = tmp_path / "bad.pt"
    small_checkpoint(max_len=8).save(path)
    ckpt = torch.load(path, weights_only=True)
    torch.save({**ckpt, "config": {**ckpt["config"], "d_ff": 64}}, path)
    with pytest.raises(MoeSpamError, match="do not match"):
        Checkpoint.load(path)


def test_load_refuses_a_vocabulary_that_does_not_fit(tmp_path):
    path = tmp_path / "ckpt.pt"
    small_checkpoint(max_len=8).save(path)
    ckpt = torch.load(path, weights_only=True)
    torch.save({**ckpt, "vocab": ckpt["vocab"][:-1]}, path)
    with pytest.raises(MoeSpamError, match="vocabulary"):
        Checkpoint.load(path)
    torch.save({**ckpt, "config": {**ckpt["config"], "n_heads": 3}}, path)
    with pytest.raises(MoeSpamError, match="do not match"):
        Checkpoint.load(path)


def test_load_refuses_a_vocabulary_without_the_special_tokens(tmp_path):
    path = tmp_path / "ckpt.pt"
    small_checkpoint().save(path)
    ckpt = torch.load(path, weights_only=True)
    torch.save({**ckpt, "vocab": ["hello", *ckpt["vocab"][1:]]}, path)
    with pytest.raises(MoeSpamError, match="special"):
        Checkpoint.load(path)


def test_record_store_round_trips_csv_and_json(tmp_path):
    store = RecordStore(tmp_path / "r[1]")  # glob characters in the folder name are no problem
    store.write_csv("h", [{"step": 0, "loss": 0.5}, {"step": 8, "loss": 0.25}])
    assert store.read_csv("h") == [{"step": "0", "loss": "0.5"}, {"step": "8", "loss": "0.25"}]
    store.write_json("training_a", {"b": 1, "a": [1, 2]})
    store.write_json("training_b", {"c": 3})
    store.write_json("other", {})
    assert store.read_json("training_a") == {"a": [1, 2], "b": 1}
    assert list(store.read_all("training_*")) == ["training_a", "training_b"]
    with pytest.raises(ValueError, match="at least one row"):
        store.write_csv("empty", [])
    assert set(store.environment()) >= {"os", "python", "packages"}
    assert len(store.sha256(store.path("other", ".json"))) == 64
    assert store.command().startswith("moe-spam ")  # no absolute paths in the records


def test_encoder_writes_mp4_and_gif_in_numeric_order(tmp_path):
    folder = tmp_path / "frames"
    folder.mkdir()
    for step in (0, 8, 16, 100):
        pixels = np.full((45, 81, 3), min(255, step * 2), dtype=np.uint8)
        Image.fromarray(pixels).save(folder / f"{step:07d}.jpg")
    assert [os.path.basename(p) for p in VideoEncoder.sorted_frames(folder)] == [
        "0000000.jpg", "0000008.jpg", "0000016.jpg", "0000100.jpg"]
    VideoEncoder(gif_width=40, gif_every=2).encode(folder, tmp_path / "media" / "run")
    assert (tmp_path / "media" / "run.mp4").stat().st_size > 0
    with Image.open(tmp_path / "media" / "run.gif") as gif:
        assert gif.size == (40, 22)


def test_encoder_refuses_an_empty_or_missing_folder(tmp_path):
    with pytest.raises(MoeSpamError, match="no frames"):
        VideoEncoder().encode(tmp_path, tmp_path / "out")
    with pytest.raises(MoeSpamError, match="not found"):
        VideoEncoder().encode(tmp_path / "missing", tmp_path / "out")


@pytest.mark.parametrize("bad", [{"gif_every": 0}, {"fps": 0}, {"fps": -3}, {"gif_width": 0},
                                 {"hold_last": -1.0}])
def test_encoder_refuses_bad_settings(bad):
    with pytest.raises(ValueError, match=next(iter(bad))):
        VideoEncoder(**bad)
