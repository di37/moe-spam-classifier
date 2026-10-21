import os

import numpy as np
import pytest
import torch
from PIL import Image

from bert import BertConfig, TinyBert
from checkpoint import load_checkpoint, save_checkpoint
from run_log import read_csv, read_json, write_csv, write_json
from video_creator import make_video, sorted_frames
from vocab import Vocab, encode


def test_checkpoint_round_trip(tmp_path):
    vocab = Vocab.build(["win a prize now", "see you at lunch"], min_count=1)
    torch.manual_seed(0)
    model = TinyBert(BertConfig(vocab_size=len(vocab), max_len=16, d_model=16, n_heads=2,
                                d_ff=32)).eval()
    path = tmp_path / "ckpt" / "moe.pt"
    save_checkpoint(str(path), model, vocab)
    loaded, loaded_vocab = load_checkpoint(str(path))
    ids = encode(vocab, ["win a prize", "see you"], max_len=16)
    with torch.no_grad():
        assert torch.equal(loaded(ids).logits, model(ids).logits)
    assert loaded_vocab == vocab
    assert loaded.config == model.config
    assert not loaded.training


def test_load_checkpoint_explains_a_missing_file(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        load_checkpoint(str(tmp_path / "nope.pt"))


def test_load_checkpoint_refuses_foreign_files(tmp_path):
    garbage = tmp_path / "garbage.pt"
    garbage.write_bytes(b"not a checkpoint")
    with pytest.raises(SystemExit, match="could not read"):
        load_checkpoint(str(garbage))
    other = tmp_path / "other.pt"
    torch.save({"model": {}}, other)
    with pytest.raises(SystemExit, match="missing"):
        load_checkpoint(str(other))


def test_load_checkpoint_refuses_weights_that_do_not_fit_the_config(tmp_path):
    vocab = Vocab.build(["a b"], min_count=1)
    model = TinyBert(BertConfig(vocab_size=len(vocab), max_len=8, d_model=16, n_heads=2, d_ff=32))
    path = tmp_path / "bad.pt"
    save_checkpoint(str(path), model, vocab)
    ckpt = torch.load(path, weights_only=True)
    torch.save({**ckpt, "config": {**ckpt["config"], "d_ff": 64}}, path)
    with pytest.raises(SystemExit, match="do not match"):
        load_checkpoint(str(path))


def test_load_checkpoint_refuses_a_vocabulary_that_does_not_fit(tmp_path):
    vocab = Vocab.build(["a b c"], min_count=1)
    model = TinyBert(BertConfig(vocab_size=len(vocab), max_len=8, d_model=16, n_heads=2, d_ff=32))
    path = tmp_path / "ckpt.pt"
    save_checkpoint(str(path), model, vocab)
    ckpt = torch.load(path, weights_only=True)
    torch.save({**ckpt, "vocab": ckpt["vocab"][:-1]}, path)
    with pytest.raises(SystemExit, match="vocabulary"):
        load_checkpoint(str(path))
    torch.save({**ckpt, "config": {**ckpt["config"], "n_heads": 3}}, path)
    with pytest.raises(SystemExit, match="do not match"):
        load_checkpoint(str(path))


def test_csv_and_json_round_trip(tmp_path):
    write_csv(str(tmp_path / "r" / "h.csv"), [{"step": 0, "loss": 0.5}, {"step": 8, "loss": 0.25}])
    assert read_csv(str(tmp_path / "r" / "h.csv")) == [{"step": "0", "loss": "0.5"},
                                                      {"step": "8", "loss": "0.25"}]
    write_json(str(tmp_path / "r.json"), {"b": 1, "a": [1, 2]})
    assert read_json(str(tmp_path / "r.json")) == {"a": [1, 2], "b": 1}
    with pytest.raises(ValueError, match="at least one row"):
        write_csv(str(tmp_path / "empty.csv"), [])


def test_make_video_writes_mp4_and_gif_in_numeric_order(tmp_path):
    folder = tmp_path / "frames"
    folder.mkdir()
    for step in (0, 8, 16, 100):
        pixels = np.full((45, 81, 3), min(255, step * 2), dtype=np.uint8)
        Image.fromarray(pixels).save(folder / f"{step:07d}.jpg")
    assert [os.path.basename(p) for p in sorted_frames(str(folder))] == [
        "0000000.jpg", "0000008.jpg", "0000016.jpg", "0000100.jpg"]
    make_video(str(folder), str(tmp_path / "media" / "run"), gif_width=40, gif_every=2)
    assert (tmp_path / "media" / "run.mp4").stat().st_size > 0
    with Image.open(tmp_path / "media" / "run.gif") as gif:
        assert gif.size == (40, 22)
