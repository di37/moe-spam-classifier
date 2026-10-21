"""
Save a trained TinyBert with everything needed to use it again (its sizes and its vocabulary),
and load it back.
"""
import os
from dataclasses import asdict

import torch

from bert import BertConfig, TinyBert
from vocab import Vocab

REQUIRED_KEYS = ("config", "model", "vocab")


def save_checkpoint(path, model, vocab):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"config": asdict(model.config), "model": model.state_dict(),
                "vocab": list(vocab.words)}, path)


def load_checkpoint(path):
    """Returns (model in eval mode, vocab)."""
    if not os.path.isfile(path):
        raise SystemExit(f"checkpoint not found: {path} (train one first)")
    try:
        # weights_only: load tensors and plain values only, never arbitrary pickled code.
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as err:  # torch raises many types for truncated or foreign files
        raise SystemExit(f"could not read checkpoint {path}: {err}") from err
    missing = [k for k in REQUIRED_KEYS if not isinstance(ckpt, dict) or k not in ckpt]
    if missing:
        raise SystemExit(f"{path} is not a checkpoint from this project (missing {missing})")
    vocab = Vocab(tuple(ckpt["vocab"]))
    try:
        config = BertConfig(**ckpt["config"])
        if config.vocab_size != len(vocab):
            raise ValueError(f"the vocabulary has {len(vocab)} words, the model "
                             f"{config.vocab_size}")
        model = TinyBert(config)
        model.load_state_dict(ckpt["model"])
    # nn.MultiheadAttention asserts that d_model divides into n_heads
    except (TypeError, ValueError, AssertionError, RuntimeError) as err:
        raise SystemExit(f"weights in {path} do not match the saved settings: {err}") from err
    return model.eval(), vocab
