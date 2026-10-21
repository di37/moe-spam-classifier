"""A trained model saved with everything needed to use it again: its sizes and vocabulary."""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from moe_spam.data.tokenizer import SPECIALS
from moe_spam.data.vocab import Vocab
from moe_spam.errors import MoeSpamError
from moe_spam.model.bert import BertConfig, TinyBert

REQUIRED_KEYS = ("config", "model", "vocab")


@dataclass(frozen=True)
class Checkpoint:
    """A model and the vocabulary it was trained with.

    Attributes:
        model: The model (in eval mode after ``load``).
        vocab: Its vocabulary.
    """

    model: TinyBert
    vocab: Vocab

    def save(self, path):
        """Saves the model's sizes, weights and vocabulary to ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(self.model.config), "model": self.model.state_dict(),
                    "vocab": list(self.vocab.words)}, path)

    @classmethod
    def load(cls, path):
        """Loads a checkpoint saved by ``save``, with the model in eval mode.

        Args:
            path: The checkpoint file.

        Returns:
            The checkpoint.

        Raises:
            MoeSpamError: If the file is missing, unreadable, not from this project, or its
                weights do not fit its saved sizes or vocabulary.
        """
        path = Path(path)
        if not path.is_file():
            raise MoeSpamError(f"checkpoint not found: {path} (train one first, from the "
                             f"repository root)")
        try:
            # weights_only: load tensors and plain values only, never arbitrary pickled code.
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as err:  # torch raises many types for truncated or foreign files
            raise MoeSpamError(f"could not read checkpoint {path}: {err}") from err
        missing = [k for k in REQUIRED_KEYS if not isinstance(ckpt, dict) or k not in ckpt]
        if missing:
            raise MoeSpamError(f"{path} is not a checkpoint from this project (missing {missing})")
        try:
            vocab = Vocab(tuple(ckpt["vocab"]))
            if vocab.words[:len(SPECIALS)] != SPECIALS:
                raise ValueError(f"the vocabulary does not start with the special tokens "
                                 f"{SPECIALS}")
            config = BertConfig(**ckpt["config"])
            if config.vocab_size != len(vocab):
                raise ValueError(f"the vocabulary has {len(vocab)} words, the model "
                                 f"{config.vocab_size}")
            model = TinyBert(config)
            model.load_state_dict(ckpt["model"])
        # nn.MultiheadAttention asserts that d_model divides into n_heads
        except (TypeError, ValueError, AssertionError, RuntimeError) as err:
            raise MoeSpamError(f"weights in {path} do not match the saved settings: {err}") from err
        return cls(model.eval(), vocab)

    def require(self, **sizes):
        """Checks the model's sizes, for code that only works with some of them.

        Args:
            **sizes: ``BertConfig`` fields and the values they must have, e.g.
                ``n_layers=2, n_experts=3``.

        Returns:
            This checkpoint, so the call can be chained after ``load``.

        Raises:
            MoeSpamError: If any size differs, naming them all.
        """
        config = self.model.config
        wrong = {k: getattr(config, k) for k, v in sizes.items() if getattr(config, k) != v}
        if wrong:
            raise MoeSpamError(f"this needs a model with {sizes}; this checkpoint has {wrong}")
        return self
