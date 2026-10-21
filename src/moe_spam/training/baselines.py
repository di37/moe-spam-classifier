"""How hard is the task? A bag-of-words logistic regression on the same split and vocabulary."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from moe_spam.data.tokenizer import PAD
from moe_spam.training.scores import COLUMNS, SpamScores

CELL_FORMATS = (".1%", "", "", "", ".3f")  # how results_table writes each of the COLUMNS


@dataclass(frozen=True)
class BagOfWordsBaseline:
    """A logistic regression on which tokens a message contains.

    Attributes:
        steps: Full-batch Adam steps.
        lr: Adam's learning rate.
        weight_decay: Adam's weight decay.
        seed: Seed of the initial weights.
    """

    steps: int = 500
    lr: float = 0.05
    weight_decay: float = 1e-4
    seed: int = 0

    @staticmethod
    def features(ids, vocab_size):
        """Returns ``[n, vocab_size]``: 1 where a message contains a token (never ``[PAD]``)."""
        present = torch.zeros(len(ids), vocab_size).scatter(1, ids, 1.0)
        return present * (torch.arange(vocab_size) != PAD)

    def fit(self, features, labels):
        """Returns a ``Linear(n_features, 1)`` fitted to ``labels`` (1 for spam)."""
        torch.manual_seed(self.seed)
        model = torch.nn.Linear(features.shape[1], 1)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for _ in range(self.steps):  # the whole training set in every step: it is small
            loss = F.binary_cross_entropy_with_logits(model(features).squeeze(-1), labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return model

    def run(self, data):
        """Fits on the training half of ``data`` (a ``SpamData``); returns the test scores."""
        vocab_size = len(data.vocab)
        model = self.fit(self.features(data.train_ids, vocab_size), data.train_labels)
        with torch.no_grad():
            logits = model(self.features(data.test_ids, vocab_size)).squeeze(-1)
        return SpamScores.from_logits(logits, data.test_labels)

    def as_record(self):
        """Returns the settings, for the run record."""
        return {"steps": self.steps, "lr": self.lr, "weight_decay": self.weight_decay,
                "seed": self.seed}


def results_table(runs):
    """Returns one aligned row per run.

    Args:
        runs: ``{name: scores}``, each with the ``COLUMNS`` keys.

    Returns:
        The table as text, with a header row.
    """
    header = f"{'run':<22}" + "".join(f"{c:>13}" for c in COLUMNS)
    rows = [f"{name:<22}" + " ".join(format(m[column], f">12{spec}")
                                     for column, spec in zip(COLUMNS, CELL_FORMATS, strict=True))
            for name, m in runs.items()]
    return "\n".join([header, *rows])
