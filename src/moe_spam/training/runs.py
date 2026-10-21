"""Finished training runs, read back from their records: final numbers and history."""

import re
from dataclasses import dataclass

import numpy as np

from moe_spam.errors import MoeSpamError
from moe_spam.records import RecordStore
from moe_spam.training.baselines import RECORD, RECORD_KEY
from moe_spam.training.trainer import RECORD_PREFIX

LOAD_COLUMN = re.compile(r"layer (\d+) load expert (\d+)")


@dataclass(frozen=True)
class RunSummary:
    """A run's settings and how its trained model did on the test set.

    Attributes:
        name: The run name, e.g. ``moe_noaux_seed2``.
        experts: Experts per MoE layer.
        aux_weight: The weight of the load-balancing loss.
        seed: The training seed.
        accuracy: Test accuracy.
        errors: Test mistakes.
        spam_f1: The F1 score of the spam class.
        loads: Per layer, the share of test tokens each expert got.
    """

    name: str
    experts: int
    aux_weight: float
    seed: int
    accuracy: float
    errors: int
    spam_f1: float
    loads: tuple

    def busiest(self, layer):
        """Returns the share of tokens the busiest expert of ``layer`` (from 1) got."""
        return max(self.loads[layer - 1])


@dataclass(frozen=True)
class RunHistory:
    """What a run logged at every evaluation.

    Attributes:
        steps: ``[n]`` the logged steps.
        loads: Per layer, ``[n, n_experts]`` the share of test tokens of each expert.
        aux: ``[n]`` the mean load-balancing loss of the steps since the last log (nan at 0).
    """

    steps: np.ndarray
    loads: tuple
    aux: np.ndarray

    @property
    def n_experts(self):
        """Experts per layer."""
        return self.loads[0].shape[1]

    def busiest(self, layer):
        """Returns ``[n]``: the busiest expert's share of ``layer`` (from 1) at every step."""
        return self.loads[layer - 1].max(axis=1)


def load_summary(name, store=None):
    """Returns the ``RunSummary`` of run ``name``.

    Raises:
        MoeSpamError: If the run has not been trained.
    """
    record = _read(name, store, "read_json")
    args, metrics = record["args"], record["final_metrics"]
    layers = sorted(k for k in metrics if re.fullmatch(r"layer \d+ load", k))
    return RunSummary(name=name, experts=args["experts"], aux_weight=args["aux_weight"],
                      seed=args["seed"], accuracy=metrics["accuracy"], errors=metrics["errors"],
                      spam_f1=metrics["spam F1"],
                      loads=tuple(tuple(metrics[k]) for k in layers))


def load_history(name, store=None):
    """Returns the ``RunHistory`` of run ``name``.

    Raises:
        MoeSpamError: If the run has not been trained.
    """
    rows = _read(name, store, "read_csv")
    columns = {}
    for key in rows[0]:
        match = LOAD_COLUMN.fullmatch(key)
        if match:
            columns.setdefault(int(match[1]), []).append(key)
    loads = tuple(np.array([[float(row[k]) for k in columns[layer]] for row in rows])
                  for layer in sorted(columns))
    return RunHistory(steps=np.array([int(row["step"]) for row in rows]), loads=loads,
                      aux=np.array([float(row["aux loss"]) for row in rows]))


def load_baseline(store=None):
    """Returns the bag-of-words baseline's test scores.

    Raises:
        MoeSpamError: If the baseline has not been fitted.
    """
    store = store or RecordStore()
    try:
        return store.read_json(RECORD)[RECORD_KEY]
    except FileNotFoundError as error:
        raise MoeSpamError("no baseline record: fit it first with `moe-spam baselines`") from error


def _read(name, store, method):
    store = store or RecordStore()
    try:
        return getattr(store, method)(f"{RECORD_PREFIX}{name}")
    except FileNotFoundError as error:
        raise MoeSpamError(f"no record of the run {name!r}: train it first with `moe-spam "
                           f"train` (see README)") from error
