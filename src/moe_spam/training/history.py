"""What a training run logs: its curves, and what the model did at the steps kept as frames."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TokenRow:
    """One example message as the training video draws it.

    Attributes:
        label: ``"ham"`` or ``"spam"``, the true label.
        words: What to print for each token, ``[CLS]`` first.
        experts: The layer-1 expert that processed each token (0-based).
        p_spam: The model's spam probability for the whole message.
    """

    label: str
    words: tuple
    experts: tuple
    p_spam: float


@dataclass(frozen=True)
class FrameState:
    """What the model did on the fixed test set at one step.

    Attributes:
        cls_probs: ``[n, n_experts]`` layer-2 router probabilities of each test ``[CLS]``.
        is_spam: ``[n]`` bool, the true labels.
        wrong: ``[n]`` bool, misclassified at this step.
        examples: The ``TokenRow`` of each example message.
        group_shares: ``[groups, n_experts]`` layer-1 share of each token group per expert.
    """

    cls_probs: np.ndarray
    is_spam: np.ndarray
    wrong: np.ndarray
    examples: tuple
    group_shares: np.ndarray


class History:
    """The curves logged during training, and the states kept for frames.

    Attributes:
        steps: The logged steps.
        losses: ``{name: values}``, one value per logged step.
        errors: ``{name: counts}``, one count per logged step.
        loads: Per logged step, each layer's ``[n_experts]`` expert load.
        extras: ``{name: values}`` saved in the CSV but not drawn.
        kept: ``(step, number of logged steps at that point, FrameState)`` per kept step.
    """

    def __init__(self):
        """Starts an empty history."""
        self.steps = []
        self.losses = {}
        self.errors = {}
        self.loads = []
        self.extras = {}
        self.kept = []

    def log(self, step, losses, errors, loads, extras=None):
        """Logs one evaluated step.

        Args:
            step: The training step.
            losses: ``{name: value}``.
            errors: ``{name: count}``.
            loads: Each layer's ``[n_experts]`` expert load.
            extras: ``{name: value}`` for the CSV only.
        """
        self.steps.append(step)
        for k, v in losses.items():
            self.losses.setdefault(k, []).append(float(v))
        for k, v in errors.items():
            self.errors.setdefault(k, []).append(int(v))
        self.loads.append(tuple(np.array(layer, dtype=float) for layer in loads))
        for k, v in (extras or {}).items():
            self.extras.setdefault(k, []).append(float(v))

    def keep(self, step, state):
        """Keeps what the model did at the step just logged, to be drawn later.

        Raises:
            RuntimeError: If ``step`` is not the step logged last.
        """
        if not self.steps or self.steps[-1] != step:
            raise RuntimeError(f"log() step {step} before keeping its frame")
        self.kept.append((step, len(self.steps), state))

    def rows(self):
        """Returns everything logged, one dict per logged step, for the CSV."""
        return [
            {"step": step,
             **{k: v[i] for k, v in self.losses.items()},
             **{k: v[i] for k, v in self.errors.items()},
             **{k: v[i] for k, v in self.extras.items()},
             **{f"layer {layer + 1} load expert {e + 1}": float(load[e])
                for layer, load in enumerate(self.loads[i]) for e in range(len(load))}}
            for i, step in enumerate(self.steps)
        ]
