"""The load-balancing video's script: its beats, titles, captions and what it compares."""

from dataclasses import dataclass

import numpy as np

from moe_spam.viz.video import Beat, numbered_titles

TITLE = "Load balancing: why an MoE needs it"
BEATS = (Beat("collapse", 9.0, 1.5), Beat("aux", 6.0, 2.0), Beat("balanced", 9.0, 1.5),
         Beat("seeds", 4.0, 3.5), Beat("accuracy", 3.0, 3.5))
SCENE_NAMES = {"collapse": "Without it, experts collapse",
               "aux": "The load-balancing loss",
               "balanced": "With it, every expert works",
               "seeds": "Not luck: 5 seeds each",
               "accuracy": "And the accuracy?"}
SCENE_TITLES = numbered_titles(BEATS, SCENE_NAMES)
AUX = r"$\mathrm{aux}=3\,{\sum}_i\,f_i\,P_i$"  # no spaces: wrapping never splits it


@dataclass(frozen=True)
class BalancingStory:
    """The runs the video compares: the same model trained with and without the loss.

    Attributes:
        balanced: ``RunSummary`` of each seed trained with the load-balancing loss.
        unbalanced: ``RunSummary`` of each seed trained without it.
        balanced_history: The ``RunHistory`` of the first balanced seed.
        unbalanced_history: The ``RunHistory`` of the first unbalanced seed.
        total_steps: Training steps of every run.
    """

    balanced: tuple
    unbalanced: tuple
    balanced_history: object
    unbalanced_history: object
    total_steps: int

    @property
    def aux_weight(self):
        """The weight of the loss in the balanced runs."""
        return self.balanced[0].aux_weight


def share_range(runs, layer):
    """Returns the busiest expert's share across ``runs``, as text: ``"74–83%"``."""
    shares = [r.busiest(layer) for r in runs]
    return f"{min(shares):.0%}–{max(shares):.0%}".replace("%–", "–")


def captions(story):
    """Returns the caption of every scene, with the runs' real numbers."""
    with_, without = story.balanced[0], story.unbalanced[0]
    final_aux = (story.balanced_history.aux[-1], story.unbalanced_history.aux[-1])
    errors = [[r.errors for r in runs] for runs in (story.balanced, story.unbalanced)]
    return {
        "collapse": f"Without load balancing, layer 2's busiest expert ends with "
                    f"{without.busiest(2):.0%} of the tokens: the expert that pulls ahead gets "
                    f"more training, so the router sends it even more.",
        "aux": f"The loss {AUX} is 1 when tokens are spread evenly. Without it in training, it "
               f"climbs to {final_aux[1]:.2f}; added with weight {story.aux_weight}, it stays "
               f"at {final_aux[0]:.2f}.",
        "balanced": f"Same model, data and seed. Layer 2's busiest expert: "
                    f"{with_.busiest(2):.0%} with the loss, {without.busiest(2):.0%} without it.",
        "seeds": f"Over {len(story.balanced)} seeds, layer 2 collapses every time without the "
                 f"loss ({share_range(story.unbalanced, 2)}); with it, the busiest expert stays "
                 f"at {share_range(story.balanced, 2)}. Layer 1 varies more.",
        "accuracy": f"Mistakes: {min(errors[0])}–{max(errors[0])} with the loss, "
                    f"{min(errors[1])}–{max(errors[1])} without. Balancing buys no accuracy here: "
                    f"it keeps every expert in use.",
    }


def revealed(history, t):
    """Returns how many logged steps of ``history`` are shown at progress ``t`` (at least 2)."""
    return max(2, int(np.ceil(t * len(history.steps))))
