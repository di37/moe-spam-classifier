"""The load-balancing video: expert collapse without the auxiliary loss, and its fix.

Everything comes from the saved training runs: the share of tokens each expert got at every
logged step, the auxiliary loss along the way, and the final numbers of five seeds each.
"""

import numpy as np

from moe_spam.paths import PATHS
from moe_spam.viz.balancing.script import (
    AUX,
    BEATS,
    SCENE_TITLES,
    TITLE,
    captions,
    revealed,
)
from moe_spam.viz.panels import draw_load
from moe_spam.viz.style import AXIS, INK, INK_2, MATH_FONT, MUTED
from moe_spam.viz.video import FPS, Video, ease

CAPTION_CHARS = 170
MATH = {"math_fontfamily": MATH_FONT}
CONDITIONS = (("with the loss", INK), ("without", MUTED))  # balanced runs first


class BalancingVideo(Video):
    """Expert collapse and the load-balancing loss, on real training runs.

    Attributes:
        story: The ``BalancingStory``.
    """

    name = "balancing"

    def __init__(self, story, paths=PATHS, fps=FPS):
        """Creates the video of ``story``, writing under ``paths``, at ``fps``."""
        super().__init__(paths, fps)
        self.story = story

    def segments(self):
        """Returns one segment: the scenes, with their captions as context."""
        return [(captions(self.story), BEATS)]

    def draw(self, canvas, context, beat, t):
        """Draws one frame of the scene ``beat`` at progress ``t``."""
        scene = beat.step
        canvas.title(TITLE, SCENE_TITLES[scene])
        canvas.caption(0.015, 0.905, context[scene], CAPTION_CHARS, **MATH)
        getattr(self, f"_{scene}")(canvas, t)

    def _collapse(self, canvas, t):
        """Both layers of the unbalanced run, step by step, with the busiest expert's share."""
        history = self.story.unbalanced_history
        n = revealed(history, t)
        for layer, rect in ((1, [0.06, 0.5, 0.5, 0.27]), (2, [0.06, 0.13, 0.5, 0.27])):
            ax = canvas.axes(rect)
            draw_load(ax, history.steps[:n], history.loads[layer - 1][:n],
                      self.story.total_steps)
            ax.set_title(f"Layer {layer} without load balancing: share of tokens per expert",
                         fontsize=11.5)
            if layer == 2:
                ax.set_xlabel("training step   (dashed: an even split)")
        _readout(canvas, history, n, step=history.steps[n - 1])

    def _aux(self, canvas, t):
        """The auxiliary loss along training, with and without it in the loss."""
        canvas.text(0.66, 0.66, AUX, fontsize=24, color=INK, **MATH)
        canvas.text(0.66, 0.58, "1 = tokens spread evenly over the experts\n"
                    f"{self.story.balanced[0].experts} = all tokens to one expert",
                    fontsize=12, color=INK_2, va="top", linespacing=1.4)
        ax = canvas.axes([0.08, 0.13, 0.5, 0.6])
        histories = (self.story.balanced_history, self.story.unbalanced_history)
        top = max(np.nanmax(h.aux) for h in histories)
        for history, (label, color) in zip(histories, CONDITIONS, strict=True):
            n = revealed(history, t)
            ax.plot(history.steps[1:n], history.aux[1:n], color=color, lw=2.2)
            if n > 1:
                ax.text(history.steps[n - 1] + 10, history.aux[n - 1], label, va="center",
                        fontsize=11, color=color)
        ax.axhline(1.0, color=INK_2, lw=0.8, ls=(0, (2, 2)))
        ax.set_xlim(0, self.story.total_steps * 1.18)
        ax.set_ylim(0.98, top + 0.04)
        ax.set_xlabel("training step")
        ax.set_title("The load-balancing loss during training", fontsize=12)
        ax.grid(color=AXIS, lw=0.5, alpha=0.6)

    def _balanced(self, canvas, t):
        """Layer 2 with and without the loss, side by side, step by step."""
        histories = (self.story.balanced_history, self.story.unbalanced_history)
        for history, (label, _), x0 in zip(histories, CONDITIONS, (0.05, 0.53), strict=True):
            n = revealed(history, t)
            ax = canvas.axes([x0, 0.3, 0.42, 0.42])
            draw_load(ax, history.steps[:n], history.loads[1][:n], self.story.total_steps)
            ax.set_title(f"Layer 2, {label}", fontsize=12.5)
            ax.set_xlabel("training step")
            canvas.text(x0, 0.16, f"busiest expert: {history.busiest(2)[n - 1]:.0%}",
                        fontsize=20, weight="bold", color=INK)

    def _seeds(self, canvas, t):
        """The busiest expert's final share in each layer, for every seed of both runs."""
        ax = canvas.axes([0.1, 0.14, 0.8, 0.58])
        groups = [(f"layer {layer}\n{label}", runs, color)
                  for layer in (1, 2)
                  for (label, color), runs in zip(CONDITIONS, (self.story.balanced,
                                                              self.story.unbalanced),
                                                  strict=True)]
        for k, (_, runs, color) in enumerate(groups):
            layer = 1 + k // 2
            shares = [r.busiest(layer) for r in runs]
            shown = shares[:max(0, int(np.ceil(ease(t * 1.3) * len(shares))))]
            ax.scatter([k] * len(shown), shown, s=140, color=color, zorder=3,
                       edgecolor="white", lw=1.2)
            if len(shown) == len(shares):
                ax.text(k + 0.12, max(shares), f"{min(shares):.0%}–{max(shares):.0%}",
                        va="bottom", fontsize=11, color=INK)
        even = 1 / self.story.balanced[0].experts
        ax.axhline(even, color=INK_2, lw=0.8, ls=(0, (2, 2)))
        ax.text(len(groups) - 0.55, even + 0.01, "even split", fontsize=10, color=INK_2)
        ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=11)
        ax.set_xlim(-0.5, len(groups) - 0.3)
        ax.set_ylim(0, 1)
        ax.set_ylabel("busiest expert's share of tokens")
        ax.set_title(f"Final busiest-expert share, {len(self.story.balanced)} seeds each",
                     fontsize=12)
        ax.grid(axis="y", color=AXIS, lw=0.5, alpha=0.6)

    def _accuracy(self, canvas, t):
        """Test mistakes of every seed, with and without the loss."""
        ax = canvas.axes([0.2, 0.14, 0.6, 0.58])
        for k, ((_, color), runs) in enumerate(zip(CONDITIONS, (self.story.balanced,
                                                                self.story.unbalanced),
                                                   strict=True)):
            errors = [r.errors for r in runs]
            shown = errors[:max(0, int(np.ceil(ease(t * 1.3) * len(errors))))]
            ax.scatter([k] * len(shown), shown, s=140, color=color, zorder=3,
                       edgecolor="white", lw=1.2)
            if len(shown) == len(errors):
                ax.text(k + 0.1, np.mean(errors), f"{min(errors)}–{max(errors)} mistakes",
                        va="center", fontsize=11, color=INK)
        ax.set_xticks([0, 1], [label for label, _ in CONDITIONS], fontsize=12)
        ax.set_xlim(-0.5, 1.8)
        ax.set_ylim(0, 1.3 * max(r.errors for r in (*self.story.balanced,
                                                     *self.story.unbalanced)))
        ax.set_ylabel("test mistakes (of 1,027 messages)")
        ax.set_title("Test mistakes per seed", fontsize=12)
        ax.grid(axis="y", color=AXIS, lw=0.5, alpha=0.6)


def _readout(canvas, history, n, step):
    """The busiest expert's share in each layer, live."""
    canvas.text(0.63, 0.74, f"step {step:,}", fontsize=13, color=INK_2)
    canvas.text(0.63, 0.68, "busiest expert's share of tokens", fontsize=13, color=INK)
    for layer, y in ((1, 0.56), (2, 0.42)):
        canvas.text(0.63, y, f"layer {layer}:  {history.busiest(layer)[n - 1]:.0%}",
                    fontsize=26, weight="bold", color=INK)
    canvas.text(0.63, 0.3, f"(an even split: {1 / history.n_experts:.0%})", fontsize=12,
                color=INK_2)
