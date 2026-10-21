"""What each step of the model does to one message, drawn in the detail panel.

Every drawing takes a progress ``t`` in [0, 1] through its step's animation.
"""

import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

from moe_spam.viz.architecture.layout import FONT
from moe_spam.viz.style import AXIS, EXPERT_COLORS, HIGHLIGHT, INK, INK_2, MONO, SURFACE, ink_on
from moe_spam.viz.video import ease

VECTOR_CELLS = 8  # embedding numbers drawn under each token


class ArchitectureScenes:
    """The detail-panel drawings of one message.

    Attributes:
        trace: What the model did to the message (a ``MessageTrace``).
        layout: Where its tokens sit (a ``TokenLayout``).
    """

    def __init__(self, trace, layout):
        """Creates the scenes of the message in ``trace``, laid out as ``layout``."""
        self.trace = trace
        self.layout = layout

    def draw(self, ax, step, t):
        """Draws ``step`` at progress ``t``; the steps are the keys of ``STEPS``."""
        method, layer = STEPS[step]
        if layer is None:
            method(self, ax, t)
        else:
            method(self, ax, t, layer)

    def text(self, ax, t):
        """The message fading in."""
        self._message(ax, alpha=ease(2 * t))

    def tokenize(self, ax, t):
        """Tokens appearing one after another, all in by ``t = 1``."""
        self._message(ax)
        n = len(self.trace.words)
        for i, (word, token, xy) in enumerate(zip(self.trace.words, self.trace.tokens,
                                                  self.layout.home, strict=True)):
            shown = ease((t - 0.8 * i / n) / 0.2)
            if shown <= 0:
                continue
            _token(ax, xy, word, alpha=shown, edge=INK if i == 0 else AXIS, lw=2 if i == 0 else 1)
            if token != word:  # [NUM], [NUM5+] or [UNK]: say what the model actually gets
                ax.text(xy[0], xy[1] + 1.0, token, family=MONO, fontsize=FONT - 3, color=INK_2,
                        alpha=shown, va="center")

    def embed(self, ax, t):
        """Each token's first embedding numbers, darker for larger."""
        self._message(ax)
        self._rows(ax)
        values = self.trace.embeddings[:, :VECTOR_CELLS]
        low, high = values.min(), values.max()
        shown = ease(t * 1.5)
        for (x, y), vector in zip(self.layout.home, values, strict=True):
            for k, v in enumerate(vector):
                grey = 0.92 - 0.8 * (v - low) / (high - low)  # darker = larger
                ax.add_patch(Rectangle((x, y + 0.7 + 0.21 * k), 1.6, 0.19,
                                       facecolor=(grey,) * 3, edgecolor="none", alpha=shown))

    def attention(self, ax, t, layer):
        """Bars of how much ``[CLS]`` takes from each token in ``layer``."""
        layout, trace = self.layout, self.trace
        self._message(ax, alpha=0.5)
        self._rows(ax, cls_edge=INK)
        weights = trace.cls_attention[layer]
        grow = ease(t * 1.4)
        for (x, y), w in zip(layout.home, weights, strict=True):
            height = 1.7 * w / weights.max() * grow
            ax.add_patch(Rectangle((x, y + 0.65), 1.2, height, facecolor=INK_2,
                                   edgecolor="none"))
        if t > 0.7:
            for i in np.argsort(weights)[::-1][:3]:
                x, y = layout.home[i]
                ax.text(x + 1.6, y + 1.9, f"{weights[i]:.1%}", fontsize=FONT - 2, color=INK,
                        va="center")
        ax.text(0.5, layout.below + 0.8, f"[CLS] reads from each token (bars, sum = 100%); "
                f"the largest share goes to “{trace.words[int(weights.argmax())]}”",
                fontsize=FONT - 1, color=INK_2, va="center", alpha=grow)

    def route(self, ax, t, layer):
        """Every token flying into the expert that processes it in ``layer``."""
        self._message(ax, alpha=0.5)
        self._bins(ax, alpha=ease(3 * t))
        self._fly(ax, self.layout.home, self.layout.bins[layer], self.trace.experts[layer], t,
                  highlight_cls=layer == len(self.trace.experts) - 1 and t > 0.8)

    def back(self, ax, t):
        """Layer 1's tokens flying back, still in their experts' colors."""
        self._message(ax, alpha=0.5)
        self._bins(ax, alpha=1 - ease(t))
        self._fly(ax, self.layout.bins[0], self.layout.home, self.trace.experts[0], t,
                  stagger=0.2, coloured=True)

    def classifier(self, ax, t):
        """The final ``[CLS]`` turned into p(spam), and the verdict."""
        layout, trace = self.layout, self.trace
        self._message(ax, alpha=0.5)
        for i, (word, xy) in enumerate(zip(trace.words, layout.home, strict=True)):
            if i:
                _token(ax, xy, word, alpha=0.3)
        move = ease(t / 0.35)
        target = (2.0, layout.below + 3.0)
        start = layout.home[0]
        _token(ax, (start[0] + move * (target[0] - start[0]),
                    start[1] + move * (target[1] - start[1])),
               "[CLS]", edge=INK, lw=2.5, zorder=5)
        if t > 0.3:
            steps = f"→  LayerNorm  →  Linear({trace.d_model} → 1)  →  sigmoid  →  p(spam)"
            ax.text(target[0] + 7, target[1], steps, fontsize=FONT, color=INK, va="center",
                    alpha=ease((t - 0.3) / 0.15))
        if t > 0.45:
            self._gauge(ax, trace.p_spam * ease((t - 0.45) / 0.3), target[1] + 3.5,
                        verdict=t > 0.8)

    def _message(self, ax, alpha=1.0):
        for i, line in enumerate(self.layout.text_lines):
            ax.text(0.5, 0.6 + i, line, fontsize=FONT + 1, color=INK_2, va="center",
                    alpha=alpha, style="italic")

    def _rows(self, ax, alpha=1.0, cls_edge=AXIS):
        for i, (word, xy) in enumerate(zip(self.trace.words, self.layout.home, strict=True)):
            _token(ax, xy, word, edge=cls_edge if i == 0 else AXIS, lw=2 if i == 0 else 1,
                   alpha=alpha)

    def _bins(self, ax, alpha=1.0):
        for e, (x0, y0, x1, y1) in enumerate(self.layout.bin_boxes):
            color = EXPERT_COLORS[e]
            ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                        boxstyle="round,pad=0,rounding_size=0.6",
                                        facecolor=color, edgecolor=color, lw=2,
                                        alpha=0.08 * alpha))
            ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                        boxstyle="round,pad=0,rounding_size=0.6",
                                        facecolor="none", edgecolor=color, lw=2, alpha=alpha))
            ax.text(x0 + 1, y0 + 0.9, f"expert {e + 1}", fontsize=FONT, weight="bold",
                    color=INK, va="center", alpha=alpha)

    def _fly(self, ax, starts, ends, experts, t, stagger=0.45, highlight_cls=False,
             coloured=False):
        """Moves every token from ``starts`` to ``ends``, one after another.

        A token takes its expert's color as it leaves, or has it from the start with
        ``coloured=True``.
        """
        n = len(self.trace.words)
        for i, word in enumerate(self.trace.words):
            local = ease((t - stagger * i / n) / (1 - stagger))
            (xa, ya), (xb, yb) = starts[i], ends[i]
            fill = EXPERT_COLORS[experts[i]] if coloured or local > 0 else SURFACE
            cls = highlight_cls and i == 0
            _token(ax, (xa + local * (xb - xa), ya + local * (yb - ya)), word, fill=fill,
                   edge=INK if cls else (AXIS if fill == SURFACE else fill),
                   lw=2.5 if cls else 1, zorder=4)

    def _gauge(self, ax, p, y, verdict):
        x0, x1 = 2.0, self.layout.width * 0.72
        ax.add_patch(Rectangle((x0, y), x1 - x0, 1.0, facecolor=HIGHLIGHT, edgecolor=AXIS))
        ax.add_patch(Rectangle((x0, y), (x1 - x0) * p, 1.0, facecolor=INK_2, edgecolor="none"))
        mid = (x0 + x1) / 2
        ax.plot([mid, mid], [y - 0.3, y + 1.3], color=INK, lw=1.2)
        ax.text(mid, y - 0.6, "0.5", ha="center", fontsize=FONT - 2, color=INK_2)
        ax.text(x0, y + 1.8, "0  ham", fontsize=FONT - 1, color=INK_2, va="center")
        ax.text(x1, y + 1.8, "spam  1", fontsize=FONT - 1, color=INK_2, va="center", ha="right")
        ax.text(x1 + 2, y + 0.5, f"p(spam) = {p:.3f}", fontsize=FONT + 2, color=INK,
                va="center")
        if verdict:
            label = "spam" if self.trace.p_spam > 0.5 else "ham"
            ax.text(x0, y + 3.6, f"The model says: {label}", fontsize=FONT + 8, weight="bold",
                    color=INK, va="center")


def _token(ax, xy, word, fill=SURFACE, edge=AXIS, lw=1.0, alpha=1.0, zorder=3):
    color = INK if fill == SURFACE else ink_on(fill)
    ax.text(*xy, word, family=MONO, fontsize=FONT, va="center", color=color, alpha=alpha,
            zorder=zorder, bbox={"boxstyle": "round,pad=0.25", "facecolor": fill,
                                 "edgecolor": edge, "linewidth": lw, "alpha": alpha})


# step -> (drawing, layer it is about)
STEPS = {
    "text": (ArchitectureScenes.text, None),
    "tokenize": (ArchitectureScenes.tokenize, None),
    "embed": (ArchitectureScenes.embed, None),
    "attn1": (ArchitectureScenes.attention, 0),
    "route1": (ArchitectureScenes.route, 0),
    "back1": (ArchitectureScenes.back, None),
    "attn2": (ArchitectureScenes.attention, 1),
    "route2": (ArchitectureScenes.route, 1),
    "classify": (ArchitectureScenes.classifier, None),
}
