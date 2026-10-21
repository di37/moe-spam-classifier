"""The README's diagram of the model, drawn left to right, part by part, with parts lit.

Coordinates are in a 100 × 19.4 box whose units are square on screen.
"""

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

from moe_spam.viz.style import AXIS, EXPERT_COLORS, HIGHLIGHT, INK, INK_2, MUTED, SURFACE, blend

WIDTH, HEIGHT = 100, 19.4
MID = 7.5           # the height of the main line
HALF = 2.2          # half the height of a box
FRAME = (43.0, 76.0, 0.8, 14.0)  # the encoder block's frame: x0, x1, y0, y1
ROUTER = (60.5, 2.6)             # the router diamond's x and half-size
EXPERT_X = (66.0, 74.5)
EXPERT_Y = (11.8, MID, 3.2)      # expert 1 on top
EXPERT_HALF = 1.45
LANE_X = 75.3       # the path between the experts and the frame's right side
LOOP = ((71.0, 14.2), (48.0, 14.2), 0.18)  # the "×2" arrow: from, to, curvature
BOXES = {  # piece: (x0, x1, label)
    "text": (0.5, 9.5, "SMS text"),
    "tokens": (12.5, 24.5, "[CLS] + word\ntokens"),
    "embed": (27.5, 40.0, "token + position\nembeddings"),
    "attn": (45.5, 55.5, "self-attention"),
    "final": (78.5, 88.0, "final [CLS]"),
    "clf": (91.0, 99.5, "classifier\n→ p(spam)"),
}
# The order the diagram is built in; each piece comes with the arrow that leads to it.
PIECES = ("text", "tokens", "embed", "block", "attn", "router", "experts", "loop", "final",
          "clf")
ARROWS = {"tokens": (9.5, 12.5), "embed": (24.5, 27.5), "attn": (40.0, 45.5),
          "router": (55.5, ROUTER[0] - ROUTER[1]), "final": (FRAME[1], 78.5),
          "clf": (88.0, 91.0)}


class Flowchart:
    """Draws the diagram; ``path`` is the way a token travels through it."""

    @classmethod
    def draw(cls, ax, shown=None, lit=(), chosen=0, layer=None, all_experts=False):
        """Draws the diagram.

        Args:
            ax: The axes to draw on.
            shown: How many pieces (in ``PIECES`` order) are drawn; a fraction fades the last
                one in. None draws them all.
            lit: The pieces to highlight.
            chosen: The expert the router's solid "top-1" arrow goes to; when the experts are
                lit, only this one is (unless ``all_experts``).
            layer: If given, the block's label says which layer (1 or 2) is running.
            all_experts: Whether lit experts are all lit, not only the chosen one.
        """
        ax.set_xlim(0, WIDTH)
        ax.set_ylim(0, HEIGHT)
        ax.axis("off")
        shown = len(PIECES) if shown is None else shown
        for k, piece in enumerate(PIECES):
            alpha = float(np.clip(shown - k, 0, 1))
            if alpha > 0:
                cls._piece(ax, piece, alpha, piece in lit, chosen, layer, all_experts)

    @classmethod
    def path(cls, experts):
        """Returns the waypoints ``[(x, y, piece)]`` of a token through both layers.

        Args:
            experts: The expert the token goes to in layer 1 and in layer 2.
        """
        (lx0, ly0), (lx1, ly1), rad = LOOP
        cx, cy = (lx0 + lx1) / 2, (ly0 + ly1) / 2 + rad * abs(lx1 - lx0)
        loop = [((1 - s) ** 2 * lx0 + 2 * s * (1 - s) * cx + s ** 2 * lx1,
                 (1 - s) ** 2 * ly0 + 2 * s * (1 - s) * cy + s ** 2 * ly1, "loop")
                for s in np.linspace(0, 1, 9)]
        start = [(cls._centre(p), MID, p) for p in ("text", "tokens", "embed")]
        layer = [(cls._centre("attn"), MID, "attn"), (ROUTER[0], MID, "router")]
        centre = (EXPERT_X[0] + EXPERT_X[1]) / 2
        first, second = ((centre, EXPERT_Y[e], "experts") for e in experts)
        # Out of an expert by the lane between the boxes and the frame, never across a box.
        up = [(LANE_X, EXPERT_Y[experts[0]], "loop"), (LANE_X, ly0, "loop")]
        end = [(LANE_X, EXPERT_Y[experts[1]], "final"), (LANE_X, MID, "final"),
               (FRAME[1], MID, "final"), (cls._centre("final"), MID, "final"),
               (cls._centre("clf"), MID, "clf")]
        return [*start, *layer, first, *up, *loop, *layer, second, *end]

    @staticmethod
    def _centre(piece):
        x0, x1, _ = BOXES[piece]
        return (x0 + x1) / 2

    @classmethod
    def _piece(cls, ax, piece, alpha, lit, chosen, layer, all_experts):
        if piece in ARROWS:
            _arrow(ax, *ARROWS[piece], alpha)
        if piece in BOXES:
            x0, x1, label = BOXES[piece]
            _box(ax, x0, x1, MID, HALF, label, alpha, lit)
        elif piece == "block":
            x0, x1, y0, y1 = FRAME
            ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                        boxstyle="round,pad=0,rounding_size=0.8",
                                        facecolor="none", edgecolor=INK if lit else AXIS,
                                        lw=1.6 if lit else 1.1, alpha=alpha))
            name = "encoder block (×2)" if layer is None else f"encoder block: layer {layer} of 2"
            ax.text(x0 + 0.8, y1 - 0.5, name, fontsize=11, weight="bold", va="top",
                    color=INK if lit or layer else INK_2, alpha=alpha)
        elif piece == "router":
            cls._router(ax, alpha, lit)
        elif piece == "experts":
            cls._experts(ax, alpha, lit, chosen, all_experts)
        elif piece == "loop":
            (lx0, ly0), (lx1, ly1), rad = LOOP
            ax.add_patch(FancyArrowPatch((lx0, ly0), (lx1, ly1),
                                         connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                         mutation_scale=12, color=INK if lit else MUTED,
                                         lw=1.4 if lit else 1.1, alpha=alpha))
            ax.text((lx0 + lx1) / 2, ly0 + 2.4, "×2", ha="center", va="bottom", fontsize=11,
                    weight="bold", color=INK if lit else INK_2, alpha=alpha)

    @staticmethod
    def _router(ax, alpha, lit):
        x, h = ROUTER
        ax.add_patch(Polygon([(x - h, MID), (x, MID + h), (x + h, MID), (x, MID - h)],
                             facecolor=HIGHLIGHT if lit else SURFACE,
                             edgecolor=INK if lit else AXIS, lw=2 if lit else 1.1, alpha=alpha))
        ax.text(x, MID, "router", ha="center", va="center", fontsize=10,
                color=INK if lit else INK_2, weight="bold" if lit else "normal", alpha=alpha)
        ax.text(x, MID - h - 0.5, "top-1", ha="center", va="top", fontsize=9, color=INK_2,
                alpha=alpha)

    @staticmethod
    def _experts(ax, alpha, lit, chosen, all_experts):
        x0, x1 = EXPERT_X
        for e, (y, color) in enumerate(zip(EXPERT_Y, EXPERT_COLORS, strict=True)):
            on = e == chosen or (lit and all_experts)
            ax.add_patch(FancyArrowPatch((ROUTER[0] + ROUTER[1], MID), (x0, y),
                                         arrowstyle="-|>", mutation_scale=11,
                                         color=INK if on else MUTED, lw=1.6 if on else 1,
                                         linestyle="-" if on else (0, (3, 3)), alpha=alpha))
            bright = lit and on
            ax.add_patch(FancyBboxPatch((x0, y - EXPERT_HALF), x1 - x0, 2 * EXPERT_HALF,
                                        boxstyle="round,pad=0,rounding_size=0.4",
                                        facecolor=color if bright else blend(color, 0.3),
                                        edgecolor="none", alpha=alpha))
            ax.text((x0 + x1) / 2, y, f"expert {e + 1}", ha="center", va="center",
                    fontsize=10, color="white" if bright else INK, alpha=alpha)


def _arrow(ax, x0, x1, alpha):
    ax.add_patch(FancyArrowPatch((x0, MID), (x1, MID), arrowstyle="-|>", mutation_scale=12,
                                 color=MUTED, lw=1.2, alpha=alpha))


def _box(ax, x0, x1, y, half, label, alpha, lit):
    ax.add_patch(FancyBboxPatch((x0, y - half), x1 - x0, 2 * half,
                                boxstyle="round,pad=0,rounding_size=0.5",
                                facecolor=HIGHLIGHT if lit else SURFACE,
                                edgecolor=INK if lit else AXIS, lw=2 if lit else 1.1,
                                alpha=alpha))
    ax.text((x0 + x1) / 2, y, label, ha="center", va="center", fontsize=11,
            color=INK if lit else INK_2, weight="bold" if lit else "normal", alpha=alpha,
            linespacing=1.15)
