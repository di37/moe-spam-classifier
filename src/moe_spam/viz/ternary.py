"""Router probabilities over three experts, drawn as points in a triangle.

Each corner is an expert: a point at a corner means everything goes to that expert, and the
centre means a three-way tie. Expert ``i``'s territory, where ``p_i`` is the largest, is the
kite between corner ``i``, the midpoints of its two edges and the centre.
"""

import math

import numpy as np
from matplotlib.patches import Polygon

from moe_spam.viz.style import AXIS, ERROR, EXPERT_COLORS, INK, INK_2, MUTED, SURFACE, ink_on

TERRITORY_ALPHA = 0.14


class Ternary:
    """The routing triangle's geometry and its two drawings.

    The drawings are the training video's panel and the architecture video's small inset.

    Attributes:
        CORNERS: The corners of experts 1, 2 and 3.
        CENTRE: Where all three get a third.
    """

    CORNERS = np.array([[0.5, math.sqrt(3) / 2], [0.0, 0.0], [1.0, 0.0]])
    CENTRE = CORNERS.mean(axis=0)
    CORNER_ARROWS = ("↑", "↙", "↘")  # the direction of each corner from the centre
    # When zoomed in, each expert's label sits at the edge of the view in its corner's
    # direction (top middle, bottom left, bottom right): (offset in half-widths, ha, va).
    ZOOMED_LABELS = (((0, 0.96), "center", "top"), ((-0.96, -0.96), "left", "bottom"),
                     ((0.96, -0.96), "right", "bottom"))
    # The zoomed view is a square around the centre, which sits CENTRE[1] above the base:
    # any wider and it would reach below the triangle, so the whole triangle is shown.
    FULL_VIEW = CENTRE[1]
    MIN_VIEW = 0.12   # never zoom in further than this (half-width around the centre)
    VIEW_MARGIN = 1.2

    @classmethod
    def xy(cls, probs):
        """Returns ``[n, 2]`` points: the probability-weighted average of the corners.

        Raises:
            ValueError: If the probabilities are not over 3 experts.
        """
        probs = np.asarray(probs, dtype=float)
        if probs.shape[-1] != 3:
            raise ValueError(f"the ternary plot needs 3 experts, got {probs.shape[-1]}")
        return probs @ cls.CORNERS

    @classmethod
    def view(cls, probs_per_frame):
        """Returns the half-width of a square view, around the centre, that fits every frame.

        A whole video shares one view. None means the whole triangle.

        Raises:
            ValueError: If there are no frames.
        """
        if not probs_per_frame:
            raise ValueError("no frames to fit the view to (was the run trained with no_video?)")
        reach = max(np.abs(cls.xy(p) - cls.CENTRE).max() for p in probs_per_frame)
        half_width = max(cls.MIN_VIEW, cls.VIEW_MARGIN * reach)
        return None if half_width >= cls.FULL_VIEW else half_width

    @classmethod
    def midpoint(cls, i, j):
        """Returns the middle of the edge between corners ``i`` and ``j``."""
        return (cls.CORNERS[i] + cls.CORNERS[j]) / 2

    @classmethod
    def kite(cls, i):
        """Returns the corners of expert ``i``'s territory."""
        j, k = (n for n in range(3) if n != i)
        return [cls.CORNERS[i], cls.midpoint(i, j), cls.CENTRE, cls.midpoint(i, k)]

    @classmethod
    def draw(cls, ax, probs, is_spam, wrong, half_width=None):
        """Draws the training video's panel: one point per test message.

        Ham are hollow circles, spam filled triangles drawn on top, mistakes red crosses.

        Args:
            ax: The axes to draw on.
            probs: ``[n, 3]`` router probabilities of each message's ``[CLS]``.
            is_spam: ``[n]`` bool, the true labels.
            wrong: ``[n]`` bool, the misclassified messages.
            half_width: From ``view``; None shows the whole triangle.
        """
        cls._draw_territories(ax)
        cls._label_corners(ax, half_width)
        xy = cls.xy(probs)
        ham, spam, bad = xy[~is_spam], xy[is_spam], xy[wrong]
        ax.scatter(ham[:, 0], ham[:, 1], s=18, facecolors="none", edgecolors=INK_2,
                   linewidths=0.7, label="ham message", zorder=2)
        ax.scatter(spam[:, 0], spam[:, 1], s=34, marker="^", color=INK, edgecolors=SURFACE,
                   linewidths=0.6, label="spam message", zorder=3)
        ax.scatter(bad[:, 0], bad[:, 1], s=70, marker="x", color=ERROR, linewidths=1.6,
                   label="misclassified", zorder=4)
        if half_width is None:
            ax.set_xlim(-0.06, 1.06)
            ax.set_ylim(-0.1, 0.98)
        else:
            ax.set_xlim(cls.CENTRE[0] - half_width, cls.CENTRE[0] + half_width)
            ax.set_ylim(cls.CENTRE[1] - half_width, cls.CENTRE[1] + half_width)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.legend(loc="upper left", fontsize=10, frameon=False, labelcolor=INK_2)

    @classmethod
    def draw_inset(cls, ax, probs):
        """Draws one ``[CLS]`` decision as a point in a small triangle.

        This is the architecture video's inset; the router's three probabilities are written
        underneath.
        """
        for i, color in enumerate(EXPERT_COLORS):
            ax.add_patch(Polygon(cls.kite(i), facecolor=color, alpha=TERRITORY_ALPHA * 1.6,
                                 edgecolor="none"))
            x, y = cls.CORNERS[i]
            ax.scatter(x, y, s=170, color=color, edgecolors=SURFACE, linewidths=2, zorder=3)
            ax.text(x, y, str(i + 1), ha="center", va="center", fontsize=9,
                    color=ink_on(color), zorder=4, weight="bold")
        ax.add_patch(Polygon(cls.CORNERS, fill=False, edgecolor=AXIS, lw=1))
        ax.scatter(*cls.CENTRE, marker="+", s=60, color=MUTED, lw=1)
        x, y = cls.xy(probs[None])[0]
        ax.scatter(x, y, s=90, color=INK, edgecolors=SURFACE, linewidths=2, zorder=5)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.32, 0.98)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(0.5, -0.17, "router: " + "  ".join(f"{p:.2f}" for p in probs), ha="center",
                fontsize=10, color=INK)
        ax.text(0.5, -0.29, "(experts 1, 2, 3)", ha="center", fontsize=9, color=INK_2)

    @classmethod
    def _draw_territories(cls, ax):
        """The kites, their borders and the lines where one expert gets half.

        The line joining the midpoints of corner ``i``'s two edges is where ``p_i = 1/2``,
        so the inner triangle they make is where no expert gets more than half.
        """
        for i, color in enumerate(EXPERT_COLORS):
            j, k = (n for n in range(3) if n != i)
            mid_ij, mid_ik = cls.midpoint(i, j), cls.midpoint(i, k)
            ax.add_patch(Polygon(cls.kite(i), facecolor=color, alpha=TERRITORY_ALPHA,
                                 edgecolor="none", zorder=0))
            mid_jk = cls.midpoint(j, k)  # the border between j's and k's territories
            ax.plot([cls.CENTRE[0], mid_jk[0]], [cls.CENTRE[1], mid_jk[1]], color=AXIS, lw=0.8,
                    zorder=1)
            ax.plot([mid_ij[0], mid_ik[0]], [mid_ij[1], mid_ik[1]], color=MUTED, lw=0.8,
                    zorder=1)
            half_mark = (mid_ij + mid_ik) / 2 + 0.02 * (cls.CORNERS[i] - cls.CENTRE)
            ax.text(*half_mark, f"50% to expert {i + 1}", ha="center", va="center", fontsize=9,
                    color=INK_2, rotation=(0, -60, 60)[i], zorder=3.5, clip_on=True,
                    bbox={"boxstyle": "round,pad=0.15", "facecolor": SURFACE,
                          "edgecolor": "none", "alpha": 0.75})
        ax.add_patch(Polygon(cls.CORNERS, fill=False, edgecolor=AXIS, lw=1, zorder=1))
        ax.scatter(*cls.CENTRE, marker="+", s=90, color=MUTED, lw=1.2, zorder=1,
                   label="centre: ⅓ to each expert")

    @classmethod
    def _label_corners(cls, ax, half_width):
        """Each expert's corner or, when zoomed in, an arrow at the edge pointing to it."""
        for i, color in enumerate(EXPERT_COLORS):
            if half_width is None:
                x, y = cls.CORNERS[i]
                ax.scatter(x, y, s=110, color=color, edgecolors=SURFACE, linewidths=2, zorder=5)
                y += 0.06 if y > 0 else -0.055  # above the top corner, below the others
                text, ha, va = f"expert {i + 1}", "center", "center"
            else:
                offset, ha, va = cls.ZOOMED_LABELS[i]
                x, y = cls.CENTRE + half_width * np.array(offset)
                text = f"{cls.CORNER_ARROWS[i]} expert {i + 1}"
            ax.text(x, y, text, ha=ha, va=va, fontsize=12, color=INK, weight="bold", zorder=6,
                    bbox={"boxstyle": "round,pad=0.3", "facecolor": SURFACE, "edgecolor": color,
                          "linewidth": 2})
