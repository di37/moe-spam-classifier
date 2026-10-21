"""The [CLS] log beside the stack: one row per step, written as the token reaches it."""

import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

from moe_spam.viz.panels import draw_cells
from moe_spam.viz.style import AXIS, EXPERT_COLORS, HIGHLIGHT, INK, INK_2

LOG = (62.0, 157.0, 73.5, 9.6)  # left, right, top of the first row, height of a row
CHART_X, CHART_W = 122.0, 33.0  # where a row's chart goes, and its width
TITLE = "What happens to [CLS], step by step"


def draw_cls_log(ax, rows, shown, current):
    """Draws the log's title and its first rows.

    Args:
        ax: The full-frame axes.
        rows: The ``LogRow`` of every step, in order.
        shown: How many rows are drawn; a fraction fades the last one in.
        current: The key of the row to highlight (the step the token is at), or None.
    """
    left, right, top, height = LOG
    ax.text(left, top + 2.6, TITLE, va="center", fontsize=12.5, weight="bold", color=INK)
    for k, row in enumerate(rows):
        alpha = min(1.0, max(0.0, shown - k))
        if alpha <= 0:
            break
        y = top - height * k  # the top of this row
        if row.key == current:
            ax.add_patch(FancyBboxPatch((left - 0.8, y - height + 0.6), right - left + 1.6,
                                        height - 0.4, boxstyle="round,pad=0,rounding_size=0.8",
                                        facecolor=HIGHLIGHT, edgecolor=AXIS, alpha=alpha))
        ax.text(left + 0.6, y - 1.6, row.title, va="center", fontsize=11, weight="bold",
                color=INK, alpha=alpha)
        ax.text(left + 0.6, y - 4.4, row.text, va="center", fontsize=10.5, color=INK_2,
                alpha=alpha)
        if row.kind:
            _CHARTS[row.kind](ax, y - 4.2, row.values, alpha)


def _strip(ax, y, values, alpha):
    """Numbers as one row of grey cells."""
    values = np.asarray(values)
    draw_cells(ax, CHART_X, y - 1.2, values, CHART_W / len(values), 2.4, len(values),
               np.abs(values).max(), alpha)


def _bars(ax, y, probs, alpha):
    """The router's probabilities, one bar per expert; the chosen one outlined."""
    chosen = int(np.argmax(probs))
    for e, (p, color) in enumerate(zip(probs, EXPERT_COLORS, strict=True)):
        yy = y + 2.0 - 2.1 * e
        ax.text(CHART_X, yy, f"expert {e + 1}", va="center", fontsize=9, color=color,
                weight="bold", alpha=alpha)
        ax.add_patch(Rectangle((CHART_X + 7.5, yy - 0.75), 20.0 * p, 1.5, facecolor=color,
                               edgecolor=INK if e == chosen else "none", lw=1.4, alpha=alpha))
        ax.text(CHART_X + 8.2 + 20.0 * p, yy, f"{p:.2f}", va="center", fontsize=9, color=INK,
                alpha=alpha)


def _gauge(ax, y, values, alpha):
    """p(spam) as a bar from 0 (ham) to 1 (spam), with the 0.5 line."""
    (p,) = values
    x0, w = CHART_X + 4.0, CHART_W - 8.0
    ax.add_patch(Rectangle((x0, y - 1.0), w, 2.0, facecolor=HIGHLIGHT, edgecolor=AXIS,
                           alpha=alpha))
    ax.add_patch(Rectangle((x0, y - 1.0), w * p, 2.0, facecolor=INK_2, edgecolor="none",
                           alpha=alpha))
    ax.plot([x0 + w / 2] * 2, [y - 1.6, y + 1.6], color=INK, lw=1.2, alpha=alpha)
    ax.text(x0 - 0.6, y, "ham", ha="right", va="center", fontsize=9, color=INK_2, alpha=alpha)
    ax.text(x0 + w + 0.6, y, "spam", va="center", fontsize=9, color=INK_2, alpha=alpha)


_CHARTS = {"strip": _strip, "bars": _bars, "gauge": _gauge}
