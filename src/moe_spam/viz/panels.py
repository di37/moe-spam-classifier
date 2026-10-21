"""Drawing helpers shared by the videos' panels, each drawing on one matplotlib axes."""

import numpy as np
from matplotlib.patches import Patch, Rectangle

from moe_spam.viz.style import (
    CHAR_EM,
    EXPERT_COLORS,
    INK_2,
    LINE_EM,
    MONO,
    MUTED,
    SURFACE,
    blend,
    ink_on,
)


def layout_tokens(words, line_chars, gap=1):
    """Lays words out in lines of a fixed-width font.

    Args:
        words: The words, in order.
        line_chars: The width of a line, in characters.
        gap: Characters between words.

    Returns:
        ``(line, column)`` of each word, wrapping before a word that would cross the width.
    """
    positions, line, col = [], 0, 0
    for word in words:
        if col > 0 and col + len(word) > line_chars:
            line, col = line + 1, 0
        positions.append((line, col))
        col += len(word) + gap
    return tuple(positions)


def draw_tokens(ax, rows, fontsize=12):
    """Draws example messages with each token boxed in its expert's color.

    The axes is measured in characters and lines of the monospace font, so the layout only
    depends on the words and never moves between frames; only colors and p(spam) change.

    Args:
        ax: The axes to draw on.
        rows: ``TokenRow``-like objects (``label``, ``words``, ``experts``, ``p_spam``).
        fontsize: The size of the token text.
    """
    box = ax.get_window_extent()
    char_px = CHAR_EM * fontsize / 72 * ax.figure.dpi
    line_px = LINE_EM * fontsize / 72 * ax.figure.dpi
    line_chars = int(box.width // char_px)
    ax.set_xlim(0, line_chars)
    ax.set_ylim(box.height / line_px, 0)
    ax.axis("off")
    y = 0.0
    for row in rows:
        ax.text(0, y, f"{row.label} message  ·  model says p(spam) = {row.p_spam:5.3f}",
                fontsize=fontsize - 1, color=INK_2, va="top")
        y += 1.1
        positions = layout_tokens(row.words, line_chars - 1)
        for word, expert, (line, col) in zip(row.words, row.experts, positions, strict=True):
            color = EXPERT_COLORS[expert]
            ax.text(col + 0.25, y + line, word, family=MONO, fontsize=fontsize, va="top",
                    color=ink_on(color),
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": color, "edgecolor": "none"})
        y += positions[-1][0] + 1.8
    handles = [Patch(color=c, label=f"expert {i + 1}") for i, c in enumerate(EXPERT_COLORS)]
    ax.legend(handles=handles, loc="lower left", ncol=3, fontsize=10, frameon=False,
              labelcolor=INK_2)


def draw_shares(ax, shares, group_labels):
    """Draws a heatmap of how each kind of token is shared out among the experts.

    Args:
        ax: The axes to draw on.
        shares: ``[groups, experts]``, rows summing to 1 (NaN for a group with no tokens).
        group_labels: The row labels.
    """
    n_groups, n_experts = shares.shape
    for g in range(n_groups):
        for e in range(n_experts):
            share = shares[g, e]
            if np.isnan(share):
                ax.text(e + 0.5, g + 0.5, "–", ha="center", va="center", color=MUTED)
                continue
            fill = blend(EXPERT_COLORS[e], share)
            ax.add_patch(Rectangle((e + 0.03, g + 0.06), 0.94, 0.88, facecolor=fill,
                                   edgecolor="none"))
            ax.text(e + 0.5, g + 0.5, f"{share:.2f}", ha="center", va="center", fontsize=10,
                    color=ink_on(fill))
    ax.set_xlim(0, n_experts)
    ax.set_ylim(n_groups, 0)
    ax.set_xticks(np.arange(n_experts) + 0.5, [f"expert {e + 1}" for e in range(n_experts)])
    ax.set_yticks(np.arange(n_groups) + 0.5, group_labels)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, labelsize=10, labelcolor=INK_2)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_load(ax, steps, loads, total_steps):
    """Draws each expert's share of all tokens over training, stacked.

    Args:
        ax: The axes to draw on.
        steps: The logged steps.
        loads: ``[n_logged, experts]``: each expert's share of the real tokens.
        total_steps: The steps of the whole run (the x axis).
    """
    shares = np.asarray(loads, dtype=float)
    n_experts = shares.shape[1]
    ax.stackplot(steps, shares.T, colors=EXPERT_COLORS[:n_experts], edgecolor=SURFACE,
                 linewidth=0.8)
    for k in range(1, n_experts):  # where the borders would sit with a perfectly even split
        ax.axhline(k / n_experts, color=INK_2, lw=0.8, ls=(0, (2, 2)))
    ax.set_xlim(0, total_steps)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 1])


def draw_curve(ax, steps, values, color, label):
    """Draws a history as a line, with a dot on the latest point (visible even alone)."""
    ax.plot(steps, values, color=color, lw=1.5, label=label)
    ax.plot(steps[-1:], values[-1:], "o", color=color, ms=4)


def draw_cells(ax, x0, y0, values, cell_w, cell_h, cols, vmax, alpha=1.0):
    """Draws numbers as grey cells, ``cols`` per row from the top, darker for larger sizes.

    Args:
        ax: The axes to draw on.
        x0: The left of the first cell.
        y0: The bottom of the first row.
        values: The numbers.
        cell_w: A cell's width.
        cell_h: A cell's height.
        cols: Cells per row.
        vmax: The size drawn darkest.
        alpha: How far the cells have faded in.
    """
    for k, v in enumerate(values):
        row, col = divmod(k, cols)
        grey = 0.94 - 0.82 * min(1.0, abs(v) / vmax) if vmax > 0 else 0.94
        ax.add_patch(Rectangle((x0 + col * cell_w, y0 - row * cell_h), cell_w * 0.92,
                               cell_h * 0.92, facecolor=(grey,) * 3, edgecolor="none",
                               alpha=alpha))
