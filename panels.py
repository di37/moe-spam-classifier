"""
The drawing pieces of a training frame, each a plain function on one matplotlib axes.

  ternary_xy     router probabilities over 3 experts -> a point in a triangle whose corners
                 are the experts (a corner = everything to that expert, the centre = undecided)
  ternary_view   how far to zoom in on the centre so that every point of a run stays in view
  draw_ternary   the [CLS] routing of every test message, over the experts' territories
  layout_tokens  where each word goes when a message is printed in a fixed-width font
  draw_tokens    example messages, each token boxed in the color of the expert that ran it
  draw_shares    heatmap: the share of each token group sent to each expert
  draw_load      the share of all tokens each expert gets, stacked, over training
  ink_on         dark or white text, whichever reads better on a fill color

Colors: the first three slots of the dataviz reference palette, which stay distinguishable
together under color-vision deficiencies. Everything that is not an expert is drawn in ink.
"""
import math

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch, Polygon, Rectangle

EXPERT_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")  # blue, orange, aqua
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
ERROR = "#d03b3b"  # status "critical", used only for misclassified messages
TERRITORY_ALPHA = 0.14
CORNERS = np.array([[0.5, math.sqrt(3) / 2], [0.0, 0.0], [1.0, 0.0]])  # experts 1, 2, 3
CENTRE = CORNERS.mean(axis=0)
CORNER_ARROWS = ("↑", "↙", "↘")  # the direction of each corner from the centre
# When zoomed in, each expert's label sits at the edge of the view in its corner's direction
# (top middle, bottom left, bottom right): (offset in half-widths, ha, va).
ZOOMED_LABELS = (((0, 0.96), "center", "top"), ((-0.96, -0.96), "left", "bottom"),
                 ((0.96, -0.96), "right", "bottom"))
# The zoomed view is a square around the centre, which sits CENTRE[1] above the base: any
# wider and it would reach below the triangle, so show the whole triangle instead.
FULL_VIEW = CENTRE[1]
MIN_VIEW = 0.12   # never zoom in further than this (half-width around the centre)
VIEW_MARGIN = 1.2
MONO = "DejaVu Sans Mono"  # ships with matplotlib, so the layout is the same everywhere
CHAR_EM = 0.602            # advance width of one DejaVu Sans Mono character, in ems
LINE_EM = 2.0              # height of one line of token boxes, in ems


def _luminance(color):
    """WCAG relative luminance."""
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in to_rgb(color)]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(a, b):
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def ink_on(fill):
    return INK if _contrast(fill, INK) >= _contrast(fill, "white") else "white"


def _blend(color, amount):
    """`amount` of `color` over the surface, as an RGB tuple (0 = surface, 1 = color)."""
    return tuple(amount * c + (1 - amount) * s
                 for c, s in zip(to_rgb(color), to_rgb(SURFACE), strict=True))


def ternary_xy(probs):
    """[n, 3] probabilities -> [n, 2] points: the probability-weighted average of the corners."""
    probs = np.asarray(probs, dtype=float)
    if probs.shape[-1] != 3:
        raise ValueError(f"the ternary plot needs 3 experts, got {probs.shape[-1]}")
    return probs @ CORNERS


def ternary_view(probs_per_frame):
    """
    Half-width of a square view around the centre that holds every point of every frame, so
    the whole video can share one view; None when that is most of the triangle anyway.
    """
    reach = max(np.abs(ternary_xy(p) - CENTRE).max() for p in probs_per_frame)
    half_width = max(MIN_VIEW, VIEW_MARGIN * reach)
    return None if half_width >= FULL_VIEW else half_width


def _draw_territories(ax):
    """
    Expert i's territory is where p_i is the largest: the kite between corner i, the midpoints
    of its two edges and the centre. The line joining those two midpoints is where p_i = 1/2,
    so the inner triangle they make is where no expert gets more than half.
    """
    for i, color in enumerate(EXPERT_COLORS):
        j, k = (n for n in range(3) if n != i)
        mid_ij, mid_ik = (CORNERS[i] + CORNERS[j]) / 2, (CORNERS[i] + CORNERS[k]) / 2
        ax.add_patch(Polygon([CORNERS[i], mid_ij, CENTRE, mid_ik], facecolor=color,
                             alpha=TERRITORY_ALPHA, edgecolor="none", zorder=0))
        mid_jk = (CORNERS[j] + CORNERS[k]) / 2  # the border between j's and k's territories
        ax.plot([CENTRE[0], mid_jk[0]], [CENTRE[1], mid_jk[1]], color=AXIS, lw=0.8, zorder=1)
        ax.plot([mid_ij[0], mid_ik[0]], [mid_ij[1], mid_ik[1]], color=MUTED, lw=0.8, zorder=1)
        half_mark = (mid_ij + mid_ik) / 2 + 0.02 * (CORNERS[i] - CENTRE)
        ax.text(*half_mark, f"50% to expert {i + 1}", ha="center", va="center", fontsize=9,
                color=INK_2, rotation=(0, -60, 60)[i], zorder=3.5, clip_on=True,
                bbox={"boxstyle": "round,pad=0.15", "facecolor": SURFACE, "edgecolor": "none",
                      "alpha": 0.75})
    ax.add_patch(Polygon(CORNERS, fill=False, edgecolor=AXIS, lw=1, zorder=1))
    ax.scatter(*CENTRE, marker="+", s=90, color=MUTED, lw=1.2, zorder=1,
               label="centre: ⅓ to each expert")


def _label_corners(ax, half_width):
    """Each expert's corner, or, when zoomed in, an arrow at the edge pointing towards it."""
    for i, color in enumerate(EXPERT_COLORS):
        if half_width is None:
            x, y = CORNERS[i]
            ax.scatter(x, y, s=110, color=color, edgecolors=SURFACE, linewidths=2, zorder=5)
            y += 0.06 if y > 0 else -0.055  # above the top corner, below the others
            text, ha, va = f"expert {i + 1}", "center", "center"
        else:
            offset, ha, va = ZOOMED_LABELS[i]
            x, y = CENTRE + half_width * np.array(offset)
            text = f"{CORNER_ARROWS[i]} expert {i + 1}"
        ax.text(x, y, text, ha=ha, va=va, fontsize=12, color=INK, weight="bold", zorder=6,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": SURFACE, "edgecolor": color,
                      "linewidth": 2})


def draw_ternary(ax, probs, is_spam, wrong, half_width=None):
    """
    Ham as hollow circles, spam as filled triangles drawn on top, mistakes as red crosses.
    half_width: from ternary_view; None shows the whole triangle.
    """
    _draw_territories(ax)
    _label_corners(ax, half_width)
    xy = ternary_xy(probs)
    ham, spam, bad = xy[~is_spam], xy[is_spam], xy[wrong]
    ax.scatter(ham[:, 0], ham[:, 1], s=18, facecolors="none", edgecolors=INK_2, linewidths=0.7,
               label="ham message", zorder=2)
    ax.scatter(spam[:, 0], spam[:, 1], s=34, marker="^", color=INK, edgecolors=SURFACE,
               linewidths=0.6, label="spam message", zorder=3)
    ax.scatter(bad[:, 0], bad[:, 1], s=70, marker="x", color=ERROR, linewidths=1.6,
               label="misclassified", zorder=4)
    if half_width is None:
        ax.set_xlim(-0.06, 1.06)
        ax.set_ylim(-0.1, 0.98)
    else:
        ax.set_xlim(CENTRE[0] - half_width, CENTRE[0] + half_width)
        ax.set_ylim(CENTRE[1] - half_width, CENTRE[1] + half_width)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="upper left", fontsize=10, frameon=False, labelcolor=INK_2)


def layout_tokens(words, line_chars, gap=1):
    """(line, column) of each word, wrapping before a word that would cross line_chars."""
    positions, line, col = [], 0, 0
    for word in words:
        if col > 0 and col + len(word) > line_chars:
            line, col = line + 1, 0
        positions.append((line, col))
        col += len(word) + gap
    return tuple(positions)


def draw_tokens(ax, rows, fontsize=12):
    """
    rows: TokenRow-like objects (label, words, experts, p_spam). The axes is measured in
    characters and lines of the monospace font, so the layout only depends on the words and
    never moves between frames; only the colors and p(spam) change.
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
    """shares [groups, experts], rows summing to 1 (NaN: no tokens of that group)."""
    n_groups, n_experts = shares.shape
    for g in range(n_groups):
        for e in range(n_experts):
            share = shares[g, e]
            if np.isnan(share):
                ax.text(e + 0.5, g + 0.5, "–", ha="center", va="center", color=MUTED)
                continue
            fill = _blend(EXPERT_COLORS[e], share)
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
    """loads [n_logged, experts]: each expert's share of all real tokens at each logged step."""
    shares = np.asarray(loads, dtype=float)
    n_experts = shares.shape[1]
    ax.stackplot(steps, shares.T, colors=EXPERT_COLORS[:n_experts], edgecolor=SURFACE,
                 linewidth=0.8)
    for k in range(1, n_experts):  # where the borders would sit with a perfectly even split
        ax.axhline(k / n_experts, color=INK_2, lw=0.8, ls=(0, (2, 2)))
    ax.set_xlim(0, total_steps)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 1])
