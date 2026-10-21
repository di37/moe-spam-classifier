"""Shapes for diagrams drawn on one axes covering the frame: boxes, panels, arrows, notes.

The axes is 160 × 90 units (10 pixels each), and every shape fades in with ``alpha``.
"""

from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path

from moe_spam.viz.style import AXIS, HIGHLIGHT, INK, INK_2, MUTED, SURFACE, ink_on

WIDTH, HEIGHT = 160, 90
DOTS = (0, (1.5, 2.5))  # the dotted line of leaders and zoom panels


def box(ax, cx, cy, w, h, label, alpha=1.0, fill=SURFACE, lit=False, fontsize=11):
    """A rounded box centred on ``(cx, cy)``, with ``label`` in the middle."""
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=0.6",
                                facecolor=HIGHLIGHT if lit and fill == SURFACE else fill,
                                edgecolor=INK if lit else INK_2, lw=2.2 if lit else 1.0,
                                alpha=alpha, zorder=3))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fontsize, linespacing=1.15,
            color=ink_on(fill) if fill != SURFACE else INK, alpha=alpha, zorder=4,
            weight="bold" if lit else "normal")


def panel(ax, x0, y0, x1, y1, alpha=1.0, fill=SURFACE, edge=AXIS, dotted=False, lw=1.2):
    """A rounded panel behind a group of boxes; dotted for a zoomed-in part."""
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=1.6", facecolor=fill,
                                edgecolor=edge, lw=lw, linestyle=DOTS if dotted else "-",
                                alpha=alpha, zorder=1))


def arrow(ax, start, end, alpha=1.0, color=MUTED, lw=1.2, dashed=False, zorder=2):
    """An arrow from ``start`` to ``end``."""
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, color=color,
                                 lw=lw, linestyle=(0, (3, 3)) if dashed else "-", alpha=alpha,
                                 zorder=zorder))


def line(ax, points, alpha=1.0, color=MUTED, lw=1.2, dotted=False, zorder=2):
    """A plain line through ``points``."""
    xs, ys = zip(*points, strict=True)
    ax.plot(xs, ys, color=color, lw=lw, linestyle=DOTS if dotted else "-", alpha=alpha,
            zorder=zorder, solid_capstyle="round")


def plus(ax, cx, cy, r, alpha=1.0, lit=False):
    """The ⊕ where a residual connection is added back."""
    ax.add_patch(Circle((cx, cy), r, facecolor=SURFACE, edgecolor=INK if lit else INK_2,
                        lw=2 if lit else 1.2, alpha=alpha, zorder=4))
    line(ax, [(cx - r * 0.6, cy), (cx + r * 0.6, cy)], alpha, INK_2, 1.2, zorder=5)
    line(ax, [(cx, cy - r * 0.6), (cx, cy + r * 0.6)], alpha, INK_2, 1.2, zorder=5)


def note(ax, x, y, text, leader_to=None, alpha=1.0, ha="left", fontsize=10.5):
    """An annotation, with a dotted leader to ``leader_to`` if given."""
    ax.text(x, y, text, ha=ha, va="center", fontsize=fontsize, color=INK, alpha=alpha,
            linespacing=1.2, zorder=4)
    if leader_to is not None:
        start = (x + (0.6 if ha == "right" else -0.6), y) if ha != "center" else (x, y)
        line(ax, [start, leader_to], alpha, INK_2, 1.1, dotted=True, zorder=2)


def brace(ax, x, y0, y1, alpha=1.0, depth=1.2):
    """A curly brace on the left of ``x``, from ``y0`` up to ``y1``."""
    mid, d = (y0 + y1) / 2, depth
    vertices = [(x, y1), (x - d / 2, y1), (x - d / 2, y1 - d), (x - d / 2, mid + d),
                (x - d / 2, mid), (x - d, mid), (x - d / 2, mid), (x - d / 2, mid - d),
                (x - d / 2, y0 + d), (x - d / 2, y0), (x, y0)]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3,
             Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3]
    ax.add_patch(PathPatch(Path(vertices, codes), facecolor="none", edgecolor=INK, lw=1.6,
                           alpha=alpha, zorder=3))


def badge(ax, x, y, text, color, alpha=1.0, size=1.9):
    """A small filled square with ``text``, centred on ``(x, y)``: an expert's number."""
    ax.add_patch(FancyBboxPatch((x - size / 2, y - size / 2), size, size,
                                boxstyle="round,pad=0,rounding_size=0.25", facecolor=color,
                                edgecolor="none", alpha=alpha, zorder=5))
    ax.text(x, y, text, ha="center", va="center", fontsize=9.5, weight="bold",
            color=ink_on(color), alpha=alpha, zorder=6)
