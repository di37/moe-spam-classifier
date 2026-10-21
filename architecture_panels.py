"""
The drawings of the architecture video: a map of the model with the current step lit up,
and what that step does to one message.

  draw_map         the model as a stack of blocks, layer 1 and layer 2 boxed and labelled
  TokenLayout      where each token box sits: in the token rows, and inside its expert's box
  token_layout     work that out from the message alone, so nothing moves between frames,
                   and refuse a message that does not fit
  draw_text ... draw_classifier   one drawing per step, each taking a progress t in [0, 1]
  draw_router_triangle            the layer-2 [CLS] decision as a point in the triangle

The detail panel is measured in characters and lines of the monospace font, as in
panels.draw_tokens, so token boxes line up whatever the message.
"""
import textwrap
from dataclasses import dataclass

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

from panels import (
    AXIS,
    CENTRE,
    CHAR_EM,
    CORNERS,
    EXPERT_COLORS,
    INK,
    INK_2,
    LINE_EM,
    MONO,
    MUTED,
    SURFACE,
    TERRITORY_ALPHA,
    ink_on,
    layout_tokens,
    ternary_xy,
)

FONT = 12
HIGHLIGHT = "#f0efec"  # neutral fill for the step being shown
ROW_GAP = 2            # characters between token boxes
ROW_STEP = 3.0         # lines from one token row to the next, with room underneath
BIN_ROW_STEP = 1.5     # lines between token rows inside an expert's box
VECTOR_CELLS = 8       # embedding numbers drawn under each token
CLASSIFIER_ROOM = 10.6  # lines the classifier step needs below the token rows
BINS_TOP = 9.6         # the expert boxes start no higher than this line, which leaves
                       # room above them for the layer-2 triangle

# The map: (step, title, subtitle, y centre in axes coordinates).
MAP_BLOCKS = (
    ("text", "Message", "an SMS, as typed", 0.95),
    ("tokenize", "Tokenizer", "[CLS] + words, numbers, punctuation", 0.865),
    ("embed", "Embeddings", "{d_model} numbers per token", 0.78),
    ("attn1", "Self-attention", "tokens share information", 0.64),
    ("route1", "Router + 3 experts", "one expert per token", 0.545),
    ("attn2", "Self-attention", "tokens share information", 0.355),
    ("route2", "Router + 3 experts", "one expert per token", 0.26),
    ("classify", "Classifier", "final [CLS] → p(spam)", 0.07),
)
# (y0, y1, title, where the training video shows this layer, in two short lines that stay
# left of the arrows)
MAP_LAYERS = ((0.445, 0.725, "LAYER 1", "in the training video:\nthe token panels"),
              (0.155, 0.435, "LAYER 2", "in the training video:\nthe triangle ([CLS])"))
MAP_ALIASES = {"back1": "route1"}  # steps drawn with another step's block lit
BLOCK_HALF_HEIGHT = 0.034


def draw_map(ax, step, d_model):
    lit = MAP_ALIASES.get(step, step)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for y0, y1, title, note in MAP_LAYERS:
        steps = [b for b in MAP_BLOCKS if y0 < b[3] < y1]
        active = any(b[0] == lit for b in steps)
        ax.add_patch(FancyBboxPatch((0.04, y0), 0.92, y1 - y0,
                                    boxstyle="round,pad=0,rounding_size=0.02", facecolor="none",
                                    edgecolor=INK if active else AXIS, lw=1.6 if active else 1))
        ax.text(0.07, y1 - 0.008, title, fontsize=11, weight="bold",
                color=INK if active else INK_2, va="top")
        ax.text(0.07, y0 + 0.008, note, fontsize=8.5, color=INK_2 if active else MUTED,
                linespacing=1.15)
    centres = [y for *_, y in MAP_BLOCKS]
    for ya, yb in zip(centres, centres[1:], strict=False):
        ax.add_patch(FancyArrowPatch((0.5, ya - BLOCK_HALF_HEIGHT), (0.5, yb + BLOCK_HALF_HEIGHT),
                                     arrowstyle="-|>", mutation_scale=10, color=MUTED, lw=1))
    for step_name, title, subtitle, y in MAP_BLOCKS:
        _map_block(ax, step_name == lit, step_name.startswith("route"), title,
                   subtitle.format(d_model=d_model), y)


def _map_block(ax, active, has_experts, title, subtitle, y):
    ax.add_patch(FancyBboxPatch((0.12, y - BLOCK_HALF_HEIGHT), 0.76, 2 * BLOCK_HALF_HEIGHT,
                                boxstyle="round,pad=0,rounding_size=0.015",
                                facecolor=HIGHLIGHT if active else SURFACE,
                                edgecolor=INK if active else AXIS, lw=2 if active else 1))
    ax.text(0.15, y + 0.009, title, fontsize=11, weight="bold" if active else "normal",
            color=INK if active else INK_2, va="center")
    ax.text(0.15, y - 0.016, subtitle, fontsize=8.5, color=INK_2 if active else MUTED,
            va="center")
    if has_experts:
        for e, color in enumerate(EXPERT_COLORS):
            x = 0.70 + 0.055 * e
            ax.add_patch(Rectangle((x, y - 0.014), 0.042, 0.028, facecolor=color,
                                   edgecolor="none", alpha=1 if active else 0.45))
            ax.text(x + 0.021, y, str(e + 1), ha="center", va="center", fontsize=8,
                    color=ink_on(color), alpha=1 if active else 0.6)


@dataclass(frozen=True)
class TokenLayout:
    width: float       # panel width, in characters
    height: float      # panel height, in lines
    text_lines: tuple  # the message, wrapped to the panel
    home: tuple        # (x, y) of each token in the token rows
    bins: tuple        # per layer: (x, y) of each token inside its expert's box
    bin_boxes: tuple   # (x0, y0, x1, y1) of each expert's box
    below: float       # first free line under the token rows


def panel_grid(ax):
    """Measure the axes in characters and lines; returns (width, height)."""
    box = ax.get_window_extent()
    width = box.width / (CHAR_EM * FONT / 72 * ax.figure.dpi)
    height = box.height / (LINE_EM * FONT / 72 * ax.figure.dpi)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    return width, height


def token_layout(text, words, experts_per_layer, width, height, n_experts=3):
    text_lines = tuple(textwrap.wrap(f"“{text}”", int(width) - 2))
    top = len(text_lines) + 1.6
    home = tuple((col + 0.5, top + line * ROW_STEP)
                 for line, col in layout_tokens(words, int(width) - 2, gap=ROW_GAP))
    below = max(y for _, y in home) + ROW_STEP - 0.6
    bin_w = (width - 1 - (n_experts - 1)) / n_experts
    x0s = [0.5 + e * (bin_w + 1) for e in range(n_experts)]
    y0 = max(below + 0.6, BINS_TOP)
    bins, lines = [], 1
    for experts in experts_per_layer:
        placed = [None] * len(words)
        for e, x0 in enumerate(x0s):
            mine = [i for i, ex in enumerate(experts) if ex == e]
            spots = layout_tokens([words[i] for i in mine], int(bin_w) - 2, gap=ROW_GAP)
            for i, (line, col) in zip(mine, spots, strict=True):
                placed[i] = (x0 + 1 + col, y0 + 2.2 + line * BIN_ROW_STEP)
                lines = max(lines, line + 1)
        bins.append(tuple(placed))
    y1 = y0 + 2.2 + lines * BIN_ROW_STEP + 0.2
    if below + CLASSIFIER_ROOM > height or y1 > height - 0.4:
        raise ValueError(f"a message of {len(words) - 1} tokens does not fit on screen")
    boxes = tuple((x0, y0, x0 + bin_w, y1) for x0 in x0s)
    return TokenLayout(width, height, text_lines, home, tuple(bins), boxes, below)


def _ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def _token(ax, xy, word, fill=SURFACE, edge=AXIS, lw=1.0, alpha=1.0, zorder=3):
    color = INK if fill == SURFACE else ink_on(fill)
    ax.text(*xy, word, family=MONO, fontsize=FONT, va="center", color=color, alpha=alpha,
            zorder=zorder, bbox={"boxstyle": "round,pad=0.25", "facecolor": fill,
                                 "edgecolor": edge, "linewidth": lw, "alpha": alpha})


def _message(ax, layout, alpha=1.0):
    for i, line in enumerate(layout.text_lines):
        ax.text(0.5, 0.6 + i, line, fontsize=FONT + 1, color=INK_2, va="center", alpha=alpha,
                style="italic")


def _rows(ax, trace, layout, alpha=1.0, cls_edge=AXIS):
    for i, (word, xy) in enumerate(zip(trace.words, layout.home, strict=True)):
        _token(ax, xy, word, edge=cls_edge if i == 0 else AXIS, lw=2 if i == 0 else 1,
               alpha=alpha)


def draw_text(ax, trace, layout, t):
    _message(ax, layout, alpha=_ease(2 * t))


def draw_tokenize(ax, trace, layout, t):
    _message(ax, layout)
    n = len(trace.words)
    for i, (word, token, xy) in enumerate(zip(trace.words, trace.tokens, layout.home,
                                              strict=True)):
        shown = _ease((t - 0.8 * i / n) / 0.2)  # one after another, all in by t = 1
        if shown <= 0:
            continue
        _token(ax, xy, word, alpha=shown, edge=INK if i == 0 else AXIS, lw=2 if i == 0 else 1)
        if token != word:  # [NUM], [NUM5+] or [UNK]: say what the model actually gets
            ax.text(xy[0], xy[1] + 1.0, token, family=MONO, fontsize=FONT - 3, color=INK_2,
                    alpha=shown, va="center")


def draw_embed(ax, trace, layout, t):
    _message(ax, layout)
    _rows(ax, trace, layout)
    values = trace.embeddings[:, :VECTOR_CELLS]
    low, high = values.min(), values.max()
    shown = _ease(t * 1.5)
    for (x, y), vector in zip(layout.home, values, strict=True):
        for k, v in enumerate(vector):
            grey = 0.92 - 0.8 * (v - low) / (high - low)  # darker = larger
            ax.add_patch(Rectangle((x, y + 0.7 + 0.21 * k), 1.6, 0.19, facecolor=(grey,) * 3,
                                   edgecolor="none", alpha=shown))


def draw_attention(ax, trace, layout, t, layer):
    _message(ax, layout, alpha=0.5)
    _rows(ax, trace, layout, cls_edge=INK)
    weights = trace.cls_attention[layer]
    grow = _ease(t * 1.4)
    for (x, y), w in zip(layout.home, weights, strict=True):
        height = 1.7 * w / weights.max() * grow
        ax.add_patch(Rectangle((x, y + 0.65), 1.2, height, facecolor=INK_2, edgecolor="none"))
    if t > 0.7:
        for i in np.argsort(weights)[::-1][:3]:
            x, y = layout.home[i]
            ax.text(x + 1.6, y + 1.9, f"{weights[i]:.1%}", fontsize=FONT - 2, color=INK,
                    va="center")
    ax.text(0.5, layout.below + 0.8, f"[CLS] reads from each token (bars, sum = 100%); the "
            f"largest share goes to “{trace.words[int(weights.argmax())]}”",
            fontsize=FONT - 1, color=INK_2, va="center", alpha=grow)


def _bins(ax, layout, alpha=1.0):
    for e, (x0, y0, x1, y1) in enumerate(layout.bin_boxes):
        color = EXPERT_COLORS[e]
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                    boxstyle="round,pad=0,rounding_size=0.6",
                                    facecolor=color, edgecolor=color, lw=2, alpha=0.08 * alpha))
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                    boxstyle="round,pad=0,rounding_size=0.6",
                                    facecolor="none", edgecolor=color, lw=2, alpha=alpha))
        ax.text(x0 + 1, y0 + 0.9, f"expert {e + 1}", fontsize=FONT, weight="bold", color=INK,
                va="center", alpha=alpha)


def _fly(ax, trace, starts, ends, experts, t, stagger=0.45, highlight_cls=False,
         coloured=False):
    """
    Move every token from starts to ends, one after another. A token takes its expert's color
    as it leaves (or has it from the start, with coloured=True).
    """
    n = len(trace.words)
    for i, word in enumerate(trace.words):
        local = _ease((t - stagger * i / n) / (1 - stagger))
        (xa, ya), (xb, yb) = starts[i], ends[i]
        fill = EXPERT_COLORS[experts[i]] if coloured or local > 0 else SURFACE
        cls = highlight_cls and i == 0
        _token(ax, (xa + local * (xb - xa), ya + local * (yb - ya)), word, fill=fill,
               edge=INK if cls else (AXIS if fill == SURFACE else fill), lw=2.5 if cls else 1,
               zorder=4)


def draw_route(ax, trace, layout, t, layer):
    _message(ax, layout, alpha=0.5)
    _bins(ax, layout, alpha=_ease(3 * t))
    _fly(ax, trace, layout.home, layout.bins[layer], trace.experts[layer], t,
         highlight_cls=layer == len(trace.experts) - 1 and t > 0.8)


def draw_back(ax, trace, layout, t):
    _message(ax, layout, alpha=0.5)
    _bins(ax, layout, alpha=1 - _ease(t))
    _fly(ax, trace, layout.bins[0], layout.home, trace.experts[0], t, stagger=0.2,
         coloured=True)


def draw_classifier(ax, trace, layout, t):
    _message(ax, layout, alpha=0.5)
    for i, (word, xy) in enumerate(zip(trace.words, layout.home, strict=True)):
        if i:
            _token(ax, xy, word, alpha=0.3)
    move = _ease(t / 0.35)
    target = (2.0, layout.below + 3.0)
    start = layout.home[0]
    _token(ax, (start[0] + move * (target[0] - start[0]), start[1] + move * (target[1] - start[1])),
           "[CLS]", edge=INK, lw=2.5, zorder=5)
    if t > 0.3:
        d_model = trace.embeddings.shape[1]
        steps = f"→  LayerNorm  →  Linear({d_model} → 1)  →  sigmoid  →  p(spam)"
        ax.text(target[0] + 7, target[1], steps, fontsize=FONT, color=INK, va="center",
                alpha=_ease((t - 0.3) / 0.15))
    if t > 0.45:
        _gauge(ax, trace.p_spam * _ease((t - 0.45) / 0.3), layout, target[1] + 3.5,
               verdict=t > 0.8, p_final=trace.p_spam)


def _gauge(ax, p, layout, y, verdict, p_final):
    x0, x1 = 2.0, layout.width * 0.72
    ax.add_patch(Rectangle((x0, y), x1 - x0, 1.0, facecolor=HIGHLIGHT, edgecolor=AXIS))
    ax.add_patch(Rectangle((x0, y), (x1 - x0) * p, 1.0, facecolor=INK_2, edgecolor="none"))
    mid = (x0 + x1) / 2
    ax.plot([mid, mid], [y - 0.3, y + 1.3], color=INK, lw=1.2)
    ax.text(mid, y - 0.6, "0.5", ha="center", fontsize=FONT - 2, color=INK_2)
    ax.text(x0, y + 1.8, "0  ham", fontsize=FONT - 1, color=INK_2, va="center")
    ax.text(x1, y + 1.8, "spam  1", fontsize=FONT - 1, color=INK_2, va="center", ha="right")
    ax.text(x1 + 2, y + 0.5, f"p(spam) = {p:.3f}", fontsize=FONT + 2, color=INK, va="center")
    if verdict:
        label = "spam" if p_final > 0.5 else "ham"
        ax.text(x0, y + 3.6, f"The model says: {label}", fontsize=FONT + 8, weight="bold",
                color=INK, va="center")


def draw_router_triangle(ax, probs):
    """The layer-2 [CLS] decision as a point in the training video's triangle."""
    for i, color in enumerate(EXPERT_COLORS):
        j, k = (n for n in range(3) if n != i)
        kite = [CORNERS[i], (CORNERS[i] + CORNERS[j]) / 2, CENTRE, (CORNERS[i] + CORNERS[k]) / 2]
        ax.add_patch(Polygon(kite, facecolor=color, alpha=TERRITORY_ALPHA * 1.6,
                             edgecolor="none"))
        x, y = CORNERS[i]
        ax.scatter(x, y, s=170, color=color, edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.text(x, y, str(i + 1), ha="center", va="center", fontsize=9, color=ink_on(color),
                zorder=4, weight="bold")
    ax.add_patch(Polygon(CORNERS, fill=False, edgecolor=AXIS, lw=1))
    ax.scatter(*CENTRE, marker="+", s=60, color=MUTED, lw=1)
    x, y = ternary_xy(probs[None])[0]
    ax.scatter(x, y, s=90, color=INK, edgecolors=SURFACE, linewidths=2, zorder=5)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.32, 0.98)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.5, -0.17, "router: " + "  ".join(f"{p:.2f}" for p in probs), ha="center",
            fontsize=10, color=INK)
    ax.text(0.5, -0.29, "(experts 1, 2, 3)", ha="center", fontsize=9, color=INK_2)
