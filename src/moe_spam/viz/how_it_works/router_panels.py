"""The routing equations, drawn with one token's real numbers.

Every drawing goes on the detail axes, a 100 × 25 box whose units are square on screen;
the left third holds the equations, the rest the drawing. Vectors are grids of cells,
darker for numbers larger in size (whatever their sign).
"""

import math

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from moe_spam.viz.style import AXIS, EXPERT_COLORS, INK, INK_2, MONO, MUTED, blend, ink_on
from moe_spam.viz.video import ease

WIDTH, HEIGHT = 100, 25
LEFT = 36            # where the drawings start; the equations are to the left of it
STRIP_X, CELL = 45, 0.7  # where the x and W_router strips start, and one number's width
BAR_X, BAR_LENGTH = 45, 40  # where the probability bars start, and the length of p = 1
SCORES_SHARE = 0.65  # the softmax scene scores the experts in this share of its animation
BALANCE_HEIGHT = 15  # the height of the tallest bar in the load-balancing scene


def setup(ax):
    """Sets the detail axes to the 100 × 25 box, without frame or ticks."""
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.axis("off")


def draw_equations(ax, equations, current, values):
    """Writes ``equations`` down the left, ``current`` in ink with ``values`` under it.

    Args:
        ax: The detail axes.
        equations: The equations, in order.
        current: The index of the one being shown, or None.
        values: Lines that put real numbers into the current equation.
    """
    y = HEIGHT - 2
    for k, equation in enumerate(equations):
        on = k == current
        ax.text(1, y, equation, family=MONO, fontsize=17, va="center",
                color=INK if on else MUTED, weight="bold" if on else "normal")
        y -= 3.2
        if on:
            for line in values:
                ax.text(2, y + 0.6, line, family=MONO, fontsize=12.5, va="center", color=INK_2)
                y -= 2.3
            y -= 0.6


def draw_softmax(ax, step, t):
    """``p = softmax(W_router · x)``: each row of weights scores one expert, then softmax."""
    vmax = np.abs(step.weights).max()  # the weights' own scale: they are much smaller than x
    ax.text(LEFT, 21.7, "x", family=MONO, fontsize=15, weight="bold", va="center", color=INK)
    _cells(ax, STRIP_X, 21.0, step.x, CELL, 1.4, len(step.x), np.abs(step.x).max())
    ax.text(STRIP_X, 23.4, f"the token “{step.word}” after self-attention: {len(step.x)} "
            f"numbers", fontsize=10.5, va="center", color=INK_2)
    ax.text(STRIP_X, 18.9, "W_router: one row of weights per expert", fontsize=10.5,
            color=INK_2, va="center")
    scored = t / SCORES_SHARE * len(step.probs)
    for e, (row, color) in enumerate(zip(step.weights, EXPERT_COLORS, strict=True)):
        y = 16.4 - 2.0 * e
        ax.text(LEFT, y + 0.7, f"expert {e + 1}", fontsize=11, va="center", color=color,
                weight="bold")
        _cells(ax, STRIP_X, y, row, CELL, 1.4, len(row), vmax)
        if e <= scored < e + 1:  # this row is being multiplied with x
            ax.add_patch(Rectangle((STRIP_X - 0.2, y - 0.2), CELL * len(row) + 0.4, 1.8,
                                   facecolor="none", edgecolor=INK, lw=1.6))
        if scored >= e + 1:
            ax.text(STRIP_X + CELL * len(row) + 1, y + 0.7, f"score {step.logits[e]:+.2f}",
                    family=MONO, fontsize=11, va="center", color=INK)
    grow = ease((t - SCORES_SHARE) / (1 - SCORES_SHARE))
    if grow > 0:
        ax.text(LEFT, 9.6, "softmax of the 3 scores:", fontsize=10.5, color=INK_2, va="center")
        draw_bars(ax, step.probs, grow)


def draw_bars(ax, probs, grow=1.0, winner=None):
    """The router's probabilities as one bar per expert, ``grow`` of their full length."""
    for e, (p, color) in enumerate(zip(probs, EXPERT_COLORS, strict=True)):
        y = 6.8 - 2.5 * e
        ax.text(LEFT, y + 0.8, f"p{e + 1}", family=MONO, fontsize=12, va="center", color=color,
                weight="bold")
        ax.add_patch(Rectangle((BAR_X, y), BAR_LENGTH * p * grow, 1.6, facecolor=color,
                               edgecolor=INK if e == winner else "none", lw=2))
        ax.text(BAR_X + BAR_LENGTH * p * grow + 0.8, y + 0.8, f"{p:.2f}", family=MONO,
                fontsize=11, va="center", color=INK)


def draw_argmax(ax, step, d_ff, t):
    """``i = argmax p``: the top bar's expert runs on the token; the others do no work."""
    i = step.expert
    ax.text(LEFT, 9.6, "the router's probabilities", fontsize=10.5, color=INK_2, va="center")
    draw_bars(ax, step.probs, winner=i)
    work = 2 * len(step.x) * d_ff
    for e, color in enumerate(EXPERT_COLORS):
        y, on = 17.5 - 5.2 * e, e == i
        ax.add_patch(FancyBboxPatch((80, y), 18, 4.0, boxstyle="round,pad=0,rounding_size=0.5",
                                    facecolor=color if on else blend(color, 0.15),
                                    edgecolor="none"))
        ax.text(89, y + 2.6, f"expert {e + 1}", ha="center", va="center", fontsize=12,
                weight="bold", color=ink_on(color) if on else INK_2)
        ax.text(89, y + 1.1, f"runs: {work:,} multiply-adds" if on else "skipped: no work",
                ha="center", va="center", fontsize=10, color=ink_on(color) if on else MUTED)
    reach = ease(t)
    if reach > 0:
        start = np.array([BAR_X + BAR_LENGTH * step.p_i + 4.5, 6.8 - 2.5 * i + 0.8])
        end = np.array([79.5, 17.5 - 5.2 * i + 2.0])
        ax.add_patch(FancyArrowPatch(tuple(start), tuple(start + reach * (end - start)),
                                     arrowstyle="-|>", mutation_scale=16, color=INK, lw=1.8))


def draw_scale(ax, step, t):
    """``y = p_i · FFN_i(x)``: the token through the chosen expert, then scaled by ``p_i``."""
    i, out_max = step.expert, np.abs(step.ffn_out).max()
    grids = ((step.x, f"x ({len(step.x)})", np.abs(step.x).max()),
             (step.hidden, f"hidden ({len(step.hidden)}), after GELU", np.abs(step.hidden).max()),
             (step.ffn_out, f"FFN_{i + 1}(x) ({len(step.ffn_out)})", out_max),
             (step.y, f"y = {step.p_i:.2f} · FFN_{i + 1}(x)", out_max))
    ax.text(LEFT, 21.5, f"expert {i + 1}: {len(step.x)} → {len(step.hidden)} → {len(step.x)}",
            fontsize=12, weight="bold", color=EXPERT_COLORS[i], va="center")
    shown = t * len(grids) * 1.1
    for k, (values, label, vmax) in enumerate(grids):
        if shown <= k:
            break
        x0 = LEFT + 16.5 * k
        _grid(ax, x0, 8.0, values, 9.6, vmax, alpha=min(1.0, shown - k))
        ax.text(x0 + 4.8, 6.6, label, ha="center", va="top", fontsize=10, color=INK_2)
        if k:
            ax.add_patch(FancyArrowPatch((x0 - 6.4, 12.8), (x0 - 0.5, 12.8), arrowstyle="-|>",
                                         mutation_scale=12, color=MUTED, lw=1.2))
            ax.text(x0 - 3.45, 13.9, ("", "W₁, GELU", "W₂", f"× {step.p_i:.2f}")[k],
                    ha="center", fontsize=10, color=INK if k == 3 else INK_2)
    if t >= 1:
        ax.text(LEFT, 2.4, "y is then added back to the token (a residual connection), and the "
                "token moves on.", fontsize=11, color=INK_2, va="center")


def draw_slopes(canvas, step, t, rects):
    """Two plots against expert i's score: ``i = argmax p`` jumps, ``p_i`` is smooth.

    Args:
        canvas: The frame's ``Canvas``.
        step: The ``RouterStep``.
        t: Progress through the scene.
        rects: Figure rectangles of the two plots.
    """
    i, scores = step.expert, step.logits
    others = np.delete(scores, i)
    z = np.linspace(scores[i] - 4, scores[i] + 4, 400)
    runner_up = int(np.delete(np.arange(len(scores)), i)[others.argmax()])
    picked = np.where(z > others.max(), i, runner_up) + 1  # 1-based, as the experts are named
    p_i = np.exp(z) / (np.exp(z) + np.exp(others).sum())
    n, marked = max(2, int(len(z) * ease(t * 1.4))), t >= 0.6

    ax = _slope_axes(canvas, rects[0], z, i, "i = argmax p: a step, with nothing to follow")
    ax.plot(z[:n], picked[:n], color=INK, lw=2, drawstyle="steps-post")
    ax.set_ylim(0.5, len(scores) + 0.5)
    ax.set_yticks(range(1, len(scores) + 1), [f"expert {e}" for e in range(1, len(scores) + 1)])
    if marked:
        ax.plot([scores[i]], [i + 1], "o", color=EXPERT_COLORS[i], ms=9, zorder=5)

    ax = _slope_axes(canvas, rects[1], z, i, "p_i: smooth, with a slope everywhere")
    ax.plot(z[:n], p_i[:n], color=INK, lw=2)
    ax.set_ylim(0, 1)
    if marked:
        slope = step.p_i * (1 - step.p_i)  # d p_i / d score_i
        ax.plot(z, step.p_i + slope * (z - scores[i]), color=EXPERT_COLORS[i], lw=1.2,
                ls=(0, (4, 3)))
        ax.plot([scores[i]], [step.p_i], "o", color=EXPERT_COLORS[i], ms=9, zorder=5)


def _slope_axes(canvas, rect, z, i, title):
    ax = canvas.axes(rect)
    ax.set_xlim(z[0], z[-1])
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(f"expert {i + 1}'s score (the others fixed)", fontsize=10)
    ax.grid(color=AXIS, lw=0.5, alpha=0.6)
    return ax


def draw_balance(ax, balance, t):
    """``f_i`` and ``P_i`` of every expert in each layer, and the ``aux`` they make."""
    grow = ease(t)
    scale = BALANCE_HEIGHT / max(max(s.load.max(), s.mean_probs.max()) for s in balance)
    for layer, stats in enumerate(balance):
        x0 = LEFT + 33 * layer
        ax.text(x0, 23.3, f"layer {layer + 1}", fontsize=12, weight="bold", color=INK,
                va="center")
        for e, color in enumerate(EXPERT_COLORS):
            x = x0 + 1 + 10 * e
            for k, (value, fill) in enumerate(((stats.load[e], color),
                                               (stats.mean_probs[e], blend(color, 0.45)))):
                height = scale * value * grow
                ax.add_patch(Rectangle((x + 3.6 * k, 4.5), 3.2, height, facecolor=fill,
                                       edgecolor="none"))
                ax.text(x + 3.6 * k + 1.6, 4.9 + height, f"{value:.2f}", ha="center",
                        fontsize=9, color=INK)
            ax.text(x + 3.4, 3.4, f"expert {e + 1}", ha="center", fontsize=10, color=INK_2,
                    va="top")
    ax.text(LEFT, 0.6, "dark: f_i, the share of tokens sent to expert i      light: P_i, the "
            "router's mean probability for it", fontsize=10, color=INK_2)


def _cells(ax, x0, y0, values, cell_w, cell_h, cols, vmax, alpha=1.0):
    """Draws ``values`` as cells, ``cols`` per row from the top, darker for larger sizes."""
    for k, v in enumerate(values):
        row, col = divmod(k, cols)
        grey = 0.94 - 0.82 * min(1.0, abs(v) / vmax) if vmax > 0 else 0.94
        ax.add_patch(Rectangle((x0 + col * cell_w, y0 - row * cell_h), cell_w * 0.92,
                               cell_h * 0.92, facecolor=(grey,) * 3, edgecolor="none",
                               alpha=alpha))


def _grid(ax, x0, y0, values, size, vmax, alpha=1.0):
    """Draws ``values`` as a square grid ``size`` wide, with its bottom-left at ``(x0, y0)``."""
    cols = math.ceil(math.sqrt(len(values)))
    cell = size / cols
    rows = math.ceil(len(values) / cols)
    _cells(ax, x0, y0 + (rows - 1) * cell, values, cell, cell, cols, vmax, alpha)
