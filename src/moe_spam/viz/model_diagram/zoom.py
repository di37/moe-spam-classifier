"""The zoomed-in parts beside the stack.

They are the MoE layer, one expert's feed-forward module, the router's readout for a real
message, and the resource savings.
"""

import textwrap

from matplotlib.patches import Rectangle

from moe_spam.viz.model_diagram.stack import BOX_H, CX, Y
from moe_spam.viz.shapes import arrow, badge, box, line, note, panel, plus
from moe_spam.viz.style import AXIS, EXPERT_COLORS, GRID, INK, INK_2, MUTED, blend

MOE_PANEL = (62.0, 12.0, 112.0, 46.0)
ROUTER = (87.0, 16.5)
EXPERT_XS, EXPERT_Y, EXPERT_W = (70.0, 87.0, 104.0), 28.5, 13.0
ADD = (87.0, 39.5)
FFN_PANEL = (72.0, 50.0, 110.0, 74.5)
FFN_X, FFN_YS = 91.0, (54.5, 61.0, 67.5)
READOUT = (114.0, 50.0, 158.0, 74.5)
SAVINGS = (116.0, 12.0, 158.0, 44.0)
SAVINGS_CHARS = 46


def draw_moe_connector(ax, alpha=1.0):
    """The dotted line from the stack's MoE box to the MoE layer."""
    line(ax, [(CX + 7.0, Y["moe"]), (MOE_PANEL[0], MOE_PANEL[3] - 2.0)], alpha, INK_2,
         1.1, dotted=True)


def draw_moe_layer(ax, alpha=1.0, expert=None, word=None):
    """The MoE layer: a router, one feed-forward network per expert, and the ⊕.

    Args:
        ax: The full-frame axes.
        alpha: How far it has faded in.
        expert: The expert the current token goes to (lit), or None.
        word: The current token, shown going into the router, or None.
    """
    panel(ax, *MOE_PANEL, alpha, fill=blend(GRID, 0.5), edge=INK_2, dotted=True)
    ax.text(MOE_PANEL[2] - 1.5, MOE_PANEL[3] - 1.8, "MoE layer", ha="right", va="center",
            fontsize=13, weight="bold", color=INK, alpha=alpha)
    rx, ry = ROUTER
    arrow(ax, (rx, MOE_PANEL[1] + 0.4), (rx, ry - BOX_H / 2), alpha)
    box(ax, rx, ry, 12.0, BOX_H, "Router", alpha, lit=expert is not None)
    for e, (x, color) in enumerate(zip(EXPERT_XS, EXPERT_COLORS, strict=True)):
        on = e == expert
        arrow(ax, (rx, ry + BOX_H / 2), (x, EXPERT_Y - 1.6), alpha,
              color=INK if on else MUTED, lw=1.8 if on else 1.0,
              dashed=expert is not None and not on)
        box(ax, x, EXPERT_Y, EXPERT_W, 3.2, "Feed forward", alpha,
            fill=color if on else blend(color, 0.25), fontsize=10.5)
        badge(ax, x + EXPERT_W / 2 - 0.9, EXPERT_Y - 1.6, str(e + 1), color, alpha)
        arrow(ax, (x, EXPERT_Y + 1.6), (ADD[0], ADD[1] - 1.3), alpha,
              color=INK if on else MUTED, lw=1.8 if on else 1.0)
    plus(ax, *ADD, 1.3, alpha, lit=expert is not None)
    arrow(ax, (ADD[0], ADD[1] + 1.3), (ADD[0], MOE_PANEL[3] - 0.2), alpha)
    if word is not None:
        ax.text(rx - 7.5, MOE_PANEL[1] + 1.6, f"“{word}”", ha="right", va="center",
                fontsize=11, weight="bold", color=INK, alpha=alpha)


def draw_ffn_module(ax, notes, shown=3.0, alpha=1.0):
    """One expert's feed-forward module, built bottom-up, with the dotted line to it.

    Args:
        ax: The full-frame axes.
        notes: ``{"first", "second", "hidden"}`` annotations, with the real sizes.
        shown: How many of its three layers are drawn (a fraction fades the last in).
        alpha: How far the panel has faded in.
    """
    line(ax, [(EXPERT_XS[0], EXPERT_Y + 1.6), (FFN_PANEL[0] + 2.0, FFN_PANEL[1])], alpha,
         INK_2, 1.1, dotted=True)
    panel(ax, *FFN_PANEL, alpha, fill=blend(GRID, 0.5), edge=INK_2, dotted=True)
    ax.text((FFN_PANEL[0] + FFN_PANEL[2]) / 2, FFN_PANEL[3] - 1.7, "FeedForward module "
            "(one expert)", ha="center", va="center", fontsize=12, weight="bold", color=INK,
            alpha=alpha)
    labels = ("Linear layer", "GELU activation", "Linear layer")
    sides = (notes["first"], None, notes["second"])
    for k, (y, label, side) in enumerate(zip(FFN_YS, labels, sides, strict=True)):
        a = alpha * min(1.0, max(0.0, shown - k))
        if a <= 0:
            continue
        arrow(ax, (FFN_X, (FFN_YS[k - 1] if k else FFN_PANEL[1]) + (BOX_H / 2 if k else 0.3)),
              (FFN_X, y - BOX_H / 2), a)
        box(ax, FFN_X, y, 16.0, BOX_H, label, a)
        if side:
            ax.text(FFN_X + 9.5, y, side, va="center", fontsize=10, color=INK_2, alpha=a)
    if shown >= 2:
        note(ax, FFN_PANEL[0] - 1.5, 57.0, notes["hidden"], (FFN_X - 8.0, FFN_YS[1]),
             alpha * min(1.0, shown - 1.5), ha="right")


def draw_readout(ax, title, word, probs, counts, alpha=1.0):
    """The router's numbers for the current token, and how many tokens each expert got.

    Args:
        ax: The full-frame axes.
        title: The readout's heading.
        word: The current token, or None once every token has gone through.
        probs: The router's probabilities for it, one per expert (or None).
        counts: Tokens sent to each expert so far.
        alpha: How far it has faded in.
    """
    x0, y0, x1, y1 = READOUT
    panel(ax, x0, y0, x1, y1, alpha, edge=AXIS)
    ax.text(x0 + 1.5, y1 - 1.8, title, va="center", fontsize=11.5, weight="bold", color=INK,
            alpha=alpha)
    if word is None:
        ax.text(x0 + 1.5, y1 - 9.0, f"Every one of the {sum(counts)} tokens went to\nexactly "
                f"one expert, chosen by the router.", va="center", fontsize=11, color=INK,
                linespacing=1.3, alpha=alpha if sum(counts) else 0.0)
    else:
        ax.text(x0 + 1.5, y1 - 5.2, f"token “{word}”", va="center", fontsize=11, color=INK,
                alpha=alpha)
    for e, color in enumerate(EXPERT_COLORS if probs is not None else ()):
        y = y1 - 9.0 - 3.2 * e
        ax.text(x0 + 1.5, y, f"p{e + 1}", va="center", fontsize=10.5, color=color,
                weight="bold", alpha=alpha)
        if probs is not None:
            on = e == int(probs.argmax())
            ax.add_patch(Rectangle((x0 + 5.0, y - 1.0), 24.0 * probs[e], 2.0, facecolor=color,
                                   edgecolor=INK if on else "none", lw=1.6, alpha=alpha))
            ax.text(x0 + 6.0 + 24.0 * probs[e], y, f"{probs[e]:.2f}", va="center",
                    fontsize=10, color=INK, alpha=alpha)
    total = max(1, sum(counts))
    ax.text(x0 + 1.5, y0 + 5.6, f"tokens per expert so far ({sum(counts)})", va="center",
            fontsize=10, color=INK_2, alpha=alpha)
    left = x0 + 1.5
    for count, color in zip(counts, EXPERT_COLORS, strict=True):
        width = (x1 - x0 - 3.0) * count / total
        if count:
            ax.add_patch(Rectangle((left, y0 + 1.4), width, 2.4, facecolor=color,
                                   edgecolor="white", lw=1.0, alpha=alpha))
            ax.text(left + width / 2, y0 + 2.6, str(count), ha="center", va="center",
                    fontsize=9.5, weight="bold", color="white", alpha=alpha)
        left += width


def draw_savings(ax, title, bullets, shown, alpha=1.0):
    """The resource savings: what the model stores, and what one token goes through.

    Args:
        ax: The full-frame axes.
        title: The heading.
        bullets: The points, with the real counts.
        shown: How many bullets are drawn (a fraction fades the last one in).
        alpha: How far the panel has faded in.
    """
    x0, y0, x1, y1 = SAVINGS
    ax.text(x0, y1 - 1.0, title, va="center", fontsize=13, weight="bold", color=INK,
            alpha=alpha)
    y = y1 - 4.5
    for k, bullet in enumerate(bullets):
        a = alpha * min(1.0, max(0.0, shown - k))
        lines = textwrap.wrap(bullet, SAVINGS_CHARS)
        if a > 0:
            ax.text(x0 + 0.5, y, "•", va="top", fontsize=12, color=INK, alpha=a)
            ax.text(x0 + 2.5, y, "\n".join(lines), va="top", fontsize=11, color=INK,
                    linespacing=1.3, alpha=a)
        y -= 2.3 * len(lines) + 1.4
