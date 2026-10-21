"""The "why MoE" drawings: a Transformer block whose FFN grows, then splits into experts.

Drawn on the full-frame axes (160 × 90 units): the block on the left, the numbers on the
right.
"""

from matplotlib.patches import Polygon, Rectangle

from moe_spam.viz.shapes import arrow, box, line
from moe_spam.viz.style import EXPERT_COLORS, INK, INK_2, LAYER_COLORS, MUTED, SURFACE, blend
from moe_spam.viz.video import ease

CX = 40.0                       # the block's centre line
CHIP_Y, ATTN_Y, ROUTER_Y, FFN_Y, JOIN_Y, OUT_Y = 14.0, 25.0, 34.5, 48.0, 58.5, 66.0
FFN_H, DENSE_W, EXPERT_W, GAP = 11.0, 22.0, 20.0, 2.0
CHAR_W = 0.8                     # the width of one character of a token chip, in units
BARS = (86.0, 104.0, 40.0)       # label x, bar start x, bar length of the largest value
LAYER_NAMES = {"dense": "dense FFN", "wide": "3× wider FFN", "moe": "MoE layer"}


def chip_positions(words, width=72.0, gap=1.6):
    """Returns the centre x of each token chip that fits in ``width``, centred on the block."""
    widths = [CHAR_W * len(w) + 1.2 for w in words]
    kept, total = [], -gap
    for w in widths:
        if total + gap + w > width:
            break
        kept.append(w)
        total += gap + w
    x, centres = CX - total / 2, []
    for w in kept:
        centres.append(x + w / 2)
        x += w + gap
    return centres


def draw_block(ax, d_model, d_ff, n_experts, widen=0.0, split=0.0, chips=(), routed=0):
    """Draws the block: tokens, self-attention, the FFN (as wide or split as asked), output.

    Args:
        ax: The full-frame axes.
        d_model: The width of a token vector.
        d_ff: The hidden width of the dense FFN and of each expert.
        n_experts: Experts per MoE layer.
        widen: 0 for the dense FFN, 1 for one ``n_experts`` times wider (in between: growing).
        split: 0 for one FFN, 1 for ``n_experts`` experts with a router.
        chips: ``(word, x, expert)`` of each token chip shown under the block.
        routed: How many chips (from the left) are coloured by their expert.
    """
    full = DENSE_W + (n_experts * (EXPERT_W + GAP) - DENSE_W) * ease(widen)
    for k, (word, x, expert) in enumerate(chips):
        color = EXPERT_COLORS[expert] if k < routed else None
        ax.text(x, CHIP_Y, word, ha="center", va="center", fontsize=10, color=INK, zorder=4,
                bbox={"boxstyle": "round,pad=0.25", "edgecolor": color or INK_2,
                      "facecolor": blend(color, 0.3) if color else SURFACE,
                      "linewidth": 1.4 if color else 0.8})
    arrow(ax, (CX, CHIP_Y + 1.6), (CX, ATTN_Y - 2.0))
    box(ax, CX, ATTN_Y, 30.0, 4.0, "Self-attention", fill=INK_2)
    if split > 0:
        _router(ax, split)
    else:
        arrow(ax, (CX, ATTN_Y + 2.0), (CX, FFN_Y - FFN_H / 2))
    if split <= 0:
        hidden = round(d_ff * (1 + (n_experts - 1) * ease(widen)))
        box(ax, CX, FFN_Y, full, FFN_H, f"FFN\n{d_model} → {hidden} → {d_model}",
            fill=SURFACE)
        arrow(ax, (CX, FFN_Y + FFN_H / 2), (CX, OUT_Y - 1.0))
    else:
        _experts(ax, d_model, d_ff, n_experts, split, chips[:routed])
        arrow(ax, (CX, JOIN_Y), (CX, OUT_Y - 1.0))
    ax.text(CX, OUT_Y, "to the next block", ha="center", va="center", fontsize=10.5,
            color=INK_2)


def expert_x(e, n_experts):
    """Returns the centre x of expert ``e`` (from 0)."""
    return CX + (e - (n_experts - 1) / 2) * (EXPERT_W + GAP)


def _router(ax, alpha):
    h = 2.6
    ax.add_patch(Polygon([(CX - h, ROUTER_Y), (CX, ROUTER_Y + h), (CX + h, ROUTER_Y),
                          (CX, ROUTER_Y - h)], facecolor=SURFACE, edgecolor=INK, lw=1.4,
                         alpha=alpha, zorder=3))
    ax.text(CX, ROUTER_Y, "router", ha="center", va="center", fontsize=9.5, color=INK,
            alpha=alpha, zorder=4)
    arrow(ax, (CX, ATTN_Y + 2.0), (CX, ROUTER_Y - h), alpha)


def _experts(ax, d_model, d_ff, n_experts, split, routed_chips):
    """The wide FFN cut into experts: gaps open, colours and the router's arrows fade in."""
    w = EXPERT_W + GAP * (1 - ease(split))
    counts = [sum(1 for *_, e in routed_chips if e == k) for k in range(n_experts)]
    for e, color in enumerate(EXPERT_COLORS[:n_experts]):
        x = expert_x(e, n_experts)
        tokens = f"\n{counts[e]} of {len(routed_chips)} tokens" if routed_chips else ""
        box(ax, x, FFN_Y, w, FFN_H, f"expert {e + 1}\n{d_model} → {d_ff} → {d_model}{tokens}",
            fill=blend(color, 0.35 * ease(split)))
        line(ax, [(CX, ROUTER_Y + 2.6), (x, FFN_Y - FFN_H / 2)], ease(split),
             color=color if counts[e] else MUTED, lw=1.6 if counts[e] else 1.0)
        line(ax, [(x, FFN_Y + FFN_H / 2), (CX, JOIN_Y)], ease(split), MUTED)


def draw_cost_bars(ax, costs, shown):
    """Weights stored and multiply-adds per token of each FFN shown so far.

    Args:
        ax: The full-frame axes.
        costs: ``{"dense" | "wide" | "moe": LayerCost}``.
        shown: ``{layer: alpha}`` of the rows to draw.
    """
    label_x, bar_x, length = BARS
    top = max(max(c.weights, c.macs_per_token) for c in costs.values())
    ax.text(label_x, 76.0, "One feed-forward layer", va="center", fontsize=13,
            weight="bold", color=INK)
    for title, y0, key in (("Weights stored", 70.0, "weights"),
                           ("Multiply-adds per token", 49.0, "macs_per_token")):
        ax.text(label_x, y0, title, va="center", fontsize=11.5, color=INK_2)
        for k, (name, alpha) in enumerate(shown.items()):
            value, y = getattr(costs[name], key), y0 - 4.5 - 4.2 * k
            ax.text(label_x, y, LAYER_NAMES[name], va="center", fontsize=10.5, color=INK,
                    alpha=alpha)
            ax.add_patch(Rectangle((bar_x, y - 1.3), length * value / top, 2.6,
                                   facecolor=LAYER_COLORS[name], edgecolor="none",
                                   alpha=alpha))
            ax.text(bar_x + length * value / top + 1.0, y, f"{value:,}", va="center",
                    fontsize=10.5, color=INK, alpha=alpha)


def draw_tradeoff(axes, costs, t):
    """Each layer as a point: weights stored (capacity) against multiply-adds (compute)."""
    top = 1.15 * max(max(c.weights, c.macs_per_token) for c in costs.values())
    axes.set_xlim(0, top)
    axes.set_ylim(0, top)
    axes.set_xlabel("weights stored (capacity)", fontsize=11)
    axes.set_ylabel("multiply-adds per token (compute)", fontsize=11)
    axes.ticklabel_format(style="plain")
    axes.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
    axes.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
    axes.grid(color="#e1e0d9", lw=0.8)
    for k, (name, c) in enumerate(costs.items()):
        alpha = ease(t * 3 - k)
        axes.scatter([c.weights], [c.macs_per_token], s=180, color=LAYER_COLORS[name],
                     zorder=3, alpha=alpha)
        axes.annotate(LAYER_NAMES[name], (c.weights, c.macs_per_token), xytext=(-12, 10),
                      textcoords="offset points", ha="right", fontsize=11, color=INK,
                      alpha=alpha)
    if t >= 1:
        dense, wide, moe = (costs[k] for k in ("dense", "wide", "moe"))
        axes.annotate("", (moe.weights, moe.macs_per_token),
                      (dense.weights, dense.macs_per_token),
                      arrowprops={"arrowstyle": "->", "color": INK, "lw": 1.4})
        axes.text((dense.weights + moe.weights) / 2, dense.macs_per_token - top * 0.06,
                  "same compute as dense", ha="center", fontsize=10.5, color=INK)
        axes.annotate("", (moe.weights, moe.macs_per_token),
                      (wide.weights, wide.macs_per_token),
                      arrowprops={"arrowstyle": "->", "color": INK, "lw": 1.4})
        axes.text(moe.weights - top * 0.02, (wide.macs_per_token + moe.macs_per_token) / 2,
                  "same capacity\nas wide", ha="right", va="center", fontsize=10.5, color=INK)


def draw_results(ax, results, t):
    """Test mistakes of each model, fewest first, as bars that grow in."""
    label_x, bar_x, _ = BARS
    length, worst = 30.0, max(errors for _, errors, _ in results)
    ax.text(label_x, 70.0, "Test mistakes, of 1,027 messages (accuracy)", va="center",
            fontsize=13, weight="bold", color=INK)
    for k, (name, errors, accuracy) in enumerate(results):
        y = 62.0 - 7.0 * k
        ax.text(label_x, y, name, va="center", fontsize=11, color=INK)
        width = length * errors / worst * ease(t * 1.5)
        ax.add_patch(Rectangle((bar_x + 6, y - 1.5), width, 3.0, facecolor=INK_2,
                               edgecolor="none"))
        if t >= 0.6:
            ax.text(bar_x + 7 + width, y, f"{errors}  ({accuracy:.1%})", va="center",
                    fontsize=11, color=INK)
