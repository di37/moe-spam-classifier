"""
The drawings of the inference video: three versions of one feed-forward layer side by side,
and bar charts of the work they do, the weights they hold and the time they take.

  lane_labels    what each layer is called, at given sizes
  draw_lanes     the dense, wide and MoE layers, with the current token and the part of each
                 layer that runs on it lit up
  draw_bars      one horizontal bar per layer, labelled with its value
  amount         1234567 -> "1.23 M", for labels

The MoE layer's experts are drawn as the wide layer cut into pieces: the same weights, of
which only one piece runs for each token.
"""
from matplotlib.patches import Polygon, Rectangle

from inference_benchmark import LAYERS
from panels import AXIS, EXPERT_COLORS, INK, INK_2, MONO, MUTED, SURFACE, ink_on

LANE_Y = {"dense": 82, "wide": 50, "moe": 18}  # lane centres, in a 0-100 panel
BLOCK_X, BLOCK_W, BLOCK_H = 30, 20, 12  # where a d_ff-wide network starts, its width, height
BAR_COLORS = {"dense": MUTED, "wide": INK_2, "moe": INK}
LIT = INK_2  # fill of a dense network that is working on the token


def lane_labels(d_ff, n_experts):
    """{layer: (title, subtitle, short name for charts)}."""
    return {
        "dense": ("Dense layer", f"1 network, {d_ff} wide", "dense"),
        "wide": (f"Dense layer, {n_experts}× wider",
                 f"1 network, {n_experts * d_ff} wide: as many weights as the MoE layer",
                 f"dense, {n_experts}× wider"),
        "moe": ("Mixture of experts",
                f"{n_experts} networks, {d_ff} wide each; the router picks 1 for each token",
                "mixture of experts"),
    }


def amount(value):
    if value >= 1e6:
        return f"{value / 1e6:.2f} M"
    if value >= 1e3:
        return f"{value / 1e3:.1f} k"
    return f"{value:.0f}"


def _block(ax, x, y, width, label, fill, edge, lw=1.2):
    ax.add_patch(Rectangle((x, y - BLOCK_H / 2), width, BLOCK_H, facecolor=fill, edgecolor=edge,
                           lw=lw))
    ax.text(x + width / 2, y, label, ha="center", va="center", fontsize=11,
            color=INK if fill == SURFACE else ink_on(fill))


def draw_lanes(ax, word, expert, lit, costs, d_ff, n_experts):
    """
    word: the current token (None before the first); expert: where the MoE router sends it;
    lit: whether the layers are working on it in this frame; costs: {layer: LayerCost}.
    """
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    labels = lane_labels(d_ff, n_experts)
    for layer in LAYERS:
        y = LANE_Y[layer]
        title, subtitle, _ = labels[layer]
        ax.text(1, y + 12, title, fontsize=13, weight="bold", color=INK, va="center")
        ax.text(1, y + 8.6, subtitle, fontsize=10, color=INK_2, va="center")
        if word is not None:
            ax.text(1, y, word, family=MONO, fontsize=13, va="center", color=INK,
                    bbox={"boxstyle": "round,pad=0.3", "facecolor": SURFACE,
                          "edgecolor": INK if lit else AXIS, "linewidth": 1.5})
        if layer == "moe":
            _draw_moe(ax, y, expert, lit, n_experts)
        else:
            ax.annotate("", (BLOCK_X - 0.5, y), (BLOCK_X - 9, y),
                        arrowprops={"arrowstyle": "-|>", "color": MUTED})
            networks = n_experts if layer == "wide" else 1
            _block(ax, BLOCK_X, y, BLOCK_W * networks, f"{networks * d_ff} wide",
                   LIT if lit and word else SURFACE, AXIS)
        macs = costs[layer].macs_per_token
        if layer == "moe":
            one_expert = costs["dense"].macs_per_token
            detail = f"{one_expert:,} + {macs - one_expert} for the router = {macs:,}"
        else:
            detail = f"{macs:,}"
        ax.text(BLOCK_X, y - 9.5, f"per token: {detail} multiply-adds", fontsize=10,
                color=INK_2, va="center")


def _draw_moe(ax, y, expert, lit, n_experts):
    rx = BLOCK_X - 5  # the router: a small diamond in front of the experts, in the color of
    choice = EXPERT_COLORS[expert % len(EXPERT_COLORS)] if lit else SURFACE  # its choice
    ax.add_patch(Polygon([(rx - 2.2, y), (rx, y + 3.5), (rx + 2.2, y), (rx, y - 3.5)],
                         facecolor=choice, edgecolor=INK, lw=1.2))
    ax.text(rx, y - 5.2, "router", ha="center", fontsize=9, color=INK_2)
    ax.annotate("", (rx - 2.4, y), (BLOCK_X - 14, y),
                arrowprops={"arrowstyle": "-|>", "color": MUTED})
    for e in range(n_experts):
        working = lit and e == expert
        color = EXPERT_COLORS[e % len(EXPERT_COLORS)]
        x = BLOCK_X + e * BLOCK_W + 0.4  # a sliver of background between the experts
        _block(ax, x, y, BLOCK_W - 0.8, f"expert {e + 1}" + ("" if working else "  (idle)"),
               color if working else SURFACE, color, lw=2 if working else 1.2)


def draw_bars(ax, values, names, x_max, title, fmt=amount, keys=LAYERS):
    """
    values, names: {key: number}, {key: label}, drawn top to bottom in the order of keys
    (the three layers, or the dense and MoE models); fmt turns a value into its label.
    """
    rows = range(len(keys))
    ax.barh(list(rows), [values[k] for k in keys], height=0.5,
            color=[BAR_COLORS[k] for k in keys])
    for row, key in zip(rows, keys, strict=True):
        ax.text(values[key] + x_max * 0.015, row, fmt(values[key]), va="center",
                fontsize=11, color=INK)
    ax.set_yticks(list(rows), [names[k] for k in keys])
    ax.invert_yaxis()
    ax.set_xlim(0, x_max)
    ax.set_xticks([])
    ax.tick_params(length=0, labelsize=11, labelcolor=INK)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, fontsize=12, loc="left")
