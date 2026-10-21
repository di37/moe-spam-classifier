"""A map of the model: a stack of blocks with layer 1 and layer 2 boxed, one block lit."""

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from moe_spam.model.bert import N_EXPERTS
from moe_spam.viz.style import AXIS, EXPERT_COLORS, HIGHLIGHT, INK, INK_2, MUTED, SURFACE, ink_on

BLOCK_HALF_HEIGHT = 0.034


class ModelMap:
    """The model drawn as the steps a message goes through.

    Attributes:
        BLOCKS: ``(step, title, subtitle, y centre)`` of each block, top to bottom.
        LAYERS: ``(y0, y1, title, note)`` of each layer's box; the note says where the
            training video shows that layer.
        ALIASES: Steps drawn with another step's block lit.
    """

    BLOCKS = (
        ("text", "Message", "an SMS, as typed", 0.95),
        ("tokenize", "Tokenizer", "[CLS] + words, numbers, punctuation", 0.865),
        ("embed", "Embeddings", "{d_model} numbers per token", 0.78),
        ("attn1", "Self-attention", "tokens share information", 0.64),
        ("route1", f"Router + {N_EXPERTS} experts", "one expert per token", 0.545),
        ("attn2", "Self-attention", "tokens share information", 0.355),
        ("route2", f"Router + {N_EXPERTS} experts", "one expert per token", 0.26),
        ("classify", "Classifier", "final [CLS] → p(spam)", 0.07),
    )
    # Notes in two short lines, so they stay left of the arrows.
    LAYERS = ((0.445, 0.725, "LAYER 1", "in the training video:\nthe token panels"),
              (0.155, 0.435, "LAYER 2", "in the training video:\nthe triangle ([CLS])"))
    ALIASES = {"back1": "route1"}

    @classmethod
    def draw(cls, ax, step, d_model):
        """Draws the map with ``step``'s block lit.

        Args:
            ax: The axes to draw on.
            step: The step being shown.
            d_model: The width of the model's token vectors, for the embeddings block.
        """
        lit = cls.ALIASES.get(step, step)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        for y0, y1, title, note in cls.LAYERS:
            active = any(b[0] == lit for b in cls.BLOCKS if y0 < b[3] < y1)
            ax.add_patch(FancyBboxPatch((0.04, y0), 0.92, y1 - y0,
                                        boxstyle="round,pad=0,rounding_size=0.02",
                                        facecolor="none", edgecolor=INK if active else AXIS,
                                        lw=1.6 if active else 1))
            ax.text(0.07, y1 - 0.008, title, fontsize=11, weight="bold",
                    color=INK if active else INK_2, va="top")
            ax.text(0.07, y0 + 0.008, note, fontsize=8.5, color=INK_2 if active else MUTED,
                    linespacing=1.15)
        centres = [y for *_, y in cls.BLOCKS]
        for ya, yb in zip(centres, centres[1:], strict=False):
            ax.add_patch(FancyArrowPatch((0.5, ya - BLOCK_HALF_HEIGHT),
                                         (0.5, yb + BLOCK_HALF_HEIGHT),
                                         arrowstyle="-|>", mutation_scale=10, color=MUTED, lw=1))
        for name, title, subtitle, y in cls.BLOCKS:
            cls._block(ax, name == lit, name.startswith("route"), title,
                       subtitle.format(d_model=d_model), y)

    @staticmethod
    def _block(ax, active, has_experts, title, subtitle, y):
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
