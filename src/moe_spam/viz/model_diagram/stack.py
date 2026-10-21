"""The model's stack, bottom to top, drawn piece by piece with its annotations.

From the input text up: tokenizer, embeddings, the repeated encoder block (LayerNorm,
self-attention, ⊕, LayerNorm, MoE, ⊕), the final LayerNorm and the output layer.
"""

from moe_spam.viz.model_diagram.shapes import arrow, box, brace, line, note, panel, plus
from moe_spam.viz.style import GRID, INK, INK_2, MONO, MUTED, SURFACE

CX, BOX_W, BOX_H = 37.0, 17.0, 2.8  # the stack's centre line, and its boxes' size
PANEL = (19.0, 9.0, 55.0, 72.5)      # the grey panel around the model
BLOCK = (24.5, 16.0, 50.5, 55.0)     # the repeated encoder block
RESIDUAL_X = 48.0                    # where the residual connections go up
LEFT_NOTES_X = 16.0                  # the right edge of the notes on the stack's left
RIGHT_NOTES_X = 58.5                 # the left edge of the notes on the stack's right
WIDE_W = 22.0                        # the width of boxes with long labels
LOOP_X = 22.0                        # where the token goes back down for the next layer
Y = {"input": 2.0, "tokens": 6.2, "embed": 12.0, "norm1": 19.5, "attn": 26.0, "add1": 32.5,
     "norm2": 37.5, "moe": 43.0, "add2": 49.5, "final": 60.0, "output": 66.0, "p": 76.0}
PLUS_R, ATTN_H = 1.1, 5.0
STACK_FILL, ATTN_FILL, MOE_FILL = GRID, INK_2, "#cfcec8"
# The order the stack is built in; each piece comes with the arrow and notes that lead to it.
PIECES = ("input", "tokens", "panel", "embed", "block", "norm1", "attn", "add1", "norm2",
          "moe", "add2", "final", "output")


class Stack:
    """Draws the stack; ``path`` is the way a token goes up it, twice through the block.

    Attributes:
        notes: ``{piece: annotation}`` written beside the stack (with real numbers).
        sample: The input message, as shown under the stack.
        n_layers: How many times the block repeats.
    """

    def __init__(self, notes, sample, n_layers):
        """Creates the stack's drawing; see the attributes."""
        self.notes = notes
        self.sample = sample
        self.n_layers = n_layers

    def draw(self, ax, shown=None, lit=(), layer=None, result=None):
        """Draws the stack.

        Args:
            ax: The full-frame axes.
            shown: How many pieces (in ``PIECES`` order) are drawn; a fraction fades the last
                one in. None draws them all.
            lit: The pieces to highlight.
            layer: If given, the brace says which layer is running.
            result: If given, what the output says, in place of "p(spam)".
        """
        shown = len(PIECES) if shown is None else shown
        alpha = {piece: min(1.0, max(0.0, shown - k)) for k, piece in enumerate(PIECES)}
        for piece in ("panel", "block", *(p for p in PIECES if p not in ("panel", "block"))):
            if alpha[piece] > 0:
                getattr(self, f"_{piece}")(ax, alpha[piece], piece in lit, layer)
        if alpha["output"] > 0:
            ax.text(CX, Y["p"], result or "p(spam)", ha="center", va="center", fontsize=11,
                    weight="bold", color=INK, alpha=alpha["output"])

    @staticmethod
    def path():
        """Returns the waypoints ``[(x, y, piece)]`` of a token through every layer."""
        up = [(CX, Y[p], p) for p in ("norm1", "attn", "add1", "norm2", "moe", "add2")]
        start = [(CX, Y["input"] + 1.5, "input"), (CX, Y["tokens"], "tokens"),
                 (CX, Y["embed"], "embed")]
        back = [(CX, BLOCK[3] + 0.6, "block"), (LOOP_X, BLOCK[3] + 0.6, "block"),
                (LOOP_X, BLOCK[1] - 0.6, "block"), (CX, BLOCK[1] - 0.6, "block")]
        end = [(CX, Y["final"], "final"), (CX, Y["output"], "output"), (CX, Y["p"], "p")]
        return [*start, *up, *back, *up, *end]

    def _input(self, ax, alpha, lit, layer):
        ax.text(CX, Y["input"], self.sample, ha="center", va="center", family=MONO,
                fontsize=10, color=INK, alpha=alpha)
        arrow(ax, (CX, Y["input"] + 1.0), (CX, Y["tokens"] - BOX_H / 2), alpha)

    def _tokens(self, ax, alpha, lit, layer):
        box(ax, CX, Y["tokens"], BOX_W, BOX_H, "Tokenized text", alpha, lit=lit)
        note(ax, LEFT_NOTES_X, Y["tokens"], self.notes["tokens"], (CX - BOX_W / 2, Y["tokens"]),
             alpha, ha="right")

    def _panel(self, ax, alpha, lit, layer):
        panel(ax, *PANEL, alpha, fill=STACK_FILL)

    def _embed(self, ax, alpha, lit, layer):
        arrow(ax, (CX, Y["tokens"] + BOX_H / 2), (CX, Y["embed"] - BOX_H / 2), alpha)
        box(ax, CX, Y["embed"], WIDE_W + 2, BOX_H, "Token + position embedding", alpha,
            lit=lit)
        note(ax, LEFT_NOTES_X, Y["embed"] + 1.2, self.notes["context"],
             (CX - (WIDE_W + 2) / 2, Y["embed"]), alpha, ha="right")
        note(ax, RIGHT_NOTES_X, 4.0, self.notes["embed"], (CX + (WIDE_W + 2) / 2, Y["embed"]),
             alpha)

    def _block(self, ax, alpha, lit, layer):
        x0, y0, x1, y1 = BLOCK
        panel(ax, x0, y0, x1, y1, alpha, fill=SURFACE, edge=INK if lit else INK_2,
              lw=2.0 if lit else 1.4)
        brace(ax, PANEL[0] - 1.0, y0, y1, alpha)
        mid = (y0 + y1) / 2
        ax.text(PANEL[0] - 3.0, mid, f"{self.n_layers}×", ha="right", va="center",
                fontsize=17, weight="bold", color=INK, alpha=alpha)
        if layer is not None:
            ax.text(PANEL[0] - 3.0, mid - 3.2, f"layer {layer}\nof {self.n_layers}", ha="right",
                    va="top", fontsize=10.5, color=INK, weight="bold", alpha=alpha)
        note(ax, LEFT_NOTES_X - 1.5, y0 + 5.0, self.notes["block"], alpha=alpha, ha="right")
        arrow(ax, (CX, Y["embed"] + BOX_H / 2), (CX, Y["norm1"] - BOX_H / 2), alpha)

    def _norm1(self, ax, alpha, lit, layer):
        box(ax, CX, Y["norm1"], BOX_W - 3, BOX_H, "LayerNorm 1", alpha, lit=lit)

    def _attn(self, ax, alpha, lit, layer):
        arrow(ax, (CX, Y["norm1"] + BOX_H / 2), (CX, Y["attn"] - ATTN_H / 2), alpha)
        box(ax, CX, Y["attn"], WIDE_W, ATTN_H, self.notes["attn"], alpha, fill=ATTN_FILL,
            lit=lit)

    def _add1(self, ax, alpha, lit, layer):
        arrow(ax, (CX, Y["attn"] + ATTN_H / 2), (CX, Y["add1"] - PLUS_R), alpha)
        self._residual(ax, Y["norm1"] - BOX_H / 2 - 0.9, Y["add1"], alpha)
        plus(ax, CX, Y["add1"], PLUS_R, alpha, lit)

    def _norm2(self, ax, alpha, lit, layer):
        arrow(ax, (CX, Y["add1"] + PLUS_R), (CX, Y["norm2"] - BOX_H / 2), alpha)
        box(ax, CX, Y["norm2"], BOX_W - 3, BOX_H, "LayerNorm 2", alpha, lit=lit)

    def _moe(self, ax, alpha, lit, layer):
        arrow(ax, (CX, Y["norm2"] + BOX_H / 2), (CX, Y["moe"] - BOX_H / 2), alpha)
        box(ax, CX, Y["moe"], BOX_W - 3, BOX_H + 0.4, "MoE", alpha, fill=MOE_FILL, lit=lit)

    def _add2(self, ax, alpha, lit, layer):
        arrow(ax, (CX, Y["moe"] + BOX_H / 2 + 0.2), (CX, Y["add2"] - PLUS_R), alpha)
        self._residual(ax, Y["add1"] + PLUS_R + 0.9, Y["add2"], alpha)
        plus(ax, CX, Y["add2"], PLUS_R, alpha, lit)

    def _final(self, ax, alpha, lit, layer):
        arrow(ax, (CX, Y["add2"] + PLUS_R), (CX, Y["final"] - BOX_H / 2), alpha)
        box(ax, CX, Y["final"], WIDE_W, BOX_H, "Final LayerNorm ([CLS])", alpha, lit=lit)

    def _output(self, ax, alpha, lit, layer):
        arrow(ax, (CX, Y["final"] + BOX_H / 2), (CX, Y["output"] - BOX_H / 2), alpha)
        box(ax, CX, Y["output"], BOX_W, BOX_H, "Linear output layer", alpha, lit=lit)
        arrow(ax, (CX, Y["output"] + BOX_H / 2), (CX, Y["p"] - 1.0), alpha)
        note(ax, RIGHT_NOTES_X, Y["output"] + 1.5, self.notes["output"],
             (CX + BOX_W / 2, Y["output"]), alpha)

    @staticmethod
    def _residual(ax, y_from, y_to, alpha):
        """The skip connection: from the stack's line, up the side, into the ⊕."""
        line(ax, [(CX, y_from), (RESIDUAL_X, y_from), (RESIDUAL_X, y_to)], alpha, MUTED)
        arrow(ax, (RESIDUAL_X, y_to), (CX + PLUS_R, y_to), alpha)
