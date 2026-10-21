"""Where every token of a message sits on screen, in the videos' detail panel.

The panel is measured in characters and lines of the monospace font, as in
``panels.draw_tokens``, so token boxes line up whatever the message.
"""

import textwrap
from dataclasses import dataclass

from moe_spam.model.bert import N_EXPERTS
from moe_spam.viz.canvas import Canvas
from moe_spam.viz.panels import layout_tokens
from moe_spam.viz.style import CHAR_EM, LINE_EM

FONT = 12
DETAIL_RECT = [0.33, 0.03, 0.655, 0.75]  # the detail panel, in figure coordinates
ROW_GAP = 2            # characters between token boxes
ROW_STEP = 3.0         # lines from one token row to the next, with room underneath
BIN_ROW_STEP = 1.5     # lines between token rows inside an expert's box
CLASSIFIER_ROOM = 10.6  # lines the classifier step needs below the token rows
BINS_TOP = 9.6         # the expert boxes start no higher than this line, which leaves room
                       # above them for the layer-2 triangle


class PanelGrid:
    """The detail panel measured in characters and lines."""

    @staticmethod
    def measure(ax):
        """Sets ``ax``'s limits to characters (x) and lines (y, downwards).

        Returns:
            ``(width, height)`` of the axes, in characters and lines.
        """
        box = ax.get_window_extent()
        width = box.width / (CHAR_EM * FONT / 72 * ax.figure.dpi)
        height = box.height / (LINE_EM * FONT / 72 * ax.figure.dpi)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.axis("off")
        return width, height

    @classmethod
    def size(cls):
        """Returns the detail panel's ``(width, height)``; it only depends on the layout."""
        with Canvas() as canvas:
            return cls.measure(canvas.axes(DETAIL_RECT))


@dataclass(frozen=True)
class TokenLayout:
    """Where each token box sits: in the token rows, and inside its expert's box.

    Attributes:
        width: The panel's width, in characters.
        height: The panel's height, in lines.
        text_lines: The message, wrapped to the panel.
        home: ``(x, y)`` of each token in the token rows.
        bins: Per layer, ``(x, y)`` of each token inside its expert's box.
        bin_boxes: ``(x0, y0, x1, y1)`` of each expert's box.
        below: The first free line under the token rows.
    """

    width: float
    height: float
    text_lines: tuple
    home: tuple
    bins: tuple
    bin_boxes: tuple
    below: float

    @classmethod
    def build(cls, text, words, experts_per_layer, width, height, n_experts=N_EXPERTS):
        """Lays a message out, from its words alone, so nothing moves between frames.

        Args:
            text: The message, shown at the top.
            words: What each token box says, ``[CLS]`` first.
            experts_per_layer: Per layer, the expert of each token.
            width: The panel's width, in characters.
            height: The panel's height, in lines.
            n_experts: The number of expert boxes.

        Returns:
            The layout.

        Raises:
            ValueError: If the message does not fit on screen.
        """
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
        return cls(width, height, text_lines, home, tuple(bins), boxes, below)

    @classmethod
    def of(cls, trace, size):
        """Returns the layout of a ``MessageTrace`` in a panel of ``size``."""
        return cls.build(trace.text, trace.words, trace.experts, *size)


def fits_on_screen(trace, size=None):
    """Returns whether a ``MessageTrace`` fits the detail panel (of ``size``, if given)."""
    try:
        TokenLayout.of(trace, size or PanelGrid.size())
    except ValueError:
        return False
    return True
