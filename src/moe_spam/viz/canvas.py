"""One frame: a styled figure that is saved and closed when its ``with`` block ends."""

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt

from moe_spam.viz.style import DPI, FIGURE_PX, INK, INK_2, STYLE


class Canvas:
    """A figure of the project's size and style, used as a context manager.

    Everything drawn inside the ``with`` block uses the shared style, and the frame is saved
    as a JPEG when the block ends (unless it raised, or no path was given).

    Attributes:
        path: Where the frame is saved, or None to only measure layouts.
        figure: The matplotlib figure, while the block runs.
    """

    def __init__(self, path=None):
        """Creates a canvas that will be saved to ``path``."""
        self.path = None if path is None else Path(path)
        self.figure = None
        self._style = None

    def __enter__(self):
        """Opens the styled figure."""
        self._style = plt.rc_context(STYLE)
        self._style.__enter__()
        self.figure = plt.figure(figsize=(FIGURE_PX[0] / DPI, FIGURE_PX[1] / DPI), dpi=DPI)
        return self

    def __exit__(self, exc_type, exc, traceback):
        """Saves the frame (if all went well), then closes the figure and the style."""
        try:
            if exc_type is None and self.path is not None:
                self.figure.savefig(self.path, pil_kwargs={"quality": 92})
        finally:
            plt.close(self.figure)
            self._style.__exit__(exc_type, exc, traceback)

    def axes(self, rect):
        """Returns new axes at ``rect`` = ``[left, bottom, width, height]`` (figure fractions)."""
        return self.figure.add_axes(rect)

    def text(self, x, y, text, **style):
        """Writes ``text`` at figure coordinates ``(x, y)``."""
        return self.figure.text(x, y, text, **style)

    def title(self, title, right=None, **style):
        """Writes the frame's title at the top left and, if given, a note at the top right.

        Args:
            title: The title.
            right: The note, or None.
            **style: Extra text properties for the note, e.g. ``math_fontfamily``.
        """
        self.text(0.015, 0.955, title, fontsize=16, color=INK, weight="bold")
        if right is not None:
            self.text(0.985, 0.955, right, ha="right", fontsize=13, color=INK_2, **style)

    def heading(self, x, y, text):
        """Writes a section heading at ``(x, y)``."""
        self.text(x, y, text, fontsize=15, color=INK, weight="bold")

    def caption(self, x, y, text, chars, **style):
        """Writes ``text`` wrapped at ``chars`` characters per line, hanging from ``(x, y)``.

        Extra ``style`` (text properties, e.g. ``math_fontfamily``) is passed on.
        """
        self.text(x, y, "\n".join(textwrap.wrap(text, chars)), fontsize=12, color=INK_2,
                  va="top", **style)
