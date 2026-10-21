"""The exercises video: a question, time to predict, then the answer from real runs.

Each exercise is one segment: the question and three choices; a countdown to make a
prediction; then the right choice, chosen by the runs' numbers, and the chart behind it.
"""

import textwrap

import numpy as np
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

from moe_spam.paths import PATHS
from moe_spam.viz.exercises.script import BEATS, TITLE
from moe_spam.viz.style import AXIS, HIGHLIGHT, INK, INK_2, MUTED, SURFACE
from moe_spam.viz.video import FPS, Video, ease

QUESTION_CHARS, TAKEAWAY_CHARS = 95, 48
CARD_XS, CARD_Y, CARD_W, CARD_H = (28.0, 80.0, 132.0), 58.0, 46.0, 8.0
CHART_RECT = [0.07, 0.08, 0.5, 0.36]
LETTERS = "ABC"


class ExercisesVideo(Video):
    """Predict-then-check exercises on the experts and the balancing.

    Attributes:
        exercises: The ``Exercise`` of each segment, in order.
    """

    name = "exercises"

    def __init__(self, exercises, paths=PATHS, fps=FPS):
        """Creates the video of ``exercises``, writing under ``paths``, at ``fps``."""
        super().__init__(paths, fps)
        self.exercises = tuple(exercises)

    def segments(self):
        """Returns one segment per exercise, each with the question, predict and reveal beats."""
        return [(exercise, BEATS) for exercise in self.exercises]

    def draw(self, canvas, context, beat, t):
        """Draws one frame of exercise ``context`` in the stage ``beat``."""
        number = self.exercises.index(context) + 1
        canvas.title(TITLE, f"exercise {number} of {len(self.exercises)}")
        canvas.text(0.03, 0.86, "\n".join(textwrap.wrap(context.question, QUESTION_CHARS)),
                    fontsize=18, weight="bold", color=INK, va="top", linespacing=1.3)
        ax = canvas.axes([0, 0, 1, 1])
        ax.set_xlim(0, 160)
        ax.set_ylim(0, 90)
        ax.axis("off")
        reveal = ease(t * 2) if beat.step == "reveal" else 0.0
        _cards(ax, context, ease(t * 2) if beat.step == "question" else 1.0, reveal)
        if beat.step == "predict":
            _countdown(ax, t, BEATS[1].seconds)
        elif beat.step == "reveal":
            _chart(canvas.axes(CHART_RECT), context, ease(t * 1.5))
            canvas.text(0.62, 0.42, "\n".join(textwrap.wrap(context.takeaway, TAKEAWAY_CHARS)),
                        fontsize=13, color=INK, va="top", linespacing=1.4, alpha=reveal)


def _cards(ax, exercise, alpha, reveal):
    """The three choices; once revealed, the right one is marked and the others fade."""
    for k, (x, option) in enumerate(zip(CARD_XS, exercise.options, strict=True)):
        right = k == exercise.answer
        fade = 1 - 0.65 * reveal if not right else 1.0
        ax.add_patch(FancyBboxPatch((x - CARD_W / 2, CARD_Y - CARD_H / 2), CARD_W, CARD_H,
                                    boxstyle="round,pad=0,rounding_size=1.0",
                                    facecolor=HIGHLIGHT if right and reveal else SURFACE,
                                    edgecolor=INK if right and reveal else AXIS,
                                    lw=1.2 + 1.8 * reveal * right, alpha=alpha * fade))
        mark = "✓  " if right and reveal > 0.5 else ""
        ax.text(x, CARD_Y, f"{mark}{LETTERS[k]}.  {option}", ha="center", va="center",
                fontsize=13, color=INK, weight="bold" if right and reveal else "normal",
                alpha=alpha * fade)


def _countdown(ax, t, seconds):
    """'Pause and predict', with the seconds left and a shrinking bar."""
    left = max(1, int(np.ceil(seconds * (1 - t))))
    ax.text(80, 40, "Pause and predict", ha="center", va="center", fontsize=20, color=INK)
    ax.text(80, 30, str(left), ha="center", va="center", fontsize=40, weight="bold",
            color=INK_2)
    ax.add_patch(Rectangle((80 - 30 * (1 - t), 21), 60 * (1 - t), 1.6, facecolor=INK_2,
                           edgecolor="none"))


def _chart(ax, exercise, grow):
    """The exercise's bars, growing in, with values on top and the bars asked about darker."""
    groups, width = len(exercise.labels), 0.8 / len(exercise.series)
    top = max(max(s.values) for s in exercise.series)
    for k, series in enumerate(exercise.series):
        xs = np.arange(groups) + (k - (len(exercise.series) - 1) / 2) * width
        for g, (x, value) in enumerate(zip(xs, series.values, strict=True)):
            color = series.color if g in exercise.highlight else MUTED
            if len(exercise.series) > 1:
                color = series.color
            ax.bar(x, value * grow, width * 0.9, color=color, edgecolor="none")
            if grow >= 1:
                ax.text(x, value + top * 0.02, exercise.fmt.format(value), ha="center",
                        va="bottom", fontsize=9.5 if len(exercise.series) > 1 else 11,
                        color=INK)
    if exercise.reference:
        value, label = exercise.reference
        ax.axhline(value, color=INK_2, lw=0.9, ls=(0, (2, 2)))
        ax.text(groups - 0.45, value, label, ha="right", va="bottom", fontsize=10, color=INK_2)
    if len(exercise.series) > 1:
        ax.legend(handles=[Patch(color=s.color, label=s.name) for s in exercise.series],
                  fontsize=10, frameon=False, loc="upper left")
    ax.set_xticks(range(groups), exercise.labels, fontsize=11)
    ax.set_ylim(0, top * 1.2)
    ax.set_yticks([])
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.set_title(exercise.chart_title, fontsize=12, loc="left")
