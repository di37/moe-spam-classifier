"""The "why MoE" video: from one dense FFN, to a wider one, to experts with a router.

Every number is measured: the layers' weights and multiply-adds are counted, the router's
choices and the experts' specialization come from the trained model, and the mistakes
from the test runs of each model.
"""

from moe_spam.model.bert import N_EXPERTS
from moe_spam.paths import PATHS
from moe_spam.viz.panels import draw_shares
from moe_spam.viz.style import INK
from moe_spam.viz.video import FPS, Video, ease
from moe_spam.viz.why_moe.diagram import (
    chip_positions,
    draw_block,
    draw_cost_bars,
    draw_results,
    draw_tradeoff,
)
from moe_spam.viz.why_moe.script import BEATS, SCENE_TITLES, TITLE, captions

CAPTION_CHARS = 170
SHARES_RECT, TRADEOFF_RECT = [0.72, 0.12, 0.24, 0.6], [0.6, 0.13, 0.36, 0.6]
STAGE = {"dense": (0, 0), "wide": (1, 0)}  # scene: (widen, split) before the split


class WhyMoeVideo(Video):
    """Why MoE: capacity without compute, shown on the trained models' real numbers.

    Attributes:
        story: The ``WhyMoeStory``.
    """

    name = "why_moe"

    def __init__(self, story, paths=PATHS, fps=FPS):
        """Creates the video of ``story``, writing under ``paths``, at ``fps``."""
        super().__init__(paths, fps)
        self.story = story
        positions = chip_positions(story.words)
        self.chips = tuple(zip(story.words, positions, story.experts, strict=False))

    def check(self):
        """Raises ``ValueError`` unless the model has three experts."""
        if self.story.n_experts != N_EXPERTS:
            raise ValueError(f"the why-MoE video needs a model with {N_EXPERTS} experts")

    def segments(self):
        """Returns one segment: the scenes, with their captions as context."""
        return [(captions(self.story), BEATS)]

    def draw(self, canvas, context, beat, t):
        """Draws one frame: the block on the left, the scene's numbers on the right."""
        scene, story = beat.step, self.story
        canvas.title(TITLE, SCENE_TITLES[scene])
        canvas.caption(0.015, 0.905, context[scene], CAPTION_CHARS)
        ax = canvas.axes([0, 0, 1, 1])
        ax.set_xlim(0, 160)
        ax.set_ylim(0, 90)
        ax.axis("off")
        widen, split = STAGE.get(scene, (1, 1))
        if scene == "wide":
            widen = t
        routed = 0 if scene in STAGE else len(self.chips)  # no router before the MoE scene
        if scene == "moe":
            split, routed = min(1.0, t * 2.5), int(len(self.chips) * ease((t - 0.35) / 0.6))
        draw_block(ax, story.d_model, story.d_ff, story.n_experts, widen, split, self.chips,
                   routed)
        if scene in ("dense", "wide", "moe"):
            order = ("dense", "wide", "moe")[:order_index(scene) + 1]
            shown = {name: 1.0 if name != scene else ease(t * 2) for name in order}
            draw_cost_bars(ax, story.costs, shown)
        elif scene == "tradeoff":
            draw_tradeoff(canvas.axes(TRADEOFF_RECT), story.costs, t)
        elif scene == "specialize":
            axes = canvas.axes(SHARES_RECT)
            draw_shares(axes, story.shares, story.group_labels)
            axes.set_title("Layer 1: share of each kind of token sent to each expert",
                           fontsize=12, color=INK, loc="right")
        else:
            draw_results(ax, story.results, t)


def order_index(scene):
    """Returns the position of ``scene`` among the dense, wide and MoE scenes."""
    return ("dense", "wide", "moe").index(scene)
