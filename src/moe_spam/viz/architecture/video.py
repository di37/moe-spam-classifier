"""The video that walks one real spam and one real ham message through the trained model.

The left of every frame is a map of the model with layer 1 and layer 2 boxed; the right shows
what the lit-up step does to the message, using the model's own attention and routing.
"""

from dataclasses import dataclass

from moe_spam.analysis.examples import Pick
from moe_spam.model.bert import N_EXPERTS, N_LAYERS
from moe_spam.paths import PATHS
from moe_spam.viz.architecture.layout import DETAIL_RECT, PanelGrid, TokenLayout
from moe_spam.viz.architecture.map import ModelMap
from moe_spam.viz.architecture.scenes import ArchitectureScenes
from moe_spam.viz.architecture.script import BEATS, caption, step_titles
from moe_spam.viz.ternary import Ternary
from moe_spam.viz.video import FPS, Video

TITLE = "How the spam classifier reads a message"
CAPTION_CHARS = 112
LAYERS, EXPERTS = N_LAYERS, N_EXPERTS  # the map shows two layers; the triangle three experts
INSET_RECT = [0.80, 0.45, 0.17, 0.22]


@dataclass(frozen=True)
class Segment:
    """One message's part of the video.

    Attributes:
        label: ``"spam"`` or ``"ham"``.
        pick: The example message.
        scenes: Its detail-panel drawings.
        number: Its place in the video, from 1.
        total: How many messages the video shows.
    """

    label: str
    pick: Pick
    scenes: ArchitectureScenes
    number: int
    total: int


class ArchitectureVideo(Video):
    """One spam and one ham example, step by step through the model.

    Attributes:
        picks: ``{"spam": Pick, "ham": Pick}``, from ``ExampleFinder.ranked``.
    """

    name = "architecture"

    def __init__(self, picks, paths=PATHS, fps=FPS):
        """Creates the video of the two ``picks``, writing under ``paths`` at ``fps``."""
        super().__init__(paths, fps)
        self.picks = picks

    def check(self):
        """Checks the model's shape, and that both messages fit on screen.

        Raises:
            ValueError: If the model is not two layers of three experts, or a message does
                not fit.
        """
        if any(len(p.trace.experts) != LAYERS or p.trace.n_experts != EXPERTS
               for p in self.picks.values()):
            raise ValueError(f"the architecture video needs a model with {LAYERS} layers of "
                             f"{EXPERTS} experts")
        size = PanelGrid.size()
        for pick in self.picks.values():
            TokenLayout.of(pick.trace, size)

    def segments(self):
        """Returns one segment per message, spam first, with its beats."""
        size = PanelGrid.size()
        return [(Segment(label, pick, ArchitectureScenes(pick.trace, TokenLayout.of(pick.trace,
                                                                                    size)),
                         number, len(self.picks)), BEATS[label])
                for number, (label, pick) in enumerate(self.picks.items(), start=1)]

    def draw(self, canvas, context, beat, t):
        """Draws one frame: the map, the step's title and caption, and its drawing."""
        step, trace = beat.step, context.pick.trace
        canvas.title(TITLE, f"message {context.number} of {context.total}: {context.label}")
        ModelMap.draw(canvas.axes([0.01, 0.02, 0.27, 0.9]), step, trace.d_model)
        canvas.heading(0.33, 0.885, step_titles(context.label)[step])
        canvas.caption(0.33, 0.862, caption(step, context.pick, context.label), CAPTION_CHARS)
        ax = canvas.axes(DETAIL_RECT)
        PanelGrid.measure(ax)
        context.scenes.draw(ax, step, t)
        if step == "route2" and t >= 1:  # once every token has landed
            Ternary.draw_inset(canvas.axes(INSET_RECT), trace.cls_probs[-1])
