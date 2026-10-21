"""The model diagram video: Tiny BERT + MoE drawn like a modern LLM architecture diagram.

The stack is built bottom to top, a real test message's [CLS] token goes up through both
blocks, then the MoE layer and one expert's feed-forward module are zoomed in on (with the
trained router's choice for every token of that message), and the parameter counts of the
trained model show what MoE saves.
"""

import numpy as np

from moe_spam.model.bert import N_EXPERTS, N_LAYERS
from moe_spam.paths import PATHS
from moe_spam.viz.model_diagram.script import (
    BEATS,
    SCENE_TITLES,
    captions,
    ffn_notes,
    sample,
    savings,
    stack_notes,
    title,
)
from moe_spam.viz.model_diagram.shapes import HEIGHT, WIDTH
from moe_spam.viz.model_diagram.stack import PIECES, Stack
from moe_spam.viz.model_diagram.zoom import (
    draw_ffn_module,
    draw_moe_connector,
    draw_moe_layer,
    draw_readout,
    draw_savings,
)
from moe_spam.viz.motion import draw_pulse, locate, progress, token_progress
from moe_spam.viz.video import FPS, Video, ease

CAPTION_CHARS = 170
BLOCK_PIECES = ("norm1", "attn", "add1", "norm2", "moe", "add2")
STREAM_START = 0.25  # the MoE scene draws its panels first, then streams the tokens
READOUT_TITLE = "Layer 1 router, token by token"


class ModelDiagramVideo(Video):
    """The architecture diagram, animated with the trained model's numbers.

    Attributes:
        facts: The trained model's ``ModelFacts``.
        trace: The followed message (a ``MessageTrace``).
        probs: ``[n, n_experts]``: layer 1's router probabilities for each of its tokens.
    """

    name = "model_diagram"

    def __init__(self, facts, trace, probs, paths=PATHS, fps=FPS):
        """Creates the video; see the attributes. It writes under ``paths``, at ``fps``."""
        super().__init__(paths, fps)
        self.facts = facts
        self.trace = trace
        self.probs = np.asarray(probs)
        self.stack = Stack(stack_notes(facts), sample(trace.text), facts.n_layers)

    def check(self):
        """Raises ``ValueError`` unless the model has two layers of three experts."""
        if (self.facts.n_layers != N_LAYERS or self.facts.n_experts != N_EXPERTS
                or self.probs.shape != (len(self.trace.words), N_EXPERTS)):
            raise ValueError(f"the model diagram video needs a model with {N_LAYERS} layers "
                             f"of {N_EXPERTS} experts")

    def segments(self):
        """Returns one segment: the scenes, with their captions as context."""
        return [(captions(self.facts, self.trace), BEATS)]

    def draw(self, canvas, context, beat, t):
        """Draws one frame: the stack on the left, the zoomed-in parts on the right."""
        scene = beat.step
        canvas.title(title(self.facts), SCENE_TITLES[scene])
        canvas.caption(0.015, 0.905, context[scene], CAPTION_CHARS)
        ax = canvas.axes([0, 0, 1, 1])
        ax.set_xlim(0, WIDTH)
        ax.set_ylim(0, HEIGHT)
        ax.axis("off")
        if scene == "stack":
            shown = t * len(PIECES)
            self.stack.draw(ax, shown, lit=(PIECES[min(int(shown), len(PIECES) - 1)],))
            return
        if scene == "flow":
            self._draw_flow(ax, t)
            return
        self.stack.draw(ax, lit=("moe",) if scene == "moe" else (), result=self._result())
        self._draw_zoom(ax, scene, t)

    def _draw_flow(self, ax, t):
        """The [CLS] token goes up the stack, back down, up again, and out as p(spam)."""
        path = Stack.path()
        x, y, piece = locate(path, t)
        last_back = max(k for k, (*_, name) in enumerate(path) if name == "block")
        second = t >= progress(path, last_back)
        layer = 1 + second if piece in (*BLOCK_PIECES, "block") else None
        self.stack.draw(ax, lit=(piece,), layer=layer, result=self._result() if t >= 1 else None)
        draw_pulse(ax, x, y, "[CLS]", radius=0.9, fontsize=10, lift=1.9)

    def _result(self):
        verdict = "spam" if self.trace.p_spam > 0.5 else "ham"
        return f"p(spam) = {self.trace.p_spam:.3f}: {verdict}"

    def _draw_zoom(self, ax, scene, t):
        """The MoE layer and its readout; then one expert's module; then the savings."""
        experts = self.trace.experts[0]
        n = len(experts)
        build = ease(t / STREAM_START) if scene == "moe" else 1.0
        done = (token_progress((t - STREAM_START) / (1 - STREAM_START), n)
                if scene == "moe" and t > STREAM_START else (0.0 if scene == "moe" else n))
        streaming = scene == "moe" and t > STREAM_START and done < n
        current = min(int(done), n - 1)
        routed = min(n, int(done) + 1) if streaming else int(done)
        counts = np.bincount(experts[:routed], minlength=self.facts.n_experts).tolist()
        draw_moe_connector(ax, build)
        draw_moe_layer(ax, build, experts[current] if streaming else None,
                       self.trace.words[current] if streaming else None)
        draw_readout(ax, READOUT_TITLE, self.trace.words[current] if streaming else None,
                     self.probs[current] if streaming else None, counts, build)
        if scene in ("ffn", "savings"):
            grow = t * 3.6 if scene == "ffn" else 3.0
            draw_ffn_module(ax, ffn_notes(self.facts), grow,
                            ease(t / 0.2) if scene == "ffn" else 1.0)
        if scene == "savings":
            heading, bullets = savings(self.facts)
            draw_savings(ax, heading, bullets, t * len(bullets) * 1.15, ease(t / 0.15))
