"""The model diagram video: Tiny BERT + MoE drawn like a modern LLM architecture diagram.

The stack is built bottom to top, and a real test message's [CLS] token goes up through both
blocks, with a log of what each step does to it. Then the MoE layer and one expert's
feed-forward module are zoomed in on (with the trained router's choice for every token of
that message), and the parameter counts of the trained model show what MoE saves.
"""

import numpy as np

from moe_spam.model.bert import N_EXPERTS, N_LAYERS
from moe_spam.paths import PATHS
from moe_spam.viz.model_diagram.cls_log import draw_cls_log
from moe_spam.viz.model_diagram.script import (
    BEATS,
    SCENE_TITLES,
    captions,
    cls_rows,
    ffn_notes,
    sample,
    savings,
    stack_notes,
    title,
)
from moe_spam.viz.model_diagram.shapes import HEIGHT, WIDTH
from moe_spam.viz.model_diagram.stack import CX, PIECES, Stack
from moe_spam.viz.model_diagram.zoom import (
    draw_ffn_module,
    draw_moe_connector,
    draw_moe_layer,
    draw_readout,
    draw_savings,
)
from moe_spam.viz.motion import draw_pulse, locate_in_steps, token_progress
from moe_spam.viz.video import FPS, Video, ease

CAPTION_CHARS = 170
BLOCK_PIECES = ("norm1", "attn", "add1", "norm2", "moe", "add2")
STREAM_START = 0.25  # the MoE scene draws its panels first, then streams the tokens
READOUT_TITLE = "Layer 1 router, token by token"
DOT_R = 0.9  # the travelling token's radius; within this much of a box, the box shows it
LOG_FADE = 0.4  # a log row fades in over this much of the step that reaches it
# The step of the token's path each log row is written at: (piece, which time it is passed).
LOG_STEPS = {"tokens": ("tokens", 1), "embed": ("embed", 1), "attn1": ("attn", 1),
             "moe1": ("moe", 1), "attn2": ("attn", 2), "moe2": ("moe", 2),
             "output": ("output", 1)}


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
        self.rows = cls_rows(trace)

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
        """The [CLS] token goes box to box, up both blocks and out as p(spam), with its log.

        Inside a box the dot is hidden: the box lights up with a "[CLS]" tag beside it, so
        no label is covered. Each log row is written as the token reaches its step.
        """
        path = Stack.path()
        names = [name for *_, name in path]
        at = t * (len(path) - 1)  # how many steps of the path are done
        x, y, heading = locate_in_steps(path, t)
        second = at >= max(k for k, name in enumerate(names) if name == "block")
        near = Stack.box_at(x, y, margin=DOT_R)
        piece = near or heading
        layer = 1 + second if piece in (*BLOCK_PIECES, "block") else None
        self.stack.draw(ax, lit=(piece,), layer=layer,
                        result=self._result() if near == "p" else None, token_at=near)
        if near is None:
            draw_pulse(ax, x, y, "[CLS]", radius=DOT_R, fontsize=10, lift=1.9,
                       side="left" if abs(x - CX) < 0.5 else "above")
        reach = [_nth(names, *LOG_STEPS[row.key]) for row in self.rows]
        shown = sum(min(1.0, max(0.0, (at - k + LOG_FADE) / LOG_FADE)) for k in reach)
        current = next((row.key for row, k in zip(self.rows[::-1], reach[::-1], strict=True)
                        if at >= k - LOG_FADE), None)
        draw_cls_log(ax, self.rows, shown, current)

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


def _nth(names, name, n):
    """Returns the index of the ``n``-th (from 1) ``name`` in ``names``."""
    return [k for k, other in enumerate(names) if other == name][n - 1]
