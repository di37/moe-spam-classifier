"""The "how it works" video: the README's diagram, then the routing equations, worked out.

The diagram is built box by box, a [CLS] token travels through it, and then one real
token's numbers go through the three routing equations: the router's softmax, the argmax,
and the expert's output scaled by ``p_i``. It ends with why ``p_i`` is there (its slope) and
the load-balancing loss measured over the test set.
"""

from moe_spam.model.bert import N_EXPERTS, N_LAYERS
from moe_spam.paths import PATHS
from moe_spam.viz.how_it_works.flowchart import PIECES, Flowchart
from moe_spam.viz.how_it_works.router_panels import (
    LEFT,
    MATH,
    draw_argmax,
    draw_balance,
    draw_equations,
    draw_scale,
    draw_slopes,
    draw_softmax,
    setup,
)
from moe_spam.viz.how_it_works.script import (
    BALANCING,
    BEATS,
    ROUTING,
    SCENE_TITLES,
    captions,
    equation_values,
    legend,
    plain,
)
from moe_spam.viz.motion import draw_pulse, locate, progress
from moe_spam.viz.style import EXPERT_COLORS, INK, INK_2, MUTED
from moe_spam.viz.video import FPS, Video

TITLE = "How Tiny BERT + Mixture of Experts works"
CAPTION_CHARS = 170
FLOW_RECT = [0.015, 0.50, 0.97, 0.335]
DETAIL_RECT = [0.015, 0.03, 0.97, 0.43]
SLOPE_RECTS = ([0.40, 0.10, 0.25, 0.27], [0.72, 0.10, 0.25, 0.27])
# scene: (equations shown, the one being worked out, the diagram's lit pieces)
ROUTING_SCENES = {"softmax": (ROUTING, 0, ("router",)),
                  "argmax": (ROUTING, 1, ("router", "experts")),
                  "scale": (ROUTING, 2, ("experts",)),
                  "gradient": (ROUTING, 2, ("router", "experts")),
                  "balance": (BALANCING, 0, ("router", "experts"))}


class HowItWorksVideo(Video):
    """The diagram and the routing equations, with a trained model's numbers.

    Attributes:
        trace: The followed message (a ``MessageTrace`` of a test spam).
        step: One of its tokens through one router (a ``RouterStep``).
        balance: One ``LayerBalance`` per layer, over the test set.
        n_messages: How many messages ``balance`` is over.
    """

    name = "how_it_works"

    def __init__(self, trace, step, balance, n_messages, paths=PATHS, fps=FPS):
        """Creates the video; see the attributes. It writes under ``paths``, at ``fps``."""
        super().__init__(paths, fps)
        self.trace = trace
        self.step = step
        self.balance = tuple(balance)
        self.n_messages = n_messages

    def check(self):
        """Raises ``ValueError`` unless the model has two layers of three experts."""
        if (len(self.trace.experts) != N_LAYERS or self.trace.n_experts != N_EXPERTS
                or len(self.balance) != N_LAYERS or len(self.step.probs) != N_EXPERTS):
            raise ValueError(f"the how-it-works video needs a model with {N_LAYERS} layers of "
                             f"{N_EXPERTS} experts")

    def segments(self):
        """Returns one segment: the scenes, with their captions as context."""
        return [(captions(self.trace, self.step, self.balance, self.n_messages), BEATS)]

    def draw(self, canvas, context, beat, t):
        """Draws one frame: the diagram on top, the scene's detail underneath."""
        scene = beat.step
        canvas.title(TITLE, SCENE_TITLES[scene], **MATH)
        canvas.caption(0.015, 0.905, context[scene], CAPTION_CHARS, **MATH)
        flow, detail = canvas.axes(FLOW_RECT), canvas.axes(DETAIL_RECT)
        setup(detail)
        if scene == "build":
            self._draw_build(flow, detail, t)
        elif scene == "flow":
            self._draw_flow(flow, detail, t)
        else:
            self._draw_routing(canvas, flow, detail, scene, t)

    def _draw_build(self, flow, detail, t):
        """The diagram appears piece by piece, each with a line on what it does."""
        shown = t * len(PIECES)
        Flowchart.draw(flow, shown=shown, lit=(PIECES[min(int(shown), len(PIECES) - 1)],))
        for k, (name, text) in enumerate(legend(self.trace, self.step)):
            alpha = min(1.0, max(0.0, shown - k))
            if alpha > 0:
                y = 23.6 - 2.45 * k
                detail.text(1, y, name, fontsize=12.5, weight="bold", color=INK, va="center",
                            alpha=alpha)
                detail.text(16, y, text, fontsize=12.5, color=INK_2, va="center", alpha=alpha,
                            **MATH)

    def _draw_flow(self, flow, detail, t):
        """The [CLS] token travels through both layers; its choices are logged below."""
        trace = self.trace
        experts = tuple(layer[0] for layer in trace.experts)
        path = Flowchart.path(experts)
        x, y, piece = locate(path, t)
        last_loop = max(k for k, (*_, name) in enumerate(path) if name == "loop")
        second = t >= progress(path, last_loop)
        Flowchart.draw(flow, lit=(piece,), chosen=experts[second], layer=1 + second)
        draw_pulse(flow, x, y, "[CLS]")
        milestones = [progress(path, k) for k, (*_, name) in enumerate(path)
                      if name == "experts"] + [1.0]
        verdict = "spam" if trace.p_spam > 0.5 else "ham"
        rows = [(EXPERT_COLORS[e], f"layer {n}: [CLS] → expert {e + 1}",
                 f"$p = {probs[e]:.2f}$")
                for n, (e, probs) in enumerate(zip(experts, trace.cls_probs, strict=True), 1)]
        rows.append((INK, "classifier",
                     rf"$p(\mathrm{{spam}}) = {trace.p_spam:.3f}$: {verdict}"))
        detail.text(1, 23.4, f"“{plain(trace.text)}”", fontsize=11, color=MUTED, va="center")
        for k, ((color, what, value), at) in enumerate(zip(rows, milestones, strict=True)):
            if t >= at - 1e-9:
                y = 18.0 - 6.0 * k
                detail.text(1, y, "●", fontsize=22, color=color, va="center")
                detail.text(4.5, y, what, fontsize=17, color=INK, va="center", weight="bold")
                detail.text(LEFT + 4, y, value, fontsize=19, color=INK_2, va="center", **MATH)

    def _draw_routing(self, canvas, flow, detail, scene, t):
        """One routing equation: the diagram's router lit, the equation and its drawing."""
        step = self.step
        equations, current, lit = ROUTING_SCENES[scene]
        balance = scene == "balance"
        Flowchart.draw(flow, lit=lit, chosen=step.expert,
                       layer=None if balance else step.layer + 1, all_experts=balance)
        draw_equations(detail, equations, current, equation_values(scene, step, self.balance))
        if scene == "softmax":
            draw_softmax(detail, step, t)
        elif scene == "argmax":
            draw_argmax(detail, step, len(step.hidden), t)
        elif scene == "scale":
            draw_scale(detail, step, t)
        elif scene == "gradient":
            draw_slopes(canvas, step, t, SLOPE_RECTS)
        else:
            draw_balance(detail, self.balance, t)
