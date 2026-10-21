"""The video on how a mixture of experts speeds up inference, and when it does not.

Three versions of one feed-forward layer see the same tokens of a real test message: a dense
network, one 3 times wider (as many weights as the MoE layer), and the MoE layer, where the
trained model's layer-1 router sends each token to one expert. Counters add up the work each
layer does; then come timings measured on this computer: many tokens at once (throughput),
one message at a time (latency), and the whole trained dense and MoE classifiers.
"""

from moe_spam.analysis.benchmark import LAYERS, LayerCost
from moe_spam.model.bert import N_EXPERTS
from moe_spam.paths import PATHS
from moe_spam.viz.inference.lanes import Lanes, draw_bars, lane_labels
from moe_spam.viz.inference.script import BEATS, SCENE_TITLES, captions
from moe_spam.viz.video import FPS, Video

TITLE = "How a mixture of experts speeds up inference"
CAPTION_CHARS = 170
EXPERTS = N_EXPERTS
HEADROOM = 1.35  # each chart's axis runs to this times its longest bar, leaving room for labels
SLOW_TOKENS, SLOW_SHARE = 4, 0.5  # the first tokens take half the stream, then it speeds up
LIT_SHARE = 0.8  # each token's layers are lit for this much of its turn
MODEL_NAMES = {"dense": "dense model (1 expert)", "moe": f"MoE model ({N_EXPERTS} experts)"}
LANES_RECT = [0.015, 0.03, 0.6, 0.76]
TOP_CHART, BOTTOM_CHART = [0.70, 0.47, 0.27, 0.25], [0.70, 0.09, 0.27, 0.25]


def token_progress(t, n):
    """Returns how many of ``n`` tokens have gone through at progress ``t``, as a float.

    The first ``SLOW_TOKENS`` take ``SLOW_SHARE`` of the stream, so they can be followed;
    the rest go faster.
    """
    slow = min(SLOW_TOKENS, n)
    if t <= SLOW_SHARE:
        return slow * t / SLOW_SHARE
    return min(n, slow + (n - slow) * (t - SLOW_SHARE) / (1 - SLOW_SHARE))


def tokens_done(t, n):
    """Returns how many whole tokens have gone through at progress ``t``."""
    return int(token_progress(t, n))


class InferenceVideo(Video):
    """Counted work and weights, then measured time, of dense and MoE layers.

    Attributes:
        trace: The streamed message (a ``MessageTrace`` of the trained MoE model).
        measured: What ``Benchmark.record`` saved.
        d_ff: The width of one expert.
        costs: ``{layer: LayerCost}`` at this project's size.
    """

    name = "inference"

    def __init__(self, trace, measured, d_ff=None, paths=PATHS, fps=FPS):
        """Creates the video of ``trace``, with ``measured`` timings.

        Args:
            trace: The streamed message.
            measured: The benchmark record, with layer timings at both sizes and model
                timings (what ``Benchmark.record`` saves).
            d_ff: The width of one expert; by default, this project's measured size.
            paths: Where the frames and the video go.
            fps: Frames per second.
        """
        super().__init__(paths, fps)
        self.trace = trace
        self.measured = measured
        self.d_ff = d_ff or measured["sizes"]["ours"]["d_ff"]
        self.costs = {c.name: c for c in LayerCost.for_sizes(trace.d_model, self.d_ff, EXPERTS)}

    def check(self):
        """Raises ``ValueError`` unless the model has three experts."""
        if self.trace.n_experts != EXPERTS:
            raise ValueError(f"the inference video needs a model with {EXPERTS} experts")

    def segments(self):
        """Returns one segment: the scenes, with this video's captions as context."""
        return [(captions(self.trace, self.costs, self.measured, EXPERTS), BEATS)]

    def draw(self, canvas, context, beat, t):
        """Draws one frame: the three lanes and the scene's charts."""
        scene, trace, n = beat.step, self.trace, len(self.trace.words)
        streaming = scene == "stream"
        done = {"setup": 0.0, "stream": token_progress(t, n)}.get(scene, float(n))
        current = min(int(done), n - 1)
        # A token counts as soon as it goes in, so the bars and the title always agree.
        counted = min(n, int(done) + 1) if streaming else int(done)
        canvas.title(TITLE, SCENE_TITLES[scene])
        canvas.caption(0.015, 0.905, context[scene], CAPTION_CHARS)
        Lanes(self.costs, self.d_ff, EXPERTS).draw(
            canvas.axes(LANES_RECT), trace.words[current] if streaming else None,
            trace.experts[0][current], streaming and done - int(done) < LIT_SHARE)
        if scene in ("throughput", "latency"):
            self._draw_layer_timings(canvas, scene)
        elif scene == "models":
            self._draw_model_timings(canvas)
        else:
            self._draw_counts(canvas, counted, n)

    def _draw_counts(self, canvas, counted, n):
        names = _short_names(self.d_ff, EXPERTS)
        work = {k: counted * self.costs[k].macs_per_token for k in LAYERS}
        draw_bars(canvas.axes(TOP_CHART), work, names,
                  HEADROOM * n * self.costs["wide"].macs_per_token,
                  f"Multiply-adds so far ({counted} of {n} tokens)")
        weights = {k: self.costs[k].weights for k in LAYERS}
        draw_bars(canvas.axes(BOTTOM_CHART), weights, names, HEADROOM * max(weights.values()),
                  "Weights held in memory")

    def _draw_layer_timings(self, canvas, scene):
        """Throughput or latency of the three layers, at this project's size and a large one."""
        for rect, key, name in ((TOP_CHART, "ours", "this project's size"),
                                (BOTTOM_CHART, "large", "a large layer")):
            size = self.measured["sizes"][key]
            names = _short_names(size["d_ff"], size["n_experts"])
            shape = f"{size['d_model']:,} → {size['d_ff']:,}, {size['n_experts']} experts"
            if scene == "throughput":
                values, fmt = size["ms_per_1k_tokens"], (lambda value: f"{value:.2f} ms")
                title = f"Many tokens at once: {name}\n{shape} · ms per 1,000 tokens"
            else:
                values, fmt = size["latency_us"], (lambda value: f"{value:,.0f} µs")
                title = (f"One message ({size['latency_tokens']} tokens): {name}\n"
                         f"{shape} · microseconds")
            draw_bars(canvas.axes(rect), values, names, HEADROOM * max(values.values()), title,
                      fmt)

    def _draw_model_timings(self, canvas):
        models, keys = self.measured["models"], tuple(MODEL_NAMES)
        one = {k: models[k]["one_message_ms"] for k in keys}
        draw_bars(canvas.axes(TOP_CHART), one, MODEL_NAMES, HEADROOM * max(one.values()),
                  "Whole classifier, one message\nmilliseconds", lambda v: f"{v:.2f} ms", keys)
        batch = {k: models[k]["ms_per_1k_messages"] for k in keys}
        draw_bars(canvas.axes(BOTTOM_CHART), batch, MODEL_NAMES, HEADROOM * max(batch.values()),
                  f"Whole classifier, all {self.measured['test_messages']:,} test messages at "
                  f"once\nmilliseconds per 1,000 messages", lambda v: f"{v:.1f} ms", keys)


def _short_names(d_ff, n_experts):
    """Returns ``{layer: short name}``, for the charts."""
    return {k: v[2] for k, v in lane_labels(d_ff, n_experts).items()}
