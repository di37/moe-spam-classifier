"""The commands that make and encode videos.

``architecture-video``, ``inference-video``, ``how-it-works-video``,
``model-diagram-video`` and ``encode-video``.
"""

from moe_spam.analysis.benchmark import RECORD, SIZES, Benchmark, saved_timings
from moe_spam.analysis.examples import ExampleFinder
from moe_spam.analysis.model_facts import ModelFacts
from moe_spam.analysis.routing import RouterProbe
from moe_spam.cli.common import Command, add_fps_option, add_video_options
from moe_spam.data.dataset import SpamData
from moe_spam.data.tokenizer import PAD
from moe_spam.errors import MoeSpamError
from moe_spam.model.bert import N_EXPERTS, N_LAYERS
from moe_spam.model.checkpoint import Checkpoint
from moe_spam.paths import PATHS
from moe_spam.training.config import DENSE_RUN, MOE_RUN
from moe_spam.viz.architecture.layout import PanelGrid, fits_on_screen
from moe_spam.viz.architecture.video import EXPERTS, LAYERS, ArchitectureVideo
from moe_spam.viz.encoder import VideoEncoder
from moe_spam.viz.how_it_works.video import HowItWorksVideo
from moe_spam.viz.inference.video import MODEL_NAMES, InferenceVideo
from moe_spam.viz.model_diagram.video import ModelDiagramVideo


def _ranked_examples(checkpoint, data):
    """Returns the example messages the videos show, chosen to fit on screen."""
    size = PanelGrid.size()
    return ExampleFinder(checkpoint.vocab).ranked(
        checkpoint.model, list(data.test_messages), fits=lambda trace: fits_on_screen(trace, size))


def _finish(video, args):
    """Renders ``video`` and, unless ``--no-video``, encodes it."""
    print(f"drew {video.render()} frames")
    if not args.no_video:
        video.encode()


class ArchitectureVideoCommand(Command):
    """``moe-spam architecture-video``: one spam and one ham message through the model."""

    name = "architecture-video"
    help = "walk one spam and one ham message through the trained model"

    def add_arguments(self, parser):
        """Adds the video options."""
        add_video_options(parser, PATHS.checkpoint(MOE_RUN))

    def run(self, args, parser):
        """Picks the two messages, then draws and encodes the video."""
        checkpoint = Checkpoint.load(args.checkpoint).require(n_layers=LAYERS, n_experts=EXPERTS)
        picks = _ranked_examples(checkpoint, SpamData.prepare())
        for label, pick in picks.items():
            print(f"{label}: {pick.text!r}  p(spam) = {pick.trace.p_spam:.3f}")
        _finish(ArchitectureVideo(picks, fps=args.fps), args)


class InferenceVideoCommand(Command):
    """``moe-spam inference-video``: why MoE is cheaper to run, counted and measured."""

    name = "inference-video"
    help = "show how a mixture of experts speeds up inference, and when it does not"

    def add_arguments(self, parser):
        """Adds the video options and ``--reuse``."""
        add_video_options(parser, PATHS.checkpoint(MOE_RUN))
        parser.add_argument("--reuse", action="store_true",
                            help=f"use the timings in results/{RECORD}.json")

    def run(self, args, parser):
        """Measures (or reuses) the timings, then draws and encodes the video."""
        d_model, d_ff, n_experts = SIZES["ours"]
        moe = Checkpoint.load(args.checkpoint).require(d_model=d_model, d_ff=d_ff,
                                                       n_experts=n_experts)
        dense = self._dense_model(d_model, d_ff)
        if dense.vocab != moe.vocab:
            raise MoeSpamError("the dense and MoE checkpoints were trained on different "
                               "vocabularies")
        data = SpamData.prepare()
        pick = _ranked_examples(moe, data)["spam"]
        print(f"message: {pick.text!r}")
        measured = saved_timings() if args.reuse else None
        if not measured or not set(MODEL_NAMES) <= set(measured.get("models") or {}):
            measured = self._measure(moe, dense, pick.text, data)
        _finish(InferenceVideo(pick.trace, measured, d_ff=d_ff, fps=args.fps), args)

    @staticmethod
    def _dense_model(d_model, d_ff):
        """Loads the dense model to compare with, checking it is the same size with 1 expert."""
        path = PATHS.checkpoint(DENSE_RUN)
        if not path.is_file():
            raise MoeSpamError(f"{path} not found: train the dense model first, with "
                               f"`moe-spam train --experts 1 --no-video`")
        return Checkpoint.load(path).require(d_model=d_model, d_ff=d_ff, n_experts=1)

    @staticmethod
    def _measure(moe, dense, text, data):
        """Times the layers and both classifiers; returns the saved record."""
        print("measuring the layers and both classifiers (about a minute)...", flush=True)
        max_len = moe.model.config.max_len
        one = moe.vocab.encode([text], max_len)
        one = one[:, :int((one != PAD).sum())]  # one message alone needs no padding
        batch = moe.vocab.encode([m.text for m in data.test_messages], max_len)
        benchmark = Benchmark()
        models = benchmark.models({"dense": dense.model, "moe": moe.model}, one, batch)
        return benchmark.record(benchmark.layers(), models, len(batch))


class HowItWorksVideoCommand(Command):
    """``moe-spam how-it-works-video``: the README's diagram and routing equations, animated."""

    name = "how-it-works-video"
    help = "animate the model's diagram and the routing equations, with real numbers"
    LAYER = 0  # the equations follow a token through layer 1, where every token's route counts

    def add_arguments(self, parser):
        """Adds the video options."""
        add_video_options(parser, PATHS.checkpoint(MOE_RUN))

    def run(self, args, parser):
        """Follows the spam example's surest word through the router, then draws the video."""
        checkpoint = Checkpoint.load(args.checkpoint).require(n_layers=N_LAYERS,
                                                              n_experts=N_EXPERTS)
        data = SpamData.prepare()
        pick = _ranked_examples(checkpoint, data)["spam"]
        probe = RouterProbe(checkpoint.model, checkpoint.vocab)
        step = probe.step(pick.text, True, self.LAYER, probe.clearest(pick.text, self.LAYER))
        balance = probe.balance([m.text for m in data.test_messages])
        print(f"message: {pick.text!r}\ntoken: {step.word!r} in layer {self.LAYER + 1}, "
              f"expert {step.expert + 1} (p = {step.p_i:.2f})")
        _finish(HowItWorksVideo(pick.trace, step, balance, len(data.test_messages),
                                fps=args.fps), args)


class ModelDiagramVideoCommand(Command):
    """``moe-spam model-diagram-video``: the architecture diagram, animated with real numbers."""

    name = "model-diagram-video"
    help = "animate the model's annotated architecture diagram, with its real numbers"

    def add_arguments(self, parser):
        """Adds the video options."""
        add_video_options(parser, PATHS.checkpoint(MOE_RUN))

    def run(self, args, parser):
        """Counts the model's parameters, routes the spam example, then draws the video."""
        checkpoint = Checkpoint.load(args.checkpoint).require(n_layers=N_LAYERS,
                                                              n_experts=N_EXPERTS)
        pick = _ranked_examples(checkpoint, SpamData.prepare())["spam"]
        facts = ModelFacts.of(checkpoint.model)
        probs = RouterProbe(checkpoint.model, checkpoint.vocab).probs(pick.text, 0)
        print(f"message: {pick.text!r}\n{facts.total_params:,} parameters, "
              f"{facts.active_params:,} active per token")
        _finish(ModelDiagramVideo(facts, pick.trace, probs, fps=args.fps), args)


class EncodeVideoCommand(Command):
    """``moe-spam encode-video``: rebuild a video from saved frames."""

    name = "encode-video"
    help = "turn a folder of numbered frames into an MP4 and a GIF"

    def add_arguments(self, parser):
        """Adds the frames folder, the output path and the encoder's settings."""
        defaults = VideoEncoder()
        parser.add_argument("frames", help="folder of frames named <number>.jpg")
        parser.add_argument("out", help="output path without extension")
        add_fps_option(parser, defaults.fps)
        parser.add_argument("--gif-every", type=int, default=defaults.gif_every,
                            choices=range(1, 31), metavar="N",
                            help="keep every N-th frame in the GIF")

    def run(self, args, parser):
        """Encodes the frames."""
        VideoEncoder(fps=args.fps, gif_every=args.gif_every).encode(args.frames, args.out)
