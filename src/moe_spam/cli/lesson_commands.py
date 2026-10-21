"""The learning-path videos: ``why-moe-video``, ``balancing-video`` and ``exercises-video``.

Each gathers its numbers from the trained models and the saved run records, then draws.
"""

from moe_spam.analysis.benchmark import LayerCost, saved_timings
from moe_spam.analysis.examples import ExampleFinder
from moe_spam.analysis.model_facts import ModelFacts
from moe_spam.cli.common import (
    Command,
    add_fps_option,
    add_video_options,
    finish,
    ranked_examples,
)
from moe_spam.data.dataset import SpamData
from moe_spam.errors import MoeSpamError
from moe_spam.model.bert import N_EXPERTS, N_LAYERS
from moe_spam.model.checkpoint import Checkpoint
from moe_spam.paths import PATHS
from moe_spam.training.config import DENSE_RUN, MOE_RUN, TrainingConfig
from moe_spam.training.evaluator import Evaluator
from moe_spam.training.runs import load_baseline, load_history, load_summary
from moe_spam.viz.balancing.script import BalancingStory
from moe_spam.viz.balancing.video import BalancingVideo
from moe_spam.viz.exercises.script import AUX_SWEEP, EXPERT_SWEEP, exercises, run_name
from moe_spam.viz.exercises.video import ExercisesVideo
from moe_spam.viz.why_moe.script import WhyMoeStory
from moe_spam.viz.why_moe.video import WhyMoeVideo

MODEL_NAMES = {"baseline": "bag-of-words", DENSE_RUN: "dense Tiny BERT",
               MOE_RUN: "Tiny BERT + MoE"}


def why_moe_story(checkpoint, data, store=None):
    """Gathers the "why MoE" video's numbers from the trained MoE model and the run records.

    Args:
        checkpoint: The trained MoE model and its vocabulary.
        data: The prepared ``SpamData``.
        store: Where the run records are (``results/`` by default).

    Returns:
        The ``WhyMoeStory``.
    """
    config = checkpoint.model.config
    trace = ranked_examples(checkpoint, data)["spam"].trace
    examples = ExampleFinder(data.vocab).first(data.test_messages)
    state = Evaluator(data, examples).evaluate(checkpoint.model).state
    baseline = load_baseline(store)
    runs = [load_summary(name, store) for name in (DENSE_RUN, MOE_RUN)]
    return WhyMoeStory(
        d_model=config.d_model, d_ff=config.d_ff, n_experts=config.n_experts,
        costs={c.name: c for c in LayerCost.for_sizes(config.d_model, config.d_ff,
                                                      config.n_experts)},
        words=trace.words, experts=trace.experts[0], shares=state.group_shares,
        group_labels=tuple(data.groups.labels(data.test_ids)),
        results=((MODEL_NAMES["baseline"], baseline["errors"], baseline["accuracy"]),
                 *((MODEL_NAMES[r.name], r.errors, r.accuracy) for r in runs)))


class WhyMoeVideoCommand(Command):
    """``moe-spam why-moe-video``: from one dense FFN to a mixture of experts."""

    name = "why-moe-video"
    help = "show why a mixture of experts: from one dense FFN to experts and a router"

    def add_arguments(self, parser):
        """Adds the video options."""
        add_video_options(parser, PATHS.checkpoint(MOE_RUN))

    def run(self, args, parser):
        """Gathers the numbers, then draws and encodes the video."""
        checkpoint = Checkpoint.load(args.checkpoint).require(n_layers=N_LAYERS,
                                                              n_experts=N_EXPERTS)
        story = why_moe_story(checkpoint, SpamData.prepare())
        print("results: " + ", ".join(f"{name} {errors} mistakes"
                                      for name, errors, _ in story.results))
        finish(WhyMoeVideo(story, fps=args.fps), args)


def balancing_story(seeds=5, store=None):
    """Gathers the load-balancing video's runs: ``seeds`` seeds with and without the loss.

    Raises:
        MoeSpamError: If a run has not been trained (``moe-spam train --seed N``, with and
            without ``--aux-weight 0``).
    """
    names = [[TrainingConfig(seed=seed, aux_weight=aux, no_video=seed > 0).run_name
              for seed in range(seeds)] for aux in (TrainingConfig().aux_weight, 0.0)]
    balanced, unbalanced = ([load_summary(name, store) for name in group] for group in names)
    return BalancingStory(balanced=tuple(balanced), unbalanced=tuple(unbalanced),
                          balanced_history=load_history(names[0][0], store),
                          unbalanced_history=load_history(names[1][0], store),
                          total_steps=TrainingConfig().steps)


class BalancingVideoCommand(Command):
    """``moe-spam balancing-video``: expert collapse without load balancing, and the fix."""

    name = "balancing-video"
    help = "show expert collapse without the load-balancing loss, over 5 seeds each"

    def add_arguments(self, parser):
        """Adds the video options (``--fps``, ``--no-video``)."""
        add_fps_option(parser)
        parser.add_argument("--no-video", action="store_true", help="only draw the frames")

    def run(self, args, parser):
        """Reads the runs, then draws and encodes the video."""
        story = balancing_story()
        print(f"layer 2 busiest expert: {story.balanced[0].busiest(2):.0%} with the loss, "
              f"{story.unbalanced[0].busiest(2):.0%} without")
        finish(BalancingVideo(story, fps=args.fps), args)


def exercises_story(store=None, paths=PATHS):
    """Gathers the exercises from the sweep runs, their checkpoints and the saved timings.

    Raises:
        MoeSpamError: If a run, a checkpoint or the timings are missing.
    """
    names = {run_name(aux_weight=w) for w in AUX_SWEEP} | {run_name(experts=k)
                                                           for k in EXPERT_SWEEP}
    runs = {name: load_summary(name, store) for name in sorted(names)}
    facts = {k: ModelFacts.of(Checkpoint.load(paths.checkpoint(run_name(experts=k))).model)
             for k in EXPERT_SWEEP}
    timings = saved_timings(store)
    if not timings or not timings.get("models"):
        raise MoeSpamError("no whole-model timings: run `moe-spam inference-video` first")
    return exercises(runs, facts, timings["models"])


class ExercisesVideoCommand(Command):
    """``moe-spam exercises-video``: predict what a change does, then see the real runs."""

    name = "exercises-video"
    help = "predict-then-check exercises on the number of experts and the balancing"

    def add_arguments(self, parser):
        """Adds the video options (``--fps``, ``--no-video``)."""
        add_fps_option(parser)
        parser.add_argument("--no-video", action="store_true", help="only draw the frames")

    def run(self, args, parser):
        """Builds the exercises from the runs, then draws and encodes the video."""
        story = exercises_story()
        for exercise in story:
            print(f"{exercise.key}: {exercise.options[exercise.answer]}")
        finish(ExercisesVideo(story, fps=args.fps), args)
