"""The exercises: questions about changing the experts or the balancing, answered by real runs.

Every answer is worked out from the runs' numbers with the rules below, so a different
training result would change the answer, not just the chart.
"""

from dataclasses import dataclass

from moe_spam.training.config import TrainingConfig
from moe_spam.viz.style import INK, MUTED
from moe_spam.viz.video import Beat

TITLE = "Predict, then check: MoE exercises"
BEATS = (Beat("question", 2.5, 0.5), Beat("predict", 3.0, 0.0), Beat("reveal", 2.5, 3.5))
SAME_ERRORS = 3           # mistakes within this many count as "about the same"
SAME_RATIO = 0.05         # times or counts within 5% count as "about the same"
EVEN_ENOUGH, COLLAPSED = 0.45, 0.6  # the busiest expert's share: about even, or collapsed
AUX_SWEEP = (0.0, 0.001, 0.01, 0.1, 1.0)
EXPERT_SWEEP = (1, 2, 3, 4, 8)


@dataclass(frozen=True)
class Series:
    """One set of bars in an exercise's chart.

    Attributes:
        name: What the bars measure (for the legend; empty for a single series).
        values: One value per bar.
        color: The bars' color.
    """

    name: str
    values: tuple
    color: str


@dataclass(frozen=True)
class Exercise:
    """A question, its choices, the answer the runs give, and the chart that shows why.

    Attributes:
        key: A short name.
        question: What the viewer predicts.
        options: The three choices.
        answer: The index of the right choice, from the runs.
        chart_title: The title of the chart revealed.
        labels: One label per bar group.
        series: The ``Series`` of bars.
        fmt: How a value is written on its bar, e.g. ``"{:.0%}"``.
        highlight: The bar groups the question is about.
        reference: ``(value, label)`` of a dashed line, or None.
        takeaway: One line on what the answer means.
    """

    key: str
    question: str
    options: tuple
    answer: int
    chart_title: str
    labels: tuple
    series: tuple
    fmt: str
    highlight: tuple
    reference: tuple | None
    takeaway: str


def closest(value, choices):
    """Returns the index of the choice nearest ``value``."""
    return min(range(len(choices)), key=lambda k: abs(choices[k] - value))


def three_way(value, reference, absolute=None, relative=None):
    """Returns 0 if ``value`` is clearly below ``reference``, 2 if clearly above, else 1.

    "Clearly" means by more than ``absolute``, or by more than ``relative`` of the reference.
    """
    margin = absolute if absolute is not None else relative * reference
    return 0 if value < reference - margin else 2 if value > reference + margin else 1


def run_name(**settings):
    """Returns the run name of a training with ``settings`` (all others at their defaults)."""
    return TrainingConfig(**settings, no_video=settings.get("experts", 3) != 3).run_name


def exercises(runs, facts, timings):
    """Builds the exercises from the trained runs.

    Args:
        runs: ``{run name: RunSummary}``, with every run of ``AUX_SWEEP`` and
            ``EXPERT_SWEEP``.
        facts: ``{experts: ModelFacts}`` for every count in ``EXPERT_SWEEP``.
        timings: The whole-model timings of the benchmark record (``dense`` and ``moe``).

    Returns:
        The exercises, in the order the video asks them.
    """
    aux = [runs[run_name(aux_weight=w)] for w in AUX_SWEEP]
    sized = [runs[run_name(experts=k)] for k in EXPERT_SWEEP]
    return (_collapse(aux), _tiny(aux), _strong(aux), _params(facts), _accuracy(sized),
            _speed(timings))


def _collapse(aux):
    off, on = aux[0].busiest(2), aux[2].busiest(2)
    shares = (1 / 3, 0.5, 0.8)
    return Exercise(
        "collapse", "Train without the load-balancing loss (aux weight 0). In layer 2, what "
        "share of the tokens will the busiest of the 3 experts get?",
        ("about 33%: an even split", "about 50%", "about 80%"), closest(off, shares),
        "Layer 2: the busiest expert's share of tokens", ("aux weight 0.01", "aux weight 0"),
        (Series("", (on, off), INK),), "{:.0%}", (1,), (1 / 3, "even split"),
        f"Without the loss one expert takes {off:.0%} of layer 2's tokens, and the other two "
        f"sit almost idle. With it: {on:.0%}.")


def _tiny(aux):
    share = aux[1].busiest(2)
    answer = 0 if share < EVEN_ENOUGH else 1 if share < COLLAPSED else 2
    stronger = [r.busiest(2) for r in aux[2:]]
    after = (f"From {aux[2].aux_weight:g} up the busiest expert gets {min(stronger):.0%}–"
             f"{max(stronger):.0%}" if all(v < EVEN_ENOUGH for v in stronger) else
             f"Stronger weights reach {min(stronger):.0%}")
    return Exercise(
        "tiny", "A balancing weight 10× smaller than ours, 0.001: is it enough to stop layer 2 "
        "from collapsing?", ("yes: about even", "partly", "no: it still collapses"), answer,
        "Layer 2: the busiest expert's share, by balancing weight",
        tuple(f"{w:g}" for w in AUX_SWEEP), (Series("", tuple(r.busiest(2) for r in aux), INK),),
        "{:.0%}", (1,), (1 / 3, "even split"),
        f"At 0.001 the busiest expert gets {share:.0%}. {after}: the weight has to be large "
        f"enough to win against the pull of the classification loss.")


def _strong(aux):
    ours, strong = aux[2].errors, aux[4].errors
    answer = (2, 1, 0)[three_way(strong, ours, absolute=SAME_ERRORS)]
    meaning = (f"more than {SAME_ERRORS} more: the balancing gets in the classifier's way",
               f"within {SAME_ERRORS}, about what a different seed alone changes",
               f"more than {SAME_ERRORS} fewer")[answer]
    return Exercise(
        "strong", "Make the balancing 100× stronger (weight 1.0 instead of 0.01). Will the model "
        "make more mistakes?", ("more mistakes", "about the same", "fewer mistakes"), answer,
        "Test mistakes, by balancing weight", tuple(f"{r.aux_weight:g}" for r in aux),
        (Series("", tuple(r.errors for r in aux), INK),), "{:d}", (2, 4), None,
        f"{strong} mistakes at 1.0 against {ours} at 0.01: {meaning}.")


def _params(facts):
    sizes = [facts[k] for k in EXPERT_SWEEP]
    three, eight = facts[3], facts[8]
    ratio = eight.active_params / three.active_params
    answer = 0 if abs(ratio - 1) <= SAME_RATIO else closest(ratio, (1, 8 / 3, 8))
    return Exercise(
        "params", "Give each MoE layer 8 experts instead of 3. How many parameters does one "
        "token use?", ("about the same", f"about {8 / 3:.1f}× as many", "about 8× as many"),
        answer, "Parameters, by experts per layer", tuple(str(k) for k in EXPERT_SWEEP),
        (Series("stored", tuple(f.total_params for f in sizes), INK),
         Series("used per token", tuple(f.active_params for f in sizes), MUTED)),
        "{:,}", (2, 4), None,
        f"The model grows from {three.total_params:,} to {eight.total_params:,} parameters, "
        f"but a token still uses {eight.active_params:,}: one expert per layer.")


def _accuracy(sized):
    three, eight = sized[2].errors, sized[4].errors
    answer = three_way(eight, three, absolute=SAME_ERRORS)
    meaning = ("the extra capacity helps", "a task this small does not need the extra capacity",
               "the extra experts get in the way")[answer]
    return Exercise(
        "accuracy", "With 8 experts instead of 3, will the model make fewer mistakes?",
        ("clearly fewer", "about the same", "clearly more"), answer,
        "Test mistakes, by experts per layer", tuple(str(r.experts) for r in sized),
        (Series("", tuple(r.errors for r in sized), INK),), "{:d}", (2, 4), None,
        f"{eight} mistakes with 8 experts, {three} with 3: {meaning}.")


def _speed(timings):
    dense, moe = timings["dense"]["one_message_ms"], timings["moe"]["one_message_ms"]
    answer = three_way(moe, dense, relative=SAME_RATIO)
    meaning = ("the experts' saved work shows up as time", "the saving and the routing cost "
               "even out", "routing each token and running the experts one by one cost more "
               "than MoE saves at this size")[answer]
    return Exercise(
        "speed", "Does the MoE model answer one message faster than the dense model, which does "
        "the same work per token?", ("faster", "about as fast", "slower"),
        answer, "One message, whole model (milliseconds)",
        ("dense", "MoE"), (Series("", (dense, moe), INK),), "{:.2f} ms", (1,), None,
        f"{moe:.1f} ms against {dense:.1f} ms: {meaning}.")
