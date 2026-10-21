r"""The "how it works" video's script: its beats, scene titles, captions and equations.

Equations are written in matplotlib's LaTeX-like mathtext (between ``$`` signs). Math in a
caption never contains a space, so wrapping a caption never splits an equation; mathtext
spaces ``=`` and the like by itself. Operators are braced, as in ``{\sum}_i``, so their
subscripts sit beside them at text size instead of underneath.
"""

from moe_spam.viz.video import Beat, numbered_titles

AUX_WEIGHT = 0.01  # the weight of the load-balancing loss in the default training
SMALL_SLOPE = 1e-3  # a gradient this small gets a word on why it is small
BEATS = (Beat("build", 6.0, 1.0), Beat("flow", 5.0, 1.5), Beat("softmax", 3.0, 3.0),
         Beat("argmax", 2.0, 2.5), Beat("scale", 3.0, 3.0), Beat("gradient", 2.5, 3.5),
         Beat("balance", 3.0, 4.0))
ROUTING = (r"$p = \mathrm{softmax}(W_{\mathrm{router}}\,x)$",
           r"$i = \mathrm{arg\,max}_j\, p_j$",
           r"$y = p_i \cdot \mathrm{FFN}_i(x)$")
BALANCING = (r"$\mathrm{aux} = 3\, {\sum}_i\, f_i\, P_i$",
             rf"$\mathcal{{L}} = \mathrm{{BCE}} + {AUX_WEIGHT} \cdot \mathrm{{aux}}$")
SCENE_NAMES = {"build": "The model, box by box",
               "flow": "One message through it",
               "softmax": ROUTING[0],
               "argmax": ROUTING[1],
               "scale": ROUTING[2],
               "gradient": "Why multiply by $p_i$?",
               "balance": "Keeping every expert busy"}
SCENE_TITLES = numbered_titles(BEATS, SCENE_NAMES)
W_ROUTER = r"$W_{\mathrm{router}}$"
SLOPE = r"\partial{L}/\partial{p_i}"  # math, without the $ signs


def plain(text):
    """Returns ``text`` with its ``$`` signs escaped, so a message is never read as math."""
    return text.replace("$", r"\$")


def scientific(value):
    r"""Returns ``value`` signed, to two figures, as a power of ten: ``-7.7 \times 10^{-6}``."""
    mantissa, exponent = f"{value:+.1e}".split("e")
    return rf"{mantissa} \times 10^{{{int(exponent)}}}"


def captions(trace, step, balance, n_messages):
    """Returns the caption of every scene.

    Args:
        trace: The message the video follows (a ``MessageTrace``).
        step: One of its tokens through one router (a ``RouterStep``).
        balance: One ``LayerBalance`` per layer, over the test messages.
        n_messages: How many messages ``balance`` is over.

    Returns:
        ``{scene: caption}``.
    """
    e1, e2 = (experts[0] + 1 for experts in trace.experts)  # [CLS]'s expert in each layer
    p1, p2 = (float(probs.max()) for probs in trace.cls_probs)
    i, n_experts, d_ff = step.expert + 1, len(step.probs), len(step.hidden)
    push = f"{'raises' if step.loss_slope < 0 else 'lowers'} $p_i$"
    if abs(step.loss_slope) < SMALL_SLOPE:
        push += ", if only a little (the model is already sure)"
    slope = scientific(step.loss_slope).replace(" ", "")
    return {
        "build": "The diagram from the README, drawn in the order a message goes through it: "
                 "text, tokens, vectors, two encoder blocks, and a classifier that reads one "
                 "token, [CLS], to decide.",
        "flow": f"Follow the [CLS] token of a test spam. In layer 1 its router picks expert "
                f"{e1} ($p={p1:.2f}$), in layer 2 expert {e2} ($p={p2:.2f}$). The final "
                rf"[CLS] goes to the classifier: $p(\mathrm{{spam}})={trace.p_spam:.3f}$.",
        "softmax": f"Inside layer {step.layer + 1}'s router, for the token “{plain(step.word)}”. "
                   f"$x$ is its {len(step.x)} numbers after self-attention. Each row of "
                   f"{W_ROUTER} scores one expert (a dot product with $x$), and softmax turns "
                   f"the {n_experts} scores into probabilities that add up to 1.",
        "argmax": f"The expert with the highest probability, expert {i} "
                  f"($p_i={step.p_i:.2f}$), is the only one that runs on this token: the other "
                  f"{n_experts - 1} do no work for it. {n_experts} experts' weights, one "
                  f"expert's work.",
        "scale": f"Expert {i} is an ordinary feed-forward network, {len(step.x)} → {d_ff} "
                 f"(GELU) → {len(step.x)}. Its output is multiplied by $p_i={step.p_i:.2f}$, "
                 f"then added back to the token, as in any residual block.",
        "gradient": f"argmax is a step: nudging the scores does not change $i$, so it gives "
                    f"nothing to learn from, while $p_i$ changes smoothly. Here "
                    f"${SLOPE}={slope}$: training {push}. Through $p_i$, the loss teaches "
                    f"{W_ROUTER}.",
        "balance": f"Over the {n_messages:,} test messages, $f_i$ is the share of tokens "
                   f"expert $i$ gets and $P_i$ the router's mean probability for it. "
                   rf"$\mathrm{{aux}}$ is 1 when both are even and 3 when one expert takes "
                   f"everything; here layer 1 has {balance[0].aux:.2f} and layer 2 "
                   f"{balance[1].aux:.2f}.",
    }


def equation_values(scene, step, balance):
    """Returns the lines that put real numbers into the scene's equation, if it has one."""
    def vector(values):
        return "[" + ", ".join(f"{v:.2f}" for v in values) + "]"

    i = step.expert + 1
    return {
        "softmax": (rf"$= \mathrm{{softmax}}({vector(step.logits)})$",
                    f"$= {vector(step.probs)}$"),
        "argmax": (f"$i = {i}$", f"$p_i = {step.p_i:.2f}$"),
        "scale": (rf"$= {step.p_i:.2f} \cdot \mathrm{{FFN}}_{{{i}}}(x)$",),
        "gradient": (f"${SLOPE} = {scientific(step.loss_slope)}$",),
        "balance": (*(rf"layer {n}: $\mathrm{{aux}} = {b.aux:.3f}$"
                      for n, b in enumerate(balance, 1)),
                    f"(1: even, {len(step.probs)}: one expert gets all)"),
    }.get(scene, ())


def legend(trace, step):
    """Returns ``(name, what it does)`` for each piece of the diagram, in the order it is built.

    Args:
        trace: The message the video follows (a ``MessageTrace``).
        step: One of its tokens through one router (a ``RouterStep``).
    """
    d, d_ff, n_experts = trace.d_model, len(step.hidden), len(step.probs)
    text = trace.text if len(trace.text) <= 70 else trace.text[:67] + "..."
    return (
        ("SMS text", f"the message, as typed: “{plain(text)}”"),
        ("tokens", f"lowercase words, numbers as [NUM] or [NUM5+], [CLS] in front: "
                   f"{len(trace.words)} tokens here"),
        ("embeddings", f"each token becomes {d} numbers: a vector for the token plus one for "
                       f"its position"),
        ("encoder block", "self-attention, then a mixture-of-experts layer"),
        ("self-attention", "every token takes in information from all the others"),
        ("router", "scores the experts for each token, and picks the top one (top-1)"),
        ("experts", f"{n_experts} feed-forward networks, {d} → {d_ff} → {d}; only the picked "
                    f"one runs"),
        ("×2", "the block runs twice: layer 1, then layer 2, each with its own weights"),
        ("final [CLS]", "after layer 2, the [CLS] token sums up the whole message"),
        ("classifier", rf"LayerNorm, Linear({d}, 1), sigmoid → $p(\mathrm{{spam}})="
                       rf"{trace.p_spam:.3f}$"),
    )
