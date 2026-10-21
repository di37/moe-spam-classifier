"""The "how it works" video's script: its beats, scene titles, captions and equations."""

from moe_spam.viz.video import Beat, numbered_titles

AUX_WEIGHT = 0.01  # the weight of the load-balancing loss in the default training
SMALL_SLOPE = 1e-3  # a gradient this small gets a word on why it is small
BEATS = (Beat("build", 6.0, 1.0), Beat("flow", 5.0, 1.5), Beat("softmax", 3.0, 3.0),
         Beat("argmax", 2.0, 2.5), Beat("scale", 3.0, 3.0), Beat("gradient", 2.5, 3.5),
         Beat("balance", 3.0, 4.0))
SCENE_NAMES = {"build": "The model, box by box",
               "flow": "One message through it",
               "softmax": "p = softmax(W_router · x)",
               "argmax": "i = argmax p",
               "scale": "y = p_i · FFN_i(x)",
               "gradient": "Why multiply by p_i?",
               "balance": "Keeping every expert busy"}
SCENE_TITLES = numbered_titles(BEATS, SCENE_NAMES)
ROUTING = ("p = softmax(W_router · x)", "i = argmax p", "y = p_i · FFN_i(x)")
BALANCING = ("aux = 3 · Σ f_i · P_i", f"loss = BCE + {AUX_WEIGHT} · aux")


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
    push = "raises" if step.loss_slope < 0 else "lowers"
    if abs(step.loss_slope) < SMALL_SLOPE:
        push += " it, though only a little, as the model is already sure of this message"
    else:
        push += " it"
    return {
        "build": "The diagram from the README, drawn in the order a message goes through it: "
                 "text, tokens, vectors, two encoder blocks, and a classifier that reads one "
                 "token, [CLS], to decide.",
        "flow": f"Follow the [CLS] token of a test spam. In layer 1 its router picks expert "
                f"{e1} (p = {p1:.2f}), in layer 2 expert {e2} (p = {p2:.2f}). The final "
                f"[CLS] goes to the classifier: p(spam) = {trace.p_spam:.3f}.",
        "softmax": f"Inside layer {step.layer + 1}'s router, for the token “{step.word}”. x is "
                   f"its {len(step.x)} numbers after self-attention. Each row of W_router scores "
                   f"one expert (a dot product with x), and softmax turns the {n_experts} scores "
                   f"into probabilities that add up to 1.",
        "argmax": f"The expert with the highest probability, expert {i} (p_i = {step.p_i:.2f}), "
                  f"is the only one that runs on this token: the other {n_experts - 1} do no "
                  f"work for it. {n_experts} experts' weights, one expert's work.",
        "scale": f"Expert {i} is an ordinary feed-forward network, {len(step.x)} → {d_ff} "
                 f"(GELU) → {len(step.x)}. Its output is multiplied by p_i = {step.p_i:.2f}, "
                 f"then added back to the token, as in any residual block.",
        "gradient": f"argmax is a step: a small change to the scores does not change i, so it "
                    f"gives the router nothing to learn from. p_i changes smoothly. For this "
                    f"message dL/dp_i = {step.loss_slope:+.1e}, so training {push}. "
                    f"Through p_i, the classification loss teaches W_router.",
        "balance": f"Over the {n_messages:,} test messages, f_i is the share of tokens expert i "
                   f"gets and P_i the router's mean probability for it. aux is 1 when both are "
                   f"even and 3 when one expert takes everything; here layer 1 has "
                   f"{balance[0].aux:.2f} and layer 2 {balance[1].aux:.2f}.",
    }


def equation_values(scene, step, balance):
    """Returns the lines that put real numbers into the scene's equation, if it has one."""
    def vector(values):
        return "[" + ", ".join(f"{v:.2f}" for v in values) + "]"

    i = step.expert + 1
    return {
        "softmax": (f"= softmax({vector(step.logits)})", f"= {vector(step.probs)}"),
        "argmax": (f"= expert {i}", f"p_i = {step.p_i:.2f}"),
        "scale": (f"= {step.p_i:.2f} · FFN_{i}(x)",),
        "gradient": (f"dL/dp_i = {step.loss_slope:+.1e}",),
        "balance": (*(f"layer {n}: aux = {b.aux:.3f}" for n, b in enumerate(balance, 1)),
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
        ("SMS text", f"the message, as typed: “{text}”"),
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
        ("classifier", f"LayerNorm, Linear({d}, 1), sigmoid → p(spam) = {trace.p_spam:.3f}"),
    )
