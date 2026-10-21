"""The model diagram video's script: its beats, titles, captions and every annotation.

Every number comes from ``ModelFacts`` (counted on the trained model) or from the trained
model's run on a real test message.
"""

from dataclasses import dataclass

from moe_spam.viz.video import Beat, numbered_titles

SAMPLE_CHARS = 44  # the input message under the stack is cut to this many characters
TOP_TOKENS = 3     # the log names the tokens [CLS] takes most from
BEATS = (Beat("stack", 7.0, 1.0), Beat("flow", 11.0, 2.0), Beat("moe", 8.0, 1.5),
         Beat("ffn", 3.5, 2.0), Beat("savings", 4.0, 4.0))
SCENE_NAMES = {"stack": "The model, bottom to top",
               "flow": "One message through it",
               "moe": "Inside the MoE layer",
               "ffn": "Inside one expert",
               "savings": "What MoE saves"}
SCENE_TITLES = numbered_titles(BEATS, SCENE_NAMES)


@dataclass(frozen=True)
class LogRow:
    """One line of the [CLS] log: what one step of the model did to [CLS].

    Attributes:
        key: The step: ``tokens``, ``embed``, ``attn1``, ``moe1``, ``attn2``, ``moe2`` or
            ``output``.
        title: The step's name.
        text: What happened, with the real numbers.
        kind: The small chart beside it: ``"strip"`` (numbers as cells), ``"bars"`` (the
            router's probabilities), ``"gauge"`` (p(spam)), or None.
        values: What the chart shows.
    """

    key: str
    title: str
    text: str
    kind: str | None = None
    values: tuple = ()


def cls_rows(trace):
    """Returns the [CLS] log of a message, one ``LogRow`` per step, from its real trace."""
    words, d = trace.words, trace.d_model
    shown = " ".join(words[:8]) + (" …" if len(words) > 8 else "")
    rows = [LogRow("tokens", "Tokenized text", f"{len(words)} tokens: {shown}"),
            LogRow("embed", "Embedding", f"[CLS] becomes {d} numbers (darker = larger)",
                   "strip", tuple(trace.embeddings[0]))]
    for layer, (weights, probs) in enumerate(zip(trace.cls_attention, trace.cls_probs,
                                                 strict=True), 1):
        top = weights.argsort()[::-1][:TOP_TOKENS]
        taken = ", ".join(f"“{words[k]}” {weights[k]:.0%}" for k in top)
        expert = trace.experts[layer - 1][0] + 1
        rows += [LogRow(f"attn{layer}", f"Layer {layer} · self-attention",
                        f"[CLS] takes most from {taken}"),
                 LogRow(f"moe{layer}", f"Layer {layer} · MoE router",
                        f"[CLS] goes to expert {expert} (p = {probs.max():.2f})", "bars",
                        tuple(probs))]
    verdict = "spam" if trace.p_spam > 0.5 else "ham"
    rows.append(LogRow("output", "Classifier", f"p(spam) = {trace.p_spam:.3f}: {verdict}",
                       "gauge", (trace.p_spam,)))
    return tuple(rows)


def title(facts):
    """Returns the video's title: the model, and its size like "(444K)"."""
    return f"Tiny BERT + Mixture of Experts ({round(facts.total_params / 1000)}K)"


def sample(text):
    """Returns the input message as shown under the stack, cut to ``SAMPLE_CHARS``."""
    shown = text if len(text) <= SAMPLE_CHARS else text[:SAMPLE_CHARS - 1].rstrip() + "…"
    return f"“{shown}”".replace("$", r"\$")


def stack_notes(facts):
    """Returns the annotations beside the stack, with the real sizes."""
    return {
        "tokens": f"Vocabulary of\n{facts.vocab_size:,} tokens",
        "context": f"Context length\nof {facts.max_len} tokens",
        "embed": f"Embedding dimension\nof {facts.d_model}",
        "block": f"Both blocks\nuse an MoE layer\n({facts.n_experts} experts)",
        "attn": f"Multi-head\nself-attention ({facts.n_heads} heads)",
        "output": "One output:\nthe spam logit",
    }


def ffn_notes(facts):
    """Returns the annotations of one expert's feed-forward module."""
    return {"first": f"{facts.d_model} → {facts.d_ff}",
            "second": f"{facts.d_ff} → {facts.d_model}",
            "hidden": f"Intermediate hidden\nlayer dimension\nof {facts.d_ff}"}


def savings(facts):
    """Returns the resource-savings heading and points, with the real parameter counts."""
    active = facts.active_params / facts.total_params
    per_token = facts.expert_params + facts.router_params
    return "Resource savings:", (
        f"Model size is {facts.total_params:,} parameters",
        f"but only 1 of {facts.n_experts} experts runs per token, in each of the "
        f"{facts.n_layers} MoE layers",
        f"so only {facts.active_params:,} parameters ({active:.0%}) are active per token",
        f"each MoE layer stores {facts.moe_params:,} weights; a token uses {per_token:,} "
        f"of them (one expert and the router)",
    )


def captions(facts, trace):
    """Returns the caption of every scene.

    Args:
        facts: The trained model's ``ModelFacts``.
        trace: The message the video follows (a ``MessageTrace``).

    Returns:
        ``{scene: caption}``, each short enough for one line.
    """
    verdict = "spam" if trace.p_spam > 0.5 else "ham"
    return {
        "stack": f"The whole model, bottom to top: tokens, embeddings, {facts.n_layers} encoder "
                 f"blocks (self-attention, then an MoE layer), and a classifier.",
        "flow": f"A test message's [CLS] token goes up through block 1, then block 2; on the "
                f"right, what each step does to it. The classifier reads it: {verdict}.",
        "moe": f"In each MoE layer the router sends every token to 1 of {facts.n_experts} "
               f"experts. These are layer 1's real choices for the {len(trace.words)} tokens.",
        "ffn": f"Each expert is a small feed-forward network, {facts.d_model} → {facts.d_ff} "
               f"(GELU) → {facts.d_model}: {facts.expert_params:,} parameters.",
        "savings": f"The model stores {facts.total_params:,} parameters, but a token goes "
                   f"through only {facts.active_params:,}: the other experts sit idle.",
    }
