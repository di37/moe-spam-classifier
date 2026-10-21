"""The model diagram video's script: its beats, titles, captions and every annotation.

Every number comes from ``ModelFacts`` (counted on the trained model) or from the trained
model's run on a real test message.
"""

from moe_spam.viz.video import Beat, numbered_titles

SAMPLE_CHARS = 44  # the input message under the stack is cut to this many characters
BEATS = (Beat("stack", 7.0, 1.0), Beat("flow", 6.0, 1.5), Beat("moe", 8.0, 1.5),
         Beat("ffn", 3.5, 2.0), Beat("savings", 4.0, 4.0))
SCENE_NAMES = {"stack": "The model, bottom to top",
               "flow": "One message through it",
               "moe": "Inside the MoE layer",
               "ffn": "Inside one expert",
               "savings": "What MoE saves"}
SCENE_TITLES = numbered_titles(BEATS, SCENE_NAMES)


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
        "flow": f"A test message's [CLS] token goes up through block 1, then block 2, and the "
                f"classifier reads it: p(spam) = {trace.p_spam:.3f}, {verdict}.",
        "moe": f"In each MoE layer the router sends every token to 1 of {facts.n_experts} "
               f"experts. These are layer 1's real choices for the {len(trace.words)} tokens.",
        "ffn": f"Each expert is a small feed-forward network, {facts.d_model} → {facts.d_ff} "
               f"(GELU) → {facts.d_model}: {facts.expert_params:,} parameters.",
        "savings": f"The model stores {facts.total_params:,} parameters, but a token goes "
                   f"through only {facts.active_params:,}: the other experts sit idle.",
    }
