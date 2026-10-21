"""The inference video's script: its beats, scene titles and captions."""

from moe_spam.analysis.benchmark import LAYERS
from moe_spam.viz.video import Beat, numbered_titles

SAME_SPEED = 0.05  # timings within 5% of each other count as "as fast"
BEATS = (Beat("setup", 1.0, 1.5), Beat("stream", 9.0, 0.5), Beat("totals", 1.0, 3.5),
         Beat("throughput", 1.0, 5.0), Beat("latency", 1.0, 5.0), Beat("models", 1.0, 5.0))
SCENE_NAMES = {"setup": "Three layers, the same tokens",
               "stream": "A message goes through",
               "totals": "Work done, weights held",
               "throughput": "Measured: many tokens at once",
               "latency": "Measured: one message at a time",
               "models": "Measured: the whole classifier"}
SCENE_TITLES = numbered_titles(BEATS, SCENE_NAMES)


def compare(value, reference):
    """Returns how a time compares with a reference.

    For example ``"2.0× slower"``, ``"4.0× faster"`` or ``"as fast"``.
    """
    ratio = value / reference
    if abs(ratio - 1) < SAME_SPEED:
        return "as fast"
    return f"{ratio:.1f}× slower" if ratio > 1 else f"{1 / ratio:.1f}× faster"


def captions(trace, costs, measured, n_experts):
    """Returns the caption of every scene.

    Args:
        trace: The message the video streams (a ``MessageTrace``).
        costs: ``{layer: LayerCost}`` at this project's size.
        measured: What ``Benchmark.record`` saved; without it, only the counted scenes have
            captions.
        n_experts: The number of experts.

    Returns:
        ``{scene: caption}``. The measured scenes say "faster" or "slower" from the numbers.
    """
    dense, wide, moe = (costs[k].macs_per_token for k in LAYERS)
    text = {
        "setup": f"Three versions of the same feed-forward layer, fed the same tokens (each a "
                 f"list of {trace.d_model} numbers). The rest of the model would be identical.",
        "stream": f"For each token the dense layer does {dense:,} multiply-adds, and the one "
                  f"{n_experts}× wider {wide:,}. The MoE layer holds the same weights as the "
                  f"wide one but runs only the expert its router picks (here, the trained "
                  f"model's layer-1 choices), so it does {moe:,}.",
        "totals": f"After {len(trace.words)} tokens the MoE layer has done {moe / wide:.0%} of "
                  f"the wide layer's work, about as much as the small dense layer, yet it holds "
                  f"as many weights as the wide one. MoE saves compute, not memory.",
    }
    if not measured:
        return text
    return {**text, **_measured_captions(trace, measured, n_experts)}


def _measured_captions(trace, measured, n_experts):
    sizes = measured["sizes"]
    ours, large = sizes["ours"]["ms_per_1k_tokens"], sizes["large"]["ms_per_1k_tokens"]
    throughput = (
        f"Measured on this computer, with many tokens at once ({sizes['ours']['batch_tokens']:,} "
        f"per batch), as a server busy with many messages would run. At this project's tiny "
        f"size the routing overhead eats the saving: the MoE layer is "
        f"{compare(ours['moe'], ours['wide'])} than the wide one (this varies a lot from run "
        f"to run). With large experts it is "
        f"{compare(large['moe'], large['wide'])} than a dense layer with the same weights "
        f"({sizes['large']['n_experts']}× would be perfect), and "
        f"{compare(large['moe'], large['dense'])} than one expert alone.")
    ours, large = sizes["ours"]["latency_us"], sizes["large"]["latency_us"]
    used = len(set(trace.experts[0]))
    used = f"all {n_experts}" if used == n_experts else f"{used} of the {n_experts}"
    latency = (
        f"One message of {sizes['ours']['latency_tokens']} tokens at a time, the way a spam filter "
        f"is called. Its tokens spread over almost every expert (this message uses {used} in "
        f"layer 1), so almost every expert's "
        f"weights must be read anyway, and each expert adds its own overhead. The MoE layer is "
        f"{compare(ours['moe'], ours['wide'])} than the wide one at this project's size, and "
        f"{compare(large['moe'], large['wide'])} at the large size. MoE's speed-up needs big "
        f"batches.")
    dense_model, moe_model = measured["models"]["dense"], measured["models"]["moe"]
    models = (
        f"The two trained classifiers, end to end: attention, embeddings and the classifier cost "
        f"the same in both. The MoE model does the dense model's work per token plus routing, so "
        f"here it can only lose: it is "
        f"{compare(moe_model['one_message_ms'], dense_model['one_message_ms'])} for one message, "
        f"and {compare(moe_model['ms_per_1k_messages'], dense_model['ms_per_1k_messages'])} for "
        f"all {measured['test_messages']:,} test messages in one batch.")
    return {"throughput": throughput, "latency": latency, "models": models}
