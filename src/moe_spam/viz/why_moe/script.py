"""The "why MoE" video's script: its beats, titles, captions and the numbers it shows."""

from dataclasses import dataclass

from moe_spam.viz.video import Beat, numbered_titles

TITLE = "Why a mixture of experts? From one FFN to many"
BEATS = (Beat("dense", 4.0, 1.5), Beat("wide", 3.0, 1.5), Beat("moe", 4.5, 2.0),
         Beat("tradeoff", 3.0, 3.0), Beat("specialize", 3.0, 3.5), Beat("verdict", 3.0, 3.5))
SCENE_NAMES = {"dense": "One FFN for every token",
               "wide": "More capacity, more compute",
               "moe": "Experts and a router",
               "tradeoff": "Capacity without the compute",
               "specialize": "The experts specialize",
               "verdict": "At this size: no gain"}
SCENE_TITLES = numbered_titles(BEATS, SCENE_NAMES)


@dataclass(frozen=True)
class WhyMoeStory:
    """Everything the video shows, measured on the trained models.

    Attributes:
        d_model: The width of a token vector.
        d_ff: The hidden width of the dense FFN and of each expert.
        n_experts: Experts per MoE layer.
        costs: ``{"dense" | "wide" | "moe": LayerCost}`` of one feed-forward layer.
        words: The first tokens of a real test message.
        experts: The expert the trained router sends each of them to, in layer 1.
        shares: ``[groups, n_experts]``: the share of each kind of test token each layer-1
            expert gets.
        group_labels: The kinds of token.
        results: ``(name, test errors, test accuracy)`` of each compared model.
    """

    d_model: int
    d_ff: int
    n_experts: int
    costs: dict
    words: tuple
    experts: tuple
    shares: object
    group_labels: tuple
    results: tuple


def captions(story):
    """Returns the caption of every scene, with the story's numbers."""
    dense, wide, moe = (story.costs[k] for k in ("dense", "wide", "moe"))
    e = story.n_experts
    best = min(story.results, key=lambda r: r[1])
    return {
        "dense": f"In a Transformer block, every token goes through the same feed-forward network "
                 f"(FFN): {dense.weights:,} weights, {dense.macs_per_token:,} multiply-adds per "
                 f"token.",
        "wide": f"To give the model more capacity, make the FFN {e}× wider: {wide.weights:,} "
                f"weights, but every token now costs {wide.macs_per_token:,} multiply-adds.",
        "moe": f"An MoE layer cuts the wide FFN into {e} experts and adds a router that sends each "
               f"token to one of them. These are the trained router's real choices.",
        "tradeoff": f"The MoE layer stores {moe.weights:,} weights, as many as the wide FFN, but a "
                    f"token costs {moe.macs_per_token:,} multiply-adds, as few as the dense one.",
        "specialize": "Each expert ends up handling different kinds of tokens: the trained "
                      "layer-1 router's real split of the test set's tokens.",
        "verdict": f"On this small task the extra capacity does not help: fewest mistakes, "
                   f"{best[1]}, by the {best[0]}. MoE pays off in models far larger than this one.",
    }


def numbers(story):
    """Returns ``{layer: (weights label, multiply-adds label)}`` for the cost bars."""
    return {name: (f"{c.weights:,}", f"{c.macs_per_token:,}")
            for name, c in story.costs.items()}
