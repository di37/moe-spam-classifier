"""The architecture video's script: its beats, step titles and captions."""

from moe_spam.data.tokenizer import TOKENIZER
from moe_spam.model.bert import N_EXPERTS
from moe_spam.viz.video import Beat, numbered_titles

CLOSE_CALL = 0.03  # say so when the router's top two probabilities for [CLS] are this close

# The ham message skips the embeddings and goes faster: the steps are the same.
BEATS = {
    "spam": (Beat("text", 1.0, 1.5), Beat("tokenize", 2.0, 1.5), Beat("embed", 1.5, 1.5),
             Beat("attn1", 1.5, 2.0), Beat("route1", 2.0, 2.5), Beat("back1", 1.0, 0.5),
             Beat("attn2", 1.5, 1.5), Beat("route2", 2.0, 3.5), Beat("classify", 2.5, 3.0)),
    "ham": (Beat("text", 0.7, 1.0), Beat("tokenize", 1.0, 0.8), Beat("attn1", 1.0, 1.0),
            Beat("route1", 1.5, 1.5), Beat("back1", 0.7, 0.3), Beat("attn2", 1.0, 0.8),
            Beat("route2", 1.5, 3.0), Beat("classify", 2.0, 1.0)),
}
STEP_NAMES = {
    "text": "A message comes in",
    "tokenize": "Tokenizer",
    "embed": "Embeddings",
    "attn1": "Layer 1: self-attention",
    "route1": "Layer 1: router and experts",
    "back1": "Layer 1: router and experts",
    "attn2": "Layer 2: self-attention",
    "route2": "Layer 2: router and experts",
    "classify": "Classifier",
}
CAPTIONS = {
    "text": "A real text message from the test set: the model never saw it during training.",
    "tokenize": "The text is cut into tokens: lowercase words, numbers and punctuation. Numbers "
                f"become [NUM] (up to {TOKENIZER.long_number - 1} digits) or [NUM5+]. A [CLS] "
                "token goes in front: it will collect a summary of the whole message.",
    "embed": "Each token becomes a list of {d_model} numbers (8 shown, darker = larger): a "
             "learned vector for the token plus one for its position.",
    "attn1": "Layer 1 starts with self-attention: every token takes in information from all the "
             "others. The bars show how much the [CLS] token takes from each one (averaged "
             "over the attention heads).",
    "route1": f"Then the router sends each token to one of {N_EXPERTS} experts, small feed-forward "
              "networks, and only that expert works on it. The training video's token panels "
              "show these choices.",
    "back1": "Each expert's output is added back to its token, and the updated tokens move on "
             "to layer 2.",
    "attn2": "Layer 2 repeats both steps with its own weights. Self-attention again: [CLS] "
             "gathers from the updated tokens.",
    "classify": "Only the final [CLS] is read: a LayerNorm, one linear layer and a sigmoid turn "
                "its {d_model} numbers into p(spam). Above 0.5 means spam.",
}


def step_titles(label):
    """Returns ``{step: title}``, numbered in the order the ``label`` message shows them."""
    return numbered_titles(BEATS[label], STEP_NAMES)


def caption(step, pick, label):
    """Returns the caption of ``step`` for an example message.

    Args:
        step: The step being shown.
        pick: The example (a ``Pick``).
        label: ``"spam"`` or ``"ham"``.

    Returns:
        The caption. Layer 2's routing step says where this message's ``[CLS]`` goes, how
        many messages of its label go there too, and whether the decision was a close call.
    """
    if step != "route2":
        return CAPTIONS[step].format(d_model=pick.trace.d_model)
    expert = pick.trace.experts[-1][0] + 1
    how_many = (f"all {pick.total}" if pick.same == pick.total
                else f"{pick.same} of the {pick.total}")
    text = (f"Layer 2 repeats the router step. The training video's triangle plots one of its "
            f"choices: where each message's [CLS] goes. This one goes to expert {expert}, as "
            f"does the [CLS] of {how_many} {label} messages in the test set.")
    first, second = sorted(pick.trace.cls_probs[-1], reverse=True)[:2]
    if first - second < CLOSE_CALL:
        text += (f" A close call: the router gives {first:.3f} to expert {expert} and "
                 f"{second:.3f} to the next.")
    return text
