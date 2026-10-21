"""Choosing example messages for the videos.

The training video shows the first short ham and spam of the test set. The architecture and
inference videos rank messages by how well they show the model at work.
"""

from dataclasses import dataclass

import torch

from moe_spam.analysis.trace import MessageTrace, MessageTracer
from moe_spam.data.tokenizer import TOKENIZER
from moe_spam.errors import MoeSpamError
from moe_spam.model.modes import evaluating

FIRST_TOKENS = {False: (8, 13), True: (8, 18)}  # token counts of the training video's examples
MIN_TOKENS = 8   # ranked examples shorter than this show too little of what attention does
MAX_TOKENS = 24  # longer ones do not fit on screen, so they are never picked


@dataclass(frozen=True)
class Pick:
    """A ranked example message.

    Attributes:
        text: The message.
        trace: What the model did to it.
        same: Messages with this label whose last-layer ``[CLS]`` goes to the same expert...
        total: ...out of this many messages with this label.
    """

    text: str
    trace: MessageTrace
    same: int
    total: int


def preference(wrong, unusual, n_tokens, n_punctuation, n_chars):
    """Returns a sort key for ranked examples: smaller is better.

    Args:
        wrong: Whether the model gets the message wrong.
        unusual: Whether its last-layer ``[CLS]`` goes elsewhere than most of its label's.
        n_tokens: Its number of tokens.
        n_punctuation: Its number of punctuation tokens.
        n_chars: Its length in characters.

    Returns:
        A tuple that sorts messages the model gets right, routed like most of their label,
        long enough to show what attention does, with little punctuation (they read like a
        sentence), and short (they fit on screen) first.
    """
    return (wrong, unusual, n_tokens < MIN_TOKENS, n_punctuation, n_chars)


class ExampleFinder:
    """Finds example messages whose every word the model knows.

    Attributes:
        vocab: The model's vocabulary.
    """

    def __init__(self, vocab, tokenizer=TOKENIZER):
        """Creates a finder for messages written in ``vocab``'s words."""
        self.vocab = vocab
        self.tokenizer = tokenizer

    def first(self, messages):
        """Finds the training video's two examples.

        Args:
            messages: The test messages, in order.

        Returns:
            ``(ham_index, spam_index)``: the first ham and the first spam (one with a number)
            that are short enough to show whole and have no ``[UNK]`` token.

        Raises:
            MoeSpamError: If either is missing.
        """
        def fits(message):
            tokens = self.tokenizer.tokenize(message.text)
            low, high = FIRST_TOKENS[message.is_spam]
            return (low <= len(tokens) <= high and self.vocab.knows(tokens)
                    and (not message.is_spam or any(map(self.tokenizer.is_number, tokens))))

        picks = [next((i for i, m in enumerate(messages) if m.is_spam == is_spam and fits(m)),
                      None) for is_spam in (False, True)]
        if None in picks:
            raise MoeSpamError("no suitable ham and spam example message in the test set")
        return tuple(picks)

    @torch.no_grad()
    def ranked(self, model, messages, fits=None):
        """Finds one spam and one ham that show the model at work, ranked by ``preference``.

        Args:
            model: The trained model.
            messages: The candidate messages (the test set).
            fits: Optional check of a ``MessageTrace``; a message it turns down is skipped
                for the next one in order (the videos pass "does it fit on screen").

        Returns:
            ``{"spam": Pick, "ham": Pick}``.

        Raises:
            MoeSpamError: If there are no messages of a label, or none is short enough with
                known words.
        """
        if not messages:
            raise MoeSpamError("no messages to pick from")
        with evaluating(model):
            out = model(self.vocab.encode([m.text for m in messages], model.config.max_len))
        cls_expert, says_spam = out.routing[-1].expert[:, 0], out.logits > 0
        tracer = MessageTracer(model, self.vocab, self.tokenizer)
        picks = {}
        for label, is_spam in (("spam", True), ("ham", False)):
            rows = [i for i, m in enumerate(messages) if m.is_spam == is_spam]
            if not rows:
                raise MoeSpamError(f"no {label} messages to pick from")
            usual = torch.bincount(cls_expert[rows], minlength=model.config.n_experts).argmax()
            tokens = {i: self.tokenizer.tokenize(messages[i].text) for i in rows}
            candidates = sorted(
                (i for i in rows if 0 < len(tokens[i]) <= MAX_TOKENS
                 and self.vocab.knows(tokens[i])),
                key=lambda i: preference(
                    bool(says_spam[i]) != is_spam, bool(cls_expert[i] != usual),
                    len(tokens[i]), sum(map(self.tokenizer.is_punctuation, tokens[i])),
                    len(messages[i].text)))
            traces = (tracer.trace(messages[i].text) for i in candidates)
            trace = next((t for t in traces if fits is None or fits(t)), None)
            if trace is None:
                raise MoeSpamError(f"no {label} message is short enough, with known words, to show")
            same = int((cls_expert[rows] == trace.experts[-1][0]).sum())
            picks[label] = Pick(trace.text, trace, same, len(rows))
        return picks
