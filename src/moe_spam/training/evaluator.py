"""Running a model on the fixed test set: the numbers and the frame state of one step."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from moe_spam.data.dataset import MAX_LEN
from moe_spam.data.tokenizer import TOKENIZER
from moe_spam.data.vocab import TokenGroups
from moe_spam.model.modes import evaluating
from moe_spam.training.history import FrameState, TokenRow
from moe_spam.training.scores import SpamScores


@dataclass(frozen=True)
class Evaluation:
    """One evaluation of the model.

    Attributes:
        metrics: Train and test loss, and the test errors by kind.
        state: What the frame for this step shows.
        loads: Each layer's ``[n_experts]`` expert load on the test set.
        scores: The test-set mistakes and the scores that follow from them.
    """

    metrics: dict
    state: FrameState
    loads: list
    scores: SpamScores

    def summary(self):
        """Returns the final numbers for the run record and the README."""
        state = self.state
        top = state.cls_probs.argmax(axis=1)
        by_class = {label: [round(float(np.mean(top[state.is_spam == is_spam] == e)), 4)
                            for e in range(state.cls_probs.shape[1])]
                    for label, is_spam in (("ham", False), ("spam", True))}
        last = len(self.loads)
        return {
            **self.metrics,
            **self.scores.as_dict(),
            **{f"layer {i + 1} load": [round(float(v), 4) for v in load]
               for i, load in enumerate(self.loads)},
            f"layer {last} [CLS] top-1 expert, ham messages": by_class["ham"],
            f"layer {last} [CLS] top-1 expert, spam messages": by_class["spam"],
        }


class Evaluator:
    """Evaluates models on one prepared dataset.

    Attributes:
        data: The prepared ``SpamData``.
        examples: Indices of the test messages the token panel shows.
    """

    def __init__(self, data, examples, tokenizer=TOKENIZER):
        """Creates an evaluator for ``data`` that also traces the ``examples`` messages."""
        self.data = data
        self.examples = examples
        self.tokenizer = tokenizer

    @torch.no_grad()
    def evaluate(self, model):
        """Runs ``model`` in eval mode on the whole test set.

        It also runs on as many training messages, for a training loss measured the same
        way (the training half is in random order, so its first rows are a random sample).

        Returns:
            The ``Evaluation``.
        """
        data = self.data
        n_train = len(data.test_ids)
        with evaluating(model):
            out = model(data.test_ids)
            train_logits = model(data.train_ids[:n_train]).logits
        labels = data.test_labels.bool()
        wrong = (out.logits > 0) != labels
        scores = SpamScores.from_logits(out.logits, data.test_labels)
        first = out.routing[0]
        p_spam = torch.sigmoid(out.logits)
        state = FrameState(
            cls_probs=out.routing[-1].probs[:, 0].numpy().copy(),  # a copy, not a view of [n, L, E]
            is_spam=labels.numpy(),
            wrong=wrong.numpy(),
            examples=tuple(self._token_row(data.test_messages[i], first.expert[i], p_spam[i])
                           for i in self.examples),
            group_shares=TokenGroups.expert_shares(data.groups.of(data.test_ids)[first.mask],
                                                   first.expert[first.mask],
                                                   model.config.n_experts),
        )
        metrics = {
            "train BCE": F.binary_cross_entropy_with_logits(
                train_logits, data.train_labels[:n_train]).item(),
            "test BCE": F.binary_cross_entropy_with_logits(out.logits, data.test_labels).item(),
            "errors": scores.errors,
            **scores.by_kind(),
        }
        return Evaluation(metrics, state, [r.load().numpy() for r in out.routing], scores)

    def _token_row(self, message, experts, p_spam):
        words = ("[CLS]", *self.tokenizer.pieces(message.text))[:MAX_LEN]
        return TokenRow("spam" if message.is_spam else "ham", words,
                        tuple(experts[:len(words)].tolist()), float(p_spam))
