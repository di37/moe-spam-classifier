"""The data as the model sees it: deduplicated, split, tokenized and encoded."""

from dataclasses import dataclass

import torch

from moe_spam.data.sms import SmsSpamCollection
from moe_spam.data.vocab import MAX_LEN, TokenGroups, Vocab

SPLIT_SEED = 0  # every run uses the same train/test split, whatever its seed
MIN_COUNT = 2   # words seen fewer times in training become [UNK]


@dataclass(frozen=True)
class SpamData:
    """Both halves of the split, encoded, plus what the training video needs.

    Attributes:
        vocab: The vocabulary, built from the training half only.
        train_ids: ``[n_train, MAX_LEN]`` token ids, in a seeded random order.
        train_labels: ``[n_train]`` floats, 1 for spam.
        test_ids: ``[n_test, MAX_LEN]`` token ids.
        test_labels: ``[n_test]`` floats, 1 for spam.
        test_messages: The test messages, in the order of ``test_ids``.
        groups: The kind of every token, from the training half's spam rates.
    """

    vocab: Vocab
    train_ids: torch.Tensor
    train_labels: torch.Tensor
    test_ids: torch.Tensor
    test_labels: torch.Tensor
    test_messages: tuple
    groups: TokenGroups

    @classmethod
    def prepare(cls, messages=None):
        """Dedupes and splits the messages, then builds the vocabulary and encodes both halves.

        Args:
            messages: The labelled messages; by default, the SMS Spam Collection.

        Returns:
            The prepared data. Everything learned from text comes from the training half.
        """
        collection = SmsSpamCollection()
        messages = collection.load() if messages is None else messages
        train, test = collection.split(collection.dedupe(messages), seed=SPLIT_SEED)
        vocab = Vocab.build([m.text for m in train], MIN_COUNT)
        return cls(vocab=vocab,
                   train_ids=vocab.encode([m.text for m in train], MAX_LEN),
                   train_labels=_labels(train),
                   test_ids=vocab.encode([m.text for m in test], MAX_LEN),
                   test_labels=_labels(test),
                   test_messages=tuple(test),
                   groups=TokenGroups.from_messages(vocab, train))

    @property
    def n_spam(self):
        """The number of spam messages in the test set."""
        return int(self.test_labels.sum())

    @property
    def n_ham(self):
        """The number of ham messages in the test set."""
        return len(self.test_labels) - self.n_spam


def _labels(messages):
    return torch.tensor([float(m.is_spam) for m in messages])
