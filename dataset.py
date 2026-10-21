"""
The data as the model sees it: the SMS messages deduplicated, split, tokenized and encoded.

  Data          token ids and labels for both halves, plus what the frames need
  prepare_data  load -> dedupe -> split -> vocabulary (training half only) -> encode
"""
from dataclasses import dataclass

import torch

from sms_data import dedupe, load_messages, split
from vocab import Vocab, encode, token_groups

SPLIT_SEED = 0  # every run uses the same train/test split, whatever its --seed
MAX_LEN = 64    # [CLS] + 63 tokens holds 98.8% of the messages whole
MIN_COUNT = 2   # words seen fewer times in training become [UNK]


@dataclass(frozen=True)
class Data:
    vocab: Vocab
    train_ids: torch.Tensor     # [n_train, MAX_LEN], in a seeded random order
    train_labels: torch.Tensor  # [n_train] float, 1 = spam
    test_ids: torch.Tensor
    test_labels: torch.Tensor
    test_messages: tuple        # the test Messages, in the same order as test_ids
    token_group: torch.Tensor   # index into vocab.GROUPS of every token id, -1 for [PAD]


def _labels(messages):
    return torch.tensor([float(m.is_spam) for m in messages])


def prepare_data():
    """Everything below the test split is learned from the training half only."""
    train, test = split(dedupe(load_messages()), seed=SPLIT_SEED)
    vocab = Vocab.build([m.text for m in train], MIN_COUNT)
    return Data(vocab=vocab,
                train_ids=encode(vocab, [m.text for m in train], MAX_LEN),
                train_labels=_labels(train),
                test_ids=encode(vocab, [m.text for m in test], MAX_LEN),
                test_labels=_labels(test),
                test_messages=tuple(test),
                token_group=token_groups(vocab, train))
