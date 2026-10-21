"""
Text -> token ids, BERT-style, but with whole words where BERT uses word pieces.

  pieces        the lowercased words, digit runs and punctuation marks of a text
  tokenize      the same pieces as tokens; digit runs become [NUM] (up to 4 digits) or [NUM5+]
                (phone numbers and short codes), so they share one common token instead of
                each being rare
  Vocab         word <-> id, built from the training messages only
  encode        texts -> [CLS] + token ids, cut or padded to max_len
  token_groups  which group (spam-leaning words, numbers, ...) each token id falls in, for
                the "which tokens each expert gets" panel
"""
import re
from collections import Counter
from dataclasses import dataclass, field

import torch

SPECIALS = ("[PAD]", "[UNK]", "[CLS]")
PAD, UNK, CLS = range(len(SPECIALS))
NUMBER_TOKENS = ("[NUM]", "[NUM5+]")
LONG_NUMBER = 5  # digits from which a number counts as a phone number or short code
GROUPS = ("[CLS]", "spam-leaning words", "ham-leaning words", "other words", "numbers",
          "punctuation", "[UNK]")

_PIECE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+|[^\sa-z\d]")


def pieces(text):
    return _PIECE.findall(text.lower())


def _token(piece):
    if piece.isdecimal():
        return NUMBER_TOKENS[len(piece) >= LONG_NUMBER]
    return piece


def tokenize(text):
    return [_token(p) for p in pieces(text)]


@dataclass(frozen=True)
class Vocab:
    words: tuple
    _index: dict = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "_index", {w: i for i, w in enumerate(self.words)})

    @classmethod
    def build(cls, texts, min_count=2):
        """Specials first, then every token seen at least min_count times, most frequent first."""
        if min_count < 1:
            raise ValueError(f"min_count must be at least 1, got {min_count}")
        counts = Counter(t for text in texts for t in tokenize(text))
        kept = sorted((w for w, c in counts.items() if c >= min_count),
                      key=lambda w: (-counts[w], w))
        return cls(SPECIALS + tuple(kept))

    def __len__(self):
        return len(self.words)

    def id(self, word):
        return self._index.get(word, UNK)


def encode(vocab, texts, max_len=64):
    """LongTensor [len(texts), max_len]: [CLS], then the tokens (cut to fit), then [PAD]s."""
    if max_len < 2:
        raise ValueError(f"max_len must be at least 2 ([CLS] and one token), got {max_len}")
    rows = [[CLS] + [vocab.id(t) for t in tokenize(text)][: max_len - 1] for text in texts]
    padded = [row + [PAD] * (max_len - len(row)) for row in rows]
    return torch.tensor(padded, dtype=torch.long).reshape(len(texts), max_len)


def token_groups(vocab, messages, min_count=5, spam_leaning=0.8, ham_leaning=0.02):
    """
    LongTensor [len(vocab)]: the index in GROUPS of each token id, -1 for [PAD].
    A word is spam-leaning if at least `spam_leaning` of its uses in `messages` are in spam,
    ham-leaning if at most `ham_leaning` are; words used fewer than min_count times are
    "other words". Pass the training messages, so the test set stays unseen.
    """
    total = Counter(t for m in messages for t in tokenize(m.text))
    spam = Counter(t for m in messages if m.is_spam for t in tokenize(m.text))

    def group(word):
        if word == "[PAD]":
            return -1
        if word in ("[CLS]", "[UNK]"):
            return GROUPS.index(word)
        if word in NUMBER_TOKENS:
            return GROUPS.index("numbers")
        if not word[0].isalpha():
            return GROUPS.index("punctuation")
        uses = total[word]
        if uses >= min_count and spam[word] / uses >= spam_leaning:
            return GROUPS.index("spam-leaning words")
        if uses >= min_count and spam[word] / uses <= ham_leaning:
            return GROUPS.index("ham-leaning words")
        return GROUPS.index("other words")

    return torch.tensor([group(w) for w in vocab.words], dtype=torch.long)
